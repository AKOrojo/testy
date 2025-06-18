# train_c4.py (Revised)
import os

import torch
import random
from datasets import load_dataset
from itertools import chain
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# Import your custom model and config from main.py
from main import T5ForConditionalGeneration, T5Config


# --- Data Preprocessing Functions (Span Corruption & Chunking) ---

def chunk_and_tokenize_examples(examples, tokenizer, chunk_size=512):
    """Tokenize and chunk a batch of text examples."""
    # First, tokenize all the text in the batch
    tokenized_texts = tokenizer(examples["text"], truncation=False, add_special_tokens=False)

    concatenated_ids = list(chain.from_iterable(tokenized_texts['input_ids']))
    total_length = len(concatenated_ids)

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= chunk_size:
        total_length = (total_length // chunk_size) * chunk_size

    # Split by chunks of chunk_size
    result = {
        "input_ids": [concatenated_ids[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
    }
    return result


def preprocess_for_t5_denoising(examples, tokenizer, corruption_rate=0.15, mean_noise_span_length=3.0):
    """Preprocesses a batch of tokenized examples for T5-style span corruption."""
    extra_id_tokens = [f"<extra_id_{i}>" for i in range(100)]
    extra_id_token_ids = tokenizer.convert_tokens_to_ids(extra_id_tokens)

    all_input_ids = []
    all_labels = []

    for input_ids in examples["input_ids"]:
        num_tokens = len(input_ids)
        num_to_corrupt = int(num_tokens * corruption_rate)

        corrupted_indices = set()
        while len(corrupted_indices) < num_to_corrupt:
            span_length = min(int(random.expovariate(1.0 / mean_noise_span_length)) + 1, 10)
            if span_length == 0:
                continue

            start_index = random.randint(0, num_tokens - span_length)
            for i in range(span_length):
                if len(corrupted_indices) < num_to_corrupt:
                    corrupted_indices.add(start_index + i)

        if not corrupted_indices:
            continue

        sorted_indices = sorted(list(corrupted_indices))

        # Ensure we don't have more spans than available sentinel tokens
        # Find continuous spans
        spans = []
        current_span = []
        for i, index in enumerate(sorted_indices):
            if not current_span or index == current_span[-1] + 1:
                current_span.append(index)
            else:
                spans.append(current_span)
                current_span = [index]
        if current_span:
            spans.append(current_span)

        # Limit to the number of available extra_id tokens
        if len(spans) > len(extra_id_token_ids):
            spans = spans[:len(extra_id_token_ids)]
            sorted_indices = [idx for span in spans for idx in span]

        new_input_ids = []
        target_ids = []

        current_extra_id_idx = 0
        last_index = 0

        for span in spans:
            # Add text before the span
            new_input_ids.extend(input_ids[last_index:span[0]])
            # Add sentinel token
            new_input_ids.append(extra_id_token_ids[current_extra_id_idx])
            target_ids.append(extra_id_token_ids[current_extra_id_idx])
            # Add original tokens to target
            target_ids.extend([input_ids[i] for i in span])

            current_extra_id_idx += 1
            last_index = span[-1] + 1

        new_input_ids.extend(input_ids[last_index:])
        target_ids.append(tokenizer.eos_token_id)

        all_input_ids.append(new_input_ids)
        all_labels.append(target_ids)

    # Pad the results
    max_input_len = 512
    max_label_len = 128

    padded_inputs = tokenizer.pad(
        {"input_ids": all_input_ids}, padding="max_length", max_length=max_input_len, return_tensors="pt"
    )
    padded_labels = tokenizer.pad(
        {"input_ids": all_labels}, padding="max_length", max_length=max_label_len, return_tensors="pt"
    )

    return {"input_ids": padded_inputs["input_ids"], "attention_mask": padded_inputs["attention_mask"],
            "labels": padded_labels["input_ids"]}


# --- Main Training Function ---

def main():
    # 2. Set the environment variable right at the beginning of your main function
    os.environ["WANDB_PROJECT"] = "c4-t5-pretraining"

    tokenizer_name = "t5-small"
    model_output_dir = "./c4-t5-from-scratch"

    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Initializing a new T5 model from scratch...")
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        _attn_implementation="flash_attention_2",
    )
    # **FIX**: Load the model with the correct dtype for Flash Attention and bf16 training
    model = T5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    print(f"Model created with {model.num_parameters():,} parameters.")

    print("Loading C4 dataset from LOCAL CACHE...")
    local_cache_dir = "../c4_dataset_cache"
    full_dataset = load_dataset("allenai/c4", "en", cache_dir=local_cache_dir)
    train_stream = full_dataset['train']

    print("Applying transformations (chunking, tokenizing, and denoising) on the fly...")
    # 1. Filter out empty or whitespace-only docs
    filtered_stream = train_stream.filter(lambda x: x['text'].strip() != "")
    # 2. Chunk and tokenize
    chunked_tokenized_stream = filtered_stream.map(
        lambda examples: chunk_and_tokenize_examples(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=["text", "timestamp", "url"]
    )
    # 3. Apply denoising
    tokenized_stream = chunked_tokenized_stream.map(
        lambda examples: preprocess_for_t5_denoising(examples, tokenizer),
        batched=True,
        batch_size=128,
    )

    # We need a data collator that doesn't try to re-pad our already-padded data
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        max_steps=1000000,
        bf16=True,
        torch_compile=False,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=2,
        logging_steps=500,
        save_steps=5000,
        save_total_limit=5,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        weight_decay=0.01,
        dataloader_num_workers=4,
        report_to=["wandb"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_stream,
        data_collator=data_collator,
    )

    print("🚀 Starting distributed training on the full C4 dataset...")
    trainer.train()
    print("✅ Training complete.")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")


if __name__ == "__main__":
    main()