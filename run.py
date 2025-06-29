import os
import torch
import random
from datasets import load_dataset, IterableDataset
from itertools import chain
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from main import T5ForConditionalGeneration, T5Config


# --- Data Preprocessing Functions (Span Corruption & Chunking) ---
def chunk_and_tokenize_stream(dataset: IterableDataset, tokenizer, chunk_size=512):
    token_buffer = []
    for example in dataset:
        # Filter out empty or whitespace-only documents on the fly
        text = example.get('text', '').strip()
        if not text:
            continue

        # Tokenize the text without special tokens
        tokens = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        token_buffer.extend(tokens)

        # Yield chunks of chunk_size from the buffer
        while len(token_buffer) >= chunk_size:
            chunk = token_buffer[:chunk_size]
            yield {"input_ids": chunk}
            # Move the buffer forward
            token_buffer = token_buffer[chunk_size:]


def preprocess_for_t5_denoising(examples, tokenizer, corruption_rate=0.15, mean_noise_span_length=3.0):
    extra_id_tokens = [f"<extra_id_{i}>" for i in range(100)]
    extra_id_token_ids = tokenizer.convert_tokens_to_ids(extra_id_tokens)

    all_input_ids = []
    all_labels = []

    for input_ids in examples["input_ids"]:
        num_tokens = len(input_ids)
        num_to_corrupt = int(num_tokens * corruption_rate)

        corrupted_indices = set()
        # Create noise spans
        while len(corrupted_indices) < num_to_corrupt:
            # Sample a span length from an exponential distribution
            span_length = min(int(random.expovariate(1.0 / mean_noise_span_length)) + 1, 10)
            if span_length == 0: continue

            start_index = random.randint(0, num_tokens - span_length)
            # Add indices from the selected span to the set of corrupted indices
            corrupted_indices.update(range(start_index, start_index + span_length))

        # Ensure we don't exceed the number of tokens to corrupt
        corrupted_indices = sorted(list(corrupted_indices))[:num_to_corrupt]

        if not corrupted_indices:
            continue

        # Find continuous spans to replace with a single sentinel token
        spans = []
        current_span = []
        for i, index in enumerate(corrupted_indices):
            if not current_span or index == current_span[-1] + 1:
                current_span.append(index)
            else:
                spans.append(current_span)
                current_span = [index]
        if current_span:
            spans.append(current_span)

        # Limit to the number of available sentinel tokens
        if len(spans) > len(extra_id_token_ids):
            spans = spans[:len(extra_id_token_ids)]

        new_input_ids = []
        target_ids = []
        current_extra_id_idx = 0
        last_index = 0

        for span in spans:
            new_input_ids.extend(input_ids[last_index:span[0]])
            new_input_ids.append(extra_id_token_ids[current_extra_id_idx])
            target_ids.append(extra_id_token_ids[current_extra_id_idx])
            target_ids.extend([input_ids[i] for i in span])
            current_extra_id_idx += 1
            last_index = span[-1] + 1

        new_input_ids.extend(input_ids[last_index:])
        target_ids.append(tokenizer.eos_token_id)

        all_input_ids.append(new_input_ids)
        all_labels.append(target_ids)

    # Padding is now handled by the Trainer's data collator
    return {"input_ids": all_input_ids, "labels": all_labels}


# --- Main Training Function ---

def main():
    os.environ["WANDB_PROJECT"] = "c4-t5-pretraining-stream"

    tokenizer_name = "t5-small"
    model_output_dir = "./c4-t5-from-scratch-stream"

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
    # Load the model with the correct dtype for Flash Attention and bf16 training
    model = T5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    print(f"Model created with {model.num_parameters():,} parameters.")

    # 1. Load C4 in streaming mode
    print("Loading C4 dataset in STREAMING mode...")
    full_dataset = load_dataset("allenai/c4", "en", streaming=True)
    train_stream = full_dataset['train']

    # 2. Shuffle the streaming dataset
    shuffled_stream = train_stream.shuffle(seed=42, buffer_size=10_000)

    # 3. Apply stateful chunking and tokenization using our generator
    print("Applying transformations (chunking, tokenizing, and denoising) on the fly...")
    chunked_tokenized_stream = chunk_and_tokenize_stream(shuffled_stream, tokenizer, chunk_size=512)

    # 4. Apply T5 denoising using .map() on our new stream of chunks
    processed_stream_generator = IterableDataset.from_generator(
        lambda: chunked_tokenized_stream
    )

    denoised_stream = processed_stream_generator.map(
        lambda examples: preprocess_for_t5_denoising(examples, tokenizer),
        batched=True,
        batch_size=256,
    )

    # Data collator for dynamic padding within each batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        max_steps=1_000_000,
        bf16=True,
        torch_compile=False,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=2,
        logging_steps=500,
        save_steps=10000,
        save_total_limit=3,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        weight_decay=0.01,
        dataloader_num_workers=4,
        report_to=["wandb"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=denoised_stream,
        data_collator=data_collator,
    )

    print("🚀 Starting distributed training on the streaming C4 dataset...")
    trainer.train()
    print("✅ Training complete.")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")


if __name__ == "__main__":
    main()