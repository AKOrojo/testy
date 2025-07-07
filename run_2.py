import os
import torch
import random
from datasets import load_dataset, IterableDataset
from itertools import chain
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,  # Changed from default_data_collator
)

# Your custom model from main.py is imported here
from main import T5ForConditionalGeneration, T5Config


# --- Data Preprocessing Functions (Span Corruption & Chunking) ---
# NO CHANGES ARE NEEDED IN THESE FUNCTIONS. They work correctly with byte-level tokenizers.
def chunk_and_tokenize_stream(dataset: IterableDataset, tokenizer, chunk_size=512):
    token_buffer = []
    for example in dataset:
        # Filter out empty or whitespace-only documents on the fly
        text = example.get('text', '').strip()
        if not text:
            continue

        # Tokenize the text without special tokens. For ByT5, this converts text to byte tokens.
        tokens = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        token_buffer.extend(tokens)

        # Yield chunks of chunk_size from the buffer
        while len(token_buffer) >= chunk_size:
            chunk = token_buffer[:chunk_size]
            yield {"input_ids": chunk}
            # Move the buffer forward
            token_buffer = token_buffer[chunk_size:]


def preprocess_for_t5_denoising(examples, tokenizer, corruption_rate=0.15, mean_noise_span_length=3.0):
    # This function is also compatible as-is.
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

    return {"input_ids": all_input_ids, "labels": all_labels}


# --- Main Training Function ---

def main():
    os.environ["WANDB_PROJECT"] = "c4-byt5-pretraining-stream"

    # --- CHANGE 1: Use ByT5 Tokenizer ---
    tokenizer_name = "google/byt5-small"
    model_output_dir = "./c4-byt5-from-scratch-stream"

    print(f"Loading tokenizer '{tokenizer_name}'...")
    # ByT5Tokenizer is a character/byte level tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # --- CHANGE 2: Use ByT5-small Model Configuration ---
    print("Initializing a new ByT5-small model from scratch with Rotary Embeddings...")
    config = T5Config(
        # Architecture from your JSON config
        d_model=1472,
        d_kv=64,
        d_ff=3584,
        num_layers=12,
        num_decoder_layers=4,
        num_heads=6,
        feed_forward_proj="gated-gelu",
        tie_word_embeddings=False,
        # Get vocab and special tokens from the ByT5 tokenizer
        vocab_size=tokenizer.vocab_size,  # Should be 384 for ByT5
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        # Use flash attention from your custom model code
        attn_implementation="flash_attention_2",
    )
    # NOTE: The model in main.py uses Rotary Position Embeddings (RoPE),
    # so `relative_attention_num_buckets` is not needed.

    # Load the model with the correct dtype for Flash Attention and bf16 training
    model = T5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    print(f"Model created with {model.num_parameters():,} parameters.")

    # 1. Load C4 in streaming mode (No changes here)
    print("Loading C4 dataset in STREAMING mode...")
    full_dataset = load_dataset("allenai/c4", "en", streaming=True)
    train_stream = full_dataset['train']

    # 2. Define a generator "factory" function.
    # This function wraps the shuffling and chunking logic.
    def processing_generator():
        # Shuffling and chunking happens inside the factory,
        # ensuring a new stream is created each time this function is called.
        shuffled_stream = train_stream.shuffle(seed=42, buffer_size=10_000)
        yield from chunk_and_tokenize_stream(shuffled_stream, tokenizer, chunk_size=1024)

    # 3. Create the IterableDataset from the factory function.
    print("Applying transformations (chunking, tokenizing, and denoising) on the fly...")
    chunked_dataset = IterableDataset.from_generator(processing_generator)

    # 4. Apply T5 denoising using .map() to the newly created dataset
    denoised_stream = chunked_dataset.map(
        lambda examples: preprocess_for_t5_denoising(examples, tokenizer, mean_noise_span_length=20.0),
        batched=True,
        batch_size=256,
    )
    
    # Data collator for dynamic padding within each batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        max_steps=1_000_000,
        optim="adafactor",
        learning_rate=None,
        #lr_scheduler_type="constant",
        bf16=True,
        torch_compile=False,
        per_device_train_batch_size=16,  # ByT5 is larger, you may need to reduce batch size
        gradient_accumulation_steps=16,  # Adjust accumulation to compensate for smaller batch
        logging_steps=500,
        save_steps=10000,
        save_total_limit=3,
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
