import argparse
import os
import random

import torch
from datasets import load_dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

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
    max_length = tokenizer.model_max_length

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

        truncated_labels = target_ids[:max_length]

        all_input_ids.append(new_input_ids)
        all_labels.append(truncated_labels)

    # Padding is now handled by the Trainer's data collator
    return {"input_ids": all_input_ids, "labels": all_labels}


# --- Main Training Function ---

# --- Main Training Function ---

def main(args):
    if "wandb" in args.report_to:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"Loading tokenizer '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Initializing a new T5 model from scratch...")
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        _attn_implementation="flash_attention_2" if args.use_flash_attention_2 else "eager"
    )
    # Load the model with the correct dtype for Flash Attention and bf16 training
    model = T5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    print(f"Model created with {model.num_parameters():,} parameters.")

    # 1. Load C4 in streaming mode
    print("Loading C4 dataset in STREAMING mode...")
    full_dataset = load_dataset("allenai/c4", "en", streaming=True)
    train_stream = full_dataset['train']

    # 2. Shuffle the streaming dataset
    shuffled_stream = train_stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    # 3. Define a new tokenization and chunking map function
    def tokenize_and_chunk(examples):
        # First, tokenize all the text examples in the batch
        tokenized_outputs = tokenizer(examples["text"], truncation=False, add_special_tokens=False)

        # Concatenate all the token lists into one super-list
        concatenated_ids = sum(tokenized_outputs["input_ids"], [])

        # Calculate the total number of chunks we can make
        total_length = len(concatenated_ids)
        num_chunks = total_length // args.chunk_size

        # Create the chunks
        chunked_ids = [
            concatenated_ids[i * args.chunk_size : (i + 1) * args.chunk_size]
            for i in range(num_chunks)
        ]
        return {"input_ids": chunked_ids}


    # 4. Apply transformations
    # First, tokenize and chunk the text stream. We use a larger batch_size for map to make chunking more efficient.
    chunked_stream = shuffled_stream.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=2000, # This can be adjusted based on memory
        remove_columns=["text", "timestamp", "url"]
    )

    # Then, apply the denoising map to the chunked stream
    denoised_stream = chunked_stream.map(
        lambda examples: preprocess_for_t5_denoising(
            examples,
            tokenizer,
            corruption_rate=args.corruption_rate,
            mean_noise_span_length=args.mean_noise_span_length
        ),
        batched=True,
        batch_size=args.map_batch_size,
    )

    # Data collator for dynamic padding within each batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        bf16=True,
        torch_compile=args.torch_compile,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        remove_unused_columns=False,
        seed=args.seed,
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

    trainer.save_model(args.model_output_dir)
    tokenizer.save_pretrained(args.model_output_dir)
    print(f"Model saved to {args.model_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5 Pre-training on C4 with Streaming")

    # Model and Tokenizer arguments
    parser.add_argument("--tokenizer_name", type=str, default="t5-small", help="Tokenizer to use.")
    parser.add_argument("--model_output_dir", type=str, default="./c4-t5-from-scratch-stream", help="Directory to save the final model.")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Enable Flash Attention 2 for faster training.")

    # Data processing arguments
    parser.add_argument("--chunk_size", type=int, default=512, help="Size of token chunks for processing.")
    parser.add_argument("--corruption_rate", type=float, default=0.15, help="Rate of token corruption for denoising.")
    parser.add_argument("--mean_noise_span_length", type=float, default=3.0, help="Mean length of noise spans.")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000, help="Buffer size for shuffling the streaming dataset.")
    parser.add_argument("--map_batch_size", type=int, default=256, help="Batch size for the .map() preprocessing step.")

    # Training arguments
    parser.add_argument("--max_steps", type=int, default=1_000_000, help="Total number of training steps.")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for optimization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps for the LR scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--optim", type=str, default="adafactor", help="Optimizer to use.")

    # Logging and Saving arguments
    parser.add_argument("--logging_steps", type=int, default=500, help="Log training metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting backend (e.g., 'wandb', 'none').")
    parser.add_argument("--wandb_project", type=str, default="c4-t5-pretraining-stream", help="WandB project name.")
    parser.add_argument("--run_name", type=str, default=None, help="A unique name for the training run, for WandB.") # <-- New argument

    args = parser.parse_args()
    main(args)