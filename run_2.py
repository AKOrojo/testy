import os
import torch
import random
from datasets import load_dataset, IterableDataset
from itertools import chain
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from main import T5ForConditionalGeneration, T5Config


import random
from datetime import datetime


def preprocess_for_byt5_masking(examples, tokenizer, corruption_rate=0.15, mean_noise_span_length=3.0):
    sentinel_tokens = [f"<sentinel_{i}>" for i in range(100)]
    sentinel_token_ids = tokenizer.convert_tokens_to_ids(sentinel_tokens)
    
    if len(sentinel_token_ids) == 0 or sentinel_token_ids[0] == tokenizer.unk_token_id:
        raise ValueError(
            "Sentinel tokens are not correctly configured in the tokenizer. "
            "Ensure they have been added as special tokens."
        )

    all_input_ids = []
    all_labels = []

    for input_ids in examples["input_ids"]:
        num_tokens = len(input_ids)
        num_to_corrupt = int(num_tokens * corruption_rate)

        corrupted_indices = set()
        while len(corrupted_indices) < num_to_corrupt:
            span_length = min(int(random.expovariate(1.0 / mean_noise_span_length)) + 1, num_tokens)
            if span_length == 0: continue
            
            start_index = random.randint(0, num_tokens - span_length)
            corrupted_indices.update(range(start_index, start_index + span_length))

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
        
        if len(spans) > len(sentinel_token_ids):
            spans = spans[:len(sentinel_token_ids)]

        corrupted_input = []
        labels = []
        last_index = 0
        
        for i, span in enumerate(spans):
            corrupted_input.extend(input_ids[last_index:span[0]])
            
            sentinel_id_for_this_span = sentinel_token_ids[i]
            corrupted_input.append(sentinel_id_for_this_span)
            
            labels.append(sentinel_id_for_this_span)
            labels.extend(input_ids[j] for j in span)
            
            last_index = span[-1] + 1
            
        corrupted_input.extend(input_ids[last_index:])
        labels.append(tokenizer.eos_token_id)
        
        all_input_ids.append(corrupted_input)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


def chunk_text_batched(examples, tokenizer, chunk_size=512):
    all_chunks = []
    
    for text in examples["text"]:
        text = text.strip()
        if not text:
            continue
        
        tokens = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) == chunk_size:
                all_chunks.append(chunk)
    
    return {"input_ids": all_chunks}


# --- Main Training Function ---

def main():
    os.environ["WANDB_PROJECT"] = "c4-byt5-pretraining-stream"

    tokenizer_name = "google/byt5-small"
    model_output_dir = "./c4-byt5-from-scratch-stream"

    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Adding 100 new sentinel tokens to the tokenizer...")
    sentinel_tokens = [f"<sentinel_{i}>" for i in range(100)]
    tokenizer.add_special_tokens({"additional_special_tokens": sentinel_tokens})

    print("Initializing a new ByT5-small model from scratch with Rotary Embeddings...")
    config = T5Config(
        d_model=1472,
        d_kv=64,
        d_ff=3584,
        num_layers=12,
        num_decoder_layers=4,
        num_heads=6,
        feed_forward_proj="gated-gelu",
        tie_word_embeddings=False,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        _attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    model = T5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    
    print(f"Resizing model token embeddings to {len(tokenizer)} to account for new tokens.")
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(dtype=torch.bfloat16)

    print(f"Model created with {model.num_parameters():,} parameters.")

    print("Loading C4 dataset in STREAMING mode...")
    full_dataset = load_dataset("allenai/c4", "en", streaming=True)
    train_stream = full_dataset['train']

    shuffled_stream = train_stream.shuffle(seed=42, buffer_size=10_000)

    print("Applying chunking and tokenization...")
    chunked_stream = shuffled_stream.map(
        lambda examples: chunk_text_batched(examples, tokenizer, chunk_size=1024),
        batched=True,
        batch_size=32,
        remove_columns=["text", "timestamp", "url"]
    )

    print("Applying ByT5-compatible masking transformation...")
    masked_stream = chunked_stream.map(
        lambda examples: preprocess_for_byt5_masking(examples, tokenizer),
        batched=True,
        batch_size=256,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="max_length",
        max_length=1024,
        pad_to_multiple_of=8,
    )

    run_name = f"c4-byt5-pretrain-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        run_name=run_name,
        max_steps=1_000_000,
        optim="adafactor",
        learning_rate=0.001,
        bf16=True,
        torch_compile=False,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=8,
        logging_steps=500,
        save_steps=10000,
        save_total_limit=3,
        warmup_steps=2000,
        weight_decay=0.01,
        dataloader_num_workers=1,
        report_to=["wandb"],
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dispatch_batches=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=masked_stream,
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