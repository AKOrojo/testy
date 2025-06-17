# In a new file, e.g., prepare_data.py

import random
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# Import your custom model and config from main.py
from main import T5ForConditionalGeneration, T5Config


def preprocess_for_t5_denoising(examples, tokenizer, corruption_rate=0.15, mean_noise_span_length=3.0):
    """
    Preprocesses a batch of text examples for T5-style span corruption.

    Args:
        examples (dict): A batch from a Hugging Face dataset, expected to have a "text" field.
        tokenizer: The tokenizer to use.
        corruption_rate (float): The fraction of tokens to corrupt.
        mean_noise_span_length (float): The average length of a corrupted span.

    Returns:
        dict: A dictionary with 'input_ids' and 'labels' for the model.
    """
    # Get the special "extra_id" tokens from the tokenizer
    extra_id_tokens = [f"<extra_id_{i}>" for i in range(100)]
    extra_id_token_ids = tokenizer.convert_tokens_to_ids(extra_id_tokens)

    inputs = []
    targets = []

    for text in examples["text"]:
        # Tokenize the text
        input_ids = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        num_tokens = len(input_ids)
        num_to_corrupt = int(num_tokens * corruption_rate)

        corrupted_indices = set()
        while len(corrupted_indices) < num_to_corrupt:
            span_length = int(random.expovariate(1.0 / mean_noise_span_length))
            span_length = min(span_length, num_to_corrupt - len(corrupted_indices), 10)  # Clamp span length
            if span_length == 0:
                continue

            start_index = random.randint(0, num_tokens - span_length)
            for i in range(span_length):
                corrupted_indices.add(start_index + i)

        if not corrupted_indices:
            continue

        # Create the corrupted input and the target sequence
        sorted_indices = sorted(list(corrupted_indices))

        new_input_ids = []
        target_ids = []

        current_extra_id_idx = 0
        last_index = 0

        in_span = False
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]

            # If we are starting a new span
            if not in_span:
                # Add the text before the span
                new_input_ids.extend(input_ids[last_index:index])
                # Add the sentinel token to input and target
                new_input_ids.append(extra_id_token_ids[current_extra_id_idx])
                target_ids.append(extra_id_token_ids[current_extra_id_idx])
                in_span = True

            # Add the original token to the target
            target_ids.append(input_ids[index])

            # If the span ends here
            if i == len(sorted_indices) - 1 or sorted_indices[i + 1] != index + 1:
                in_span = False
                current_extra_id_idx += 1

            last_index = index + 1

        # Add remaining text after the last span
        new_input_ids.extend(input_ids[last_index:])

        # Add EOS token to target
        target_ids.append(tokenizer.eos_token_id)

        inputs.append(tokenizer.decode(new_input_ids))
        targets.append(tokenizer.decode(target_ids))

    # Final tokenization to create model inputs
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def chunk_examples(examples, chunk_size=512):
    """Chunks a batch of long text examples into smaller pieces."""
    all_chunks = []
    for text in examples["text"]:
        # Simple chunking by words, can be improved
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:  # Ensure chunk is not empty
                all_chunks.append(chunk)
    return {"text": all_chunks}

def main():
    # Model name and tokenizer
    # We use a pretrained tokenizer but will initialize the model from scratch
    tokenizer_name = "t5-small"
    model_output_dir = "./c4-t5-from-scratch"

    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 1. Initialize the Model from Scratch using your custom T5 implementation
    print("Initializing a new T5 model from scratch...")
    # T5Config from your main.py has the t5-small parameters by default
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        # Use 'flash_attention_2' for efficiency if your hardware supports it
        # _attn_implementation="flash_attention_2",
    )
    # Use the T5ForConditionalGeneration class from your main.py
    model = T5ForConditionalGeneration(config)
    print(f"Model created with {model.num_parameters():,} parameters.")

    # 2. Load and Prepare the Dataset
    # WARNING: The full C4 dataset is massive (~750GB).
    # For a real run, you'd use the full dataset. For testing, we use a small part.
    print("Loading and preparing C4 dataset...")
    # Use streaming to avoid downloading the whole dataset at once
    # For a test run, just take a small sample
    full_dataset = load_dataset("allenai/c4", "en", streaming=True)
    train_dataset_stream = full_dataset['train']

    # Let's take 10,000 examples for this test run
    # For a real run, you would not use .take()
    train_dataset_sample = train_dataset_stream.take(10000)

    # Convert the IterableDataset to a standard Dataset for easier processing
    # This will download the 10,000 examples.
    from datasets import Dataset
    train_dataset = Dataset.from_generator(lambda: (yield from train_dataset_sample))

    chunked_dataset = train_dataset.map(
        chunk_examples,
        batched=True,
        remove_columns=train_dataset.column_names  # Remove old columns
    )
    print(f"Dataset chunked. Original size: {len(train_dataset)}, New size: {len(chunked_dataset)}")

    # Apply the denoising preprocessing function
    tokenized_dataset = chunked_dataset.map(
        lambda examples: preprocess_for_t5_denoising(examples, tokenizer),
        batched=True,
        remove_columns=["text"]  # The only remaining column is "text"
    )
    tokenized_dataset.set_format(type="torch")
    print("Dataset prepared.")

    # 3. Set up Training
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=8,  # Adjust based on your GPU memory
        gradient_accumulation_steps=4,  # Effective batch size = 8 * 4 = 32
        learning_rate=1e-3,  # T5 pre-training often uses a larger learning rate
        num_train_epochs=1,  # For a real run, you would train for many more epochs or steps
        logging_steps=100,
        save_steps=1000,
        fp16=True,  # Use mixed-precision training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 4. Start Training
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 5. Save the final model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")


if __name__ == "__main__":
    main()