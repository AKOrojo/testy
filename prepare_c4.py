import os
import shutil
import glob
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from itertools import chain
import random

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


if __name__ == "__main__":
    # --- Configuration ---
    tokenizer_name = "t5-small"

    # === STEP 1: Point this to the directory with your existing .arrow files ===
    initial_arrow_cache_dir = "../../Downloads/c4_dataset_cache/allenai___c4/en/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2"

    # Define directories for temporary and final datasets
    processed_shards_dir = "./c4_processed_shards"
    final_dataset_path = "./c4_processed_final"

    os.makedirs(processed_shards_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # === STEP 2: Find all the raw .arrow train files ===
    file_pattern = os.path.join(initial_arrow_cache_dir, "c4-train-*.arrow")
    local_arrow_files = sorted(glob.glob(file_pattern))

    if not local_arrow_files:
        print(f"ERROR: No 'c4-train-*.arrow' files found in '{initial_arrow_cache_dir}'.")
        print("Please make sure the `initial_arrow_cache_dir` variable points to the correct directory.")
        exit()

    print(f"Found {len(local_arrow_files)} source Arrow files to process.")

    # --- Main Loop: Process one LOCAL .arrow file at a time ---
    for i, arrow_file_path in enumerate(local_arrow_files):
        shard_name = os.path.basename(arrow_file_path)
        processed_shard_path = os.path.join(processed_shards_dir, f"shard_{i}")

        if os.path.exists(processed_shard_path):
            print(f"--- Shard {i + 1}/{len(local_arrow_files)} already processed. Skipping. ---")
            continue

        print(f"--- Processing Shard {i + 1}/{len(local_arrow_files)}: {shard_name} ---")

        # 1. Load just that one .arrow file from your disk
        print("1. Loading source Arrow file...")
        temp_dataset = load_dataset("arrow", data_files=arrow_file_path, split="train")

        # 2. Apply the full processing pipeline
        print("2. Filtering, chunking, and denoising...")
        filtered_dataset = temp_dataset.filter(lambda x: x['text'] and x['text'].strip() != "", num_proc=16)
        chunked_dataset = filtered_dataset.map(
            lambda exs: chunk_and_tokenize_examples(exs, tokenizer),
            batched=True, batch_size=500, remove_columns=["text", "timestamp", "url"], num_proc=16
        )
        processed_shard = chunked_dataset.map(
            lambda exs: preprocess_for_t5_denoising(exs, tokenizer),
            batched=True, batch_size=256, num_proc=16
        )

        # 3. Save the processed shard
        print("3. Saving processed shard...")
        processed_shard.save_to_disk(processed_shard_path)

        # 4. KEY STEP: Delete the original .arrow file to save space
        print(f"4. Deleting source Arrow file: {arrow_file_path}...")
        os.remove(arrow_file_path)

        print(f"--- Shard {i + 1} complete. --- \n")

    # --- Final Step: Combine all processed shards ---
    print("All shards processed. Now combining them into a final dataset...")

    shard_paths = [os.path.join(processed_shards_dir, f"shard_{i}") for i in range(len(local_arrow_files))]
    processed_shards_list = [load_from_disk(path) for path in shard_paths]

    final_dataset = concatenate_datasets(processed_shards_list)

    print(f"Combining complete. Saving final dataset to {final_dataset_path}...")
    final_dataset.save_to_disk(final_dataset_path)

    # --- Cleanup ---
    print("Cleaning up temporary directories...")
    shutil.rmtree(processed_shards_dir)

    print(f"✅✅✅ All done! Your final dataset is ready at: {final_dataset_path}")
