# download_c4.py
from datasets import load_dataset
import os

if __name__ == "__main__":
    # Define a specific cache directory on a drive with ample space
    cache_directory = "../c4_dataset_cache"
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    print(f"Starting download of allenai/c4 to cache directory: {cache_directory}")
    print("This will take a very long time and a lot of disk space...")

    # Load the dataset, which forces it to be downloaded and processed into the cache
    load_dataset(
        "allenai/c4",
        "en",
        cache_dir=cache_directory
    )

    print("✅ C4 dataset download and caching complete.")