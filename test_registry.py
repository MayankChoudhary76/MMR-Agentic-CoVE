# test_registry.py

from src.data.registry import get_dataset_paths

if __name__ == "__main__":
    dataset_name = "beauty"
    paths = get_dataset_paths(dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Raw path: {paths['raw']}")
    print(f"Processed path: {paths['processed']}")