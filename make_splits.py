from utils import load_dataset, create_and_save_splits

DATASET_NAME = "madelon"  # Cambia por tu dataset
X, y, _ = load_dataset(DATASET_NAME)
create_and_save_splits(X, y, n_repeats=3, n_folds=5, seed=42, file_path=f"splits_{DATASET_NAME}.json")