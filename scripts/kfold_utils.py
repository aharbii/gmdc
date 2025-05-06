"""
Utility to generate stratified 5-fold splits for classification tasks.
Each fold uses a rotating 80-10-10 train-val-test split strategy.

Outputs:
- A CSV with fold assignments
- A dictionary containing DataFrames for each fold
"""

from sklearn.model_selection import StratifiedKFold
import pandas as pd
from utils.log_utils import get_logger

logger = get_logger(__name__, "kfold_utils.log")


def generate_stratified_folds(index_csv_path: str, n_splits: int = 5, seed: int = 42, output_csv: str = None) -> tuple[dict, pd.DataFrame]:
    """
    Generates stratified k-fold splits and assigns fold indices.

    Args:
        index_csv_path (str): Path to the CSV file with image filenames and labels.
        n_splits (int): Number of folds. Default is 5.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        dict: A dictionary with fold indices mapping to train/val/test DataFrames.
    """
    try:
        df = pd.read_csv(index_csv_path)
    except Exception as e:
        logger.exception(f"Failed to read index CSV: {e}")
        raise

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    logger.info(f"Assigned fold indices using stratified split.")

    fold_sets = {}
    for current_fold in range(n_splits):
        test_fold = current_fold
        val_fold = (current_fold + 1) % n_splits
        train_folds = [f for f in range(n_splits) if f not in [
            test_fold, val_fold]]

        train_df = df[df.fold.isin(train_folds)].copy()
        val_df = df[df.fold == val_fold].copy()
        test_df = df[df.fold == test_fold].copy()

        fold_sets[current_fold] = {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }

        logger.info(
            f"Fold {current_fold}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        logger.info(
            f"Fold {current_fold}: Test={test_fold}, Val={val_fold}, Train={train_folds}")

    fold_csv_path = output_csv or index_csv_path.replace(".csv", "_folds.csv")
    try:
        df.to_csv(fold_csv_path, index=False)
        logger.info(f"Saved fold assignments to {fold_csv_path}")
    except Exception as e:
        logger.exception(f"Failed to save fold CSV: {e}")
        raise

    return fold_sets, df
