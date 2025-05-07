"""
Main pipeline script for the Glioma-MDC 2025 challenge.

Steps:
1. Visualize annotated polygons from JSON files.
2. Extract image patches centered on polygon centroids.
3. Perform 5-fold stratified cross-validation.
4. Train a CNN model on each fold and log metrics.
"""

import os
from scripts.kfold_utils import generate_stratified_folds
from scripts.parse_and_visualize import visualize_all_annotations
from scripts.patch_extractor import process_dataset
from scripts.train import train_fold
from utils.log_utils import LOG_DIR, get_logger
from configs.constants import (
    FOLDS_DIR,
    MODEL_NAMES,
    PATCH_SIZE,
    FOLDS_COUNT,
    SELECTED_MODEL_NAMES,
    TEST_PATCH_DIR,
    TESTING_PATH,
    TRAIN_PATCH_DIR,
    TRAINING_PATH,
    VISUAL_ANNOTATION_DIR,
    PATCHED_DIR,
)
from utils.plot_utils import plot_val_metrics_json_comparison

logger = get_logger("main", "main.log")
logger.info(f"Log directory created: {LOG_DIR}")


def main():
    # try:
    #     os.makedirs(VISUAL_ANNOTATION_DIR, exist_ok=True)
    #     logger.info(f"Using training dataset from: {TRAINING_PATH}")
    #     logger.info("Starting visualization of training annotations.")
    #     visualize_all_annotations(TRAINING_PATH, TRAINING_PATH, VISUAL_ANNOTATION_DIR)
    #     logger.info("Completed annotation visualization.")
    # except Exception as e:
    #     logger.exception("Failed during annotation visualization.")
    #     raise

    try:
        os.makedirs(TRAIN_PATCH_DIR, exist_ok=True)
        logger.info("Extracting patches...")
        process_dataset(TRAINING_PATH, TRAIN_PATCH_DIR, PATCH_SIZE)
        logger.info("Completed patches extraction")
    except Exception as e:
        logger.exception("Failed during patch extraction.")
        raise

    try:
        logger.info("Generating 5-fold cross validation...")
        PATCH_INDEX_PATH = os.path.join(TRAIN_PATCH_DIR, "patches_index.csv")
        fold_sets, df = generate_stratified_folds(
            PATCH_INDEX_PATH, n_splits=FOLDS_COUNT, seed=42
        )
        logger.info(f"Generated {FOLDS_COUNT}-folds cross validation")
    except Exception as e:
        logger.exception("Failed during fold generation.")
        raise

    try:
        logger.info("Training models...")
        for model_name in SELECTED_MODEL_NAMES:
            for fold_id in range(FOLDS_COUNT):
                logger.info(
                    f"Starting training for Fold {fold_id} (Fold {fold_id+1} of {FOLDS_COUNT})"
                )
                train_fold(
                    fold_id=fold_id,
                    patch_dir=TRAIN_PATCH_DIR,
                    fold_sets=fold_sets,
                    epochs=10,
                    batch_size=32,
                    lr=1e-3,
                    model_name=model_name,
                )
                logger.info(
                    f"Completed training for Fold {fold_id} (Fold {fold_id+1} of {FOLDS_COUNT})"
                )
                plot_val_metrics_json_comparison(FOLDS_DIR, fold_id)
    except Exception as e:
        logger.exception("Failed during model training.")
        raise

    logger.info("Pipeline completed successfully. All steps executed.")


if __name__ == "__main__":
    main()
