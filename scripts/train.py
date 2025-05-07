"""
Training pipeline for glioma mitosis classification using a custom CNN.
Performs k-fold training and logs performance metrics per fold.

Saves:
- Trained model to outputs/models/
- Training logs to outputs/logs/
"""

"""
Usage:
    python scripts/train.py --fold 0 --patch_dir outputs/patches --index_csv outputs/patches/patches_index.csv --epochs 10 --model CustomCNN
"""


from scripts.model import get_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from scripts.dataset import GliomaPatchDataset
from scripts.validate import validate
from utils.log_utils import get_logger
import os
from configs.constants import (
    FOLDS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    SUBMISSION_DIR,
    TEST_PATCH_DIR,
    TRAIN_PATCH_DIR,
)
import pandas as pd
import argparse
import time

from utils.plot_utils import plot_metric


logger = get_logger(__name__, "train.log")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Train the model for one epoch.

    Returns:
        tuple[float, float, float]: Average loss, accuracy, and F1 score.
    """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            logger.error(f"Invalid labels found: {labels}")
            raise ValueError("Label out of range for CrossEntropyLoss.")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, float]:
    """
    Evaluate the model on a validation or test set.

    Returns:
        tuple[float, float, float]: Average loss, accuracy, and F1 score.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1


def train_fold(
    fold_id: int = -1,
    patch_dir: str = TRAIN_PATCH_DIR,
    fold_sets: dict = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    model_name: str = "customcnn",
) -> None:
    """
    Train and validate the model on a single fold.
    Saves the model and metrics to disk.

    Args:
        fold_id: ID of the fold to train on (0-4)
        patch_dir: Directory containing patch images
        fold_sets: A dictionary mapping each fold ID to a dict containing
                   'train', 'val' and 'test' DataFrames with patch_info and labels.
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        model_name: Model class to be instantiated. Default is "customcnn"
    """

    if fold_sets:
        logger.info(f"Starting training for fold {fold_id}")
        fold_data = fold_sets[fold_id]

        log_dir = os.path.join(FOLDS_DIR, str(fold_id))
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logger.exception(f"Failed to create log directory {log_dir}: {e}")
            raise
        train_csv_path = os.path.join(log_dir, "train.csv")
        val_csv_path = os.path.join(log_dir, "val.csv")
        fold_data["train"].to_csv(train_csv_path, index=False)
        fold_data["val"].to_csv(val_csv_path, index=False)

        train_ds = GliomaPatchDataset(train_csv_path, patch_dir)
        val_ds = GliomaPatchDataset(val_csv_path, patch_dir)
    else:
        logger.info(f"Starting training for the complete dataset")
        log_dir = SUBMISSION_DIR
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logger.exception(f"Failed to create log directory {log_dir}: {e}")
            raise
        train_csv_path = os.path.join(TRAIN_PATCH_DIR, "patches_index.csv")
        val_csv_path = os.path.join(TEST_PATCH_DIR, "patches_index.csv")

        train_ds = GliomaPatchDataset(train_csv_path, TRAIN_PATCH_DIR)
        val_ds = GliomaPatchDataset(val_csv_path, TEST_PATCH_DIR)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = get_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics = []

    for epoch in range(epochs):
        start_time = time.time()
        try:
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
            )
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Val   loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
            )

            metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                }
            )
        except Exception as e:
            logger.exception(f"Error during epoch {epoch+1}: {e}")
            continue
        logger.info(
            f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f}s"
        )

    if fold_sets is None:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.exception(
                f"Failed to create model directory {os.path.dirname(model_path)}: {e}"
            )
            raise

    try:
        metrics_df = pd.DataFrame(metrics)
        metrics_path = os.path.join(log_dir, f"{model_name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved training metrics to {metrics_path}")

        plot_metric(
            metrics_df,
            "loss",
            "Training vs Validation Loss",
            "Loss",
            os.path.join(log_dir, f"{model_name}_loss.png"),
        )
        plot_metric(
            metrics_df,
            "acc",
            "Training vs Validation Accuracy",
            "Accuracy",
            os.path.join(log_dir, f"{model_name}_accuracy.png"),
        )
        plot_metric(
            metrics_df,
            "f1",
            "Training vs Validation F1 Score",
            "F1 Score",
            os.path.join(log_dir, f"{model_name}_f1.png"),
        )
    except Exception as e:
        logger.exception(f"Failed to save model or metrics: {e}")
        raise

    try:
        validate(
            model=model,
            dataloader=val_loader,
            output_dir=log_dir,
            model_name=model_name,
            device=device,
            train_accuracy=train_acc,
        )
    except Exception as e:
        logger.exception(f"Failed to validate model: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument(
        "--patch_dir",
        type=str,
        required=True,
        help="Path to directory with patch images",
    )
    parser.add_argument(
        "--index_csv", type=str, required=True, help="Path to patches_index.csv"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--model",
        type=str,
        default="customcnn",
        help="Model class name to use (e.g., CustomCNN, ResNet18)",
    )

    args = parser.parse_args()

    from scripts.kfold_utils import load_folds

    try:
        fold_sets = load_folds(args.index_csv)
    except Exception as e:
        logger.exception(f"Could not load folds from {args.index_csv}: {e}")
        raise SystemExit(1)

    train_fold(
        fold_id=args.fold,
        patch_dir=args.patch_dir,
        fold_sets=fold_sets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model,
    )
