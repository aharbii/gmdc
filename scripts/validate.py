"""
Validation script for evaluating trained models on held-out folds
in the Glioma-MDC 2025 challenge. This module loads trained model
weights and evaluates their performance on validation datasets.

Usage:
    Run this script to evaluate a specified model across all validation folds.
    It computes classification metrics, generates ROC and PR curves, and saves
    the results for each fold.

Functions:
    - validate_model: Runs inference on the validation dataset and collects predictions.
    - compute_metrics: Calculates performance metrics from predictions.
    - plot_curves: Generates and saves ROC and PR curves.
    - evaluate_model_across_folds: Coordinates evaluation across all folds.
"""

import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Any
from configs.constants import FOLDS_DIR, NUM_CLASSES
from scripts.dataset import GliomaPatchDataset
from scripts.model import get_model
from utils.log_utils import get_logger
from utils.plot_utils import plot_curves

logger = get_logger(__name__, "validate.log")


def validate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the validation dataset and collect true labels, predicted labels, and predicted probabilities.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of true labels, predicted labels, and predicted probabilities for the positive class.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    logger.info("Starting validation loop.")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())
    logger.info("Validation loop completed.")

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute classification metrics given true labels, predicted labels, and predicted probabilities.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        Dict[str, Any]: Dictionary containing accuracy, precision, recall, f1_score, roc_auc, pr_auc, and confusion matrix.
    """
    logger.info("Computing metrics.")
    metrics = {}
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["f1_score"] = f1_score(y_true, y_pred)
    except Exception as e:
        logger.error(f"Error computing basic classification metrics: {e}")
        metrics["accuracy"] = None
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["f1_score"] = None

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logger.error(f"Error computing ROC AUC: {e}")
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception as e:
        logger.error(f"Error computing PR AUC: {e}")
        metrics["pr_auc"] = None

    try:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    except Exception as e:
        logger.error(f"Error computing confusion matrix: {e}")
        metrics["confusion_matrix"] = None

    logger.info("Metrics computation completed.")
    return metrics


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: str,
    model_name: str,
    device: torch.device,
    train_accuracy: float = None,
) -> Dict[str, Any]:
    """
    Validate a model on a single fold or the complete test, compute metrics, generate plots, and save the results.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for the validation dataset.
        output_dir (str): Directory to save metrics and plots.
        model_name (str): Name of the model (used in filenames and plots).
        device (torch.device): Device to run the model on.
        training_accuracy: Training accuracy value to be used for metrics visualization.

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics for the model.
    """
    logger.info(
        f"Starting validation for model '{model_name}' in directory '{output_dir}'."
    )
    try:
        y_true, y_pred, y_prob = validate_model(model, dataloader, device)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["train_accuracy"] = train_accuracy
        plot_curves(y_true, y_prob, output_dir, model_name)
        metrics_path = os.path.join(output_dir, f"{model_name}_val_metrics.json")
        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved validation metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics to {metrics_path}: {e}")
        return metrics
    except Exception as e:
        logger.error(
            f"Validation failed for model '{model_name}' in '{output_dir}': {e}"
        )
        return {}
