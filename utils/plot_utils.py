import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from configs.constants import SUBMISSION_DIR
from utils.log_utils import get_logger

logger = get_logger(__name__, "plot.log")

sns.set(style="whitegrid")


def plot_metric(
    metrics_df: pd.DataFrame,
    metric_name: str,
    title: str,
    ylabel: str,
    file_path: str,
) -> None:
    """
    Plot a training and validation metric over epochs and save the plot as a PNG file.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing epoch, train_metric, and val_metric columns.
        metric_name (str): The metric to plot (e.g., 'accuracy', 'f1_score', 'loss').
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        file_path (str): Path to save the plot.

    Returns:
        None

    Raises:
        KeyError: If the specified metric is not found in the DataFrame.
        Exception: If an error occurs during plotting or saving the file.
    """
    try:
        plt.figure()
        plt.plot(metrics_df["epoch"], metrics_df[f"train_{metric_name}"], label="Train")
        plt.plot(
            metrics_df["epoch"], metrics_df[f"val_{metric_name}"], label="Validation"
        )
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()
        logger.info(f"Saved plot for {metric_name} to {file_path}")
    except KeyError as e:
        logger.error(f"Metric {metric_name} not found in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in plot_metric function: {e}")
        raise


def plot_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold_dir: str,
    model_name: str,
) -> None:
    """
    Generate and save ROC and Precision-Recall curves for the model's predictions.

    Parameters:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        fold_dir (str): Directory to save the plots.
        model_name (str): Name of the model (used in plot titles and filenames).
    """
    logger.info(f"Generating ROC and PR curves for model {model_name} in {fold_dir}.")
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()
        roc_path = os.path.join(fold_dir, f"{model_name}_roc.png")
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"Saved ROC curve to {roc_path}")
    except Exception as e:
        logger.error(f"Failed to generate/save ROC curve: {e}")

    try:
        prec, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(recall, prec, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} PR Curve")
        plt.legend()
        pr_path = os.path.join(fold_dir, f"{model_name}_pr.png")
        plt.savefig(pr_path)
        plt.close()
        logger.info(f"Saved PR curve to {pr_path}")
    except Exception as e:
        logger.error(f"Failed to generate/save PR curve: {e}")


def plot_val_metrics_json_comparison(folds_dir: str, fold: int):
    """
    Compare validation metrics from JSON files for a specific fold.

    Args:
        folds_dir (str): Path to the parent folds directory.
        fold (int): Fold number to analyze.

    Saves:
        A bar chart comparing accuracy, precision, recall, f1, roc_auc, pr_auc across models.
    """
    fold_path = os.path.join(folds_dir, str(fold))
    if not os.path.isdir(fold_path):
        logger.info(f"Fold directory not found: {fold_path}")
        return

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "train_accuracy",
    ]
    data = []
    acc = []
    confusion_data = []

    for file in os.listdir(fold_path):
        if file.endswith("_val_metrics.json"):
            model_name = file.replace("_val_metrics.json", "")
            file_path = os.path.join(fold_path, file)
            try:
                with open(file_path, "r") as f:
                    content = json.load(f)
                    for metric in metrics:
                        if metric in content:
                            data.append(
                                {
                                    "Model": model_name,
                                    "Metric": metric,
                                    "Score": content[metric],
                                }
                            )
                            if metric == "accuracy" or metric == "train_accuracy":
                                acc.append(
                                    {
                                        "Model": model_name,
                                        "Metric": metric,
                                        "Score": content[metric],
                                    }
                                )
                    if "confusion_matrix" in content:
                        cm_df = pd.DataFrame(content["confusion_matrix"])
                        confusion_data.append((model_name, cm_df))
            except Exception as e:
                logger.info(f"Error reading {file_path}: {e}")
                raise

    if not data:
        logger.warning(f"No metrics found in fold {fold}")
        return

    try:
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
    except Exception as e:
        logger.exception(f"Failed to create log directory {SUBMISSION_DIR}: {e}")
        raise

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title(f"Fold {fold} - Validation Metrics Comparison per Model")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.tight_layout()

    output_path = os.path.join(
        SUBMISSION_DIR, f"val_metrics_detailed_comparison_{fold}.png"
    )
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved comparison plot: {output_path}")

    acc_df = pd.DataFrame(acc)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=acc_df, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title(f"Fold {fold} - Training vs Validation Accuracy")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.tight_layout()

    output_path = os.path.join(SUBMISSION_DIR, f"train_val_acc_{fold}.png")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved comparison plot: {output_path}")

    if confusion_data:
        fig, axes = plt.subplots(
            1, len(confusion_data), figsize=(6 * len(confusion_data), 5)
        )
        if len(confusion_data) == 1:
            axes = [axes]
        for ax, (model_name, cm_df) in zip(axes, confusion_data):
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"{model_name} - Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        plt.tight_layout()
        cm_plot_path = os.path.join(SUBMISSION_DIR, f"confusion_matrices_{fold}.png")
        plt.savefig(cm_plot_path)
        plt.close()
        logger.info(f"Saved confusion matrices plot: {cm_plot_path}")


def plot_patches_distribution(patches_data: dict, output_path: str) -> None:
    """
    Plot the distribution of patches for each class based on the data generated in patch_extractor.

    Args:
        patches_data (dict): Dictionary containing class names as keys and patch counts as values.
        output_path (str): Path to save the output plot.

    Returns:
        None

    Raises:
        Exception: If an error occurs during plotting or saving the file.
    """

    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        logger.exception(f"Failed to create log directory {output_path}: {e}")
        raise

    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=list(patches_data.keys()), y=list(patches_data.values()), palette="Set2"
        )
        plt.title("Patches Distribution per Class")
        plt.xlabel("Class")
        plt.ylabel("Number of Patches")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "patches_distribution.png"))
        plt.close()
        logger.info(f"Saved patches distribution plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot patches distribution: {e}")
        raise
