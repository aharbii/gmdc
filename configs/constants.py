"""
Global constants for the Glioma-MDC 2025 challenge project.

Centralized configuration for dataset paths, patch size, and fold count.
"""

import os
from datetime import datetime

# Dataset Configuration
PATCH_SIZE = 64
FOLDS_COUNT = 5
DATASET_PATH = "dataset/Data_122824"
TRAINING_PATH = os.path.join(DATASET_PATH, "Glioma_MDC_2025_training")
TESTING_PATH = os.path.join(DATASET_PATH, "Glioma_MDC_2025_test")
NUM_CLASSES = 2
LABEL_MAP = {"Non-mitosis": 0, "Mitosis": 1}

# Output Directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUTS_DIR = "outputs"
OUTPUTS_BASE = os.path.join(OUTPUTS_DIR, TIMESTAMP)
VISUAL_ANNOTATION_DIR = os.path.join(OUTPUTS_BASE, "annotations")
PATCHED_DIR = os.path.join(OUTPUTS_BASE, "patches")
TRAIN_PATCH_DIR = os.path.join(PATCHED_DIR, "train")
TEST_PATCH_DIR = os.path.join(PATCHED_DIR, "test")
LOG_DIR = os.path.join(OUTPUTS_BASE, "logs")
MODELS_DIR = os.path.join(OUTPUTS_BASE, "models")
FOLDS_DIR = os.path.join(OUTPUTS_BASE, "folds")
SUBMISSION_DIR = os.path.join(OUTPUTS_BASE, "submission")

# MODELS
MODEL_NAMES = [
    "customcnn",
    "customcnn_v2",
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b2",
    "densenet121",
    "mobilenet_v2",
    "convnext_tiny",
    "convnext_small",
    "vgg16_bn",
]

SELECTED_MODEL_NAMES = ["customcnn_v2", "resnet50", "efficientnet_b2"]
