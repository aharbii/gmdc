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
NUM_CLASSES = 2
LABEL_MAP = {"Non-mitosis": 0, "Mitosis": 1}

# Output Directories
OUTPUTS_BASE = "outputs"
VISUAL_ANNOTATION_DIR = os.path.join(OUTPUTS_BASE, "visualizations", "polygon_visuals")
PATCHED_DIR = os.path.join(OUTPUTS_BASE, "patches")

# Timestamped Logging
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(OUTPUTS_BASE, "logs", TIMESTAMP)

# MODELS
MODEL_NAMES = [
    "customcnn",
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
