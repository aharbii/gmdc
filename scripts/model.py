"""
Defines a custom convolutional neural network (CNN) for classifying mitotic figures
in glioma patches. This model serves as a baseline architecture using a 3-layer CNN.

Includes logging and basic error handling for forward pass and instantiation.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from configs.constants import MODEL_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomCNN(nn.Module):
    """
    A simple CNN baseline for mitosis classification using RGB patches of configurable size.
    Outputs logits for classification into a specified number of classes.

    Attributes:
        features (nn.Sequential): Feature extraction layers using convolutional blocks.
        classifier (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, input_size: int = 64, num_classes: int = NUM_CLASSES) -> None:
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flattened_dim = 128 * (input_size // 8) * (input_size // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, input_size, input_size)

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes)
        """
        try:
            x = self.features(x)
            x = self.classifier(x)
            return x
        except Exception as e:
            logger.error(f"Error during forward pass: {e}", exc_info=True)
            raise

    def __repr__(self):
        return f"{self.__class__.__name__}(input=64x64x3, output={self.classifier[-1].out_features})"


def get_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Retrieves a classification model by name with its final layer adjusted
    for the specified number of classes.

    Args:
        model_name (str): The name of the model architecture to instantiate.
        num_classes (int): The number of output classes for the classification task. Default is NUM_CLASSES
        pretrained (bool): Whether to load pretrained weights. Default is True.
        freeze_backbone (bool): Whether to freeze the backbone feature extractor layers when pretrained is True.

    Returns:
        nn.Module: Instantiated model ready for training or evaluation.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    try:
        if model_name not in MODEL_NAMES:
            logger.error(f"Unsupported model name requested: {model_name}")
            raise ValueError(f"Unsupported model_name: {model_name}")

        logger.info(f"Loading model: {model_name} with pretrained={pretrained}")

        if model_name == "customcnn":
            model = CustomCNN(num_classes=num_classes)

        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )

        elif model_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b2(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )

        elif model_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        elif model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )

        elif model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier[2] = nn.Linear(
                model.classifier[2].in_features, num_classes
            )

        elif model_name == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = models.convnext_small(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier[2] = nn.Linear(
                model.classifier[2].in_features, num_classes
            )

        elif model_name == "vgg16_bn":
            weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
            model = models.vgg16_bn(weights=weights)
            if freeze_backbone and pretrained:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier[6] = nn.Linear(
                model.classifier[6].in_features, num_classes
            )

        logger.info(
            f"Model {model_name} loaded with pretrained={pretrained}, freeze_backbone={freeze_backbone}"
        )
        return model
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info(f"Instantiating {CustomCNN.__name__} for test run.")
        model = CustomCNN(input_size=64)
        dummy = torch.randn(4, 3, 64, 64)
        out = model(dummy)
        logger.info(f"Output shape: {out.shape}")
    except Exception as e:
        logger.exception(f"Failed to execute model test run: {e}")
