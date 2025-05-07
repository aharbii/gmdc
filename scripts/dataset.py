"""
PyTorch Dataset class for loading preprocessed glioma mitosis classification image patches.

Features:
- Reads image patch paths and labels from a CSV file.
- Applies optional or default image transformations.
- Supports resizing to a custom image size.
- Maps string labels to integer values using a predefined dictionary.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from configs.constants import LABEL_MAP, PATCH_SIZE

from utils.log_utils import get_logger

logger = get_logger(__name__, "dataset.log")


class GliomaPatchDataset(Dataset):
    """
    Dataset for mitosis classification patches.

    Args:
        index_csv (str): Path to the CSV containing filenames and labels.
        patches_dir (str): Directory where patch images are stored.
        transform (callable, optional): Transformations to apply on the images.
        image_size (int): Desired image size for resizing (height, width). Default is {PATCH_SIZE}.
        apply_augmentation (bool): Whether to apply data augmentation. Default is False.
    """

    def __init__(
        self,
        index_csv: str,
        patches_dir: str,
        image_size: int = PATCH_SIZE,
        apply_augmentation: bool = False,
    ) -> None:
        self.data = pd.read_csv(index_csv)
        self.patches_dir = patches_dir
        self.label_map = LABEL_MAP

        base_transform = [T.Resize((image_size, image_size)), T.ToTensor()]
        augment_transform = [
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        if apply_augmentation:
            logger.info("Applying data augmentation")
            self.transform = T.Compose(augment_transform + base_transform)
        else:
            self.transform = T.Compose(base_transform)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the image and label at the specified index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: (transformed image tensor, label)
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.patches_dir, row["filename"])
        label_raw = row["label"]
        try:
            label = int(label_raw)
        except ValueError:
            label = self.label_map.get(label_raw, -1)

        if label < 0 or label >= len(self.label_map):
            raise ValueError(f"Invalid label '{label_raw}' at index {idx}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        image = self.transform(image)

        return image, label

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(samples={len(self)}, dir='{self.patches_dir}')"
        )
