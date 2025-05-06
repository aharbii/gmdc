"""
Extracts fixed-size image patches centered on annotated polygons from histopathology images.

For each polygon in the JSON annotations, a patch is extracted using the polygon centroid.
The extracted patches are saved to disk, and a CSV index of filename-label pairs is created.
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from utils.log_utils import get_logger

logger = get_logger(__name__, "patch_extractor.log")


def load_json(json_path: Path) -> dict:
    """
    Load a single JSON annotation file.

    Args:
        json_path (Path): Path to the annotation file.

    Returns:
        dict: Parsed JSON content.
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Failed to load JSON file {json_path}: {e}")
        raise


def get_centroid(points: np.ndarray) -> tuple[int, int]:
    """
    Compute the centroid of a polygon.

    Args:
        points (np.ndarray): Polygon points as (N, 2) array.

    Returns:
        tuple[int, int]: (x, y) coordinates of the centroid.
    """
    points = np.array(points, dtype=np.float32)
    centroid = np.mean(points, axis=0)
    return int(round(centroid[0])), int(round(centroid[1]))


def extract_patch(image: np.ndarray, center_x: int, center_y: int, size: int) -> np.ndarray:
    """
    Extract a fixed-size patch centered on (center_x, center_y).

    Args:
        image (np.ndarray): Input image.
        center_x (int): X coordinate of patch center.
        center_y (int): Y coordinate of patch center.
        size (int): Patch size in pixels.

    Returns:
        np.ndarray: Extracted image patch.
    """
    h, w = image.shape[:2]
    half = size // 2
    x1, y1 = center_x - half, center_y - half
    x2, y2 = center_x + half, center_y + half

    pad_x1 = max(0, -x1)
    pad_y1 = max(0, -y1)
    pad_x2 = max(0, x2 - w)
    pad_y2 = max(0, y2 - h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    patch = image[y1:y2, x1:x2]
    if pad_x1 or pad_y1 or pad_x2 or pad_y2:
        patch = cv2.copyMakeBorder(
            patch, pad_y1, pad_y2, pad_x1, pad_x2, borderType=cv2.BORDER_CONSTANT, value=0)
    return patch


def process_dataset(image_dir: str, output_dir: str, patch_size: int = 64) -> None:
    """
    Process a dataset of images and annotations to extract centered patches.

    Args:
        image_dir (str): Path to directory containing images and JSON annotations.
        output_dir (str): Path to output directory for patches and CSV index.
        patch_size (int): Size of the square patch to extract. Defaults to 64.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ["filename,label"]

    json_paths = sorted(image_dir.glob("*.json"))
    for json_path in tqdm(json_paths, desc="Extracting patches"):
        try:
            annotation = load_json(json_path)
            image_name = annotation.get(
                "imagePath", json_path.stem + ".jpg").replace("\\", "/").split("/")[-1]
            image_path = image_dir / image_name

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Image not found: {image_path}")
                continue

            img_h, img_w = image.shape[:2]
            ann_h = annotation.get("imageHeight", img_h)
            ann_w = annotation.get("imageWidth", img_w)

            scale_x = img_w / ann_w
            scale_y = img_h / ann_h

            for i, shape in enumerate(annotation.get("shapes", [])):
                label = shape.get("label", "").lower()
                label_int = 1 if label == "mitosis" else 0

                raw_points = np.array(
                    shape.get("points", []), dtype=np.float32)
                raw_points *= [scale_x, scale_y]

                cx, cy = get_centroid(raw_points)
                patch = extract_patch(image, cx, cy, patch_size)

                patch_filename = f"{json_path.stem}_{i}.jpg"
                patch_path = output_dir / patch_filename
                cv2.imwrite(str(patch_path), patch)
                index_lines.append(f"{patch_filename},{label_int}")

        except Exception as e:
            logger.exception(f"Failed to process {json_path}: {e}")

    csv_path = output_dir / "patches_index.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(index_lines))
    logger.info(f"Saved patch index to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract image patches centered on polygon centroids.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to the image and JSON annotation directory.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted patches and index CSV.")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Size of each square patch.")
    args = parser.parse_args()

    process_dataset(args.image_dir, args.output_dir, args.patch_size)
