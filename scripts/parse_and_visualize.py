"""
Parses annotation JSON files and overlays labeled polygons on histopathological images.

Outputs visualizations with drawn polygon annotations for mitosis and non-mitosis cells.
Supports coordinate scaling if annotation and image resolutions differ.
"""

import os
import json
import cv2
import numpy as np
from utils.log_utils import get_logger

DEFAULT_IMAGE_SIZE = (512, 512)

logger = get_logger(__name__, "parse_and_visualize.log")


def load_json(json_path: str) -> dict:
    """
    Load a single annotation JSON file.

    Args:
        json_path (str): Path to the annotation JSON file.

    Returns:
        dict: Parsed annotation data.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_path}: {e}")
        raise
    return data


def draw_polygons_on_image(
    image_path: str, annotation: dict, output_path: str = None
) -> np.ndarray:
    """
    Draws labeled polygons on the image and saves/returns the result.

    Args:
        image_path (str): Path to the image file.
        annotation (dict): Annotation dictionary from the corresponding JSON file.
        output_path (str, optional): Path to save the output image. Defaults to None.

    Returns:
        np.ndarray: Image with drawn polygons (if successful), else None.
    """

    image = cv2.imread(image_path)

    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return

    try:
        for shape in annotation.get("shapes", []):
            label = shape.get("label")
            points = shape.get("points", [])
            points = np.array(points, dtype=np.float32)

            if shape.get("shape_type", "polygon") != "polygon" or len(points) < 3:
                logger.warning(f"Skipping invalid polygon in {image_path}")
                continue

            ann_h, ann_w = annotation.get(
                "imageHeight", DEFAULT_IMAGE_SIZE[0]
            ), annotation.get("imageWidth", DEFAULT_IMAGE_SIZE[1])
            img_h, img_w = image.shape[:2]

            if (ann_h != img_h) or (ann_w != img_w):
                scale_x = img_w / ann_w
                scale_y = img_h / ann_h
                points *= [scale_x, scale_y]

            points = np.round(points).astype(np.int32)
            points = points.reshape((-1, 1, 2))

            color = (0, 255, 0) if label.lower() == "mitosis" else (0, 0, 255)
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            centroid = np.mean(points[:, 0, :], axis=0).astype(int)
            cv2.putText(
                image, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
    except Exception as e:
        logger.exception(f"Failed to draw polygons for {image_path}: {e}")
        return

    if output_path:
        cv2.imwrite(output_path, image)
    return image


def visualize_all_annotations(image_dir: str, json_dir: str, output_dir: str) -> None:
    """
    Iterate through images and draw annotation polygons on them.

    Args:
        image_dir (str): Directory containing image files.
        json_dir (str): Directory containing JSON annotation files.
        output_dir (str): Directory to save the annotated visualizations.
    """

    logger.info(f"Drawing polygons on images...")
    image_files = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    logger.info(f"Found {len(image_files)} images and {len(json_files)} JSON files")

    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        json_path = os.path.join(json_dir, image_id + ".json")

        if not os.path.exists(json_path):
            logger.warning(f"JSON file missing for image: {image_file}")
            continue

        try:
            annotation = load_json(json_path)
        except Exception as e:
            logger.exception(f"Failed to load JSON file {json_path}: {e}")
            continue

        try:
            output_path = os.path.join(output_dir, f"vis_{image_id}.jpg")
            draw_polygons_on_image(image_path, annotation, output_path=output_path)
        except Exception as e:
            logger.exception(f"Error processing {image_file}: {e}")
