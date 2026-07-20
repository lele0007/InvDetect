from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from invdetect.paths import find_images, path_identifier


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def precision_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    predicted_positive = int(prediction.sum())
    if predicted_positive == 0:
        return 1.0 if not target.any() else 0.0
    return float(np.logical_and(prediction, target).sum()) / predicted_positive


def evaluate_directories(
    prediction_dir: str | Path,
    target_dir: str | Path,
    threshold: int = 127,
) -> dict[str, object]:
    prediction_dir = Path(prediction_dir)
    target_dir = Path(target_dir)
    targets = {
        path_identifier(target_dir, path): path for path in find_images(target_dir)
    }
    rows: list[dict[str, float | str]] = []
    for prediction_path in find_images(prediction_dir):
        identifier = path_identifier(prediction_dir, prediction_path)
        target_path = targets.get(identifier)
        if target_path is None:
            raise FileNotFoundError(f"No ground-truth mask found for {identifier}.")
        prediction = np.asarray(Image.open(prediction_path).convert("L")) > threshold
        target = np.asarray(Image.open(target_path).convert("L")) > threshold
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {identifier}: {prediction.shape} vs {target.shape}"
            )
        rows.append(
            {
                "image": identifier,
                "dice": dice_score(prediction, target),
                "precision": precision_score(prediction, target),
            }
        )
    if not rows:
        raise ValueError("No prediction masks were evaluated.")
    return {
        "images": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "per_image": rows,
    }
