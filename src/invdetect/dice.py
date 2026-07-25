from pathlib import Path

import numpy as np
from PIL import Image

from invdetect.data import list_images


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    total = int(prediction.sum()) + int(target.sum())
    if total == 0:
        return 1.0
    intersection = int(np.logical_and(prediction, target).sum())
    return 2.0 * intersection / total


def evaluate_dice(
    prediction_dir: str | Path, target_dir: str | Path, threshold: int = 127
) -> tuple[list[tuple[str, float]], float]:
    ground_truth = {path.stem: path for path in list_images(target_dir)}
    results = []
    for prediction_path in list_images(prediction_dir):
        target_path = ground_truth.get(prediction_path.stem)
        if target_path is None:
            raise FileNotFoundError(f"Missing ground-truth mask: {prediction_path.name}")
        with Image.open(prediction_path) as image:
            prediction = np.asarray(image.convert("L")) > threshold
        with Image.open(target_path) as image:
            target = np.asarray(image.convert("L")) > threshold
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {prediction_path.name}: "
                f"{prediction.shape} vs {target.shape}"
            )
        results.append((prediction_path.name, dice_score(prediction, target)))
    mean_dice = float(np.mean([score for _, score in results]))
    return results, mean_dice
