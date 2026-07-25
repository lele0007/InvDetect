import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from invdetect.data import list_images

PATCH_NAME = re.compile(r"^(?P<image_id>.+)__x=(?P<x>\d+)_y=(?P<y>\d+)$")


def parse_patch_name(filename: str) -> tuple[str, int, int]:
    match = PATCH_NAME.match(Path(filename).stem)
    if match is None:
        raise ValueError(
            f"Invalid patch name '{filename}'. Expected: case001__x=0_y=0.png"
        )
    return match["image_id"], int(match["x"]), int(match["y"])


def _load_predictions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Prediction CSV is empty.")
    return rows


def reconstruct_images(
    patch_dir: str | Path,
    predictions_csv: str | Path,
    output_dir: str | Path,
    threshold: float = 0.0,
) -> None:
    patch_paths = {}
    for path in list_images(patch_dir):
        if path.name in patch_paths:
            raise ValueError(f"Duplicate patch filename: {path.name}")
        patch_paths[path.name] = path

    groups = defaultdict(list)
    for row in _load_predictions(predictions_csv):
        filename = row["filename"]
        image_id, x, y = parse_patch_name(filename)
        if filename not in patch_paths:
            raise FileNotFoundError(f"Patch listed in CSV was not found: {filename}")
        groups[image_id].append(
            (patch_paths[filename], x, y, float(row["anomaly_score"]))
        )

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    score_dir = output_dir / "score_maps"
    for directory in (image_dir, mask_dir, score_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for image_id, records in groups.items():
        with Image.open(records[0][0]) as first:
            mode = "RGB" if first.mode == "RGB" else "L"
            patch_width, patch_height = first.size
        width = max(x + patch_width for _, x, _, _ in records)
        height = max(y + patch_height for _, _, y, _ in records)
        channels = 3 if mode == "RGB" else 1

        image_sum = np.zeros((height, width, channels), dtype=np.float32)
        image_count = np.zeros((height, width, 1), dtype=np.float32)
        score_sum = np.zeros((height, width), dtype=np.float32)
        score_count = np.zeros((height, width), dtype=np.float32)

        for path, x, y, score in records:
            with Image.open(path) as patch_image:
                patch = np.asarray(patch_image.convert(mode), dtype=np.float32)
            if channels == 1:
                patch = patch[:, :, None]
            h, w = patch.shape[:2]
            image_sum[y : y + h, x : x + w] += patch
            image_count[y : y + h, x : x + w] += 1.0
            score_sum[y : y + h, x : x + w] += score
            score_count[y : y + h, x : x + w] += 1.0

        image = np.divide(
            image_sum,
            image_count,
            out=np.zeros_like(image_sum),
            where=image_count > 0,
        )
        score_map = np.divide(
            score_sum,
            score_count,
            out=np.zeros_like(score_sum),
            where=score_count > 0,
        )
        image = np.clip(image, 0, 255).astype(np.uint8)
        if channels == 1:
            image = image[:, :, 0]
        Image.fromarray(image).save(image_dir / f"{image_id}.png")
        Image.fromarray(((score_map > threshold) * 255).astype(np.uint8)).save(
            mask_dir / f"{image_id}.png"
        )
        np.save(score_dir / f"{image_id}.npy", score_map)
