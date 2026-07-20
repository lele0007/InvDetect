from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from invdetect.paths import find_images, path_identifier


def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Return sliding-window positions and force coverage of the final boundary."""
    if length < patch_size:
        raise ValueError(f"Image side {length} is smaller than patch size {patch_size}.")
    positions = list(range(0, length - patch_size + 1, stride))
    final_position = length - patch_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def _image_mode(channels: int) -> str:
    if channels == 1:
        return "L"
    if channels == 3:
        return "RGB"
    raise ValueError("Only one-channel and three-channel images are supported.")


def pil_to_normalized_tensor(image: Image.Image, channels: int) -> torch.Tensor:
    array = np.asarray(image.convert(_image_mode(channels)), dtype=np.float32)
    if channels == 1:
        array = array[None, :, :]
    else:
        array = np.transpose(array, (2, 0, 1))
    normalized = np.ascontiguousarray(array / 127.5 - 1.0)
    return torch.from_numpy(normalized)


@dataclass(frozen=True)
class PatchRecord:
    image_path: Path
    image_id: str
    x: int
    y: int


class ImagePatchDataset(Dataset[dict[str, object]]):
    """Extract overlapped patches from images without writing a patch cache to disk."""

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 32,
        stride: int = 16,
        channels: int = 1,
    ) -> None:
        self.root = Path(root)
        self.patch_size = patch_size
        self.stride = stride
        self.channels = channels
        self.records: list[PatchRecord] = []

        for image_path in find_images(self.root):
            with Image.open(image_path) as image:
                width, height = image.size
            xs = sliding_positions(width, patch_size, stride)
            ys = sliding_positions(height, patch_size, stride)
            image_id = path_identifier(self.root, image_path)
            self.records.extend(
                PatchRecord(image_path=image_path, image_id=image_id, x=x, y=y)
                for y in ys
                for x in xs
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            patch = image.crop(
                (
                    record.x,
                    record.y,
                    record.x + self.patch_size,
                    record.y + self.patch_size,
                )
            )
            tensor = pil_to_normalized_tensor(patch, self.channels)
        return {
            "image": tensor,
            "image_id": record.image_id,
            "x": record.x,
            "y": record.y,
        }


@dataclass(frozen=True)
class ImagePatchBatch:
    patches: torch.Tensor
    coordinates: list[tuple[int, int]]
    image_shape: tuple[int, int]


def extract_image_patches(
    image_path: str | Path,
    patch_size: int,
    stride: int,
    channels: int,
) -> ImagePatchBatch:
    with Image.open(image_path) as image:
        image = image.convert(_image_mode(channels))
        width, height = image.size
        xs = sliding_positions(width, patch_size, stride)
        ys = sliding_positions(height, patch_size, stride)
        coordinates = [(x, y) for y in ys for x in xs]
        patches = [
            pil_to_normalized_tensor(
                image.crop((x, y, x + patch_size, y + patch_size)), channels
            )
            for x, y in coordinates
        ]
    return ImagePatchBatch(
        patches=torch.stack(patches),
        coordinates=coordinates,
        image_shape=(height, width),
    )
