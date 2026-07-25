from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No image patches found in: {directory}")
    return images


def load_patch(path: str | Path, channels: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode), dtype=np.float32)
    if channels == 1:
        array = array[None, :, :]
    else:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(array / 127.5 - 1.0))


class BraTS2021PatchDataset(Dataset):
    """Pre-cut BraTS 2021 brain-tumor patch dataset.

    Training mode reads normal patches directly from ``root``.
    Labeled test mode expects ``root/normal`` and ``root/abnormal``.
    """

    def __init__(self, root: str | Path, channels: int = 1, labeled: bool = False):
        self.channels = channels
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)

        if labeled:
            self.samples.extend((path, 0) for path in list_images(root / "normal"))
            self.samples.extend((path, 1) for path in list_images(root / "abnormal"))
        else:
            self.samples.extend((path, 0) for path in list_images(root))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return load_patch(path, self.channels), path.name, label
