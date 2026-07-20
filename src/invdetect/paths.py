from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def find_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found under: {root}")
    return images


def path_identifier(root: str | Path, path: str | Path) -> str:
    relative = Path(path).relative_to(Path(root)).with_suffix("")
    return "__".join(relative.parts)

