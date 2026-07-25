from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load the project YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Must be a mapping.")
    return config
