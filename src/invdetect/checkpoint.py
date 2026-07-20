from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from invdetect.config import InvDetectConfig
from invdetect.model import TimeConditionedUNet


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: InvDetectConfig,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": epoch,
        "config": config.to_dict(),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def _strip_data_parallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_model(
    checkpoint_path: str | Path,
    config: InvDetectConfig,
    device: torch.device,
) -> tuple[TimeConditionedUNet, dict[str, Any]]:
    model = TimeConditionedUNet(**config.model.__dict__).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
        metadata = {key: value for key, value in payload.items() if key != "model_state"}
    elif isinstance(payload, dict) and all(isinstance(key, str) for key in payload):
        state_dict = payload
        metadata = {"format_version": 0, "source": "raw_state_dict"}
    state_dict = _strip_data_parallel_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata
