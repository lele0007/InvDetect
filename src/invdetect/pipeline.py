from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from invdetect.aggregation import aggregate_anomaly_map, latent_pairwise_weights
from invdetect.config import InvDetectConfig
from invdetect.data import extract_image_patches
from invdetect.diffusion import DiffusionSchedule, ddim_invert
from invdetect.normality import NoiseLatentNormalityModel
from invdetect.scr import spatial_contiguity_refinement


@dataclass(frozen=True)
class DetectionResult:
    anomaly_map: np.ndarray
    mask: np.ndarray
    patch_scores: np.ndarray
    patch_coordinates: list[tuple[int, int]]


class InvDetectPipeline:
    def __init__(
        self,
        model: nn.Module,
        normality_model: NoiseLatentNormalityModel,
        config: InvDetectConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.normality_model = normality_model
        self.config = config
        self.device = device
        self.schedule = DiffusionSchedule(**config.diffusion.__dict__).to(device)

    @torch.inference_mode()
    def _invert_batches(self, patches: torch.Tensor, batch_size: int) -> np.ndarray:
        latent_batches: list[np.ndarray] = []
        for start in range(0, len(patches), batch_size):
            batch = patches[start : start + batch_size].to(self.device, non_blocking=True)
            latents = ddim_invert(self.model, self.schedule, batch)
            latent_batches.append(latents.cpu().numpy())
        return np.concatenate(latent_batches, axis=0)

    def detect(
        self,
        image_path: str | Path,
        batch_size: int = 32,
        use_scr: bool = True,
    ) -> DetectionResult:
        patches = extract_image_patches(
            image_path=image_path,
            patch_size=self.config.patch.size,
            stride=self.config.patch.stride,
            channels=self.config.model.input_channels,
        )
        latents = self._invert_batches(patches.patches, batch_size)
        patch_scores = self.normality_model.anomaly_score(latents)
        anomaly_map, coverage = aggregate_anomaly_map(
            image_shape=patches.image_shape,
            coordinates=patches.coordinates,
            patch_scores=patch_scores,
            patch_size=self.config.patch.size,
            sigma=self.config.patch.sigma,
        )

        if use_scr:
            horizontal, vertical = latent_pairwise_weights(
                image_shape=patches.image_shape,
                coordinates=patches.coordinates,
                patch_latents=latents,
                coverage=coverage,
                patch_size=self.config.patch.size,
                sigma=self.config.patch.sigma,
                tau=self.config.scr.tau,
                lambda_smooth=self.config.scr.lambda_smooth,
                chunk_rows=self.config.scr.latent_chunk_rows,
            )
            mask = spatial_contiguity_refinement(anomaly_map, horizontal, vertical)
        else:
            # Paper ablation: threshold the boundary-aligned anomaly score at eta=0.
            mask = (anomaly_map > 0.0).astype(np.uint8)

        return DetectionResult(
            anomaly_map=anomaly_map,
            mask=mask,
            patch_scores=patch_scores,
            patch_coordinates=patches.coordinates,
        )

