from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.svm import OneClassSVM


def flatten_latents(latents: np.ndarray) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim < 2:
        raise ValueError("Expected a batch of latents with at least two dimensions.")
    return latents.reshape(latents.shape[0], -1)


class NoiseLatentNormalityModel:
    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str | float = "scale",
    ) -> None:
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.estimator = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.latent_dimension: int | None = None

    def fit(self, latents: np.ndarray) -> NoiseLatentNormalityModel:
        features = flatten_latents(latents)
        self.estimator.fit(features)
        self.latent_dimension = features.shape[1]
        return self

    def anomaly_score(self, latents: np.ndarray) -> np.ndarray:
        features = flatten_latents(latents)
        if self.latent_dimension is None:
            raise RuntimeError("The normality model has not been fitted.")
        if features.shape[1] != self.latent_dimension:
            raise ValueError(
                f"Expected latent dimension {self.latent_dimension}, got {features.shape[1]}."
            )
        return -self.estimator.decision_function(features).reshape(-1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": 1,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "latent_dimension": self.latent_dimension,
            "estimator": self.estimator,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> NoiseLatentNormalityModel:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported normality model file.")
        instance = cls(nu=payload["nu"], kernel=payload["kernel"], gamma=payload["gamma"])
        instance.estimator = payload["estimator"]
        instance.latent_dimension = payload["latent_dimension"]
        return instance

