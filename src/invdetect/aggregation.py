from __future__ import annotations

import numpy as np


def gaussian_patch_weights(patch_size: int = 32, sigma: float = 8.0) -> np.ndarray:
    """Equation (1): Gaussian proximity to the patch center."""
    axis = np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0
    squared_distance = axis[:, None] ** 2 + axis[None, :] ** 2
    return np.exp(-squared_distance / (2.0 * sigma**2)).astype(np.float32)


def _validate_patch_outputs(
    coordinates: list[tuple[int, int]],
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != len(coordinates):
        raise ValueError(
            f"Expected {len(coordinates)} patch outputs, received {values.shape[0]}."
        )
    return values


def aggregate_anomaly_map(
    image_shape: tuple[int, int],
    coordinates: list[tuple[int, int]],
    patch_scores: np.ndarray,
    patch_size: int = 32,
    sigma: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _validate_patch_outputs(coordinates, np.asarray(patch_scores).reshape(-1))
    height, width = image_shape
    kernel = gaussian_patch_weights(patch_size, sigma)
    weighted_scores = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for (x, y), score in zip(coordinates, scores, strict=True):
        if x < 0 or y < 0 or x + patch_size > width or y + patch_size > height:
            raise ValueError(f"Patch at ({x}, {y}) falls outside image shape {image_shape}.")
        weighted_scores[y : y + patch_size, x : x + patch_size] += kernel * float(score)
        weight_sum[y : y + patch_size, x : x + patch_size] += kernel

    anomaly_map = np.zeros_like(weighted_scores)
    np.divide(weighted_scores, weight_sum, out=anomaly_map, where=weight_sum > 0)
    return anomaly_map, weight_sum


def latent_pairwise_weights(
    image_shape: tuple[int, int],
    coordinates: list[tuple[int, int]],
    patch_latents: np.ndarray,
    coverage: np.ndarray,
    patch_size: int = 32,
    sigma: float = 8.0,
    tau: float = 10.0,
    lambda_smooth: float = 25.0,
    chunk_rows: int = 8,
) -> tuple[np.ndarray, np.ndarray]:

    latents = _validate_patch_outputs(coordinates, patch_latents)
    latents = np.asarray(latents, dtype=np.float32).reshape(latents.shape[0], -1)
    height, width = image_shape
    if coverage.shape != image_shape:
        raise ValueError("Coverage shape must equal image_shape.")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")

    kernel = gaussian_patch_weights(patch_size, sigma)
    horizontal = np.zeros((height, max(width - 1, 0)), dtype=np.float32)
    vertical = np.zeros((max(height - 1, 0), width), dtype=np.float32)
    denominator = 2.0 * tau**2

    for row_start in range(0, height, chunk_rows):
        output_rows = min(chunk_rows, height - row_start)
        block_end = min(height, row_start + output_rows + 1)
        block_height = block_end - row_start
        latent_sum = np.zeros((block_height, width, latents.shape[1]), dtype=np.float32)

        for index, (x, y) in enumerate(coordinates):
            patch_bottom = y + patch_size
            if patch_bottom <= row_start or y >= block_end:
                continue
            overlap_top = max(y, row_start)
            overlap_bottom = min(patch_bottom, block_end)
            block_top = overlap_top - row_start
            block_bottom = overlap_bottom - row_start
            kernel_top = overlap_top - y
            kernel_bottom = overlap_bottom - y
            local_weights = kernel[kernel_top:kernel_bottom, :]
            latent_sum[block_top:block_bottom, x : x + patch_size, :] += (
                local_weights[:, :, None] * latents[index][None, None, :]
            )

        block_coverage = coverage[row_start:block_end]
        aggregated = np.zeros_like(latent_sum)
        np.divide(
            latent_sum,
            block_coverage[:, :, None],
            out=aggregated,
            where=block_coverage[:, :, None] > 0,
        )

        if width > 1:
            delta_h = aggregated[:output_rows, 1:, :] - aggregated[:output_rows, :-1, :]
            distance_h = np.einsum("...d,...d->...", delta_h, delta_h, optimize=True)
            valid_h = (
                block_coverage[:output_rows, 1:] > 0
            ) & (block_coverage[:output_rows, :-1] > 0)
            horizontal[row_start : row_start + output_rows] = (
                lambda_smooth * np.exp(-distance_h / denominator) * valid_h
            ).astype(np.float32)

        vertical_rows = min(output_rows, height - 1 - row_start)
        if vertical_rows > 0:
            delta_v = aggregated[1 : vertical_rows + 1] - aggregated[:vertical_rows]
            distance_v = np.einsum("...d,...d->...", delta_v, delta_v, optimize=True)
            valid_v = (block_coverage[1 : vertical_rows + 1] > 0) & (
                block_coverage[:vertical_rows] > 0
            )
            vertical[row_start : row_start + vertical_rows] = (
                lambda_smooth * np.exp(-distance_v / denominator) * valid_v
            ).astype(np.float32)

    return horizontal, vertical

