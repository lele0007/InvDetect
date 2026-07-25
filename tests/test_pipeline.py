import csv

import numpy as np
import torch
from PIL import Image

from invdetect.dice import dice_score
from invdetect.diffusion import (
    DiffusionDenoiser,
    DiffusionSchedule,
    ddim_invert,
    diffusion_loss,
)
from invdetect.reconstruction import reconstruct_images


def test_small_diffusion_forward_and_inversion():
    model = DiffusionDenoiser(channels=1, base_channels=8, timesteps=4)
    schedule = DiffusionSchedule(timesteps=4)
    images = torch.randn(2, 1, 8, 8)
    loss = diffusion_loss(model, schedule, images)
    loss.backward()
    latents = ddim_invert(model, schedule, images)
    assert torch.isfinite(loss)
    assert latents.shape == images.shape
    assert torch.isfinite(latents).all()


def test_reconstruction_and_dice(tmp_path):
    patch_dir = tmp_path / "patches" / "normal"
    patch_dir.mkdir(parents=True)
    Image.fromarray(np.full((4, 4), 50, dtype=np.uint8)).save(
        patch_dir / "case1__x=0_y=0.png"
    )
    Image.fromarray(np.full((4, 4), 100, dtype=np.uint8)).save(
        patch_dir / "case1__x=4_y=0.png"
    )

    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "true_label", "predicted_label", "anomaly_score"])
        writer.writerow(["case1__x=0_y=0.png", 0, 0, -1.0])
        writer.writerow(["case1__x=4_y=0.png", 1, 1, 1.0])

    output_dir = tmp_path / "output"
    reconstruct_images(tmp_path / "patches", predictions, output_dir)
    mask = np.asarray(Image.open(output_dir / "masks" / "case1.png")) > 127
    expected = np.zeros((4, 8), dtype=bool)
    expected[:, 4:] = True
    assert dice_score(mask, expected) == 1.0
