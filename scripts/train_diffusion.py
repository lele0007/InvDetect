import argparse

import torch
from torch.utils.data import DataLoader

from invdetect.config import load_config
from invdetect.data import BraTS2021PatchDataset
from invdetect.diffusion import (
    DiffusionDenoiser,
    DiffusionSchedule,
    choose_device,
    diffusion_loss,
    save_diffusion,
)


def main():
    parser = argparse.ArgumentParser(description="Train.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    torch.manual_seed(config["seed"])
    device = choose_device(config["runtime"]["device"])
    data_config = config["data"]
    model_config = config["diffusion"]
    dataset = BraTS2021PatchDataset(
        config["paths"]["train_dir"], channels=data_config["channels"]
    )
    loader = DataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        num_workers=config["runtime"]["num_workers"],
    )
    model = DiffusionDenoiser(
        channels=data_config["channels"],
        base_channels=model_config["base_channels"],
        timesteps=model_config["timesteps"],
    ).to(device)
    schedule = DiffusionSchedule(model_config["timesteps"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])

    print(f"device={device}, patches={len(dataset)}")
    for epoch in range(1, model_config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for images, _, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(model, schedule, images)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
        print(f"epoch={epoch}, loss={total_loss / len(dataset):.6f}")

    save_diffusion(
        config["paths"]["diffusion_checkpoint"],
        model,
        channels=data_config["channels"],
        base_channels=model_config["base_channels"],
        timesteps=model_config["timesteps"],
    )
    print(f"saved: {config['paths']['diffusion_checkpoint']}")


if __name__ == "__main__":
    main()
