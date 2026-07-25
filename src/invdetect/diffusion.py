from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class DiffusionDenoiser(nn.Module):
    """Compact time-conditioned convolutional diffusion model."""

    def __init__(self, channels: int = 1, base_channels: int = 32, timesteps: int = 100):
        super().__init__()
        self.time_embedding = nn.Embedding(timesteps, base_channels)
        self.time_projection = nn.Linear(base_channels, base_channels)
        self.conv1 = nn.Conv2d(channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.output = nn.Conv2d(base_channels, channels, 3, padding=1)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time = F.silu(self.time_projection(self.time_embedding(timesteps)))
        hidden = F.silu(self.conv1(images))
        hidden = F.silu(self.conv2(hidden) + time[:, :, None, None])
        hidden = F.silu(self.conv3(hidden))
        return self.output(hidden)


class DiffusionSchedule:
    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        self.sqrt_alpha_bars = torch.sqrt(torch.cumprod(alphas, dim=0))
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.sqrt_alpha_bars**2)

    @staticmethod
    def _at(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        selected = values.gather(0, timesteps)
        return selected.reshape(len(timesteps), *((1,) * (len(shape) - 1)))

    def add_noise(
        self, clean: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean)
        alpha = self._at(self.sqrt_alpha_bars, timesteps, clean.shape)
        sigma = self._at(self.sqrt_one_minus_alpha_bars, timesteps, clean.shape)
        return alpha * clean + sigma * noise, noise


def diffusion_loss(
    model: nn.Module, schedule: DiffusionSchedule, clean: torch.Tensor
) -> torch.Tensor:
    timesteps = torch.randint(
        0, schedule.timesteps, (len(clean),), device=clean.device, dtype=torch.long
    )
    noisy, noise = schedule.add_noise(clean, timesteps)
    return F.mse_loss(model(noisy, timesteps), noise)


@torch.inference_mode()
def ddim_invert(
    model: nn.Module, schedule: DiffusionSchedule, clean: torch.Tensor
) -> torch.Tensor:
    """Map clean patches to deterministic diffusion noise latents."""
    model.eval()
    latent = clean
    for timestep in range(schedule.timesteps - 1):
        time_batch = torch.full(
            (len(clean),), timestep, device=clean.device, dtype=torch.long
        )
        predicted_noise = model(latent, time_batch)
        predicted_clean = (
            latent - schedule.sqrt_one_minus_alpha_bars[timestep] * predicted_noise
        ) / schedule.sqrt_alpha_bars[timestep].clamp_min(1e-12)
        latent = (
            schedule.sqrt_alpha_bars[timestep + 1] * predicted_clean
            + schedule.sqrt_one_minus_alpha_bars[timestep + 1] * predicted_noise
        )
    return latent


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_diffusion(
    path: str | Path,
    model: DiffusionDenoiser,
    channels: int,
    base_channels: int,
    timesteps: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "channels": channels,
            "base_channels": base_channels,
            "timesteps": timesteps,
        },
        path,
    )


def load_diffusion(
    path: str | Path, device: torch.device
) -> tuple[DiffusionDenoiser, DiffusionSchedule, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DiffusionDenoiser(
        channels=checkpoint["channels"],
        base_channels=checkpoint["base_channels"],
        timesteps=checkpoint["timesteps"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    schedule = DiffusionSchedule(checkpoint["timesteps"], device=device)
    return model, schedule, checkpoint
