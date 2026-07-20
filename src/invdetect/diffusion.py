from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        selected = values.gather(0, timesteps)
        return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        alpha = self._extract(self.sqrt_alpha_bars, timesteps, clean.shape)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, clean.shape)
        return alpha * clean + sigma * noise, noise


def diffusion_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    clean: torch.Tensor,
) -> torch.Tensor:
    timesteps = torch.randint(
        0, schedule.timesteps, (clean.shape[0],), device=clean.device, dtype=torch.long
    )
    noisy, noise = schedule.add_noise(clean, timesteps)
    predicted_noise = model(noisy, timesteps)
    return F.mse_loss(predicted_noise, noise)


@torch.inference_mode()
def ddim_invert(
    model: nn.Module,
    schedule: DiffusionSchedule,
    clean: torch.Tensor,
) -> torch.Tensor:
    """Deterministically map image patches x_0 to their DDIM noise latents."""
    model.eval()
    latent = clean
    for timestep in range(schedule.timesteps - 1):
        time_batch = torch.full(
            (clean.shape[0],), timestep, device=clean.device, dtype=torch.long
        )
        predicted_noise = model(latent, time_batch)
        alpha_t = schedule.sqrt_alpha_bars[timestep]
        sigma_t = schedule.sqrt_one_minus_alpha_bars[timestep]
        alpha_next = schedule.sqrt_alpha_bars[timestep + 1]
        sigma_next = schedule.sqrt_one_minus_alpha_bars[timestep + 1]
        predicted_clean = (latent - sigma_t * predicted_noise) / alpha_t.clamp_min(1e-12)
        latent = alpha_next * predicted_clean + sigma_next * predicted_noise
    return latent


@torch.inference_mode()
def ddim_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Deterministic DDIM sampling, mainly used for training sanity checks."""
    model.eval()
    sample = noise
    for timestep in reversed(range(schedule.timesteps)):
        time_batch = torch.full(
            (noise.shape[0],), timestep, device=noise.device, dtype=torch.long
        )
        predicted_noise = model(sample, time_batch)
        alpha_t = schedule.sqrt_alpha_bars[timestep]
        sigma_t = schedule.sqrt_one_minus_alpha_bars[timestep]
        predicted_clean = (sample - sigma_t * predicted_noise) / alpha_t.clamp_min(1e-12)
        if timestep == 0:
            sample = predicted_clean
        else:
            sample = (
                schedule.sqrt_alpha_bars[timestep - 1] * predicted_clean
                + schedule.sqrt_one_minus_alpha_bars[timestep - 1] * predicted_noise
            )
    return sample.clamp(-1.0, 1.0)

