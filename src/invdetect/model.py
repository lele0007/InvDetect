from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Time embedding retained from the user-provided, working U-Net."""

    def __init__(self, embedding_dim: int, frequency_scale: float = 5000.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.frequency_scale = frequency_scale

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(self.frequency_scale)
            * torch.arange(half_dim, device=timestep.device)
            / half_dim
        )
        angles = timestep[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.time_dense = nn.Sequential(
            nn.Linear(time_embedding_dim, out_channels),
            nn.ReLU(),
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(inputs)
        hidden = self.norm1(self.conv1(inputs))
        hidden = hidden + self.time_dense(time_embedding).unsqueeze(-1).unsqueeze(-1)
        hidden = self.relu(hidden)
        hidden = self.norm2(self.conv2(hidden))
        return self.relu(hidden + residual)


class MultiHeadAttention(nn.Module):
    """Attention block retained exactly for checkpoint and behavior compatibility."""

    def __init__(self, in_channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.head_dim**-0.5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.size()
        query = self.query(inputs).view(batch, self.num_heads, self.head_dim, -1)
        key = self.key(inputs).view(batch, self.num_heads, self.head_dim, -1)
        value = self.value(inputs).view(batch, self.num_heads, self.head_dim, -1)
        attention_scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * self.scale
        attention = self.softmax(attention_scores)
        hidden = torch.einsum("bhqk,bhvd->bhqd", attention, value)
        hidden = hidden.contiguous().view(batch, channels, height, width)
        return self.out(hidden) + inputs


class TimeConditionedUNet(nn.Module):
    """The four-stage time-conditioned U-Net from the supplied experiment code."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        time_embedding_dim: int = 32,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        base_filters = base_channels
        embed_dim = time_embedding_dim
        self.time_embedding = SinusoidalPositionalEmbedding(embed_dim)

        self.enc1 = ResidualBlock(input_channels, base_filters, embed_dim)
        self.enc2 = ResidualBlock(base_filters, base_filters * 2, embed_dim)
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4, embed_dim)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8, embed_dim)
        self.attn1 = MultiHeadAttention(base_filters * 8, attention_heads)

        self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16, embed_dim)
        self.attn2 = MultiHeadAttention(base_filters * 16, attention_heads)

        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8, embed_dim)
        self.attn3 = MultiHeadAttention(base_filters * 8, attention_heads)
        self.dec3 = ResidualBlock(base_filters * 12, base_filters * 4, embed_dim)
        self.dec2 = ResidualBlock(base_filters * 6, base_filters * 2, embed_dim)
        self.dec1 = ResidualBlock(base_filters * 3, base_filters, embed_dim)

        self.final = nn.Conv2d(base_filters, output_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.ConvTranspose2d(
            base_filters * 16,
            base_filters * 8,
            kernel_size=2,
            stride=2,
        )

    def forward(self, inputs: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timestep)
        enc1 = self.enc1(inputs, time_embedding)
        enc2 = self.enc2(self.pool(enc1), time_embedding)
        enc3 = self.enc3(self.pool(enc2), time_embedding)
        enc4 = self.attn1(self.enc4(self.pool(enc3), time_embedding))
        bottleneck = self.attn2(self.bottleneck(self.pool(enc4), time_embedding))

        dec4 = self.attn3(
            self.dec4(
                torch.cat([self.upsample(bottleneck), enc4], dim=1),
                time_embedding,
            )
        )
        dec3 = self.dec3(
            torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1),
            time_embedding,
        )
        dec2 = self.dec2(
            torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1),
            time_embedding,
        )
        dec1 = self.dec1(
            torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1),
            time_embedding,
        )
        return self.final(dec1)


# Familiar alias retained for compatibility with the original experiment code.
UNet = TimeConditionedUNet
