from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.layers(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = self.mlp(self.avg_pool(x))
        max_value = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg + max_value)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg, max_value], dim=1)))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.residual(self.block(x))
        return features, self.pool(features)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.block = ConvBlock(out_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        self.attention = CBAM(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.residual(self.block(x))
        return self.attention(x)


class ResUNetCBAM(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = EncoderStage(in_channels, base_channels)
        self.enc2 = EncoderStage(base_channels, base_channels * 2)
        self.enc3 = EncoderStage(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderStage(base_channels * 4, base_channels * 8)

        self.bridge = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            ResidualBlock(base_channels * 16),
            CBAM(base_channels * 16),
        )

        self.dec4 = DecoderStage(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderStage(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderStage(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderStage(base_channels * 2, base_channels, base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        skip1, down1 = self.enc1(x)
        skip2, down2 = self.enc2(down1)
        skip3, down3 = self.enc3(down2)
        skip4, down4 = self.enc4(down3)

        bridge = self.bridge(down4)
        up4 = self.dec4(bridge, skip4)
        up3 = self.dec3(up4, skip3)
        up2 = self.dec2(up3, skip2)
        up1 = self.dec1(up2, skip1)
        return self.head(up1)


class LearnablePhysicalModel(nn.Module):
    """A lightweight differentiable branch that mimics underwater image formation inversion."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, base_channels),
            nn.MaxPool2d(2),
            ConvBlock(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 2, base_channels * 4),
        )
        self.transmission_head = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.atmospheric_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 4, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refinement = nn.Sequential(
            ConvBlock(6, base_channels),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        transmission = self.transmission_head(features).clamp(0.15, 0.95)
        atmospheric = self.atmospheric_head(features)
        restored = (x - atmospheric * (1.0 - transmission)) / transmission
        restored = restored.clamp(0.0, 1.0)
        return self.refinement(torch.cat([x, restored], dim=1))


class MuNet(nn.Module):
    def __init__(self, in_channels: int = 9, hidden_channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(in_channels, hidden_channels),
            nn.MaxPool2d(2),
            ConvBlock(hidden_channels, hidden_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(hidden_channels * 2, hidden_channels * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, fst: Tensor, fsr: Tensor) -> Tensor:
        fused_features = torch.cat([x, fst, fsr], dim=1)
        return self.head(self.network(fused_features))


@dataclass
class BACForwardOutput:
    fst: Tensor
    fsr: Tensor
    mu: Tensor
    fused: Tensor


class BACModel(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.semantic_branch = ResUNetCBAM(base_channels=base_channels)
        self.physical_branch = LearnablePhysicalModel(base_channels=base_channels)
        self.mu_branch = MuNet(hidden_channels=base_channels)

    def forward(self, x: Tensor) -> BACForwardOutput:
        fst = self.semantic_branch(x)
        fsr = self.physical_branch(x)
        mu = self.mu_branch(x, fst, fsr).view(-1, 1, 1, 1)
        fused = (mu * fst) + ((1.0 - mu) * fsr)
        return BACForwardOutput(fst=fst, fsr=fsr, mu=mu, fused=fused.clamp(0.0, 1.0))
