from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import compute_psnr, compute_ssim, compute_uiqm
from .networks import BACForwardOutput, BACModel


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.epsilon**2))


class StructuralSimilarityLoss(nn.Module):
    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size
        self.avg_pool = nn.AvgPool2d(window_size, stride=1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        c1 = 0.01**2
        c2 = 0.03**2

        mu_x = self.avg_pool(x)
        mu_y = self.avg_pool(y)
        sigma_x = self.avg_pool(x * x) - (mu_x * mu_x)
        sigma_y = self.avg_pool(y * y) - (mu_y * mu_y)
        sigma_xy = self.avg_pool(x * y) - (mu_x * mu_y)

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim_map = numerator / (denominator + 1e-8)
        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))


@dataclass
class LossBreakdown:
    total: Tensor
    fused: Tensor
    fst: Tensor
    fsr: Tensor
    mu_regularization: Tensor


class BACLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = StructuralSimilarityLoss()
        self.gradient = GradientLoss()

    def _branch_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (
            self.charbonnier(prediction, target)
            + 0.4 * self.ssim(prediction, target)
            + 0.1 * self.gradient(prediction, target)
        )

    def forward(self, outputs: BACForwardOutput, target: Tensor) -> LossBreakdown:
        fused_loss = self._branch_loss(outputs.fused, target)
        fst_loss = self._branch_loss(outputs.fst, target)
        fsr_loss = self._branch_loss(outputs.fsr, target)
        mu_reg = torch.mean((outputs.mu - 0.5) ** 2)
        total = fused_loss + 0.35 * fst_loss + 0.35 * fsr_loss + 0.02 * mu_reg
        return LossBreakdown(total=total, fused=fused_loss, fst=fst_loss, fsr=fsr_loss, mu_regularization=mu_reg)


@dataclass
class EpochSummary:
    loss: float
    psnr: float
    ssim: float
    uiqm: float


def train_one_epoch(
    model: BACModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: BACLoss,
    device: torch.device,
) -> EpochSummary:
    model.train()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "uiqm": 0.0}
    count = 0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        losses.total.backward()
        optimizer.step()

        metrics = batch_metrics(outputs.fused.detach(), targets.detach())
        totals["loss"] += float(losses.total.item())
        totals["psnr"] += metrics["psnr"]
        totals["ssim"] += metrics["ssim"]
        totals["uiqm"] += metrics["uiqm"]
        count += 1

    return EpochSummary(
        loss=totals["loss"] / max(count, 1),
        psnr=totals["psnr"] / max(count, 1),
        ssim=totals["ssim"] / max(count, 1),
        uiqm=totals["uiqm"] / max(count, 1),
    )


@torch.no_grad()
def validate_one_epoch(
    model: BACModel,
    dataloader: DataLoader,
    criterion: BACLoss,
    device: torch.device,
) -> EpochSummary:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "uiqm": 0.0}
    count = 0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        metrics = batch_metrics(outputs.fused, targets)
        totals["loss"] += float(losses.total.item())
        totals["psnr"] += metrics["psnr"]
        totals["ssim"] += metrics["ssim"]
        totals["uiqm"] += metrics["uiqm"]
        count += 1

    return EpochSummary(
        loss=totals["loss"] / max(count, 1),
        psnr=totals["psnr"] / max(count, 1),
        ssim=totals["ssim"] / max(count, 1),
        uiqm=totals["uiqm"] / max(count, 1),
    )


def batch_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    preds_np = tensor_to_uint8(predictions)
    targets_np = tensor_to_uint8(targets)
    psnr_values = []
    ssim_values = []
    uiqm_values = []

    for pred, target in zip(preds_np, targets_np, strict=False):
        pred_rgb = np.transpose(pred, (1, 2, 0))
        target_rgb = np.transpose(target, (1, 2, 0))
        psnr_values.append(compute_psnr(target_rgb, pred_rgb))
        ssim_values.append(compute_ssim(target_rgb, pred_rgb))
        uiqm_values.append(compute_uiqm(pred_rgb))

    return {
        "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
        "uiqm": float(np.mean(uiqm_values)) if uiqm_values else 0.0,
    }


def tensor_to_uint8(tensor: Tensor) -> np.ndarray:
    clipped = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    return (clipped * 255.0).round().astype(np.uint8)


def save_checkpoint(
    path: Path,
    model: BACModel,
    optimizer: Optimizer,
    epoch: int,
    best_val_psnr: float,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_psnr": best_val_psnr,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def save_training_history(path: Path, history: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
