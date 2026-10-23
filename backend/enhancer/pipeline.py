from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .metrics import compute_psnr, compute_ssim, compute_uiqm
from .networks import BACModel


@dataclass
class EnhancementResult:
    fused_image: np.ndarray
    fst_image: np.ndarray
    fsr_image: np.ndarray
    mu: float
    psnr: float
    ssim: float
    uiqm: float
    notes: str


class ResUNetCBAMModule:
    def infer(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(rgb, -1, sharpen_kernel)


class PhysicalReconstructionModule:
    def infer(self, image: np.ndarray) -> np.ndarray:
        rgb = image.astype(np.float32) / 255.0
        channel_means = np.maximum(rgb.mean(axis=(0, 1)), 1e-4)
        balanced = rgb / channel_means.reshape(1, 1, 3)
        balanced = np.clip(balanced / balanced.max(), 0.0, 1.0)
        gamma_corrected = np.power(balanced, 0.9)
        return (gamma_corrected * 255.0).astype(np.uint8)


class MuNetModule:
    def infer(self, fst_image: np.ndarray, fsr_image: np.ndarray) -> float:
        fst_gray = cv2.cvtColor(fst_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fsr_gray = cv2.cvtColor(fsr_image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        fst_score = np.std(fst_gray) + 0.15 * np.mean(cv2.Laplacian(fst_gray, cv2.CV_32F))
        fsr_score = np.std(fsr_gray) + 0.15 * np.mean(cv2.Laplacian(fsr_gray, cv2.CV_32F))
        raw_mu = fst_score / (abs(fst_score) + abs(fsr_score) + 1e-6)
        return float(np.clip(raw_mu, 0.15, 0.85))


class BalancedAdaptionCompensationPipeline:
    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir
        self.resunet_cbam = ResUNetCBAMModule()
        self.physical_model = PhysicalReconstructionModule()
        self.munet = MuNetModule()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learned_model = self._load_trained_model()

    def enhance(self, image: np.ndarray) -> EnhancementResult:
        if self.learned_model is not None:
            fst, fsr, mu, fused = self._enhance_with_trained_model(image)
            notes = (
                "Inference uses trained BAC weights: ResUNet+CBAM for FST, learnable physical branch "
                "for FSR, MuNet for \u03bc, and final fusion output = \u03bc\u00b7FST + (1\u2212\u03bc)\u00b7FSR."
            )
        else:
            fst = self.resunet_cbam.infer(image)
            fsr = self.physical_model.infer(image)
            mu = self.munet.infer(fst, fsr)
            fused = np.clip((mu * fst.astype(np.float32)) + ((1.0 - mu) * fsr.astype(np.float32)), 0, 255).astype(np.uint8)
            notes = (
                "Inference follows the PPT BAC flow: ResUNet+CBAM -> FST, physical model -> FSR, "
                "MuNet -> \u03bc, final output = \u03bc\u00b7FST + (1\u2212\u03bc)\u00b7FSR. The current repository uses a "
                "deterministic prototype pipeline until trained weights are added."
            )

        return EnhancementResult(
            fused_image=fused,
            fst_image=fst,
            fsr_image=fsr,
            mu=mu,
            psnr=compute_psnr(image, fused),
            ssim=compute_ssim(image, fused),
            uiqm=compute_uiqm(fused),
            notes=notes,
        )

    def _load_trained_model(self) -> BACModel | None:
        if self.weights_dir is None:
            return None
        checkpoint_path = self.weights_dir / "bac_model.pt"
        if not checkpoint_path.exists():
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        model = BACModel(base_channels=int(config.get("base_channels", 32)))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _enhance_with_trained_model(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.learned_model(tensor)
        fst = self._tensor_to_rgb(outputs.fst)
        fsr = self._tensor_to_rgb(outputs.fsr)
        fused = self._tensor_to_rgb(outputs.fused)
        mu = float(outputs.mu.mean().item())
        return fst, fsr, mu, fused

    @staticmethod
    def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
        image = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
