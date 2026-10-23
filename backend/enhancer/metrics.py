import math

import cv2
import numpy as np


def compute_psnr(reference: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((reference.astype(np.float32) - test.astype(np.float32)) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20 * math.log10(255.0 / math.sqrt(mse)))


def _ssim_channel(channel_a: np.ndarray, channel_b: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    a = channel_a.astype(np.float64)
    b = channel_b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(np.mean(numerator / (denominator + 1e-12)))


def compute_ssim(reference: np.ndarray, test: np.ndarray) -> float:
    ref = _ensure_uint8(reference)
    tst = _ensure_uint8(test)
    scores = [_ssim_channel(ref[:, :, idx], tst[:, :, idx]) for idx in range(3)]
    return float(np.mean(scores))


def compute_uiqm(image: np.ndarray) -> float:
    img = _ensure_uint8(image).astype(np.float32)
    r, g, b = cv2.split(img)
    rg = r - g
    yb = ((r + g) / 2.0) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.mean(np.abs(laplacian)))
    contrast = float(np.std(gray))
    return float(0.0282 * colorfulness + 0.2953 * sharpness + 3.5753 * contrast)


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)
