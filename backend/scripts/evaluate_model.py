from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from enhancer.dataset import collect_images
from enhancer.pipeline import BalancedAdaptionCompensationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BAC model on the UIEB dataset.")
    parser.add_argument("--uieb-root", type=Path, required=True, help="Path to the UIEB dataset root.")
    args = parser.parse_args()

    pipeline = BalancedAdaptionCompensationPipeline()
    images = collect_images(args.uieb_root)
    if not images:
        raise SystemExit("No test images found under the provided UIEB path.")

    psnr_values = []
    ssim_values = []
    uiqm_values = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(cv2.resize(image, (512, 512)), cv2.COLOR_BGR2RGB)
        result = pipeline.enhance(image)
        psnr_values.append(result.psnr)
        ssim_values.append(result.ssim)
        uiqm_values.append(result.uiqm)

    print("UIEB evaluation summary")
    print(f"Images evaluated: {len(psnr_values)}")
    print(f"Average PSNR: {sum(psnr_values) / len(psnr_values):.4f}")
    print(f"Average SSIM: {sum(ssim_values) / len(ssim_values):.4f}")
    print(f"Average UIQM: {sum(uiqm_values) / len(uiqm_values):.4f}")
    print("Formula used: output = μ·FST + (1−μ)·FSR")


if __name__ == "__main__":
    main()
