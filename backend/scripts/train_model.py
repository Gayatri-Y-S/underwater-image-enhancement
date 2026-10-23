from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from enhancer.dataset import collect_images, split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Training entry point for the BAC underwater enhancement model.")
    parser.add_argument("--euvp-root", type=Path, required=True, help="Path to the EUVP dataset root.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    images = collect_images(args.euvp_root)
    split = split_dataset(images, train_ratio=0.8)

    print("Training configuration")
    print(f"EUVP images found: {len(images)}")
    print(f"Train split: {len(split.train)}")
    print(f"Validation split: {len(split.val)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print()
    print("Planned training stages")
    print("1. Train ResUNet + CBAM to learn semantic feature transfer output FST.")
    print("2. Train the physical-model reconstruction branch to estimate FSR.")
    print("3. Train MuNet to predict μ for BAC fusion.")
    print("4. Save best validation weights into backend/weights/.")
    print()
    print("This scaffold intentionally leaves the heavy PyTorch training loop for your experiment phase.")
    print("You can now plug in your dataloader, loss functions, and checkpointing strategy.")


if __name__ == "__main__":
    main()
