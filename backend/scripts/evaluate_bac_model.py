from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from enhancer.dataset import UnderwaterImagePairDataset, discover_paired_samples
from enhancer.networks import BACModel
from enhancer.training import BACLoss, validate_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the BAC model on the UIEB dataset.")
    parser.add_argument("--uieb-root", type=Path, required=True, help="Path to the UIEB dataset root.")
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parents[1] / "weights" / "bac_model.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    samples = discover_paired_samples(args.uieb_root)
    if not samples:
        raise SystemExit("No paired UIEB samples were discovered. Provide the reference subset with matching filenames.")
    if not args.weights.exists():
        raise SystemExit(f"Checkpoint not found: {args.weights}")

    checkpoint = torch.load(args.weights, map_location="cpu")
    config = checkpoint.get("config", {})
    model = BACModel(base_channels=int(config.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = UnderwaterImagePairDataset(samples, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = BACLoss()
    summary = validate_one_epoch(model, dataloader, criterion, device)

    print("UIEB evaluation summary")
    print(f"Images evaluated: {len(samples)}")
    print(f"Average loss: {summary.loss:.4f}")
    print(f"Average PSNR: {summary.psnr:.4f}")
    print(f"Average SSIM: {summary.ssim:.4f}")
    print(f"Average UIQM: {summary.uiqm:.4f}")
    print("Formula used: output = mu·FST + (1-mu)·FSR")


if __name__ == "__main__":
    main()
