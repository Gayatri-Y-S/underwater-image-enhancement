from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from enhancer.dataset import UnderwaterImagePairDataset, discover_paired_samples, split_dataset
from enhancer.networks import BACModel
from enhancer.training import BACLoss, save_checkpoint, save_training_history, train_one_epoch, validate_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BAC underwater image enhancement model on EUVP.")
    parser.add_argument("--euvp-root", type=Path, required=True, help="Path to the EUVP dataset root.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    args = parser.parse_args()

    samples = discover_paired_samples(args.euvp_root)
    if not samples:
        raise SystemExit(
            "No paired EUVP samples were discovered. Arrange the dataset into input/raw and reference/gt folders."
        )

    split = split_dataset(samples, train_ratio=0.8)
    train_dataset = UnderwaterImagePairDataset(split.train, image_size=args.image_size)
    val_dataset = UnderwaterImagePairDataset(split.val, image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BACModel(base_channels=args.base_channels).to(device)
    criterion = BACLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    config = {
        "base_channels": args.base_channels,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_dataset": "EUVP",
        "validation_split": "20%",
        "fusion_formula": "output = mu·FST + (1-mu)·FSR",
    }
    checkpoint_path = args.weights_dir / "bac_model.pt"
    history_path = args.weights_dir / "training_history.json"
    best_val_psnr = float("-inf")
    history: list[dict[str, float | int]] = []

    print("Training configuration")
    print(f"EUVP paired samples found: {len(samples)}")
    print(f"Train split: {len(split.train)}")
    print(f"Validation split: {len(split.val)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Device: {device}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_summary = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_summary = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_summary.loss,
                "train_psnr": train_summary.psnr,
                "train_ssim": train_summary.ssim,
                "train_uiqm": train_summary.uiqm,
                "val_loss": val_summary.loss,
                "val_psnr": val_summary.psnr,
                "val_ssim": val_summary.ssim,
                "val_uiqm": val_summary.uiqm,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_summary.loss:.4f} psnr {train_summary.psnr:.3f} ssim {train_summary.ssim:.4f} | "
            f"val loss {val_summary.loss:.4f} psnr {val_summary.psnr:.3f} ssim {val_summary.ssim:.4f}"
        )

        if val_summary.psnr > best_val_psnr:
            best_val_psnr = val_summary.psnr
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_psnr, config)
            print(f"  Saved new best checkpoint to {checkpoint_path}")

    save_training_history(history_path, history)
    print()
    print(f"Training finished. Best validation PSNR: {best_val_psnr:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
