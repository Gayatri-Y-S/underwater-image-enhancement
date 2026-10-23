from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_KEYWORDS = ("raw", "input", "inp", "degraded", "distorted", "underwater", "traina", "test")
TARGET_KEYWORDS = ("gt", "ref", "reference", "label", "clear", "clean", "target", "trainb")


@dataclass
class DatasetSplit:
    train: list["PairedSample"]
    val: list["PairedSample"]


@dataclass
class PairedSample:
    input_path: Path
    target_path: Path
    sample_id: str


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def split_dataset(samples: list[PairedSample], train_ratio: float = 0.8, seed: int = 42) -> DatasetSplit:
    items = list(samples)
    random.Random(seed).shuffle(items)
    split_index = int(len(items) * train_ratio)
    return DatasetSplit(train=items[:split_index], val=items[split_index:])


def discover_paired_samples(root: Path) -> list[PairedSample]:
    files = collect_images(root)
    if not files:
        return []

    sample_map: dict[str, dict[str, Path]] = {}
    for path in files:
        normalized_parent = path.parent.as_posix().lower()
        role = _infer_role(normalized_parent)
        sample_id = _normalize_stem(path.stem)
        sample_map.setdefault(sample_id, {})[role] = path

    paired = [
        PairedSample(input_path=paths["input"], target_path=paths["target"], sample_id=sample_id)
        for sample_id, paths in sample_map.items()
        if "input" in paths and "target" in paths
    ]
    return sorted(paired, key=lambda sample: sample.sample_id)


class UnderwaterImagePairDataset(Dataset):
    def __init__(self, samples: list[PairedSample], image_size: int = 512) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        input_image = _load_image(sample.input_path, self.image_size)
        target_image = _load_image(sample.target_path, self.image_size)
        return {
            "input": torch.from_numpy(input_image).permute(2, 0, 1).float(),
            "target": torch.from_numpy(target_image).permute(2, 0, 1).float(),
            "sample_id": sample.sample_id,
        }


def _load_image(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def _normalize_stem(stem: str) -> str:
    normalized = stem.lower()
    for token in INPUT_KEYWORDS + TARGET_KEYWORDS:
        normalized = normalized.replace(token, "")
    normalized = normalized.replace("-", "").replace("_", "").replace(" ", "")
    return normalized


def _infer_role(parent: str) -> str:
    if any(token in parent for token in TARGET_KEYWORDS):
        return "target"
    return "input"
