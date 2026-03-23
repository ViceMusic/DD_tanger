"""Minimal MNIST loader."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


def mnist_tensor(raw_dir: str | Path = "datasets/raw/mnist") -> tuple[TensorDataset, TensorDataset, dict[str, tuple[int, ...]]]:
    raw_dir = Path(raw_dir)

    def read_idx(path: Path) -> torch.Tensor:
        with gzip.open(path, "rb") as handle:
            _, data_type, dims = struct.unpack(">HBB", handle.read(4))
            if data_type != 0x08:
                raise ValueError(f"Unsupported IDX data type: {data_type}")
            shape = struct.unpack(">" + "I" * dims, handle.read(4 * dims))
            data = handle.read()
        return torch.frombuffer(data, dtype=torch.uint8).clone().reshape(shape)

    train_images = read_idx(raw_dir / "train-images-idx3-ubyte.gz").unsqueeze(1).float() / 255.0
    train_labels = read_idx(raw_dir / "train-labels-idx1-ubyte.gz").long()
    test_images = read_idx(raw_dir / "t10k-images-idx3-ubyte.gz").unsqueeze(1).float() / 255.0
    test_labels = read_idx(raw_dir / "t10k-labels-idx1-ubyte.gz").long()

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    shape_info = {
        "train_images": tuple(train_images.shape),
        "train_labels": tuple(train_labels.shape),
        "test_images": tuple(test_images.shape),
        "test_labels": tuple(test_labels.shape),
    }
    return train_dataset, test_dataset, shape_info
