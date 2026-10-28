"""Paths, device selection, seeding and the Moving MNIST split."""
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]          # .../Question2
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
GIF_DIR = ROOT / "gifs"
CKPT_DIR = ROOT / "checkpoints"
RES_DIR = ROOT / "results"
for d in (DATA_DIR, FIG_DIR, GIF_DIR, CKPT_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_IDX = range(0, 8000)
VAL_IDX = range(8000, 9000)
TEST_IDX = range(9000, 10000)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_moving_mnist():
    """All 10,000 videos, uint8, shape (10000, 20, 1, 64, 64)."""
    from torchvision.datasets import MovingMNIST

    f = DATA_DIR / "MovingMNIST" / "mnist_test_seq.npy"
    if f.exists() and f.stat().st_size < 819_200_000:
        raise RuntimeError(f"{f} is incomplete ({f.stat().st_size:,} bytes); finish the download first")
    ds = MovingMNIST(root=str(DATA_DIR), split=None, download=True)
    data = ds.data if hasattr(ds, "data") else torch.stack([ds[i] for i in range(len(ds))])
    data = torch.as_tensor(data)
    assert data.shape == (10000, 20, 1, 64, 64), data.shape
    assert data.dtype == torch.uint8
    return data


def splits(data):
    """Train 0-7999, val 8000-8999, test 9000-9999."""
    return data[0:8000], data[8000:9000], data[9000:10000]


def to_float(x_uint8):
    return x_uint8.float().div_(255.0)


def save_json(obj, name):
    path = RES_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name):
    with open(RES_DIR / name) as f:
        return json.load(f)
