"""Training: Adam 1e-3, batch 32, 30 epochs, one random window per video per epoch.

Usage:  python -m q2.train --model A --seed 0 [--epochs 1]
"""
import argparse
import json
import time

import torch

from .common import CKPT_DIR, RES_DIR, get_device, load_moving_mnist, splits, to_float
from .models import n_params
from .transforms import G, N_OUT, STARTS_AUG, STARTS_NORMAL, apply_g
from .zoo import CONFIGS, build


def sample_windows(n, speeds, gen):
    """Per-video speed k and start t -> 1-based frame indices and target mask."""
    ks, ts = [], []
    for _ in range(n):
        k = 1 if speeds == "normal" else [1, 2, 3][torch.randint(3, (1,), generator=gen).item()]
        starts = STARTS_NORMAL[1] if speeds == "normal" else STARTS_AUG[k]
        ks.append(k)
        ts.append(starts[torch.randint(len(starts), (1,), generator=gen).item()])
    k = torch.tensor(ks)[:, None]
    t = torch.tensor(ts)[:, None]
    j = torch.arange(5)[None]
    inp = t + j * k                     # (n, 5)
    tgt = t + (5 + j) * k               # (n, 5)
    mask = tgt <= 20
    tgt = torch.where(mask, tgt, torch.ones_like(tgt))   # placeholder frame for missing targets
    return inp - 1, tgt - 1, mask


def make_batch(vids_u8, speeds, sym, gen, device):
    n = vids_u8.shape[0]
    i_idx, t_idx, mask = sample_windows(n, speeds, gen)
    rows = torch.arange(n)[:, None]
    X = to_float(vids_u8[rows, i_idx])
    Y = to_float(vids_u8[rows, t_idx])
    if sym:
        gs = torch.randint(len(G), (n,), generator=gen)
        X = torch.stack([apply_g(X[b], G[g]) for b, g in enumerate(gs.tolist())])
        Y = torch.stack([apply_g(Y[b], G[g]) for b, g in enumerate(gs.tolist())])
    return X.to(device), Y.to(device), mask.to(device)


def masked_loss(pred, Y, mask):
    """Mean over pixels and existing targets per video, then over videos."""
    per = ((pred - Y) ** 2).mean(dim=(2, 3, 4))            # (B, 5)
    per = (per * mask).sum(1) / mask.sum(1)
    return per.mean()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def val_e5(model, val_u8, device, bs=100):
    model.eval()
    tot = 0.0
    for s in range(0, len(val_u8), bs):
        v = to_float(val_u8[s:s + bs]).to(device)
        pred = model(v[:, 0:5])
        tot += ((pred - v[:, 5:10]) ** 2).mean(dim=(1, 2, 3, 4)).sum().item()
    model.train()
    return tot / len(val_u8)


def train(name, seed, epochs=30, bs=32, data=None, log=True):
    cfg = CONFIGS[name]
    device = get_device()
    if data is None:
        data = load_moving_mnist()
    train_u8, val_u8, _ = splits(data)

    torch.manual_seed(seed)                          # initialisation
    gen = torch.Generator().manual_seed(seed)        # shuffling, windows, augmentation
    model = build(cfg["arch"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    hist = dict(model=name, seed=seed, params=n_params(model), device=str(device),
                epoch_time=[], train_loss=[], val_e5=[])
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(train_u8), generator=gen)
        sync(device); t0 = time.perf_counter()
        run = 0.0
        for s in range(0, len(perm), bs):
            X, Y, mask = make_batch(train_u8[perm[s:s + bs]], cfg["speeds"], cfg["sym"], gen, device)
            loss = masked_loss(model(X, N_OUT), Y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * X.shape[0]
        sync(device); dt = time.perf_counter() - t0
        hist["epoch_time"].append(dt)
        hist["train_loss"].append(run / len(perm))
        hist["val_e5"].append(val_e5(model, val_u8, device))
        if log:
            print(f"[{name} s{seed}] epoch {ep:2d}/{epochs}  loss {hist['train_loss'][-1]:.4e}  "
                  f"val E5 {hist['val_e5'][-1]:.4e}  {dt:.1f}s", flush=True)

    hist["mean_epoch_time"] = sum(hist["epoch_time"]) / len(hist["epoch_time"])
    tag = f"{name}_s{seed}" + ("" if epochs == 30 else f"_ep{epochs}")
    torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
    with open(RES_DIR / f"train_{tag}.json", "w") as f:
        json.dump(hist, f, indent=2)
    return model, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    data = load_moving_mnist()
    for sd in a.seed:
        train(a.model, sd, a.epochs, data=data)
