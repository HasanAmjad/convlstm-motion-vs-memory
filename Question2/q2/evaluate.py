"""Metrics: MSE_h, E(5), E(2), consistency gaps C_T and transformed errors E_T."""
import time

import numpy as np
import torch

from .common import CKPT_DIR, RES_DIR, get_device, to_float
from .transforms import get_window, mirror, rotate
from .zoo import CONFIGS, build

TRANSFORMS = {"M": mirror, "R": rotate}


def load_model(name, seed, device=None):
    device = device or get_device()
    m = build(CONFIGS[name]["arch"])
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_s{seed}.pt", map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def predict(model, X, bs=32, device=None):
    """X (N, 5, 1, 64, 64) on CPU -> predictions on CPU."""
    device = device or get_device()
    out = [model(X[s:s + bs].to(device)).cpu() for s in range(0, len(X), bs)]
    return torch.cat(out)


def per_horizon(P, Y):
    """(N, 5) per-video MSE of each forecast horizon."""
    return ((P - Y) ** 2).mean(dim=(2, 3, 4))


def test_windows(videos_u8):
    """Windows at t = 1 for speeds 1x, 2x, 3x (used for the test and validation splits)."""
    v = to_float(videos_u8)
    return {k: get_window(v, t=1, k=k) for k in (1, 2, 3)}


def evaluate(model, windows, device=None):
    """All Table 2-4 quantities. Per-video arrays are kept for histograms."""
    X1, Y1, _ = windows[1]
    P1 = predict(model, X1, device=device)
    mse_h = per_horizon(P1, Y1)
    r = dict(MSE_h=mse_h.mean(0).tolist(), E5_1x=mse_h.mean().item(), E2_1x=mse_h[:, :2].mean().item())
    per_video = dict(E5_1x=mse_h.mean(1).numpy())
    preds = {"I": P1}
    for name, T in TRANSFORMS.items():
        PT = predict(model, T(X1), device=device)
        C = ((PT - T(P1)) ** 2).mean(dim=(1, 2, 3, 4))
        E = ((PT - T(Y1)) ** 2).mean(dim=(1, 2, 3, 4))
        r[f"C_{name}"], r[f"E_{name}"] = C.mean().item(), E.mean().item()
        per_video[f"C_{name}"], per_video[f"E_{name}"] = C.numpy(), E.numpy()
        preds[name] = PT
    for k in (2, 3):
        Xk, Yk, mask = windows[k]
        Pk = predict(model, Xk, device=device)
        m = per_horizon(Pk, Yk)
        r[f"E2_{k}x"] = m[:, :2].mean().item()
        if mask.all():
            r[f"E5_{k}x"] = m.mean().item()
        preds[f"{k}x"] = Pk
    return r, per_video, preds


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def inference_time(model, X1, bs=32, device=None):
    """Wall-clock to predict all 1x test windows; batches moved to device before timing."""
    device = device or get_device()
    batches = [X1[s:s + bs].to(device) for s in range(0, len(X1), bs)]
    model(batches[0]); sync(device)                     # warm-up
    t0 = time.perf_counter()
    for b in batches:
        model(b)
    sync(device)
    return time.perf_counter() - t0


# ------------------------------------------------------------- formatting
def fmt(x):
    """3 significant figures; scientific notation below 1e-3."""
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    s = f"{x:#.3g}"
    return s.rstrip(".") if "e" not in s else s


def cell(vals):
    """'mean (seed 0, seed 1)' for learned models, single value otherwise."""
    if len(vals) == 1:
        return fmt(vals[0])
    return f"{fmt(float(np.mean(vals)))} ({fmt(vals[0])}, {fmt(vals[1])})"
