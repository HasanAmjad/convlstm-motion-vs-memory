"""Figures 2-5 and GIFs."""
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import to_float


def n_blobs(frame, thr=0.3, min_area=15):
    """Number of separate bright blobs (dilated by 1 px so touching digits merge)."""
    m = (frame.squeeze().numpy() > thr).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum())


def separated(video, frames):
    """True if the two digits are apart in every listed frame (1-based)."""
    return all(n_blobs(video[f - 1]) >= 2 for f in frames)


def overlap_frames(video, frames):
    """How many of the listed frames show merged digits."""
    return sum(n_blobs(video[f - 1]) < 2 for f in frames)


def _show(ax, img, title=None):
    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def figure2(X, Y, P, ids, path):
    fig, axs = plt.subplots(3 * len(ids), 5, figsize=(6, 3.8 * len(ids)))
    for n, i in enumerate(ids):
        for h in range(5):
            _show(axs[3 * n, h], X[i, h], f"input {h + 1}")
            _show(axs[3 * n + 1, h], P[i, h], f"pred {h + 6}")
            _show(axs[3 * n + 2, h], Y[i, h], f"true {h + 6}")
        axs[3 * n, 0].set_ylabel(f"video {9000 + i}", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def four_rows(X, Y, P, PT, T):
    """Rows: (X | f(X)), (T(X) | T(Y)), (T(X) | f(T(X))), (T(X) | T(f(X)))."""
    TX = T(X)
    return [(X, P), (TX, T(Y)), (TX, PT), (TX, T(P))]


def figure3_panel(rows, label, path):
    fig, axs = plt.subplots(4, 10, figsize=(10, 4.4), gridspec_kw=dict(wspace=0.05, hspace=0.08))
    for r, (left, right) in enumerate(rows):
        for c in range(5):
            _show(axs[r, c], left[c], f"in {c + 1}" if r == 0 else None)
            _show(axs[r, 5 + c], right[c], f"h={c + 1}" if r == 0 else None)
        axs[r, 0].set_ylabel(f"({r + 1})", rotation=0, labelpad=12, fontsize=9, va="center")
    fig.suptitle(label, fontsize=9, y=0.995)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def gif(rows, path, scale=3):
    """Animated GIF over the 5 horizons: each row shows its 5 inputs and horizon h."""
    frames = []
    for h in range(5):
        lines = []
        for left, right in rows:
            tiles = [left[c, 0] for c in range(5)] + [torch.ones(64, 4), right[h, 0]]
            lines.append(torch.cat(tiles, dim=1))
            lines.append(torch.ones(4, lines[-1].shape[1]))
        img = (torch.cat(lines[:-1], dim=0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        img = np.kron(img, np.ones((scale, scale), dtype=np.uint8))
        frames.append(img)
    imageio.mimsave(path, frames, duration=0.8, loop=0)


def figure4(per_video, marks, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key in zip(axs, ["C_M", "C_R"]):
        v = per_video[key]
        bins = np.logspace(np.log10(max(v.min(), 1e-8)), np.log10(v.max()), 40)
        ax.hist(v, bins=bins, color="#3a7ca5", edgecolor="white")
        for i, ls in zip(marks[key], ["-", "--"]):
            ax.axvline(v[i], color="#c0392b", lw=1.3, ls=ls, label=f"video {9000 + i}")
        ax.set_xscale("log"); ax.set_xlabel(f"per-video ${key[0]}_{key[2]}$"); ax.set_ylabel("videos")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def figure5(windows, preds, i, path):
    fig, axs = plt.subplots(3, 9, figsize=(10, 3.6))
    for r, k in enumerate((1, 2, 3)):
        X, Y, _ = windows[k]
        P = preds["I" if k == 1 else f"{k}x"]
        for c in range(5):
            _show(axs[r, c], X[i, c], "input" if r == 0 else None)
        for h in range(2):
            _show(axs[r, 5 + h], P[i, h], f"pred h={h + 1}" if r == 0 else None)
            _show(axs[r, 7 + h], Y[i, h], f"true h={h + 1}" if r == 0 else None)
        axs[r, 0].set_ylabel(f"{k}x", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
