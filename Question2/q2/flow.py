"""Figure 1: direction and speed of motion in the training videos (Farneback flow)."""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def pooled_flow(train_uint8, n_videos=50):
    """Flow on every consecutive pair (x_s, x_{s+1}), s = 1..19, of the first n videos.

    Keeps pixels brighter than 0.2 (on [0,1]) at the same location in both frames
    and with flow magnitude > 0.5 px. Returns dx, dy (pixels) and the pair count.
    """
    dxs, dys, n_pairs = [], [], 0
    for v in range(n_videos):
        vid = train_uint8[v, :, 0].numpy()               # (20, 64, 64) uint8
        for s in range(19):
            prev, nxt = vid[s], vid[s + 1]
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]          # dx along columns, dy along rows
            bright = (prev.astype(np.float32) / 255 > 0.2) & (nxt.astype(np.float32) / 255 > 0.2)
            moving = np.sqrt(dx ** 2 + dy ** 2) > 0.5
            keep = bright & moving
            dxs.append(dx[keep]); dys.append(dy[keep])
            n_pairs += 1
    return np.concatenate(dxs), np.concatenate(dys), n_pairs


def figure1(dx, dy, path):
    theta = np.degrees(np.arctan2(-dy, dx)) % 360.0      # 0 = right, 90 = up on screen
    speed = np.sqrt(dx ** 2 + dy ** 2)
    n = len(speed)
    w = np.full(n, 1.0 / n)
    med = float(np.median(speed))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(theta, bins=np.arange(0, 361, 20), weights=w, color="#3a7ca5", edgecolor="white")
    ax[0].axhline(1 / 18, color="k", ls="--", lw=1, label="uniform (1/18)")
    ax[0].set_xticks(range(0, 361, 60))
    ax[0].set_xlabel(r"flow direction $\theta$ (degrees; 0$^\circ$ right, 90$^\circ$ up)")
    ax[0].set_ylabel("fraction of kept pixels")
    ax[0].set_title("(a) direction")
    ax[0].legend(frameon=False)

    hi = np.percentile(speed, 99.5)
    ax[1].hist(speed, bins=np.linspace(0.5, hi, 40), weights=w, color="#3a7ca5", edgecolor="white")
    ax[1].axvline(med, color="#c0392b", lw=1.5, label=f"median = {med:.3g} px/frame")
    ax[1].set_xlabel("flow speed (pixels per frame)")
    ax[1].set_ylabel("fraction of kept pixels")
    ax[1].set_title("(b) speed")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")

    hist_dir, _ = np.histogram(theta, bins=np.arange(0, 361, 20))
    return fig, dict(n_pixels=int(n), median_speed=med,
                     mean_speed=float(speed.mean()),
                     direction_fractions=(hist_dir / n).round(4).tolist(),
                     speed_p10_p90=[float(np.percentile(speed, 10)), float(np.percentile(speed, 90))])
