"""Compare model-D candidates on the VALIDATION split (seed 0).

Usage:  python -m q2.select_d
"""
import torch

from .common import CKPT_DIR, get_device, load_moving_mnist, save_json, splits, to_float
from .evaluate import per_horizon, predict
from .models import n_params
from .transforms import get_window
from .zoo import D_CHOICE, D_DESIGNS, build, reach_px


def main():
    device = get_device()
    _, val_u8, _ = splits(load_moving_mnist())
    v = to_float(val_u8)
    W = {k: get_window(v, t=1, k=k) for k in (1, 2, 3)}
    out = {}
    for name in ["C"] + list(D_DESIGNS):
        arch = "A" if name == "C" else name
        f = CKPT_DIR / ("D_s0.pt" if name == D_CHOICE else f"{name}_s0.pt")   # chosen candidate's run is D's seed 0
        if not f.exists():
            print("missing", f.name); continue
        m = build(arch); m.load_state_dict(torch.load(f, map_location="cpu")); m.to(device).eval()
        r = dict(params=n_params(m), reach_px=reach_px(arch))
        for k, (X, Y, _) in W.items():
            mh = per_horizon(predict(m, X, device=device), Y)
            r[f"val_E2_{k}x"] = mh[:, :2].mean().item()
            if k == 1:
                r["val_E5_1x"] = mh.mean().item()
        r["val_E2_mean"] = sum(r[f"val_E2_{k}x"] for k in (1, 2, 3)) / 3
        out[name] = r
        print(name, {a: (f"{b:.4g}" if isinstance(b, float) else b) for a, b in r.items()})
    save_json(out, "d_selection.json")


if __name__ == "__main__":
    main()
