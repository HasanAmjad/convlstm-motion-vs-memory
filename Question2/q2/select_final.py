"""Choose the final model's base (D or D+B) on the VALIDATION split.

Criterion (same as for model D): mean validation E(2) over 1x, 2x, 3x of the
symmetrised model, averaged over seeds 0 and 1.
Usage:  python -m q2.select_final
"""
import numpy as np

from .common import get_device, load_moving_mnist, save_json, splits
from .evaluate import evaluate, load_model, test_windows
from .models import Symmetrised


def main():
    device = get_device()
    _, val_u8, _ = splits(load_moving_mnist())
    W = test_windows(val_u8)                      # same windows, validation videos
    out = {}
    for name in ("D", "DB"):
        per_seed = []
        for sd in (0, 1):
            r, _, _ = evaluate(Symmetrised(load_model(name, sd, device)).eval(), W, device=device)
            r["val_E2_mean"] = float(np.mean([r["E2_1x"], r["E2_2x"], r["E2_3x"]]))
            per_seed.append(r)
        out[f"sym{name}"] = {k: float(np.mean([p[k] for p in per_seed]))
                             for k in ("E5_1x", "E2_1x", "E2_2x", "E2_3x", "val_E2_mean")}
        print(f"sym{name}", {k: f"{v:.4g}" for k, v in out[f"sym{name}"].items()})
    best = min(out, key=lambda k: out[k]["val_E2_mean"])
    out["choice"] = best
    print("choice:", best)
    save_json(out, "final_selection.json")


if __name__ == "__main__":
    main()
