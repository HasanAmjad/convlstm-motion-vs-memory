"""Model and training-regime definitions (A-D, D+B)."""
from .models import ConvLSTMForecaster, model_A

# Candidate designs for model D (<= 721,176 params, reach > 8 px). Reach per step
# = (sum over the two layers of dilation * (kernel - 1) / 2) grid cells x 4 px.
D_DESIGNS = {
    "Dd12": dict(patch=4, hidden=(64, 64), kernels=(3, 3), dilations=(1, 2)),   # 3 cells = 12 px
    "Dd13": dict(patch=4, hidden=(64, 64), kernels=(3, 3), dilations=(1, 3)),   # 4 cells = 16 px
    "Dk5":  dict(patch=4, hidden=(46, 46), kernels=(5, 5), dilations=(1, 1)),   # 4 cells = 16 px
}
D_CHOICE = "Dd12"   # chosen on the validation split (results/d_selection.json)

# speeds: "normal" = 1x only, t in 1..11;  "aug" = k ~ U{1,2,3}, t from allowed starts
CONFIGS = {
    "A":  dict(arch="A", speeds="normal", sym=False),
    "B":  dict(arch="A", speeds="normal", sym=True),
    "C":  dict(arch="A", speeds="aug", sym=False),
    "D":  dict(arch=D_CHOICE, speeds="aug", sym=False),
    "DB": dict(arch=D_CHOICE, speeds="aug", sym=True),
}
for _n in D_DESIGNS:
    CONFIGS[_n] = dict(arch=_n, speeds="aug", sym=False)


def build(arch):
    if arch == "A":
        return model_A()
    if arch in D_DESIGNS:
        return ConvLSTMForecaster(**D_DESIGNS[arch])
    raise ValueError(arch)


def reach_px(arch):
    cfg = dict(patch=4, kernels=(3, 3), dilations=(1, 1)) if arch == "A" else D_DESIGNS[arch]
    cells = sum(d * (k - 1) // 2 for k, d in zip(cfg["kernels"], cfg["dilations"]))
    return cells * cfg["patch"]
