"""Model and training-regime definitions (A-D, D+B)."""
from .models import ConvLSTMForecaster, model_A

# speeds: "normal" = 1x only, t in 1..11;  "aug" = k ~ U{1,2,3}, t from allowed starts
CONFIGS = {
    "A":  dict(arch="A", speeds="normal", sym=False),
    "B":  dict(arch="A", speeds="normal", sym=True),
    "C":  dict(arch="A", speeds="aug", sym=False),
    "D":  dict(arch="D", speeds="aug", sym=False),
    "DB": dict(arch="D", speeds="aug", sym=True),
}


def build(arch):
    if arch == "A":
        return model_A()
    if arch == "D":
        raise NotImplementedError("model D is defined after Q2.11")
    raise ValueError(arch)
