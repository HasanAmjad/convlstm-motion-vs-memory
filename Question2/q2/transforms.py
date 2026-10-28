"""Mirror, rotation, symmetry group G and speed windows (frames numbered 1-20)."""
import torch

# ---------------------------------------------------------------- symmetries
def mirror(x):
    """(M x)[i, j] = x[i, 63 - j]  (flip columns)."""
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    """R^a: a quarter-turns anticlockwise."""
    return torch.rot90(x, k=a, dims=(-2, -1))


# G = {R^a, M R^a : a = 0..3}; element (m, a) means x -> M^m R^a x (rotate first, then mirror)
G = [(m, a) for m in (0, 1) for a in range(4)]
G_NAMES = {(0, 0): "I", (0, 1): "R", (0, 2): "R2", (0, 3): "R3",
           (1, 0): "M", (1, 1): "MR", (1, 2): "MR2", (1, 3): "MR3"}


def apply_g(x, g):
    m, a = g
    y = rotate(x, a) if a % 4 else x
    return mirror(y) if m else y


def inverse_g(x, g):
    """g^{-1}: R^a -> R^{-a};  each M R^a is its own inverse."""
    m, a = g
    if m:
        return apply_g(x, g)
    return rotate(x, -a) if a % 4 else x


# ------------------------------------------------------------------ windows
N_FRAMES = 20
N_IN = 5
N_OUT = 5

# allowed start frames t (1-based)
STARTS_NORMAL = {1: list(range(1, 12))}                       # models A, B
STARTS_AUG = {1: list(range(1, 15)), 2: list(range(1, 9)), 3: [1, 2]}  # models C, D


def window_frames(t, k):
    """1-based frame numbers (inputs, existing targets) for start t and speed k."""
    inp = [t + j * k for j in range(N_IN)]
    tgt = [t + (N_IN + j) * k for j in range(N_OUT)]
    tgt = [f for f in tgt if f <= N_FRAMES]
    assert inp[-1] <= N_FRAMES, "input frames must all exist"
    return inp, tgt


def get_window(video, t=1, k=1):
    """video (..., 20, C, H, W) -> X, Y (zero-padded), mask of existing targets."""
    inp, tgt = window_frames(t, k)
    tdim = video.dim() - 4
    X = video.index_select(tdim, torch.tensor([f - 1 for f in inp], device=video.device))
    Yv = video.index_select(tdim, torch.tensor([f - 1 for f in tgt], device=video.device))
    pad_shape = list(Yv.shape)
    pad_shape[tdim] = N_OUT - len(tgt)
    Y = torch.cat([Yv, Yv.new_zeros(pad_shape)], dim=tdim) if len(tgt) < N_OUT else Yv
    mask = torch.zeros(N_OUT, dtype=torch.bool)
    mask[: len(tgt)] = True
    return X, Y, mask
