# ConvLSTM: Motion vs Memory

AI651 – Deep Learning for Space, Time and Graphs (LUMS, Fall 2026), Assignment 2
Hasan Amjad (25280042)

Does a video forecaster learn *how things move*, or only *what training videos look like*?
A ConvLSTM is trained on Moving MNIST (5 frames in, 5 frames out) and tested under
transformations that real motion should not care about: mirror, 90° rotation, and 2×/3× speed-up.

## Findings

| Failure of the baseline (model A) | Cause | Evidence |
|---|---|---|
| Larger error under rotation than mirror | Digit appearance | Symmetry augmentation (B) halves the rotation gap |
| Breaks down at 2× and 3× | Mostly unfamiliar motion, slightly reach | Speed training (C) cuts 3× error by 35%; larger reach (D) adds ~3.5% |

The final model, model D wrapped as a forecaster averaged over the 8 symmetries of the square,
is exactly consistent under mirror and rotation (gaps ≈ 1e-16) and robust to speed, but predictions
blur within a few frames. Verdict: motion is only partly learned.

## Repository layout

| Path | Contents |
|---|---|
| `Question2/q2/` | Code: transforms and windows, models, training, evaluation, figures |
| `Question2/Question2.ipynb` | Runs every check, table and figure in the order of the handout |
| `Question2/checkpoints/` | Trained weights for A, B, C, D, D+B (seeds 0, 1) and the D candidates |
| `Question2/results/` | Metrics, design-selection results and training logs (JSON) |
| `Question2/figures/`, `Question2/gifs/` | Figures and animated GIFs used in the report |
| `Question2/preregistration/` | Q2.5 and Q2.11 predictions, committed before the experiments |
| `report/` | LaTeX source and PDF of the report |

Question 1 is a written design proposal with no code (no implementation is required by the handout).

## Models

| Model | Architecture | Training data | Reach |
|---|---|---|---|
| A | 2-layer ConvLSTM, 64 channels, 3×3 kernels, 4×4 patch folding (480,784 params) | 1× | 8 px |
| B | as A | 1×, random mirror/rotation | 8 px |
| C | as A | 1×, 2×, 3× | 8 px |
| D | as A, layer 2 dilated by 2 (480,784 params) | 1×, 2×, 3× | 12 px |
| Final | D averaged over the 8 symmetries | — | 12 px |

Design choices (model D, and D vs D+B for the final model) were made on the validation split only.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r Question2/requirements.txt
cd Question2

python -m q2.train --model A          # seeds 0 and 1; likewise B, C, D, DB
python -m q2.train --model Dd13 --seed 0   # other D candidates (Dd12 is D)
python -m q2.select_d                 # compare D candidates on validation
python -m q2.select_final             # symmetrised D vs D+B on validation

jupyter notebook Question2.ipynb      # tables and figures
```

Moving MNIST is downloaded by `torchvision.datasets.MovingMNIST` into `Question2/data/`.
Videos 0–7,999 train, 8,000–8,999 validation, 9,000–9,999 test.
Hardware used: MacBook Pro (2021), Apple M1 Pro, 32 GB, PyTorch `mps` backend (about 25 s per epoch).
