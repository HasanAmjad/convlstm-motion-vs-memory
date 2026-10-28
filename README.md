# ConvLSTM: Motion vs Memory

Code for **AI651 – Deep Learning for Space, Time and Graphs (LUMS, Fall 2026), Assignment 2**.
Author: Hasan Amjad (25280042)

Does a video predictor learn *how things move*, or only *what training videos look like*?
A ConvLSTM forecaster is trained on Moving MNIST (5 frames in, 5 frames out) and stress-tested
with transformations that real motion should not care about: mirror, 90° rotation, and 2×/3× speed-up.
The failures are traced to appearance, unfamiliar motion, or architectural reach, and the final
model is a larger-reach ConvLSTM wrapped as a forecaster that is exactly symmetric under the 8 symmetries of the square.

## Repository layout

| Path | Contents |
|---|---|
| `Question2/` | All code, notebook, results, figures and GIFs for Question 2 |
| `Question2/preregistration/` | Predictions for Q2.5 and Q2.11, committed **before** the corresponding experiments (see commit times) |
| `report/` | LaTeX source and PDF of the submitted report |

**Question 1** (river forecasting proposal) is a written design proposal. The handout states that no
dataset, implementation or training run is required, so it has no code. See the report.

## Reproducing Question 2

```bash
pip install -r Question2/requirements.txt
```
Moving MNIST is downloaded automatically by `torchvision.datasets.MovingMNIST`.
Videos 0–7,999 train, 8,000–8,999 validation, 9,000–9,999 test. Every learned model is trained with seeds 0 and 1.
