# ConvLSTM: Motion vs Memory

AI651 – Deep Learning for Space, Time and Graphs (LUMS, Fall 2026), Assignment 2
Hasan Amjad (25280042)

Question 2 code. Question 1 is a written proposal with no code.

## Layout

| Path | Contents |
|---|---|
| `Question2/q2/` | Code |
| `Question2/Question2.ipynb` | Tables and figures |
| `Question2/checkpoints/` | Trained weights |
| `Question2/results/` | Metrics and training logs |
| `Question2/figures/`, `Question2/gifs/` | Figures and GIFs |
| `Question2/preregistration/` | Q2.5 and Q2.11 predictions |
| `report/` | Report |

## Run

```bash
pip install -r Question2/requirements.txt
cd Question2
python -m q2.train --model A      # likewise B, C, D, DB
jupyter notebook Question2.ipynb
```
