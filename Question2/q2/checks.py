"""Table 1: checks that the transform and window functions match the handout."""
import torch

from .transforms import mirror, rotate, get_window, window_frames


def _maxabs(a, b):
    return (a - b).abs().max().item()


def _bright(v):
    """(frame index, row, column) of the bright pixel."""
    idx = (v == 1).nonzero()
    assert len(idx) == 1, f"expected one bright pixel, found {len(idx)}"
    f, _, r, c = idx[0].tolist()
    return (f, r, c)


def _frame_ids(t, k):
    """Run get_window on a video whose frame f is filled with the value f."""
    v = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()
    X, Y, mask = get_window(v, t=t, k=k)
    inp = [int(X[i, 0, 0, 0].item()) for i in range(X.shape[0])]
    tgt = [int(Y[i, 0, 0, 0].item()) for i in range(Y.shape[0]) if mask[i]]
    return inp, tgt


def _fmt(frames):
    return ", ".join(map(str, frames))


def run_table1():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1                       # stored at index 7, as in the handout

    rows = []
    # (a)-(c): exact identities
    for key, lhs, rhs, label in [
        ("a", mirror(mirror(x)), x, "max|M(M(x)) - x|"),
        ("b", rotate(x, 4), x, "max|R^4(x) - x|"),
        ("c", mirror(rotate(mirror(x))), rotate(x, -1), "max|M R M(x) - R^-1(x)|"),
    ]:
        rows.append(dict(check=key, what=label, expected="0",
                         yours=f"{_maxabs(lhs, rhs):g} (torch.equal: {torch.equal(lhs, rhs)})",
                         ok=torch.equal(lhs, rhs)))

    # (d)-(e): bright-pixel locations
    fM = _bright(mirror(p)); fR = _bright(rotate(p))
    rows.append(dict(check="d", what="bright pixel of M(p) (frame, row, column)",
                     expected="(7, 10, 43)", yours=str(fM), ok=fM == (7, 10, 43)))
    rows.append(dict(check="e", what="bright pixel of R(p) (frame, row, column)",
                     expected="(7, 43, 10)", yours=str(fR), ok=fR == (7, 43, 10)))

    # (f)-(h): window frames at t = 1
    expected = {1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
                2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
                3: ([1, 4, 7, 10, 13], [16, 19])}
    for key, k in zip("fgh", (1, 2, 3)):
        inp, tgt = _frame_ids(1, k)
        assert (inp, tgt) == window_frames(1, k)
        ei, et = expected[k]
        rows.append(dict(check=key, what=f"{k}x input / target frames",
                         expected=f"{_fmt(ei)} / {_fmt(et)}",
                         yours=f"{_fmt(inp)} / {_fmt(tgt)}",
                         ok=(inp, tgt) == (ei, et)))
    return rows


def print_table1(rows):
    w = max(len(r["what"]) for r in rows)
    print(f"{'':4}{'Check':{w}}  {'Expected':38}  Yours")
    for r in rows:
        flag = "OK " if r["ok"] else "FAIL"
        print(f"({r['check']}) {r['what']:{w}}  {r['expected']:38}  {r['yours']}  [{flag}]")
    print("\nALL CHECKS PASS" if all(r["ok"] for r in rows) else "\nSOME CHECKS FAIL")
