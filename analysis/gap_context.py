"""Where does the competition tracker lose the ball? Aggregate rates only.

For every graded frame where the ball was visible, record whether it is missing 5 frames later,
bucketed by how crowded the ball's surroundings are and by ball speed. If losses cluster in
crowds, random synthetic gaps under-represent the hard moments.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from src.atd import load_atd  # noqa: E402


def loss_rates(px, py, bx, by, pitch, lag=5):
    L, W = pitch
    vis = np.isfinite(bx)
    d = np.hypot((px - bx[:, None]) * L, (py - by[:, None]) * W)
    crowd = (np.where(np.isfinite(d), d, np.inf) < 5).sum(1)
    v = np.full(len(bx), np.nan)
    v[1:] = np.hypot(np.diff(bx) * L, np.diff(by) * W) * 25
    base = vis[:-lag] & vis[1:-lag + 1] if lag > 1 else vis[:-lag]
    lost = ~vis[lag:]
    out = {}
    for name, bins, val in (("crowd<5m", [0, 1, 2, 3, 5, 99], crowd), ("speed m/s", [0, 3, 8, 15, 25, 999], v)):
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = base & (val[:-lag] >= lo) & (val[:-lag] < hi)
            rows.append("[{},{}) n={} loss={:.4f}".format(lo, hi, int(m.sum()), lost[m].mean() if m.any() else np.nan))
        out[name] = rows
    return out


if __name__ == "__main__":
    atd = load_atd()
    for _, a, b in half_video_blocks("BAR-ATH")[:1]:
        pass
    sl = np.zeros(atd.n_frames, bool)
    for _, a, b in half_video_blocks("BAR-ATH"):
        sl[a:b + 1] = True
    idx = np.flatnonzero(sl)
    r = loss_rates(atd.px[idx], atd.py[idx], atd.bx[idx], atd.by[idx], atd.pitch)
    for k, rows in r.items():
        print(k)
        for row in rows:
            print("  ", row)
