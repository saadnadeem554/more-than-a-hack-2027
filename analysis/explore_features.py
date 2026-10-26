"""Candidate per-frame features: nearest-player distance and team, ball motion from positions.

Prints aggregates only. The public ground truth (5 minutes, published) is used to sanity-check
what each feature means, never to pick parameter values.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.atd import load_atd  # noqa: E402


def pct(a, q=(10, 25, 50, 75, 90)):
    a = a[np.isfinite(a)]
    return [round(float(np.percentile(a, k)), 2) for k in q] if len(a) else []


def main():
    a = load_atd()
    L, W = a.pitch
    vis = np.isfinite(a.bx)
    dx = (a.px - a.bx[:, None]) * L
    dy = (a.py - a.by[:, None]) * W
    dist = np.hypot(dx, dy)                       # NaN where ball or player missing
    dist_h = np.where(a.team[None, :] == 0, dist, np.nan)
    dist_a = np.where(a.team[None, :] == 1, dist, np.nan)
    with np.errstate(all="ignore"):
        dh = np.nanmin(np.where(np.isnan(dist_h), np.inf, dist_h), axis=1)
        da = np.nanmin(np.where(np.isnan(dist_a), np.inf, dist_a), axis=1)
    dh[~np.isfinite(dh)] = np.nan
    da[~np.isfinite(da)] = np.nan
    near = np.fmin(dh, da)
    near_team = np.where(dh <= da, 0, 1)

    # ball velocity (m/s) from positions over +/- 5 frames, only where both ends are visible
    k = 5
    v = np.full(a.n_frames, np.nan)
    ok = vis[2 * k:] & vis[:-2 * k]
    v[k:-k][ok] = np.hypot((a.bx[2 * k:] - a.bx[:-2 * k]) * L,
                           (a.by[2 * k:] - a.by[:-2 * k]) * W)[ok] / (2 * k / 25)

    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    lab = gt["label"].values
    print("pct = 10/25/50/75/90")
    for c in "HAD":
        m = (lab == c)
        print("{}: n={}  nearest dist {}  ball v {}".format(
            c, m.sum(), pct(near[m]), pct(v[m])))
    alive = np.isin(lab, ["H", "A"]) & np.isfinite(near)
    for r in (1.5, 2.5, 4, 6, 10):
        m = alive & (near <= r)
        acc = (near_team[m] == (lab[m] == "A")).mean()
        print("alive, nearest within {:>4} m: frames {:>5} ({:.0%} of alive-visible)  team acc {:.3f}".format(
            r, m.sum(), m.sum() / alive.sum(), acc))
    # whole match (graded-agnostic) distribution of nearest distance
    print("full-match nearest dist:", pct(near))


if __name__ == "__main__":
    main()
