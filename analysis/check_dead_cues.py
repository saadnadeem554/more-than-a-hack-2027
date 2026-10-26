"""Which cheap signals separate dead ball from alive on the public window? Aggregates only."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.atd import load_atd  # noqa: E402

a = load_atd()
L, W = a.pitch
print("player speed finite overall: {:.3f}".format(np.isfinite(a.ps).mean()))
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
lab = gt["label"].values
lo, hi = 11185, 18685

# mean player speed per frame, smoothed over 2 s
with np.errstate(all="ignore"):
    ms = np.nanmean(a.ps, axis=1)
k = 25
ms_s = pd.Series(ms).rolling(2 * k + 1, center=True, min_periods=5).mean().values

# team spread: std of all tracked player positions (metres), smoothed
with np.errstate(all="ignore"):
    sx = np.nanstd(a.px * L, axis=1)
    sy = np.nanstd(a.py * W, axis=1)
vis = np.isfinite(a.bx)
vis_s = pd.Series(vis.astype(float)).rolling(2 * k + 1, center=True, min_periods=1).mean().values
edge = np.minimum.reduce([a.bx * L, (1 - a.bx) * L, a.by * W, (1 - a.by) * W])
out = (edge < 0.0)

for name, arr in (("mean player speed (2s)", ms_s), ("player x-spread", sx), ("player y-spread", sy),
                  ("ball visible frac (2s)", vis_s), ("ball edge dist m", edge)):
    row = []
    for c in ("H", "A", "D"):
        m = np.zeros(len(lab), bool)
        m[lo:hi] = lab[lo:hi] == c
        v = arr[m]
        v = v[np.isfinite(v)]
        row.append("{}: {}".format(c, np.round(np.percentile(v, [10, 50, 90]), 2).tolist() if len(v) else "-"))
    print("{:<24} {}".format(name, "   ".join(row)))
