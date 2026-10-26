"""Is the ATD ball speed field usable, and does it match the displacement between frames?"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.atd import load_atd  # noqa: E402

a = load_atd()
vis = np.isfinite(a.bx)
idx = np.arange(len(vis))
print("speed finite among visible:", np.isfinite(a.bs[vis]).mean())
w = slice(11185, 18685)
print("public window: vis", vis[w].mean(), "speed finite", np.isfinite(a.bs[w]).mean())
L, W = a.pitch
d = np.hypot(np.diff(a.bx) * L, np.diff(a.by) * W) * 25
ok = vis[1:] & vis[:-1] & np.isfinite(a.bs[1:])
print("pairs:", ok.sum(), "corr speed vs displacement*25:",
      np.corrcoef(d[ok], a.bs[1:][ok])[0, 1],
      "median ratio", np.median(a.bs[1:][ok] / np.maximum(d[ok], 1e-6)))
print("displacement*25 pct:", np.percentile(d[ok], [5, 50, 95, 99]))
