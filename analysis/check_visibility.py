"""Mean ball visibility of degraded external tracking for a few visible-run scales (target ~0.593)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import external_sportec as ext
from src import degrade as dg

for scale in (1.0, 1.5, 2.0, 2.5):
    rng = np.random.default_rng(0)
    vals = []
    for mid in ext.match_ids()[:3]:
        parts, _ = ext.load_match(mid)
        for p in parts:
            n = len(p["bx"])
            mask = dg.synthetic_pattern(n, rng, scale)
            out = (p["bx"] < 0) | (p["bx"] > 1) | (p["by"] < 0) | (p["by"] > 1)
            vals.append((mask & ~out).mean())
    print("visible_scale {:.1f}: mean visible {:.3f}  (out-of-pitch share {:.3f})".format(scale, np.mean(vals), out.mean()))
