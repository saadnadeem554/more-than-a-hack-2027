"""Mean ball visibility of degraded external tracking (target ~0.593 on the competition)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import external_sportec as ext
from src.degrade import degrade

for context in (False, True):
    rng = np.random.default_rng(0)
    vals = []
    for mid in ext.match_ids()[:3]:
        parts, pitch = ext.load_match(mid)
        for p in parts:
            _, _, bx, _ = degrade(p["px"], p["py"], p["bx"], p["by"], rng, pitch=pitch, context=context)
            vals.append(np.isfinite(bx).mean())
    print("context={}: mean visible {:.3f} (range {:.3f}-{:.3f})".format(context, np.mean(vals), min(vals), max(vals)))
