"""Sweep one decoder parameter and print structural statistics (not tuned on labels).

    python analysis/sweep.py dead_bias 0.0 0.3 0.6 0.9
The public-window spellF1 is printed as a sanity check only; do not pick values by it.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from src.atd import load_atd
from src.decoder import decode, PARAMS
import possession_metrics as pm

def stats(lab):
    g = lab != "X"
    out = {"D%": round(100 * (lab[g] == "D").mean(), 1)}
    runs = []; cur = None; n = 0
    for c in lab[g]:
        if c == cur: n += 1
        else:
            if cur in ("H", "A"): runs.append(n)
            cur, n = c, 1
    if cur in ("H", "A"): runs.append(n)
    out["poss"] = len(runs); out["mean_s"] = round(np.mean(runs) / 25, 1)
    return out

if __name__ == "__main__":
    key, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]
    atd = load_atd()
    for v in vals:
        p = dict(PARAMS); p[key] = type(PARAMS[key])(v)
        lab = decode(atd, p)
        print(key, v, stats(lab))
