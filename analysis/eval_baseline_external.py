"""Starter baseline (nearest player, majority vote, D where ball missing) on degraded external matches.
Calibrates how external-match spellF1 relates to the competition's published numbers."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation")); sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm
from baseline import smooth
from src import external_sportec as ext
from src.degrade import degrade, degraded_team

for window in (25, 50, 125):
    res = []
    for mid in ext.match_ids():
        rng = np.random.default_rng(0)
        parts, pitch = ext.load_match(mid)
        sp = []
        for part in parts:
            px, py, bx, by = degrade(part["px"], part["py"], part["bx"], part["by"], rng, pitch=pitch)
            L, W = pitch
            d = np.hypot((px - bx[:, None]) * L, (py - by[:, None]) * W)
            d = np.where(np.isfinite(d), d, np.inf)
            near = d.argmin(1)
            codes = degraded_team(part["team"])[near].astype(np.int8)
            codes[~np.isfinite(bx) | ~np.isfinite(d.min(1))] = -1
            lab = smooth(codes, window)
            pred = np.array([{"H": 0, "A": 1, "D": 2}[c] for c in lab], dtype=np.int8)
            gt = np.where(part["label"] >= 0, part["label"], 3).astype(np.int8)
            sp.append(pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"])
        res.append(np.mean(sp))
    print("baseline smooth{}: external spellF1 {:.4f}".format(window, np.mean(res)))
