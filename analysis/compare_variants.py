"""Compare model sets on the competition match: public-window spellF1 (with bootstrap interval)
and full-match structure (possessions per live minute, dead share). Sanity only -- the public
window holds 17 possessions and cannot rank (EVALUATION.md).

    python analysis/compare_variants.py name=glob@cost [name=glob@cost ...]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from src.atd import load_atd  # noqa: E402
from src import seq_model  # noqa: E402

gt = pm.label_codes(pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
atd = load_atd()
for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1)
    pattern, cost = rest.rsplit("@", 1)
    cfg = dict(seq_model.SEQ, switch_cost=float(cost))
    labels = seq_model.decode_seq(atd, sorted(glob.glob(pattern)), cfg)
    pred = pm.label_codes(labels)
    runs = pm.valid_runs(gt)
    sp = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    lo, hi = pm.bootstrap_spell_f1(gt, pred, runs, 1000)[:2]
    alive = np.isin(labels, ["H", "A"])
    n_poss = int(np.sum(alive[1:] & (labels[1:] != labels[:-1])) + alive[0])
    g = labels != "X"
    print("{:<14} public spellF1 {:.3f} [{:.3f}, {:.3f}]  poss {:>2}/17 | full: {:.2f} poss/live-min, D {:.3f}".format(
        name, sp["mean_f1"], lo, hi, sp["n_pred"], n_poss / (alive.sum() / 1500), (labels[g] == "D").mean()))
