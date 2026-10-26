"""How fine-grained is each label source? Possessions per minute of live play and short-spell share.

Compares the DFL provider labels (raw, and with short "sandwiched" spells merged into the
surrounding team -- a touch without control, as the competition definition describes) against the
public 5-minute ground truth, which the organisers publish as the reference for how cases are
labelled. Aggregate counts only.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import external_sportec as ext  # noqa: E402
from src.labels import merge_sandwiched  # noqa: E402


def spells(lab):
    """[(start, stop, code)] maximal runs of 0/1 (possessions) in an array with 2 = dead."""
    out, start = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            if lab[start] in (0, 1):
                out.append((start, i, lab[start]))
            start = i
    return out


def stats(labels):
    n_sp, alive, short2, lens = 0, 0, 0, []
    for lab in labels:
        sp = spells(lab)
        n_sp += len(sp)
        alive += int(np.isin(lab, (0, 1)).sum())
        lens += [(b - a) / 25 for a, b, _ in sp]
    lens = np.array(lens)
    return n_sp / (alive / 25 / 60), (lens < 2).mean(), np.median(lens), n_sp


if __name__ == "__main__":
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
    pub = np.array([{"H": 0, "A": 1, "D": 2}.get(c, -1) for c in gt[11185:18685]])
    r = stats([pub])
    print("public GT        : {:.2f} poss/alive-min  <2s {:.0%}  median {:.1f}s  n={}".format(*r))
    dfl = [p["label"] for mid in ext.match_ids() for p in ext.load_match(mid)[0]]
    for t in (0, 1, 2, 3, 4, 6):
        lab = [merge_sandwiched(l, int(t * 25)) for l in dfl] if t else dfl
        r = stats(lab)
        print("DFL merge <{:>2}s  : {:.2f} poss/alive-min  <2s {:.0%}  median {:.1f}s  n={}".format(t, *r))
