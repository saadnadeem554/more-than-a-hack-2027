"""Score the rule-based decoder on the degraded external matches (honest, labelled, other matches).

    python analysis/eval_rule_external.py [key value]   # optional single PARAMS override
Tag evidence is off (the external data has no Smart Tagging).
"""
import os, sys
from types import SimpleNamespace
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm
from src import external_sportec as ext
from src.degrade import degrade, degraded_team
from src.decoder import PARAMS, build_emissions, decode_block


def score_match(mid, p, seed=0):
    rng = np.random.default_rng(seed)
    parts, pitch = ext.load_match(mid)
    spells, macros = [], []
    for part in parts:
        px, py, bx, by = degrade(part["px"], part["py"], part["bx"], part["by"], rng, pitch=pitch)
        ns = SimpleNamespace(px=px, py=py, bx=bx, by=by, team=degraded_team(part["team"]), pitch=pitch, n_frames=len(bx))
        pred = decode_block(build_emissions(ns, p, use_tags=False), p).astype(np.int8)
        gt = np.where(part["label"] >= 0, part["label"], 3).astype(np.int8)
        runs = pm.valid_runs(gt)
        spells.append(pm.spell_scores(gt, pred, runs)["mean_f1"])
        macros.append(pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))))
    return np.mean(spells), np.mean(macros)


if __name__ == "__main__":
    p = dict(PARAMS)
    if len(sys.argv) == 3:
        p[sys.argv[1]] = type(PARAMS[sys.argv[1]])(float(sys.argv[2]))
    res = [score_match(m, p) for m in ext.match_ids()]
    print("rule-based on degraded external matches: spellF1 {:.4f}  macroF1 {:.4f}  per-match spell {}".format(
        np.mean([r[0] for r in res]), np.mean([r[1] for r in res]), [round(r[0], 3) for r in res]))
