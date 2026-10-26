"""Where does held-out spellF1 go? Oracle decompositions on dumped external-fold predictions.

    python analysis/oracle_errors.py kaggle/out            # reads dump_*_val*.npz

For each held-out half (frame-level labels), score:
  model          the shipped decoding (step Viterbi, cost 1.0)
  oracle_dead    model, but every frame's alive/dead status replaced by the truth
                 (team of alive frames: model's, or the model's last team where it said dead)
  oracle_team    model's alive/dead status, but alive frames get the true team where the truth
                 is alive
plus counts of true possessions by length and how many the model matched at IoU 0.5.
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from train_seq import viterbi_logp  # noqa: E402

STEP, COST = 10, 1.0


def ffill_team(pred):
    """Team at each frame: pred where alive, else the last alive team (or 0)."""
    team = np.where(pred <= 1, pred, -1).astype(int)
    idx = np.where(team >= 0, np.arange(len(team)), 0)
    np.maximum.accumulate(idx, out=idx)
    t = team[idx]
    return np.where(t >= 0, t, 0)


def spell(gt, pred):
    return pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]


def main(folder):
    res = {"model": [], "oracle_dead": [], "oracle_team": []}
    length_bins = [0, 2, 5, 10, 20, 1e9]
    found = np.zeros(len(length_bins) - 1)
    total = np.zeros(len(length_bins) - 1)
    n_gt, n_pred = 0, 0
    for path in sorted(glob.glob(os.path.join(folder, "dump_*_val*.npz"))):
        z = np.load(path)
        for key in [k for k in z.files if k.startswith("logp_")]:
            i = key.split("_")[1]
            yf = z["yf_" + i].astype(np.int8)
            n = len(yf)
            gt = np.where(yf >= 0, yf, 3).astype(np.int8)
            pred = np.repeat(viterbi_logp(z[key], COST), STEP)[:n].astype(np.int8)
            od = np.where(gt == 2, 2, ffill_team(pred)).astype(np.int8)
            od[gt == 3] = pred[gt == 3]
            ot = pred.copy()
            m = (pred <= 1) & (gt <= 1)
            ot[m] = gt[m]
            res["model"].append(spell(gt, pred))
            res["oracle_dead"].append(spell(gt, od))
            res["oracle_team"].append(spell(gt, ot))

            runs = pm.valid_runs(gt)
            g_sp = pm.possession_spells(gt, runs)
            p_sp = pm.possession_spells(pred, runs)
            n_gt += len(g_sp)
            n_pred += len(p_sp)
            ov = pm.spell_overlaps(g_sp, p_sp)
            hit = pm.matched_gt_flags(ov, len(g_sp), 0.5)
            lens = np.array([(s[1] - s[0]) / 25.0 for s in g_sp]) if len(g_sp) else np.array([])
            for b in range(len(length_bins) - 1):
                sel = (lens >= length_bins[b]) & (lens < length_bins[b + 1])
                total[b] += sel.sum()
                found[b] += np.asarray(hit)[sel].sum()
    for k, v in res.items():
        print("{:<12} spellF1 {:.4f}  (halves {})".format(k, np.mean(v), len(v)))
    print("possessions: truth {}  predicted {}".format(n_gt, n_pred))
    for b in range(len(length_bins) - 1):
        print("  true length [{:>4}, {:>4}) s: n={:>5}  matched@0.5 {:.2f}".format(
            length_bins[b], length_bins[b + 1] if length_bins[b + 1] < 1e9 else "inf",
            int(total[b]), found[b] / max(total[b], 1)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "kaggle", "out"))
