"""Score ensembles of held-out predictions offline (from train_seq.py --dump files).

    python analysis/ensemble_eval.py kaggle/out base touch          # each alone + all together
    python analysis/ensemble_eval.py kaggle/out base:tta touch:tta  # ":tta" = team-swap TTA

For each fold, averages the step log-probs of the named arms, decodes with Viterbi at several
switching costs, and scores spellF1 against the frame labels stored in the dump.
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

COSTS = (1.0, 2.0, 3.0, 4.0, 6.0)


def fold_scores(folder, arms, fold):
    names = [a.split(":")[0] for a in arms]
    tta = [a.endswith(":tta") for a in arms]
    dumps = [np.load(os.path.join(folder, "dump_{}_val{}.npz".format(a, fold))) for a in names]
    keys = sorted(k for k in dumps[0].files if k.startswith("logp_"))
    out = {c: [] for c in COSTS}
    for key in keys:
        i = key.split("_")[1]
        member = [0.5 * (d[key] + d["logpS_" + i]) if t else d[key] for d, t in zip(dumps, tta)]
        logp = np.mean(member, axis=0)
        yf = dumps[0]["yf_" + i]
        gt = np.where(yf >= 0, yf, 3).astype(np.int8)
        runs = pm.valid_runs(gt)
        for c in COSTS:
            pred = np.repeat(viterbi_logp(logp, c), 10)[:len(gt)].astype(np.int8)
            out[c].append(pm.spell_scores(gt, pred, runs)["mean_f1"])
    return {c: float(np.mean(v)) for c, v in out.items()}


def main(folder, groups):
    folds = sorted(int(p.rsplit("val", 1)[1].split(".")[0])
                   for p in glob.glob(os.path.join(folder, "dump_{}_val*.npz".format(groups[0][0].split(":")[0]))))
    table = {}
    for arms in groups:
        per = [fold_scores(folder, arms, f) for f in folds]
        table["+".join(arms)] = per
        means = {c: np.mean([p[c] for p in per]) for c in COSTS}
        best = max(means, key=means.get)
        print("{:<12} ".format("+".join(arms)) + " ".join("c={}:{:.4f}".format(c, means[c]) for c in COSTS)
              + "   best {:.4f}@{}".format(means[best], best))
    return table


if __name__ == "__main__":
    folder = sys.argv[1]
    arms = sys.argv[2:]
    groups = [[a] for a in arms] + ([arms] if len(arms) > 1 else [])
    t = main(folder, groups)
    if len(arms) > 1:
        ens, base = t["+".join(arms)], t[arms[0]]
        for c in COSTS:
            d = [e[c] - b[c] for e, b in zip(ens, base)]
            print("ensemble vs {} @cost {}: mean {:+.4f}, better on {}/{}".format(
                arms[0], c, np.mean(d), sum(x > 0 for x in d), len(d)))
