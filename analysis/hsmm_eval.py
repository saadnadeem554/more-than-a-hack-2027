"""Offline test of semi-Markov decoding on held-out dumps (base + touch ensemble log-probs).

    python analysis/hsmm_eval.py kaggle/out external/train/features_v6_lab.npz

Duration distributions for fold k come from the training labels of the OTHER six matches.
Compared against the shipped flat-cost Viterbi (cost 2.0).
"""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from train_seq import viterbi_logp  # noqa: E402
from src import hsmm  # noqa: E402

MAX_LEN = 600  # steps = 240 s


def main(dump_dir, dataset):
    z = np.load(dataset, allow_pickle=True)
    Y, lens, match, rep = z["Y"], z["lens"], z["match"], z["rep"]
    ends = np.cumsum(lens)
    series_y = [Y[e - n:e] for e, n in zip(ends, lens)]
    folds = sorted(set(int(m) for m in match))
    grid = list(itertools.product((0.3, 0.5, 1.0), (0.5, 1.0, 2.0), (0.0, 2.0)))
    res = {g: [] for g in grid}
    base = []
    for f in folds:
        train_y = [series_y[i] for i in range(len(series_y)) if match[i] != f and rep[i] == 0]
        dur = hsmm.duration_logpmf(train_y, MAX_LEN)
        dumps = [np.load(os.path.join(dump_dir, "dump_{}_val{}.npz".format(a, f))) for a in ("base", "touch")]
        keys = sorted(k for k in dumps[0].files if k.startswith("logp_"))
        rows = {g: [] for g in grid}
        brow = []
        for key in keys:
            i = key.split("_")[1]
            logp = np.mean([d[key] for d in dumps], axis=0)
            yf = dumps[0]["yf_" + i]
            gt = np.where(yf >= 0, yf, 3).astype(np.int8)
            runs = pm.valid_runs(gt)
            score = lambda path: pm.spell_scores(gt, np.repeat(path, 10)[:len(gt)].astype(np.int8), runs)["mean_f1"]
            brow.append(score(viterbi_logp(logp, 2.0)))
            for g in grid:
                rows[g].append(score(hsmm.decode(logp, dur, *g)))
        base.append(np.mean(brow))
        for g in grid:
            res[g].append(np.mean(rows[g]))
        print("fold", f, "viterbi {:.4f}".format(base[-1]), flush=True)
    print("viterbi cost 2.0: {:.4f}".format(np.mean(base)))
    for g, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
        d = np.array(v) - np.array(base)
        print("hsmm emis={} dur={} switch={}: {:.4f}  vs viterbi {:+.4f}, better {}/{}".format(
            *g, np.mean(v), d.mean(), int((d > 0).sum()), len(d)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
