"""Write a copy of a training dataset with transformed labels (features untouched).

    python relabel_dataset.py IN.npz OUT.npz --train-merge 3 --eval-merge 3

--train-merge sets the step labels the model learns from (Y); --eval-merge sets the frame labels
held-out matches are scored against (YF). Both apply src/labels.merge_sandwiched per series.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.features import step_labels  # noqa: E402
from src.labels import merge_sandwiched  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--train-merge", type=float, default=0.0, help="seconds")
    ap.add_argument("--eval-merge", type=float, default=0.0, help="seconds")
    args = ap.parse_args()
    z = np.load(args.inp, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    fl = data["flens"]
    fe = np.cumsum(fl)
    ys, yfs = [], []
    for s, e, n_steps in zip(fe - fl, fe, data["lens"]):
        yf = data["YF"][s:e]
        y = step_labels(merge_sandwiched(yf, int(args.train_merge * 25)))
        assert len(y) == n_steps
        ys.append(y)
        yfs.append(merge_sandwiched(yf, int(args.eval_merge * 25)).astype(np.int8))
    data["Y"] = np.concatenate(ys)
    data["YF"] = np.concatenate(yfs)
    np.savez_compressed(args.out, **data)
    print("saved", args.out)


if __name__ == "__main__":
    main()
