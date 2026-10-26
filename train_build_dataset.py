"""Build the training set from the external matches: degraded tracking -> step features + labels.

    python train_build_dataset.py [--reps 6] [--out external/train/features.npz]

Only external (CC-BY) tracking goes in; the competition data is not read.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src import external_sportec as ext  # noqa: E402
from src.degrade import degrade, degraded_team  # noqa: E402
from src.features import compute_features, step_labels, N_FEATURES, COLUMNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "external", "train", "features.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    X, Y, M, H, R = [], [], [], [], []
    for mi, mid in enumerate(ext.match_ids()):
        parts, pitch = ext.load_match(mid)
        for part in parts:
            y = step_labels(part["label"])
            for rep in range(args.reps):
                px, py, bx, by = degrade(part["px"], part["py"], part["bx"], part["by"], rng, pitch=pitch)
                X.append(compute_features(px, py, bx, by, degraded_team(part["team"]), pitch=pitch))
                Y.append(y); M.append(mi); H.append(int(part["half"][0])); R.append(rep)
            print(mid, "half", part["half"][0], "done", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lens = np.array([len(x) for x in X])
    np.savez_compressed(args.out, X=np.concatenate(X), Y=np.concatenate(Y), lens=lens,
                        match=np.array(M), half=np.array(H), rep=np.array(R),
                        columns=np.array(COLUMNS))
    print("saved", args.out, "rows", int(lens.sum()), "features", N_FEATURES)


if __name__ == "__main__":
    main()
