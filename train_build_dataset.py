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
from src.features import compute_features, step_labels, columns  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "external", "train", "features.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-orient", action="store_true", help="omit the direction-aware features")
    ap.add_argument("--touch", action="store_true", help="add the touch features")
    ap.add_argument("--random-gaps", action="store_true", help="context-free ball gaps (older degradation)")
    ap.add_argument("--metrica", action="store_true",
                    help="append Metrica sample games 1-2 (match indices after the DFL ones)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    X, Y, M, H, R, YF = [], [], [], [], [], []
    sources = [(lambda m=mid: ext.load_match(m), mid) for mid in ext.match_ids()]
    if args.metrica:
        from src import external_metrica as met
        sources += [(lambda g=g: met.load_game(g), "metrica%d" % g) for g in met.GAMES]
    for mi, (loader, mid) in enumerate(sources):
        parts, pitch = loader()
        for part in parts:
            y = step_labels(part["label"])
            for rep in range(args.reps):
                px, py, bx, by = degrade(part["px"], part["py"], part["bx"], part["by"], rng, pitch=pitch,
                                         context=not args.random_gaps)
                X.append(compute_features(px, py, bx, by, degraded_team(part["team"]), pitch=pitch,
                                          orient=not args.no_orient, touch=args.touch))
                Y.append(y); M.append(mi); H.append(int(part["half"][0])); R.append(rep)
                YF.append(part["label"].astype(np.int8))
            print(mid, "half", part["half"][0], "done", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lens = np.array([len(x) for x in X])
    np.savez_compressed(args.out, X=np.concatenate(X), Y=np.concatenate(Y), lens=lens,
                        YF=np.concatenate(YF), flens=np.array([len(y) for y in YF]),
                        match=np.array(M), half=np.array(H), rep=np.array(R),
                        columns=np.array(columns(not args.no_orient, args.touch)))
    print("saved", args.out, "rows", int(lens.sum()), "features", len(columns(not args.no_orient, args.touch)))


if __name__ == "__main__":
    main()
