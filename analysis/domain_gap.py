"""Compare step-feature distributions: competition ATD vs external (clean and degraded).

Prints per-feature quantiles only. A large gap means the degradation does not yet look like the
real automatic tracking, so a model trained on it will be miscalibrated on the competition.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from src.atd import load_atd  # noqa: E402
from src import external_sportec as ext  # noqa: E402
from src.degrade import degrade, degraded_team  # noqa: E402
from src.features import compute_features, COLUMNS  # noqa: E402

Q = (10, 50, 90)


def quant(x):
    return np.percentile(x, Q, axis=0)


def main():
    atd = load_atd()
    ours = np.concatenate([compute_features(atd.px[a:b + 1], atd.py[a:b + 1], atd.bx[a:b + 1],
                                            atd.by[a:b + 1], atd.team, pitch=atd.pitch)
                           for _, a, b in half_video_blocks("BAR-ATH")])
    rng = np.random.default_rng(0)
    clean, deg = [], []
    for mid in ext.match_ids()[:3]:
        parts, pitch = ext.load_match(mid)
        for p in parts:
            clean.append(compute_features(p["px"], p["py"], p["bx"], p["by"], p["team"], pitch=pitch))
            px, py, bx, by = degrade(p["px"], p["py"], p["bx"], p["by"], rng, pitch=pitch)
            deg.append(compute_features(px, py, bx, by, degraded_team(p["team"]), pitch=pitch))
    clean, deg = np.concatenate(clean), np.concatenate(deg)
    qo, qc, qd = quant(ours), quant(clean), quant(deg)
    print("{:<11} {:>22} {:>22} {:>22}".format("feature", "ours p10/50/90", "ext clean", "ext degraded"))
    for j, c in enumerate(COLUMNS):
        f = lambda q: "/".join("{:.2f}".format(v) for v in q[:, j])
        print("{:<11} {:>22} {:>22} {:>22}".format(c, f(qo), f(qc), f(qd)))
    # visible-only ball-to-nearest-player distance (the most important cue)
    vo, vd = ours[:, 0] > 0.99, deg[:, 0] > 0.99
    for name, X, v in (("ours", ours, vo), ("degraded", deg, vd)):
        near = np.minimum(X[v, COLUMNS.index("h_near1")], X[v, COLUMNS.index("a_near1")])
        print("visible-step nearest player {:<9} p10/50/90/99 {}".format(
            name, np.round(np.percentile(near, (10, 50, 90, 99)), 2)))


if __name__ == "__main__":
    main()
