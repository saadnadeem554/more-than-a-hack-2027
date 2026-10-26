"""Phase 1: aggregate-only look at the tracking. Prints summary statistics, never raw rows.

    python analysis/explore_atd.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import half_video_blocks  # noqa: E402
from src.atd import load_atd  # noqa: E402
import load_smart_tagging  # noqa: E402


def runs(mask):
    """(start, length) of each run of True."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e - s


def pct(a, q=(5, 25, 50, 75, 95)):
    a = a[np.isfinite(a)]
    return {k: round(float(np.percentile(a, k)), 3) for k in q} if len(a) else {}


def main():
    atd = load_atd()
    n = atd.n_frames
    L, W = atd.pitch
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks("BAR-ATH"):
        graded[a:b + 1] = True

    vis = np.isfinite(atd.bx)
    print("ball visible: all {:.1%}  graded {:.1%}".format(vis.mean(), vis[graded].mean()))

    # --- gaps in the ball track (graded only)
    for name, mask in (("missing", ~vis & graded), ("visible", vis & graded)):
        s, l = runs(mask)
        print("{} runs: n={} len(frames) pct={}".format(name, len(l), pct(l.astype(float))))

    # --- ball geometry when visible
    inside = (atd.bx >= 0) & (atd.bx <= 1) & (atd.by >= 0) & (atd.by <= 1)
    g = vis & graded
    print("ball visible but outside pitch rect: {:.2%} of visible graded".format(
        (~inside[g]).mean()))
    sp = atd.bs * 1.0
    print("ball speed (tracker units, m/s?) pct:", pct(sp[g]))

    # --- player counts per frame
    cnt = np.isfinite(atd.px).sum(1)
    print("tracked players/frame graded pct:", pct(cnt[graded].astype(float)))
    home = np.isfinite(atd.px[:, atd.team == 0]).sum(1)
    away = np.isfinite(atd.px[:, atd.team == 1]).sum(1)
    print("home/away tracked mean graded: {:.1f} / {:.1f}".format(
        home[graded].mean(), away[graded].mean()))

    # --- Smart Tagging structure
    st = load_smart_tagging.load()
    sp_ = [i for i in st if i["code"].lower() == "set pieces"]
    print("set pieces:", len(sp_))
    types = {}
    for i in sp_:
        t = i["tags"].get("Type", "?")
        types.setdefault(t, []).append(i)
    for t, items in sorted(types.items()):
        dur = np.array([(i["end_frame"] - i["start_frame"]) / 25 for i in items])
        ballvis = np.mean([vis[i["start_frame"]:i["end_frame"]].mean() for i in items])
        print("  {:<24} n={:>3} dur_s pct={} ball_vis={:.2f}".format(
            t, len(items), pct(dur), ballvis))

    # --- public ground truth (explicitly allowed, 5 minutes only)
    import pandas as pd
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    lab = gt["label"].values
    m = lab != "X"
    for c in "HAD":
        k = m & (lab == c)
        print("public GT {}: n={} ball_vis={:.2f} outside_rect={:.2f} speed={}".format(
            c, k.sum(), vis[k].mean(), (~inside[k & vis]).mean() if (k & vis).any() else float("nan"),
            pct(sp[k & vis], (25, 50, 75))))
    s, l = runs((lab == "D"))
    print("public GT dead stretches (s):", np.round(l / 25, 1).tolist())


if __name__ == "__main__":
    main()
