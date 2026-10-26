"""Which ball smoothing window / noise makes external ball_acc look like the competition's?"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks
from src.atd import load_atd
from src import external_sportec as ext
from src.degrade import smooth_ball
from src.features import compute_features, columns

j = columns(True, True).index("ball_acc")
atd = load_atd()
_, a, b = half_video_blocks("BAR-ATH")[0]
ours = compute_features(atd.px[a:b + 1], atd.py[a:b + 1], atd.bx[a:b + 1], atd.by[a:b + 1], atd.team,
                        pitch=atd.pitch, touch=True)[:, j]
print("ours            ", np.round(np.percentile(ours, (50, 90, 99)), 2).tolist())
parts, pitch = ext.load_match(ext.match_ids()[0])
p = parts[0]
rng = np.random.default_rng(0)
for w in (1, 5, 9, 15, 25):
    for noise in (0.0, 0.1):
        bx, by = smooth_ball(p["bx"], p["by"], w)
        bx = bx + rng.normal(0, noise / pitch[0], len(bx))
        by = by + rng.normal(0, noise / pitch[1], len(by))
        f = compute_features(p["px"], p["py"], bx, by, p["team"], pitch=pitch, touch=True)[:, j]
        print("w={:>2} noise={:.1f}  ".format(w, noise), np.round(np.percentile(f, (50, 90, 99)), 2).tolist())
