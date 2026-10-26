"""Touch-feature distributions: competition ATD vs degraded external (aggregate quantiles only)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks
from src.atd import load_atd
from src import external_sportec as ext
from src.degrade import degrade, degraded_team
from src.features import compute_features, columns

cols = columns(True, True)
atd = load_atd()
_, a, b = half_video_blocks("BAR-ATH")[0]
ours = compute_features(atd.px[a:b + 1], atd.py[a:b + 1], atd.bx[a:b + 1], atd.by[a:b + 1], atd.team,
                        pitch=atd.pitch, touch=True)
rng = np.random.default_rng(0)
parts, pitch = ext.load_match(ext.match_ids()[0])
p = parts[0]
px, py, bx, by = degrade(p["px"], p["py"], p["bx"], p["by"], rng, pitch=pitch)
deg = compute_features(px, py, bx, by, degraded_team(p["team"]), pitch=pitch, touch=True)
clean = compute_features(p["px"], p["py"], p["bx"], p["by"], p["team"], pitch=pitch, touch=True)
for name in ("ball_acc", "touch_ctrl_s", "ball_vx_s", "h_appr", "a_appr"):
    j = cols.index(name)
    q = lambda X: np.round(np.percentile(X[:, j], (10, 50, 90, 99)), 2).tolist()
    print("{:<13} ours {}  degraded {}  clean {}".format(name, q(ours), q(deg), q(clean)))
