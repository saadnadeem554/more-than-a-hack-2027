"""The inferred home attacking direction must flip between halves, on every match."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks
from src.atd import load_atd
from src import external_sportec as ext
from src.degrade import degrade, degraded_team
from src.features import home_attack_sign
import numpy as np

atd = load_atd()
print("BAR-ATH", [home_attack_sign(atd.px[a:b + 1], atd.team) for _, a, b in half_video_blocks("BAR-ATH")])
rng = np.random.default_rng(0)
for mid in ext.match_ids():
    parts, pitch = ext.load_match(mid)
    clean = [home_attack_sign(p["px"], p["team"]) for p in parts]
    deg = []
    for p in parts:
        px, py, bx, by = degrade(p["px"], p["py"], p["bx"], p["by"], rng, pitch=pitch)
        deg.append(home_attack_sign(px, degraded_team(p["team"])))
    print(mid, "clean", clean, "degraded", deg)
