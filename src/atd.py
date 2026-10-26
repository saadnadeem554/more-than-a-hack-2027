"""Load the tactical ATD into dense numpy arrays indexed by VIDEO frame.

    from src.atd import load_atd
    atd = load_atd()
    atd.px, atd.py, atd.ps   # (n_frames, 46) player x, y (normalised 0-1) and speed, NaN = missing
    atd.bx, atd.by, atd.bs   # (n_frames,) ball
    atd.team                 # (46,) 0 = home, 1 = away

The parse is cached as a .npz under data/cache/ (gitignored with the rest of data/), so the
~1 minute text parse happens once.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

N_SLOTS = 46


def _parse(raw_path, n_frames):
    px = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    py = px.copy()
    ps = px.copy()
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    def triple(group):
        if group[0] == "N":
            return None
        a, b, c = group.split(",")
        return float(a), float(b), float(c)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            for slot, group in enumerate(parts[1].split(";")):
                t = triple(group)
                if t is not None:
                    px[f, slot], py[f, slot], ps[f, slot] = t
            t = triple(parts[2].strip())
            if t is not None:
                ball[f] = t
    return px, py, ps, ball


def load_atd(game=GAME, data_dir=None, rebuild=False):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    cache_dir = os.path.join(data_dir or DATA_DIR, "cache")
    cache = os.path.join(cache_dir, "atd_tactical.npz")

    if rebuild or not os.path.isfile(cache):
        px, py, ps, ball = _parse(raw_path, n_frames)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, px=px, py=py, ps=ps, ball=ball)
    z = np.load(cache)
    return SimpleNamespace(
        px=z["px"], py=z["py"], ps=z["ps"],
        bx=z["ball"][:, 0], by=z["ball"][:, 1], bs=z["ball"][:, 2],
        team=team_of_each_track(meta_path),
        pitch=GAMES[game]["pitch"], n_frames=n_frames,
    )
