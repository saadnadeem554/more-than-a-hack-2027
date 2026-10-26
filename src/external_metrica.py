"""Parse Metrica Sports' open sample games 1 and 2 (tracking + events) into the external format.

Source: https://github.com/metrica-sports/sample-data (anonymised; "please acknowledge the source").
Tracking: 25 fps, normalised 0-1 like the competition ATD, one CSV per team, ball in the home file.

These games have no per-frame ball status, so labels are derived from the synchronised events:
  dead  from a BALL OUT (its end frame), a FAULT RECEIVED, or a goal, until the next restart
        event (the next event of any other type);
  alive the team of the latest controlling event (PASS, SET PIECE, SHOT, RECOVERY, BALL LOST),
        unchanged by CHALLENGE -- contested and loose balls stay with the last team in control,
        as the competition rules state.
Frames before the first event of a period are left unlabelled (-1).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET_DIR = os.path.join(ROOT, "external", "metrica")
GAMES = (1, 2)
CONTROL = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST"}
STOP = {"BALL OUT", "FAULT RECEIVED"}
IGNORE = {"CARD", "CHALLENGE"}


def _tracking(game, side):
    path = os.path.join(MET_DIR, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
    df = pd.read_csv(path, skiprows=2)
    names = list(df.columns)
    players = [c for c in names if c.startswith("Player")]
    xs, ys = [], []
    for c in players:
        i = names.index(c)
        xs.append(df.iloc[:, i].values)
        ys.append(df.iloc[:, i + 1].values)
    ball = None
    if "Ball" in names:
        i = names.index("Ball")
        ball = (df.iloc[:, i].values, df.iloc[:, i + 1].values)
    return df["Period"].values, df["Frame"].values, np.stack(xs, 1), np.stack(ys, 1), ball


def _labels(game, frames, period):
    ev = pd.read_csv(os.path.join(MET_DIR, "Sample_Game_{}_RawEventsData.csv".format(game)))
    n = len(frames)
    label = np.full(n, -1, dtype=np.int8)
    first = int(frames[0])
    for p in (1, 2):
        e = ev[ev["Period"] == p].sort_values("Start Frame")
        in_p = np.flatnonzero(period == p)
        if not len(in_p) or e.empty:
            continue
        lo, hi = in_p[0], in_p[-1]
        cur, dead = -1, False
        k0 = int(e["Start Frame"].iloc[0]) - first
        rows = e.to_dict("records")
        marks = []  # (frame index, state) where state is 0/1 team or 2 dead
        for r in rows:
            t, sub = r["Type"], str(r["Subtype"])
            team = 0 if r["Team"] == "Home" else 1
            s, f_end = int(r["Start Frame"]) - first, int(r["End Frame"]) - first
            if t in IGNORE:
                continue
            if t in STOP:
                marks.append((f_end if t == "BALL OUT" else s, 2))
                continue
            if t == "SHOT" and "GOAL" in sub and "OUT" not in sub:
                marks.append((s, team))
                marks.append((f_end, 2))
                continue
            if t in CONTROL:
                marks.append((s, team))
        marks.sort(key=lambda m: m[0])
        state, pos = -1, max(k0, lo)
        for idx, st in marks:
            idx = min(max(idx, lo), hi + 1)
            if idx > pos and state >= 0:
                label[pos:idx] = state
            if idx >= pos:
                pos = idx
            state = st
        if state >= 0:
            label[pos:hi + 1] = state
    return label


def load_game(game):
    period, frames, hx, hy, ball = _tracking(game, "Home")
    _, _, ax, ay, _ = _tracking(game, "Away")
    px = np.concatenate([hx, ax], 1).astype(np.float32)
    py = np.concatenate([hy, ay], 1).astype(np.float32)
    team = np.array([0] * hx.shape[1] + [1] * ax.shape[1], dtype=np.int8)
    bx, by = (b.astype(np.float32) for b in ball)
    label = _labels(game, frames, period)
    parts = []
    for p in (1, 2):
        m = period == p
        parts.append(dict(px=px[m], py=py[m], team=team, bx=bx[m], by=by[m], label=label[m],
                          half=np.full(int(m.sum()), p, dtype=np.int8)))
    return parts, (105.0, 68.0)


if __name__ == "__main__":
    for g in GAMES:
        parts, _ = load_game(g)
        for p in parts:
            lab, vis = p["label"], np.isfinite(p["bx"])
            print("game", g, "half", p["half"][0], "frames", len(lab),
                  "H {:.2f} A {:.2f} D {:.2f} unl {:.3f}".format(*[(lab == k).mean() for k in (0, 1, 2, -1)]),
                  "ball vis: alive {:.2f} dead {:.2f}".format(vis[lab <= 1].mean(), vis[lab == 2].mean()),
                  "dead runs", int(np.sum(np.diff((lab == 2).astype(int)) == 1)))
