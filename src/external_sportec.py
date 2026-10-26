"""Parse one DFL/Sportec open-data match (CC-BY 4.0) into dense arrays plus frame labels.

Source: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data, https://doi.org/10.6084/m9.figshare.28196177 .

Labels come from the provider's own per-frame fields on the ball:
    BallStatus 0 -> dead ball (D);  BallStatus 1 + BallPossession 1 -> home (H), 2 -> away (A).
That is the same definition the organisers use for the competition ground truth.

Output coordinates are normalised to 0-1 exactly like the competition ATD (centre-origin metres
divided by the pitch size, shifted by 0.5), so the same feature code runs on both.
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(ROOT, "external", "sportec")
CACHE_DIR = os.path.join(ROOT, "external", "cache")

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
_POSS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def match_ids():
    return sorted(re.search(r"(DFL-MAT-\w+)\.xml$", p).group(1)
                  for p in glob.glob(os.path.join(EXT_DIR, "DFL_04_03_positions_raw_observed_*.xml")))


def _find(prefix, match_id):
    hits = glob.glob(os.path.join(EXT_DIR, "{}*{}.xml".format(prefix, match_id)))
    if not hits:
        raise IOError("missing {} for {}".format(prefix, match_id))
    return hits[0]


def _team_roles(match_id):
    root = ET.parse(_find("DFL_02_01_matchinformation", match_id)).getroot()
    roles = {t.get("TeamId"): t.get("Role") for t in root.iter("Team")}
    pitch = root.find(".//Environment")
    return roles, float(pitch.get("PitchX")), float(pitch.get("PitchY"))


def parse_match(match_id):
    """dict: px, py (n, P), team (P,) 0 home / 1 away, bx, by (n,), label (n,) 0 H / 1 A / 2 D / -1,
    half (n,) 1 or 2. Frames are the concatenated halves, in play order."""
    roles, pitch_x, pitch_y = _team_roles(match_id)
    sections = {}  # section -> {"ball": {n: (x, y, poss, status)}, "players": {pid: (team, {n: (x, y)})}}
    cur = None
    with open(_find("DFL_04_03_positions_raw_observed", match_id), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                m = _FRAMESET.match(line)
                sec, team_id, pid = m.groups()
                s = sections.setdefault(sec, {"ball": {}, "players": {}})
                if team_id == "BALL":
                    cur = ("ball", s["ball"])
                elif team_id in roles:
                    team = 0 if roles[team_id] == "home" else 1
                    d = {}
                    s["players"][pid] = (team, d)
                    cur = ("player", d)
                else:  # referee
                    cur = None
            elif line.startswith("<Frame") and cur is not None:
                m = _FRAME.match(line)
                if m is None:
                    continue
                n, x, y = m.groups()
                n = int(n)
                if cur[0] == "ball":
                    q = _POSS.search(line)
                    poss, status = (int(q.group(1)), int(q.group(2))) if q else (0, 1)
                    cur[1][n] = (float(x), float(y), poss, status)
                else:
                    cur[1][n] = (float(x), float(y))

    parts = []
    for half, sec in ((1, "firstHalf"), (2, "secondHalf")):
        s = sections[sec]
        frames = sorted(s["ball"])
        lo, hi = frames[0], frames[-1]
        n = hi - lo + 1
        pids = sorted(s["players"])
        px = np.full((n, len(pids)), np.nan, dtype=np.float32)
        py = px.copy()
        team = np.array([s["players"][p][0] for p in pids], dtype=np.int8)
        for j, p in enumerate(pids):
            for f, (x, y) in s["players"][p][1].items():
                if lo <= f <= hi:
                    px[f - lo, j] = x / pitch_x + 0.5
                    py[f - lo, j] = y / pitch_y + 0.5
        bx = np.full(n, np.nan, dtype=np.float32)
        by = bx.copy()
        label = np.full(n, -1, dtype=np.int8)
        for f, (x, y, poss, status) in s["ball"].items():
            i = f - lo
            bx[i] = x / pitch_x + 0.5
            by[i] = y / pitch_y + 0.5
            label[i] = 2 if status == 0 else (0 if poss == 1 else 1 if poss == 2 else -1)
        parts.append(dict(px=px, py=py, team=team, bx=bx, by=by, label=label,
                          half=np.full(n, half, dtype=np.int8)))
    return parts, (pitch_x, pitch_y)


def load_match(match_id, rebuild=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, match_id + ".npz")
    if rebuild or not os.path.isfile(path):
        parts, pitch = parse_match(match_id)
        out = {}
        for i, part in enumerate(parts):
            for k, v in part.items():
                out["h{}_{}".format(i + 1, k)] = v
        out["pitch"] = np.array(pitch)
        np.savez_compressed(path, **out)
    z = np.load(path)
    return [{k: z["h{}_{}".format(h, k)] for k in ("px", "py", "team", "bx", "by", "label", "half")}
            for h in (1, 2)], tuple(z["pitch"])


if __name__ == "__main__":
    for mid in match_ids():
        parts, pitch = load_match(mid)
        for p in parts:
            lab = p["label"]
            vis = np.isfinite(p["bx"]).mean()
            print(mid, "half", p["half"][0], "frames", len(lab), "ball_vis {:.3f}".format(vis),
                  "H {:.2f} A {:.2f} D {:.2f} unk {:.3f}".format(*[(lab == k).mean() for k in (0, 1, 2, -1)]),
                  "players", p["px"].shape[1])
