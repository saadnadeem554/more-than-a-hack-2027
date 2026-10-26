"""Possession decoder v1: hand-built evidence + Viterbi over {H, A, D}.

Nothing here is fitted to the competition match's labels. Every constant is a football prior
chosen by reasoning, and is named in PARAMS so that the technical document can list them.

Pipeline
  1. per-frame evidence from the tracking (who is near the ball, is the game stopped?)
  2. per-frame evidence from Smart Tagging (phase team tags, set-piece windows)
  3. average to 0.4 s steps, decode each half with Viterbi (switching costs = possession inertia)
  4. merge runs shorter than a minimum duration, write 'X' outside the graded halves
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import half_video_blocks  # noqa: E402
import load_smart_tagging  # noqa: E402

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

PARAMS = dict(
    step=10,                 # frames per decoding step (0.4 s)
    tau=2.0,                 # m, distance scale of a player's influence on the ball
    ctrl_clip=4.0,           # max |log-odds| of team control from one frame
    interp_tau=50.0,         # frames; ball confidence e-folding time away from an observation
    ctrl_far=12.0,           # m, ball further than this from every player gives no team evidence
    gain_team=1.5,           # nats per unit control log-odds
    gain_dead=1.0,           # nats per unit dead evidence
    dead_bias=1.0,           # nats against D when there is no evidence either way (ball is in play ~55-60% of a match)
    speed_k=12,              # frames either side for player/ball speed (+/- 0.5 s)
    slow_speed=1.8,          # m/s, median player speed below which the game looks stopped
    slow_scale=0.8,
    edge_out=2.0,            # dead evidence when the ball is outside the pitch rectangle
    edge_near_m=2.0,         # m from the line
    edge_near=0.8,
    edge_far=-0.2,
    setpiece_secs=12.0,      # dead evidence in [window start, start + this]
    setpiece=1.0,
    tag_w=0.3,               # nats from a phase tag's team
    cost_switch=8.0,         # H <-> A
    cost_dead_in=6.0,        # alive -> D
    cost_dead_out=6.0,       # D -> alive
    min_alive=50,            # frames; shorter alive runs are merged into neighbours
    min_dead=50,
)

ATTACK_CODES = {"build up", "progression", "creating chances", "ball in final third",
                "ball in the box", "players in the box", "attacking transition", "long ball"}
DEFEND_CODES = {"defending in attacking third", "defending in middle third",
                "defending in defensive third", "defensive transition"}
TEAM_NAMES = {"barcelona": H, "atletic club": A, "athletic club": A}


def fill_ball(atd, p):
    """Ball track with gaps linearly interpolated, plus a confidence in [0, 1].

    Confidence decays with the time to the nearest real observation, so a long occlusion gives
    weight to its edges and almost none to its middle.
    """
    vis = np.isfinite(atd.bx)
    idx = np.arange(atd.n_frames)
    if not vis.any():
        return atd.bx, atd.by, np.zeros(atd.n_frames)
    bx = np.interp(idx, idx[vis], atd.bx[vis])
    by = np.interp(idx, idx[vis], atd.by[vis])
    last = np.maximum.accumulate(np.where(vis, idx, -10**9))
    nxt = np.minimum.accumulate(np.where(vis, idx, 10**9)[::-1])[::-1]
    gap = np.minimum(idx - last, nxt - idx).astype(float)
    return bx, by, np.exp(-gap / p["interp_tau"])


def control_logit(atd, p):
    """Per-frame log-odds that the home team controls the ball, weighted by ball confidence."""
    L, W = atd.pitch
    bx, by, conf = fill_ball(atd, p)
    dist = np.hypot((atd.px - bx[:, None]) * L, (atd.py - by[:, None]) * W)
    w = np.exp(-dist / p["tau"])
    w = np.where(np.isfinite(w), w, 0.0)
    home = w[:, atd.team == 0].sum(1)
    away = w[:, atd.team == 1].sum(1)
    c = np.log(home + 1e-3) - np.log(away + 1e-3)
    nearest = np.where(np.isfinite(dist), dist, np.inf).min(1)
    c = np.clip(c, -p["ctrl_clip"], p["ctrl_clip"]) * conf
    c[nearest > p["ctrl_far"]] = 0.0
    return c


def player_speed(atd, p):
    """Per-frame median speed (m/s) over tracked slots, from positions +/- speed_k frames."""
    k = p["speed_k"]
    L, W = atd.pitch
    n = atd.n_frames
    out = np.full(n, np.nan)
    d = np.hypot((atd.px[2 * k:] - atd.px[:-2 * k]) * L, (atd.py[2 * k:] - atd.py[:-2 * k]) * W)
    d = d / (2 * k / 25.0)
    d[d > 12.0] = np.nan  # identity switches / glitches
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out[k:n - k] = np.nanmedian(d, axis=1)
    return out


def dead_evidence(atd, p, speed):
    """Per-frame evidence (positive = ball dead) from player slowness and the ball at the line."""
    L, W = atd.pitch
    e = np.zeros(atd.n_frames)
    s = np.where(np.isfinite(speed), np.clip((p["slow_speed"] - speed) / p["slow_scale"], -1.5, 1.5), 0.0)
    e += s
    edge = np.minimum.reduce([atd.bx * L, (1 - atd.bx) * L, atd.by * W, (1 - atd.by) * W])
    vis = np.isfinite(edge)
    e[vis & (edge < 0)] += p["edge_out"]
    e[vis & (edge >= 0) & (edge < p["edge_near_m"])] += p["edge_near"]
    e[vis & (edge >= p["edge_near_m"])] += p["edge_far"]
    return e


def tag_evidence(n, p):
    """(team_logit, setpiece_dead) per frame from Smart Tagging."""
    votes_h = np.zeros(n)
    votes_a = np.zeros(n)
    setp = np.zeros(n)
    for inst in load_smart_tagging.load():
        code = inst["code"].strip().lower()
        a, b = max(inst["start_frame"], 0), min(inst["end_frame"], n)
        team = TEAM_NAMES.get((inst["tags"].get("Team") or "").strip().lower())
        if code == "set pieces":
            typ = (inst["tags"].get("Type") or "").lower()
            if not typ.startswith("kick off"):
                setp[a:min(b, a + int(p["setpiece_secs"] * 25))] = 1.0
            continue
        if team is None:
            continue
        has_ball = team if code in ATTACK_CODES else (1 - team) if code in DEFEND_CODES else None
        if has_ball is None:
            continue
        (votes_h if has_ball == H else votes_a)[a:b] += 1.0
    tot = votes_h + votes_a
    net = np.where(tot > 0, (votes_h - votes_a) / np.maximum(tot, 1), 0.0)
    return net, setp


def build_emissions(atd, p=PARAMS, use_tags=True):
    """(n_frames, 3) log-scores for H, A, D."""
    c = control_logit(atd, p)
    speed = player_speed(atd, p)
    de = dead_evidence(atd, p, speed)
    net, setp = tag_evidence(atd.n_frames, p) if use_tags else (np.zeros(atd.n_frames),) * 2
    dead = p["gain_dead"] * (de + p["setpiece"] * setp) - p["dead_bias"]
    team = p["gain_team"] * c + p["tag_w"] * net
    em = np.empty((atd.n_frames, 3))
    em[:, H] = -0.5 * dead + 0.5 * team
    em[:, A] = -0.5 * dead - 0.5 * team
    em[:, D] = dead
    return em


def viterbi(em, cost_switch, cost_in, cost_out):
    """Max-score path over 3 states with a transition-cost matrix."""
    n = len(em)
    trans = np.array([[0, -cost_switch, -cost_in],
                      [-cost_switch, 0, -cost_in],
                      [-cost_out, -cost_out, 0]], dtype=float)
    score = np.empty((n, 3))
    back = np.zeros((n, 3), dtype=np.int8)
    score[0] = em[0]
    for t in range(1, n):
        cand = score[t - 1][:, None] + trans          # from x to
        back[t] = cand.argmax(0)
        score[t] = cand.max(0) + em[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def merge_short(labels, min_alive, min_dead):
    """Absorb runs shorter than the minimum into the neighbouring run that is longer."""
    labels = labels.copy()
    for _ in range(10):
        change = np.flatnonzero(np.diff(labels)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(labels)]))
        lens = ends - starts
        changed = False
        for i in np.argsort(lens):
            lab = labels[starts[i]]
            need = min_dead if lab == D else min_alive
            if lens[i] >= need or len(starts) == 1:
                continue
            left = lens[i - 1] if i > 0 else -1
            right = lens[i + 1] if i + 1 < len(lens) else -1
            if max(left, right) < 0:
                continue
            labels[starts[i]:ends[i]] = labels[starts[i - 1]] if left >= right else labels[starts[i + 1]]
            changed = True
            break
        if not changed:
            break
    return labels


def decode_block(em, p=PARAMS):
    """Frame labels (0 H, 1 A, 2 D) for one contiguous block of frame emissions."""
    s = p["step"]
    n = len(em)
    steps = -(-n // s)
    pad = np.zeros((steps * s, 3))
    pad[:n] = em
    pad[n:] = pad[n - 1]
    step_em = pad.reshape(steps, s, 3).mean(1)
    path = viterbi(step_em, p["cost_switch"], p["cost_dead_in"], p["cost_dead_out"])
    return merge_short(np.repeat(path, s)[:n], p["min_alive"], p["min_dead"])


def decode(atd, p=PARAMS):
    """Full-length array of letters ('H','A','D','X')."""
    em = build_emissions(atd, p)
    out = np.full(atd.n_frames, "X", dtype="<U1")
    for half, first, last in half_video_blocks("BAR-ATH"):
        out[first:last + 1] = LETTERS[decode_block(em[first:last + 1], p)]
    # FAQ.md: the second half is graded from 110,808 but kicks off at 113,742; all of it is D.
    out[110808:113742] = "D"
    return out
