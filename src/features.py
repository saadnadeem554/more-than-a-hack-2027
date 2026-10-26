"""Per-step features from tracking, identical for the competition ATD and the external matches.

Inputs are normalised 0-1 coordinates (players: n x P, ball: n) and a team array (0 home, 1 away).
Everything is identity-free and orientation-free, so a model trained on other matches transfers.
Output rows are 0.4 s steps (STEP frames), averaged over the step.

Column layout (see COLUMNS): shared block, then a block per team. swap_teams() exchanges the two
team blocks and negates the control logit, which is the augmentation that stops a model from
learning "home" as a prior.
"""

import numpy as np

STEP = 10
PITCH = (105.0, 68.0)
INTERP_TAU = 50.0  # frames; ball confidence e-folding time away from a real observation

TEAM_COLS = ["near1", "near2", "near3", "n5", "n10", "tracked", "sx", "sy"]
SHARED = ["vis", "conf", "ball_ax", "ball_ay", "edge", "ball_v", "ctrl", "pspeed"]
COLUMNS = SHARED + ["h_" + c for c in TEAM_COLS] + ["a_" + c for c in TEAM_COLS]
N_FEATURES = len(COLUMNS)
_CTRL = COLUMNS.index("ctrl")
_HOME = [COLUMNS.index("h_" + c) for c in TEAM_COLS]
_AWAY = [COLUMNS.index("a_" + c) for c in TEAM_COLS]


def swap_teams(x):
    """Copy of a (..., N_FEATURES) array with the home and away blocks exchanged."""
    y = x.copy()
    y[..., _HOME], y[..., _AWAY] = x[..., _AWAY], x[..., _HOME]
    y[..., _CTRL] = -x[..., _CTRL]
    return y


def _fill_ball(bx, by):
    n = len(bx)
    vis = np.isfinite(bx)
    idx = np.arange(n)
    if not vis.any():
        return np.full(n, 0.5), np.full(n, 0.5), np.zeros(n)
    ix = np.interp(idx, idx[vis], bx[vis])
    iy = np.interp(idx, idx[vis], by[vis])
    last = np.maximum.accumulate(np.where(vis, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(vis, idx, 10 ** 9)[::-1])[::-1]
    gap = np.minimum(idx - last, nxt - idx).astype(float)
    return ix, iy, np.exp(-gap / INTERP_TAU)


def _step_mean(a, step):
    n = len(a)
    steps = -(-n // step)
    pad = np.full((steps * step,) + a.shape[1:], np.nan, dtype=np.float64)
    pad[:n] = a
    with np.errstate(all="ignore"):
        out = np.nanmean(pad.reshape((steps, step) + a.shape[1:]), axis=1)
    return np.nan_to_num(out)


def compute_features(px, py, bx, by, team, pitch=PITCH, step=STEP):
    """(n_steps, N_FEATURES) float32."""
    L, W = pitch
    n = len(bx)
    px = px.astype(np.float64)
    py = py.astype(np.float64)
    vis = np.isfinite(bx)
    ix, iy, conf = _fill_ball(bx.astype(np.float64), by.astype(np.float64))

    dist = np.hypot((px - ix[:, None]) * L, (py - iy[:, None]) * W)  # NaN where player missing
    edge = np.minimum.reduce([ix * L, (1 - ix) * L, iy * W, (1 - iy) * W])
    k = 5
    v = np.zeros(n)
    v[k:-k] = np.hypot((ix[2 * k:] - ix[:-2 * k]) * L, (iy[2 * k:] - iy[:-2 * k]) * W) / (2 * k / 25.0)

    kp = 12
    ps = np.full(n, np.nan)
    d = np.hypot((px[2 * kp:] - px[:-2 * kp]) * L, (py[2 * kp:] - py[:-2 * kp]) * W) / (2 * kp / 25.0)
    d[d > 12.0] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ps[kp:n - kp] = np.nanmedian(d, axis=1)

    cols = {}
    cols["vis"] = vis.astype(float)
    cols["conf"] = conf
    cols["ball_ax"] = np.abs(ix - 0.5) * 2
    cols["ball_ay"] = np.abs(iy - 0.5) * 2
    cols["edge"] = np.clip(edge, -3, 25)
    cols["ball_v"] = np.clip(v, 0, 40) * conf
    cols["pspeed"] = ps

    w = np.where(np.isfinite(dist), np.exp(-dist / 2.0), 0.0)
    wh, wa = w[:, team == 0].sum(1), w[:, team == 1].sum(1)
    c = np.clip(np.log(wh + 1e-3) - np.log(wa + 1e-3), -4, 4)
    nearest = np.where(np.isfinite(dist), dist, np.inf).min(1)
    c[nearest > 12.0] = 0.0
    cols["ctrl"] = c * conf

    for t, tag in ((0, "h_"), (1, "a_")):
        dt = np.where(team[None, :] == t, dist, np.inf)
        s = np.sort(np.where(np.isfinite(dt), dt, np.inf), axis=1)
        for j in range(3):
            cols[tag + "near%d" % (j + 1)] = np.minimum(s[:, j], 25.0) if s.shape[1] > j else np.full(n, 25.0)
        fin = np.isfinite(dt)
        cols[tag + "n5"] = ((dt < 5) & fin).sum(1)
        cols[tag + "n10"] = ((dt < 10) & fin).sum(1)
        m = (team == t)
        pxm, pym = px[:, m] * L, py[:, m] * W
        cols[tag + "tracked"] = np.isfinite(pxm).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols[tag + "sx"] = np.nanstd(pxm, axis=1)
            cols[tag + "sy"] = np.nanstd(pym, axis=1)

    mat = np.stack([cols[name] for name in COLUMNS], axis=1)
    return _step_mean(mat, step).astype(np.float32)


def step_labels(frame_labels, step=STEP):
    """Majority label per step; -1 where a step has no labelled frame."""
    n = len(frame_labels)
    steps = -(-n // step)
    pad = np.full(steps * step, -1, dtype=np.int16)
    pad[:n] = frame_labels
    blocks = pad.reshape(steps, step)
    counts = np.stack([(blocks == k).sum(1) for k in range(3)], axis=1)
    out = counts.argmax(1).astype(np.int8)
    out[counts.sum(1) == 0] = -1
    return out
