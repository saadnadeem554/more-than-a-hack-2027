"""Per-step features from tracking, identical for the competition ATD and the external matches.

Inputs are normalised 0-1 coordinates (players: n x P, ball: n) and a team array (0 home, 1 away).
Everything is identity-free and orientation-free, so a model trained on other matches transfers.
Output rows are 0.4 s steps (STEP frames), averaged over the step.

Column layout (see COLUMNS): shared block, then a block per team. Columns ending in "_s" are
signed so that positive favours the home team (control log-odds, positions and velocities along
the home team's attacking direction). swap_teams() exchanges the two team blocks and negates every
signed column, which is the augmentation that stops a model from learning "home" as a prior.

The attacking direction is inferred per half from the tracking alone: the team whose deepest
player (its goalkeeper, most of the time) sits nearer x = 0 defends that end.
"""

import numpy as np

STEP = 10
PITCH = (105.0, 68.0)
INTERP_TAU = 50.0  # frames; ball confidence e-folding time away from a real observation

TEAM_COLS = ["near1", "near2", "near3", "n5", "n10", "tracked", "sx", "sy"]
SHARED = ["vis", "conf", "ball_ax", "ball_ay", "edge", "ball_v", "ctrl_s", "pspeed"]
TEAM_ORIENT = ["cx_s", "vx_s", "depth_s"]
SHARED_ORIENT = ["ball_x_s", "vx_all_s"]


def columns(orient=True):
    team = TEAM_COLS + (TEAM_ORIENT if orient else [])
    shared = SHARED + (SHARED_ORIENT if orient else [])
    return shared + ["h_" + c for c in team] + ["a_" + c for c in team]


COLUMNS = columns(True)
N_FEATURES = len(COLUMNS)


def swap_teams(x, cols=COLUMNS):
    """Copy of a (..., n_cols) array with home and away exchanged (see module docstring)."""
    cols = list(cols)
    home = [i for i, c in enumerate(cols) if c.startswith("h_")]
    away = [cols.index("a_" + cols[i][2:]) for i in home]
    y = x.copy()
    y[..., home], y[..., away] = x[..., away], x[..., home]
    signed = [i for i, c in enumerate(cols) if c.endswith("_s")]
    y[..., signed] = -y[..., signed]
    return y


def home_attack_sign(px, team):
    """+1 if the home team attacks towards x = 1 in this half, else -1."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_deep = np.nanmedian(np.nanmin(px[:, team == 0], axis=1))
        away_deep = np.nanmedian(np.nanmin(px[:, team == 1], axis=1))
    return 1.0 if home_deep < away_deep else -1.0


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


def compute_features(px, py, bx, by, team, pitch=PITCH, step=STEP, orient=True):
    """(n_steps, len(columns(orient))) float32. One call per half (direction is per half)."""
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
    cols["ctrl_s"] = c * conf

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

    if orient:
        sgn = home_attack_sign(px, team)
        # per-player x velocity (m/s) over +/- kp frames, along the home attacking direction
        vxp = np.full(px.shape, np.nan)
        vxp[kp:n - kp] = (px[2 * kp:] - px[:-2 * kp]) * L / (2 * kp / 25.0)
        vxp[np.abs(vxp) > 12.0] = np.nan
        vxp *= sgn
        xs = (px - 0.5) * L * sgn  # metres from halfway, positive = home attacking half
        cols["ball_x_s"] = (ix - 0.5) * L * sgn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols["vx_all_s"] = np.nanmean(vxp, axis=1)
            for t, tag in ((0, "h_"), (1, "a_")):
                m = team == t
                cols[tag + "cx_s"] = np.nanmean(xs[:, m], axis=1)
                cols[tag + "vx_s"] = np.nanmean(vxp[:, m], axis=1)
                # deepest outfield-ish line: 2nd-most-defensive player (skips the goalkeeper)
                own = np.sort(np.where(np.isfinite(xs[:, m]), xs[:, m] * (1 if t == 0 else -1), np.inf), axis=1)
                second = own[:, 1] if own.shape[1] > 1 else own[:, 0]
                cols[tag + "depth_s"] = np.where(np.isfinite(second), second, 0.0) * (1 if t == 0 else -1)

    mat = np.stack([cols[name] for name in columns(orient)], axis=1)
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
