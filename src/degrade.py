"""Make clean external tracking look like the competition's automatic tracking.

Calibrated against aggregate feature quantiles of the competition ATD (analysis/domain_gap.py):

- the ball is lost on ~41% of graded frames in long gaps (median ~10 s): alternating visible /
  missing runs are sampled from the run-length quantiles of that tracking;
- the automatic tracker rarely reports a ball beyond the lines, so an out-of-pitch ball is hidden
  most of the time;
- players drop out in short blocks, and the tracker adds spurious detections (staff, substitutes
  warming up), mostly near the touchlines;
- its ball track is heavily smoothed (velocity changes ~2x softer than raw optical tracking), so
  the external ball is passed through a 15-31 frame moving average.

Each call draws its strengths at random (domain randomisation), so the real tracking sits inside
the range the model sees in training rather than at one guessed setting.
"""

import numpy as np

# Run-length quantiles (frames) of the competition tracking's ball track on graded frames, at the
# 0/5/25/50/75/95/100 percentiles. Only these aggregate numbers are used; no positions, no labels.
MISSING_Q = ([0, 5, 25, 50, 75, 95, 100], [1, 2, 152, 268, 526, 1270, 3000])
VISIBLE_Q = ([0, 5, 25, 50, 75, 95, 100], [5, 40, 137, 345, 758, 2267, 6000])


def synthetic_pattern(n, rng, visible_scale=1.0):
    """Boolean ball-visibility array of length n built from alternating sampled run lengths."""
    out = np.zeros(n, dtype=bool)
    pos, visible = 0, bool(rng.integers(0, 2))
    while pos < n:
        q = VISIBLE_Q if visible else MISSING_Q
        length = int(np.interp(rng.random() * 100, q[0], q[1]) * (visible_scale if visible else 1.0))
        out[pos:pos + length] = visible
        pos += max(length, 1)
        visible = not visible
    return out


# Relative ball-loss hazard of the competition tracker by context (analysis/gap_context.py):
# losses are ~2x more likely when the ball is slow (often dead) and ~2x more likely in a crowd.
SPEED_BINS, SPEED_MULT = [3, 8, 15, 25], [1.97, 0.84, 0.43, 0.24, 0.8]
CROWD_BINS, CROWD_MULT = [3, 5], [1.0, 1.45, 2.3]


def context_pattern(bx, by, px, py, rng, pitch, base_hazard):
    """Ball visibility where each visible frame is lost with a context-dependent hazard and each
    gap lasts a length sampled from the competition run-length quantiles."""
    n = len(bx)
    L, W = pitch
    v = np.zeros(n)
    v[1:] = np.hypot(np.diff(bx) * L, np.diff(by) * W) * 25
    v = np.nan_to_num(v)
    d = np.hypot((px - bx[:, None]) * L, (py - by[:, None]) * W)
    crowd = (np.where(np.isfinite(d), d, np.inf) < 5).sum(1)
    h = base_hazard * np.array(SPEED_MULT)[np.digitize(v, SPEED_BINS)]         * np.array(CROWD_MULT)[np.digitize(crowd, CROWD_BINS)]
    starts = np.flatnonzero(rng.random(n) < h)
    out = np.ones(n, dtype=bool)
    free = 0
    for t in starts:
        if t < free:
            continue
        length = int(np.interp(rng.random() * 100, MISSING_Q[0], MISSING_Q[1]))
        out[t:t + length] = False
        free = t + length + 1
    return out


def smooth_ball(bx, by, window):
    """Centred moving average of the ball track (NaN-aware). The competition tracker's ball is
    much smoother than raw optical positions (analysis/ball_smoothing.py)."""
    if window <= 1:
        return bx.copy(), by.copy()
    k = np.ones(int(window))
    out = []
    for a in (bx, by):
        ok = np.isfinite(a)
        num = np.convolve(np.where(ok, a, 0.0), k, mode="same")
        den = np.convolve(ok.astype(float), k, mode="same")
        out.append(np.where(ok, num / np.maximum(den, 1), np.nan).astype(np.float32))
    return out[0], out[1]


def degrade(px, py, bx, by, rng, pitch=(105.0, 68.0), randomise=True, context=True):
    n = len(bx)
    L, W = pitch
    u = (lambda lo, hi: rng.uniform(lo, hi)) if randomise else (lambda lo, hi: (lo + hi) / 2)
    player_drop = u(0.005, 0.05)
    ball_noise_m = u(0.0, 0.15)
    ball_window = int(u(15, 31))
    out_hide = u(0.6, 1.0)
    spurious_rate = u(0.0, 0.25)

    # ball: sampled gaps, plus hidden when outside the pitch rectangle
    # Visible runs are stretched because hiding the off-pitch ball (about a third of external
    # frames) adds gaps of its own; 2-3x lands the overall visibility near the competition's 59%.
    if context:
        mask = context_pattern(bx, by, px, py, rng, pitch, base_hazard=u(0.0003, 0.0007))
    else:
        mask = synthetic_pattern(n, rng, visible_scale=u(2.0, 3.0))
    outside = (bx < 0) | (bx > 1) | (by < 0) | (by > 1)
    hide_out = outside & (np.repeat(rng.random(-(-n // 25)), 25)[:n] < out_hide)
    mask &= ~hide_out
    bx2, by2 = smooth_ball(bx, by, ball_window)
    bx2[~mask] = np.nan
    by2[~mask] = np.nan
    bx2 = bx2 + rng.normal(0, ball_noise_m / L, n)
    by2 = by2 + rng.normal(0, ball_noise_m / W, n)

    # players: block dropout
    block = 50
    px2, py2 = px.copy(), py.copy()
    blocks = -(-n // block)
    drop = np.repeat(rng.random((blocks, px.shape[1])) < player_drop, block, axis=0)[:n]
    px2[drop] = np.nan
    py2[drop] = np.nan

    # spurious detections: one extra slot per team that appears in runs near a touchline
    extra_x = np.full((n, 2), np.nan, dtype=np.float32)
    extra_y = np.full((n, 2), np.nan, dtype=np.float32)
    for j in range(2):
        on = np.repeat(rng.random(-(-n // 250)) < spurious_rate, 250)[:n]
        walk_x = np.clip(rng.uniform(0.1, 0.9) + np.cumsum(rng.normal(0, 0.002, n)), 0.0, 1.0)
        side = rng.choice([-0.03, 1.03])
        walk_y = side + np.cumsum(rng.normal(0, 0.0005, n))
        extra_x[on, j] = walk_x[on]
        extra_y[on, j] = walk_y[on]
    px2 = np.concatenate([px2, extra_x], axis=1)
    py2 = np.concatenate([py2, extra_y], axis=1)
    return px2, py2, bx2.astype(np.float32), by2.astype(np.float32)


def degraded_team(team):
    """Team array matching degrade()'s two appended spurious slots (one per team)."""
    return np.concatenate([team, np.array([0, 1], dtype=team.dtype)])
