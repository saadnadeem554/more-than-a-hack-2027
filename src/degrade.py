"""Make clean external tracking look like the competition's automatic tracking.

Calibrated against aggregate feature quantiles of the competition ATD (analysis/domain_gap.py):

- the ball is lost on ~41% of graded frames in long gaps (median ~10 s): alternating visible /
  missing runs are sampled from the run-length quantiles of that tracking;
- the automatic tracker rarely reports a ball beyond the lines, so an out-of-pitch ball is hidden
  most of the time;
- players drop out in short blocks, and the tracker adds spurious detections (staff, substitutes
  warming up), mostly near the touchlines.

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


def degrade(px, py, bx, by, rng, pitch=(105.0, 68.0), randomise=True):
    n = len(bx)
    L, W = pitch
    u = (lambda lo, hi: rng.uniform(lo, hi)) if randomise else (lambda lo, hi: (lo + hi) / 2)
    player_drop = u(0.005, 0.05)
    ball_noise_m = u(0.05, 0.4)
    out_hide = u(0.6, 1.0)
    spurious_rate = u(0.0, 0.25)

    # ball: sampled gaps, plus hidden when outside the pitch rectangle
    # Visible runs are stretched because hiding the off-pitch ball (about a third of external
    # frames) adds gaps of its own; 2-3x lands the overall visibility near the competition's 59%.
    mask = synthetic_pattern(n, rng, visible_scale=u(2.0, 3.0))
    outside = (bx < 0) | (bx > 1) | (by < 0) | (by > 1)
    hide_out = outside & (np.repeat(rng.random(-(-n // 25)), 25)[:n] < out_hide)
    mask &= ~hide_out
    bx2, by2 = bx.copy(), by.copy()
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
