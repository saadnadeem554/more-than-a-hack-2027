"""Make clean external tracking look like the competition's automatic tracking.

The competition ATD loses the ball on ~41% of graded frames, in long gaps (median ~10 s), and
drops players. We sample alternating visible/missing runs from the run-length quantiles of that
tracking (aggregate numbers only) and drop player slots in short blocks.
"""

import numpy as np


# Run-length quantiles (frames) of the competition tracking's ball track on graded frames, at the
# 0/5/25/50/75/95/100 percentiles. Only these aggregate numbers are used; no positions, no labels.
MISSING_Q = ([0, 5, 25, 50, 75, 95, 100], [1, 2, 152, 268, 526, 1270, 3000])
VISIBLE_Q = ([0, 5, 25, 50, 75, 95, 100], [5, 40, 137, 345, 758, 2267, 6000])


def synthetic_pattern(n, rng):
    """Boolean ball-visibility array of length n built from alternating sampled run lengths."""
    out = np.zeros(n, dtype=bool)
    pos, visible = 0, bool(rng.integers(0, 2))
    while pos < n:
        q = VISIBLE_Q if visible else MISSING_Q
        length = int(np.interp(rng.random() * 100, q[0], q[1]))
        out[pos:pos + length] = visible
        pos += max(length, 1)
        visible = not visible
    return out


def degrade(px, py, bx, by, rng, player_drop=0.08, block=50, ball_noise_m=0.4,
            pitch=(105.0, 68.0)):
    n = len(bx)
    mask = synthetic_pattern(n, rng)
    bx2, by2 = bx.copy(), by.copy()
    bx2[~mask] = np.nan
    by2[~mask] = np.nan
    L, W = pitch
    bx2 = bx2 + rng.normal(0, ball_noise_m / L, n)
    by2 = by2 + rng.normal(0, ball_noise_m / W, n)

    px2, py2 = px.copy(), py.copy()
    blocks = -(-n // block)
    drop = rng.random((blocks, px.shape[1])) < player_drop
    drop = np.repeat(drop, block, axis=0)[:n]
    px2[drop] = np.nan
    py2[drop] = np.nan
    return px2, py2, bx2.astype(np.float32), by2.astype(np.float32)
