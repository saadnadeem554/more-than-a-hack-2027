"""Explicit-duration (semi-Markov) decoding over {H, A, D}.

A plain Viterbi charges a flat cost per label change, so it has no idea how long possessions and
stoppages usually last. Here the path is a sequence of segments, each scored as

    emission_weight * sum of step log-probs of its state
    + duration_weight * log p(duration | state)
    + transition cost of entering it,

with consecutive segments in different states. Duration distributions are learned from training
labels (steps of 0.4 s), smoothed on a log-length scale.
"""
import numpy as np

H, A, D = 0, 1, 2


def duration_logpmf(step_label_series, max_len, n_states=3, bins_per_octave=4, floor=1e-6):
    """(n_states, max_len + 1) log-pmf of segment length in steps, from label arrays (-1 ignored).

    Lengths are histogrammed on a log2 scale (smooth for long, rare durations) and spread back to
    integer lengths; segments longer than max_len are counted at max_len.
    """
    edges = 2.0 ** (np.arange(0, np.log2(max_len) + 1.0 / bins_per_octave, 1.0 / bins_per_octave))
    edges = np.unique(np.concatenate(([1.0], edges, [max_len + 1.0])))
    out = np.full((n_states, max_len + 1), np.log(floor))
    for s in range(n_states):
        lengths = []
        for y in step_label_series:
            change = np.flatnonzero(np.diff(y)) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [len(y)]))
            for a, b in zip(starts, ends):
                if y[a] == s:
                    lengths.append(min(b - a, max_len))
        if not lengths:
            continue
        counts, _ = np.histogram(lengths, bins=edges)
        pmf = np.zeros(max_len + 1)
        for k in range(len(edges) - 1):
            lo, hi = int(np.ceil(edges[k])), int(np.ceil(edges[k + 1])) - 1
            hi = min(hi, max_len)
            if hi >= lo and counts[k]:
                pmf[lo:hi + 1] += counts[k] / (hi - lo + 1)
        pmf /= pmf.sum()
        out[s] = np.log(np.maximum(pmf, floor))
    out[:, 0] = -np.inf
    return out


def decode(logp, dur_logpmf, emission_weight=1.0, duration_weight=1.0, switch_cost=0.0):
    """Best segmentation of a (T, 3) step log-prob array. Returns a (T,) state path."""
    T, S = logp.shape
    max_len = dur_logpmf.shape[1] - 1
    cum = np.vstack([np.zeros((1, S)), np.cumsum(logp * emission_weight, axis=0)])
    dur = dur_logpmf * duration_weight
    best = np.full((T + 1, S), -np.inf)       # best score of a path whose last segment ends at t
    back_len = np.zeros((T + 1, S), dtype=np.int32)
    back_prev = np.zeros((T + 1, S), dtype=np.int8)
    best[0] = 0.0
    for t in range(1, T + 1):
        lens = np.arange(1, min(max_len, t) + 1)
        starts = t - lens
        for s in range(S):
            seg = cum[t, s] - cum[starts, s] + dur[s, lens]
            if t == T:  # last segment may be truncated: do not charge it for being short
                seg = cum[t, s] - cum[starts, s] + np.maximum(dur[s, lens], dur[s].max() - 2.0)
            prev = best[starts].copy()
            prev[:, s] = -np.inf
            prev[starts > 0] -= switch_cost
            first = starts == 0
            prev[first] = 0.0  # a path may start with any state
            k_prev = prev.argmax(1)
            total = prev[np.arange(len(lens)), k_prev] + seg
            if first.any():  # first segment: also truncated at the start
                i0 = np.flatnonzero(first)[0]
                total[i0] = cum[t, s] - cum[0, s] + np.maximum(dur[s, lens[i0]], dur[s].max() - 2.0)
            j = int(total.argmax())
            best[t, s] = total[j]
            back_len[t, s] = lens[j]
            back_prev[t, s] = k_prev[j]
    path = np.empty(T, dtype=np.int8)
    t, s = T, int(best[T].argmax())
    while t > 0:
        L = back_len[t, s]
        path[t - L:t] = s
        ps = back_prev[t, s]
        t -= L
        s = int(ps)
    return path
