"""Inference with the trained GRU ensemble on the competition tracking.

    from src.seq_model import decode_seq
    labels = decode_seq(atd, ["models/seq_all_s0.pt", ...])

Pipeline per graded half: step features -> GRU log-probs (averaged over the ensemble) ->
add Smart Tagging priors -> Viterbi with a switching cost -> frame labels.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from src.features import compute_features, columns, STEP  # noqa: E402
from src import decoder as rule  # noqa: E402

SEQ = dict(
    switch_cost=2.0,     # nats per label change; held-out CV flat over 2-4, 2 matches the public density best
    tag_team=0.0,        # weight of Smart Tagging team votes (nats)
    tag_setpiece=0.0,    # nats of extra D inside the first seconds of a set-piece window
    frame_decode=False,  # interpolate step log-probs to frames and decode per frame (cost x STEP)
    tta=True,            # team-swap test-time augmentation (+0.012 held-out, 6-7/7 folds)
)


def _load(paths):
    import torch
    sys.path.insert(0, ROOT)
    from train_seq import build_model
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = build_model(ck.get("arch", "gru"), len(ck["columns"]), ck["hidden"], ck.get("layers", 2))
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((m, ck["mu"], ck["sd"], list(ck["columns"])))
    return models


def predict_logp(models, feats, tta=False):
    """Mean step log-probs over models; with tta, each model also sees the team-swapped input
    and its H/A outputs are swapped back (makes the ensemble exactly home/away symmetric)."""
    import torch
    from train_seq import swap_np
    out = 0.0
    with torch.no_grad():
        for m, mu, sd, cols in models:
            x = torch.from_numpy(((feats - mu) / sd)[None]).float()
            lp = torch.log_softmax(m(x)[0], -1).numpy()
            if tta:
                xs = torch.from_numpy(((swap_np(feats, cols) - mu) / sd)[None]).float()
                lp = 0.5 * (lp + torch.log_softmax(m(xs)[0], -1).numpy()[:, [1, 0, 2]])
            out = out + lp
    return out / len(models)


def decode_seq(atd, model_paths, cfg=SEQ):
    from train_seq import viterbi_logp, frame_logp
    models = _load(model_paths)
    out = np.full(atd.n_frames, "X", dtype="<U1")
    net, setp = rule.tag_evidence(atd.n_frames, rule.PARAMS)
    for half, first, last in half_video_blocks("BAR-ATH"):
        sl = slice(first, last + 1)
        # models may use different feature sets (e.g. with and without touch features):
        # compute each set once and average log-probs over every model
        logp, groups = 0.0, {}
        for mdl in models:
            groups.setdefault(tuple(mdl[3]), []).append(mdl)
        for cols, group in groups.items():
            feats = compute_features(atd.px[sl], atd.py[sl], atd.bx[sl], atd.by[sl], atd.team,
                                     pitch=atd.pitch, orient="ball_x_s" in cols,
                                     touch="ball_acc" in cols)
            assert list(cols) == columns("ball_x_s" in cols, "ball_acc" in cols)
            logp = logp + predict_logp(group, feats, cfg["tta"]) * len(group)
        logp = logp / len(models)
        n_steps = len(logp)
        pad = lambda a: np.pad(a, (0, n_steps * STEP - len(a)), mode="edge").reshape(n_steps, STEP).mean(1)
        logp = logp.copy()
        logp[:, 0] += cfg["tag_team"] * pad(net[sl]) * 0.5
        logp[:, 1] -= cfg["tag_team"] * pad(net[sl]) * 0.5
        logp[:, 2] += cfg["tag_setpiece"] * pad(setp[sl])
        n = last - first + 1
        if cfg["frame_decode"]:
            path = viterbi_logp(frame_logp(logp, n, STEP), cfg["switch_cost"] * STEP)
        else:
            path = np.repeat(viterbi_logp(logp, cfg["switch_cost"]), STEP)[:n]
        out[sl] = rule.LETTERS[path]
    out[110808:113742] = "D"  # FAQ.md: graded but the ball is dead until the 113,742 kickoff
    return out
