"""Train a bidirectional GRU that labels each 0.4 s step H / A / D from tracking features.

Trained only on external matches (DFL open data, CC-BY 4.0) whose tracking has been degraded to
look like the competition ATD -- see train_build_dataset.py. Validation is on held-out external
matches, which is allowed (hand-labelled / provider-labelled data from OTHER matches).

    python train_seq.py --features external/train/features.npz --val 0 1 --out models/seq_val01.pt
    python train_seq.py --features external/train/features.npz --val -1 --out models/seq_all.pt   # all matches

Runs on CPU or GPU (Kaggle: --device cuda). Needs numpy and torch only; the optional spellF1
report uses evaluation/possession_metrics.py when it can be imported.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
try:
    from src.features import swap_teams  # noqa: E402
except Exception:  # Kaggle copy without src/: columns are stored in the npz
    swap_teams = None
try:
    import possession_metrics as pm  # noqa: E402
except Exception:
    pm = None


class SeqNet(nn.Module):
    def __init__(self, n_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def swap_np(x, columns):
    """Team-swap augmentation without importing src/: exchange h_/a_ blocks, negate *_s columns."""
    cols = list(columns)
    home = [i for i, c in enumerate(cols) if c.startswith("h_")]
    away = [cols.index("a_" + cols[i][2:]) for i in home]
    y = x.copy()
    y[..., home], y[..., away] = x[..., away], x[..., home]
    signed = [i for i, c in enumerate(cols) if c.endswith("_s")]
    y[..., signed] = -y[..., signed]
    return y


def load(path, ykey="Y", yfkey="YF"):
    """[(X, step_labels, frame_labels or None)], plus per-series match, half, rep and columns.

    ykey / yfkey pick which stored label set to train on / score against (relabel_dataset.py).
    """
    z = np.load(path, allow_pickle=True)
    X, Y, lens = z["X"], z[ykey], z["lens"]  # read each array once: npz access decompresses
    ends = np.cumsum(lens)
    starts = ends - lens
    if yfkey in z.files:
        YF, fl = z[yfkey], z["flens"]
        fe = np.cumsum(fl)
        frames = [YF[s:e] for s, e in zip(fe - fl, fe)]
    else:
        frames = [None] * len(lens)
    series = [(X[s:e], Y[s:e], f) for (s, e), f in zip(zip(starts, ends), frames)]
    return series, z["match"], z["half"], z["rep"], list(z["columns"])


def frame_logp(logp, n_frames, step=10):
    """Step log-probs linearly interpolated to frames (step i is centred on frame i*step + step/2)."""
    centres = np.arange(len(logp)) * step + (step - 1) / 2.0
    f = np.arange(n_frames)
    return np.stack([np.interp(f, centres, logp[:, k]) for k in range(logp.shape[1])], axis=1)


def viterbi_logp(logp, cost):
    n = len(logp)
    trans = np.full((3, 3), -cost)
    np.fill_diagonal(trans, 0.0)
    score = np.empty((n, 3))
    back = np.zeros((n, 3), dtype=np.int8)
    score[0] = logp[0]
    for t in range(1, n):
        cand = score[t - 1][:, None] + trans
        back[t] = cand.argmax(0)
        score[t] = cand.max(0) + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def transition_weights(yb, radius, weight):
    """(batch, steps) weights: `weight` within `radius` steps of an H<->A change, else 1."""
    team = yb.clone()
    team[(team != 0) & (team != 1)] = -1
    change = (team[:, 1:] != team[:, :-1]) & (team[:, 1:] >= 0) & (team[:, :-1] >= 0)
    mark = torch.zeros_like(yb, dtype=torch.float32)
    mark[:, 1:] = change.float()
    k = torch.ones(1, 1, 2 * radius + 1, device=yb.device)
    near = torch.nn.functional.conv1d(mark[:, None], k, padding=radius)[:, 0] > 0
    return torch.where(near, torch.full_like(mark, weight), torch.ones_like(mark))


def evaluate(model, series, idx, mu, sd, device, step=10, switch_cost=2.0):
    """Held-out scores against FRAME-level labels.

    spellF1 / macroF1: step decisions repeated to frames.  spellF1_fr / macroF1_fr: step log-probs
    interpolated to frames and decoded per frame (cost x step, the same stiffness), which places
    boundaries between step edges.
    """
    model.eval()
    accs, rows = [], []
    with torch.no_grad():
        for i in idx:
            x, y, yf = series[i]
            xt = torch.from_numpy(((x - mu) / sd)[None]).float().to(device)
            logp = torch.log_softmax(model(xt)[0], -1).cpu().numpy()
            path = viterbi_logp(logp, switch_cost)
            ok = y >= 0
            accs.append(float((path[ok] == y[ok]).mean()))
            if yf is None:  # older datasets: step labels only
                yf = np.repeat(y, step)
            n = len(yf)
            gt = np.where(yf >= 0, yf, 3).astype(np.int8)
            rows.append((gt, np.repeat(path, step)[:n].astype(np.int8),
                         viterbi_logp(frame_logp(logp, n, step), switch_cost * step).astype(np.int8)))
    out = {"acc": float(np.mean(accs))}
    if pm is not None and rows:
        for key, j in (("", 1), ("_fr", 2)):
            spell, macro = [], []
            for r in rows:
                g, p = r[0], r[j]
                runs = pm.valid_runs(g)
                spell.append(pm.spell_scores(g, p, runs)["mean_f1"])
                macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(g, p))))
            out["spellF1" + key] = float(np.mean(spell))
            out["macroF1" + key] = float(np.mean(macro))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "external", "train", "features.npz"))
    ap.add_argument("--val", type=int, nargs="+", default=[0, 1], help="match indices held out; -1 = none")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], help="match indices never used")
    ap.add_argument("--dump", default=None, help="write held-out step log-probs + frame labels (npz)")
    ap.add_argument("--ykey", default="Y", help="training label set in the npz")
    ap.add_argument("--yfkey", default="YF", help="frame label set held-out matches are scored on")
    ap.add_argument("--trans-weight", type=float, default=1.0,
                    help="loss weight for steps within --trans-radius of a possession change")
    ap.add_argument("--trans-radius", type=int, default=5, help="steps (0.4 s each)")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "seq.pt"))
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=400, help="steps per training chunk (0.4 s each)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    series, match, half, rep, columns = load(args.features, args.ykey, args.yfkey)
    val_matches = set(v for v in args.val if v >= 0)
    excluded = set(args.exclude)
    train_idx = [i for i in range(len(series)) if match[i] not in val_matches | excluded]
    val_idx = [i for i in range(len(series)) if match[i] in val_matches and rep[i] == 0]
    print("series: train {} val {} | features {} | device {}".format(
        len(train_idx), len(val_idx), len(columns), args.device), flush=True)

    allx = np.concatenate([series[i][0] for i in train_idx])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    model = SeqNet(len(columns), args.hidden, args.layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        model.train()
        xb, yb = [], []
        for _ in range(args.batch):
            x, y, _ = series[train_idx[rng.integers(len(train_idx))]]
            s = int(rng.integers(0, max(1, len(x) - args.chunk)))
            xc, yc = x[s:s + args.chunk].copy(), y[s:s + args.chunk].astype(np.int64).copy()
            if len(xc) < args.chunk:
                pad = args.chunk - len(xc)
                xc = np.pad(xc, ((0, pad), (0, 0)), mode="edge")
                yc = np.pad(yc, (0, pad), constant_values=-1)
            if rng.random() < 0.5:  # swap home/away: features and labels together
                xc = swap_np(xc, columns)
                yc = np.where(yc == 0, 1, np.where(yc == 1, 0, yc))
            xb.append((xc - mu) / sd)
            yb.append(yc)
        xb = torch.from_numpy(np.stack(xb)).float().to(args.device)
        yb = torch.from_numpy(np.stack(yb)).to(args.device)
        xb = xb + 0.02 * torch.randn_like(xb)
        per = loss_fn(model(xb).reshape(-1, 3), yb.reshape(-1)).reshape(yb.shape)
        valid = (yb >= 0).float()
        w = valid
        if args.trans_weight != 1.0:
            w = valid * transition_weights(yb, args.trans_radius, args.trans_weight)
        loss = (per * w).sum() / valid.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 250 == 0 or it == args.iters:
            msg = "iter {:>5} loss {:.4f} ({:.0f}s)".format(it, loss.item(), time.time() - t0)
            if val_idx and (it % 1000 == 0 or it == args.iters):
                msg += "  val " + " ".join("{} {:.4f}".format(k, v) for k, v in
                                          evaluate(model, series, val_idx, mu, sd, args.device).items())
            print(msg, flush=True)

    if val_idx and args.dump:  # held-out predictions for offline error analysis
        model.eval()
        dump = {}
        with torch.no_grad():
            for i in val_idx:
                x, y, yf = series[i]
                xt = torch.from_numpy(((x - mu) / sd)[None]).float().to(args.device)
                dump["logp_%d" % i] = torch.log_softmax(model(xt)[0], -1).cpu().numpy()
                dump["yf_%d" % i] = yf if yf is not None else np.repeat(y, 10)
                dump["x_%d" % i] = x
        np.savez_compressed(args.dump, columns=np.array(columns), **dump)

    if val_idx:  # choose the decoding switch cost by held-out matches
        for cost in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
            r = evaluate(model, series, val_idx, mu, sd, args.device, switch_cost=cost)
            print("RESULT val={} cost={} {}".format(",".join(map(str, sorted(val_matches))), cost,
                  " ".join("{}={:.4f}".format(k, v) for k, v in r.items())), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "columns": columns,
                "hidden": args.hidden, "layers": args.layers,
                "val": sorted(val_matches)}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
