"""Regenerate the possession file from the provided data.

    python make_submission.py [--out evaluation/results/leti_gang.csv] [--rule]

Default: GRU ensemble trained on external matches (models/seq_*_s*.pt) + Viterbi.
--rule: the hand-built v1 decoder (no learned weights).
"""
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import write_submission  # noqa: E402
from src.atd import load_atd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "leti_gang.csv"))
    ap.add_argument("--rule", action="store_true")
    args = ap.parse_args()
    atd = load_atd()
    if args.rule:
        from src.decoder import decode
        labels = decode(atd)
    else:
        from src.seq_model import decode_seq
        paths = sorted(glob.glob(os.path.join(ROOT, "models", "seq_*_s*.pt")))
        labels = decode_seq(atd, paths)
    write_submission(args.out, labels)
    print("wrote", args.out, {k: int((labels == k).sum()) for k in "HADX"})


if __name__ == "__main__":
    main()
