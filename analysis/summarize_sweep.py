"""Rank (config, decoding cost) pairs from a Kaggle sweep's results.txt by mean held-out spellF1.

    python analysis/summarize_sweep.py kaggle/out/results.txt [rank_key, e.g. spellF1_fr]
"""
import re
import sys
from collections import defaultdict

import numpy as np

rows = defaultdict(lambda: defaultdict(list))
for line in open(sys.argv[1]):
    m = re.match(r"(\S+) RESULT val=(\S+) cost=(\S+) (.*)", line.strip())
    if not m:
        continue
    name, fold, cost, rest = m.groups()
    for k, v in re.findall(r"(\w+)=([\d.]+)", rest):
        rows[(name, float(cost))][k].append(float(v))

key = sys.argv[2] if len(sys.argv) > 2 else "spellF1"
ranked = sorted(rows.items(), key=lambda kv: -np.mean(kv[1][key]))
extra = [k for k in ("spellF1_fr", "macroF1_fr") if any(k in d for d in rows.values())]
print("{:<10} {:>5} {:>8} {:>8} {:>8}".format("config", "cost", "spellF1", "sd", "macroF1")
      + "".join(" {:>11}".format(k) for k in extra) + " folds")
for (name, cost), d in ranked:
    print("{:<10} {:>5} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        name, cost, np.mean(d["spellF1"]), np.std(d["spellF1"]), np.mean(d["macroF1"]))
        + "".join(" {:>11.4f}".format(np.mean(d[k])) for k in extra) + " {:>5}".format(len(d["spellF1"])))
