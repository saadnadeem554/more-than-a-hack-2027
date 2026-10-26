"""Rank (config, decoding cost) pairs from a Kaggle sweep's results.txt by mean held-out spellF1.

    python analysis/summarize_sweep.py kaggle/out/results.txt
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

ranked = sorted(rows.items(), key=lambda kv: -np.mean(kv[1]["spellF1"]))
print("{:<10} {:>5} {:>8} {:>8} {:>8} {:>6}".format("config", "cost", "spellF1", "sd", "macroF1", "folds"))
for (name, cost), d in ranked:
    print("{:<10} {:>5} {:>8.4f} {:>8.4f} {:>8.4f} {:>6}".format(
        name, cost, np.mean(d["spellF1"]), np.std(d["spellF1"]), np.mean(d["macroF1"]), len(d["spellF1"])))
