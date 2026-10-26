"""How do Smart Tagging windows line up with the dead-ball stretches in the public 5 minutes?"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
lab = gt["label"].values
lo, hi = 11185, 18684
print("GT runs in window (label, start, end, seconds):")
start = lo
for f in range(lo + 1, hi + 2):
    if f == hi + 1 or lab[f] != lab[start]:
        print("  {} {}-{} {:.1f}s".format(lab[start], start, f - 1, (f - start) / 25))
        start = f
print("Smart tagging overlapping window (code, team, type, start, end):")
for i in load_smart_tagging.load():
    if i["end_frame"] >= lo and i["start_frame"] <= hi:
        print("  {:<28} {:<14} {:<22} {}-{}".format(
            i["code"], i["tags"].get("Team"), i["tags"].get("Type") or "", i["start_frame"], i["end_frame"]))
