"""Label transforms that bring provider possession labels closer to the competition definition.

The competition counts a change of possession only on *controlled* possession: a touch without
control, a deflection, or an aerial duel won without control does not change it (CHALLENGE.md).
Provider labels (DFL BallPossession) switch on much shorter spells. merge_sandwiched() removes a
short spell of one team that sits between spells of the other team inside the same live stretch:
the ball went back to the team that had it, so the touch never became a possession.
"""
import numpy as np


def merge_sandwiched(label, min_frames):
    """Copy of a frame label array (0 H, 1 A, 2 D, -1 unknown) with every team spell shorter than
    min_frames that is directly preceded and followed by the other team relabelled to that team.
    Repeats until stable, since a merge can create a new short sandwich."""
    lab = np.asarray(label).copy()
    if min_frames <= 0:
        return lab
    while True:
        change = np.flatnonzero(np.diff(lab)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(lab)]))
        codes = lab[starts]
        merged = False
        for i in range(1, len(starts) - 1):
            c = codes[i]
            if c not in (0, 1) or ends[i] - starts[i] >= min_frames:
                continue
            if codes[i - 1] == codes[i + 1] == 1 - c:
                lab[starts[i]:ends[i]] = 1 - c
                merged = True
        if not merged:
            return lab
