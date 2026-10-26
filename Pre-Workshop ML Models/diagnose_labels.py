"""
Diagnose label coverage across the barrothandendpoints dataset.

For every channel/label, report:
  - how many frames in each split actually contain it
  - average number of points and average segment length (for diameter lines)

This tells us whether the weak channels (diameter_40pct / diameter_60pct)
are a DATA problem (missing/sparse/degenerate labels) vs a model problem.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import math
from pathlib import Path
from collections import defaultdict

from jet_multimask import CHANNELS

DATA_DIR = Path(r"c:\Users\nmana\Documents\VSCode\OpenCV MEW Jet Analysis\data\barrothandendpoints")
SPLITS = ["train", "val", "test"]


def seg_len(pts):
    if len(pts) < 2:
        return 0.0
    (x1, y1), (x2, y2) = pts[0], pts[1]
    return math.hypot(x2 - x1, y2 - y1)


def main():
    # per split: label -> count of frames containing it
    present = {s: defaultdict(int) for s in SPLITS}
    totals  = {s: 0 for s in SPLITS}
    # diameter length stats: label -> list of lengths
    diam_lengths = defaultdict(list)
    degenerate   = defaultdict(int)   # label -> frames where it exists but is degenerate

    for split in SPLITS:
        for jp in sorted((DATA_DIR / split).glob("*.json")):
            totals[split] += 1
            data = json.loads(jp.read_text(encoding="utf-8"))
            seen = set()
            for sh in data.get("shapes", []):
                lab = sh.get("label")
                if lab not in CHANNELS:
                    continue
                pts = sh.get("points", [])
                seen.add(lab)
                if lab.startswith("diameter_"):
                    L = seg_len(pts)
                    diam_lengths[lab].append(L)
                    if L < 1.0 or len(pts) < 2:
                        degenerate[lab] += 1
            for lab in seen:
                present[split][lab] += 1

    print("=" * 72)
    print("LABEL PRESENCE  (frames containing the label / total frames in split)")
    print("=" * 72)
    header = f"{'label':18s} " + "".join(f"{s:>16s}" for s in SPLITS)
    print(header)
    print("-" * 72)
    for lab in CHANNELS:
        row = f"{lab:18s} "
        for s in SPLITS:
            c = present[s][lab]
            t = totals[s]
            pct = 100.0 * c / t if t else 0.0
            row += f"{c:5d}/{t:<5d}({pct:3.0f}%)"
            row += " "
        print(row)

    print("\n" + "=" * 72)
    print("DIAMETER SEGMENT LENGTHS (native px) — low/degenerate lengths hurt Dice")
    print("=" * 72)
    print(f"{'label':18s}{'n':>6s}{'min':>8s}{'mean':>8s}{'max':>8s}{'<2px':>8s}")
    print("-" * 72)
    for lab in CHANNELS:
        if not lab.startswith("diameter_"):
            continue
        Ls = diam_lengths[lab]
        if not Ls:
            print(f"{lab:18s}{0:>6d}{'-':>8s}{'-':>8s}{'-':>8s}{'-':>8s}")
            continue
        n = len(Ls)
        short = sum(1 for L in Ls if L < 2.0)
        print(f"{lab:18s}{n:>6d}{min(Ls):>8.1f}{sum(Ls)/n:>8.1f}{max(Ls):>8.1f}{short:>8d}")


if __name__ == "__main__":
    main()
