"""Find AReM recordings that are copies of each other.

Each file should be a separate repetition of an activity. This compares every pair of
the 88 files (across folders too) and reports pairs sharing more than half their rows
verbatim. Sensor readings are noisy, so two independent sessions share almost no rows.

Usage: python data/check_independence.py [data/AReM]
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

COLUMNS = ["time", "avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23"]
THRESHOLD = 0.5


def read_rows(path: Path) -> list[bytes]:
    frame = pd.read_csv(
        path, comment="#", header=None, sep=r"[,\s]+", engine="python", usecols=range(7), names=COLUMNS
    )
    return [row.tobytes() for row in frame.drop(columns="time").to_numpy()]


def duplicate_pairs(root: Path) -> list[tuple[str, str, float]]:
    rows = {f"{p.parent.name}/{p.stem}": read_rows(p) for p in sorted(root.glob("*/dataset*.csv"))}
    pairs = []
    for a, b in combinations(rows, 2):
        other = set(rows[b])
        shared = sum(r in other for r in rows[a]) / min(len(rows[a]), len(rows[b]))
        if shared > THRESHOLD:
            pairs.append((a, b, shared))
    return pairs


def main(root: Path) -> None:
    pairs = duplicate_pairs(root)
    for a, b, shared in pairs:
        print(f"{a:20s} {b:20s} {shared:6.1%} of rows shared")
    print(f"{len(pairs)} duplicate pairs across {len(list(root.glob('*/dataset*.csv')))} files")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "AReM"))
