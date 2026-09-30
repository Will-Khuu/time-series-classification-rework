"""Load AReM recordings into one tidy frame, one row per 250 ms sample."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "AReM"

SERIES = ["avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23"]

# Copy -> original. Found by data/check_independence.py; tests/test_loading.py re-runs the scan.
COPIES = {
    "cycling/dataset15": "cycling/dataset1",
    "lying/dataset10": "lying/dataset1",
    "lying/dataset11": "lying/dataset3",
    "lying/dataset12": "lying/dataset5",
    "lying/dataset13": "lying/dataset7",
    "lying/dataset14": "lying/dataset9",
    "lying/dataset15": "lying/dataset1",
}


def activity_of(folder: str) -> str:
    return "bending" if folder.startswith("bending") else folder


def read_recording(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        comment="#",
        header=None,
        sep=r"[,\s]+",
        engine="python",
        usecols=range(7),
        names=["time", *SERIES],
    )


def load_recordings(root: Path = DATA_DIR, *, drop_copies: bool = True) -> pd.DataFrame:
    """Columns: recording_id, folder, activity, session, time, then SERIES.

    session is the recording_id of the original, so a copy and its original share one.
    """
    frames = []
    for path in sorted(root.glob("*/dataset*.csv")):
        recording_id = f"{path.parent.name}/{path.stem}"
        if drop_copies and recording_id in COPIES:
            continue
        frame = read_recording(path)
        frame.insert(0, "recording_id", recording_id)
        frame.insert(1, "folder", path.parent.name)
        frame.insert(2, "activity", activity_of(path.parent.name))
        frame.insert(3, "session", COPIES.get(recording_id, recording_id))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def legacy_split(recordings: pd.DataFrame) -> tuple[list[str], list[str]]:
    """The HW3/HW4 notebook split, kept to reproduce the original numbers.

    Filenames were sorted as text, so the 15-file folders test on dataset1, dataset10 and
    dataset11. Both lists are text-sorted to match the notebook's groupby("filename") row
    order, which sets the CV folds under shuffle=True.
    """
    if recordings["recording_id"].nunique() != 88:
        raise ValueError("legacy_split needs all 88 files: load_recordings(drop_copies=False)")
    train, test = [], []
    for folder, ids in recordings.groupby("folder")["recording_id"]:
        ids = sorted(ids.unique())
        n_test = 2 if folder.startswith("bending") else 3
        test += ids[:n_test]
        train += ids[n_test:]
    return sorted(train), sorted(test)
