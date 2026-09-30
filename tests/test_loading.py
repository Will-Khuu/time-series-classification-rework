import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arwsn.loading import COPIES, SERIES, legacy_split, load_recordings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from check_independence import duplicate_pairs  # noqa: E402


@pytest.fixture(scope="module")
def everything(data_dir):
    return load_recordings(data_dir, drop_copies=False)


@pytest.fixture(scope="module")
def unique(data_dir):
    return load_recordings(data_dir)


def test_counts(everything, unique):
    assert everything["recording_id"].nunique() == 88
    assert unique["recording_id"].nunique() == 81


def test_columns_are_clean(everything):
    assert list(everything.columns) == ["recording_id", "folder", "activity", "session", "time", *SERIES]
    assert not everything.isna().any().any()
    assert all(everything[c].dtype == np.float64 for c in SERIES)


def test_rows_per_recording(everything):
    sizes = everything.groupby("recording_id").size()
    assert sizes["sitting/dataset8"] == 479
    assert (sizes.drop("sitting/dataset8") == 480).all()


def test_matches_notebook_parser(data_dir, everything):
    for rid, frame in everything.groupby("recording_id"):
        if rid == "bending2/dataset4":
            continue
        original = pd.read_csv(data_dir / f"{rid}.csv", skiprows=4, sep=None, engine="python", usecols=range(7))
        np.testing.assert_array_equal(frame[["time", *SERIES]].to_numpy(), original.to_numpy(), err_msg=rid)


def test_space_separated_file(data_dir, everything):
    path = data_dir / "bending2/dataset4.csv"
    frame = everything[everything["recording_id"] == "bending2/dataset4"]
    np.testing.assert_array_equal(frame[["time", *SERIES]].to_numpy(), np.loadtxt(path, comments="#"))


def test_activity_merges_bending_folders(everything):
    by_folder = everything.groupby("folder")["activity"].first()
    assert by_folder["bending1"] == by_folder["bending2"] == "bending"
    assert set(by_folder) == {"bending", "cycling", "lying", "sitting", "standing", "walking"}


def test_copies_constant_matches_content_scan(data_dir):
    groups = {}
    for a, b, _ in duplicate_pairs(data_dir):
        groups.setdefault(COPIES.get(a, a), set()).update({a, b})
    scanned = {rid: original for original, members in groups.items() for rid in members if rid != original}
    assert scanned == COPIES


def test_copies_share_session_with_original(everything):
    session = everything.groupby("recording_id")["session"].first()
    for copy, original in COPIES.items():
        assert session[copy] == session[original] == original


def test_drop_copies_keeps_originals(unique):
    kept = set(unique["recording_id"])
    assert kept.isdisjoint(COPIES)
    assert set(COPIES.values()) <= kept


def test_legacy_split_reproduces_notebook(everything):
    train, test = legacy_split(everything)
    assert len(train) == 69
    assert test == [
        f"{folder}/dataset{n}"
        for folder in ["bending1", "bending2", "cycling", "lying", "sitting", "standing", "walking"]
        for n in ([1, 2] if folder.startswith("bending") else [1, 10, 11])
    ]
    assert set(train).isdisjoint(test)
    assert train[:3] == ["bending1/dataset3", "bending1/dataset4", "bending1/dataset5"]
    assert train.index("cycling/dataset15") < train.index("cycling/dataset2")


def test_legacy_split_refuses_deduplicated_input(unique):
    with pytest.raises(ValueError, match="drop_copies=False"):
        legacy_split(unique)
