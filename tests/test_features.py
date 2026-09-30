import numpy as np
import pytest

from arwsn.features import STATS, extract_features, feature_names, labels, sessions
from arwsn.loading import SERIES, load_recordings
from notebook_reference import feature_extraction_ts2, notebook_columns

L_VALUES = range(1, 21)


@pytest.fixture(scope="module")
def recordings(data_dir):
    return load_recordings(data_dir, drop_copies=False)


@pytest.mark.parametrize("l", L_VALUES)
def test_one_row_per_recording(recordings, l):
    features = extract_features(recordings, l)
    assert features.index.is_unique
    assert list(features.index) == sorted(recordings["recording_id"].unique())
    assert features.shape == (88, 42 * l)


@pytest.mark.parametrize("l", L_VALUES)
def test_values_match_notebook(recordings, l):
    features = extract_features(recordings, l)
    for rid, group in recordings.groupby("recording_id"):
        expected = feature_extraction_ts2(group[["time", *SERIES]], l)
        np.testing.assert_allclose(features.loc[rid].to_numpy(), expected, rtol=1e-12, err_msg=f"{rid} l={l}")


def test_names_describe_values(recordings):
    walk = recordings[recordings["recording_id"] == "walking/dataset2"]
    features = extract_features(walk, 3)
    for s, series in enumerate(SERIES, start=1):
        for seg, chunk in enumerate(np.array_split(walk[series].to_numpy(), [160, 320]), start=1):
            assert features.iloc[0][f"min{s}_sub{seg}"] == chunk.min()
            assert features.iloc[0][f"std{s}_sub{seg}"] == pytest.approx(chunk.std(ddof=1))


def test_notebook_names_were_wrong_beyond_l1():
    assert feature_names(1) == notebook_columns(1)
    wrong = sum(a != b for a, b in zip(feature_names(2), notebook_columns(2)))
    assert wrong == 70


def test_short_recording_segments(recordings):
    sitting8 = recordings[recordings["recording_id"] == "sitting/dataset8"]
    features = extract_features(sitting8, 2)
    assert features.iloc[0]["max1_sub2"] == sitting8["avg_rss12"].iloc[239:].max()


def test_labels_align_with_features(recordings):
    y = labels(recordings)
    assert list(y.index) == list(extract_features(recordings, 1).index)
    assert y["bending2/dataset4"] == "bending"
    assert y.value_counts().to_dict() == {
        "cycling": 15, "lying": 15, "sitting": 15, "standing": 15, "walking": 15, "bending": 13
    }


def test_stat_order():
    assert STATS == ["min", "q1", "median", "mean", "q3", "max", "std"]


def test_sessions_align_and_group_copies(recordings):
    groups = sessions(recordings)
    assert list(groups.index) == list(extract_features(recordings, 1).index)
    assert groups["lying/dataset15"] == groups["lying/dataset10"] == groups["lying/dataset1"] == "lying/dataset1"
    assert groups.nunique() == 81
