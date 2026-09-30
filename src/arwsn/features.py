"""Turn each recording into one row of time-domain statistics."""

import numpy as np
import pandas as pd

from arwsn.loading import SERIES

STATS = ["min", "q1", "median", "mean", "q3", "max", "std"]


def segment_bounds(n: int, l: int) -> np.ndarray:
    return np.linspace(0, n, l + 1, dtype=int)


def segment_stats(values: np.ndarray) -> np.ndarray:
    """Rows of values in, one row of STATS per column out."""
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    return np.stack(
        [values.min(axis=0), q1, median, values.mean(axis=0), q3, values.max(axis=0), values.std(axis=0, ddof=1)]
    )


def feature_names(l: int) -> list[str]:
    return [f"{stat}{s}_sub{seg}" for s in range(1, len(SERIES) + 1) for seg in range(1, l + 1) for stat in STATS]


def recording_features(values: np.ndarray, l: int) -> np.ndarray:
    """Series-major order: every segment of series 1, then series 2, as the notebook computed them."""
    edges = segment_bounds(len(values), l)
    per_segment = np.stack([segment_stats(values[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return per_segment.transpose(2, 0, 1).ravel()


def extract_features(recordings: pd.DataFrame, l: int) -> pd.DataFrame:
    """One row per recording, indexed by recording_id, 42 * l columns."""
    rows = {rid: recording_features(g[SERIES].to_numpy(), l) for rid, g in recordings.groupby("recording_id")}
    return pd.DataFrame.from_dict(rows, orient="index", columns=feature_names(l)).rename_axis("recording_id")


def labels(recordings: pd.DataFrame) -> pd.Series:
    """Activity per recording, in the same order as extract_features rows."""
    return recordings.groupby("recording_id")["activity"].first()


def sessions(recordings: pd.DataFrame) -> pd.Series:
    """Original recording per row, in extract_features order; use as CV groups so copies share a fold."""
    return recordings.groupby("recording_id")["session"].first()
