"""The original notebook's feature code, copied verbatim as a test oracle."""

import numpy as np


def split_time_series(df, l):
    n = len(df)
    edges = np.linspace(0, n, l + 1, dtype=int)
    return [df.iloc[edges[i]:edges[i + 1]] for i in range(l)]


def feature_extraction_ts2(df, l):
    ts_cols = [1, 2, 3, 4, 5, 6]
    features = []

    for i in ts_cols:
        subseries_list = split_time_series(df.iloc[:, i], l)
        for sub in subseries_list:
            sub = sub.dropna()
            features.extend([
                sub.min(),
                sub.quantile(0.25),
                sub.median(),
                sub.mean(),
                sub.quantile(0.75),
                sub.max(),
                sub.std(),
            ])
    return features


def notebook_columns(l):
    cols = []
    for i in range(1, l + 1):
        for j in [1, 2, 3, 4, 5, 6]:
            for stat in ['min', 'q1', 'median', 'mean', 'q3', 'max', 'std']:
                cols.append(f"{stat}{j}_sub{i}")
    return cols
