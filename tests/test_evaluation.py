import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from statsmodels.stats.proportion import proportion_confint

from arwsn.evaluation import cv_scores_by_l, majority_baseline, nested_cv, per_class_report, wilson_interval


@pytest.mark.parametrize("k, n, low, high", [(19, 19, 0.832, 1.000), (17, 19, 0.686, 0.971)])
def test_wilson_known_values(k, n, low, high):
    assert wilson_interval(k, n) == pytest.approx((low, high), abs=5e-4)


@pytest.mark.parametrize("k, n", [(0, 10), (3, 10), (74, 81), (81, 81)])
def test_wilson_matches_statsmodels(k, n):
    assert wilson_interval(k, n) == pytest.approx(proportion_confint(k, n, method="wilson"), abs=1e-12)


def test_majority_baseline():
    assert majority_baseline(pd.Series(["other"] * 15 + ["bending"] * 4)) == pytest.approx(15 / 19)


def make_data(seed=0, n=60, informative_l=3, max_l=6):
    rng = np.random.default_rng(seed)
    ids = pd.Index([f"r{i}" for i in range(n)], name="recording_id")
    y = pd.Series(np.repeat(["a", "b", "c"], n // 3), index=ids)
    features = {}
    for l in range(1, max_l + 1):
        X = rng.normal(size=(n, 4))
        if l == informative_l:
            X[:, 0] += pd.factorize(y)[0] * 3
        features[l] = pd.DataFrame(X, index=ids)
    return features, y


def logistic():
    return LogisticRegression(max_iter=1000)


def test_nested_cv_finds_the_informative_l():
    features, y = make_data()
    result = nested_cv(features, y, logistic, StratifiedKFold(5, shuffle=True, random_state=0),
                       StratifiedKFold(4, shuffle=True, random_state=1))
    assert result.chosen_l == [3] * 5
    assert result.mean > 0.8
    assert result.predictions.notna().all() and (result.predictions.index == y.index).all()


def test_naive_max_is_optimistic_on_noise():
    gaps = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ids = pd.Index([f"r{i}" for i in range(40)])
        y = pd.Series(np.repeat(["a", "b"], 20), index=ids)
        features = {l: pd.DataFrame(rng.normal(size=(40, 3)), index=ids) for l in range(1, 21)}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        naive = cv_scores_by_l(features, y, logistic, cv).max()
        honest = nested_cv(features, y, logistic, cv, StratifiedKFold(4, shuffle=True, random_state=seed)).mean
        gaps.append(naive - honest)
    assert np.mean(gaps) > 0.05


class Spy(BaseEstimator, ClassifierMixin):
    seen: list = []

    def fit(self, X, y):
        Spy.seen.append(set(X.index))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_outer_test_rows_never_reach_selection_or_fitting():
    features, y = make_data()
    outer = KFold(4, shuffle=True, random_state=0)
    folds = list(outer.split(y.index))
    Spy.seen = []
    nested_cv(features, y, Spy, outer, KFold(3))
    fits_per_fold = len(Spy.seen) // len(folds)
    for f, (_, test) in enumerate(folds):
        held_out = set(y.index[test])
        for fitted in Spy.seen[f * fits_per_fold:(f + 1) * fits_per_fold]:
            assert fitted.isdisjoint(held_out)


def test_groups_keep_copies_in_one_fold():
    features, y = make_data()
    groups = pd.Series([f"g{i // 2}" for i in range(len(y))], index=y.index)
    Spy.seen = []
    outer = StratifiedGroupKFold(3, shuffle=True, random_state=0)
    nested_cv(features, y, Spy, outer, StratifiedGroupKFold(2), groups=groups)
    for fitted in Spy.seen:
        assert all((a in fitted) == (b in fitted) for a, b in zip(y.index[::2], y.index[1::2]))


def test_groups_matched_by_id_not_position():
    features, y = make_data()
    groups = pd.Series([f"g{i // 2}" for i in range(len(y))], index=y.index).sample(frac=1, random_state=0)
    Spy.seen = []
    nested_cv(features, y, Spy, StratifiedGroupKFold(3, shuffle=True, random_state=0), StratifiedGroupKFold(2),
              groups=groups)
    for fitted in Spy.seen:
        assert all((a in fitted) == (b in fitted) for a, b in zip(y.index[::2], y.index[1::2]))


def test_ties_go_to_smallest_l():
    features, y = make_data(informative_l=99)
    result = nested_cv(features, y, lambda: DummyClassifier(), StratifiedKFold(3), StratifiedKFold(2))
    assert result.chosen_l == [1, 1, 1]


def test_per_class_report():
    y_true = pd.Series(["sit", "sit", "stand", "stand", "walk"])
    y_pred = pd.Series(["sit", "stand", "stand", "stand", "walk"])
    report, matrix = per_class_report(y_true, y_pred)
    assert report.loc["sit", "recall"] == 0.5
    assert report.loc["stand", "precision"] == pytest.approx(2 / 3)
    assert report["support"].tolist() == [2, 2, 1]
    assert matrix.loc["sit", "stand"] == 1
    assert int(np.trace(matrix)) == 4


def test_runs_on_real_recordings(data_dir):
    from arwsn.features import extract_features, labels, sessions
    from arwsn.loading import load_recordings

    recordings = load_recordings(data_dir)
    y, groups = labels(recordings), sessions(recordings)
    features = {l: extract_features(recordings, l) for l in (1, 2)}
    result = nested_cv(features, y, lambda: DummyClassifier(), StratifiedGroupKFold(5, shuffle=True, random_state=0),
                       StratifiedGroupKFold(3), groups=groups)
    assert len(result.predictions) == 81 and result.predictions.notna().all()
    assert result.mean == pytest.approx(majority_baseline(y), abs=0.05)
