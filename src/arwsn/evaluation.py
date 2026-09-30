"""Baselines, confidence intervals, and nested cross-validation."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import BaseCrossValidator, cross_val_score

ModelFactory = Callable[[], BaseEstimator]


def majority_baseline(y: pd.Series) -> float:
    """Accuracy of always predicting the most common class."""
    return y.value_counts(normalize=True).iloc[0]


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, min(center + half, 1.0)


def cv_scores_by_l(
    features_by_l: dict[int, pd.DataFrame],
    y: pd.Series,
    make_model: ModelFactory,
    cv: BaseCrossValidator,
    groups: pd.Series | None = None,
    n_jobs: int | None = None,
) -> pd.Series:
    """Mean CV accuracy for each l. Its maximum is the notebook's selection-biased estimate."""
    scores = {}
    for l, X in features_by_l.items():
        g = groups.loc[X.index] if groups is not None else None
        scores[l] = cross_val_score(make_model(), X, y.loc[X.index], cv=cv, groups=g, n_jobs=n_jobs).mean()
    return pd.Series(scores, name="cv_accuracy").rename_axis("l")


@dataclass(frozen=True)
class NestedResult:
    fold_scores: list[float]
    chosen_l: list[int]
    predictions: pd.Series

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores, ddof=1))


def nested_cv(
    features_by_l: dict[int, pd.DataFrame],
    y: pd.Series,
    make_model: ModelFactory,
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    groups: pd.Series | None = None,
    n_jobs: int | None = None,
) -> NestedResult:
    """Choose l inside each outer training fold, then score on recordings that choice never saw.

    Ties between l values go to the smallest l.
    """
    ids = y.index
    groups = groups.loc[ids] if groups is not None else None
    predictions = pd.Series(index=ids, dtype=object, name="predicted")
    fold_scores, chosen_l = [], []
    for train, test in outer_cv.split(ids, y, groups):
        train_ids, test_ids = ids[train], ids[test]
        inner_groups = groups.loc[train_ids] if groups is not None else None
        inner = cv_scores_by_l(
            {l: X.loc[train_ids] for l, X in features_by_l.items()}, y.loc[train_ids], make_model, inner_cv, inner_groups,
            n_jobs,
        )
        best_l = int(inner.index[inner.to_numpy().argmax()])
        model = make_model().fit(features_by_l[best_l].loc[train_ids], y.loc[train_ids])
        predicted = model.predict(features_by_l[best_l].loc[test_ids])
        predictions.loc[test_ids] = predicted
        fold_scores.append(float(np.mean(predicted == y.loc[test_ids].to_numpy())))
        chosen_l.append(best_l)
    return NestedResult(fold_scores, chosen_l, predictions)


def per_class_report(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precision, recall, F1, and support per class, plus a confusion matrix (rows are true classes)."""
    classes = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    report = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)
    return report, matrix
