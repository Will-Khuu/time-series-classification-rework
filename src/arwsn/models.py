"""Model factories. Each returns a fresh, unfitted estimator so every CV fold refits from scratch."""

from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def notebook_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def notebook_binary_rfecv() -> RFECV:
    """HW4 cell 25: a nearly unregularized logistic regression (C=1e10) with recursive feature elimination.

    RFECV keeps the feature count with the best CV score, and the notebook then reported that best
    score, so its "CV accuracy" is a maximum over feature counts and over l.
    """
    return RFECV(
        LogisticRegression(solver="liblinear", C=1e10, max_iter=1000, random_state=SEED),
        step=1,
        cv=notebook_cv(),
        scoring="accuracy",
    )


def l1_pipeline() -> Pipeline:
    """HW4 cell 47: scale, keep features an L1 model with CV-tuned strength selects, then fit an L1 model.

    Every step refits inside each CV fold. Used for both the multiclass and the binary task.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectFromModel(
            LogisticRegressionCV(l1_ratios=(1,), solver="saga", max_iter=1000, cv=notebook_cv(),
                                 scoring="accuracy", random_state=SEED)
        )),
        ("clf", LogisticRegression(l1_ratio=1, solver="saga", max_iter=1000, random_state=SEED)),
    ])
