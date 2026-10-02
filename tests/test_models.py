import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from arwsn.models import l1_pipeline, notebook_binary_rfecv
import pytest


@pytest.mark.parametrize("make", [l1_pipeline, notebook_binary_rfecv])
def test_factories_return_fresh_unfitted_models(make):
    a, b = make(), make()
    assert a is not b
    with pytest.raises(Exception):
        check_is_fitted(a)
    assert clone(a).get_params().keys() == a.get_params().keys()


def test_l1_pipeline_selects_informative_features():
    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat(["sit", "stand", "walk"], 20))
    X = pd.DataFrame(rng.normal(size=(60, 10)), columns=[f"f{i}" for i in range(10)])
    X["f0"] += pd.factorize(y)[0] * 4
    model = l1_pipeline().fit(X, y)
    kept = X.columns[model.named_steps["feature_selection"].get_support()]
    assert "f0" in kept
    assert (model.predict(X) == y).mean() > 0.9


def test_notebook_rfecv_is_nearly_unregularized():
    estimator = notebook_binary_rfecv().estimator
    assert estimator.C == 1e10 and estimator.solver == "liblinear"
