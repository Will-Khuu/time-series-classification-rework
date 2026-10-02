"""Run every experiment and write results/metrics.json.

Parts:
  reproduce  the notebook's method on the notebook's split, compared against its printed scores
  nested     nested cross-validation on the 81 unique recordings
  curve      plain CV accuracy for each l on the 81 unique recordings

Usage: python scripts/run_experiments.py [reproduce|nested|curve ...]
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from arwsn.evaluation import cv_scores_by_l, majority_baseline, nested_cv, per_class_report, wilson_interval
from arwsn.features import extract_features, labels, sessions
from arwsn.loading import legacy_split, load_recordings
from arwsn.models import SEED, l1_pipeline, notebook_binary_rfecv, notebook_cv

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics.json"
L_VALUES = range(1, 21)

# Printed by HW4 cells 25 and 47 for l = 1..20.
NOTEBOOK_BINARY_CV = [1.0, 0.9571428571428571, 0.9714285714285715, 0.9714285714285715, 0.9857142857142858,
                      0.9714285714285715, 0.9560439560439562, 0.9714285714285715, 0.9703296703296704,
                      0.956043956043956, 0.9714285714285715, 1.0, 0.9857142857142858, 0.9703296703296704,
                      0.9857142857142858, 0.9857142857142858, 0.9857142857142858, 1.0, 1.0, 1.0]
NOTEBOOK_MULTI_CV = [0.913, 0.868, 0.840, 0.852, 0.824, 0.823, 0.807, 0.795, 0.793, 0.767,
                     0.737, 0.812, 0.780, 0.781, 0.854, 0.678, 0.695, 0.781, 0.723, 0.810]
NOTEBOOK_BINARY_TEST, NOTEBOOK_MULTI_TEST = 1.0, 1 - 0.10526315789473684


def summarize(correct: int, n: int, baseline: float) -> dict:
    low, high = wilson_interval(correct, n)
    return {"correct": correct, "n": n, "accuracy": correct / n, "ci95": [low, high], "majority_baseline": baseline}


def reproduce() -> dict:
    recordings = load_recordings(drop_copies=False)
    train, test = legacy_split(recordings)
    activity, session = labels(recordings), sessions(recordings)
    leaked = [t for t in test if session[t] in set(session[train])]
    tasks = {
        "binary": ((activity == "bending").astype(int), NOTEBOOK_BINARY_CV, NOTEBOOK_BINARY_TEST),
        "multiclass": (activity, NOTEBOOK_MULTI_CV, NOTEBOOK_MULTI_TEST),
    }
    out = {"test_recordings_with_copy_in_train": leaked}
    for task, (y, notebook_cv_scores, notebook_test) in tasks.items():
        cv_scores, models = [], []
        for l in L_VALUES:
            X, y_train = extract_features(recordings, l).loc[train], y.loc[train]
            if task == "binary":
                model = notebook_binary_rfecv().fit(X, y_train)
                cv_scores.append(float(np.max(model.cv_results_["mean_test_score"])))
            else:
                cv_scores.append(float(cross_val_score(l1_pipeline(), X, y_train, cv=notebook_cv(), n_jobs=-1).mean()))
                model = l1_pipeline().fit(X, y_train)
            models.append(model)
            print(f"  {task} l={l:2d} cv={cv_scores[-1]:.3f} notebook={notebook_cv_scores[l - 1]:.3f}", flush=True)
        best = int(np.argmax(cv_scores))
        X_test = extract_features(recordings, best + 1).loc[test]
        predicted = pd.Series(models[best].predict(X_test), index=test)
        clean = [t for t in test if t not in leaked]
        out[task] = {
            "cv_by_l": cv_scores,
            "notebook_cv_by_l": notebook_cv_scores,
            "max_abs_diff_vs_notebook": float(np.max(np.abs(np.array(cv_scores) - np.array(notebook_cv_scores)))),
            "best_l": best + 1,
            "reported_cv": cv_scores[best],
            "test": summarize(int((predicted == y.loc[test]).sum()), len(test), majority_baseline(y.loc[test])),
            "notebook_test_accuracy": notebook_test,
            "test_without_leaked": summarize(int((predicted.loc[clean] == y.loc[clean]).sum()), len(clean),
                                             majority_baseline(y.loc[clean])),
        }
    return out


def unique_data() -> tuple[dict[int, pd.DataFrame], pd.Series, pd.Series]:
    recordings = load_recordings()
    return {l: extract_features(recordings, l) for l in L_VALUES}, labels(recordings), sessions(recordings)


NESTED_TASKS = {
    "multiclass": (lambda y: y, l1_pipeline),
    "binary": (lambda y: y == "bending", l1_pipeline),
    "binary_notebook_rfecv": (lambda y: y == "bending", notebook_binary_rfecv),
}


def nested() -> dict:
    features, activity, groups = unique_data()
    out = {}
    for task, (target, make_model) in NESTED_TASKS.items():
        y = target(activity)
        result = nested_cv(features, y, make_model,
                           StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                           StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1), groups=groups, n_jobs=-1)
        correct = int((result.predictions == y).sum())
        report, matrix = per_class_report(y.astype(str), result.predictions.astype(str))
        out[task] = {
            **summarize(correct, len(y), majority_baseline(y)),
            "fold_accuracy": result.fold_scores,
            "chosen_l": result.chosen_l,
            "per_class": report.to_dict(orient="index"),
            "confusion_matrix": {"labels": list(matrix.index), "rows": matrix.to_numpy().tolist()},
            "misclassified": {rid: [str(y[rid]), str(p)] for rid, p in result.predictions.items() if p != y[rid]},
        }
        print(f"  {task}: {correct}/{len(y)} chosen_l={result.chosen_l}", flush=True)
    return out


def curve() -> dict:
    features, activity, groups = unique_data()
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    out = {}
    for task, target in [("multiclass", lambda y: y), ("binary", lambda y: y == "bending")]:
        scores = cv_scores_by_l(features, target(activity), l1_pipeline, cv, groups=groups, n_jobs=-1)
        out[task] = {"cv_by_l": scores.tolist(), "n_features_by_l": [42 * l for l in L_VALUES],
                     "naive_best_l": int(scores.idxmax()), "naive_best_cv": float(scores.max())}
        print(f"  {task}: best l={out[task]['naive_best_l']} cv={out[task]['naive_best_cv']:.3f}", flush=True)
    return out


def main(parts: list[str]) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*(did not converge|l1_ratios|fitted attributes).*")
    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    runners = {"reproduce": reproduce, "nested": nested, "curve": curve}
    for part in parts:
        start = time.time()
        print(f"running {part}", flush=True)
        metrics[part] = runners[part]()
        metrics[part]["seconds"] = round(time.time() - start, 1)
        METRICS.parent.mkdir(exist_ok=True)
        METRICS.write_text(json.dumps(metrics, indent=2))
        print(f"{part} done in {metrics[part]['seconds']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["reproduce", "nested", "curve"])
