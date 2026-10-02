"""Draw the README figures from results/metrics.json into results/figures/.

Usage: python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from arwsn.evaluation import wilson_interval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HONEST, COURSE, SECOND = "#2a78d6", "#8a8984", "#eb6834"
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def dimensionality(metrics: dict) -> None:
    curve = metrics["curve"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for task, color, name in [("multiclass", HONEST, "6 activities"), ("binary", SECOND, "Bending vs other")]:
        x, y = curve[task]["n_features_by_l"], np.array(curve[task]["cv_by_l"]) * 100
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE)
        ax.annotate(name, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK_2)
    best = curve["multiclass"]
    ax.annotate(f"{best['naive_best_cv'] * 100:.1f}% with 42 features", (42, best["naive_best_cv"] * 100),
                xytext=(10, -16), textcoords="offset points", color=INK)
    ax.set(xlabel="Features per recording", ylabel="Cross-validated accuracy (%)",
           ylim=(60, 100), xlim=(0, 1000))
    ax.set_title("Six-activity accuracy falls as features grow (5-fold CV, 81 recordings)")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_vs_features.png", dpi=200)
    plt.close(fig)


def confusion(metrics: dict) -> None:
    cm = metrics["nested"]["multiclass"]["confusion_matrix"]
    labels, rows = cm["labels"], np.array(cm["rows"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    cmap = matplotlib.colors.ListedColormap(SEQUENTIAL)
    ax.imshow(rows, cmap=cmap, vmin=0, vmax=rows.max())
    for i, j in np.ndindex(rows.shape):
        if rows[i, j]:
            ax.text(j, i, rows[i, j], ha="center", va="center", color=SURFACE if rows[i, j] > rows.max() / 2 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    m = metrics["nested"]["multiclass"]
    ax.set_title(f"6 activities: {m['correct']} of {m['n']} correct")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def original_vs_honest(metrics: dict) -> None:
    rep, nest = metrics["reproduce"], metrics["nested"]
    rows = [
        ("6 activities, course version (19)", rep["multiclass"]["test"], COURSE),
        ("6 activities, nested CV (81)", nest["multiclass"], HONEST),
        ("Bending, course version (19)", rep["binary"]["test"], COURSE),
        ("Bending, nested CV, course method (81)", nest["binary_notebook_rfecv"], HONEST),
        ("Bending, nested CV, L1 pipeline (81)", nest["binary"], HONEST),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    for i, (name, r, color) in enumerate(reversed(rows)):
        low, high = wilson_interval(r["correct"], r["n"])
        ax.plot([low * 100, high * 100], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(r["accuracy"] * 100, i, "o", color=color, markersize=8, markeredgewidth=0)
        ax.plot(r["majority_baseline"] * 100, i, "|", color=INK_2, markersize=12, markeredgewidth=2)
        ax.annotate(f"{r['accuracy'] * 100:.1f}%", (high * 100, i), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK)
    ax.set_yticks(range(len(rows)), [name for name, _, _ in reversed(rows)])
    ax.set(xlim=(0, 115), xlabel="Accuracy (%)")
    ax.text(0, -0.28, "Dot: estimate.  Bar: 95% Wilson interval.  |: majority-class baseline.", transform=ax.transAxes,
            color=INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title("Course version vs nested cross-validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "original_vs_honest.png", dpi=200)
    plt.close(fig)


def main() -> None:
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    dimensionality(metrics)
    confusion(metrics)
    original_vs_honest(metrics)
    print(f"wrote {sorted(p.name for p in FIGURES.glob('*.png'))}")


if __name__ == "__main__":
    main()
