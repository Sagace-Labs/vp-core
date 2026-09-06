"""Binary classification metrics.

Every metric a protocol may name is defined here, so a protocol's ``metrics``
tuple is checkable against ``METRICS``. A fold with a single class yields NaN.
"""

from __future__ import annotations

import numpy as np

__all__ = ["METRICS", "aggregate", "binary_metrics"]

METRICS: tuple[str, ...] = ("auc_roc", "auprc", "mcc", "brier", "balanced_acc")


def binary_metrics(
    y_true, y_prob, *, threshold: float = 0.5
) -> dict[str, float]:
    """All of ``METRICS`` plus fold descriptors, for one set of predictions."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2

    return {
        "auc_roc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "auprc": float("nan")
        if single_class
        else float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "balanced_acc": float("nan")
        if single_class
        else float(balanced_accuracy_score(y_true, y_pred)),
        "n": len(y_true),
        "pos_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }


def aggregate(
    per_seed: list[dict[str, float]], metrics: tuple[str, ...]
) -> dict[str, dict]:
    """Collapse per-seed metric dicts into mean, std and the raw per-seed values."""
    out: dict[str, dict] = {}
    for name in metrics:
        vals = [float(m[name]) for m in per_seed]
        finite = [v for v in vals if np.isfinite(v)]
        out[name] = {
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "std": float(np.std(finite, ddof=0)) if finite else float("nan"),
            "per_seed": [round(v, 6) for v in vals],
            "n_finite": len(finite),
        }
    return out
