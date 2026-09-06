"""Shared XGBoost fitting.

The trainer lives here and the hyperparameters live in each pathway, because
the predecessor repo copied one identical XGB recipe into nine ``model.py``
files that then drifted. A pathway supplies its own ``params`` dict; this
module owns only the mechanics that must not vary: balanced positive weighting,
early stopping on the validation fold and nothing else, and device selection.

Early stopping watches the validation fold only. The test fold is never seen by
any fitting decision.
"""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np

__all__ = ["device", "fit_binary", "predict_proba"]


def device() -> str:
    """``"cuda"`` when an NVIDIA driver answers, else ``"cpu"``."""
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=3,
        )
        return "cuda"
    except Exception:
        return "cpu"


def fit_binary(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    params: dict[str, Any],
    seed: int = 0,
) -> Any:
    """Fit one binary XGBoost classifier with balanced class weighting."""
    import xgboost as xgb

    y_train = np.asarray(y_train).astype(int)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0

    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device=device(),
        eval_metric="auc",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, np.asarray(y_val).astype(int))], verbose=False)
    return model


def predict_proba(model, X) -> np.ndarray:
    """Positive-class probability as a 1-D float array."""
    return np.asarray(model.predict_proba(X))[:, 1].astype(np.float64)
