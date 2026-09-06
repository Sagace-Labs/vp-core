"""Shared XGBoost fitting.

The trainer lives here, the hyperparameters per pathway. A pathway
supplies its own ``params`` dict; this module owns only the mechanics that must
not vary across pathways: balanced positive weighting, early stopping, device
selection and the censored-potency fit.

Early stopping watches the validation (not test) fold only.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CensoredModel",
    "device",
    "fit_binary",
    "fit_censored",
    "predict_proba",
]


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


@dataclass
class CensoredModel:
    """A potency regressor plus the calibration that turns it into a probability.

    ``cutoff_um`` is the threshold the probability refers to.
    """

    booster: Any
    slope: float
    intercept: float
    cutoff_um: float

    def potency(self, X) -> np.ndarray:
        """Predicted potency in the units the interval was given in."""
        import xgboost as xgb

        return np.asarray(self.booster.predict(xgb.DMatrix(X)), dtype=np.float64)

    def margin(self, X) -> np.ndarray:
        """Decades of potency below the cutoff; positive means more potent."""
        return _margin(self.potency(X), self.cutoff_um)

    def probability(self, X) -> np.ndarray:
        """P(potency is below ``cutoff_um``)."""
        z = self.slope * self.margin(X) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def _margin(potency_um, cutoff_um: float) -> np.ndarray:
    potency = np.clip(np.asarray(potency_um, dtype=np.float64), 1e-9, None)
    return np.log10(cutoff_um) - np.log10(potency)


def fit_censored(
    X_train,
    lower_train,
    upper_train,
    X_val,
    lower_val,
    upper_val,
    y_val,
    *,
    params: dict[str, Any],
    cutoff_um: float,
    seed: int = 0,
) -> CensoredModel:
    """Fit potency as an interval, then calibrate it against the cutoff.

    Bounds are ``(lower, upper)`` per compound: equal for an exact measurement,
    and ``upper = inf`` for one observed only above an assay ceiling.

    The calibration is a two-parameter logistic on the validation fold.
    """
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

    dtrain, dvalid = xgb.DMatrix(X_train), xgb.DMatrix(X_val)
    bounds = ((dtrain, lower_train, upper_train), (dvalid, lower_val, upper_val))
    for matrix, low, high in bounds:
        matrix.set_float_info("label_lower_bound", np.asarray(low, dtype=np.float64))
        matrix.set_float_info("label_upper_bound", np.asarray(high, dtype=np.float64))

    booster = xgb.train(
        {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "device": device(),
            "eta": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 6),
            "min_child_weight": params.get("min_child_weight", 1.0),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "lambda": params.get("reg_lambda", 2.0),
            "alpha": params.get("reg_alpha", 0.05),
            "gamma": params.get("gamma", 0.1),
            "seed": seed,
            "nthread": -1,
        },
        dtrain,
        num_boost_round=params.get("n_estimators", 2000),
        evals=[(dvalid, "valid")],
        early_stopping_rounds=params.get("early_stopping_rounds", 40),
        verbose_eval=False,
    )

    margin = _margin(np.asarray(booster.predict(dvalid), dtype=np.float64), cutoff_um)
    y_val = np.asarray(y_val).astype(int)
    if len(np.unique(y_val)) < 2:
        # Nothing to calibrate against; fall back to a fixed half-decade width.
        return CensoredModel(booster, slope=2.0, intercept=0.0, cutoff_um=cutoff_um)
    calibrator = LogisticRegression(max_iter=5000)
    calibrator.fit(margin.reshape(-1, 1), y_val)
    return CensoredModel(
        booster,
        slope=float(calibrator.coef_[0][0]),
        intercept=float(calibrator.intercept_[0]),
        cutoff_um=cutoff_um,
    )
