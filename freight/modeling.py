"""Model definition, target-noise cleaning and evaluation helpers."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEEDS = (0, 1, 2)
CLEAN_THRESHOLD = 0.30  # |log residual| above this => corrupted target (~9x the MAD)

BASE_PARAMS = dict(
    objective="regression", learning_rate=0.03, n_estimators=600, num_leaves=15,
    min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
)


def flag_corrupted_targets(X: pd.DataFrame, y_log: pd.Series, params: dict | None = None,
                           threshold: float = CLEAN_THRESHOLD, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold residual screen. Returns a boolean array: True = keep row.

    A robust (Huber) model is fit on 4/5 of the data and predicts the other 1/5,
    so a corrupted row can never "explain itself". Rows whose actual log-rate is
    more than `threshold` away from the out-of-fold prediction are dropped from
    TRAINING only.
    """
    p = {**BASE_PARAMS, **(params or {}), "n_estimators": 300, "objective": "huber", "alpha": 0.3}
    oof = np.zeros(len(X))
    for tr, va in KFold(n_splits, shuffle=True, random_state=1).split(X):
        m = lgb.LGBMRegressor(**p).fit(X.iloc[tr], y_log.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    return np.abs(y_log.values - oof) < threshold


def fit_ensemble(X: pd.DataFrame, y_log: pd.Series, params: dict | None = None, seeds=SEEDS):
    p = {**BASE_PARAMS, **(params or {})}
    return [lgb.LGBMRegressor(**p, random_state=s).fit(X, y_log) for s in seeds]


def predict_rate(models, X: pd.DataFrame) -> np.ndarray:
    """Average in log space, then convert back to dollars."""
    return np.exp(np.mean([m.predict(X) for m in models], axis=0))


def metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    ape = np.abs(actual - pred) / actual
    return {
        "MAE": float(np.mean(np.abs(actual - pred))),
        "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "MAPE_%": float(ape.mean() * 100),
        "MedAPE_%": float(np.median(ape) * 100),
        "n": int(len(actual)),
    }
