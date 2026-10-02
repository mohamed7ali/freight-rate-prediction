"""Validation experiments: time-based split, rolling-origin folds, ablations.

Run:  python validate.py
Writes results/*.csv|json and figures/*.png used in the report.

Split logic
-----------
data/validation.csv covers 2025-11-01..2025-12-31 -- strictly AFTER the labeled
data (2025-01-01..2025-10-31). The final model is therefore a 1-2 month-ahead
forecast, so validation must be out-of-time too. A random split would leak
neighbouring-day information and be optimistic.

  * Fold 1: train Jan-Jun  -> test Jul-Aug
  * Fold 2: train Jan-Jul  -> test Aug-Sep   (folds 1-2 are used for tuning only)
  * Fold 3: train Jan-Aug  -> test Sep-Oct   (final untouched holdout, 2 months ahead)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from freight.features import build_features, city_levels
from freight.modeling import BASE_PARAMS, fit_ensemble, flag_corrupted_targets, metrics, predict_rate

DATA = Path("data")
RES, FIG = Path("results"), Path("figures")
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

FOLDS = [("2025-06-30", "2025-08-31"), ("2025-07-31", "2025-09-30"), ("2025-08-31", "2025-10-31")]
GRID = [
    dict(num_leaves=15, min_child_samples=60, n_estimators=600),
    dict(num_leaves=31, min_child_samples=40, n_estimators=600),
    dict(num_leaves=15, min_child_samples=100, n_estimators=900),
    dict(num_leaves=63, min_child_samples=30, n_estimators=400),
]


def split(df, end, test_end):
    d = pd.to_datetime(df["date"])
    return (d <= end).values, ((d > end) & (d <= test_end)).values


def run_fold(df, X, y_log, tr, te, params=None, clean=True):
    keep = flag_corrupted_targets(X[tr], y_log[tr], params) if clean else np.ones(tr.sum(), bool)
    Xt, yt = X[tr][keep], y_log[tr][keep]
    models = fit_ensemble(Xt, yt, params, seeds=(0,))
    pred = predict_rate(models, X[te])
    return pred, int((~keep).sum())


def main():
    dev = pd.read_csv(DATA / "train_test.csv")
    val = pd.read_csv(DATA / "validation.csv")
    cities = city_levels(dev, val)
    y_log = np.log(dev["posted_rate"])
    X = build_features(dev, cities)
    actual = dev["posted_rate"]

    # ---- 1. tune on folds 1-2 only ---------------------------------------
    tune_rows = []
    for i, params in enumerate(GRID):
        scores = []
        for end, tend in FOLDS[:2]:
            tr, te = split(dev, end, tend)
            pred, _ = run_fold(dev, X, y_log, tr, te, params)
            scores.append(metrics(actual[te], pred)["MAPE_%"])
        tune_rows.append({"config": i, **params, "mean_MAPE_%": float(np.mean(scores))})
        print("tune", tune_rows[-1])
    tune = pd.DataFrame(tune_rows); tune.to_csv(RES / "tuning.csv", index=False)
    best = GRID[int(tune["mean_MAPE_%"].idxmin())]
    json.dump(best, open(RES / "best_params.json", "w"), indent=2)
    print("best params:", best)

    # ---- 2. final holdout (fold 3) + ablations ------------------------------
    end, tend = FOLDS[2]
    tr, te = split(dev, end, tend)

    # Reporting-only corruption flag using the whole dev set (never used for training the holdout model)
    keep_all = flag_corrupted_targets(X, y_log, best)
    clean_te = keep_all[te]

    rows = []

    def record(name, pred, note=""):
        m_all = metrics(actual[te], pred)
        m_clean = metrics(actual[te][clean_te], pred[clean_te])
        rows.append({"model": name, **{f"{k}": v for k, v in m_all.items() if k != "n"},
                     "MAPE_clean_%": m_clean["MAPE_%"], "MAE_clean": m_clean["MAE"], "note": note})

    # baseline: median $/mile per distance bucket
    bins = [0, 150, 300, 600, 1000, 1500, 2000, 3500]
    b = pd.cut(dev["distance"], bins)
    rpm = (dev["posted_rate"] / dev["distance"])[tr].groupby(b[tr]).median()
    record("Baseline: median $/mile by distance bucket x distance",
           (b[te].map(rpm).astype(float) * dev["distance"][te]).values)

    pred, dropped = run_fold(dev, X, y_log, tr, te, best, clean=False)
    record("LightGBM, no target cleaning", pred)
    pred_final, dropped = run_fold(dev, X, y_log, tr, te, best, clean=True)
    record("LightGBM + target cleaning (FINAL)", pred_final, f"{dropped} train rows dropped")

    Xm = build_features(dev, cities, extra=["market_index", "quote_signal"])
    pred, _ = run_fold(dev, Xm, y_log, tr, te, best, clean=True)
    record("  + market_index & quote_signal as features", pred, "worse out-of-time")

    # random 80/20 split: shows how optimistic a non-temporal split would be
    rs = np.zeros(len(dev), bool)
    for a, b_ in KFold(5, shuffle=True, random_state=0).split(dev):
        rs[b_] = True; break
    rtr, rte = ~rs, rs
    keep = flag_corrupted_targets(X[rtr], y_log[rtr], best)
    mdl = fit_ensemble(X[rtr][keep], y_log[rtr][keep], best, seeds=(0,))
    mr = metrics(actual[rte], predict_rate(mdl, X[rte]))
    keepm = flag_corrupted_targets(Xm[rtr], y_log[rtr], best)
    mdlm = fit_ensemble(Xm[rtr][keepm], y_log[rtr][keepm], best, seeds=(0,))
    mrm = metrics(actual[rte], predict_rate(mdlm, Xm[rte]))
    rows.append({"model": "(random 80/20 split) final features", "MAE": mr["MAE"], "RMSE": mr["RMSE"],
                 "MAPE_%": mr["MAPE_%"], "MedAPE_%": mr["MedAPE_%"], "note": "random split; not representative of a forward forecast"})
    rows.append({"model": "(random 80/20 split) + market_index & quote_signal", "MAE": mrm["MAE"], "RMSE": mrm["RMSE"],
                 "MAPE_%": mrm["MAPE_%"], "MedAPE_%": mrm["MedAPE_%"], "note": "looks great on a random split (leakage) but fails out-of-time"})
    out = pd.DataFrame(rows).round(3); out.to_csv(RES / "holdout_metrics.csv", index=False)
    print(out.to_string())

    # ---- 3. rolling-origin summary for the final config --------------------
    roll = []
    for k, (e, t) in enumerate(FOLDS, 1):
        a, c = split(dev, e, t)
        p, dr = run_fold(dev, X, y_log, a, c, best)
        roll.append({"fold": k, "train_end": e, "test_window": f"{pd.Timestamp(e) + pd.Timedelta(days=1):%Y-%m-%d}..{t}",
                     **metrics(actual[c], p), "dropped_train_rows": dr})
    pd.DataFrame(roll).round(3).to_csv(RES / "rolling_origin.csv", index=False)
    print(pd.DataFrame(roll).round(3).to_string())

    # ---- 4. cleaning summary + figures --------------------------------------
    summary = {
        "dev_rows": int(len(dev)),
        "negative_weight_rows": int((dev["weight"] < 0).sum()),
        "missing_weight_rows": int(dev["weight"].isna().sum()),
        "missing_market_index_rows": int(dev["market_index"].isna().sum()),
        "corrupted_target_rows_flagged": int((~keep_all).sum()),
        "corrupted_target_share_%": round(float((~keep_all).mean() * 100), 2),
        "val_negative_weight_rows": int((val["weight"] < 0).sum()),
        "val_missing_weight_rows": int(val["weight"].isna().sum()),
    }
    json.dump(summary, open(RES / "data_quality_summary.json", "w"), indent=2); print(summary)

    # residual histogram on the holdout
    res = np.log(actual[te].values) - np.log(pred_final)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    ax[0].hist(np.clip(res, -1.5, 1.8), bins=120, color="#064A56"); ax[0].set_yscale("log")
    ax[0].axvline(-0.3, color="crimson", ls="--"); ax[0].axvline(0.3, color="crimson", ls="--")
    ax[0].set_title("Holdout log-residuals (dashed = cleaning cutoff)"); ax[0].set_xlabel("log(actual) - log(predicted)")
    samp = np.random.default_rng(0).choice(te.sum(), 3000, replace=False)
    a_, p_ = actual[te].values[samp], pred_final[samp]
    ax[1].scatter(p_, a_, s=4, alpha=.4, color="#064A56"); lim = [0, max(a_.max(), p_.max())]
    ax[1].plot(lim, lim, color="crimson", lw=1); ax[1].set_xlabel("Predicted ($)"); ax[1].set_ylabel("Actual ($)")
    ax[1].set_title("Holdout (Sep-Oct): actual vs predicted, 3,000 sampled loads")
    plt.tight_layout(); plt.savefig(FIG / "holdout_diagnostics.png"); plt.close()


if __name__ == "__main__":
    main()
