"""Train the final model on ALL labeled data and write the submission files.

Run:  python predict.py
Outputs:
  validation_predictions.csv          (load_id,predicted_rate; 12,000 rows)
  data/december_chart_inputs.csv      (same 7 columns, predicted_rate filled)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from freight.features import build_features, city_levels
from freight.modeling import BASE_PARAMS, fit_ensemble, flag_corrupted_targets, predict_rate

DATA = Path("data")


def main():
    dev = pd.read_csv(DATA / "train_test.csv")
    val = pd.read_csv(DATA / "validation.csv")
    tmpl = pd.read_csv(DATA / "validation_predictions_template.csv")
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")

    params_path = Path("results/best_params.json")
    params = json.load(open(params_path)) if params_path.exists() else {}
    cities = city_levels(dev, val, dec)

    # --- train on cleaned labeled data ---------------------------------------
    X = build_features(dev, cities)
    y_log = np.log(dev["posted_rate"])
    keep = flag_corrupted_targets(X, y_log, params)
    print(f"Dropping {(~keep).sum()} corrupted-target rows ({(~keep).mean():.2%}) from training")
    models = fit_ensemble(X[keep], y_log[keep], params)

    # --- 12,000 validation loads ---------------------------------------------
    pred = predict_rate(models, build_features(val, cities))
    out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(pred, 2)})
    out = tmpl[["load_id"]].merge(out, on="load_id", how="left")  # keep template order
    assert out["predicted_rate"].notna().all() and len(out) == 12_000 and (out["predicted_rate"] > 0).all()
    out.to_csv("validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv", out.shape)

    # --- December chart: fixed lane, only the date changes --------------------
    coords = {}
    for side in ("pickup", "delivery"):
        c = dev.groupby(side)[[f"{side}_lat", f"{side}_lon"]].first()
        coords[side] = c
    d = dec.copy()
    for side in ("pickup", "delivery"):
        d[f"{side}_lat"] = d[side].map(coords[side][f"{side}_lat"])
        d[f"{side}_lon"] = d[side].map(coords[side][f"{side}_lon"])
    assert d[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()
    dec["predicted_rate"] = np.round(predict_rate(models, build_features(d, cities)), 2)
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print(dec[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
