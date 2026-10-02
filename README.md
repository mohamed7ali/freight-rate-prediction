# Freight Rate Prediction Challenge

Predicts the posted rate ($) of freight loads. Final model: LightGBM ensemble (3 seeds) on log(rate),
trained on cleaned data, validated with a time-based rolling-origin split.

## Run

```bash
python -m pip install -r requirements.txt
python validate.py     # time-based validation + ablations -> results/, figures/
python predict.py      # trains on all labeled data -> validation_predictions.csv + December predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python build_report.py # optional: rebuilds reports/report.pdf
```

## Layout

| Path | Purpose |
|---|---|
| `data/` | Provided CSVs (`train_test.csv`, `validation.csv`, templates) |
| `freight/features.py` | Cleaning + feature engineering (shared by validation and prediction) |
| `freight/modeling.py` | Model, target-noise screen, metrics |
| `validate.py` | Rolling-origin validation, tuning, ablations |
| `predict.py` | Final fit + submission files |
| `score.py` | Provided scorer |
| `results/` | Metrics tables written by `validate.py` |
| `reports/report.pdf` | Submission report |
| `validation_predictions.csv` | Submission file (`load_id,predicted_rate`) |

## Key decisions

- **Time-based split.** Labeled data is Jan-Oct 2025, validation is Nov-Dec 2025, so validation is forward-looking.
- **Cleaning.** Negative weights sign-flipped, missing weights kept as NaN + flag, ~1.4% corrupted targets
  (rate off by 3-5x or 0.2-0.4x) dropped from training only via out-of-fold residuals.
- **market_index / quote_signal excluded.** They hurt out-of-time accuracy and are not available for the December chart.
