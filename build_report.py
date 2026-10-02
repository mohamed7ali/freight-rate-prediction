"""Builds reports/report.pdf from results/ and figures/."""
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ss = getSampleStyleSheet()
H = ParagraphStyle("H", parent=ss["Heading2"], textColor=colors.HexColor("#064A56"), spaceBefore=10)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
doc = SimpleDocTemplate("reports/report.pdf", pagesize=letter, leftMargin=.8*inch, rightMargin=.8*inch, topMargin=.7*inch, bottomMargin=.7*inch)

hold = pd.read_csv("results/holdout_metrics.csv")
roll = pd.read_csv("results/rolling_origin.csv")
dq = json.load(open("results/data_quality_summary.json"))

def table(df, cols, widths):
    data = [cols] + [[f"{v:.2f}" if isinstance(v, float) else str(v) for v in r] for r in df[cols].values.tolist()]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#064A56")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t

S = []
S.append(Paragraph("Freight Rate Prediction: Validation Report", ss["Title"]))
S.append(Paragraph("1. Approach", H))
S.append(Paragraph("Model: LightGBM regression on log(posted_rate), averaged over 3 seeds, predictions converted back to dollars. "
    "Inputs: distance, log distance, |weight| + missing flag, equipment, pickup/delivery city and coordinates, straight-line distance, "
    "road/straight-line ratio, day of week. A median $/mile-by-distance baseline is used as the reference point.", B))
S.append(Paragraph("2. Data quality and handling", H))
S.append(Paragraph(f"&bull; <b>Negative weights</b> ({dq['negative_weight_rows']} dev, {dq['val_negative_weight_rows']} validation rows): the absolute values match the positive distribution (5k-47.5k lb), so they are treated as sign errors and flipped.<br/>"
    f"&bull; <b>Missing weight</b> ({dq['missing_weight_rows']} dev, {dq['val_missing_weight_rows']} validation): left as NaN with a missing flag; LightGBM handles it natively.<br/>"
    f"&bull; <b>Missing market_index</b> ({dq['missing_market_index_rows']} dev rows): irrelevant to the final model, which does not use it.<br/>"
    f"&bull; <b>Corrupted targets</b>: ~{dq['corrupted_target_share_%']}% of labeled rows ({dq['corrupted_target_rows_flagged']}) have a rate roughly 3-5x or 0.2-0.4x the expected value for the lane, spread evenly across months and unrelated to any feature. "
    "They are flagged with an out-of-fold Huber model (|log residual| &gt; 0.30) and removed from training only. Evaluation rows are never removed.<br/>"
    "&bull; <b>Short-haul distances</b>: distances appear floored near 70 miles, so a few short lanes have a distance far above straight-line distance. The rate follows the listed distance, so these were kept.", B))
S.append(Paragraph("3. Split and validation", H))
S.append(Paragraph("The labeled data covers Jan-Oct 2025 and the 12,000 validation loads cover Nov-Dec 2025, so the task is a forecast. "
    "I used expanding-window, time-based folds instead of a random split. Folds 1-2 were used for hyperparameter selection; fold 3 (train Jan-Aug, test Sep-Oct) is the final holdout, evaluated once after tuning.", B))
S.append(table(roll, ["fold", "test_window", "MAE", "RMSE", "MAPE_%", "MedAPE_%"], [.5*inch, 1.9*inch, .8*inch, .8*inch, .8*inch, .9*inch]))
S.append(Spacer(1, 8))
S.append(Paragraph("Final holdout (Sep-Oct) and ablations", H))
h = hold.copy(); h["model"] = h["model"].str.slice(0, 62)
S.append(table(h, ["model", "MAE", "RMSE", "MAPE_%", "MedAPE_%"], [3.2*inch, .65*inch, .65*inch, .65*inch, .75*inch]))
S.append(Spacer(1, 6))
S.append(Paragraph("<b>Findings.</b> (a) Cleaning corrupted targets cut holdout MAPE from 5.4% to 4.4%. "
    "(b) RMSE stays near 630 because the holdout itself contains ~1.5% corrupted rows that no model should predict; on non-corrupted rows MAPE is about 1.9%. "
    "(c) market_index and quote_signal improve a random split but hurt the forward split, so they are excluded. "
    "(d) The random split is a less reliable guide to forward performance, which is why the time-based split was used.", B))
S.append(Image("figures/holdout_diagnostics.png", width=6.6*inch, height=2.4*inch))
S.append(Paragraph("4. Fixed December chart (produced by score.py)", H))
S.append(Image("figures/candidate_december.png", width=6.6*inch, height=2.95*inch))
S.append(Paragraph("The chart shows a weekly cycle (lowest on Sundays, highest on Thursdays) of roughly $808-832 for the fixed Lexington to Fort Wayne dry-van load. "
    "Because market_index is not an input, the model has no month-level trend beyond the day-of-week pattern, which is why the pattern repeats each week.", B))
doc.build(S)
