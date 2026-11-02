from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
REPORT_PATH = ROOT / "reports" / "THRESHOLD_TRADEOFF_REPORT.md"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

target = "fault_d7"
excluded = [
    "timestamp",
    target,
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
]

X = df.drop(columns=excluded)
y = df[target].astype(int)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

lines = [
    "# AquaSenseAI — Validation Threshold Trade-off Report",
    "",
    "## Evaluation boundaries",
    "",
    "- Chronological training split: first 70% of records.",
    "- Validation split: next 15%; used for this threshold comparison.",
    "- Final 15% test split: not evaluated or used for threshold selection.",
    "- Two groundwater anomaly features excluded for this experiment.",
    "- Target is `fault_d7`, a leak-proximity label, not confirmation of an active leak.",
    "",
    "## Validation results",
    "",
    "| Threshold | TN | FP | FN | TP | False-alarm rate | Positive recall | Negative recall |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    positive_recall = tp / (tp + fn) if (tp + fn) else 0.0
    negative_recall = tn / (tn + fp) if (tn + fp) else 0.0
    lines.append(
        f"| {threshold:.2f} | {tn} | {fp} | {fn} | {tp} | "
        f"{false_alarm_rate:.1%} | {positive_recall:.1%} | {negative_recall:.1%} |"
    )

lines.extend([
    "",
    "## Interpretation",
    "",
    "- False-alarm rate is the fraction of actual class-0 validation rows predicted as class 1.",
    "- Positive recall is the fraction of actual class-1 rows predicted as class 1.",
    "- Negative recall is the fraction of actual class-0 rows predicted as class 0.",
    "- Higher thresholds generally reduce false alarms while potentially increasing missed positive cases.",
    "- Threshold choice requires an explicit operational trade-off; no threshold is declared final here.",
    "- These results concern the dataset label and do not establish real-world leak detection performance.",
    "",
])

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Report created: {REPORT_PATH}")
print(f"Validation rows analyzed: {len(X_val)}")
print("Final test split: NOT evaluated")
