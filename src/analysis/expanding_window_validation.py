from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

X = df.drop(columns=[
    "timestamp", "fault_d7",
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
])
y = df["fault_d7"].astype(int)

n = len(df)
test_start = int(n * 0.85)

print("\nEXPANDING-WINDOW TIME VALIDATION")
print("=" * 65)
print("Final 15% test period remains untouched.")

for train_fraction, val_end_fraction in [
    (0.45, 0.55),
    (0.55, 0.65),
    (0.65, 0.75),
    (0.75, 0.85),
]:
    train_end = int(n * train_fraction)
    val_end = min(int(n * val_end_fraction), test_start)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

    print(f"\nTrain {train_fraction:.0%}; validation "
          f"{train_fraction:.0%}-{val_end_fraction:.0%}")
    print(f"Rows: train={len(X_train)}, validation={len(X_val)}")
    print(f"Validation positive rate: {y_val.mean():.4f}")

    models = [
        ("Majority baseline",
         DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
             ("scaler", StandardScaler()),
             ("classifier", LogisticRegression(
                 class_weight=None, max_iter=2000, random_state=42
             )),
         ])),
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        print(f"{name}: accuracy={accuracy_score(y_val, pred):.4f}; "
              f"balanced_accuracy={balanced_accuracy_score(y_val, pred):.4f}")

        if name == "Logistic Regression":
            prob = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
            print(f"  ROC-AUC={roc_auc_score(y_val, prob):.4f}")

print("\nCompleted. Final test set was not used.")
