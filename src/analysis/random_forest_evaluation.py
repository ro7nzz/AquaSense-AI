from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)
df = df.sort_values("timestamp").reset_index(drop=True)

target = "fault_d7"
X = df.drop(columns=["timestamp", target])
y = df[target].astype(int)

n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:validation_end], y.iloc[train_end:validation_end]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

print("\nAquaSenseAI - Random Forest Validation")
print("=" * 45)
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print("Final test split: NOT evaluated")
print(f"Validation positive rate: {y_val.mean():.4f}")
print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_val, predictions):.4f}")
print(f"PR-AUC (average precision): {average_precision_score(y_val, probabilities):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")
print("\nClassification report:")
print(classification_report(
    y_val,
    predictions,
    labels=[0, 1],
    target_names=["No leak-proximity label", "Leak-proximity label"],
    zero_division=0,
))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_val, predictions, labels=[0, 1]))

print("\nFeature importance (top 10):")
classifier = model.named_steps["classifier"]
importance = pd.Series(
    classifier.feature_importances_,
    index=model.named_steps["imputer"].get_feature_names_out(X.columns),
).sort_values(ascending=False)
print(importance.head(10).to_string())
