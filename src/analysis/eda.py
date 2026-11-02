from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "timestamp" not in df.columns or "fault_d7" not in df.columns:
        raise ValueError("Required columns 'timestamp' and 'fault_d7' are missing.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    lines = [
        "AquaSenseAI - Exploratory Data Analysis Summary",
        "=" * 52,
        "",
        f"Source file: {DATA_PATH.relative_to(PROJECT_ROOT)}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Duplicate complete rows: {df.duplicated().sum()}",
        f"Invalid timestamps: {df['timestamp'].isna().sum()}",
        "",
        "COLUMN DATA TYPES",
        "-" * 30,
        df.dtypes.to_string(),
        "",
        "MISSING VALUES",
        "-" * 30,
        df.isna().sum().sort_values(ascending=False).to_string(),
        "",
        "TARGET DISTRIBUTION (fault_d7)",
        "-" * 30,
        df["fault_d7"].value_counts(dropna=False).sort_index().to_string(),
        "",
        "TARGET PERCENTAGES",
        "-" * 30,
        (
            df["fault_d7"].value_counts(normalize=True, dropna=False)
            .sort_index().mul(100).round(2).astype(str).add("%").to_string()
        ),
        "",
        "NUMERIC FEATURE SUMMARY",
        "-" * 30,
        df.select_dtypes(include="number").describe().T.round(3).to_string(),
        "",
    ]

    valid_timestamps = df["timestamp"].dropna()

    if not valid_timestamps.empty:
        lines.extend([
            "TIMESTAMP RANGE",
            "-" * 30,
            f"Start: {valid_timestamps.min()}",
            f"End: {valid_timestamps.max()}",
            "",
        ])

    time_df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if len(time_df) > 1:
        intervals = time_df["timestamp"].diff().dropna()
        lines.extend([
            "CHRONOLOGICAL INTERVAL CHECK",
            "-" * 30,
            f"One-hour intervals: {(intervals == pd.Timedelta(hours=1)).sum()}",
            f"Intervals longer than one hour: {(intervals > pd.Timedelta(hours=1)).sum()}",
            f"Non-positive intervals: {(intervals <= pd.Timedelta(0)).sum()}",
            "",
        ])

    (REPORT_DIR / "eda_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    numeric_features = df.select_dtypes(include="number").columns.drop(
        "fault_d7", errors="ignore"
    )

    results = []

    for feature in numeric_features:
        zero_rates = df.groupby("fault_d7")[feature].apply(
            lambda values: round(values.eq(0).mean() * 100, 2)
        )

        results.append({
            "feature": feature,
            "zero_pct_label_0": zero_rates.get(0, float("nan")),
            "zero_pct_label_1": zero_rates.get(1, float("nan")),
            "missing_count": int(df[feature].isna().sum()),
        })

    results_df = pd.DataFrame(results).sort_values(
        "zero_pct_label_1", ascending=False
    )

    (REPORT_DIR / "zero_pattern_analysis.txt").write_text(
        results_df.to_string(index=False), encoding="utf-8"
    )

    print("EDA completed successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Invalid timestamps: {df['timestamp'].isna().sum()}")
    print("\nZero patterns by target class:")
    print(results_df.to_string(index=False))
    print("\nReports saved:")
    print(REPORT_DIR / "eda_summary.txt")
    print(REPORT_DIR / "zero_pattern_analysis.txt")


if __name__ == "__main__":
    main()
