from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "missingness_periods.txt"

FEATURE_PAIRS = [
    ("326417_ws_temp", "326416_ws_vigor"),
    ("4408411600_kW", "4408411600_power_hour"),
    ("279625_ws_temp", "279623_ws_vigor"),
    ("gw_lvl_site2_anomaly_score", "gw_temp_site2_anomaly_score"),
    ("temp_site1_anomaly_score", "sfc_temp_site1_anomaly_score"),
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    lines = [
        "AquaSenseAI - Continuous Missingness Periods",
        "=" * 52,
        f"Dataset rows: {len(df)}",
        "",
    ]

    for feature_a, feature_b in FEATURE_PAIRS:
        if feature_a not in df.columns or feature_b not in df.columns:
            lines.extend([
                f"Pair: {feature_a} | {feature_b}",
                "Skipped: one or both columns are missing.",
                "",
            ])
            continue

        # Both features previously showed identical missingness.
        missing = df[feature_a].isna() & df[feature_b].isna()
        missing_rows = df.loc[missing, ["timestamp"]].copy()

        lines.extend([
            f"Pair: {feature_a} | {feature_b}",
            "-" * 52,
            f"Rows where both are missing: {int(missing.sum())}",
        ])

        if missing_rows.empty:
            lines.extend(["No missing periods.", ""])
            continue

        # A new period begins if timestamps are not exactly one hour apart
        # or if there is a non-missing row between missing observations.
        missing_rows["new_period"] = (
            missing_rows["timestamp"].diff().ne(pd.Timedelta(hours=1))
        )
        missing_rows["period_id"] = missing_rows["new_period"].cumsum()

        periods = (
            missing_rows.groupby("period_id")
            .agg(
                start=("timestamp", "min"),
                end=("timestamp", "max"),
                rows=("timestamp", "size"),
            )
            .reset_index(drop=True)
        )

        lines.append(f"Number of separate missing periods: {len(periods)}")
        lines.append("")
        lines.append("Start | End | Missing hours")

        for _, period in periods.iterrows():
            lines.append(
                f"{period['start']} | {period['end']} | {int(period['rows'])}"
            )

        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
