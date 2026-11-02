from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

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

    if "timestamp" not in df.columns:
        raise ValueError("Required column 'timestamp' is missing.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    lines = [
        "AquaSenseAI - Missingness Overlap Analysis",
        "=" * 48,
        f"Dataset rows: {len(df)}",
        "",
        "PAIRWISE MISSINGNESS",
        "-" * 48,
    ]

    for feature_a, feature_b in FEATURE_PAIRS:
        if feature_a not in df.columns or feature_b not in df.columns:
            lines.extend([
                f"{feature_a} | {feature_b}",
                "Skipped: one or both columns are missing.",
                "",
            ])
            continue

        missing_a = df[feature_a].isna()
        missing_b = df[feature_b].isna()

        both_missing = missing_a & missing_b
        only_a_missing = missing_a & ~missing_b
        only_b_missing = ~missing_a & missing_b
        neither_missing = ~missing_a & ~missing_b

        lines.extend([
            f"Feature A: {feature_a}",
            f"Feature B: {feature_b}",
            f"A missing: {int(missing_a.sum())}",
            f"B missing: {int(missing_b.sum())}",
            f"Both missing: {int(both_missing.sum())}",
            f"Only A missing: {int(only_a_missing.sum())}",
            f"Only B missing: {int(only_b_missing.sum())}",
            f"Neither missing: {int(neither_missing.sum())}",
            "",
        ])

        if both_missing.any():
            lines.append("Sample timestamps where both are missing:")
            lines.extend(
                df.loc[both_missing, "timestamp"]
                .head(10)
                .astype(str)
                .tolist()
            )
            lines.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "missingness_analysis.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
