from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

st.set_page_config(page_title="AquaSenseAI", page_icon="💧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at top right, #20283a 0%, #0b1020 45%, #080b14 100%); color: #e8edf8; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #111827, #090e19); border-right: 1px solid #30394b; }
[data-testid="stMetric"] { background: linear-gradient(145deg, #1a2233, #111827); border: 1px solid #303b50; padding: 20px; border-radius: 16px; transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; animation: fadeUp .6s ease both; }
[data-testid="stMetric"]:hover { transform: translateY(-5px); border-color: #ff914d; box-shadow: 0 10px 30px #ff914d18; }
[data-testid="stMetricLabel"] { color: #aab8ce; }
[data-testid="stMetricValue"] { color: #ffffff; font-weight: 800; }
h1, h2, h3 { color: #f8fafc !important; }
.hero { background: linear-gradient(115deg, #202b40, #172238 60%, #38251f); padding: 30px; border-radius: 20px; border: 1px solid #3c465b; margin-bottom: 22px; animation: fadeUp .7s ease both, glow 5s ease-in-out infinite alternate; }
.hero h1 { margin: 8px 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -1px; }
.hero p { color: #b8c5d9; margin-bottom: 0; }
.pill { display: inline-block; padding: 6px 12px; border-radius: 20px; background: #412719; color: #ffad70; font-size: .78rem; font-weight: 700; }
div[data-testid="stAlert"] { border-radius: 12px; }
.stButton > button { border-radius: 10px; border: 1px solid #ff914d; transition: all .2s ease; }
.stButton > button:hover { border-color: #ffb17d; transform: translateY(-2px); }
[data-testid="stPlotlyChart"] { background: #101827; border: 1px solid #29364a; border-radius: 14px; padding: 8px; animation: fadeUp .8s ease both; }
.small-muted { color: #9aabc5; font-size: .88rem; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes glow { from { box-shadow: 0 0 12px #ff914d08; } to { box-shadow: 0 0 28px #ff914d20; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }

/* AquaSenseAI command-center redesign */
.block-container {
    max-width: 1800px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}
[data-testid="stHeader"] {
    background: rgba(7, 11, 19, .88);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c121d, #080c13);
    border-right: 1px solid #263344;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.35rem;
    letter-spacing: -.5px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 5px 8px;
    border-radius: 7px;
}
.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(ellipse at 88% 35%, #0c827a30, transparent 36%),
        radial-gradient(ellipse at 75% 100%, #ff914d20, transparent 45%),
        linear-gradient(115deg, #101a29, #0b111d 65%, #111a25);
    border: 1px solid #2a3a4d;
    border-radius: 18px;
    padding: 28px 32px;
    margin: 4px 0 24px;
    box-shadow: 0 12px 45px #00000030;
}
.hero:after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -80px;
    top: -120px;
    border-radius: 50%;
    border: 1px solid #42d6c220;
    box-shadow: 0 0 70px #42d6c210;
    pointer-events: none;
}
.hero h1 {
    font-size: clamp(1.8rem, 3vw, 2.7rem);
    letter-spacing: -1.3px;
    margin: 12px 0 8px;
}
.hero p {
    max-width: 700px;
    line-height: 1.7;
    color: #a9bad0;
}
.hero .pill {
    background: #12332f;
    color: #62e0c5;
    border: 1px solid #23594e;
    letter-spacing: .7px;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #151f2d, #0d141f);
    border: 1px solid #28394b;
    border-radius: 13px;
    padding: 18px 20px;
    box-shadow: 0 8px 22px #00000015;
}
[data-testid="stMetric"]:hover {
    border-color: #e88d50;
    transform: translateY(-3px);
}
[data-testid="stMetricLabel"] {
    font-size: .78rem;
    letter-spacing: .5px;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-size: clamp(1.35rem, 2vw, 2rem);
}
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"] {
    border: 1px solid #273648;
    border-radius: 12px;
    overflow: hidden;
}
h2, h3 {
    letter-spacing: -.35px;
}
hr {
    border-color: #263344;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #91a5bd;
}
@keyframes commandEnter {
    from { opacity: 0; transform: translateY(9px); }
    to { opacity: 1; transform: translateY(0); }
}
.hero {
    animation: commandEnter .65s ease-out both;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df

with st.sidebar:
    st.title("💧 AquaSenseAI")
    st.caption("URBAN WATER INTELLIGENCE")
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
    page = st.radio(
    "NAVIGATION",
    ["Overview", "ML Prediction", "SQL Analytics", "Sensor Explorer", "Data Quality", "Risk Analysis", "About"],
)
    st.divider()
    st.caption("Historical data • Decision-support prototype")

if uploaded is not None:
    try:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding='utf-8')
    except UnicodeDecodeError:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding='cp1252')
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
else:
    df = load_data()

st.markdown("""
<div class="hero">
    <span class="pill">● AQUASENSEAI · HISTORICAL INTELLIGENCE</span>
    <h1>Water Network Command Center</h1>
    <p>Explore network anomaly indicators, historical leak-proximity patterns,
    sensor data quality, and model-assisted investigation priorities.</p>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.error("Dataset not found. Upload the project CSV in the sidebar.")
    st.code(str(DATA_PATH))
    st.stop()

if df.empty:
    st.error("The dataset has no usable rows.")
    st.stop()

target = "fault_d7" if "fault_d7" in df.columns else None
numeric = df.select_dtypes(include=np.number).columns.tolist()
features = [c for c in numeric if c != target]
timestamps = "timestamp" if "timestamp" in df.columns else None

with st.sidebar:
    if timestamps and df[timestamps].notna().any():
        valid_dates = df[timestamps].dropna()
        date_range = st.date_input(
            "Time window",
            value=(valid_dates.min().date(), valid_dates.max().date()),
            min_value=valid_dates.min().date(),
            max_value=valid_dates.max().date()
        )
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start, end = date_range
            df = df[(df[timestamps].dt.date >= start) & (df[timestamps].dt.date <= end)]

if page == "Overview":
    import plotly.graph_objects as go

    # TOP KPI STRIP
    st.markdown("### NETWORK OPERATIONS / HISTORICAL ANALYTICS")
    labels = pd.to_numeric(df[target], errors="coerce") if target else pd.Series(dtype=float)
    missing_cells = int(df.isna().sum().sum())
    positive_count = int((labels == 1).sum()) if target else 0
    negative_count = int((labels == 0).sum()) if target else 0
    positive_rate = float((labels == 1).mean() * 100) if target and len(labels) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("HISTORICAL RECORDS", f"{len(df):,}")
    k2.metric("ANOMALY INDICATORS", f"{len(features)}")
    k3.metric("MISSING CELLS", f"{missing_cells:,}")
    k4.metric("PROXIMITY-POSITIVE", f"{positive_rate:.1f}%" if target else "N/A")

    st.markdown("---")

    # MAIN TREND + RIGHT PRIORITY RAIL
    main, rail = st.columns([2.25, 1], gap="medium")

    with main:
        st.markdown("### Historical Indicator Activity")
        st.caption("Select indicators to inspect patterns across the selected time window.")

        selected = st.multiselect(
            "Indicators",
            features,
            default=features[:min(3, len(features))],
            key="grid_indicators"
        )

        if timestamps and selected:
            plot_df = df[[timestamps] + selected].copy()
            plot_df = plot_df.sort_values(timestamps)
            long_df = plot_df.melt(
                id_vars=timestamps, var_name="Indicator", value_name="Score"
            )
            fig = px.line(
                long_df, x=timestamps, y="Score", color="Indicator",
                template="plotly_dark"
            )
            fig.update_traces(line=dict(width=2))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(l=8, r=8, t=18, b=8),
                legend=dict(orientation="h", y=1.12),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("Select indicators and ensure timestamps are available.")

    with rail:
        st.markdown("### Investigation Priorities")
        st.caption("Historical label summary — not live alerts")

        if target:
            st.metric("Proximity-positive records", f"{positive_count:,}")
            st.metric("Negative records", f"{negative_count:,}")

            # Rank records by combined available indicator magnitude.
            # This is a screening heuristic, not a leak probability.
            if features:
                candidate = df.copy()
                candidate["_screening_score"] = candidate[features].abs().mean(
                    axis=1, skipna=True
                )
                candidate = candidate.nlargest(5, "_screening_score")

                display_cols = ([timestamps] if timestamps else []) + ["_screening_score"]
                if target:
                    display_cols.append(target)
                display_cols = [c for c in display_cols if c in candidate.columns]
                display = candidate[display_cols].rename(
                    columns={"_screening_score": "Indicator magnitude",
                             target: "fault_d7"} if target else
                            {"_screening_score": "Indicator magnitude"}
                )
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.caption(
                    "Ranked by average absolute indicator magnitude. "
                    "This heuristic is not a trained-model score or confirmed leak."
                )
        else:
            st.info("No fault_d7 label exists in this dataset.")

    st.markdown("---")

    # LOWER ANALYTICS GRID
    left, middle, right = st.columns([1.15, 1, 1.15], gap="medium")

    with left:
        st.markdown("### Indicator Ranking")
        if features:
            ranking = df[features].abs().mean().sort_values(ascending=True).tail(8)
            rank_df = ranking.rename("Mean absolute value").reset_index()
            rank_df.columns = ["Indicator", "Mean absolute value"]
            fig = px.bar(
                rank_df, x="Mean absolute value", y="Indicator",
                orientation="h", template="plotly_dark"
            )
            fig.update_traces(marker_color="#ff965c")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280, margin=dict(l=4, r=8, t=8, b=8)
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    with middle:
        st.markdown("### Label Distribution")
        if target:
            counts = pd.DataFrame({
                "Class": ["Proximity-positive", "Negative"],
                "Records": [positive_count, negative_count]
            })
            fig = px.pie(
                counts, names="Class", values="Records", hole=.68,
                template="plotly_dark",
                color="Class",
                color_discrete_map={
                    "Proximity-positive": "#ff965c",
                    "Negative": "#42d6c2"
                }
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=4, r=4, t=8, b=8),
                legend=dict(orientation="h", y=-.12)
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption("fault_d7 is approximate proximity to recorded leak events.")

    with right:
        st.markdown("### Data Quality")
        total_cells = int(df.size)
        complete_cells = max(0, total_cells - missing_cells)
        quality_df = pd.DataFrame({
            "Status": ["Complete", "Missing"],
            "Cells": [complete_cells, missing_cells]
        })
        fig = px.bar(
            quality_df, x="Status", y="Cells", color="Status",
            template="plotly_dark",
            color_discrete_map={"Complete": "#42d6c2", "Missing": "#ff965c"}
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=280,
            margin=dict(l=4, r=4, t=8, b=8)
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(f"{missing_cells:,} missing cells across {total_cells:,} total cells.")

    st.markdown("---")
    st.markdown("### Recent Historical Records")
    st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
    st.download_button(
        "Export filtered dataset",
        df.to_csv(index=False).encode("utf-8"),
        "aquasense_data.csv",
        "text/csv"
    )


elif page == "ML Prediction":
    import json
    import joblib

    st.subheader("AI Leak-Proximity Prediction")
    st.caption("Experimental model ? Historical dataset ? Decision support only")

    model_path = ROOT / "models" / "aquasense_logistic_regression.joblib"
    schema_path = ROOT / "models" / "feature_schema.json"

    if not model_path.exists() or not schema_path.exists():
        st.error("Model files not found. Run: python .\\src\\train_model.py")
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        model_features = schema["features"]

        missing_columns = [c for c in model_features if c not in df.columns]
        if missing_columns:
            st.error("The loaded dataset is missing required model features:")
            st.write(missing_columns)
        else:
            st.info(
                "Choose a historical record to score. The model was trained on "
                "the earliest 70% of the dataset. This score is experimental and "
                "does not confirm a real leak."
            )

            prediction_df = df.copy()
            if timestamps:
                prediction_df = prediction_df.sort_values(timestamps).reset_index(drop=True)

            row_index = st.slider(
                "Historical record index",
                min_value=0,
                max_value=len(prediction_df) - 1,
                value=max(0, int(len(prediction_df) * 0.85)),
                step=1,
            )
            row = prediction_df.iloc[[row_index]]
            threshold = st.slider(
                "Demo decision threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                help="Configurable demonstration threshold; not an operationally validated threshold.",
            )

            X_one = row[model_features].apply(pd.to_numeric, errors="coerce")
            score = float(model.predict_proba(X_one)[0, 1])
            predicted_label = int(score >= threshold)

            if timestamps:
                st.write("**Selected timestamp:**", str(row.iloc[0][timestamps]))
            a, b, c = st.columns(3)
            a.metric("Model score", f"{score:.1%}")
            b.metric("Demo classification", "Positive" if predicted_label else "Negative")
            c.metric("Decision threshold", f"{threshold:.0%}")

            if predicted_label:
                st.warning("The model flags this historical record as positive for leak-proximity.")
            else:
                st.success("The model does not flag this historical record as positive at the selected threshold.")

            st.markdown("### Input feature snapshot")
            st.dataframe(
                X_one.T.rename(columns={X_one.index[0]: "Value"}),
                use_container_width=True,
            )

            if target and target in row.columns:
                actual = pd.to_numeric(row[target], errors="coerce").iloc[0]
                st.markdown("### Historical label (reference only)")
                if pd.notna(actual):
                    st.write(
                        "Recorded `fault_d7` label:",
                        "Positive" if int(actual) == 1 else "Negative",
                    )
                    st.caption(
                        "This label marks proximity to a recorded leak event within an approximate "
                        "seven-day window; it is not proof of a leak at this exact hour."
                    )

            st.warning(
                "Limitations: this is a historical-data prototype, not a live sensor system. "
                "The score is not established as a calibrated probability of an actual leak. "
                "Do not use it alone for operational decisions."
            )

elif page == "SQL Analytics":
    import sqlite3

    st.subheader("SQL Analytics")
    st.caption("Read-only analytics from the local SQLite database")
    db_path = ROOT / "data" / "aquasense.db"

    if not db_path.exists():
        st.error("Database not found. Run: python .\\sql\\build_database.py")
    else:
        try:
            with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM water_sensor_records"
                ).fetchone()[0]
                distribution = pd.read_sql_query("""
                    SELECT fault_d7 AS historical_label,
                           COUNT(*) AS records,
                           ROUND(100.0 * COUNT(*) /
                           (SELECT COUNT(*) FROM water_sensor_records), 2) AS percentage
                    FROM water_sensor_records
                    GROUP BY fault_d7 ORDER BY fault_d7
                """, conn)
                daily = pd.read_sql_query("""
                    SELECT DATE(timestamp) AS record_date,
                           COUNT(*) AS records,
                           SUM(CASE WHEN fault_d7 = 1 THEN 1 ELSE 0 END) AS positive_labels
                    FROM water_sensor_records
                    GROUP BY DATE(timestamp) ORDER BY record_date
                """, conn)

            a, b = st.columns(2)
            a.metric("Records in SQL database", f"{total:,}")
            b.metric("Daily summaries", f"{len(daily):,}")

            st.markdown("### Historical label distribution")
            st.dataframe(distribution, use_container_width=True, hide_index=True)

            st.markdown("### Daily record trends")
            if not daily.empty:
                st.line_chart(daily.set_index("record_date")[["records", "positive_labels"]])
                st.dataframe(daily, use_container_width=True, hide_index=True)

            st.caption(
                "`fault_d7` indicates approximate proximity to a recorded leak event; "
                "it does not confirm a leak at every positive timestamp."
            )
        except Exception as exc:
            st.error(f"Unable to read SQL analytics: {exc}")

elif page == "Sensor Explorer":
    st.subheader("Sensor Indicator Explorer")
    if not features:
        st.warning("No numeric indicators found.")
    else:
        feature = st.selectbox("Select indicator", features)
        series = pd.to_numeric(df[feature], errors="coerce")
        a, b, c, d = st.columns(4)
        a.metric("Available readings", f"{series.notna().sum():,}")
        b.metric("Missing readings", f"{series.isna().sum():,}")
        c.metric("Median score", f"{series.median():.2f}" if series.notna().any() else "N/A")
        d.metric("95th percentile", f"{series.quantile(.95):.2f}" if series.notna().any() else "N/A")
        if timestamps:
            fig = px.line(df, x=timestamps, y=feature, template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(series.dropna(), nbins=40, template="plotly_dark", title="Score Distribution")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("These are derived anomaly indicators, not raw water-flow or pressure readings.")

elif page == "Data Quality":
    st.subheader("Data Quality & Sensor Availability")
    missing = df.isna().sum().sort_values(ascending=False)
    quality = pd.DataFrame({"Column": missing.index, "Missing cells": missing.values,
                            "Missing %": (missing.values / max(len(df), 1) * 100).round(2)})
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Columns with missing data", int((missing > 0).sum()))
    chart_data = quality[quality["Missing cells"] > 0].sort_values("Missing %")
    if not chart_data.empty:
        fig = px.bar(chart_data, x="Missing %", y="Column", orientation="h",
                     color="Missing %", template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=max(350, len(chart_data)*25))
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.warning("Missing readings are preserved as missing; they are not automatically replaced with zero.")

elif page == "Risk Analysis":
    st.subheader("Leak-Proximity Risk Analysis")
    if target and features:
        feature = st.selectbox("Compare indicator", features)
        compare = df[[feature, target]].copy()
        compare[feature] = pd.to_numeric(compare[feature], errors="coerce")
        compare[target] = pd.to_numeric(compare[target], errors="coerce")
        compare = compare.dropna()
        if not compare.empty:
            summary = compare.groupby(target)[feature].agg(["count", "mean", "median", "std"]).reset_index()
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
            fig = px.box(compare, x=target, y=feature, color=target, points="outliers", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            corr = df[features + [target]].corr(numeric_only=True)[target].drop(target).dropna().sort_values()
            corr_df = corr.rename("Correlation").reset_index().rename(columns={"index":"Indicator"})
            fig = px.bar(corr_df, x="Correlation", y="Indicator", orientation="h", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=max(350, len(corr_df)*24))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Exploratory associations only. Correlation does not establish causation.")
        else:
            st.info("No overlapping values for the selected indicator and label.")
    else:
        st.warning("This analysis requires numeric indicators and the fault_d7 label.")

else:
    st.subheader("About AquaSenseAI")
    st.markdown("""
    AquaSenseAI explores urban water-network anomaly indicators, data quality,
    and historical proximity to recorded leak events.

    **Interpretation and limitations**
    - `fault_d7` marks an approximate seven-day window around recorded leak events.
    - A positive label does not prove a leak occurred at that exact hour.
    - This dataset does not establish exact leak location or leaked water volume.
    - No live sensor feed is connected.
    - Correlation charts are exploratory and do not prove causality.
    - Anomaly features are derived scores rather than raw pressure or flow readings.
    """)
    st.write("**Stack:** Python · Pandas · NumPy · Streamlit · Plotly")
    st.write(f"**Dataset:** {len(df):,} rows × {len(df.columns)} columns")

st.divider()
st.caption("AquaSenseAI | Urban Water Intelligence | Historical-data prototype")





