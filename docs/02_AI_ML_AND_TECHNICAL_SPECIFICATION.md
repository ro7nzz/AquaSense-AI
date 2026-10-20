# AquaSenseAI — AI/ML & Data Specification
## Document 02: How the AI Works

**Project:** AquaSenseAI — Urban Water Intelligence  
**Document Version:** 1.0  
**Status:** Initial Technical Specification  
**Primary Language:** Python

---

## 1. Purpose of This Document

This document defines the proposed data architecture, preprocessing workflow, feature engineering, machine-learning methods, model evaluation strategy, and prediction outputs for AquaSenseAI.

It is intended to guide implementation consistently across the project.

The final models and dataset schema will be confirmed after the available data has been inspected. No model accuracy, leakage-detection capability, or forecasting performance will be claimed before appropriate evaluation.

## 2. AI/ML Problem Definition

AquaSenseAI addresses three related analytical problems.

### 2.1 Problem A: Water Consumption Anomaly Detection

**Objective:** Identify consumption observations that differ substantially from expected patterns.

Potential anomalies include:
- Sudden increases in consumption.
- Persistent consumption outside expected operating hours, if the data has suitable time resolution.
- Unusual deviations from a property's or zone's historical baseline.
- Unexpected changes compared with similar customers or periods.

**Proposed approach:** Begin with statistical baselines and an unsupervised anomaly-detection method. Compare alternatives where the data supports meaningful evaluation.

**Output:** An anomaly score, a flag, and an explanation based on observed features.

An anomaly indicates unusual behavior, not proof of a physical leak.

### 2.2 Problem B: Potential Water-Loss Intelligence

**Objective:** Help users investigate and prioritize potential water losses.

Possible inputs include:
- Consumption anomalies.
- Meter readings.
- Distribution-zone measurements.
- Water entering and leaving a defined system.
- Historical incident records.
- Location and infrastructure information, if available.

**Proposed approach:** Combine validated analytical indicators into a transparent prioritization score. Use supervised risk prediction only if sufficiently reliable historical labels exist.

**Output:** A potential-risk category, supporting evidence, and a recommended investigation priority.

A prioritization score is not automatically a calibrated probability of leakage.

### 2.3 Problem C: Water Demand Forecasting

**Objective:** Estimate future water consumption using historical observations.

Potential forecasting horizons include the next day, week, or another operationally relevant period.

**Proposed approach:** Establish a simple baseline before testing more complex time-series or machine-learning models.

**Output:** Forecasted consumption, forecast horizon, and evaluation results. Prediction intervals may be included when supported by the chosen method.

Forecasting is conditional on having adequate time coverage, consistent measurements, and suitable features.

## 3. Data Requirements

### 3.1 Candidate Data Sources

The project may use:
- Public water-consumption datasets.
- Water-distribution or utility datasets.
- Meter-level consumption records.
- Synthetic data for development and software testing.
- Public weather or calendar information, where relevant and appropriately licensed.

The actual dataset must be selected based on availability, quality, licensing, and alignment with the project objectives.

Synthetic data must be clearly identified and must not be presented as real utility measurements.

### 3.2 Candidate Data Tables

The following tables are proposed logical structures. They are not a claim that every dataset will contain these fields.

#### A. Consumption Records

| Field | Suggested Type | Description |
|---|---|---|
| record_id | Integer or string | Unique observation identifier |
| meter_id | String | Meter or customer identifier |
| timestamp | Datetime | Time of observation |
| consumption_value | Numeric | Measured consumption |
| consumption_unit | String | Unit of measurement |
| zone_id | String, optional | Distribution or service zone |
| meter_status | String, optional | Measurement or device status |

The meaning of `consumption_value` must be established. It may represent interval consumption or a cumulative meter reading. These are different quantities and must be processed differently.

#### B. Operational or Zone Measurements

| Field | Suggested Type | Description |
|---|---|---|
| measurement_id | String | Unique measurement identifier |
| timestamp | Datetime | Measurement time |
| zone_id | String | Zone identifier |
| inlet_volume | Numeric, optional | Volume entering a defined zone |
| outlet_volume | Numeric, optional | Volume measured leaving the zone |
| pressure | Numeric, optional | Pressure measurement, if available |
| flow_rate | Numeric, optional | Flow-rate measurement, if available |

Water-loss estimation requires compatible measurements and a clearly defined system boundary. Inlet and outlet measurements may not alone account for storage changes, authorized uses, timing differences, or measurement errors.

#### C. Historical Incidents

| Field | Suggested Type | Description |
|---|---|---|
| incident_id | String | Incident identifier |
| zone_id or meter_id | String | Related asset, if known |
| incident_timestamp | Datetime | Incident time |
| incident_type | String | Recorded incident category |
| verified_leak | Boolean, optional | Verified outcome, if documented |
| resolution_timestamp | Datetime, optional | Resolution time |

Incident data may be used for supervised learning or retrospective evaluation only when labels are reliable and their timing is handled correctly.

## 4. Data Quality and Preprocessing

Preprocessing will be implemented as reusable Python functions rather than repeated manual notebook operations.

### 4.1 Schema Validation

- Verify required columns and data types.
- Parse timestamps consistently.
- Validate measurement units.
- Check whether identifiers are unique where expected.
- Confirm that the dataset's time zone and sampling frequency are understood.

### 4.2 Missing Values

- Calculate missing-value counts and percentages.
- Identify patterns of missingness.
- Avoid silently replacing unknown consumption with zero.
- Use imputation only when justified by the data and analytical purpose.
- Document rows excluded from each analysis.

### 4.3 Duplicate Records

- Detect exact duplicate rows.
- Identify duplicate keys such as meter and timestamp.
- Determine whether duplicates represent ingestion errors or legitimate repeated measurements.
- Apply a documented resolution policy.

### 4.4 Invalid Measurements

Check for:
- Negative consumption where physically invalid for the defined measurement.
- Impossible or implausible values.
- Unexpected unit changes.
- Irregular sampling intervals.
- Meter resets or rollovers in cumulative readings.
- Timestamp inconsistencies.

Valid zero readings must not automatically be classified as errors.

### 4.5 Cumulative Versus Interval Readings

For cumulative meter readings, interval consumption can be derived from consecutive readings:

`interval_consumption = current_reading - previous_reading`

This calculation requires careful treatment of meter resets, rollovers, replacement, missing readings, and negative differences.

For already aggregated interval consumption, the calculation must not be applied again.

### 4.6 Time Aggregation

Depending on the source data, observations may be aggregated to hourly, daily, or monthly intervals.

Aggregation must preserve the correct measurement meaning and unit. For example, summing interval volumes may be valid, while summing cumulative meter readings is generally not.

## 5. Exploratory Data Analysis

Exploratory analysis will establish the dataset's behavior before model development.

Required investigations include:
- Dataset dimensions and column types.
- Coverage across dates and identifiers.
- Missing values and duplicate records.
- Consumption distributions and extreme values.
- Trends by day, week, and month where temporal resolution permits.
- Differences across zones, meters, or customer groups.
- Seasonal patterns, if sufficient history exists.
- Relationships between consumption and available external variables.
- Data-quality limitations that could affect model validity.

Deliverables will include reproducible notebooks, visualizations, and written findings.

## 6. Feature Engineering

Features will be created only when the source data supports them.

### 6.1 Time-Based Features

Possible features:
- Hour of day.
- Day of week.
- Weekend indicator.
- Month or season.
- Holiday indicator, where relevant.

These features require correctly interpreted timestamps and time zones.

### 6.2 Historical Consumption Features

Possible features:
- Previous interval consumption.
- Previous-day consumption.
- Rolling mean.
- Rolling standard deviation.
- Difference from a historical baseline.
- Percentage deviation from baseline.
- Recent consumption trend.

Rolling features must use past observations only. Future values must never enter features used to predict earlier observations.

### 6.3 Anomaly Features

Potential features include:
- Deviation from a meter-specific baseline.
- Deviation from a time-of-day baseline.
- Unusual increases compared with recent history.
- Persistence of elevated consumption.
- Robust statistical deviation from a historical distribution.

Features such as overnight usage are appropriate only if timestamps and sampling frequency allow overnight behavior to be measured meaningfully.

### 6.4 Operational Features

If available:
- Pressure.
- Flow rate.
- Zone inlet and outlet measurements.
- Customer or property category.
- Historical incidents.
- Infrastructure age or characteristics.

Identifiers should not be treated as numeric measurements without a clear rationale. High-cardinality identifiers may require grouping, encoding, or separate baseline strategies.

## 7. Model Development Strategy

AquaSenseAI will use a baseline-first approach.

### 7.1 Baseline 1: Rule-Based Detection

Develop transparent rules based on consumption thresholds or deviations from historical baselines.

Purpose:
- Establish a simple reference.
- Help validate feature calculations.
- Provide an interpretable comparison for machine-learning models.

Thresholds must be selected from data analysis, domain requirements, or documented assumptions rather than chosen to create attractive results.

### 7.2 Baseline 2: Statistical Anomaly Detection

Potential methods:
- Rolling mean and standard deviation.
- Median and median absolute deviation.
- Time-of-day or day-of-week baselines.

Statistical methods provide a simple way to identify deviations and help explain why an observation was flagged.

### 7.3 Candidate Model: Isolation Forest

Isolation Forest is a candidate unsupervised anomaly-detection model available in Scikit-learn.

It can identify observations that are unusual relative to the feature distribution.

Implementation considerations:
- Select features appropriate to the measurement frequency.
- Fit preprocessing and the model using training data only.
- Evaluate contamination or alert thresholds rather than assuming a fixed anomaly proportion is correct.
- Compare flagged observations with the statistical baseline.
- Review sensitivity to feature selection and time periods.

Isolation Forest does not directly prove that a leak exists and may not capture every type of temporal behavior.

### 7.4 Alternative Models

Depending on the dataset:
- Local Outlier Factor for local density-based outlier detection.
- One-Class SVM when scale and data characteristics make it appropriate.
- Supervised classifiers if reliable incident labels are available.
- Time-series forecasting residuals for detecting deviations from predicted consumption.

These are candidates, not mandatory implementations. Model selection will depend on the quality and structure of the available data.

## 8. Anomaly Scoring and Alert Classification

The system should distinguish between model output and operational interpretation.

A proposed alert record may contain:

| Field | Description |
|---|---|
| observation_id | Identifier of the evaluated observation |
| anomaly_score | Score or transformed measure used by the selected method |
| anomaly_flag | Whether the chosen threshold was exceeded |
| risk_category | Configured alert category |
| supporting_features | Features that contributed to the alert |
| generated_at | Time the alert was generated |
| model_version | Model or configuration identifier |

Score direction and scale vary by algorithm. The implementation must document whether higher or lower values indicate greater anomaly likelihood.

Possible operational categories are:
- Normal: no alert under the current rule or threshold.
- Low priority: modest deviation requiring context.
- Medium priority: unusual pattern that merits review.
- High priority: substantial or persistent deviation warranting prompt investigation.

These labels must be calibrated to the selected model and operational requirements. They must not be described as verified leak severity without supporting evidence.

## 9. Potential Water-Loss Estimation

Where compatible measurements exist, a basic water-balance calculation may be considered:

`estimated_difference = measured_inlet_volume - measured_authorized_outflow`

This difference is not necessarily leakage. Interpretation may require accounting for:
- Authorized consumption.
- Storage changes.
- Measurement uncertainty.
- Timing and aggregation differences.
- Unmetered legitimate uses.
- Transfers between system boundaries.

Where those factors are unknown, the result must be labeled as an unaccounted-for difference or preliminary estimate rather than confirmed physical leakage.

Consumption anomalies can support investigation, but they cannot independently quantify actual leakage volume unless a validated estimation method and sufficient measurements are available.

## 10. Demand Forecasting

Forecasting will be implemented only if the dataset has sufficient history and an appropriate time structure.

### 10.1 Baseline Forecast

Begin with a simple method such as:
- Last-observation or persistence forecast.
- Seasonal naive forecast when a stable seasonal period is supported.

### 10.2 Candidate Models

Potential models include:
- Moving-average or exponential-smoothing methods.
- Suitable time-series models.
- Random Forest or gradient-boosting regressors using lagged and calendar features.

The selected approach must be justified by evaluation rather than algorithm complexity.

### 10.3 Forecast Validation

Use chronological splits:
- Training period: earlier observations.
- Validation period: subsequent observations used for model selection.
- Test period: later observations reserved for final evaluation.

Do not randomly shuffle time-series observations when doing so would allow future information to leak into training.

### 10.4 Forecast Metrics

Potential metrics:
- MAE (Mean Absolute Error).
- RMSE (Root Mean Squared Error).
- MASE or another appropriate scale-aware comparison.

MAPE should be avoided or used carefully when actual consumption can be zero or close to zero.

All metrics must be reported with the forecast horizon, target unit, test period, and baseline comparison.

## 11. Model Evaluation

### 11.1 When Ground-Truth Labels Exist

For leakage or anomaly labels, possible metrics include:
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Precision-recall curve.
- Alert volume and false-positive rate.

The appropriate threshold should reflect the trade-off between missed incidents and unnecessary investigations.

### 11.2 When Ground-Truth Labels Do Not Exist

For unsupervised anomaly detection:
- Compare against transparent statistical baselines.
- Review sampled alerts for plausibility.
- Inspect flagged time periods and known data-quality events.
- Test stability across different periods and feature configurations.
- Obtain expert or documented validation if possible.

Without reliable labels or an independent validation method, true leakage-detection accuracy cannot be established.

### 11.3 Data Leakage Prevention

- Split data before fitting learned preprocessing.
- Calculate historical features using past data only.
- Avoid using future incident outcomes as predictors.
- Keep repeated observations from the same event from improperly crossing evaluation boundaries.
- Use chronological or group-aware splitting when appropriate.
- Fit scalers, imputers, and encoders on training data only.

## 12. Reproducibility and Model Management

Each model experiment should record:
- Dataset version or source.
- Feature definitions.
- Training and evaluation periods.
- Model type and configuration.
- Random seed where relevant.
- Evaluation metrics.
- Known limitations.
- Model artifact path and version.

Reusable model code should live in the project's source directories, while notebooks should focus on exploration and experimentation.

Persisted model artifacts should be loaded only from trusted sources. Training and inference must use compatible preprocessing and feature schemas.

## 13. Proposed Python Components

| Component | Responsibility |
|---|---|
| Data loader | Read source data |
| Data validator | Check schema, units, and quality |
| Preprocessor | Clean and transform observations |
| Feature engineer | Build historical and time-based features |
| Baseline detector | Calculate transparent anomaly rules |
| Anomaly model | Fit or apply the selected anomaly detector |
| Risk scorer | Convert supported evidence into a documented priority score |
| Forecaster | Train and evaluate demand forecasting models, if included |
| Evaluator | Calculate appropriate metrics |
| Prediction service | Generate consistent outputs for the application |
| Reporting module | Prepare metrics and outputs for dashboards |

The exact module names and folder structure will be finalized in Document 03.

## 14. Expected Outputs

The AI/ML pipeline should produce, as applicable:
- Cleaned analytical datasets.
- Feature tables.
- Consumption summaries.
- Anomaly scores and flags.
- Alert explanations.
- Potential-risk categories.
- Water-loss estimates with documented assumptions.
- Demand forecasts and evaluation metrics.
- Model evaluation reports.
- Saved model artifacts and experiment metadata.

## 15. Technical Limitations

The following limitations must be documented where applicable:
- Sparse or irregular measurements.
- Missing historical observations.
- Lack of confirmed leak labels.
- Incomplete system water-balance measurements.
- Changes in meter behavior or consumption patterns.
- Seasonal shifts or changing operating conditions.
- Poor generalization to unseen zones or customers.
- False positives and false negatives.
- Limited evidence for live or real-time performance.

Model outputs must remain proportionate to the evidence available.

## 16. Acceptance Criteria

The AI/ML component will be considered implemented when:
1. Data validation and preprocessing are reproducible.
2. Feature definitions and units are documented.
3. At least one anomaly-detection method is implemented.
4. A baseline comparison is available.
5. Evaluation is performed without inappropriate data leakage.
6. Output scores and thresholds are explained.
7. Forecasting, if implemented, is evaluated chronologically.
8. Results and limitations are recorded.
9. The application or dashboard consumes model outputs consistently.

Performance thresholds will be established after dataset inspection and baseline evaluation.

## 17. Final AI/ML Definition

AquaSenseAI will use a structured analytical pipeline combining data validation, consumption feature engineering, statistical baselines, machine-learning-based anomaly detection, and transparent prioritization of potential water-loss incidents.

Demand forecasting and quantitative water-loss estimation are conditional capabilities that will be implemented only when appropriate data and validation methods are available.

The objective is not simply to produce predictions, but to produce explainable, reproducible, and appropriately evaluated water intelligence.

**Document status:** Initial AI/ML specification. Dataset selection, feature availability, model choice, and evaluation targets will be confirmed during implementation.
