# AquaSenseAI — System Design & Implementation Specification
## Document 03: How We Build It

**Project:** AquaSenseAI — AI-Powered Urban Water Intelligence System  
**Document Version:** 1.0  
**Status:** Initial System Design  
**Implementation Language:** Python

---

## 1. Purpose

This document defines the system architecture, project structure, database design, software modules, application integration, testing strategy, and implementation sequence for AquaSenseAI.

It translates the project objectives and AI/ML specifications into a practical engineering blueprint.

The architecture is modular so that data processing, machine learning, database operations, and presentation can be developed and tested independently.

## 2. System Architecture

The system will follow a layered architecture.

### Layer 1: Data Sources

Possible sources include public water datasets, meter consumption records, operational measurements, historical incidents, and optional contextual data.

Only sources that are available, documented, and appropriate for the project will be used.

### Layer 2: Data Processing

Python, Pandas, and NumPy will be used to:
- Load source data.
- Validate schemas and units.
- Clean and transform records.
- Engineer analytical features.
- Prepare model inputs and reporting datasets.

### Layer 3: Database

SQL will provide structured storage and analytical querying.

The database engine will be finalized during implementation based on the selected data volume, integration requirements, and available tools.

SQLite is a reasonable initial option for a local prototype. A different engine can be selected if requirements justify it.

### Layer 4: Analytics and Machine Learning

This layer will contain:
- Consumption KPI calculations.
- Statistical anomaly baselines.
- The selected anomaly-detection model.
- Potential water-loss calculations.
- Transparent investigation-priority scoring.
- Demand forecasting, if supported by the data.

### Layer 5: Presentation

Power BI will provide interactive analytical dashboards.

A lightweight application interface may be built using Streamlit. FastAPI is an alternative if a separate HTTP API is required.

The initial architecture will avoid unnecessary duplication. A web API will be added only when it provides a clear integration benefit.

### Layer 6: Testing and Reporting

The project will include automated tests where appropriate, model evaluation reports, documentation, and reproducible setup instructions.

## 3. High-Level Data Flow

The proposed data flow is:

1. Obtain source data.
2. Validate the original schema and measurement units.
3. Store or preserve the raw data.
4. Clean and transform the data.
5. Load validated records into the analytical database where appropriate.
6. Calculate consumption statistics and features.
7. Apply the selected anomaly-detection model.
8. Generate documented scores, alerts, and analytical outputs.
9. Export or expose validated results to Power BI and the application.
10. Evaluate the outputs and document limitations.

Raw inputs should remain distinguishable from cleaned data and derived results.

## 4. Proposed Project Structure

The existing project will evolve toward the following structure:

```text
AquaSenseAI/
│
├── .venv/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── docs/
│   ├── 01_PROJECT_MASTER_SPECIFICATION.md
│   ├── 02_AI_ML_AND_TECHNICAL_SPECIFICATION.md
│   ├── 03_SYSTEM_DESIGN_AND_IMPLEMENTATION_SPECIFICATION.md
│   └── 04_DEVELOPMENT_ROADMAP.md
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── model_evaluation/
│
├── sql/
│   ├── schema/
│   └── queries/
│
├── src/
│   └── aquasense/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       ├── features/
│       ├── analytics/
│       ├── models/
│       ├── database/
│       └── evaluation/
│
├── tests/
│
├── app/
│
├── .gitignore
├── README.md
└── requirements.txt
```

This is the target structure, not a requirement to create every folder immediately. Directories will be added when their corresponding phase begins.

### Directory Responsibilities

| Directory | Responsibility |
|---|---|
| `data/raw/` | Original source data preserved without analytical modifications |
| `data/interim/` | Temporary or partially processed data |
| `data/processed/` | Validated datasets ready for analysis |
| `data/external/` | Supplementary external datasets |
| `docs/` | Specifications and project documentation |
| `models/` | Saved model artifacts and related metadata |
| `notebooks/` | Exploratory analysis and experiments |
| `reports/figures/` | Generated plots and visual outputs |
| `reports/model_evaluation/` | Model evaluation results |
| `sql/schema/` | Database schema definitions |
| `sql/queries/` | Reusable SQL queries |
| `src/aquasense/` | Reusable application and analytical code |
| `tests/` | Automated tests |
| `app/` | User-facing application code |

The virtual environment must not be committed to Git.

Large datasets, private records, secrets, and generated model files should be handled according to the project's version-control and privacy policy.

## 5. Software Module Design

### 5.1 Configuration Module

Responsibilities:
- Define project paths and configuration values.
- Read environment-specific settings where needed.
- Avoid hardcoding machine-specific paths throughout the codebase.
- Keep credentials and secrets outside source control.

### 5.2 Data Module

Responsibilities:
- Read supported data formats.
- Validate expected columns.
- Check timestamps and units.
- Detect missing values and duplicates.
- Apply documented cleaning rules.

Data transformations should be deterministic wherever practical.

### 5.3 Feature Module

Responsibilities:
- Calculate time-based features.
- Build historical consumption baselines.
- Calculate rolling statistics and deviations.
- Ensure that historical features do not use future observations.

### 5.4 Analytics Module

Responsibilities:
- Calculate consumption summaries.
- Produce defined water-use KPIs.
- Calculate water-balance differences where measurements permit.
- Prepare results for reports and dashboards.

### 5.5 Model Module

Responsibilities:
- Fit the selected model.
- Apply consistent preprocessing during inference.
- Generate anomaly scores and flags.
- Save and load approved model artifacts.
- Record the model version and relevant configuration.

### 5.6 Database Module

Responsibilities:
- Manage database connections.
- Create or validate tables.
- Load analytical records.
- Execute parameterized queries.
- Handle database errors appropriately.

### 5.7 Evaluation Module

Responsibilities:
- Compare models against baselines.
- Calculate appropriate metrics.
- Record evaluation periods and assumptions.
- Generate evaluation reports.

## 6. Database Design

The final schema will be determined after inspecting the chosen dataset.

The following logical tables provide a starting point.

### 6.1 Consumption Table

Potential fields:
- `record_id`
- `meter_id`
- `timestamp`
- `consumption_value`
- `consumption_unit`
- `zone_id`, where available

A unique key should be defined based on the actual source schema. Meter and timestamp may form a composite key when the data guarantees one observation per meter per timestamp.

### 6.2 Zone Measurements Table

Potential fields:
- `measurement_id`
- `zone_id`
- `timestamp`
- `inlet_volume`
- `outlet_volume`
- `pressure`, where available
- `flow_rate`, where available

Measurements must include enough context to establish compatible units and time intervals.

### 6.3 Incident Table

Potential fields:
- `incident_id`
- `meter_id` or `zone_id`
- `incident_timestamp`
- `incident_type`
- `verified_leak`, where documented
- `resolution_timestamp`, where available

This table is optional and depends on whether reliable incident records exist.

### 6.4 Prediction and Alert Table

Potential fields:
- `prediction_id`
- `observation_id`
- `model_version`
- `anomaly_score`
- `anomaly_flag`
- `risk_category`
- `generated_at`

The score definition and alert threshold must be documented for each relevant model version.

### 6.5 Database Principles

- Use appropriate primary and foreign keys.
- Store timestamps consistently.
- Define units explicitly.
- Validate data before insertion.
- Use parameterized SQL queries.
- Separate source observations from predictions.
- Avoid storing unnecessary personal information.
- Document table relationships and assumptions.

## 7. SQL Analytics

SQL will support calculations and reporting such as:
- Total consumption by time period.
- Average and peak consumption.
- Consumption by zone or meter.
- Period-over-period changes.
- Counts of anomaly alerts.
- Distribution of risk categories.
- Historical summaries for dashboards.

Where a KPI is calculated in both SQL and Python, the results should be reconciled using the same filters, units, and aggregation rules.

## 8. Power BI Integration

Power BI will serve as the primary analytical dashboard tool.

Potential dashboard pages include:

### Page 1: Executive Overview
- Total measured consumption.
- Consumption trends.
- Number of flagged anomalies.
- Highest-priority investigations.
- Data coverage and freshness indicators.

### Page 2: Consumption Analytics
- Daily and monthly trends.
- Meter or zone comparisons.
- Peak-use analysis.
- Historical baseline comparisons.

### Page 3: Anomaly Monitoring
- Flagged observations.
- Anomaly scores.
- Supporting features.
- Time-based patterns and investigation priorities.

### Page 4: Water-Loss Analysis
- Water-balance indicators, if measurements support them.
- Zone comparisons.
- Estimated differences and assumptions.
- Historical trends.

### Page 5: Forecasting
- Forecast versus actual consumption.
- Evaluation metrics.
- Forecast horizon and limitations.

Dashboard pages will be included only when their underlying data and analytical outputs are available.

### Integration Strategy

The preferred initial workflow is to prepare validated database tables or analysis-ready exports and connect Power BI to them.

The precise connector and refresh process will be chosen after the database engine is finalized.

## 9. Application Integration

A user-facing interface may be implemented using Streamlit for rapid development.

Potential capabilities:
- View consumption summaries.
- Filter results by date or zone.
- Inspect anomaly alerts.
- Review explanations and scores.
- View forecasts, if implemented.
- Export analytical results where appropriate.

FastAPI may be used instead if the project needs a dedicated REST API or multiple independent clients.

The application must distinguish measured values, calculated estimates, and model predictions.

## 10. Model Inference Workflow

The intended inference workflow is:

1. Receive or load a validated observation.
2. Apply the same preprocessing and feature definitions used during training.
3. Generate a model score.
4. Apply the documented decision threshold.
5. Generate an alert record.
6. Attach supporting information where available.
7. Persist or expose the result.
8. Make the result available to the dashboard or application.

The system must not silently accept incompatible columns, units, or model artifacts.

## 11. Testing Strategy

### 11.1 Unit Testing

Test individual functions such as:
- Data validation.
- Consumption calculations.
- Feature generation.
- Anomaly scoring.
- KPI calculations.
- Database operations.

### 11.2 Integration Testing

Test:
- Data loading and preprocessing.
- Database insertion and retrieval.
- Model inference with expected feature schemas.
- Output generation for dashboards and the application.

### 11.3 Data Quality Testing

Test for:
- Missing required columns.
- Invalid timestamps.
- Duplicate records.
- Unexpected units.
- Invalid numeric values.
- Incorrect aggregation behavior.

### 11.4 Model Testing

Verify:
- Consistent inference output.
- Correct feature ordering and preprocessing.
- Model artifact compatibility.
- Evaluation reproducibility.
- Appropriate handling of missing or invalid inputs.

### 11.5 End-to-End Testing

Confirm that a representative dataset can pass through the supported pipeline and produce valid analytical results.

Testing must use appropriately documented fixtures or test data. Synthetic test cases must not be confused with real operational evidence.

## 12. Logging and Error Handling

The implementation should:
- Record important pipeline events.
- Report invalid input data clearly.
- Provide useful error messages.
- Avoid exposing credentials or sensitive records in logs.
- Track model and data versions where appropriate.
- Fail clearly when required inputs are missing.

## 13. Security and Privacy

- Keep credentials outside source code.
- Use environment variables or suitable configuration mechanisms for secrets.
- Exclude `.env` files from version control.
- Avoid committing the virtual environment.
- Restrict access to sensitive datasets.
- Minimize personal data collection.
- Validate user inputs and database operations.
- Use appropriate authentication if the deployed application requires it.

## 14. Version Control

Git will be used to track source code and documentation.

The repository should include:
- Source files.
- SQL definitions and queries.
- Tests.
- Documentation.
- Dependency specifications.
- Appropriate small configuration files.

The repository should normally exclude:
- `.venv/`
- Secret files.
- Temporary files.
- Private datasets.
- Large generated artifacts unless deliberately managed.
- Cache directories.

A suitable `.gitignore` will be verified during Phase 2.

## 15. Implementation Sequence

Development will follow this order:

1. Complete project specifications and roadmap.
2. Verify the Python environment and project structure.
3. Select and inspect the dataset.
4. Implement data validation and cleaning.
5. Perform exploratory data analysis.
6. Design and populate the SQL database.
7. Build analytical KPIs and baseline methods.
8. Engineer model features.
9. Implement and evaluate anomaly detection.
10. Add supported water-loss intelligence and forecasting.
11. Build Power BI dashboards.
12. Build the application interface and integrations.
13. Test the complete workflow.
14. Finalize documentation and deployment.

The order may be adjusted when dataset limitations or technical dependencies require it, but changes must be documented.

## 16. Definition of Done

A feature is complete when:
- Its requirements are clear.
- Its implementation exists.
- Its behavior has been tested.
- Its outputs are interpretable.
- Relevant assumptions are documented.
- Related files are saved and organized.

A phase is complete only when its required deliverables have been verified.

## 17. Technical Decisions Still Open

The following decisions must be finalized during implementation:
- Source dataset and license.
- Database engine.
- Final schema.
- Anomaly-detection algorithm and threshold.
- Whether supervised leakage prediction is feasible.
- Whether demand forecasting is supported.
- Power BI connection and refresh strategy.
- Whether the application requires Streamlit, FastAPI, or both.
- Deployment platform.

These decisions will be based on project requirements, data quality, implementation complexity, and evaluation results.

## 18. Final System Definition

AquaSenseAI will be developed as a modular, data-driven water intelligence system combining Python, Pandas, SQL, machine learning, and Power BI, with an optional application interface.

The system will transform validated water-related observations into consumption analytics, anomaly alerts, explainable investigation priorities, and other outputs supported by the available data.

The design prioritizes reproducibility, maintainability, measurable evaluation, and transparent reporting over unnecessary complexity or unsupported claims.

**Document status:** Initial system design specification. Final implementation decisions will be confirmed during the relevant development phases.
