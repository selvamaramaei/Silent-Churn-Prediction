# 📊 Silent Churn Prediction & Monitoring System

## About the Project

This project was developed to strengthen a platform's user retention strategy by detecting **Silent Churn** users — users who haven't cancelled their subscription, but whose usage frequency or duration has quietly declined.

Unlike traditional "explicit churn" models, this system analyzes negative trends in user behavioral data (particularly `usage_drop_rate`). Despite severe class imbalance in the dataset (1.84% churn rate), a high-**Recall**-focused engineering approach is used to generate "at-risk user lists" that give the business a chance to intervene.

### Problem Solved & Business Value
* **Business Problem:** Users often stop using a platform long before they actually cancel their subscription, which leads to revenue loss that goes unnoticed until it's too late.
* **Engineering Solution:** This system processes raw usage logs into engineered features, identifies high-risk users (labels), and provides an interactive dashboard for analyzing those at-risk users.
* **Prioritized Metric: RECALL (Class 1).** As an engineer, I made a deliberate trade-off: "I'd rather risk asking a healthy customer 'is everything okay?' (a False Positive) than miss a user quietly churning out the door (a False Negative)."

---

## Project Architecture & Data Flow

The project follows a fully modular and scalable MLOps structure, consisting of 4 main layers — from raw data generation all the way to the end user viewing the report.

```text
[ DATA GENERATION ] --> [ DATABASE ] --> [ FEATURE ENGINEERING ] --> [ MODELING ] --> [ MONITORING ]
        |                    |                     |                      |                |
 data_generation.py    PostgreSQL DB       feature_pipeline.py      XGBoost & RF      dashboard.py
 (Raw CSV Generation)  (SQL Storage)       (7d/30d Averages)        (Training & DVC)  (Risk Analysis)
```

## Folder Structure

```text
 Silent-Churn/
├── data/                    # Raw and processed data (tracked with .dvc)
│   ├── processed/           # Metadata after SQL ingestion
│   ├── quarantine/          # Records that failed quality checks
│   └── raw/                 # feature_usage.csv and raw logs
├── models/                  # Trained model weights
│   ├── rf_silent_churn_v1.joblib  # Random Forest model
│   ├── silent_churn_v1.json       # XGBoost model
│   └── *.dvc                # Version tracking for large models
├── notebooks/               # Analysis and EDA work
│   ├── EDA.ipynb            # Exploratory Data Analysis
│   └── Model_Analysis.ipynb # In-depth model performance analysis
├── src/                     # Source code (logic layer)
│   ├── databases/           # DB connection: db_loader.py
│   ├── features/            # Feature pipeline: feature_pipeline.py
│   ├── flows/                # Ingestion flow: ingestion_flow.py
│   ├── ingestion/            # Data generation: data_generation.py, data_quality.py
│   ├── labeling/             # Churn labeling: labeler.py
│   ├── training/              # Training: train_xgboots.py, train_random_forest.py
│   ├── dashboard.py          # Interactive Streamlit dashboard
│   ├── inference.py          # XGBoost inference script
│   └── inference_rf.py       # Random Forest inference script
└── .gitignore               # Filters out venv and cache files
```

## Model Performance Analysis
Due to the severe class imbalance during training (1.84% churn rate), the models were configured to give higher weight to class 1 (churn) via `scale_pos_weight`/`class_weight`.

```text
| Metric (Class 1 - Churn) | XGBoost (Threshold 0.85) | Random Forest (Threshold 0.50) |
|---------------------------|---------------------------|----------------------------------|
| Recall                    | 0.71                      | 0.98                             |
| Precision                  | 0.18                      | 0.12                              |
| F1-Score                   | 0.28                      | 0.21                              |
```

## Dataset Architecture & PostgreSQL Integration
The data layer of the project is built on a structured usage-log architecture designed to capture behavioral churn signals.

### Raw Dataset Schema (`feature_usage.csv`)

Each row represents a single user's interaction with a specific feature:

| Field | Description |
|-------|-------------|
| usage_id | Unique usage record identifier |
| subscription_id | Subscription identifier |
| usage_date | Date of usage |
| feature_name | The feature that was used |
| usage_count | Daily usage count |
| usage_duration_secs | Total usage duration (seconds) |
| error_count | Number of errors that occurred |

This structure is normalized to allow user behavior to be analyzed from a time-series perspective.

---

### PostgreSQL Feature Engineering Layer

After the raw data is ingested into the PostgreSQL database, subscription-level behavioral metrics are computed. This layer handles data enrichment and aggregation before the modeling stage.

Key engineered features:

- **daily_usage**
  Total usage volume for a given subscription on a given day.

- **usage_7d_avg**
  7-day moving average of usage (short-term trend indicator).

- **usage_30d_avg**
  30-day moving average of usage (long-term behavioral baseline).

- **usage_drop_rate**
  The ratio `usage_7d_avg / usage_30d_avg`.
  As this ratio drops, the risk of the user drifting away from the platform (behaviorally) increases.

This approach aims for early risk detection by analyzing declines in usage behavior, rather than relying on explicit churn (subscription cancellation).

---

## MLOps: DVC (Data Version Control)

In this project, large data files and model weights are not stored directly in the Git repository. Instead, they are versioned using **DVC (Data Version Control)**.

### Versioning Strategy

- After each model training run:
  - The dataset used is saved as a snapshot
  - The trained model file is tracked with DVC
  - A `.dvc` reference file is generated

- The GitHub repository:
  - Contains only reference files
  - Large data and model files are stored in DVC storage

### Benefits

- Reproducible model training
- Comparable experiments
- Lightweight, clean Git repository structure
- Data management suited for production environments

---
## Setup

```bash
git clone https://github.com/username/Silent-Churn.git
cd Silent-Churn
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Model Training

```bash
python src/training/train_xgboots.py
```

---

## Interactive Dashboard (Streamlit)

An interactive Streamlit dashboard was developed to allow analysis of the model outputs.

The dashboard supports:

- **Model selection** (XGBoost / Random Forest)
- **Risk threshold adjustment**
- **Setting the number of users to analyze**
- **Risk score distribution histogram (log scale)**
- **High-risk user list**
- **Individual user lookup by Account ID**
- **Per-user usage_drop_rate trend chart**

The dashboard allows switching between a global risk overview and micro-level user analysis.

---

## Running the Dashboard

```bash
streamlit run src/dashboard.py
```
