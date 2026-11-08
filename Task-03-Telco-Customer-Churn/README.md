# Task 3: Telco Customer Churn — Ensemble Classification

Train an ensemble model to predict customer churn from telecom subscription
records, and compare **Random Forest** and **XGBoost** on ROC-AUC.

## Problem Statement

Train an ensemble machine learning model to predict customer churn using
telecom subscription records. Preprocess categorical features using
OneHotEncoder, balance class distributions with SMOTE or class weighting, train
Random Forest and XGBoost classifiers, and evaluate ROC-AUC curve (>80%).

## Dataset

`data/raw/Telco-Customer-Churn.csv` — 7,043 rows × 21 columns. Confirmed present
and profiled.

```
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn
```

Four properties drive the workflow:

- **`TotalCharges`** is stored as **text** and contains **11 blank strings**. Every
  blank belongs to a `tenure == 0` customer (signed up, never billed), so the
  notebook coerces it to numeric and fills those with **0.0** — not the column
  median (1,397.48), which would invent charges they never had.
- **`Churn`** is the binary target (`Yes` / `No`): 1,869 churners vs 5,174
  non-churners, a **26.54%** churn rate — imbalanced, so it must be balanced.
- **`customerID`** is an identifier with no predictive value and is dropped.
- **15 categorical columns** (2–4 levels each) are one-hot encoded; the first
  level of each is dropped.

If the file is ever replaced with one using different column names or a
different shape, the encoding and column-mapping steps are the only parts that
need adjusting.

## Planned Workflow

1. Load and profile the dataset; drop `customerID`.
2. Coerce `TotalCharges` to numeric and handle the resulting nulls.
3. Encode categorical features with `OneHotEncoder`.
4. Balance the target classes with SMOTE (`imbalanced-learn`), with
   `class_weight` / `scale_pos_weight` as a comparison.
5. Train `RandomForestClassifier` and `XGBClassifier` with cross-validation.
6. Plot ROC curves and report ROC-AUC (target > 0.80).
7. Plot the confusion matrix and the top 10 feature importances.
8. Save figures, metrics, and fitted models.

## Project Structure

```
Task-03-Telco-Customer-Churn/
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv    # <-- paste the dataset here
├── models/                            # Saved fitted models + metrics
├── notebooks/
│   └── 01_telco_churn_ensemble.ipynb  # main deliverable
├── reports/
│   └── figures/                       # Generated visualizations
├── src/
│   └── check_data.py                  # Data sanity check (exit 1 on failure)
├── tests/                             # Regression suite
├── requirements.txt                   # Pinned dependencies
└── README.md
```

## Expected Deliverable

A Jupyter notebook with a ROC curve, confusion matrix, and feature importance
bar chart, comparing Random Forest and XGBoost.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Reproducibility

- All randomness seeded (`random_state=42`).
- `requirements.txt` is a full `pip freeze` of the virtual environment.
- The notebook writes its own figures and models.
