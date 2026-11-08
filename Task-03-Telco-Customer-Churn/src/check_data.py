"""Sanity-check the Telco Customer Churn dataset before any modelling.

Usage (from anywhere in the repo):
    python src/check_data.py

The dataset path is resolved from this file's location, so the working
directory does not matter. Exits with status 1 if any check fails, which makes
the script usable as a pre-flight check in a pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
EXPECTED_ROWS = 7043
EXPECTED_CHURN = {"No": 5174, "Yes": 1869}
EXPECTED_BLANK_TOTAL_CHARGES = 11


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        return 1

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {DATA_PATH.relative_to(REPO_ROOT)}")
    print(f"Shape: {df.shape}")

    print("\nDtypes:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())
    print(f"\nDuplicated rows: {df.duplicated().sum()}")
    print(f"Duplicated customerID: {df[ID_COLUMN].duplicated().sum()}")

    print("\nTarget distribution:")
    print(df[TARGET].value_counts().to_string())

    print("\nNumeric summary:")
    print(df[["tenure", "SeniorCitizen", "MonthlyCharges"]].describe().T.round(2).to_string())

    blank = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
    print(f"\nBlank TotalCharges strings: {blank}")

    failures = []

    if list(df.columns) != EXPECTED_COLUMNS:
        failures.append(f"columns are {df.columns.tolist()}, expected {EXPECTED_COLUMNS}")

    if len(df) != EXPECTED_ROWS:
        failures.append(f"row count is {len(df)}, expected {EXPECTED_ROWS}")

    churn_counts = {str(k): int(v) for k, v in df[TARGET].value_counts().items()}
    if churn_counts != EXPECTED_CHURN:
        failures.append(f"churn counts are {churn_counts}, expected {EXPECTED_CHURN}")

    if blank != EXPECTED_BLANK_TOTAL_CHARGES:
        failures.append(
            f"blank TotalCharges count is {blank}, expected {EXPECTED_BLANK_TOTAL_CHARGES}"
        )

    if df.duplicated().sum() != 0:
        failures.append("dataset contains duplicated rows")

    if df[ID_COLUMN].duplicated().sum() != 0:
        failures.append("customerID contains duplicates")

    if int(df["TotalCharges"].isnull().sum()) != 0:
        failures.append("TotalCharges already has real nulls (only blanks are expected)")

    if failures:
        print("\nCHECK FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nCHECK PASSED: schema, row count, target balance and TotalCharges blanks are as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
