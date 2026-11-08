"""Sanity-check the California Housing dataset before any modelling.

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
DATA_PATH = REPO_ROOT / "data" / "raw" / "housing.csv"

TARGET = "median_house_value"
EXPECTED_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    TARGET,
    "ocean_proximity",
]
NUMERIC_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in (TARGET, "ocean_proximity")]
EXPECTED_ROWS = 20640
EXPECTED_MISSING = {"total_bedrooms": 207}
EXPECTED_CATEGORIES = {"<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"}


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        return 1

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {DATA_PATH.relative_to(REPO_ROOT)}")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes.to_string())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print(f"\nDuplicated rows: {df.duplicated().sum()}")

    print("\nDescriptive statistics (numeric columns):")
    print(df[NUMERIC_COLUMNS + [TARGET]].describe().T.round(3).to_string())

    print("\nocean_proximity distribution:")
    print(df["ocean_proximity"].value_counts().to_string())

    failures = []

    if list(df.columns) != EXPECTED_COLUMNS:
        failures.append(f"columns are {df.columns.tolist()}, expected {EXPECTED_COLUMNS}")

    if len(df) != EXPECTED_ROWS:
        failures.append(f"row count is {len(df)}, expected {EXPECTED_ROWS}")

    missing = {c: int(n) for c, n in df.isnull().sum().items() if n > 0}
    if missing != EXPECTED_MISSING:
        failures.append(f"missing values are {missing}, expected {EXPECTED_MISSING}")

    categories = set(df["ocean_proximity"].unique())
    if categories != EXPECTED_CATEGORIES:
        failures.append(
            f"ocean_proximity categories are {sorted(categories)}, "
            f"expected {sorted(EXPECTED_CATEGORIES)}"
        )

    if df.duplicated().sum() != 0:
        failures.append("dataset contains duplicated rows")

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_float_dtype(df[column]):
            failures.append(f"column '{column}' is not float dtype")

    if not pd.api.types.is_float_dtype(df[TARGET]):
        failures.append(f"target '{TARGET}' is not float dtype")

    if failures:
        print("\nCHECK FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nCHECK PASSED: schema, row count, missing values and categories are as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
