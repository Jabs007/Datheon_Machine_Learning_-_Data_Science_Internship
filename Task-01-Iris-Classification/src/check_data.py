"""Sanity-check the raw Iris dataset before any modelling.

Usage (from anywhere in the repo):
    python src/check_data.py

The dataset path is resolved from this file's location, so the working
directory does not matter. Exits with status 1 if any check fails, which
makes the script usable as a pre-flight check in a pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "iris.csv"

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["species"]
EXPECTED_ROWS = 150
EXPECTED_CLASSES = {"setosa", "versicolor", "virginica"}
# The benchmark CSV is known to contain exactly one duplicated row
# (indices 101 and 142: 5.8/2.7/5.1/1.9 virginica). Duplicates are
# reported, not treated as a failure - the notebook removes them.
EXPECTED_DUPLICATES = 1


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

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nSpecies distribution:")
    print(df["species"].value_counts().to_string())

    failures = []

    if list(df.columns) != EXPECTED_COLUMNS:
        failures.append(f"columns are {df.columns.tolist()}, expected {EXPECTED_COLUMNS}")

    if len(df) != EXPECTED_ROWS:
        failures.append(f"row count is {len(df)}, expected {EXPECTED_ROWS}")

    if set(df["species"].unique()) != EXPECTED_CLASSES:
        failures.append(
            f"species are {sorted(df['species'].unique())}, "
            f"expected {sorted(EXPECTED_CLASSES)}"
        )

    if df.isnull().any().any():
        failures.append("dataset contains missing values")

    if df.duplicated().sum() != EXPECTED_DUPLICATES:
        failures.append(
            f"duplicate count is {df.duplicated().sum()}, expected {EXPECTED_DUPLICATES}"
        )

    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_float_dtype(df[column]):
            failures.append(f"column '{column}' is not float dtype")

    if failures:
        print("\nCHECK FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nCHECK PASSED: schema, row count, class labels and dtypes are as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
