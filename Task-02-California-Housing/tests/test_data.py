"""Integrity checks for the California Housing dataset."""

import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "housing.csv"

TARGET = "median_house_value"
NUMERIC_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
# column order in the raw CSV: numeric features, target, then the categorical
EXPECTED_COLUMNS = NUMERIC_COLUMNS + [TARGET, "ocean_proximity"]


class TestRawData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_file_exists(self):
        self.assertTrue(DATA_PATH.is_file(), f"missing dataset: {DATA_PATH}")

    def test_schema(self):
        self.assertEqual(list(self.df.columns), EXPECTED_COLUMNS)

    def test_row_count(self):
        self.assertEqual(len(self.df), 20640)

    def test_only_total_bedrooms_has_missing_values(self):
        missing = {c: int(n) for c, n in self.df.isnull().sum().items() if n > 0}
        self.assertEqual(missing, {"total_bedrooms": 207})

    def test_no_duplicates(self):
        self.assertEqual(self.df.duplicated().sum(), 0)

    def test_ocean_proximity_categories(self):
        counts = self.df["ocean_proximity"].value_counts()
        self.assertEqual(
            set(counts.index),
            {"<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"},
        )

    def test_numeric_columns_are_float(self):
        for column in NUMERIC_COLUMNS + [TARGET]:
            self.assertTrue(
                pd.api.types.is_float_dtype(self.df[column]),
                f"{column} is {self.df[column].dtype}",
            )

    def test_ocean_proximity_is_categorical(self):
        self.assertTrue(pd.api.types.is_string_dtype(self.df["ocean_proximity"]))

    def test_target_is_capped(self):
        """The census ceiling is a known property of this dataset."""
        self.assertEqual(float(self.df[TARGET].max()), 500001.0)
        self.assertEqual(int((self.df[TARGET] >= 500_000).sum()), 992)


if __name__ == "__main__":
    unittest.main()
