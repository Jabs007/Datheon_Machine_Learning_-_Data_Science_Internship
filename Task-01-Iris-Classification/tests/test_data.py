"""Integrity checks for the raw benchmark dataset."""

import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "iris.csv"

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class TestRawData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_file_exists(self):
        self.assertTrue(DATA_PATH.is_file(), f"missing dataset: {DATA_PATH}")

    def test_schema(self):
        self.assertEqual(
            list(self.df.columns),
            FEATURE_COLUMNS + ["species"],
        )

    def test_row_count(self):
        self.assertEqual(len(self.df), 150)

    def test_no_missing_values(self):
        self.assertFalse(self.df.isnull().any().any())

    def test_classes_are_balanced(self):
        counts = self.df["species"].value_counts()
        self.assertEqual(set(counts.index), {"setosa", "versicolor", "virginica"})
        self.assertTrue((counts == 50).all(), f"unbalanced classes:\n{counts}")

    def test_single_duplicate_row(self):
        self.assertEqual(self.df.duplicated().sum(), 1)

    def test_features_are_float(self):
        for column in FEATURE_COLUMNS:
            self.assertTrue(
                pd.api.types.is_float_dtype(self.df[column]),
                f"{column} is {self.df[column].dtype}",
            )

    def test_species_is_string(self):
        self.assertTrue(pd.api.types.is_string_dtype(self.df["species"]))


if __name__ == "__main__":
    unittest.main()
