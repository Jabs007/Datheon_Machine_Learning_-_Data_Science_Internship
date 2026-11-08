"""Integrity checks for the Telco Customer Churn dataset."""

import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

TARGET = "Churn"
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


class TestRawData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_file_exists(self):
        self.assertTrue(DATA_PATH.is_file(), f"missing dataset: {DATA_PATH}")

    def test_schema(self):
        self.assertEqual(list(self.df.columns), EXPECTED_COLUMNS)

    def test_row_count(self):
        self.assertEqual(len(self.df), 7043)

    def test_no_missing_values_in_the_raw_file(self):
        """The 11 bad TotalCharges are blank strings, not true nulls."""
        self.assertEqual(int(self.df.isnull().sum().sum()), 0)

    def test_total_charges_is_text_with_eleven_blanks(self):
        self.assertEqual(int((self.df["TotalCharges"].astype(str).str.strip() == "").sum()), 11)

    def test_total_charges_blanks_are_all_new_customers(self):
        blank = pd.to_numeric(self.df["TotalCharges"], errors="coerce").isna()
        self.assertTrue(bool((self.df.loc[blank, "tenure"] == 0).all()))

    def test_churn_distribution(self):
        counts = {str(k): int(v) for k, v in self.df[TARGET].value_counts().items()}
        self.assertEqual(counts, {"No": 5174, "Yes": 1869})

    def test_churn_rate_is_imbalanced(self):
        rate = float((self.df[TARGET] == "Yes").mean())
        self.assertAlmostEqual(rate, 0.2654, places=4)
        self.assertLess(rate, 0.35)

    def test_no_duplicates(self):
        self.assertEqual(int(self.df.duplicated().sum()), 0)

    def test_customer_id_is_unique(self):
        self.assertEqual(int(self.df["customerID"].duplicated().sum()), 0)

    def test_numeric_columns(self):
        self.assertTrue(pd.api.types.is_integer_dtype(self.df["tenure"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.df["MonthlyCharges"]))

    def test_categorical_cardinality(self):
        expected = {
            "gender": 2, "Partner": 2, "Dependents": 2, "PhoneService": 2,
            "MultipleLines": 3, "InternetService": 3, "OnlineSecurity": 3,
            "OnlineBackup": 3, "DeviceProtection": 3, "TechSupport": 3,
            "StreamingTV": 3, "StreamingMovies": 3, "Contract": 3,
            "PaperlessBilling": 2, "PaymentMethod": 4,
        }
        for column, cardinality in expected.items():
            with self.subTest(column=column):
                self.assertEqual(int(self.df[column].nunique()), cardinality)

    def test_expected_feature_groups(self):
        self.assertEqual(len(CATEGORICAL), 15)
        self.assertEqual(len(NUMERIC), 3)


if __name__ == "__main__":
    unittest.main()
