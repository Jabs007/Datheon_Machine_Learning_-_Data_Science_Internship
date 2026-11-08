"""Check the saved models and figures the notebook is supposed to produce."""

import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
FIGURE_DIR = REPO_ROOT / "reports" / "figures"
DATA_PATH = REPO_ROOT / "data" / "raw" / "iris.csv"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_FIGURES = ["pairplot.png", "confusion_matrix.png", "decision_boundary.png"]
EXPECTED_MODELS = ["scaler.pkl", "logistic_regression.pkl", "decision_tree.pkl"]
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def _test_split():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X = df.drop("species", axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


class TestSavedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = _test_split()
        cls.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        cls.logistic = joblib.load(MODEL_DIR / "logistic_regression.pkl")
        cls.tree = joblib.load(MODEL_DIR / "decision_tree.pkl")

    def test_all_model_files_exist(self):
        for name in EXPECTED_MODELS:
            path = MODEL_DIR / name
            self.assertTrue(path.is_file(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty: {path}")

    def test_models_record_the_feature_names(self):
        """A saved model that knows its inputs fails loudly on mismatches."""
        self.assertEqual(list(self.logistic.feature_names_in_), FEATURE_COLUMNS)
        self.assertEqual(list(self.tree.feature_names_in_), FEATURE_COLUMNS)

    def test_models_record_the_classes(self):
        expected = ["setosa", "versicolor", "virginica"]
        self.assertEqual(list(self.logistic.classes_), expected)
        self.assertEqual(list(self.tree.classes_), expected)

    def test_logistic_regression_reproduces_benchmark(self):
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index,
        )
        accuracy = accuracy_score(self.y_test, self.logistic.predict(X_test_scaled))
        self.assertAlmostEqual(accuracy, 0.9333, places=4)

    def test_decision_tree_reproduces_benchmark(self):
        accuracy = accuracy_score(self.y_test, self.tree.predict(self.X_test))
        self.assertAlmostEqual(accuracy, 0.9333, places=4)

    def test_scaler_matches_a_fresh_fit(self):
        """The saved scaler must be the one the benchmark used."""
        from sklearn.preprocessing import StandardScaler

        fresh = StandardScaler().fit(self.X_train)
        self.assertTrue(
            (self.scaler.mean_ == fresh.mean_).all(),
            "saved scaler mean does not match a fresh fit",
        )
        self.assertTrue(
            (self.scaler.scale_ == fresh.scale_).all(),
            "saved scaler scale does not match a fresh fit",
        )


class TestSavedFigures(unittest.TestCase):
    def test_all_figures_exist_and_are_non_trivial_pngs(self):
        for name in EXPECTED_FIGURES:
            path = FIGURE_DIR / name
            self.assertTrue(path.is_file(), f"missing {path}")
            size = path.stat().st_size
            self.assertGreater(size, 5_000, f"{name} is suspiciously small ({size} B)")
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(8), PNG_SIGNATURE, f"{name} is not a PNG")


if __name__ == "__main__":
    unittest.main()
