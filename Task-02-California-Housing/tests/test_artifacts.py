"""Check the saved models, metrics table, and figures the notebook produces."""

import unittest
from pathlib import Path

import joblib
import pandas as pd

try:
    from tests.test_pipeline import build_pipeline, EXPECTED_METRICS, rmse
except ImportError:  # pragma: no cover - fallback when run from inside tests/
    from test_pipeline import build_pipeline, EXPECTED_METRICS, rmse

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
FIGURE_DIR = REPO_ROOT / "reports" / "figures"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

EXPECTED_MODELS = [
    "scaler.pkl",
    "ols_linear_regression.pkl",
    "ridge.pkl",
    "lasso.pkl",
]
EXPECTED_TABULAR = ["metrics.csv", "cv_results.pkl"]
EXPECTED_FIGURES = [
    "target_distribution.png",
    "feature_distributions.png",
    "correlation_heatmap.png",
    "vif_after.png",
    "cv_curves.png",
    "predicted_vs_actual.png",
    "residual_distribution.png",
    "residual_vs_predicted.png",
    "coefficients.png",
]


class TestSavedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.dropped, cls.encoded, _ = (
            build_pipeline()
        )
        cls.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        cls.ols = joblib.load(MODEL_DIR / "ols_linear_regression.pkl")
        cls.ridge = joblib.load(MODEL_DIR / "ridge.pkl")
        cls.lasso = joblib.load(MODEL_DIR / "lasso.pkl")

    def test_all_model_files_exist(self):
        for name in EXPECTED_MODELS + EXPECTED_TABULAR:
            path = MODEL_DIR / name
            self.assertTrue(path.is_file(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty: {path}")

    def test_metrics_csv_matches_the_published_results(self):
        metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
        self.assertEqual(len(metrics), 3)
        for _, row in metrics.iterrows():
            expected = EXPECTED_METRICS[row["Model"]]
            self.assertAlmostEqual(row["RMSE"], expected["RMSE"], places=2, msg=row["Model"])
            self.assertAlmostEqual(row["MAE"], expected["MAE"], places=2, msg=row["Model"])
            self.assertAlmostEqual(row["R2"], expected["R2"], places=6, msg=row["Model"])

    def test_models_were_trained_on_the_selected_features(self):
        expected = list(self.X_train.columns)
        for name, model in [("OLS", self.ols), ("Ridge", self.ridge), ("Lasso", self.lasso)]:
            self.assertEqual(list(model.feature_names_in_), expected, name)
        # the scaler is fitted on the continuous columns only
        continuous = [c for c in expected if c not in self.encoded]
        self.assertEqual(list(self.scaler.feature_names_in_), continuous)

    def test_saved_models_reproduce_the_benchmark(self):
        for name, model in [("OLS", self.ols), ("Ridge", self.ridge), ("Lasso", self.lasso)]:
            score = rmse(self.y_test, model.predict(self.X_test))
            expected = EXPECTED_METRICS[
                {"OLS": "OLS (LinearRegression)", "Ridge": "Ridge (L2)", "Lasso": "Lasso (L1)"}[name]
            ]["RMSE"]
            self.assertAlmostEqual(score, expected, places=2, msg=name)

    def test_cv_results_record_the_alpha_grid(self):
        cv = joblib.load(MODEL_DIR / "cv_results.pkl")
        self.assertEqual(len(cv["alphas"]), 15)
        self.assertAlmostEqual(cv["ridge_best_alpha"], 0.3162, places=4)
        self.assertAlmostEqual(cv["lasso_best_alpha"], 1.0, places=4)
        self.assertEqual(len(cv["ridge_cv"]), 15)
        self.assertEqual(len(cv["lasso_cv"]), 15)


class TestSavedFigures(unittest.TestCase):
    def test_all_figures_exist_and_are_valid_pngs(self):
        for name in EXPECTED_FIGURES:
            path = FIGURE_DIR / name
            self.assertTrue(path.is_file(), f"missing {path}")
            size = path.stat().st_size
            self.assertGreater(size, 5_000, f"{name} is suspiciously small ({size} B)")
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(8), PNG_SIGNATURE, f"{name} is not a PNG")


if __name__ == "__main__":
    unittest.main()
