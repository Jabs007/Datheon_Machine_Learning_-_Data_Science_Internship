"""Check the saved models, metrics table, and figures the notebook produces."""

import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from tests.test_pipeline import build_pipeline
except ImportError:  # pragma: no cover - fallback when run from inside tests/
    from test_pipeline import build_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
FIGURE_DIR = REPO_ROOT / "reports" / "figures"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

EXPECTED_MODELS = ["encoder.pkl", "random_forest.pkl", "xgboost.pkl"]
EXPECTED_TABULAR = ["metrics.csv", "model_config.pkl"]
EXPECTED_FIGURES = [
    "churn_distribution.png",
    "numeric_by_churn.png",
    "categorical_churn_rates.png",
    "numeric_correlation.png",
    "class_imbalance.png",
    "roc_auc_comparison.png",
    "roc_curves.png",
    "confusion_matrices.png",
    "feature_importance.png",
]


class TestSavedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, _encoded, _encoder = (
            build_pipeline()
        )
        cls.encoder = joblib.load(MODEL_DIR / "encoder.pkl")
        cls.random_forest = joblib.load(MODEL_DIR / "random_forest.pkl")
        cls.xgboost = joblib.load(MODEL_DIR / "xgboost.pkl")

    def test_all_artifact_files_exist(self):
        for name in EXPECTED_MODELS + EXPECTED_TABULAR:
            path = MODEL_DIR / name
            self.assertTrue(path.is_file(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty: {path}")

    def test_metrics_csv_has_all_six_configurations(self):
        metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
        self.assertEqual(len(metrics), 6)
        self.assertEqual(set(metrics["Model"]), {"Random Forest", "XGBoost"})
        self.assertEqual(set(metrics["Balancing"]), {"none", "class_weight", "smote"})

    def test_best_configuration_beats_the_auc_target(self):
        metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
        best = float(metrics["Test ROC-AUC"].max())
        self.assertGreater(best, 0.80, f"best test ROC-AUC {best:.4f} missed the target")

    def test_models_were_trained_on_the_encoded_features(self):
        expected = list(self.X_train.columns)
        self.assertEqual(
            list(self.random_forest.named_steps["model"].feature_names_in_), expected
        )
        self.assertEqual(
            list(self.xgboost.named_steps["model"].feature_names_in_), expected
        )

    def test_saved_models_reproduce_the_metrics_table(self):
        metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
        config = joblib.load(MODEL_DIR / "model_config.pkl")
        checks = [
            ("Random Forest", self.random_forest, config["random_forest_strategy"]),
            ("XGBoost", self.xgboost, config["xgboost_strategy"]),
        ]
        for name, model, strategy in checks:
            with self.subTest(model=name):
                score = float(roc_auc_score(
                    self.y_test, model.predict_proba(self.X_test)[:, 1]
                ))
                expected = float(metrics.loc[
                    (metrics["Model"] == name) & (metrics["Balancing"] == strategy),
                    "Test ROC-AUC",
                ].iloc[0])
                self.assertAlmostEqual(score, expected, places=6)

    def test_encoder_round_trips(self):
        self.assertEqual(len(self.encoder.categories_), 15)
        self.assertIn("Contract", [c for c in self.encoder.feature_names_in_])


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
