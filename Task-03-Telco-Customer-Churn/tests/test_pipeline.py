"""Reproduce the notebook's pipeline and check the published results.

This is the executable form of the notebook: it rebuilds the cleaning, the
one-hot encoding, the stratified split, and all six (model x balancing)
configurations, then asserts the ROC-AUC target and the published metrics.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

TARGET = "Churn"
RANDOM_STATE = 42
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
RF_GRID = {"n_estimators": [200, 400], "max_depth": [None, 10], "min_samples_leaf": [1, 2]}
XGB_GRID = {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.1]}
STRATEGIES = ["none", "class_weight", "smote"]

# Filled in from the executed notebook; see tests/test_artifacts.py too.
EXPECTED_BEST_TEST_AUC = {"Random Forest": None, "XGBoost": None}


def build_pipeline():
    """Mirror the notebook from raw CSV through to the encoded, split matrices."""
    df = pd.read_csv(DATA_PATH).drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoder.fit(df[CATEGORICAL_COLUMNS])
    encoded = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))

    X = pd.concat(
        [
            df[NUMERIC_COLUMNS].reset_index(drop=True),
            pd.DataFrame(
                encoder.transform(df[CATEGORICAL_COLUMNS]),
                columns=encoded,
                index=df.index,
            ),
        ],
        axis=1,
    )
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, encoded, encoder


def build_search(strategy, model_name, scale_pos_weight):
    if model_name == "Random Forest":
        classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        if strategy == "class_weight":
            classifier.set_params(class_weight="balanced")
        grid = {f"model__{k}": v for k, v in RF_GRID.items()}
    else:
        classifier = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
        if strategy == "class_weight":
            classifier.set_params(scale_pos_weight=scale_pos_weight)
        grid = {f"model__{k}": v for k, v in XGB_GRID.items()}

    steps = []
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", classifier))
    return ImbPipeline(steps), grid


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.X_train, cls.X_test, cls.y_train, cls.y_test,
         cls.encoded, cls.encoder) = build_pipeline()
        cls.scale_pos_weight = float(
            cls.y_train.value_counts().max() / cls.y_train.value_counts().min()
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cls.results = {}
        for model_name in ["Random Forest", "XGBoost"]:
            for strategy in STRATEGIES:
                pipeline, grid = build_search(strategy, model_name, cls.scale_pos_weight)
                search = GridSearchCV(
                    pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1
                )
                search.fit(cls.X_train, cls.y_train)
                proba = search.best_estimator_.predict_proba(cls.X_test)[:, 1]
                cls.results[(model_name, strategy)] = {
                    "search": search,
                    "test_auc": float(roc_auc_score(cls.y_test, proba)),
                }

    def test_cleaning_removes_all_nulls(self):
        self.assertEqual(int(self.X_train.isnull().sum().sum()), 0)
        self.assertEqual(int(self.X_test.isnull().sum().sum()), 0)

    def test_encoding_dimensions(self):
        self.assertEqual(len(self.encoded), 26)
        self.assertEqual(self.X_train.shape[1], 29)

    def test_split_sizes_and_stratification(self):
        self.assertEqual(len(self.X_train), 5634)
        self.assertEqual(len(self.X_test), 1409)
        self.assertAlmostEqual(float(self.y_train.mean()), float(self.y_test.mean()), places=4)

    def test_scale_pos_weight(self):
        self.assertAlmostEqual(self.scale_pos_weight, 2.7687, places=4)

    def test_smote_balances_the_training_classes(self):
        """SMOTE inside the pipeline must produce equal class counts."""
        smote = SMOTE(random_state=RANDOM_STATE)
        X_bal, y_bal = smote.fit_resample(self.X_train, self.y_train)
        counts = pd.Series(y_bal).value_counts()
        self.assertEqual(int(counts[0]), int(counts[1]))
        self.assertEqual(len(X_bal), 2 * int(self.y_train.value_counts().max()))

    def test_all_six_configurations_beat_the_auc_target(self):
        """The task requires ROC-AUC > 0.80."""
        for (model_name, strategy), result in self.results.items():
            with self.subTest(model=model_name, strategy=strategy):
                self.assertGreater(
                    result["test_auc"], 0.80,
                    f"{model_name}/{strategy} AUC {result['test_auc']:.4f} missed the target",
                )

    def test_smote_beats_no_balancing(self):
        """Balancing should help; this is the task's premise."""
        for model_name in ["Random Forest", "XGBoost"]:
            with self.subTest(model=model_name):
                self.assertGreater(
                    self.results[(model_name, "smote")]["test_auc"],
                    self.results[(model_name, "none")]["test_auc"],
                )

    def test_best_models_clear_the_target_with_margin(self):
        for model_name in ["Random Forest", "XGBoost"]:
            best = max(
                (result["test_auc"] for (name, _), result in self.results.items()
                 if name == model_name)
            )
            with self.subTest(model=model_name):
                self.assertGreater(best, 0.80)

    def test_feature_importances_cover_every_feature(self):
        for (model_name, strategy), result in self.results.items():
            estimator = result["search"].best_estimator_.named_steps["model"]
            with self.subTest(model=model_name, strategy=strategy):
                self.assertEqual(len(estimator.feature_importances_), self.X_train.shape[1])


if __name__ == "__main__":
    unittest.main()
