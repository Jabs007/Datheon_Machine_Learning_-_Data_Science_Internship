"""Reproduce the notebook's full pipeline and check the published results.

This is the executable form of the notebook's Summary: it rebuilds the imputation,
the interaction features, the one-hot encoding, the split, the VIF-based feature
selection, the scaling, and all three models, then asserts the published metrics.
If any of those steps silently changes, these tests fail.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "housing.csv"

TARGET = "median_house_value"
RANDOM_STATE = 42
VIF_THRESHOLD = 10.0
ALPHAS = np.logspace(-3, 4, 15)

EXPECTED_DROPPED = {"total_bedrooms", "latitude", "households"}
EXPECTED_METRICS = {
    "OLS (LinearRegression)": {"RMSE": 69777.7340, "MAE": 48694.6598, "R2": 0.6284418},
    "Ridge (L2)": {"RMSE": 69782.0023, "MAE": 48698.5016, "R2": 0.6283963},
    "Lasso (L1)": {"RMSE": 69777.2384, "MAE": 48695.2616, "R2": 0.6284471},
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_vif(frame):
    values = {}
    for column in frame.columns:
        target = frame[column].to_numpy()
        others = frame.drop(columns=[column]).to_numpy()
        r2 = LinearRegression().fit(others, target).score(others, target)
        values[column] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(values).sort_values(ascending=False)


def build_pipeline():
    """Mirror the notebook from raw CSV through to scaled, selected features."""
    df = pd.read_csv(DATA_PATH)

    # 2. imputation
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

    # 4. interaction features
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    df["income_per_capita"] = df["median_income"] / df["population_per_household"]

    # 5. split
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # categorical encoding, fit on train only
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[["ocean_proximity"]])
    encoded = list(encoder.get_feature_names_out(["ocean_proximity"]))
    X_train = pd.concat(
        [
            X_train.drop(columns=["ocean_proximity"]),
            pd.DataFrame(
                encoder.transform(X_train[["ocean_proximity"]]),
                columns=encoded,
                index=X_train.index,
            ),
        ],
        axis=1,
    )
    X_test = pd.concat(
        [
            X_test.drop(columns=["ocean_proximity"]),
            pd.DataFrame(
                encoder.transform(X_test[["ocean_proximity"]]),
                columns=encoded,
                index=X_test.index,
            ),
        ],
        axis=1,
    )

    # 6. multicollinearity, train only
    kept = X_train.copy()
    dropped = set()
    while True:
        vif = compute_vif(kept)
        worst = vif.index[0]
        if vif.iloc[0] <= VIF_THRESHOLD:
            break
        dropped.add(worst)
        kept = kept.drop(columns=[worst])
    X_test = X_test[kept.columns]

    # 7. scaling, fit on train only
    continuous = [c for c in kept.columns if c not in encoded]
    scaler = StandardScaler()
    X_train_scaled = kept.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[continuous] = scaler.fit_transform(kept[continuous])
    X_test_scaled[continuous] = scaler.transform(X_test[continuous])

    return X_train_scaled, X_test_scaled, y_train, y_test, dropped, encoded, scaler


def train_models(X_train, y_train):
    ols = LinearRegression().fit(X_train, y_train)

    ridge_grid = GridSearchCV(
        Ridge(random_state=RANDOM_STATE),
        param_grid={"alpha": ALPHAS},
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    ).fit(X_train, y_train)

    lasso_grid = GridSearchCV(
        Lasso(random_state=RANDOM_STATE, max_iter=100000, tol=1e-4),
        param_grid={"alpha": ALPHAS},
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    ).fit(X_train, y_train)

    return ols, ridge_grid, lasso_grid


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.X_train, cls.X_test, cls.y_train, cls.y_test,
         cls.dropped, cls.encoded, cls.scaler) = build_pipeline()
        cls.ols, cls.ridge_grid, cls.lasso_grid = train_models(cls.X_train, cls.y_train)

    def test_imputation_and_engineering(self):
        self.assertEqual(len(self.X_train), 16512)
        self.assertEqual(len(self.X_test), 4128)
        self.assertIn("rooms_per_household", self.X_train.columns)
        self.assertIn("bedrooms_per_room", self.X_train.columns)
        self.assertIn("population_per_household", self.X_train.columns)
        self.assertIn("income_per_capita", self.X_train.columns)

    def test_split_is_reproducible(self):
        self.assertAlmostEqual(float(self.y_train.mean()), 207194.69, places=2)
        self.assertAlmostEqual(float(self.y_test.mean()), 205500.31, places=2)

    def test_vif_drops_the_expected_features(self):
        self.assertEqual(self.dropped, EXPECTED_DROPPED)

    def test_all_remaining_vifs_are_below_threshold(self):
        vif = compute_vif(self.X_train)
        self.assertLess(float(vif.max()), VIF_THRESHOLD)
        self.assertEqual(self.X_train.shape[1], 13)

    def test_encoding_produced_four_indicators(self):
        self.assertEqual(len(self.encoded), 4)
        self.assertTrue(all(c.startswith("ocean_proximity_") for c in self.encoded))
        # the dropped reference level must not be present
        self.assertNotIn("ocean_proximity_<1H OCEAN", self.encoded)

    def test_scaling_centre_and_unit_variance_on_train(self):
        continuous = [c for c in self.X_train.columns if c not in self.encoded]
        # ddof=0: StandardScaler produces the population standard deviation
        self.assertTrue(abs(self.X_train[continuous].mean()).max() < 1e-12)
        self.assertTrue(abs(self.X_train[continuous].std(ddof=0) - 1.0).max() < 1e-12)

    def test_one_hot_columns_are_left_unscaled(self):
        """Binary indicators must stay 0/1, not be rescaled."""
        for column in self.encoded:
            values = set(self.X_train[column].unique())
            self.assertEqual(values, {0, 1})

    def test_tuned_alphas(self):
        self.assertAlmostEqual(float(self.ridge_grid.best_params_["alpha"]), 0.3162, places=4)
        self.assertAlmostEqual(float(self.lasso_grid.best_params_["alpha"]), 1.0, places=4)

    def test_ols_metrics(self):
        prediction = self.ols.predict(self.X_test)
        self.assertAlmostEqual(rmse(self.y_test, prediction),
                               EXPECTED_METRICS["OLS (LinearRegression)"]["RMSE"], places=2)
        self.assertAlmostEqual(float(mean_absolute_error(self.y_test, prediction)),
                               EXPECTED_METRICS["OLS (LinearRegression)"]["MAE"], places=2)
        self.assertAlmostEqual(float(r2_score(self.y_test, prediction)),
                               EXPECTED_METRICS["OLS (LinearRegression)"]["R2"], places=6)

    def test_ridge_metrics(self):
        prediction = self.ridge_grid.best_estimator_.predict(self.X_test)
        self.assertAlmostEqual(rmse(self.y_test, prediction),
                               EXPECTED_METRICS["Ridge (L2)"]["RMSE"], places=2)
        self.assertAlmostEqual(float(mean_absolute_error(self.y_test, prediction)),
                               EXPECTED_METRICS["Ridge (L2)"]["MAE"], places=2)
        self.assertAlmostEqual(float(r2_score(self.y_test, prediction)),
                               EXPECTED_METRICS["Ridge (L2)"]["R2"], places=6)

    def test_lasso_metrics(self):
        prediction = self.lasso_grid.best_estimator_.predict(self.X_test)
        self.assertAlmostEqual(rmse(self.y_test, prediction),
                               EXPECTED_METRICS["Lasso (L1)"]["RMSE"], places=2)
        self.assertAlmostEqual(float(mean_absolute_error(self.y_test, prediction)),
                               EXPECTED_METRICS["Lasso (L1)"]["MAE"], places=2)
        self.assertAlmostEqual(float(r2_score(self.y_test, prediction)),
                               EXPECTED_METRICS["Lasso (L1)"]["R2"], places=6)

    def test_models_agree_within_noise(self):
        """The honest headline: no model beats OLS by a meaningful margin."""
        scores = {}
        for name, model in [
            ("OLS (LinearRegression)", self.ols),
            ("Ridge (L2)", self.ridge_grid.best_estimator_),
            ("Lasso (L1)", self.lasso_grid.best_estimator_),
        ]:
            scores[name] = rmse(self.y_test, model.predict(self.X_test))
        spread = max(scores.values()) - min(scores.values())
        self.assertLess(spread, 100.0, f"models diverged: {scores}")

    def test_residuals_are_centred_on_zero(self):
        prediction = self.ols.predict(self.X_test)
        residual = self.y_test - prediction
        self.assertAlmostEqual(float(residual.mean()), -35.79, places=2)
        self.assertAlmostEqual(float(residual.std()), 69786.18, places=2)


if __name__ == "__main__":
    unittest.main()
