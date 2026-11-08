"""Reproduce the notebook's pipeline and check the documented benchmark.

These tests are the executable form of the README's Results section: they
rebuild the exact 80/20 stratified split and both models, then assert the
published accuracy and confusion matrices. If a change to the data, the
split, or the model settings silently moves the numbers, these fail.
"""

import unittest
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "iris.csv"

EXPECTED_ACCURACY = 0.9333
EXPECTED_CM = [[10, 0, 0], [0, 9, 1], [0, 1, 9]]


def build_split():
    """Mirror the notebook: drop duplicates, stratified 80/20, random_state=42."""
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X = df.drop("species", axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logistic = LogisticRegression(max_iter=1000, random_state=42)
    logistic.fit(
        pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
        y_train,
    )

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)

    return scaler, logistic, tree


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = build_split()
        cls.scaler, cls.logistic, cls.tree = train_models(cls.X_train, cls.y_train)

    def test_split_is_stratified_and_disjoint(self):
        self.assertEqual(len(self.X_train), 119)
        self.assertEqual(len(self.X_test), 30)
        self.assertEqual(self.y_train.value_counts().to_dict(),
                         {"setosa": 40, "versicolor": 40, "virginica": 39})
        self.assertEqual(self.y_test.value_counts().to_dict(),
                         {"setosa": 10, "versicolor": 10, "virginica": 10})
        # train and test indices must not overlap
        self.assertEqual(
            len(set(self.X_train.index) & set(self.X_test.index)), 0
        )

    def test_scaling_zero_centres_training_features(self):
        scaled = self.scaler.transform(self.X_train)
        self.assertTrue(abs(scaled.mean()) < 1e-12)
        self.assertTrue(abs(scaled.std() - 1.0) < 1e-12)

    def test_logistic_regression_accuracy(self):
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index,
        )
        accuracy = accuracy_score(self.y_test, self.logistic.predict(X_test_scaled))
        self.assertAlmostEqual(accuracy, EXPECTED_ACCURACY, places=4)

    def test_decision_tree_accuracy(self):
        accuracy = accuracy_score(self.y_test, self.tree.predict(self.X_test))
        self.assertAlmostEqual(accuracy, EXPECTED_ACCURACY, places=4)

    def test_logistic_regression_confusion_matrix(self):
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index,
        )
        cm = confusion_matrix(
            self.y_test,
            self.logistic.predict(X_test_scaled),
            labels=self.logistic.classes_,
        )
        self.assertEqual(cm.tolist(), EXPECTED_CM)

    def test_decision_tree_confusion_matrix(self):
        cm = confusion_matrix(
            self.y_test,
            self.tree.predict(self.X_test),
            labels=self.tree.classes_,
        )
        self.assertEqual(cm.tolist(), EXPECTED_CM)

    def test_setosa_is_perfectly_separated(self):
        """Both models get every setosa test sample right."""
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index,
        )
        for model, X in ((self.logistic, X_test_scaled), (self.tree, self.X_test)):
            cm = confusion_matrix(
                self.y_test, model.predict(X), labels=model.classes_
            )
            setosa_row = list(model.classes_).index("setosa")
            self.assertEqual(cm[setosa_row].sum(), 10)
            self.assertEqual(cm[setosa_row][setosa_row], 10)


if __name__ == "__main__":
    unittest.main()
