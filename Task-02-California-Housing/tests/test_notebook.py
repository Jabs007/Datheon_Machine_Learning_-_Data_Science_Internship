"""Structural checks on the deliverable notebook itself.

The pipeline tests re-derive the numbers from the raw data; these tests make sure
the notebook is the artifact that actually produces them - valid nbformat, all
required steps present, and wired up to save its own figures and models.
"""

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "01_california_housing_regression.ipynb"

REQUIRED_CODE_SNIPPETS = [
    "pd.read_csv",              # 1. load the dataset
    "fillna",                   # 2. impute missing values
    "sns.heatmap",              # 2. correlation heatmap
    "StandardScaler",           # 3. standardize continuous variables
    "OneHotEncoder",            # 3. encode categorical attributes
    "train_test_split",         # split
    "compute_vif",              # multicollinearity
    "LinearRegression",         # OLS
    "Ridge",                    # Ridge (L2)
    "Lasso",                    # Lasso (L1)
    "GridSearchCV",             # hyperparameter search
    "mean_squared_error",       # RMSE
    "mean_absolute_error",      # MAE
    "r2_score",                 # R2
    "Predicted vs Actual",      # predicted vs actual plot
    "Residual Distribution",    # residual distribution plot
    "plt.savefig",              # figures persisted
    "joblib.dump",              # models persisted
]


class TestNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK_PATH.read_text())
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.nb["cells"]
            if cell["cell_type"] == "code"
        )
        cls.markdown = "\n".join(
            "".join(cell["source"])
            for cell in cls.nb["cells"]
            if cell["cell_type"] == "markdown"
        )

    def test_notebook_is_valid_nbformat(self):
        self.assertEqual(self.nb["nbformat"], 4)
        self.assertIn("cells", self.nb)
        self.assertIn("metadata", self.nb)
        self.assertGreater(len(self.nb["cells"]), 0)

    def test_every_cell_is_markdown_or_code(self):
        for index, cell in enumerate(self.nb["cells"]):
            self.assertIn(
                cell["cell_type"],
                {"markdown", "code"},
                f"cell {index} has unexpected type {cell['cell_type']}",
            )
            self.assertTrue(cell["source"], f"cell {index} is empty")

    def test_all_required_steps_are_present(self):
        for snippet in REQUIRED_CODE_SNIPPETS:
            self.assertIn(snippet, self.code, f"notebook is missing: {snippet}")

    def test_title_cell_contains_no_code(self):
        first_markdown = next(
            cell for cell in self.nb["cells"] if cell["cell_type"] == "markdown"
        )
        source = "".join(first_markdown["source"])
        self.assertFalse(
            source.startswith(("from ", "import ")),
            f"title markdown cell contains code: {source[:60]!r}",
        )

    def test_summary_does_not_overstate_the_winner(self):
        """The models are within noise; the summary must say so."""
        lowered = self.markdown.lower()
        self.assertIn("caveat", lowered)
        self.assertTrue(
            "essentially nothing" in lowered or "within noise" in lowered,
            "summary should state that regularization bought nothing meaningful",
        )
        self.assertNotIn("Best model: Lasso", self.markdown)

    def test_notebook_records_the_artifact_paths(self):
        self.assertIn("../reports/figures", self.code)
        self.assertIn("../models", self.code)

    def test_notebook_points_at_the_real_dataset(self):
        self.assertIn("../data/raw/housing.csv", self.code)

    def test_check_data_script_is_runnable_from_any_directory(self):
        script = (REPO_ROOT / "src" / "check_data.py").read_text()
        self.assertIn("__file__", script)


if __name__ == "__main__":
    unittest.main()
