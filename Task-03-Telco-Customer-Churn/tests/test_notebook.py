"""Structural checks on the deliverable notebook itself.

The pipeline tests re-derive the numbers from the raw data; these tests make sure
the notebook is the artifact that actually produces them - valid nbformat, all
required steps present, and wired up to save its own figures and models.
"""

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "01_telco_churn_ensemble.ipynb"

REQUIRED_CODE_SNIPPETS = [
    "pd.read_csv",            # load the dataset
    "pd.to_numeric",          # convert TotalCharges to numeric
    "OneHotEncoder",          # encode categorical features
    "train_test_split",       # split
    "SMOTE",                  # balance class distribution
    "class_weight",           # alternative balancing
    "scale_pos_weight",       # alternative balancing
    "RandomForestClassifier", # model 1
    "XGBClassifier",          # model 2
    "GridSearchCV",           # cross-validation / tuning
    "roc_auc_score",          # ROC-AUC
    "roc_curve",              # ROC curve plot
    "confusion_matrix",       # confusion matrix
    "feature_importances_",   # feature importance
    "plt.savefig",            # figures persisted
    "joblib.dump",            # models persisted
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

    def test_summary_states_the_auc_target(self):
        """The task requires ROC-AUC > 0.80; the summary must address it."""
        lowered = self.markdown.lower()
        self.assertIn("0.80", lowered)

    def test_notebook_records_the_artifact_paths(self):
        self.assertIn("../reports/figures", self.code)
        self.assertIn("../models", self.code)

    def test_notebook_points_at_the_real_dataset(self):
        self.assertIn("../data/raw/Telco-Customer-Churn.csv", self.code)

    def test_smote_is_inside_the_cv_pipeline(self):
        """SMOTE must not be applied before the split."""
        self.assertIn("ImbPipeline", self.code)
        self.assertIn('("smote", SMOTE(', self.code)

    def test_check_data_script_is_runnable_from_any_directory(self):
        script = (REPO_ROOT / "src" / "check_data.py").read_text()
        self.assertIn("__file__", script)


if __name__ == "__main__":
    unittest.main()
