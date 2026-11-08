"""Structural checks on the deliverable notebook itself.

The pipeline tests re-derive the numbers from the raw data; these tests make
sure the notebook is the artifact that actually produces them - valid nbformat,
all required steps present, and wired up to save its own figures and models.
"""

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "01_iris_exploration.ipynb"

REQUIRED_CODE_SNIPPETS = [
    "StandardScaler",          # feature scaling
    "LogisticRegression",      # model 1
    "DecisionTreeClassifier",  # model 2
    "train_test_split",        # stratified split
    "confusion_matrix",        # benchmark metric
    "sns.pairplot",            # exploratory pairplot
    "sns.heatmap",             # confusion matrix heatmap
    "contourf",                # decision boundary
    "plt.savefig",             # figures persisted to reports/figures
    "joblib.dump",             # models persisted to models/
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

    def test_required_steps_are_present(self):
        for snippet in REQUIRED_CODE_SNIPPETS:
            self.assertIn(snippet, self.code, f"notebook is missing: {snippet}")

    def test_title_cell_contains_no_code(self):
        """A markdown cell must not start with a Python statement."""
        first_markdown = next(
            c for c in self.nb["cells"] if c["cell_type"] == "markdown"
        )
        source = "".join(first_markdown["source"])
        self.assertFalse(
            source.startswith(("from ", "import ")),
            f"title markdown cell contains code: {source[:60]!r}",
        )

    def test_summary_reports_the_tie(self):
        """Both models score identically; the summary must not crown a winner."""
        self.assertIn("tied", self.markdown.lower())
        self.assertNotIn("Best model", self.markdown)

    def test_notebook_records_the_artifact_paths(self):
        self.assertIn("../reports/figures", self.code)
        self.assertIn("../models", self.code)

    def test_check_data_script_is_runnable_from_any_directory(self):
        """check_data.py must resolve its path from __file__, not the cwd."""
        script = (REPO_ROOT / "src" / "check_data.py").read_text()
        self.assertIn("__file__", script)
        self.assertNotIn('DATA_PATH = "data/raw/iris.csv"', script)


if __name__ == "__main__":
    unittest.main()
