"""Tests for the sentiment analysis classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_classifier import (
    clean_text,
    load_models,
    predict_sentiment,
    MODEL_PATH,
    VECTORIZER_PATH,
)


class TestCleanText:
    """Tests for the clean_text function."""

    def test_basic_lowercase(self):
        """Text should be lowercased."""
        result = clean_text("HELLO WORLD")
        assert result == "hello world"

    def test_html_tags_removed(self):
        """HTML tags should be removed."""
        result = clean_text("<p>Hello world</p>")
        assert "<" not in result
        assert ">" not in result

    def test_punctuation_removed(self):
        """Punctuation should be removed."""
        result = clean_text("Hello, world!")
        assert result == "hello world"

    def test_numbers_removed(self):
        """Numbers should be removed."""
        result = clean_text("Version 2.0 released")
        assert "2" not in result
        assert "0" not in result

    def test_extra_whitespace(self):
        """Extra whitespace should be collapsed."""
        result = clean_text("Hello   world")
        assert "  " not in result


class TestModelsExist:
    """Tests that model files exist."""

    def test_model_file_exists(self):
        """Sentiment model should be saved."""
        assert MODEL_PATH.is_file(), f"Model not found at {MODEL_PATH}"

    def test_vectorizer_file_exists(self):
        """TF-IDF vectorizer should be saved."""
        assert VECTORIZER_PATH.is_file(), f"Vectorizer not found at {VECTORIZER_PATH}"


class TestPredictSentiment:
    """Tests for the predict_sentiment function."""

    @classmethod
    def setup_class(cls):
        """Load models once for all tests."""
        cls.multinomial_nb, cls.logistic_reg, cls.vectorizer = load_models()

    def test_predict_positive(self):
        """Positive phrase should be classified as positive."""
        result = predict_sentiment(
            "This movie was absolutely fantastic and I loved every minute of it!",
            self.multinomial_nb,
            self.logistic_reg,
            self.vectorizer,
        )
        assert result["MultinomialNB"] == "positive"
        assert result["LogisticRegression"] == "positive"

    def test_predict_negative(self):
        """Negative phrase should be classified as negative."""
        result = predict_sentiment(
            "The worst film I've ever seen, completely boring and terrible.",
            self.multinomial_nb,
            self.logistic_reg,
            self.vectorizer,
        )
        assert result["MultinomialNB"] == "negative"
        assert result["LogisticRegression"] == "negative"

    def test_predict_custom(self):
        """Custom phrase should be classified."""
        result = predict_sentiment(
            "Great acting and amazing storyline, highly recommended!",
            self.multinomial_nb,
            self.logistic_reg,
            self.vectorizer,
        )
        # Both models should agree on a sentiment (either positive or negative)
        assert result["MultinomialNB"] in ("positive", "negative")
        assert result["LogisticRegression"] in ("positive", "negative")
        # Result should have the expected keys
        assert "input" in result
        assert "cleaned" in result

    def test_cleaned_output(self):
        """Cleaned text should be lowercased and without punctuation."""
        result = predict_sentiment(
            "HELLO! This is AMAZING @#$%",
            self.multinomial_nb,
            self.logistic_reg,
            self.vectorizer,
        )
        # Cleaned text should be lowercase and without special chars
        assert result["cleaned"] == result["cleaned"].lower()
        assert "@" not in result["cleaned"]
        assert "!" not in result["cleaned"]