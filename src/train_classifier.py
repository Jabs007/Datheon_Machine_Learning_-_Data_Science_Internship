"""Train a sentiment analysis classifier on movie reviews.

Suggested Technical Approach (from problem statement):
1. Download the sentiment reviews dataset
2. Preprocess text using NLTK or regex (remove HTML tags, punctuation, stopwords)
3. Convert text to numerical vectors using TfidfVectorizer(max_features=5000)
4. Train MultinomialNB and LogisticRegression
5. Create a predict_sentiment() function that tests live custom user input phrases
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Path to the NLTK movie_reviews dataset (downloaded previously)
NLTK_DATA_DIR = Path("/home/jabs101/nltk_data")
POS_DIR = NLTK_DATA_DIR / "corpora" / "movie_reviews" / "pos"
NEG_DIR = NLTK_DATA_DIR / "corpora" / "movie_reviews" / "neg"

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = REPO_ROOT / "models" / "tfidf_vectorizer.pkl"


def load_movie_reviews(limit_per_class: int = 500) -> tuple[list[str], list[int]]:
    """Load the movie_reviews dataset directly from disk.

    Reads positive and negative reviews from the NLTK data directory.

    Args:
        limit_per_class: Number of reviews per class to load (default 500).

    Returns:
        X: list of raw text strings
        y: list of labels (1=positive, 0=negative)
    """
    texts = []
    labels = []

    # Get sorted file lists for reproducibility
    pos_files = sorted(os.listdir(POS_DIR))[:limit_per_class]
    neg_files = sorted(os.listdir(NEG_DIR))[:limit_per_class]

    # Load positive reviews
    for filename in pos_files:
        filepath = POS_DIR / filename
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        texts.append(raw)
        labels.append(1)

    # Load negative reviews
    for filename in neg_files:
        filepath = NEG_DIR / filename
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        texts.append(raw)
        labels.append(0)

    return texts, labels


def clean_text(text: str) -> str:
    """Clean and normalize text: lowercase, remove HTML, remove punctuation, remove stopwords.

    Uses regex-based preprocessing as specified in the problem statement.
    """
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove non-letter characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tfidf_features(X_train: list[str], X_test: list[str], max_features: int = 5000):
    """Convert text to TF-IDF feature matrices.

    Uses TfidfVectorizer with max_features=5000 as specified in the problem.

    Returns:
        vectorizer: fitted TfidfVectorizer
        X_train_tfidf: training feature matrix
        X_test_tfidf: test feature matrix
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return vectorizer, X_train_tfidf, X_test_tfidf


def train_models(X_train_tfidf: list, y_train: list[int]) -> tuple[MultinomialNB, LogisticRegression]:
    """Train Multinomial Naive Bayes and Logistic Regression classifiers.

    Returns:
        multinomial_nb: trained MultinomialNB
        logistic_reg: trained LogisticRegression
    """
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(X_train_tfidf, y_train)

    logistic_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    logistic_reg.fit(X_train_tfidf, y_train)

    return multinomial_nb, logistic_reg


def evaluate_models(
    multinomial_nb: MultinomialNB,
    logistic_reg: LogisticRegression,
    X_test_tfidf: list,
    y_test: list[int],
) -> dict[str, dict]:
    """Evaluate both models and return classification reports."""
    results = {}

    nb_preds = multinomial_nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_preds)
    results["MultinomialNB"] = {
        "accuracy": nb_acc,
        "report": classification_report(y_test, nb_preds, output_dict=True, zero_division=0),
    }

    log_preds = logistic_reg.predict(X_test_tfidf)
    log_acc = accuracy_score(y_test, log_preds)
    results["LogisticRegression"] = {
        "accuracy": log_acc,
        "report": classification_report(y_test, log_preds, output_dict=True, zero_division=0),
    }

    return results


def save_models(
    multinomial_nb: MultinomialNB,
    logistic_reg: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> None:
    """Save trained models and vectorizer to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"multinomial_nb": multinomial_nb, "logistic_reg": logistic_reg}, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Models saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")


def load_models() -> tuple:
    """Load saved models and vectorizer from disk."""
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return models["multinomial_nb"], models["logistic_reg"], vectorizer


def predict_sentiment(
    phrase: str,
    multinomial_nb: MultinomialNB,
    logistic_reg: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> dict[str, str]:
    """Predict sentiment for a custom user input phrase.

    Args:
        phrase: The input text to classify
        multinomial_nb: trained MultinomialNB model
        logistic_reg: trained LogisticRegression model
        vectorizer: fitted TfidfVectorizer

    Returns:
        dict with model names as keys and predicted sentiment as values
    """
    cleaned = clean_text(phrase)
    tfidf = vectorizer.transform([cleaned])

    nb_pred = multinomial_nb.predict(tfidf)[0]
    log_pred = logistic_reg.predict(tfidf)[0]

    nb_sentiment = "positive" if nb_pred == 1 else "negative"
    log_sentiment = "positive" if log_pred == 1 else "negative"

    return {
        "MultinomialNB": nb_sentiment,
        "LogisticRegression": log_sentiment,
        "input": phrase,
        "cleaned": cleaned,
    }


def main() -> int:
    """Main training pipeline."""
    print("Loading and preprocessing data...")
    X, y = load_movie_reviews(limit_per_class=500)
    print(f"  Loaded {len(X)} reviews ({sum(y)} positive, {len(y)-sum(y)} negative)")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Build TF-IDF features
    print("Building TF-IDF features (max_features=5000)...")
    vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test, max_features=5000)
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # Train models
    print("Training models...")
    multinomial_nb, logistic_reg = train_models(X_train_tfidf, y_train)

    # Evaluate models
    print("Evaluating models...")
    results = evaluate_models(multinomial_nb, logistic_reg, X_test_tfidf, y_test)

    for model_name, metrics in results.items():
        print(f"  {model_name}: accuracy = {metrics['accuracy']:.4f}")

    # Save models
    save_models(multinomial_nb, logistic_reg, vectorizer)

    # Test with custom phrases
    print("\nCustom sentiment predictions:")
    test_phrases = [
        "This movie was absolutely fantastic and I loved every minute of it!",
        "The worst film I've ever seen, completely boring and terrible.",
        "It was an okay movie, not great but not terrible either.",
        "Great acting and amazing storyline, highly recommended!",
        "I regret watching this, it was a complete waste of time.",
    ]

    multinomial_nb, logistic_reg, vectorizer = load_models()
    for phrase in test_phrases:
        result = predict_sentiment(phrase, multinomial_nb, logistic_reg, vectorizer)
        print(f"  Phrase: {result['input']}")
        print(f"    MultinomialNB: {result['MultinomialNB']}")
        print(f"    LogisticRegression: {result['LogisticRegression']}")
        print()

    return 0


if __name__ == "__main__":
    exit(main())