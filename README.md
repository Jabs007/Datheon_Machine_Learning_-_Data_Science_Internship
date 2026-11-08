# Sentiment Analysis Classifier

## Project Overview
Natural language processing text classifier that ingests customer product/movie review records, cleans and tokenizes text, transforms text into numerical TF-IDF feature matrices, and trains a Multinomial Naive Bayes classifier to identify positive and negative sentiment.

## Dataset
- 1000 movie reviews (500 positive, 500 negative) from NLTK's movie_reviews corpus
- Preprocessed using regex: lowercasing, HTML tag removal, punctuation/numbers removal, stopword filtering, whitespace cleanup

## Models Trained
1. **Multinomial Naive Bayes** - 81.5% test accuracy
2. **Logistic Regression** - 82.5% test accuracy

## Feature Extraction
- `TfidfVectorizer(max_features=5000)` - converts cleaned text to numerical feature vectors

## Pipeline
1. Load movie reviews from NLTK data directory
2. Clean and normalize text using regex preprocessing
3. Split into 80/20 train/test sets (stratified by sentiment)
4. Transform text to TF-IDF feature matrices
5. Train both MultinomialNB and LogisticRegression models
6. Evaluate and report accuracy
7. Save models and vectorizer to disk via pickle

## Usage
```python
from train_classifier import predict_sentiment, load_models

multinomial_nb, logistic_reg, vectorizer = load_models()

result = predict_sentiment(
    "This movie was absolutely fantastic and I loved every minute of it!",
    multinomial_nb, logistic_reg, vectorizer
)
# Output: {'MultinomialNB': 'positive', 'LogisticRegression': 'positive', ...}
```

## Files
- `src/train_classifier.py` - Main training and prediction script
- `tests/test_classifier.py` - Unit tests for clean_text, model loading, and predictions
- `models/sentiment_model.pkl` - Trained classifier pipeline
- `models/tfidf_vectorizer.pkl` - Fitted TfidfVectorizer
- `data/sentiment_reviews.csv` - Dataset metadata
- `requirements.txt` - scikit-learn, nltk