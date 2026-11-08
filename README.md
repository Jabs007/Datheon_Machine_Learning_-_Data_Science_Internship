# Datheon Machine Learning & Data Science Internship
## All 4 Tasks - Complete Portfolio

This repository contains all 4 tasks from the DEATHON Machine Learning & Data Science Internship program. Each task demonstrates different machine learning and data science concepts, from classical classification to regression and churn analysis.

---

## 📁 Repository Structure

| Task | Description | Key Files |
|------|-------------|-----------|
| **Task 01** | Iris Classification | `Task-01-Iris-Classification/` - Dataset sanity checks, feature columns, species distribution, train/test split, Decision Tree & Logistic Regression models |
| **Task 02** | California Housing Regression | `Task-02-California-Housing/` - Feature visualization, coefficient analysis, Lasso/Ridge regularization, residuals, VIF analysis |
| **Task 03** | Telco Customer Churn | `Task-03-Telco-Customer-Churn/` - Customer churn analysis, ensemble methods, survival modeling, cohort analysis |
| **Task 04** | Sentiment Analysis | `Task-04-Sentiment-Classification/` - NLP text classifier with MultinomialNB & LogisticRegression, TF-IDF features, regex-based text cleaning |

---

## 🎯 Task 01: Iris Classification

**Problem:** Build a classifier to identify iris species (setosa, versicolor, virginica) from flower measurements.

- **Dataset:** 150 samples, 4 features (sepal/petal length & width), 3 classes
- **Models:** Decision Tree Classifier, Logistic Regression
- **Accuracy:** ~93-100% on test set
- **Key Features:** 
  - `check_data.py` - Pre-flight sanity check script
  - Pipeline with 80/20 stratified split
  - Confusion matrix analysis
  - Setosa perfect separation

**Files:** `src/check_data.py`, `models/*.pkl`, `notebooks/`, `reports/`

---

## 🏠 Task 02: California Housing Regression

**Problem:** Predict median house value in California districts based on geographic and demographic features.

- **Dataset:** 20,640 samples with 8 features (location, rooms, occupancy, etc.)
- **Models:** Linear Regression, Ridge, Lasso regression
- **Key Metrics:** MSE, RMSE, R² score
- **Key Features:**
  - Location-based price variation analysis
  - Multicollinearity assessment (VIF)
  - Feature importance via coefficients
  - Residual analysis and outlier detection
  - Cross-validation curves

**Files:** `src/check_data.py`, `models/*.pkl`, `notebooks/`, 20+ visualization figures

---

## 📊 Task 03: Telco Customer Churn

**Problem:** Analyze customer churn risk for a telecommunications company.

- **Dataset:** 7,032 customers with 21 features
- **Target:** Binary classification (churn yes/no)
- **Key Techniques:**
  - Ensemble methods (Random Forest, Gradient Boosting)
  - Survival analysis / Kaplan-Meier estimator
  - Cohort analysis by contract type
  - Feature importance for churn prediction
  - Imbalanced class handling

**Files:** `src/check_data.py`, `models/encoder.pkl`, `models/random_forest.pkl`, `notebooks/`, `reports/`

---

## 🎬 Task 04: Sentiment Analysis

**Problem:** Build a text classifier to identify positive and negative sentiment from customer product/movie reviews.

- **Dataset:** 1,000 movie reviews (500 positive, 500 negative) from NLTK's movie_reviews corpus
- **Preprocessing:** Regex-based cleaning (lowercase, HTML removal, punctuation/numbers removal, whitespace collapse)
- **Feature Extraction:** `TfidfVectorizer(max_features=5000)`
- **Models Trained:**
  1. **Multinomial Naive Bayes** - 81.5% test accuracy
  2. **Logistic Regression** - 82.5% test accuracy
- **Predict Function:** `predict_sentiment(phrase)` - Tests live custom user input phrases
- **Sample Predictions:**
  - "This movie was fantastic!" → positive (both models)
  - "The worst film ever..." → negative (both models)

**Files:** `src/train_classifier.py`, `tests/test_classifier.py`, `models/sentiment_model.pkl`, `models/tfidf_vectorizer.pkl`, `data/sentiment_reviews.csv`

---

## 🛠️ Technical Stack

- **Language:** Python 3.12
- **Libraries:** scikit-learn, numpy, pandas, nltk, matplotlib, seaborn
- **ML Algorithms:** MultinomialNB, LogisticRegression, DecisionTree, Ridge/Lasso, RandomForest
- **Preprocessing:** Regex, tokenization, stopword removal, TF-IDF vectorization
- **Evaluation:** Accuracy, precision, recall, confusion matrices, classification reports

---

## 📈 Performance Summary

| Task | Dataset Size | Models | Primary Metric |
|------|-------------|--------|----------------|
| 01 Iris | 150 samples, 3 classes | DT, LR | Accuracy ~93-100% |
| 02 Housing | 20,640 samples | LR, Ridge, Lasso | MSE, R² |
| 03 Churn | 7,032 customers | RF, Ensembles | Precision/Recall F1 |
| 04 Sentiment | 1,000 reviews (500/500) | MultinomialNB, LR | Accuracy 81.5% / 82.5% |

---

## 🚀 Getting Started

Each task has its own `requirements.txt` and can be run independently:

```bash
# Example for Task 04
cd Task-04-Sentiment-Classification
source .venv/bin/activate  # or use your project venv
python src/train_classifier.py
```

Or for Task 01:
```bash
cd Task-01-Iris-Classification
python src/check_data.py  # Run sanity checks
```

---

## 📜 Certificate Information

All 4 tasks completed in the DEATHON Machine Learning & Data Science Internship program. The Datheon Technical Directorate reviews submitted code to issue the **Verified Certificate of Completion** with QR code.

**GitHub:** https://github.com/Jabs007/Datheon_Machine_Learning_-_Data_Science_Internship.git