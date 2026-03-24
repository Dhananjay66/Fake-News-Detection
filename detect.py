import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# Load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Debug: Check data shapes and sample content
print("Fake news dataset shape:", fake.shape)
print("Real news dataset shape:", real.shape)
print("\nSample fake news:")
print(fake.head(2))
print("\nSample real news:")
print(real.head(2))

# Add labels
fake['label'] = 0  # 0 = fake
real['label'] = 1  # 1 = real

# Combine datasets
data = pd.concat([fake, real], axis=0)

# Debug: Check label distribution
print("\nLabel distribution:")
print(data['label'].value_counts())

# Keep only text and label columns
data = data[['text', 'label']]
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle with fixed seed

# Improved text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text.strip()

data['text'] = data['text'].apply(clean_text)

# Remove empty texts
data = data[data['text'].str.len() > 0]
print(f"\nDataset shape after cleaning: {data.shape}")

X = data['text']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Debug: Check train/test distribution
print("\nTrain set label distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest set label distribution:")
print(pd.Series(y_test).value_counts())

# Vectorize with better parameters
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_features=10000,  # Limit features to avoid overfitting
    ngram_range=(1, 2)  # Include unigrams and bigrams
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nFeature matrix shape: {X_train_vec.shape}")

# Train model with better parameters
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0  # Regularization parameter
)
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Test with sample news
sample_news = [
    "Government confirms economic recovery is real",
    "Scientists discover breakthrough in renewable energy",
    "Breaking: Aliens found in basement of pizza shop",
    "Stock market reaches new highs amid positive economic data"
]

sample_vec = vectorizer.transform(sample_news)
predictions = model.predict(sample_vec)
probabilities = model.predict_proba(sample_vec)

print("\nSample Predictions:")
for i, news in enumerate(sample_news):
    pred_label = "Real" if predictions[i] == 1 else "Fake"
    confidence = max(probabilities[i])
    print(f"News: {news}")
    print(f"Prediction: {pred_label} (Confidence: {confidence:.3f})")
    print(f"Probabilities - Fake: {probabilities[i][0]:.3f}, Real: {probabilities[i][1]:.3f}")
    print()

# Debug: Check most important features
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Features that indicate fake news (negative coefficients)
fake_indicators = np.argsort(coefficients)[:20]
print("Top 20 features indicating FAKE news:")
for idx in fake_indicators:
    print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")

print()

# Features that indicate real news (positive coefficients)
real_indicators = np.argsort(coefficients)[-20:]
print("Top 20 features indicating REAL news:")
for idx in real_indicators:
    print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")