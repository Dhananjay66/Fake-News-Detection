import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import re
import numpy as np

def improved_preprocessing(text):
    """Same preprocessing as used in the Streamlit app"""
    if pd.isnull(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove very short texts
    if len(text) < 10:
        return ""
    
    return text

def load_and_prepare_data():
    """Load and prepare the dataset with improved preprocessing"""
    # Check if files exist
    if not os.path.exists("Fake.csv") or not os.path.exists("True.csv"):
        print("Error: Fake.csv or True.csv not found!")
        return None, None

    try:
        # Load datasets
        print("Loading datasets...")
        fake = pd.read_csv("Fake.csv")
        real = pd.read_csv("True.csv")
        
        print(f"Fake news dataset shape: {fake.shape}")
        print(f"Real news dataset shape: {real.shape}")
        print(f"Fake news columns: {fake.columns.tolist()}")
        print(f"Real news columns: {real.columns.tolist()}")
        
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None, None

    # Add labels
    fake['label'] = 0  # 0 = Fake
    real['label'] = 1  # 1 = Real

    # Determine which column to use for text
    text_column = None
    if 'text' in fake.columns and 'text' in real.columns:
        text_column = 'text'
    elif 'title' in fake.columns and 'title' in real.columns:
        text_column = 'title'
    else:
        print("Error: No common 'text' or 'title' column found!")
        print(f"Fake columns: {fake.columns.tolist()}")
        print(f"Real columns: {real.columns.tolist()}")
        return None, None

    print(f"Using column: {text_column}")

    # Select relevant columns
    fake_data = fake[[text_column, 'label']].copy()
    real_data = real[[text_column, 'label']].copy()
    
    # Rename column to 'text' for consistency
    fake_data = fake_data.rename(columns={text_column: 'text'})
    real_data = real_data.rename(columns={text_column: 'text'})

    # Remove missing values
    fake_data = fake_data.dropna()
    real_data = real_data.dropna()
    
    print(f"After removing NaN - Fake: {len(fake_data)}, Real: {len(real_data)}")

    # Apply preprocessing
    print("Applying text preprocessing...")
    fake_data['text'] = fake_data['text'].apply(improved_preprocessing)
    real_data['text'] = real_data['text'].apply(improved_preprocessing)
    
    # Remove empty texts after preprocessing
    fake_data = fake_data[fake_data['text'] != ""]
    real_data = real_data[real_data['text'] != ""]
    
    print(f"After preprocessing - Fake: {len(fake_data)}, Real: {len(real_data)}")

    # Balance the dataset (optional - comment out if you want to keep original distribution)
    min_len = min(len(fake_data), len(real_data))
    if min_len > 0:
        fake_data = fake_data.sample(min_len, random_state=42)
        real_data = real_data.sample(min_len, random_state=42)
        print(f"Balanced dataset - Each class: {min_len} samples")

    # Combine and shuffle
    data = pd.concat([fake_data, real_data], axis=0)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Final dataset shape: {data.shape}")
    print(f"Label distribution:")
    print(data['label'].value_counts())
    
    # Show sample data
    print("\nSample fake news:")
    print(data[data['label'] == 0]['text'].iloc[0][:200] + "...")
    print("\nSample real news:")
    print(data[data['label'] == 1]['text'].iloc[0][:200] + "...")

    return data['text'], data['label']

def train_model():
    """Train the fake news detection model"""
    print("Starting model training...")
    
    # Load data
    X, y = load_and_prepare_data()
    if X is None:
        return

    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Train label distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test label distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # TF-IDF Vectorization
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_df=0.7,          # Ignore terms that appear in more than 70% of documents
        min_df=3,            # Ignore terms that appear in less than 3 documents
        max_features=10000   # Limit vocabulary size
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Train multiple models and compare
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            random_state=42,
            n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Final evaluation with best model
    y_pred = best_model.predict(X_test_vec)
    
    print(f"\nFinal Model Performance ({best_model_name}):")
    print(f"Accuracy: {best_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Test with sample predictions
    print("\nTesting with sample texts...")
    sample_texts = [
        "Government confirms economic recovery is real",
        "Scientists discover breakthrough in renewable energy",
        "Breaking: Aliens found in basement of pizza shop",
        "Stock market reaches new highs amid positive economic data"
    ]
    
    for text in sample_texts:
        processed_text = improved_preprocessing(text)
        sample_vec = vectorizer.transform([processed_text])
        prediction = best_model.predict(sample_vec)[0]
        probability = best_model.predict_proba(sample_vec)[0]
        
        label = "Real" if prediction == 1 else "Fake"
        confidence = max(probability)
        
        print(f"Text: {text}")
        print(f"Prediction: {label} (Confidence: {confidence:.3f})")
        print()

    # Save model and vectorizer
    print("Saving model and vectorizer...")
    try:
        with open("model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        print("✅ Model and vectorizer saved successfully!")
        
        # Save model info
        model_info = {
            'model_type': best_model_name,
            'accuracy': best_accuracy,
            'vocabulary_size': len(vectorizer.vocabulary_),
            'feature_shape': X_train_vec.shape,
            'preprocessing_function': 'improved_preprocessing'
        }
        
        with open("model_info.pkl", "wb") as f:
            pickle.dump(model_info, f)
            
        print("✅ Model info saved!")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")

if __name__ == "__main__":
    train_model()