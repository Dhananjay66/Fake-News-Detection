import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def debug_dataset():
    """Debug the dataset to identify issues"""
    print("=== DATASET DEBUGGING ===")
    
    # Check if files exist
    if not os.path.exists("Fake.csv") or not os.path.exists("True.csv"):
        print("❌ Dataset files not found!")
        return None, None
    
    # Load datasets
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")
    
    print(f"Fake dataset shape: {fake.shape}")
    print(f"Real dataset shape: {real.shape}")
    print(f"Fake columns: {fake.columns.tolist()}")
    print(f"Real columns: {real.columns.tolist()}")
    
    # Check for text column
    text_column = None
    if 'text' in fake.columns:
        text_column = 'text'
    elif 'title' in fake.columns:
        text_column = 'title'
    elif 'content' in fake.columns:
        text_column = 'content'
    else:
        print("❌ No suitable text column found!")
        return None, None
    
    print(f"Using column: {text_column}")
    
    # Sample data inspection
    print("\n=== SAMPLE FAKE NEWS ===")
    print(fake[text_column].iloc[0][:200] + "...")
    print("\n=== SAMPLE REAL NEWS ===")
    print(real[text_column].iloc[0][:200] + "...")
    
    # Check for missing values
    print(f"\nFake missing values: {fake[text_column].isnull().sum()}")
    print(f"Real missing values: {real[text_column].isnull().sum()}")
    
    # Check text lengths
    fake_lengths = fake[text_column].str.len()
    real_lengths = real[text_column].str.len()
    print(f"\nFake text avg length: {fake_lengths.mean():.0f}")
    print(f"Real text avg length: {real_lengths.mean():.0f}")
    
    return fake, real, text_column

def improved_preprocessing(text):
    """Improved text preprocessing"""
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

def train_improved_model():
    """Train an improved model with better preprocessing"""
    print("\n=== TRAINING IMPROVED MODEL ===")
    
    # Debug dataset first
    fake, real, text_column = debug_dataset()
    if fake is None:
        return
    
    # Prepare data with improved preprocessing
    fake['label'] = 0  # Fake = 0
    real['label'] = 1  # Real = 1
    
    # Apply preprocessing
    fake['processed_text'] = fake[text_column].apply(improved_preprocessing)
    real['processed_text'] = real[text_column].apply(improved_preprocessing)
    
    # Remove empty texts
    fake = fake[fake['processed_text'] != ""]
    real = real[real['processed_text'] != ""]
    
    print(f"After preprocessing - Fake: {len(fake)}, Real: {len(real)}")
    
    # Balance the dataset if needed
    min_samples = min(len(fake), len(real))
    fake_balanced = fake.sample(n=min_samples, random_state=42)
    real_balanced = real.sample(n=min_samples, random_state=42)
    
    print(f"Balanced dataset - Fake: {len(fake_balanced)}, Real: {len(real_balanced)}")
    
    # Combine datasets
    data = pd.concat([fake_balanced, real_balanced], axis=0)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = data['processed_text']
    y = data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set - Fake: {sum(y_train == 0)}, Real: {sum(y_train == 1)}")
    print(f"Test set - Fake: {sum(y_test == 0)}, Real: {sum(y_test == 1)}")
    
    # Improved vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.8,  # Increased from 0.7
        min_df=2,    # Words must appear at least twice
        max_features=5000,  # Limit features
        ngram_range=(1, 2),  # Include bigrams
        lowercase=True,
        strip_accents='unicode'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model with different parameters
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0,  # Regularization parameter
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"[[True Fake: {cm[0,0]}, False Real: {cm[0,1]}]")
    print(f" [False Fake: {cm[1,0]}, True Real: {cm[1,1]}]]")
    
    # Test with sample predictions
    print(f"\n=== SAMPLE PREDICTIONS ===")
    test_samples = [
        "Breaking: Scientists discover aliens living among us",
        "The Federal Reserve announced new interest rates today",
        "Government confirms economic recovery is underway",
        "Miracle cure that doctors don't want you to know about",
        "Stock market shows steady growth this quarter"
    ]
    
    for sample in test_samples:
        processed_sample = improved_preprocessing(sample)
        sample_vec = vectorizer.transform([processed_sample])
        prediction = model.predict(sample_vec)[0]
        probability = model.predict_proba(sample_vec)[0]
        
        result = "REAL" if prediction == 1 else "FAKE"
        confidence = max(probability) * 100
        
        print(f"Text: '{sample[:50]}...'")
        print(f"Prediction: {result} (Confidence: {confidence:.1f}%)")
        print(f"Probabilities: Fake={probability[0]:.3f}, Real={probability[1]:.3f}")
        print("-" * 50)
    
    # Save improved model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("\n✅ Improved model saved successfully!")
    
    return model, vectorizer

def test_saved_model():
    """Test the saved model"""
    print("\n=== TESTING SAVED MODEL ===")
    
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        test_cases = [
            ("Government announces new policy changes", "Should be REAL"),
            ("Scientists confirm climate change effects", "Should be REAL"),
            ("Breaking: Aliens control world governments", "Should be FAKE"),
            ("Miracle diet loses 50 pounds overnight", "Should be FAKE"),
            ("Stock market analysis shows positive trends", "Should be REAL")
        ]
        
        for text, expected in test_cases:
            processed_text = improved_preprocessing(text)
            text_vec = vectorizer.transform([processed_text])
            prediction = model.predict(text_vec)[0]
            probability = model.predict_proba(text_vec)[0]
            
            result = "REAL" if prediction == 1 else "FAKE"
            confidence = max(probability) * 100
            
            print(f"Text: '{text}'")
            print(f"Expected: {expected}")
            print(f"Predicted: {result} (Confidence: {confidence:.1f}%)")
            print(f"Probabilities: Fake={probability[0]:.3f}, Real={probability[1]:.3f}")
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    # Run the complete debugging and training process
    train_improved_model()
    test_saved_model()