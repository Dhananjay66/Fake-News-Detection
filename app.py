# app.py
import streamlit as st
import pickle
import os
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

def improved_preprocessing(text):
    """Same preprocessing as used in training"""
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

@st.cache_resource
def load_model():
    """Load the trained model and vectorizer"""
    try:
        if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
            st.error("❌ Model files not found! Please run the training script first.")
            st.info("Run: `python train_improved_model.py`")
            return None, None
        
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        st.success("✅ Model loaded successfully!")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def predict_news(text, model, vectorizer):
    """Predict if news is fake or real with proper preprocessing"""
    try:
        # Apply the same preprocessing as used in training
        processed_text = improved_preprocessing(text)
        
        if processed_text == "":
            return None, None, "Text too short or empty after preprocessing"
        
        # Transform text
        input_vector = vectorizer.transform([processed_text])
        
        # Get prediction and probability
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0]
        
        return prediction, probability, None
    except Exception as e:
        return None, None, f"Error making prediction: {str(e)}"

def main():
    st.title("📰 Fake News Detection App")
    st.markdown("Enter a news article below to check if it's likely to be fake or real.")
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Input section
    st.subheader("📝 Enter News Text")
    
    # Sample texts for testing
    with st.expander("🧪 Try these sample texts"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Real News Examples:**")
            st.code("The Federal Reserve announced new interest rates today")
            st.code("Scientists at MIT published research on renewable energy")
            st.code("Stock market shows steady growth this quarter")
        
        with col2:
            st.markdown("**Fake News Examples:**")
            st.code("Breaking: Aliens control world governments")
            st.code("Miracle cure that doctors don't want you to know")
            st.code("Government confirms weather is controlled by aliens")
    
    user_input = st.text_area(
        "Paste your news article here:",
        height=150,
        placeholder="Enter the news text you want to check..."
    )
    
    # Analysis section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze News", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Text"):
            st.rerun()
    
    with col3:
        debug_mode = st.checkbox("🔧 Debug Mode")
    
    if analyze_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some news text to analyze!")
        else:
            with st.spinner("Analyzing news article..."):
                prediction, probability, error = predict_news(user_input, model, vectorizer)
                
                if error:
                    st.error(f"❌ {error}")
                elif prediction is not None:
                    # Debug information
                    if debug_mode:
                        st.subheader("🔧 Debug Information")
                        processed_text = improved_preprocessing(user_input)
                        st.write(f"**Processed text:** {processed_text[:200]}...")
                        st.write(f"**Text length:** {len(processed_text)} characters")
                        st.write(f"**Raw prediction:** {prediction}")
                        st.write(f"**Raw probabilities:** {probability}")
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Remember: 0 = Fake, 1 = Real
                    if prediction == 1:
                        st.success("✅ This news appears to be **REAL**")
                        confidence = probability[1] * 100
                        st.info(f"Real News Confidence: {confidence:.1f}%")
                    else:
                        st.error("❌ This news appears to be **FAKE**")
                        confidence = probability[0] * 100
                        st.info(f"Fake News Confidence: {confidence:.1f}%")
                    
                    # Show probability breakdown
                    st.subheader("📈 Probability Breakdown")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Fake Probability", f"{probability[0]:.2%}")
                        st.progress(probability[0])
                    
                    with col2:
                        st.metric("Real Probability", f"{probability[1]:.2%}")
                        st.progress(probability[1])
                    
                    # Confidence interpretation
                    max_prob = max(probability)
                    if max_prob > 0.8:
                        st.success("🎯 High confidence prediction")
                    elif max_prob > 0.6:
                        st.info("⚖️ Medium confidence prediction")
                    else:
                        st.warning("❓ Low confidence prediction - treat with caution")
    
    # Information section
    st.sidebar.markdown("## ℹ️ About This Model")
    st.sidebar.info(
        "This app uses a Logistic Regression model trained on news articles. "
        "It analyzes text patterns using TF-IDF vectorization to classify news as real or fake."
    )
    
    st.sidebar.markdown("## 🎯 Model Labels")
    st.sidebar.markdown(
        "- **0 = Fake News** ❌\n"
        "- **1 = Real News** ✅"
    )
    
    st.sidebar.markdown("## ⚠️ Important Notes")
    st.sidebar.warning(
        "- This tool is for educational purposes only\n"
        "- Always verify news from multiple reliable sources\n"
        "- Model accuracy depends on training data quality\n"
        "- Results should be interpreted with caution"
    )
    
    st.sidebar.markdown("## 🔧 How It Works")
    st.sidebar.markdown(
        "1. **Text Preprocessing** - Clean and normalize text\n"
        "2. **TF-IDF Vectorization** - Convert text to numerical features\n"
        "3. **Logistic Regression** - Classify as fake (0) or real (1)\n"
        "4. **Probability Calculation** - Provide confidence scores"
    )

if __name__ == "__main__":
    main()