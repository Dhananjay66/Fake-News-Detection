# 📰 Fake News Detection App

A Machine Learning web application that detects whether a news article is **Real or Fake** using Natural Language Processing (NLP).

## 🚀 Live Demo
[Click here to view the app](#) <!-- Replace with your Streamlit Cloud URL after deployment -->

## 📌 Features
- Detects fake vs real news articles
- Simple and clean web interface
- Trained on real-world news datasets
- Fast predictions using TF-IDF + ML model

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **ML Libraries:** Scikit-learn, Pandas, NumPy
- **Model:** TF-IDF Vectorizer + Classification Model

## 📂 Project Structure
```
Fake-News-Detection/
├── app.py              # Main Streamlit app
├── train_model.py      # Model training script
├── detect.py           # Detection logic
├── Fake.csv            # Fake news dataset
├── True.csv            # Real news dataset
├── model.pkl           # Trained model
├── vectorizer.pkl      # TF-IDF vectorizer
└── requirements.txt    # Dependencies
```

## ⚙️ How to Run Locally
```bash
# Clone the repo
git clone https://github.com/Dhananjay66/Fake-News-Detection.git
cd Fake-News-Detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the app
streamlit run app.py
```

## 📊 Dataset
- `Fake.csv` — Collection of fake news articles
- `True.csv` — Collection of real news articles

## 🙋‍♂️ Author
**Dhananjay** — [GitHub](https://github.com/Dhananjay66)