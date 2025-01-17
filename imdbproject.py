import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess data
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        data = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    # Validate required columns
    if 'review' not in data.columns or 'sentiment' not in data.columns:
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")
    
    data.dropna(subset=['review', 'sentiment'], inplace=True)
    data['review'] = data['review'].astype(str)  # Ensure all reviews are strings
    return data

# Preprocess and vectorize text
def preprocess_and_vectorize(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000, dtype=np.float32)
    X = tfidf.fit_transform(data['review'])  # Kept sparse format
    y = data['sentiment'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0)
    return X, y, tfidf

# Train models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
    return lr_model, dt_model, accuracy_score(y_test, lr_pred), accuracy_score(y_test, dt_pred)

# Streamlit App
def main():
    st.title("Sentiment Analysis App")
    st.write("Upload a CSV file to train the model or enter a review for sentiment prediction.")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            X, y, tfidf = preprocess_and_vectorize(data)
            lr_model, dt_model, lr_acc, dt_acc = train_models(X, y)
            joblib.dump((lr_model, dt_model, tfidf), 'models.pkl')
            st.success(f"Models trained. Logistic Regression Accuracy: {lr_acc:.2f}, Decision Tree Accuracy: {dt_acc:.2f}")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    if st.button("Retrain Model"):
        if uploaded_file is not None:
            try:
                data = load_data(uploaded_file)
                X, y, tfidf = preprocess_and_vectorize(data)
                lr_model, dt_model, lr_acc, dt_acc = train_models(X, y)
                joblib.dump((lr_model, dt_model, tfidf), 'models.pkl')
                st.success(f"Models retrained. Logistic Regression Accuracy: {lr_acc:.2f}, Decision Tree Accuracy: {dt_acc:.2f}")
            except Exception as e:
                st.error(f"Error during retraining: {e}")
        else:
            st.error("Please upload a new dataset to retrain the models.")

    review_input = st.text_area("Enter a review for sentiment prediction")

    if st.button("Predict Sentiment"):
        try:
            lr_model, dt_model, tfidf = joblib.load('models.pkl')
            review_vector = tfidf.transform([review_input])  # Kept sparse format
            lr_pred = lr_model.predict(review_vector)[0]
            dt_pred = dt_model.predict(review_vector)[0]
            sentiment = "Positive" if lr_pred == 1 else "Negative"
            st.write(f"Logistic Regression Prediction: {sentiment}")
            sentiment_dt = "Positive" if dt_pred == 1 else "Negative"
            st.write(f"Decision Tree Prediction: {sentiment_dt}")
        except FileNotFoundError:
            st.error("Please train the model first by uploading a dataset.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
