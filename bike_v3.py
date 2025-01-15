import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def preprocess_data(df):
    try:
        # Handle date columns
        if 'dteday' in df.columns:
            st.write("### Parsing date column 'dteday'")
            df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
            df['year'] = df['dteday'].dt.year
            df['month'] = df['dteday'].dt.month
            df['day'] = df['dteday'].dt.day
            df['weekday'] = df['dteday'].dt.weekday
            df.drop('dteday', axis=1, inplace=True)

        # Remove known target-leaking columns
        if 'casual' in df.columns or 'registered' in df.columns:
            st.write("### Removing target-leaking columns: 'casual' and 'registered'")
            df = df.drop(columns=['casual', 'registered'], errors='ignore')

        # Convert non-numeric columns to numeric if possible
        non_numeric_cols = df.select_dtypes(include=['object']).columns
        if not non_numeric_cols.empty:
            st.write("### Non-Numeric Columns Identified:", list(non_numeric_cols))
            df[non_numeric_cols] = df[non_numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        original_shape = df.shape
        df = df.dropna()
        st.write(f"Dropped {original_shape[0] - df.shape[0]} rows due to missing values.")

        if df.shape[0] == 0:
            raise ValueError("No valid data available after preprocessing.")

        return df
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")

def check_multicollinearity(df, threshold=0.8):
    corr_matrix = df.corr()
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    highly_correlated = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                highly_correlated.add(corr_matrix.columns[i])

    return highly_correlated

def main():
    st.title("Model Evaluation App")

    # Upload training data
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"], key="train")
    test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"], key="test")

    if train_file is not None:
        try:
            train_data = pd.read_csv(train_file)
            st.write("### Training Data Preview")
            st.dataframe(train_data.head())

            if "cnt" in train_data.columns:
                train_data = preprocess_data(train_data)

                y_train = train_data["cnt"]
                X_train = train_data.drop("cnt", axis=1)

                # Check for multicollinearity
                correlated_features = check_multicollinearity(X_train)
                if correlated_features:
                    st.write("### Highly Correlated Features (Threshold > 0.8):")
                    st.write(correlated_features)
                    X_train = X_train.drop(columns=list(correlated_features))

                model = LinearRegression()
                model.fit(X_train, y_train)

                # Training metrics
                r2_train = model.score(X_train, y_train)
                y_train_pred = model.predict(X_train)
                rmse_train = calculate_rmse(y_train, y_train_pred)

                st.write("### Training Metrics")
                st.write(f"R-squared: {r2_train:.4f}")
                st.write(f"RMSE: {rmse_train:.4f}")

                # Handle test data
                if test_file is not None:
                    try:
                        test_data = pd.read_csv(test_file)
                        st.write("### Test Data Preview")
                        st.dataframe(test_data.head())

                        if "cnt" in test_data.columns:
                            test_data = preprocess_data(test_data)
                            X_test = test_data.drop("cnt", axis=1)

                            # Drop correlated features from test set
                            X_test = X_test.drop(columns=list(correlated_features), errors='ignore')
                            y_test = test_data["cnt"]

                            y_test_pred = model.predict(X_test)
                            rmse_test = calculate_rmse(y_test, y_test_pred)

                            st.write("### Test Metrics")
                            st.write(f"RMSE: {rmse_test:.4f}")
                        else:
                            st.error("Test data must contain a 'cnt' column for the dependent variable.")
                    except Exception as e:
                        st.error(f"Error processing test data: {e}")
            else:
                st.error("Training data must contain a 'cnt' column for the dependent variable.")
        except Exception as e:
            st.error(f"Error processing training data: {e}")

if __name__ == "__main__":
    main()
