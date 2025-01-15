import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define a function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Streamlit app
def main():
    st.title("Model Evaluation App")

    # Upload training data
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"], key="train")
    test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"], key="test")

    if train_file is not None:
        train_data = pd.read_csv(train_file)
        st.write("### Training Data Preview")
        st.dataframe(train_data.head())

        if "cnt" in train_data.columns:
            y_train = train_data["cnt"]
            X_train = train_data.drop("cnt", axis=1)

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
                test_data = pd.read_csv(test_file)
                st.write("### Test Data Preview")
                st.dataframe(test_data.head())

                if "cnt" in test_data.columns:
                    y_test = test_data["cnt"]
                    X_test = test_data.drop("cnt", axis=1)

                    y_test_pred = model.predict(X_test)
                    rmse_test = calculate_rmse(y_test, y_test_pred)

                    st.write("### Test Metrics")
                    st.write(f"RMSE: {rmse_test:.4f}")
                else:
                    st.error("Test data must contain a 'cnt' column for the dependent variable.")
        else:
            st.error("Training data must contain a 'cnt' column for the dependent variable.")

if __name__ == "__main__":
    main()
