import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to preprocess the data
def preprocess_data(data):
    data['mileage'] = pd.to_numeric(data['mileage'].str.replace('kmpl', ''), errors='coerce')
    data['engine'] = pd.to_numeric(data['engine'].str.replace('CC', ''), errors='coerce')
    data['max_power'] = pd.to_numeric(data['max_power'].str.replace('bhp', ''), errors='coerce')
    data['torq'] = pd.to_numeric(data['torque'].str.extract(r'(\d+rpm)')[0].str.replace('rpm', ''), errors='coerce')

    numcols = data.select_dtypes(include=np.number)
    for col in numcols.columns:
        numcols[col] = numcols[col].fillna(numcols[col].median())

    numcols['age'] = 2024 - numcols['year']
    numcols = numcols.drop(['year'], axis=1)

    objcols = data.select_dtypes(include=['object'])
    objcols = objcols.drop(['name', 'torque'], axis=1)
    objcols = pd.get_dummies(objcols, columns=['fuel', 'seller_type', 'transmission', 'owner'])

    scaler = StandardScaler()
    numcols_scaled = pd.DataFrame(scaler.fit_transform(numcols), columns=numcols.columns)

    final_data = pd.concat([numcols_scaled, objcols], axis=1)
    return final_data

# Streamlit app
def main():
    st.title("Car Price Prediction App")
    st.write("Upload a dataset to predict car prices")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if 'selling_price' in data.columns:
            st.write("Dataset Loaded. Training the model...")
            X = preprocess_data(data.drop(['selling_price'], axis=1))
            y = data['selling_price']

            model = LinearRegression()
            model.fit(X, np.log(y))  # Log transformation on target variable

            # Predictions for the training data
            y_pred = np.exp(model.predict(X))

            # Calculate RMSE for training data
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            st.write("Model trained successfully!")
            st.write(f"Training R-Square: {model.score(X, np.log(y)):.4f}")
            st.write(f"Training RMSE: {rmse:.4f}")

        else:
            st.error("Uploaded dataset must contain a 'selling_price' column for training.")

        # Allow user to upload a new dataset for predictions
        st.write("\nUpload another dataset for predictions:")
        prediction_file = st.file_uploader("Choose another CSV file for predictions", type="csv")

        if prediction_file is not None:
            new_data = pd.read_csv(prediction_file)

            if 'selling_price' in new_data.columns:
                # Separate the actual selling price for RMSE calculation
                actual_selling_price = new_data['selling_price']
                prediction_data = new_data.drop(['selling_price'], axis=1)
            else:
                st.warning("The uploaded dataset for predictions does not contain 'selling_price'. RMSE cannot be calculated.")
                prediction_data = new_data

            # Preprocess the data for predictions
            processed_data = preprocess_data(prediction_data)

            # Ensure the columns in processed_data match the ones used in the model training
            required_columns = model.feature_names_in_  # Get the columns used during training
            missing_cols = [col for col in required_columns if col not in processed_data.columns]
            for col in missing_cols:
                processed_data[col] = 0  # Add missing column with a default value (e.g., 0)

            # Reorder columns to match the model's training data
            processed_data = processed_data[required_columns]

            # Generate predictions
            predictions = np.exp(model.predict(processed_data))

            st.write("Predictions generated successfully!")
            st.write(pd.DataFrame({"Predicted Selling Price": predictions}))

            # Calculate RMSE if actual selling price is provided
            if 'selling_price' in new_data.columns:
                rmse_new = np.sqrt(mean_squared_error(actual_selling_price, predictions))
                st.write(f"New Dataset RMSE: {rmse_new:.4f}")

                # Calculate R-Squared for new dataset
                r2_new = r2_score(actual_selling_price, predictions)
                st.write(f"New Dataset R-Squared: {r2_new:.4f}")

if __name__ == "__main__":
    main()
