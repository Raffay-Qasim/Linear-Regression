import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Function to perform Linear Regression
def perform_linear_regression(data, x_column, y_column):
    # Prepare data
    X = data[[x_column]]
    y = data[y_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Return results
    return model.coef_, model.intercept_, mse, r2, y_pred, y_test

# Streamlit Interface
st.title("Linear Regression on CSV Data")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Show the first few rows of the dataset
    st.write("Data preview:", data.head())

    # Select columns for X and Y
    x_column = st.selectbox("Select the independent variable (X)", data.columns)
    y_column = st.selectbox("Select the dependent variable (Y)", data.columns)

    # Perform linear regression when the button is pressed
    if st.button("Run Linear Regression"):
        if x_column and y_column:
            coef, intercept, mse, r2, y_pred, y_test = perform_linear_regression(data, x_column, y_column)

            # Display the results
            st.write(f"Linear Regression Results:")
            st.write(f"Coefficient (Slope): {coef[0]}")
            st.write(f"Intercept: {intercept}")
            st.write(f"Mean Squared Error (MSE): {mse}")
            st.write(f"R-squared: {r2}")

            # Display prediction vs actual values
            results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.write("Predictions vs Actual values:", results)
