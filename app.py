import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv('flights.csv')  # Ensure 'flights.csv' file is in the same directory

# Convert date_oftravel to datetime
df['date_oftravel'] = pd.to_datetime(df['date_oftravel'])

# Build the regression model
X_reg = df[["source", "destination", "flightType", "time", "distance", "agency", "date_oftravel"]]
y_reg = df["price"]

# One-hot encoding for categorical variables
categorical_features = ["source", "destination", "flightType", "agency"]
numeric_features = ["time", "distance"]

preprocessor_reg = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(sparse_output=False), categorical_features)  # Set sparse_output to False
    ]
)

# Create pipelines for regression models
linear_model = Pipeline(steps=[("preprocessor", preprocessor_reg), ("regressor", LinearRegression())])
tree_model = Pipeline(steps=[("preprocessor", preprocessor_reg), ("regressor", DecisionTreeRegressor())])

# Fit both models
linear_model.fit(X_reg, y_reg)
tree_model.fit(X_reg, y_reg)

# Calculate predictions and metrics for Linear Regression
linear_predictions = linear_model.predict(X_reg)
linear_mse = mean_squared_error(y_reg, linear_predictions)
linear_average_price = np.mean(linear_predictions)
linear_accuracy = r2_score(y_reg, linear_predictions) * 100

# Calculate predictions and metrics for Decision Tree
tree_predictions = tree_model.predict(X_reg)
tree_mse = mean_squared_error(y_reg, tree_predictions)
tree_average_price = np.mean(tree_predictions)
tree_accuracy = r2_score(y_reg, tree_predictions) * 100

# Build the classification model
# Create price categories
price_thresholds = [0, 500, 1000, 1500, 2000]
price_labels = [0, 1, 2, 3]  # Categories for <500, 500-1000, 1000-1500, >1500
df['price_category'] = pd.cut(df['price'], bins=price_thresholds, labels=price_labels, right=False)

X_class = df[["source", "destination", "flightType", "time", "distance", "agency", "date_oftravel"]]
y_class = df["price_category"]

# Build the classification pipeline
preprocessor_class = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(sparse_output=False), categorical_features)  # Set sparse_output to False
    ]
)

# Pipeline for Gaussian Naive Bayes classifier
nb_model = Pipeline(steps=[("preprocessor", preprocessor_class), ("classifier", GaussianNB())])
nb_model.fit(X_class, y_class)

# Streamlit UI
st.title("Flight Ticket Price Prediction and Classification")

# User inputs for regression
st.header("Predict Ticket Price")
source = st.selectbox("Select Source City", df["source"].unique())
destination = st.selectbox("Select Destination City", df["destination"].unique())
flight_type = st.selectbox("Select Flight Type", df["flightType"].unique())
time = st.number_input("Enter Time (hours)", min_value=0.0, step=0.5)
distance = st.number_input("Enter Distance (km)", min_value=0, step=10)
agency = st.selectbox("Select Travel Agency", df["agency"].unique())
date_of_travel = st.date_input("Select Date of Travel")

# Prepare the input data for prediction
input_data = {
    "source": source,
    "destination": destination,
    "flightType": flight_type,
    "time": time,
    "distance": distance,
    "agency": agency,
    "date_oftravel": pd.to_datetime(date_of_travel),
}
input_df = pd.DataFrame([input_data])

# Select model type for price prediction
model_type = st.selectbox("Select Model for Price Prediction", ["Linear Regression", "Decision Tree"])

# Predict price based on selected model
if st.button("Predict Price"):
    if model_type == "Linear Regression":
        price_prediction = linear_model.predict(input_df)
        mse = linear_mse
        average_price = linear_average_price
        accuracy = linear_accuracy
    else:
        price_prediction = tree_model.predict(input_df)
        mse = tree_mse
        average_price = tree_average_price
        accuracy = tree_accuracy
        
    st.success(f"The predicted ticket price is: ${price_prediction[0]:.2f}")

    # Show model metrics on prediction
    st.subheader("Model Evaluation Metrics")
    st.write(f"Average Price: ${average_price:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Model Accuracy: {accuracy:.2f}%")

# User inputs for classification
st.header("Classify Ticket Price Category")
if st.button("Classify Price Category"):
    # Use Naive Bayes for classification
    category_prediction = nb_model.predict(input_df)
    categories = {0: "<$500", 1: "$500-$1000", 2: "$1000-$1500", 3: ">$1500"}
    predicted_category = categories[category_prediction[0]]
    st.success(f"The predicted price category is: {predicted_category}")

# Show data for insights
if st.button("Show Insights"):
    st.subheader("Dataset Insights")
    st.write(df.describe())
