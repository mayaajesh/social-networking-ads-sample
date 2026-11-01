
import streamlit as st
import pandas as pd
import pickle

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Purchase Prediction",
    page_icon="🛍️",
    layout="centered"
)

st.title("Customer Purchase Prediction System")

st.write("Predict whether a customer will purchase a product.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input(
    "Age", min_value=18, max_value=100, value=30
)

salary = st.number_input(
    "Estimated Salary", min_value=0, value=50000
)

# Prediction
if st.button("Predict Purchase"):

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "EstimatedSalary": [salary]
    })

    # Encode gender
    input_data = pd.get_dummies(
        input_data, columns=["Gender"]
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns, fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success("The customer is likely to purchase!")
    else:
        st.warning("The customer is unlikely to purchase.")

    st.write("Purchase Probability:", round(probability * 100, 2), "%")
