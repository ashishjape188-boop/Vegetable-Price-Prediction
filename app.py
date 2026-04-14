# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sqlalchemy import create_engine
import datetime

# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="Vegetable Price Predictor",
    layout="centered"
)

st.title("🥕 Vegetable Price Prediction App")

st.write(
    "Predict next-day vegetable prices using Machine Learning."
)

# -------------------------------
# LOAD MODEL FILES
# -------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("vegetable_price_model.pkl")

    features = joblib.load("model_features.pkl")

    encoder = joblib.load("veg_label_encoder.pkl")

    data = pd.read_csv("vegetable_features_ready.csv")

    data["date"] = pd.to_datetime(data["date"])

    return model, features, encoder, data


model, features, encoder, df = load_model()

# -------------------------------
# USER INPUT
# -------------------------------

vegetables = sorted(df["vegetable"].unique())

veg_name = st.selectbox(
    "Select Vegetable",
    vegetables
)

# -------------------------------
# GET LATEST DATA
# -------------------------------

veg_df = df[df["vegetable"] == veg_name]

veg_df = veg_df.sort_values("date")

latest_row = veg_df.iloc[-1].copy()

# Encode vegetable
veg_code = encoder.transform([veg_name])[0]

latest_row["veg_code"] = veg_code

# -------------------------------
# DATE INPUT
# -------------------------------

selected_date = st.date_input(
    "Select Prediction Date",
    datetime.today()
)

latest_row["day"] = selected_date.day
latest_row["month"] = selected_date.month
latest_row["weekday"] = selected_date.weekday()

# -------------------------------
# PREDICT BUTTON
# -------------------------------

if st.button("Predict Price"):

    try:

        input_data = pd.DataFrame(
            [latest_row[features]]
        )

        prediction = model.predict(input_data)[0]

        st.success(
            f"💰 Predicted Price of {veg_name}: ₹ {prediction:.2f}"
        )

    except Exception as e:

        st.error("Prediction failed.")
        st.write(e)


# -------------------------------
# SHOW LATEST DATA
# -------------------------------

with st.expander("Show Latest Data"):

    st.dataframe(
        veg_df.tail(10)
    )
