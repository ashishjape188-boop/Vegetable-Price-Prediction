import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Vegetable Price Prediction",
    layout="wide"
)

st.title("🥦 Vegetable Price Prediction Dashboard")

# -----------------------------
# LOAD FILES
# -----------------------------
model = joblib.load("vegetable_price_model.pkl")
encoder = joblib.load("veg_label_encoder.pkl")
model_features = joblib.load("model_features.pkl")


# Load dataset
df = pd.read_csv("vegetable_features_ready.csv")
st.write(df.shape)

# Fix date parsing
df["date"] = pd.to_datetime(
    df["date"],
    format="%Y-%m-%d",
    errors="coerce"
)

df = df.dropna(subset=["date"])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_multiple_days(veg_name, days):

    veg_code = encoder.transform([veg_name])[0]

    veg_df = df[df["vegetable"] == veg_name].copy()

    veg_df = veg_df.sort_values("date")

    last_rows = veg_df.tail(30).copy()

    predictions = []

    current_date = last_rows["date"].iloc[-1]

    for i in range(days):

        next_date = current_date + timedelta(days=1)

        # Safe lag handling
        lag_1 = last_rows["price"].iloc[-1]

        lag_2 = last_rows["price"].iloc[-2] if len(last_rows) >= 2 else lag_1
        lag_3 = last_rows["price"].iloc[-3] if len(last_rows) >= 3 else lag_1

        lag_7 = (
            last_rows["price"].iloc[-7]
            if len(last_rows) >= 7
            else lag_1
        )

        lag_14 = (
            last_rows["price"].iloc[-14]
            if len(last_rows) >= 14
            else lag_1
        )

        lag_21 = (
            last_rows["price"].iloc[-21]
            if len(last_rows) >= 21
            else lag_1
        )

        lag_30 = (
            last_rows["price"].iloc[-30]
            if len(last_rows) >= 30
            else lag_1
        )

        # Create input dictionary
        # Create input dictionary
        input_dict = {
        
            "veg_code": veg_code,
        
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_21": lag_21,
            "lag_30": lag_30,
        
            "day": next_date.day,
            "month": next_date.month,
            "weekday": next_date.weekday(),
        
            "rolling_mean_7": (
                last_rows["price"].tail(7).mean()
                if len(last_rows) >= 7
                else lag_1
            ),
        
            "rolling_std_7": (
                last_rows["price"].tail(7).std()
                if len(last_rows) >= 7
                else 0
            ),
        
            "price_diff": lag_1 - lag_2
        }

        # Convert to dataframe
        input_data = pd.DataFrame([input_dict])

        input_data = input_data.fillna(0)
        
        input_data = input_data[model_features]

        # Predict
        raw_pred = model.predict(input_data)[0]

        st.write("Raw prediction:", raw_pred)
        
        # Use raw value for recursion
        new_row = {
            "date": next_date,
            "price": raw_pred
        }
        
        last_rows = pd.concat(
            [last_rows, pd.DataFrame([new_row])],
            ignore_index=True
        )
        
        last_rows = last_rows.tail(30)
        
        current_date = next_date
        
        # Clip only for display
        display_price = max(1, min(raw_pred, 500))
        
        predictions.append(float(display_price))

        last_rows = pd.concat(
            [last_rows, pd.DataFrame([new_row])],
            ignore_index=True
        )

        last_rows = last_rows.tail(30)

        current_date = next_date

    return predictions


# -----------------------------
# UI INPUTS
# -----------------------------
vegetables = sorted(df["vegetable"].unique())

veg_name = st.selectbox(
    "Select Vegetable",
    vegetables
)

days = st.slider(
    "Select number of days to predict",
    min_value=1,
    max_value=30,
    value=7
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Price"):

    preds = predict_multiple_days(
        veg_name,
        days
    )

    st.success("Prediction Completed!")

    # Show values
    result_df = pd.DataFrame({
        "Day": list(range(1, days + 1)),
        "Predicted Price": preds
    })

    st.dataframe(result_df)

    # Plot chart
    fig, ax = plt.subplots()

    ax.plot(
        result_df["Day"],
        result_df["Predicted Price"]
    )

    ax.set_title(
        f"{veg_name} Price Forecast"
    )

    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Price")

    st.pyplot(fig)
