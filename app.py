import streamlit as st
import joblib
import pandas as pd
from sqlalchemy import create_engine
import datetime

# Load saved files
model = joblib.load("vegetable_price_rf_model.pkl")
le = joblib.load("veg_label_encoder.pkl")
features = joblib.load("model_features.pkl")
accuracy_data = joblib.load("model_accuracy.pkl")

# Load dataset
df = pd.read_csv("vegetable_prices_final.csv")

df["date"] = pd.to_datetime(
    df["date"],
    format="%d-%m-%Y"
)

# Prediction function
def predict_multiple_days(
        vegetable,
        days_ahead):

    predictions = []

    veg_df = df[
        df["vegetable"] == vegetable
    ].sort_values("date")

    last_row = veg_df.iloc[-1]

    lag_1  = last_row["price"]
    lag_7  = veg_df.iloc[-7]["price"]
    lag_14 = veg_df.iloc[-14]["price"]
    lag_30 = veg_df.iloc[-30]["price"]

    day      = last_row["date"].day
    month    = last_row["date"].month
    weekday  = last_row["date"].weekday()

    veg_code = le.transform([vegetable])[0]

    for i in range(days_ahead):

        input_data = pd.DataFrame([[
            veg_code,
            lag_1,
            lag_7,
            lag_14,
            lag_30,
            day,
            month,
            weekday
        ]], columns=features)

        predicted_price = model.predict(
            input_data
        )[0]

        predicted_price = float(
            round(predicted_price, 2)
        )

        predictions.append(predicted_price)

        lag_30 = lag_14
        lag_14 = lag_7
        lag_7  = lag_1
        lag_1  = predicted_price

    return predictions


# ---------------- UI ----------------

st.title("🥦 Vegetable Price Prediction Dashboard")

vegetables = list(le.classes_)

selected_veg = st.selectbox(
    "Select Vegetable",
    vegetables
)

days = st.slider(
    "Select number of days to predict",
    1,
    30,
    7
)

if st.button("Predict Price"):

    preds = predict_multiple_days(
        selected_veg,
        days
    )

    st.success("Prediction Complete!")

    st.subheader("Predicted Prices")

    st.write(preds)

    st.subheader("Model Accuracy")

    st.write(
        f"{accuracy_data['test_accuracy']:.2f}%"
    )

    chart_df = pd.DataFrame({
        "Day": list(range(1, days+1)),
        "Predicted Price": preds
    })

    st.line_chart(
        chart_df.set_index("Day")
    )
