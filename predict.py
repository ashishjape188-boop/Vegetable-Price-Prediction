import pandas as pd
import joblib
from datetime import timedelta

# Load artifacts
model = joblib.load("vegetable_price_model.pkl")
model_features = joblib.load("model_features.pkl")
le = joblib.load("label_encoder.pkl")


def predict_multiple_days(df, vegetable, days):

    veg_code = le.transform([vegetable])[0]

    df_veg = df[df["vegetable"] == vegetable].copy()
    df_veg = df_veg.sort_values("date")

    last_rows = df_veg.tail(30).copy()

    current_date = last_rows["date"].iloc[-1]

    predictions = []

    for i in range(days):

        next_date = current_date + timedelta(days=1)

        # LAG FEATURES
        lag_1 = last_rows["price"].iloc[-1]
        lag_2 = last_rows["price"].iloc[-2]
        lag_3 = last_rows["price"].iloc[-3]
        lag_7 = last_rows["price"].iloc[-7]
        lag_14 = last_rows["price"].iloc[-14]
        lag_21 = last_rows["price"].iloc[-21]
        lag_30 = last_rows["price"].iloc[-30]

        rolling_mean_7 = last_rows["price"].tail(7).mean()
        rolling_std_7 = last_rows["price"].tail(7).std()

        price_diff = lag_1 - lag_2

        input_dict = {
            "veg_code": veg_code,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_21": lag_21,
            "lag_30": lag_30,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "price_diff": price_diff,
            "day": next_date.day,
            "month": next_date.month,
            "weekday": next_date.weekday(),
        }

        input_df = pd.DataFrame([input_dict])

        input_df = input_df[model_features]

        pred_price = model.predict(input_df)[0]

        predictions.append(float(pred_price))

        new_row = pd.DataFrame({
            "date": [next_date],
            "price": [pred_price]
        })

        last_rows = pd.concat([last_rows, new_row], ignore_index=True)

        current_date = next_date

    return predictions