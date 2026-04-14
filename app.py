import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from predict import predict_multiple_days
from sqlalchemy import create_engine

# Database connection
DATABASE_URL = st.secrets["DATABASE_URL"]

engine = create_engine(DATABASE_URL)


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Vegetable Price Prediction",
    layout="wide"
)

st.title("🥦 Vegetable Price Prediction Dashboard")


# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("vegetable_prices_final.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_data()


# -----------------------------
# Sidebar Controls
# -----------------------------

vegetables = df["vegetable"].unique()

selected_veg = st.selectbox(
    "Select Vegetable",
    vegetables
)

days_to_predict = st.slider(
    "Select number of days to predict",
    min_value=1,
    max_value=30,
    value=7
)


# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Price"):

    with st.spinner("Predicting prices..."):

        preds = predict_multiple_days(
            df,
            selected_veg,
            days_to_predict
        )

        st.success("Prediction Completed!")


        # -------------------------
        # Create Prediction Table
        # -------------------------

        result_df = pd.DataFrame({
            "Day": list(range(1, days_to_predict + 1)),
            "Predicted Price": preds
        })

        st.subheader("Predicted Prices")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.dataframe(
                result_df,
                use_container_width=True
        )

        # -------------------------
        # Clean Green Chart Styling
        # -------------------------

        st.subheader(f"{selected_veg} Price Forecast")

        fig, ax = plt.subplots(figsize=(10, 5))

        # Remove background
        fig.patch.set_alpha(0)     # Figure background
        ax.set_facecolor("none")   # Plot background

        # Green prediction line
        ax.plot(
            result_df["Day"],
            result_df["Predicted Price"],
            color="green",
            marker="o",
            linewidth=2
        )

        # Labels
        ax.set_title(
            f"{selected_veg} Price Forecast",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel("Days Ahead", fontsize=12)
        ax.set_ylabel("Predicted Price (₹)", fontsize=12)

        # Light grid
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3
        )

        ax.tick_params(colors="white")

        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")

        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

        # Remove top/right borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.pyplot(fig)
