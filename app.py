# app.py
import os

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# --------- USER: path to saved model (change if needed) ----------
MODEL_PATH = os.path.join("model", "house_price_model.pkl")
# -----------------------------------------------------------------

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "YearBuilt",
]

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Train model first and put file there.")
else:
    model = load(MODEL_PATH)

    st.header("Enter house features")
    # Provide sensible default values and ranges (you can adjust)
    overallqual = st.slider(
        "Overall Quality (OverallQual)", min_value=1, max_value=10, value=6
    )
    gr_liv_area = st.number_input(
        "Above grade (ground) living area square feet (GrLivArea)",
        value=1500,
        min_value=200,
    )
    total_bsmt_sf = st.number_input(
        "Total basement square feet (TotalBsmtSF)", value=800, min_value=0
    )
    garage_cars = st.slider(
        "Garage capacity (GarageCars)", min_value=0, max_value=4, value=1
    )
    full_bath = st.slider(
        "Full bathrooms (FullBath)", min_value=0, max_value=4, value=2
    )
    year_built = st.number_input(
        "Year built (YearBuilt)", value=1980, min_value=1850, max_value=2026
    )

    if st.button("Predict House Price"):
        input_df = pd.DataFrame(
            [
                [
                    overallqual,
                    gr_liv_area,
                    total_bsmt_sf,
                    garage_cars,
                    full_bath,
                    year_built,
                ]
            ],
            columns=FEATURES,
        )

        pred = model.predict(input_df)[0]
        st.success(f"Predicted Sale Price: ${pred:,.2f}")

        # Optional: show model input for transparency
        with st.expander("Show input data"):
            st.write(input_df)
