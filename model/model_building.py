# /model/model_development.py
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------- USER: set dataset path here ----------
DATA_PATH = os.path.join(
    "..", "data", "house_prices.csv"
)  # if executed from /model/ ; adjust if running from repo root
# If you run script from project root, change to: DATA_PATH = "data/house_prices.csv"
# ------------------------------------------------

# Features chosen (6) and target
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "YearBuilt",
]
TARGET = "SalePrice"


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_data(df):
    # Keep only selected features + target
    df = df[FEATURES + [TARGET]].copy()
    # Basic cleaning: drop rows where target missing
    df = df.dropna(subset=[TARGET])
    return df


def train_and_save(df, model_path="house_price_model.pkl"):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: imputer -> scaler -> RandomForest
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    # Train
    num_pipeline.fit(X_train, y_train)

    # Evaluate
    preds = num_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("Evaluation on test set:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.4f}")

    # Save model pipeline (includes imputer & scaler)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(num_pipeline, model_path)
    print(f"Saved model pipeline to: {model_path}")

    return num_pipeline, (mae, mse, rmse, r2)


if __name__ == "__main__":
    # If running from repo root, change DATA_PATH accordingly
    # Example: python model/model_development.py
    # Adjust DATA_PATH at top if required
    df = load_data(DATA_PATH)
    df = prepare_data(df)
    model_file = os.path.join(
        "..", "model", "house_price_model.pkl"
    )  # path relative to /model/ when script executed there
    # If you're running from project root: model_file = "model/house_price_model.pkl"
    # Ensure the correct path before running.
    # For simplicity, if running from project root:
    #    set DATA_PATH = "data/house_prices.csv" and model_file = "model/house_price_model.pkl"
    # Train and save
    train_and_save(df, model_path=model_file)
