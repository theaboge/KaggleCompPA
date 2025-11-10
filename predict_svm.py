import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib

# Load test data
test = pd.read_json("test_data.jsonlines", lines=True)

# Load saved preprocessing objects and model
model = joblib.load("final_model.pkl")
tfidf = joblib.load("tfidf.pkl")
ohe = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# --- Build structured features (same as training) ---
def build_structured_features(df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler) -> csr_matrix:
    """
    Build structured features from the dataframe using provided encoders/scalers.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - ohe (OneHotEncoder): Pre-fitted OneHotEncoder.
    - scaler (StandardScaler): Pre-fitted StandardScaler.

    Returns:
    - csr_matrix: Structured feature matrix.
    """
    base = pd.DataFrame({
        "price": df["price"].fillna(0),
        "base_price": df["base_price"].fillna(0),
        "sold_quantity": df["sold_quantity"].fillna(0),
        "available_quantity": df["available_quantity"].fillna(0),
        "accepts_mercadopago": df["accepts_mercadopago"].astype(int),
        "listing_type_id": df["listing_type_id"],
        "buying_mode": df["buying_mode"],
        "category_id": df["category_id"],
    })

    cat_cols = base.select_dtypes("object").columns
    base[cat_cols] = base[cat_cols].fillna("missing")

    # Apply one-hot encoding and scaling
    cat_encoded = ohe.transform(base[cat_cols])
    num = base.drop(columns=cat_cols).select_dtypes(include=[np.number])
    num_scaled = scaler.transform(num)

    # Combine numeric and categorical parts
    X_struct = hstack([csr_matrix(num_scaled), cat_encoded])
    return X_struct


# --- Text features ---
test["title"] = test["title"].fillna("")
X_text = tfidf.transform(test["title"])

# --- Structured features ---
X_struct = build_structured_features(test, ohe, scaler)

# --- Combine all features ---
X_final = hstack([X_text, X_struct])

# --- Predict ---
test_pred = model.predict(X_final)

# --- Prepare submission ---
submission = pd.DataFrame({
    "ID": np.arange(1, len(test_pred) + 1),
    "condition": test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully!")
