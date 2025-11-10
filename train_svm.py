import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
import joblib

# Load training data
train = pd.read_json("train_data.jsonlines-kopi", lines=True)
y = train["condition"]  # target labels ("new"/"used")

# Build TF-IDF features from 'title'
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(train["title"].fillna(""))


def build_structured_features(df: pd.DataFrame, fit_encoder: bool = True, ohe: OneHotEncoder = None, scaler: StandardScaler = None) -> tuple:
    """
    Extract relevant structured features and one-hot encode categorical columns.

    Parameters:
    - df: DataFrame containing the data.
    - fit_encoder: Boolean indicating whether to fit new encoders/scalers.
    - ohe: Pre-fitted OneHotEncoder (if fit_encoder is False).
    - scaler: Pre-fitted StandardScaler (if fit_encoder is False).  

    Returns:
    - X_struct: Sparse matrix of structured features.
    - ohe: Fitted OneHotEncoder (if fit_encoder is True).
    - scaler: Fitted StandardScaler (if fit_encoder is True).
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

    # Separate categorical and numeric columns
    cat_cols = base.select_dtypes("object").columns
    base[cat_cols] = base[cat_cols].fillna("missing")

    # One-hot encode categorical columns
    if fit_encoder:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        cat_encoded = ohe.fit_transform(base[cat_cols])
    else:
        cat_encoded = ohe.transform(base[cat_cols])

    # Normalize numeric columns
    num = base.drop(columns=cat_cols).select_dtypes(include=[np.number])
    if fit_encoder:
        scaler = StandardScaler(with_mean=False)
        num_scaled = scaler.fit_transform(num)
    else:
        num_scaled = scaler.transform(num)

    X_struct = hstack([csr_matrix(num_scaled), cat_encoded])
    return X_struct, ohe, scaler

# Build structured features
X_struct, ohe, scaler = build_structured_features(train)

# Combine text + structured features
X_final = hstack([X_text, X_struct])

# Train model
X_tr, X_val, y_tr, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

model = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)
model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")

# Save model and preprocessing objects
joblib.dump(model, "final_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(ohe, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved successfully!")