import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent



def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "conf.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


# Column registry
COLUMN_REGISTRY = load_config()

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    numerical_cols = COLUMN_REGISTRY["numerical"]
    # binary_cols = COLUMN_REGISTRY["binary_categorical"]
    # multi_categorical_cols = COLUMN_REGISTRY["multi_categorical"]

    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    id_col = df[COLUMN_REGISTRY["id"]].copy()
    df.drop(columns=COLUMN_REGISTRY["id"], inplace=True)

    # Fit and save label encoders per binary column
    label_encoders = {}
    for col in COLUMN_REGISTRY["binary_categorical"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # save fitted encoder

    # Encode target separately
    le_target = LabelEncoder()
    df["Churn"] = le_target.fit_transform(df["Churn"])
    label_encoders["Churn"] = le_target

    # One-hot encode
    df = pd.get_dummies(df, columns=COLUMN_REGISTRY["multi_categorical"], drop_first=True)

    # Save the final column order (critical for prediction alignment)
    feature_columns = df.drop("Churn", axis=1).columns.tolist()

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Bundle everything needed for prediction
    transformation_bundle = {
        "label_encoders": label_encoders,   # fitted LabelEncoders
        "feature_columns": feature_columns, # column order after get_dummies
        "column_registry": COLUMN_REGISTRY, # column type registry
    }

    return id_col, X, y, transformation_bundle


def save_transformations(bundle, path=None):
    if path is None:
        path = PROJECT_ROOT / "models" / "transformations.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_transformations(path=None):
    if path is None:
        path = PROJECT_ROOT / "models" / "transformations.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def save_splits(X_train, X_test, y_train, y_test, path=None):
    if path is None:
        path = PROJECT_ROOT / "data" / "telecom_churn" / "processed"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    pickle.dump(X_train, open(path / "X_train.pkl", "wb"))
    pickle.dump(X_test,  open(path / "X_test.pkl",  "wb"))
    pickle.dump(y_train, open(path / "y_train.pkl", "wb"))
    pickle.dump(y_test,  open(path / "y_test.pkl",  "wb"))


def load_splits(path=None):
    if path is None:
        path = PROJECT_ROOT / "data" / "telecom_churn" / "processed"
    path = Path(path)

    X_train = pickle.load(open(path / "X_train.pkl", "rb"))
    X_test  = pickle.load(open(path / "X_test.pkl",  "rb"))
    y_train = pickle.load(open(path / "y_train.pkl", "rb"))
    y_test  = pickle.load(open(path / "y_test.pkl",  "rb"))

    return X_train, X_test, y_train, y_test