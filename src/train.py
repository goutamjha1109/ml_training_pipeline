from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml

from logger import logger
from data_loader import load_data, preprocess_data, save_transformations,save_splits

PROJECT_ROOT = Path(__file__).parent.parent

def load_params(params_path=None):
    if params_path is None:
        params_path = PROJECT_ROOT/ "config" / "params.yaml"
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def read_data(path):
    logger.info(f"Loading data from {path}")
    df = load_data(path)
    id_col, X, y, bundle = preprocess_data(df)
    save_transformations(bundle)
    logger.info("Transformations saved to models/transformations.pkl")
    logger.info(f"Data preprocessed. Shape: {X.shape}")
    return id_col, X, y, bundle


def split_data(X, y, test_size, random_state):
    logger.info("Splitting data into train/test sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    save_splits(X_train, X_test, y_train, y_test)
    logger.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, params):
    model = None
    try:
        logger.info("Starting training process.")
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)
        logger.info("Training completed successfully.")

        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to models/model.pkl.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")


if __name__ == "__main__":
    # Load params
    params = load_params()
    data_params = params["data"]
    model_params = params["model"]

    validation_params = params["validation_data"]

    DATA_PATH = PROJECT_ROOT / data_params["path"]
    PROCESSED_PATH = PROJECT_ROOT / validation_params["path"]


    id_col, X, y, bundle = read_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=data_params["test_size"],
        random_state=data_params["random_state"]
    )
    train(X_train, y_train, model_params)