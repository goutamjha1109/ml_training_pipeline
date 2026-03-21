from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import os
from logger import logger
from data_loader import load_data, preprocess_data, save_transformations,save_splits
from utils import get_train_args, load_schema, load_params
PROJECT_ROOT = Path(__file__).parent.parent

def load_params(params_path):
    # if params_path is None:
    #     params_path = PROJECT_ROOT/ "config" / "params.yaml"
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def read_data(path, save_path, schema):
    logger.info(f"Loading data from {path}")
    df = load_data(path)
    id_col, X, y, bundle = preprocess_data(df, schema=schema)
    save_transformations(bundle, save_path)
    logger.info("Transformations saved to models/transformations.pkl")
    logger.info(f"Data preprocessed. Shape: {X.shape}")
    return id_col, X, y, bundle


def split_data(X, y, test_size,save_path, random_state):
    logger.info("Splitting data into train/test sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    save_splits(X_train, X_test, y_train, y_test, save_path)
    logger.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, params):
    model = None
    try:
        model_path = Path(params["artifacts_path"]) / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Starting training process.")
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)
        logger.info("Training completed successfully.")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to models/model.pkl.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")


if __name__ == "__main__":
    # Load params
    args = get_train_args()
    schema = load_schema(args.schema)
    params = load_params(args.params)
    data_params = params["data"]
    model_params = params["model"]

    validation_params = params["validation_data"]

    DATA_PATH = data_params["path"]
    PROCESSED_PATH = validation_params["path"]
    ARTIFACTS_PATH = model_params["artifacts_path"]
    # DATA_PATH = PROJECT_ROOT / data_params["path"]
    # PROCESSED_PATH = PROJECT_ROOT / validation_params["path"]


    id_col, X, y, bundle = read_data(DATA_PATH, ARTIFACTS_PATH, schema)
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=data_params["test_size"],
        save_path=PROCESSED_PATH,
        random_state=data_params["random_state"],
        
    )
    train(X_train, y_train, model_params)