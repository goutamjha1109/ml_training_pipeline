import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from logger import logger
from dataclasses import dataclass


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    le_target: LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df, schema):
    for col in schema["numerical"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    le_target = LabelEncoder()
    y = le_target.fit_transform(df[schema["target"][0]])
    y = pd.Series(y, name=schema["target"][0])

    X = df.drop(columns=schema["id"] + schema["target"])
    return X, y, le_target


def load_and_split_data(path: Path, schema: dict, params: dict) -> DataSplit:
    test_size    = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    df = load_data(path)
    X, y, le_target = clean_data(df, schema)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return DataSplit(X_train, X_test, y_train, y_test, le_target)


def build_preprocessor(schema):
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", schema["numerical"]),
        ("bin", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), schema["binary_categorical"]),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False
        ), schema["multi_categorical"]),
    ], remainder="drop")
    return preprocessor


def save_pipeline(pipeline, le_target, artifacts_dir):
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    pickle.dump(pipeline, open(path / "pipeline.pkl", "wb"))
    pickle.dump(le_target, open(path / "le_target.pkl", "wb"))
    logger.info(f"Pipeline saved to {path / 'pipeline.pkl'}")
    logger.info(f"Target encoder saved to {path / 'le_target.pkl'}")


def load_pipeline(artifacts_dir):
    logger.info("Loading artifacts from saved location...")
    try:
        artifacts_dir = Path(artifacts_dir)
        pipeline  = pickle.load(open(artifacts_dir / "pipeline.pkl", "rb"))
        le_target = pickle.load(open(artifacts_dir / "le_target.pkl", "rb"))
        logger.info("Pipeline and target encoder loaded successfully.")
        return pipeline, le_target
    except FileNotFoundError as e:
        logger.error(f"Artifact file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        raise


def save_splits(X_train, X_test, y_train, y_test, path):
    logger.info("Started data save process......")
    path.mkdir(parents=True, exist_ok=True)
    pickle.dump(X_train, open(path / "X_train.pkl", "wb"))
    pickle.dump(X_test,  open(path / "X_test.pkl",  "wb"))
    pickle.dump(y_train, open(path / "y_train.pkl", "wb"))
    pickle.dump(y_test,  open(path / "y_test.pkl",  "wb"))
    logger.info("Data dumping successful ......")


def load_splits(path):
    path = Path(path)
    X_train = pickle.load(open(path / "X_train.pkl", "rb"))
    X_test  = pickle.load(open(path / "X_test.pkl",  "rb"))
    y_train = pickle.load(open(path / "y_train.pkl", "rb"))
    y_test  = pickle.load(open(path / "y_test.pkl",  "rb"))
    return X_train, X_test, y_train, y_test