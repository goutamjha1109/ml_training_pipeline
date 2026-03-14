from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

from logger import logger


DATA_PATH = "data/raw/telecom_churn.csv"


def read_data(path):
    df = load_data(path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return df, X, y



def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    try:
        logger.info("Starting training process.")
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        logger.info("Training completed successfully.")


        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to models/model.pkl.")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
    
    return model 
