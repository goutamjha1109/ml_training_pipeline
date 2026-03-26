import pickle
import pandas as pd
from pathlib import Path 
from src.logger import logger
from typing import List


class ChurnPredictor:
    def __init__(self, artifact_dir: Path):
        self.pipeline , self.le_target = self._load_artifacts(artifact_dir)
        pass

    def _load_artifacts(self, artifact_dir:Path):
        try:
            pipeline = pickle.load(open(artifact_dir / "pipeline.pkl", "rb"))
            le_target = pickle.load(open(artifact_dir / "le_target.pkl", "rb"))
            logger.info(f"artifacts loaded successfully")
            return pipeline, le_target
        except FileNotFoundError as e:
            logger.error(f"Artifact not found {e}")
            raise

    def _to_dataframe(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame([data])
    
    def predict_single(self, data:dict) -> dict:
        df = self._to_dataframe(data)
        probability = float(self.pipeline.predict_proba_(df)[0][1])
        predicted_index = 1 if probability >= 0.5 else 0
        predicted_label = self.le_target.inverse_transform([predicted_index])[0]
        return {
            "churn_probability" : probability,
            "churn_prediction" : predicted_label 
        }
    
    def predict_batch(self, data: List[dict]) -> dict:
        df = pd.DataFrame(data["customers"])
        probabilities = self.pipeline.predict_proba(df)[:,-1]
        predicted_index = [1 if prob > 0.5 else 0 for prob in probabilities]
        return [{
            "churn_probability" : round(float(prob), 4),
            "churn_prediction" : self.le_target.inverse_transform([index])[0]

            
        } for (prob, index) in zip(probabilities, predicted_index)]