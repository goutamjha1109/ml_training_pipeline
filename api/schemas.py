from pydantic import BaseModel
from typing import List
 
class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    PaperlessBilling: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaymentMethod: str


class PredictionResponse(BaseModel):
    churn_probability:float
    churn_prediction:bool 

class BatchPrediction(BaseModel):
    customers: List[CustomerFeatures]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResponse]