from fastapi import FastAPI, HTTPException
from pathlib import Path
from contextlib import asynccontextmanager
import yaml

from api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPrediction,
    BatchPredictResponse
)

from api.predictor import ChurnPredictor

def load_params(path = "config/params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
    
predictor = None

@asynccontextmanager   
async def lifespan(app:FastAPI):
    global predictor
    params = load_params()
    artifacts_dir = Path(params["paths"]["artifacts"])
    predictor = ChurnPredictor(artifacts_dir)
    yield

app = FastAPI(
    title= "Telecom Churn Predictor API",
    version= "1.0.0",
    lifespan=lifespan
)


@app.get("/health") 
def health(): 
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer : CustomerFeatures):
    try:
        result = predictor.predict_single(customer.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request : BatchPrediction):
    try:
        results = predictor.predict_batch(request.model_dump())
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)