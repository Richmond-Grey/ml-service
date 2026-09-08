import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define Model Path (strictly use the model inside the ml-service directory)
MODEL_PATH = Path(__file__).resolve().parent / "leak_model.pkl"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Attempting to load model from {MODEL_PATH.resolve()}...")
    if not MODEL_PATH.exists():
        logger.error(f"CRITICAL: Model file '{MODEL_PATH}' does not exist.")
        sys.exit(1)
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model file '{MODEL_PATH}'. Error: {e}")
        sys.exit(1)
    
    yield


app = FastAPI(title="Leak Detection ML Service", lifespan=lifespan)


class PredictRequest(BaseModel):
    pressure: float = Field(..., description="Pressure value")
    flow_rate: float = Field(..., description="Flow rate value")
    temperature: float = Field(..., description="Temperature value")


class PredictResponse(BaseModel):
    leak_probability: float
    prediction: str


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Features in order: pressure, flow_rate, temperature
        features = np.array([[data.pressure, data.flow_rate, data.temperature]])
        probabilities = model.predict_proba(features)
        
        # Binary classification assumption: proba for class 1 (LEAK) is at index 1
        # If model outputs 1D array or binary class probabilities
        if probabilities.shape[1] >= 2:
            leak_prob = float(probabilities[0][1])
        else:
            leak_prob = float(probabilities[0][0])
            
        prediction = "LEAK" if leak_prob > 0.5 else "NORMAL"
        
        return PredictResponse(
            leak_probability=round(leak_prob, 4),
            prediction=prediction
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
