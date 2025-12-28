from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import glob

# =====================================================
# MLflow Configuration
# =====================================================
MLFLOW_ARTIFACTS_PATH = r"C:\Users\bough\OneDrive\Bureau\MLOps_Project\mlartifacts"
mlflow.set_tracking_uri(f"file:///{MLFLOW_ARTIFACTS_PATH.replace(os.sep, '/')}")

# =====================================================
# Dynamically find all MLflow model artifacts
# =====================================================
loaded_models = []
model_ids = []

# Search for MLflow model artifacts under mlartifacts folder
model_dirs = glob.glob(os.path.join(MLFLOW_ARTIFACTS_PATH, "0", "models", "*", "artifacts"))

if not model_dirs:
    raise FileNotFoundError(f"No MLflow model artifacts found in {MLFLOW_ARTIFACTS_PATH}")

for path in model_dirs:
    mlmodel_file = os.path.join(path, "MLmodel")
    if os.path.exists(mlmodel_file):
        model = mlflow.sklearn.load_model(path)
        loaded_models.append(model)
        model_ids.append(os.path.basename(os.path.dirname(path)))
        print(f"✅ Model loaded from: {path}")

# =====================================================
# FastAPI App
# =====================================================
app = FastAPI(
    title="SECOM Classifier API",
    description="MLflow + FastAPI inference service",
    version="1.0.0"
)

# Input data schema
class PredictionInput(BaseModel):
    features: list[float]

# Health check endpoint
@app.get("/")
def health_check():
    return {
        "status": "running",
        "loaded_models": model_ids
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        predictions = []

        for model in loaded_models:
            expected_features = len(model.feature_names_in_)
            
            if len(data.features) != expected_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {expected_features} features, got {len(data.features)}"
                )
            
            X = pd.DataFrame([data.features], columns=model.feature_names_in_)
            prediction = model.predict(X)
            predictions.append(int(prediction[0]))
        
        return {
            "predictions": predictions,
            "model_ids": model_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
