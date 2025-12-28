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
MLRUNS_PATH = os.getenv("MLRUNS_PATH", "/app/mlruns")
EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "419088383897925458")

mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

# =====================================================
# Locate MLflow Model Artifacts
# =====================================================
models_root = os.path.join(MLRUNS_PATH, EXPERIMENT_ID, "models")

if not os.path.exists(models_root):
    raise FileNotFoundError(f"Models directory not found: {models_root}")

model_dirs = sorted(glob.glob(os.path.join(models_root, "*")), reverse=True)

MODEL_PATH = None
for model_dir in model_dirs:
    artifacts_path = os.path.join(model_dir, "artifacts")
    if os.path.exists(os.path.join(artifacts_path, "MLmodel")):
        MODEL_PATH = artifacts_path
        break

if MODEL_PATH is None:
    raise FileNotFoundError("❌ No valid MLflow model found")

print(f"✅ Using MLflow model at: {MODEL_PATH}")

# =====================================================
# Load Model
# =====================================================
def load_model():
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print("✅ Model successfully loaded")
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

model = load_model()

# =====================================================
# FastAPI App
# =====================================================
app = FastAPI(
    title="SECOM Classifier API",
    description="MLflow + FastAPI inference service",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def health_check():
    return {"status": "running", "model_path": MODEL_PATH}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        expected_features = len(model.feature_names_in_)

        if len(data.features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_features} features, got {len(data.features)}"
            )

        X = pd.DataFrame([data.features], columns=model.feature_names_in_)
        prediction = model.predict(X)

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
