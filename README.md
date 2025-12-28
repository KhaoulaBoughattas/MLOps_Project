# SECOM MLOps Project: Defect Prediction for Manufactured Products

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-20.10-blue)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-brightgreen)](https://github.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.7.0-orange)](https://mlflow.org/)

---

## 🌟 Project Objective
Detect whether a product is defective or not using sensor measurements collected from manufacturing processes.  
Binary classification task using the **SECOM dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom).

---

## 🧩 Dataset
- **Source:** SECOM dataset, UCI ML Repository  
- **Features:** 590 sensor measurements  
- **Samples:** 1567 products (rows)  
- **Target:** `Pass/Fail` label indicating defect status  
- **Challenges:** Missing values, noisy features → requires preprocessing and feature selection  

---

## 🏗 Project Architecture
### ASCII Overview

Data (SECOM CSV)
↓
Data Preprocessing & Feature Engineering
↓
ML Pipeline (Train/Test Split, Model Training)
↓
MLflow Tracking & Experiment Logging
↓
MLflow Model Registry
↓
FastAPI Inference API
↓
Docker Container
↓
Kubernetes Deployment


### Mermaid Flow Diagram

```mermaid
flowchart LR
    A[SECOM Dataset] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[ML Pipeline]
    D --> E[MLflow Experiment Tracking]
    E --> F[MLflow Model Registry]
    F --> G[FastAPI Inference API]
    G --> H[Docker Container]
    H --> I[Kubernetes Deployment]


---

## ⚙️ Tech Stack
- **Python** 3.10+
- **Data Science & ML:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM  
- **MLOps:** MLflow, Docker, Kubernetes  
- **API:** FastAPI, Uvicorn  
- **CI/CD:** GitHub Actions  
- **Version Control:** Git, GitHub  

---

## 🛠 End-to-End Pipeline

1. **Data Preprocessing**
   - Handle missing values and outliers  
   - Feature scaling and normalization  
   - Feature selection  

2. **Model Training**
   - Train multiple classifiers (Random Forest, XGBoost, LightGBM)  
   - Hyperparameter tuning  
   - Cross-validation  

3. **Experiment Tracking**
   - Log metrics, parameters, and artifacts in MLflow  
   - Compare multiple runs  

4. **Model Registry**
   - Register the best-performing model in MLflow  
   - Promote model to **Production stage**  

5. **API Deployment**
   - Serve predictions using FastAPI  
   - Input JSON → Output prediction  

6. **Containerization**
   - Dockerize API and model  
   - Push Docker image to GitHub Container Registry or DockerHub  

7. **Kubernetes Deployment**
   - Deploy Docker container as a scalable Kubernetes service  

---

## 🔄 CI/CD Pipeline

**GitHub Actions Workflow:** `.github/workflows/ci-cd.yml`
- **Trigger:** Push or pull request on `main` branch  
- **Steps:**
  1. Checkout repository  
  2. Set up Python 3.10  
  3. Install dependencies (`requirements.txt`)  
  4. Run tests (unit tests for API & pipeline)  
  5. Build Docker image  
  6. Push Docker image to GitHub Container Registry  
  7. Deploy to Kubernetes cluster (optional: production)  

**Mermaid CI/CD Diagram:**
flowchart TD
    A[GitHub Push/PR] --> B[GitHub Actions CI/CD]
    B --> C[Run Tests]
    C --> D[Build Docker Image]
    D --> E[Push Docker Image to Registry]
    E --> F[Kubernetes Deployment]
    F --> G[Live API Service]



## 🚀 How to Run Locally
1️⃣ Train Model
python -m pipelines.train_pipeline

2️⃣ Launch MLflow UI
mlflow ui


Access: http://localhost:5000

3️⃣ Run FastAPI Server
uvicorn api.main:app --reload

4️⃣ Test API
POST http://127.0.0.1:8000/predict
Body:
{
  "features": [val1, val2, ..., valN]
}

📦 Docker Deployment
# Build Docker image
docker build -t secom-api:latest -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 secom-api:latest

☸️ Kubernetes Deployment
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml


Access API through Kubernetes Service on port 8000.

✅ MLflow Model Registry Example
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="SECOM_Classifier"
)

# Promote to Production
client.transition_model_version_stage(
    name="SECOM_Classifier",
    version=1,
    stage="Production"
)

🎯 Final Deployment Flow
flowchart LR
    A[MLflow Model (Production)] --> B[FastAPI Inference Service]
    B --> C[Docker Container]
    C --> D[Kubernetes Deployment]


📌 Key Highlights

Fully automated MLOps pipeline

CI/CD with GitHub Actions

Real-time inference API via FastAPI

Scalable deployment with Docker + Kubernetes

MLflow experiment tracking & model registry

This README demonstrates a complete end-to-end MLOps workflow suitable for production evaluation.
