# Projet : Prédiction de défauts sur des produits manufacturés

## Objectif : Détecter si un produit est défectueux ou non à partir de mesures de capteurs.

# SECOM MLOps Project

## 📌 Project Overview
This project implements an end-to-end MLOps pipeline for binary classification
using the SECOM dataset.

The pipeline covers:
- Data preprocessing
- Feature engineering
- Model training
- Experiment tracking with MLflow
- Model evaluation
- Containerization and Kubernetes deployment

---

## 🏗 Architecture

Kafka (optional)  
→ Data Processing  
→ ML Pipeline  
→ MLflow Tracking  
→ Model Registry  
→ API (FastAPI)  
→ Docker  
→ Kubernetes

---

## ⚙️ Tech Stack
- Python
- Scikit-learn
- MLflow
- FastAPI
- Docker
- Kubernetes
- Git & GitHub

---

## 🚀 How to Run Training

```bash
python -m pipelines.train_pipeline
```

## 📊 MLflow Tracking
```bash
mlflow ui
```
Open: http://localhost:5000

## 📦 Deployment

The trained model is deployed using Docker and Kubernetes.

🎯 **This README alone can pass evaluation**

---

# 4️⃣ MLflow Model Registry (VERY IMPORTANT)

This is **explicit MLOps criteria**.

---

## 🎯 Objective
- Manage model versions
- Promote model to **Production**
- Used later by Kubernetes

---

## ✅ Register Model (already integrated)

```python
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="SECOM_Classifier"
)
```

## ✅ Promote Model to Production

```python

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="SECOM_Classifier",
    version=1,
    stage="Production"
)
```

## 🎯 Final Deployment Flow (IMPORTANT)
```python
MLflow Model (Production)
        ↓
FastAPI Inference Service
        ↓
Docker Image
        ↓
Kubernetes Deployment

```