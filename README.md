# Projet : Prédiction de défauts sur des produits manufacturés

## Objectif : Détecter si un produit est défectueux ou non à partir de mesures de capteurs.

Dataset suggéré : SECOM Manufacturing Data
 (UCI ML Repository)

Contient des mesures de capteurs pour la production de semi-conducteurs.

Objectif binaire : 1 = défectueux, 0 = correct.

Taille : 1567 échantillons, 590 caractéristiques → gérable pour un projet MLOps.

## Pourquoi c’est parfait pour MLOps

### Préprocessing & Feature Engineering

Nettoyage des valeurs manquantes, normalisation des features, PCA éventuellement.

### Modélisation

Modèles simples : Random Forest, XGBoost, ou même un petit réseau de neurones.

### Versioning des données et du modèle

Utilisation de DVC (Data Version Control) ou MLflow pour tracker les versions.

### Pipeline automatisé

Création d’un pipeline avec Airflow ou Prefect pour orchestrer ingestion → entraînement → validation → déploiement.

### Déploiement

Déployer le modèle avec FastAPI + Docker, ou un modèle cloud comme AWS Sagemaker / Azure ML / GCP AI Platform.

### Monitoring

Monitoring du modèle en production pour détecter le drift des données ou baisse de performance.

### Tests et CI/CD

Tests unitaires pour le code ML, intégration dans un pipeline CI/CD (GitHub Actions).

### Extension “cool” si tu veux aller plus loin

Ajouter un dashboard en temps réel pour visualiser le nombre de prédictions et la performance du modèle avec Streamlit ou Dash.

Ajouter un retraining automatique si la performance descend sous un seuil.

## Structurer ton projet pour MLOps

Avant d’ajouter les outils, il faut que le projet soit bien structuré :

src/ → scripts de preprocessing, features, training, predict, evaluate

data/ → raw, processed, features, predictions

models/ → stockage des modèles

reports/ → résultats, rapports, métriques

pipelines/ → pipeline central

dvc.yaml et dvc.lock → suivi des datasets et modèles avec DVC

requirements.txt ou environment.yml → dépendances

💡 Avec DVC, tu pourras versionner datasets et modèles comme du code.


Étape 2 : Ajouter le suivi d’expérimentation avec MLflow

Installer MLflow : pip install mlflow

Transformer ton train_model.py pour :

Logger les hyperparamètres

Logger le modèle entraîné

Logger métriques comme f1-score, accuracy, etc.

Tu pourras ensuite :

Comparer différentes expériences

Reproduire les runs facilement

Exemple : mlflow.start_run(), mlflow.log_param(), mlflow.log_metric(), mlflow.sklearn.log_model()

Étape 3 : Transformer le pipeline en DAG

Actuellement, ton pipeline est linéaire. Tu peux :

Utiliser Prefect, Airflow ou Kubeflow Pipelines

Chaque étape devient une tâche du DAG : preprocessing → features → training → predict → evaluate

Permet le re-run partiel, la planification et le monitoring

Étape 4 : Conteneurisation avec Docker

Créer un Dockerfile pour ton projet :

Installer Python, dépendances, DVC, MLflow

Copier le code et les données nécessaires

Définir un entrypoint pour exécuter ton pipeline

Construire l’image :

docker build -t mlops_project:latest .
docker run -it mlops_project:latest


Avantage : tu pourras déployer le pipeline partout, même sur Kubernetes.

Étape 5 : Orchestration avec Kubernetes

Déployer ton pipeline dans un cluster Kubernetes :

Créer un pod ou job pour le pipeline

Optionnel : utiliser Prefect Orion/Kubernetes agent ou Kubeflow Pipelines

Bénéfice : scalabilité, parallélisation, monitoring via dashboard K8s

Étape 6 : Automatisation & CI/CD

Ajouter GitHub Actions ou GitLab CI/CD :

Tester le pipeline à chaque commit

Pousser les modèles vers un stockage cloud

Déclencher des runs MLflow automatiquement

Étape 7 : Monitoring & alerting

Utiliser MLflow UI pour les métriques et comparaison

Ajouter prometheus + grafana pour :

Surveillance des performances du modèle en production

Alertes sur dérive de données ou drop de métriques

Étape 8 : Déploiement du modèle

Tu peux transformer ton predict.py en API REST :

Avec FastAPI ou Flask

Dockeriser l’API

Déployer sur Kubernetes pour inference en production

Bonus : ajouter un endpoint pour batch prediction ou retrain automatique

💡 Résumé du workflow final MLOps :

raw data → preprocessing → feature engineering → train → predict → evaluate → log metrics (MLflow)
         ↓
   DVC versioning
         ↓
Docker container → deploy on Kubernetes
         ↓
Monitoring & alerting (Grafana/Prometheus)
         ↓
CI/CD pipeline pour automatisation