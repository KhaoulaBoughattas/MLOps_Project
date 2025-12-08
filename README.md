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
