# pipelines/train_pipeline.py

import os
import sys

# --- Ajouter le chemin racine du projet pour que 'src' soit trouvé ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess_secom
from src.feature_engineering import add_features
from src.train_model import train_model
from src.predict import predict
from src.evaluate import evaluate

def full_ml_pipeline():
    print("=== PIPELINE COMPLET MLOPS ===")

    # --- Créer les dossiers si manquants ---
    for folder in ["data/processed", "data/features", "data/predictions", "models", "reports"]:
        os.makedirs(folder, exist_ok=True)

    # 1️⃣ Prétraitement
    print("\n--- Étape 1 : Prétraitement ---")
    preprocess_secom(
        data_path="data/secom/secom.data",
        labels_path="data/secom/secom_labels.data",
        output_path="data/processed/clean_data.csv"
    )

    # 2️⃣ Feature Engineering
    print("\n--- Étape 2 : Feature Engineering ---")
    add_features(
        input_path="data/processed/clean_data.csv",
        output_path="data/features/engineered_data.csv"
    )

    # 3️⃣ Entraînement
    print("\n--- Étape 3 : Entraînement ---")
    train_model(
        data_path="data/features/engineered_data.csv",
        model_path="models/rf_model.pkl",
        model_type="randomforest",
        balance_method="smote"
    )

    # 4️⃣ Prédictions
    print("\n--- Étape 4 : Prédictions ---")
    predict(
        data_path="data/features/engineered_data.csv",
        model_path="models/rf_model.pkl",
        output_path="data/predictions/preds.csv"
    )

    # 5️⃣ Évaluation
    print("\n--- Étape 5 : Évaluation ---")
    evaluate(
        data_path="data/features/engineered_data.csv",
        preds_path="data/predictions/preds.csv",
        report_path="reports/classification_report.txt",
        cm_path="reports/confusion_matrix.png"
    )

    print("\n=== PIPELINE TERMINÉ ===")


if __name__ == "__main__":
    full_ml_pipeline()
