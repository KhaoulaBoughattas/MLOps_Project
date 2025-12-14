# src/train_model.py

import os
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# --- Config MLflow : fichier local mlruns compatible Windows ---
mlruns_path = os.path.abspath("mlruns").replace("\\", "/")
MLFLOW_TRACKING_URI = f"file:///{mlruns_path}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Créer ou utiliser un experiment nommé ---
EXPERIMENT_NAME = "SECOM_RF"
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception:
    # Si l'expérience n'existe pas, elle sera créée
    mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_model(
    data_path="data/features/engineered_data.csv",
    model_path="models/rf_model.pkl",
    model_type="randomforest",
    balance_method="smote",
    threshold=None
):
    print("=== TRAINING MODEL ===")

    # --- Charger les données ---
    df = pd.read_csv(data_path)
    if "Y" not in df.columns:
        raise ValueError("La colonne cible 'Y' est manquante")

    X = df.drop(columns=["Y"])
    y = df["Y"]

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- SMOTE multi-classes ---
    if balance_method == "smote":
        print("✔ SMOTE multi-classes appliqué")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # --- Sélection du modèle ---
    if model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced" if balance_method == "class_weight" else None
        )
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            eval_metric="mlogloss"
        )
    elif model_type == "lightgbm":
        model = LGBMClassifier(
            n_estimators=300,
            class_weight="balanced" if balance_method == "class_weight" else None
        )
    else:
        raise ValueError(f"Modèle inconnu : {model_type}")

    # --- MLflow run ---
    with mlflow.start_run(run_name=f"train_{model_type}"):
        # Log hyperparamètres
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("balance_method", balance_method)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])

        # --- Entraînement ---
        print("✔ Entraînement du modèle...")
        model.fit(X_train, y_train)

        # --- Prédictions sur test set pour MLflow ---
        y_pred_test = model.predict(X_test)
        f1_macro_test = f1_score(y_test, y_pred_test, average='macro')
        mlflow.log_metric("f1_macro_testset", f1_macro_test)
        print("\n=== F1-SCORE MACRO (test set) ===")
        print(f1_macro_test)
        print("\n=== CLASSIFICATION REPORT (test set) ===")
        print(classification_report(y_test, y_pred_test, digits=4))

        # --- Prédictions sur tout le dataset pour rapport final ---
        y_pred_all = model.predict(X)
        f1_macro_all = f1_score(y, y_pred_all, average='macro')
        mlflow.log_metric("f1_macro_full", f1_macro_all)
        print("\n=== F1-SCORE MACRO (full dataset) ===")
        print(f1_macro_all)
        print("\n=== CLASSIFICATION REPORT (full dataset) ===")
        print(classification_report(y, y_pred_all, digits=4))

        # --- Sauvegarde locale ---
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        features_path = model_path.replace(".pkl", "_features.pkl")
        joblib.dump(list(X.columns), features_path)

        # --- Log modèle et features dans MLflow ---
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(features_path)

        print("✔ Modèle et features sauvegardés")
        print("=== TRAINING DONE ===")

    return model


if __name__ == "__main__":
    train_model()
