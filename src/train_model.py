# =========================================================
# File: src/train_model.py
# Purpose: Train ML model + track with MLflow + register model
# =========================================================

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

# =========================================================
# MLflow configuration (LOCAL, Windows compatible)
# =========================================================
mlruns_path = os.path.abspath("mlruns").replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{mlruns_path}")

EXPERIMENT_NAME = "SECOM_RF"
mlflow.set_experiment(EXPERIMENT_NAME)

REGISTERED_MODEL_NAME = "SECOM_Classifier"

# =========================================================
# Training function
# =========================================================
def train_model(
    data_path="data/features/engineered_data.csv",
    model_path="models/rf_model.pkl",
    model_type="randomforest",
    balance_method="smote"
):
    print("\n==============================")
    print("🚀 START TRAINING")
    print("==============================")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = pd.read_csv(data_path)

    if "Y" not in df.columns:
        raise ValueError("❌ Target column 'Y' is missing")

    X = df.drop(columns=["Y"])
    y = df["Y"]

    # -----------------------------------------------------
    # Train / Test split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------------------------------
    # Handle class imbalance
    # -----------------------------------------------------
    if balance_method == "smote":
        print("✔ Applying SMOTE on training data")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------
    if model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=300,
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
        raise ValueError(f"❌ Unknown model type: {model_type}")

    # -----------------------------------------------------
    # MLflow Run
    # -----------------------------------------------------
    with mlflow.start_run(run_name=f"{model_type}_training"):

        # ---- Log parameters
        mlflow.log_params({
            "model_type": model_type,
            "balance_method": balance_method,
            "n_features": X.shape[1],
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0]
        })

        # ---- Train model
        print("✔ Training model...")
        model.fit(X_train, y_train)

        # -------------------------------------------------
        # Evaluation on TEST SET
        # -------------------------------------------------
        y_pred_test = model.predict(X_test)
        f1_macro_test = f1_score(y_test, y_pred_test, average="macro")

        mlflow.log_metric("f1_macro_test", f1_macro_test)

        print("\n📊 TEST SET RESULTS")
        print(f"F1 Macro (test): {f1_macro_test:.4f}")
        print(classification_report(y_test, y_pred_test, digits=4))

        # -------------------------------------------------
        # Evaluation on FULL DATASET
        # -------------------------------------------------
        y_pred_all = model.predict(X)
        f1_macro_full = f1_score(y, y_pred_all, average="macro")

        mlflow.log_metric("f1_macro_full", f1_macro_full)

        print("\n📊 FULL DATASET RESULTS")
        print(f"F1 Macro (full): {f1_macro_full:.4f}")
        print(classification_report(y, y_pred_all, digits=4))

        # -------------------------------------------------
        # Save model locally
        # -------------------------------------------------
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        features_path = model_path.replace(".pkl", "_features.pkl")
        joblib.dump(list(X.columns), features_path)

        # -------------------------------------------------
        # Log model + register in MLflow Model Registry
        # -------------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        mlflow.log_artifact(features_path)

        print("\n✅ Model logged to MLflow")
        print(f"✅ Registered as: {REGISTERED_MODEL_NAME}")
        print("==============================")
        print("🎉 TRAINING COMPLETED")
        print("==============================")

    return model


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    train_model()
