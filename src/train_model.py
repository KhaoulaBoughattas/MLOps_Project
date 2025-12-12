# src/train_model.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")


def train_model(
    data_path="data/features/engineered_data.csv",
    model_path="models/rf_model.pkl",
    model_type="randomforest",
    balance_method="smote",   # ⭐ recommandé pour SECOM
    threshold=None
):

    print("=== TRAINING MODEL ===")

    df = pd.read_csv(data_path)
    if "Y" not in df.columns:
        raise ValueError("La colonne cible 'Y' est manquante")

    X = df.drop(columns=["Y"])
    y = df["Y"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE multi-classes
    if balance_method == "smote":
        print("✔ SMOTE multi-classes appliqué")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # MODELS
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

    print("✔ Entraînement...")
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    print("\n=== F1-SCORE MACRO ===")
    print(f1_score(y_test, y_pred, average='macro'))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, digits=4))

    # Sauvegarde
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    features_path = model_path.replace(".pkl", "_features.pkl")
    joblib.dump(list(X.columns), features_path)

    print("✔ Modèle et features sauvegardés")
    print("=== TRAINING DONE ===")

    return model


if __name__ == "__main__":
    train_model()
