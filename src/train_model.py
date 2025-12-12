# src/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import os

def train_model(data_path="data/features/engineered_data.csv",
                model_path="models/rf_model.pkl"):
    """
    Entraîne un modèle RandomForest sur le dataset fourni et sauvegarde le modèle.
    Utilise le dataset avec les features déjà générées (feature engineering).
    """

    # Charger le dataset
    df = pd.read_csv(data_path)

    # Vérifier que la colonne cible existe
    if "Y" not in df.columns:
        raise ValueError("La colonne cible 'Y' est manquante dans le dataset")

    # Séparer X et y
    X = df.drop(columns=["Y"])
    y = df["Y"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Créer et entraîner le modèle
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prédictions et score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"✔ F1-score: {f1}")

    # Sauvegarde du modèle et de ses colonnes d'entrée
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✔ Modèle sauvegardé dans {model_path}")

    # Sauvegarder les noms des features pour prédiction future
    features_path = model_path.replace(".pkl", "_features.pkl")
    joblib.dump(list(X.columns), features_path)
    print(f"✔ Liste des features sauvegardée dans {features_path}")

    return model


if __name__ == "__main__":
    train_model()
