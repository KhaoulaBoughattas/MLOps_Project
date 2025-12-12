# src/predict.py

import pandas as pd
import joblib
import os
import sys

def predict(data_path="data/features/engineered_data.csv",
            model_path="models/rf_model.pkl",
            output_path="data/predictions/preds.csv"):

    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"Erreur : le fichier du modèle '{model_path}' est introuvable.")
        sys.exit(1)

    # Vérifier que les données existent
    if not os.path.exists(data_path):
        print(f"Erreur : le fichier des données '{data_path}' est introuvable.")
        sys.exit(1)

    # Charger les données
    df = pd.read_csv(data_path)

    # Charger le modèle
    model = joblib.load(model_path)

    # Séparer features
    if "Y" in df.columns:
        X = df.drop(columns=["Y"])
    else:
        X = df

    # Faire la prédiction
    preds = model.predict(X)

    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder les prédictions
    pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)

    print(f"✔ Prédictions sauvegardées dans : {output_path}")


if __name__ == "__main__":
    predict()
