# src/feature_engineering.py

import pandas as pd
import numpy as np
import os

def add_features(input_path="data/processed/clean_data.csv",
                 output_path="data/features/engineered_data.csv"):

    # Charger les données pré-traitées
    df = pd.read_csv(input_path)

    # Exemple simple : features statistiques
    df["feature_mean"] = df.iloc[:, :-1].mean(axis=1)
    df["feature_std"] = df.iloc[:, :-1].std(axis=1)

    # Créer le dossier de sortie si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder le fichier final dans le dossier attendu par DVC
    df.to_csv(output_path, index=False)

    print(f"✔ Feature engineering terminé. Fichier généré : {output_path}")
    return df


if __name__ == "__main__":
    add_features()
