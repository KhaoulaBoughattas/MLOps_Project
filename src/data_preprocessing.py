# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_secom(
    data_path="data/secom/secom.data",
    labels_path="data/secom/secom_labels.data",
    output_path="data/processed/clean_data.csv"
):

    # Assurer que le dossier processed existe
    os.makedirs("data/processed", exist_ok=True)

    # ----- 1. Lire les features -----
    # SECOM contient des espaces multiples → utiliser \s+
    df_features = pd.read_csv(data_path, sep=r"\s+", header=None)

    # Remplacer les "NaN" textuels par np.nan
    df_features = df_features.replace("NaN", np.nan)

    # Convertir en float
    df_features = df_features.astype(float)

    # ----- 2. Imputer les valeurs manquantes -----
    imputer = SimpleImputer(strategy='mean')
    df_features_imputed = pd.DataFrame(imputer.fit_transform(df_features))

    # ----- 3. Normalisation -----
    scaler = StandardScaler()
    df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features_imputed))

    # ----- 4. Lire les labels -----
    df_labels = pd.read_csv(labels_path, sep=r"\s+", header=None)

    # SECOM possède 2 colonnes: label + timestamp → garder juste label
    df_labels = df_labels.iloc[:, 0].rename("Y")

    # ----- 5. Fusion features + labels -----
    df = pd.concat([df_features_scaled, df_labels], axis=1)

    # ----- 6. Sauvegarder -----
    df.to_csv(output_path, index=False)
    print(f"✔ Fichier prétraité sauvegardé dans : {output_path}")

    return df


# ---------- Exécution directe ----------
if __name__ == "__main__":
    df = preprocess_secom()
    print("Shape :", df.shape)
    print(df.head())
