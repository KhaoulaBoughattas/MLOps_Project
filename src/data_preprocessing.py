# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_secom(data_path="C:/Users/bough/OneDrive/Bureau/MLOps_Project/data/secom/secom.data",
                     labels_path="C:/Users/bough/OneDrive/Bureau/MLOps_Project/data/secom/secom_labels.data"):


    # Lire les features (SECOM utilise des espaces multiples)
    df_features = pd.read_csv(data_path, sep=r"\s+", header=None)

    # Remplacer les chaînes "NaN" (texte) par np.nan
    df_features = df_features.replace("NaN", np.nan)

    # Convertir en float
    df_features = df_features.astype(float)

    # Imputer les valeurs manquantes avec la moyenne
    imputer = SimpleImputer(strategy='mean')
    df_features_imputed = pd.DataFrame(imputer.fit_transform(df_features))

    # Normalisation
    scaler = StandardScaler()
    df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features_imputed))

    # Lire les labels
    df_labels = pd.read_csv(labels_path, sep=r"\s+", header=None)
    df_labels.columns = ['Y']

    # Fusion features + labels
    df = pd.concat([df_features_scaled, df_labels], axis=1)

    return df


if __name__ == "__main__":
    df = preprocess_secom()
    print("Shape :", df.shape)
    print(df.head())
