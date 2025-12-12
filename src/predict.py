import pandas as pd
import joblib
import os
import sys

def predict(
    data_path="data/features/engineered_data.csv",
    model_path="models/rf_model.pkl",
    output_path="data/predictions/preds.csv",
    threshold=None  # seuil optionnel uniquement pour modèles binaires
):

    # --- Vérifications ---
    if not os.path.exists(model_path):
        print(f"❌ Modèle '{model_path}' introuvable.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"❌ Fichier de données '{data_path}' introuvable.")
        sys.exit(1)

    print("⏳ Chargement des données...")
    df = pd.read_csv(data_path)

    print("⏳ Chargement du modèle...")
    model = joblib.load(model_path)

    # --- Chargement des features ---
    features_path = model_path.replace(".pkl", "_features.pkl")
    if not os.path.exists(features_path):
        print(f"❌ Fichier des features '{features_path}' introuvable.")
        sys.exit(1)
    features = joblib.load(features_path)

    # Vérification que toutes les features existent
    missing = [f for f in features if f not in df.columns]
    if missing:
        print("❌ Certaines features manquent dans les données :")
        for m in missing:
            print(f"   - {m}")
        sys.exit(1)

    X = df[features].copy()
    print("⏳ Génération des prédictions...")

    # --- Cas multi-classes ou binaire ---
    n_classes = len(getattr(model, "classes_", []))
    if n_classes > 2:
        preds = model.predict(X)
    else:
        if hasattr(model, "predict_proba") and threshold is not None:
            proba = model.predict_proba(X)[:, 1]
            preds = (proba >= threshold).astype(int)
        else:
            preds = model.predict(X)

    # --- Sauvegarde ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
    print(f"✔ Prédictions sauvegardées dans : {output_path}")
    print(f"✔ Nombre total : {len(preds)}")
    print("✔ Distribution des classes :")
    print(pd.Series(preds).value_counts())

if __name__ == "__main__":
    predict()
