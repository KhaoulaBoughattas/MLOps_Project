import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(
    data_path="data/features/engineered_data.csv",
    preds_path="data/predictions/preds.csv",
    report_path="reports/classification_report.txt",
    cm_path="reports/confusion_matrix.png"
):

    # --- Vérifications ---
    for path, name in [(data_path, "données"), (preds_path, "prédictions")]:
        if not os.path.exists(path):
            print(f"❌ Fichier {name} '{path}' introuvable.")
            sys.exit(1)

    # --- Chargement des données et prédictions ---
    df = pd.read_csv(data_path)
    if "Y" not in df.columns:
        print("❌ La colonne cible 'Y' est manquante.")
        sys.exit(1)
    y_true = df["Y"]

    preds_df = pd.read_csv(preds_path)
    if "prediction" not in preds_df.columns:
        print("❌ La colonne 'prediction' est manquante dans le fichier de prédictions.")
        sys.exit(1)
    y_pred = preds_df["prediction"]

    # --- Assurer que y_true et y_pred ont les mêmes classes ---
    all_labels = sorted(list(set(y_true) | set(y_pred)))

    # --- Création dossier reports ---
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # --- Classification report ---
    report = classification_report(y_true, y_pred, labels=all_labels, digits=4, zero_division=0)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✔ Rapport de classification sauvegardé dans : {report_path}")
    print(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    plt.close()
    print(f"✔ Confusion matrix sauvegardée dans : {cm_path}")

if __name__ == "__main__":
    evaluate()
