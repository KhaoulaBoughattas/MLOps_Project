# src/evaluate.py
import pandas as pd
from sklearn.metrics import classification_report
import joblib
import os

def evaluate(data_path="data/features/engineered_data.csv",
             model_path="models/rf_model.pkl",
             output_path="reports/classification_report.txt"):

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    X = df.drop(columns=["Y"])
    y = df["Y"]
    y_pred = model.predict(X)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(classification_report(y, y_pred))

    print(f"✔ Rapport de classification sauvegardé dans : {output_path}")


if __name__ == "__main__":
    evaluate()
