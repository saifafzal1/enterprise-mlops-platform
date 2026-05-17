import os, mlflow, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

DATA_PATH  = "data/raw/heart_disease.csv"
MODEL_PATH = "models/heart_disease.pkl"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_data():
    df = pd.read_csv(DATA_PATH, names=COLUMNS, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train():
    mlflow.set_experiment("heart-disease")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf.fit(X_train_sc, y_train)

        preds = clf.predict(X_test_sc)
        proba = clf.predict_proba(X_test_sc)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1":       round(f1_score(y_test, preds), 4),
            "roc_auc":  round(roc_auc_score(y_test, proba), 4),
        }
        mlflow.log_params({"model": "RandomForest", "n_estimators": 100, "max_depth": 8})
        mlflow.log_metrics(metrics)

        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": clf, "scaler": scaler}, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"Saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
