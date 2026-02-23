import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from config import META_PATH, MODEL_PATH
from features import load_data, split_features_target


def evaluate_model() -> None:
    """Evaluate the trained model on the full dataset and print metrics."""
    df = load_data()
    X, y = split_features_target(df)

    model = joblib.load(MODEL_PATH)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc = roc_auc_score(y, y_proba)
    report = classification_report(y, y_pred)

    print(f"ROC-AUC (full dataset): {auc:.4f}")
    print(report)

    meta_path = Path(META_PATH)
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("\nSaved training metadata:")
        print(json.dumps(metadata, indent=4))


def main() -> None:
    """Entry point for running model evaluation from the command line."""
    evaluate_model()


if __name__ == "__main__":
    main()
