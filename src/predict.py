import argparse
from typing import Tuple

import joblib
import numpy as np

from config import MODEL_PATH


def make_prediction(features: list[float]) -> Tuple[int, float]:
    """Generate a prediction and positive-class probability for one patient."""
    model = joblib.load(MODEL_PATH)
    array = np.asarray([features], dtype=float)
    proba = float(model.predict_proba(array)[0][1])
    label = int(proba >= 0.5)
    return label, proba


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single patient example."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a single prediction using the trained heart disease model.\n"
            "Features: age, sex, chest_pain, blood_pressure, cholesterol, "
            "max_hr, st_depression."
        )
    )

    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", type=int, required=True)
    parser.add_argument("--chest-pain", type=int, required=True, dest="chest_pain")
    parser.add_argument("--blood-pressure", type=float, required=True, dest="blood_pressure")
    parser.add_argument("--cholesterol", type=float, required=True)
    parser.add_argument("--max-hr", type=float, required=True, dest="max_hr")
    parser.add_argument("--st-depression", type=float, required=True, dest="st_depression")

    return parser.parse_args()


def main() -> None:
    """Entry point for running a CLI prediction."""
    args = parse_args()
    features = [
        args.age,
        args.sex,
        args.chest_pain,
        args.blood_pressure,
        args.cholesterol,
        args.max_hr,
        args.st_depression,
    ]

    label, proba = make_prediction(features)
    print(f"Predicted label: {label}")
    print(f"Positive class probability: {proba:.4f}")


if __name__ == "__main__":
    main()
