from pathlib import Path
from typing import Tuple

import pandas as pd

from config import DATA_PATH

FEATURE_COLUMNS = [
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "max_hr",
    "st_depr",
]

TARGET_COLUMN = "heart_disease"


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the heart disease dataset and apply basic preprocessing."""
    df = pd.read_csv(path)
    if df["sex"].dtype == object:
        df["sex"] = df["sex"].replace({"male": 0, "female": 1})
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split an input dataframe into feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
