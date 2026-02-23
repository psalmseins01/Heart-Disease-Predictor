import json
from datetime import datetime

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CV_FOLDS,
    LOG_REG_C_GRID,
    LOG_REG_MAX_ITER,
    LOG_REG_PENALTY,
    LOG_REG_SOLVER,
    META_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from features import load_data, split_features_target


def build_pipeline() -> Pipeline:
    """Create the preprocessing and logistic regression pipeline."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=LOG_REG_MAX_ITER),
            ),
        ]
    )
    return pipeline


def train() -> None:
    """Train the logistic regression model and persist artifacts to disk."""
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()

    param_grid = {
        "classifier__C": LOG_REG_C_GRID,
        "classifier__penalty": [LOG_REG_PENALTY],
        "classifier__solver": [LOG_REG_SOLVER],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV_FOLDS,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print("Best Parameters:", grid.best_params_)
    print("ROC-AUC:", auc)
    print(classification_report(y_test, best_model.predict(X_test)))

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "trained_at": str(datetime.now()),
        "roc_auc": auc,
        "best_params": grid.best_params_,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    train()
