from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "heart-disease-advanced" / "data" / "heart-disease.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model_v1.pkl"
META_PATH = MODEL_DIR / "metadata.json"

TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_FOLDS = 5

LOG_REG_MAX_ITER = 1000
LOG_REG_C_GRID = [0.01, 0.1, 1.0, 10.0]
LOG_REG_PENALTY = "l2"
LOG_REG_SOLVER = "lbfgs"
