from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Heart Disease ML API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


class PatientData(BaseModel):
    age: float
    sex: int
    chest_pain: int
    blood_pressure: float
    cholesterol: float
    max_hr: float
    st_depression: float


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the trained model used for prediction."""
    try:
        return joblib.load("models/model_v1.pkl")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Trained model not found. Run training before serving predictions.",
        ) from exc


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Render the main HTML interface for heart disease risk assessment."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PatientData):
    features = np.array(
        [
            [
                data.age,
                data.sex,
                data.chest_pain,
                data.blood_pressure,
                data.cholesterol,
                data.max_hr,
                data.st_depression,
            ]
        ]
    )

    model = get_model()
    proba = model.predict_proba(features)[0][1]
    prediction = int(proba >= 0.5)

    return {
        "prediction": prediction,
        "probability": round(float(proba), 4),
        "risk_level": risk_bucket(proba),
    }


def risk_bucket(p: float) -> str:
    if p < 0.3:
        return "Low Risk"
    elif p < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"
