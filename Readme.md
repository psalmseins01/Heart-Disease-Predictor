<h1 align="center">Predictive Cardiology – Heart Disease Risk Studio</h1>

This project is an end‑to‑end heart disease risk assessment application built around an advanced logistic regression pipeline, exposed via a FastAPI backend and a professional web UI written in vanilla HTML, CSS, and JavaScript.

The goal is to take structured cardiovascular features for a patient and estimate the probability of heart disease in a way that is explainable, reproducible, and deployable.

---

## System Diagram

The following diagram summarizes the overall system flow, from data to deployment:

![System Diagram](System%20Diagram.png)

---

## Features

- **Logistic regression pipeline**
  - Standardized features and hyperparameter tuning with cross‑validation (GridSearchCV).
  - Optimized for ROC‑AUC.
- **Reproducible training and evaluation**
  - Centralized configuration for paths, splits, and hyperparameters.
  - Scripts for training, evaluation, and CLI‑based prediction.
- **FastAPI inference service**
  - `/health` endpoint for monitoring.
  - `/predict` endpoint for JSON‑based inference.
  - `GET /` serving a professional, responsive web dashboard.
- **Vanilla JS Web UI**
  - Clean two‑panel layout: explainer + patient profile form and results.
  - Client‑side validation and concise risk explanation.
- **Deployment‑ready**
  - Dockerfile for containerization.
  - Documented Render (cloud) deployment from GitHub.

---

## Project Structure

```text
.
├── app/
│   ├── main.py               # FastAPI app and routes
│   ├── templates/
│   │   └── index.html        # Web UI template
│   └── static/
│       ├── style.css         # UI styling
│       └── script.js         # Vanilla JS client logic
├── src/
│   ├── config.py             # Central configuration and paths
│   ├── features.py           # Data loading and feature/target splitting
│   ├── pipeline.py           # Training pipeline and GridSearchCV
│   ├── train.py              # Training entry‑point
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # CLI prediction script
├── models/
│   ├── model_v1.pkl          # Trained logistic regression model
│   └── metadata.json         # Training metadata and metrics
├── notebooks/
│   └── EDA.ipynb             # Exploratory data analysis (optional)
├── heart-disease-advanced/
│   └── data/
│       └── heart-disease.csv # Source dataset
├── requirements.txt          # Python dependencies
├── Dockerfile                # Containerization
├── ENDPOINTS.md              # Endpoint usage (ignored in Git)
├── DEPLOY_RENDER.md          # Render deployment guide (ignored in Git)
└── ai_history.json           # AI assistant change log
```

---

## Requirements

All Python dependencies are defined in:

- [`requirements.txt`](requirements.txt)

Install them into your active virtual environment with:

```bash
pip install -r requirements.txt
```

This includes:

- `pandas` (>=1.5,<3)
- `numpy`
- `scikit-learn`
- `fastapi`, `uvicorn`
- `pydantic`
- `jinja2`

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/psalmseins01/Predictive-Cardiology.git
cd Predictive-Cardiology
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

This will:

- Load the dataset.
- Perform train/validation split and cross‑validation.
- Save the best model to `models/model_v1.pkl`.
- Save training metadata to `models/metadata.json`.

### 5. (Optional) Evaluate the model

```bash
python src/evaluate.py
```

### 6. Run the API

```bash
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/` – Web UI.
- `http://127.0.0.1:8000/health` – Health check.
- `http://127.0.0.1:8000/docs` – Interactive OpenAPI docs.

---

## API Overview

- `GET /health`
  - Returns a simple JSON payload indicating the service is healthy.
- `GET /`
  - Serves the heart disease risk dashboard UI.
- `POST /predict`
  - Accepts a JSON body with numeric fields such as:
    - `age`, `sex`, `chest_pain`, `blood_pressure`,
      `cholesterol`, `max_hr`, `st_depression`.
  - Returns:
    - `prediction` (0/1),
    - `probability` (0–1),
    - `risk_level` (“Low Risk”, “Moderate Risk”, “High Risk”).

For detailed examples using cURL, Postman, and the browser UI, see:

- `ENDPOINTS.md`

---

## Deployment

You can deploy this project in multiple ways:

- **Docker**
  - Build and run the included Dockerfile to containerize the FastAPI service.
- **Render (GitHub → cloud)**
  - A step‑by‑step Render deployment guide is available in:
    - `DEPLOY_RENDER.md`

In a typical Render setup:

- Build command:

  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```

- Start command:

  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```

---

## License

You can add the license of your choice here (for example, MIT, Apache‑2.0, or a custom license), depending on how you intend others to use this project.
