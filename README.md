# ML Leak Detection Service

A lightweight FastAPI service serving a trained scikit-learn `RandomForestClassifier` leak detection model.

## Features
- Lifespan model loading (loads `leak_model.pkl` once at startup).
- Graceful startup handling (logs error and exits cleanly if model is missing or corrupt).
- Input validation using Pydantic.
- `POST /predict` endpoint returning leak probability and prediction (`LEAK` | `NORMAL`).
- `GET /health` endpoint returning `{"status": "ok"}` when operational.

---

## Getting Started

### 1. Installation
Ensure Python 3.9+ is installed. Navigate to the `ml-service` directory and install dependencies:

```bash
cd ml-service
pip install -r requirements.txt
```

### 2. Running Locally (Development Mode)
To run the server with auto-reload:

```bash
uvicorn main:app --reload --port 8000
```

---

## API Endpoints

### 1. Health Check
- **GET** `/health`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Predict Leak Probability
- **POST** `/predict`
- **Payload**:
  ```json
  {
    "pressure": 45.2,
    "flow_rate": 12.8,
    "temperature": 65.0
  }
  ```
- **Response**:
  ```json
  {
    "leak_probability": 0.82,
    "prediction": "LEAK"
  }
  ```
