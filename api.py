"""
FastAPI backend for the EV Battery End-of-Life (EOL) classifier.
Exposes standard prediction, health check, and random telemetry data fetching.
"""

import logging
import os
import random
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eol_api")

MODEL_PATH = "models/rf_pipeline.pkl"
DATASET_PATH = "data/dataset.csv"

model_store: dict = {"model": None, "load_error": None}


def _diagnose_load_error(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, AttributeError) and "_RemainderColsList" in msg:
        return (
            f"Failed to load '{MODEL_PATH}': this pickle was created with an "
            "older scikit-learn (1.6.1) and the ColumnTransformer it contains "
            f"cannot be unpickled by the scikit-learn version installed here "
            f"({sklearn.__version__}). This is a known scikit-learn regression "
            "(github.com/scikit-learn/scikit-learn issue #32090). Fix by either: "
            "(1) installing scikit-learn==1.6.1 in this environment, or "
            "(2) re-running train_sklearn.py to regenerate the pickle."
        )
    if isinstance(exc, FileNotFoundError):
        return f"Could not find {MODEL_PATH}. Run train_sklearn.py first."
    return f"Failed to load '{MODEL_PATH}': {type(exc).__name__}: {msg}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once
    try:
        model_store["model"] = joblib.load(MODEL_PATH)
        model_store["load_error"] = None
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as exc:
        diagnosis = _diagnose_load_error(exc)
        logger.error(diagnosis)
        model_store["model"] = None
        model_store["load_error"] = diagnosis
    yield


app = FastAPI(title="EV Battery EOL Diagnostics API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eol-diagnostics.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatteryTelemetry(BaseModel):
    battery_id: str = Field(default="BAT-2024-0457")
    car_model: Literal[
        "Chevy Bolt EV",
        "BYD Blade EV",
        "Model Y Performance",
        "Model 3 Long Range",
        "Tesla Model S",
    ] = "Tesla Model S"
    battery_type: Literal["LFP", "NMC", "NCA"] = "NMC"
    charging_speed: Literal["Slow (AC)", "Fast (DC)", "Ultra-Fast (HPC)"] = "Fast (DC)"
    climate: Literal["Cold", "Temperate", "Hot"] = "Temperate"
    depth_of_discharge: Literal[
        "Deep (0-100%)", "Shallow (20-80%)", "Moderate (10-90%)"
    ] = "Moderate (10-90%)"
    cycle_index: int = Field(default=850, ge=0)
    calendar_age_months: float = Field(default=18.5, ge=0)
    mileage_km: float = Field(default=45000.0, ge=0)
    soh_percentage: float = Field(default=91.5, ge=0, le=100)
    capacity_Ah: float = Field(default=74.5, gt=0)
    voltage_V: float = Field(default=354.0, gt=0)
    temperature_C: float = Field(default=25.0)
    internal_resistance_ohm: float = Field(default=0.061, gt=0)
    energy_throughput_kwh: float = Field(default=16500.0, ge=0)


class PredictionResponse(BaseModel):
    battery_id: str
    prediction: int
    eol_reached: bool
    status_message: str
    confidence_pct: Optional[float] = None


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model_store["model"] is not None else "degraded",
        "model_loaded": model_store["model"] is not None,
        "load_error": model_store["load_error"],
    }


# ✅ NEW ENDPOINT: Fetches a real sample row from your CSV dataset
@app.get("/random-battery")
def get_random_battery_row():
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found at '{DATASET_PATH}'. Please ensure data folder is mapped.",
        )
    try:
        df = pd.read_csv(DATASET_PATH)
        # Grab a random row and convert to dictionary format
        random_row = df.sample(n=1).to_dict(orient="records")[0]
        return random_row
    except Exception as exc:
        logger.exception("Failed to sample data row")
        raise HTTPException(
            status_code=500, detail=f"Dataset sample operation failed: {exc}"
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: BatteryTelemetry):
    model = model_store["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_store["load_error"] or "Model registry unavailable.",
        )

    input_df = pd.DataFrame([payload.model_dump()])

    try:
        prediction = int(model.predict(input_df)[0])
        confidence_pct = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            confidence_pct = round(float(proba[prediction]) * 100, 2)
    except Exception as exc:
        logger.exception("Prediction calculation error")
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}")

    is_eol = prediction == 1
    status_message = (
        "End of Life — replacement recommended"
        if is_eol
        else "Healthy — continue normal operation"
    )

    return PredictionResponse(
        battery_id=payload.battery_id,
        prediction=prediction,
        eol_reached=is_eol,
        status_message=status_message,
        confidence_pct=confidence_pct,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn running on http://0.0.0.0:10000