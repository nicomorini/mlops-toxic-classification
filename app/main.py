import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# Force MLflow to allow local file system storage
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global dictionary to store loaded models
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the ML model on startup.
    It attempts to fetch the Production model from MLflow Registry.
    If it fails (e.g., MLflow server down or running in CI/CD cloud),
    it falls back to the local .joblib file.
    """
    model_name = "ToxicClassifier"
    model_stage = "Production"
    model_uri = f"models:/{model_name}/{model_stage}"

    # 1. Try Loading from MLflow Registry
    try:
        # Define tracking URI (defaults to local SQLite DB)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)

        logger.info(f"Attempting to load model from MLflow Registry: {model_uri}")
        ml_models["toxic_classifier"] = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow Registry.")

    except Exception as e:
        logger.warning(f"MLflow load failed: {e!s}. Initiating fallback mechanism...")

        # 2. Fallback: Load from local file
        model_path = Path("models") / "model.joblib"
        logger.info(f"Loading fallback model from {model_path}...")

        if not model_path.exists():
            logger.error("Fallback model file not found. API cannot start.")
            raise RuntimeError(
                "CRITICAL: No models available (MLflow down & local file missing)."
            ) from e

        ml_models["toxic_classifier"] = joblib.load(model_path)
        logger.info("Fallback model loaded successfully.")

    yield  # The API is running here

    # Clean up resources on shutdown
    logger.info("Shutting down and clearing model from memory...")
    ml_models.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Toxic Text Classification API",
    description="MLOps API with MLflow dynamic loading and Fallback",
    version="0.2.0",
    lifespan=lifespan,
)

# --- Observability: Prometheus Instrumentation ---
Instrumentator().instrument(app).expose(app)


class PredictRequest(BaseModel):
    """Schema for the incoming prediction request."""

    text: str = Field(..., description="The text to classify", min_length=1)


class PredictResponse(BaseModel):
    """Schema for the outgoing prediction response."""

    is_toxic: bool
    toxicity_probability: float


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Endpoint to classify text."""
    model = ml_models.get("toxic_classifier")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        prediction = model.predict([request.text])[0]
        probabilities = model.predict_proba([request.text])[0]
        toxic_prob = float(probabilities[1])

        return PredictResponse(
            is_toxic=bool(prediction == 1), toxicity_probability=toxic_prob
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during prediction: {e!s}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction."
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": "toxic_classifier" in ml_models}
