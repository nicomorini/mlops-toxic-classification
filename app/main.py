import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize a specific logger for this module
logger = logging.getLogger(__name__)

# Global dictionary to store loaded models
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the ML model on startup
    and clean up on shutdown.
    """
    model_path = Path("models") / "model.joblib"

    logger.info(f"Loading model from {model_path}...")
    if not model_path.exists():
        logger.error(
            f"Model file not found at {model_path}. Please run train.py first."
        )
        raise RuntimeError("Model file missing.")

    # Load the scikit-learn pipeline
    ml_models["toxic_classifier"] = joblib.load(model_path)
    logger.info("Model loaded successfully.")

    yield  # The API is running and receiving requests here

    # Clean up resources on shutdown
    logger.info("Shutting down and clearing model from memory...")
    ml_models.clear()


# Initialize FastAPI app with the lifespan manager
app = FastAPI(
    title="Toxic Text Classification API",
    description="MLOps portfolio project for real-time text classification",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    """Schema for the incoming prediction request."""

    text: str = Field(..., description="The text to classify", min_length=1)


class PredictResponse(BaseModel):
    """Schema for the outgoing prediction response."""

    is_toxic: bool
    toxicity_probability: float


# --- Endpoints ---
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Endpoint to classify text as toxic or non-toxic.
    """
    model = ml_models.get("toxic_classifier")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # Predict class (0 or 1)
        prediction = model.predict([request.text])[0]

        # Predict probability (returns an array [prob_class_0, prob_class_1])
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
    """Simple health check endpoint."""
    return {"status": "healthy", "model_loaded": "toxic_classifier" in ml_models}
