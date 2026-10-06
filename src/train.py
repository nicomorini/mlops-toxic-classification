import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_dummy_data() -> tuple[list[str], list[int]]:
    """Generate a small dummy dataset for toxic/non-toxic text classification."""
    logger.info("Generating dummy dataset...")
    X = [
        "You are an absolute idiot",
        "This project is very interesting",
        "I hate you, go away",
        "Hello, how are you today?",
        "You are a piece of trash",
        "Great job on this code",
        "Go to hell you and your family",
        "Have a wonderful day everyone",
        "You are so stupid",
        "I love this new feature",
    ]
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return X, y


def train_baseline_model() -> None:
    """Create, train, evaluate and log the baseline model using MLflow."""

    # 1. Fetch and split Data
    X, y = get_dummy_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 2. Define Hyperparameters
    params = {
        "tfidf__max_features": 1000,
        "clf__C": 1.0,
        "clf__penalty": "l2",
        "clf__random_state": 42,
    }

    # 3. Initialize MLflow Experiment
    # This creates a folder named 'mlruns' locally to store the tracking data
    mlflow.set_experiment("Toxic_Classification_Baseline")

    with mlflow.start_run():
        logger.info("MLflow run started.")

        # Log Hyperparameters to MLflow
        mlflow.log_params(params)

        # 4. Define and Train the Pipeline
        logger.info("Initializing and training the Pipeline...")
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=params["tfidf__max_features"])),
                (
                    "clf",
                    LogisticRegression(
                        C=params["clf__C"],
                        penalty=params["clf__penalty"],
                        random_state=params["clf__random_state"],
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)

        # 5. Evaluate the model
        logger.info("Evaluating the model...")
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        # Log Metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        logger.info(f"Metrics - Accuracy: {acc:.2f}, F1: {f1:.2f}")

        # 6. Log the model inside MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        logger.info("Model logged to MLflow.")

        # 7. (Fallback) Save the model locally for the FastAPI app and CI/CD
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Model successfully saved locally to: {model_path}")


if __name__ == "__main__":
    train_baseline_model()
