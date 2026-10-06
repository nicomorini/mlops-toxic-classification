import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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
    """Create, train, evaluate, and register the model into MLflow Production."""

    X, y = get_dummy_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    params = {
        "tfidf__max_features": 1000,
        "clf__C": 1.0,
        "clf__penalty": "l2",
        "clf__random_state": 42,
    }

    mlflow.set_experiment("Toxic_Classification_Baseline")

    with mlflow.start_run():
        logger.info("MLflow run started.")
        mlflow.log_params(params)

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

        logger.info("Evaluating the model...")
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        logger.info(f"Metrics - Accuracy: {acc:.2f}, F1: {f1:.2f}")

        # 1. Log AND Register the model in one step
        model_name = "ToxicClassifier"
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, artifact_path="model", registered_model_name=model_name
        )

        model_version = model_info.registered_model_version
        logger.info(f"Model registered as '{model_name}', Version: {model_version}")

        # 2. Promote the model to "Production" stage
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True,  # Automatically demote the old production model
        )
        logger.info(f"Version {model_version} promoted to 'Production' stage.")

        # 3. (Fallback) Save locally for FastAPI
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved locally to: {model_path}")


if __name__ == "__main__":
    train_baseline_model()
