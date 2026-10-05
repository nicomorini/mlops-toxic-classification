import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize a specific logger for this module
logger = logging.getLogger(__name__)


def get_dummy_data() -> tuple[list[str], list[int]]:
    """
    Generate a small dummy dataset for toxic/non-toxic text classification.
    Returns a tuple containing a list of texts and a list of labels.
    """
    logger.info("Generating dummy dataset...")

    # Dummy data: texts and corresponding labels (1 = Toxic, 0 = Non-Toxic)
    X = [
        "You are an absolute idiot",
        "This project is very interesting",
        "I hate you, go away",
        "Hello, how are you today?",
        "You are a piece of trash",
        "Great job on this code",
        "Go to hell you and your family",
        "Have a wonderful day everyone",
    ]
    y = [1, 0, 1, 0, 1, 0, 1, 0]

    return X, y


def train_baseline_model() -> None:
    """
    Create, train, and save the baseline model pipeline.
    """
    # 1. Fetch Data
    X_train, y_train = get_dummy_data()

    # 2. Define the Pipeline
    logger.info("Initializing the Pipeline (TF-IDF + Logistic Regression)...")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True)),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )

    # 3. Training
    logger.info("Training the model...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed successfully.")

    # 4. Save the Model
    models_dir = Path("models")
    # Create the models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model successfully saved to: {model_path}")


if __name__ == "__main__":
    train_baseline_model()
