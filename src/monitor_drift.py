import json
import logging
import sys
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_reference_data(ref_path: Path) -> pd.DataFrame:
    """Load or mock reference training data."""
    if ref_path.exists():
        return pd.read_csv(ref_path)

    logger.info(
        "Reference file not found. Creating a dummy baseline for the prototype..."
    )
    # These represent the style of data the model was trained on
    df = pd.DataFrame(
        {
            "text": [
                "You are an absolute idiot",
                "This project is very interesting",
                "I hate you, go away",
                "Hello, how are you today?",
            ]
        }
    )
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ref_path, index=False)
    return df


def monitor_data_drift() -> None:
    """Analyze production logs against reference data to detect Data Drift."""

    data_dir = Path("data")
    prod_path = data_dir / "production_logs.csv"
    ref_path = data_dir / "reference.csv"

    # 1. Check if we have production logs (If missing, generate dummy data for CI/CD)
    if not prod_path.exists():
        logger.warning(
            "Production logs not found. Generating dummy production data for CI/CD demonstration..."
        )
        prod_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a dummy CSV containing drifted data (new slang)
        pd.DataFrame(
            {
                "timestamp": ["2026-01-01T12:00:00", "2026-01-01T12:05:00"],
                "text": ["ur trash bro tbh", "cancel this guy fr fr"],
                "is_toxic": [True, True],
                "probability": [0.95, 0.88],
            }
        ).to_csv(prod_path, index=False)

    # 2. Load Datasets
    logger.info("Loading reference and production data...")
    ref_df = get_reference_data(ref_path)
    prod_df = pd.read_csv(prod_path)

    # We only care about the input feature 'text' for data drift here
    ref_df = ref_df[["text"]]
    prod_df = prod_df[["text"]]

    # 3. Generate Evidently Report
    logger.info("Running Evidently Data Drift analysis...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_df, current_data=prod_df)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save HTML for human visualization
    html_path = reports_dir / "drift_report.html"
    drift_report.save_html(str(html_path))
    logger.info(f"Visual HTML report saved to: {html_path}")

    # Save JSON for machine automation
    json_path = reports_dir / "drift_report.json"
    drift_report.save_json(str(json_path))
    logger.info(f"Automated JSON report saved to: {json_path}")

    # 4. Parse JSON to extract the boolean drift flag
    logger.info("Parsing JSON to evaluate drift status...")
    with open(json_path, encoding="utf-8") as f:
        report_data = json.load(f)

    # Extract the overall dataset drift boolean flag
    drift_detected = report_data["metrics"][0]["result"]["dataset_drift"]

    if drift_detected:
        logger.warning(
            "🚨 DATA DRIFT DETECTED! Production data distribution has changed."
        )
        logger.warning("Action required: Trigger Continuous Training pipeline (CI/CT).")
        # Exit with error code 1 to fail the GitHub Actions workflow and trigger an email alert
        sys.exit(1)
    else:
        logger.info(
            "✅ No data drift detected. The model is still operating in a familiar environment."
        )
        sys.exit(0)


if __name__ == "__main__":
    monitor_data_drift()
