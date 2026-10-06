import logging
import sys

from mlflow.tracking import MlflowClient

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_and_promote(
    model_name: str, metric_name: str, improvement_threshold: float
) -> None:
    """
    Compare the Staging (new) model against the Production (old) model.
    Promote the new model ONLY if its metric is strictly better than the old one
    by at least the improvement_threshold.
    """
    # 1. Initialize MLflow Client connecting to our central database
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

    logger.info(f"Evaluating models for '{model_name}' based on '{metric_name}'...")

    # 2. Fetch all versions of the model
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not find model '{model_name}' in MLflow Registry: {e}")
        sys.exit(1)

    production_version = None
    staging_version = None

    # Identify which version is in Production and which is in Staging
    for v in versions:
        if v.current_stage == "Production":
            production_version = v
        elif v.current_stage == "Staging":
            staging_version = v

    if not staging_version:
        logger.warning(
            f"No model found in 'Staging' for '{model_name}'. Nothing to evaluate."
        )
        sys.exit(1)

    # 3. Retrieve Metrics for Staging (Challenger)
    staging_run_id = staging_version.run_id
    staging_metrics = client.get_run(staging_run_id).data.metrics
    staging_score = staging_metrics.get(metric_name)

    if staging_score is None:
        logger.error(
            f"Metric '{metric_name}' not found in Staging run {staging_run_id}."
        )
        sys.exit(1)

    # 4. Retrieve Metrics for Production (Champion)
    production_score = 0.0  # Default if there is no production model yet
    if production_version:
        prod_run_id = production_version.run_id
        prod_metrics = client.get_run(prod_run_id).data.metrics
        production_score = prod_metrics.get(metric_name, 0.0)
        logger.info(
            f"[CHAMPION] Production Version {production_version.version} score: {production_score:.4f}"
        )
    else:
        logger.info("[CHAMPION] No existing Production model. Baseline score is 0.0")

    logger.info(
        f"[CHALLENGER] Staging Version {staging_version.version} score: {staging_score:.4f}"
    )

    # 5. The Battle: Compare Scores
    score_diff = staging_score - production_score
    logger.info(
        f"Score difference: {score_diff:.4f} (Threshold needed: {improvement_threshold})"
    )

    if score_diff >= improvement_threshold:
        logger.info("🏆 Challenger wins! Promoting Staging model to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production",
            archive_existing_versions=True,  # Automatically demotes the old Champion
        )
        logger.info(f"Version {staging_version.version} is now in Production.")
    else:
        logger.info("🛑 Challenger failed to beat the Champion. Promotion rejected.")
        # Optionally, archive the failed staging model
        client.transition_model_version_stage(
            name=model_name, version=staging_version.version, stage="Archived"
        )


if __name__ == "__main__":
    # Define our evaluation rules
    MODEL_NAME = "ToxicClassifier"
    METRIC_TO_OPTIMIZE = "f1_score"
    IMPROVEMENT_THRESHOLD = 0.01  # The new model must be at least 1% better

    evaluate_and_promote(
        model_name=MODEL_NAME,
        metric_name=METRIC_TO_OPTIMIZE,
        improvement_threshold=IMPROVEMENT_THRESHOLD,
    )
