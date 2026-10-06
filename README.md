# 🚀 End-to-End MLOps Pipeline: Toxic Text Classification

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-blue.svg?logo=mlflow)
![Terraform](https://img.shields.io/badge/Terraform-IaC-623CE4.svg?logo=terraform)
![Google Cloud](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg?logo=googlecloud)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)
![EvidentlyAI](https://img.shields.io/badge/EvidentlyAI-Drift_Detection-orange.svg)

## 📌 Project Overview
This repository contains a complete, production-ready **MLOps architecture** for a Machine Learning model that classifies text as toxic or non-toxic. 

It is a fully automated system that handles the entire ML lifecycle: from code versioning and model tracking to infrastructure provisioning, continuous deployment (CI/CD), real-time monitoring, and automated continuous training (CT) upon data drift detection.

## 🏗️ Architecture & Workflow

The system is designed following best practices for MLOps, ensuring scalability, reproducibility, and high availability.

1. **Development & Tracking:** Code is formatted with `Ruff`, tested with `Pytest`, and ML experiments are tracked using **MLflow**.
2. **Infrastructure as Code (IaC):** **Terraform** automatically provisions a Docker registry and a Serverless environment on **Google Cloud Platform (GCP)**.
3. **CI/CD:** Every push to `main` triggers **GitHub Actions** to test the code, build a multi-stage **Docker** image, and deploy it to GCP Cloud Run with zero downtime.
4. **Serving & Fallback:** **FastAPI** serves the model. Upon startup, the API dynamically pulls the latest "Production" model from the MLflow Registry. If the registry is down, a **Critical Fallback Mechanism** loads a local `.joblib` model to ensure the API never crashes.
5. **Observability:** **Prometheus** tracks API metrics (requests, latency), while a background asynchronous task logs user predictions for drift analysis.
6. **Continuous Monitoring & Training (CT):** A nightly cron job runs **EvidentlyAI** to detect Data Drift. If drift is detected, an automated retraining pipeline is triggered. The new model is trained, evaluated against the current production model, and promoted *only* if it mathematically outperforms the old one (A/B Testing logic).

## 🛠️ Tech Stack
* **Machine Learning:** `scikit-learn`, `pandas`
* **MLOps & Tracking:** `MLflow`, `DVC` (Data Version Control)
* **API Serving:** `FastAPI`, `Uvicorn`
* **Infrastructure & DevOps:** `Docker`, `Terraform`, `Google Cloud Platform` (Cloud Run, Artifact Registry)
* **CI/CD:** `GitHub Actions`, `Ruff` (Linting), `Pytest`
* **Monitoring:** `Prometheus` (Telemetry), `EvidentlyAI` (Data Drift)

---

## 🧠 Deep Dive: How the System Works

### 1. Dynamic Model Loading & High Availability
The FastAPI application (`app/main.py`) does not rely on static files. During the `lifespan` startup event, it connects to the MLflow Model Registry to download the latest certified "Production" model. To guarantee High Availability, a `try-except` fallback mechanism is implemented: if the MLflow server is unreachable, it loads a local serialized model.

### 2. Infrastructure as Code (IaC)
The `infra/` folder contains Terraform scripts (`main.tf`, `variables.tf`). Running `terraform apply` idempotently creates a GCP Artifact Registry for Docker images and a Cloud Run service configured for serverless autoscaling (scale-to-zero to optimize costs). 

### 3. Asynchronous Data Collection
To perform Drift Detection in production without adding latency to the user's API response, FastAPI's `BackgroundTasks` are utilized. Once the API computes the prediction, a background thread safely appends the user input and the model's confidence score to a local production dataset (`data/production_logs.csv`) using UTC timestamps.

### 4. Automated Drift Detection & Continuous Training (CT)
A GitHub Actions cron job (`.github/workflows/drift_check.yml`) runs nightly. It triggers `src/monitor_drift.py`, which uses **EvidentlyAI** to compare the reference dataset with the collected production logs. 
If Data Drift is detected (e.g., users are using new slang), the script exits with code `1`. This fails the pipeline, triggering an alert and initiating the Continuous Training workflow (`retrain.yml`).

### 5. The "Gatekeeper" Evaluation
When retraining occurs, the new model is not blindly deployed. It is placed in the "Staging" environment in MLflow. The script `src/evaluate_and_promote.py` acts as a judge: it compares the F1-score of the new Staging model against the current Production model. The Staging model is promoted to Production *only* if it beats the current champion by a defined threshold (e.g., +1%).

---

## 📂 Project Structure

```text
├── .github/workflows/       # CI/CD pipelines (CI, CD, Drift Check, Retrain)
├── app/
│   ├── main.py              # FastAPI application with dynamic MLflow loading
├── infra/
│   ├── main.tf              # Terraform infrastructure definition for GCP
│   ├── variables.tf         # Terraform variables
├── src/
│   ├── train.py                 # ML training script & MLflow tracking
│   ├── monitor_drift.py         # EvidentlyAI drift detection script
│   ├── retrain.py               # Continuous training pipeline trigger
│   ├── evaluate_and_promote.py  # A/B testing logic for model promotion
├── tests/
│   ├── test_api.py          # Pytest API integration tests
├── Dockerfile               # Multi-stage optimized Docker build
├── requirements.txt         # Python dependencies
└── pytest.ini               # Pytest configuration
```

## 💻 How to Run Locally

1. **Clone the repository and set up the environment:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mlops-toxic-classification.git
   cd mlops-toxic-classification
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Train the baseline model and populate the MLflow Registry:**
   ```bash
   python src/train.py
   ```

3. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   *Access the interactive API documentation at `http://127.0.0.1:8000/docs`.*

4. **View MLflow Tracking Dashboard:**
   ```bash
   mlflow ui
   ```
   *Access the dashboard at `http://127.0.0.1:5000` to view experiments and the Model Registry.*

5. **Run Drift Detection manually:**
   ```bash
   python src/monitor_drift.py
   ```
   *An interactive HTML report will be generated in the `reports/` folder.*

---
*Built by Nicolò Morini (nicomorini25@gmail.com)*
