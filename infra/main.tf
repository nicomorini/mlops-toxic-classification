# --- Terraform Block ---
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# --- Provider Configuration ---
provider "google" {
  credentials = file(var.gcp_credentials)
  project     = var.project_id
  region      = var.region
}

# --- 1. Artifact Registry (Docker Image Storage) ---
resource "google_artifact_registry_repository" "ml_repo" {
  location      = var.region
  repository_id = "mlops-repo"
  description   = "Docker repository for the MLOps API"
  format        = "DOCKER"
}

# --- 2. Cloud Run Service ---
resource "google_cloud_run_v2_service" "ml_api" {
  name     = "mlops-toxic-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      # We use a public dummy image to initialize the service.
      # The CI/CD pipeline will later override this with our actual image from ml_repo.
      image = "us-docker.pkg.dev/cloudrun/container/hello"
      
      ports {
        container_port = 8000 # Matches the EXPOSE port in our Dockerfile
      }
      
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi" # Keeping it very small for the free tier
        }
      }
    }
  }
}

# --- 3. IAM Policy (Make the API public) ---
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = google_cloud_run_v2_service.ml_api.project
  location = google_cloud_run_v2_service.ml_api.location
  name     = google_cloud_run_v2_service.ml_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- 4. Outputs (Print useful info after creation) ---
output "artifact_registry_url" {
  description = "The URL of the Artifact Registry"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_repo.repository_id}"
}

output "cloud_run_service_url" {
  description = "The public URL of our deployed API"
  value       = google_cloud_run_v2_service.ml_api.uri
}