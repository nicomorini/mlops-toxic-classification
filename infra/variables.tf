variable "project_id" {
  description = "The GCP Project ID (NOT the project name, but the unique ID)"
  type        = string
  default     = "mlops-toxic-project" 
}

variable "region" {
  description = "The GCP Region for deploying resources"
  type        = string
  default     = "europe-west1" # Belgium, excellent and cheap for Europe
}

variable "gcp_credentials" {
  description = "Path to the GCP Service Account JSON key"
  type        = string
  default     = "../gcp-key.json" # Relative path going up one folder from /infra
}