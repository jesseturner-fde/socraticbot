variable "project_id" {
  description = "The Google Cloud Project ID where resources will be provisioned."
  type        = string
}

variable "region" {
  description = "The Google Cloud region for compute and storage resources."
  type        = string
  default     = "us-central1"
}

variable "app_name" {
  description = "Name of the application and resource prefix."
  type        = string
  default     = "socratic-study-agent"
}

variable "environment" {
  description = "Deployment environment (development, staging, production)."
  type        = string
  default     = "production"
}

variable "container_image" {
  description = "Artifact Registry container image URI for Cloud Run deployment."
  type        = string
  default     = "us-central1-docker.pkg.dev/google-course-fde/studyagent/agent:latest"
}

variable "gemini_model" {
  description = "Default Gemini model for Socratic orchestration."
  type        = string
  default     = "gemini-3.5-flash"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances (0 for serverless scale-to-zero)."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of autoscaled Cloud Run instances."
  type        = number
  default     = 5
}
