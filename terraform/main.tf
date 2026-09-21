terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.20"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# =====================================================================
# Enable Required Google Cloud APIs
# =====================================================================

locals {
  services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "discoveryengine.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
  ]
}

resource "google_project_service" "enabled_services" {
  for_each                   = toset(local.services)
  project                    = var.project_id
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}

# =====================================================================
# Artifact Registry for Docker Containers
# =====================================================================

resource "google_artifact_registry_repository" "studyagent_repo" {
  provider      = google
  location      = var.region
  repository_id = "${var.app_name}-repo"
  description   = "Docker repository for Socratic Technical Study Agent"
  format        = "DOCKER"

  depends_on = [google_project_service.enabled_services]
}

# =====================================================================
# Dedicated Cloud Run Service Account & Least-Privilege IAM
# =====================================================================

resource "google_service_account" "agent_sa" {
  account_id   = "${var.app_name}-sa"
  display_name = "Service Account for Socratic Study Agent"
}

# Secret Manager Accessor for API Keys
resource "google_project_iam_member" "sa_secret_access" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

# Storage Object User for Session Persistence
resource "google_project_iam_member" "sa_storage_access" {
  project = var.project_id
  role    = "roles/storage.objectUser"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

# Discovery Engine / Vertex AI Search Viewer
resource "google_project_iam_member" "sa_discovery_access" {
  project = var.project_id
  role    = "roles/discoveryengine.viewer"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

# =====================================================================
# Secret Manager for Gemini API Key
# =====================================================================

resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "${var.app_name}-gemini-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_services]
}

# =====================================================================
# Google Cloud Storage Bucket for Session & Profile Persistence
# =====================================================================

resource "google_storage_bucket" "session_storage" {
  name                        = "${var.project_id}-${var.app_name}-sessions-${var.environment}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 90
    }
  }

  depends_on = [google_project_service.enabled_services]
}

# =====================================================================
# Vertex AI Search / Discovery Engine Datastore
# =====================================================================

resource "google_discovery_engine_data_store" "paper_datastore" {
  provider                    = google-beta
  location                    = "global"
  data_store_id               = "${var.app_name}-datastore"
  display_name                = "Attention Research Literature Datastore"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false

  depends_on = [google_project_service.enabled_services]
}

# =====================================================================
# Google Cloud Run v2 Service Deployment
# =====================================================================

resource "google_cloud_run_v2_service" "studyagent_service" {
  name     = var.app_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.agent_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      env {
        name  = "GEMINI_MODEL"
        value = var.gemini_model
      }
      env {
        name  = "GCS_SESSION_BUCKET"
        value = google_storage_bucket.session_storage.name
      }
      env {
        name  = "DISCOVERY_ENGINE_DATASTORE_ID"
        value = google_discovery_engine_data_store.paper_datastore.data_store_id
      }

      # Mount API key from Secret Manager
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.enabled_services,
    google_artifact_registry_repository.studyagent_repo,
  ]
}

# Allow public or authenticated invocations based on policy
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.studyagent_service.name
  role     = "roles/run.invoker"
  member   = "allAuthenticatedUsers"
}
