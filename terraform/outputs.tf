output "cloud_run_service_url" {
  description = "The deployed URL of the Socratic Study Agent Cloud Run service."
  value       = google_cloud_run_v2_service.studyagent_service.uri
}

output "artifact_registry_repository_url" {
  description = "Artifact Registry Docker repository endpoint."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.studyagent_repo.repository_id}"
}

output "session_storage_bucket" {
  description = "GCS bucket name storing persistent sessions and learner profiles."
  value       = google_storage_bucket.session_storage.name
}

output "discovery_engine_datastore_id" {
  description = "Vertex AI Search / Discovery Engine datastore identifier."
  value       = google_discovery_engine_data_store.paper_datastore.data_store_id
}

output "service_account_email" {
  description = "Service account assigned to the Cloud Run workload."
  value       = google_service_account.agent_sa.email
}
