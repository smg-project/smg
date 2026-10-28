//! Health and server-info answers for the OpenAI-compatible router, judged
//! over the workers that front third-party providers.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::{error, worker::WorkerSource};

pub(super) fn health_generate(source: &dyn WorkerSource) -> Response {
    let workers = source.external_workers();
    if workers.is_empty() {
        return error::service_unavailable("service_unavailable", "No external workers registered");
    }

    let (healthy, unhealthy): (Vec<_>, Vec<_>) = workers.iter().partition(|w| w.is_healthy());

    if unhealthy.is_empty() {
        (
            StatusCode::OK,
            format!("OK - {} workers healthy", healthy.len()),
        )
            .into_response()
    } else {
        let unhealthy_info: Vec<_> = unhealthy
            .iter()
            .map(|w| format!("{} ({})", w.model_id(), w.redacted_url()))
            .collect();
        error::service_unavailable(
            "service_unavailable",
            format!(
                "{}/{} workers unhealthy: {}",
                unhealthy.len(),
                workers.len(),
                unhealthy_info.join(", ")
            ),
        )
    }
}

pub(super) fn get_server_info(source: &dyn WorkerSource) -> Response {
    let stats = source.stats();
    let workers = source.external_workers();
    let worker_urls: Vec<_> = workers.iter().map(|w| w.redacted_url()).collect();

    let info = json!({
        "router_type": "openai",
        "total_workers": stats.total_workers,
        "external_workers": workers.len(),
        "healthy_workers": stats.healthy_workers,
        "total_models": stats.total_models,
        "worker_urls": worker_urls
    });
    (StatusCode::OK, Json(info)).into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, http::StatusCode};

    use super::{get_server_info, health_generate};
    use crate::test_support::{
        StubWorker, StubWorkers, CREDENTIALED_WORKER_URL, MASKED_WORKER_URL,
    };

    /// The health answer names an unhealthy worker by its address masked:
    /// credentials in a worker URL never reach a health reader.
    #[tokio::test]
    async fn an_unhealthy_worker_is_named_masked() {
        let source = StubWorkers(vec![StubWorker::unhealthy_at(CREDENTIALED_WORKER_URL)]);

        let response = health_generate(&source);

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        let text = std::str::from_utf8(&body).expect("utf-8");
        assert!(!text.contains("secret"), "{text}");
        assert!(text.contains(&format!("m ({MASKED_WORKER_URL})")), "{text}");
    }

    /// The server-info answer lists every worker by its address masked: it is
    /// read without authentication, so a credential in a worker URL never
    /// reaches it either.
    #[tokio::test]
    async fn the_server_info_names_every_worker_masked() {
        let source = StubWorkers(vec![StubWorker::at(CREDENTIALED_WORKER_URL)]);

        let response = get_server_info(&source);

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        let text = std::str::from_utf8(&body).expect("utf-8");
        assert!(!text.contains("secret"), "{text}");
        let info: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(info["worker_urls"], serde_json::json!([MASKED_WORKER_URL]));
    }
}
