use axum::{body::to_bytes, http::StatusCode};

use super::{get_server_info, health_generate};
use crate::test_support::{
    text_of, StubWorker, StubWorkers, CREDENTIALED_WORKER_URL, MASKED_WORKER_URL,
};

/// The health answer names an unhealthy worker by its address masked:
/// credentials in a worker URL never reach a health reader.
#[tokio::test]
async fn an_unhealthy_worker_is_named_masked() {
    let source = StubWorkers(vec![StubWorker::unhealthy_at(CREDENTIALED_WORKER_URL)]);

    let response = health_generate(&source);

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let text = text_of(response).await;
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
