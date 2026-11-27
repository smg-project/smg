use axum::{routing::get, Json, Router};
use serde_json::json;
use tokio::net::TcpListener;

use super::probe_models_owned_by;
use crate::server::http_client;

#[tokio::test]
async fn probe_models_owned_by_accepts_nvidia_as_local() {
    async fn models() -> Json<serde_json::Value> {
        Json(json!({
            "data": [{
                "id": "test-model",
                "object": "model",
                "owned_by": "nvidia"
            }],
            "object": "list"
        }))
    }

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only mock /v1/models server; handle is aborted at test end"
    )]
    let server = tokio::spawn(async move {
        axum::serve(listener, Router::new().route("/v1/models", get(models)))
            .await
            .unwrap();
    });

    let owned_by = probe_models_owned_by(&format!("http://{addr}"), 5, &http_client(), None)
        .await
        .unwrap();
    server.abort();

    assert_eq!(owned_by, "nvidia");
}
