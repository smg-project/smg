//! Prefill-decode disaggregation integration tests, and the placement read the
//! prefill-addition and prefill-removal tests share: a completion routed with its
//! decision record, the prefill worker the bucket policy chose read from it.

use openai_protocol::completion::CompletionRequest;
use smg::{middleware::TenantRequestMeta, routers::RouterTrait};

mod grpc_pd_client_departure_test;
mod grpc_pd_decode_start_cancelled_test;
mod grpc_pd_fanout_test;
mod grpc_pd_kv_store_connector_test;
mod grpc_pd_prefill_addition_test;
mod grpc_pd_prefill_removal_test;
mod grpc_pd_prefill_silence_test;

/// A non-streaming completion of `words` prompt words for `model`, eight output tokens.
#[expect(clippy::unwrap_used, reason = "test helper")]
fn completion_request(model: &str, words: usize) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "prompt": "test ".repeat(words),
        "max_tokens": 8,
        "stream": false,
    }))
    .unwrap()
}

/// The prefill worker the bucket policy chose for a completion of `words` prompt
/// words, read from the decision record the request asks for.
#[expect(clippy::expect_used, clippy::unwrap_used, reason = "test helper")]
async fn prefill_worker_for(
    router: &dyn RouterTrait,
    tenant: &TenantRequestMeta,
    model: &str,
    words: usize,
) -> String {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
    let response = router
        .route_completion(
            Some(&headers),
            tenant,
            completion_request(model, words),
            model,
        )
        .await;
    let status = response.status();
    let trace = response
        .headers()
        .get("x-smg-cache-trace")
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    assert_eq!(
        status,
        http::StatusCode::OK,
        "{words} tokens: {}",
        String::from_utf8_lossy(&bytes)
    );
    let trace: serde_json::Value =
        serde_json::from_str(&trace.expect("the request asked for its decision record")).unwrap();
    trace["selections"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|selection| selection["policy"] == "bucket")
        .and_then(|selection| selection["worker"].as_str())
        .map(str::to_owned)
        .expect("the record names the prefill worker the bucket policy chose")
}
