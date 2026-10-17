//! End-to-end check of the PD admission gate against a mock prefill/decode
//! pair, on the running window the decode worker reports.
//!
//! The gate claims a room in the decode engine's running window before a
//! dispatch goes out, and abstains for a worker that reports none. The decode
//! worker here carries the `max_num_seqs` label a vLLM servicer's
//! `GetServerInfo` flattens into (as TokenSpeed's and SGLang's already did),
//! set to a window of one, and the admission wait is zero: a request arriving
//! while another still holds the room is shed with the overload 503, where
//! the same pair without the label admits both.

#[path = "common/mod.rs"]
mod common;

use std::{collections::HashMap, sync::Arc, time::Duration};

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_worker, serve_canned_worker, Canned,
};
use openai_protocol::{completion::CompletionRequest, model_card::ModelCard};
use smg::{
    middleware::TenantRequestMeta,
    routers::{common::pd_admission, RouterTrait},
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};

const MODEL: &str = "pd-admission-test-model";
/// How long the canned mock holds each request before answering, so that a
/// second request lands while the first still holds its decode room.
const HOLD: Duration = Duration::from_secs(2);
/// Head start the first request gets before the second is sent.
const HEAD_START: Duration = Duration::from_millis(200);

/// The canned mock worker: it holds each request for `HOLD`.
async fn start_mock_grpc_worker() -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: HOLD,
        output_tokens: 3,
    })
    .await
}

/// A PD gRPC router over one mock prefill and one mock decode worker, the
/// decode carrying `decode_labels` as discovery would have set them, with a
/// zero admission wait so a full window sheds at once.
async fn build_pd_router(
    prefill_port: u16,
    decode_port: u16,
    decode_labels: HashMap<String, String>,
) -> Box<dyn RouterTrait> {
    // The test context skips the start-up path that latches the config's
    // wait, so latch it here: a full window answers at once.
    pd_admission::set_pd_admission_wait_secs(0);
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .random_policy()
        .pd_admission_wait_secs(0)
        .build_unchecked();
    let prefill = grpc_worker(
        prefill_port,
        WorkerType::Prefill,
        RuntimeType::TokenSpeed,
        ModelCard::new(MODEL),
    );
    let decode = grpc_worker(
        decode_port,
        WorkerType::Decode,
        RuntimeType::TokenSpeed,
        ModelCard::new(MODEL),
    )
    .labels(decode_labels);
    build_grpc_router(
        config,
        MODEL,
        vec![Arc::new(prefill.build()), Arc::new(decode.build())],
    )
    .await
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request() -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "Hello world",
        "max_tokens": 8,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-admission-test-tenant"))
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn read_body(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

/// Two requests that overlap on the pair: the status of each, in the order
/// they were sent.
async fn two_overlapping_requests(router: &dyn RouterTrait) -> [(http::StatusCode, String); 2] {
    let tenant = tenant();
    let first = router.route_completion(None, &tenant, completion_request(), MODEL);
    let second = async {
        tokio::time::sleep(HEAD_START).await;
        router
            .route_completion(None, &tenant, completion_request(), MODEL)
            .await
    };
    let (first, second) = tokio::join!(first, second);
    let mut out = Vec::with_capacity(2);
    for response in [first, second] {
        let status = response.status();
        out.push((status, read_body(response).await));
    }
    [out.swap_remove(0), out.swap_remove(0)]
}

/// A decode worker reporting a window of one admits one request at a time:
/// the second, arriving while the first holds the room, is shed with the
/// overload 503 rather than posted to the engines, and the room is free
/// again once the first completes.
#[tokio::test]
async fn a_reported_window_of_one_sheds_the_overlapping_request() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let labels = HashMap::from([("max_num_seqs".to_string(), "1".to_string())]);
    let router = build_pd_router(prefill, decode, labels).await;

    let outcomes = two_overlapping_requests(router.as_ref()).await;
    let shed: Vec<_> = outcomes
        .iter()
        .filter(|(status, _)| *status == http::StatusCode::SERVICE_UNAVAILABLE)
        .collect();
    assert_eq!(shed.len(), 1, "exactly one request is shed: {outcomes:?}");
    assert!(
        shed[0].1.contains("worker_overload_protection_shed"),
        "the shed is the overload guard's answer: {}",
        shed[0].1
    );
    assert!(
        outcomes
            .iter()
            .any(|(status, _)| *status == http::StatusCode::OK),
        "the request that holds the room completes: {outcomes:?}"
    );

    let response = router
        .route_completion(None, &tenant(), completion_request(), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "the room is released: {body}");
}

/// A decode worker that reports no window is not gated: both overlapping
/// requests go to the engines, as dispatch always did.
#[tokio::test]
async fn an_unreported_window_admits_overlapping_requests() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode, HashMap::new()).await;

    let outcomes = two_overlapping_requests(router.as_ref()).await;
    for (status, body) in &outcomes {
        assert_eq!(*status, http::StatusCode::OK, "{body}");
    }
}
