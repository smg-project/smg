//! A prefill-decode completion that asks for its decision record reads the
//! pair's two rows: the response header names the prefill pick and the
//! decode pick, each under its side's policy.

use std::time::Duration;

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_workers, serve_canned_worker, Canned,
};
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};

use super::completion_request;
use crate::common;

const MODEL: &str = "pd-pair-trace-test-model";

/// A PD gRPC router, round robin on both roles, over one canned prefill
/// worker and one canned decode worker; returned with the two ports.
async fn build_pd_router() -> (Box<dyn RouterTrait>, u16, u16) {
    let canned = || {
        serve_canned_worker(Canned {
            model: MODEL,
            gen_delay: Duration::ZERO,
            output_tokens: 3,
        })
    };
    let (prefill, decode) = (canned().await, canned().await);
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .build_unchecked();
    let workers = grpc_workers(
        &[(prefill, WorkerType::Prefill), (decode, WorkerType::Decode)],
        RuntimeType::TokenSpeed,
        MODEL,
    );
    (
        build_grpc_router(config, MODEL, workers).await,
        prefill,
        decode,
    )
}

/// The selections of the decision record one completion asks for, each as
/// its policy and its worker.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn traced_picks(router: &dyn RouterTrait) -> Vec<(String, String)> {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
    let tenant = TenantRequestMeta::new(TenantKey::new("pd-pair-trace-test-tenant"));
    let response = router
        .route_completion(Some(&headers), &tenant, completion_request(MODEL, 8), MODEL)
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    let trace = response
        .headers()
        .get("x-smg-cache-trace")
        .expect("the request asked for its decision record")
        .to_str()
        .expect("the record is ASCII")
        .to_owned();
    let record: serde_json::Value =
        serde_json::from_str(&trace).expect("the header is the record's JSON");
    record["selections"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|selection| {
            let field = |name: &str| selection[name].as_str().unwrap_or_default().to_owned();
            (field("policy"), field("worker"))
        })
        .collect()
}

/// One traced completion: the record lists the prefill pick, then the
/// decode pick, each under round robin.
#[tokio::test]
async fn a_traced_pair_names_both_picks() {
    let (router, prefill, decode) = build_pd_router().await;
    let pick = |port: u16| ("round_robin".to_owned(), format!("grpc://127.0.0.1:{port}"));
    assert_eq!(
        traced_picks(router.as_ref()).await,
        [pick(prefill), pick(decode)],
        "the prefill pick, then the decode pick"
    );
}
