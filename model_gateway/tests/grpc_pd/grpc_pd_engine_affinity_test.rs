//! Prefill-decode completions of one prompt under `cache_aware` on both
//! roles keep the pair that served the prompt first, whichever
//! implementation places the pair: under `--policy-engine engine` each side
//! reads its affinity from its own policy, as each of the legacy pair's two
//! selections does.

use std::time::Duration;

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_workers, serve_canned_worker, Canned,
};
use smg::{
    config::PolicyEngineMode,
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};

use super::completion_request;
use crate::common;

const MODEL: &str = "pd-engine-affinity-test-model";

/// The prompt's size in words: several blocks of the policy's tree.
const WORDS: usize = 96;

/// The completions that follow the prompt's first.
const FOLLOWERS: usize = 7;

/// A PD gRPC router under `mode`, `cache_aware` on both roles, over two
/// canned prefill workers and two canned decode workers.
async fn build_pd_router(mode: PolicyEngineMode) -> Box<dyn RouterTrait> {
    let canned = || {
        serve_canned_worker(Canned {
            model: MODEL,
            gen_delay: Duration::ZERO,
            output_tokens: 3,
        })
    };
    let ports = [
        (canned().await, WorkerType::Prefill),
        (canned().await, WorkerType::Prefill),
        (canned().await, WorkerType::Decode),
        (canned().await, WorkerType::Decode),
    ];
    let mut config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .cache_aware_policy(0.5, 32, 1.1, 0, 4096)
        .build_unchecked();
    config.policy_engine = mode;
    let workers = grpc_workers(&ports, RuntimeType::TokenSpeed, MODEL);
    build_grpc_router(config, MODEL, workers).await
}

/// The pair one traced completion of the prompt was placed on: the workers
/// of its record's prefill row and decode row.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn placed_pair(router: &dyn RouterTrait) -> (String, String) {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
    let tenant = TenantRequestMeta::new(TenantKey::new("pd-engine-affinity-test-tenant"));
    let response = router
        .route_completion(
            Some(&headers),
            &tenant,
            completion_request(MODEL, WORDS),
            MODEL,
        )
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
    let workers: Vec<String> = record["selections"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|selection| selection["policy"] == "cache_aware")
        .filter_map(|selection| selection["worker"].as_str().map(str::to_owned))
        .collect();
    let [prefill, decode] =
        <[String; 2]>::try_from(workers).expect("a prefill row and a decode row under cache_aware");
    (prefill, decode)
}

/// Eight completions of one prompt per switch value. The first is a miss
/// that either implementation places by load; the seven that follow keep
/// its prefill and its decode, seven of seven under the engine as under the
/// legacy policies.
#[tokio::test]
async fn a_cache_aware_pair_keeps_its_prompts_pair_under_both_switch_values() {
    for mode in [PolicyEngineMode::Legacy, PolicyEngineMode::Engine] {
        let router = build_pd_router(mode).await;
        let first = placed_pair(router.as_ref()).await;
        let mut kept = 0;
        for _ in 0..FOLLOWERS {
            kept += usize::from(placed_pair(router.as_ref()).await == first);
        }
        assert_eq!(
            kept, FOLLOWERS,
            "{mode:?}: every later completion keeps the pair {first:?}"
        );
    }
}
