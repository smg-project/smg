//! A prefill worker leaves a PD deployment whose prefill policy partitions
//! requests by size across the prefill workers (`bucket`). The removal runs
//! the real removal workflow; afterwards a request of a size the departed
//! worker owned has to find a prefill worker still. The partition must
//! forget the worker, not keep handing its size range to a URL that is gone.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use common::grpc_harness::{
    grpc_router, grpc_router_config, grpc_test_context, grpc_worker, health_checks_off,
    serve_canned_worker, Canned,
};
use openai_protocol::{
    completion::CompletionRequest,
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus},
};
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, Worker, WorkerType},
    workflow::Job,
};

use crate::common;

const MODEL: &str = "pd-prefill-removal-test-model";
/// The bucket's first split hands 4096 size units out evenly: with three
/// prefill workers the middle one owns [1365, 2729]. The mock tokenizer
/// turns every `test` of a prompt into one token, so a prompt's word count
/// is the size the policy sees.
const IN_THE_FIRST_RANGE: usize = 100;
const IN_THE_MIDDLE_RANGE: usize = 2000;
const IN_THE_LAST_RANGE: usize = 3000;

/// Spawn a canned mock gRPC worker in-process.
async fn start_mock_grpc_worker() -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 3,
    })
    .await
}

struct Bed {
    app_context: Arc<AppContext>,
    router: Box<dyn RouterTrait>,
    prefill_urls: Vec<String>,
}

/// A PD gRPC router with the bucket prefill policy over `prefill_ports` mock
/// prefill workers and one mock decode worker, all registered as the
/// rendezvous-room runtime the canned mock serves, each with a drain window
/// of `drain_settle_secs` before its removal.
async fn build_bed(prefill_ports: &[u16], decode_port: u16, drain_settle_secs: u64) -> Bed {
    let config = grpc_router_config()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: Some(PolicyConfig::Bucket {
                // Selection follows the size partition alone: no load spill
                // and no boundary adjustment within the test.
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: 1.0001,
                bucket_adjust_interval_secs: 3600,
            }),
            decode_policy: None,
        })
        .random_policy()
        .build_unchecked();
    let app_context = grpc_test_context(config, MODEL).await;

    let roles = prefill_ports
        .iter()
        .map(|port| (*port, WorkerType::Prefill))
        .chain([(decode_port, WorkerType::Decode)]);
    let workers: Vec<Arc<dyn Worker>> = roles
        .map(|(port, worker_type)| {
            Arc::new(
                grpc_worker(
                    port,
                    worker_type,
                    RuntimeType::TokenSpeed,
                    ModelCard::new(MODEL),
                )
                .health_config(HealthCheckConfig {
                    drain_settle_secs,
                    ..health_checks_off()
                })
                .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let prefill_workers: Vec<Arc<dyn Worker>> = workers
        .iter()
        .filter(|worker| matches!(worker.worker_type(), WorkerType::Prefill))
        .cloned()
        .collect();
    let router = grpc_router(&app_context, workers).await;
    // What the registration workflow's policy step does once prefill workers
    // are in: the bucket policy learns their URLs and splits the sizes.
    app_context
        .policy_registry
        .init_pd_bucket_policies(&prefill_workers);
    let prefill_urls = prefill_workers
        .iter()
        .map(|worker| worker.url().to_string())
        .collect();
    Bed {
        app_context,
        router,
        prefill_urls,
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request(words: usize) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "test ".repeat(words),
        "max_tokens": 8,
        "stream": false,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-prefill-removal-test-tenant"))
}

/// Route one completion of `words` tokens; the status and the body.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn route(bed: &Bed, words: usize) -> (http::StatusCode, String) {
    let response = bed
        .router
        .route_completion(None, &tenant(), completion_request(words), MODEL)
        .await;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// Submit the removal of `url` the way the control plane does: a removal
/// job, which runs the removal workflow (drain, then the removal).
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn submit_removal(bed: &Bed, url: &str) {
    bed.app_context
        .worker_job_queue
        .get()
        .expect("the test context wires the job queue")
        .submit(Job::RemoveWorker {
            url: url.to_string(),
            expected_revisions: None,
        })
        .await
        .expect("submit the removal");
}

/// Wait until the registry holds `url` with `status`.
async fn wait_for_status(bed: &Bed, url: &str, status: WorkerStatus) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let current = bed
            .app_context
            .worker_registry
            .get_by_url(url)
            .map(|worker| worker.status());
        if current == Some(status) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{url} did not reach {status:?}: {current:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Wait until `url` has left the registry and its removal job has settled.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn await_removed(bed: &Bed, url: &str) {
    let job_queue = bed
        .app_context
        .worker_job_queue
        .get()
        .expect("the test context wires the job queue");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let gone = bed.app_context.worker_registry.get_by_url(url).is_none();
        let status = job_queue.get_status(url);
        let settled = status
            .as_ref()
            .is_none_or(|s| s.status != "pending" && s.status != "processing");
        if gone && settled {
            assert!(
                status.as_ref().is_none_or(|s| s.status != "failed"),
                "removal of {url} failed: {status:?}"
            );
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "removal of {url} did not settle: gone={gone}, status={status:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Three prefill workers; the middle one is removed through the removal
/// workflow while requests of every size flow. Afterwards every size still
/// finds a prefill worker, the removed worker's own range included.
#[tokio::test]
async fn requests_in_a_removed_prefill_workers_size_range_still_find_a_worker() {
    let prefill = [
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    ];
    let decode = start_mock_grpc_worker().await;
    let bed = build_bed(&prefill, decode, 0).await;

    for words in [IN_THE_FIRST_RANGE, IN_THE_MIDDLE_RANGE, IN_THE_LAST_RANGE] {
        let (status, body) = route(&bed, words).await;
        assert_eq!(
            status,
            http::StatusCode::OK,
            "{words} tokens before the removal: {body}"
        );
    }

    submit_removal(&bed, &bed.prefill_urls[1]).await;
    await_removed(&bed, &bed.prefill_urls[1]).await;

    for round in 0..3 {
        for words in [IN_THE_FIRST_RANGE, IN_THE_MIDDLE_RANGE, IN_THE_LAST_RANGE] {
            let (status, body) = route(&bed, words).await;
            assert_eq!(
                status,
                http::StatusCode::OK,
                "{words} tokens after the removal, round {round}: {body}"
            );
        }
    }
}

/// Route one completion of `words` tokens, asking for its decision record,
/// and return the prefill worker the bucket policy chose for it.
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn prefill_worker_for(bed: &Bed, words: usize) -> String {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
    let response = bed
        .router
        .route_completion(Some(&headers), &tenant(), completion_request(words), MODEL)
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

/// The removal drains the worker first: for its drain window it is still
/// listed by the bucket but no longer offered to the policy. From the first
/// draining moment on, a request of the middle size goes to the previous
/// worker, the neighbour the removal hands the range to afterwards: the range
/// moves once, and no request of it is scattered over the other workers. The
/// other sizes keep their owners throughout.
#[tokio::test]
async fn a_draining_prefill_workers_size_range_goes_to_its_neighbour_at_once() {
    let prefill = [
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    ];
    let decode = start_mock_grpc_worker().await;
    let bed = build_bed(&prefill, decode, 3).await;
    let (first, middle, last) = (
        bed.prefill_urls[0].clone(),
        bed.prefill_urls[1].clone(),
        bed.prefill_urls[2].clone(),
    );
    assert_eq!(prefill_worker_for(&bed, IN_THE_FIRST_RANGE).await, first);
    assert_eq!(prefill_worker_for(&bed, IN_THE_MIDDLE_RANGE).await, middle);
    assert_eq!(prefill_worker_for(&bed, IN_THE_LAST_RANGE).await, last);

    submit_removal(&bed, &middle).await;
    wait_for_status(&bed, &middle, WorkerStatus::Draining).await;
    let mut during_the_drain = Vec::new();
    while bed
        .app_context
        .worker_registry
        .get_by_url(&middle)
        .is_some()
    {
        during_the_drain.push(prefill_worker_for(&bed, IN_THE_MIDDLE_RANGE).await);
        assert_eq!(prefill_worker_for(&bed, IN_THE_FIRST_RANGE).await, first);
        assert_eq!(prefill_worker_for(&bed, IN_THE_LAST_RANGE).await, last);
    }
    await_removed(&bed, &middle).await;
    let mut after_the_removal = Vec::new();
    for _ in 0..3 {
        after_the_removal.push(prefill_worker_for(&bed, IN_THE_MIDDLE_RANGE).await);
    }

    assert!(
        during_the_drain.len() >= 4,
        "the drain window saw only {} requests of the middle size",
        during_the_drain.len()
    );
    let mut per_worker: BTreeMap<&str, usize> = BTreeMap::new();
    for worker in &during_the_drain {
        *per_worker.entry(worker.as_str()).or_default() += 1;
    }
    assert_eq!(
        per_worker,
        BTreeMap::from([(first.as_str(), during_the_drain.len())]),
        "middle-size requests during the drain, per prefill worker (the previous worker is {first})"
    );
    assert_eq!(
        after_the_removal,
        vec![first.clone(); 3],
        "after the removal the previous worker keeps the range"
    );
}
