//! Regular placement through the policy engine: under `--policy-engine
//! engine` each preset the engine serves routes every request of the HTTP
//! router, as it does under the legacy default. So does each of them a
//! document carries, written as a document and read from a file as
//! `--policy-spec` is; and under a document a worker's `policy` label takes
//! no model off it.

use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{PolicyConfig, PolicyEngineMode, RouterConfig};
use smg_policy_controller::spec::PolicySpec;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, MockWorker, RequestRecorder},
    wait_for_worker_registration, AppTestContext, TestRouterConfig, TestWorkerConfig,
};

/// The harness's regular configuration under `policy`, decided through the
/// engine.
fn engine_config(port: u16, policy: PolicyConfig) -> RouterConfig {
    let mut config = TestRouterConfig::round_robin(port);
    config.policy = policy;
    config.policy_engine = PolicyEngineMode::Engine;
    config
}

/// The harness's regular configuration under the document of the preset
/// `name`, written to a file in `dir` and read as `--policy-spec` reads it.
#[expect(clippy::expect_used, reason = "test helper")]
fn spec_file_config(port: u16, name: &str, dir: &tempfile::TempDir) -> RouterConfig {
    let path = dir.path().join(format!("{name}.yaml"));
    let document = serde_yaml::to_string(&PolicySpec::preset(name).expect("a preset document"))
        .expect("the document as YAML");
    std::fs::write(&path, document).expect("the document written");
    let mut config = TestRouterConfig::round_robin(port);
    config.policy_engine = PolicyEngineMode::Engine;
    config.policy_spec = Some(path);
    config
}

/// What the gateway under test decides with: a preset through the flags, or
/// a preset written as a document into a directory and read from that file
/// as `--policy-spec` reads it.
enum Decider<'a> {
    Flags(PolicyConfig),
    Document(&'a str, &'a tempfile::TempDir),
}

impl Decider<'_> {
    fn name(&self) -> &str {
        match self {
            Self::Flags(policy) => policy.name(),
            Self::Document(name, _) => name,
        }
    }

    /// The harness's configuration. It is built inside the awaited
    /// expression, so the test's future carries the decider and not a
    /// configuration.
    fn config(self, port: u16) -> RouterConfig {
        match self {
            Self::Flags(policy) => engine_config(port, policy),
            Self::Document(name, dir) => spec_file_config(port, name, dir),
        }
    }
}

/// `workers` healthy workers behind a gateway whose engine decides as
/// `decider` says take twenty generate requests, every one placed by a
/// decision of the engine.
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
async fn placed_through_the_engine(
    port: u16,
    worker_port: u16,
    workers: u16,
    decider: Decider<'_>,
) {
    let name = decider.name().to_owned();
    let documented = matches!(decider, Decider::Document(..));
    let ctx = AppTestContext::new_with_config(
        decider.config(port),
        TestWorkerConfig::healthy_workers(worker_port, workers),
    )
    .await;
    if documented {
        assert_eq!(
            ctx.app_context.policy_registry.get_default_policy().name(),
            "random",
            "{name}: a document's registry stands over the neutral configuration, not the flags'"
        );
    }
    let app = ctx.create_app();
    let requests = 20;
    let mut placed = 0;
    for i in 0..requests {
        let payload = json!({ "text": format!("Request {i}"), "stream": false });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        if resp.status() == StatusCode::OK {
            placed += 1;
        }
    }
    assert_eq!(placed, requests, "{name}: every request is placed");
    let decider = ctx
        .app_context
        .policy_registry
        .decider()
        .expect("the engine is hosted under the switch");
    assert_eq!(
        decider.decisions(),
        requests as u64,
        "{name}: every request was decided through the engine"
    );
    ctx.shutdown().await;
}

#[tokio::test]
async fn round_robin_routes_through_the_engine() {
    placed_through_the_engine(3920, 19920, 3, Decider::Flags(PolicyConfig::RoundRobin)).await;
}

#[tokio::test]
async fn random_routes_through_the_engine() {
    placed_through_the_engine(3921, 19924, 2, Decider::Flags(PolicyConfig::Random)).await;
}

#[tokio::test]
async fn passthrough_routes_through_the_engine() {
    placed_through_the_engine(3922, 19927, 2, Decider::Flags(PolicyConfig::Passthrough)).await;
}

#[tokio::test]
async fn power_of_two_routes_through_the_engine() {
    placed_through_the_engine(
        3923,
        19930,
        2,
        Decider::Flags(PolicyConfig::PowerOfTwo {
            load_metric: Default::default(),
        }),
    )
    .await;
}

#[tokio::test]
async fn least_load_routes_through_the_engine() {
    placed_through_the_engine(
        3924,
        19933,
        2,
        Decider::Flags(PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        }),
    )
    .await;
}

#[tokio::test]
async fn cache_aware_routes_through_the_engine() {
    placed_through_the_engine(
        3929,
        19948,
        2,
        Decider::Flags(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096)),
    )
    .await;
}

/// Each preset of the flag cases above that a document carries, written as
/// a document and read from a file as `--policy-spec` is, routes as the flag
/// does: every request placed, every one decided through the engine.
/// `cache_aware` is the flag's alone: nothing feeds a document the prefix
/// index, so its document is refused at start (the registry's
/// `a_document_whose_stage_reads_an_unfed_input_is_refused_at_the_apply`).
#[tokio::test]
async fn each_preset_as_a_spec_file_routes_as_the_flag_does() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let presets = [
        "round_robin",
        "random",
        "passthrough",
        "power_of_two",
        "least_load",
    ];
    for (i, name) in presets.into_iter().enumerate() {
        let i = u16::try_from(i).expect("five presets");
        placed_through_the_engine(3930 + i, 19960 + 2 * i, 2, Decider::Document(name, &dir)).await;
    }
}

/// Under a document nothing but the document chooses a policy. A worker
/// registered with a `policy` label is one more worker of the document's:
/// the label is not read, as `--policy` is not, and no legacy policy is built
/// for its model. Two workers serve one model under the round-robin
/// document, the first registered with the label `passthrough`, the policy
/// that sends every request to the first healthy worker: each worker takes a
/// share of the model's requests, as the document says. (The decider's
/// `decisions()` cannot tell the two apart: the regular path draws a
/// sequence for every request at its capture, whoever places it.)
#[tokio::test]
async fn a_workers_policy_label_is_not_read_under_a_document() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let ctx =
        AppTestContext::new_with_config(spec_file_config(3935, "round_robin", &dir), vec![]).await;
    let app = ctx.create_app();

    let mut workers = Vec::new();
    let mut recorders = Vec::new();
    for (port, labels) in [
        (19970, json!({ "policy": "passthrough" })),
        (19971, json!({})),
    ] {
        let recorder = RequestRecorder::new();
        set_request_recorder(port, recorder.clone());
        let mut worker = MockWorker::new(TestWorkerConfig::healthy(port));
        let url = worker.start().await.expect("the worker's address");
        let registration = json!({ "url": url, "labels": labels });
        let req = Request::builder()
            .method("POST")
            .uri("/workers")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(registration.to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED, "worker {port}");
        wait_for_worker_registration(&ctx.app_context, &url, Duration::from_secs(10)).await;
        workers.push(worker);
        recorders.push(recorder);
    }
    let model = ctx
        .app_context
        .worker_registry
        .get_all()
        .first()
        .map(|worker| worker.model_id().to_string())
        .expect("a registered worker");

    let requests = 20;
    for i in 0..requests {
        let payload = json!({
            "model": model,
            "messages": [{ "role": "user", "content": format!("Request {i}") }],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "request {i} is placed");
    }
    let shares: Vec<usize> = recorders
        .iter()
        .map(|recorder| recorder.bodies().len())
        .collect();
    assert_eq!(
        shares.iter().sum::<usize>(),
        requests,
        "every request reached a worker: {shares:?}"
    );
    assert!(
        shares.iter().all(|share| *share > 0),
        "the document's round-robin gives each worker a share, where the label's passthrough \
         gives the first worker every request: {shares:?}"
    );
    for mut worker in workers {
        worker.stop().await;
    }
    ctx.shutdown().await;
}
