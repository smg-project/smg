//! Regular placement through the policy engine: under `--policy-engine
//! engine` each preset the engine serves routes every request of the HTTP
//! router, as it does under the legacy default, and so does each preset
//! written as a document and read from a file as `--policy-spec` is.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{PolicyConfig, PolicyEngineMode, RouterConfig};
use smg_policy_controller::spec::PolicySpec;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

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

/// Each preset the engine serves, written as a document and read from a
/// file as `--policy-spec` is, routes as the flag does: every request
/// placed, every one decided through the engine.
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
