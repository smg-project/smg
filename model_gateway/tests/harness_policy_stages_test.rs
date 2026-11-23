//! The test harness starts the policy engine the way a served gateway does.
//! Where a test's configuration runs the policy controller, a context the
//! harness builds starts the engine over the document in force and the table
//! of stages the test lent it. With the policy controller off the harness
//! starts no engine.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use openai_protocol::completion::CompletionRequest;
use serde_json::json;
use smg::{config::RouterConfig, middleware::TenantRequestMeta, tenant::TenantKey};
use smg_policy_controller::{
    document::PolicySpec,
    engine::{Candidate, RequestFacts, Scorer, Target, TieBand},
    stages::{Lifecycle, StageRegistry},
    tracking::{EventSet, Lease, Outcome, ReleaseReason, Side},
};
use smg_test_harness::{
    self as common,
    grpc_harness::{
        build_grpc_router, grpc_router_config, grpc_workers, serve_canned_worker, Canned,
    },
    mock_worker::MockWorkerConfig,
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};
use smg_worker::identity::basic::{RuntimeType, WorkerType};
use tower::ServiceExt;

const MODEL: &str = "harness-policy-stages-model";
/// The test's own stage, as its table and its document name it.
const STAGE: &str = "hears";

/// What the test's stage did and was told, in order.
type Heard = Arc<Mutex<Vec<&'static str>>>;

#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn note(heard: &Heard, word: &'static str) {
    heard.lock().expect("the stage heard").push(word);
}

/// A scorer of the test's own. It ranks every worker alike and takes every
/// word of a lease, so the engine leases each request it places under it.
#[derive(Debug)]
struct Hears(Heard);

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        note(&self.0, "dispatched");
    }

    fn on_completed(&self, _: &Lease, _: &Outcome) {
        note(&self.0, "ended");
    }

    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        note(&self.0, "released");
    }
}

impl Scorer for Hears {
    fn name(&self) -> &'static str {
        STAGE
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        note(&self.0, "scored");
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// An engine of the harness on a port of its own, which answers every
/// request.
async fn engine() -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 2,
    })
    .await
}

#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn completion() -> CompletionRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "prompt": "Hello Hello Hello",
        "max_tokens": 2,
        "stream": false,
    }))
    .expect("a completion request")
}

/// One completion over the gRPC route of a gateway of two workers, built by
/// the harness with the test's table lent. The configuration supplies a
/// document whose one profile ranks by the test's stage, and
/// `policy_controller` says whether the gateway runs it. The answer is what
/// the stage did and was told from the gateway's start on: `built` each time
/// the table built it, `scored` each time the engine asked it, `dispatched`
/// and `ended` for each dispatch and each end it heard, `released` for a
/// lease given up with neither.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn one_request(policy_controller: bool) -> Vec<&'static str> {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    stages
        .scorer(STAGE, move || {
            note(&kept, "built");
            Box::new(Hears(Arc::clone(&kept)))
        })
        .expect("a free name");
    // The table built the stage once as it took it, to read its name: what
    // is kept from here on is the gateway's doing.
    heard.lock().expect("the stage heard").clear();
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": "smg.policy/v1",
        "generation": 1,
        "profiles": { "default": { "score": [STAGE], "pick": "best" } },
    }))
    .expect("a policy document");
    let mut config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .policy_controller(policy_controller)
        .build_unchecked();
    config.policy_spec = Some(document);
    let ports = [engine().await, engine().await];
    let workers = grpc_workers(
        &ports.map(|port| (port, WorkerType::Regular)),
        RuntimeType::TokenSpeed,
        MODEL,
    );

    let status = common::with_policy_stages(stages, async {
        let router = build_grpc_router(config, MODEL, workers).await;
        let tenant = TenantRequestMeta::new(TenantKey::new("harness-policy-stages-tenant"));
        router
            .route_completion(None, &tenant, completion(), MODEL)
            .await
            .status()
    })
    .await;
    let said = heard.lock().expect("the stage heard").clone();
    assert_eq!(status, StatusCode::OK, "{said:?}");
    said
}

/// What the stage says of one request the engine placed under it: the start
/// built it for the engine, the engine asked it for the request's scores, and
/// it heard the request's dispatch and its end.
const PLACED: [&str; 4] = ["built", "scored", "dispatched", "ended"];

/// A test lends a table that holds a scorer of its own, and its configuration
/// runs the policy controller under a document that names the scorer. The
/// gateway the harness builds places a gRPC request by that scorer, which
/// hears the request's dispatch and its end.
#[tokio::test]
async fn a_scorer_the_test_lent_places_a_grpc_request_and_hears_its_dispatch_and_end() {
    assert_eq!(one_request(true).await, PLACED);
}

/// With the policy controller off the harness starts no engine, though the
/// test lends a table and supplies a document: the scorer is never built, and
/// the request is answered by the classic policy. The run's own value
/// ([`common::POLICY_CONTROLLER_ENV`]) turns the controller on for this
/// context as for every other, and the same request is then the engine's.
#[tokio::test]
async fn a_configuration_that_leaves_the_policy_controller_off_starts_no_engine() {
    let heard = one_request(false).await;
    if common::policy_controller_forced() {
        assert_eq!(heard, PLACED);
    } else {
        assert!(heard.is_empty(), "{heard:?}");
    }
}

/// Serves one request on a gateway of `config` whose `workers` went through
/// the gateway's registration steps, and answers what the gateway's registry
/// counted: the classic policy objects it made, and the requests that
/// reached a classic policy's place under the policy engine.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn made_by_a_gateway(config: RouterConfig, workers: Vec<MockWorkerConfig>) -> usize {
    let gateway = AppTestContext::new_with_config(config, workers).await;
    let request = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"text": "one request", "stream": false}).to_string(),
        ))
        .expect("a request");
    let answer = gateway.create_app().oneshot(request).await;
    let status = answer.expect("an answer").status();
    assert_eq!(status, StatusCode::OK, "the request is served");
    let made = gateway.app_context.policy_registry.classic_made_or_asked();
    gateway.shutdown().await;
    made
}

/// What the registries of two gateways counted, each started by the harness
/// with the policy controller as `policy_controller` says. The first
/// registered a regular worker and served a request. The second registered a
/// prefill and a decode worker under the two port names and served a pair's
/// request.
async fn counted_by_two_gateways(
    policy_controller: bool,
    [prefill, decode]: [u16; 2],
) -> [usize; 2] {
    let mut single = TestRouterConfig::round_robin(0);
    single.policy_controller = policy_controller;
    // A gateway's start and its one request make a large future: each is boxed.
    let single = Box::pin(made_by_a_gateway(
        single,
        vec![TestWorkerConfig::healthy(0)],
    ))
    .await;

    let url = |port: u16| format!("http://127.0.0.1:{port}");
    let mut pair = RouterConfig::builder()
        .prefill_decode_mode(vec![(url(prefill), None)], vec![url(decode)])
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .build_unchecked();
    pair.health_check.disable_health_check = true;
    pair.policy_controller = policy_controller;
    let workers = vec![
        TestWorkerConfig::prefill(prefill),
        TestWorkerConfig::decode(decode),
    ];
    [single, Box::pin(made_by_a_gateway(pair, workers)).await]
}

/// Where the policy controller runs, a gateway makes no classic policy
/// object and hands no request to one: not at its start, not when a regular,
/// a prefill or a decode worker registers, and not to serve a request or a
/// pair's request.
#[tokio::test]
async fn a_gateway_that_runs_the_policy_controller_makes_and_asks_no_classic_policy() {
    assert_eq!(
        counted_by_two_gateways(true, [19_960, 19_961]).await,
        [0, 0]
    );
}

/// The same steps with the policy controller off make the configured
/// policies: the default policy, and for a pair each side's own beside it.
#[tokio::test]
async fn a_gateway_that_leaves_the_policy_controller_off_makes_its_classic_policies() {
    let made = if common::policy_controller_forced() {
        [0, 0]
    } else {
        [1, 3]
    };
    assert_eq!(counted_by_two_gateways(false, [19_962, 19_963]).await, made);
}
