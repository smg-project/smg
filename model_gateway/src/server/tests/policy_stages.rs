//! A host's own policy stages through the context built at startup.

use std::sync::Arc;

use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg_policy_controller::{
    document::PolicySpec,
    engine::{Candidate, Constraints, Filter, RequestFacts, WorkerId},
    stages::StageRegistry,
};
use smg_worker::identity::basic::{ConnectionMode, WorkerType};
use tempfile::tempdir;

use super::super::{build_app_context, Extensions, ServerConfig};
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    routers::{
        common::{
            placement::{select_single, PlacementInputs, Stated},
            snapshot::RoutingPool,
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};

const MODEL: &str = "sample-model";
const WORKERS: [&str; 3] = [
    "http://worker-1.invalid:8000",
    "http://worker-2.invalid:8000",
    "http://worker-3.invalid:8000",
];
/// The one worker the host's stage keeps.
const KEPT: &str = WORKERS[1];
/// The host's stage, as its table and its document name it.
const STAGE: &str = "keeps-one";

/// A host's own stage: of a request's candidates it keeps the worker at
/// [`KEPT`] alone.
#[derive(Debug)]
struct KeepsOne;

impl Filter for KeepsOne {
    fn name(&self) -> &'static str {
        STAGE
    }

    fn keep(&self, _: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        candidate.id == WorkerId::from_address(KEPT)
    }
}

/// The table of a host: the stock stages and its own.
fn host_stages() -> StageRegistry {
    let mut stages = StageRegistry::builtin();
    stages
        .filter(STAGE, || Box::new(KeepsOne))
        .expect("a free name");
    stages
}

/// A gateway that runs the policy controller under a supplied document. Its
/// one profile filters by the host's stage and picks at random among the
/// workers the stage kept.
fn configuration() -> RouterConfig {
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": "smg.policy/v1",
        "generation": 1,
        "profiles": { "default": { "filter": [STAGE], "pick": "random" } },
    }))
    .expect("a policy document");
    RouterConfig {
        policy_controller: true,
        policy_spec: Some(document),
        ..RouterConfig::default()
    }
}

/// The start of a gateway of `config` over what `host` added, where the
/// start is refused: the words it ends with.
async fn refusal(config: RouterConfig, host: Extensions) -> String {
    build_app_context(&ServerConfig::minimal(config), host)
        .await
        .expect_err("the start is refused")
}

fn register_workers(app: &AppContext) {
    for url in WORKERS {
        let worker = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(MODEL))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        app.worker_registry
            .register(Arc::new(worker))
            .expect("the worker registers");
    }
}

/// A host registers a stage of its own and its document names it. The
/// gateway starts, and the engine it started runs the host's stage: every
/// request lands on the one worker that stage keeps, where a random pick
/// over the three workers would spread them.
#[tokio::test]
async fn a_hosts_stage_named_by_its_document_places_the_gateways_requests() {
    let host = Extensions::new().policy_stages(host_stages());
    let app = build_app_context(&ServerConfig::minimal(configuration()), host)
        .await
        .expect("the host's table holds the stage");
    assert!(app.policy_registry.engine().is_some());
    register_workers(&app);
    for _ in 0..32 {
        let placed = select_single(
            &app.worker_registry,
            &app.policy_registry,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            PlacementInputs::default(),
            Stated::nothing,
        )
        .expect("the stage keeps one worker");
        assert_eq!(placed.worker().url(), KEPT);
    }
}

/// The same document on a gateway whose host added no table. The stock
/// table does not hold the stage, and the start ends by the stage's name in
/// the validator's own words.
#[tokio::test]
async fn the_stock_table_refuses_a_document_that_names_a_hosts_stage() {
    let config = configuration();
    let document = config.policy_spec.clone().expect("the supplied document");
    let refused = config
        .hosted_policy(&document, &StageRegistry::builtin())
        .expect_err("the stock table holds no such stage");
    let ended = refusal(config, Extensions::new()).await;
    assert_eq!(ended, format!("policy controller: {refused}"));
    assert!(ended.contains(&format!("{STAGE:?}")), "{ended}");
}

/// The rank option of the classic policies has no place in a document.
/// Beside a supplied document the start refuses it by its name.
#[tokio::test]
async fn the_rank_option_beside_a_supplied_document_is_refused_by_name() {
    let config = RouterConfig {
        dp_minimum_tokens_scheduler: true,
        ..configuration()
    };
    let host = Extensions::new().policy_stages(host_stages());
    let ended = refusal(config, host).await;
    assert!(
        ended.starts_with("policy controller: dp_minimum_tokens_scheduler = true: "),
        "{ended}"
    );
}

/// A refused document ends the start before the gateway reads its storage
/// hook or opens a store. The hook's file is missing here too, and the
/// start ends in the document's words.
#[tokio::test]
async fn a_refused_document_ends_the_start_ahead_of_the_stores() {
    let dir = tempdir().expect("a temporary directory");
    let hook = dir.path().join("missing.wasm");
    let config = RouterConfig {
        storage_hook_wasm_path: Some(hook.to_str().expect("a UTF-8 path").to_owned()),
        ..configuration()
    };
    // The same configuration under the host's table reaches the hook.
    let host = Extensions::new().policy_stages(host_stages());
    let at_the_hook = refusal(config.clone(), host).await;
    assert!(
        at_the_hook.starts_with("failed to read WASM storage hook at "),
        "{at_the_hook}"
    );
    let ended = refusal(config, Extensions::new()).await;
    assert!(ended.starts_with("policy controller: "), "{ended}");
}
