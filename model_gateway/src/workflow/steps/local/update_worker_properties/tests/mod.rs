mod label_refresh;

use std::{collections::HashMap, time::Duration};

use engine_zmq_client::{
    mock_engine::{connect_to_frontend, default_ready_response, MockEngine},
    EngineId,
};
use openai_protocol::{
    model_card::ModelCard,
    model_type::ModelType,
    worker::{
        HealthCheckConfig, HealthCheckUpdate, OverloadUpdate, WorkerSpec, WorkerType,
        WorkerUpdateRequest,
    },
};
use tracing_test::traced_test;
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    app_context::AppContext,
    routers::{
        common::kv_transfer::{connector_mode_for_worker, KvConnectorMode},
        grpc::{
            backend_client::BackendClient,
            zmq_client::{EosTokenIds, ZmqEngineClient},
        },
    },
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        registry::WorkerId,
        BasicWorker, ConnectionMode, RuntimeType,
    },
    workflow::steps::local::test_support::labels_of,
};

fn make_app_context(workers: &[Arc<dyn Worker>]) -> Arc<AppContext> {
    make_app_context_with(workers, None)
}

/// With the tool parsers the gateway reads with, when given: a parser
/// override is validated against them.
fn make_app_context_with(
    workers: &[Arc<dyn Worker>],
    tool_parser_factory: Option<tool_parser::ParserFactory>,
) -> Arc<AppContext> {
    use crate::{
        app_context::testing::app_context_for_tests, config::RouterConfig, worker::WorkerRegistry,
    };

    let router_config = RouterConfig::builder()
        .worker_startup_timeout_secs(1)
        .build_unchecked();
    let registry = Arc::new(WorkerRegistry::new());
    for w in workers {
        registry.register(Arc::clone(w)).unwrap();
    }
    app_context_for_tests(router_config, registry, tool_parser_factory)
}

/// Each worker's id and revision, as the find step captures them.
fn snapshot_of(app_context: &AppContext, workers: &[Arc<dyn Worker>]) -> Vec<(WorkerId, u64)> {
    workers
        .iter()
        .map(|worker| {
            let id = app_context
                .worker_registry
                .get_id_by_url(worker.url())
                .expect("the worker is registered");
            (id, worker.revision())
        })
        .collect()
}

fn make_context(
    app_context: Arc<AppContext>,
    worker: Arc<dyn Worker>,
    labels: HashMap<String, String>,
) -> WorkflowContext<WorkerUpdateWorkflowData> {
    let snapshot = snapshot_of(&app_context, std::slice::from_ref(&worker));
    let data = WorkerUpdateWorkflowData {
        config: WorkerUpdateRequest {
            priority: None,
            cost: None,
            labels: Some(labels),
            api_key: None,
            health: None,
        },
        worker_url: worker.url().to_string(),
        app_context: Some(app_context),
        workers_to_update: Some(vec![worker]),
        updated_workers: None,
        refresh_discovered_labels: false,
        fresh_engine_labels: None,
        snapshot: Some(snapshot),
    };
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// A `constrained_decoding` label sent in an update must name a policy,
/// as at registration: a typo is refused where the labels are merged,
/// and a policy goes through.
#[tokio::test]
async fn a_constrained_decoding_label_update_must_name_a_policy() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8000")
            .status(WorkerStatus::Ready)
            .build(),
    );
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut typo = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("constrained_decoding".to_string(), "sometimes".to_string())]),
    );
    let err = UpdateWorkerPropertiesStep
        .execute(&mut typo)
        .await
        .expect_err("a label that names no policy");
    assert!(err.to_string().contains("'requested' or 'always'"), "{err}");
    let mut always = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("constrained_decoding".to_string(), "always".to_string())]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut always).await);
}

fn make_refresh_context(
    app_context: Arc<AppContext>,
    worker: Arc<dyn Worker>,
    fresh: HashMap<String, String>,
) -> WorkflowContext<WorkerUpdateWorkflowData> {
    let snapshot = vec![(
        app_context
            .worker_registry
            .get_id_by_url(worker.url())
            .expect("the worker is registered"),
        worker.revision(),
    )];
    let data = WorkerUpdateWorkflowData {
        config: WorkerUpdateRequest {
            priority: None,
            cost: None,
            labels: None,
            api_key: None,
            health: None,
        },
        worker_url: worker.url().to_string(),
        app_context: Some(app_context),
        workers_to_update: Some(vec![worker]),
        updated_workers: None,
        refresh_discovered_labels: true,
        fresh_engine_labels: Some(fresh),
        snapshot: Some(snapshot),
    };
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// Prove the step's `result` is a success; a failure names the caller's line.
#[track_caller]
fn step_succeeds(result: WorkflowResult<StepResult>) {
    assert_eq!(result.unwrap(), StepResult::Success);
}

/// A gRPC decode worker over `labels`, with `discovered` on record as the
/// engine's read, and `dp` its rank and size when it is a DP rank.
fn grpc_decode_worker(
    dp: Option<(usize, usize)>,
    labels: HashMap<String, String>,
    discovered: HashMap<String, String>,
) -> Arc<dyn Worker> {
    let builder = BasicWorkerBuilder::new("grpc://decode:8000")
        .worker_type(WorkerType::Decode)
        .connection_mode(ConnectionMode::Grpc);
    let builder = match dp {
        Some((rank, size)) => builder.dp_config(rank, size),
        None => builder,
    };
    Arc::new(
        builder
            .labels(labels)
            .discovered_labels(discovered)
            .status(WorkerStatus::Ready)
            .build(),
    )
}

/// A DP flag only the operator set still sizes the engine's share on a
/// refresh: the fresh window is mapped with the width the worker's labels
/// use, so the rank follows the engine's new window and the record keeps
/// the share, not the whole window.
#[tokio::test]
async fn a_rank_whose_dp_flag_only_the_operator_set_follows_the_engines_window() {
    // Registered dp-aware over an engine that reports no DP label, with
    // `enable_dp_attention` set in the worker's config: the rank carries
    // the share of the engine's 256, and the record has it in rank space.
    let worker = grpc_decode_worker(
        Some((2, 8)),
        labels_of(&[
            ("max_running_requests", "32"),
            ("enable_dp_attention", "true"),
        ]),
        labels_of(&[("max_running_requests", "32")]),
    );
    assert_eq!(worker.max_running_requests(), Some(32));
    let app_ctx = make_app_context(std::slice::from_ref(&worker));

    // The engine restarts with a window of 512, still naming no DP label.
    let fresh = labels_of(&[("max_running_requests", "512")]);
    let mut refresh = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    let rebuilt = app_ctx
        .worker_registry
        .get_by_url(worker.url())
        .expect("the rank stays registered");
    assert_eq!(
        rebuilt.max_running_requests(),
        Some(64),
        "the rank's share follows the engine's new window"
    );
    assert_eq!(
        *rebuilt.metadata().discovered_labels,
        labels_of(&[("max_running_requests", "64")]),
        "the record keeps the share in rank space"
    );
    assert_eq!(
        rebuilt
            .metadata()
            .spec
            .labels
            .get("enable_dp_attention")
            .map(String::as_str),
        Some("true"),
        "the operator's flag stands"
    );
}

/// A label refresh replaces what discovery read before with what the
/// engine reports now, keeps the operator's labels and the worker's
/// status, records the new discovered set, and lets the decode admission
/// window follow; the same report again rebuilds nothing.
#[tokio::test]
async fn a_label_refresh_follows_the_engine_and_keeps_the_operators_labels() {
    let discovered = labels_of(&[("max_num_seqs", "256"), ("page_size", "128")]);
    let mut labels = discovered.clone();
    labels.insert("tier".to_string(), "gold".to_string());
    let worker = grpc_decode_worker(None, labels, discovered);
    assert_eq!(worker.max_running_requests(), Some(256));
    let app_ctx = make_app_context(std::slice::from_ref(&worker));

    let fresh = labels_of(&[
        ("max_num_seqs", "8"),
        ("page_size", "1152"),
        ("supports_vision", "true"),
    ]);
    let mut refresh =
        make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh.clone());
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    let rebuilt = app_ctx
        .worker_registry
        .get_by_url("grpc://decode:8000")
        .expect("the worker stays registered");
    assert_eq!(
        rebuilt.metadata().spec.labels,
        labels_of(&[
            ("max_num_seqs", "8"),
            ("page_size", "1152"),
            ("supports_vision", "true"),
            ("tier", "gold"),
        ]),
        "engine labels follow the engine, the operator's label stays"
    );
    assert_eq!(*rebuilt.metadata().discovered_labels, fresh);
    assert_eq!(
        rebuilt.max_running_requests(),
        Some(8),
        "the decode admission window follows the engine"
    );
    assert_eq!(
        rebuilt.status(),
        WorkerStatus::Ready,
        "a refresh keeps the worker routable"
    );

    // The same report again changes nothing: no rebuild, no updated worker.
    let revision = rebuilt.revision();
    let mut again = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&rebuilt), fresh);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut again).await);
    assert_eq!(
        app_ctx
            .worker_registry
            .get_by_url("grpc://decode:8000")
            .unwrap()
            .revision(),
        revision,
        "an unchanged engine report rebuilds nothing"
    );
    assert!(again.data.updated_workers.as_ref().unwrap().is_empty());
}

/// The engine's KV transfer metadata arrives in its report under label
/// names: it is never a label (as at registration), a reported role fills
/// a field the operator left unset while a configured connector stands,
/// and the engine's id is the worker's live id from then on, rebuilt or
/// not.
#[tokio::test]
async fn a_label_refresh_keeps_the_engines_kv_metadata_out_of_the_labels_and_in_its_fields() {
    let discovered = labels_of(&[("max_num_seqs", "256")]);
    let mut spec = WorkerSpec::new("grpc://decode:8000");
    spec.worker_type = WorkerType::Decode;
    spec.connection_mode = ConnectionMode::Grpc;
    spec.kv_connector = Some("NixlConnector".to_string());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::from_spec(spec)
            .labels(discovered.clone())
            .discovered_labels(discovered)
            .status(WorkerStatus::Ready)
            .build(),
    );
    assert_eq!(worker.kv_engine_id(), None);
    let app_ctx = make_app_context(std::slice::from_ref(&worker));

    let fresh = labels_of(&[
        ("max_num_seqs", "8"),
        ("kv_connector", "MooncakeConnector"),
        ("kv_role", "kv_consumer"),
        ("kv_engine_id", "engine-late"),
    ]);
    let mut refresh = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    let rebuilt = app_ctx
        .worker_registry
        .get_by_url("grpc://decode:8000")
        .expect("the worker stays registered");
    assert_eq!(
        rebuilt.metadata().spec.labels,
        labels_of(&[("max_num_seqs", "8")]),
        "the KV metadata is not a label"
    );
    assert_eq!(
        *rebuilt.metadata().discovered_labels,
        labels_of(&[("max_num_seqs", "8")])
    );
    assert_eq!(
        rebuilt.metadata().spec.kv_connector.as_deref(),
        Some("NixlConnector"),
        "a configured connector stands"
    );
    assert_eq!(
        rebuilt.metadata().spec.kv_role.as_deref(),
        Some("kv_consumer"),
        "a role the operator left unset follows the engine"
    );
    assert_eq!(
        rebuilt.kv_engine_id().as_deref(),
        Some("engine-late"),
        "the engine's id is the live id"
    );
    assert!(rebuilt.kv_engine_id_confirmed());

    // The same labels with a new engine id: no rebuild, the live id follows.
    let revision = rebuilt.revision();
    let fresh = labels_of(&[("max_num_seqs", "8"), ("kv_engine_id", "engine-restarted")]);
    let mut again = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&rebuilt), fresh);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut again).await);
    let current = app_ctx
        .worker_registry
        .get_by_url("grpc://decode:8000")
        .unwrap();
    assert_eq!(
        current.revision(),
        revision,
        "an unchanged engine report rebuilds nothing"
    );
    assert_eq!(current.kv_engine_id().as_deref(), Some("engine-restarted"));
}

/// An engine that reports a `constrained_decoding` naming no policy is
/// refused by the refresh as at registration: the job fails with the
/// reason and the worker keeps the labels it had.
#[tokio::test]
async fn a_label_refresh_refuses_a_constrained_decoding_that_names_no_policy() {
    let discovered = labels_of(&[("max_num_seqs", "256"), ("constrained_decoding", "always")]);
    let worker = grpc_decode_worker(None, discovered.clone(), discovered);
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let revision = worker.revision();
    let mut refresh = make_refresh_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        labels_of(&[("max_num_seqs", "8"), ("constrained_decoding", "sometimes")]),
    );
    let err = UpdateWorkerPropertiesStep
        .execute(&mut refresh)
        .await
        .expect_err("a report that names no policy");
    assert!(err.to_string().contains("'requested' or 'always'"), "{err}");
    let kept = app_ctx
        .worker_registry
        .get_by_url("grpc://decode:8000")
        .expect("the worker stays registered");
    assert_eq!(kept.revision(), revision, "the worker keeps its labels");
    assert_eq!(
        kept.metadata()
            .spec
            .labels
            .get("constrained_decoding")
            .map(String::as_str),
        Some("always")
    );
}

/// A key an update sets is the operator's from then on, even when the
/// value it sets is the one the engine reported: a later refresh keeps it
/// while the engine's own value moves.
#[tokio::test]
async fn a_label_set_by_an_update_stays_the_operators_through_a_refresh() {
    let discovered = labels_of(&[("max_num_seqs", "256"), ("page_size", "128")]);
    let worker = grpc_decode_worker(None, discovered.clone(), discovered);
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    // The update sets the window to the very value the engine had reported.
    let mut update = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("max_num_seqs".to_string(), "256".to_string())]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut update).await);
    let updated = app_ctx
        .worker_registry
        .get_by_url("grpc://decode:8000")
        .expect("the worker stays registered");
    assert!(
        !updated
            .metadata()
            .discovered_labels
            .contains_key("max_num_seqs"),
        "the key the update set left the discovered set"
    );
    // The engine restarts with the same window and another page size: the
    // worker is rebuilt, and the key the operator owns stays out of the
    // recorded report.
    let mut rebuild = make_refresh_context(
        Arc::clone(&app_ctx),
        Arc::clone(&updated),
        labels_of(&[("max_num_seqs", "256"), ("page_size", "1152")]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut rebuild).await);
    let rebuilt = app_ctx
        .worker_registry
        .get_by_url("grpc://decode:8000")
        .unwrap();
    assert_eq!(
        rebuilt
            .metadata()
            .spec
            .labels
            .get("page_size")
            .map(String::as_str),
        Some("1152"),
        "the page size rebuilt the worker"
    );
    assert!(
        !rebuilt
            .metadata()
            .discovered_labels
            .contains_key("max_num_seqs"),
        "the operator's key stays out of the recorded report through a rebuild"
    );
    // The engine now reports another window: the operator's stays.
    let mut refresh = make_refresh_context(
        Arc::clone(&app_ctx),
        Arc::clone(&rebuilt),
        labels_of(&[("max_num_seqs", "8"), ("page_size", "1152")]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    let refreshed = app_ctx
        .worker_registry
        .get_by_url("grpc://decode:8000")
        .unwrap();
    assert_eq!(
        refreshed.max_running_requests(),
        Some(256),
        "the operator's window stays through the refresh"
    );
    assert_eq!(
        refreshed
            .metadata()
            .spec
            .labels
            .get("page_size")
            .map(String::as_str),
        Some("1152"),
        "the engine's own keys still follow it"
    );
}

/// On an attention-DP rank worker the window label is the rank's share,
/// recorded in rank space at registration; a refresh maps the engine's
/// fresh labels through the same rule, so the share follows the engine's
/// new window, and an operator's hand-set window still wins.
#[tokio::test]
async fn a_rank_workers_window_follows_the_engine_in_rank_space() {
    // The engine runs attention DP over 8 ranks with a window of 256: a
    // rank worker carries the share, 32, and records it in rank space.
    let discovered = labels_for_rank(
        &labels_of(&[("max_running_requests", "256"), ("attn_dp_size", "8")]),
        Some((0, 8)),
        &HashMap::new(),
    );
    assert_eq!(
        discovered.get("max_running_requests").map(String::as_str),
        Some("32")
    );
    let worker = grpc_decode_worker(Some((0, 8)), discovered.clone(), discovered.clone());
    assert_eq!(worker.max_running_requests(), Some(32));
    // The operator pinned another rank's window by hand: registration kept
    // only the hand-set key (`labels_for_rank` with the operator's labels).
    let pinned = grpc_decode_worker(
        Some((1, 8)),
        labels_of(&[("max_num_seqs", "16"), ("attn_dp_size", "8")]),
        discovered,
    );
    assert_eq!(pinned.max_running_requests(), Some(16));
    let app_ctx = make_app_context(&[Arc::clone(&worker), Arc::clone(&pinned)]);

    // The engine restarts with a window of 512: the share is 64.
    let fresh = labels_of(&[("max_running_requests", "512"), ("attn_dp_size", "8")]);
    for rank in [&worker, &pinned] {
        let mut refresh =
            make_refresh_context(Arc::clone(&app_ctx), Arc::clone(rank), fresh.clone());
        step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    }
    let rebuilt = app_ctx
        .worker_registry
        .get_by_url(worker.url())
        .expect("the rank stays registered");
    assert_eq!(
        rebuilt.max_running_requests(),
        Some(64),
        "the rank's share follows the engine's new window"
    );
    assert_eq!(
        rebuilt
            .metadata()
            .discovered_labels
            .get("max_running_requests")
            .map(String::as_str),
        Some("64"),
        "the discovered record stays in rank space"
    );
    let pinned_now = app_ctx
        .worker_registry
        .get_by_url(pinned.url())
        .expect("the pinned rank stays registered");
    assert_eq!(
        pinned_now.max_running_requests(),
        Some(16),
        "the operator's hand-set window wins over the engine's share"
    );
    assert_eq!(
        pinned_now.revision(),
        pinned.revision(),
        "nothing to follow on the pinned rank: it is not rebuilt"
    );
}

/// A rank registered from a provisional read (no attention-DP label yet)
/// carries the whole window; once the engine's answer names the width,
/// the refresh moves the rank to its share instead of the whole figure.
#[tokio::test]
async fn a_rank_registered_from_a_provisional_read_takes_its_share_on_refresh() {
    let discovered = labels_of(&[("max_running_requests", "256")]);
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:8001")
            .worker_type(WorkerType::Decode)
            .connection_mode(ConnectionMode::Grpc)
            .dp_config(3, 8)
            .labels(discovered.clone())
            .discovered_labels(discovered)
            .status(WorkerStatus::Ready)
            .build(),
    );
    assert_eq!(worker.max_running_requests(), Some(256));
    let app_ctx = make_app_context(std::slice::from_ref(&worker));

    let fresh = labels_of(&[("max_running_requests", "256"), ("attn_dp_size", "8")]);
    let mut refresh = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    let rebuilt = app_ctx
        .worker_registry
        .get_by_url(worker.url())
        .expect("the rank stays registered");
    assert_eq!(
        rebuilt.max_running_requests(),
        Some(32),
        "the rank takes its share once the engine names its width"
    );
    assert_eq!(
        rebuilt
            .metadata()
            .spec
            .labels
            .get("attn_dp_size")
            .map(String::as_str),
        Some("8")
    );
}

/// A refresh writes back against the revision it read: a worker that
/// changed during the read (an update, a replacement) is left as it is,
/// and one that was removed is not re-added.
#[tokio::test]
async fn a_label_refresh_is_dropped_when_the_worker_changed_or_left_during_the_read() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode-b:8000")
            .connection_mode(ConnectionMode::Grpc)
            .labels(labels_of(&[("max_num_seqs", "256")]))
            .discovered_labels(labels_of(&[("max_num_seqs", "256")]))
            .status(WorkerStatus::Ready)
            .build(),
    );
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let fresh = labels_of(&[("max_num_seqs", "8")]);

    // Changed during the read: the snapshot's revision is stale.
    let mut refresh =
        make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh.clone());
    // An operator update lands while the read is in flight: the registry
    // replaces the worker and bumps its revision.
    let id = app_ctx
        .worker_registry
        .get_id_by_url("grpc://decode-b:8000")
        .unwrap();
    let patched: Arc<dyn Worker> = Arc::new(
        worker
            .as_any()
            .downcast_ref::<BasicWorker>()
            .expect("a BasicWorker")
            .clone(),
    );
    assert!(app_ctx.worker_registry.replace(&id, patched));
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    assert_eq!(
        app_ctx
            .worker_registry
            .get_by_url("grpc://decode-b:8000")
            .unwrap()
            .max_running_requests(),
        Some(256),
        "a stale snapshot is not written back"
    );
    assert!(refresh.data.updated_workers.as_ref().unwrap().is_empty());

    // Removed during the read, and registered again at the same address:
    // the refresh must neither bring the old worker back nor touch the new
    // one (a fresh registration starts at the revision the snapshot had).
    let current = app_ctx
        .worker_registry
        .get_by_url("grpc://decode-b:8000")
        .unwrap();
    let mut refresh = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&current), fresh);
    let id = app_ctx
        .worker_registry
        .get_id_by_url("grpc://decode-b:8000")
        .unwrap();
    app_ctx.worker_registry.remove(&id);
    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode-b:8000")
            .connection_mode(ConnectionMode::Grpc)
            .labels(labels_of(&[("max_num_seqs", "999")]))
            .status(WorkerStatus::Ready)
            .build(),
    );
    app_ctx.worker_registry.register(replacement).unwrap();
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);
    assert_eq!(
        app_ctx
            .worker_registry
            .get_by_url("grpc://decode-b:8000")
            .unwrap()
            .max_running_requests(),
        Some(999),
        "a worker registered again at the address keeps its own labels"
    );
    assert!(refresh.data.updated_workers.as_ref().unwrap().is_empty());
}

/// The cap is held where the labels are merged, not only at the admin
/// check: a worker already at the cap refuses an update with a new key
/// (updates queued together each passed the early check), while one that
/// rewrites keys the worker has goes through.
#[tokio::test]
async fn the_label_cap_holds_over_the_labels_the_step_merges() {
    let mut builder = BasicWorkerBuilder::new("http://full:8000").status(WorkerStatus::Ready);
    for i in 0..MAX_WORKER_LABELS {
        builder = builder.label(format!("k{i}"), "v");
    }
    let worker: Arc<dyn Worker> = Arc::new(builder.build());
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut over = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("k_new".to_string(), "v".to_string())]),
    );
    let err = UpdateWorkerPropertiesStep
        .execute(&mut over)
        .await
        .expect_err("a 129th label at merge time");
    assert!(err.to_string().contains("129 labels on worker"), "{err}");
    let mut rewrite = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("k0".to_string(), "w".to_string())]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut rewrite).await);
}

/// A worker a provider registered after its update was accepted keeps
/// its labels: the step refuses them against the worker it would change.
/// An update of what no provider publishes still applies to it.
#[tokio::test]
async fn an_update_of_labels_is_refused_for_a_worker_discovered_since_it_was_accepted() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8000")
            .status(WorkerStatus::Ready)
            .label(DISCOVERY_PROVIDER_LABEL, "kubernetes")
            .build(),
    );
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut labels = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("env".to_string(), "prod".to_string())]),
    );
    let err = UpdateWorkerPropertiesStep
        .execute(&mut labels)
        .await
        .expect_err("labels for a discovered worker");
    assert!(
        err.to_string()
            .contains("owned by the 'kubernetes' discovery provider"),
        "{err}"
    );
    let mut priority = make_context(Arc::clone(&app_ctx), Arc::clone(&worker), HashMap::new());
    priority.data.config.labels = None;
    priority.data.config.priority = Some(3);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut priority).await);
    let updated = priority.data.updated_workers.expect("the updated worker");
    assert_eq!(updated[0].priority(), 3);
}

/// An update writes back against the registration its find step found:
/// one replaced since (a provider registering the address) keeps what
/// replaced it, and one removed since is not brought back. The update
/// fails instead.
#[tokio::test]
async fn an_update_is_dropped_for_a_worker_replaced_or_removed_since_it_was_found() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8000")
            .status(WorkerStatus::Ready)
            .build(),
    );
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let registry = &app_ctx.worker_registry;
    let id = registry.get_id_by_url("http://w:8000").unwrap();
    let mut update = make_context(Arc::clone(&app_ctx), Arc::clone(&worker), HashMap::new());
    update.data.config.labels = None;
    update.data.config.priority = Some(7);
    let discovered: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8000")
            .status(WorkerStatus::Ready)
            .label(DISCOVERY_PROVIDER_LABEL, "file")
            .build(),
    );
    assert!(registry.replace(&id, discovered));
    let err = UpdateWorkerPropertiesStep
        .execute(&mut update)
        .await
        .expect_err("the worker was replaced since it was found");
    assert!(err.to_string().contains("replaced or removed"), "{err}");
    let current = registry.get_by_url("http://w:8000").unwrap();
    assert_eq!(
        current.metadata().spec.labels.get(DISCOVERY_PROVIDER_LABEL),
        Some(&"file".to_string()),
        "the provider's registration stands"
    );
    assert_ne!(current.priority(), 7);

    let mut update = make_context(Arc::clone(&app_ctx), Arc::clone(&current), HashMap::new());
    update.data.config.labels = None;
    update.data.config.priority = Some(7);
    registry.remove(&id);
    UpdateWorkerPropertiesStep
        .execute(&mut update)
        .await
        .expect_err("the worker was removed since it was found");
    assert!(
        registry.get_by_url("http://w:8000").is_none(),
        "a removed worker is not brought back"
    );
}

#[tokio::test]
async fn pd_update_preserves_connector_and_pair_descriptor() {
    for (connector, mode, protocol, transport, pair_key, expected_mode) in [
        (
            "MooncakeConnector",
            ConnectionMode::Http,
            None,
            "mooncake",
            "vllm/mooncake/page=16",
            KvConnectorMode::Mooncake {
                host: "prefill".to_string(),
                port: 9123,
                engine_id: Some("engine-1_dp1".to_string()),
            },
        ),
        (
            "NixlConnector",
            ConnectionMode::Grpc,
            Some("cluster-blue"),
            "nixl",
            "cluster-blue",
            KvConnectorMode::Nixl,
        ),
    ] {
        let mut spec = WorkerSpec::new("http://prefill:8000");
        spec.worker_type = WorkerType::Prefill;
        spec.runtime_type = RuntimeType::Vllm;
        spec.connection_mode = mode;
        spec.kv_connector = Some(connector.to_string());
        spec.kv_role = Some("kv_producer".to_string());
        spec.kv_engine_id = Some("engine-1".to_string());
        spec.bootstrap_port = Some(9123);
        spec.pairing_protocol = protocol.map(str::to_string);
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::from_spec(spec)
                .dp_config(1, 2)
                .label("block_size", "16")
                .label("tier", "silver")
                .label("max_num_seqs", "256")
                .priority(2)
                .cost(1.5)
                .health_config(HealthCheckConfig {
                    timeout_secs: 3,
                    check_interval_secs: 11,
                    ..Default::default()
                })
                .status(WorkerStatus::Ready)
                .build(),
        );
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            HashMap::from([
                ("tier".to_string(), "gold".to_string()),
                // A window set by the update is the worker's only window:
                // the discovered one under the other name goes.
                ("max_running_requests".to_string(), "512".to_string()),
            ]),
        );
        ctx.data.config.priority = Some(9);
        ctx.data.config.health = Some(HealthCheckUpdate {
            timeout_secs: Some(7),
            ..Default::default()
        });

        step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

        let updated = app_ctx
            .worker_registry
            .get_by_url("http://prefill:8000@1")
            .expect("worker stays registered");
        assert!(Arc::ptr_eq(
            &updated,
            &ctx.data.updated_workers.as_ref().unwrap()[0]
        ));
        assert_eq!(updated.base_url(), "http://prefill:8000");
        assert_eq!(updated.url(), "http://prefill:8000@1");
        assert_eq!(updated.dp_rank(), Some(1));
        assert_eq!(updated.dp_size(), Some(2));
        assert_eq!(
            updated.metadata().spec.kv_role.as_deref(),
            Some("kv_producer")
        );
        assert_eq!(connector_mode_for_worker(updated.as_ref()), expected_mode);
        assert_eq!(updated.pd_pairing().transport(), Some(transport));
        assert_eq!(updated.pd_pairing().explicit(), protocol);
        assert_eq!(updated.pd_pairing().key(), pair_key);
        assert_eq!(updated.metadata().spec.labels["tier"], "gold");
        assert_eq!(updated.metadata().spec.labels["block_size"], "16");
        assert_eq!(
            updated.metadata().spec.labels["max_running_requests"],
            "512"
        );
        assert_eq!(updated.metadata().spec.labels.get("max_num_seqs"), None);
        assert_eq!(updated.max_running_requests(), Some(512));
        assert_eq!(updated.priority(), 9);
        assert_eq!(updated.cost(), 1.5);
        assert_eq!(updated.metadata().health_config.timeout_secs, 7);
        assert_eq!(updated.metadata().health_config.check_interval_secs, 11);
        assert_eq!(updated.status(), WorkerStatus::Ready);
    }
}

/// A PATCH rebuilds the worker. The breaker it was registered with
/// (gateway defaults plus overrides) and the breaker's live state must
/// come through; a rebuild on the built-in defaults would silently
/// re-arm a worker the breaker had taken out of rotation.
#[tokio::test]
async fn update_keeps_the_circuit_breaker_config_and_state() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        success_threshold: 1,
        timeout_duration: Duration::from_secs(7),
        window_duration: Duration::from_secs(9),
    };
    assert_ne!(config, CircuitBreakerConfig::default());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8000")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(config.clone())
            .status(WorkerStatus::Ready)
            .build(),
    );
    worker.record_circuit_breaker_outcome(false);
    worker.record_circuit_breaker_outcome(false);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);

    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());
    UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

    let updated = &ctx.data.updated_workers.as_ref().unwrap()[0];
    let basic = updated.as_any().downcast_ref::<BasicWorker>().unwrap();
    assert_eq!(basic.circuit_breaker_config(), config);
    assert_eq!(updated.circuit_breaker_state(), CircuitState::Open);
}

#[tokio::test]
async fn pd_update_retains_live_kv_engine_state() {
    for current_id in [Some("engine-current"), None] {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill:8000")
                .worker_type(WorkerType::Prefill)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .kv_connector("MooncakeConnector")
                .kv_engine_id("engine-initial")
                .bootstrap_port(Some(9123))
                .status(WorkerStatus::Ready)
                .build(),
        );
        worker.refresh_kv_engine_id(current_id.map(str::to_string));
        worker.set_kv_engine_id_confirmed(false);
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

        UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

        let updated = &ctx.data.updated_workers.as_ref().unwrap()[0];
        assert_eq!(updated.kv_engine_id().as_deref(), current_id);
        assert!(!updated.kv_engine_id_confirmed());

        // A probe that began before the update still holds the old worker.
        worker.refresh_kv_engine_id(Some("engine-late".to_string()));
        worker.set_kv_engine_id_confirmed(true);
        assert_eq!(updated.kv_engine_id().as_deref(), Some("engine-late"));
        assert!(updated.kv_engine_id_confirmed());
        assert_eq!(
            connector_mode_for_worker(updated.as_ref()),
            KvConnectorMode::Mooncake {
                host: "prefill".to_string(),
                port: 9123,
                engine_id: Some("engine-late".to_string()),
            }
        );
    }
}

/// A label-only PATCH must not disturb a ZMQ worker's transport: the engine
/// handshakes once at its own startup, so losing the connected client (or
/// the addresses and engine count that define the sockets) would black-hole
/// every request until the engine group is restarted.
#[tokio::test]
async fn zmq_update_preserves_transport_state_and_the_live_client() {
    let dir = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::Vllm,
            Duration::from_secs(10)
        ),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let _engine = engine.expect("mock engine");
    let client = Arc::new(BackendClient::Zmq(client.expect("adapter connect")));

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(ep("ts0.ipc"))
            .connection_mode(ConnectionMode::Zmq)
            .runtime_type(RuntimeType::Vllm)
            .zmq_handshake_address(handshake.clone())
            .zmq_engine_group(2)
            .zmq_startup_timeout(Duration::from_secs(7200))
            .health_config(HealthCheckConfig::default())
            .overload(OverloadUpdate {
                waiting_requests: Some(8),
                token_usage: None,
            })
            .status(WorkerStatus::Ready)
            .build(),
    );
    let old = worker
        .as_any()
        .downcast_ref::<BasicWorker>()
        .expect("BasicWorker");
    old.backend_client
        .load_full()
        .set(Arc::clone(&client))
        .ok()
        .expect("cell empty");

    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("tier".to_string(), "gold".to_string())]),
    );

    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

    let updated = ctx.data.updated_workers.as_ref().expect("updated workers");
    assert_eq!(updated.len(), 1);
    let new = updated[0]
        .as_any()
        .downcast_ref::<BasicWorker>()
        .expect("BasicWorker");

    assert_eq!(
        new.metadata().spec.labels.get("tier"),
        Some(&"gold".to_string())
    );
    // The handshake bound an operator set survives the rebuild: the live value,
    // not the builder's default.
    assert_eq!(
        new.metadata().zmq_startup_timeout,
        Duration::from_secs(7200),
        "the ZMQ startup bound survives the update"
    );
    assert_eq!(
        new.metadata().spec.zmq_handshake_address.as_deref(),
        Some(handshake.as_str())
    );
    assert_eq!(new.metadata().zmq_engine_count(), Some(2));
    // The overload block survives a metadata-only update, and so does the
    // effective threshold resolved from it.
    assert_eq!(new.metadata().spec.overload.waiting_requests, Some(8));
    assert_eq!(new.metadata().overload.waiting_requests, Some(8));
    assert!(
        new.connect_signal_tx.is_some(),
        "ZMQ promotion stays event-driven after an update"
    );

    let adopted = new
        .backend_client
        .load_full()
        .get()
        .cloned()
        .expect("replacement must inherit the connected client");
    assert!(Arc::ptr_eq(&adopted, &client));
    assert_eq!(new.status(), WorkerStatus::Ready);

    // Neither side of a ZMQ replacement may materialize an HTTP client
    // it will never use.
    assert!(old.http_client.cell_is_empty());
    assert!(new.http_client.cell_is_empty());
}

/// A ZMQ worker whose link is up against a mock engine that advertised
/// `max_model_len` at its handshake: `recorded` is what registration read
/// from the model's files, `by_hand` the operator's own labels on top. The
/// engine and its socket directory ride along to stay alive.
async fn zmq_worker_with_its_link(
    max_model_len: u64,
    recorded: HashMap<String, String>,
    by_hand: HashMap<String, String>,
) -> (
    Arc<dyn Worker>,
    Arc<BackendClient>,
    MockEngine,
    tempfile::TempDir,
) {
    use engine_zmq_client::protocol::handshake::EngineCoreReadyResponse;

    let dir = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let ready = EngineCoreReadyResponse {
        max_model_len,
        ..default_ready_response()
    };
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::Vllm,
            Duration::from_secs(10)
        ),
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
    );
    let engine = engine.expect("mock engine");
    let client = Arc::new(BackendClient::Zmq(client.expect("adapter connect")));
    let mut labels = recorded.clone();
    labels.extend(by_hand);
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(ep("ts0.ipc"))
            .connection_mode(ConnectionMode::Zmq)
            .runtime_type(RuntimeType::Vllm)
            .zmq_handshake_address(handshake)
            .model(ModelCard::new("m"))
            .labels(labels)
            .discovered_labels(recorded)
            .status(WorkerStatus::Ready)
            .build(),
    );
    worker
        .as_any()
        .downcast_ref::<BasicWorker>()
        .expect("BasicWorker")
        .backend_client
        .load_full()
        .set(Arc::clone(&client))
        .ok()
        .expect("cell empty");
    (worker, client, engine, dir)
}

/// A ZMQ worker's label refresh (the connect signal's, once its handshake is
/// in) reads the window the link learned there: the rebuilt card carries it
/// for admission to read, what registration recorded stays, and the live
/// link is kept.
#[tokio::test]
async fn a_zmq_refresh_gives_the_card_the_handshake_window() {
    use crate::workflow::steps::local::RediscoverLabelsStep;

    let recorded = HashMap::from([(
        "constrained_decoding_start".to_string(),
        "first_token".to_string(),
    )]);
    let (worker, client, _engine, _dir) =
        zmq_worker_with_its_link(4096, recorded, HashMap::new()).await;
    assert_eq!(
        worker.context_length("m"),
        None,
        "registration read no window"
    );

    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(Arc::clone(&app_ctx), Arc::clone(&worker), HashMap::new());
    ctx.data.config.labels = None;
    ctx.data.refresh_discovered_labels = true;
    step_succeeds(RediscoverLabelsStep.execute(&mut ctx).await);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

    let updated = ctx.data.updated_workers.as_ref().expect("updated workers");
    assert_eq!(updated.len(), 1);
    assert_eq!(
        updated[0].context_length("m"),
        Some(4096),
        "the card carries the handshake window"
    );
    let labels = &updated[0].metadata().spec.labels;
    assert_eq!(
        labels.get("max_context_length").map(String::as_str),
        Some("4096")
    );
    assert_eq!(
        labels.get("constrained_decoding_start").map(String::as_str),
        Some("first_token"),
        "what registration recorded stays"
    );
    let new = updated[0]
        .as_any()
        .downcast_ref::<BasicWorker>()
        .expect("BasicWorker");
    let adopted = new
        .backend_client
        .load_full()
        .get()
        .cloned()
        .expect("replacement must inherit the connected client");
    assert!(Arc::ptr_eq(&adopted, &client), "the live link is kept");
}

/// A context window the operator set by hand on a ZMQ worker stays its only
/// window when the handshake names a wider one under the name the reader
/// prefers: the refresh merges as registration does (`merge_labels`), the
/// handshake's figure is not added beside the operator's limit, and the
/// worker is kept as it is.
#[tokio::test]
async fn a_zmq_refresh_keeps_the_window_the_operator_set() {
    use crate::workflow::steps::local::RediscoverLabelsStep;

    let recorded = HashMap::from([(
        "constrained_decoding_start".to_string(),
        "first_token".to_string(),
    )]);
    let by_hand = HashMap::from([("max_model_len".to_string(), "2048".to_string())]);
    let (worker, _client, _engine, _dir) = zmq_worker_with_its_link(4096, recorded, by_hand).await;

    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(Arc::clone(&app_ctx), Arc::clone(&worker), HashMap::new());
    ctx.data.config.labels = None;
    ctx.data.refresh_discovered_labels = true;
    step_succeeds(RediscoverLabelsStep.execute(&mut ctx).await);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

    let updated = ctx.data.updated_workers.clone().unwrap_or_default();
    let after = updated
        .first()
        .cloned()
        .unwrap_or_else(|| Arc::clone(&worker));
    let labels = &after.metadata().spec.labels;
    assert_eq!(
        labels.get("max_context_length"),
        None,
        "the handshake's figure is not added beside the operator's limit"
    );
    assert_eq!(
        labels.get("max_model_len").map(String::as_str),
        Some("2048")
    );
    assert!(
        updated.is_empty(),
        "nothing changed: the worker is kept as it is"
    );
}

/// An HTTP worker's replacement must adopt the same shared client — a
/// fresh client would silently detach it from the worker-client cache
/// entry the old worker was keeping alive.
#[tokio::test]
async fn http_update_adopts_the_materialized_shared_client() {
    let client = Arc::new(crate::server::http_client());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8080")
            .http_client(Arc::clone(&client))
            .status(WorkerStatus::Ready)
            .build(),
    );
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

    let updated = &ctx.data.updated_workers.as_ref().expect("updated workers")[0];
    let adopted = updated
        .http_client_handle_if_initialized()
        .expect("materialized client is adopted");
    assert!(Arc::ptr_eq(&adopted, &client));
}

/// The adopted client speaks one HTTP version; the replacement must keep
/// describing it correctly.
#[tokio::test]
async fn http_update_preserves_http2() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8080")
            .http_client(Arc::new(crate::server::http_client()))
            .http2(true)
            .status(WorkerStatus::Ready)
            .build(),
    );
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

    UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

    let updated = &ctx.data.updated_workers.as_ref().expect("updated workers")[0];
    assert!(updated.http2());
}

/// A worker over the cap through the labels discovered at registration
/// still takes an update that adds no key: a priority alone, or a rewrite
/// of keys it has; one that would add a key is refused.
#[tokio::test]
async fn a_worker_over_the_cap_takes_updates_that_add_no_key() {
    let mut builder = BasicWorkerBuilder::new("http://over:8000").status(WorkerStatus::Ready);
    for i in 0..MAX_WORKER_LABELS + 20 {
        builder = builder.label(format!("k{i}"), "v");
    }
    let worker: Arc<dyn Worker> = Arc::new(builder.build());
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut priority = make_context(Arc::clone(&app_ctx), Arc::clone(&worker), HashMap::new());
    priority.data.config.labels = None;
    priority.data.config.priority = Some(7);
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut priority).await);
    let mut rewrite = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("k3".to_string(), "w".to_string())]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut rewrite).await);
    let mut adds = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        HashMap::from([("k_new".to_string(), "v".to_string())]),
    );
    let err = UpdateWorkerPropertiesStep
        .execute(&mut adds)
        .await
        .expect_err("a new key on a worker over the cap");
    assert!(
        err.to_string().contains("149 labels on worker") && err.to_string().contains("1 new"),
        "{err}"
    );
}

/// Every rank is checked before any is replaced: when the second of two
/// ranks refuses the merge, the first keeps its labels too.
#[tokio::test]
async fn a_refused_rank_leaves_no_partial_update_behind() {
    let first: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://rank:8000@0")
            .status(WorkerStatus::Ready)
            .label("k0", "v")
            .build(),
    );
    let mut builder = BasicWorkerBuilder::new("http://rank:8000@1").status(WorkerStatus::Ready);
    for i in 0..MAX_WORKER_LABELS {
        builder = builder.label(format!("k{i}"), "v");
    }
    let second: Arc<dyn Worker> = Arc::new(builder.build());
    let app_ctx = make_app_context(&[Arc::clone(&first), Arc::clone(&second)]);
    let data = WorkerUpdateWorkflowData {
        config: WorkerUpdateRequest {
            priority: None,
            cost: None,
            labels: Some(HashMap::from([("k_new".to_string(), "v".to_string())])),
            api_key: None,
            health: None,
        },
        worker_url: "http://rank:8000".to_string(),
        app_context: Some(Arc::clone(&app_ctx)),
        workers_to_update: Some(vec![Arc::clone(&first), Arc::clone(&second)]),
        updated_workers: None,
        refresh_discovered_labels: false,
        fresh_engine_labels: None,
        snapshot: Some(snapshot_of(
            &app_ctx,
            &[Arc::clone(&first), Arc::clone(&second)],
        )),
    };
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    let err = UpdateWorkerPropertiesStep
        .execute(&mut context)
        .await
        .expect_err("the second rank refuses the merge");
    assert!(err.to_string().contains("http://rank:8000@1"), "{err}");
    let kept = app_ctx
        .worker_registry
        .get_by_url("http://rank:8000@0")
        .expect("the first rank is still registered");
    assert!(
        !kept.metadata().spec.labels.contains_key("k_new"),
        "the first rank was not updated before the second was checked"
    );
}

/// A worker as registration builds it: its card derived from the spec and
/// the labels, the labels on the worker.
fn registered(spec: WorkerSpec, labels: HashMap<String, String>) -> Arc<dyn Worker> {
    registered_worker(spec, labels, HashMap::new())
}

/// A worker registered from a discovery read: the labels are on record as
/// the engine's, so a refresh lets them follow it.
fn discovered(spec: WorkerSpec, labels: HashMap<String, String>) -> Arc<dyn Worker> {
    registered_worker(spec, labels.clone(), labels)
}

/// A worker as registration builds it: the model card derived from the
/// labels, with `discovered_labels` as the record of what discovery read.
fn registered_worker(
    spec: WorkerSpec,
    labels: HashMap<String, String>,
    discovered_labels: HashMap<String, String>,
) -> Arc<dyn Worker> {
    use super::super::create_worker::{build_model_card, resolve_model_id};
    let model_id = resolve_model_id(&spec, &labels).to_string();
    let card = build_model_card(&model_id, &spec, &labels, &HashMap::new());
    Arc::new(
        BasicWorkerBuilder::from_spec(spec)
            .model(card)
            .labels(labels)
            .discovered_labels(discovered_labels)
            .status(WorkerStatus::Ready)
            .build(),
    )
}

/// The card registration derived from the engine's provisional read
/// follows a refresh as it follows a label update: a worker registered
/// while its engine was still loading reads with the engine's final
/// context window and vision flag once the refresh lands.
#[tokio::test]
async fn a_label_refresh_re_derives_the_model_card_from_the_engines_labels() {
    let worker = discovered(
        WorkerSpec::new("http://w:8000"),
        labels_of(&[("served_model_name", "m"), ("max_model_len", "4096")]),
    );
    assert_eq!(
        worker.context_length("m"),
        Some(4096),
        "as registered from the provisional read"
    );
    assert!(!worker.models().remove(0).model_type.supports_vision());
    let app_ctx = make_app_context(std::slice::from_ref(&worker));

    let fresh = labels_of(&[
        ("served_model_name", "m"),
        ("max_model_len", "32768"),
        ("supports_vision", "true"),
    ]);
    let mut refresh =
        make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh.clone());
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut refresh).await);

    let rebuilt = app_ctx
        .worker_registry
        .get_by_url("http://w:8000")
        .expect("the worker stays registered");
    assert_eq!(
        rebuilt.context_length("m"),
        Some(32768),
        "the context window follows the engine's final read"
    );
    let card = rebuilt.models().remove(0);
    assert_eq!(card.id, "m", "the id stays");
    assert!(
        card.model_type.supports_vision(),
        "the vision flag follows the engine's final read"
    );
    assert_eq!(*rebuilt.metadata().discovered_labels, fresh);
    assert_eq!(rebuilt.status(), WorkerStatus::Ready);
}

/// The card registration derived from the labels follows a label update:
/// the context window, the model type and architectures, the tokenizer
/// path and the vision flag, so request validation and routing see what
/// the worker's labels show. The card's id is the worker's identity in
/// the registry and stays.
#[tokio::test]
async fn the_model_card_follows_a_label_update() {
    let worker = registered(
        WorkerSpec::new("http://w:8000"),
        labels_of(&[
            ("served_model_name", "m"),
            ("max_model_len", "4096"),
            ("model_type", "llama"),
            ("architectures", r#"["LlamaForCausalLM"]"#),
        ]),
    );
    assert_eq!(worker.context_length("m"), Some(4096), "as registered");
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        labels_of(&[
            ("served_model_name", "other"),
            ("max_model_len", "8192"),
            ("model_type", "qwen3"),
            ("architectures", r#"["Qwen3ForCausalLM"]"#),
            ("supports_vision", "true"),
            ("tokenizer_path", "/models/m"),
        ]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

    let updated = app_ctx
        .worker_registry
        .get_by_url("http://w:8000")
        .expect("worker stays registered");
    assert_eq!(
        updated.context_length("m"),
        Some(8192),
        "the context window follows the label"
    );
    let card = updated.models().remove(0);
    assert_eq!(card.id, "m", "the id stays");
    assert_eq!(card.hf_model_type.as_deref(), Some("qwen3"));
    assert_eq!(card.architectures, ["Qwen3ForCausalLM"]);
    assert_eq!(card.tokenizer_path.as_deref(), Some("/models/m"));
    assert!(card.model_type.supports_vision());
    assert_eq!(updated.metadata().spec.labels["max_model_len"], "8192");
    assert_eq!(app_ctx.worker_registry.get_by_model("m").len(), 1);
}

/// A value the operator set by hand stands: a card with its own context
/// window keeps it when the label changes; the labels show the engine's
/// value all the same.
#[tokio::test]
async fn a_card_value_set_by_hand_stands_over_a_label_update() {
    let mut spec = WorkerSpec::new("http://w:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m").with_context_length(2048)));
    let worker = registered(spec, labels_of(&[("max_model_len", "4096")]));
    assert_eq!(worker.context_length("m"), Some(2048), "as registered");
    let app_ctx = make_app_context(std::slice::from_ref(&worker));
    let mut ctx = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        labels_of(&[("max_model_len", "8192")]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

    let updated = app_ctx
        .worker_registry
        .get_by_url("http://w:8000")
        .expect("worker stays registered");
    assert_eq!(updated.context_length("m"), Some(2048), "set by hand");
    assert_eq!(updated.metadata().spec.labels["max_model_len"], "8192");
}

/// A parser override a label update names must be one the gateway reads
/// with, as at registration: an unknown name is refused and the worker is
/// left as it was; a known one reaches the card.
#[tokio::test]
async fn a_parser_override_from_a_label_update_must_be_known() {
    let worker = registered(
        WorkerSpec::new("http://w:8000"),
        labels_of(&[("served_model_name", "m")]),
    );
    let app_ctx = make_app_context_with(
        std::slice::from_ref(&worker),
        Some(tool_parser::ParserFactory::default()),
    );
    let mut unknown = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        labels_of(&[("tool_parser", "definitely-not-a-parser")]),
    );
    let err = UpdateWorkerPropertiesStep
        .execute(&mut unknown)
        .await
        .expect_err("an unknown parser override");
    assert!(err.to_string().contains("definitely-not-a-parser"), "{err}");
    let kept = app_ctx
        .worker_registry
        .get_by_url("http://w:8000")
        .expect("worker stays registered");
    assert!(Arc::ptr_eq(&kept, &worker), "the worker was left as it was");

    let mut known = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        labels_of(&[("tool_parser", "json")]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut known).await);
    let updated = app_ctx
        .worker_registry
        .get_by_url("http://w:8000")
        .expect("worker stays registered");
    assert_eq!(
        updated.models().remove(0).tool_parser.as_deref(),
        Some("json")
    );
}

/// An external worker's cards come from the provider's model list, not
/// from labels: a label update leaves them as they are, whether the
/// worker serves one model or several.
#[tokio::test]
async fn an_external_workers_cards_are_left_as_they_are() {
    for cards in [
        vec![ModelCard::new("a")],
        vec![ModelCard::new("a"), ModelCard::new("b")],
    ] {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.example.com/v1")
                .runtime_type(RuntimeType::External)
                .models(cards.clone())
                .status(WorkerStatus::Ready)
                .build(),
        );
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            labels_of(&[
                ("is_embedding", "true"),
                ("model_type", "chat"),
                ("max_model_len", "8192"),
            ]),
        );
        step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);

        let updated = app_ctx
            .worker_registry
            .get_by_url("https://api.example.com/v1")
            .expect("worker stays registered");
        assert_eq!(updated.models().len(), cards.len());
        for card in updated.models() {
            assert_eq!(card.model_type, ModelType::LLM, "{}", card.id);
            assert_eq!(card.hf_model_type, None, "{}", card.id);
            assert_eq!(card.context_length, None, "{}", card.id);
        }
        assert_eq!(updated.metadata().spec.labels["max_model_len"], "8192");
    }
}

/// A parser override a label update brings in is checked against the
/// other workers of the model, as at registration: the warning names the
/// peer whose override differs, not the worker itself, whose old card is
/// still registered while the new one is checked.
#[tokio::test]
#[traced_test]
async fn a_conflicting_parser_override_from_a_label_update_is_warned_about() {
    let peer = registered(
        WorkerSpec::new("http://peer:8000"),
        labels_of(&[("served_model_name", "m"), ("tool_parser", "json")]),
    );
    let worker = registered(
        WorkerSpec::new("http://w:8000"),
        labels_of(&[("served_model_name", "m"), ("tool_parser", "json")]),
    );
    let app_ctx = make_app_context_with(
        &[Arc::clone(&peer), Arc::clone(&worker)],
        Some(tool_parser::ParserFactory::default()),
    );
    let mut ctx = make_context(
        Arc::clone(&app_ctx),
        Arc::clone(&worker),
        labels_of(&[("tool_parser", "pythonic")]),
    );
    step_succeeds(UpdateWorkerPropertiesStep.execute(&mut ctx).await);
    assert!(logs_contain("conflicting parser overrides"));
    assert!(logs_contain("existing_worker=\"http://peer:8000\""));
    assert!(!logs_contain("existing_worker=\"http://w:8000\""));
    let updated = app_ctx
        .worker_registry
        .get_by_url("http://w:8000")
        .expect("worker stays registered");
    assert_eq!(
        updated.models().remove(0).tool_parser.as_deref(),
        Some("pythonic")
    );
}
