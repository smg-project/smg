//! The DP read's cases.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use http::StatusCode;
use serde_json::json;
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    health::overload::OverloadThresholds,
    identity::basic::{RuntimeType, Worker, WorkerType},
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    labels::flat_labels,
    load::monitor::poll_group_once_for_test,
    model_card::ModelCard,
    settings::RegistrationSettings,
    test_support::{labels_of, LoadReaders, RefusedAddress, ScriptedConnector},
    worker::{WorkerGroupKey, WorkerModels, WorkerStatus},
    workflow::{
        create_worker::CreateLocalWorkerStep,
        host::testing::{registration_settings, serve_router, ProbingHost},
        metadata::ServerInfo,
        runtimes::HTTP_RUNTIMES,
    },
};

/// The shape `labels` hold, for labels that name a usable DP size.
fn shape(labels: &HashMap<String, String>) -> DpInfo {
    dp_info_from_labels(labels)
        .expect("a usable DP size")
        .expect("a DP size")
}

#[test]
fn dp_info_from_labels_reads_dp_size_and_model() {
    for widths in [
        vec![("dp_size", "4")],
        vec![("dp_size", "1"), ("attn_dp_size", "4")],
        vec![("attn_dp_size", "4")],
        vec![("dp_size", "2"), ("attn_dp_size", "2")],
    ] {
        let mut metadata = labels_of(&widths);
        metadata.insert("served_model_name".to_string(), "m".to_string());
        let info = shape(&metadata);
        assert_eq!(info.dp_size, 4);
        assert_eq!(info.model_id, "m");
    }
}

#[test]
fn dp_info_from_labels_model_path_fallback() {
    let info = shape(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo")]));
    assert_eq!(info.model_id, "repo");
    let info = shape(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo/")]));
    assert_eq!(info.model_id, "repo");
}

#[test]
fn dp_info_from_labels_unknown_model_when_absent() {
    let info = shape(&labels_of(&[("dp_size", "2")]));
    assert_eq!(info.model_id, UNKNOWN_MODEL_ID);
}

#[test]
fn zmq_dp_info_uses_the_spec_dp_size() {
    // Nothing to read over ZMQ: an ungrouped worker is dp_size 1 and a
    // grouped one reports its spec size, so the creation gets a dp_info to
    // expand.
    let mut spec = WorkerSpec::new("ipc:///tmp/smg-zmq/engine.ipc");
    let info = zmq_dp_info(&spec, &labels_of(&[("model_id", "m")]));
    assert_eq!(info.dp_size, 1);
    assert_eq!(info.model_id, "m");

    spec.dp_size = Some(4);
    assert_eq!(zmq_dp_info(&spec, &HashMap::new()).dp_size, 4);

    spec.dp_size = Some(0);
    assert_eq!(zmq_dp_info(&spec, &HashMap::new()).dp_size, 1);
}

/// Labels that name no DP size hold no shape, and that is no error: an
/// engine of one rank, or one that takes no rank pin, names none.
#[test]
fn dp_info_from_labels_holds_none_where_no_dp_size_is_named() {
    for labels in [labels_of(&[]), labels_of(&[("served_model_name", "m")])] {
        assert!(
            matches!(dp_info_from_labels(&labels), Ok(None)),
            "{labels:?}"
        );
    }
}

/// A DP size that is named and cannot be used is an error, which names what
/// was wrong with it.
#[test]
fn dp_info_from_labels_rejects_a_dp_size_it_cannot_use() {
    assert!(dp_info_from_labels(&labels_of(&[("dp_size", "0")])).is_err());
    assert!(dp_info_from_labels(&labels_of(&[("dp_size", "abc")])).is_err());
    assert!(dp_info_from_labels(&labels_of(&[("dp_size", "1"), ("attn_dp_size", "0")])).is_err());
    assert!(dp_info_from_labels(&labels_of(&[("dp_size", "1"), ("attn_dp_size", "abc")])).is_err());
    let overflow = labels_of(&[("dp_size", &usize::MAX.to_string()), ("attn_dp_size", "2")]);
    assert!(dp_info_from_labels(&overflow).is_err());
}

/// The DP width reads the same from a server info's flattened labels,
/// whichever way the server splits it.
#[test]
fn dp_info_from_labels_reads_a_flattened_server_info() {
    for (dp_size, attn_dp_size) in [(4, None), (1, Some(4))] {
        assert_eq!(shape(&server_info_labels(dp_size, attn_dp_size)).dp_size, 4);
    }
}

/// The labels the metadata read leaves of a server info that names these
/// sizes and a model path.
fn server_info_labels(dp_size: usize, attn_dp_size: Option<usize>) -> HashMap<String, String> {
    let body = serde_json::json!({
        "model_path": "org/model-a",
        "dp_size": dp_size,
        "attn_dp_size": attn_dp_size,
    });
    let info: ServerInfo = serde_json::from_value(body).unwrap();
    flat_labels(&info)
}

/// A local registration on a host whose settings name nothing of the DP
/// read: there is no setting for it.
fn local_context(
    spec: WorkerSpec,
    connection_mode: ConnectionMode,
    runtime: Option<&str>,
) -> (
    Arc<ProbingHost>,
    WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>>,
) {
    let host = ProbingHost::new(false, ScriptedConnector::default());
    let mut data = WorkerRegistrationWorkflowData::new(
        spec,
        WorkerRegistrationMode::Upsert,
        Arc::clone(&host),
        WorkerJobAsker::Api,
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(connection_mode);
    data.detected_runtime_type = runtime.map(str::to_string);
    (host, WorkflowContext::new(WorkflowInstanceId::new(), data))
}

/// A runtime whose row of the table says its server info reports the DP
/// size (`true`), or one whose row says it does not.
fn a_runtime(reports_dp: bool) -> RuntimeType {
    HTTP_RUNTIMES
        .iter()
        .find(|row| row.reports_dp == reports_dp)
        .map(|row| row.runtime)
        .unwrap()
}

/// An HTTP worker at an address that refuses every connection, of
/// `runtime`: a step that asks it anything fails. The address is refused
/// while the reservation handed back lives, so a case binds it by name.
fn refused_http_context(
    runtime: Option<&str>,
) -> (
    RefusedAddress,
    WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>>,
) {
    let refused = RefusedAddress::reserve();
    let spec = WorkerSpec::new(format!("http://{}", refused.addr()));
    let ctx = local_context(spec, ConnectionMode::Http, runtime).1;
    (refused, ctx)
}

/// A load answer of one engine that holds a row for each of `ranks`.
fn loads_of(ranks: &[usize]) -> serde_json::Value {
    let rows: Vec<_> = ranks
        .iter()
        .map(|rank| json!({"dp_rank": rank, "num_running_reqs": 1}))
        .collect();
    json!({ "loads": rows })
}

/// An engine whose load route answers every read with `status` and `body`,
/// and the count of its reads. Any other path of it is a 404.
async fn engine_with_loads(
    status: StatusCode,
    body: serde_json::Value,
) -> (String, Arc<AtomicUsize>) {
    let reads = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&reads);
    let route = axum::routing::get(move || {
        counter.fetch_add(1, Ordering::SeqCst);
        let body = body.clone();
        async move { (status, axum::Json(body)) }
    });
    let router = axum::Router::new().route(ENGINE_LOADS_ROUTE, route);
    (serve_router(router).await, reads)
}

/// An HTTP worker at `base` of the runtime whose row reports the DP size,
/// whose server info counted `ranks`.
fn http_context(
    base: &str,
    ranks: usize,
) -> WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>> {
    let runtime = a_runtime(true);
    let spec = WorkerSpec::new(base);
    let mut ctx = local_context(spec, ConnectionMode::Http, Some(runtime.as_str())).1;
    ctx.data.discovered_labels = server_info_labels(ranks, None);
    ctx
}

/// A gRPC worker's shape is read from the labels the metadata read left, on
/// a host given no setting for it: no connection is asked for.
#[tokio::test]
async fn a_grpc_workers_shape_is_read_from_its_labels() {
    let (host, mut ctx) = local_context(
        WorkerSpec::new("grpc://worker-a:9000"),
        ConnectionMode::Grpc,
        None,
    );
    ctx.data.discovered_labels = labels_of(&[
        ("dp_size", "2"),
        ("attn_dp_size", "2"),
        ("served_model_name", "m"),
    ]);

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    let dp_info = ctx.data.dp_info.as_ref().unwrap();
    assert_eq!((dp_info.dp_size, dp_info.model_id.as_str()), (4, "m"));
    assert_eq!(host.connects(), 0);
}

/// A gRPC worker whose labels hold no DP size registers as one plain worker:
/// the step passes and leaves no shape.
#[tokio::test]
async fn a_grpc_worker_with_no_dp_size_passes_with_no_shape() {
    let (_, mut ctx) = local_context(
        WorkerSpec::new("grpc://worker-a:9000"),
        ConnectionMode::Grpc,
        None,
    );
    ctx.data.discovered_labels = labels_of(&[("served_model_name", "m")]);

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert!(ctx.data.dp_info.is_none());
}

/// A ZMQ worker's shape is its spec's, with the model its labels name.
#[tokio::test]
async fn a_zmq_workers_shape_is_its_specs() {
    let mut spec = WorkerSpec::new("ipc:///tmp/engine-a.sock");
    spec.dp_size = Some(4);
    let (_, mut ctx) = local_context(spec, ConnectionMode::Zmq, None);
    ctx.data.discovered_labels = labels_of(&[("model_id", "m")]);

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    let dp_info = ctx.data.dp_info.as_ref().unwrap();
    assert_eq!((dp_info.dp_size, dp_info.model_id.as_str()), (4, "m"));
}

/// An HTTP worker's shape is read from the labels the metadata read left of
/// its server info, for a runtime whose row says the size stands there, on
/// a host given no setting for it. A worker of several ranks is asked its
/// load route once, and keeps its shape where the answer holds a row for
/// every rank, in whatever order.
#[tokio::test]
async fn an_http_workers_shape_is_read_from_its_labels() {
    let (base, reads) = engine_with_loads(StatusCode::OK, loads_of(&[3, 0, 2, 1])).await;
    let mut ctx = http_context(&base, 1);
    ctx.data.discovered_labels = server_info_labels(2, Some(2));

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    let dp_info = ctx.data.dp_info.as_ref().unwrap();
    assert_eq!((dp_info.dp_size, dp_info.model_id.as_str()), (4, "model-a"));
    assert_eq!(reads.load(Ordering::SeqCst), 1);
}

/// An HTTP worker of several ranks is left with no shape, and so registers
/// as one plain worker, where one read of its load route does not hold a
/// row for every rank. The route is not served, or answers what is no load
/// report. Its answer names no rank, misses one, or is a roll-up whose rows
/// name their own workers. Or the read learned nothing: a server error, a
/// refused key, an address that refuses the connection. The step passes
/// each time, and the route is read once.
#[tokio::test]
async fn an_http_worker_keeps_no_shape_where_its_load_route_misses_a_rank() {
    let rollup = json!({"loads": [
        {"worker": "http://engine-a", "dp_rank": 0},
        {"worker": "http://engine-b", "dp_rank": 1},
    ]});
    for (status, body) in [
        (StatusCode::NOT_FOUND, json!({})),
        (StatusCode::METHOD_NOT_ALLOWED, json!({})),
        (StatusCode::NOT_IMPLEMENTED, json!({})),
        (StatusCode::OK, json!({"version": "1"})),
        (StatusCode::OK, loads_of(&[])),
        (StatusCode::OK, loads_of(&[0])),
        (StatusCode::OK, loads_of(&[1])),
        (StatusCode::OK, rollup),
        (StatusCode::INTERNAL_SERVER_ERROR, json!({})),
        (StatusCode::UNAUTHORIZED, json!({})),
    ] {
        let case = format!("{status} {body}");
        let (base, reads) = engine_with_loads(status, body).await;
        let mut ctx = http_context(&base, 2);

        let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success, "{case}");
        assert!(ctx.data.dp_info.is_none(), "{case}");
        assert_eq!(reads.load(Ordering::SeqCst), 1, "{case}");
    }

    let (_refused, mut ctx) = refused_http_context(Some(a_runtime(true).as_str()));
    ctx.data.discovered_labels = server_info_labels(2, None);

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert!(ctx.data.dp_info.is_none());
}

/// Only an HTTP worker that counts several ranks is asked its load route. A
/// gRPC worker and a ZMQ worker keep the shape their labels and their spec
/// hold, an HTTP worker of one rank keeps its width of one, and an HTTP
/// worker whose runtime reports no size has no shape. None of them reads
/// the route, which here would answer that it is not served.
#[tokio::test]
async fn only_an_http_worker_of_several_ranks_is_asked_its_load_route() {
    let (base, reads) = engine_with_loads(StatusCode::NOT_FOUND, json!({})).await;
    let address = base.strip_prefix("http://").unwrap();
    let four = labels_of(&[("dp_size", "4"), ("served_model_name", "m")]);

    let grpc_spec = WorkerSpec::new(format!("grpc://{address}"));
    let mut grpc = local_context(grpc_spec, ConnectionMode::Grpc, None).1;
    grpc.data.discovered_labels = four.clone();
    let mut zmq_spec = WorkerSpec::new(format!("tcp://{address}"));
    zmq_spec.dp_size = Some(4);
    let zmq = local_context(zmq_spec, ConnectionMode::Zmq, None).1;
    let one_rank = http_context(&base, 1);
    let silent = a_runtime(false);
    let silent_spec = WorkerSpec::new(base.as_str());
    let mut silent = local_context(silent_spec, ConnectionMode::Http, Some(silent.as_str())).1;
    silent.data.discovered_labels = four;

    for (mut ctx, ranks) in [
        (grpc, Some(4)),
        (zmq, Some(4)),
        (one_rank, Some(1)),
        (silent, None),
    ] {
        let case = format!("{:?}, {ranks:?}", ctx.data.connection_mode);

        let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success, "{case}");
        let shape = ctx.data.dp_info.as_ref().map(|info| info.dp_size);
        assert_eq!(shape, ranks, "{case}");
    }
    assert_eq!(reads.load(Ordering::SeqCst), 0);
}

/// An HTTP worker whose host keeps its own load source for it is not asked
/// its load route, and keeps its ranks: the host's source serves each
/// rank's load, whatever the route would answer. Here the route is not
/// served, and the source's calls fail: neither is read.
#[tokio::test]
async fn an_http_worker_whose_host_keeps_a_load_source_keeps_its_ranks_unasked() {
    let (base, reads) = engine_with_loads(StatusCode::NOT_FOUND, json!({})).await;
    let host = ProbingHost::new(false, ScriptedConnector::default().keeping_a_load_source());
    let mut ctx = http_context(&base, 2);
    ctx.data.host = Some(host);

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.dp_info.as_ref().map(|info| info.dp_size), Some(2));
    assert_eq!(reads.load(Ordering::SeqCst), 0);
}

/// Each outcome of the one read that leaves no shape has its own words, and
/// an answer with a row for every rank lacks nothing. The warning names the
/// engine with its credentials masked, the route, what was read and how the
/// rank workers come back.
#[test]
fn the_plain_worker_warning_names_the_engine_the_route_and_what_was_read() {
    let answer =
        |ranks: &[usize]| NativeLoads::Available(serde_json::from_value(loads_of(ranks)).unwrap());
    assert_eq!(missing_rank_loads(&answer(&[1, 0]), 2), None);
    for (loads, read) in [
        (answer(&[0]), "does not name every rank"),
        (NativeLoads::Absent, "is not served"),
        (NativeLoads::Unreadable, "answered what is no load report"),
        (NativeLoads::Inconclusive, "gave no answer"),
    ] {
        assert_eq!(missing_rank_loads(&loads, 2), Some(read));
    }

    assert_eq!(
        plain_worker_warning("http://user:key@engine-a:8000", 4, "is not served"),
        "http://***@engine-a:8000 reports 4 DP ranks, but its load route /v1/loads is not \
         served; registering as a single non-DP worker with no rank pinning (remove the worker \
         and register it again once the route answers by rank)"
    );
}

/// The gauges an engine of the runtime whose row reports the DP size serves
/// on its `/metrics` page: nine requests wait.
const GAUGES: &str = "sglang:num_running_reqs{model=\"m\"} 2.0\n\
     sglang:num_queue_reqs{model=\"m\"} 9.0\n\
     sglang:token_usage{model=\"m\"} 0.5\n";

/// An engine of two ranks that serves no load route registers as one plain
/// worker, and that worker is one the load monitor can read: one tick takes
/// its report from the engine's gauges and holds it back at the host's
/// threshold. Two rank workers would have had no report and no veto.
#[tokio::test]
async fn an_engine_with_no_load_route_registers_plain_and_is_held_back_at_the_threshold() {
    let gauges = axum::routing::get(|| async { GAUGES });
    let base = serve_router(axum::Router::new().route("/metrics", gauges)).await;
    let settings = RegistrationSettings {
        overload_defaults: OverloadThresholds {
            waiting_requests: Some(9),
            token_usage: None,
        },
        ..registration_settings()
    };
    let host = ProbingHost::with_settings(settings, ScriptedConnector::default());
    let mut spec = WorkerSpec::new(base.as_str());
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    let mut data = WorkerRegistrationWorkflowData::new(
        spec,
        WorkerRegistrationMode::Upsert,
        Arc::clone(&host),
        WorkerJobAsker::Api,
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.detected_runtime_type = Some(a_runtime(true).as_str().to_string());
    data.discovered_labels = server_info_labels(2, None);
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    let read = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();
    let created = CreateLocalWorkerStep.execute(&mut ctx).await.unwrap();

    assert_eq!((read, created), (StepResult::Success, StepResult::Success));
    let workers = ctx.data.actual_workers.take().unwrap();
    let registry = Arc::clone(host.worker_registry());
    let readers = Arc::new(LoadReaders::Blind);
    let monitor = Arc::new(WorkerMonitor::new(
        Arc::clone(&registry),
        readers,
        1,
        false,
        false,
    ));
    for worker in &workers {
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(worker)).unwrap();
    }
    let group = WorkerGroupKey {
        model_id: "m".to_string(),
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Http,
    };
    poll_group_once_for_test(&monitor, &group, Duration::from_secs(10), Instant::now()).await;

    let snapshot = monitor.load_snapshot();
    let count = |holds: &dyn Fn(&Arc<dyn Worker>) -> bool| {
        workers.iter().filter(|worker| holds(worker)).count()
    };
    let rank_workers = count(&|worker| worker.dp_rank().is_some());
    let reports = count(&|worker| snapshot.get(worker.url()).is_some());
    let held_back = count(&|worker| worker.is_overloaded());
    assert_eq!(
        (workers.len(), rank_workers, reports, held_back),
        (1, 0, 1, 1),
        "workers, rank workers among them, reports after one tick, held back at the threshold"
    );
    let waiting = snapshot
        .get(&base)
        .map(|report| report.total_waiting_reqs());
    assert_eq!(waiting, Some(9), "the report is the engine's gauges");
}

/// An HTTP worker is left with no shape where its runtime's row says the
/// size does not stand in its server info, where its runtime has no row, or
/// where no runtime was detected, whatever its labels hold: a host that
/// fronts an engine of four ranks answers with that engine's labels, and
/// registers as one worker.
#[tokio::test]
async fn an_http_worker_whose_runtime_reports_no_dp_size_has_no_shape_whatever_its_labels() {
    let silent = a_runtime(false);
    for runtime in [Some(silent.as_str()), Some("a-runtime-with-no-row"), None] {
        let (_refused, mut ctx) = refused_http_context(runtime);
        ctx.data.discovered_labels = server_info_labels(4, None);

        let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success, "{runtime:?}");
        assert!(ctx.data.dp_info.is_none(), "{runtime:?}");
    }
}

/// An HTTP worker whose labels hold no DP size, because its server info
/// could not be read or names none, registers as one plain worker: the step
/// passes and leaves no shape, and so it does for a size that cannot be
/// used. No outcome of the read fails the registration.
#[tokio::test]
async fn an_http_worker_with_no_usable_dp_size_passes_with_no_shape() {
    for labels in [
        labels_of(&[]),
        labels_of(&[("model_path", "org/model-a")]),
        labels_of(&[("dp_size", "0")]),
        labels_of(&[("dp_size", "2"), ("attn_dp_size", "abc")]),
    ] {
        let (_refused, mut ctx) = refused_http_context(Some(a_runtime(true).as_str()));
        ctx.data.discovered_labels = labels.clone();

        let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success, "{labels:?}");
        assert!(ctx.data.dp_info.is_none(), "{labels:?}");
    }
}

/// A worker that is not local is skipped.
#[tokio::test]
async fn a_worker_that_is_not_local_is_skipped() {
    let (_, mut ctx) = local_context(
        WorkerSpec::new("https://api.example.com"),
        ConnectionMode::Http,
        None,
    );
    ctx.data.worker_kind = Some(WorkerKind::External);

    assert_eq!(
        DiscoverDpInfoStep.execute(&mut ctx).await.unwrap(),
        StepResult::Skip
    );
    assert!(ctx.data.dp_info.is_none());
}
