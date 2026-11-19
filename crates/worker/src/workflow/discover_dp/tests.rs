//! The DP read's cases.

use std::sync::Arc;

use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    identity::basic::RuntimeType,
    jobs::WorkerRegistrationMode,
    labels::flat_labels,
    test_support::{labels_of, RefusedAddress, ScriptedConnector},
    workflow::{
        host::testing::{serve_json_routes, ProbingHost},
        metadata::ServerInfo,
        runtimes::HTTP_RUNTIMES,
    },
};

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
        let info = dp_info_from_labels(&metadata).unwrap();
        assert_eq!(info.dp_size, 4);
        assert_eq!(info.model_id, "m");
    }
}

#[test]
fn dp_info_from_labels_model_path_fallback() {
    let info =
        dp_info_from_labels(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo")])).unwrap();
    assert_eq!(info.model_id, "repo");
    let info =
        dp_info_from_labels(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo/")])).unwrap();
    assert_eq!(info.model_id, "repo");
}

#[test]
fn dp_info_from_labels_unknown_model_when_absent() {
    let info = dp_info_from_labels(&labels_of(&[("dp_size", "2")])).unwrap();
    assert_eq!(info.model_id, UNKNOWN_MODEL_ID);
}

#[test]
fn zmq_dp_info_uses_the_spec_dp_size() {
    // Nothing to read over ZMQ: an ungrouped worker is dp_size 1 and a
    // grouped one reports its spec size, so the creation gets a dp_info to
    // check where the DP read is on.
    let mut spec = WorkerSpec::new("ipc:///tmp/smg-zmq/engine.ipc");
    let info = zmq_dp_info(&spec, &labels_of(&[("model_id", "m")]));
    assert_eq!(info.dp_size, 1);
    assert_eq!(info.model_id, "m");

    spec.dp_size = Some(4);
    assert_eq!(zmq_dp_info(&spec, &HashMap::new()).dp_size, 4);

    spec.dp_size = Some(0);
    assert_eq!(zmq_dp_info(&spec, &HashMap::new()).dp_size, 1);
}

#[test]
fn dp_info_from_labels_rejects_missing_or_zero_dp_size() {
    assert!(dp_info_from_labels(&labels_of(&[])).is_err());
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
        let body = serde_json::json!({
            "model_path": "org/model-a",
            "dp_size": dp_size,
            "attn_dp_size": attn_dp_size,
        });
        let info: ServerInfo = serde_json::from_value(body).unwrap();
        let dp_info = dp_info_from_labels(&flat_labels(&info)).unwrap();
        assert_eq!(dp_info.dp_size, 4);
    }
}

fn local_context(
    host: Arc<ProbingHost>,
    spec: WorkerSpec,
    connection_mode: ConnectionMode,
    runtime: Option<&str>,
) -> WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>> {
    let mut data = WorkerRegistrationWorkflowData::new(spec, WorkerRegistrationMode::Upsert, host);
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(connection_mode);
    data.detected_runtime_type = runtime.map(str::to_string);
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// A host whose settings turn the DP read on.
fn reading_dp() -> Arc<ProbingHost> {
    ProbingHost::reading_dp(ScriptedConnector::default())
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

/// With the read off in the host's settings nothing is read: a worker whose
/// address refuses every connection passes with no shape.
#[tokio::test]
async fn with_the_read_off_nothing_is_read() {
    let refused = RefusedAddress::reserve();
    let mut ctx = local_context(
        ProbingHost::new(false, ScriptedConnector::default()),
        WorkerSpec::new(format!("http://{}", refused.addr())),
        ConnectionMode::Http,
        Some(a_runtime(true).as_str()),
    );

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert!(ctx.data.dp_info.is_none());
}

/// A gRPC worker's shape is read from the labels the metadata read left: no
/// connection is asked for.
#[tokio::test]
async fn a_grpc_workers_shape_is_read_from_its_labels() {
    let host = reading_dp();
    let mut ctx = local_context(
        Arc::clone(&host),
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
    let mut ctx = local_context(
        reading_dp(),
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
    let mut ctx = local_context(reading_dp(), spec, ConnectionMode::Zmq, None);
    ctx.data.discovered_labels = labels_of(&[("model_id", "m")]);

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    let dp_info = ctx.data.dp_info.as_ref().unwrap();
    assert_eq!((dp_info.dp_size, dp_info.model_id.as_str()), (4, "m"));
}

/// An HTTP worker's shape is read from its server info, with the host's
/// worker client, for a runtime whose row says the size stands there.
#[tokio::test]
async fn an_http_workers_shape_is_read_from_its_server_info() {
    let base = serve_json_routes(vec![(
        "/server_info",
        http::StatusCode::OK,
        serde_json::json!({"model_path": "org/model-a", "dp_size": 2, "attn_dp_size": 2}),
    )])
    .await;
    let mut ctx = local_context(
        reading_dp(),
        WorkerSpec::new(base),
        ConnectionMode::Http,
        Some(a_runtime(true).as_str()),
    );

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    let dp_info = ctx.data.dp_info.as_ref().unwrap();
    assert_eq!((dp_info.dp_size, dp_info.model_id.as_str()), (4, "model-a"));
}

/// An HTTP worker is not asked where its runtime's row says the size does
/// not stand in its server info, where its runtime has no row, or where no
/// runtime was detected. It passes with no shape, though its address refuses
/// every connection.
#[tokio::test]
async fn an_http_worker_whose_runtime_reports_no_dp_size_is_not_asked() {
    let refused = RefusedAddress::reserve();
    let silent = a_runtime(false);
    for runtime in [Some(silent.as_str()), Some("a-runtime-with-no-row"), None] {
        let mut ctx = local_context(
            reading_dp(),
            WorkerSpec::new(format!("http://{}", refused.addr())),
            ConnectionMode::Http,
            runtime,
        );

        let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success, "{runtime:?}");
        assert!(ctx.data.dp_info.is_none(), "{runtime:?}");
    }
}

/// An HTTP read that fails, or a server info that names no DP size, fails
/// the step under its own id and leaves no shape.
#[tokio::test]
async fn an_http_read_that_fails_fails_the_step_under_its_id() {
    let refused = RefusedAddress::reserve();
    let sizeless = serve_json_routes(vec![(
        "/server_info",
        http::StatusCode::OK,
        serde_json::json!({"model_path": "org/model-a"}),
    )])
    .await;
    for (url, said) in [
        (
            format!("http://{}", refused.addr()),
            "Failed to get DP info: ",
        ),
        (sizeless, "Failed to get DP info: Invalid DP metadata from "),
    ] {
        let mut ctx = local_context(
            reading_dp(),
            WorkerSpec::new(url),
            ConnectionMode::Http,
            Some(a_runtime(true).as_str()),
        );

        let error = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap_err();

        assert!(
            matches!(
                &error,
                WorkflowError::StepFailed { step_id, message }
                    if *step_id == StepId::new(DISCOVER_DP_STEP) && message.starts_with(said)
            ),
            "{error:?}"
        );
        assert!(ctx.data.dp_info.is_none());
    }
}

/// A worker that is not local is skipped.
#[tokio::test]
async fn a_worker_that_is_not_local_is_skipped() {
    let mut ctx = local_context(
        reading_dp(),
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
