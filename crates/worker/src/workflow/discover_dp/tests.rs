//! The DP read's cases.

use std::sync::Arc;

use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    identity::basic::RuntimeType,
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    labels::flat_labels,
    test_support::{labels_of, RefusedAddress, ScriptedConnector},
    workflow::{host::testing::ProbingHost, metadata::ServerInfo, runtimes::HTTP_RUNTIMES},
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
/// a host given no setting for it. The worker is asked nothing: its address
/// refuses every connection.
#[tokio::test]
async fn an_http_workers_shape_is_read_from_its_labels() {
    let (_refused, mut ctx) = refused_http_context(Some(a_runtime(true).as_str()));
    ctx.data.discovered_labels = server_info_labels(2, Some(2));

    let result = DiscoverDpInfoStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    let dp_info = ctx.data.dp_info.as_ref().unwrap();
    assert_eq!((dp_info.dp_size, dp_info.model_id.as_str()), (4, "model-a"));
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
