use std::{collections::HashMap, sync::Arc};

use openai_protocol::worker::{WorkerStatus, WorkerUpdateRequest};
use smg_worker::{identity::basic::Worker, jobs::WorkerJobAsker};
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    app_context::testing::create_test_app_context,
    routers::grpc::backend_client::BasicWorkerBuilder,
};

fn context(worker: Arc<dyn Worker>, refresh: bool) -> WorkflowContext<WorkerUpdateWorkflowData> {
    let data = WorkerUpdateWorkflowData {
        asker: if refresh {
            WorkerJobAsker::Health
        } else {
            WorkerJobAsker::Api
        },
        config: WorkerUpdateRequest {
            priority: None,
            cost: None,
            labels: None,
            api_key: None,
            health: None,
        },
        worker_url: worker.url().to_string(),
        app_context: Some(create_test_app_context()),
        workers_to_update: Some(vec![worker]),
        updated_workers: None,
        refresh_discovered_labels: refresh,
        fresh_engine_labels: None,
        snapshot: None,
    };
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// A ZMQ worker whose link is not up has nothing to re-read yet: the
/// labels stay, and the next transition into service refreshes again.
#[tokio::test]
async fn a_zmq_worker_without_its_link_is_skipped() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("ipc:///tmp/engine.sock")
            .connection_mode(ConnectionMode::Zmq)
            .status(WorkerStatus::Ready)
            .build(),
    );
    let mut ctx = context(worker, true);
    assert_eq!(
        RediscoverLabelsStep.execute(&mut ctx).await.unwrap(),
        StepResult::Skip
    );
    assert!(ctx.data.fresh_engine_labels.is_none());
}

/// A ZMQ worker's refresh reads the window its link learned at the
/// handshake and keeps what registration read from the model's files.
#[tokio::test]
async fn a_zmq_worker_reads_its_handshake_window() {
    use std::time::Duration;

    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::handshake::EngineCoreReadyResponse,
        EngineId,
    };
    use openai_protocol::worker::RuntimeType;

    use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

    let dir = tempfile::tempdir().unwrap();
    let ep = |socket: &str| format!("ipc://{}", dir.path().join(socket).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let ready = EngineCoreReadyResponse {
        max_model_len: 4096,
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
    let _engine = engine.expect("mock engine");
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(ep("worker.ipc"))
            .connection_mode(ConnectionMode::Zmq)
            .runtime_type(RuntimeType::Vllm)
            .engine_connection(BackendClient::Zmq(client.expect("adapter connect")))
            .discovered_labels(HashMap::from([(
                "constrained_decoding_start".to_string(),
                "first_token".to_string(),
            )]))
            .status(WorkerStatus::Ready)
            .build(),
    );
    let mut ctx = context(worker, true);
    assert_eq!(
        RediscoverLabelsStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let fresh = ctx.data.fresh_engine_labels.expect("the link's labels");
    assert_eq!(
        fresh.get("max_context_length").map(String::as_str),
        Some("4096"),
        "the handshake's window"
    );
    assert_eq!(
        fresh.get("constrained_decoding_start").map(String::as_str),
        Some("first_token"),
        "what registration read stays"
    );
    assert_eq!(
        fresh.len(),
        2,
        "nothing else of the handshake's report: {fresh:?}"
    );
}

#[tokio::test]
async fn without_a_refresh_request_the_step_is_skipped() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:1")
            .connection_mode(ConnectionMode::Grpc)
            .status(WorkerStatus::Ready)
            .build(),
    );
    let mut ctx = context(worker, false);
    assert_eq!(
        RediscoverLabelsStep.execute(&mut ctx).await.unwrap(),
        StepResult::Skip
    );
    assert!(ctx.data.fresh_engine_labels.is_none());
}

/// A read that fails leaves nothing for the update step: the worker keeps
/// its labels, and the step reports a retryable error.
#[tokio::test]
async fn a_failed_read_leaves_the_worker_unchanged() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:1")
            .connection_mode(ConnectionMode::Grpc)
            .label("max_num_seqs", "256")
            .status(WorkerStatus::Ready)
            .build(),
    );
    let mut ctx = context(Arc::clone(&worker), true);
    let err = RediscoverLabelsStep
        .execute(&mut ctx)
        .await
        .expect_err("nothing listens on the port");
    assert!(RediscoverLabelsStep.is_retryable(&err));
    assert!(ctx.data.fresh_engine_labels.is_none());
    assert_eq!(
        worker
            .metadata()
            .spec
            .labels
            .get("max_num_seqs")
            .map(String::as_str),
        Some("256")
    );
}
