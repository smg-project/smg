//! Harmony encoding registration gating (issue #1608).
//!
//! Runs as its own test binary: `TIKTOKEN_ENCODINGS_BASE` points at a
//! nonexistent directory for every test here, so encoding loads fail
//! deterministically without touching the network and without racing other
//! test binaries' successful loads.

use std::sync::Arc;

use smg::{
    worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, Worker, WorkerType},
    workflow::{
        data::{WorkerKind, WorkerRegistrationMode, WorkerWorkflowData},
        steps::local::EnsureHarmonyEncodingStep,
    },
};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowInstanceId};

fn force_offline_vocab() {
    std::env::set_var("TIKTOKEN_ENCODINGS_BASE", "/nonexistent-tiktoken-encodings");
}

fn worker(model_id: &str, connection: ConnectionMode) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:1")
            .worker_type(WorkerType::Regular)
            .connection_mode(connection)
            .model(ModelCard::new(model_id))
            .build(),
    )
}

fn context(
    kind: WorkerKind,
    connection: ConnectionMode,
    workers: Vec<Arc<dyn Worker>>,
) -> WorkflowContext<WorkerWorkflowData> {
    WorkflowContext::new(
        WorkflowInstanceId::new(),
        WorkerWorkflowData {
            config: openai_protocol::worker::WorkerSpec::new("grpc://127.0.0.1:1"),
            registration_mode: WorkerRegistrationMode::Upsert,
            worker_kind: Some(kind),
            connection_mode: Some(connection),
            http2: None,
            http_client_handle: None,
            detected_runtime_type: None,
            discovered_labels: Default::default(),
            dp_info: None,
            model_cards: Vec::new(),
            workers: None,
            final_labels: Default::default(),
            app_context: None,
            actual_workers: Some(workers),
        },
    )
}

/// The step only concerns gRPC gpt-oss workers: HTTP mode (which proxies to
/// the backend), non-gpt-oss models, and external workers all skip.
#[tokio::test]
async fn skips_unless_worker_is_grpc_gpt_oss() {
    force_offline_vocab();
    let step = EnsureHarmonyEncodingStep;

    let mut http_gpt_oss = context(
        WorkerKind::Local,
        ConnectionMode::Http,
        vec![worker("openai/gpt-oss-20b", ConnectionMode::Http)],
    );
    assert!(matches!(
        step.execute(&mut http_gpt_oss).await,
        Ok(StepResult::Skip)
    ));

    let mut grpc_other_model = context(
        WorkerKind::Local,
        ConnectionMode::Grpc,
        vec![worker("deepseek-v4-flash", ConnectionMode::Grpc)],
    );
    assert!(matches!(
        step.execute(&mut grpc_other_model).await,
        Ok(StepResult::Skip)
    ));

    let mut external = context(
        WorkerKind::External,
        ConnectionMode::Grpc,
        vec![worker("openai/gpt-oss-20b", ConnectionMode::Grpc)],
    );
    assert!(matches!(
        step.execute(&mut external).await,
        Ok(StepResult::Skip)
    ));
}

/// A gRPC gpt-oss worker whose vocab cannot be loaded fails registration with
/// a typed error — never a panic — and the failure does not poison later
/// attempts.
#[tokio::test]
async fn fails_registration_for_grpc_gpt_oss_without_vocab() {
    force_offline_vocab();
    let step = EnsureHarmonyEncodingStep;
    let mut ctx = context(
        WorkerKind::Local,
        ConnectionMode::Grpc,
        vec![worker("openai/gpt-oss-20b", ConnectionMode::Grpc)],
    );

    let error = step
        .execute(&mut ctx)
        .await
        .expect_err("gpt-oss registration must fail without the Harmony vocab");
    assert!(
        error.to_string().contains("Harmony encoding"),
        "unexpected error: {error}"
    );

    let error = step
        .execute(&mut ctx)
        .await
        .expect_err("a later attempt must fail cleanly, not deadlock or panic");
    assert!(error.to_string().contains("Harmony encoding"));
}
