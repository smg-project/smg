//! Behavioral tests for `AbortOnDropStream`'s deferred-abort mode, driven
//! through the real `VllmEngineClient` against an in-process mock engine.
//!
//! The deferral contract: with `defer_abort_until_first_item`, dropping the
//! stream before the backend produced any response holds the abort until the
//! first response arrives (the disaggregated decode leg uses this so a client
//! disconnect can never tear down a request mid-KV-handoff); dropping after
//! the first response — or without the deferral — aborts promptly, and
//! `mark_completed` suppresses the abort entirely.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::StreamExt;
use smg_grpc_client::{vllm_proto as proto, VllmEngineClient};
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Request, Response, Status};

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

#[derive(Default)]
struct MockState {
    /// Sender feeding the currently open generate stream.
    generate_tx: Mutex<Option<GenerateTx>>,
    /// Request ids passed to Abort.
    aborted: Mutex<Vec<String>>,
}

impl MockState {
    fn lock_tx(&self) -> std::sync::MutexGuard<'_, Option<GenerateTx>> {
        self.generate_tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn aborted_ids(&self) -> Vec<String> {
        self.aborted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[derive(Clone, Default)]
struct MockEngine {
    state: Arc<MockState>,
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for MockEngine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(4);
        *self.state.lock_tx() = Some(tx);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        self.state
            .aborted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(request.into_inner().request_ids);
        Ok(Response::new(proto::AbortResponse::default()))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_loads(
        &self,
        _request: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    type GetTokenizerStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<smg_grpc_client::common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    type SubscribeKvEventsStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<smg_grpc_client::common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }
}

async fn spawn_mock() -> Result<(String, Arc<MockState>), Box<dyn std::error::Error>> {
    let mock = MockEngine::default();
    let state = Arc::clone(&mock.state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    #[expect(
        clippy::disallowed_methods,
        reason = "mock server for one test; lives until the test process exits"
    )]
    tokio::spawn(
        Server::builder()
            .add_service(proto::vllm_engine_server::VllmEngineServer::new(mock))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    Ok((format!("http://{addr}"), state))
}

fn chunk() -> proto::GenerateResponse {
    proto::GenerateResponse::default()
}

/// Wait (bounded) until the mock has recorded at least one abort.
async fn wait_for_abort(state: &MockState) -> Vec<String> {
    for _ in 0..100 {
        let ids = state.aborted_ids();
        if !ids.is_empty() {
            return ids;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    state.aborted_ids()
}

fn generate_request(id: &str) -> proto::GenerateRequest {
    proto::GenerateRequest {
        request_id: id.to_string(),
        ..Default::default()
    }
}

#[tokio::test]
async fn default_mode_aborts_immediately_on_drop() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client.generate(generate_request("imm-1")).await.unwrap();
    drop(stream);

    assert_eq!(wait_for_abort(&state).await, ["imm-1"]);
}

#[tokio::test]
async fn deferred_mode_holds_abort_until_first_item() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client
        .generate(generate_request("def-1"))
        .await
        .unwrap()
        .defer_abort_until_first_item();
    drop(stream);

    // No abort while the backend hasn't produced anything.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        state.aborted_ids().is_empty(),
        "abort must be held until the first response"
    );

    // The backend produces its first response (handoff complete) — the
    // deferred abort now fires.
    let tx = state.lock_tx().clone().unwrap();
    tx.send(Ok(chunk())).await.unwrap();
    assert_eq!(wait_for_abort(&state).await, ["def-1"]);
}

#[tokio::test]
async fn deferred_mode_aborts_immediately_after_first_item_was_consumed() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let mut stream = client
        .generate(generate_request("def-2"))
        .await
        .unwrap()
        .defer_abort_until_first_item();

    let tx = state.lock_tx().clone().unwrap();
    tx.send(Ok(chunk())).await.unwrap();
    let first = stream.next().await;
    assert!(matches!(first, Some(Ok(_))));

    // The protected window is over — dropping now aborts promptly.
    drop(stream);
    assert_eq!(wait_for_abort(&state).await, ["def-2"]);
}

#[tokio::test]
async fn mark_completed_suppresses_abort_in_deferred_mode() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client
        .generate(generate_request("done-1"))
        .await
        .unwrap()
        .defer_abort_until_first_item();
    stream.mark_completed();
    drop(stream);

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(state.aborted_ids().is_empty());
}
