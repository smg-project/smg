//! Behavioral tests for `AbortOnDropStream::with_deadline`, driven through
//! the real `VllmEngineClient` against an in-process mock engine.
//!
//! The deadline's contract: a stream polled at or past its deadline ends with
//! the deadline's status (and nothing after it), the abort goes to the
//! backend at once rather than when the reader lets go of the stream, and a
//! stream that ends before the deadline is unchanged by it.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use futures::StreamExt;
use smg_grpc_client::{is_request_timeout, vllm_proto as proto, RequestDeadline, VllmEngineClient};
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
    fn take_tx(&self) -> Option<GenerateTx> {
        self.generate_tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
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
    async fn start_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StartProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn stop_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StopProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(4);
        *self
            .state
            .generate_tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(tx);
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

    async fn flush_cache(
        &self,
        _request: Request<smg_grpc_client::common_proto::FlushCacheRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
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

fn generate_request(id: &str) -> proto::GenerateRequest {
    proto::GenerateRequest {
        request_id: id.to_string(),
        ..Default::default()
    }
}

/// Wait (bounded) until the mock has recorded at least one abort.
async fn wait_for_abort(state: &MockState) -> Vec<String> {
    for _ in 0..200 {
        let ids = state.aborted_ids();
        if !ids.is_empty() {
            return ids;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    state.aborted_ids()
}

/// The stream's first message is the one message a 300 ms deadline cuts in
/// an hour of silence; the message's shape is the deadline's own; nothing
/// follows it.
const DEADLINE: Duration = Duration::from_millis(300);

#[tokio::test]
async fn a_silent_stream_ends_at_the_deadline_with_its_status_and_the_abort_goes_out_at_once() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let started = Instant::now();
    let mut stream = client
        .generate(generate_request("silent-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(DEADLINE));
    // The engine holds the stream open and says nothing.
    let _held_open = state.take_tx().expect("the mock opened a stream");

    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("the deadline ends the wait");
    let elapsed = started.elapsed();
    let status = first
        .expect("a message, not the end")
        .expect_err("the deadline's status");
    assert!(
        elapsed >= DEADLINE && elapsed < DEADLINE + Duration::from_secs(5),
        "ended after {elapsed:?}"
    );
    assert!(is_request_timeout(&status), "{status:?}");
    assert!(
        status.message().contains("request timed out after 300ms"),
        "{status:?}"
    );

    // The abort reaches the engine while the reader still holds the stream.
    assert_eq!(wait_for_abort(&state).await, ["silent-1"]);
    // Nothing follows the deadline's status.
    assert!(stream.next().await.is_none());
    drop(stream);
    // The drop sends no second abort.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(state.aborted_ids(), ["silent-1"]);
}

#[tokio::test]
async fn a_stream_that_goes_silent_after_its_first_message_is_cut_at_the_deadline_too() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let mut stream = client
        .generate(generate_request("after-first-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(DEADLINE));
    let tx = state.take_tx().expect("the mock opened a stream");
    tx.send(Ok(proto::GenerateResponse::default()))
        .await
        .unwrap();

    // The message that came in time passes; the silence after it ends at
    // the deadline.
    assert!(stream.next().await.unwrap().is_ok());
    let status = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("the deadline ends the wait")
        .expect("a message, not the end")
        .expect_err("the deadline's status");
    assert!(is_request_timeout(&status), "{status:?}");
    assert_eq!(wait_for_abort(&state).await, ["after-first-1"]);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn a_stream_that_ends_before_the_deadline_is_unchanged_by_it() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let mut stream = client
        .generate(generate_request("in-time-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(Duration::from_secs(30)));
    let tx = state.take_tx().expect("the mock opened a stream");
    for _ in 0..3 {
        tx.send(Ok(proto::GenerateResponse::default()))
            .await
            .unwrap();
    }
    drop(tx);

    let mut messages = 0;
    while let Some(item) = stream.next().await {
        item.expect("a message of the engine's");
        messages += 1;
    }
    assert_eq!(messages, 3);
    stream.mark_completed();
    drop(stream);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        state.aborted_ids().is_empty(),
        "a completed stream sends no abort"
    );
}

#[tokio::test]
async fn a_completed_stream_meets_its_end_not_the_deadline() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let mut stream = client
        .generate(generate_request("completed-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(Duration::from_millis(300)));
    let tx = state.take_tx().expect("the mock opened a stream");
    tx.send(Ok(proto::GenerateResponse::default()))
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_ok());

    // The consumer took the final message and marked the request complete,
    // the worker ended its stream, and the deadline passes before the
    // consumer's next poll: that poll meets the end, not the timeout.
    stream.mark_completed();
    drop(tx);
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        stream.next().await.is_none(),
        "a completed stream met the deadline instead of its end"
    );
    assert!(
        state.aborted_ids().is_empty(),
        "a completed stream sends no abort"
    );
}
