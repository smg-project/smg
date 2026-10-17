//! Behavioral tests for the call tap, driven through the real
//! `VllmEngineClient` against an in-process mock engine.
//!
//! The tap is process-wide, so every test holds `TAP_LOCK` while it sets one.

use std::{
    error::Error,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::Duration,
};

use futures::StreamExt;
use prost::Message;
use smg_grpc_client::{
    common_proto, set_call_tap, vllm_proto as proto, CallEnd, CallRecord, CallStart, CallTap,
    VllmEngineClient,
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Code, Request, Response, Status};

static TAP_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const GENERATE: &str = "/vllm.grpc.engine.VllmEngine/Generate";
const HEALTH_CHECK: &str = "/vllm.grpc.engine.VllmEngine/HealthCheck";

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---- mock engine: the generate request's id picks the behavior ----

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

#[derive(Clone, Default)]
struct MockEngine {
    /// Streams of `held` requests, kept open until the test ends.
    held: Arc<Mutex<Vec<GenerateTx>>>,
}

fn chunk(token: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                token_ids: vec![token],
                ..Default::default()
            },
        )),
    }
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for MockEngine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(4);
        match request.into_inner().request_id.as_str() {
            "rejected" => return Err(Status::resource_exhausted("queue full")),
            "complete" => {
                tx.send(Ok(chunk(7))).await.ok();
                tx.send(Ok(chunk(8))).await.ok();
            }
            "failed" => {
                tx.send(Ok(chunk(7))).await.ok();
                tx.send(Err(Status::internal("engine fault"))).await.ok();
            }
            _ => lock(&self.held).push(tx),
        }
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Ok(Response::new(proto::AbortResponse::default()))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Ok(Response::new(proto::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn flush_cache(
        &self,
        _request: Request<common_proto::FlushCacheRequest>,
    ) -> Result<Response<common_proto::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
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

    type GetTokenizerStream = ReceiverStream<Result<common_proto::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    type SubscribeKvEventsStream = ReceiverStream<Result<common_proto::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }
}

/// A running mock engine; dropping `shutdown` stops it.
struct Mock {
    endpoint: String,
    shutdown: oneshot::Sender<()>,
    stopped: tokio::task::JoinHandle<()>,
}

async fn spawn_mock() -> Result<Mock, Box<dyn Error>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}", listener.local_addr()?);
    let (shutdown, stop) = oneshot::channel::<()>();
    #[expect(
        clippy::disallowed_methods,
        reason = "mock server for one test; stopped by the test or at process exit"
    )]
    let stopped = tokio::spawn(async move {
        Server::builder()
            .add_service(proto::vllm_engine_server::VllmEngineServer::new(
                MockEngine::default(),
            ))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                stop.await.ok();
            })
            .await
            .ok();
    });
    Ok(Mock {
        endpoint,
        shutdown,
        stopped,
    })
}

// ---- a tap that keeps every call it sees ----

#[derive(Default)]
struct Call {
    uri: String,
    method: String,
    requests: Vec<Vec<u8>>,
    responses: Vec<Vec<u8>>,
    end: Option<CallEnd>,
}

#[derive(Clone, Default)]
struct Recorder {
    calls: Arc<Mutex<Vec<Arc<Mutex<Call>>>>>,
}

struct Record(Arc<Mutex<Call>>);

impl CallTap for Recorder {
    fn begin(&self, call: &CallStart<'_>) -> Option<Box<dyn CallRecord>> {
        let call = Arc::new(Mutex::new(Call {
            uri: call.uri.to_string(),
            method: call.uri.path().to_string(),
            ..Default::default()
        }));
        lock(&self.calls).push(Arc::clone(&call));
        Some(Box::new(Record(call)))
    }
}

impl CallRecord for Record {
    fn request(&mut self, message: &[u8]) {
        let mut call = lock(&self.0);
        assert!(call.end.is_none(), "request message after the end");
        call.requests.push(message.to_vec());
    }

    fn response(&mut self, message: &[u8]) {
        let mut call = lock(&self.0);
        assert!(call.end.is_none(), "response message after the end");
        call.responses.push(message.to_vec());
    }

    fn end(self: Box<Self>, end: CallEnd) {
        let mut call = lock(&self.0);
        assert!(call.end.is_none(), "call ended twice");
        call.end = Some(end);
    }
}

impl Recorder {
    /// Wait (bounded) for the first call to `method` to end, and take it.
    async fn ended(&self, method: &str) -> Option<Call> {
        for _ in 0..200 {
            let found = lock(&self.calls)
                .iter()
                .find(|call| {
                    let call = lock(call);
                    call.method == method && call.end.is_some()
                })
                .cloned();
            if let Some(call) = found {
                return Some(std::mem::take(&mut *lock(&call)));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        None
    }
}

fn install(recorder: &Recorder) {
    set_call_tap(Some(Arc::new(recorder.clone())));
}

fn generate_request(id: &str) -> proto::GenerateRequest {
    proto::GenerateRequest {
        request_id: id.to_string(),
        ..Default::default()
    }
}

/// The engine status a call ended with, if it ended with one.
fn status(end: Option<CallEnd>) -> Option<(Code, String)> {
    match end {
        Some(CallEnd::Status(status)) => Some((status.code(), status.message().to_string())),
        _ => None,
    }
}

// ---- tests ----

#[tokio::test]
async fn streamed_call_records_request_responses_and_ok() {
    let _guard = TAP_LOCK.lock().await;
    let mock = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&mock.endpoint).await.unwrap();
    let recorder = Recorder::default();
    install(&recorder);

    let request = generate_request("complete");
    let received: Vec<_> = client
        .generate(request.clone())
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    let call = recorder.ended(GENERATE).await.unwrap();
    set_call_tap(None);

    assert_eq!(call.uri, format!("{}{GENERATE}", mock.endpoint));
    assert_eq!(call.requests, [request.encode_to_vec()]);
    let responses: Vec<_> = call
        .responses
        .iter()
        .map(|message| proto::GenerateResponse::decode(&message[..]).unwrap())
        .collect();
    assert_eq!(responses, received);
    assert_eq!(responses, [chunk(7), chunk(8)]);
    assert_eq!(status(call.end), Some((Code::Ok, String::new())));
}

#[tokio::test]
async fn engine_error_after_responses_is_the_end_status() {
    let _guard = TAP_LOCK.lock().await;
    let mock = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&mock.endpoint).await.unwrap();
    let recorder = Recorder::default();
    install(&recorder);

    let mut stream = client.generate(generate_request("failed")).await.unwrap();
    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert!(matches!(stream.next().await, Some(Err(_))));
    let call = recorder.ended(GENERATE).await.unwrap();
    set_call_tap(None);

    assert_eq!(call.responses, [chunk(7).encode_to_vec()]);
    assert_eq!(
        status(call.end),
        Some((Code::Internal, "engine fault".to_string()))
    );
}

#[tokio::test]
async fn rejected_call_records_the_trailers_only_status() {
    let _guard = TAP_LOCK.lock().await;
    let mock = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&mock.endpoint).await.unwrap();
    let recorder = Recorder::default();
    install(&recorder);

    let rejected = client.generate(generate_request("rejected")).await;
    let call = recorder.ended(GENERATE).await.unwrap();
    set_call_tap(None);

    assert_eq!(
        rejected.err().map(|e| e.code()),
        Some(Code::ResourceExhausted)
    );
    assert_eq!(call.requests.len(), 1);
    assert!(call.responses.is_empty());
    assert_eq!(
        status(call.end),
        Some((Code::ResourceExhausted, "queue full".to_string()))
    );
}

#[tokio::test]
async fn dropped_call_records_dropped() {
    let _guard = TAP_LOCK.lock().await;
    let mock = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&mock.endpoint).await.unwrap();
    let recorder = Recorder::default();
    install(&recorder);

    // The engine never ends a `held` stream, so the dropped stream is
    // released by the reset after the drain bound, not by the engine.
    let stream = client
        .generate(generate_request("held"))
        .await
        .unwrap()
        .with_release_drain_bound(Duration::from_millis(50));
    drop(stream);
    let call = recorder.ended(GENERATE).await.unwrap();
    set_call_tap(None);

    assert!(matches!(call.end, Some(CallEnd::Dropped)), "{:?}", call.end);
}

#[tokio::test]
async fn unary_call_records_an_empty_request_message() {
    let _guard = TAP_LOCK.lock().await;
    let mock = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&mock.endpoint).await.unwrap();
    let recorder = Recorder::default();
    install(&recorder);

    let health = client.health_check().await.unwrap();
    let call = recorder.ended(HEALTH_CHECK).await.unwrap();
    set_call_tap(None);

    // `HealthCheckRequest {}` encodes to zero bytes: still one message.
    assert_eq!(call.requests, [Vec::<u8>::new()]);
    assert_eq!(call.responses, [health.encode_to_vec()]);
    assert_eq!(status(call.end), Some((Code::Ok, String::new())));
}

#[tokio::test]
async fn unreachable_engine_records_failed() {
    let _guard = TAP_LOCK.lock().await;
    let mock = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&mock.endpoint).await.unwrap();
    mock.shutdown.send(()).ok();
    mock.stopped.await.unwrap();
    let recorder = Recorder::default();
    install(&recorder);

    let unreachable = client.health_check().await;
    let call = recorder.ended(HEALTH_CHECK).await.unwrap();
    set_call_tap(None);

    assert!(unreachable.is_err());
    assert!(
        matches!(call.end, Some(CallEnd::Failed(_))),
        "{:?}",
        call.end
    );
}

#[tokio::test]
async fn declined_call_passes_through() {
    struct Decline;
    impl CallTap for Decline {
        fn begin(&self, _call: &CallStart<'_>) -> Option<Box<dyn CallRecord>> {
            None
        }
    }

    let _guard = TAP_LOCK.lock().await;
    let mock = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&mock.endpoint).await.unwrap();
    set_call_tap(Some(Arc::new(Decline)));

    let received: Vec<_> = client
        .generate(generate_request("complete"))
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    set_call_tap(None);

    assert_eq!(received, [chunk(7), chunk(8)]);
}
