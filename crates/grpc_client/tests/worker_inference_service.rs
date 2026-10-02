//! In-process tonic tests for the Worker's `WorkerInference` service
//! (`EngineWorkerInference` over a scripted `EngineTransport`) and for the
//! vLLM adapter's abort guard against a scripted engine.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::{
    common_proto as common, vllm_proto as vllm,
    worker_inference::{
        connect_engine_transport, EngineTransport, EngineTransportStream, EngineWorkerInference,
        WorkerInferenceClient,
    },
    worker_inference_proto as proto,
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
    task::JoinSet,
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Code, Request, Response, Status};

const DEADLINE: Duration = Duration::from_secs(5);

/// One stream the fake transport handed to the Worker: the test feeds frames
/// through `frames` and learns from `dropped` when the Worker let go of it.
struct OpenStream {
    request_id: String,
    frames: mpsc::UnboundedSender<Result<proto::GenerateResponse, Status>>,
    dropped: oneshot::Receiver<()>,
}

struct FakeStream {
    frames: mpsc::UnboundedReceiver<Result<proto::GenerateResponse, Status>>,
    dropped: Option<oneshot::Sender<()>>,
}

impl Stream for FakeStream {
    type Item = Result<proto::GenerateResponse, Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.frames.poll_recv(cx)
    }
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        if let Some(dropped) = self.dropped.take() {
            let _ = dropped.send(());
        }
    }
}

#[derive(Clone)]
struct FakeTransport {
    opened: mpsc::UnboundedSender<OpenStream>,
    aborts: Arc<Mutex<Vec<proto::AbortRequest>>>,
    abort_succeeds: Arc<AtomicBool>,
}

#[tonic::async_trait]
impl EngineTransport for FakeTransport {
    async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<EngineTransportStream, Status> {
        let (frames_tx, frames_rx) = mpsc::unbounded_channel();
        let (dropped_tx, dropped_rx) = oneshot::channel();
        self.opened
            .send(OpenStream {
                request_id: request.request_id,
                frames: frames_tx,
                dropped: dropped_rx,
            })
            .map_err(|_| Status::internal("stream observer is gone"))?;
        Ok(Box::pin(FakeStream {
            frames: frames_rx,
            dropped: Some(dropped_tx),
        }))
    }

    async fn abort(&self, request: proto::AbortRequest) -> Result<proto::AbortResponse, Status> {
        self.aborts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request);
        let success = self.abort_succeeds.load(Ordering::SeqCst);
        Ok(proto::AbortResponse {
            success,
            message: if success {
                String::new()
            } else {
                "no such request".to_string()
            },
        })
    }
}

struct Worker {
    client: WorkerInferenceClient,
    opened: mpsc::UnboundedReceiver<OpenStream>,
    transport: FakeTransport,
    serving: Arc<AtomicBool>,
    /// Aborts the tonic server when the test ends.
    _servers: JoinSet<Result<(), tonic::transport::Error>>,
}

impl Worker {
    async fn start(
        max_concurrent_requests: u32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (opened_tx, opened) = mpsc::unbounded_channel();
        let transport = FakeTransport {
            opened: opened_tx,
            aborts: Arc::default(),
            abort_succeeds: Arc::new(AtomicBool::new(true)),
        };
        let serving = Arc::new(AtomicBool::new(true));
        let service = EngineWorkerInference::from_transport(
            Arc::new(transport.clone()) as Arc<dyn EngineTransport>,
            max_concurrent_requests,
        )
        .with_serving_flag(Arc::clone(&serving));
        let mut servers = JoinSet::new();
        servers.spawn(
            Server::builder()
                .add_service(proto::worker_inference_server::WorkerInferenceServer::new(
                    service,
                ))
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        let client = WorkerInferenceClient::connect(&format!("grpc://{address}")).await?;
        Ok(Self {
            client,
            opened,
            transport,
            serving,
            _servers: servers,
        })
    }

    /// The next stream the transport handed out, or `None` past the deadline.
    async fn opened(&mut self) -> Option<OpenStream> {
        tokio::time::timeout(DEADLINE, self.opened.recv())
            .await
            .ok()
            .flatten()
    }

    /// The status a `generate` call is refused with, or `None` if admitted.
    async fn refusal(&self) -> Option<Status> {
        self.client
            .generate(generate_request("refused"))
            .await
            .err()
    }
}

fn generate_request(request_id: &str) -> proto::GenerateRequest {
    proto::GenerateRequest {
        request_id: request_id.to_string(),
        stream: true,
        ..Default::default()
    }
}

fn chunk(request_id: &str) -> proto::GenerateResponse {
    proto::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                token_ids: vec![7],
                completion_tokens: 1,
                ..Default::default()
            },
        )),
    }
}

/// Whether the Worker released the stream before the deadline.
async fn released(dropped: oneshot::Receiver<()>) -> bool {
    tokio::time::timeout(DEADLINE, dropped)
        .await
        .is_ok_and(|signal| signal.is_ok())
}

#[tokio::test]
async fn not_serving_is_rejected_before_admission() {
    let mut worker = Worker::start(1).await.unwrap();
    let mut held = worker
        .client
        .generate(generate_request("held"))
        .await
        .unwrap();
    let held_stream = worker.opened().await.expect("held stream opened");
    assert_eq!(held_stream.request_id, "held");
    held_stream.frames.send(Ok(chunk("held"))).unwrap();
    assert!(held.next().await.unwrap().is_ok());

    // The single permit is taken, yet a draining Worker answers UNAVAILABLE,
    // not RESOURCE_EXHAUSTED: the serving check precedes admission and never
    // reaches the transport.
    worker.serving.store(false, Ordering::Release);
    let refusal = worker.refusal().await.expect("draining Worker refuses");
    assert_eq!(refusal.code(), Code::Unavailable, "{refusal:?}");
    assert!(matches!(
        worker.opened.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));

    worker.serving.store(true, Ordering::Release);
    let refusal = worker.refusal().await.expect("full Worker refuses");
    assert_eq!(refusal.code(), Code::ResourceExhausted, "{refusal:?}");
    drop(held);
}

#[tokio::test]
async fn admission_permits_are_released_on_drop_and_on_end() {
    let mut worker = Worker::start(1).await.unwrap();

    let mut first = worker
        .client
        .generate(generate_request("first"))
        .await
        .unwrap();
    let first_stream = worker.opened().await.expect("first stream opened");
    first_stream.frames.send(Ok(chunk("first"))).unwrap();
    assert!(first.next().await.unwrap().is_ok());
    let refusal = worker.refusal().await.expect("full Worker refuses");
    assert_eq!(refusal.code(), Code::ResourceExhausted, "{refusal:?}");

    // Ungraceful client disconnect mid-stream.
    drop(first);
    assert!(
        released(first_stream.dropped).await,
        "Worker reclaims a stream its client dropped"
    );
    let mut second = worker
        .client
        .generate(generate_request("second"))
        .await
        .expect("permit released when the dropped stream was reclaimed");
    let OpenStream {
        frames, dropped, ..
    } = worker.opened().await.expect("second stream opened");
    let refusal = worker.refusal().await.expect("full Worker refuses");
    assert_eq!(refusal.code(), Code::ResourceExhausted, "{refusal:?}");

    // Orderly end of stream.
    frames.send(Ok(chunk("second"))).unwrap();
    drop(frames);
    assert!(second.next().await.unwrap().is_ok());
    assert!(second.next().await.is_none());
    assert!(
        released(dropped).await,
        "Worker releases a stream that ended"
    );
    worker
        .client
        .generate(generate_request("third"))
        .await
        .expect("permit released when the stream ended");
}

#[tokio::test]
async fn abort_delegates_to_the_transport() {
    let worker = Worker::start(0).await.unwrap();
    worker
        .client
        .abort_request("r1".to_string(), "client gone".to_string())
        .await
        .unwrap();

    worker
        .transport
        .abort_succeeds
        .store(false, Ordering::SeqCst);
    let status = worker
        .client
        .abort_request("r2".to_string(), "client gone".to_string())
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::FailedPrecondition);
    assert_eq!(status.message(), "no such request");

    let aborts = worker
        .transport
        .aborts
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        aborts,
        ["r1", "r2"].map(|request_id| proto::AbortRequest {
            request_id: request_id.to_string(),
            reason: "client gone".to_string(),
        })
    );
}

/// vLLM engine that answers a request with one `Complete` per sampled index.
#[derive(Clone)]
struct FakeVllm {
    healthy: Arc<AtomicBool>,
    aborts: mpsc::UnboundedSender<String>,
}

type FakeVllmStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl vllm::vllm_engine_server::VllmEngine for FakeVllm {
    type GenerateStream = FakeVllmStream<vllm::GenerateResponse>;
    type GetTokenizerStream = FakeVllmStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = FakeVllmStream<common::KvEventBatch>;

    async fn generate(
        &self,
        request: Request<vllm::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let sampled = request.sampling_params.map_or(1, |params| params.n.max(1));
        let frames: Vec<_> = (0..sampled)
            .map(|index| {
                Ok(vllm::GenerateResponse {
                    response: Some(vllm::generate_response::Response::Complete(
                        vllm::GenerateComplete {
                            output_ids: vec![index],
                            finish_reason: "stop".to_string(),
                            index,
                            ..Default::default()
                        },
                    )),
                })
            })
            .collect();
        Ok(Response::new(Box::pin(stream::iter(frames))))
    }

    async fn abort(
        &self,
        request: Request<vllm::AbortRequest>,
    ) -> Result<Response<vllm::AbortResponse>, Status> {
        for request_id in request.into_inner().request_ids {
            self.aborts
                .send(request_id)
                .map_err(|_| Status::internal("abort observer is gone"))?;
        }
        Ok(Response::new(vllm::AbortResponse::default()))
    }

    async fn health_check(
        &self,
        _request: Request<vllm::HealthCheckRequest>,
    ) -> Result<Response<vllm::HealthCheckResponse>, Status> {
        let healthy = self.healthy.load(Ordering::SeqCst);
        Ok(Response::new(vllm::HealthCheckResponse {
            healthy,
            message: if healthy { "ok" } else { "loading weights" }.to_string(),
        }))
    }

    async fn embed(
        &self,
        _request: Request<vllm::EmbedRequest>,
    ) -> Result<Response<vllm::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_model_info(
        &self,
        _request: Request<vllm::GetModelInfoRequest>,
    ) -> Result<Response<vllm::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_server_info(
        &self,
        _request: Request<vllm::GetServerInfoRequest>,
    ) -> Result<Response<vllm::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_loads(
        &self,
        _request: Request<vllm::GetLoadsRequest>,
    ) -> Result<Response<vllm::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }
}

async fn start_engine(
    engine: FakeVllm,
) -> Result<(String, JoinSet<Result<(), tonic::transport::Error>>), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let mut servers = JoinSet::new();
    servers.spawn(
        Server::builder()
            .add_service(vllm::vllm_engine_server::VllmEngineServer::new(engine))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    Ok((format!("grpc://{address}"), servers))
}

fn multi_index_request(request_id: &str) -> proto::GenerateRequest {
    proto::GenerateRequest {
        request_id: request_id.to_string(),
        tokenized: Some(proto::TokenizedInput {
            input_ids: vec![1, 2, 3],
            ..Default::default()
        }),
        sampling_params: Some(proto::SamplingParams {
            n: 2,
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    }
}

/// The index of a `Complete` frame; `None` for anything else.
fn complete_index(item: Option<Result<proto::GenerateResponse, Status>>) -> Option<u32> {
    match item {
        Some(Ok(proto::GenerateResponse {
            response: Some(proto::generate_response::Response::Complete(complete)),
            ..
        })) => Some(complete.index),
        _ => None,
    }
}

#[tokio::test]
async fn vllm_adapter_probes_health_and_aborts_until_every_index_completes() {
    let (aborts_tx, mut aborts) = mpsc::unbounded_channel();
    let engine = FakeVllm {
        healthy: Arc::new(AtomicBool::new(false)),
        aborts: aborts_tx,
    };
    let (endpoint, _servers) = start_engine(engine.clone()).await.unwrap();

    let error = connect_engine_transport("vllm", &endpoint)
        .await
        .err()
        .expect("an unhealthy engine is refused")
        .to_string();
    assert_eq!(error, "vLLM engine reports unhealthy: loading weights");
    engine.healthy.store(true, Ordering::SeqCst);
    let transport = connect_engine_transport("VLLM", &endpoint).await.unwrap();

    // Both indexes consumed: the guard is released and the drop stays silent.
    let mut finished = transport
        .generate(multi_index_request("finished"))
        .await
        .unwrap();
    assert_eq!(complete_index(finished.next().await), Some(0));
    assert_eq!(complete_index(finished.next().await), Some(1));
    assert!(finished.next().await.is_none());
    drop(finished);

    // One of two indexes consumed: the drop must abort the request.
    let mut partial = transport
        .generate(multi_index_request("partial"))
        .await
        .unwrap();
    assert_eq!(complete_index(partial.next().await), Some(0));
    drop(partial);

    let aborted = tokio::time::timeout(DEADLINE, aborts.recv())
        .await
        .expect("abort within the deadline")
        .expect("abort observer");
    assert_eq!(aborted, "partial");
    assert!(matches!(
        aborts.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
}
