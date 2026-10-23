use std::sync::{Arc, Mutex};

use futures::stream;
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};
use tonic_health::pb::{
    health_check_response::ServingStatus,
    health_server::{Health, HealthServer},
    HealthCheckRequest, HealthCheckResponse,
};

use super::*;
use crate::worker::BasicWorkerBuilder;

#[derive(Clone)]
struct HealthPeer {
    status: ServingStatus,
    requests: Arc<Mutex<Vec<String>>>,
}

#[tonic::async_trait]
impl Health for HealthPeer {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        self.requests
            .lock()
            .unwrap()
            .push(request.into_inner().service);
        Ok(Response::new(HealthCheckResponse {
            status: self.status as i32,
        }))
    }

    type WatchStream = stream::Empty<Result<HealthCheckResponse, Status>>;

    async fn watch(
        &self,
        _: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        Err(Status::unimplemented("unary health fixture"))
    }
}

struct Peer {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    task: JoinHandle<()>,
}

impl Drop for Peer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn peer(status: ServingStatus) -> Peer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let service = HealthPeer {
        status,
        requests: Arc::clone(&requests),
    };
    let task = tokio::spawn(async move {
        Server::builder()
            .add_service(HealthServer::new(service))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    Peer {
        url,
        requests,
        task,
    }
}

fn worker(url: &str, runtime: RuntimeType, role: WorkerType) -> BasicWorker {
    BasicWorkerBuilder::new(url)
        .runtime_type(runtime)
        .worker_type(role)
        .connection_mode(ConnectionMode::Grpc)
        .health_config(HealthCheckConfig {
            timeout_secs: 1,
            ..HealthCheckConfig::default()
        })
        .build()
}

#[tokio::test]
async fn epd_sglang_encoder_health_requires_serving_on_the_named_service() {
    for status in [
        ServingStatus::Serving,
        ServingStatus::NotServing,
        ServingStatus::Unknown,
        ServingStatus::ServiceUnknown,
    ] {
        let peer = peer(status).await;
        let result = worker(&peer.url, RuntimeType::Sglang, WorkerType::Encode)
            .check_health_async()
            .await;
        assert_eq!(result.is_ok(), status == ServingStatus::Serving);
        assert_eq!(
            *peer.requests.lock().unwrap(),
            ["sglang.grpc.encoder.SglangEncoder"]
        );
    }
}

#[tokio::test]
async fn epd_sglang_encoder_health_bounds_connect_and_rpc_together() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    // A real TCP peer accepts the dial but never answers the HTTP/2 handshake/RPC.
    let stalled = tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let started = time::Instant::now();
    let result = time::timeout(
        Duration::from_secs(3),
        worker(&url, RuntimeType::Sglang, WorkerType::Encode).check_health_async(),
    )
    .await;
    stalled.abort();
    assert!(result.unwrap().is_err());
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[tokio::test]
async fn epd_sglang_encoder_health_does_not_replace_scheduler_probes() {
    let peer = peer(ServingStatus::Serving).await;
    for (runtime, role) in [
        (RuntimeType::Sglang, WorkerType::Regular),
        (RuntimeType::Sglang, WorkerType::Prefill),
        (RuntimeType::Sglang, WorkerType::Decode),
        (RuntimeType::TokenSpeed, WorkerType::Encode),
        (RuntimeType::Vllm, WorkerType::Encode),
    ] {
        assert!(worker(&peer.url, runtime, role)
            .check_health_async()
            .await
            .is_err());
    }
    let http = BasicWorkerBuilder::new(&peer.url)
        .runtime_type(RuntimeType::Sglang)
        .worker_type(WorkerType::Encode)
        .connection_mode(ConnectionMode::Http)
        .build();
    assert!(!http.grpc_health_check().await.unwrap());
    assert!(peer.requests.lock().unwrap().is_empty());
}
