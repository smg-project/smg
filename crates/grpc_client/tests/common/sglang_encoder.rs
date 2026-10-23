//! Real loopback transport; encoder computation and scheduler delivery are stand-ins.

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use smg_grpc_client::{sglang_encoder_proto as ep, SglangEncoderClient};
use tokio::{net::TcpListener, sync::Notify, task::JoinHandle};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

#[derive(Default)]
pub struct State {
    registered: Mutex<HashSet<String>>,
    changed: Notify,
    pub finished: Notify,
    pub requests: Mutex<Vec<ep::EncodeRequest>>,
    pub fail: AtomicUsize,
    pub timeouts: Mutex<Vec<String>>,
}

#[tonic::async_trait]
impl ep::sglang_encoder_server::SglangEncoder for State {
    async fn encode(
        &self,
        request: Request<ep::EncodeRequest>,
    ) -> Result<Response<ep::EncodeResponse>, Status> {
        self.timeouts.lock().unwrap().push(
            request
                .metadata()
                .get("grpc-timeout")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("")
                .to_string(),
        );
        let request = request.into_inner();
        self.requests.lock().unwrap().push(request.clone());
        if self.fail.load(Ordering::SeqCst) == 1 {
            return Err(Status::internal("encode control"));
        }
        if self.fail.load(Ordering::SeqCst) == 2 {
            return Err(Status::cancelled("Timeout expired"));
        }
        loop {
            let changed = self.changed.notified();
            if self.registered.lock().unwrap().contains(&request.req_id) {
                break;
            }
            changed.await;
        }
        self.finished.notify_one();
        Ok(Response::new(ep::EncodeResponse::default()))
    }
    async fn send(
        &self,
        _: Request<ep::SendRequest>,
    ) -> Result<Response<ep::SendResponse>, Status> {
        Err(Status::unimplemented("not the scheduler-receive path"))
    }
    async fn scheduler_receive_url(
        &self,
        request: Request<ep::SchedulerReceiveUrlRequest>,
    ) -> Result<Response<ep::SchedulerReceiveUrlResponse>, Status> {
        let request = request.into_inner();
        self.registered.lock().unwrap().insert(request.req_id);
        self.changed.notify_waiters();
        Ok(Response::new(ep::SchedulerReceiveUrlResponse {}))
    }
}

pub struct Peer {
    pub url: String,
    pub client: SglangEncoderClient,
    pub raw: ep::sglang_encoder_client::SglangEncoderClient<tonic::transport::Channel>,
    pub state: Arc<State>,
    server: JoinHandle<Result<(), tonic::transport::Error>>,
}
impl Peer {
    pub async fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let state = Arc::new(State::default());
        let service = ep::sglang_encoder_server::SglangEncoderServer::from_arc(state.clone());
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        let url = format!("grpc://{address}");
        let client = SglangEncoderClient::connect(&url).await.unwrap();
        let raw =
            ep::sglang_encoder_client::SglangEncoderClient::connect(format!("http://{address}"))
                .await
                .unwrap();
        Self {
            url,
            client,
            raw,
            state,
            server,
        }
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.server.abort();
    }
}

pub async fn register(peer: &Peer, id: &str) -> Result<(), Status> {
    peer.raw
        .clone()
        .scheduler_receive_url(ep::SchedulerReceiveUrlRequest {
            req_id: id.into(),
            receive_url: "127.0.0.1:7001".into(),
            receive_count: 1,
        })
        .await
        .map(|_| ())
}
