//! Real loopback transport; encoder computation and scheduler delivery are stand-ins.

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use smg_grpc_client::{sglang_encoder_proto as ep, SglangEncoderClient};
use tokio::{net::TcpListener, sync::Notify, task::JoinSet};
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
        self.timeouts
            .lock()
            .map_err(|_| Status::internal("encoder timeout recorder lock poisoned"))?
            .push(
                request
                    .metadata()
                    .get("grpc-timeout")
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("")
                    .to_string(),
            );
        let request = request.into_inner();
        self.requests
            .lock()
            .map_err(|_| Status::internal("encoder request recorder lock poisoned"))?
            .push(request.clone());
        if self.fail.load(Ordering::SeqCst) == 1 {
            return Err(Status::internal("encode control"));
        }
        if self.fail.load(Ordering::SeqCst) == 2 {
            return Err(Status::cancelled("Timeout expired"));
        }
        loop {
            let changed = self.changed.notified();
            let registered = self
                .registered
                .lock()
                .map_err(|_| Status::internal("encoder registration lock poisoned"))?
                .contains(&request.req_id);
            if registered {
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
        self.registered
            .lock()
            .map_err(|_| Status::internal("encoder registration lock poisoned"))?
            .insert(request.req_id);
        self.changed.notify_waiters();
        Ok(Response::new(ep::SchedulerReceiveUrlResponse {}))
    }
}

pub struct Peer {
    pub url: String,
    pub client: SglangEncoderClient,
    pub raw: ep::sglang_encoder_client::SglangEncoderClient<tonic::transport::Channel>,
    pub state: Arc<State>,
    server: JoinSet<Result<(), tonic::transport::Error>>,
}
impl Peer {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let state = Arc::new(State::default());
        let service = ep::sglang_encoder_server::SglangEncoderServer::from_arc(state.clone());
        // The set owns the server even if a later connection fails. Dropping
        // it aborts its tasks; a JoinHandle alone would detach on that path.
        let mut server = JoinSet::new();
        server.spawn(
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        let url = format!("grpc://{address}");
        let client = SglangEncoderClient::connect(&url).await?;
        let raw =
            ep::sglang_encoder_client::SglangEncoderClient::connect(format!("http://{address}"))
                .await?;
        Ok(Self {
            url,
            client,
            raw,
            state,
            server,
        })
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.server.abort_all();
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
