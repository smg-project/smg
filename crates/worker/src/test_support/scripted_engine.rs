//! An engine whose two pushed streams follow a script, for the cases of the
//! KV event collector: tonic's client half and server half joined in
//! process, so a case reads tonic's own stream with no socket and no
//! generated service.

use std::{
    collections::HashMap,
    convert::Infallible,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use parking_lot::Mutex;
use tonic::{
    body::Body,
    codegen::{http, BoxFuture, Service},
    server::ServerStreamingService,
    Request, Response, Status, Streaming,
};
use tonic_prost::ProstCodec;

use super::{no_engine, no_health_check, BasicWorkerBuilder};
use crate::{
    common_proto::{KvEventBatch, SubscribeKvEventsRequest},
    connection::engine::{AdminAnswer, EngineConnection, EngineConnector, HealthAnswer, ZmqLink},
    identity::basic::{ConnectionMode, Worker},
    worker::{ProfileOptions, RuntimeType, WorkerLoadResponse, WorkerType},
};

/// What a [`ScriptedEngine`] does with a subscribe call.
#[derive(Clone, Debug)]
pub enum Script {
    /// Accepts the call and never answers it, as an engine still starting
    /// behind an open port does.
    Hang,
    /// Refuses the call with this status.
    Refuse(Status),
    /// Answers at once with a stream that yields the items, then ends.
    Stream(Vec<Result<KvEventBatch, Status>>),
    /// Answers at once with a stream that yields nothing and never ends.
    Open,
}

/// An engine connection whose two pushed streams follow a [`Script`]: each
/// subscribe call is noted with its request and its arrival, then answered
/// as the script says, with tonic's own stream. Every other call is refused
/// as unavailable. It is the connector of the workers built over it and
/// their one connection.
#[derive(Clone, Debug)]
pub struct ScriptedEngine(Arc<Scripted>);

#[derive(Debug)]
struct Scripted {
    script: Script,
    calls: Mutex<Vec<(SubscribeKvEventsRequest, Instant)>>,
}

impl ScriptedEngine {
    /// An engine that answers every subscribe call as `script` says.
    pub fn new(script: Script) -> Self {
        Self(Arc::new(Scripted {
            script,
            calls: Mutex::default(),
        }))
    }

    /// A gRPC worker at `url` connected to this engine, with no health
    /// checker behind it.
    pub fn worker(&self, url: &str) -> Arc<dyn Worker> {
        Arc::new(self.builder(url).build())
    }

    /// A rank worker connected to this engine: rank `rank` of the `size`
    /// ranks of the data-parallel engine at `url`.
    pub fn rank_worker(&self, url: &str, rank: usize, size: usize) -> Arc<dyn Worker> {
        Arc::new(self.builder(url).dp_config(rank, size).build())
    }

    fn builder(&self, url: &str) -> BasicWorkerBuilder {
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .engine_connector(Arc::new(self.clone()))
    }

    /// The subscribe calls the engine got so far: each request with its
    /// arrival.
    pub fn calls(&self) -> Vec<(SubscribeKvEventsRequest, Instant)> {
        self.0.calls.lock().clone()
    }

    /// One subscribe call through tonic's client half.
    async fn subscribe(
        &self,
        request: SubscribeKvEventsRequest,
    ) -> Result<Streaming<KvEventBatch>, Status> {
        let path = http::uri::PathAndQuery::from_static("/scripted.Engine/SubscribeKvEvents");
        let codec = ProstCodec::<SubscribeKvEventsRequest, KvEventBatch>::default();
        tonic::client::Grpc::new(Joined(Arc::clone(&self.0)))
            .server_streaming(Request::new(request), path, codec)
            .await
            .map(Response::into_inner)
    }
}

/// The two halves joined: what the client half sends is handed to tonic's
/// server half as it stands, and its answer handed back.
#[derive(Clone)]
struct Joined(Arc<Scripted>);

impl Service<http::Request<Body>> for Joined {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let answer = Answer(Arc::clone(&self.0));
        Box::pin(async move {
            let codec = ProstCodec::<KvEventBatch, SubscribeKvEventsRequest>::default();
            Ok(tonic::server::Grpc::new(codec)
                .server_streaming(answer, request)
                .await)
        })
    }
}

/// The engine's answer to one subscribe call: the call noted, then the
/// script.
struct Answer(Arc<Scripted>);

impl ServerStreamingService<SubscribeKvEventsRequest> for Answer {
    type Response = KvEventBatch;
    type ResponseStream = BoxStream<'static, Result<KvEventBatch, Status>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<SubscribeKvEventsRequest>) -> Self::Future {
        let engine = Arc::clone(&self.0);
        Box::pin(async move {
            engine
                .calls
                .lock()
                .push((request.into_inner(), Instant::now()));
            let batches: Self::ResponseStream = match &engine.script {
                Script::Hang => std::future::pending().await,
                Script::Refuse(status) => return Err(status.clone()),
                Script::Stream(items) => Box::pin(stream::iter(items.clone())),
                Script::Open => Box::pin(stream::pending()),
            };
            Ok(Response::new(batches))
        })
    }
}

#[async_trait]
impl EngineConnection for ScriptedEngine {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, Status> {
        no_engine()
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, Status> {
        no_engine()
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, Status> {
        no_engine()
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, Status> {
        no_engine()
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, Status> {
        no_engine()
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, Status> {
        no_engine()
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, Status> {
        no_engine()
    }

    async fn subscribe_kv_events(&self, start_seq: u64) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe(SubscribeKvEventsRequest {
            start_sequence_number: start_seq,
            ..Default::default()
        })
        .await
    }

    async fn subscribe_load_records(&self) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe(SubscribeKvEventsRequest {
            loads_only: true,
            ..Default::default()
        })
        .await
    }
}

#[async_trait]
impl EngineConnector for ScriptedEngine {
    async fn connect_grpc(
        &self,
        _base_url: &str,
        _runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        Ok(Arc::new(self.clone()))
    }

    async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        Ok(Arc::new(self.clone()))
    }

    async fn dial_grpc(&self, _base_url: &str, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}
