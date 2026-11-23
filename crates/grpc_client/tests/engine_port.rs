//! The engine port over each family's client, against a scripted engine of
//! every family behind one address: each call of the port answers what the
//! engine said, in the worker's own words; a call a family's wire lacks
//! answers `Unimplemented` and reaches no engine; a runtime with no gRPC
//! family is refused by name; and the gRPC connector connects, dials and
//! refuses the host-local wire.

use std::{
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::Duration,
};

use smg_grpc_client::{
    common_proto as common, engine_load::LOAD_SECTIONS, engine_port, mlx_proto, sglang_proto,
    tokenspeed_proto, trtllm_proto, vllm_proto,
};
use smg_worker::{
    connection::engine::{EngineConnection, EngineConnector, ZmqLink},
    labels::SUPPORTS_VISION_LABEL,
    worker::{ProfileOptions, RuntimeType, WorkerType},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Code, Request, Response, Status};

/// The health word of an engine that can serve: the message beside a healthy
/// verdict, and the whole answer of a family that reports a status alone.
const READY: &str = "OK";
/// What a subscription answers on the wire of the one family with no stream:
/// the gateway's own words there.
const NO_STREAM: &str = "SubscribeKvEvents RPC not supported for MLX backend";
/// The model every scripted engine serves.
const MODEL: &str = "scripted-model";

/// The five families: the runtime, the name its scripted engine notes a call
/// under, and what its wire carries beside the calls every family has (the
/// load read, the admin calls, the two streams).
const FAMILIES: [(RuntimeType, &str, bool, bool, bool); 5] = [
    (RuntimeType::Sglang, "sglang", true, true, true),
    (RuntimeType::Vllm, "vllm", true, true, true),
    (RuntimeType::Trtllm, "trtllm", true, false, true),
    (RuntimeType::Mlx, "mlx", false, false, false),
    (RuntimeType::TokenSpeed, "tokenspeed", true, true, true),
];

/// The named value a scripted engine states on its rank's load, on each
/// wire whose answer has the list.
fn stated_value() -> common::NamedValue {
    common::NamedValue {
        name: "queue_age".to_string(),
        unit: "ms".to_string(),
        value: 6.5,
        age_ms: Some(37),
    }
}

type EventStream = ReceiverStream<Result<common::KvEventBatch, Status>>;

/// What the scripted engines were asked.
#[derive(Default)]
struct Seen {
    /// Each scripted call an engine served, as `<family> <call>`, in order.
    calls: Mutex<Vec<&'static str>>,
    /// The sections of the last load read.
    load_sections: Mutex<Vec<String>>,
    /// The wait of the last flush.
    flush: Mutex<Option<f32>>,
    /// The last profile request.
    profile: Mutex<Option<common::StartProfileRequest>>,
    /// Every subscription request, in order.
    subscribes: Mutex<Vec<common::SubscribeKvEventsRequest>>,
}

fn lock<T>(held: &Mutex<T>) -> MutexGuard<'_, T> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A scripted engine of every family: one value serves the five services.
#[derive(Clone)]
struct Engine {
    health: &'static str,
    seen: Arc<Seen>,
}

impl Engine {
    fn note(&self, call: &'static str) {
        lock(&self.seen.calls).push(call);
    }

    fn flushed(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Response<common::FlushCacheResponse> {
        *lock(&self.seen.flush) = Some(request.into_inner().timeout_s);
        Response::new(common::FlushCacheResponse {
            success: true,
            message: "flushed".to_string(),
        })
    }

    fn profile_started(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Response<common::ProfileResponse> {
        *lock(&self.seen.profile) = Some(request.into_inner());
        Response::new(common::ProfileResponse {
            success: true,
            message: "started".to_string(),
        })
    }

    fn profile_stopped() -> Response<common::ProfileResponse> {
        Response::new(common::ProfileResponse {
            success: false,
            message: "none in flight".to_string(),
        })
    }

    /// One batch numbered as the request asked, then the stream's end.
    async fn subscribed(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<EventStream>, Status> {
        let asked = request.into_inner();
        let (tx, rx) = mpsc::channel(1);
        let batch = common::KvEventBatch {
            sequence_number: asked.start_sequence_number,
            ..Default::default()
        };
        tx.send(Ok(batch))
            .await
            .map_err(|_| Status::internal("the stream's reader is gone"))?;
        lock(&self.seen.subscribes).push(asked);
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// The five scripted engines behind one address, each reporting `health`,
/// with what they were asked.
#[expect(
    clippy::expect_used,
    reason = "a host that gives the engines no port stops the case"
)]
async fn engines(health: &'static str) -> (String, Arc<Seen>) {
    let engine = Engine {
        health,
        seen: Arc::default(),
    };
    let seen = Arc::clone(&engine.seen);
    let router = Server::builder()
        .add_service(
            sglang_proto::sglang_scheduler_server::SglangSchedulerServer::new(engine.clone()),
        )
        .add_service(vllm_proto::vllm_engine_server::VllmEngineServer::new(
            engine.clone(),
        ))
        .add_service(
            tokenspeed_proto::token_speed_scheduler_server::TokenSpeedSchedulerServer::new(
                engine.clone(),
            ),
        )
        .add_service(trtllm_proto::trtllm_service_server::TrtllmServiceServer::new(engine.clone()))
        .add_service(mlx_proto::mlx_engine_server::MlxEngineServer::new(engine));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a port");
    let addr = listener.local_addr().expect("the port's address");
    #[expect(
        clippy::disallowed_methods,
        reason = "scripted engines for one test; they live until the test process exits"
    )]
    tokio::spawn(router.serve_with_incoming(TcpListenerStream::new(listener)));
    (format!("http://{addr}"), seen)
}

/// The port of `runtime`'s family at `url`.
#[expect(
    clippy::panic,
    reason = "a family that does not connect stops the case, by the runtime's name"
)]
async fn connect(url: &str, runtime: RuntimeType) -> Arc<dyn EngineConnection> {
    engine_port::connect(url, runtime)
        .await
        .unwrap_or_else(|reason| panic!("no connection for {runtime}: {reason}"))
}

/// The answer of a call a family's wire lacks: `Unimplemented` in `words`,
/// with no engine asked (`seen` is what the engines served before it).
fn assert_refused<T>(answer: Result<T, Status>, words: &str, seen: &Seen, served: usize) {
    let status = answer.err();
    assert_eq!(status.as_ref().map(Status::code), Some(Code::Unimplemented));
    assert_eq!(status.as_ref().map(Status::message), Some(words));
    assert_eq!(lock(&seen.calls).len(), served, "an engine was asked");
}

/// [`assert_refused`] in the words that name the call.
fn assert_lacks<T>(answer: Result<T, Status>, call: &str, seen: &Seen, served: usize) {
    let words = format!("{call} RPC not supported for this backend");
    assert_refused(answer, &words, seen, served);
}

#[tokio::test]
async fn each_runtime_connects_to_its_own_family() {
    let (url, seen) = engines(READY).await;
    for (runtime, family, ..) in FAMILIES {
        let port = connect(&url, runtime).await;
        port.health_answer().await.expect("a health answer");
        let served = lock(&seen.calls).last().copied();
        assert_eq!(
            served.and_then(|call| call.split_once(' ')),
            Some((family, "HealthCheck")),
            "{runtime}"
        );
        assert!(port.is_alive(), "{runtime}");
        assert!(port.carries_admin_calls(), "{runtime}");
    }
}

#[tokio::test]
async fn a_runtime_with_no_grpc_family_is_refused_by_name() {
    let (url, seen) = engines(READY).await;
    for runtime in [
        RuntimeType::Unspecified,
        RuntimeType::Generic,
        RuntimeType::External,
    ] {
        let refused = engine_port::connect(&url, runtime).await.err();
        assert_eq!(refused, Some(format!("Unknown runtime type: {runtime}")));
    }
    assert!(lock(&seen.calls).is_empty());
}

#[tokio::test]
async fn a_health_answer_carries_the_engines_verdict_and_its_words() {
    for health in [READY, "loading"] {
        let (url, _seen) = engines(health).await;
        for (runtime, ..) in FAMILIES {
            let port = connect(&url, runtime).await;
            let answer = port.health_answer().await.expect("a health answer");
            assert_eq!(
                (answer.healthy, answer.message.as_str()),
                (health == READY, health),
                "{runtime}"
            );
            let verdict = port.health_check().await.expect("a verdict");
            assert_eq!(verdict, health == READY, "{runtime}");
        }
    }
}

#[tokio::test]
async fn a_load_read_asks_every_section_and_answers_the_ranks() {
    let (url, seen) = engines(READY).await;
    for (runtime, _, loads, ..) in FAMILIES {
        let port = connect(&url, runtime).await;
        let served = lock(&seen.calls).len();
        let answer = port.get_loads().await;
        if loads {
            let answer = answer.expect("a load answer");
            assert_eq!(answer.dp_rank_count, 1, "{runtime}");
            let running: Vec<_> = answer
                .loads
                .iter()
                .map(|rank| rank.num_running_reqs)
                .collect();
            assert_eq!(running, [3], "{runtime}");
            // A rank's named values arrive as the engine stated them, on
            // each wire whose answer has the list.
            let values: Vec<_> = answer
                .loads
                .iter()
                .flat_map(|rank| &rank.values)
                .map(|value| (&*value.name, &*value.unit, value.value, value.age_ms))
                .collect();
            let stated: &[(&str, &str, f64, Option<u32>)] = if runtime == RuntimeType::Trtllm {
                &[]
            } else {
                &[("queue_age", "ms", 6.5, Some(37))]
            };
            assert_eq!(values, stated, "{runtime}");
            assert_eq!(*lock(&seen.load_sections), LOAD_SECTIONS, "{runtime}");
            lock(&seen.load_sections).clear();
        } else {
            assert_lacks(answer, "GetLoads", &seen, served);
        }
    }
}

#[tokio::test]
async fn the_admin_calls_answer_the_engine_or_unimplemented() {
    let (url, seen) = engines(READY).await;
    let options = ProfileOptions {
        output_dir: Some("/tmp/profile".to_string()),
        start_step: Some(2),
        num_steps: Some(5),
        activities: Some(vec!["CPU".to_string()]),
        with_stack: Some(true),
        record_shapes: Some(false),
        profile_by_stage: true,
    };
    for (runtime, _, _, admin, _) in FAMILIES {
        let port = connect(&url, runtime).await;
        let served = lock(&seen.calls).len();
        if admin {
            let flushed = port.flush_cache(1.5).await.expect("a flush answer");
            assert_eq!(
                (flushed.success, flushed.message.as_str()),
                (true, "flushed")
            );
            assert_eq!(lock(&seen.flush).take(), Some(1.5), "{runtime}");
            let started = port.start_profile(&options).await.expect("a start answer");
            assert_eq!(
                (started.success, started.message.as_str()),
                (true, "started")
            );
            let asked = common::StartProfileRequest {
                output_dir: Some("/tmp/profile".to_string()),
                start_step: Some(2),
                num_steps: Some(5),
                activities: vec!["CPU".to_string()],
                with_stack: Some(true),
                record_shapes: Some(false),
                profile_by_stage: true,
            };
            assert_eq!(lock(&seen.profile).take(), Some(asked), "{runtime}");
            let stopped = port.stop_profile().await.expect("a stop answer");
            assert_eq!(
                (stopped.success, stopped.message.as_str()),
                (false, "none in flight"),
                "{runtime}"
            );
            assert_eq!(lock(&seen.calls).len(), served + 3, "{runtime}");
        } else {
            assert_lacks(port.flush_cache(1.5).await, "FlushCache", &seen, served);
            assert_lacks(
                port.start_profile(&options).await,
                "StartProfile",
                &seen,
                served,
            );
            assert_lacks(port.stop_profile().await, "StopProfile", &seen, served);
        }
    }
}

#[tokio::test]
async fn the_label_reads_answer_what_the_engine_reports() {
    let (url, _seen) = engines(READY).await;
    // The model's name and one fact of the server, under each family's keys.
    for (runtime, model_key, server_key, server_value) in [
        (RuntimeType::Sglang, "served_model_name", "version", "1.0"),
        (
            RuntimeType::Vllm,
            "served_model_name",
            "kv_engine_id",
            "engine-1",
        ),
        (RuntimeType::Trtllm, "model_id", "version", "1.0"),
        (RuntimeType::Mlx, "served_model_name", "server_type", "grpc"),
        (
            RuntimeType::TokenSpeed,
            "served_model_name",
            "version",
            "1.0",
        ),
    ] {
        let port = connect(&url, runtime).await;
        let model = port.model_labels().await.expect("the model's labels");
        assert_eq!(
            model.get(model_key).map(String::as_str),
            Some(MODEL),
            "{runtime}"
        );
        // No scripted engine sets its vision field. One family's conversion
        // keeps an unset field as a verdict, and the others drop it.
        assert_eq!(
            model.contains_key(SUPPORTS_VISION_LABEL),
            runtime == RuntimeType::Vllm,
            "{runtime}"
        );
        let server = port.server_labels().await.expect("the server's labels");
        assert_eq!(
            server.get(server_key).map(String::as_str),
            Some(server_value),
            "{runtime}"
        );
        // The derived read of the transfer engine id stands on the server's labels.
        let engine_id = port.kv_engine_id().await.expect("the engine id's read");
        assert_eq!(
            engine_id.as_deref(),
            (runtime == RuntimeType::Vllm).then_some("engine-1"),
            "{runtime}"
        );
    }
}

#[tokio::test]
async fn an_event_subscription_asks_for_the_load_records_first() {
    let (url, seen) = engines(READY).await;
    for (runtime, _, _, _, streams) in FAMILIES {
        let port = connect(&url, runtime).await;
        let served = lock(&seen.calls).len();
        if streams {
            let mut events = port.subscribe_kv_events(7).await.expect("the events");
            let batch = events.message().await.expect("a read").expect("a batch");
            assert_eq!(batch.sequence_number, 7, "{runtime}");
            let mut loads = port.subscribe_load_records().await.expect("the records");
            loads.message().await.expect("a read").expect("a batch");
            let asked = [
                common::SubscribeKvEventsRequest {
                    start_sequence_number: 7,
                    load_records_before_first_batch: true,
                    loads_only: false,
                },
                common::SubscribeKvEventsRequest {
                    start_sequence_number: 0,
                    load_records_before_first_batch: true,
                    loads_only: true,
                },
            ];
            assert_eq!(
                std::mem::take(&mut *lock(&seen.subscribes)),
                asked,
                "{runtime}"
            );
        } else {
            assert_refused(port.subscribe_kv_events(7).await, NO_STREAM, &seen, served);
            assert_refused(
                port.subscribe_load_records().await,
                NO_STREAM,
                &seen,
                served,
            );
        }
    }
}

#[tokio::test]
async fn the_grpc_connector_connects_dials_and_refuses_the_host_local_wire() {
    let (url, seen) = engines(READY).await;
    let connector = engine_port::GrpcConnector;
    let port = connector
        .connect_grpc(&url, RuntimeType::Vllm)
        .await
        .unwrap_or_else(|reason| panic!("no connection: {reason}"));
    assert!(port.health_check().await.expect("a verdict"));
    assert_eq!(lock(&seen.calls).last().copied(), Some("vllm HealthCheck"));

    // A dial asks the server nothing, and fails where nothing listens.
    let served = lock(&seen.calls).len();
    let bound = Duration::from_secs(5);
    connector.dial_grpc(&url, bound).await.expect("the dial");
    assert_eq!(lock(&seen.calls).len(), served);
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a port");
        format!(
            "http://{}",
            listener.local_addr().expect("the port's address")
        )
    };
    assert!(connector.dial_grpc(&closed, bound).await.is_err());

    let link = ZmqLink {
        base_url: "tcp://127.0.0.1:1".to_string(),
        model_id: MODEL.to_string(),
        runtime: RuntimeType::Vllm,
        worker_type: WorkerType::Regular,
        handshake_address: None,
        engine_count: None,
        pinned_by_rank: false,
        startup: bound,
        request_timeout: None,
    };
    let refused = connector.connect_zmq(link).await.err();
    assert_eq!(refused.as_deref(), Some(engine_port::NO_HOST_LOCAL_WIRE));
}

#[tonic::async_trait]
impl sglang_proto::sglang_scheduler_server::SglangScheduler for Engine {
    type GenerateStream = ReceiverStream<Result<sglang_proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<sglang_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn embed(
        &self,
        _request: Request<sglang_proto::EmbedRequest>,
    ) -> Result<Response<sglang_proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn health_check(
        &self,
        _request: Request<sglang_proto::HealthCheckRequest>,
    ) -> Result<Response<sglang_proto::HealthCheckResponse>, Status> {
        self.note("sglang HealthCheck");
        Ok(Response::new(sglang_proto::HealthCheckResponse {
            healthy: self.health == READY,
            message: self.health.to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<sglang_proto::AbortRequest>,
    ) -> Result<Response<sglang_proto::AbortResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn get_model_info(
        &self,
        _request: Request<sglang_proto::GetModelInfoRequest>,
    ) -> Result<Response<sglang_proto::GetModelInfoResponse>, Status> {
        self.note("sglang GetModelInfo");
        Ok(Response::new(sglang_proto::GetModelInfoResponse {
            served_model_name: MODEL.to_string(),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<sglang_proto::GetServerInfoRequest>,
    ) -> Result<Response<sglang_proto::GetServerInfoResponse>, Status> {
        self.note("sglang GetServerInfo");
        Ok(Response::new(sglang_proto::GetServerInfoResponse {
            sglang_version: "1.0".to_string(),
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        request: Request<sglang_proto::GetLoadsRequest>,
    ) -> Result<Response<sglang_proto::GetLoadsResponse>, Status> {
        self.note("sglang GetLoads");
        *lock(&self.seen.load_sections) = request.into_inner().include;
        Ok(Response::new(sglang_proto::GetLoadsResponse {
            dp_rank_count: 1,
            loads: vec![sglang_proto::SchedulerLoad {
                num_running_reqs: 3,
                values: vec![stated_value()],
                ..Default::default()
            }],
            ..Default::default()
        }))
    }

    async fn flush_cache(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        self.note("sglang FlushCache");
        Ok(self.flushed(request))
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.note("sglang StartProfile");
        Ok(self.profile_started(request))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.note("sglang StopProfile");
        Ok(Self::profile_stopped())
    }

    type GetTokenizerStream = ReceiverStream<Result<common::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    type SubscribeKvEventsStream = EventStream;

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.note("sglang SubscribeKvEvents");
        self.subscribed(request).await
    }

    async fn load_lo_ra_adapter(
        &self,
        _request: Request<sglang_proto::LoadLoRaAdapterRequest>,
    ) -> Result<Response<sglang_proto::LoadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn unload_lo_ra_adapter(
        &self,
        _request: Request<sglang_proto::UnloadLoRaAdapterRequest>,
    ) -> Result<Response<sglang_proto::UnloadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn list_loaded_lo_ra_adapters(
        &self,
        _request: Request<sglang_proto::ListLoadedLoRaAdaptersRequest>,
    ) -> Result<Response<sglang_proto::ListLoadedLoRaAdaptersResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }
}

#[tonic::async_trait]
impl vllm_proto::vllm_engine_server::VllmEngine for Engine {
    type GenerateStream = ReceiverStream<Result<vllm_proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<vllm_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn embed(
        &self,
        _request: Request<vllm_proto::EmbedRequest>,
    ) -> Result<Response<vllm_proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn health_check(
        &self,
        _request: Request<vllm_proto::HealthCheckRequest>,
    ) -> Result<Response<vllm_proto::HealthCheckResponse>, Status> {
        self.note("vllm HealthCheck");
        Ok(Response::new(vllm_proto::HealthCheckResponse {
            healthy: self.health == READY,
            message: self.health.to_string(),
        }))
    }

    async fn flush_cache(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        self.note("vllm FlushCache");
        Ok(self.flushed(request))
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.note("vllm StartProfile");
        Ok(self.profile_started(request))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.note("vllm StopProfile");
        Ok(Self::profile_stopped())
    }

    async fn abort(
        &self,
        _request: Request<vllm_proto::AbortRequest>,
    ) -> Result<Response<vllm_proto::AbortResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn get_model_info(
        &self,
        _request: Request<vllm_proto::GetModelInfoRequest>,
    ) -> Result<Response<vllm_proto::GetModelInfoResponse>, Status> {
        self.note("vllm GetModelInfo");
        Ok(Response::new(vllm_proto::GetModelInfoResponse {
            served_model_name: MODEL.to_string(),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<vllm_proto::GetServerInfoRequest>,
    ) -> Result<Response<vllm_proto::GetServerInfoResponse>, Status> {
        self.note("vllm GetServerInfo");
        Ok(Response::new(vllm_proto::GetServerInfoResponse {
            kv_engine_id: "engine-1".to_string(),
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        request: Request<vllm_proto::GetLoadsRequest>,
    ) -> Result<Response<vllm_proto::GetLoadsResponse>, Status> {
        self.note("vllm GetLoads");
        *lock(&self.seen.load_sections) = request.into_inner().include;
        Ok(Response::new(vllm_proto::GetLoadsResponse {
            dp_rank_count: 1,
            loads: vec![vllm_proto::SchedulerLoad {
                num_running_reqs: 3,
                values: vec![stated_value()],
                ..Default::default()
            }],
            ..Default::default()
        }))
    }

    type GetTokenizerStream = ReceiverStream<Result<common::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    type SubscribeKvEventsStream = EventStream;

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.note("vllm SubscribeKvEvents");
        self.subscribed(request).await
    }
}

#[tonic::async_trait]
impl tokenspeed_proto::token_speed_scheduler_server::TokenSpeedScheduler for Engine {
    type GenerateStream = ReceiverStream<Result<tokenspeed_proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<tokenspeed_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn health_check(
        &self,
        _request: Request<tokenspeed_proto::HealthCheckRequest>,
    ) -> Result<Response<tokenspeed_proto::HealthCheckResponse>, Status> {
        self.note("tokenspeed HealthCheck");
        Ok(Response::new(tokenspeed_proto::HealthCheckResponse {
            healthy: self.health == READY,
            message: self.health.to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<tokenspeed_proto::AbortRequest>,
    ) -> Result<Response<tokenspeed_proto::AbortResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn get_model_info(
        &self,
        _request: Request<tokenspeed_proto::GetModelInfoRequest>,
    ) -> Result<Response<tokenspeed_proto::GetModelInfoResponse>, Status> {
        self.note("tokenspeed GetModelInfo");
        Ok(Response::new(tokenspeed_proto::GetModelInfoResponse {
            served_model_name: MODEL.to_string(),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<tokenspeed_proto::GetServerInfoRequest>,
    ) -> Result<Response<tokenspeed_proto::GetServerInfoResponse>, Status> {
        self.note("tokenspeed GetServerInfo");
        Ok(Response::new(tokenspeed_proto::GetServerInfoResponse {
            tokenspeed_version: "1.0".to_string(),
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        request: Request<tokenspeed_proto::GetLoadsRequest>,
    ) -> Result<Response<tokenspeed_proto::GetLoadsResponse>, Status> {
        self.note("tokenspeed GetLoads");
        *lock(&self.seen.load_sections) = request.into_inner().include;
        Ok(Response::new(tokenspeed_proto::GetLoadsResponse {
            dp_rank_count: 1,
            loads: vec![tokenspeed_proto::SchedulerLoad {
                num_running_reqs: 3,
                values: vec![stated_value()],
                ..Default::default()
            }],
            ..Default::default()
        }))
    }

    async fn flush_cache(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        self.note("tokenspeed FlushCache");
        Ok(self.flushed(request))
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.note("tokenspeed StartProfile");
        Ok(self.profile_started(request))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.note("tokenspeed StopProfile");
        Ok(Self::profile_stopped())
    }

    type GetTokenizerStream = ReceiverStream<Result<common::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    type SubscribeKvEventsStream = EventStream;

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.note("tokenspeed SubscribeKvEvents");
        self.subscribed(request).await
    }
}

#[tonic::async_trait]
impl trtllm_proto::trtllm_service_server::TrtllmService for Engine {
    type GenerateStream = ReceiverStream<Result<trtllm_proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<trtllm_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn embed(
        &self,
        _request: Request<trtllm_proto::EmbedRequest>,
    ) -> Result<Response<trtllm_proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn health_check(
        &self,
        _request: Request<trtllm_proto::HealthCheckRequest>,
    ) -> Result<Response<trtllm_proto::HealthCheckResponse>, Status> {
        self.note("trtllm HealthCheck");
        Ok(Response::new(trtllm_proto::HealthCheckResponse {
            status: self.health.to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<trtllm_proto::AbortRequest>,
    ) -> Result<Response<trtllm_proto::AbortResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn get_model_info(
        &self,
        _request: Request<trtllm_proto::GetModelInfoRequest>,
    ) -> Result<Response<trtllm_proto::GetModelInfoResponse>, Status> {
        self.note("trtllm GetModelInfo");
        Ok(Response::new(trtllm_proto::GetModelInfoResponse {
            model_id: MODEL.to_string(),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<trtllm_proto::GetServerInfoRequest>,
    ) -> Result<Response<trtllm_proto::GetServerInfoResponse>, Status> {
        self.note("trtllm GetServerInfo");
        Ok(Response::new(trtllm_proto::GetServerInfoResponse {
            version: "1.0".to_string(),
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        request: Request<trtllm_proto::GetLoadsRequest>,
    ) -> Result<Response<trtllm_proto::GetLoadsResponse>, Status> {
        self.note("trtllm GetLoads");
        *lock(&self.seen.load_sections) = request.into_inner().include;
        Ok(Response::new(trtllm_proto::GetLoadsResponse {
            dp_rank_count: 1,
            loads: vec![trtllm_proto::SchedulerLoad {
                num_running_reqs: 3,
                ..Default::default()
            }],
            ..Default::default()
        }))
    }

    type GetTokenizerStream = ReceiverStream<Result<common::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    type SubscribeKvEventsStream = EventStream;

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.note("trtllm SubscribeKvEvents");
        self.subscribed(request).await
    }
}

#[tonic::async_trait]
impl mlx_proto::mlx_engine_server::MlxEngine for Engine {
    type GenerateStream = ReceiverStream<Result<mlx_proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<mlx_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn health_check(
        &self,
        _request: Request<mlx_proto::HealthCheckRequest>,
    ) -> Result<Response<mlx_proto::HealthCheckResponse>, Status> {
        self.note("mlx HealthCheck");
        Ok(Response::new(mlx_proto::HealthCheckResponse {
            healthy: self.health == READY,
            message: self.health.to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<mlx_proto::AbortRequest>,
    ) -> Result<Response<mlx_proto::AbortResponse>, Status> {
        Err(Status::unimplemented("not scripted"))
    }

    async fn get_model_info(
        &self,
        _request: Request<mlx_proto::GetModelInfoRequest>,
    ) -> Result<Response<mlx_proto::GetModelInfoResponse>, Status> {
        self.note("mlx GetModelInfo");
        Ok(Response::new(mlx_proto::GetModelInfoResponse {
            served_model_name: MODEL.to_string(),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<mlx_proto::GetServerInfoRequest>,
    ) -> Result<Response<mlx_proto::GetServerInfoResponse>, Status> {
        self.note("mlx GetServerInfo");
        Ok(Response::new(mlx_proto::GetServerInfoResponse {
            server_type: "grpc".to_string(),
            ..Default::default()
        }))
    }

    type GetTokenizerStream = ReceiverStream<Result<common::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("not scripted"))
    }
}
