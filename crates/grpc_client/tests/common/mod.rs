//! The mock engine the client's behaviour tests drive: an in-process server
//! of the engine's wire that records the request ids it is asked to abort,
//! the trace header each call carried, and keeps the sender of the open
//! generate stream for the test, which plays the engine's side of it
//! (silence, a chunk, the end of the stream).
#![allow(
    dead_code,
    clippy::allow_attributes,
    reason = "each test binary reads the helpers it needs; the others are unused there"
)]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use smg_grpc_client::{sglang_proto as sglang, vllm_proto as proto};
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Request, Response, Status};

pub type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;
pub type SglangGenerateTx = mpsc::Sender<Result<sglang::GenerateResponse, Status>>;

/// The metadata key the client's trace injector writes (the W3C trace
/// context's header), read by the mock from every call that injects it.
pub const TRACE_HEADER: &str = "traceparent";

#[derive(Default)]
pub struct MockState {
    /// Sender feeding the currently open generate stream.
    generate_tx: Mutex<Option<GenerateTx>>,
    /// Request ids passed to Abort.
    aborted: Mutex<Vec<String>>,
    /// The calls served that the client injects the trace context into,
    /// each with the `TRACE_HEADER` value it carried, in arrival order.
    traced: Mutex<Vec<(&'static str, Option<String>)>>,
    /// The last generate request the mock decoded, whole.
    generated: Mutex<Option<proto::GenerateRequest>>,
    /// The status every generate call fails with before its stream opens,
    /// when set (both wires).
    generate_status: Mutex<Option<tonic::Code>>,
    /// Sender feeding the currently open generate stream of the SGLang wire.
    generate_tx_sglang: Mutex<Option<SglangGenerateTx>>,
    /// The last generate request the mock decoded on the SGLang wire, whole.
    generated_sglang: Mutex<Option<sglang::GenerateRequest>>,
}

impl MockState {
    pub fn lock_tx(&self) -> std::sync::MutexGuard<'_, Option<GenerateTx>> {
        self.generate_tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn aborted_ids(&self) -> Vec<String> {
        self.aborted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The injecting calls served so far, each with the trace header it carried.
    pub fn traced(&self) -> Vec<(&'static str, Option<String>)> {
        self.traced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn lock_tx_sglang(&self) -> std::sync::MutexGuard<'_, Option<SglangGenerateTx>> {
        self.generate_tx_sglang
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The last generate request the mock decoded on the SGLang wire.
    pub fn last_generated_sglang(&self) -> Option<sglang::GenerateRequest> {
        self.generated_sglang
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The last generate request the mock decoded.
    pub fn last_generated(&self) -> Option<proto::GenerateRequest> {
        self.generated
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Every generate call from now on fails with `code` before its
    /// stream opens.
    pub fn fail_generate_with(&self, code: tonic::Code) {
        *self
            .generate_status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(code);
    }

    /// Records `call` with the trace header `request` carried, if any.
    fn record_trace<T>(&self, call: &'static str, request: &Request<T>) {
        let header = request
            .metadata()
            .get(TRACE_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        self.traced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((call, header));
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
        request: Request<smg_grpc_client::common_proto::StartProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        self.state.record_trace("StartProfile", &request);
        Ok(Response::new(
            smg_grpc_client::common_proto::ProfileResponse::default(),
        ))
    }

    async fn stop_profile(
        &self,
        request: Request<smg_grpc_client::common_proto::StopProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        self.state.record_trace("StopProfile", &request);
        Ok(Response::new(
            smg_grpc_client::common_proto::ProfileResponse::default(),
        ))
    }

    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.state.record_trace("Generate", &request);
        if let Some(code) = *self
            .state
            .generate_status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Err(Status::new(code, "scripted"));
        }
        *self
            .state
            .generated
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(request.into_inner());
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

    async fn flush_cache(
        &self,
        request: Request<smg_grpc_client::common_proto::FlushCacheRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::FlushCacheResponse>, Status> {
        self.state.record_trace("FlushCache", &request);
        Ok(Response::new(
            smg_grpc_client::common_proto::FlushCacheResponse {
                success: request.into_inner().timeout_s == 1.5,
                message: "reset result".to_string(),
            },
        ))
    }

    async fn embed(
        &self,
        request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        self.state.record_trace("Embed", &request);
        Ok(Response::new(proto::EmbedResponse::default()))
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

/// The same mock on the SGLang wire: the generate stream is fed by the
/// test the same way, the request it decoded is recorded, the status switch
/// is shared; every other call is unused in tests.
#[tonic::async_trait]
impl sglang::sglang_scheduler_server::SglangScheduler for MockEngine {
    type GenerateStream = ReceiverStream<Result<sglang::GenerateResponse, Status>>;

    async fn generate(
        &self,
        request: Request<sglang::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.state.record_trace("Generate", &request);
        if let Some(code) = *self
            .state
            .generate_status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Err(Status::new(code, "scripted"));
        }
        *self
            .state
            .generated_sglang
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(request.into_inner());
        let (tx, rx) = mpsc::channel(4);
        *self.state.lock_tx_sglang() = Some(tx);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn embed(
        &self,
        _request: Request<sglang::EmbedRequest>,
    ) -> Result<Response<sglang::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn health_check(
        &self,
        _request: Request<sglang::HealthCheckRequest>,
    ) -> Result<Response<sglang::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn abort(
        &self,
        _request: Request<sglang::AbortRequest>,
    ) -> Result<Response<sglang::AbortResponse>, Status> {
        Ok(Response::new(sglang::AbortResponse::default()))
    }

    async fn get_model_info(
        &self,
        _request: Request<sglang::GetModelInfoRequest>,
    ) -> Result<Response<sglang::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_server_info(
        &self,
        _request: Request<sglang::GetServerInfoRequest>,
    ) -> Result<Response<sglang::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn get_loads(
        &self,
        _request: Request<sglang::GetLoadsRequest>,
    ) -> Result<Response<sglang::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn flush_cache(
        &self,
        _request: Request<smg_grpc_client::common_proto::FlushCacheRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn start_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StartProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn stop_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StopProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
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

    async fn load_lo_ra_adapter(
        &self,
        _request: Request<sglang::LoadLoRaAdapterRequest>,
    ) -> Result<Response<sglang::LoadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn unload_lo_ra_adapter(
        &self,
        _request: Request<sglang::UnloadLoRaAdapterRequest>,
    ) -> Result<Response<sglang::UnloadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }

    async fn list_loaded_lo_ra_adapters(
        &self,
        _request: Request<sglang::ListLoadedLoRaAdaptersRequest>,
    ) -> Result<Response<sglang::ListLoadedLoRaAdaptersResponse>, Status> {
        Err(Status::unimplemented("unused in tests"))
    }
}

/// The wire a mock serves: the gateway's clients speak one per engine.
#[derive(Clone, Copy)]
pub enum Wire {
    Vllm,
    Sglang,
}

pub async fn spawn_mock(
) -> Result<(String, Arc<MockState>), Box<dyn std::error::Error + Send + Sync>> {
    serve(Wire::Vllm, None).await
}

/// The mock on the SGLang wire, tonic's default decode limit.
pub async fn spawn_sglang_mock(
) -> Result<(String, Arc<MockState>), Box<dyn std::error::Error + Send + Sync>> {
    serve(Wire::Sglang, None).await
}

/// The mock on the SGLang wire with its decoder's message limit raised to
/// `max_message_bytes`.
pub async fn spawn_sglang_mock_decoding_up_to(
    max_message_bytes: usize,
) -> Result<(String, Arc<MockState>), Box<dyn std::error::Error + Send + Sync>> {
    serve(Wire::Sglang, Some(max_message_bytes)).await
}

/// The mock with its decoder's message limit raised to `max_message_bytes`
/// (tonic's default is 4 MiB; the engine servicers raise theirs for the
/// image tensors a request carries).
pub async fn spawn_mock_decoding_up_to(
    max_message_bytes: usize,
) -> Result<(String, Arc<MockState>), Box<dyn std::error::Error + Send + Sync>> {
    serve(Wire::Vllm, Some(max_message_bytes)).await
}

async fn serve(
    wire: Wire,
    max_message_bytes: Option<usize>,
) -> Result<(String, Arc<MockState>), Box<dyn std::error::Error + Send + Sync>> {
    let mock = MockEngine::default();
    let state = Arc::clone(&mock.state);
    let router = match wire {
        Wire::Vllm => {
            let mut service = proto::vllm_engine_server::VllmEngineServer::new(mock);
            if let Some(bytes) = max_message_bytes {
                service = service.max_decoding_message_size(bytes);
            }
            Server::builder().add_service(service)
        }
        Wire::Sglang => {
            let mut service = sglang::sglang_scheduler_server::SglangSchedulerServer::new(mock);
            if let Some(bytes) = max_message_bytes {
                service = service.max_decoding_message_size(bytes);
            }
            Server::builder().add_service(service)
        }
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    #[expect(
        clippy::disallowed_methods,
        reason = "mock server for one test; lives until the test process exits"
    )]
    tokio::spawn(router.serve_with_incoming(TcpListenerStream::new(listener)));
    Ok((format!("http://{addr}"), state))
}

pub fn generate_request(id: &str) -> proto::GenerateRequest {
    proto::GenerateRequest {
        request_id: id.to_string(),
        ..Default::default()
    }
}

/// Wait (bounded) until the mock has recorded at least one abort.
pub async fn wait_for_abort(state: &MockState) -> Vec<String> {
    for _ in 0..200 {
        let ids = state.aborted_ids();
        if !ids.is_empty() {
            return ids;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    state.aborted_ids()
}
