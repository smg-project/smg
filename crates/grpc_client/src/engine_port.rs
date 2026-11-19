//! The worker runtime's engine port, stated on each engine family's client.
//!
//! [`EngineConnection`] is the worker crate's: the calls its runtime makes on
//! a connected engine, in the worker crate's own types. Each client of this
//! crate answers it here, so every host of the runtime reads an engine the
//! same way: one health verdict, one load answer, one set of labels
//! ([`crate::engine_labels`]). A call a family's wire lacks answers
//! `Unimplemented`. [`connect`] makes the connection of a runtime's family,
//! and [`GrpcConnector`] is the port's other half for a host that reaches its
//! engines over gRPC alone.
//!
//! Where a client has a method of a port call's own name, that method is the
//! wire call the port call makes. The compiler prefers a type's own method to
//! a trait's, so no port call comes back to itself.
//!
//! Two answers are the same for every client here. A gRPC channel reconnects
//! in place and keeps no local flag of its engine, so each answers alive and
//! a probe's health call tells. Each also answers that its wire carries the
//! admin calls: one a family lacks answers `Unimplemented`, as a missing call
//! does, where a wire that carries none is refused before any call.
//!
//! An event subscription asks the servicer for its load records ahead of the
//! first batch. The runtime's stream reader skips a load-only batch before
//! any sequence check.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use smg_worker::{
    connection::engine::{AdminAnswer, EngineConnection, EngineConnector, HealthAnswer, ZmqLink},
    worker::{ProfileOptions, RuntimeType, WorkerLoadResponse},
};
use tonic::{Status, Streaming};

use crate::{
    channel::connect_channel_with_timeout,
    common_proto::{FlushCacheResponse, KvEventBatch, ProfileResponse, StartProfileRequest},
    engine_labels::{ModelInfo, ServerInfo},
    engine_load::LOAD_SECTIONS,
    trtllm_service, MlxEngineClient, SglangSchedulerClient, TokenSpeedSchedulerClient,
    TrtllmServiceClient, VllmEngineClient,
};

/// The sections a load read asks for, as the call takes them.
fn load_sections() -> Vec<String> {
    LOAD_SECTIONS.into_iter().map(String::from).collect()
}

/// The engine's profile request, from the worker's options.
fn profile_request(options: &ProfileOptions) -> StartProfileRequest {
    StartProfileRequest {
        output_dir: options.output_dir.clone(),
        start_step: options.start_step,
        num_steps: options.num_steps,
        activities: options.activities.clone().unwrap_or_default(),
        with_stack: options.with_stack,
        record_shapes: options.record_shapes,
        profile_by_stage: options.profile_by_stage,
    }
}

/// A flush's answer in the worker's words.
fn flushed(answer: FlushCacheResponse) -> AdminAnswer {
    AdminAnswer {
        success: answer.success,
        message: answer.message,
    }
}

/// A profile call's answer in the worker's words.
fn profiled(answer: ProfileResponse) -> AdminAnswer {
    AdminAnswer {
        success: answer.success,
        message: answer.message,
    }
}

/// The answer of a call the family's wire lacks.
fn lacks(call: &str) -> Status {
    Status::unimplemented(format!("{call} RPC not supported for this backend"))
}

/// What a subscription answers on the wire of the one family that streams
/// neither events nor load records: the words the gateway's dispatch answers
/// there today.
const NO_STREAM: &str = "SubscribeKvEvents RPC not supported for MLX backend";

/// Connect to the gRPC server of an engine of `runtime` at `base_url`: the
/// family's client, behind the port. Every runtime is named here, so a new
/// one is answered before this builds. An error is the reason as a worker
/// reports it.
pub async fn connect(
    base_url: &str,
    runtime: RuntimeType,
) -> Result<Arc<dyn EngineConnection>, String> {
    fn port<C: EngineConnection>(
        connected: Result<C, Box<dyn std::error::Error + Send + Sync>>,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        match connected {
            Ok(client) => Ok(Arc::new(client)),
            Err(e) => Err(e.to_string()),
        }
    }
    match runtime {
        RuntimeType::Sglang => port(SglangSchedulerClient::connect(base_url).await),
        RuntimeType::Vllm => port(VllmEngineClient::connect(base_url).await),
        RuntimeType::Trtllm => port(TrtllmServiceClient::connect(base_url).await),
        RuntimeType::Mlx => port(MlxEngineClient::connect(base_url).await),
        RuntimeType::TokenSpeed => port(TokenSpeedSchedulerClient::connect(base_url).await),
        RuntimeType::Unspecified | RuntimeType::Generic | RuntimeType::External => {
            Err(format!("Unknown runtime type: {runtime}"))
        }
    }
}

/// What [`GrpcConnector`] answers a worker of the host-local wire.
pub const NO_HOST_LOCAL_WIRE: &str = "this host reaches an engine over gRPC alone: \
    a worker of the host-local wire connects through its gateway";

/// The connector of a host that reaches its engines over gRPC alone: the
/// port's other half, over [`connect`] and the crate's own dial. It refuses a
/// worker of the host-local wire with [`NO_HOST_LOCAL_WIRE`]: that wire runs
/// between an engine and the gateway on its host.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrpcConnector;

#[async_trait]
impl EngineConnector for GrpcConnector {
    async fn connect_grpc(
        &self,
        base_url: &str,
        runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        connect(base_url, runtime).await
    }

    async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        Err(NO_HOST_LOCAL_WIRE.to_string())
    }

    /// The crate's own dial: the channel every family's client connects
    /// through, so the dial answers as a connect would.
    async fn dial_grpc(&self, base_url: &str, timeout: Duration) -> Result<(), String> {
        connect_channel_with_timeout(base_url, timeout)
            .await
            .map(drop)
            .map_err(|e| e.to_string())
    }
}

#[async_trait]
impl EngineConnection for SglangSchedulerClient {
    fn carries_admin_calls(&self) -> bool {
        true
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, Status> {
        let answer = self.health_check().await?;
        Ok(HealthAnswer {
            healthy: answer.healthy,
            message: answer.message,
        })
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, Status> {
        Ok(self.get_loads(load_sections()).await?.into())
    }

    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, Status> {
        Ok(flushed(self.flush_cache(timeout_s).await?))
    }

    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, Status> {
        Ok(profiled(
            self.start_profile(profile_request(options)).await?,
        ))
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, Status> {
        Ok(profiled(self.stop_profile().await?))
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ModelInfo::Sglang(Box::new(self.get_model_info().await?)).to_labels())
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ServerInfo::Sglang(Box::new(self.get_server_info().await?)).to_labels())
    }

    async fn subscribe_kv_events(&self, start_seq: u64) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_kv_events_with_load_records(start_seq).await
    }

    async fn subscribe_load_records(&self) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_load_records().await
    }
}

#[async_trait]
impl EngineConnection for VllmEngineClient {
    fn carries_admin_calls(&self) -> bool {
        true
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, Status> {
        let answer = self.health_check().await?;
        Ok(HealthAnswer {
            healthy: answer.healthy,
            message: answer.message,
        })
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, Status> {
        Ok(self.get_loads(load_sections()).await?.into())
    }

    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, Status> {
        Ok(flushed(self.flush_cache(timeout_s).await?))
    }

    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, Status> {
        Ok(profiled(
            self.start_profile(profile_request(options)).await?,
        ))
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, Status> {
        Ok(profiled(self.stop_profile().await?))
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ModelInfo::Vllm(self.get_model_info().await?).to_labels())
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ServerInfo::Vllm(Box::new(self.get_server_info().await?)).to_labels())
    }

    async fn subscribe_kv_events(&self, start_seq: u64) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_kv_events_with_load_records(start_seq).await
    }

    async fn subscribe_load_records(&self) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_load_records().await
    }
}

#[async_trait]
impl EngineConnection for TrtllmServiceClient {
    fn carries_admin_calls(&self) -> bool {
        true
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, Status> {
        let answer = self.health_check().await?;
        Ok(HealthAnswer {
            healthy: trtllm_service::status_healthy(&answer.status),
            message: answer.status,
        })
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, Status> {
        Ok(self.get_loads(load_sections()).await?.into())
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, Status> {
        Err(lacks("FlushCache"))
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, Status> {
        Err(lacks("StartProfile"))
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, Status> {
        Err(lacks("StopProfile"))
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ModelInfo::Trtllm(self.get_model_info().await?).to_labels())
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ServerInfo::Trtllm(self.get_server_info().await?).to_labels())
    }

    async fn subscribe_kv_events(&self, start_seq: u64) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_kv_events_with_load_records(start_seq).await
    }

    async fn subscribe_load_records(&self) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_load_records().await
    }
}

#[async_trait]
impl EngineConnection for MlxEngineClient {
    fn carries_admin_calls(&self) -> bool {
        true
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, Status> {
        let answer = self.health_check().await?;
        Ok(HealthAnswer {
            healthy: answer.healthy,
            message: answer.message,
        })
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, Status> {
        Err(lacks("GetLoads"))
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, Status> {
        Err(lacks("FlushCache"))
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, Status> {
        Err(lacks("StartProfile"))
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, Status> {
        Err(lacks("StopProfile"))
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ModelInfo::Mlx(self.get_model_info().await?).to_labels())
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ServerInfo::Mlx(self.get_server_info().await?).to_labels())
    }

    async fn subscribe_kv_events(
        &self,
        _start_seq: u64,
    ) -> Result<Streaming<KvEventBatch>, Status> {
        Err(Status::unimplemented(NO_STREAM))
    }

    async fn subscribe_load_records(&self) -> Result<Streaming<KvEventBatch>, Status> {
        Err(Status::unimplemented(NO_STREAM))
    }
}

#[async_trait]
impl EngineConnection for TokenSpeedSchedulerClient {
    fn carries_admin_calls(&self) -> bool {
        true
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, Status> {
        let answer = self.health_check().await?;
        Ok(HealthAnswer {
            healthy: answer.healthy,
            message: answer.message,
        })
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, Status> {
        Ok(self.get_loads(load_sections()).await?.into())
    }

    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, Status> {
        Ok(flushed(self.flush_cache(timeout_s).await?))
    }

    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, Status> {
        Ok(profiled(
            self.start_profile(profile_request(options)).await?,
        ))
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, Status> {
        Ok(profiled(self.stop_profile().await?))
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ModelInfo::TokenSpeed(Box::new(self.get_model_info().await?)).to_labels())
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, Status> {
        Ok(ServerInfo::TokenSpeed(Box::new(self.get_server_info().await?)).to_labels())
    }

    async fn subscribe_kv_events(&self, start_seq: u64) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_kv_events_with_load_records(start_seq).await
    }

    async fn subscribe_load_records(&self) -> Result<Streaming<KvEventBatch>, Status> {
        self.subscribe_load_records().await
    }
}
