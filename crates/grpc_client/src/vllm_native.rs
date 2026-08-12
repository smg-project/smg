//! Client for vLLM's first-party gRPC services.
//!
//! vLLM serves two native gRPC services from the engine process itself
//! (vendored protos: `proto/vllm_native/`, see its README for provenance):
//!
//! - `vllm.Inference` — `Generate` / `GenerateStream`
//! - `vllm.Control` — `GetServerInfo`, `GetModelInfo`, `Abort`,
//!   `GetKvEventSources`
//!
//! Connecting to these directly removes the injected Python servicer from the
//! serving path for engines that expose them. This module is the transport
//! layer only — worker detection and routing-pipeline integration land
//! separately; nothing constructs this client in production yet.
//!
//! Notable protocol properties, load-bearing for the follow-up wiring:
//!
//! - `ModelInfo` declares the engine's own `tool_call_parser` /
//!   `reasoning_parser` names — a native source for per-model parser
//!   overrides (today those come from worker labels).
//! - `ServerInfo` carries `kv_block_size`, `total_kv_blocks`, scheduler
//!   limits, and full parallelism info (including decode-context-parallel
//!   size), so capacity and block-size registration need no side channel.
//! - `GetKvEventSources` returns per-data-parallel-rank event endpoints with
//!   replay endpoints — the discovery half of event-driven cache-aware
//!   routing without a subscription shim.
//! - Data-parallel rank targeting is out-of-band: requests carry the
//!   `x-data-parallel-rank` metadata header rather than a body field.
//! - Cancellation is explicit (`Control.Abort` with request ids); dropping a
//!   response stream does not abort the engine-side request by itself.

use std::{future::Future, pin::Pin};

use tonic::{transport::Channel, Request, Streaming};
use tracing::{debug, warn};

use crate::{AbortOnDropClient, BoxedTraceInjector};

// Include the generated protobuf code. Unlike the sibling proto modules this
// one contains enums, whose generated `as_str_name(&self)` trips the
// pass-by-value lint — silence it for generated code only.
#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    tonic::include_proto!("vllm");
}

/// Metadata key selecting a data-parallel rank for a request.
pub const DATA_PARALLEL_RANK_METADATA_KEY: &str = "x-data-parallel-rank";

/// Streaming `generate_stream()` response that auto-aborts on drop. Concrete
/// alias for the generic `crate::AbortOnDropStream`.
pub type AbortOnDropStream = crate::AbortOnDropStream<proto::GenerateResponse, VllmNativeClient>;

/// Client for vLLM's first-party `Inference` + `Control` gRPC services.
///
/// Both service clients share one HTTP/2 channel.
#[derive(Clone)]
pub struct VllmNativeClient {
    inference: proto::inference_client::InferenceClient<Channel>,
    control: proto::control_client::ControlClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl AbortOnDropClient for VllmNativeClient {
    fn abort_for_drop(
        self,
        request_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>> {
        Box::pin(async move {
            let mut control = self.control;
            control
                .abort(Request::new(proto::AbortRequest {
                    request_ids: vec![request_id],
                }))
                .await
                .map(|_| ())
        })
    }
}

impl VllmNativeClient {
    /// Create a new client and connect to the engine's gRPC listener.
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, std::sync::Arc::new(crate::NoopTraceInjector))
            .await
    }

    /// Create a new client with a custom trace injector.
    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to vLLM native gRPC server at {endpoint}");
        let channel = crate::channel::connect_channel(endpoint).await?;
        Ok(Self {
            inference: proto::inference_client::InferenceClient::new(channel.clone()),
            control: proto::control_client::ControlClient::new(channel),
            trace_injector,
        })
    }

    /// Set or replace the trace injector.
    #[must_use]
    pub fn with_trace_injector(mut self, trace_injector: BoxedTraceInjector) -> Self {
        self.trace_injector = trace_injector;
        self
    }

    /// Engine/server capabilities: versions, parallelism, KV block size and
    /// capacity, scheduler limits.
    pub async fn get_server_info(&self) -> Result<proto::ServerInfo, tonic::Status> {
        let mut request = Request::new(proto::GetServerInfoRequest {});
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {e}");
        }
        let mut client = self.control.clone();
        Ok(client.get_server_info(request).await?.into_inner())
    }

    /// Model identity and capabilities, including the engine's own
    /// `tool_call_parser` / `reasoning_parser` declarations.
    pub async fn get_model_info(&self) -> Result<proto::ModelInfo, tonic::Status> {
        let mut request = Request::new(proto::GetModelInfoRequest {});
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {e}");
        }
        let mut client = self.control.clone();
        Ok(client.get_model_info(request).await?.into_inner())
    }

    /// Per-data-parallel-rank KV event source descriptors (transport,
    /// endpoint, topic, replay endpoint).
    pub async fn get_kv_event_sources(
        &self,
    ) -> Result<proto::GetKvEventSourcesResponse, tonic::Status> {
        let mut request = Request::new(proto::GetKvEventSourcesRequest {});
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {e}");
        }
        let mut client = self.control.clone();
        Ok(client.get_kv_event_sources(request).await?.into_inner())
    }

    /// Abort in-flight requests by id. Idempotent server-side.
    pub async fn abort(&self, request_ids: Vec<String>) -> Result<(), tonic::Status> {
        let mut request = Request::new(proto::AbortRequest { request_ids });
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {e}");
        }
        let mut client = self.control.clone();
        client.abort(request).await.map(|_| ())
    }

    /// Streaming generation. `data_parallel_rank`, when set, rides the
    /// `x-data-parallel-rank` metadata header (the protocol carries no body
    /// field for it). The returned stream is raw; wrap it in
    /// [`AbortOnDropStream`] at the call site that owns cancellation policy —
    /// dropping the raw stream does NOT abort the engine-side request.
    pub async fn generate_stream(
        &self,
        generate_request: proto::GenerateRequest,
        data_parallel_rank: Option<u32>,
    ) -> Result<Streaming<proto::GenerateResponse>, tonic::Status> {
        let mut request = Request::new(generate_request);
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {e}");
        }
        if let Some(rank) = data_parallel_rank {
            request.metadata_mut().insert(
                DATA_PARALLEL_RANK_METADATA_KEY,
                rank.to_string()
                    .parse()
                    .map_err(|_| tonic::Status::internal("invalid dp rank metadata"))?,
            );
        }
        let mut client = self.inference.clone();
        Ok(client.generate_stream(request).await?.into_inner())
    }
}
