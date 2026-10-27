//! Unified gRPC client wrapper for SGLang, vLLM, and TensorRT-LLM backends

use std::{collections::HashMap, sync::Arc};

use openai_protocol::{
    chat::ChatCompletionRequest,
    completion::CompletionRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
    worker::{ConstrainedDecoding, WorkerLoadResponse},
};
use smg_grpc_client::{
    common_proto, mlx_proto, sglang_proto, tokenizer_bundle, tokenizer_bundle::StreamBundle,
    tokenspeed_proto, trtllm_proto, vllm_proto, BoxedTraceInjector, MlxEngineClient,
    SglangGenerateRequestOptions, SglangSchedulerClient, TokenSpeedSchedulerClient,
    TrtllmServiceClient, VllmEngineClient,
};

use super::multimodal::SUPPORTS_VISION_LABEL;
use crate::{
    observability::otel_trace::OtelTraceInjector,
    routers::grpc::{
        proto_wrapper::{
            cleanup_mm_shm_handles, collect_tokenspeed_generate_request_shm_handles,
            collect_vllm_generate_request_shm_handles, finish_tokenspeed_request,
            finish_vllm_request, ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateRequest,
            ProtoStream,
        },
        MultimodalData,
    },
};

/// Health check response (common across backends)
#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

/// TRT-LLM reports liveness via a free-form status string whose only healthy
/// value is `"OK"` (per `trtllm_service.proto`); anything else is an error
/// description and must be treated as unhealthy.
fn trtllm_status_healthy(status: &str) -> bool {
    status.trim().eq_ignore_ascii_case("ok")
}

/// Wraps the per-backend gRPC clients. RPCs absent on a backend's wire
/// return `Status::unimplemented`.
#[derive(Clone)]
pub enum GrpcClient {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
    Trtllm(TrtllmServiceClient),
    Mlx(MlxEngineClient),
    TokenSpeed(TokenSpeedSchedulerClient),
}

#[derive(Default)]
pub struct GenerateRequestBuildOptions {
    pub multimodal_inputs: Option<MultimodalData>,
    pub tool_constraints: Option<(String, String)>,
    pub require_reasoning: bool,
    /// `tool_constraints` was led with the model's thought and its close for
    /// a worker whose engine starts the grammar at the first generated token:
    /// the request says so where its dialect has a switch for the gate behind
    /// the engine's reasoning parser
    /// (`ProtoGenerateRequest::set_grammar_from_first_token`).
    pub first_token_grammar: bool,
}

impl GrpcClient {
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    pub fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_tokenspeed() check"
    )]
    pub fn as_tokenspeed(&self) -> &TokenSpeedSchedulerClient {
        match self {
            Self::TokenSpeed(client) => client,
            _ => panic!("Expected TokenSpeed client"),
        }
    }

    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    /// Runtime type backing this client. Lets shared logic (e.g. the multimodal
    /// capability matrix) key on the backend without matching every variant.
    pub fn runtime_type(&self) -> crate::worker::RuntimeType {
        use crate::worker::RuntimeType;
        match self {
            Self::Sglang(_) => RuntimeType::Sglang,
            Self::Vllm(_) => RuntimeType::Vllm,
            Self::Trtllm(_) => RuntimeType::Trtllm,
            Self::Mlx(_) => RuntimeType::Mlx,
            Self::TokenSpeed(_) => RuntimeType::TokenSpeed,
        }
    }

    pub async fn connect(
        url: &str,
        runtime_type: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Every RPC carries the current span's W3C trace context in its
        // metadata, so the engine side can continue the gateway's trace (a
        // no-op while tracing is off).
        let trace_injector: BoxedTraceInjector = Arc::new(OtelTraceInjector);
        match runtime_type {
            "sglang" => Ok(Self::Sglang(
                SglangSchedulerClient::connect_with_trace_injector(url, trace_injector).await?,
            )),
            "vllm" => Ok(Self::Vllm(
                VllmEngineClient::connect_with_trace_injector(url, trace_injector).await?,
            )),
            "trtllm" | "tensorrt-llm" => Ok(Self::Trtllm(
                TrtllmServiceClient::connect_with_trace_injector(url, trace_injector).await?,
            )),
            "mlx" => Ok(Self::Mlx(
                MlxEngineClient::connect_with_trace_injector(url, trace_injector).await?,
            )),
            "tokenspeed" => Ok(Self::TokenSpeed(
                TokenSpeedSchedulerClient::connect_with_trace_injector(url, trace_injector).await?,
            )),
            _ => Err(format!("Unknown runtime type: {runtime_type}").into()),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Sglang(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Vllm(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Trtllm(client) => {
                let resp = client.health_check().await?;
                let healthy = trtllm_status_healthy(&resp.status);
                Ok(HealthCheckResponse {
                    healthy,
                    message: resp.status,
                })
            }
            Self::Mlx(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::TokenSpeed(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Sglang(client) => Ok(ModelInfo::Sglang(Box::new(client.get_model_info().await?))),
            Self::Vllm(client) => Ok(ModelInfo::Vllm(client.get_model_info().await?)),
            Self::Trtllm(client) => Ok(ModelInfo::Trtllm(client.get_model_info().await?)),
            Self::Mlx(client) => Ok(ModelInfo::Mlx(client.get_model_info().await?)),
            Self::TokenSpeed(client) => Ok(ModelInfo::TokenSpeed(Box::new(
                client.get_model_info().await?,
            ))),
        }
    }

    /// Get the full load response from the backend.
    /// Returns `Unimplemented` for backends without scheduler load metrics.
    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        // Optional sections beyond `core` (disagg/queues/memory) are dropped by
        // engines that do not report them, so requesting them is always safe and
        // leaves routing consumers, which only read `core`, unaffected.
        let include = || {
            ["core", "disagg", "queues", "memory"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        };
        match self {
            Self::Sglang(client) => {
                let resp = client.get_loads(include()).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            Self::TokenSpeed(client) => {
                let resp = client.get_loads(include()).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            Self::Vllm(client) => {
                let resp = client.get_loads(include()).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            _ => Err(tonic::Status::unimplemented(
                "GetLoads RPC not supported for this backend",
            )),
        }
    }

    /// Flush the KV cache on the backend. Returns `Unimplemented` for
    /// backends without the FlushCache RPC.
    pub async fn flush_cache(
        &self,
        timeout_s: f32,
    ) -> Result<common_proto::FlushCacheResponse, tonic::Status> {
        match self {
            Self::Sglang(client) => client.flush_cache(timeout_s).await,
            Self::TokenSpeed(client) => client.flush_cache(timeout_s).await,
            Self::Vllm(client) => client.flush_cache(timeout_s).await,
            Self::Trtllm(_) | Self::Mlx(_) => Err(tonic::Status::unimplemented(
                "FlushCache RPC not supported for this backend",
            )),
        }
    }

    /// Start the profiler on the backend. Returns `Unimplemented` for
    /// backends without the StartProfile RPC.
    pub async fn start_profile(
        &self,
        req: common_proto::StartProfileRequest,
    ) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Sglang(client) => client.start_profile(req).await,
            Self::TokenSpeed(client) => client.start_profile(req).await,
            Self::Vllm(client) => client.start_profile(req).await,
            Self::Trtllm(_) | Self::Mlx(_) => Err(tonic::Status::unimplemented(
                "StartProfile RPC not supported for this backend",
            )),
        }
    }

    /// Stop the profiler on the backend. Returns `Unimplemented` for
    /// backends without the StopProfile RPC.
    pub async fn stop_profile(&self) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Sglang(client) => client.stop_profile().await,
            Self::TokenSpeed(client) => client.stop_profile().await,
            Self::Vllm(client) => client.stop_profile().await,
            Self::Trtllm(_) | Self::Mlx(_) => Err(tonic::Status::unimplemented(
                "StopProfile RPC not supported for this backend",
            )),
        }
    }

    /// Subscribe to KV cache events, asking the servicer for its load records
    /// before the publisher's first batch: the gateway's stream reader skips a
    /// `load_only` batch before any sequence check (the KV-event monitor's
    /// `process_stream`), so the records numbered 0 cost it nothing and a
    /// worker's first request is judged like any later one. Returns
    /// `Unimplemented` on backends without KV-event streaming.
    pub async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Sglang(client) => {
                client
                    .subscribe_kv_events_with_load_records(start_seq)
                    .await
            }
            Self::Vllm(client) => {
                client
                    .subscribe_kv_events_with_load_records(start_seq)
                    .await
            }
            Self::Trtllm(client) => {
                client
                    .subscribe_kv_events_with_load_records(start_seq)
                    .await
            }
            Self::TokenSpeed(client) => {
                client
                    .subscribe_kv_events_with_load_records(start_seq)
                    .await
            }
            Self::Mlx(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents RPC not supported for MLX backend",
            )),
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Sglang(client) => Ok(ServerInfo::Sglang(Box::new(
                client.get_server_info().await?,
            ))),
            Self::Vllm(client) => Ok(ServerInfo::Vllm(Box::new(client.get_server_info().await?))),
            Self::Trtllm(client) => Ok(ServerInfo::Trtllm(client.get_server_info().await?)),
            Self::Mlx(client) => Ok(ServerInfo::Mlx(client.get_server_info().await?)),
            Self::TokenSpeed(client) => Ok(ServerInfo::TokenSpeed(Box::new(
                client.get_server_info().await?,
            ))),
        }
    }

    /// Fetch tokenizer bundle from backend runtime and validate integrity/safety.
    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        let bundle = match self {
            Self::Sglang(client) => client.get_tokenizer().await,
            Self::Vllm(client) => client.get_tokenizer().await,
            Self::Trtllm(client) => client.get_tokenizer().await,
            Self::Mlx(client) => client.get_tokenizer().await,
            Self::TokenSpeed(client) => client.get_tokenizer().await,
        }?;

        tokenizer_bundle::validate_bundle_sha256(&bundle).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Tokenizer bundle SHA256 validation failed: {e}"),
            )
        })?;

        Ok(bundle)
    }

    /// Generate streaming response from request
    ///
    /// Dispatches to the appropriate backend client and wraps the result in ProtoStream.
    /// Returns `tonic::Status` on error so callers can inspect the gRPC status code directly.
    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match (self, req) {
            (Self::Sglang(client), ProtoGenerateRequest::Sglang(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Sglang(stream))
            }
            (Self::Vllm(client), ProtoGenerateRequest::Vllm(boxed_req)) => {
                let shm_handles = collect_vllm_generate_request_shm_handles(&boxed_req);
                match client.generate(*boxed_req).await {
                    Ok(stream) => Ok(ProtoStream::Vllm(stream)),
                    Err(error) => {
                        cleanup_mm_shm_handles(&shm_handles);
                        Err(error)
                    }
                }
            }
            (Self::Trtllm(client), ProtoGenerateRequest::Trtllm(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Trtllm(stream))
            }
            (Self::Mlx(client), ProtoGenerateRequest::Mlx(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Mlx(stream))
            }
            (Self::TokenSpeed(client), ProtoGenerateRequest::TokenSpeed(boxed_req)) => {
                let shm_handles = collect_tokenspeed_generate_request_shm_handles(&boxed_req);
                match client.generate(*boxed_req).await {
                    Ok(stream) => Ok(ProtoStream::TokenSpeed(stream)),
                    Err(error) => {
                        cleanup_mm_shm_handles(&shm_handles);
                        Err(error)
                    }
                }
            }
            #[expect(
                clippy::panic,
                reason = "client and request types are always matched by construction in the pipeline"
            )]
            _ => panic!("Mismatched client and request types"),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match (self, req) {
            (Self::Sglang(client), ProtoEmbedRequest::Sglang(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedComplete::Sglang(resp))
            }
            (Self::Vllm(client), ProtoEmbedRequest::Vllm(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedComplete::Vllm(resp))
            }
            (Self::TokenSpeed(_), _) => Err(tonic::Status::unimplemented(
                "TokenSpeed backend does not support embedding",
            )),
            (Self::Mlx(_), _) => Err(tonic::Status::unimplemented(
                "MLX backend does not support embedding",
            )),
            #[expect(
                clippy::panic,
                reason = "client and request types are always matched by construction in the pipeline"
            )]
            _ => panic!("Mismatched client and request types or unsupported embedding backend"),
        }
    }

    /// Build the engine's generate request for a chat completion body.
    pub fn build_chat_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        self.build_engine_request(
            request_id,
            body,
            processed_text,
            token_ids,
            options,
            EngineBuilders {
                sglang: SglangSchedulerClient::build_generate_request_from_chat,
                vllm: VllmEngineClient::build_generate_request_from_chat,
                trtllm: TrtllmServiceClient::build_generate_request_from_chat,
                mlx: MlxEngineClient::build_generate_request_from_chat,
                tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_chat,
            },
        )
    }

    /// Build the engine's generate request for a Messages body.
    pub fn build_messages_request(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        self.build_engine_request(
            request_id,
            body,
            processed_text,
            token_ids,
            options,
            EngineBuilders {
                sglang: SglangSchedulerClient::build_generate_request_from_messages,
                vllm: VllmEngineClient::build_generate_request_from_messages,
                trtllm: TrtllmServiceClient::build_generate_request_from_messages,
                mlx: MlxEngineClient::build_generate_request_from_messages,
                tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_messages,
            },
        )
    }

    /// Build the engine's generate request for a completion body.
    pub fn build_completion_request(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        self.build_engine_plain_request(
            request_id,
            body,
            original_text,
            token_ids,
            EnginePlainBuilders {
                sglang: SglangSchedulerClient::build_generate_request_from_completion,
                vllm: VllmEngineClient::build_generate_request_from_completion,
                trtllm: TrtllmServiceClient::build_generate_request_from_completion,
                mlx: MlxEngineClient::build_generate_request_from_completion,
                tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_completion,
            },
        )
    }

    /// Build the engine's generate request for a plain generate body.
    pub fn build_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        self.build_engine_plain_request(
            request_id,
            body,
            original_text,
            token_ids,
            EnginePlainBuilders {
                sglang: SglangSchedulerClient::build_plain_generate_request,
                vllm: VllmEngineClient::build_plain_generate_request,
                trtllm: TrtllmServiceClient::build_plain_generate_request,
                mlx: MlxEngineClient::build_plain_generate_request,
                tokenspeed: TokenSpeedSchedulerClient::build_plain_generate_request,
            },
        )
    }

    /// The per-engine dispatch of a request kind that carries multimodal inputs
    /// and tool constraints: the engine's own multimodal variant out of the
    /// assembled data, its options, its builder, and the boxing into the
    /// request enum. vLLM and TokenSpeed build through the wrappers that unlink
    /// the `/dev/shm` segments of a request that fails to build.
    fn build_engine_request<B>(
        &self,
        request_id: String,
        body: &B,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
        builders: EngineBuilders<B>,
    ) -> Result<ProtoGenerateRequest, String> {
        let GenerateRequestBuildOptions {
            multimodal_inputs,
            tool_constraints,
            require_reasoning,
            first_token_grammar,
        } = options;
        let EngineBuilders {
            sglang,
            vllm,
            trtllm,
            mlx,
            tokenspeed,
        } = builders;
        let mut request = match self {
            Self::Sglang(client) => {
                let sglang_mm = engine_multimodal(multimodal_inputs, |mm| match mm {
                    MultimodalData::Sglang(data) => Some(data.into_proto()),
                    _ => None,
                });
                let req = sglang(
                    client,
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    SglangGenerateRequestOptions {
                        multimodal_inputs: sglang_mm,
                        tool_call_constraint: tool_constraints,
                        require_reasoning,
                    },
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(_) => {
                let vllm_mm = engine_multimodal(multimodal_inputs, |mm| match mm {
                    MultimodalData::Vllm(data) => Some(data.into_protos()),
                    _ => None,
                });
                finish_vllm_request(vllm_mm, |mm| {
                    vllm(
                        request_id,
                        body,
                        processed_text,
                        token_ids,
                        mm,
                        tool_constraints,
                    )
                })
            }
            Self::Trtllm(client) => {
                let trtllm_mm = engine_multimodal(multimodal_inputs, |mm| match mm {
                    MultimodalData::Trtllm(data) => Some(data.into_proto()),
                    _ => None,
                });
                let req = trtllm(
                    client,
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    trtllm_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            // MLX: the assembly stage rejects multimodal before this path.
            Self::Mlx(client) => {
                let req = mlx(
                    client,
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(_) => {
                let tokenspeed_mm = engine_multimodal(multimodal_inputs, |mm| match mm {
                    MultimodalData::TokenSpeed(data) => Some(data.into_proto(true)),
                    _ => None,
                });
                finish_tokenspeed_request(tokenspeed_mm, |mm| {
                    tokenspeed(
                        request_id,
                        body,
                        processed_text,
                        token_ids,
                        mm,
                        tool_constraints,
                    )
                })
            }
        }?;
        request.set_grammar_from_first_token(first_token_grammar);
        Ok(request)
    }

    /// The per-engine dispatch of a request kind with neither multimodal inputs
    /// nor tool constraints: the engine's builder, boxed into the request enum.
    fn build_engine_plain_request<B, T>(
        &self,
        request_id: String,
        body: &B,
        text: T,
        token_ids: Vec<u32>,
        builders: EnginePlainBuilders<B, T>,
    ) -> Result<ProtoGenerateRequest, String> {
        let EnginePlainBuilders {
            sglang,
            vllm,
            trtllm,
            mlx,
            tokenspeed,
        } = builders;
        Ok(match self {
            Self::Sglang(client) => ProtoGenerateRequest::Sglang(Box::new(sglang(
                client, request_id, body, text, token_ids,
            )?)),
            Self::Vllm(_) => {
                ProtoGenerateRequest::Vllm(Box::new(vllm(request_id, body, text, token_ids)?))
            }
            Self::Trtllm(client) => ProtoGenerateRequest::Trtllm(Box::new(trtllm(
                client, request_id, body, text, token_ids,
            )?)),
            Self::Mlx(client) => {
                ProtoGenerateRequest::Mlx(Box::new(mlx(client, request_id, body, text, token_ids)?))
            }
            Self::TokenSpeed(_) => ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed(
                request_id, body, text, token_ids,
            )?)),
        })
    }
}

/// A vLLM builder for a request kind carrying multimodal inputs and tool
/// constraints (chat, Messages). The ZMQ dispatch in `backend_client` builds
/// through the same functions.
pub(crate) type VllmMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    Option<vllm_proto::MultimodalInputs>,
    Option<(String, String)>,
) -> Result<vllm_proto::GenerateRequest, String>;

/// The TokenSpeed counterpart of [`VllmMmBuilder`].
pub(crate) type TokenSpeedMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    Option<tokenspeed_proto::MultimodalInputs>,
    Option<(String, String)>,
) -> Result<tokenspeed_proto::GenerateRequest, String>;

/// The SGLang counterpart of [`VllmMmBuilder`] on the connected client; its
/// builders take the multimodal inputs and constraints bundled as options.
type SglangClientMmBuilder<B> = fn(
    &SglangSchedulerClient,
    String,
    &B,
    String,
    Vec<u32>,
    SglangGenerateRequestOptions,
) -> Result<sglang_proto::GenerateRequest, String>;

/// The TRT-LLM counterpart of [`VllmMmBuilder`] on the connected client.
type TrtllmClientMmBuilder<B> = fn(
    &TrtllmServiceClient,
    String,
    &B,
    String,
    Vec<u32>,
    Option<trtllm_proto::MultimodalInput>,
    Option<(String, String)>,
) -> Result<trtllm_proto::GenerateRequest, String>;

/// The MLX counterpart of [`VllmMmBuilder`] on the connected client; MLX
/// takes no multimodal inputs (the assembly stage refuses them earlier).
type MlxClientMmBuilder<B> = fn(
    &MlxEngineClient,
    String,
    &B,
    String,
    Vec<u32>,
    Option<(String, String)>,
) -> Result<mlx_proto::GenerateRequest, String>;

/// A vLLM builder for a request kind with neither multimodal inputs nor tool
/// constraints (completion, plain generate); `T` is the builder's text
/// parameter. The ZMQ dispatch in `backend_client` builds through the same
/// functions.
pub(crate) type VllmPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<vllm_proto::GenerateRequest, String>;

/// The TokenSpeed counterpart of [`VllmPlainBuilder`].
pub(crate) type TokenSpeedPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<tokenspeed_proto::GenerateRequest, String>;

/// The SGLang counterpart of [`VllmPlainBuilder`] on the connected client.
type SglangClientPlainBuilder<B, T> = fn(
    &SglangSchedulerClient,
    String,
    &B,
    T,
    Vec<u32>,
) -> Result<sglang_proto::GenerateRequest, String>;

/// The TRT-LLM counterpart of [`VllmPlainBuilder`] on the connected client.
type TrtllmClientPlainBuilder<B, T> = fn(
    &TrtllmServiceClient,
    String,
    &B,
    T,
    Vec<u32>,
) -> Result<trtllm_proto::GenerateRequest, String>;

/// The MLX counterpart of [`VllmPlainBuilder`] on the connected client.
type MlxClientPlainBuilder<B, T> =
    fn(&MlxEngineClient, String, &B, T, Vec<u32>) -> Result<mlx_proto::GenerateRequest, String>;

/// The engines' builders of one request kind that carries multimodal inputs and
/// tool constraints (chat, Messages): the only thing that differs between the
/// two build surfaces, so the per-engine dispatch lives once in
/// [`GrpcClient::build_engine_request`]. SGLang, TRT-LLM and MLX build on their
/// client; vLLM and TokenSpeed on the type.
struct EngineBuilders<B> {
    sglang: SglangClientMmBuilder<B>,
    vllm: VllmMmBuilder<B>,
    trtllm: TrtllmClientMmBuilder<B>,
    mlx: MlxClientMmBuilder<B>,
    tokenspeed: TokenSpeedMmBuilder<B>,
}

/// The engines' builders of one request kind with neither multimodal inputs nor
/// tool constraints (completion, plain generate); `T` is the builders' text
/// parameter (`String` for completion, `Option<String>` for plain generate).
/// The per-engine dispatch lives once in
/// [`GrpcClient::build_engine_plain_request`].
struct EnginePlainBuilders<B, T> {
    sglang: SglangClientPlainBuilder<B, T>,
    vllm: VllmPlainBuilder<B, T>,
    trtllm: TrtllmClientPlainBuilder<B, T>,
    mlx: MlxClientPlainBuilder<B, T>,
    tokenspeed: TokenSpeedPlainBuilder<B, T>,
}

/// The assembled multimodal data as the proto of the worker's engine, `own`
/// picking the engine's variant. The assembly stage builds the variant of the
/// worker's engine, so any other variant is a gateway bug.
#[expect(
    clippy::unreachable,
    reason = "assembly stage guarantees matching MultimodalData variant for each backend"
)]
fn engine_multimodal<T>(
    inputs: Option<MultimodalData>,
    own: impl FnOnce(MultimodalData) -> Option<T>,
) -> Option<T> {
    inputs.map(|mm| {
        let Some(proto) = own(mm) else {
            unreachable!("caller guarantees matching variant")
        };
        proto
    })
}

// ---------------------------------------------------------------------------
// Metadata wrappers
// ---------------------------------------------------------------------------

pub enum ModelInfo {
    Sglang(Box<sglang_proto::GetModelInfoResponse>),
    Vllm(vllm_proto::GetModelInfoResponse),
    Trtllm(trtllm_proto::GetModelInfoResponse),
    Mlx(mlx_proto::GetModelInfoResponse),
    TokenSpeed(Box<tokenspeed_proto::GetModelInfoResponse>),
}

pub enum ServerInfo {
    Sglang(Box<sglang_proto::GetServerInfoResponse>),
    Vllm(Box<vllm_proto::GetServerInfoResponse>),
    Trtllm(trtllm_proto::GetServerInfoResponse),
    Mlx(mlx_proto::GetServerInfoResponse),
    TokenSpeed(Box<tokenspeed_proto::GetServerInfoResponse>),
}

impl ModelInfo {
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            // The vLLM servicer computes `supports_vision` itself (the
            // engine's own verdict; `false` = `--language-model-only` or a
            // text model), so its `false` is a fact worth a label. One
            // caveat: when that computation fails on an unexpected config
            // shape, the servicer logs a warning and reports `false` as
            // well, and a vision model in that state is refused images
            // with the label's message instead of being served.
            ModelInfo::Vllm(info) => flat_labels(info),
            ModelInfo::Sglang(info) => without_unset_vision(flat_labels(info)),
            ModelInfo::Trtllm(info) => without_unset_vision(flat_labels(info)),
            ModelInfo::Mlx(info) => without_unset_vision(flat_labels(info)),
            ModelInfo::TokenSpeed(info) => without_unset_vision(flat_labels(info)),
        }
    }
}

/// A proto3 `bool` cannot tell "unset" from `false`. The SGLang Python
/// servicer reads `supports_vision` from its scheduler info with a default
/// of `false` (a vision model served by it was refused images on that
/// default), and a servicer that predates the field sends the proto default;
/// TokenSpeed's servicer derives the field from its static modality list, an
/// accurate answer, but the field cannot tell that derivation from an older
/// servicer's default. For every runtime but vLLM a `false` is therefore
/// dropped as no verdict: no label, so nothing downstream reads a refusal
/// into it; a `true` is kept.
fn without_unset_vision(mut labels: HashMap<String, String>) -> HashMap<String, String> {
    if labels
        .get(SUPPORTS_VISION_LABEL)
        .is_some_and(|value| value == "false")
    {
        labels.remove(SUPPORTS_VISION_LABEL);
    }
    labels
}

impl ServerInfo {
    /// Convert to labels. SGLang needs special handling because its `server_args`
    /// is a `prost_types::Struct` (not Serialize). vLLM/TRT-LLM are plain structs.
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            ServerInfo::Sglang(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, SGLANG_GRPC_KEYS);
                }
                if !info.sglang_version.is_empty() {
                    labels.insert("version".to_string(), info.sglang_version.clone());
                }
                labels
            }
            ServerInfo::Vllm(info) => flat_labels(info),
            ServerInfo::Trtllm(info) => flat_labels(info),
            ServerInfo::Mlx(info) => flat_labels(info),
            ServerInfo::TokenSpeed(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, TOKENSPEED_GRPC_KEYS);
                }
                // TokenSpeed's global request window is shared by its attention-DP ranks.
                labels.insert("enable_dp_attention".to_string(), "true".to_string());
                if !info.tokenspeed_version.is_empty() {
                    labels.insert("version".to_string(), info.tokenspeed_version.clone());
                }
                // The engine's declared constrained-decoding policy, a field of
                // its own rather than a `server_args` key: the `constrained_decoding`
                // label the gateway follows per worker. "" is no declaration.
                if !info.constrained_decoding.is_empty() {
                    labels.insert(
                        ConstrainedDecoding::LABEL.to_string(),
                        info.constrained_decoding.clone(),
                    );
                }
                // Carry the worker's /dev/shm namespace identity (advertised in
                // scheduler_info). The router compares it to its own to decide the
                // SHM tensor transport by *verifying* a shared /dev/shm rather than
                // inferring it from the worker URL. See `worker_shares_dev_shm`.
                if let Some(ref sched) = info.scheduler_info {
                    pick_prost_fields(&mut labels, sched, &["shm_namespace_id"]);
                    if let Some(prost_types::Value {
                        kind: Some(prost_types::value::Kind::ListValue(values)),
                    }) = sched.fields.get("cache_trace_epochs")
                    {
                        let epochs: Option<Vec<&str>> = values
                            .values
                            .iter()
                            .map(|value| match &value.kind {
                                Some(prost_types::value::Kind::StringValue(epoch))
                                    if !epoch.is_empty() =>
                                {
                                    Some(epoch.as_str())
                                }
                                _ => None,
                            })
                            .collect();
                        if let Some(epochs) = epochs.filter(|epochs| !epochs.is_empty()) {
                            labels.insert(
                                "cache_trace_epochs".to_string(),
                                serde_json::json!(epochs).to_string(),
                            );
                        }
                    }
                }
                labels
            }
        }
    }
}

/// Keys worth extracting from SGLang gRPC `server_args` (which contains the full config).
const SGLANG_GRPC_KEYS: &[&str] = &[
    "model_path",
    "served_model_name",
    "tokenizer_path",
    "tp_size",
    "dp_size",
    "pp_size",
    "context_length",
    "max_total_tokens",
    "max_running_requests",
    // Whether the engine splits `max_running_requests` across its
    // attention-DP workers (the flag, or the width its newer releases resolve
    // it into): under `--dp-aware` each rank worker then gets its share of
    // the window (see `create_worker::labels_for_rank`).
    "enable_dp_attention",
    "attn_dp_size",
    "load_balance_method",
    "disaggregation_mode",
    "is_embedding",
    "vocab_size",
    "weight_version",
    // PD pairing protocol: the transport and KV layout a prefill and a decode
    // must share for a handoff to work (the engines validate TP themselves).
    "disaggregation_transfer_backend",
    "disaggregation_bootstrap_port",
    "kv_cache_dtype",
    "page_size",
    "attention_backend",
    // The operator's explicit protocol, which the servicer reads from the
    // engine's SMG_PAIRING_PROTOCOL environment.
    "pairing_protocol",
    // The engine's own per-request media limits (`--limit-mm-data-per-request`),
    // flattened by the servicer to `image=1,video=1`: the gateway's own media
    // pipeline holds requests to them, as the engine's front end would.
    "mm_item_limits",
    // The engine's reasoning parser, under a name of its own (`reasoning_parser`
    // is the per-model parser override read off a worker's labels), and where
    // the engine starts a request's grammar on a thinking prompt, which shapes
    // a forced tool call's grammar (`utils::tool_constraints_for_worker`); the
    // servicer writes both.
    "engine_reasoning_parser",
    "constrained_decoding_start",
    // A servicer's own vision verdict. `GetModelInfo.supports_vision` is a
    // proto3 bool, so a `false` there cannot be told from a field the servicer
    // never set and is dropped for this runtime (`ModelInfo::to_labels`); a
    // servicer that knows it carries no media to its engine writes the
    // verdict here on purpose, and the gateway refuses image and video input
    // for the worker at its gate instead of after fetching it.
    "supports_vision",
];

/// Keys worth extracting from TokenSpeed gRPC `server_args` (post-rename: bare
/// names, not `_path` variants — TokenSpeed dropped the legacy suffixes).
/// TokenSpeed spells its parallelism `attn_tp_size` / `pipeline_parallel_size`
/// / `data_parallel_size`; `normalize_grpc_keys` folds those into the
/// canonical `tp_size` / `pp_size` / `dp_size`.
const TOKENSPEED_GRPC_KEYS: &[&str] = &[
    "model",
    "served_model_name",
    "tokenizer",
    "tp_size",
    "attn_tp_size",
    "dp_size",
    "data_parallel_size",
    "pp_size",
    "pipeline_parallel_size",
    "context_length",
    "max_total_tokens",
    // TokenSpeed's spelling of the scheduler's running window; the fleet
    // capacity accounting and the PD admission gate read either name.
    "max_num_seqs",
    "max_running_requests",
    "load_balance_method",
    "is_embedding",
    "vocab_size",
    "weight_version",
    // PD pairing protocol (see SGLANG_GRPC_KEYS). TokenSpeed has no page
    // size: its KV layout is the cache contract exchanged at rendezvous.
    "disaggregation_mode",
    "disaggregation_transfer_backend",
    "disaggregation_bootstrap_port",
    "kv_cache_dtype",
    "attention_backend",
    "pairing_protocol",
    // The engine's own per-request media limits, flattened by the servicer as
    // for SGLang (`image=4`); TokenSpeed's engine has none to advertise today.
    "mm_item_limits",
    // The engine's reasoning parser and where it starts a request's grammar
    // on a thinking prompt, as for SGLang above; the servicer writes both.
    "engine_reasoning_parser",
    "constrained_decoding_start",
    // RL control plane (crates/rl): where the engine's SGLang-compatible
    // control app listens and what it implements. The engine advertises
    // these; SMG's discovery turns them into the labels the RL crate reads.
    "rl.control_url",
    "rl.pause_modes",
    "rl.update_from",
    "rl.abort",
    "rl.flush_cache",
    "rl.sleep_wake",
    "rl.reports_weight_version",
];

// ---------------------------------------------------------------------------
// Label helpers
// ---------------------------------------------------------------------------

/// Serialize to flat label map, skipping nulls/zeros/empty.
///
/// Booleans are emitted as `"true"` / `"false"` so downstream consumers
/// (e.g. `is_generation == "false"` for embedding detection) work correctly.
pub(crate) fn flat_labels<T: serde::Serialize>(value: &T) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(value) {
        for (key, val) in obj {
            match val {
                serde_json::Value::String(s) if !s.is_empty() && s != "null" => {
                    labels.insert(key, s);
                }
                serde_json::Value::Number(n) if n.as_f64().is_some_and(|v| v != 0.0) => {
                    // Format integers without decimal point
                    let formatted = n
                        .as_i64()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| n.to_string());
                    labels.insert(key, formatted);
                }
                serde_json::Value::Bool(b) => {
                    labels.insert(key, b.to_string());
                }
                serde_json::Value::Array(arr) if !arr.is_empty() => {
                    if let Ok(s) = serde_json::to_string(&arr) {
                        labels.insert(key, s);
                    }
                }
                _ => {}
            }
        }
    }
    labels
}

/// Pick specific keys from a `prost_types::Struct`.
fn pick_prost_fields(labels: &mut HashMap<String, String>, s: &prost_types::Struct, keys: &[&str]) {
    for key in keys {
        if let Some(val) = s.fields.get(*key) {
            if let Some(ref kind) = val.kind {
                match kind {
                    prost_types::value::Kind::StringValue(s) if !s.is_empty() && s != "null" => {
                        labels.insert((*key).to_string(), s.clone());
                    }
                    prost_types::value::Kind::NumberValue(n) if *n != 0.0 => {
                        let formatted = if *n == (*n as i64) as f64 {
                            (*n as i64).to_string()
                        } else {
                            n.to_string()
                        };
                        labels.insert((*key).to_string(), formatted);
                    }
                    prost_types::value::Kind::BoolValue(b) => {
                        labels.insert((*key).to_string(), b.to_string());
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use smg_grpc_client::{sglang_proto, tokenspeed_proto, vllm_proto};

    use super::{trtllm_status_healthy, ModelInfo, ServerInfo, SUPPORTS_VISION_LABEL};

    /// A vLLM worker's `GetServerInfo` fields flatten into labels as they
    /// are: the engine's per-prompt media limits arrive as `mm_item_limits`
    /// and its encoder budget as `mm_encoder_budget`, and an engine without
    /// them (a text model) sets no such labels.
    #[test]
    fn server_info_to_labels_vllm_carries_the_engines_item_limits() {
        let info = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            server_type: "vllm-grpc".to_string(),
            mm_item_limits: "image=8,video=2".to_string(),
            mm_encoder_budget: 192_000,
            ..Default::default()
        }));
        let labels = info.to_labels();
        assert_eq!(
            labels.get("mm_item_limits").map(String::as_str),
            Some("image=8,video=2")
        );
        assert_eq!(
            labels.get("mm_encoder_budget").map(String::as_str),
            Some("192000")
        );
        let text_only = ServerInfo::Vllm(Box::default()).to_labels();
        assert!(!text_only.contains_key("mm_item_limits"));
        assert!(!text_only.contains_key("mm_encoder_budget"));
    }

    /// A servicer's reasoning-parser facts flatten into the
    /// `engine_reasoning_parser` and `constrained_decoding_start` labels the
    /// forced tool-call grammar is shaped by, and never into
    /// `reasoning_parser`, the per-model parser override registration
    /// validates against the router's own registry; an engine whose servicer
    /// reports neither (the proto default) sets no such labels.
    #[test]
    fn server_info_to_labels_carries_where_the_engine_starts_a_grammar() {
        let info = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            engine_reasoning_parser: "glm45".to_string(),
            constrained_decoding_start: "after_reasoning".to_string(),
            ..Default::default()
        }));
        let labels = info.to_labels();
        assert_eq!(
            labels.get("engine_reasoning_parser").map(String::as_str),
            Some("glm45")
        );
        assert_eq!(
            labels.get("constrained_decoding_start").map(String::as_str),
            Some("after_reasoning")
        );
        assert!(!labels.contains_key("reasoning_parser"));
        let silent = ServerInfo::Vllm(Box::default()).to_labels();
        assert!(!silent.contains_key("engine_reasoning_parser"));
        assert!(!silent.contains_key("constrained_decoding_start"));
    }

    /// vLLM discovery preserves an explicit policy and leaves old engines silent.
    #[test]
    fn vllm_server_info_discovers_decoding_policy() {
        for policy in ["", "requested", "always"] {
            let labels = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
                constrained_decoding: policy.to_string(),
                ..Default::default()
            }))
            .to_labels();
            assert_eq!(
                labels.get("constrained_decoding").map(String::as_str),
                if policy.is_empty() {
                    None
                } else {
                    Some(policy)
                }
            );
        }
    }

    /// A TokenSpeed servicer's declared constrained-decoding policy travels as
    /// a field of its own and becomes the `constrained_decoding` label the
    /// gateway follows per worker; an engine that declares none (the proto
    /// default) sets no such label.
    #[test]
    fn server_info_to_labels_carries_the_engines_constrained_decoding_policy() {
        let info = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            constrained_decoding: "always".to_string(),
            ..Default::default()
        }));
        assert_eq!(
            info.to_labels()
                .get("constrained_decoding")
                .map(String::as_str),
            Some("always")
        );
        let silent = ServerInfo::TokenSpeed(Box::default()).to_labels();
        assert!(!silent.contains_key("constrained_decoding"));
    }

    /// An SGLang servicer's own vision verdict travels in `server_args`,
    /// where a `false` is deliberate (the proto3 `GetModelInfo` bool cannot
    /// say so): kept as the label either way, absent when the servicer wrote
    /// none.
    #[test]
    fn server_info_to_labels_sglang_keeps_the_servicers_own_vision_verdict() {
        let info = |verdict: Option<bool>| {
            let mut fields = BTreeMap::from([("tp_size".to_string(), number_value(2.0))]);
            if let Some(verdict) = verdict {
                fields.insert(
                    "supports_vision".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::BoolValue(verdict)),
                    },
                );
            }
            ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
                server_args: Some(prost_types::Struct { fields }),
                ..Default::default()
            }))
        };
        assert_eq!(
            info(Some(false))
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("false")
        );
        assert_eq!(
            info(Some(true))
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("true")
        );
        let silent = info(None).to_labels();
        assert!(!silent.contains_key(SUPPORTS_VISION_LABEL));
        assert_eq!(silent.get("tp_size").map(String::as_str), Some("2"));
    }

    /// `supports_vision` is a label only when a servicer set it on purpose:
    /// vLLM's `false` stays (language-model-only / text model); the SGLang
    /// and TokenSpeed protos carry the same `bool`, which their servicers may
    /// leave at the proto default even for a vision model, so their `false`
    /// yields no label (unknown) and their `true` is kept.
    #[test]
    fn model_info_to_labels_keeps_only_a_deliberate_vision_verdict() {
        let vllm_text = ModelInfo::Vllm(vllm_proto::GetModelInfoResponse {
            supports_vision: false,
            ..Default::default()
        });
        assert_eq!(
            vllm_text
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("false")
        );
        let vllm_vision = ModelInfo::Vllm(vllm_proto::GetModelInfoResponse {
            supports_vision: true,
            ..Default::default()
        });
        assert_eq!(
            vllm_vision
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("true")
        );

        let sglang_unset = ModelInfo::Sglang(Box::default());
        assert!(!sglang_unset.to_labels().contains_key(SUPPORTS_VISION_LABEL));
        let sglang_vision = ModelInfo::Sglang(Box::new(sglang_proto::GetModelInfoResponse {
            supports_vision: true,
            ..Default::default()
        }));
        assert_eq!(
            sglang_vision
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("true")
        );

        let tokenspeed_unset = ModelInfo::TokenSpeed(Box::default());
        assert!(!tokenspeed_unset
            .to_labels()
            .contains_key(SUPPORTS_VISION_LABEL));
        // Other booleans keep flattening as before, `false` included.
        assert_eq!(
            ModelInfo::Sglang(Box::new(sglang_proto::GetModelInfoResponse {
                is_generation: false,
                ..Default::default()
            }))
            .to_labels()
            .get("is_generation")
            .map(String::as_str),
            Some("false")
        );
    }

    #[test]
    fn trtllm_status_healthy_matches_ok_exactly() {
        assert!(trtllm_status_healthy("ok"));
        assert!(trtllm_status_healthy("OK"));
        assert!(trtllm_status_healthy("  OK  "));

        assert!(!trtllm_status_healthy("not ok"));
        assert!(!trtllm_status_healthy("checking"));
        assert!(!trtllm_status_healthy("degraded"));
        assert!(!trtllm_status_healthy(""));
    }

    fn string_value(s: &str) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::StringValue(s.to_string())),
        }
    }

    fn number_value(n: f64) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(n)),
        }
    }

    /// The vLLM servicer's running window flattens into the `max_num_seqs`
    /// label the PD admission gate and the capacity tracker read; an engine
    /// that reports none (the proto default) leaves no label, as before.
    #[test]
    fn server_info_to_labels_vllm_carries_the_running_window() {
        let info = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            kv_connector: "NixlConnector".to_string(),
            max_num_seqs: 64,
            ..Default::default()
        }));
        let labels = info.to_labels();
        assert_eq!(labels.get("max_num_seqs").map(String::as_str), Some("64"));
        assert_eq!(
            labels.get("kv_connector").map(String::as_str),
            Some("NixlConnector")
        );

        let unknown = ServerInfo::Vllm(Box::default());
        assert!(!unknown.to_labels().contains_key("max_num_seqs"));
    }

    /// The `/workers` metadata path for TokenSpeed: curated `server_args` keys
    /// become labels, everything else (unlisted keys, scheduler_info, runtime
    /// state) stays out.
    #[test]
    fn server_info_to_labels_tokenspeed_picks_curated_keys_and_version() {
        let info = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("model".to_string(), string_value("Qwen/Qwen3-8B")),
                    ("tokenizer".to_string(), string_value("Qwen/Qwen3-8B")),
                    ("tp_size".to_string(), number_value(2.0)),
                    ("max_total_tokens".to_string(), number_value(8192.0)),
                    ("disaggregation_mode".to_string(), string_value("prefill")),
                    (
                        "disaggregation_transfer_backend".to_string(),
                        string_value("mooncake"),
                    ),
                    (
                        "disaggregation_bootstrap_port".to_string(),
                        number_value(8998.0),
                    ),
                    ("kv_cache_dtype".to_string(), string_value("auto")),
                    ("attention_backend".to_string(), string_value("flashinfer")),
                    ("pipeline_parallel_size".to_string(), number_value(2.0)),
                    (
                        "rl.control_url".to_string(),
                        string_value("http://10.0.0.5:40100"),
                    ),
                    (
                        "rl.pause_modes".to_string(),
                        string_value("wait,abort,keep"),
                    ),
                    (
                        "rl.reports_weight_version".to_string(),
                        string_value("true"),
                    ),
                    ("mm_item_limits".to_string(), string_value("image=4")),
                    // Not in TOKENSPEED_GRPC_KEYS — must not become a label.
                    ("host".to_string(), string_value("127.0.0.1")),
                ]),
            }),
            scheduler_info: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("status".to_string(), string_value("ready")),
                    (
                        "cache_trace_epochs".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::ListValue(
                                prost_types::ListValue {
                                    values: vec![
                                        string_value("replica-one"),
                                        string_value("replica-two"),
                                    ],
                                },
                            )),
                        },
                    ),
                ]),
            }),
            active_requests: 3,
            uptime_seconds: 12.5,
            max_total_num_tokens: 8192,
            tokenspeed_version: "0.1.0".to_string(),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("model").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        assert_eq!(
            labels.get("tokenizer").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        // Integral numbers are formatted without a decimal point.
        assert_eq!(labels.get("tp_size").map(String::as_str), Some("2"));
        assert_eq!(
            labels.get("max_total_tokens").map(String::as_str),
            Some("8192")
        );
        assert_eq!(labels.get("version").map(String::as_str), Some("0.1.0"));
        assert_eq!(
            labels.get("cache_trace_epochs").map(String::as_str),
            Some(r#"["replica-one","replica-two"]"#)
        );
        assert!(!labels.contains_key("host"));
        // The pairing-protocol facts survive, under the engine's own names;
        // the discovery step canonicalises the parallelism spellings.
        assert_eq!(
            labels.get("disaggregation_mode").map(String::as_str),
            Some("prefill")
        );
        assert_eq!(
            labels
                .get("disaggregation_transfer_backend")
                .map(String::as_str),
            Some("mooncake")
        );
        assert_eq!(
            labels
                .get("disaggregation_bootstrap_port")
                .map(String::as_str),
            Some("8998")
        );
        assert_eq!(
            labels.get("kv_cache_dtype").map(String::as_str),
            Some("auto")
        );
        assert_eq!(
            labels.get("attention_backend").map(String::as_str),
            Some("flashinfer")
        );
        assert_eq!(
            labels.get("pipeline_parallel_size").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            labels.get("rl.control_url").map(String::as_str),
            Some("http://10.0.0.5:40100")
        );
        assert_eq!(
            labels.get("rl.pause_modes").map(String::as_str),
            Some("wait,abort,keep")
        );
        assert_eq!(
            labels.get("rl.reports_weight_version").map(String::as_str),
            Some("true")
        );
        // The engine's per-request media limits, for the media pipeline.
        assert_eq!(
            labels.get("mm_item_limits").map(String::as_str),
            Some("image=4")
        );
        // scheduler_info and transient runtime state never become labels.
        assert!(!labels.contains_key("status"));
        assert!(!labels.contains_key("active_requests"));
        assert!(!labels.contains_key("uptime_seconds"));
    }

    /// The TokenSpeed servicer's grammar-start facts ride with the server args
    /// and become the `engine_reasoning_parser` and `constrained_decoding_start`
    /// labels, as for SGLang; the engine's own `reasoning_parser` argument is
    /// not the router's parser override and stays out.
    #[test]
    fn server_info_to_labels_tokenspeed_carries_where_the_engine_starts_a_grammar() {
        let info = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("reasoning_parser".to_string(), string_value("kimi_k25")),
                    (
                        "engine_reasoning_parser".to_string(),
                        string_value("kimi_k25"),
                    ),
                    (
                        "constrained_decoding_start".to_string(),
                        string_value("first_token"),
                    ),
                ]),
            }),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("engine_reasoning_parser").map(String::as_str),
            Some("kimi_k25")
        );
        assert_eq!(
            labels.get("constrained_decoding_start").map(String::as_str),
            Some("first_token")
        );
        assert!(!labels.contains_key("reasoning_parser"));

        // A servicer that writes neither key: no label, the router keeps the
        // grammar shape that fits either engine.
        let silent = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([(
                    "reasoning_parser".to_string(),
                    string_value("kimi_k25"),
                )]),
            }),
            ..Default::default()
        }))
        .to_labels();
        assert!(!silent.contains_key("engine_reasoning_parser"));
        assert!(!silent.contains_key("constrained_decoding_start"));
        assert!(!silent.contains_key("reasoning_parser"));
    }

    #[test]
    fn server_info_to_labels_sglang_picks_curated_keys_and_version() {
        let info = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("model_path".to_string(), string_value("Qwen/Qwen3-8B")),
                    ("dp_size".to_string(), number_value(4.0)),
                    (
                        "is_embedding".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::BoolValue(false)),
                        },
                    ),
                    (
                        "disaggregation_transfer_backend".to_string(),
                        string_value("nixl"),
                    ),
                    (
                        "disaggregation_bootstrap_port".to_string(),
                        number_value(8998.0),
                    ),
                    ("kv_cache_dtype".to_string(), string_value("fp8_e5m2")),
                    ("page_size".to_string(), number_value(64.0)),
                    ("attention_backend".to_string(), string_value("fa3")),
                    ("pairing_protocol".to_string(), string_value("kv-v1")),
                    (
                        "mm_item_limits".to_string(),
                        string_value("image=1,video=1"),
                    ),
                    // Not in SGLANG_GRPC_KEYS — must not become a label.
                    ("api_key".to_string(), string_value("secret")),
                ]),
            }),
            sglang_version: "0.4.0".to_string(),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("model_path").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        assert_eq!(labels.get("dp_size").map(String::as_str), Some("4"));
        // Booleans are kept even when false (embedding detection relies on it).
        assert_eq!(
            labels.get("is_embedding").map(String::as_str),
            Some("false")
        );
        assert_eq!(labels.get("version").map(String::as_str), Some("0.4.0"));
        assert!(!labels.contains_key("api_key"));
        // The pairing-protocol facts survive: transport, bootstrap port, KV layout.
        assert_eq!(
            labels
                .get("disaggregation_transfer_backend")
                .map(String::as_str),
            Some("nixl")
        );
        assert_eq!(
            labels
                .get("disaggregation_bootstrap_port")
                .map(String::as_str),
            Some("8998")
        );
        assert_eq!(
            labels.get("kv_cache_dtype").map(String::as_str),
            Some("fp8_e5m2")
        );
        assert_eq!(labels.get("page_size").map(String::as_str), Some("64"));
        assert_eq!(
            labels.get("attention_backend").map(String::as_str),
            Some("fa3")
        );
        // The explicit protocol the servicer read from the engine's environment.
        assert_eq!(
            labels.get("pairing_protocol").map(String::as_str),
            Some("kv-v1")
        );
        // The engine's per-request media limits, for the media pipeline.
        assert_eq!(
            labels.get("mm_item_limits").map(String::as_str),
            Some("image=1,video=1")
        );
    }

    /// The servicer's grammar-start facts ride with the server args and become
    /// the `engine_reasoning_parser` and `constrained_decoding_start` labels. The
    /// engine's own `reasoning_parser` argument, which the server args carry
    /// as well, is not the router's parser override and stays out.
    #[test]
    fn server_info_to_labels_sglang_carries_where_the_engine_starts_a_grammar() {
        let info = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("reasoning_parser".to_string(), string_value("qwen3")),
                    ("engine_reasoning_parser".to_string(), string_value("qwen3")),
                    (
                        "constrained_decoding_start".to_string(),
                        string_value("after_reasoning"),
                    ),
                ]),
            }),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("engine_reasoning_parser").map(String::as_str),
            Some("qwen3")
        );
        assert_eq!(
            labels.get("constrained_decoding_start").map(String::as_str),
            Some("after_reasoning")
        );
        assert!(
            !labels.contains_key("reasoning_parser"),
            "the engine's parser argument is not the router's parser override"
        );

        // A servicer that writes neither key (one from before the facts): no
        // label, so the router keeps the grammar shape that fits either engine.
        let silent = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([("reasoning_parser".to_string(), string_value("qwen3"))]),
            }),
            ..Default::default()
        }))
        .to_labels();
        assert!(!silent.contains_key("engine_reasoning_parser"));
        assert!(!silent.contains_key("constrained_decoding_start"));
        assert!(!silent.contains_key("reasoning_parser"));
    }

    /// `GetModelInfoResponse` is flat for every backend, so it serializes via
    /// `flat_labels`: empty strings and zero numbers are skipped, booleans are
    /// kept (except a non-vLLM `supports_vision: false`, which is no verdict),
    /// arrays are JSON-encoded.
    #[test]
    fn model_info_to_labels_tokenspeed_flat_serializes_skipping_empty_and_zero() {
        let info = ModelInfo::TokenSpeed(Box::new(tokenspeed_proto::GetModelInfoResponse {
            model_path: "Qwen/Qwen3-8B".to_string(),
            tokenizer_path: String::new(), // empty — skipped
            served_model_name: "qwen3-8b".to_string(),
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            max_context_length: 32768,
            vocab_size: 151_936,
            pad_token_id: 0, // zero — skipped
            supports_vision: false,
            supports_multimodal: false,
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("model_path").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        assert_eq!(
            labels.get("served_model_name").map(String::as_str),
            Some("qwen3-8b")
        );
        assert_eq!(
            labels.get("max_context_length").map(String::as_str),
            Some("32768")
        );
        assert_eq!(labels.get("vocab_size").map(String::as_str), Some("151936"));
        assert_eq!(
            labels.get("architectures").map(String::as_str),
            Some(r#"["Qwen3ForCausalLM"]"#)
        );
        // A proto-default `false` is no vision verdict on this runtime...
        assert!(!labels.contains_key("supports_vision"));
        // ...while other booleans flatten as before, `false` included.
        assert_eq!(
            labels.get("supports_multimodal").map(String::as_str),
            Some("false")
        );
        assert!(!labels.contains_key("tokenizer_path"));
        assert!(!labels.contains_key("pad_token_id"));
    }
}
