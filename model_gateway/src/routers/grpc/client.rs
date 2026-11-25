//! Unified gRPC client wrapper for SGLang, vLLM, and TensorRT-LLM backends

use std::sync::Arc;

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use smg_grpc_client::{
    common_proto,
    engine_labels::{ModelInfo, ServerInfo},
    mlx_proto, sglang_proto, tokenizer_bundle,
    tokenizer_bundle::StreamBundle,
    tokenspeed_proto, trtllm_proto, trtllm_service, vllm_proto, BoxedTraceInjector, FormatSetAside,
    MlxEngineClient, OutputConstraint, SglangGenerateRequestOptions, SglangSchedulerClient,
    TokenSpeedSchedulerClient, TrtllmServiceClient, VllmEngineClient,
};
use smg_worker::identity::basic::RuntimeType;

use crate::{
    observability::otel_trace::OtelTraceInjector,
    routers::grpc::{
        proto_wrapper::{
            cleanup_generate_request_shm, cleanup_mm_shm_handles,
            collect_tokenspeed_generate_request_shm_handles,
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
    /// Which of a chat body's own format fields `tool_constraints` answers (a
    /// table's tag for a JSON `response_format`; the whole turn's grammar for
    /// the `regex` and the `ebnf` too): the builders read them as absent, so
    /// the engine's request carries the constraint alone.
    pub format_set_aside: FormatSetAside,
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
        reason = "typed accessor: the caller guarantees the variant"
    )]
    pub fn as_tokenspeed(&self) -> &TokenSpeedSchedulerClient {
        match self {
            Self::TokenSpeed(client) => client,
            _ => panic!("Expected TokenSpeed client"),
        }
    }

    /// Runtime type backing this client. Lets shared logic (e.g. the multimodal
    /// capability matrix) key on the backend without matching every variant.
    pub fn runtime_type(&self) -> RuntimeType {
        match self {
            Self::Sglang(_) => RuntimeType::Sglang,
            Self::Vllm(_) => RuntimeType::Vllm,
            Self::Trtllm(_) => RuntimeType::Trtllm,
            Self::Mlx(_) => RuntimeType::Mlx,
            Self::TokenSpeed(_) => RuntimeType::TokenSpeed,
        }
    }

    /// Whether the client has a wire for `runtime`: [`Self::connect`] takes
    /// its name. Any other runtime is refused there ahead of a dial.
    pub fn has_wire(runtime: RuntimeType) -> bool {
        match runtime {
            RuntimeType::Sglang
            | RuntimeType::Vllm
            | RuntimeType::Trtllm
            | RuntimeType::Mlx
            | RuntimeType::TokenSpeed => true,
            RuntimeType::Unspecified | RuntimeType::Generic | RuntimeType::External => false,
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
                let healthy = trtllm_service::status_healthy(&resp.status);
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
        let include = || {
            smg_grpc_client::engine_load::LOAD_SECTIONS
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
            Self::Trtllm(client) => {
                let resp = client.get_loads(include()).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            Self::Mlx(_) => Err(tonic::Status::unimplemented(
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

    /// The servicer's load records alone, no events (`SubscribeKvEvents`
    /// with `loads_only`): what a worker streams when routing reads its load
    /// and keeps no KV index. A servicer that predates the field streams its
    /// events too; the subscriber drops them. `Unimplemented` on backends
    /// without the stream.
    pub async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Sglang(client) => client.subscribe_load_records().await,
            Self::Vllm(client) => client.subscribe_load_records().await,
            Self::Trtllm(client) => client.subscribe_load_records().await,
            Self::TokenSpeed(client) => client.subscribe_load_records().await,
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
    /// A request in another runtime's message type than the client's is
    /// answered `UNIMPLEMENTED` with both runtimes named, as an RPC absent
    /// on a backend's wire is.
    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match (self, req) {
            (Self::Sglang(client), ProtoGenerateRequest::Sglang(boxed_req)) => {
                let stream = client.generate_framed(*boxed_req).await?;
                Ok(ProtoStream::Sglang(stream))
            }
            (Self::Vllm(client), ProtoGenerateRequest::Vllm(boxed_req)) => {
                let shm_handles = collect_vllm_generate_request_shm_handles(&boxed_req);
                match client.generate_framed(*boxed_req).await {
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
                match client.generate_framed(*boxed_req).await {
                    Ok(stream) => Ok(ProtoStream::TokenSpeed(stream)),
                    Err(error) => {
                        cleanup_mm_shm_handles(&shm_handles);
                        Err(error)
                    }
                }
            }
            // A request in another runtime's message type than this client's.
            // The pair index keeps known runtimes apart on this wire and a
            // single placement builds its request with the client that sends
            // it, so nothing is known to reach this arm. A pair that did is
            // one this wire has no call for: the status is the caller's to
            // answer, where a panic ended the request's task with no answer.
            // No worker receives the request, so its shared-memory segments
            // are unlinked here, as the arms above unlink theirs on a send
            // error.
            (client, req) => {
                cleanup_generate_request_shm(&req);
                Err(tonic::Status::unimplemented(format!(
                    "a {} request cannot be sent to a {} worker",
                    req.runtime_type(),
                    client.runtime_type()
                )))
            }
        }
    }

    pub async fn embed(&self, req: ProtoEmbedRequest) -> Result<ProtoEmbedComplete, tonic::Status> {
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
            // A request in another runtime's message type than this client's,
            // or a backend with no embedding call. The embedding stage builds
            // each request with the client that sends it, so nothing is known
            // to reach this arm; one that did is answered as the generate
            // dispatch answers it.
            (client, req) => Err(tonic::Status::unimplemented(format!(
                "a {} embedding request cannot be sent to a {} worker",
                req.runtime_type(),
                client.runtime_type()
            ))),
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
            format_set_aside,
            require_reasoning,
            first_token_grammar,
        } = options;
        let constraint = OutputConstraint {
            tool: tool_constraints,
            set_aside: format_set_aside,
        };
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
                    MultimodalData::Sglang(data) => Some(data.into_protos()),
                    _ => None,
                });
                let (sglang_mm, extras) =
                    sglang_mm.map_or((None, Vec::new()), |(first, extras)| (Some(first), extras));
                let mut req = sglang(
                    client,
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    SglangGenerateRequestOptions {
                        multimodal_inputs: sglang_mm,
                        tool_call_constraint: constraint.tool,
                        format_set_aside: constraint.set_aside,
                        require_reasoning,
                    },
                )?;
                req.extra_mm_inputs = extras;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(_) => {
                let vllm_mm = engine_multimodal(multimodal_inputs, |mm| match mm {
                    MultimodalData::Vllm(data) => Some(data.into_protos()),
                    _ => None,
                });
                finish_vllm_request(vllm_mm, |mm| {
                    vllm(request_id, body, processed_text, token_ids, mm, constraint)
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
                    constraint,
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
                    constraint,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(_) => {
                let tokenspeed_mm = engine_multimodal(multimodal_inputs, |mm| match mm {
                    MultimodalData::TokenSpeed(data) => Some(data.into_proto(true)),
                    _ => None,
                });
                finish_tokenspeed_request(tokenspeed_mm, |mm| {
                    tokenspeed(request_id, body, processed_text, token_ids, mm, constraint)
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
    OutputConstraint,
) -> Result<vllm_proto::GenerateRequest, String>;

/// The TokenSpeed counterpart of [`VllmMmBuilder`].
pub(crate) type TokenSpeedMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    Option<tokenspeed_proto::MultimodalInputs>,
    OutputConstraint,
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
    OutputConstraint,
) -> Result<trtllm_proto::GenerateRequest, String>;

/// The MLX counterpart of [`VllmMmBuilder`] on the connected client; MLX
/// takes no multimodal inputs (the assembly stage refuses them earlier).
type MlxClientMmBuilder<B> = fn(
    &MlxEngineClient,
    String,
    &B,
    String,
    Vec<u32>,
    OutputConstraint,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The predicate and the connect name the same runtimes. A connect
    /// with a wire dials, and nothing listens here. A connect with none
    /// refuses the name ahead of the dial.
    #[tokio::test]
    async fn the_client_has_a_wire_for_each_runtime_its_connect_takes() {
        let all = [
            RuntimeType::Unspecified,
            RuntimeType::Sglang,
            RuntimeType::Vllm,
            RuntimeType::Trtllm,
            RuntimeType::Mlx,
            RuntimeType::TokenSpeed,
            RuntimeType::Generic,
            RuntimeType::External,
        ];
        for runtime in all {
            let refused = GrpcClient::connect("grpc://127.0.0.1:1", &runtime.to_string()).await;
            let refused = refused.err().map(|error| error.to_string());
            let refused = refused.expect("nothing listens");
            let takes_the_name = !refused.starts_with("Unknown runtime type");
            assert_eq!(
                GrpcClient::has_wire(runtime),
                takes_the_name,
                "{runtime}: {refused}"
            );
        }
    }
}
