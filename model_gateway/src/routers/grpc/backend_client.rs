//! Backend client: the polymorphism point over a worker's transport.
//!
//! A worker's backend is reached either via gRPC (the [`GrpcClient`] multiplexer
//! over SGLang/vLLM/TRT-LLM/MLX/TokenSpeed) or via a direct ZMQ connection to a
//! same-host engine — vLLM EngineCore or TokenSpeed ([`ZmqEngineClient`]).
//! `BackendClient` keeps those
//! first-class siblings — `GrpcClient` stays pure gRPC — while the execution
//! pipeline (which works against [`ProtoStream`]/[`ProtoGenerateRequest`]) is
//! shared unchanged.

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use smg_grpc_client::{
    common_proto, tokenizer_bundle::StreamBundle, tokenspeed_proto, vllm_proto,
    SglangSchedulerClient, TokenSpeedSchedulerClient, VllmEngineClient,
};

use crate::{
    routers::grpc::{
        client::{
            GenerateRequestBuildOptions, GrpcClient, HealthCheckResponse, ModelInfo, ServerInfo,
        },
        common::stages::helpers,
        proto_wrapper::{
            finish_tokenspeed_request, finish_vllm_request, ProtoEmbedComplete, ProtoEmbedRequest,
            ProtoGenerateRequest, ProtoStream,
        },
        zmq_client::{fold_tokenizer_eos_backstop, ZmqDialect, ZmqEngineClient},
        MultimodalData,
    },
    worker::RuntimeType,
};

/// A backend connection: gRPC (any engine) or direct ZMQ (vLLM EngineCore or
/// TokenSpeed).
#[derive(Clone)]
pub enum BackendClient {
    Grpc(GrpcClient),
    Zmq(ZmqEngineClient),
}

/// The native request builders of both ZMQ dialects for one request kind — the
/// only thing that differs between the ZMQ build surfaces, so the dialect
/// dispatch itself lives once in [`build_zmq_request`]/[`build_zmq_plain_request`].
struct ZmqBuilders<V, T> {
    vllm: V,
    tokenspeed: T,
}

/// A vLLM builder for a request kind carrying multimodal inputs and tool
/// constraints (chat, messages).
type VllmMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    Option<vllm_proto::MultimodalInputs>,
    Option<(String, String)>,
) -> Result<vllm_proto::GenerateRequest, String>;

/// The TokenSpeed counterpart of [`VllmMmBuilder`].
type TokenSpeedMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    Option<tokenspeed_proto::MultimodalInputs>,
    Option<(String, String)>,
) -> Result<tokenspeed_proto::GenerateRequest, String>;

/// A vLLM builder for a request kind with neither multimodal inputs nor tool
/// constraints (completion, plain generate); `T` is the builder's text
/// parameter.
type VllmPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<vllm_proto::GenerateRequest, String>;

/// The TokenSpeed counterpart of [`VllmPlainBuilder`].
type TokenSpeedPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<tokenspeed_proto::GenerateRequest, String>;

impl BackendClient {
    /// Runtime type backing this client.
    pub fn runtime_type(&self) -> RuntimeType {
        match self {
            Self::Grpc(client) => client.runtime_type(),
            Self::Zmq(client) => client.runtime(),
        }
    }

    /// True if this is a direct-ZMQ backend (the engine receives token ids only
    /// and cannot match string stops itself).
    pub fn is_zmq(&self) -> bool {
        matches!(self, Self::Zmq(_))
    }

    /// Finalize a built generate request for this backend's wire: resolve
    /// string `stop`s the engine cannot match (token-only wires and SGLang's
    /// `skip_tokenizer_init` workers) into `stop_token_ids`, folding in EOS
    /// where the frontend owns stopping.
    ///
    /// Returns the router's residual obligation: the stop strings the engine
    /// will never see, which response processing must trim from output text.
    /// Empty when the engine matches stops server-side. This is the client's
    /// own policy — callers need no transport knowledge.
    pub fn finalize_generate_request(
        &self,
        request: &mut ProtoGenerateRequest,
        tokenizer: Option<&std::sync::Arc<dyn llm_tokenizer::traits::Tokenizer>>,
    ) -> Vec<String> {
        let token_only_wire = self.is_zmq();
        let router_stops = helpers::resolve_string_stops(request, tokenizer, token_only_wire);
        if token_only_wire {
            // EngineCore has no tokenizer, so stopping at EOS is this
            // frontend's job; TokenSpeed's scheduler stops at EOS itself, and
            // its requests are a different proto variant — the fold's own
            // variant match is the single dispatch point.
            fold_tokenizer_eos_backstop(request, tokenizer);
        }
        router_stops
    }

    /// Local liveness. gRPC has no cheap local flag (it uses a health RPC), so
    /// this reports `true` for gRPC; ZMQ reflects its connection liveness.
    pub fn is_alive(&self) -> bool {
        match self {
            Self::Grpc(_) => true,
            Self::Zmq(client) => client.is_alive(),
        }
    }

    /// Mutable SGLang client accessor. Only valid for a gRPC-SGLang backend;
    /// callers guard with a runtime/sglang check.
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees an SGLang gRPC backend"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Grpc(client) => client.as_sglang_mut(),
            Self::Zmq(_) => panic!("Expected SGLang client, got ZMQ backend"),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.health_check().await,
            Self::Zmq(client) => {
                let resp = client.health_check();
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_model_info().await,
            Self::Zmq(client) => Ok(client.get_model_info()),
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_server_info().await,
            Self::Zmq(client) => Ok(client.get_server_info()),
        }
    }

    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_loads().await,
            Self::Zmq(client) => Ok(client.get_loads()),
        }
    }

    pub async fn flush_cache(
        &self,
        timeout_s: f32,
    ) -> Result<common_proto::FlushCacheResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.flush_cache(timeout_s).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "FlushCache not supported over ZMQ",
            )),
        }
    }

    pub async fn start_profile(
        &self,
        req: common_proto::StartProfileRequest,
    ) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.start_profile(req).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "StartProfile not supported over ZMQ",
            )),
        }
    }

    pub async fn stop_profile(&self) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.stop_profile().await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "StopProfile not supported over ZMQ",
            )),
        }
    }

    pub async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Grpc(client) => client.subscribe_kv_events(start_seq).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents not supported over ZMQ",
            )),
        }
    }

    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Grpc(client) => client.get_tokenizer().await,
            // EngineCore does not serve tokenizer artifacts over ZMQ; the
            // tokenizer is configured at worker registration instead.
            Self::Zmq(_) => Err("ZMQ backend does not serve a tokenizer bundle".into()),
        }
    }

    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match self {
            Self::Grpc(client) => client.generate(req).await,
            Self::Zmq(client) => Ok(ProtoStream::Zmq(client.generate(req).await?)),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match self {
            Self::Grpc(client) => client.embed(req).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "ZMQ backend does not support embedding yet",
            )),
        }
    }

    pub fn build_chat_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_chat_request(request_id, body, processed_text, token_ids, options)
            }
            // A ZMQ backend speaks vLLM EngineCore or TokenSpeed directly; build
            // the native request for its dialect, mirroring the gRPC per-engine
            // dispatch in `GrpcClient::build_chat_request`.
            Self::Zmq(client) => build_zmq_request(
                client.dialect(),
                request_id,
                body,
                processed_text,
                token_ids,
                options,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_chat,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_chat,
                },
            ),
        }
    }

    pub fn build_messages_request(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_messages_request(request_id, body, processed_text, token_ids, options)
            }
            // Mirrors the gRPC per-engine dispatch: build the request natively for
            // the ZMQ backend's dialect (vLLM EngineCore or TokenSpeed).
            Self::Zmq(client) => build_zmq_request(
                client.dialect(),
                request_id,
                body,
                processed_text,
                token_ids,
                options,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_messages,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_messages,
                },
            ),
        }
    }

    pub fn build_completion_request(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_completion_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => build_zmq_plain_request(
                client.dialect(),
                request_id,
                body,
                original_text,
                token_ids,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_completion,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_completion,
                },
            ),
        }
    }

    pub fn build_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_generate_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => build_zmq_plain_request(
                client.dialect(),
                request_id,
                body,
                original_text,
                token_ids,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_plain_generate_request,
                    tokenspeed: TokenSpeedSchedulerClient::build_plain_generate_request,
                },
            ),
        }
    }
}

/// Build a multimodal-carrying request (chat, messages) for a ZMQ backend: one
/// dispatch over the closed [`ZmqDialect`], converting the assembled multimodal
/// data to the dialect's proto and finishing through its SHM-cleanup wrapper.
fn build_zmq_request<B>(
    dialect: ZmqDialect,
    request_id: String,
    body: &B,
    processed_text: String,
    token_ids: Vec<u32>,
    options: GenerateRequestBuildOptions,
    builders: ZmqBuilders<VllmMmBuilder<B>, TokenSpeedMmBuilder<B>>,
) -> Result<ProtoGenerateRequest, String> {
    let ZmqBuilders { vllm, tokenspeed } = builders;
    match dialect {
        ZmqDialect::Vllm => {
            let vllm_mm = zmq_vllm_mm(options.multimodal_inputs)?;
            finish_vllm_request(vllm_mm, |mm| {
                vllm(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    mm,
                    options.tool_constraints,
                )
            })
        }
        ZmqDialect::TokenSpeed => {
            let tokenspeed_mm = zmq_tokenspeed_mm(options.multimodal_inputs)?;
            finish_tokenspeed_request(tokenspeed_mm, |mm| {
                tokenspeed(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    mm,
                    options.tool_constraints,
                )
            })
        }
    }
}

/// Build a request kind with no multimodal or tool inputs (completion, plain
/// generate) for a ZMQ backend. `text` is the builders' shared text parameter
/// (`String` for completion, `Option<String>` for plain generate).
fn build_zmq_plain_request<B, T>(
    dialect: ZmqDialect,
    request_id: String,
    body: &B,
    text: T,
    token_ids: Vec<u32>,
    builders: ZmqBuilders<VllmPlainBuilder<B, T>, TokenSpeedPlainBuilder<B, T>>,
) -> Result<ProtoGenerateRequest, String> {
    let ZmqBuilders { vllm, tokenspeed } = builders;
    match dialect {
        ZmqDialect::Vllm => Ok(ProtoGenerateRequest::Vllm(Box::new(vllm(
            request_id, body, text, token_ids,
        )?))),
        ZmqDialect::TokenSpeed => Ok(ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed(
            request_id, body, text, token_ids,
        )?))),
    }
}

/// Convert assembled multimodal data for a vLLM ZMQ backend. A backend/variant
/// mismatch is a gateway bug (the assembly stage should produce the backend's
/// own variant), surfaced as a build error rather than a panic.
fn zmq_vllm_mm(
    inputs: Option<MultimodalData>,
) -> Result<Option<vllm_proto::MultimodalInputs>, String> {
    inputs
        .map(|mm| match mm {
            MultimodalData::Vllm(data) => Ok(data.into_proto()),
            other => Err(mm_variant_mismatch("vLLM", &other)),
        })
        .transpose()
}

/// Convert assembled multimodal data for a TokenSpeed ZMQ backend. See
/// [`zmq_vllm_mm`] for the mismatch semantics.
fn zmq_tokenspeed_mm(
    inputs: Option<MultimodalData>,
) -> Result<Option<tokenspeed_proto::MultimodalInputs>, String> {
    inputs
        .map(|mm| match mm {
            MultimodalData::TokenSpeed(data) => Ok(data.into_proto(true)),
            other => Err(mm_variant_mismatch("TokenSpeed", &other)),
        })
        .transpose()
}

/// Name the variant of a mismatched `MultimodalData` without dumping its tensor
/// payloads into the error string.
fn mm_variant_mismatch(expected: &str, got: &MultimodalData) -> String {
    let got = match got {
        MultimodalData::Sglang(_) => "SGLang",
        MultimodalData::Vllm(_) => "vLLM",
        MultimodalData::Trtllm(_) => "TRT-LLM",
        MultimodalData::TokenSpeed(_) => "TokenSpeed",
    };
    format!("multimodal data variant mismatch: {expected} ZMQ backend got {got} data")
}
