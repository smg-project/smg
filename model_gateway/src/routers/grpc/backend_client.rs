//! Backend client: the polymorphism point over a worker's transport.
//!
//! A worker's backend is reached via gRPC (the [`GrpcClient`] multiplexer
//! over SGLang/vLLM/TRT-LLM/MLX/TokenSpeed), via a direct ZMQ connection to a
//! same-host engine — vLLM EngineCore or TokenSpeed ([`ZmqEngineClient`]) —
//! or via the engine-neutral SMG Worker data plane ([`SmgBackendClient`]).
//! `BackendClient` keeps those first-class siblings — `GrpcClient` stays pure
//! gRPC — while the execution pipeline (which works against
//! [`ProtoStream`]/[`ProtoGenerateRequest`]) is shared unchanged.

use std::sync::Arc;

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use smg_grpc_client::{
    common_proto, tokenizer_bundle::StreamBundle, tokenspeed_proto, vllm_proto,
    worker_inference::from_tokenspeed_request, SglangSchedulerClient, TokenSpeedSchedulerClient,
    VllmEngineClient, WorkerInferenceClient,
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
        zmq_client::{
            fold_eos_into_stop_token_ids, fold_tokenizer_eos_backstop, ZmqDialect, ZmqEngineClient,
        },
        MultimodalData,
    },
    worker::RuntimeType,
};

/// A backend connection: gRPC (any engine) or direct ZMQ (vLLM EngineCore or
/// TokenSpeed).
#[derive(Clone)]
pub enum BackendClient {
    Grpc(GrpcClient),
    /// Stable Worker SMG data plane. `runtime` describes the colocated engine
    /// for capability checks, but never changes the Router-to-Worker wire.
    Smg(Arc<SmgBackendClient>),
    Zmq(ZmqEngineClient),
}

#[derive(Clone)]
pub struct SmgBackendClient {
    inference: WorkerInferenceClient,
    runtime: RuntimeType,
    token_only_wire: bool,
}

impl SmgBackendClient {
    pub fn new(
        inference: WorkerInferenceClient,
        runtime: RuntimeType,
        token_only_wire: bool,
    ) -> Self {
        Self {
            inference,
            runtime,
            token_only_wire,
        }
    }

    /// vLLM EngineCore behind a token-only Worker has no tokenizer: the
    /// Router supplies both its EOS stop ids and its primary EOS id.
    fn router_owns_eos(&self) -> bool {
        self.token_only_wire && self.runtime == RuntimeType::Vllm
    }

    /// A client over a bound socket nobody answers: enough for build-time
    /// and stamping behavior, never for a dispatch.
    #[cfg(test)]
    pub(crate) async fn unanswered_for_tests(runtime: RuntimeType, token_only_wire: bool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let address = listener.local_addr().expect("local addr");
        let inference = WorkerInferenceClient::connect(&format!("grpc://{address}"))
            .await
            .expect("connect WorkerInference");
        Self::new(inference, runtime, token_only_wire)
    }
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
    /// True if this is a direct-ZMQ backend (the engine receives token ids only
    /// and cannot match string stops itself).
    pub fn is_zmq(&self) -> bool {
        matches!(self, Self::Zmq(_))
    }

    /// Runtime type backing this client.
    pub fn runtime_type(&self) -> RuntimeType {
        match self {
            Self::Grpc(client) => client.runtime_type(),
            Self::Smg(client) => client.runtime,
            Self::Zmq(client) => client.runtime(),
        }
    }

    /// True when the engine-facing wire accepts token ids but cannot match
    /// string stops. This includes direct ZMQ and an SMG Worker that advertises
    /// a colocated ZMQ engine transport.
    pub fn uses_token_only_wire(&self) -> bool {
        match self {
            Self::Smg(client) => client.token_only_wire,
            Self::Zmq(_) => true,
            Self::Grpc(_) => false,
        }
    }

    /// True when the wire refuses a request id it still holds from an earlier
    /// attempt (the SMG Worker's duplicate-id guard), so every attempt needs a
    /// fresh engine id even for a bare client `rid`.
    pub fn rejects_duplicate_request_ids(&self) -> bool {
        matches!(self, Self::Smg(_))
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
        tokenizer: Option<&Arc<dyn llm_tokenizer::traits::Tokenizer>>,
    ) -> Vec<String> {
        let token_only_wire = self.uses_token_only_wire();
        let router_stops = helpers::resolve_string_stops(request, tokenizer, token_only_wire);
        match self {
            // Per request, from this request's tokenizer: one Worker can
            // serve several models.
            Self::Smg(client) if client.router_owns_eos() => {
                fold_smg_vllm_eos_stops(request, tokenizer);
            }
            // EngineCore has no tokenizer: the stop set and the request's
            // `_eos_token_id` both come from this frontend. TokenSpeed stops
            // at EOS itself; the fold's variant match skips its requests.
            Self::Zmq(client) => {
                client.adopt_tokenizer_eos(tokenizer);
                fold_tokenizer_eos_backstop(request, tokenizer);
            }
            Self::Smg(_) | Self::Grpc(_) => {}
        }
        router_stops
    }

    /// Local liveness. gRPC has no cheap local flag (it uses a health RPC), so
    /// this reports `true` for gRPC; ZMQ reflects its connection liveness.
    pub fn is_alive(&self) -> bool {
        match self {
            Self::Grpc(_) => true,
            Self::Smg(_) => true,
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
            Self::Smg(_) => panic!("Worker SMG backend does not expose an engine-specific client"),
            Self::Zmq(_) => panic!("Expected SGLang client, got ZMQ backend"),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.health_check().await,
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "Worker SMG health is served by WorkerControl",
            )),
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
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "Worker SMG model metadata is served by WorkerControl",
            )),
            Self::Zmq(client) => Ok(client.get_model_info()),
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_server_info().await,
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "Worker SMG server metadata is served by WorkerControl",
            )),
            Self::Zmq(client) => Ok(client.get_server_info()),
        }
    }

    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_loads().await,
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "WorkerInference v1 does not expose scheduler loads",
            )),
            Self::Zmq(client) => Ok(client.get_loads()),
        }
    }

    pub async fn flush_cache(
        &self,
        timeout_s: f32,
    ) -> Result<common_proto::FlushCacheResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.flush_cache(timeout_s).await,
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "WorkerInference v1 does not expose cache administration",
            )),
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
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "WorkerInference v1 does not expose profiling",
            )),
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "StartProfile not supported over ZMQ",
            )),
        }
    }

    pub async fn stop_profile(&self) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.stop_profile().await,
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "WorkerInference v1 does not expose profiling",
            )),
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
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "WorkerInference v1 does not expose KV events",
            )),
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
            Self::Smg(_) => Err("WorkerInference v1 does not serve a tokenizer bundle".into()),
            // EngineCore does not serve tokenizer artifacts over ZMQ; the
            // tokenizer is configured at worker registration instead.
            Self::Zmq(_) => Err("ZMQ backend does not serve a tokenizer bundle".into()),
        }
    }

    /// `primary_eos` is the request tokenizer's primary EOS id; only an SMG
    /// Worker whose engine cannot resolve EOS itself reads it.
    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
        primary_eos: Option<u32>,
    ) -> Result<ProtoStream, tonic::Status> {
        match self {
            Self::Grpc(client) => client.generate(req).await,
            Self::Smg(client) => {
                let ProtoGenerateRequest::TokenSpeed(request) = req else {
                    return Err(tonic::Status::invalid_argument(
                        "Worker SMG requires the engine-neutral request representation",
                    ));
                };
                let primary_eos = primary_eos.filter(|_| client.router_owns_eos());
                let request = from_tokenspeed_request(*request, primary_eos)?;
                Ok(ProtoStream::Smg(client.inference.generate(request).await?))
            }
            Self::Zmq(client) => Ok(ProtoStream::Zmq(client.generate(req).await?)),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match self {
            Self::Grpc(client) => client.embed(req).await,
            Self::Smg(_) => Err(tonic::Status::unimplemented(
                "WorkerInference v1 does not support embedding",
            )),
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
            Self::Smg(_) => {
                if options.multimodal_inputs.is_some() {
                    return Err("WorkerInference v1 does not support multimodal inputs".to_string());
                }
                let request = TokenSpeedSchedulerClient::build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    None,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::TokenSpeed(Box::new(request)))
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
            Self::Smg(_) => {
                if options.multimodal_inputs.is_some() {
                    return Err("WorkerInference v1 does not support multimodal inputs".to_string());
                }
                let request = TokenSpeedSchedulerClient::build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    None,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::TokenSpeed(Box::new(request)))
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
            Self::Smg(_) => Ok(ProtoGenerateRequest::TokenSpeed(Box::new(
                TokenSpeedSchedulerClient::build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?,
            ))),
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
            Self::Smg(_) => Ok(ProtoGenerateRequest::TokenSpeed(Box::new(
                TokenSpeedSchedulerClient::build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?,
            ))),
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

/// Fold the tokenizer's EOS ids into the portable request's `stop_token_ids`
/// for a Worker fronting tokenizer-less vLLM EngineCore; the Worker forwards
/// them unchanged. The primary EOS id travels separately, at dispatch.
fn fold_smg_vllm_eos_stops(
    request: &mut ProtoGenerateRequest,
    tokenizer: Option<&Arc<dyn llm_tokenizer::traits::Tokenizer>>,
) {
    let ProtoGenerateRequest::TokenSpeed(request) = request else {
        return;
    };
    let Some(params) = request.sampling_params.as_mut() else {
        return;
    };
    fold_eos_into_stop_token_ids(params.ignore_eos, &mut params.stop_token_ids, tokenizer);
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
            // The ZMQ wire carries one modality batch (see `zmq_vllm_mm`).
            finish_vllm_request(vllm_mm.map(|mm| (mm, Vec::new())), |mm| {
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
            MultimodalData::Vllm(data) => {
                let (primary, extra) = data.into_protos();
                if !extra.is_empty() {
                    return Err(
                        "the vLLM ZMQ backend takes one modality per request; mixed image and \
                         video requests need the gRPC backend"
                            .to_string(),
                    );
                }
                Ok(primary)
            }
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
            // No RDMA staging: the ZMQ engine reads tensors inline off the wire
            // and has no puller for `remote` payloads.
            MultimodalData::TokenSpeed(data) => Ok(data.into_proto(false)),
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

#[cfg(test)]
mod tests {
    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};

    use super::*;

    fn tokenspeed_request(stop: &[&str]) -> ProtoGenerateRequest {
        ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
            sampling_params: Some(tokenspeed_proto::SamplingParams {
                stop: stop.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn tokenspeed_params(request: &ProtoGenerateRequest) -> &tokenspeed_proto::SamplingParams {
        match request {
            ProtoGenerateRequest::TokenSpeed(request) => {
                request.sampling_params.as_ref().expect("sampling params")
            }
            _ => panic!("expected TokenSpeed request"),
        }
    }

    #[test]
    fn smg_vllm_eos_stops_fold_into_the_portable_request() {
        let mut request = tokenspeed_request(&[]);
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());

        fold_smg_vllm_eos_stops(&mut request, Some(&tokenizer));

        assert_eq!(
            tokenspeed_params(&request).stop_token_ids,
            tokenizer.eos_token_ids()
        );
    }

    /// Only a Worker fronting tokenizer-less vLLM takes the Router's EOS: its
    /// stop ids at finalize, and its primary id at dispatch.
    #[tokio::test]
    async fn only_a_token_only_vllm_worker_takes_the_router_eos() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        for (runtime, token_only_wire, owns) in [
            (RuntimeType::Vllm, true, true),
            (RuntimeType::Vllm, false, false),
            (RuntimeType::TokenSpeed, true, false),
        ] {
            let client = SmgBackendClient::unanswered_for_tests(runtime, token_only_wire).await;
            assert_eq!(
                client.router_owns_eos(),
                owns,
                "{runtime:?}/{token_only_wire}"
            );

            let backend = BackendClient::Smg(Arc::new(client));
            let mut request = tokenspeed_request(&["Hello world"]);
            let router_stops = backend.finalize_generate_request(&mut request, Some(&tokenizer));
            let params = tokenspeed_params(&request);
            if owns {
                assert_eq!(params.stop_token_ids, tokenizer.eos_token_ids());
            } else {
                assert!(params.stop_token_ids.is_empty());
            }
            // A token-only wire never sees the string stop; a gRPC-fronted
            // engine matches it itself.
            assert_eq!(params.stop.is_empty(), token_only_wire);
            assert_eq!(router_stops.is_empty(), !token_only_wire);
        }
    }

    /// The SMG Worker refuses a request id it still holds, so its lane cannot
    /// replay a bare client `rid` across attempts.
    #[tokio::test]
    async fn smg_wire_rejects_duplicate_request_ids() {
        let smg = BackendClient::Smg(Arc::new(
            SmgBackendClient::unanswered_for_tests(RuntimeType::Vllm, false).await,
        ));
        assert!(smg.rejects_duplicate_request_ids());
    }
}
