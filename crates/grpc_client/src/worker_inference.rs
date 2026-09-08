//! Engine-neutral Router-to-Worker inference client.
//!
//! The wire contract is deliberately independent of the engine behind the
//! Worker. Conversion helpers bridge the first text-generation implementation
//! to the router's existing TokenSpeed-shaped internal request/response model;
//! that shape does not escape onto the WorkerInference wire.

use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use futures::{Stream, StreamExt};
use proto::worker_inference_server::WorkerInference as _;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::{transport::Channel, Request, Response, Status};
use tracing::{debug, warn};

// `worker_inference.proto` shares the `smg.worker.v1` package with
// `worker_control.proto` and lands in the same generated file. Re-export the
// single include from [`crate::worker_control`] so `worker_proto::GenerateRequest`
// and `worker_inference_proto::GenerateRequest` stay one type.
pub use crate::worker_control::proto;
use crate::{
    sglang_runtime as sglang, tokenspeed_proto as ts, vllm_proto as vllm, AbortOnDropClient,
    BoxedTraceInjector, NoopTraceInjector, TokenSpeedSchedulerClient, VllmEngineClient,
};

pub type AbortOnDropStream =
    crate::AbortOnDropStream<proto::GenerateResponse, WorkerInferenceClient>;

/// Client for the stable Worker SMG data plane.
#[derive(Clone)]
pub struct WorkerInferenceClient {
    client: proto::worker_inference_client::WorkerInferenceClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl AbortOnDropClient for WorkerInferenceClient {
    fn abort_for_drop(
        self,
        request_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), Status>> + Send>> {
        Box::pin(async move {
            self.abort_request(request_id, "Stream dropped".to_string())
                .await
        })
    }
}

impl WorkerInferenceClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!(endpoint, "Connecting to WorkerInference");
        let channel = crate::channel::connect_channel(endpoint).await?;
        Ok(Self {
            client: proto::worker_inference_client::WorkerInferenceClient::new(channel),
            trace_injector,
        })
    }

    pub async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<AbortOnDropStream, Status> {
        let request_id = request.request_id.clone();
        let mut request = Request::new(request);
        if let Err(error) = self.trace_injector.inject(request.metadata_mut()) {
            warn!(%error, "Failed to inject WorkerInference trace context");
        }
        let response = self.client.clone().generate(request).await?;
        Ok(AbortOnDropStream::new(
            response.into_inner(),
            request_id,
            self.clone(),
        ))
    }

    pub async fn abort_request(&self, request_id: String, reason: String) -> Result<(), Status> {
        let mut request = Request::new(proto::AbortRequest { request_id, reason });
        if let Err(error) = self.trace_injector.inject(request.metadata_mut()) {
            warn!(%error, "Failed to inject WorkerInference trace context");
        }
        let response = self.client.clone().abort(request).await?.into_inner();
        if response.success {
            Ok(())
        } else {
            Err(Status::failed_precondition(response.message))
        }
    }
}

/// Worker-side adapter for SGLang's native Rust gRPC service.
///
/// The Router only sees [`proto::WorkerInference`]. This adapter owns the
/// engine-specific translation and can later be embedded directly in the
/// engine coordinator through the Python binding.
#[derive(Clone)]
pub struct SglangWorkerInference {
    client: sglang::sglang_service_client::SglangServiceClient<Channel>,
}

impl SglangWorkerInference {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let channel = crate::channel::connect_channel(endpoint).await?;
        Ok(Self {
            client: sglang::sglang_service_client::SglangServiceClient::new(channel),
        })
    }
}

pub type EngineTransportStream =
    Pin<Box<dyn Stream<Item = Result<proto::GenerateResponse, Status>> + Send>>;

/// Worker-local transport to an inference engine.
///
/// The Router-facing service remains the stable [`proto::WorkerInference`]
/// gRPC contract. Implementations of this trait may use engine-native gRPC,
/// same-host ZMQ IPC, or an in-process channel without changing that wire.
#[tonic::async_trait]
pub trait EngineTransport: Send + Sync {
    async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<EngineTransportStream, Status>;

    async fn abort(&self, request: proto::AbortRequest) -> Result<proto::AbortResponse, Status>;
}

#[tonic::async_trait]
impl proto::worker_inference_server::WorkerInference for SglangWorkerInference {
    type GenerateStream = EngineTransportStream;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let request_id = request.request_id.clone();
        let request = into_sglang_request(request)?;
        let stream = self
            .client
            .clone()
            .generate(Request::new(request))
            .await?
            .into_inner()
            .scan(HashMap::new(), move |emitted_by_index, item| {
                let result = item.and_then(|response| {
                    from_sglang_response(&request_id, response, emitted_by_index)
                });
                futures::future::ready(Some(result))
            });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        let request = request.into_inner();
        let response = self
            .client
            .clone()
            .abort(Request::new(sglang::AbortRequest {
                rid: request.request_id,
                abort_all: false,
            }))
            .await?
            .into_inner();
        Ok(Response::new(proto::AbortResponse {
            success: response.success,
            message: if response.success {
                String::new()
            } else {
                "SGLang rejected the abort request".to_string()
            },
        }))
    }
}

/// Worker-side adapter for the SMG vLLM scheduler gRPC service.
#[derive(Clone)]
pub struct VllmWorkerInference {
    client: VllmEngineClient,
}

impl VllmWorkerInference {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            client: VllmEngineClient::connect(endpoint).await?,
        })
    }
}

#[tonic::async_trait]
impl proto::worker_inference_server::WorkerInference for VllmWorkerInference {
    type GenerateStream = EngineTransportStream;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let request_id = request.request_id.clone();
        let stream = self.client.generate(into_vllm_request(request)?).await?;
        let stream = futures::stream::unfold(stream, move |mut stream| {
            let request_id = request_id.clone();
            async move {
                let item = stream.next().await?;
                let mapped = item.map(|response| from_vllm_response(&request_id, response));
                if matches!(
                    &mapped,
                    Ok(proto::GenerateResponse {
                        response: Some(proto::generate_response::Response::Complete(_)),
                        ..
                    })
                ) {
                    stream.mark_completed();
                }
                Some((mapped, stream))
            }
        });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        let request = request.into_inner();
        self.client
            .abort_request(request.request_id, request.reason)
            .await?;
        Ok(Response::new(proto::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }
}

/// Worker-side adapter for the TokenSpeed scheduler gRPC service.
#[derive(Clone)]
pub struct TokenSpeedWorkerInference {
    client: TokenSpeedSchedulerClient,
}

impl TokenSpeedWorkerInference {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            client: TokenSpeedSchedulerClient::connect(endpoint).await?,
        })
    }
}

#[tonic::async_trait]
impl proto::worker_inference_server::WorkerInference for TokenSpeedWorkerInference {
    type GenerateStream = EngineTransportStream;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let stream = self
            .client
            .generate(into_tokenspeed_request(request.into_inner()))
            .await?;
        // TokenSpeed's gRPC chunks already carry the delta shape that
        // `GenerateStreamChunk` specifies: `tokenspeed_scheduler.proto`
        // documents `token_ids` as "generated tokens since the previous
        // chunk", and the servicer slices logprobs down to the same frame.
        // Pass them through unchanged -- re-deriving deltas here would treat
        // every chunk after the first as an already-emitted prefix and drain
        // it to nothing. Only `Complete` is cumulative, as on every lane.
        let stream = futures::stream::unfold(stream, |mut stream| async move {
            let item = stream.next().await?;
            let mapped = item.map(from_tokenspeed_response);
            if matches!(
                &mapped,
                Ok(proto::GenerateResponse {
                    response: Some(proto::generate_response::Response::Complete(_)),
                    ..
                })
            ) {
                stream.mark_completed();
            }
            Some((mapped, stream))
        });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        let request = request.into_inner();
        self.client
            .abort_request(request.request_id, request.reason)
            .await?;
        Ok(Response::new(proto::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }
}

/// Engine-specific transport hidden behind the stable Worker service.
#[derive(Clone)]
enum EngineAdapter {
    Sglang(SglangWorkerInference),
    Vllm(VllmWorkerInference),
    TokenSpeed(TokenSpeedWorkerInference),
}

/// Connect the engine-native gRPC adapter for `engine_type` and return it as a
/// bare [`EngineTransport`], so a caller can wrap it in its own admission and
/// lifecycle gates (or install it into a lazily-bound one).
pub async fn connect_engine_transport(
    engine_type: &str,
    endpoint: &str,
) -> Result<Arc<dyn EngineTransport>, Box<dyn std::error::Error + Send + Sync>> {
    let adapter = match engine_type.to_ascii_lowercase().as_str() {
        "sglang" => EngineAdapter::Sglang(SglangWorkerInference::connect(endpoint).await?),
        "vllm" => EngineAdapter::Vllm(VllmWorkerInference::connect(endpoint).await?),
        "tokenspeed" | "ts" => {
            EngineAdapter::TokenSpeed(TokenSpeedWorkerInference::connect(endpoint).await?)
        }
        other => {
            return Err(format!("WorkerInference adapter is not implemented for {other}").into())
        }
    };
    // Opening a channel proves only that something listens on the port. The
    // Worker owns engine readiness, and callers announce SERVING as soon as
    // this returns, so refuse to come up in front of an engine that is
    // absent, still loading, or speaks a different service.
    adapter.verify_engine_ready().await?;
    Ok(Arc::new(adapter))
}

impl EngineAdapter {
    /// One engine health probe over the adapter's own client. SGLang's
    /// runtime service has no health RPC, so that adapter can only be trusted
    /// by whoever launched the engine.
    async fn verify_engine_ready(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (engine, healthy, message) = match self {
            Self::Vllm(adapter) => {
                let health = adapter.client.health_check().await?;
                ("vLLM", health.healthy, health.message)
            }
            Self::TokenSpeed(adapter) => {
                let health = adapter.client.health_check().await?;
                ("TokenSpeed", health.healthy, health.message)
            }
            Self::Sglang(_) => return Ok(()),
        };
        if healthy {
            Ok(())
        } else {
            Err(format!("{engine} engine reports unhealthy: {message}").into())
        }
    }
}

#[tonic::async_trait]
impl EngineTransport for EngineAdapter {
    async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<EngineTransportStream, Status> {
        let request = Request::new(request);
        let response = match self {
            Self::Sglang(service) => service.generate(request).await,
            Self::Vllm(service) => service.generate(request).await,
            Self::TokenSpeed(service) => service.generate(request).await,
        }?;
        Ok(response.into_inner())
    }

    async fn abort(&self, request: proto::AbortRequest) -> Result<proto::AbortResponse, Status> {
        let request = Request::new(request);
        let response = match self {
            Self::Sglang(service) => service.abort(request).await,
            Self::Vllm(service) => service.abort(request).await,
            Self::TokenSpeed(service) => service.abort(request).await,
        }?;
        Ok(response.into_inner())
    }
}

/// One admission-controlled tonic service type for the Python binding
/// regardless of engine.
#[derive(Clone)]
pub struct EngineWorkerInference {
    transport: Arc<dyn EngineTransport>,
    permits: Option<Arc<Semaphore>>,
    serving: Option<Arc<AtomicBool>>,
}

impl EngineWorkerInference {
    pub async fn connect(
        engine_type: &str,
        endpoint: &str,
        max_concurrent_requests: u32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let transport = connect_engine_transport(engine_type, endpoint).await?;
        Ok(Self::from_transport(transport, max_concurrent_requests))
    }

    /// Wrap an engine-native transport with the common Worker admission and
    /// lifecycle gates.
    #[must_use]
    pub fn from_transport(
        transport: Arc<dyn EngineTransport>,
        max_concurrent_requests: u32,
    ) -> Self {
        let permits = (max_concurrent_requests > 0)
            .then(|| Arc::new(Semaphore::new(max_concurrent_requests as usize)));
        Self {
            transport,
            permits,
            serving: None,
        }
    }

    #[must_use]
    pub fn with_serving_flag(mut self, serving: Arc<AtomicBool>) -> Self {
        self.serving = Some(serving);
        self
    }
}

fn try_acquire_worker_permit(
    permits: Option<&Arc<Semaphore>>,
) -> Result<Option<OwnedSemaphorePermit>, Status> {
    permits
        .map(|permits| Arc::clone(permits).try_acquire_owned())
        .transpose()
        .map_err(|_| Status::resource_exhausted("Worker request limit reached"))
}

/// `WorkerInference` v1 has no input-logprobs lane: neither
/// `GenerateStreamChunk` nor `GenerateComplete` carries them. A request that
/// asks for prompt logprobs (`return_logprob` with a non-negative
/// `logprob_start_len`; `-1` means "output logprobs only") would make the
/// engine compute them and then have the adapter drop them, so the Router
/// saw a normal stream with the requested data silently missing. Refuse it at
/// the boundary that can name the gap.
fn reject_input_logprobs(request: &proto::GenerateRequest) -> Result<(), Status> {
    if request.return_logprob && request.logprob_start_len.is_some_and(|start| start >= 0) {
        return Err(Status::unimplemented(
            "WorkerInference v1 does not carry input (prompt) logprobs",
        ));
    }
    Ok(())
}

fn ensure_worker_serving(serving: Option<&Arc<AtomicBool>>) -> Result<(), Status> {
    if serving.is_some_and(|serving| !serving.load(Ordering::Acquire)) {
        return Err(Status::unavailable("Worker is not serving"));
    }
    Ok(())
}

#[tonic::async_trait]
impl proto::worker_inference_server::WorkerInference for EngineWorkerInference {
    type GenerateStream = EngineTransportStream;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        ensure_worker_serving(self.serving.as_ref())?;
        let request = request.into_inner();
        reject_input_logprobs(&request)?;
        let permit = try_acquire_worker_permit(self.permits.as_ref())?;
        let stream = self.transport.generate(request).await?;
        let stream = stream.map(move |item| {
            let _permit = &permit;
            item
        });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        self.transport
            .abort(request.into_inner())
            .await
            .map(Response::new)
    }
}

pub fn into_vllm_request(request: proto::GenerateRequest) -> Result<vllm::GenerateRequest, Status> {
    reject_input_logprobs(&request)?;
    let tokenized = request.tokenized.ok_or_else(missing_tokenized_input)?;
    let sampling_params = request
        .sampling_params
        .map(|params| {
            into_vllm_sampling(
                params,
                request.return_logprob,
                request.top_logprobs_num,
                &request.token_ids_logprob,
            )
        })
        .transpose()?;
    Ok(vllm::GenerateRequest {
        request_id: request.request_id,
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: tokenized.original_text,
                input_ids: tokenized.input_ids,
            },
        )),
        sampling_params,
        stream: request.stream,
        kv_transfer_params: None,
        mm_inputs: None,
        kv_transfer_params_json: None,
        data_parallel_rank: request.data_parallel_rank,
    })
}

fn into_vllm_sampling(
    params: proto::SamplingParams,
    return_logprob: bool,
    top_logprobs_num: i32,
    token_ids_logprob: &[u32],
) -> Result<vllm::SamplingParams, Status> {
    if params.engine_parameters.is_some() {
        return Err(Status::invalid_argument(
            "vLLM adapter does not accept untyped engine parameters",
        ));
    }
    if !token_ids_logprob.is_empty() {
        return Err(Status::unimplemented(
            "vLLM adapter does not support token_ids_logprob",
        ));
    }
    let logprobs = return_logprob.then_some(top_logprobs_num.max(0));
    // Never ask vLLM for prompt logprobs: the wire cannot carry them back
    // (`reject_input_logprobs` refuses the requests that want them), and
    // computing them makes V1 skip prefix caching for the prefill.
    let prompt_logprobs = None;
    let logit_bias = params
        .logit_bias
        .into_iter()
        .map(|(key, value)| {
            key.parse::<i32>()
                .map(|key| (key, value))
                .map_err(|_| Status::invalid_argument("vLLM logit_bias keys must be token IDs"))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    // vLLM's proto seed is i32 while the request seed is u64. Saturate rather
    // than reject so a large seed keeps working here exactly as it does on the
    // direct vLLM path (`vllm_engine.rs`), preserving the "set" vs "unset"
    // distinction.
    let seed = params
        .sampling_seed
        .map(|value| i32::try_from(value).unwrap_or(i32::MAX));
    let constraint = params.constraint.map(|constraint| match constraint {
        proto::sampling_params::Constraint::Regex(value) => {
            vllm::sampling_params::Constraint::Regex(value)
        }
        proto::sampling_params::Constraint::JsonSchema(value) => {
            vllm::sampling_params::Constraint::JsonSchema(value)
        }
        proto::sampling_params::Constraint::EbnfGrammar(value) => {
            vllm::sampling_params::Constraint::Grammar(value)
        }
        proto::sampling_params::Constraint::StructuralTag(value) => {
            vllm::sampling_params::Constraint::StructuralTag(value)
        }
    });

    Ok(vllm::SamplingParams {
        temperature: params.temperature,
        top_p: params.top_p.unwrap_or(1.0),
        top_k: params.top_k.unwrap_or_default().max(0) as u32,
        min_p: params.min_p.unwrap_or_default(),
        frequency_penalty: params.frequency_penalty.unwrap_or_default(),
        presence_penalty: params.presence_penalty.unwrap_or_default(),
        repetition_penalty: params.repetition_penalty.unwrap_or(1.0),
        max_tokens: params.max_new_tokens,
        min_tokens: params.min_new_tokens,
        stop: params.stop,
        stop_token_ids: params.stop_token_ids,
        // vLLM's own defaults for the omitted case, matching the direct path.
        skip_special_tokens: params.skip_special_tokens.unwrap_or(true),
        spaces_between_special_tokens: params.spaces_between_special_tokens.unwrap_or(true),
        ignore_eos: params.ignore_eos,
        n: params.n.max(1),
        logprobs,
        prompt_logprobs,
        seed,
        include_stop_str_in_output: params.no_stop_trim.unwrap_or(false),
        logit_bias,
        truncate_prompt_tokens: None,
        eos_token_id: params.eos_token_id,
        constraint,
    })
}

fn missing_tokenized_input() -> Status {
    // `unwrap_or_default()` here would turn an omitted field into a legitimate
    // zero-token prompt: the engine either errors opaquely or generates
    // unconditioned output, and the Router sees a normal stream either way.
    // This is a server-side boundary reachable by any gRPC peer, so name the
    // failure where it can still be named.
    Status::invalid_argument("WorkerInference GenerateRequest requires tokenized input")
}

pub fn from_vllm_response(
    request_id: &str,
    response: vllm::GenerateResponse,
) -> proto::GenerateResponse {
    use vllm::generate_response::Response;

    let response = response.response.map(|response| match response {
        Response::Chunk(chunk) => {
            proto::generate_response::Response::Chunk(proto::GenerateStreamChunk {
                token_ids: chunk.token_ids,
                prompt_tokens: chunk.prompt_tokens,
                completion_tokens: chunk.completion_tokens,
                cached_tokens: chunk.cached_tokens,
                output_logprobs: chunk.output_logprobs.map(from_vllm_logprobs),
                index: chunk.index,
            })
        }
        Response::Complete(complete) => {
            proto::generate_response::Response::Complete(proto::GenerateComplete {
                output_ids: complete.output_ids,
                finish_reason: complete.finish_reason,
                prompt_tokens: complete.prompt_tokens,
                completion_tokens: complete.completion_tokens,
                cached_tokens: complete.cached_tokens,
                output_logprobs: complete.output_logprobs.map(from_vllm_logprobs),
                matched_stop: complete.matched_stop.map(|matched| match matched {
                    vllm::generate_complete::MatchedStop::MatchedTokenId(id) => {
                        proto::generate_complete::MatchedStop::MatchedTokenId(id)
                    }
                    vllm::generate_complete::MatchedStop::MatchedStopStr(value) => {
                        proto::generate_complete::MatchedStop::MatchedStopStr(value)
                    }
                }),
                index: complete.index,
            })
        }
    });
    proto::GenerateResponse {
        request_id: request_id.to_string(),
        response,
    }
}

/// Inverse of [`from_vllm_response`], for the Router side of the Worker wire.
///
/// The vLLM response *shape* is what the Router's accumulation is keyed on
/// (delta chunks, cumulative `Complete`), and that is exactly the
/// `WorkerInference` contract -- so this is the shape an SMG stream maps onto,
/// whichever engine the Worker actually fronts.
pub fn into_vllm_response(response: proto::GenerateResponse) -> vllm::GenerateResponse {
    use proto::generate_response::Response;

    vllm::GenerateResponse {
        response: response.response.map(|response| match response {
            Response::Chunk(chunk) => {
                vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                    token_ids: chunk.token_ids,
                    prompt_tokens: chunk.prompt_tokens,
                    completion_tokens: chunk.completion_tokens,
                    cached_tokens: chunk.cached_tokens,
                    output_logprobs: chunk.output_logprobs.map(into_vllm_logprobs),
                    input_logprobs: None,
                    index: chunk.index,
                })
            }
            Response::Complete(complete) => {
                vllm::generate_response::Response::Complete(vllm::GenerateComplete {
                    output_ids: complete.output_ids,
                    finish_reason: complete.finish_reason,
                    prompt_tokens: complete.prompt_tokens,
                    completion_tokens: complete.completion_tokens,
                    cached_tokens: complete.cached_tokens,
                    output_logprobs: complete.output_logprobs.map(into_vllm_logprobs),
                    input_logprobs: None,
                    kv_transfer_params: None,
                    kv_transfer_params_json: None,
                    matched_stop: complete.matched_stop.map(|matched| match matched {
                        proto::generate_complete::MatchedStop::MatchedTokenId(id) => {
                            vllm::generate_complete::MatchedStop::MatchedTokenId(id)
                        }
                        proto::generate_complete::MatchedStop::MatchedStopStr(value) => {
                            vllm::generate_complete::MatchedStop::MatchedStopStr(value)
                        }
                    }),
                    index: complete.index,
                })
            }
        }),
    }
}

fn into_vllm_logprobs(logprobs: proto::OutputLogProbs) -> vllm::OutputLogProbs {
    vllm::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top_logprobs
            .into_iter()
            .map(|top| vllm::TopLogProbs {
                values: top.values,
                token_ids: top.token_ids,
            })
            .collect(),
    }
}

fn from_vllm_logprobs(logprobs: vllm::OutputLogProbs) -> proto::OutputLogProbs {
    proto::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top_logprobs
            .into_iter()
            .map(|item| proto::TopLogProbs {
                values: item.values,
                token_ids: item.token_ids,
            })
            .collect(),
    }
}

fn into_sglang_request(request: proto::GenerateRequest) -> Result<sglang::GenerateRequest, Status> {
    if request.return_logprob
        || request.top_logprobs_num != 0
        || !request.token_ids_logprob.is_empty()
    {
        return Err(Status::unimplemented(
            "SGLang native gRPC does not expose token logprobs on GenerateResponse",
        ));
    }
    let input_ids = request
        .tokenized
        .ok_or_else(missing_tokenized_input)?
        .input_ids
        .into_iter()
        .map(|id| i32::try_from(id).map_err(|_| numeric_range_error("input token id")))
        .collect::<Result<Vec<_>, _>>()?;
    let sampling_params = request
        .sampling_params
        .map(into_sglang_sampling)
        .transpose()?;
    Ok(sglang::GenerateRequest {
        input_ids,
        sampling_params,
        stream: Some(request.stream),
        return_logprob: Some(false),
        top_logprobs_num: None,
        logprob_start_len: request.logprob_start_len,
        rid: Some(request.request_id),
        routed_dp_rank: request.data_parallel_rank,
        priority: None,
        require_reasoning: None,
        max_thinking_tokens: None,
    })
}

fn into_sglang_sampling(params: proto::SamplingParams) -> Result<sglang::SamplingParams, Status> {
    if !params.logit_bias.is_empty() || params.engine_parameters.is_some() {
        return Err(Status::invalid_argument(
            "SGLang native gRPC does not support WorkerInference engine parameters or logit bias",
        ));
    }
    // `sglang_runtime.proto` has no field for these, and SGLang's own defaults
    // are `skip_special_tokens=true`, `spaces_between_special_tokens=true`,
    // `no_stop_trim=false`. Silently dropping a non-default value would flip
    // detokenization behind the caller's back -- the Harmony path relies on
    // `skip_special_tokens=false`. Fail closed, matching the checks above.
    //
    // Only an *explicit* override is a failure: the fields are presence-tracked
    // precisely so an omission can fall through to SGLang's defaults, which are
    // what the adapter would have produced anyway.
    if params.skip_special_tokens == Some(false)
        || params.spaces_between_special_tokens == Some(false)
        || params.no_stop_trim == Some(true)
    {
        return Err(Status::unimplemented(
            "SGLang native gRPC cannot express skip_special_tokens, \
             spaces_between_special_tokens, or no_stop_trim overrides",
        ));
    }
    if params.eos_token_id.is_some() {
        return Err(Status::unimplemented(
            "SGLang native gRPC has no eos_token_id override; SGLang resolves EOS itself",
        ));
    }
    Ok(sglang::SamplingParams {
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        min_p: params.min_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        repetition_penalty: params.repetition_penalty,
        max_new_tokens: params
            .max_new_tokens
            .map(|value| i32::try_from(value).map_err(|_| numeric_range_error("max_new_tokens")))
            .transpose()?,
        min_new_tokens: Some(
            i32::try_from(params.min_new_tokens)
                .map_err(|_| numeric_range_error("min_new_tokens"))?,
        ),
        stop: params.stop,
        stop_token_ids: params
            .stop_token_ids
            .into_iter()
            .map(|id| i32::try_from(id).map_err(|_| numeric_range_error("stop token id")))
            .collect::<Result<Vec<_>, _>>()?,
        ignore_eos: Some(params.ignore_eos),
        n: (params.n != 0)
            .then(|| i32::try_from(params.n).map_err(|_| numeric_range_error("n")))
            .transpose()?,
        seed: params
            .sampling_seed
            .map(|value| i64::try_from(value).map_err(|_| numeric_range_error("sampling_seed")))
            .transpose()?,
        guided_decoding: params.constraint.map(|constraint| sglang::GuidedDecoding {
            constraint: Some(match constraint {
                proto::sampling_params::Constraint::Regex(value) => {
                    sglang::guided_decoding::Constraint::Regex(value)
                }
                proto::sampling_params::Constraint::JsonSchema(value) => {
                    sglang::guided_decoding::Constraint::JsonSchema(value)
                }
                proto::sampling_params::Constraint::EbnfGrammar(value) => {
                    sglang::guided_decoding::Constraint::Ebnf(value)
                }
                proto::sampling_params::Constraint::StructuralTag(value) => {
                    sglang::guided_decoding::Constraint::StructuralTag(value)
                }
            }),
        }),
    })
}

fn numeric_range_error(field: &str) -> Status {
    Status::invalid_argument(format!("{field} is outside SGLang's signed 32-bit range"))
}

fn from_sglang_response(
    request_id: &str,
    response: sglang::GenerateResponse,
    emitted_by_index: &mut HashMap<u32, usize>,
) -> Result<proto::GenerateResponse, Status> {
    let mut output_ids = response
        .output_ids
        .into_iter()
        .map(|id| {
            u32::try_from(id).map_err(|_| Status::internal("SGLang returned a negative token id"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prompt_tokens = meta_u32(&response.meta_info, "prompt_tokens");
    let completion_tokens = meta_u32(&response.meta_info, "completion_tokens");
    let cached_tokens = meta_u32(&response.meta_info, "cached_tokens");
    let index = meta_u32(&response.meta_info, "index");

    if !response.finished {
        let emitted = emitted_by_index.entry(index).or_default();
        if output_ids.len() < *emitted {
            return Err(Status::internal(
                "SGLang returned a shorter cumulative token sequence",
            ));
        }
        output_ids.drain(..*emitted);
        *emitted += output_ids.len();
    }

    let response = if response.finished {
        let (finish_reason, matched_stop) = finish_metadata(&response.meta_info);
        proto::generate_response::Response::Complete(proto::GenerateComplete {
            output_ids,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            output_logprobs: None,
            matched_stop,
            index,
        })
    } else {
        proto::generate_response::Response::Chunk(proto::GenerateStreamChunk {
            token_ids: output_ids,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            output_logprobs: None,
            index,
        })
    };

    Ok(proto::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(response),
    })
}

fn meta_u32(meta: &HashMap<String, String>, key: &str) -> u32 {
    meta.get(key)
        .and_then(|value| serde_json::from_str::<u32>(value).ok())
        .unwrap_or_default()
}

fn finish_metadata(
    meta: &HashMap<String, String>,
) -> (String, Option<proto::generate_complete::MatchedStop>) {
    let Some(value) = meta
        .get("finish_reason")
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
    else {
        return ("stop".to_string(), None);
    };

    match value {
        serde_json::Value::String(reason) => (reason, None),
        serde_json::Value::Object(object) => {
            let reason = object
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("stop")
                .to_string();
            let matched = object.get("matched").and_then(|value| {
                if let Some(id) = value.as_u64().and_then(|id| u32::try_from(id).ok()) {
                    Some(proto::generate_complete::MatchedStop::MatchedTokenId(id))
                } else {
                    value.as_str().map(|value| {
                        proto::generate_complete::MatchedStop::MatchedStopStr(value.to_string())
                    })
                }
            });
            (reason, matched)
        }
        _ => ("stop".to_string(), None),
    }
}

/// Convert the router's mature text-generation representation to the stable
/// Worker wire. Unsupported extension lanes fail explicitly rather than being
/// silently discarded.
pub fn from_tokenspeed_request(
    request: ts::GenerateRequest,
) -> Result<proto::GenerateRequest, Status> {
    if request.mm_inputs.is_some() {
        return Err(Status::unimplemented(
            "WorkerInference v1 does not support multimodal inputs",
        ));
    }
    if request.encode_bootstrap_info.is_some() || request.kv_bootstrap_info.is_some() {
        return Err(Status::unimplemented(
            "WorkerInference v1 does not support disaggregated execution",
        ));
    }

    Ok(proto::GenerateRequest {
        request_id: request.request_id,
        tokenized: request.tokenized.map(|input| proto::TokenizedInput {
            input_ids: input.input_ids,
            original_text: input.original_text,
        }),
        sampling_params: request.sampling_params.map(from_tokenspeed_sampling),
        return_logprob: request.return_logprob,
        logprob_start_len: request.logprob_start_len,
        top_logprobs_num: request.top_logprobs_num,
        token_ids_logprob: request.token_ids_logprob,
        stream: request.stream,
        data_parallel_rank: request.data_parallel_rank,
    })
}

/// Adapter-side conversion used by the mock worker and, later, each engine
/// binding. It is kept in the protocol crate so every engine receives exactly
/// the same argument semantics.
pub fn into_tokenspeed_request(request: proto::GenerateRequest) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: request.request_id,
        tokenized: request.tokenized.map(|input| ts::TokenizedInput {
            input_ids: input.input_ids,
            original_text: input.original_text,
        }),
        sampling_params: request.sampling_params.map(into_tokenspeed_sampling),
        return_logprob: request.return_logprob,
        logprob_start_len: request.logprob_start_len,
        top_logprobs_num: request.top_logprobs_num,
        token_ids_logprob: request.token_ids_logprob,
        stream: request.stream,
        data_parallel_rank: request.data_parallel_rank,
        ..Default::default()
    }
}

pub fn from_tokenspeed_response(response: ts::GenerateResponse) -> proto::GenerateResponse {
    use ts::generate_response::Response;
    proto::GenerateResponse {
        request_id: response.request_id,
        response: response.response.map(|response| match response {
            Response::Chunk(chunk) => {
                proto::generate_response::Response::Chunk(proto::GenerateStreamChunk {
                    token_ids: chunk.token_ids,
                    prompt_tokens: chunk.prompt_tokens,
                    completion_tokens: chunk.completion_tokens,
                    cached_tokens: chunk.cached_tokens,
                    output_logprobs: chunk.output_logprobs.map(from_tokenspeed_logprobs),
                    index: chunk.index,
                })
            }
            Response::Complete(complete) => {
                proto::generate_response::Response::Complete(proto::GenerateComplete {
                    output_ids: complete.output_ids,
                    finish_reason: complete.finish_reason,
                    prompt_tokens: complete.prompt_tokens,
                    completion_tokens: complete.completion_tokens,
                    cached_tokens: complete.cached_tokens,
                    output_logprobs: complete.output_logprobs.map(from_tokenspeed_logprobs),
                    matched_stop: complete.matched_stop.map(|matched| match matched {
                        ts::generate_complete::MatchedStop::MatchedTokenId(id) => {
                            proto::generate_complete::MatchedStop::MatchedTokenId(id)
                        }
                        ts::generate_complete::MatchedStop::MatchedStopStr(value) => {
                            proto::generate_complete::MatchedStop::MatchedStopStr(value)
                        }
                    }),
                    index: complete.index,
                })
            }
        }),
    }
}

pub fn into_tokenspeed_response(response: proto::GenerateResponse) -> ts::GenerateResponse {
    use proto::generate_response::Response;
    ts::GenerateResponse {
        request_id: response.request_id,
        response: response.response.map(|response| match response {
            Response::Chunk(chunk) => {
                ts::generate_response::Response::Chunk(ts::GenerateStreamChunk {
                    token_ids: chunk.token_ids,
                    prompt_tokens: chunk.prompt_tokens,
                    completion_tokens: chunk.completion_tokens,
                    cached_tokens: chunk.cached_tokens,
                    output_logprobs: chunk.output_logprobs.map(into_tokenspeed_logprobs),
                    index: chunk.index,
                })
            }
            Response::Complete(complete) => {
                ts::generate_response::Response::Complete(ts::GenerateComplete {
                    output_ids: complete.output_ids,
                    finish_reason: complete.finish_reason,
                    prompt_tokens: complete.prompt_tokens,
                    completion_tokens: complete.completion_tokens,
                    cached_tokens: complete.cached_tokens,
                    output_logprobs: complete.output_logprobs.map(into_tokenspeed_logprobs),
                    matched_stop: complete.matched_stop.map(|matched| match matched {
                        proto::generate_complete::MatchedStop::MatchedTokenId(id) => {
                            ts::generate_complete::MatchedStop::MatchedTokenId(id)
                        }
                        proto::generate_complete::MatchedStop::MatchedStopStr(value) => {
                            ts::generate_complete::MatchedStop::MatchedStopStr(value)
                        }
                    }),
                    index: complete.index,
                })
            }
        }),
    }
}

fn from_tokenspeed_sampling(params: ts::SamplingParams) -> proto::SamplingParams {
    proto::SamplingParams {
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        min_p: params.min_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        repetition_penalty: params.repetition_penalty,
        max_new_tokens: params.max_new_tokens,
        min_new_tokens: params.min_new_tokens,
        stop: params.stop,
        stop_token_ids: params.stop_token_ids,
        ignore_eos: params.ignore_eos,
        // The TokenSpeed request these are read from has plain bools, so the
        // Router always states them explicitly; presence only carries meaning
        // for a peer that builds a WorkerInference request directly.
        skip_special_tokens: Some(params.skip_special_tokens),
        spaces_between_special_tokens: Some(params.spaces_between_special_tokens),
        n: params.n,
        logit_bias: params.logit_bias,
        constraint: params.constraint.map(|constraint| match constraint {
            ts::sampling_params::Constraint::Regex(value) => {
                proto::sampling_params::Constraint::Regex(value)
            }
            ts::sampling_params::Constraint::JsonSchema(value) => {
                proto::sampling_params::Constraint::JsonSchema(value)
            }
            ts::sampling_params::Constraint::EbnfGrammar(value) => {
                proto::sampling_params::Constraint::EbnfGrammar(value)
            }
            ts::sampling_params::Constraint::StructuralTag(value) => {
                proto::sampling_params::Constraint::StructuralTag(value)
            }
        }),
        engine_parameters: params.custom_params,
        no_stop_trim: Some(params.no_stop_trim),
        sampling_seed: params.sampling_seed,
        // Resolved per request by the Router's tokenizer before this
        // conversion (`fold_smg_vllm_eos_backstop`).
        eos_token_id: params.eos_token_id,
    }
}

fn into_tokenspeed_sampling(params: proto::SamplingParams) -> ts::SamplingParams {
    ts::SamplingParams {
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        min_p: params.min_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        repetition_penalty: params.repetition_penalty,
        max_new_tokens: params.max_new_tokens,
        min_new_tokens: params.min_new_tokens,
        stop: params.stop,
        stop_token_ids: params.stop_token_ids,
        ignore_eos: params.ignore_eos,
        skip_special_tokens: params.skip_special_tokens.unwrap_or(true),
        spaces_between_special_tokens: params.spaces_between_special_tokens.unwrap_or(true),
        n: params.n,
        logit_bias: params.logit_bias,
        constraint: params.constraint.map(|constraint| match constraint {
            proto::sampling_params::Constraint::Regex(value) => {
                ts::sampling_params::Constraint::Regex(value)
            }
            proto::sampling_params::Constraint::JsonSchema(value) => {
                ts::sampling_params::Constraint::JsonSchema(value)
            }
            proto::sampling_params::Constraint::EbnfGrammar(value) => {
                ts::sampling_params::Constraint::EbnfGrammar(value)
            }
            proto::sampling_params::Constraint::StructuralTag(value) => {
                ts::sampling_params::Constraint::StructuralTag(value)
            }
        }),
        custom_params: params.engine_parameters,
        no_stop_trim: params.no_stop_trim.unwrap_or(false),
        sampling_seed: params.sampling_seed,
        eos_token_id: params.eos_token_id,
    }
}

fn from_tokenspeed_logprobs(logprobs: ts::OutputLogProbs) -> proto::OutputLogProbs {
    proto::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top_logprobs
            .into_iter()
            .map(|item| proto::TopLogProbs {
                values: item.values,
                token_ids: item.token_ids,
            })
            .collect(),
    }
}

fn into_tokenspeed_logprobs(logprobs: proto::OutputLogProbs) -> ts::OutputLogProbs {
    ts::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top_logprobs
            .into_iter()
            .map(|item| ts::TopLogProbs {
                values: item.values,
                token_ids: item.token_ids,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };

    use futures::{stream, Stream, StreamExt};
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{transport::Server, Response, Status};

    use super::*;

    #[derive(Clone, Default)]
    struct TestInference {
        aborts: Arc<AtomicUsize>,
    }

    #[tonic::async_trait]
    impl proto::worker_inference_server::WorkerInference for TestInference {
        type GenerateStream =
            Pin<Box<dyn Stream<Item = Result<proto::GenerateResponse, Status>> + Send>>;

        async fn generate(
            &self,
            request: Request<proto::GenerateRequest>,
        ) -> Result<Response<Self::GenerateStream>, Status> {
            let request_id = request.into_inner().request_id;
            let first = proto::GenerateResponse {
                request_id,
                response: Some(proto::generate_response::Response::Chunk(
                    proto::GenerateStreamChunk {
                        token_ids: vec![42],
                        completion_tokens: 1,
                        ..Default::default()
                    },
                )),
            };
            Ok(Response::new(Box::pin(
                stream::once(async move { Ok(first) }).chain(stream::pending()),
            )))
        }

        async fn abort(
            &self,
            _request: Request<proto::AbortRequest>,
        ) -> Result<Response<proto::AbortResponse>, Status> {
            self.aborts.fetch_add(1, Ordering::SeqCst);
            Ok(Response::new(proto::AbortResponse {
                success: true,
                message: String::new(),
            }))
        }
    }

    #[test]
    fn text_request_round_trips_without_engine_fields() {
        let request = ts::GenerateRequest {
            request_id: "req-1".to_string(),
            tokenized: Some(ts::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: "hello".to_string(),
            }),
            sampling_params: Some(ts::SamplingParams {
                temperature: Some(0.25),
                max_new_tokens: Some(8),
                stop: vec!["done".to_string()],
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };

        let worker = from_tokenspeed_request(request.clone()).expect("portable request");
        assert_eq!(into_tokenspeed_request(worker), request);
    }

    #[test]
    fn disaggregated_request_is_rejected() {
        let request = ts::GenerateRequest {
            kv_bootstrap_info: Some(ts::KvBootstrapInfo::default()),
            ..Default::default()
        };
        let status = from_tokenspeed_request(request).expect_err("unsupported extension");
        assert_eq!(status.code(), tonic::Code::Unimplemented);
    }

    #[test]
    fn sglang_native_request_preserves_portable_arguments() {
        let request = proto::GenerateRequest {
            request_id: "native-1".to_string(),
            tokenized: Some(proto::TokenizedInput {
                input_ids: vec![10, 20],
                original_text: "ignored by tokenized native RPC".to_string(),
            }),
            sampling_params: Some(proto::SamplingParams {
                temperature: Some(0.2),
                max_new_tokens: Some(16),
                stop_token_ids: vec![99],
                sampling_seed: Some(7),
                // The Router always sets these explicitly; SGLang's native
                // proto has no field for them, so only its own defaults pass.
                skip_special_tokens: Some(true),
                spaces_between_special_tokens: Some(true),
                ..Default::default()
            }),
            stream: true,
            data_parallel_rank: Some(2),
            ..Default::default()
        };

        let native = into_sglang_request(request).expect("native request");
        assert_eq!(native.input_ids, vec![10, 20]);
        assert_eq!(native.rid.as_deref(), Some("native-1"));
        assert_eq!(native.routed_dp_rank, Some(2));
        let sampling = native.sampling_params.expect("sampling params");
        assert_eq!(sampling.temperature, Some(0.2));
        assert_eq!(sampling.max_new_tokens, Some(16));
        assert_eq!(sampling.stop_token_ids, vec![99]);
        assert_eq!(sampling.seed, Some(7));
    }

    #[test]
    fn sglang_native_rejects_inexpressible_detokenization_flags() {
        // `sglang_runtime.proto` cannot carry these, and SGLang's defaults are
        // the opposite of what is being asked for. Dropping them silently would
        // flip detokenization behind the caller's back (the Harmony path relies
        // on `skip_special_tokens: false`), so the adapter must fail closed.
        for params in [
            proto::SamplingParams {
                skip_special_tokens: Some(false),
                ..Default::default()
            },
            proto::SamplingParams {
                spaces_between_special_tokens: Some(false),
                ..Default::default()
            },
            proto::SamplingParams {
                no_stop_trim: Some(true),
                ..Default::default()
            },
            proto::SamplingParams {
                eos_token_id: Some(128009),
                ..Default::default()
            },
        ] {
            let request = proto::GenerateRequest {
                request_id: "native-flags".to_string(),
                tokenized: Some(proto::TokenizedInput {
                    input_ids: vec![1],
                    ..Default::default()
                }),
                sampling_params: Some(params),
                ..Default::default()
            };
            let status = into_sglang_request(request).expect_err("unsupported flag");
            assert_eq!(status.code(), tonic::Code::Unimplemented);
        }
    }

    #[test]
    fn sglang_native_lets_omitted_detokenization_flags_fall_through() {
        // Proto3 decodes an omitted scalar bool as `false`, so gating on the
        // value alone would reject every request that simply did not mention
        // these fields -- and SGLang's own defaults are exactly what the
        // adapter would have produced.
        let request = proto::GenerateRequest {
            request_id: "native-defaults".to_string(),
            tokenized: Some(proto::TokenizedInput {
                input_ids: vec![10, 20],
                ..Default::default()
            }),
            sampling_params: Some(proto::SamplingParams {
                temperature: Some(0.3),
                ..Default::default()
            }),
            ..Default::default()
        };

        let native = into_sglang_request(request).expect("defaults are expressible");
        assert_eq!(
            native.sampling_params.expect("sampling params").temperature,
            Some(0.3)
        );
    }

    #[test]
    fn vllm_request_carries_the_router_supplied_eos() {
        let sampling = into_vllm_request(proto::GenerateRequest {
            request_id: "vllm-eos".to_string(),
            tokenized: Some(proto::TokenizedInput {
                input_ids: vec![1],
                ..Default::default()
            }),
            sampling_params: Some(proto::SamplingParams {
                eos_token_id: Some(128009),
                ..Default::default()
            }),
            ..Default::default()
        })
        .expect("vLLM request")
        .sampling_params
        .expect("sampling params");
        assert_eq!(sampling.eos_token_id, Some(128009));
    }

    #[test]
    fn worker_responses_round_trip_through_the_vllm_shape() {
        // The Router maps an SMG stream onto the vLLM response shape, which is
        // what `ChunkSemantics::Delta` describes -- so this conversion has to be
        // lossless in both directions for the fields the Router reads.
        for response in [
            proto::GenerateResponse {
                request_id: "rt-chunk".to_string(),
                response: Some(proto::generate_response::Response::Chunk(
                    proto::GenerateStreamChunk {
                        token_ids: vec![7, 8],
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        cached_tokens: 1,
                        output_logprobs: Some(proto::OutputLogProbs {
                            token_logprobs: vec![-0.5],
                            token_ids: vec![7],
                            top_logprobs: vec![proto::TopLogProbs {
                                values: vec![-0.5, -1.5],
                                token_ids: vec![7, 8],
                            }],
                        }),
                        index: 1,
                    },
                )),
            },
            proto::GenerateResponse {
                request_id: "rt-complete".to_string(),
                response: Some(proto::generate_response::Response::Complete(
                    proto::GenerateComplete {
                        output_ids: vec![7, 8],
                        finish_reason: "stop".to_string(),
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        cached_tokens: 1,
                        output_logprobs: None,
                        matched_stop: Some(proto::generate_complete::MatchedStop::MatchedStopStr(
                            "END".to_string(),
                        )),
                        index: 1,
                    },
                )),
            },
        ] {
            let request_id = response.request_id.clone();
            let vllm_shaped = into_vllm_response(response.clone());
            assert_eq!(from_vllm_response(&request_id, vllm_shaped), response);
        }
    }

    #[test]
    fn requests_without_tokenized_input_are_rejected() {
        // An empty `TokenizedInput` default would reach the engine as a
        // legitimate zero-token prompt, so the boundary that can name the
        // problem has to reject it.
        for build in [
            (|request| into_vllm_request(request).map(|_| ())) as fn(_) -> Result<(), Status>,
            |request| into_sglang_request(request).map(|_| ()),
        ] {
            let status = build(proto::GenerateRequest {
                request_id: "no-input".to_string(),
                sampling_params: Some(proto::SamplingParams::default()),
                ..Default::default()
            })
            .expect_err("tokenized input is required");
            assert_eq!(status.code(), tonic::Code::InvalidArgument);
        }
    }

    #[test]
    fn vllm_request_preserves_portable_arguments() {
        let request = proto::GenerateRequest {
            request_id: "vllm-1".to_string(),
            tokenized: Some(proto::TokenizedInput {
                input_ids: vec![10, 20],
                original_text: "hello".to_string(),
            }),
            sampling_params: Some(proto::SamplingParams {
                temperature: Some(0.2),
                max_new_tokens: Some(16),
                stop_token_ids: vec![99],
                sampling_seed: Some(7),
                n: 2,
                ..Default::default()
            }),
            return_logprob: true,
            top_logprobs_num: 3,
            stream: true,
            data_parallel_rank: Some(2),
            ..Default::default()
        };

        let native = into_vllm_request(request).expect("vLLM request");
        assert_eq!(native.request_id, "vllm-1");
        assert_eq!(native.data_parallel_rank, Some(2));
        let Some(vllm::generate_request::Input::Tokenized(input)) = native.input else {
            panic!("expected tokenized input")
        };
        assert_eq!(input.input_ids, vec![10, 20]);
        let sampling = native.sampling_params.expect("sampling params");
        assert_eq!(sampling.temperature, Some(0.2));
        assert_eq!(sampling.max_tokens, Some(16));
        assert_eq!(sampling.stop_token_ids, vec![99]);
        assert_eq!(sampling.seed, Some(7));
        assert_eq!(sampling.logprobs, Some(3));
        assert_eq!(sampling.n, 2);
    }

    fn vllm_logprob_request(
        return_logprob: bool,
        logprob_start_len: Option<i32>,
    ) -> proto::GenerateRequest {
        proto::GenerateRequest {
            request_id: "vllm-logprobs".to_string(),
            tokenized: Some(proto::TokenizedInput {
                input_ids: vec![1],
                ..Default::default()
            }),
            sampling_params: Some(proto::SamplingParams::default()),
            return_logprob,
            logprob_start_len,
            top_logprobs_num: 3,
            ..Default::default()
        }
    }

    #[test]
    fn input_logprob_requests_are_refused_not_dropped() {
        // The Router populates `logprob_start_len` on every request (-1 means
        // "output logprobs only"). Only a non-negative start with
        // `return_logprob` asks for prompt logprobs, which no v1 response
        // frame can carry -- so that combination is an error at the boundary,
        // and everything else passes through without asking vLLM for them.
        for (return_logprob, start) in [
            (true, Some(-1)),
            (false, Some(-1)),
            (false, Some(0)),
            (true, None),
        ] {
            let request = vllm_logprob_request(return_logprob, start);
            assert!(
                reject_input_logprobs(&request).is_ok(),
                "{return_logprob} {start:?}"
            );
            let sampling = into_vllm_request(request)
                .expect("vLLM request")
                .sampling_params
                .expect("sampling params");
            assert_eq!(sampling.prompt_logprobs, None);
            assert_eq!(sampling.logprobs, return_logprob.then_some(3));
        }
        for start in [Some(0), Some(4)] {
            let request = vllm_logprob_request(true, start);
            assert_eq!(
                reject_input_logprobs(&request).expect_err("gate").code(),
                tonic::Code::Unimplemented
            );
            assert_eq!(
                into_vllm_request(request).expect_err("vLLM lane").code(),
                tonic::Code::Unimplemented
            );
        }
    }

    #[test]
    fn vllm_saturates_seeds_beyond_the_proto_range() {
        // vLLM's proto seed is i32 while the request seed is u64. The direct
        // vLLM path saturates; rejecting here would 400 a request that works
        // today, and unset must stay unset so vLLM picks its own seed.
        let request = |seed: Option<u64>| proto::GenerateRequest {
            request_id: "vllm-seed".to_string(),
            tokenized: Some(proto::TokenizedInput {
                input_ids: vec![1],
                ..Default::default()
            }),
            sampling_params: Some(proto::SamplingParams {
                sampling_seed: seed,
                ..Default::default()
            }),
            ..Default::default()
        };
        let seed_of = |seed| {
            into_vllm_request(request(seed))
                .expect("vLLM request")
                .sampling_params
                .expect("sampling params")
                .seed
        };
        assert_eq!(seed_of(Some(7)), Some(7));
        assert_eq!(seed_of(Some(3_000_000_000)), Some(i32::MAX));
        assert_eq!(seed_of(Some(u64::MAX)), Some(i32::MAX));
        assert_eq!(seed_of(None), None);
    }

    #[test]
    fn vllm_response_maps_to_worker_contract() {
        let response = from_vllm_response(
            "vllm-2",
            vllm::GenerateResponse {
                response: Some(vllm::generate_response::Response::Complete(
                    vllm::GenerateComplete {
                        output_ids: vec![42, 43],
                        finish_reason: "stop".to_string(),
                        completion_tokens: 2,
                        matched_stop: Some(vllm::generate_complete::MatchedStop::MatchedTokenId(
                            43,
                        )),
                        ..Default::default()
                    },
                )),
            },
        );

        let Some(proto::generate_response::Response::Complete(complete)) = response.response else {
            panic!("expected completion")
        };
        assert_eq!(response.request_id, "vllm-2");
        assert_eq!(complete.output_ids, vec![42, 43]);
        assert_eq!(complete.completion_tokens, 2);
        assert_eq!(
            complete.matched_stop,
            Some(proto::generate_complete::MatchedStop::MatchedTokenId(43))
        );
    }

    #[test]
    fn worker_admission_rejects_overload_and_recovers_after_drop() {
        let permits = Arc::new(Semaphore::new(1));
        let first = try_acquire_worker_permit(Some(&permits)).expect("first permit");
        let overloaded = try_acquire_worker_permit(Some(&permits)).expect_err("limit reached");
        assert_eq!(overloaded.code(), tonic::Code::ResourceExhausted);
        drop(first);
        assert!(try_acquire_worker_permit(Some(&permits)).is_ok());
        assert!(try_acquire_worker_permit(None).is_ok());
    }

    #[test]
    fn worker_lifecycle_rejects_new_requests_while_draining() {
        let serving = Arc::new(AtomicBool::new(true));
        assert!(ensure_worker_serving(Some(&serving)).is_ok());
        serving.store(false, Ordering::Release);
        let draining = ensure_worker_serving(Some(&serving)).expect_err("draining rejects");
        assert_eq!(draining.code(), tonic::Code::Unavailable);
        assert!(ensure_worker_serving(None).is_ok());
    }

    #[test]
    fn sglang_native_finish_metadata_maps_to_worker_contract() {
        let response = from_sglang_response(
            "native-2",
            sglang::GenerateResponse {
                output_ids: vec![42, 43],
                meta_info: HashMap::from([
                    ("prompt_tokens".to_string(), "3".to_string()),
                    ("completion_tokens".to_string(), "2".to_string()),
                    (
                        "finish_reason".to_string(),
                        r#"{"type":"stop","matched":43}"#.to_string(),
                    ),
                ]),
                finished: true,
            },
            &mut HashMap::new(),
        )
        .expect("worker response");

        let Some(proto::generate_response::Response::Complete(complete)) = response.response else {
            panic!("expected completion")
        };
        assert_eq!(complete.output_ids, vec![42, 43]);
        assert_eq!(complete.prompt_tokens, 3);
        assert_eq!(complete.completion_tokens, 2);
        assert_eq!(complete.finish_reason, "stop");
        assert_eq!(
            complete.matched_stop,
            Some(proto::generate_complete::MatchedStop::MatchedTokenId(43))
        );
    }

    #[test]
    fn sglang_native_stream_is_converted_from_cumulative_to_delta_tokens() {
        let mut emitted_by_index = HashMap::new();
        let first = from_sglang_response(
            "native-stream",
            sglang::GenerateResponse {
                output_ids: vec![10],
                meta_info: HashMap::new(),
                finished: false,
            },
            &mut emitted_by_index,
        )
        .expect("first chunk");
        let second = from_sglang_response(
            "native-stream",
            sglang::GenerateResponse {
                output_ids: vec![10, 20],
                meta_info: HashMap::new(),
                finished: false,
            },
            &mut emitted_by_index,
        )
        .expect("second chunk");

        let Some(proto::generate_response::Response::Chunk(first)) = first.response else {
            panic!("expected first chunk")
        };
        let Some(proto::generate_response::Response::Chunk(second)) = second.response else {
            panic!("expected second chunk")
        };
        assert_eq!(first.token_ids, vec![10]);
        assert_eq!(second.token_ids, vec![20]);
    }

    #[tokio::test]
    async fn stream_drop_sends_abort_over_worker_service() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let address = listener.local_addr().expect("listener address");
        let service = TestInference::default();
        let aborts = Arc::clone(&service.aborts);
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only tonic server is explicitly aborted before return"
        )]
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(proto::worker_inference_server::WorkerInferenceServer::new(
                    service,
                ))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
        });

        let client = WorkerInferenceClient::connect(&format!("grpc://{address}"))
            .await
            .expect("connect WorkerInference");
        let mut response = client
            .generate(proto::GenerateRequest {
                request_id: "drop-me".to_string(),
                stream: true,
                ..Default::default()
            })
            .await
            .expect("generate");
        assert!(response.next().await.expect("first item").is_ok());
        drop(response);

        tokio::time::timeout(Duration::from_secs(2), async {
            while aborts.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("drop-triggered abort");
        server.abort();
    }
}
