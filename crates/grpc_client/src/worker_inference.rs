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
    task::{ready, Context, Poll},
};

use futures::{Stream, StreamExt};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::{transport::Channel, Request, Response, Status};
use tracing::{debug, warn};

// `worker_inference.proto` shares the `smg.worker.v1` package with
// `worker_control.proto` and lands in the same generated file. Re-export the
// single include from [`crate::worker_control`] so `worker_proto::GenerateRequest`
// and `worker_inference_proto::GenerateRequest` stay one type.
pub use crate::worker_control::proto;
use crate::{
    tokenspeed_proto as ts, vllm_proto as vllm, AbortOnDropClient, BoxedTraceInjector,
    NoopTraceInjector, TokenSpeedSchedulerClient, VllmEngineClient,
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

/// Engines emit one `Complete` per sampled index; `n == 0` leaves the engine
/// default of one.
fn sampled_indexes(request: &proto::GenerateRequest) -> u32 {
    request
        .sampling_params
        .as_ref()
        .map_or(1, |params| params.n.max(1))
}

/// Engine frames mapped onto the Worker wire. The engine stream's abort-on-drop
/// guard stays armed until every sampled index has reported `Complete`, so a
/// Router that disconnects between two indexes still aborts the rest.
struct GuardedEngineStream<S, F, M> {
    inner: S,
    map: F,
    release_guard: M,
    pending_completes: u32,
}

impl<S, T, F, M> GuardedEngineStream<S, F, M>
where
    S: Stream<Item = Result<T, Status>> + Unpin + Send + 'static,
    F: FnMut(T) -> proto::GenerateResponse + Unpin + Send + 'static,
    M: Fn(&S) + Unpin + Send + 'static,
{
    fn boxed(inner: S, pending_completes: u32, map: F, release_guard: M) -> EngineTransportStream {
        Box::pin(Self {
            inner,
            map,
            release_guard,
            pending_completes,
        })
    }
}

impl<S, T, F, M> Stream for GuardedEngineStream<S, F, M>
where
    S: Stream<Item = Result<T, Status>> + Unpin,
    F: FnMut(T) -> proto::GenerateResponse + Unpin,
    M: Fn(&S) + Unpin,
{
    type Item = Result<proto::GenerateResponse, Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let Some(item) = ready!(Pin::new(&mut this.inner).poll_next(cx)) else {
            return Poll::Ready(None);
        };
        let mapped = item.map(&mut this.map);
        let completed = matches!(
            &mapped,
            Ok(proto::GenerateResponse {
                response: Some(proto::generate_response::Response::Complete(_)),
                ..
            })
        );
        if completed && this.pending_completes > 0 {
            this.pending_completes -= 1;
            if this.pending_completes == 0 {
                (this.release_guard)(&this.inner);
            }
        }
        Poll::Ready(Some(mapped))
    }
}

fn ensure_engine_healthy(
    engine: &str,
    healthy: bool,
    message: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if healthy {
        Ok(())
    } else {
        Err(format!("{engine} engine reports unhealthy: {message}").into())
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

    async fn verify_engine_ready(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let health = self.client.health_check().await?;
        ensure_engine_healthy("vLLM", health.healthy, &health.message)
    }
}

#[tonic::async_trait]
impl EngineTransport for VllmWorkerInference {
    async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<EngineTransportStream, Status> {
        let request_id = request.request_id.clone();
        let sampled_indexes = sampled_indexes(&request);
        let stream = self.client.generate(into_vllm_request(request)?).await?;
        Ok(GuardedEngineStream::boxed(
            stream,
            sampled_indexes,
            move |response| from_vllm_response(&request_id, response),
            crate::vllm_engine::AbortOnDropStream::mark_completed,
        ))
    }

    async fn abort(&self, request: proto::AbortRequest) -> Result<proto::AbortResponse, Status> {
        self.client
            .abort_request(request.request_id, request.reason)
            .await?;
        Ok(proto::AbortResponse {
            success: true,
            message: String::new(),
        })
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

    async fn verify_engine_ready(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let health = self.client.health_check().await?;
        ensure_engine_healthy("TokenSpeed", health.healthy, &health.message)
    }
}

#[tonic::async_trait]
impl EngineTransport for TokenSpeedWorkerInference {
    async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<EngineTransportStream, Status> {
        let sampled_indexes = sampled_indexes(&request);
        let stream = self
            .client
            .generate(into_tokenspeed_request(request))
            .await?;
        // TokenSpeed chunks are already deltas; only `Complete` is cumulative.
        Ok(GuardedEngineStream::boxed(
            stream,
            sampled_indexes,
            from_tokenspeed_response,
            crate::tokenspeed_scheduler::AbortOnDropStream::mark_completed,
        ))
    }

    async fn abort(&self, request: proto::AbortRequest) -> Result<proto::AbortResponse, Status> {
        self.client
            .abort_request(request.request_id, request.reason)
            .await?;
        Ok(proto::AbortResponse {
            success: true,
            message: String::new(),
        })
    }
}

/// Connect the engine-native gRPC adapter for `engine_type` as a bare
/// [`EngineTransport`], for callers that add their own admission and
/// lifecycle gates.
///
/// An open channel only proves that something listens on the port, and
/// callers announce SERVING as soon as this returns, so the engine's health
/// RPC must also pass.
pub async fn connect_engine_transport(
    engine_type: &str,
    endpoint: &str,
) -> Result<Arc<dyn EngineTransport>, Box<dyn std::error::Error + Send + Sync>> {
    match engine_type.to_ascii_lowercase().as_str() {
        "vllm" => {
            let adapter = VllmWorkerInference::connect(endpoint).await?;
            adapter.verify_engine_ready().await?;
            Ok(Arc::new(adapter))
        }
        "tokenspeed" | "ts" => {
            let adapter = TokenSpeedWorkerInference::connect(endpoint).await?;
            adapter.verify_engine_ready().await?;
            Ok(Arc::new(adapter))
        }
        other => Err(format!(
            "unsupported Worker engine type {other:?}; supported engine types: vllm, tokenspeed"
        )
        .into()),
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
        extra_mm_inputs: Vec::new(),
        media_refs: None,
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
    // The wire carries no prompt logprobs, and asking vLLM for them disables
    // prefix caching for the prefill.
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
    // vLLM's seed is i32; saturate so a large u64 seed stays set, as on the
    // direct vLLM path.
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
                spec_accepted_tokens: complete.spec_accepted_tokens,
                spec_draft_tokens: complete.spec_draft_tokens,
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
                    media_identity: None,
                    matched_stop: complete.matched_stop.map(|matched| match matched {
                        proto::generate_complete::MatchedStop::MatchedTokenId(id) => {
                            vllm::generate_complete::MatchedStop::MatchedTokenId(id)
                        }
                        proto::generate_complete::MatchedStop::MatchedStopStr(value) => {
                            vllm::generate_complete::MatchedStop::MatchedStopStr(value)
                        }
                    }),
                    index: complete.index,
                    spec_accepted_tokens: complete.spec_accepted_tokens,
                    spec_draft_tokens: complete.spec_draft_tokens,
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

/// Convert the router's mature text-generation representation to the stable
/// Worker wire. Unsupported extension lanes fail explicitly rather than being
/// silently discarded. `primary_eos` is the Router tokenizer's primary EOS id
/// for a Worker whose engine cannot resolve it; `None` leaves EOS to the
/// Worker.
pub fn from_tokenspeed_request(
    request: ts::GenerateRequest,
    primary_eos: Option<u32>,
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
        sampling_params: request
            .sampling_params
            .map(|params| from_tokenspeed_sampling(params, primary_eos)),
        return_logprob: request.return_logprob,
        logprob_start_len: request.logprob_start_len,
        top_logprobs_num: request.top_logprobs_num,
        token_ids_logprob: request.token_ids_logprob,
        stream: request.stream,
        data_parallel_rank: request.data_parallel_rank,
    })
}

/// Adapter-side inverse of [`from_tokenspeed_request`], shared by every
/// transport that speaks the TokenSpeed request shape.
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
                    spec_accepted_tokens: complete.spec_accepted_tokens,
                    spec_draft_tokens: complete.spec_draft_tokens,
                })
            }
        }),
    }
}

fn from_tokenspeed_sampling(
    params: ts::SamplingParams,
    primary_eos: Option<u32>,
) -> proto::SamplingParams {
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
        eos_token_id: primary_eos,
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

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::{atomic::AtomicUsize, Arc},
        time::Duration,
    };

    use futures::{stream, Stream, StreamExt};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{transport::Server, Response, Status};

    use super::*;

    struct TestInference {
        aborts: mpsc::UnboundedSender<String>,
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
            request: Request<proto::AbortRequest>,
        ) -> Result<Response<proto::AbortResponse>, Status> {
            self.aborts
                .send(request.into_inner().request_id)
                .map_err(|_| Status::internal("abort observer is gone"))?;
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

        let worker = from_tokenspeed_request(request.clone(), None).expect("portable request");
        assert_eq!(into_tokenspeed_request(worker), request);
    }

    #[test]
    fn disaggregated_request_is_rejected() {
        let request = ts::GenerateRequest {
            kv_bootstrap_info: Some(ts::KvBootstrapInfo::default()),
            ..Default::default()
        };
        let status = from_tokenspeed_request(request, None).expect_err("unsupported extension");
        assert_eq!(status.code(), tonic::Code::Unimplemented);
    }

    #[test]
    fn vllm_request_carries_the_router_supplied_eos() {
        let request = ts::GenerateRequest {
            request_id: "vllm-eos".to_string(),
            tokenized: Some(ts::TokenizedInput {
                input_ids: vec![1],
                ..Default::default()
            }),
            sampling_params: Some(ts::SamplingParams::default()),
            ..Default::default()
        };

        // The Router's dispatch-time EOS rides the wire and reaches vLLM.
        let wire =
            from_tokenspeed_request(request.clone(), Some(128009)).expect("portable request");
        assert_eq!(
            wire.sampling_params.as_ref().and_then(|p| p.eos_token_id),
            Some(128009)
        );
        let sampling = into_vllm_request(wire)
            .expect("vLLM request")
            .sampling_params
            .expect("sampling params");
        assert_eq!(sampling.eos_token_id, Some(128009));

        // Without one, the Worker resolves EOS itself.
        let wire = from_tokenspeed_request(request, None).expect("portable request");
        assert_eq!(
            wire.sampling_params.as_ref().and_then(|p| p.eos_token_id),
            None
        );
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
                        spec_accepted_tokens: 5,
                        spec_draft_tokens: 6,
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
        let status = into_vllm_request(proto::GenerateRequest {
            request_id: "no-input".to_string(),
            sampling_params: Some(proto::SamplingParams::default()),
            ..Default::default()
        })
        .expect_err("tokenized input is required");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
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

    #[tokio::test]
    async fn unsupported_engine_types_are_rejected_by_name() {
        for engine_type in ["sglang", "trtllm", ""] {
            let error = connect_engine_transport(engine_type, "grpc://127.0.0.1:1")
                .await
                .err()
                .expect("unsupported engine type")
                .to_string();
            assert!(
                error.contains(&format!("{engine_type:?}"))
                    && error.contains("vllm")
                    && error.contains("tokenspeed"),
                "{error}"
            );
        }
    }

    type Frames = stream::Iter<std::vec::IntoIter<Result<proto::GenerateResponse, Status>>>;

    fn frame(response: proto::generate_response::Response) -> proto::GenerateResponse {
        proto::GenerateResponse {
            request_id: "multi-index".to_string(),
            response: Some(response),
        }
    }

    #[tokio::test]
    async fn guard_is_released_only_after_every_sampled_index_completes() {
        for (n, completes_until_release) in [(0, 1), (1, 1), (3, 3)] {
            let request = proto::GenerateRequest {
                sampling_params: Some(proto::SamplingParams {
                    n,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let releases = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&releases);
            let frames: Vec<_> = (0..3)
                .flat_map(|index| {
                    [
                        Ok(frame(proto::generate_response::Response::Chunk(
                            proto::GenerateStreamChunk {
                                index,
                                ..Default::default()
                            },
                        ))),
                        Err(Status::internal("passed through")),
                        Ok(frame(proto::generate_response::Response::Complete(
                            proto::GenerateComplete {
                                index,
                                ..Default::default()
                            },
                        ))),
                    ]
                })
                .collect();
            let mut stream = GuardedEngineStream::boxed(
                stream::iter(frames),
                sampled_indexes(&request),
                std::convert::identity,
                move |_: &Frames| {
                    observed.fetch_add(1, Ordering::SeqCst);
                },
            );

            let (mut completes, mut errors) = (0, 0);
            while let Some(item) = stream.next().await {
                match item {
                    Ok(proto::GenerateResponse {
                        response: Some(proto::generate_response::Response::Complete(_)),
                        ..
                    }) => completes += 1,
                    Ok(_) => {}
                    Err(_) => errors += 1,
                }
                assert_eq!(
                    releases.load(Ordering::SeqCst),
                    usize::from(completes >= completes_until_release),
                    "n={n} after {completes} completes"
                );
            }
            assert_eq!((completes, errors), (3, 3));
            assert_eq!(releases.load(Ordering::SeqCst), 1, "released exactly once");
        }
    }

    #[tokio::test]
    async fn stream_drop_sends_abort_over_worker_service() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let address = listener.local_addr().expect("listener address");
        let (aborts, mut aborted) = mpsc::unbounded_channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only tonic server is explicitly aborted before return"
        )]
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(proto::worker_inference_server::WorkerInferenceServer::new(
                    TestInference { aborts },
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

        let request_id = tokio::time::timeout(Duration::from_secs(5), aborted.recv())
            .await
            .expect("drop-triggered abort within the deadline")
            .expect("abort observer");
        assert_eq!(request_id, "drop-me");
        server.abort();
    }
}
