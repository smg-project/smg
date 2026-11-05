//! The `Generate` RPC: the one request path. Takes the string stops the
//! token-only scheduler cannot match, submits per-choice engine streams,
//! matches the stops on the decoded output, and merges the choices into one
//! response stream that the `Abort` RPC can end.

use std::{
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use engine_zmq_adapter::{
    stops::take_tokenspeed_string_stops, to_tokenspeed_response, TokenSpeedGenerateStream,
};
use futures::Stream;
use llm_tokenizer::stop::StopSequenceDecoder;
use smg_grpc_client::{tokenspeed_proto as ts, vllm_proto as vllm};
use tonic::Status;

use super::State;
use crate::{
    load_tracker::{rank_of, ChoiceLoad, PendingGuard, PrefillProbe, Submission},
    metrics::{complete_finish_reason, ChoiceMetrics},
    requests::HeldTerminal,
    servicer::RequestOrigin,
    stop_match::{stop_decoder, StopMatcher},
    stream::{poll_engine_item, Choice, Choices, RequestStream},
    BoxStream,
};

/// Handle one `Generate` request against the connected scheduler. `origin`
/// says whose request it is, which decides what is counted
/// ([`RequestOrigin`]).
pub(super) async fn generate(
    state: &Arc<State>,
    mut req: ts::GenerateRequest,
    origin: RequestOrigin,
) -> Result<BoxStream<ts::GenerateResponse>, Status> {
    // The request's arrival, the instant its latencies are measured from.
    let arrival = Instant::now();
    let client = state.shared.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    let request_id: Arc<str> = Arc::from(req.request_id.as_str());
    // Register before the submit: an `Abort` landing in the gap must find the
    // entry, and a duplicate id is refused up front.
    let (registration, cancel) = state.shared.registry.register(&request_id)?;
    let streaming = req.stream;
    let (skip_special_tokens, min_tokens) = req
        .sampling_params
        .as_ref()
        .map_or((false, 0), |sp| (sp.skip_special_tokens, sp.min_new_tokens));
    // String stops leave the engine request (single-token ones become stop
    // ids) and come back as this servicer's own obligation; the scheduler
    // stops at EOS itself.
    let tokenizer = state.tokenizer();
    let stops = req
        .sampling_params
        .as_mut()
        .map(|params| take_tokenspeed_string_stops(params, tokenizer))
        .unwrap_or_default();
    let decoder = stop_decoder(tokenizer, &stops, skip_special_tokens)?;
    let prompt_tokens = req.tokenized.as_ref().map_or(0, |tokenized| {
        u32::try_from(tokenized.input_ids.len()).unwrap_or(u32::MAX)
    });
    let (max_tokens, n) = req
        .sampling_params
        .as_ref()
        .map_or((None, 1), |params| (params.max_new_tokens, params.n));
    let subs = client
        .generate_tokenspeed_streams_for_model(req, state.model.grid_tensors())
        .await?;
    // A client's request is in the load tracker's view (in flight on the
    // rank its first choice went to, the choice that reports the prompt,
    // from here until the engine's first output or the stream's end, each
    // choice's prefill sampled on the servicer's clock from here to its
    // first output: this wire carries no engine timing) and in the request
    // metrics; the servicer's own health probe is in neither.
    let (probe, loads, metrics) = match origin {
        RequestOrigin::Client => {
            let at = Instant::now();
            let rank = rank_of(
                subs.iter()
                    .find(|sub| sub.index() == 0)
                    .or_else(|| subs.first())
                    .and_then(TokenSpeedGenerateStream::engine_index),
            );
            (
                state.shared.loads.submit(Submission {
                    request_id: &request_id,
                    rank,
                    prompt_tokens,
                    at,
                }),
                Some(PendingGuard::new(Arc::clone(state), &request_id)),
                state.shared.metrics.request(
                    arrival,
                    subs.first()
                        .and_then(TokenSpeedGenerateStream::engine_index),
                    prompt_tokens,
                    max_tokens,
                    n,
                ),
            )
        }
        RequestOrigin::HealthProbe => (
            PrefillProbe::default(),
            None,
            state.shared.metrics.uncounted(),
        ),
    };
    let mut choices = Choices::new();
    for sub in subs {
        // Each choice decodes its own text: the matcher is per sequence.
        let decoder = match &decoder {
            Some(_) => stop_decoder(tokenizer, &stops, skip_special_tokens)?,
            None => None,
        };
        // Each choice's engine request went to its own rank (an unpinned
        // fan-out is placed per choice): its tokens are credited there.
        let load = match origin {
            RequestOrigin::Client => {
                ChoiceLoad::new(rank_of(sub.engine_index()), sub.index(), probe)
            }
            RequestOrigin::HealthProbe => ChoiceLoad::uncounted(),
        };
        choices.push(ChoiceStream::new(
            Arc::clone(&request_id),
            Arc::clone(state),
            sub,
            decoder,
            streaming,
            min_tokens,
            metrics.choice(),
            load,
        ));
    }
    Ok(Box::pin(
        RequestStream::new(choices, cancel, registration, metrics).holding(loads),
    ))
}

/// One choice of a generate request: the ZMQ-mapped stream plus the string
/// stop matcher the scheduler cannot run itself.
struct ChoiceStream {
    /// The request's id, stamped on every response (the choices of an
    /// `n > 1` fan-out ride under the parent's id, told apart by `index`).
    request_id: Arc<str>,
    /// `None` once this choice ended (terminal response yielded or error).
    /// Dropping it before the engine's own terminal output aborts the
    /// engine-side request.
    inner: Option<TokenSpeedGenerateStream>,
    stops: StopMatcher,
    /// Whether the client asked for incremental chunks. A non-streaming
    /// request receives only the terminal `Complete`.
    streaming: bool,
    /// A frontend-synthesized terminal `Complete` to yield next.
    pending: Option<vllm::GenerateResponse>,
    /// This choice's observation of the request's metrics.
    metrics: ChoiceMetrics,
    /// The servicer state, for the load tracker.
    state: Arc<State>,
    /// This choice's share of the load tracker's bookkeeping.
    load: ChoiceLoad,
}

impl ChoiceStream {
    #[expect(clippy::too_many_arguments)]
    fn new(
        request_id: Arc<str>,
        state: Arc<State>,
        inner: TokenSpeedGenerateStream,
        decoder: Option<StopSequenceDecoder>,
        streaming: bool,
        min_tokens: u32,
        metrics: ChoiceMetrics,
        load: ChoiceLoad,
    ) -> Self {
        Self {
            request_id,
            inner: Some(inner),
            stops: StopMatcher::new(decoder, min_tokens),
            streaming,
            pending: None,
            metrics,
            state,
            load,
        }
    }

    fn convert(&self, response: vllm::GenerateResponse) -> ts::GenerateResponse {
        to_tokenspeed_response(&self.request_id, response)
    }
}

impl Choice for ChoiceStream {
    /// End this choice on an abort: the `Complete` it was about to yield, the
    /// engine's parked one, or a synthesized `abort` one; `None` once it has
    /// already ended. Dropping the engine stream aborts the engine side.
    fn abort(&mut self) -> Option<ts::GenerateResponse> {
        if let Some(complete) = self.pending.take() {
            self.inner = None;
            return Some(self.convert(complete));
        }
        let mut inner = self.inner.take()?;
        let complete = inner.complete_aborted();
        if let Some(reason) = complete_finish_reason(&complete) {
            self.metrics.ended(reason);
        }
        Some(self.convert(complete))
    }
}

impl HeldTerminal for ChoiceStream {
    type Response = ts::GenerateResponse;

    /// The terminal this choice already holds, when the engine (or a stop
    /// string) finished it before the poll that would have yielded it: the
    /// `Complete` it was about to yield, or the engine's parked one. `None`
    /// for a choice still open, which is left as it is.
    fn finished(&mut self) -> Option<ts::GenerateResponse> {
        if let Some(complete) = self.pending.take() {
            self.inner = None;
            return Some(self.convert(complete));
        }
        let inner = self.inner.as_mut()?;
        if !inner.has_parked_complete() {
            return None;
        }
        // `complete_aborted` yields the parked `Complete` when there is one.
        let complete = inner.complete_aborted();
        if let Some(reason) = complete_finish_reason(&complete) {
            self.metrics.ended(reason);
        }
        self.inner = None;
        Some(self.convert(complete))
    }

    /// Whether this choice is still open: neither ended nor holding its
    /// terminal.
    fn is_open(&self) -> bool {
        self.pending.is_none() && self.inner.is_some()
    }
}

type ChoiceItem = Result<ts::GenerateResponse, Status>;

impl Stream for ChoiceStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(complete) = this.pending.take() {
                // The frontend ended this choice: drop the engine stream so the
                // engine-side request is aborted, then deliver the Complete.
                this.inner = None;
                return Poll::Ready(Some(Ok(this.convert(complete))));
            }
            let item = match poll_engine_item(&mut this.inner, &mut this.metrics, cx) {
                ControlFlow::Continue(item) => item,
                ControlFlow::Break(poll) => return poll,
            };
            let Some(vllm::generate_response::Response::Chunk(chunk)) = item.response.as_ref()
            else {
                // The engine's own terminal `Complete` passes through as-is;
                // the engine stream ends after it, and the choice is no
                // longer open once the `Complete` is out (a shutdown landing
                // before the next poll must not cut it; the engine stream has
                // already seen the finish, so dropping it sends no abort).
                let mut ended = false;
                if let Some(vllm::generate_response::Response::Complete(complete)) =
                    item.response.as_ref()
                {
                    this.load.count_prompt(
                        &this.state.shared.loads,
                        &this.request_id,
                        complete.prompt_tokens,
                        complete.cached_tokens,
                        Instant::now(),
                    );
                    this.metrics.complete(
                        &complete.finish_reason,
                        complete.prompt_tokens,
                        complete.cached_tokens,
                    );
                    ended = true;
                }
                let response = this.convert(item);
                if ended {
                    this.inner = None;
                }
                return Poll::Ready(Some(Ok(response)));
            };
            let tokens = u32::try_from(chunk.token_ids.len()).unwrap_or(u32::MAX);
            // The metrics' clock read serves the load tracker too.
            let at = this
                .metrics
                .chunk(tokens, chunk.prompt_tokens, chunk.cached_tokens)
                .unwrap_or_else(Instant::now);
            let loads = &this.state.shared.loads;
            this.load.count_prompt(
                loads,
                &this.request_id,
                chunk.prompt_tokens,
                chunk.cached_tokens,
                at,
            );
            this.load.generated(loads, tokens, at);
            let matched = match this.stops.feed(&chunk.token_ids) {
                Ok(matched) => matched,
                Err(status) => {
                    this.inner = None;
                    this.metrics.failed(status.code());
                    return Poll::Ready(Some(Err(status)));
                }
            };
            if let Some(matched) = matched {
                if let Some(inner) = this.inner.as_mut() {
                    // A stop string that is also a single token reaches the
                    // engine as a stop id, so the engine may finish on the
                    // very tick the string matched; its parked `Complete`
                    // already holds the accumulated ids and logprobs.
                    if !inner.has_parked_complete() {
                        this.pending = Some(inner.complete_with_matched_stop(matched));
                        this.metrics.string_stop();
                    }
                }
                if !this.streaming {
                    continue;
                }
                // The triggering delta goes out first.
                return Poll::Ready(Some(Ok(this.convert(item))));
            }
            if !this.streaming {
                continue;
            }
            return Poll::Ready(Some(Ok(this.convert(item))));
        }
    }
}
