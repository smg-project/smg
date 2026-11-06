//! The `Generate` RPC: the one request path. Validates, registers the request
//! (so `Abort` can end it), submits the per-choice engine streams and merges
//! them into one response stream. The engine keeps its tokenizer on this
//! wire, so string stops ride the request and the engine matches them
//! itself; the servicer has no stop matcher of its own.

use std::{
    collections::BTreeMap,
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use engine_zmq_adapter::{to_trtllm_response, TrtllmGenerateStream};
use futures::Stream;
use smg_grpc_client::{trtllm_proto as trt, vllm_proto as vllm};
use tonic::Status;

use super::State;
use crate::{
    load_tracker::{rank_of, ChoiceLoad, PendingGuard, Submission},
    metrics::{complete_finish_reason, ChoiceMetrics},
    requests::HeldTerminal,
    stream::{poll_engine_item, Choice, Choices, RequestStream},
    BoxStream,
};

/// Handle one `Generate` request against the connected engine.
/// `trace_headers` is the caller's W3C trace context (see
/// [`crate::trace_context::trace_headers`]), carried to the engine beside
/// each choice's request so the engine's spans join the Router's trace.
pub(super) async fn generate(
    state: &Arc<State>,
    req: trt::GenerateRequest,
    trace_headers: Option<BTreeMap<String, String>>,
) -> Result<BoxStream<trt::GenerateResponse>, Status> {
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
    let streaming = req.streaming;
    let prompt_tokens = req.tokenized.as_ref().map_or(0, |tokenized| {
        u32::try_from(tokenized.input_token_ids.len()).unwrap_or(u32::MAX)
    });
    let max_tokens = (req.max_tokens > 0).then_some(req.max_tokens);
    let n = req
        .sampling_config
        .as_ref()
        .map_or(1, |config| config.num_return_sequences.max(1));
    let subs = client.generate_trtllm_streams(req, trace_headers).await?;
    // The load tracker's view of the request: in flight on the rank its
    // first choice went to (the choice that reports the prompt) from here
    // until the shim's first output (or the stream's end), each choice's
    // prefill sampled on the servicer's clock from here to its first output
    // (this wire carries no engine timing).
    let at = Instant::now();
    let rank = rank_of(
        subs.iter()
            .find(|sub| sub.index() == 0)
            .or_else(|| subs.first())
            .and_then(TrtllmGenerateStream::engine_index),
    );
    let probe = state.shared.loads.submit(Submission {
        request_id: &request_id,
        rank,
        prompt_tokens,
        at,
    });
    let loads = PendingGuard::new(Arc::clone(state), &request_id);
    let metrics = state.shared.metrics.request(
        arrival,
        subs.first().and_then(TrtllmGenerateStream::engine_index),
        prompt_tokens,
        max_tokens,
        n,
    );
    let mut choices = Choices::new();
    for sub in subs {
        // Each choice's engine request went to its own rank (an unpinned
        // fan-out is placed per choice): its tokens are credited there.
        let load = ChoiceLoad::new(rank_of(sub.engine_index()), sub.index(), probe);
        choices.push(ChoiceStream {
            request_id: Arc::clone(&request_id),
            state: Arc::clone(state),
            inner: Some(sub),
            streaming,
            metrics: metrics.choice(),
            load,
        });
    }
    let stream = RequestStream::new(choices, cancel, registration, metrics).holding(loads);
    Ok(Box::pin(stream))
}

/// One choice of a generate request: the ZMQ-mapped stream under the
/// request's id.
struct ChoiceStream {
    /// The request's id, stamped on every response (the choices of an
    /// `n > 1` fan-out ride under the parent's id, told apart by
    /// `sequence_index`).
    request_id: Arc<str>,
    /// `None` once this choice ended (terminal response yielded or error).
    /// Dropping it before the engine's own terminal output aborts the
    /// engine-side request.
    inner: Option<TrtllmGenerateStream>,
    /// Whether the client asked for incremental chunks. A non-streaming
    /// request receives only the terminal `Complete`.
    streaming: bool,
    /// This choice's observation of the request's metrics.
    metrics: ChoiceMetrics,
    /// The servicer state, for the load tracker.
    state: Arc<State>,
    /// This choice's share of the load tracker's bookkeeping.
    load: ChoiceLoad,
}

impl ChoiceStream {
    fn convert(&self, response: vllm::GenerateResponse) -> trt::GenerateResponse {
        to_trtllm_response(&self.request_id, response)
    }
}

impl Choice for ChoiceStream {
    /// End this choice on an abort: the engine's parked `Complete` or a
    /// synthesized `abort` one; `None` once it has already ended. Dropping the
    /// engine stream aborts the engine side.
    fn abort(&mut self) -> Option<trt::GenerateResponse> {
        let complete = self.inner.as_mut()?.complete_aborted();
        if let Some(reason) = complete_finish_reason(&complete) {
            self.metrics.ended(reason);
        }
        let response = self.convert(complete);
        self.inner = None;
        Some(response)
    }
}

impl HeldTerminal for ChoiceStream {
    type Response = trt::GenerateResponse;

    /// The terminal this choice already holds, when the engine finished it on
    /// the tick before the poll that would have yielded it (the parked
    /// `Complete`). `None` for a choice still open, which is left as it is.
    fn finished(&mut self) -> Option<trt::GenerateResponse> {
        let inner = self.inner.as_mut()?;
        if !inner.has_parked_complete() {
            return None;
        }
        // `complete_aborted` yields the parked `Complete` when there is one.
        let complete = inner.complete_aborted();
        if let Some(reason) = complete_finish_reason(&complete) {
            self.metrics.ended(reason);
        }
        let response = self.convert(complete);
        self.inner = None;
        Some(response)
    }

    /// Whether this choice is still open: neither ended nor holding its
    /// terminal.
    fn is_open(&self) -> bool {
        self.inner.is_some()
    }
}

type ChoiceItem = Result<trt::GenerateResponse, Status>;

impl Stream for ChoiceStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            let item = match poll_engine_item(&mut this.inner, &mut this.metrics, cx) {
                ControlFlow::Continue(item) => item,
                ControlFlow::Break(poll) => return poll,
            };
            let is_chunk = match item.response.as_ref() {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
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
                    true
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
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
                    false
                }
                _ => false,
            };
            if is_chunk && !this.streaming {
                continue;
            }
            // The engine's own terminal `Complete` passes through as-is; the
            // engine stream ends after it, and the choice is no longer open
            // once the `Complete` is out (a shutdown landing before the next
            // poll must not cut it; the engine stream has already seen the
            // finish, so dropping it sends no abort).
            let ended = matches!(
                item.response,
                Some(vllm::generate_response::Response::Complete(_))
            );
            let response = this.convert(item);
            if ended {
                this.inner = None;
            }
            return Poll::Ready(Some(Ok(response)));
        }
    }
}
