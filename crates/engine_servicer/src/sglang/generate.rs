//! The `Generate` RPC: the one request path. Validates, registers the request
//! (so `Abort` can end it), submits the per-choice engine streams and merges
//! them into one response stream. The headless scheduler keeps its tokenizer
//! on this wire, so string stops ride the request and the scheduler matches
//! them itself; the servicer has no stop matcher of its own.

use std::{
    collections::BTreeMap,
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use engine_zmq_adapter::{to_sglang_response, SglangGenerateStream};
use futures::{stream::SelectAll, Stream};
use smg_grpc_client::{sglang_proto as sg, vllm_proto as vllm};
use tonic::Status;

use super::{hidden_states_cap, HiddenStatesCap, SglangModelInfo, State};
use crate::{
    metrics::{complete_finish_reason, ChoiceMetrics},
    requests::HeldTerminal,
    stream::{poll_engine_item, Choice, RequestStream},
    BoxStream,
};

/// Handle one `Generate` request against the connected scheduler.
/// `trace_headers` is the caller's W3C trace context (see
/// [`crate::trace_context::trace_headers`]), carried to the scheduler beside
/// each choice's request so the engine's spans join the Router's trace.
pub(super) async fn generate(
    state: &Arc<State>,
    req: sg::GenerateRequest,
    trace_headers: Option<BTreeMap<String, String>>,
) -> Result<BoxStream<sg::GenerateResponse>, Status> {
    // The request's arrival, the instant its latencies are measured from.
    let arrival = Instant::now();
    let client = state.shared.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    if req.return_hidden_states {
        refuse_hidden_states_above_cap(&state.model)?;
    }
    let request_id: Arc<str> = Arc::from(req.request_id.as_str());
    // Register before the submit: an `Abort` landing in the gap must find the
    // entry, and a duplicate id is refused up front.
    let (registration, cancel) = state.shared.registry.register(&request_id)?;
    let streaming = req.stream;
    let prompt_tokens = req.tokenized.as_ref().map_or(0, |tokenized| {
        u32::try_from(tokenized.input_ids.len()).unwrap_or(u32::MAX)
    });
    let (max_tokens, n) = req
        .sampling_params
        .as_ref()
        .map_or((None, 1), |params| (params.max_new_tokens, params.n));
    let subs = client
        .generate_sglang_streams_with_trace(req, trace_headers)
        .await?;
    let metrics = state.shared.metrics.request(
        arrival,
        subs.first().and_then(SglangGenerateStream::engine_index),
        prompt_tokens,
        max_tokens,
        n,
    );
    let mut choices = SelectAll::new();
    for sub in subs {
        choices.push(ChoiceStream {
            request_id: Arc::clone(&request_id),
            inner: Some(sub),
            streaming,
            metrics: metrics.choice(),
        });
    }
    Ok(Box::pin(RequestStream::new(
        choices,
        cancel,
        registration,
        metrics,
    )))
}

/// `return_hidden_states` on a scheduler launched below `full`
/// (`--return-hidden-states-mode`): refused ahead of the scheduler with the
/// engine frontend's own words, as the Python servicer refuses it. A launcher
/// whose server args name no cap leaves the decision to the scheduler.
fn refuse_hidden_states_above_cap(model: &SglangModelInfo) -> Result<(), Status> {
    const LAUNCH: &str =
        "Please launch with `--return-hidden-states-mode full` to allow return_hidden_states=True.";
    match hidden_states_cap(model) {
        None | Some(HiddenStatesCap::Full) => Ok(()),
        Some(HiddenStatesCap::Last) => Err(Status::invalid_argument(format!(
            "The requested return_hidden_states mode exceeds the server maximum `last`. {LAUNCH}"
        ))),
        Some(HiddenStatesCap::Null) => Err(Status::invalid_argument(format!(
            "The server is not configured to return hidden states. {LAUNCH}"
        ))),
    }
}

/// One choice of a generate request: the ZMQ-mapped stream under the
/// request's id.
struct ChoiceStream {
    /// The request's id, stamped on every response (the choices of an
    /// `n > 1` fan-out ride under the parent's id, told apart by `index`).
    request_id: Arc<str>,
    /// `None` once this choice ended (terminal response yielded or error).
    /// Dropping it before the engine's own terminal output aborts the
    /// engine-side request.
    inner: Option<SglangGenerateStream>,
    /// Whether the client asked for incremental chunks. A non-streaming
    /// request receives only the terminal `Complete`.
    streaming: bool,
    /// This choice's observation of the request's metrics.
    metrics: ChoiceMetrics,
}

impl ChoiceStream {
    /// The SGLang-proto response, including the choice's reasoning-token
    /// count and selected scores, which the vLLM intermediate cannot carry.
    fn convert(&self, response: vllm::GenerateResponse) -> sg::GenerateResponse {
        match self.inner.as_ref() {
            Some(inner) => inner.to_sglang_response(&self.request_id, response),
            None => to_sglang_response(&self.request_id, response, 0),
        }
    }
}

impl Choice for ChoiceStream {
    /// End this choice on an abort: the engine's parked `Complete` or a
    /// synthesized `abort` one; `None` once it has already ended. Dropping the
    /// engine stream aborts the engine side.
    fn abort(&mut self) -> Option<sg::GenerateResponse> {
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
    type Response = sg::GenerateResponse;

    /// The terminal this choice already holds, when the engine finished it on
    /// the tick before the poll that would have yielded it (the parked
    /// `Complete`). `None` for a choice still open, which is left as it is.
    fn finished(&mut self) -> Option<sg::GenerateResponse> {
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

type ChoiceItem = Result<sg::GenerateResponse, Status>;

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
                    this.metrics.chunk(
                        u32::try_from(chunk.token_ids.len()).unwrap_or(u32::MAX),
                        chunk.prompt_tokens,
                        chunk.cached_tokens,
                    );
                    true
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
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
            // finish, so dropping it sends no abort). `convert` reads the
            // engine stream, so it goes first.
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
