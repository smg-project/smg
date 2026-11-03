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
use futures::{stream::SelectAll, Stream};
use smg_grpc_client::{trtllm_proto as trt, vllm_proto as vllm};
use tonic::Status;

use super::State;
use crate::{
    metrics::{complete_finish_reason, ChoiceMetrics},
    requests::HeldTerminal,
    stream::{poll_engine_item, Choice, RequestStream},
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
    let metrics = state.shared.metrics.request(
        arrival,
        subs.first().and_then(TrtllmGenerateStream::engine_index),
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
