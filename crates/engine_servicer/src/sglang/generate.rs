//! The `Generate` RPC: the one request path. Validates, registers the request
//! (so `Abort` can end it), submits the per-choice engine streams and merges
//! them into one response stream. The headless scheduler keeps its tokenizer
//! on this wire, so string stops ride the request and the scheduler matches
//! them itself; the servicer has no stop matcher of its own.

use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use engine_zmq_adapter::{to_sglang_response, SglangGenerateStream};
use futures::{stream::SelectAll, Stream};
use smg_grpc_client::{sglang_proto as sg, vllm_proto as vllm};
use tokio::sync::oneshot;
use tonic::Status;

use super::State;
use crate::{
    requests::{settle_shutdown, CancelReason, HeldTerminal, Registration},
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
    let client = state.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    if req
        .tokenized
        .as_ref()
        .is_none_or(|tokenized| tokenized.input_ids.is_empty())
    {
        return Err(Status::invalid_argument(
            "tokenized.input_ids is required: the Router tokenizes before it dispatches",
        ));
    }
    let request_id: Arc<str> = Arc::from(req.request_id.as_str());
    // Register before the submit: an `Abort` landing in the gap must find the
    // entry, and a duplicate id is refused up front.
    let (registration, cancel) = state.registry.register(&request_id)?;
    let streaming = req.stream;
    let subs = client
        .generate_sglang_streams_with_trace(req, trace_headers)
        .await?;
    let mut choices = SelectAll::new();
    for sub in subs {
        choices.push(ChoiceStream {
            request_id: Arc::clone(&request_id),
            inner: Some(sub),
            streaming,
        });
    }
    Ok(Box::pin(GenerateStream {
        choices,
        cancel,
        aborted: None,
        cut: None,
        _registration: registration,
    }))
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

    /// End this choice on an abort: the engine's parked `Complete` or a
    /// synthesized `abort` one; `None` once it has already ended. Dropping the
    /// engine stream aborts the engine side.
    fn abort(&mut self) -> Option<sg::GenerateResponse> {
        let complete = self.inner.as_mut()?.complete_aborted();
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
            let Some(inner) = this.inner.as_mut() else {
                return Poll::Ready(None);
            };
            let item = match Pin::new(inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.inner = None;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(status))) => {
                    this.inner = None;
                    return Poll::Ready(Some(Err(status)));
                }
                Poll::Ready(Some(Ok(item))) => item,
            };
            let is_chunk = matches!(
                item.response,
                Some(vllm::generate_response::Response::Chunk(_))
            );
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

/// The merged response stream of one generate request: its choices polled
/// together (interleaved, each tagged with its `index`), ended early by the
/// `Abort` RPC.
struct GenerateStream {
    choices: SelectAll<ChoiceStream>,
    cancel: oneshot::Receiver<CancelReason>,
    /// Set once aborted: the terminal `Complete` of every choice still open
    /// at that point, yielded before the stream ends.
    aborted: Option<VecDeque<sg::GenerateResponse>>,
    /// The status a shutdown ends the stream with once `aborted` is drained,
    /// when it cut a choice still open; `None` ends it plainly.
    cut: Option<Status>,
    _registration: Registration,
}

impl Stream for GenerateStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(aborted) = this.aborted.as_mut() {
                if let Some(response) = aborted.pop_front() {
                    return Poll::Ready(Some(Ok(response)));
                }
                // The completions the choices already held went out; a
                // shutdown that cut a choice still open ends the stream with
                // its status, a plain abort just ends it.
                return Poll::Ready(this.cut.take().map(Err));
            }
            // The cancellation names its cause. An `Abort` RPC is the caller's
            // own abort: the request ends as the Python servicer ends it, with
            // each open choice's `Complete(finish_reason = "abort")`. A
            // shutdown with the request still open is its engine side going
            // away under it (the lifecycle owner stopping the pair after the
            // engine exited, a drain whose grace ran out): no completion the
            // caller could read as a short answer, so the stream ends with
            // UNAVAILABLE, which the Router classes as the worker's failure.
            // Either way dropping the choices aborts the engine side.
            if let Poll::Ready(reason) = Pin::new(&mut this.cancel).poll(cx) {
                if reason == Ok(CancelReason::Shutdown) {
                    // A choice the engine (or a stop string) finished before
                    // this poll keeps its completion; only a choice still
                    // open is cut, and only then does the stream end with the
                    // status (see `settle_shutdown`).
                    let (finished, cut) = settle_shutdown(&mut this.choices);
                    this.aborted = Some(finished);
                    this.cut = cut.then(|| Status::unavailable(crate::requests::SHUTDOWN_MESSAGE));
                    continue;
                }
                // The abort, or a sender dropped without a word (the registry
                // entry went away): the completion it always was.
                let aborted = this
                    .choices
                    .iter_mut()
                    .filter_map(ChoiceStream::abort)
                    .collect();
                this.choices.clear();
                this.aborted = Some(aborted);
                continue;
            }
            return Pin::new(&mut this.choices).poll_next(cx);
        }
    }
}
