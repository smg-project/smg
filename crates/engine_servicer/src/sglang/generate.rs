//! The `Generate` RPC: the one request path. Validates, registers the request
//! (so `Abort` can end it), submits the per-choice engine streams and merges
//! them into one response stream. The headless scheduler keeps its tokenizer
//! on this wire, so string stops ride the request and the scheduler matches
//! them itself; the servicer has no stop matcher of its own.

use std::{
    collections::VecDeque,
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
use crate::{requests::Registration, BoxStream};

/// Handle one `Generate` request against the connected scheduler.
pub(super) async fn generate(
    state: &Arc<State>,
    req: sg::GenerateRequest,
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
    let subs = client.generate_sglang_streams(req).await?;
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
    /// The SGLang-proto response, with the scheduler's reasoning-token count
    /// for this choice (the vLLM-proto intermediate has no slot for it).
    fn convert(&self, response: vllm::GenerateResponse) -> sg::GenerateResponse {
        let reasoning_tokens = self
            .inner
            .as_ref()
            .map_or(0, SglangGenerateStream::reasoning_tokens);
        to_sglang_response(&self.request_id, response, reasoning_tokens)
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
            // engine stream ends after it.
            return Poll::Ready(Some(Ok(this.convert(item))));
        }
    }
}

/// The merged response stream of one generate request: its choices polled
/// together (interleaved, each tagged with its `index`), ended early by the
/// `Abort` RPC.
struct GenerateStream {
    choices: SelectAll<ChoiceStream>,
    cancel: oneshot::Receiver<()>,
    /// Set once aborted: the terminal `Complete` of every choice still open
    /// at that point, yielded before the stream ends.
    aborted: Option<VecDeque<sg::GenerateResponse>>,
    _registration: Registration,
}

impl Stream for GenerateStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(aborted) = this.aborted.as_mut() {
                return Poll::Ready(aborted.pop_front().map(Ok));
            }
            // Either an `Abort` fired the sender or shutdown dropped the
            // registry: both end the request as the Python servicer does, with
            // each open choice's `Complete(finish_reason = "abort")`. Dropping
            // the choices aborts the engine side.
            if Pin::new(&mut this.cancel).poll(cx).is_ready() {
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
