//! The `Generate` RPC: the one request path. Takes the string stops the
//! token-only scheduler cannot match, submits per-choice engine streams,
//! matches the stops on the decoded output, and merges the choices into one
//! response stream that the `Abort` RPC can end.

use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use engine_zmq_adapter::{
    stops::take_tokenspeed_string_stops, to_tokenspeed_response, TokenSpeedGenerateStream,
};
use futures::{stream::SelectAll, Stream};
use llm_tokenizer::stop::StopSequenceDecoder;
use smg_grpc_client::{tokenspeed_proto as ts, vllm_proto as vllm};
use tokio::sync::oneshot;
use tonic::Status;

use super::State;
use crate::{
    requests::{settle_shutdown, CancelReason, HeldTerminal, Registration},
    stop_match::{stop_decoder, StopMatcher},
    BoxStream,
};

/// Handle one `Generate` request against the connected scheduler.
pub(super) async fn generate(
    state: &Arc<State>,
    mut req: ts::GenerateRequest,
) -> Result<BoxStream<ts::GenerateResponse>, Status> {
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
    let subs = client.generate_tokenspeed_streams(req).await?;
    let mut choices = SelectAll::new();
    for sub in subs {
        // Each choice decodes its own text: the matcher is per sequence.
        let decoder = match &decoder {
            Some(_) => stop_decoder(tokenizer, &stops, skip_special_tokens)?,
            None => None,
        };
        choices.push(ChoiceStream::new(
            Arc::clone(&request_id),
            sub,
            decoder,
            streaming,
            min_tokens,
        ));
    }
    Ok(Box::pin(GenerateStream {
        choices,
        cancel,
        aborted: None,
        cut: None,
        _registration: registration,
    }))
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
}

impl ChoiceStream {
    fn new(
        request_id: Arc<str>,
        inner: TokenSpeedGenerateStream,
        decoder: Option<StopSequenceDecoder>,
        streaming: bool,
        min_tokens: u32,
    ) -> Self {
        Self {
            request_id,
            inner: Some(inner),
            stops: StopMatcher::new(decoder, min_tokens),
            streaming,
            pending: None,
        }
    }

    fn convert(&self, response: vllm::GenerateResponse) -> ts::GenerateResponse {
        to_tokenspeed_response(&self.request_id, response)
    }

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
            let Some(vllm::generate_response::Response::Chunk(chunk)) = item.response.as_ref()
            else {
                // The engine's own terminal `Complete` passes through as-is;
                // the engine stream ends after it, and the choice is no
                // longer open once the `Complete` is out (a shutdown landing
                // before the next poll must not cut it; the engine stream has
                // already seen the finish, so dropping it sends no abort).
                let ended = matches!(
                    item.response,
                    Some(vllm::generate_response::Response::Complete(_))
                );
                let response = this.convert(item);
                if ended {
                    this.inner = None;
                }
                return Poll::Ready(Some(Ok(response)));
            };
            let matched = match this.stops.feed(&chunk.token_ids) {
                Ok(matched) => matched,
                Err(status) => {
                    this.inner = None;
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

/// The merged response stream of one generate request: its choices polled
/// together (interleaved, each tagged with its `index`), ended early by the
/// `Abort` RPC.
struct GenerateStream {
    choices: SelectAll<ChoiceStream>,
    cancel: oneshot::Receiver<CancelReason>,
    /// Set once aborted: the terminal `Complete` of every choice still open
    /// at that point, yielded before the stream ends.
    aborted: Option<VecDeque<ts::GenerateResponse>>,
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
