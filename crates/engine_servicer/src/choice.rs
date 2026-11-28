//! The choice streams every wire's `Generate` builds its request stream
//! from, written once over the adapter's per-wire engine streams: the choice
//! that passes the engine's items through as the wire's proto responses
//! ([`Plain`]: the wires whose engine matches string stops itself), and the
//! choice that matches string stops on the decoded output in front of an
//! engine that cannot ([`StopMatched`]: the token-only wires). A wire supplies
//! its proto conversion, what it stamps on the way out and what it checks on
//! the engine's own terminal ([`Shape`]); the metrics and load-tracker feed,
//! the abort, the held terminal and the poll loop are decided here, one way
//! for every wire.

use std::{
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use engine_zmq_adapter::{
    FrontendFinish, SglangGenerateStream, TokenSpeedGenerateStream, TrtllmGenerateStream,
    VllmGenerateStream,
};
use futures::Stream;
use llm_tokenizer::stop::StopSequenceDecoder;
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;

use crate::{
    load_tracker::ChoiceLoad,
    metrics::{complete_finish_reason, ChoiceMetrics},
    requests::HeldTerminal,
    servicer::WireState,
    stop_match::StopMatcher,
    stream::{poll_engine_item, Choice},
};

/// One choice's engine stream as the choices read it: the adapter's per-wire
/// stream types, which yield the intermediate proto responses, park the
/// engine's terminal `Complete` for the poll that follows its finish tick and
/// end from the frontend's side through [`FrontendFinish`].
pub(crate) trait EngineChoice:
    FrontendFinish + Stream<Item = Result<vllm::GenerateResponse, Status>> + Unpin + Send
{
    /// The prefill the engine timed itself, taken once: the computed tokens
    /// with the engine's start and end seconds. `None` on a wire without
    /// engine timing, whose choices sample their prefill on the servicer's
    /// clock through the probe they were armed with.
    fn take_prefill_sample(&mut self) -> Option<(u32, f64, f64)> {
        None
    }
}

/// An engine stream of a token-only wire: the servicer matches the string
/// stops, and ends the engine request with the matched stop on its
/// `Complete` ([`FrontendFinish::complete_with_matched_stop`]).
pub(crate) trait MatchesStops: EngineChoice {}

impl EngineChoice for VllmGenerateStream {
    fn take_prefill_sample(&mut self) -> Option<(u32, f64, f64)> {
        VllmGenerateStream::take_prefill_sample(self)
    }
}

impl EngineChoice for SglangGenerateStream {}

impl EngineChoice for TokenSpeedGenerateStream {}

impl EngineChoice for TrtllmGenerateStream {}

impl MatchesStops for VllmGenerateStream {}

impl MatchesStops for TokenSpeedGenerateStream {}

/// What a wire makes of a choice's items: its proto response for one engine
/// item under the request's id, with whatever it stamps on the way out, and
/// its check of the engine's own terminal `Complete`.
pub(crate) trait Shape: Send + Unpin {
    /// The adapter's engine stream of the wire.
    type Engine: EngineChoice;
    /// The wire's servicer state; its load tracker takes every choice's feed.
    type State: WireState;
    /// The wire's proto response.
    type Response: Send + 'static;

    /// `response` as the wire's response under `request_id`. `engine` is the
    /// choice's engine stream while it is open (a wire that reads per-stream
    /// rows off it converts with them) and `None` once it ended.
    fn convert(
        &self,
        request_id: &str,
        engine: Option<&Self::Engine>,
        response: vllm::GenerateResponse,
    ) -> Self::Response;

    /// The wire's check of the engine's own terminal `Complete` before it
    /// goes out; an error ends the choice with it instead. Nothing to check,
    /// by default.
    fn completed(
        &self,
        _state: &Self::State,
        _complete: &vllm::GenerateComplete,
    ) -> Result<(), Status> {
        Ok(())
    }
}

/// One choice of a generate request on a wire whose engine matches the
/// string stops itself: the engine's items pass through as the wire's
/// responses, the terminal `Complete` as it came.
pub(crate) struct Plain<S: Shape> {
    /// `None` once this choice ended (terminal response yielded or error).
    /// Dropping it before the engine's own terminal output aborts the
    /// engine-side request.
    inner: Option<S::Engine>,
    /// Whether the client asked for incremental chunks. A non-streaming
    /// request receives only the terminal `Complete`.
    streaming: bool,
    /// This choice's observation of the request's metrics.
    metrics: ChoiceMetrics,
    /// The servicer state, for the load tracker and the wire's checks.
    state: Arc<S::State>,
    /// The request's id: the load tracker's key, and what the wire stamps on
    /// every response (the choices of an `n > 1` fan-out ride under the
    /// parent's id, told apart by their index).
    request_id: Arc<str>,
    /// This choice's share of the load tracker's bookkeeping.
    load: ChoiceLoad,
    /// The wire's conversion, with what it stamps and checks.
    shape: S,
}

impl<S: Shape> Plain<S> {
    pub(crate) fn new(
        state: Arc<S::State>,
        request_id: Arc<str>,
        inner: S::Engine,
        streaming: bool,
        metrics: ChoiceMetrics,
        load: ChoiceLoad,
        shape: S,
    ) -> Self {
        Self {
            inner: Some(inner),
            streaming,
            metrics,
            state,
            request_id,
            load,
            shape,
        }
    }

    /// The engine's `item` accounted: its tokens and the request's prompt
    /// figures fed to the metrics and the load tracker (with the engine-timed
    /// prefill, on a wire that has one), and the wire's check of the engine's
    /// own `Complete`.
    #[inline]
    fn account(&mut self, item: &vllm::GenerateResponse) -> Result<(), Status> {
        let loads = &self.state.shared().loads;
        if let Some((tokens, start, end)) = self
            .inner
            .as_mut()
            .and_then(EngineChoice::take_prefill_sample)
        {
            loads.prefill_computed(self.load.rank(), tokens, start, end);
        }
        match item.response.as_ref() {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                let tokens = u32::try_from(chunk.token_ids.len()).unwrap_or(u32::MAX);
                // The metrics' clock read serves the load tracker too.
                let at = self
                    .metrics
                    .chunk(tokens, chunk.prompt_tokens, chunk.cached_tokens)
                    .unwrap_or_else(Instant::now);
                self.load.count_prompt(
                    loads,
                    &self.request_id,
                    chunk.prompt_tokens,
                    chunk.cached_tokens,
                    at,
                );
                self.load.generated(loads, tokens, at);
            }
            Some(vllm::generate_response::Response::Complete(complete)) => {
                self.load.count_prompt(
                    loads,
                    &self.request_id,
                    complete.prompt_tokens,
                    complete.cached_tokens,
                    Instant::now(),
                );
                self.metrics.complete(
                    &complete.finish_reason,
                    complete.prompt_tokens,
                    complete.cached_tokens,
                );
                self.shape.completed(&self.state, complete)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Whether `item` is dropped: a chunk of a request that asked for the
    /// terminal `Complete` alone.
    #[inline(always)]
    fn skipped(&self, item: &vllm::GenerateResponse) -> bool {
        !self.streaming
            && matches!(
                item.response,
                Some(vllm::generate_response::Response::Chunk(_))
            )
    }

    /// `item` as the wire's response. The engine's own terminal `Complete`
    /// passes through as it came; the engine stream ends after it, and the
    /// choice is no longer open once the `Complete` is out (a shutdown
    /// landing before the next poll must not cut it; the engine stream has
    /// already seen the finish, so dropping it sends no abort). The
    /// conversion reads the engine stream, so it goes first.
    #[inline(always)]
    fn yielded(&mut self, item: vllm::GenerateResponse) -> S::Response {
        let ended = matches!(
            item.response,
            Some(vllm::generate_response::Response::Complete(_))
        );
        let response = self
            .shape
            .convert(&self.request_id, self.inner.as_ref(), item);
        if ended {
            self.inner = None;
        }
        response
    }

    /// The engine's parked `Complete` or a synthesized `abort` one, converted
    /// while the engine stream is still there to read, the choice ended with
    /// it; `None` once it has already ended.
    fn terminal(&mut self) -> Option<S::Response> {
        let complete = self.inner.as_mut()?.complete_aborted();
        if let Some(reason) = complete_finish_reason(&complete) {
            self.metrics.ended(reason);
        }
        let response = self
            .shape
            .convert(&self.request_id, self.inner.as_ref(), complete);
        self.inner = None;
        Some(response)
    }

    /// A frontend-synthesized terminal: the engine stream dropped (which
    /// aborts the engine side), the choice ended with `complete`.
    fn ended_with(&mut self, complete: vllm::GenerateResponse) -> S::Response {
        self.inner = None;
        self.shape.convert(&self.request_id, None, complete)
    }

    /// The choice failed with `status`: the engine stream dropped so the
    /// choice stays ended, the error recorded on its metrics.
    fn failed(&mut self, status: Status) -> Status {
        self.inner = None;
        self.metrics.failed(status.code());
        status
    }
}

impl<S: Shape> Plain<S>
where
    S::Engine: MatchesStops,
{
    /// The engine's `Complete` for a stop string the matcher found in the
    /// last chunk, with the ids and logprobs so far; `None` once the choice
    /// has ended, or when the engine parked its own `Complete` on the same
    /// tick (a stop string that is also a single token reaches the engine as
    /// a stop id, so the engine may finish on the very tick the string
    /// matched; its parked `Complete` already holds the accumulated ids and
    /// logprobs, and goes through instead of a synthesized empty one).
    fn complete_with_matched_stop(&mut self, matched: String) -> Option<vllm::GenerateResponse> {
        let inner = self.inner.as_mut()?;
        if inner.has_parked_complete() {
            return None;
        }
        let complete = inner.complete_with_matched_stop(matched);
        self.metrics.string_stop();
        Some(complete)
    }
}

impl<S: Shape> Choice for Plain<S> {
    /// End this choice on an abort: the engine's parked `Complete` or a
    /// synthesized `abort` one; `None` once it has already ended. Dropping the
    /// engine stream aborts the engine side.
    fn abort(&mut self) -> Option<S::Response> {
        self.terminal()
    }
}

impl<S: Shape> HeldTerminal for Plain<S> {
    type Response = S::Response;

    /// The terminal this choice already holds, when the engine finished it on
    /// the tick before the poll that would have yielded it (the parked
    /// `Complete`). `None` for a choice still open, which is left as it is.
    fn finished(&mut self) -> Option<S::Response> {
        if !self.inner.as_ref()?.has_parked_complete() {
            return None;
        }
        self.terminal()
    }

    /// Whether this choice is still open: neither ended nor holding its
    /// terminal.
    fn is_open(&self) -> bool {
        self.inner.is_some()
    }
}

impl<S: Shape> Stream for Plain<S> {
    type Item = Result<S::Response, Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            let item = match poll_engine_item(&mut this.inner, &mut this.metrics, cx) {
                ControlFlow::Continue(item) => item,
                ControlFlow::Break(poll) => return poll,
            };
            if let Err(status) = this.account(&item) {
                return Poll::Ready(Some(Err(this.failed(status))));
            }
            if this.skipped(&item) {
                continue;
            }
            return Poll::Ready(Some(Ok(this.yielded(item))));
        }
    }
}

/// One choice of a generate request on a token-only wire: a [`Plain`] choice
/// with the string stop matcher the engine cannot run itself in front, which
/// ends the choice with a frontend-synthesized `Complete` on a match.
pub(crate) struct StopMatched<S: Shape> {
    choice: Plain<S>,
    /// Decodes the output incrementally and matches the request's string
    /// stops, honouring `min_tokens`.
    stops: StopMatcher,
    /// A frontend-synthesized terminal `Complete` to yield next.
    pending: Option<vllm::GenerateResponse>,
}

impl<S: Shape> StopMatched<S>
where
    S::Engine: MatchesStops,
{
    pub(crate) fn new(
        choice: Plain<S>,
        decoder: Option<StopSequenceDecoder>,
        min_tokens: u32,
    ) -> Self {
        Self {
            choice,
            stops: StopMatcher::new(decoder, min_tokens),
            pending: None,
        }
    }

    /// The `Complete` this choice was about to yield, if the matcher ended
    /// it: the engine stream dropped (which aborts the engine side) and the
    /// choice ended with it.
    fn take_pending(&mut self) -> Option<S::Response> {
        let complete = self.pending.take()?;
        Some(self.choice.ended_with(complete))
    }
}

impl<S: Shape> Choice for StopMatched<S>
where
    S::Engine: MatchesStops,
{
    /// End this choice on an abort: the `Complete` it was about to yield, the
    /// engine's parked one, or a synthesized `abort` one; `None` once it has
    /// already ended. Dropping the engine stream aborts the engine side.
    fn abort(&mut self) -> Option<S::Response> {
        if let Some(response) = self.take_pending() {
            return Some(response);
        }
        self.choice.terminal()
    }
}

impl<S: Shape> HeldTerminal for StopMatched<S>
where
    S::Engine: MatchesStops,
{
    type Response = S::Response;

    /// The terminal this choice already holds, when the engine (or a stop
    /// string) finished it before the poll that would have yielded it: the
    /// `Complete` it was about to yield, or the engine's parked one. `None`
    /// for a choice still open, which is left as it is.
    fn finished(&mut self) -> Option<S::Response> {
        if let Some(response) = self.take_pending() {
            return Some(response);
        }
        self.choice.finished()
    }

    /// Whether this choice is still open: neither ended nor holding its
    /// terminal.
    fn is_open(&self) -> bool {
        self.pending.is_none() && self.choice.is_open()
    }
}

impl<S: Shape> Stream for StopMatched<S>
where
    S::Engine: MatchesStops,
{
    type Item = Result<S::Response, Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(response) = this.take_pending() {
                // The frontend ended this choice: the engine stream is
                // dropped so the engine-side request is aborted, then the
                // Complete is delivered.
                return Poll::Ready(Some(Ok(response)));
            }
            let choice = &mut this.choice;
            let item = match poll_engine_item(&mut choice.inner, &mut choice.metrics, cx) {
                ControlFlow::Continue(item) => item,
                ControlFlow::Break(poll) => return poll,
            };
            if let Err(status) = choice.account(&item) {
                return Poll::Ready(Some(Err(choice.failed(status))));
            }
            if let Some(vllm::generate_response::Response::Chunk(chunk)) = item.response.as_ref() {
                let matched = match this.stops.feed(&chunk.token_ids) {
                    Ok(matched) => matched,
                    Err(status) => return Poll::Ready(Some(Err(choice.failed(status)))),
                };
                if let Some(matched) = matched {
                    this.pending = choice.complete_with_matched_stop(matched);
                }
            }
            if choice.skipped(&item) {
                continue;
            }
            // The triggering delta goes out first, as the engine's own
            // frontend delivers the step that matched; the pending `Complete`
            // follows on the next poll.
            return Poll::Ready(Some(Ok(choice.yielded(item))));
        }
    }
}
