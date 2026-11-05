//! Shared streaming machinery: the per-request token accounting the dialects
//! accumulate, the mapped-stream trait, the merged per-choice stream and the
//! request deadline that ends it.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use engine_zmq_client::protocol::sglang::output::{MatchedStop, SglangOutput};
use futures::{stream::SelectAll, Stream, StreamExt};
use smg_grpc_client::vllm_proto as vllm;
use tokio::time::{Instant, Sleep};
use tracing::warn;

use crate::{
    client::zmq_status, sglang::SglangGenerateStream, tokenspeed::TokenSpeedGenerateStream,
    trtllm::TrtllmGenerateStream, vllm::VllmGenerateStream,
};

/// Streaming generate output over ZMQ, presented as vLLM-proto
/// `GenerateResponse`. One sub-stream per fanned-out engine request (n>1);
/// they are polled together and yield as ready (interleaved), each tagging its
/// responses with its choice `index`. The merged stream ends when every sub
/// has delivered its terminal `Complete`. Dropping it before that aborts all
/// still-running engine-side sub-requests, so no explicit abort or
/// `mark_completed` is required. Which variant is active is fixed by the
/// backend protocol chosen at connect time. A request deadline
/// (`Self::with_deadline`) ends a stream still waiting on the engine the
/// same way from the inside.
pub struct ZmqGenerateStream {
    subs: SubStreams,
    /// The request's deadline and its timer, when the connection bounds its
    /// requests (see [`Self::with_deadline`]).
    deadline: Option<StreamDeadline>,
}

/// The merged per-choice sub-streams of one dialect.
enum SubStreams {
    /// vLLM EngineCore outputs.
    Vllm(Subs<VllmGenerateStream>),
    /// TokenSpeed outputs.
    TokenSpeed(Subs<TokenSpeedGenerateStream>),
    /// SGLang outputs.
    Sglang(Subs<SglangGenerateStream>),
    /// TensorRT-LLM outputs (the shim's slim records).
    Trtllm(Subs<TrtllmGenerateStream>),
}

/// The subs of one request: the one sub of a request without fan-out, polled
/// directly, or the `SelectAll` merge of an `n > 1` fan-out. The merge
/// re-queues a sub behind every item it yields (a task node allocated and the
/// sub moved out and back in per output), which a single sub never needs. The
/// one sub is boxed once at the start (a dialect stream is about 1 KB) so the
/// merged stream, and every wrapper holding it, stays the size it was.
enum Subs<S> {
    /// The one sub of a request without fan-out; `None` once it ended.
    One(Option<Box<S>>),
    /// The `n > 1` fan-out, polled together and yielding as ready.
    Many(SelectAll<S>),
}

impl<S: Stream + Unpin> Subs<S> {
    fn new(mut subs: Vec<S>) -> Self {
        if subs.len() == 1 {
            Self::One(subs.pop().map(Box::new))
        } else {
            Self::Many(subs.into_iter().collect())
        }
    }

    /// The next item of the subs still open; `None` once every sub ended
    /// (and on every poll after that).
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        match self {
            Self::One(sub) => {
                let Some(stream) = sub.as_mut() else {
                    return Poll::Ready(None);
                };
                let next = stream.poll_next_unpin(cx);
                if matches!(next, Poll::Ready(None)) {
                    *sub = None;
                }
                next
            }
            Self::Many(subs) => subs.poll_next_unpin(cx),
        }
    }

    /// `id` of every sub still open.
    fn ids<'a>(&'a self, id: fn(&'a S) -> &'a str) -> Vec<&'a str> {
        match self {
            Self::One(sub) => sub.as_deref().into_iter().map(id).collect(),
            Self::Many(subs) => subs.iter().map(id).collect(),
        }
    }

    /// Drop every sub still open.
    fn clear(&mut self) {
        match self {
            Self::One(sub) => *sub = None,
            Self::Many(subs) => subs.clear(),
        }
    }
}

impl SubStreams {
    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<vllm::GenerateResponse, tonic::Status>>> {
        match self {
            Self::Vllm(subs) => subs.poll_next(cx),
            Self::TokenSpeed(subs) => subs.poll_next(cx),
            Self::Sglang(subs) => subs.poll_next(cx),
            Self::Trtllm(subs) => subs.poll_next(cx),
        }
    }

    /// The engine-side request ids of the subs still open.
    fn request_ids(&self) -> Vec<&str> {
        match self {
            Self::Vllm(subs) => subs.ids(VllmGenerateStream::request_id),
            Self::TokenSpeed(subs) => subs.ids(TokenSpeedGenerateStream::request_id),
            Self::Sglang(subs) => subs.ids(SglangGenerateStream::request_id),
            Self::Trtllm(subs) => subs.ids(TrtllmGenerateStream::request_id),
        }
    }

    /// Drop every sub still open, which aborts its engine-side request.
    fn clear(&mut self) {
        match self {
            Self::Vllm(subs) => subs.clear(),
            Self::TokenSpeed(subs) => subs.clear(),
            Self::Sglang(subs) => subs.clear(),
            Self::Trtllm(subs) => subs.clear(),
        }
    }
}

/// The deadline of one generate request on the wire: the connection's request
/// timeout, counted from the request's dispatch (see
/// [`ZmqEngineClient::set_request_timeout`](crate::ZmqEngineClient::set_request_timeout)).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestDeadline {
    at: Instant,
    timeout: Duration,
}

impl RequestDeadline {
    /// The deadline `timeout` from now; one the clock cannot add never passes.
    pub(crate) fn after(timeout: Duration) -> Self {
        let now = Instant::now();
        let at = now.checked_add(timeout).unwrap_or_else(|| far_future(now));
        Self { at, timeout }
    }

    /// The status the stream ends with when the deadline passes: a
    /// `DEADLINE_EXCEEDED` of the frontend's own, which the gateway answers
    /// as it answers a worker's timeout (504, the stream's error event).
    fn status(self) -> tonic::Status {
        tonic::Status::deadline_exceeded(format!(
            "request timed out after {:?}: the engine did not finish it within the request timeout",
            self.timeout
        ))
    }
}

/// An instant the clock cannot reach: a deadline that never passes.
fn far_future(now: Instant) -> Instant {
    // tokio's timers reach about two years ahead; a longer sleep is clamped
    // there, which is as never as this needs.
    now + Duration::from_secs(365 * 24 * 60 * 60)
}

/// A stream's armed deadline; disarmed at the cut and at the stream's own
/// end.
struct StreamDeadline {
    deadline: RequestDeadline,
    sleep: Pin<Box<Sleep>>,
}

impl ZmqGenerateStream {
    pub(crate) fn vllm(subs: Vec<VllmGenerateStream>) -> Self {
        Self::new(SubStreams::Vllm(Subs::new(subs)))
    }

    pub(crate) fn tokenspeed(subs: Vec<TokenSpeedGenerateStream>) -> Self {
        Self::new(SubStreams::TokenSpeed(Subs::new(subs)))
    }

    pub(crate) fn sglang(subs: Vec<SglangGenerateStream>) -> Self {
        Self::new(SubStreams::Sglang(Subs::new(subs)))
    }

    pub(crate) fn trtllm(subs: Vec<TrtllmGenerateStream>) -> Self {
        Self::new(SubStreams::Trtllm(Subs::new(subs)))
    }

    fn new(subs: SubStreams) -> Self {
        Self {
            subs,
            deadline: None,
        }
    }

    /// Bound the stream by the request's `deadline`, when there is one: a
    /// stream still waiting on the engine at the deadline ends there (a
    /// message the engine has already delivered is read first, as under
    /// `tokio::time::timeout`), every sub still open is dropped, which aborts
    /// its engine-side request at once rather than when the reader lets go of
    /// the stream, and the reader meets the deadline's status, then the end.
    /// A stream that ended before the deadline is out of its reach. `None`
    /// leaves the stream unbounded.
    #[must_use]
    pub(crate) fn with_deadline(mut self, deadline: Option<RequestDeadline>) -> Self {
        self.deadline = deadline.map(|deadline| StreamDeadline {
            deadline,
            sleep: Box::pin(tokio::time::sleep_until(deadline.at)),
        });
        self
    }

    /// Next vLLM-proto response, or `None` when the stream ends.
    pub async fn next(&mut self) -> Option<Result<vllm::GenerateResponse, tonic::Status>> {
        futures::future::poll_fn(|cx| self.poll_next(cx)).await
    }

    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<vllm::GenerateResponse, tonic::Status>>> {
        // The engine's messages first: one already delivered is read even at
        // the deadline, and a stream whose subs all ended is out of the
        // timer's reach. Only a stream that has to wait reads the timer,
        // whose waker sees to the poll when the engine sends nothing at all.
        match self.subs.poll_next(cx) {
            Poll::Pending => {}
            ready @ Poll::Ready(_) => {
                if matches!(ready, Poll::Ready(None)) {
                    self.deadline = None;
                }
                return ready;
            }
        }
        let Some(armed) = self.deadline.as_mut() else {
            return Poll::Pending;
        };
        if armed.sleep.as_mut().poll(cx).is_pending() {
            return Poll::Pending;
        }
        let deadline = armed.deadline;
        warn!(
            request_ids = ?self.subs.request_ids(),
            timeout = ?deadline.timeout,
            "Request timed out: the engine did not finish it within the request timeout; aborting it on the engine"
        );
        // The cut: the subs go (each one's drop aborts its engine-side
        // request), the timer with them; the empty merge is the end from
        // here on.
        self.subs.clear();
        self.deadline = None;
        Poll::Ready(Some(Err(deadline.status())))
    }

    /// No-op: the ZMQ stream aborts natively on drop, so there is nothing to
    /// mark. Present for parity with the tonic abort-on-drop streams.
    #[expect(
        clippy::unused_self,
        reason = "kept for API parity with the tonic abort-on-drop streams"
    )]
    pub fn mark_completed(&mut self) {}
}

/// Accumulated per-request token counts shared by the stream mappers.
#[derive(Default)]
pub(crate) struct StreamState {
    pub(crate) output_ids: Vec<u32>,
    pub(crate) completion_tokens: u32,
    pub(crate) prompt_tokens: u32,
    pub(crate) cached_tokens: u32,
    /// Cumulative sampled-token logprobs across ticks. The proto contract is
    /// incremental per streaming chunk but cumulative on the terminal
    /// `Complete`, so accumulate here and drain into `Complete`.
    pub(crate) output_logprobs_val: Vec<f32>,
    pub(crate) output_logprobs_idx: Vec<u32>,
    /// Prompt (input) logprobs, accumulated across chunked-prefill ticks.
    pub(crate) prompt_logprobs: Vec<vllm::InputTokenLogProb>,
    pub(crate) prompt_token_ids: Vec<u32>,
    pub(crate) prompt_top_logprobs: Vec<vllm::TopLogProbs>,
    /// Cumulative per-position ranked candidates (`top_logprobs`), accumulated
    /// alongside the sampled logprobs and drained into the terminal `Complete`.
    pub(crate) output_top_logprobs: Vec<vllm::TopLogProbs>,
}

impl StreamState {
    /// Cumulative sampled-token logprobs for the terminal `Complete`, drained
    /// from the accumulated state (`None` when logprobs were not requested).
    pub(crate) fn take_complete_logprobs(&mut self) -> Option<vllm::OutputLogProbs> {
        (!self.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
            token_logprobs: std::mem::take(&mut self.output_logprobs_val),
            token_ids: std::mem::take(&mut self.output_logprobs_idx),
            top_logprobs: std::mem::take(&mut self.output_top_logprobs),
        })
    }

    /// Attach the accumulated prompt logprobs to `response`: once on the first
    /// token-bearing chunk (the proto carries them in the first chunk only;
    /// `emitted` remembers it) and on every `Complete`, including one parked in
    /// `pending`. Prefill precedes the first sampled token, so the set is whole
    /// by the time a chunk carries tokens. Nothing happens when no prompt
    /// logprobs were accumulated.
    pub(crate) fn attach_input_logprobs(
        &self,
        pending: &mut Option<vllm::GenerateResponse>,
        emitted: &mut bool,
        response: &mut vllm::GenerateResponse,
    ) {
        if self.prompt_logprobs.is_empty() {
            return;
        }
        // Built per attachment site rather than up front: with
        // `prompt_logprobs` requested this runs on every decode tick, and the
        // common tick (a later chunk) attaches nothing.
        let build = || vllm::InputLogProbs {
            token_logprobs: self.prompt_logprobs.clone(),
            token_ids: self.prompt_token_ids.clone(),
            top_logprobs: self.prompt_top_logprobs.clone(),
        };
        if let Some(vllm::generate_response::Response::Complete(parked)) = pending
            .as_mut()
            .and_then(|pending| pending.response.as_mut())
        {
            parked.input_logprobs = Some(build());
        }
        match response.response.as_mut() {
            Some(vllm::generate_response::Response::Chunk(chunk))
                if !*emitted && !chunk.token_ids.is_empty() =>
            {
                chunk.input_logprobs = Some(build());
                *emitted = true;
            }
            // Later chunks never repeat them (the proto carries them in the
            // first chunk only), and neither do token-less prefill chunks.
            Some(vllm::generate_response::Response::Chunk(_)) => {}
            Some(vllm::generate_response::Response::Complete(complete)) => {
                complete.input_logprobs = Some(build());
            }
            None => {}
        }
    }

    /// Emit one engine tick as vLLM-proto responses. On a finish tick the
    /// `Complete` (with the engine-specific finish reason and matched stop) is
    /// returned directly — unless the tick also carried new tokens, in which
    /// case a `Chunk` goes out first and the `Complete` is parked in `pending`
    /// for the next poll. Non-finish ticks emit a plain `Chunk`.
    pub(crate) fn emit_tick(
        &mut self,
        index: u32,
        token_ids: Vec<u32>,
        chunk_logprobs: Option<vllm::OutputLogProbs>,
        finish: Option<(String, Option<vllm::generate_complete::MatchedStop>)>,
        pending: &mut Option<vllm::GenerateResponse>,
    ) -> vllm::GenerateResponse {
        let chunk = |state: &Self, token_ids, chunk_logprobs| {
            vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                token_ids,
                prompt_tokens: state.prompt_tokens,
                completion_tokens: state.completion_tokens,
                cached_tokens: state.cached_tokens,
                output_logprobs: chunk_logprobs,
                index,
                ..Default::default()
            })
        };
        let response = match finish {
            Some((finish_reason, matched_stop)) => {
                let complete = vllm::GenerateResponse {
                    response: Some(vllm::generate_response::Response::Complete(
                        vllm::GenerateComplete {
                            output_ids: std::mem::take(&mut self.output_ids),
                            finish_reason,
                            prompt_tokens: self.prompt_tokens,
                            completion_tokens: self.completion_tokens,
                            cached_tokens: self.cached_tokens,
                            matched_stop,
                            output_logprobs: self.take_complete_logprobs(),
                            index,
                            ..Default::default()
                        },
                    )),
                };
                if token_ids.is_empty() {
                    return complete;
                }
                let chunk = chunk(self, token_ids, chunk_logprobs);
                *pending = Some(complete);
                chunk
            }
            None => chunk(self, token_ids, chunk_logprobs),
        };
        vllm::GenerateResponse {
            response: Some(response),
        }
    }
}

/// A slim record's matched stop in the vLLM-proto shape.
pub(crate) fn matched_stop_of(matched: MatchedStop) -> vllm::generate_complete::MatchedStop {
    match matched {
        MatchedStop::TokenId(id) => vllm::generate_complete::MatchedStop::MatchedTokenId(id),
        MatchedStop::Text(text) => vllm::generate_complete::MatchedStop::MatchedStopStr(text),
    }
}

/// One tick of the slim record, mapped the way every wire that answers with
/// it maps it: the per-request counts (reported directly, cumulative for the
/// completion), the sampled and ranked logprob columns (this tick's on the
/// chunk, accumulated for the `Complete`), the prompt-logprob columns (on
/// the prefill tick, once per request, chunked prefill may split them), then
/// the tick as a response with the prompt logprobs attached where the proto
/// carries them. `finish` is the terminal verdict the wire's module resolved
/// first (each applies its own abort policy before this; an abort never
/// reaches here).
pub(crate) fn map_slim_output(
    state: &mut StreamState,
    index: u32,
    output: SglangOutput,
    finish: Option<(String, Option<vllm::generate_complete::MatchedStop>)>,
    pending: &mut Option<vllm::GenerateResponse>,
    input_logprobs_emitted: &mut bool,
) -> vllm::GenerateResponse {
    if output.prompt_tokens > 0 {
        state.prompt_tokens = output.prompt_tokens;
    }
    if output.cached_tokens > 0 {
        state.cached_tokens = output.cached_tokens;
    }
    state.completion_tokens = output.completion_tokens;
    state.output_ids.extend(output.output_ids.iter().copied());
    if !output.input_logprobs_val.is_empty() {
        state
            .prompt_logprobs
            .extend(
                output
                    .input_logprobs_val
                    .iter()
                    .map(|lp| vllm::InputTokenLogProb {
                        value: lp.map(|lp| lp as f32),
                    }),
            );
        state
            .prompt_token_ids
            .extend(output.input_logprobs_idx.iter().copied());
        state.prompt_top_logprobs.extend(
            output
                .input_top_logprobs_val
                .iter()
                .zip(&output.input_top_logprobs_idx)
                .map(|(values, token_ids)| vllm::TopLogProbs {
                    values: values.iter().map(|&lp| lp as f32).collect(),
                    token_ids: token_ids.clone(),
                }),
        );
    }
    // Ranked candidates per position, as the engine reports them.
    let tick_top_logprobs: Vec<vllm::TopLogProbs> = output
        .output_top_logprobs_val
        .iter()
        .zip(&output.output_top_logprobs_idx)
        .map(|(values, token_ids)| vllm::TopLogProbs {
            values: values.iter().map(|&lp| lp as f32).collect(),
            token_ids: token_ids.clone(),
        })
        .collect();
    let chunk_logprobs = (!output.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
        token_logprobs: output
            .output_logprobs_val
            .iter()
            .map(|&lp| lp as f32)
            .collect(),
        token_ids: output.output_logprobs_idx.clone(),
        top_logprobs: tick_top_logprobs.clone(),
    });
    state
        .output_logprobs_val
        .extend(output.output_logprobs_val.iter().map(|&lp| lp as f32));
    state
        .output_logprobs_idx
        .extend(output.output_logprobs_idx.iter().copied());
    state.output_top_logprobs.extend(tick_top_logprobs);
    let mut response = state.emit_tick(index, output.output_ids, chunk_logprobs, finish, pending);
    state.attach_input_logprobs(pending, input_logprobs_emitted, &mut response);
    response
}

/// The per-dialect half of a ZMQ generate stream: the wire stream, the parked
/// terminal `Complete`, and the mapping from one wire output to a vLLM-proto
/// response. [`poll_mapped`] writes the `Stream` machinery once for every
/// dialect that implements this.
pub(crate) trait MappedGenerateStream {
    /// One tick of engine output on this dialect's wire.
    type Output;
    /// The wire stream carrying those ticks.
    type Inner: Stream<Item = Result<Self::Output, engine_zmq_client::Error>> + Unpin;

    fn inner(&mut self) -> &mut Self::Inner;

    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens; yielded before the wire stream is polled again.
    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse>;

    fn map_output(&mut self, output: Self::Output)
        -> Result<vllm::GenerateResponse, tonic::Status>;
}

/// `Stream::poll_next` for any [`MappedGenerateStream`]: drain the parked
/// `Complete` first, otherwise poll the wire and map the tick.
pub(crate) fn poll_mapped<S: MappedGenerateStream>(
    stream: &mut S,
    cx: &mut Context<'_>,
) -> Poll<Option<Result<vllm::GenerateResponse, tonic::Status>>> {
    if let Some(pending) = stream.pending().take() {
        return Poll::Ready(Some(Ok(pending)));
    }
    match Pin::new(stream.inner()).poll_next(cx) {
        Poll::Ready(Some(Ok(output))) => Poll::Ready(Some(stream.map_output(output))),
        Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(zmq_status(error)))),
        Poll::Ready(None) => Poll::Ready(None),
        Poll::Pending => Poll::Pending,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_zmq_client::{
        mock_engine::{EngineInbound, MockEngine, MockEngineInput},
        protocol::vllm::output::{
            EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
        },
    };
    use openai_protocol::worker::RuntimeType;
    use tokio::time::Instant;
    use tonic::Code;

    use super::*;
    use crate::{client::ZmqEngineClient, test_support::connected_pair_in};

    fn chunk(token_ids: Vec<u32>) -> vllm::GenerateResponse {
        vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Chunk(
                vllm::GenerateStreamChunk {
                    token_ids,
                    ..Default::default()
                },
            )),
        }
    }

    fn complete() -> vllm::GenerateResponse {
        vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Complete(
                vllm::GenerateComplete::default(),
            )),
        }
    }

    fn input_logprobs_of(response: &vllm::GenerateResponse) -> Option<&vllm::InputLogProbs> {
        match response.response.as_ref()? {
            vllm::generate_response::Response::Chunk(chunk) => chunk.input_logprobs.as_ref(),
            vllm::generate_response::Response::Complete(complete) => {
                complete.input_logprobs.as_ref()
            }
        }
    }

    /// The accumulated prompt logprobs ride the first token-bearing chunk once,
    /// every `Complete` (a parked one included), and nothing else.
    #[test]
    fn prompt_logprobs_attach_once_to_a_chunk_and_to_every_complete() {
        let state = StreamState {
            prompt_logprobs: vec![vllm::InputTokenLogProb::default()],
            prompt_token_ids: vec![7],
            ..Default::default()
        };
        let mut emitted = false;
        let mut pending = None;

        let mut empty_chunk = chunk(vec![]);
        state.attach_input_logprobs(&mut pending, &mut emitted, &mut empty_chunk);
        assert!(
            input_logprobs_of(&empty_chunk).is_none(),
            "a token-less chunk carries none"
        );
        assert!(!emitted);

        let mut first = chunk(vec![10]);
        state.attach_input_logprobs(&mut pending, &mut emitted, &mut first);
        assert_eq!(
            input_logprobs_of(&first).map(|l| l.token_ids.clone()),
            Some(vec![7])
        );
        assert!(emitted);

        let mut later = chunk(vec![11]);
        state.attach_input_logprobs(&mut pending, &mut emitted, &mut later);
        assert!(
            input_logprobs_of(&later).is_none(),
            "a later chunk never repeats them"
        );

        let mut pending = Some(complete());
        let mut last = complete();
        state.attach_input_logprobs(&mut pending, &mut emitted, &mut last);
        assert!(
            input_logprobs_of(&last).is_some(),
            "every Complete carries them"
        );
        assert!(
            pending.as_ref().and_then(input_logprobs_of).is_some(),
            "a parked Complete carries them too"
        );

        let silent = StreamState::default();
        let mut untouched = chunk(vec![10]);
        let mut emitted = false;
        silent.attach_input_logprobs(&mut None, &mut emitted, &mut untouched);
        assert!(
            input_logprobs_of(&untouched).is_none() && !emitted,
            "nothing accumulated, nothing attached"
        );
    }

    /// A connected vLLM-dialect client over a throwaway ipc endpoint, its
    /// requests bounded by `request_timeout` when one is given, and the mock
    /// engine behind it.
    async fn connected(
        dir: &std::path::Path,
        request_timeout: Option<Duration>,
    ) -> (ZmqEngineClient, MockEngine) {
        let (client, engine) = connected_pair_in(dir, RuntimeType::Vllm).await;
        if let Some(timeout) = request_timeout {
            client.set_request_timeout(timeout);
        }
        (client, engine)
    }

    /// One engine tick for `request_id`: a token, and the finish when `finish`
    /// is set.
    fn tick(
        request_id: &str,
        token: u32,
        finish: Option<EngineCoreFinishReason>,
    ) -> EngineCoreOutputs {
        let finished = finish.map(|_| std::collections::BTreeSet::from([request_id.to_string()]));
        EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: request_id.to_string(),
                new_token_ids: vec![token],
                finish_reason: finish,
                ..Default::default()
            }],
            finished_requests: finished,
            ..Default::default()
        })
    }

    fn request(request_id: &str, prompt: Vec<u32>) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            request_id: request_id.to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: prompt,
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(8),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        }
    }

    /// The engine's add request for `request_id`, received by the mock.
    async fn expect_add(input: &mut MockEngineInput, request_id: &str) {
        match input.recv().await.expect("inbound") {
            EngineInbound::Add(add) => assert_eq!(add.request_id, request_id),
            other => panic!("expected the add request, got {other:?}"),
        }
    }

    /// The token ids of the chunk `item` carries.
    fn chunk_tokens(item: Option<Result<vllm::GenerateResponse, tonic::Status>>) -> Vec<u32> {
        match item.expect("an item").expect("no error").response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => chunk.token_ids,
            other => panic!("expected a chunk, got {other:?}"),
        }
    }

    /// Nothing reaches the mock engine within `window`: in particular no
    /// abort for a request that ended on its own.
    async fn expect_nothing(input: &mut MockEngineInput, window: Duration) {
        if let Ok(inbound) = tokio::time::timeout(window, input.recv()).await {
            panic!("the engine received {inbound:?}, expected nothing");
        }
    }

    /// The issue's case: a stream that went silent after its first tokens
    /// (the engine stopped answering this request) ends at the request
    /// timeout counted from its dispatch, not when the client gives up. The
    /// reader meets a `DEADLINE_EXCEEDED` naming the timeout, the abort frame
    /// for the request reaches the engine at once (the reader still holds the
    /// stream), nothing follows the cut, and the connection's in-flight
    /// accounting is released: the engine's answer, had it come, would find
    /// no request.
    #[tokio::test]
    #[expect(
        clippy::print_stderr,
        reason = "the measured cut and abort times are the test's report"
    )]
    async fn a_stream_that_goes_silent_ends_at_the_request_timeout_and_is_aborted() {
        const TIMEOUT: Duration = Duration::from_millis(300);
        let dir = tempfile::tempdir().unwrap();
        let (client, engine) = connected(dir.path(), Some(TIMEOUT)).await;
        let (mut input, mut output) = engine.split();

        let dispatched = Instant::now();
        let mut stream = client
            .generate_vllm(request("r-silent", vec![1, 2, 3]))
            .await
            .expect("generate");
        expect_add(&mut input, "r-silent").await;
        // Two deltas, then the engine says nothing more about the request.
        output
            .send_outputs(&tick("r-silent", 10, None))
            .await
            .unwrap();
        output
            .send_outputs(&tick("r-silent", 11, None))
            .await
            .unwrap();
        assert_eq!(chunk_tokens(stream.next().await), vec![10]);
        assert_eq!(chunk_tokens(stream.next().await), vec![11]);

        let status = stream
            .next()
            .await
            .expect("the deadline ends the stream")
            .expect_err("with the deadline's status");
        let cut_after = dispatched.elapsed();
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert!(
            status
                .message()
                .starts_with("request timed out after 300ms"),
            "{status:?}"
        );
        assert!(
            cut_after >= TIMEOUT && cut_after < TIMEOUT * 4,
            "cut {cut_after:?} after the dispatch, timeout {TIMEOUT:?}"
        );
        // The abort goes to the engine now, while the reader still holds the
        // stream.
        let abort_after = Instant::now();
        match input.recv().await.expect("inbound") {
            EngineInbound::Abort(ids) => assert_eq!(ids, vec!["r-silent".to_string()]),
            other => panic!("expected the abort, got {other:?}"),
        }
        assert!(
            abort_after.elapsed() < Duration::from_secs(2),
            "the abort frame took {:?} after the cut",
            abort_after.elapsed()
        );
        eprintln!(
            "silent after two deltas: cut {cut_after:?} after the dispatch, the abort frame at the engine {:?} after the cut",
            abort_after.elapsed()
        );
        // Nothing after the cut, whatever the engine sends since.
        output
            .send_outputs(&tick("r-silent", 12, None))
            .await
            .unwrap();
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
        // The request is off the connection's books: nothing in flight.
        assert_eq!(client.prefill_pending_tokens(), 0);
        assert!(client.output_silence().is_none());
        drop(stream);
        expect_nothing(&mut input, Duration::from_millis(200)).await;
    }

    /// A request the engine never starts answering (queued behind others, or
    /// the engine stuck in its first step) is cut the same way, and its
    /// prompt leaves the connection's prefill accounting with it.
    #[tokio::test]
    #[expect(
        clippy::print_stderr,
        reason = "the measured cut time is the test's report"
    )]
    async fn a_request_the_engine_never_answers_is_cut_at_the_request_timeout() {
        const TIMEOUT: Duration = Duration::from_millis(300);
        let dir = tempfile::tempdir().unwrap();
        let (client, engine) = connected(dir.path(), Some(TIMEOUT)).await;
        let (mut input, _output) = engine.split();

        let dispatched = Instant::now();
        let mut stream = client
            .generate_vllm(request("r-unanswered", vec![1, 2, 3, 4]))
            .await
            .expect("generate");
        expect_add(&mut input, "r-unanswered").await;
        assert_eq!(client.prefill_pending_tokens(), 4);

        let status = stream
            .next()
            .await
            .expect("the deadline ends the stream")
            .expect_err("with the deadline's status");
        assert_eq!(status.code(), Code::DeadlineExceeded);
        let cut_after = dispatched.elapsed();
        assert!(
            cut_after >= TIMEOUT && cut_after < TIMEOUT * 4,
            "cut {cut_after:?} after the dispatch"
        );
        match input.recv().await.expect("inbound") {
            EngineInbound::Abort(ids) => assert_eq!(ids, vec!["r-unanswered".to_string()]),
            other => panic!("expected the abort, got {other:?}"),
        }
        eprintln!("never answered: cut {cut_after:?} after the dispatch, the abort frame received");
        assert!(stream.next().await.is_none());
        // The abort retires the request on the dispatcher's side as well.
        let settled = Instant::now() + Duration::from_secs(5);
        while client.prefill_pending_tokens() != 0 {
            assert!(Instant::now() < settled, "the prompt stayed pending");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(client.output_silence().is_none());
    }

    /// A stream that ends within the timeout is unchanged by it: every
    /// delta, the completion, the end, and no abort to the engine.
    #[tokio::test]
    async fn a_stream_that_finishes_within_the_request_timeout_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let (client, engine) = connected(dir.path(), Some(Duration::from_secs(5))).await;
        let (mut input, mut output) = engine.split();

        let mut stream = client
            .generate_vllm(request("r-live", vec![1, 2]))
            .await
            .expect("generate");
        expect_add(&mut input, "r-live").await;
        output
            .send_outputs(&tick("r-live", 10, None))
            .await
            .unwrap();
        output
            .send_outputs(&tick("r-live", 11, Some(EngineCoreFinishReason::Stop)))
            .await
            .unwrap();
        assert_eq!(chunk_tokens(stream.next().await), vec![10]);
        // The finish tick carried a token: its delta, then the completion.
        assert_eq!(chunk_tokens(stream.next().await), vec![11]);
        match stream.next().await.expect("complete").expect("ok").response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.finish_reason, "stop");
                assert_eq!(complete.output_ids, vec![10, 11]);
            }
            other => panic!("expected the completion, got {other:?}"),
        }
        assert!(stream.next().await.is_none());
        stream.mark_completed();
        drop(stream);
        expect_nothing(&mut input, Duration::from_millis(300)).await;
    }

    /// A pause shorter than the timeout is the engine's to take: a stream
    /// that resumes and finishes before the deadline is unchanged, and the
    /// deadline counts from the dispatch, not from the last message.
    #[tokio::test]
    async fn a_stream_that_resumes_before_the_request_timeout_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let (client, engine) = connected(dir.path(), Some(Duration::from_secs(2))).await;
        let (mut input, mut output) = engine.split();

        let mut stream = client
            .generate_vllm(request("r-paused", vec![1]))
            .await
            .expect("generate");
        expect_add(&mut input, "r-paused").await;
        output
            .send_outputs(&tick("r-paused", 10, None))
            .await
            .unwrap();
        assert_eq!(chunk_tokens(stream.next().await), vec![10]);
        // The engine pauses, then resumes well inside the deadline.
        let ((), next) = tokio::join!(
            async {
                tokio::time::sleep(Duration::from_millis(400)).await;
                output
                    .send_outputs(&tick("r-paused", 11, Some(EngineCoreFinishReason::Length)))
                    .await
                    .unwrap();
            },
            stream.next()
        );
        assert_eq!(chunk_tokens(next), vec![11]);
        match stream.next().await.expect("complete").expect("ok").response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.finish_reason, "length");
            }
            other => panic!("expected the completion, got {other:?}"),
        }
        assert!(stream.next().await.is_none());
        drop(stream);
        expect_nothing(&mut input, Duration::from_millis(300)).await;
    }

    /// A deadline that passes after the stream ended is a no-op: the reader's
    /// last poll (the one that finds the end) finds the end, not a timeout,
    /// and no abort goes to the engine for a request that finished on its own.
    #[tokio::test]
    async fn a_deadline_that_passes_after_the_stream_ended_is_a_no_op() {
        const TIMEOUT: Duration = Duration::from_millis(300);
        let dir = tempfile::tempdir().unwrap();
        let (client, engine) = connected(dir.path(), Some(TIMEOUT)).await;
        let (mut input, mut output) = engine.split();

        let mut stream = client
            .generate_vllm(request("r-ended", vec![1]))
            .await
            .expect("generate");
        expect_add(&mut input, "r-ended").await;
        output
            .send_outputs(&tick("r-ended", 10, Some(EngineCoreFinishReason::Stop)))
            .await
            .unwrap();
        assert_eq!(chunk_tokens(stream.next().await), vec![10]);
        match stream.next().await.expect("complete").expect("ok").response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.finish_reason, "stop");
            }
            other => panic!("expected the completion, got {other:?}"),
        }
        // The deadline passes before the reader asks for the end.
        tokio::time::sleep(TIMEOUT * 2).await;
        assert!(
            stream.next().await.is_none(),
            "the end of a finished stream, not a timeout"
        );
        assert!(stream.next().await.is_none());
        assert_eq!(client.prefill_pending_tokens(), 0);
        assert!(client.output_silence().is_none());
        drop(stream);
        expect_nothing(&mut input, Duration::from_millis(300)).await;
    }

    /// A connection nobody bounds (a servicer's own link) keeps its streams
    /// open: the silent stream is still waiting well past where a bound
    /// would have cut it, and ends only when the reader lets go.
    #[tokio::test]
    #[expect(
        clippy::print_stderr,
        reason = "the observed wait is the test's report"
    )]
    async fn a_connection_without_a_request_timeout_leaves_a_silent_stream_open() {
        let dir = tempfile::tempdir().unwrap();
        let (client, engine) = connected(dir.path(), None).await;
        let (mut input, mut output) = engine.split();

        let mut stream = client
            .generate_vllm(request("r-unbounded", vec![1]))
            .await
            .expect("generate");
        expect_add(&mut input, "r-unbounded").await;
        output
            .send_outputs(&tick("r-unbounded", 10, None))
            .await
            .unwrap();
        assert_eq!(chunk_tokens(stream.next().await), vec![10]);
        assert!(
            tokio::time::timeout(Duration::from_millis(600), stream.next())
                .await
                .is_err(),
            "an unbounded silent stream yields nothing"
        );
        eprintln!(
            "unbounded connection: the silent stream is still open 600ms after its last delta"
        );
        drop(stream);
        match input.recv().await.expect("inbound") {
            EngineInbound::Abort(ids) => assert_eq!(ids, vec!["r-unbounded".to_string()]),
            other => panic!("expected the drop's abort, got {other:?}"),
        }
    }

    /// An `n > 1` request is a fan-out of `n` engine requests merged into the
    /// one stream: every sub's chunks and completion arrive, each tagged with
    /// its choice index, in whatever order the engine answers; the stream ends
    /// once every sub completed and stays ended, and nothing is aborted.
    #[tokio::test]
    async fn a_fanned_out_request_merges_its_subs() {
        let dir = tempfile::tempdir().unwrap();
        let (client, engine) = connected(dir.path(), None).await;
        let (mut input, mut output) = engine.split();

        let mut req = request("r-fan", vec![1, 2]);
        req.sampling_params.as_mut().unwrap().n = 2;
        let mut stream = client.generate_vllm(req).await.expect("generate");
        expect_add(&mut input, "r-fan-0").await;
        expect_add(&mut input, "r-fan-1").await;
        // The second sub finishes first; the first one takes two ticks.
        output
            .send_outputs(&tick("r-fan-1", 21, Some(EngineCoreFinishReason::Stop)))
            .await
            .unwrap();
        output
            .send_outputs(&tick("r-fan-0", 10, None))
            .await
            .unwrap();
        output
            .send_outputs(&tick("r-fan-0", 11, Some(EngineCoreFinishReason::Stop)))
            .await
            .unwrap();
        let mut chunks = Vec::new();
        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("ok").response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    chunks.push((chunk.index, chunk.token_ids));
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    completes.push((complete.index, complete.output_ids));
                }
                other => panic!("unexpected response {other:?}"),
            }
        }
        chunks.sort();
        completes.sort();
        assert_eq!(chunks, vec![(0, vec![10]), (0, vec![11]), (1, vec![21])]);
        assert_eq!(completes, vec![(0, vec![10, 11]), (1, vec![21])]);
        assert!(stream.next().await.is_none(), "an ended stream stays ended");
        drop(stream);
        expect_nothing(&mut input, Duration::from_millis(300)).await;
    }
}
