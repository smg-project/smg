//! Shared streaming machinery: the per-request token accounting the dialects
//! accumulate, the mapped-stream trait, the merged per-choice stream and the
//! request deadline that ends it.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use futures::{stream::SelectAll, Stream, StreamExt};
use smg_grpc_client::vllm_proto as vllm;
use tokio::time::{Instant, Sleep};
use tracing::warn;

use crate::{
    client::zmq_status, sglang::SglangGenerateStream, tokenspeed::TokenSpeedGenerateStream,
    vllm::VllmGenerateStream,
};

/// Streaming generate output over ZMQ, presented as vLLM-proto
/// `GenerateResponse`. One sub-stream per fanned-out engine request (n>1);
/// they are polled together and yield as ready (interleaved), each tagging its
/// responses with its choice `index`. The merged stream ends when every sub
/// has delivered its terminal `Complete`. Dropping it before that aborts all
/// still-running engine-side sub-requests, so no explicit abort or
/// `mark_completed` is required. Which variant is active is fixed by the
/// backend protocol chosen at connect time. A request deadline
/// ([`Self::with_deadline`]) ends a stream still waiting on the engine the
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
    Vllm(SelectAll<VllmGenerateStream>),
    /// TokenSpeed outputs.
    TokenSpeed(SelectAll<TokenSpeedGenerateStream>),
    /// SGLang outputs.
    Sglang(SelectAll<SglangGenerateStream>),
}

impl SubStreams {
    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<vllm::GenerateResponse, tonic::Status>>> {
        match self {
            Self::Vllm(streams) => streams.poll_next_unpin(cx),
            Self::TokenSpeed(streams) => streams.poll_next_unpin(cx),
            Self::Sglang(streams) => streams.poll_next_unpin(cx),
        }
    }

    /// The engine-side request ids of the subs still open.
    fn request_ids(&self) -> Vec<&str> {
        match self {
            Self::Vllm(streams) => streams.iter().map(VllmGenerateStream::request_id).collect(),
            Self::TokenSpeed(streams) => streams
                .iter()
                .map(TokenSpeedGenerateStream::request_id)
                .collect(),
            Self::Sglang(streams) => streams
                .iter()
                .map(SglangGenerateStream::request_id)
                .collect(),
        }
    }

    /// Drop every sub still open, which aborts its engine-side request.
    fn clear(&mut self) {
        match self {
            Self::Vllm(streams) => streams.clear(),
            Self::TokenSpeed(streams) => streams.clear(),
            Self::Sglang(streams) => streams.clear(),
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
    pub(crate) fn vllm(streams: SelectAll<VllmGenerateStream>) -> Self {
        Self::new(SubStreams::Vllm(streams))
    }

    pub(crate) fn tokenspeed(streams: SelectAll<TokenSpeedGenerateStream>) -> Self {
        Self::new(SubStreams::TokenSpeed(streams))
    }

    pub(crate) fn sglang(streams: SelectAll<SglangGenerateStream>) -> Self {
        Self::new(SubStreams::Sglang(streams))
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
}
