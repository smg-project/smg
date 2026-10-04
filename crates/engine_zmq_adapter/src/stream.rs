//! Shared streaming machinery: the per-request token accounting the dialects
//! accumulate, the mapped-stream trait, and the merged per-choice stream.

use futures::{stream::SelectAll, Stream, StreamExt};
use smg_grpc_client::vllm_proto as vllm;

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
/// backend protocol chosen at connect time.
pub enum ZmqGenerateStream {
    /// vLLM EngineCore outputs.
    Vllm(SelectAll<VllmGenerateStream>),
    /// TokenSpeed outputs.
    TokenSpeed(SelectAll<TokenSpeedGenerateStream>),
    /// SGLang outputs.
    Sglang(SelectAll<SglangGenerateStream>),
}

impl ZmqGenerateStream {
    /// Next vLLM-proto response, or `None` when the stream ends.
    pub async fn next(&mut self) -> Option<Result<vllm::GenerateResponse, tonic::Status>> {
        match self {
            Self::Vllm(streams) => streams.next().await,
            Self::TokenSpeed(streams) => streams.next().await,
            Self::Sglang(streams) => streams.next().await,
        }
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
    cx: &mut std::task::Context<'_>,
) -> std::task::Poll<Option<Result<vllm::GenerateResponse, tonic::Status>>> {
    use std::task::Poll;
    if let Some(pending) = stream.pending().take() {
        return Poll::Ready(Some(Ok(pending)));
    }
    match std::pin::Pin::new(stream.inner()).poll_next(cx) {
        Poll::Ready(Some(Ok(output))) => Poll::Ready(Some(stream.map_output(output))),
        Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(zmq_status(error)))),
        Poll::Ready(None) => Poll::Ready(None),
        Poll::Pending => Poll::Pending,
    }
}
