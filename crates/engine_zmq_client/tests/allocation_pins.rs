//! What the decode of an output with logprobs allocates, pinned at today's
//! count, so the next change that adds a copy fails a test rather than
//! moving a bench inside its noise. A separate executable, so it can declare
//! the counting allocator the pin needs. `ZMQ_DECODE_COUNTED=<n>` runs the
//! decode `n` times and exits, for a counter-based reading (`perf stat`).

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]
#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]

mod common;

use std::hint::black_box;

use bytes::Bytes;
use common::counting_alloc::{calls, Counting};
use engine_zmq_client::{
    codec::encode_msgpack,
    protocol::vllm::{
        logprobs::{Logprobs, PositionLogprobs, TokenLogprob},
        output::{
            decode_engine_core_outputs, EngineCoreFinishReason, EngineCoreOutput,
            EngineCoreOutputs, RequestBatchOutputs,
        },
    },
};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = calls();
    let out = work();
    (out, calls() - before)
}

/// One engine step for one request: one new token with its logprob and five
/// ranked candidates (the sampled token first, as the engine lists them).
fn one_token_five_candidates() -> Vec<Bytes> {
    let entries = (0..6)
        .map(|rank| TokenLogprob {
            token_id: 100 + rank,
            logprob: -0.25 * (rank as f32 + 1.0),
            rank: if rank == 0 { 7 } else { rank },
        })
        .collect();
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![100],
            new_logprobs: Some(Logprobs {
                positions: vec![PositionLogprobs { entries }],
            }),
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        }],
        scheduler_stats: None,
        kv_transfer: None,
        timestamp: 0.0,
        finished_requests: None,
    });
    vec![Bytes::from(encode_msgpack(&outputs).expect("encode"))]
}

/// Decoding one output whose logprobs ride inline: the outputs vector, the
/// request id, the token ids, the three arrays' dtype strings, shape vectors
/// and raw views, the three decoded columns, the positions and the entries.
/// The three field names a failed decode would name are formatted on the
/// error path alone.
#[test]
fn decoding_one_output_with_inline_logprobs_allocates_this_much() {
    let frames = one_token_five_candidates();
    let warm = decode_engine_core_outputs(&frames).expect("decode");
    let (decoded, allocations) = allocations_during(|| decode_engine_core_outputs(&frames));
    let decoded = decoded.expect("decode");
    assert_eq!(decoded, warm);
    let outputs = &decoded.as_request_batch().expect("a batch").outputs;
    assert_eq!(outputs[0].new_logprobs.as_ref().map(Logprobs::len), Some(1));
    eprintln!("allocation calls per decode: {allocations}");
    assert_eq!(allocations, 25, "allocation calls per decode");
}

/// `ZMQ_DECODE_COUNTED=<n>`: the decode `n` times, for a counter-based
/// reading of the instructions per decode (the difference of two counts).
#[test]
fn counted_decodes() {
    let Some(count) = std::env::var("ZMQ_DECODE_COUNTED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    else {
        return;
    };
    let frames = one_token_five_candidates();
    for _ in 0..count {
        black_box(decode_engine_core_outputs(black_box(&frames)).expect("decode"));
    }
}
