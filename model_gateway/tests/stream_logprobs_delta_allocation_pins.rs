//! What a chat stream's text delta carrying logprobs allocates, pinned on the
//! scripted engine seam against the plain delta, on the plain chat stream and
//! under Symphony's table (the step writer's path): the logprobs add the engine
//! message's own lists (prost grows them as it decodes) and one decode per
//! token text, and the frame writer adds nothing of its own (the entries are
//! written from the message's view), so the next change that builds an
//! object per chunk, copies a byte list or collects the candidates into a
//! list fails here rather than moving the mock bed's allocation count inside
//! its noise. A separate executable, so it can declare the counting
//! allocator the pin needs.

#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a stream allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use prost::Message as _;
use smg::routers::grpc::{EngineScript, StreamBench};
use smg_grpc_client::vllm_proto as proto;

thread_local! {
    /// Allocation and reallocation calls this thread made.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The candidates per token the logprobs stream asks for and the chunks carry.
const CANDIDATES: u32 = 5;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = CALLS.with(Cell::get);
    let out = work();
    (out, CALLS.with(Cell::get) - before)
}

/// The text of `tokens` one-character tokens.
fn text(tokens: usize) -> String {
    "abcdefghij".chars().cycle().take(tokens).collect()
}

/// A chunk of the token `id`, with its logprobs (the sampled one and
/// `CANDIDATES` ranked alternatives) or without, encoded as the engine sends it.
fn chunk_bytes(id: u32, logprobs: bool) -> Vec<u8> {
    let value = |k: u32| -0.5 - f32::from(u8::try_from(k).unwrap_or(u8::MAX));
    let output_logprobs = logprobs.then(|| proto::OutputLogProbs {
        token_logprobs: vec![-0.125],
        token_ids: vec![id],
        top_logprobs: vec![proto::TopLogProbs {
            values: (0..CANDIDATES).map(value).collect(),
            token_ids: (0..CANDIDATES).map(|k| id + 1 + k).collect(),
        }],
    });
    proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                index: 0,
                token_ids: vec![id],
                completion_tokens: 1,
                output_logprobs,
                ..Default::default()
            },
        )),
    }
    .encode_to_vec()
}

/// The allocation calls one token's chunk adds to a stream, read as the
/// slope between a 32-token and a 64-token run of the same stream (the
/// stream's fixed costs cancel); each length is run once warm before the
/// counted run. Both runs go through the chat stream that asks for logprobs
/// (no tools, no reasoning split; under Symphony's table when `symphony`, the
/// step writer's path), the plain one with chunks carrying none, so the
/// difference is the logprobs' own cost.
async fn per_token(bench: &StreamBench, logprobs: bool, symphony: bool) -> usize {
    let mut counts = [0usize; 2];
    for (slot, tokens) in [32usize, 64].into_iter().enumerate() {
        let script = if logprobs {
            EngineScript::single_tokens_with_logprobs(&text(tokens), CANDIDATES)
        } else {
            EngineScript::single_tokens(&text(tokens))
        };
        let run = || async {
            if symphony {
                bench.chat_symphony_logprobs(&script).await
            } else {
                bench.chat_logprobs(&script).await
            }
        };
        let warm = run().await;
        let before = CALLS.with(Cell::get);
        let bytes = run().await;
        counts[slot] = CALLS.with(Cell::get) - before;
        assert!(
            bytes > 0 && bytes == warm,
            "the stream wrote {bytes} bytes ({warm} warm)"
        );
    }
    counts[1].saturating_sub(counts[0]) / 32
}

/// A logprobs delta's allocation calls over the plain delta's are the engine
/// message's lists (decoded by prost) and one decode per token text (the
/// token and its candidates); the one-pass writer adds none of its own: no
/// object per chunk, no byte-list copy, no candidates or entries list. The
/// margin of one covers the larger frames' flush boundaries. Under `symphony`
/// the stream runs under Symphony's table, the step writer's path.
async fn a_logprobs_delta_adds_only_the_lists_and_the_decodes(symphony: bool) {
    let path = if symphony {
        "the chat stream under Symphony's table"
    } else {
        "the plain chat stream"
    };
    let bench = StreamBench::start().await;
    let plain = per_token(&bench, false, symphony).await;
    let with_logprobs = per_token(&bench, true, symphony).await;
    let extra = with_logprobs.saturating_sub(plain);

    let id = u32::from('a');
    let (with, without) = (chunk_bytes(id, true), chunk_bytes(id, false));
    let decode = |bytes: &[u8]| proto::GenerateResponse::decode(bytes).map(drop);
    let (decoded, lists_with) = allocations_during(|| decode(&with));
    let (decoded_plain, lists_without) = allocations_during(|| decode(&without));
    assert!(
        decoded.is_ok() && decoded_plain.is_ok(),
        "the chunks decode"
    );
    let message_lists = lists_with.saturating_sub(lists_without);

    let tokenizer = bench.tokenizer();
    let texts = usize::try_from(CANDIDATES).unwrap_or(usize::MAX) + 1;
    let (decodes_ok, decodes) = allocations_during(|| {
        (0..texts).all(|k| {
            tokenizer
                .decode(&[id + u32::try_from(k).unwrap_or(u32::MAX)], false)
                .is_ok()
        })
    });
    assert!(decodes_ok, "the texts decode");

    eprintln!(
        "allocation calls per token on {path}: plain {plain}, with logprobs {with_logprobs}, extra {extra}; \
         the message's lists {message_lists}, the {texts} decodes {decodes}"
    );
    assert!(
        extra <= message_lists + decodes + 1,
        "a logprobs delta on {path} adds {extra} allocation calls per token over the plain delta; \
         the message's lists are {message_lists} and its {texts} decodes {decodes}: the writer \
         builds something of its own per chunk"
    );
}

/// The plain chat stream (no parser): the delta goes through the emission's
/// content path.
#[tokio::test]
async fn a_logprobs_delta_adds_the_message_lists_and_one_decode_per_text_and_nothing_of_its_own() {
    a_logprobs_delta_adds_only_the_lists_and_the_decodes(false).await;
}

/// The same stream under Symphony's table: the delta goes through the step
/// writer, whose first text event carries the logprobs from the view.
#[tokio::test]
async fn under_the_symphony_table_a_logprobs_delta_adds_the_same_and_nothing_of_its_own() {
    a_logprobs_delta_adds_only_the_lists_and_the_decodes(true).await;
}
