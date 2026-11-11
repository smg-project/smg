//! What taking a request's string stops allocates, pinned at today's count,
//! so the next change that copies the list on the way to the engine fails a
//! test rather than moving a bench inside its noise. A separate executable,
//! so it can declare the counting allocator the pin needs.

#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a call allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
};

use engine_zmq_adapter::stops::take_vllm_string_stops;
use llm_tokenizer::{traits::Tokenizer, MockTokenizer};
use smg_grpc_client::vllm_proto::SamplingParams;

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

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = CALLS.with(Cell::get);
    let out = work();
    (out, CALLS.with(Cell::get) - before)
}

/// The two single-token string stops of the pinned request, as a chat
/// request with `stop: ["Hello", "world"]` reaches the dispatch stage.
const TWO_STOPS: [&str; 2] = ["Hello", "world"];

/// A request's sampling params with the two stops and no stop ids yet.
fn params_with_two_stops() -> SamplingParams {
    SamplingParams {
        stop: Vec::from(TWO_STOPS.map(String::from)),
        ..SamplingParams::default()
    }
}

/// Allocation calls of taking two single-token stops off a request beyond
/// the tokenizer's own two encodes: the first push into `stop_token_ids`,
/// and nothing for the strings themselves, which the caller keeps and the
/// encoder borrows. The tokenizer's share is measured beside the call, so
/// an allocation added to the mock's `encode` moves nothing here.
const TAKE_TWO_STOPS_OWN_CALLS: usize = 1;

#[test]
fn taking_two_single_token_stops_allocates_this_much() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    // A warm-up call pays whatever the tokenizer sets up on its first encode.
    let mut warm = params_with_two_stops();
    let _ = take_vllm_string_stops(&mut warm, Some(&tokenizer));
    let ((), encoder_calls) = allocations_during(|| {
        for stop in TWO_STOPS {
            let _ = tokenizer.encode(stop, false);
        }
    });
    let mut params = params_with_two_stops();
    let (taken, calls) =
        allocations_during(|| take_vllm_string_stops(&mut params, Some(&tokenizer)));
    eprintln!(
        "take_vllm_string_stops with two single-token stops: {calls} allocation calls, \
         {encoder_calls} of them the tokenizer's two encodes"
    );
    assert_eq!(
        taken, TWO_STOPS,
        "the taken strings come back for the router-side decoder"
    );
    assert_eq!(
        params.stop_token_ids.len(),
        2,
        "both single-token stops reach stop_token_ids"
    );
    assert_eq!(
        calls,
        encoder_calls + TAKE_TWO_STOPS_OWN_CALLS,
        "taking two single-token stops allocates this much beyond the tokenizer's encodes"
    );
}
