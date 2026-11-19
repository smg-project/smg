//! What a typed JSON answer allocates, pinned at today's count: the body's
//! one exact-size allocation beside the response parts' own, so the next
//! change that hands the body a buffer with spare capacity, sizes a buffer
//! per answer or writes through a growing writer fails a test rather than
//! moving the mock bed's allocation count inside its noise. A separate
//! executable, so it can declare the counting allocator the pin needs.

#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what an answer allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
};

use axum::{
    body::{to_bytes, Bytes},
    http::{header::CONTENT_TYPE, HeaderValue},
    response::{IntoResponse, Response},
};
use serde::Serialize;
use smg::routers::common::json_response;

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

/// A chat completion's shape, sized like a short answer's response body.
#[derive(Serialize)]
struct Completion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
    system_fingerprint: Option<String>,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: Message,
    logprobs: Option<()>,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: String,
    refusal: Option<()>,
    tool_calls: Option<()>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// A completion whose body is `sentences` repeats of one sentence long.
fn completion(sentences: usize) -> Completion {
    Completion {
        id: format!("chatcmpl-{}", "0123456789abcdef".repeat(2)),
        object: "chat.completion",
        created: 1_762_800_000,
        model: "a-served-model-name".to_owned(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant",
                content: "The quick brown fox jumps over the lazy dog. ".repeat(sentences),
                refusal: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens: 64,
            completion_tokens: 8,
            total_tokens: 72,
        },
        system_fingerprint: None,
    }
}

/// Polls a ready future to completion without a runtime: a response body
/// built from bytes resolves on its first poll.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::yield_now();
    }
}

/// The response's content type and body bytes, or the body's own error.
fn body_of(response: Response) -> Result<(Option<String>, Vec<u8>), axum::Error> {
    let (parts, body) = response.into_parts();
    let content_type = parts
        .headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let bytes = block_on(to_bytes(body, usize::MAX))?;
    Ok((content_type, bytes.to_vec()))
}

/// The allocation calls of the response parts alone: the same content type
/// around an empty body, which allocates nothing of its own.
fn parts_allocations() -> usize {
    let (response, calls) = allocations_during(|| {
        (
            [(CONTENT_TYPE, HeaderValue::from_static("application/json"))],
            Bytes::new(),
        )
            .into_response()
    });
    drop(response);
    calls
}

#[test]
fn a_json_answer_allocates_once_for_its_body() {
    let value = completion(9);
    let expected = serde_json::to_vec(&value).expect("a serializable answer");
    assert!(
        (513..=1024).contains(&expected.len()),
        "the body is sized like a short completion's: {} bytes",
        expected.len()
    );

    // Warm the thread once: its buffer reaches the body's size.
    drop(json_response(&value));
    parts_allocations();

    let parts = parts_allocations();
    let (response, calls) = allocations_during(|| json_response(&value));
    eprintln!("allocation calls per answer: {calls} (the response parts alone: {parts})");

    let (content_type, bytes) = body_of(response).expect("a body built from bytes");
    assert_eq!(content_type.as_deref(), Some("application/json"));
    assert_eq!(bytes, expected, "the same bytes as serde_json writes");
    assert_eq!(
        calls,
        parts + 1,
        "the body's one exact-size allocation beside the response parts' own"
    );
}

#[test]
fn a_json_answer_past_the_retained_size_is_exact_and_leaves_the_thread_warm() {
    // About 90 KiB: past the capacity a thread keeps between answers.
    let large = completion(2_000);
    let expected = serde_json::to_vec(&large).expect("a serializable answer");
    assert!(expected.len() > 64 * 1024, "{} bytes", expected.len());

    let (_, bytes) = body_of(json_response(&large)).expect("a body built from bytes");
    assert_eq!(bytes, expected, "the same bytes as serde_json writes");

    // The thread's buffer went with the large body (the retention cap): the
    // next short answer grows a new one, more than the one allocation of a
    // warm thread, and the one after that is back to one. Without the cap
    // the large buffer would stay and the first count would be the warm one.
    let short = completion(9);
    let parts = parts_allocations();
    let (response, regrown) = allocations_during(|| json_response(&short));
    drop(response);
    assert!(
        regrown > parts + 1,
        "the buffer grows again after the cap dropped it: {regrown} calls, the response parts alone {parts}"
    );
    eprintln!("allocation calls for the first short answer after the cap: {regrown} (the response parts alone: {parts})");
    let (response, calls) = allocations_during(|| json_response(&short));
    drop(response);
    assert_eq!(
        calls,
        parts + 1,
        "one allocation for the body on a warm thread"
    );
}
