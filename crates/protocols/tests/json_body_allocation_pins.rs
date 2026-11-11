//! What a JSON response body allocates, pinned at today's count: the sized
//! responder (`JsonBody`) makes one allocation for the body where axum's
//! `Json` grows a 128-byte buffer three times for a body of 513 to 1,024
//! bytes. A separate executable, so it can declare the counting allocator the
//! pin needs.

#![cfg(feature = "axum")]
#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a response allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
};

use axum::{
    body::to_bytes,
    http::header::CONTENT_TYPE,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::validated::JsonBody;
use serde::Serialize;

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
static ALLOCATOR: Counting = Counting;

/// `f`'s result and the allocation calls it made on this thread.
fn allocations_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = CALLS.with(Cell::get);
    let value = f();
    (value, CALLS.with(Cell::get) - before)
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

fn completion() -> Completion {
    Completion {
        id: format!("chatcmpl-{}", "0123456789abcdef".repeat(2)),
        object: "chat.completion",
        created: 1_762_800_000,
        model: "a-served-model-name".to_owned(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant",
                content: "The quick brown fox jumps over the lazy dog. ".repeat(9),
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

#[test]
fn a_json_body_allocates_once_for_its_bytes() {
    let expected = serde_json::to_vec(&completion()).expect("a serializable body");
    assert!(
        (513..=1024).contains(&expected.len()),
        "the body is sized for axum's third growth: {} bytes",
        expected.len()
    );

    // Warm both responders once: the thread's buffer reaches the body's size.
    drop(JsonBody(completion()).into_response());
    drop(Json(completion()).into_response());

    let value = completion();
    let (response, sized) = allocations_during(|| JsonBody(&value).into_response());
    let (response_of_axum, grown) = allocations_during(|| Json(&value).into_response());
    eprintln!("allocation calls per response: JsonBody {sized}, axum Json {grown}");

    let (content_type, bytes) = body_of(response).expect("a body built from bytes");
    assert_eq!(content_type.as_deref(), Some("application/json"));
    assert_eq!(bytes, expected, "the same bytes as serde_json writes");
    let (_, bytes_of_axum) = body_of(response_of_axum).expect("a body built from bytes");
    assert_eq!(bytes_of_axum, expected, "the same bytes as axum writes");
    assert_eq!(
        sized, 4,
        "the body's one exact-size allocation and the response parts' own"
    );
    assert!(
        grown >= sized + 3,
        "axum's buffer grows three times for this body: {grown} against {sized}"
    );
}
