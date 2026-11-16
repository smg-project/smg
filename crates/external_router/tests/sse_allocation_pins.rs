//! What the SSE encoder allocates per frame, pinned at today's counts, so the
//! next change that adds a copy or a block per frame fails a test rather than
//! moving a bench inside its noise. A separate executable, so it can declare
//! the counting allocator the pins need.
#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]

mod common;

use bytes::Bytes;
use common::counting_alloc::{calls, Counting};
use smg_external_router::sse::SseEncoder;

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = calls();
    let out = work();
    (out, calls() - before)
}

/// A frame of a few hundred bytes, the size of a stream's delta event.
fn delta(i: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_text.delta",
        "sequence_number": i,
        "item_id": "msg_0123456789abcdef0123456789abcdef",
        "output_index": 0,
        "content_index": 0,
        "delta": "a dozen words of answer text that a model streams per step ",
    })
}

/// A frame of several blocks, the size of a stream's terminal event once the
/// answer is long.
fn terminal() -> serde_json::Value {
    serde_json::json!({"type": "response.completed", "text": "x".repeat(40 * 1024)})
}

/// Ten delta frames after the first are cut from the encoder's block: no
/// allocation per frame (the first frame's split promotes the block once,
/// the frames after it share that block until it is used up).
#[test]
fn delta_frames_are_cut_from_one_block() {
    let mut enc = SseEncoder::new();
    let deltas: Vec<serde_json::Value> = (0..=10).map(delta).collect();
    let mut frames: Vec<Bytes> = Vec::with_capacity(deltas.len());
    frames.push(enc.encode_data(&deltas[0]).expect("a frame"));
    let ((), allocations) = allocations_during(|| {
        for value in &deltas[1..] {
            frames.push(enc.encode_data(value).expect("a frame"));
        }
    });
    assert_eq!(frames.len(), 11);
    assert!(frames[0].len() > 200, "the delta is a realistic frame");
    eprintln!("allocation calls for ten delta frames after the first: {allocations}");
    assert_eq!(allocations, 0, "allocation calls for ten delta frames");
}

/// Ten delta frames after the first delta that follows a large frame, every
/// frame still held: no allocation. The first delta took a block of the large
/// frame's size; the deltas after it are cut from that block's tail (a
/// reservation keyed on the largest frame seen took a fresh block of that
/// size for every one of them).
#[test]
fn delta_frames_after_a_large_frame_are_cut_from_one_block() {
    let mut enc = SseEncoder::new();
    let _large = enc.encode_data(&terminal()).expect("a frame");
    let deltas: Vec<serde_json::Value> = (0..=10).map(delta).collect();
    let mut frames: Vec<Bytes> = Vec::with_capacity(deltas.len());
    frames.push(enc.encode_data(&deltas[0]).expect("a frame"));
    let ((), allocations) = allocations_during(|| {
        for value in &deltas[1..] {
            frames.push(enc.encode_data(value).expect("a frame"));
        }
    });
    assert_eq!(frames.len(), 11);
    eprintln!("allocation calls for ten delta frames after a large frame: {allocations}");
    assert_eq!(
        allocations, 0,
        "allocation calls for ten delta frames after a large frame"
    );
}

/// A large frame after a large frame, the first one still held, is written
/// into one block: that block and the split's shared header, nothing grown
/// or copied by doublings. The second frame wraps the first (the way a
/// stream's `content_part.done` wraps its `output_text.done`), so it is a
/// little longer than the length the reservation is keyed on: the block's
/// headroom takes the growth (a block of exactly the last frame's length
/// doubled with a copy on the first byte past it, three calls). The first
/// large frame of an encoder grows as a `Vec` does; a base encoder paid one
/// allocation and one whole-frame copy per frame on top.
#[test]
fn a_longer_frame_after_a_large_frame_takes_one_block() {
    let mut enc = SseEncoder::new();
    let text = "x".repeat(40 * 1024);
    let first = enc
        .encode_data(&serde_json::json!({"type": "response.output_text.done", "text": text}))
        .expect("a frame");
    // The value is built before the count: the pin reads the encoder's calls, not the map's.
    let wrapped = serde_json::json!({
        "type": "response.content_part.done",
        "part": {"type": "output_text", "text": text}
    });
    let (second, allocations) = allocations_during(|| enc.encode_data(&wrapped).expect("a frame"));
    assert!(
        second.len() > first.len(),
        "the second frame wraps the first"
    );
    eprintln!("allocation calls for a longer frame after a large frame: {allocations}");
    assert_eq!(
        allocations, 2,
        "allocation calls for a longer frame after a large frame"
    );
}

/// A frame after large frames that nothing holds any more (a stream paused
/// long enough for the channel to write and drop them: a tool list's done
/// event before the model's first token) takes their block back whole: no
/// allocation. The second frame, a little longer than the first, is written
/// into a fresh block of exactly the first one's length and a block more
/// (`bytes` gives a fresh block no more than asked, floored at the encoder's
/// first block), so it leaves that block a tail under a block: a reservation
/// of a block more on that tail reallocated the dead block, one call, where
/// taking it back costs none.
#[test]
fn a_frame_after_dropped_large_frames_takes_their_block_back() {
    let mut enc = SseEncoder::new();
    let text = "x".repeat(100 * 1024);
    let first = enc
        .encode_data(&serde_json::json!({"type": "response.output_text.done", "text": text}))
        .expect("a frame");
    let second = enc
        .encode_data(&serde_json::json!({
            "type": "response.content_part.done",
            "part": {"type": "output_text", "text": text}
        }))
        .expect("a frame");
    assert!(
        second.len() > first.len(),
        "the second frame wraps the first"
    );
    drop(first);
    drop(second);
    let next = delta(0);
    let (frame, allocations) = allocations_during(|| enc.encode_data(&next).expect("a frame"));
    assert!(!frame.is_empty(), "a frame");
    eprintln!("allocation calls for a frame after dropped large frames: {allocations}");
    assert_eq!(
        allocations, 0,
        "allocation calls for a frame after dropped large frames"
    );
}
