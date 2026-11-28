//! What the Responses id generator and the strict schema walk allocate, pinned
//! at today's counts, so the next change that formats an id or a path piece by
//! piece fails a test rather than moving a number inside a benchmark's noise.
//! A separate executable, so it can declare the counting allocator the pins
//! need.

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
};

use openai_protocol::{
    common::strict_schema_missing_additional_properties, responses::generate_id,
};
use serde_json::json;

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

/// An id is its prefix, an underscore and 50 lowercase hex digits, built in
/// the one string that is returned: one allocation per id.
#[test]
fn generating_an_id_allocates_once() {
    // Warm: the thread's random generator is set up by the first call, not the pin's.
    assert!(generate_id("warm").starts_with("warm_"));
    let (id, calls) = allocations_during(|| generate_id("resp"));
    eprintln!("allocation calls per generated id: {calls}");
    assert_eq!(&id[..5], "resp_");
    assert_eq!(id.len(), "resp_".len() + 50);
    assert!(
        id[5..]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "the digits after the underscore are lowercase hex: {id}"
    );
    assert_eq!(calls, 1, "one allocation per id");
}

/// A strict schema that pins every object node is the common request: the
/// walk that looks for an unpinned node allocates nothing on it.
#[test]
fn walking_a_pinned_strict_schema_allocates_nothing() {
    let schema = json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "array", "items": {"type": "integer"}},
            "place": {
                "type": "object",
                "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
                "additionalProperties": false
            },
            "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]}
        },
        "additionalProperties": false
    });
    let (missing, calls) =
        allocations_during(|| strict_schema_missing_additional_properties(&schema));
    eprintln!("allocation calls per walk of a pinned strict schema: {calls}");
    assert_eq!(missing, None);
    assert_eq!(
        calls, 0,
        "a schema that holds the rule is walked without an allocation"
    );
}
