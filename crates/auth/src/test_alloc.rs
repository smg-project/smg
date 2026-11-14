//! The crate's unit tests run under a counting allocator, so an allocation
//! pin can read what a call allocated: the system allocator with a per-thread
//! count of allocation and reallocation calls (the form of the crate's api-key
//! pins executable). Test builds only.
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a call allocated; test builds only"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    /// Allocation and reallocation calls this thread made.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Allocation and reallocation calls this thread made so far.
pub(crate) fn calls() -> usize {
    CALLS.with(Cell::get)
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: the caller's layout contract, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: `ptr` came from this allocator with `layout`; the new size is the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The count moves by one for one allocation and one reallocation, and not
/// for a free: what every pin in the crate reads.
#[test]
fn the_count_follows_this_threads_allocation_calls() {
    let before = calls();
    let mut bytes: Vec<u8> = Vec::with_capacity(8);
    assert_eq!(calls() - before, 1, "one allocation");
    bytes.extend(std::iter::repeat_n(0u8, 64));
    assert_eq!(calls() - before, 2, "one reallocation on top");
    drop(bytes);
    assert_eq!(calls() - before, 2, "a free is not counted");
}
