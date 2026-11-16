//! A counting global allocator for the tests that read how many times a
//! path allocated, on every thread.
//! The binary that measures declares it as its allocator
//! (`#[global_allocator] static GLOBAL: Counting = Counting;`) and reads the
//! counter it needs.

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to count what a path allocates"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

/// Allocation and reallocation calls on every thread so far: what a path
/// that crosses threads costs as a whole (a servicer serves on its own
/// runtime thread; the client and the mock engine run on the test's).
pub static CALLS_TOTAL: AtomicUsize = AtomicUsize::new(0);

/// Allocation and reallocation calls every thread has made so far.
pub fn total_calls() -> usize {
    CALLS_TOTAL.load(Ordering::Relaxed)
}

fn count_call() {
    CALLS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_call();
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
