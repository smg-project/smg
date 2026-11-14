//! A counting global allocator for the tests that read how many times a path
//! allocated. The binary that measures declares it as its allocator
//! (`#[global_allocator] static GLOBAL: Counting = Counting;`) and reads the
//! calls its thread made.

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a decision allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    /// Allocation and reallocation calls this thread made. Per thread, so a
    /// pin counts its own path whatever runs beside it; a constant without a
    /// destructor, so the allocator reaches it without one.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Allocation and reallocation calls the calling thread has made so far.
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}

fn count_call() {
    CALLS.with(|calls| calls.set(calls.get() + 1));
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
