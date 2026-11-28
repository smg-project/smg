//! A counting global allocator for the tests that read what a path allocates
//! on the thread that runs it. The binary that measures declares it as its
//! allocator (`#[global_allocator] static GLOBAL: Counting = Counting;`) and
//! reads a count through [`allocations_during`].

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a path allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    /// Allocation and reallocation calls this thread made.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

pub struct Counting;

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

/// What `work` returns, and the allocation calls this thread made during it.
pub fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = CALLS.with(Cell::get);
    let out = work();
    (out, CALLS.with(Cell::get) - before)
}
