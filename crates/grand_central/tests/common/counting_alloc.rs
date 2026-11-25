//! A counting global allocator for the tests that read how many times a path
//! allocated. The binary that measures declares it as its allocator
//! (`#[global_allocator] static GLOBAL: Counting = Counting(System);`) and
//! reads the calls its thread made. It counts over the allocator it is
//! handed: a bench hands it the one a gateway's binary registers.

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

pub struct Counting<A = System>(pub A);

unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the same layout contract as the caller's, forwarded to the allocator counted over.
        unsafe { self.0.alloc(layout) }
    }

    // Forwarded: the allocator counted over may zero for less than a fill.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the same layout contract as the caller's.
        unsafe { self.0.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { self.0.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_call();
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { self.0.realloc(ptr, layout, new_size) }
    }
}
