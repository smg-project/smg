//! A counting global allocator for the tests that read what a thread's
//! cache kept, what an encode allocated, or how many times a path allocated.
//! The binary that measures declares it as its allocator
//! (`#[global_allocator] static GLOBAL: Counting = Counting;`) and reads the
//! counter it needs.

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a thread kept or an encode allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Bytes currently allocated through the global allocator.
pub static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Bytes requested from the global allocator so far.
pub static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Allocation and reallocation calls this thread made. Per thread, so a
    /// pin counts its own path whatever runs beside it; a constant without a
    /// destructor, so the allocator reaches it without one.
    static CALLS: Cell<usize> = const { Cell::new(0) };
    /// Bytes this thread requested, counted the same way.
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Allocation and reallocation calls the calling thread has made so far.
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}

/// Bytes the calling thread has requested so far (the growth of a
/// reallocation counts once).
pub fn bytes() -> usize {
    BYTES.with(Cell::get)
}

fn count_call(bytes: usize) {
    CALLS.with(|calls| calls.set(calls.get() + 1));
    BYTES.with(|total| total.set(total.get() + bytes));
}

pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_call(layout.size());
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let growth = new_size.saturating_sub(layout.size());
        count_call(growth);
        ALLOCATED.fetch_add(growth, Ordering::Relaxed);
        if new_size >= layout.size() {
            LIVE.fetch_add(new_size - layout.size(), Ordering::Relaxed);
        } else {
            LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
