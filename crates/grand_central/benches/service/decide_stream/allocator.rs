//! The bench's global allocator: the one the binary registers, with its
//! calls counted while a count row measures.

use std::{
    alloc::{GlobalAlloc, Layout},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

/// Whether the allocator counts: on for a count row's measured decisions.
pub(super) static COUNTING: AtomicBool = AtomicBool::new(false);
/// The allocation and reallocation calls of every thread while it counts.
pub(super) static CALLS: AtomicU64 = AtomicU64::new(0);

/// The allocator every row runs on: the one the binary registers
/// (`src/bin/smg-grand-central.rs`), on the targets it registers it. A
/// served decision's lists are made on one thread and freed on another,
/// and what that costs is the allocator's: a row read over another one is
/// not the binary's row.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
static SHIPPED: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The system allocator, on a target where the binary registers none.
#[cfg(any(target_env = "msvc", target_env = "musl"))]
static SHIPPED: std::alloc::System = std::alloc::System;

/// The shipped allocator, counting its calls while [`COUNTING`] is on. The
/// rows that read time pay one read of a flag nobody writes.
struct Counted;

fn count_call() {
    if COUNTING.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the caller's layout contract, forwarded to the shipped allocator.
        unsafe { SHIPPED.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the caller's layout contract, forwarded to the shipped allocator.
        unsafe { SHIPPED.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator with this layout.
        unsafe { SHIPPED.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_call();
        // SAFETY: `ptr` came from this allocator with this layout; the caller upholds `new_size`.
        unsafe { SHIPPED.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counted = Counted;

/// Opens the allocator's count.
pub(super) fn counting() {
    CALLS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
}
