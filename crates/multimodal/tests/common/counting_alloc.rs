//! A counting global allocator for the pins that read what a path allocates.
//! The binary that measures declares it as its allocator
//! (`#[global_allocator] static GLOBAL: Counting = Counting;`) and reads the
//! counters it needs.

#![expect(unsafe_code, reason = "a global allocator that counts allocations")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

/// An allocation at or above this size is a buffer of the image's own scale (a decoded
/// frame, a resized plane, a wire), not bookkeeping.
pub const LARGE: usize = 256 * 1024;

/// Allocation and reallocation calls so far.
pub static CALLS: AtomicUsize = AtomicUsize::new(0);

/// Bytes requested so far; a reallocation counts its new size.
pub static BYTES: AtomicUsize = AtomicUsize::new(0);

/// Calls of [`LARGE`] bytes or more so far.
pub static LARGE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// The allocator: the system allocator, counted.
pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        if layout.size() >= LARGE {
            LARGE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        if new_size >= LARGE {
            LARGE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
