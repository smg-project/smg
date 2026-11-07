//! Recycling pools for the large per-image vision buffers.
//!
//! The preprocess pipeline allocates tens of MB per image (the [C, H, W] f32
//! tensor, the batched patch buffer and the Pillow-exact resample's
//! intermediate rows are each large).
//! Freshly-allocated buffers of this size bypass the allocator's reuse paths
//! (glibc caps non-main-arena chunks at 64 MB and mmaps anything larger or
//! colder), so every image pays tens of thousands of minor page faults; the
//! fault path serializes process-wide and caps the data plane's effective
//! parallelism. Recycling keeps the pages mapped and hot.
//!
//! A lock-free thread-local pool serves same-thread take/give. Preprocess
//! internals run on blocking-pool threads, which exit when idle, and on the
//! crate's preprocessing pool, whose workers live as long as the process, so
//! what a thread keeps is what the process keeps: at most
//! [`MAX_THREAD_POOLED`] buffers per element type, each as large as the
//! largest request it served; the byte pool drops a buffer an image needed
//! less than half of, so an outlier image does not pin its size on a thread.

use std::cell::RefCell;

/// Max recycled buffers kept per thread per element type; excess is dropped.
/// The vision path holds at most a couple of live buffers of a type per
/// request, so a small cap captures same-thread reuse.
const MAX_THREAD_POOLED: usize = 2;

/// A thread's stack of recycled buffers of one element type.
struct Pool<T>(RefCell<Vec<Vec<T>>>);

impl<T> Pool<T> {
    const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    /// The buffer given last, its length and contents as given, or a new
    /// empty one.
    fn take(&self) -> Vec<T> {
        self.0.borrow_mut().pop().unwrap_or_default()
    }

    /// Keep `v` for a later same-thread take; beyond the cap it is dropped.
    fn give(&self, v: Vec<T>) {
        if v.capacity() == 0 {
            return;
        }
        let mut pooled = self.0.borrow_mut();
        if pooled.len() < MAX_THREAD_POOLED {
            pooled.push(v);
        }
    }
}

thread_local! {
    static F32_LOCAL: Pool<f32> = const { Pool::new() };
    static U8_LOCAL: Pool<u8> = const { Pool::new() };
}

/// Take an empty `Vec<f32>` with at least `cap` capacity, reusing pooled storage.
pub fn take_f32_cap(cap: usize) -> Vec<f32> {
    let mut v = F32_LOCAL.with(Pool::take);
    v.clear();
    v.reserve(cap);
    v
}

/// Return an `f32` buffer for reuse by a later same-thread take.
pub fn give_f32(v: Vec<f32>) {
    F32_LOCAL.with(|pool| pool.give(v));
}

/// Take a zero-filled `Vec<f32>` of exactly `len`, reusing pooled storage.
pub fn take_f32(len: usize) -> Vec<f32> {
    let mut v = take_f32_cap(len);
    v.resize(len, 0.0);
    v
}

/// A buffer of `len` bytes for a kernel that writes every one of them: a plain
/// allocation filled in place, never a zeroed allocation. The allocators
/// serve a large zeroed block by handing its pages back to the kernel and
/// taking them again page fault by page fault (jemalloc zeroes a recycled
/// block that way; glibc's zeroed allocation of a fresh heap top relies on
/// fresh pages), which costs more than the kernel's own pass on a buffer of
/// the image's scale; a plain allocation reuses the pages the allocator kept.
#[expect(
    clippy::slow_vector_initialization,
    reason = "the zero-filled `vec!` is the calloc this helper exists to avoid; see its doc comment"
)]
pub(crate) fn plain_buffer(len: usize) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(len);
    buffer.resize(len, 0);
    buffer
}

/// [`plain_buffer`] for a kernel that writes `f32` values.
pub(crate) fn plain_f32(len: usize) -> Vec<f32> {
    let mut buffer = Vec::with_capacity(len);
    buffer.resize(len, 0.0);
    buffer
}

/// Take a `Vec<u8>` of exactly `len` bytes whose contents are unspecified (a
/// recycled buffer keeps the bytes it was given back with, a new one is a
/// [`plain_buffer`]), reusing pooled storage: for a writer that fills every byte
/// before any is read, which spares a fill the size of the buffer. A pooled
/// buffer shorter than `len` is dropped for a fresh one: growing it would
/// copy its stale bytes and round its capacity up to twice the old size,
/// where a fresh buffer is exactly `len`, and a capacity is then always a
/// length some image was served at.
pub fn take_u8_unspecified(len: usize) -> Vec<u8> {
    let mut v = U8_LOCAL.with(Pool::take);
    if v.capacity() < len {
        return plain_buffer(len);
    }
    v.resize(len, 0);
    v
}

/// Return a byte buffer for reuse by a later same-thread take. A buffer
/// whose capacity is more than twice the length it comes back with is
/// dropped instead: the capacity is a length an earlier image was served at
/// (takes never grow a buffer), so the thread keeps the size of the images
/// it serves, not of the largest one it ever saw.
pub fn give_u8(v: Vec<u8>) {
    if v.capacity() <= 2 * v.len() {
        U8_LOCAL.with(|pool| pool.give(v));
    }
}

#[cfg(test)]
mod tests {
    use super::{give_u8, take_u8_unspecified, MAX_THREAD_POOLED};

    /// A byte buffer given back comes back on the same thread with its bytes
    /// and the asked length, newest first; a fresh one is zeroed; a buffer
    /// given beyond the cap is dropped, and so is one the last image needed
    /// less than half of; a request longer than the pooled buffer takes a
    /// fresh exact one, which later shorter requests recycle.
    #[test]
    fn a_given_byte_buffer_comes_back_with_its_bytes() {
        let mut first = take_u8_unspecified(8);
        first.fill(7);
        let storage = first.as_ptr();
        give_u8(first);
        let again = take_u8_unspecified(4);
        assert_eq!(again.as_ptr(), storage, "the given buffer is the one taken");
        assert_eq!(again, [7; 4], "a recycled buffer keeps its bytes");
        let mut fresh = take_u8_unspecified(16);
        assert_eq!(fresh, [0; 16], "a new buffer is zeroed");
        fresh.fill(9);
        give_u8(fresh);
        give_u8(again);
        give_u8(vec![1_u8; 4]);
        let kept: Vec<Vec<u8>> = (0..MAX_THREAD_POOLED)
            .map(|_| take_u8_unspecified(1))
            .collect();
        assert_eq!(kept, [[7], [9]], "the kept buffers come back newest first");
        assert_eq!(
            take_u8_unspecified(1),
            [0],
            "the buffer given beyond the cap was dropped: the pool is empty again"
        );
        let mut outlier = take_u8_unspecified(64);
        outlier.fill(5);
        outlier.truncate(16);
        give_u8(outlier);
        assert_eq!(
            take_u8_unspecified(16),
            [0; 16],
            "a buffer the last image needed less than half of is dropped, not kept"
        );
        let mut steady = take_u8_unspecified(64);
        steady.fill(6);
        steady.truncate(40);
        give_u8(steady);
        assert_eq!(
            take_u8_unspecified(40),
            [6; 40],
            "a buffer the last image needed more than half of is kept"
        );
        give_u8(take_u8_unspecified(64));
        let grown = take_u8_unspecified(70);
        assert_eq!(
            grown.capacity(),
            70,
            "a request longer than the pooled buffer takes a fresh one of exactly its length"
        );
        give_u8(grown);
        assert_eq!(
            take_u8_unspecified(60).capacity(),
            70,
            "the buffer served at 70 is recycled for 60: no growth inflated its capacity"
        );
    }
}
