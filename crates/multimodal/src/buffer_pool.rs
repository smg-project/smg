//! Large per-request buffers kept between requests.
//!
//! An image request builds its pixels in a few buffers of megabytes: the
//! decoded frame, the patch tensor, the tensor's wire encoding. Allocated
//! zero-filled per request (`vec![0u8; n]`), each came back from the
//! allocator as fresh pages (a zeroed allocation of that size is a new
//! mapping, or pages the allocator had given back), so every request faulted
//! them in page by page and the kernel zeroed each page right before the
//! producer overwrote it: about 5,200 minor faults and 20 MB of page zeroing
//! per 1024x768 request. The producers write every byte they ask for, so
//! nothing in those buffers has to be zero: this pool keeps them. A producer
//! takes a buffer of the length it needs and fills it; the buffer comes back
//! when the request is done with it: the frame dropped
//! ([`ImageFrame`](crate::types::ImageFrame) returns its pixels), the wire
//! payload sent ([`recycled_bytes`]), the patches written in another dtype
//! ([`EncoderInput::into_bytes_as`](crate::encoder_inputs::EncoderInput::into_bytes_as)).
//!
//! Geometry: a length between [`SMALLEST`] and [`LARGEST`] is served from the
//! class of its next power of two (nine classes, 256 KiB to 64 MiB), so one
//! class serves every length up to its size and the pages past the length
//! are never touched; a length outside the range is allocated as before. The
//! pool holds at most [`BOUND`] bytes of capacity in all, process-wide; a
//! buffer offered beyond that is freed as before. Resident memory grows by
//! at most the bytes requests wrote into the kept buffers, so by at most the
//! bound: a 1024x768 request writes 2.4 MB into a 4 MiB-class frame and
//! 18.9 MB into a 32 MiB-class `float32` wire, and the bound holds seven
//! such pairs; at eight runtime threads the steady state is the in-flight
//! count's worth of them.
//!
//! One pool for the process rather than one per thread: a buffer is taken on
//! the thread that decodes or preprocesses and comes back from the thread
//! that finishes the request, so a per-thread pool would never see its
//! buffers again. The lock is held for one push or pop per buffer.
//!
//! In a debug build a buffer is handed out filled with [`SENTINEL`] rather
//! than a previous request's bytes, so a producer that left a byte unwritten
//! shows it in every test that compares its output with a reference.

use std::sync::{Mutex, MutexGuard, PoisonError};

use bytes::Bytes;
use image::DynamicImage;

/// The shortest buffer the pool serves; a shorter one is allocated as before.
/// The smallest image wire a processor emits per request, a single 336x336
/// tile in bfloat16 (677,376 bytes), sits in the 1 MiB class above this floor.
pub const SMALLEST: usize = 256 << 10;
/// The longest buffer the pool serves; a longer one is allocated as before.
pub const LARGEST: usize = 64 << 20;
/// The capacity the pool holds in all, the idle buffers of every class
/// together.
pub const BOUND: usize = 256 << 20;
/// What a buffer holds when handed out in a debug build.
pub const SENTINEL: u8 = 0xA5;

/// One class per power of two from [`SMALLEST`] to [`LARGEST`].
const CLASSES: usize = (LARGEST / SMALLEST).trailing_zeros() as usize + 1;

/// The idle buffers, by class.
pub(crate) struct BufferPool {
    inner: Mutex<Idle>,
}

struct Idle {
    /// The idle buffers of each class, each with the class's capacity.
    classes: [Vec<Vec<u8>>; CLASSES],
    /// The capacity of every idle buffer together.
    held: usize,
}

/// The process's pool.
static POOL: BufferPool = BufferPool::new();

#[cfg(test)]
thread_local! {
    /// The pool the seams use on this thread under [`with_pool`]: a test's
    /// own instance, out of reach of the other tests' traffic.
    static TEST_POOL: std::cell::Cell<Option<&'static BufferPool>> =
        const { std::cell::Cell::new(None) };
}

/// The pool the crate's seams use.
fn shared() -> &'static BufferPool {
    #[cfg(test)]
    if let Some(pool) = TEST_POOL.with(std::cell::Cell::get) {
        return pool;
    }
    &POOL
}

/// Run `work` with every seam on this thread using `pool`.
#[cfg(test)]
pub(crate) fn with_pool<T>(pool: &'static BufferPool, work: impl FnOnce() -> T) -> T {
    TEST_POOL.with(|cell| cell.set(Some(pool)));
    let result = work();
    TEST_POOL.with(|cell| cell.set(None));
    result
}

/// The class capacity a length is served with: its next power of two, or
/// `None` for a length outside the pooled range.
fn class_capacity(len: usize) -> Option<usize> {
    (SMALLEST..=LARGEST)
        .contains(&len)
        .then(|| len.next_power_of_two())
}

/// The capacity a buffer of `len` bytes will retain.
pub fn allocation_capacity(len: usize) -> usize {
    class_capacity(len).unwrap_or(len)
}

/// The class of a buffer, by its capacity: `None` unless the capacity is a
/// class's.
fn class_index(capacity: usize) -> Option<usize> {
    (capacity.is_power_of_two() && (SMALLEST..=LARGEST).contains(&capacity))
        .then(|| (capacity / SMALLEST).trailing_zeros() as usize)
}

impl BufferPool {
    pub(crate) const fn new() -> Self {
        Self {
            inner: Mutex::new(Idle {
                classes: [const { Vec::new() }; CLASSES],
                held: 0,
            }),
        }
    }

    fn idle(&self) -> MutexGuard<'_, Idle> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A buffer of `len` bytes for a producer that writes every one of them.
    /// Its bytes are a previous request's (the [`SENTINEL`] in a debug
    /// build), zeros only when it is fresh or outside the pooled range.
    pub(crate) fn take(&self, len: usize) -> Vec<u8> {
        let Some(capacity) = class_capacity(len) else {
            return vec![0u8; len];
        };
        let index = (capacity / SMALLEST).trailing_zeros() as usize;
        let kept = {
            let mut idle = self.idle();
            let kept = idle.classes[index].pop();
            if kept.is_some() {
                idle.held -= capacity;
            }
            kept
        };
        // A fresh buffer is a zeroed allocation of the whole class: the pages
        // past `len` stay untouched, the ones below it are faulted in by the
        // producer's first write, once in the buffer's life.
        let mut buf = kept.unwrap_or_else(|| vec![0u8; capacity]);
        buf.resize(len, 0);
        #[cfg(debug_assertions)]
        buf.fill(SENTINEL);
        buf
    }

    /// `buf` back for the next request: kept when it is a class's buffer and
    /// the bound has room, freed as before otherwise. Any buffer may be
    /// offered.
    pub(crate) fn recycle(&self, buf: Vec<u8>) {
        let capacity = buf.capacity();
        let Some(index) = class_index(capacity) else {
            return;
        };
        // A buffer the bound turns away is freed after the lock is released.
        let turned_away = {
            let mut idle = self.idle();
            if idle.held + capacity > BOUND {
                Some(buf)
            } else {
                idle.held += capacity;
                idle.classes[index].push(buf);
                None
            }
        };
        drop(turned_away);
    }

    /// The capacity held in all.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.idle().held
    }
}

/// [`BufferPool::take`] on the process's pool.
pub(crate) fn take(len: usize) -> Vec<u8> {
    shared().take(len)
}

/// A buffer back to the process's pool: kept for the next request of its
/// class, or freed when it is outside the classes or the pool is at its bound.
pub fn recycle(buf: Vec<u8>) {
    shared().recycle(buf);
}

/// The pixel buffer of `image` back to the pool: the eight-bit layouts own a
/// byte vector, any other layout is freed as before.
pub(crate) fn recycle_image(image: DynamicImage) {
    match image {
        DynamicImage::ImageRgb8(pixels) => recycle(pixels.into_raw()),
        DynamicImage::ImageRgba8(pixels) => recycle(pixels.into_raw()),
        DynamicImage::ImageLuma8(pixels) => recycle(pixels.into_raw()),
        DynamicImage::ImageLumaA8(pixels) => recycle(pixels.into_raw()),
        _ => {}
    }
}

/// `buf` as a payload: a `Bytes` over it whose buffer returns to the pool
/// when the last reference drops (the request encoded and sent), where
/// `Bytes::from(buf)` would free it. A buffer that is not a class's is
/// wrapped as before.
pub fn recycled_bytes(buf: Vec<u8>) -> Bytes {
    if class_index(buf.capacity()).is_none() {
        return Bytes::from(buf);
    }
    Bytes::from_owner(Recycled {
        buf,
        pool: shared(),
    })
}

/// A pooled buffer behind a [`Bytes`]: its drop is the buffer's return.
struct Recycled {
    buf: Vec<u8>,
    pool: &'static BufferPool,
}

impl AsRef<[u8]> for Recycled {
    fn as_ref(&self) -> &[u8] {
        &self.buf
    }
}

impl Drop for Recycled {
    fn drop(&mut self) {
        self.pool.recycle(std::mem::take(&mut self.buf));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_length_in_range_takes_its_class_and_one_outside_is_allocated_as_before() {
        let pool = BufferPool::new();
        let exact = pool.take(SMALLEST);
        assert_eq!((exact.len(), exact.capacity()), (SMALLEST, SMALLEST));
        let above = pool.take(SMALLEST + 1);
        assert_eq!(
            (above.len(), above.capacity()),
            (SMALLEST + 1, 2 * SMALLEST)
        );
        let largest = pool.take(LARGEST);
        assert_eq!((largest.len(), largest.capacity()), (LARGEST, LARGEST));
        let below = pool.take(SMALLEST - 1);
        assert_eq!(
            (below.len(), below.capacity()),
            (SMALLEST - 1, SMALLEST - 1)
        );
        assert!(
            below.iter().all(|&byte| byte == 0),
            "allocated as before: zeroed"
        );
        let beyond = pool.take(LARGEST + 1);
        assert_eq!(
            (beyond.len(), beyond.capacity()),
            (LARGEST + 1, LARGEST + 1)
        );
        assert!(pool.take(0).is_empty());
    }

    #[test]
    fn the_smallest_image_wire_is_served_from_the_pool() {
        // A single 336x336 tile on the bfloat16 wire, the smallest image wire
        // a processor emits per request; below the former 1 MiB floor it was
        // allocated zero-filled on every request.
        let wire = 3 * 336 * 336 * 2;
        assert_eq!(class_capacity(wire), Some(1 << 20));
        let pool = BufferPool::new();
        let first = pool.take(wire);
        let address = first.as_ptr();
        pool.recycle(first);
        let again = pool.take(wire);
        assert_eq!(again.as_ptr(), address);
    }

    #[test]
    fn a_recycled_buffer_serves_the_next_take_of_its_class_at_the_new_length() {
        let pool = BufferPool::new();
        let first = pool.take(3 * SMALLEST / 2);
        let address = first.as_ptr();
        pool.recycle(first);
        assert_eq!(pool.held(), 2 * SMALLEST);
        // A longer length of the same class: the same buffer, resized.
        let second = pool.take(2 * SMALLEST - 7);
        assert_eq!(second.as_ptr(), address);
        assert_eq!(second.len(), 2 * SMALLEST - 7);
        assert_eq!(pool.held(), 0);
        // Another class misses it.
        pool.recycle(second);
        let other = pool.take(SMALLEST);
        assert_ne!(other.as_ptr(), address);
        assert_eq!(pool.held(), 2 * SMALLEST);
    }

    #[test]
    fn a_debug_build_hands_a_buffer_out_filled_with_the_sentinel() {
        let pool = BufferPool::new();
        let mut buf = pool.take(SMALLEST);
        buf.fill(0x17);
        pool.recycle(buf);
        let again = pool.take(SMALLEST);
        if cfg!(debug_assertions) {
            assert!(again.iter().all(|&byte| byte == SENTINEL));
        } else {
            assert!(again.iter().all(|&byte| byte == 0x17));
        }
    }

    #[test]
    fn a_buffer_of_another_capacity_is_freed_as_before() {
        let pool = BufferPool::new();
        pool.recycle(Vec::with_capacity(SMALLEST + 1));
        pool.recycle(Vec::with_capacity(SMALLEST / 2));
        pool.recycle(Vec::with_capacity(2 * LARGEST));
        assert_eq!(pool.held(), 0);
    }

    #[test]
    fn the_bound_turns_a_buffer_away() {
        let pool = BufferPool::new();
        for _ in 0..BOUND / LARGEST {
            pool.recycle(Vec::with_capacity(LARGEST));
        }
        assert_eq!(pool.held(), BOUND);
        pool.recycle(Vec::with_capacity(SMALLEST));
        assert_eq!(pool.held(), BOUND);
        drop(pool.take(LARGEST));
        assert_eq!(pool.held(), BOUND - LARGEST);
        pool.recycle(Vec::with_capacity(SMALLEST));
        assert_eq!(pool.held(), BOUND - LARGEST + SMALLEST);
    }

    #[test]
    fn a_payload_returns_its_buffer_when_the_last_reference_drops() {
        static MINE: BufferPool = BufferPool::new();
        with_pool(&MINE, || {
            let buf = take(SMALLEST + 3);
            let address = buf.as_ptr();
            let payload = recycled_bytes(buf);
            assert_eq!(payload.len(), SMALLEST + 3);
            let copy = payload.clone();
            drop(payload);
            assert_eq!(MINE.held(), 0, "a reference is still out");
            assert_eq!(copy.as_ptr(), address, "the payload is the buffer itself");
            drop(copy);
            assert_eq!(MINE.held(), 2 * SMALLEST);
            let again = take(SMALLEST + 3);
            assert_eq!(again.as_ptr(), address);
        });
    }

    #[test]
    fn a_payload_over_a_buffer_of_another_capacity_is_a_plain_bytes() {
        static MINE: BufferPool = BufferPool::new();
        with_pool(&MINE, || {
            drop(recycled_bytes(vec![1, 2, 3]));
            drop(recycled_bytes(Vec::with_capacity(SMALLEST + 1)));
            assert_eq!(MINE.held(), 0);
        });
    }

    #[test]
    fn an_eight_bit_image_returns_its_pixels_and_a_wider_one_is_freed_as_before() {
        static MINE: BufferPool = BufferPool::new();
        with_pool(&MINE, || {
            let pixels = take(1024 * 1024 * 3);
            let address = pixels.as_ptr();
            let rgb = image::RgbImage::from_raw(1024, 1024, pixels).expect("sized");
            recycle_image(DynamicImage::ImageRgb8(rgb));
            // The 4 MiB class of a 3 MiB buffer, whatever the floor.
            assert_eq!(MINE.held(), 4 << 20);
            let again = take(1024 * 1024 * 3);
            assert_eq!(again.as_ptr(), address);
            let wide = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(1024, 1024);
            recycle_image(DynamicImage::ImageRgb16(wide));
            assert_eq!(MINE.held(), 0);
        });
    }
}
