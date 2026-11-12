//! Recycling pools for the large per-image vision buffers.
//!
//! The preprocess pipeline allocates tens of MB per image (the [C, H, W] f32
//! tensor and the batched patch buffer are each large).
//! Freshly-allocated buffers of this size bypass the allocator's reuse paths
//! (glibc caps non-main-arena chunks at 64 MB and mmaps anything larger or
//! colder), so every image pays tens of thousands of minor page faults; the
//! fault path serializes process-wide and caps the data plane's effective
//! parallelism. Recycling keeps the pages mapped and hot.
//!
//! A lock-free thread-local pool serves same-thread take/give (preprocess
//! internals run on blocking-pool threads). The pool is capped to bound
//! residency; buffers beyond the cap are dropped. A thread keeps buffers only
//! once it has taken from the pool: a tensor's consuming write may run on a
//! thread that never preprocesses (a runtime worker), and a pool there would
//! hold the storage for the life of the thread without ever handing it out,
//! so a give on such a thread is a plain drop.

use std::cell::RefCell;

/// Max recycled buffers kept per thread per class; excess is dropped. The
/// vision path holds at most a couple of live tensors per request, so a small
/// cap captures same-thread reuse.
const MAX_THREAD_POOLED: usize = 2;

thread_local! {
    /// The thread's kept buffers: `None` until its first take, so a thread
    /// that only consumes tensors made elsewhere keeps nothing.
    static F32_LOCAL: RefCell<Option<Vec<Vec<f32>>>> = const { RefCell::new(None) };
}

macro_rules! pool_impl {
    ($take_cap:ident, $give:ident, $held:ident, $ty:ty, $local:ident) => {
        /// Take an empty `Vec` with at least `cap` capacity, reusing pooled storage.
        pub fn $take_cap(cap: usize) -> Vec<$ty> {
            let mut v = $local
                .with(|p| p.borrow_mut().get_or_insert_with(Vec::new).pop())
                .unwrap_or_default();
            v.clear();
            v.reserve(cap);
            v
        }

        /// Return a buffer for reuse by a later same-thread take. Kept only on
        /// a thread that has taken from the pool (one of its producers); on
        /// any other thread the buffer is dropped, since nothing there would
        /// ever take it.
        pub fn $give(v: Vec<$ty>) {
            if v.capacity() == 0 {
                return;
            }
            $local.with(|p| {
                let mut p = p.borrow_mut();
                if let Some(kept) = p.as_mut().filter(|kept| kept.len() < MAX_THREAD_POOLED) {
                    kept.push(v);
                }
            });
        }

        /// How many buffers this thread keeps.
        #[cfg(test)]
        fn $held() -> usize {
            $local.with(|p| p.borrow().as_ref().map_or(0, Vec::len))
        }
    };
}

pool_impl!(take_f32_cap, give_f32, held_f32, f32, F32_LOCAL);

/// Take a zero-filled `Vec<f32>` of exactly `len`, reusing pooled storage.
pub fn take_f32(len: usize) -> Vec<f32> {
    let mut v = take_f32_cap(len);
    v.resize(len, 0.0);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_producer_thread_keeps_what_it_gives_back_up_to_the_cap() {
        std::thread::spawn(|| {
            let taken = take_f32(16);
            give_f32(taken);
            assert_eq!(held_f32(), 1);
            let again = take_f32(16);
            assert_eq!(held_f32(), 0);
            give_f32(again);
            give_f32(vec![0.0; 16]);
            give_f32(vec![0.0; 16]);
            assert_eq!(held_f32(), MAX_THREAD_POOLED, "the cap holds");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn a_thread_that_never_took_keeps_nothing() {
        std::thread::spawn(|| {
            give_f32(vec![0.0; 16]);
            assert_eq!(held_f32(), 0, "a give before any take is a plain drop");
            let _ = take_f32(16);
            give_f32(vec![0.0; 16]);
            assert_eq!(held_f32(), 1, "after its first take the thread keeps");
        })
        .join()
        .unwrap();
    }
}
