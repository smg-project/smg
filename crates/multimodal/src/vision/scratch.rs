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
        /// Take an empty `Vec` with at least `cap` capacity: the smallest kept
        /// buffer that holds it, or new storage when none does, so the kept
        /// buffers stay for the takes they hold.
        pub fn $take_cap(cap: usize) -> Vec<$ty> {
            let kept = $local.with(|p| {
                let mut p = p.borrow_mut();
                let kept = p.get_or_insert_with(Vec::new);
                let fit = kept
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.capacity() >= cap)
                    .min_by_key(|(_, v)| v.capacity())
                    .map(|(at, _)| at);
                fit.map(|at| kept.swap_remove(at))
            });
            match kept {
                Some(mut v) => {
                    v.clear();
                    v
                }
                None => Vec::with_capacity(cap),
            }
        }

        /// Return a buffer for reuse by a later same-thread take. Kept only on
        /// a thread that has taken from the pool (one of its producers); on
        /// any other thread the buffer is dropped, since nothing there would
        /// ever take it. A full pool gives up its smallest buffer for a
        /// larger one.
        pub fn $give(v: Vec<$ty>) {
            if v.capacity() == 0 {
                return;
            }
            $local.with(|p| {
                let mut p = p.borrow_mut();
                let Some(kept) = p.as_mut() else {
                    return;
                };
                if kept.len() < MAX_THREAD_POOLED {
                    kept.push(v);
                } else if let Some(smallest) =
                    kept.iter_mut().min_by_key(|buffer| buffer.capacity())
                {
                    if smallest.capacity() < v.capacity() {
                        *smallest = v;
                    }
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

    #[test]
    fn a_take_no_kept_buffer_holds_leaves_the_kept_buffers_kept() {
        std::thread::spawn(|| {
            let mut kept = take_f32_cap(1_000);
            kept.push(1.0);
            give_f32(kept);
            let large = take_f32_cap(2_000);
            assert!(large.capacity() >= 2_000);
            assert_eq!(held_f32(), 1, "the kept buffer waits for a take it holds");
            let small = take_f32_cap(1_000);
            assert_eq!(held_f32(), 0, "and serves it");
            assert!(small.is_empty() && small.capacity() >= 1_000);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn a_take_gets_the_smallest_kept_buffer_that_holds_it() {
        std::thread::spawn(|| {
            for (first, second) in [(1_000, 3_000), (3_000, 1_000)] {
                let (first, second) = (take_f32_cap(first), take_f32_cap(second));
                give_f32(first);
                give_f32(second);
                let small = take_f32_cap(500);
                assert!(small.capacity() < 3_000, "the large buffer is left");
                let _large = take_f32_cap(2_000);
                assert_eq!(held_f32(), 0, "for the large take, which it serves");
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn a_full_pool_gives_up_its_smallest_buffer_for_a_larger_one() {
        std::thread::spawn(|| {
            let buffers = [1_000, 2_000, 3_000, 500].map(take_f32_cap);
            for buffer in buffers {
                give_f32(buffer);
            }
            assert_eq!(held_f32(), MAX_THREAD_POOLED);
            let _large = take_f32_cap(3_000);
            assert_eq!(held_f32(), 1, "the largest buffer given back was kept");
            let _middle = take_f32_cap(2_000);
            assert_eq!(held_f32(), 0, "beside the next largest");
        })
        .join()
        .unwrap();
    }
}
