//! The eviction sampler the caches share: a position drawn uniformly from an
//! index of the keys, so every entry is a candidate at every round, whatever
//! its place in its map.

use std::sync::atomic::{AtomicU64, Ordering};

/// The generator a cache draws its eviction samples from (xorshift; the
/// quality a uniform sample needs, no more). It starts from one fixed seed,
/// so the draws of a cache driven the same way are the same in every process.
pub(super) struct Sampler(AtomicU64);

impl Sampler {
    pub(super) const fn new() -> Self {
        Self(AtomicU64::new(0x9E37_79B9_7F4A_7C15))
    }

    /// A position below `len`, drawn uniformly; `len` is at least one. The
    /// caller holds the lock of the index it draws from, so the state is read
    /// and written by one thread at a time.
    pub(super) fn draw(&self, len: usize) -> usize {
        let mut x = self.0.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0.store(x, Ordering::Relaxed);
        (x % len as u64) as usize
    }

    /// The position of the oldest of `samples` draws below `len` (every
    /// position when `len` is no more than `samples`), `timestamp_at` reading
    /// the entry at a position; ties go to the first draw, and an entry gone
    /// from its map reads as the oldest. `None` when `len` is zero. The one
    /// selection rule of the caches: each keeps its sample size and its own
    /// removal and accounting.
    pub(super) fn oldest(
        &self,
        len: usize,
        samples: usize,
        mut timestamp_at: impl FnMut(usize) -> u64,
    ) -> Option<usize> {
        let mut oldest: Option<(usize, u64)> = None;
        let mut consider = |position: usize| {
            let timestamp = timestamp_at(position);
            if oldest.is_none_or(|(_, seen)| timestamp < seen) {
                oldest = Some((position, timestamp));
            }
        };
        if len <= samples {
            (0..len).for_each(&mut consider);
        } else {
            (0..samples).for_each(|_| consider(self.draw(len)));
        }
        oldest.map(|(position, _)| position)
    }
}

#[cfg(test)]
mod tests {
    use super::Sampler;

    #[test]
    fn the_oldest_is_the_least_timestamp_ties_to_the_first_and_a_gone_entry_counts_as_oldest() {
        let sampler = Sampler::new();
        let stamps = [5u64, 3, 3, 9];
        assert_eq!(
            sampler.oldest(stamps.len(), 32, |at| stamps[at]),
            Some(1),
            "every position read, the first of a tie"
        );
        assert_eq!(sampler.oldest(0, 32, |_| 0), None, "nothing to draw from");
        let gone = [5u64, 0, 3, 9];
        assert_eq!(
            sampler.oldest(gone.len(), 32, |at| gone[at]),
            Some(1),
            "an entry gone from its map reads as 0, the oldest"
        );
        // More entries than samples: the draws are positions below the length, the result the
        // least of them.
        let many: Vec<u64> = (0..100).map(|k| 1000 - k).collect();
        let mut drawn = Vec::new();
        let oldest = sampler.oldest(many.len(), 8, |at| {
            drawn.push(at);
            many[at]
        });
        assert_eq!(drawn.len(), 8, "one read per draw");
        assert!(drawn.iter().all(|&at| at < many.len()));
        assert_eq!(
            oldest,
            drawn.iter().copied().min_by_key(|&at| many[at]),
            "the least of the draws"
        );
    }
}
