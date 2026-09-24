//! Process-lifetime cache activity across all tokenizer instances.
//!
//! Fixed-size snapshots require no cache locks or entry scans. Individual fields
//! can straddle concurrent operations; clearing/dropping a cache never resets them.

use std::sync::atomic::{AtomicU64, Ordering};

pub(super) static L0: Activity = Activity::new();
pub(super) static L1: Activity = Activity::new();

pub(super) struct Activity {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    reused_bytes: AtomicU64,
}

impl Activity {
    const fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            reused_bytes: AtomicU64::new(0),
        }
    }

    pub(super) fn hit(&self, bytes: usize) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.reused_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(super) fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn evict(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
}

/// Cumulative activity for one cache layer across all tokenizer instances.
#[derive(Debug, Clone, Copy)]
pub struct CacheActivityStats {
    /// Either `l0` (whole input) or `l1` (special-token boundary prefix).
    pub layer: &'static str,
    /// Successful lookups, counted once per lookup, not once per prefix probe.
    pub hits: u64,
    /// Unsuccessful lookups, including L1 inputs without usable boundaries.
    pub misses: u64,
    /// Entries actually removed for capacity; excludes clear, drop and replacement.
    pub evictions: u64,
    /// UTF-8 input bytes served by hits (whole input for L0, matched prefix for L1).
    /// This is reuse volume, not memory usage or a measurement of CPU time saved.
    pub reused_bytes: u64,
}

/// Read process-lifetime totals, including activity before a metrics recorder exists.
/// Disabled layers produce no activity; L0 hits do not perform an L1 lookup.
pub fn cache_activity_stats() -> [CacheActivityStats; 2] {
    [("l0", &L0), ("l1", &L1)].map(|(layer, counters)| CacheActivityStats {
        layer,
        hits: counters.hits.load(Ordering::Relaxed),
        misses: counters.misses.load(Ordering::Relaxed),
        evictions: counters.evictions.load(Ordering::Relaxed),
        reused_bytes: counters.reused_bytes.load(Ordering::Relaxed),
    })
}
