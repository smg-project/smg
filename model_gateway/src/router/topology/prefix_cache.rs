// model_gateway/src/router/topology/prefix_cache.rs

use mini_moka::sync::Cache;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;

/// A high‑performance, lock‑free LRU cache mapping prefix hashes to worker IDs.
///
/// This cache uses `mini‑moka` for the forward lookup and a `DashMap` for
/// reverse indexing (worker → set of hashes). An eviction listener ensures
/// the reverse index stays in sync when entries are evicted.
///
/// Worker IDs are stored as `Arc<str>` to make cloning cheap on hot paths.
pub struct PrefixCache {
    inner: Cache<u64, Arc<str>>,
    reverse: Arc<DashMap<Arc<str>, HashSet<u64>>>,
}

impl PrefixCache {
    /// Creates a new cache with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        let reverse: Arc<DashMap<Arc<str>, HashSet<u64>>> = Arc::new(DashMap::new());
        let reverse_clone = Arc::clone(&reverse);

        let inner = Cache::builder()
            .max_capacity(capacity as u64)
            .eviction_listener(move |key: Arc<u64>, value: Arc<str>, _cause| {
                // When LRU evicts an entry, remove it from the reverse index
                if let Some(mut hashes) = reverse_clone.get_mut(&value) {
                    hashes.remove(&*key);
                }
            })
            .build();

        Self { inner, reverse }
    }

    /// Inserts a mapping from a prefix hash to a worker ID.
    ///
    /// If the hash was previously assigned to a different worker, the old
    /// reverse entry is cleaned up automatically.
    pub fn insert(&self, prefix_hash: u64, worker_id: Arc<str>) {
        // Step 1: If this hash belonged to a different worker, clean the old reverse entry
        if let Some(old_worker) = self.inner.get(&prefix_hash) {
            if &*old_worker != &*worker_id {
                if let Some(mut old_set) = self.reverse.get_mut(&old_worker) {
                    old_set.remove(&prefix_hash);
                }
            }
        }

        // Step 2: Add to new worker's reverse index
        self.reverse
            .entry(worker_id.clone())
            .or_default()
            .insert(prefix_hash);

        // Step 3: Insert into cache
        self.inner.insert(prefix_hash, worker_id);
    }

    /// Looks up the worker ID for a given prefix hash.
    pub fn lookup(&self, prefix_hash: u64) -> Option<Arc<str>> {
        self.inner.get(&prefix_hash)
    }

    /// Removes all entries belonging to a specific worker.
    pub fn remove_worker(&self, worker_id: &str) {
        let worker_id: Arc<str> = worker_id.into();

        if let Some((_, hashes)) = self.reverse.remove(&worker_id) {
            for hash in hashes {
                self.inner.invalidate(&hash);
            }
        }
    }

    /// Returns the current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.inner.entry_count() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let cache = PrefixCache::new(10);
        cache.insert(0x1234, "worker-a".into());
        assert_eq!(cache.lookup(0x1234).map(|s| s.to_string()), Some("worker-a".to_string()));
        assert_eq!(cache.lookup(0x5678), None);
    }

    #[test]
    fn remove_worker() {
        let cache = PrefixCache::new(10);
        cache.insert(0x1234, "worker-a".into());
        cache.insert(0x5678, "worker-a".into());
        cache.insert(0xABCD, "worker-b".into());

        cache.remove_worker("worker-a");
        assert_eq!(cache.lookup(0x1234), None);
        assert_eq!(cache.lookup(0x5678), None);
        assert_eq!(cache.lookup(0xABCD).map(|s| s.to_string()), Some("worker-b".to_string()));
    }

    #[test]
    fn overwrite_cleans_reverse() {
        let cache = PrefixCache::new(10);
        cache.insert(0x1234, "worker-a".into());
        cache.insert(0x1234, "worker-b".into());
        cache.remove_worker("worker-a");
        assert_eq!(cache.lookup(0x1234).map(|s| s.to_string()), Some("worker-b".to_string()));
    }

    #[test]
    fn lru_eviction() {
        let cache = PrefixCache::new(3);
        cache.insert(1, "w1".into());
        cache.insert(2, "w2".into());
        cache.insert(3, "w3".into());
        assert_eq!(cache.len(), 3);

        cache.insert(4, "w4".into());
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.lookup(1), None);
        assert_eq!(cache.lookup(2).map(|s| s.to_string()), Some("w2".to_string()));
        assert_eq!(cache.lookup(3).map(|s| s.to_string()), Some("w3".to_string()));
        assert_eq!(cache.lookup(4).map(|s| s.to_string()), Some("w4".to_string()));
    }
}