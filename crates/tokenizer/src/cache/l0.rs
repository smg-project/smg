//! L0 Cache: Whole-string exact match cache
//!
//! This is the simplest and most effective cache layer.
//! Key: (input string, add_special_tokens) → Value: full encoding result (Arc-wrapped for zero-copy cache hits)
//!
//! Expected hit rate: 60-90% for workloads with repeated system prompts
//!
//! ## Eviction strategy: Approximate LRU
//!
//! Uses an approximate LRU strategy (sample + evict oldest) instead of arbitrary
//! eviction. This is critical for the main use case of caching system prompts:
//! - System prompts are inserted early and accessed on every request
//! - Arbitrary eviction could remove these high-value entries
//! - FIFO would be even worse: it would evict the oldest entries first, which
//!   are exactly the system prompts we want to keep
//! - Full LRU requires O(n) scanning; approximate LRU via sampling gives
//!   excellent results with O(SAMPLE_SIZE) work per eviction
//!
//! Each cache entry tracks a `last_accessed` timestamp (monotonic counter).
//! On eviction, we sample a few entries and remove the least-recently-used one.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use dashmap::DashMap;

use super::activity::L0;
use crate::traits::Encoding;

/// Number of entries to sample when looking for an eviction candidate.
/// Higher values give better LRU approximation but cost more per eviction.
/// 8 is a good balance: P(evicting an entry in the oldest 10%) ≈ 57% even
/// with just 8 samples from a 10K-entry cache.
const EVICTION_SAMPLE_SIZE: usize = 8;

/// A cached encoding entry with access tracking for approximate LRU eviction.
struct CachedEntry {
    /// The cached encoding result
    encoding: Arc<Encoding>,
    /// Monotonic timestamp of last access (for LRU eviction)
    last_accessed: AtomicU64,
}

/// L0 cache implementation using DashMap for lock-free reads.
///
/// Uses two separate maps (one per `add_special_tokens` value) so that
/// lookups can borrow the key as `&str` without allocating a `String`.
///
/// Eviction uses approximate LRU: when capacity is reached, sample a few
/// entries and evict the one with the oldest `last_accessed` timestamp.
pub struct L0Cache {
    /// Cache for encode(input, add_special_tokens = false)
    map_plain: Arc<DashMap<String, CachedEntry>>,
    /// Cache for encode(input, add_special_tokens = true)
    map_special: Arc<DashMap<String, CachedEntry>>,
    /// Maximum number of entries (across both maps) before eviction
    max_entries: usize,
    /// Cache hit counter
    hits: AtomicU64,
    /// Cache miss counter
    misses: AtomicU64,
    /// Monotonic counter for LRU timestamps
    access_counter: AtomicU64,
}

impl L0Cache {
    /// Create a new L0 cache with the specified capacity
    pub fn new(max_entries: usize) -> Self {
        let per_map = max_entries.min(1024) / 2 + 1;
        Self {
            map_plain: Arc::new(DashMap::with_capacity(per_map)),
            map_special: Arc::new(DashMap::with_capacity(per_map)),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
        }
    }

    #[inline]
    fn map_for(&self, add_special_tokens: bool) -> &DashMap<String, CachedEntry> {
        if add_special_tokens {
            &self.map_special
        } else {
            &self.map_plain
        }
    }

    /// Get the next monotonic timestamp for access tracking.
    #[inline]
    fn next_timestamp(&self) -> u64 {
        self.access_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Get an encoding from the cache (returns Arc for zero-copy access).
    /// Zero-allocation on the lookup path.
    #[inline]
    pub fn get(&self, key: &str, add_special_tokens: bool) -> Option<Arc<Encoding>> {
        match self.map_for(add_special_tokens).get(key) {
            Some(entry) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                L0.hit(key.len());
                // Update last-accessed timestamp for LRU tracking.
                // This is a single atomic store -- no contention on the map lock.
                let ts = self.next_timestamp();
                entry.value().last_accessed.store(ts, Ordering::Relaxed);
                Some(Arc::clone(&entry.value().encoding))
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                L0.miss();
                None
            }
        }
    }

    /// Evict the least-recently-used entry (approximately) if total capacity is reached.
    ///
    /// Uses approximate LRU via sampling: picks EVICTION_SAMPLE_SIZE entries from
    /// the larger map and evicts the one with the smallest (oldest) `last_accessed`
    /// timestamp. This avoids scanning all entries while still providing good LRU
    /// behavior in practice.
    fn maybe_evict(&self) {
        if self.len() >= self.max_entries {
            let victim_map = if self.map_plain.len() >= self.map_special.len() {
                &self.map_plain
            } else {
                &self.map_special
            };

            // Sample up to EVICTION_SAMPLE_SIZE entries and find the oldest.
            // Scope the iterator so all DashMap shard read-locks are released
            // before we call remove().
            let key_to_remove = {
                let mut oldest_key: Option<String> = None;
                let mut oldest_ts = u64::MAX;

                for (i, entry) in victim_map.iter().enumerate() {
                    let ts = entry.value().last_accessed.load(Ordering::Relaxed);
                    if ts < oldest_ts {
                        oldest_ts = ts;
                        oldest_key = Some(entry.key().clone());
                    }
                    if i + 1 >= EVICTION_SAMPLE_SIZE {
                        break;
                    }
                }
                oldest_key
            };

            if let Some(k) = key_to_remove {
                if victim_map.remove(&k).is_some() {
                    L0.evict();
                }
            }
        }
    }

    /// Insert an encoding into the cache
    pub fn insert(&self, key: String, add_special_tokens: bool, value: Encoding) {
        self.maybe_evict();
        let ts = self.next_timestamp();
        let entry = CachedEntry {
            encoding: Arc::new(value),
            last_accessed: AtomicU64::new(ts),
        };
        self.map_for(add_special_tokens).insert(key, entry);
    }

    /// Get the current number of entries in the cache
    pub fn len(&self) -> usize {
        self.map_plain.len() + self.map_special.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.map_plain.is_empty() && self.map_special.is_empty()
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        CacheStats {
            hits,
            misses,
            entries: self.len(),
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.map_plain.clear();
        self.map_special.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.access_counter.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use crate::{traits::Encoding, *};

    fn mock_encoding(tokens: Vec<u32>) -> Encoding {
        Encoding::Plain(tokens)
    }

    #[test]
    fn test_basic_get_set() {
        let cache = L0Cache::new(10);

        // Miss
        assert!(cache.get("hello", false).is_none());

        // Insert
        cache.insert("hello".to_string(), false, mock_encoding(vec![1, 2, 3]));

        // Hit
        let result = cache.get("hello", false);
        assert!(result.is_some());
        assert_eq!(result.unwrap().token_ids(), &[1, 2, 3]);
    }

    #[test]
    fn test_add_special_tokens_flag_separates_entries() {
        let cache = L0Cache::new(10);

        cache.insert("hello".to_string(), false, mock_encoding(vec![1, 2, 3]));
        cache.insert(
            "hello".to_string(),
            true,
            mock_encoding(vec![100, 1, 2, 3, 101]),
        );

        // Different flags should return different results
        let without = cache.get("hello", false).unwrap();
        let with = cache.get("hello", true).unwrap();
        assert_eq!(without.token_ids(), &[1, 2, 3]);
        assert_eq!(with.token_ids(), &[100, 1, 2, 3, 101]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_eviction() {
        let cache = L0Cache::new(2);

        cache.insert("a".to_string(), false, mock_encoding(vec![1]));
        cache.insert("b".to_string(), false, mock_encoding(vec![2]));

        // Should evict when adding third
        cache.insert("c".to_string(), false, mock_encoding(vec![3]));

        // Cache should have exactly 2 entries
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_eviction_across_maps() {
        let cache = L0Cache::new(2);

        // Fill up map_plain to capacity
        cache.insert("a".to_string(), false, mock_encoding(vec![1]));
        cache.insert("b".to_string(), false, mock_encoding(vec![2]));
        assert_eq!(cache.len(), 2);

        // Insert into map_special — should evict from map_plain (the larger map)
        cache.insert("c".to_string(), true, mock_encoding(vec![3]));
        assert_eq!(cache.len(), 2, "total entries must not exceed max_entries");
    }

    #[test]
    fn test_stats() {
        let cache = L0Cache::new(10);

        cache.insert("test".to_string(), false, mock_encoding(vec![1, 2, 3]));

        // 1 miss
        let _ = cache.get("missing", false);

        // 1 hit
        let _ = cache.get("test", false);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_clear() {
        let cache = L0Cache::new(10);

        cache.insert("test".to_string(), false, mock_encoding(vec![1, 2, 3]));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get("test", false).is_none());
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let cache = Arc::new(L0Cache::new(1000));
        let mut handles = vec![];

        // Spawn 10 threads
        for i in 0..10 {
            let cache_clone = cache.clone();
            handles.push(thread::spawn(move || {
                let key = format!("key_{i}");
                cache_clone.insert(key.clone(), false, mock_encoding(vec![i as u32]));

                let result = cache_clone.get(&key, false);
                assert!(result.is_some());
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_arc_reuse() {
        let cache = L0Cache::new(10);
        cache.insert("test".to_string(), false, mock_encoding(vec![1, 2, 3]));

        let arc1 = cache.get("test", false).unwrap();
        let arc2 = cache.get("test", false).unwrap();

        // Both should point to the same allocation
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }

    /// Verify that approximate LRU eviction keeps frequently-accessed entries
    /// and evicts stale ones. This simulates the system-prompt use case:
    /// a "system_prompt" entry is inserted first and accessed on every request,
    /// while one-off queries are inserted and never accessed again.
    /// Under the old arbitrary eviction, the system prompt could be evicted.
    /// Under approximate LRU, it should survive because its last_accessed
    /// timestamp is continuously refreshed by each get().
    #[test]
    fn test_lru_eviction_keeps_frequently_accessed() {
        // Small cache: capacity 4
        let cache = L0Cache::new(4);

        // Insert a "system prompt" — the high-value entry we want to keep
        cache.insert(
            "system_prompt".to_string(),
            false,
            mock_encoding(vec![10, 20, 30]),
        );

        // Insert 3 one-off queries (fills cache to capacity = 4)
        cache.insert("query_1".to_string(), false, mock_encoding(vec![1]));
        cache.insert("query_2".to_string(), false, mock_encoding(vec![2]));
        cache.insert("query_3".to_string(), false, mock_encoding(vec![3]));
        assert_eq!(cache.len(), 4);

        // Simulate realistic workload: each new request accesses the system
        // prompt (cache hit) and then inserts a new one-off query.
        // This interleaved access pattern keeps the system prompt's timestamp
        // fresh relative to all the one-off queries.
        for i in 4..12 {
            // Every request hits the system prompt first (like a real API server)
            let result = cache.get("system_prompt", false);
            assert!(
                result.is_some(),
                "system_prompt should still be in the cache after query_{} insertion",
                i - 1
            );

            // Then a new one-off query is inserted, triggering eviction
            cache.insert(format!("query_{i}"), false, mock_encoding(vec![i]));
        }

        // The system prompt should still be present — LRU protects it
        // because it was accessed more recently than the eviction victims.
        let system_prompt = cache.get("system_prompt", false);
        assert!(
            system_prompt.is_some(),
            "system_prompt should survive eviction because it was recently accessed"
        );
        assert_eq!(system_prompt.unwrap().token_ids(), &[10, 20, 30]);

        // Cache size should still be at capacity
        assert!(cache.len() <= 4);

        // The early one-off queries should all be evicted by now
        let early_queries_remaining = (1..=3)
            .filter(|i| cache.get(&format!("query_{i}"), false).is_some())
            .count();
        assert_eq!(
            early_queries_remaining, 0,
            "all early one-off queries should have been evicted"
        );
    }

    /// Verify that entries without any get() access are evicted before
    /// entries that have been accessed, even when inserted in the same order.
    #[test]
    fn test_lru_eviction_prefers_untouched_entries() {
        let cache = L0Cache::new(3);

        // Insert three entries
        cache.insert("keep_me".to_string(), false, mock_encoding(vec![1]));
        cache.insert("stale_1".to_string(), false, mock_encoding(vec![2]));
        cache.insert("stale_2".to_string(), false, mock_encoding(vec![3]));

        // Access "keep_me" to make it the most recently used
        let _ = cache.get("keep_me", false);

        // Insert a new entry, forcing eviction. The eviction should pick
        // one of the stale entries (stale_1 or stale_2) rather than keep_me.
        cache.insert("new_entry".to_string(), false, mock_encoding(vec![4]));

        assert_eq!(cache.len(), 3);

        // "keep_me" should survive because it was accessed
        assert!(
            cache.get("keep_me", false).is_some(),
            "keep_me should survive eviction because it was recently accessed"
        );

        // At least one of the stale entries should have been evicted
        let stale_remaining = ["stale_1", "stale_2"]
            .iter()
            .filter(|k| cache.get(k, false).is_some())
            .count();
        assert!(
            stale_remaining < 2,
            "at least one stale entry should have been evicted"
        );
    }
}
