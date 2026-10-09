//! L0: the exact-match cache from an input text to its token ids.
//!
//! An entry is the text's plain ids (4 bytes per token; nothing else the
//! encoding carried is kept), so the cache's memory is the texts, the ids
//! and a fixed per-entry overhead, and it is bounded two ways: by entries
//! and by bytes. An input that would take more than a quarter of the byte
//! budget on its own is not cached.
//!
//! Eviction is approximate LRU by sampling: [`EVICTION_SAMPLE_SIZE`] entries
//! are drawn uniformly from an index of all keys and the least recently used
//! of them goes. Uniform sampling is what gives the approximation its
//! guarantee (the chance that an eviction hits the oldest tenth of the cache
//! is 1 - 0.9^8, about 57 %), and what keeps a hot set that recurs every few
//! seconds resident under a stream of cold inserts: a recently touched entry
//! is practically never the oldest of eight random ones.
//!
//! Reads take no lock beyond the map's shard; inserts and evictions
//! serialize on the index, which they also keep consistent with the maps.

use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};

use dashmap::DashMap;

use super::activity::L0;
use crate::traits::Encoding;

/// Entries sampled per eviction; the oldest of them is removed.
const EVICTION_SAMPLE_SIZE: usize = 8;
/// What an entry costs beyond its text and ids: the map slot, the shared
/// key, the value and its timestamp, the index entry.
const ENTRY_OVERHEAD_BYTES: usize = 128;
/// Byte budget when a caller gives none (`L0Cache::new`).
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

struct CachedEntry {
    encoding: Arc<Encoding>,
    last_accessed: AtomicU64,
    /// The bytes this entry counts against the budget.
    bytes: usize,
}

/// One key in the eviction index, with the map it lives in.
struct IndexEntry {
    key: Arc<str>,
    special: bool,
}

/// L0 cache implementation using DashMap for lock-free reads.
pub struct L0Cache {
    /// Entries encoded without special tokens.
    map_plain: DashMap<Arc<str>, CachedEntry>,
    /// Entries encoded with special tokens.
    map_special: DashMap<Arc<str>, CachedEntry>,
    /// Every key of both maps, for uniform eviction sampling. Also the lock
    /// that serializes inserts and evictions.
    index: Mutex<Vec<IndexEntry>>,
    max_entries: usize,
    max_bytes: usize,
    bytes: AtomicUsize,
    entries: AtomicUsize,
    hits: AtomicU64,
    misses: AtomicU64,
    access_counter: AtomicU64,
    /// State of the sampler's generator.
    sample_state: AtomicU64,
}

impl L0Cache {
    /// A cache of at most `max_entries` entries and [`DEFAULT_MAX_BYTES`].
    pub fn new(max_entries: usize) -> Self {
        Self::with_limits(max_entries, DEFAULT_MAX_BYTES)
    }

    /// A cache of at most `max_entries` entries and `max_bytes` bytes (texts,
    /// ids and per-entry overhead).
    pub fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        let per_map = max_entries.min(1024) / 2 + 1;
        Self {
            map_plain: DashMap::with_capacity(per_map),
            map_special: DashMap::with_capacity(per_map),
            index: Mutex::new(Vec::with_capacity(max_entries.min(1024))),
            max_entries,
            max_bytes,
            bytes: AtomicUsize::new(0),
            entries: AtomicUsize::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
            sample_state: AtomicU64::new(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Get the appropriate map based on add_special_tokens flag
    #[inline]
    fn map_for(&self, add_special_tokens: bool) -> &DashMap<Arc<str>, CachedEntry> {
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

    /// Look up a cached encoding for the given input text and
    /// add_special_tokens flag. Returns an Arc clone (cheap) if found.
    #[inline]
    pub fn get(&self, key: &str, add_special_tokens: bool) -> Option<Arc<Encoding>> {
        match self.map_for(add_special_tokens).get(key) {
            Some(entry) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                L0.hit(key.len());
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

    /// The next pseudo-random number of the sampler (xorshift; the quality a
    /// uniform sample needs, no more).
    fn random(&self) -> u64 {
        let mut x = self.sample_state.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.sample_state.store(x, Ordering::Relaxed);
        x
    }

    /// Removes the least recently used of [`EVICTION_SAMPLE_SIZE`] entries
    /// drawn uniformly from the index (every entry when the cache holds no
    /// more than that). The caller holds the index lock.
    fn evict_one(&self, index: &mut Vec<IndexEntry>) {
        if index.is_empty() {
            return;
        }
        let mut oldest: Option<(usize, u64)> = None;
        let mut consider = |cache: &Self, position: usize| {
            let entry = &index[position];
            let timestamp = cache
                .map_for(entry.special)
                .get(&*entry.key)
                .map_or(0, |found| found.last_accessed.load(Ordering::Relaxed));
            if oldest.is_none_or(|(_, ts)| timestamp < ts) {
                oldest = Some((position, timestamp));
            }
        };
        if index.len() <= EVICTION_SAMPLE_SIZE {
            for position in 0..index.len() {
                consider(self, position);
            }
        } else {
            for _ in 0..EVICTION_SAMPLE_SIZE {
                let position = (self.random() % index.len() as u64) as usize;
                consider(self, position);
            }
        }
        let Some((position, _)) = oldest else {
            return;
        };
        let victim = index.swap_remove(position);
        if let Some((_, removed)) = self.map_for(victim.special).remove(&*victim.key) {
            self.bytes.fetch_sub(removed.bytes, Ordering::Relaxed);
            self.entries.fetch_sub(1, Ordering::Relaxed);
            L0.evict();
        }
    }

    /// Insert an encoding into the cache. Only the ids are kept. An input
    /// whose entry would exceed a quarter of the byte budget is not cached.
    pub fn insert(&self, key: String, add_special_tokens: bool, value: Encoding) {
        let ids = match value {
            Encoding::Plain(ids) => ids,
            other => other.token_ids().to_vec(),
        };
        let bytes = key.len() + ids.len() * 4 + ENTRY_OVERHEAD_BYTES;
        if bytes > self.max_bytes / 4 {
            return;
        }
        let key: Arc<str> = key.into();
        let entry = CachedEntry {
            encoding: Arc::new(Encoding::Plain(ids)),
            last_accessed: AtomicU64::new(self.next_timestamp()),
            bytes,
        };
        let mut index = self
            .index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let map = self.map_for(add_special_tokens);
        let replacing = map.get(&*key).map(|existing| existing.bytes);
        let incoming = bytes.saturating_sub(replacing.unwrap_or(0));
        let new_entry = replacing.is_none();
        while !index.is_empty()
            && ((new_entry && self.entries.load(Ordering::Relaxed) >= self.max_entries)
                || self.bytes.load(Ordering::Relaxed) + incoming > self.max_bytes)
        {
            self.evict_one(&mut index);
        }
        match map.insert(Arc::clone(&key), entry) {
            Some(old) => {
                self.bytes.fetch_sub(old.bytes, Ordering::Relaxed);
            }
            None => {
                index.push(IndexEntry {
                    key,
                    special: add_special_tokens,
                });
                self.entries.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.load(Ordering::Relaxed)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes the entries count against the budget.
    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
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

    /// Clear all entries
    pub fn clear(&self) {
        let mut index = self
            .index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.map_plain.clear();
        self.map_special.clear();
        index.clear();
        self.bytes.store(0, Ordering::Relaxed);
        self.entries.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.access_counter.store(0, Ordering::Relaxed);
    }
}

/// Cache statistics
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

    #[test]
    fn entries_hold_plain_ids_whatever_came_in() {
        let cache = L0Cache::new(10);
        cache.insert("t".to_string(), false, Encoding::Tiktoken(vec![7, 8, 9]));
        let got = cache.get("t", false).unwrap();
        assert!(matches!(*got, Encoding::Plain(_)));
        assert_eq!(got.token_ids(), &[7, 8, 9]);
    }

    #[test]
    fn the_byte_bound_holds() {
        // Each entry: 4-byte key + 100 ids * 4 + overhead = 532 bytes.
        let cache = L0Cache::with_limits(1_000, 532 * 10);
        for i in 0..100u32 {
            cache.insert(format!("k{i:03}"), false, mock_encoding(vec![i; 100]));
            assert!(
                cache.bytes() <= 532 * 10,
                "{} bytes after {i}",
                cache.bytes()
            );
        }
        assert_eq!(cache.len(), 10);
        assert_eq!(cache.bytes(), 532 * 10);
    }

    #[test]
    fn an_entry_above_a_quarter_of_the_budget_is_not_cached() {
        let cache = L0Cache::with_limits(1_000, 4_000);
        cache.insert("big".to_string(), false, mock_encoding(vec![1; 300]));
        assert!(cache.get("big", false).is_none());
        assert_eq!(cache.len(), 0);
        cache.insert("small".to_string(), false, mock_encoding(vec![1; 10]));
        assert!(cache.get("small", false).is_some());
    }

    #[test]
    fn replacing_a_key_does_not_double_count() {
        let cache = L0Cache::with_limits(10, 100_000);
        cache.insert("k".to_string(), false, mock_encoding(vec![1; 10]));
        let once = cache.bytes();
        cache.insert("k".to_string(), false, mock_encoding(vec![2; 20]));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), once + 40);
        assert_eq!(cache.get("k", false).unwrap().token_ids(), &[2; 20]);
    }

    #[test]
    fn a_hot_set_survives_turnover() {
        // 100 entries, a hot set of 10 touched every five inserts, 20,000 cold
        // inserts: 200 turnovers of the cache. With uniform sampling a hot key
        // is never the oldest of eight; the hot set stays.
        let cache = L0Cache::new(100);
        let hot: Vec<String> = (0..10).map(|i| format!("hot{i}")).collect();
        for key in &hot {
            cache.insert(key.clone(), false, mock_encoding(vec![1]));
        }
        for i in 0..20_000u32 {
            cache.insert(format!("cold{i}"), false, mock_encoding(vec![i]));
            if i % 5 == 0 {
                for key in &hot {
                    assert!(
                        cache.get(key, false).is_some(),
                        "{key} lost after {i} inserts"
                    );
                }
            }
        }
        assert_eq!(cache.len(), 100);
    }
}
