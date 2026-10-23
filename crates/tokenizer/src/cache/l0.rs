//! L0: the exact-match cache from an input text to its token ids.
//!
//! An entry is the text's plain ids (4 bytes per token; nothing else the
//! encoding carried is kept), so the cache's memory is the texts, the ids
//! and a fixed per-entry overhead, and it is bounded two ways: by entries
//! and by bytes. An input that would take more than a quarter of the byte
//! budget on its own is not cached.
//!
//! The map is keyed on a 64-bit hash of the text and the `add_special_tokens`
//! flag. The hash is a keyed one that runs at memory speed (foldhash, seeded
//! per cache), computed once per request: a [`L0Cache::lookup`] that misses
//! hands it to the [`L0Cache::fill`] that follows, and the map's own hasher
//! passes it through. A hit is never taken on the hash alone: the entry's
//! text is compared byte for byte, so two texts with one hash can cost the
//! later one its slot, never a wrong answer.
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
//! serialize on the index, which they also keep consistent with the map.

use std::{
    hash::{BuildHasher, BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use dashmap::DashMap;

use super::activity::L0;
use crate::traits::{Encoding, TokenIdType};

/// Entries sampled per eviction; the oldest of them is removed.
const EVICTION_SAMPLE_SIZE: usize = 8;
/// What an entry costs beyond its text and ids: the map slot, the value and
/// its timestamp, the shared ids, the index entry.
const ENTRY_OVERHEAD_BYTES: usize = 128;
/// Byte budget when a caller gives none (`L0Cache::new`).
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// The map's hasher. Its keys are key hashes already (see [`L0Cache::key`]),
/// so a key passes through unchanged.
#[derive(Default, Clone, Copy)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // The map's keys are `u64`s and arrive through `write_u64`; anything
        // else is folded in so the hasher stays total.
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3);
        }
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

struct CachedEntry {
    /// The text, compared byte for byte on every hit.
    text: Box<str>,
    add_special_tokens: bool,
    encoding: Arc<Encoding>,
    last_accessed: AtomicU64,
    /// The bytes this entry counts against the budget.
    bytes: usize,
}

/// What a [`L0Cache::lookup`] found: the ids, or what the [`L0Cache::fill`]
/// after a miss needs.
pub(crate) enum L0Lookup {
    Hit(Arc<Encoding>),
    Miss(L0Miss),
}

/// A miss: the key the lookup computed, carried to [`L0Cache::fill`] so the
/// text is hashed once per request.
#[derive(Debug)]
pub(crate) struct L0Miss {
    hash: u64,
    add_special_tokens: bool,
}

/// L0 cache implementation using DashMap for lock-free reads.
pub struct L0Cache {
    /// Entries by key hash.
    map: DashMap<u64, CachedEntry, BuildHasherDefault<PassThrough>>,
    /// The key of every entry, for uniform eviction sampling. Also the lock
    /// that serializes inserts and evictions.
    index: Mutex<Vec<u64>>,
    /// The keyed hash of texts, seeded for this cache.
    keys: foldhash::fast::RandomState,
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
        let initial = max_entries.min(1024);
        Self {
            map: DashMap::with_capacity_and_hasher(initial, BuildHasherDefault::default()),
            index: Mutex::new(Vec::with_capacity(initial)),
            keys: foldhash::fast::RandomState::default(),
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

    /// The key of a text under a flag: one pass over the bytes.
    #[inline]
    fn key(&self, text: &str, add_special_tokens: bool) -> u64 {
        let mut hasher = self.keys.build_hasher();
        hasher.write(text.as_bytes());
        hasher.write_u8(u8::from(add_special_tokens));
        hasher.finish()
    }

    /// Get the next monotonic timestamp for access tracking.
    #[inline]
    fn next_timestamp(&self) -> u64 {
        self.access_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// The cached encoding of a text under `add_special_tokens`, or the miss
    /// that [`L0Cache::fill`] completes once the text is encoded. A hit is an
    /// entry whose text equals `text` byte for byte.
    #[inline]
    pub(crate) fn lookup(&self, text: &str, add_special_tokens: bool) -> L0Lookup {
        let hash = self.key(text, add_special_tokens);
        if let Some(entry) = self.map.get(&hash) {
            if entry.add_special_tokens == add_special_tokens && *entry.text == *text {
                self.hits.fetch_add(1, Ordering::Relaxed);
                L0.hit(text.len());
                let ts = self.next_timestamp();
                entry.last_accessed.store(ts, Ordering::Relaxed);
                return L0Lookup::Hit(Arc::clone(&entry.encoding));
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        L0.miss();
        L0Lookup::Miss(L0Miss {
            hash,
            add_special_tokens,
        })
    }

    /// Look up a cached encoding for the given input text and
    /// add_special_tokens flag. Returns an Arc clone (cheap) if found.
    #[inline]
    pub fn get(&self, key: &str, add_special_tokens: bool) -> Option<Arc<Encoding>> {
        match self.lookup(key, add_special_tokens) {
            L0Lookup::Hit(encoding) => Some(encoding),
            L0Lookup::Miss(_) => None,
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
    fn evict_one(&self, index: &mut Vec<u64>) {
        if index.is_empty() {
            return;
        }
        let mut oldest: Option<(usize, u64)> = None;
        let mut consider = |cache: &Self, position: usize| {
            let timestamp = cache
                .map
                .get(&index[position])
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
        if let Some((_, removed)) = self.map.remove(&victim) {
            self.bytes.fetch_sub(removed.bytes, Ordering::Relaxed);
            self.entries.fetch_sub(1, Ordering::Relaxed);
            L0.evict();
        }
    }

    /// Caches the ids of the text a [`L0Cache::lookup`] missed on, under the
    /// key that lookup computed: `text` must be the text given to it. The
    /// text is copied once, into the entry. An entry that would exceed a
    /// quarter of the byte budget is not cached.
    pub(crate) fn fill(&self, miss: L0Miss, text: &str, ids: Vec<TokenIdType>) {
        let L0Miss {
            hash,
            add_special_tokens,
        } = miss;
        let bytes = text.len() + ids.len() * 4 + ENTRY_OVERHEAD_BYTES;
        if bytes > self.max_bytes / 4 {
            return;
        }
        let entry = CachedEntry {
            text: text.into(),
            add_special_tokens,
            encoding: Arc::new(Encoding::Plain(ids)),
            last_accessed: AtomicU64::new(self.next_timestamp()),
            bytes,
        };
        let mut index = self
            .index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // What the slot holds: the same text (a concurrent miss filled it
        // first; replaced below) or, under the same hash, another text, which
        // keeps its slot so that every key stays one entry in the map and in
        // the index.
        let replacing = match self.map.get(&hash) {
            Some(resident)
                if resident.add_special_tokens == add_special_tokens && *resident.text == *text =>
            {
                Some(resident.bytes)
            }
            Some(_) => return,
            None => None,
        };
        let incoming = bytes.saturating_sub(replacing.unwrap_or(0));
        let new_entry = replacing.is_none();
        while !index.is_empty()
            && ((new_entry && self.entries.load(Ordering::Relaxed) >= self.max_entries)
                || self.bytes.load(Ordering::Relaxed) + incoming > self.max_bytes)
        {
            self.evict_one(&mut index);
        }
        match self.map.insert(hash, entry) {
            Some(old) => {
                self.bytes.fetch_sub(old.bytes, Ordering::Relaxed);
            }
            None => {
                index.push(hash);
                self.entries.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Insert an encoding into the cache. Only the ids are kept. The request
    /// path uses [`L0Cache::lookup`] and [`L0Cache::fill`], which hash the
    /// text once between them.
    pub fn insert(&self, key: String, add_special_tokens: bool, value: Encoding) {
        let ids = match value {
            Encoding::Plain(ids) => ids,
            other => other.token_ids().to_vec(),
        };
        let miss = L0Miss {
            hash: self.key(&key, add_special_tokens),
            add_special_tokens,
        };
        self.fill(miss, &key, ids);
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
        self.map.clear();
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
    use super::*;
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

    #[test]
    fn a_miss_filled_is_a_hit_with_the_same_ids() {
        let cache = L0Cache::new(10);
        let text = "<|im_start|>user\nhello<|im_end|>";
        let L0Lookup::Miss(miss) = cache.lookup(text, true) else {
            panic!("an empty cache misses");
        };
        cache.fill(miss, text, vec![5, 6, 7]);
        let L0Lookup::Hit(hit) = cache.lookup(text, true) else {
            panic!("the filled text hits");
        };
        assert_eq!(hit.token_ids(), &[5, 6, 7]);
        // The other flag value is another key.
        assert!(matches!(cache.lookup(text, false), L0Lookup::Miss(_)));
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (1, 2, 1));
    }

    #[test]
    fn texts_that_differ_in_one_byte_are_distinct_entries() {
        // Long texts whose difference sits at the end or in the middle: each
        // is its own entry and only its own text hits it.
        let cache = L0Cache::new(10);
        let base = "the same long paragraph, repeated ".repeat(400);
        let variants = [
            base.clone(),
            format!("{base}."),
            format!(
                "{}X{}",
                &base[..base.len() / 2],
                &base[base.len() / 2 + 1..]
            ),
            base[..base.len() - 1].to_string(),
        ];
        for (i, text) in variants.iter().enumerate() {
            assert!(
                cache.get(text, false).is_none(),
                "variant {i} before its insert"
            );
            cache.insert(text.clone(), false, mock_encoding(vec![i as u32; 3]));
        }
        for (i, text) in variants.iter().enumerate() {
            assert_eq!(
                cache.get(text, false).unwrap().token_ids(),
                &[i as u32; 3],
                "variant {i}"
            );
        }
        assert_eq!(cache.len(), variants.len());
    }

    #[test]
    fn a_fill_for_a_text_filled_meanwhile_replaces_it_once() {
        // Two misses on one text (two threads encoding it at once), two fills:
        // one entry, counted once, holding the later ids.
        let cache = L0Cache::with_limits(10, 100_000);
        let text = "shared prompt";
        let L0Lookup::Miss(first) = cache.lookup(text, false) else {
            panic!("miss");
        };
        let L0Lookup::Miss(second) = cache.lookup(text, false) else {
            panic!("miss");
        };
        cache.fill(first, text, vec![1; 10]);
        let once = cache.bytes();
        cache.fill(second, text, vec![2; 20]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), once + 40);
        assert_eq!(cache.get(text, false).unwrap().token_ids(), &[2; 20]);
    }

    #[test]
    fn two_texts_under_one_key_keep_the_resident_and_never_cross() {
        // A 64-bit collision cannot be produced on demand, so the misses are
        // forged with one key: the later text is not cached, the resident
        // keeps its slot, and a lookup of either text by its real key never
        // returns the other's ids.
        let cache = L0Cache::new(10);
        let forged = |hash| L0Miss {
            hash,
            add_special_tokens: false,
        };
        cache.fill(forged(7), "first", vec![1]);
        cache.fill(forged(7), "second", vec![2]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.index.lock().unwrap().as_slice(), &[7]);
        let resident = cache.map.get(&7).unwrap();
        assert_eq!(&*resident.text, "first");
        drop(resident);
        assert!(cache.get("second", false).is_none());
        // A real lookup of "first" computes its own key, not the forged one.
        assert!(cache.get("first", false).is_none());
    }

    #[test]
    fn eviction_tracks_entries_by_key() {
        // Entries filled through lookup/fill evict like inserted ones: the
        // index holds one key per entry, and removing a victim frees its bytes.
        let cache = L0Cache::with_limits(3, 1 << 20);
        for i in 0..6u32 {
            let text = format!("prompt {i}");
            let L0Lookup::Miss(miss) = cache.lookup(&text, false) else {
                panic!("miss");
            };
            cache.fill(miss, &text, vec![i; 4]);
            assert!(cache.len() <= 3, "{} entries after fill {i}", cache.len());
            assert_eq!(cache.index.lock().unwrap().len(), cache.len());
        }
        let live: usize = (0..6u32)
            .filter(|i| cache.get(&format!("prompt {i}"), false).is_some())
            .count();
        assert_eq!(live, 3);
        let expected: usize = (0..6u32)
            .filter(|i| cache.get(&format!("prompt {i}"), false).is_some())
            .map(|i| format!("prompt {i}").len() + 16 + ENTRY_OVERHEAD_BYTES)
            .sum();
        assert_eq!(cache.bytes(), expected);
    }
}
