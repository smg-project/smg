//! Consistent hash ring for O(log n) worker selection.
//!
//! The ring maps a routing key to a worker URL using consistent hashing over
//! virtual nodes. The registry reconciles one ring per model when workers are
//! added or removed via [`HashRing::updated`], which rehashes only the URLs
//! that changed — individual lookups only pay an `O(log n)` binary search plus
//! a small bounded dedupe set to skip virtual-node duplicates. See
//! [`HashRing::find_healthy_url`] for details.
//!
//! The type intentionally has no dependency on the `Worker` trait — it is
//! constructed from URLs — so policies and tests can build rings without
//! materializing fake workers.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// Consistent hash ring for O(log n) worker selection.
///
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on `hash(worker_url + vnode_index)`. This provides:
/// - Even key distribution across workers
/// - Minimal key redistribution when workers are added/removed (~1/N keys move)
/// - O(log n) lookup via binary search
///
/// Uses blake3 for stable, fast hashing that is consistent across Rust versions.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted list of `(ring_position, url_index)`.
    ///
    /// Multiple entries per worker (virtual nodes) for even distribution.
    /// Entries index into `urls` so reconciliation copies plain data instead
    /// of touching a refcount per virtual node.
    entries: Arc<[(u64, u32)]>,
    /// Unique worker URLs, indexed by the entries.
    urls: Arc<[Arc<str>]>,
}

/// The `(position, url_index)` virtual nodes for one URL, unsorted.
fn vnodes_for(url: &str, url_index: u32) -> impl Iterator<Item = (u64, u32)> + '_ {
    (0..VIRTUAL_NODES_PER_WORKER).map(move |vnode| {
        (
            HashRing::hash_position(&format!("{url}#{vnode}")),
            url_index,
        )
    })
}

impl HashRing {
    /// Build a hash ring from a collection of worker URLs.
    ///
    /// Creates `VIRTUAL_NODES_PER_WORKER` entries per unique URL for even
    /// distribution; repeated URLs keep only their first occurrence, matching
    /// [`Self::updated`], so a duplicate never carries extra ring weight.
    /// Accepts any iterable of string-like items, so callers can pass the
    /// output of `workers.iter().map(|w| w.url())` without allocating a Vec.
    pub fn new<I>(urls: I) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let iter = urls.into_iter();
        let (lower, _) = iter.size_hint();
        let mut seen: HashSet<Arc<str>> = HashSet::with_capacity(lower);
        let mut urls: Vec<Arc<str>> = Vec::with_capacity(lower);
        for url in iter {
            let url: Arc<str> = Arc::from(url.as_ref());
            if seen.insert(Arc::clone(&url)) {
                urls.push(url);
            }
        }
        let mut entries: Vec<(u64, u32)> =
            Vec::with_capacity(urls.len().saturating_mul(VIRTUAL_NODES_PER_WORKER));

        for (index, url) in urls.iter().enumerate() {
            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let vnode_key = format!("{url}#{vnode}");
                entries.push((Self::hash_position(&vnode_key), index as u32));
            }
        }

        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
            urls: Arc::from(urls.into_boxed_slice()),
        }
    }

    /// Reconcile this ring to exactly `urls`, rehashing only URLs the ring
    /// does not already contain. Content-equivalent to `HashRing::new(urls)`
    /// — retained URLs keep their virtual-node positions by construction —
    /// but costs one merge pass instead of a full rehash and sort, which is
    /// what keeps per-registration cost flat while a large fleet registers.
    pub fn updated<I>(&self, urls: I) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut desired: HashSet<String> = HashSet::new();
        let mut added: Vec<Arc<str>> = Vec::new();
        let existing: HashMap<&str, u32> = self
            .urls
            .iter()
            .enumerate()
            .map(|(index, url)| (&**url, index as u32))
            .collect();
        for url in urls {
            let url = url.as_ref();
            if desired.insert(url.to_string()) && !existing.contains_key(url) {
                added.push(Arc::from(url));
            }
        }

        if added.is_empty() && desired.len() == self.urls.len() {
            return self.clone();
        }

        // Remap retained URLs to their new indices; u32::MAX marks removal.
        let mut new_urls: Vec<Arc<str>> = Vec::with_capacity(desired.len());
        let mut remap: Vec<u32> = vec![u32::MAX; self.urls.len()];
        for (index, url) in self.urls.iter().enumerate() {
            if desired.contains(&**url) {
                remap[index] = new_urls.len() as u32;
                new_urls.push(Arc::clone(url));
            }
        }

        let mut added_vnodes: Vec<(u64, u32)> =
            Vec::with_capacity(added.len().saturating_mul(VIRTUAL_NODES_PER_WORKER));
        for url in added {
            let index = new_urls.len() as u32;
            added_vnodes.extend(vnodes_for(&url, index));
            new_urls.push(url);
        }
        added_vnodes.sort_unstable_by_key(|(pos, _)| *pos);

        // Single merge pass: retained entries (already sorted) + added vnodes.
        let mut merged: Vec<(u64, u32)> =
            Vec::with_capacity(new_urls.len().saturating_mul(VIRTUAL_NODES_PER_WORKER));
        let mut pending = added_vnodes.into_iter().peekable();
        for &(pos, index) in self.entries.iter() {
            let new_index = remap[index as usize];
            if new_index == u32::MAX {
                continue;
            }
            while let Some(vnode) = pending.next_if(|&(added_pos, _)| added_pos <= pos) {
                merged.push(vnode);
            }
            merged.push((pos, new_index));
        }
        merged.extend(pending);

        Self {
            entries: Arc::from(merged.into_boxed_slice()),
            urls: Arc::from(new_urls.into_boxed_slice()),
        }
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "blake3 always produces 32 bytes — converting a fixed 8-byte slice to [u8; 8] is infallible"
    )]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        u64::from_le_bytes(
            hash.as_bytes()[..8]
                .try_into()
                .expect("blake3 hash is always 32 bytes, slicing first 8 is infallible"),
        )
    }

    /// Find a worker URL for a key using consistent hashing.
    ///
    /// Returns the first healthy worker URL at or after the key's position
    /// (clockwise). Skips virtual nodes for workers already checked.
    ///
    /// Cost per call: `O(log n)` binary search to find the start position
    /// plus one small `HashSet` allocation bounded by
    /// `min(worker_count(), 16)` slots to dedupe virtual-node hits while
    /// walking clockwise. The dedupe set is dropped before return.
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // Walk clockwise from start, wrapping around. Track visited URLs to
        // avoid calling `is_healthy` multiple times for the same worker when
        // we hit its virtual nodes. Capacity is bounded by the physical worker
        // count — typically a handful of entries — so the per-lookup
        // allocation is negligible relative to the hashing itself.
        let mut checked_urls = HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, index) = self.entries[(start + i) % self.entries.len()];
            let url_str: &str = &self.urls[index as usize];

            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// Check if the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the ring (including virtual nodes).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of unique workers in the ring.
    pub fn worker_count(&self) -> usize {
        self.urls.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same_selection(a: &HashRing, b: &HashRing, keys: usize) {
        for i in 0..keys {
            let key = format!("key-{i}");
            assert_eq!(
                a.find_healthy_url(&key, |_| true),
                b.find_healthy_url(&key, |_| true),
                "selection diverged for {key}"
            );
        }
    }

    #[test]
    fn empty_ring_returns_none() {
        let ring = HashRing::new(std::iter::empty::<&str>());
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.worker_count(), 0);
        assert_eq!(ring.find_healthy_url("any-key", |_| true), None);
    }

    #[test]
    fn len_scales_with_virtual_nodes() {
        let ring = HashRing::new(["http://a", "http://b", "http://c"]);
        assert!(!ring.is_empty());
        assert_eq!(ring.len(), 3 * VIRTUAL_NODES_PER_WORKER);
        assert_eq!(ring.worker_count(), 3);
    }

    #[test]
    fn find_healthy_url_is_deterministic() {
        let ring = HashRing::new(["http://a", "http://b", "http://c"]);
        let first = ring.find_healthy_url("routing-key", |_| true).unwrap();
        for _ in 0..10 {
            assert_eq!(ring.find_healthy_url("routing-key", |_| true), Some(first));
        }
    }

    #[test]
    fn find_healthy_url_skips_unhealthy() {
        let ring = HashRing::new(["http://a", "http://b", "http://c"]);
        let picked = ring.find_healthy_url("routing-key", |url| url != "http://a");
        assert!(matches!(picked, Some("http://b") | Some("http://c")));
    }

    #[test]
    fn find_healthy_url_returns_none_when_all_unhealthy() {
        let ring = HashRing::new(["http://a", "http://b"]);
        assert_eq!(ring.find_healthy_url("k", |_| false), None);
    }

    #[test]
    fn accepts_owned_string_iterators() {
        let urls = vec!["http://a".to_string(), "http://b".to_string()];
        let ring = HashRing::new(urls);
        assert_eq!(ring.worker_count(), 2);
    }

    #[test]
    fn updated_with_addition_matches_full_rebuild() {
        let base = HashRing::new(["http://a", "http://b"]);
        let incremental = base.updated(["http://a", "http://b", "http://c"]);
        let rebuilt = HashRing::new(["http://a", "http://b", "http://c"]);

        assert_eq!(incremental.worker_count(), 3);
        assert_eq!(incremental.len(), rebuilt.len());
        assert_same_selection(&incremental, &rebuilt, 500);
    }

    #[test]
    fn updated_with_removal_matches_full_rebuild() {
        let base = HashRing::new(["http://a", "http://b", "http://c"]);
        let incremental = base.updated(["http://a", "http://c"]);
        let rebuilt = HashRing::new(["http://a", "http://c"]);

        assert_eq!(incremental.worker_count(), 2);
        assert_eq!(incremental.len(), rebuilt.len());
        assert_same_selection(&incremental, &rebuilt, 500);
    }

    #[test]
    fn updated_with_same_membership_is_a_cheap_clone() {
        let base = HashRing::new(["http://a", "http://b"]);
        let same = base.updated(["http://b", "http://a"]);
        assert!(Arc::ptr_eq(&base.entries, &same.entries));
        assert!(Arc::ptr_eq(&base.urls, &same.urls));
    }

    #[test]
    fn updated_survives_a_registration_wave() {
        // Grow one URL at a time — the registration pattern — and verify the
        // final ring is indistinguishable from a single full build.
        let urls: Vec<String> = (0..40).map(|i| format!("http://w{i}:8000")).collect();
        let mut ring = HashRing::new(std::iter::empty::<&str>());
        for i in 0..urls.len() {
            ring = ring.updated(&urls[..=i]);
        }
        let rebuilt = HashRing::new(&urls);
        assert_eq!(ring.len(), rebuilt.len());
        assert_eq!(ring.worker_count(), rebuilt.worker_count());
        assert_same_selection(&ring, &rebuilt, 1000);
    }

    #[test]
    fn updated_to_empty_clears_the_ring() {
        let base = HashRing::new(["http://a"]);
        let cleared = base.updated(std::iter::empty::<&str>());
        assert!(cleared.is_empty());
        assert_eq!(cleared.worker_count(), 0);
    }

    #[test]
    fn updated_dedupes_repeated_urls() {
        let base = HashRing::new(["http://a"]);
        let ring = base.updated(["http://a", "http://b", "http://b"]);
        assert_eq!(ring.worker_count(), 2);
        assert_eq!(ring.len(), 2 * VIRTUAL_NODES_PER_WORKER);
    }

    #[test]
    fn new_dedupes_repeated_urls() {
        // A repeated URL must not carry extra virtual-node weight.
        let deduped = HashRing::new(["http://a", "http://a"]);
        let single = HashRing::new(["http://a"]);
        assert_eq!(deduped.worker_count(), 1);
        assert_eq!(deduped.len(), VIRTUAL_NODES_PER_WORKER);
        assert_eq!(deduped.len(), single.len());
        assert_same_selection(&deduped, &single, 100);
    }

    #[test]
    fn duplicate_inputs_build_the_same_ring_through_both_paths() {
        let duplicated = ["http://a", "http://b", "http://b"];
        let rebuilt = HashRing::new(duplicated);
        let incremental = HashRing::new(["http://a"]).updated(duplicated);

        assert_eq!(rebuilt.worker_count(), incremental.worker_count());
        assert_eq!(rebuilt.len(), incremental.len());
        assert_same_selection(&rebuilt, &incremental, 500);
    }
}
