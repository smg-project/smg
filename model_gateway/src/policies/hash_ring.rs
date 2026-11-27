//! Consistent hash ring for O(log n) worker selection.
//!
//! The ring maps a routing key to a worker URL using consistent hashing over
//! virtual nodes. Each membership generation has one ring, reconciled from
//! the ring kept for its model via [`HashRing::updated`], which rehashes only
//! the URLs that changed — individual lookups only pay an `O(log n)` binary
//! search plus a visited mark per worker (on the stack up to 512 workers, one
//! vector of words beyond) to skip virtual-node duplicates. See
//! [`HashRing::find_healthy_url`] for details.
//!
//! The type intentionally has no dependency on the `Worker` trait — it is
//! constructed from URLs — so policies and tests can build rings without
//! materializing fake workers.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use smg_grand_central::{
    engine::WorkerId,
    policies::consistent_hashing::{Head, LentRing},
};

use super::prefix_hash::PrefixHashPolicy;

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// The words of visited marks a lookup keeps on the stack: one bit per
/// worker, so a fleet of up to `64 * INLINE_VISITED_WORDS` workers is walked
/// without an allocation.
const INLINE_VISITED_WORDS: usize = 8;

/// The workers a lookup has asked about, marked by URL index: a bit per
/// worker in a few words on the stack for the fleets the ring usually holds,
/// a vector of words for a larger one.
enum Visited {
    Inline([u64; INLINE_VISITED_WORDS]),
    Heap(Vec<u64>),
}

impl Visited {
    /// Marks for a ring of `workers` URLs, none set.
    fn for_workers(workers: usize) -> Self {
        let words = workers.div_ceil(64);
        if words <= INLINE_VISITED_WORDS {
            Self::Inline([0; INLINE_VISITED_WORDS])
        } else {
            Self::Heap(vec![0; words])
        }
    }

    /// Marks the worker at `index`; whether this is its first mark.
    fn mark(&mut self, index: usize) -> bool {
        let words: &mut [u64] = match self {
            Self::Inline(words) => words,
            Self::Heap(words) => words,
        };
        let bit = 1u64 << (index % 64);
        let word = &mut words[index / 64];
        let first = *word & bit == 0;
        *word |= bit;
        first
    }
}

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
    /// Each URL's worker id ([`WorkerId::from_address`]), hashed once where
    /// the URL joins the ring: Grand Central's walk reads it
    /// ([`LentRing::walk_ids`]).
    ids: Arc<[WorkerId]>,
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
            entries.extend(vnodes_for(url, index as u32));
        }

        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
            ids: urls.iter().map(|url| WorkerId::from_address(url)).collect(),
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
        let mut new_ids: Vec<WorkerId> = Vec::with_capacity(desired.len());
        let mut remap: Vec<u32> = vec![u32::MAX; self.urls.len()];
        for (index, url) in self.urls.iter().enumerate() {
            if desired.contains(&**url) {
                remap[index] = new_urls.len() as u32;
                new_urls.push(Arc::clone(url));
                new_ids.push(self.ids[index]);
            }
        }

        let mut added_vnodes: Vec<(u64, u32)> =
            Vec::with_capacity(added.len().saturating_mul(VIRTUAL_NODES_PER_WORKER));
        for url in added {
            let index = new_urls.len() as u32;
            added_vnodes.extend(vnodes_for(&url, index));
            new_ids.push(WorkerId::from_address(&url));
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
            ids: Arc::from(new_ids.into_boxed_slice()),
        }
    }

    /// Whether `urls` are this ring's URLs in this ring's order: the ring is
    /// then the ring over them, and nothing is hashed or allocated to learn
    /// it. A ring keeps each URL where its first list put it and new ones
    /// behind, so a list given again answers true unless a list moved a URL
    /// the ring kept. False decides nothing: the same URLs in another order
    /// make the same ring, and [`Self::updated`] finds it.
    pub(crate) fn is_over<I>(&self, urls: I) -> bool
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut own = self.urls.iter();
        urls.into_iter()
            .all(|url| own.next().is_some_and(|own| &**own == url.as_ref()))
            && own.next().is_none()
    }

    /// This ring's URLs in the ring's own order: each where its first list
    /// put it, new ones behind.
    pub(crate) fn urls(&self) -> &Arc<[Arc<str>]> {
        &self.urls
    }

    /// Whether this ring and `other` are one table of positions, for the
    /// cases that say a ring was taken from another and not built again.
    #[cfg(test)]
    pub(crate) fn shares_positions_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.entries, &other.entries)
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "blake3 always produces 32 bytes — converting a fixed 8-byte slice to [u8; 8] is infallible"
    )]
    pub(crate) fn hash_position(s: &str) -> u64 {
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
    /// plus a visited mark per worker, to ask `is_healthy` once per worker
    /// while walking clockwise over its virtual nodes. The marks sit on the
    /// stack for a fleet of up to `64 * INLINE_VISITED_WORDS` workers, so
    /// such a lookup allocates nothing; a larger fleet allocates one vector
    /// of words per lookup.
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        self.walk_from(Self::hash_position(key), |_, url| is_healthy(url))
    }

    /// The walk behind every lookup: the ring's workers clockwise from
    /// `position`, each asked once by its place among the URLs and its URL
    /// until `take` takes one. The URL taken, `None` where the ring is
    /// empty or none is taken. [`Self::find_healthy_url`] starts it where
    /// its key stands, and Grand Central starts it at the place a
    /// request states and reads the id kept at a worker's place
    /// ([`LentRing::walk_ids`]): one walk, so the two visit the workers in
    /// one order.
    #[inline]
    fn walk_from(&self, position: u64, mut take: impl FnMut(usize, &str) -> bool) -> Option<&str> {
        if self.entries.is_empty() {
            return None;
        }

        let start = self.entries.partition_point(|(pos, _)| *pos < position);

        // Walk clockwise from start, wrapping around. A worker is asked once
        // whatever its virtual nodes: the entries index the URLs, so a bit per
        // index marks the ones asked, without a set to allocate.
        let mut visited = Visited::for_workers(self.worker_count());

        for i in 0..self.entries.len() {
            let (_, index) = self.entries[(start + i) % self.entries.len()];
            if !visited.mark(index as usize) {
                continue;
            }

            let url_str: &str = &self.urls[index as usize];
            if take(index as usize, url_str) {
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

/// The ring as Grand Central asks it, for a request its caller lends
/// the ring with: where a key's text stands, where a head of a request
/// stands, and the walk from a stated place. Nothing of the ring is copied.
/// The position is the one a lookup starts from and the walk is the
/// lookup's own (`HashRing::walk_from`), so the engine places a key on the
/// worker a ring policy finds for it. A head stands where the
/// `prefix_hash` policy looks it up (`PrefixHashPolicy::head_position`),
/// so the engine places a prompt on the worker that policy finds for it.
impl LentRing for HashRing {
    fn position(&self, key: &str) -> u64 {
        Self::hash_position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        Some(PrefixHashPolicy::head_position(head))
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.walk_from(position, |_, url| take(url));
    }

    fn walk_ids(&self, position: u64, take: &mut dyn FnMut(WorkerId) -> bool) {
        self.walk_from(position, |place, _| take(self.ids[place]));
    }
}

/// A model's ring before it is filled, as Grand Central is lent it: a
/// key has its position and no worker to walk to. The engine then places
/// the key at that position over the workers it keeps, as a ring policy
/// does with no ring (`ConsistentHashingPolicy::find_by_consistent_hash`:
/// the key's position modulo the eligible workers).
///
/// A head has its position too. The `prefix_hash` policy takes the least
/// loaded worker with no ring, and the engine's prefix form leaves a head
/// that reaches no worker to the stages behind its ring, the fewest
/// requests in flight. A head with no position would be placed on the
/// engine's own ring instead.
#[derive(Debug)]
pub(crate) struct Unfilled;

impl LentRing for Unfilled {
    fn position(&self, key: &str) -> u64 {
        HashRing::hash_position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        Some(PrefixHashPolicy::head_position(head))
    }

    fn walk(&self, _: u64, _: &mut dyn FnMut(&str) -> bool) {}
}

#[cfg(test)]
#[path = "../../tests/unit/policies/hash_ring/tests.rs"]
mod tests;
