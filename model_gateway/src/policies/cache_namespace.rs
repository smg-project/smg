//! Cache namespaces: the routing-side mirror of an engine's prefix-cache
//! partition.
//!
//! Engines partition their prefix caches by client-supplied request fields
//! (a cache salt, an extra classification key, the LoRA adapter). A router
//! that keys affinity on the prompt alone asserts prefix hits the engine
//! cannot serve whenever two requests share a prompt but differ in one of
//! those fields: it pins the request to a worker for a reuse that never
//! happens, and the operator sees a healthy router-side match rate over a
//! dead engine-side cache.
//!
//! A [`CacheNamespace`] is a fixed-width hash of the partition fields placed
//! at the head of every routing key, so requests in different namespaces
//! diverge at the first position and can never match each other, while
//! requests in the same namespace keep their full affinity. Only the hash
//! travels: no client value reaches the trees, the hash index or mesh
//! gossip. Requests without any partition field get no namespace, and their
//! routing keys are byte-identical to before.
//!
//! The event-driven index is keyed differently: the engines fold the LoRA
//! name and the cache salt into their block hashes, and the KV event monitor
//! recomputes stored blocks under the same XXH3 seed (`kv_index::salt`). A
//! namespace therefore also carries that seed, so a salted or LoRA request
//! is hashed under it and matches only its own blocks; a plain request keeps
//! the plain hash. The extra key is not part of the engines' seed.
//!
//! The marker is excluded from the match ratio, so it cannot by itself push
//! two unrelated prompts of one namespace over the cache threshold.
//!
//! Unpartitioned keys never start with marker material: real vocabulary ids
//! never carry the marker bit and prompt text does not begin with a control
//! character, and [`CacheNamespace::unpartitioned_tokens`] /
//! [`CacheNamespace::unpartitioned_text`] strip any that a client does send,
//! so the two key spaces are disjoint by construction and an unpartitioned
//! request cannot forge its way into a namespace's subtree.
//!
//! Occupancy: every live namespace holds its own copy of a shared prefix, so
//! the approximate trees' `max_tree_size` bounds the sum across namespaces.
//! Size it for tenants × working set; a per-request salt makes every request
//! a unique path.

use std::borrow::Cow;

use kv_index::salt::namespace_seed;
use openai_protocol::common::CachePartition;
use xxhash_rust::xxh3::Xxh3;

/// Domain separator: a change to the encoding changes every namespace.
const DOMAIN: &[u8] = b"smg/cache-namespace/v1";

/// Distinct token ids in a namespace marker (see
/// [`CacheNamespace::token_marker`]). The token-tree marker is padded to
/// the tree's page size ([`CacheNamespace::token_marker_len`]).
pub const TOKEN_MARKER_LEN: usize = 2;

/// Width in chars of the string-tree marker (see
/// [`CacheNamespace::text_marker`]).
pub const TEXT_MARKER_LEN: usize = 18;

/// Set on every marker token id. Vocabularies are orders of magnitude
/// smaller than 2^31, so a marker can never collide with a prompt token.
const MARKER_TOKEN_BIT: u32 = 1 << 31;

/// Pads a token-tree marker out to a whole page.
const MARKER_PAD_TOKEN: u32 = MARKER_TOKEN_BIT;

/// Delimits the string-tree marker; a control character that prompt text
/// does not begin with.
const TEXT_MARKER_DELIM: char = '\u{1}';

/// Fixed-width identity of a request's cache partition, and the XXH3 seed
/// under which the engines' KV events for that partition are hashed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheNamespace {
    marker: u64,
    event_seed: u64,
}

impl CacheNamespace {
    /// Derive the namespace from a request's partition fields; `None` when
    /// no field is set (an empty string counts as unset, as it does on the
    /// engine). Each field is encoded with a presence byte and its length,
    /// so an absent field and adjacent fields with shifted boundaries hash
    /// differently.
    pub fn derive(partition: &CachePartition<'_>) -> Option<Self> {
        if partition.is_empty() {
            return None;
        }
        let mut hasher = Xxh3::new();
        hasher.update(DOMAIN);
        for field in [
            partition.cache_salt,
            partition.extra_key,
            partition.lora_path,
        ] {
            match field.filter(|value| !value.is_empty()) {
                Some(value) => {
                    hasher.update(&[1u8]);
                    hasher.update(&(value.len() as u64).to_le_bytes());
                    hasher.update(value.as_bytes());
                }
                None => hasher.update(&[0u8]),
            }
        }
        Some(Self {
            marker: hasher.digest(),
            event_seed: namespace_seed(partition.lora_path, partition.cache_salt),
        })
    }

    /// The XXH3 seed the engines' blocks for this partition are hashed under
    /// (see `kv_index::salt::namespace_seed`); the plain seed when neither a
    /// LoRA name nor a cache salt is set.
    pub fn event_seed(self) -> u64 {
        self.event_seed
    }

    /// The partition's whole identity as an XXH3 seed, the extra key
    /// included. A host names a request's prefix under it, so two requests
    /// share a named prefix only where their partitions are one, as they
    /// share an engine's cache. A text request's windows are hashed under
    /// it too: no engine hashes text, so that seed need match none.
    pub fn partition_seed(self) -> u64 {
        self.marker
    }

    /// The namespace as token ids for the token tree and hash mode: two ids
    /// with the reserved high bit set, so they never match prompt tokens.
    pub fn token_marker(self) -> [u32; TOKEN_MARKER_LEN] {
        [
            (self.marker >> 32) as u32 | MARKER_TOKEN_BIT,
            self.marker as u32 | MARKER_TOKEN_BIT,
        ]
    }

    /// The namespace as a fixed-width text prefix for the string tree.
    pub fn text_marker(self) -> String {
        format!("{d}{:016x}{d}", self.marker, d = TEXT_MARKER_DELIM)
    }

    /// Length of the token-tree marker for a tree with `page_size`-token
    /// pages: the marker fills whole pages, so the prompt's own page grid is
    /// exactly what it is without a namespace and the match count for a given
    /// prompt is unchanged.
    pub fn token_marker_len(page_size: usize) -> usize {
        let page_size = page_size.max(1);
        TOKEN_MARKER_LEN.div_ceil(page_size) * page_size
    }

    /// `tokens` with the marker at its head, padded to whole `page_size` pages.
    pub fn prefixed_tokens(self, tokens: &[u32], page_size: usize) -> Vec<u32> {
        let marker_len = Self::token_marker_len(page_size);
        let mut keyed = Vec::with_capacity(marker_len + tokens.len());
        keyed.extend_from_slice(&self.token_marker());
        keyed.resize(marker_len, MARKER_PAD_TOKEN);
        keyed.extend_from_slice(tokens);
        keyed
    }

    /// `text` with the marker at its head.
    pub fn prefixed_text(self, text: &str) -> String {
        let mut keyed = String::with_capacity(TEXT_MARKER_LEN + text.len());
        keyed.push_str(&self.text_marker());
        keyed.push_str(text);
        keyed
    }

    /// The routing key of an unpartitioned request: leading ids that carry
    /// the marker bit are dropped so the key can never enter a namespace's
    /// subtree. A no-op for vocabulary ids.
    pub fn unpartitioned_tokens(tokens: &[u32]) -> &[u32] {
        let start = tokens
            .iter()
            .position(|&id| id & MARKER_TOKEN_BIT == 0)
            .unwrap_or(tokens.len());
        &tokens[start..]
    }

    /// The token tree's key for `tokens` under `namespace`, in a tree whose
    /// pages hold `page_size` tokens, and the length of the marker at its
    /// head. A partitioned request keys under its namespace marker: another
    /// namespace diverges at position 0 and can never match, and the marker
    /// never counts toward the match ratio, so a match count read from the
    /// tree has the marker length subtracted. The marker fills whole pages,
    /// so the prompt's page grid is the unpartitioned one. An unpartitioned
    /// request keys under its tokens stripped of marker material, marker
    /// length 0.
    #[inline]
    pub fn keyed_tokens(
        namespace: Option<Self>,
        tokens: &[u32],
        page_size: usize,
    ) -> (Cow<'_, [u32]>, usize) {
        match namespace {
            Some(namespace) => (
                Cow::Owned(namespace.prefixed_tokens(tokens, page_size)),
                Self::token_marker_len(page_size),
            ),
            None => (Cow::Borrowed(Self::unpartitioned_tokens(tokens)), 0),
        }
    }

    /// The routing key of an unpartitioned request: leading marker delimiters
    /// are dropped so the key can never enter a namespace's subtree. A no-op
    /// for prompt text.
    pub fn unpartitioned_text(text: &str) -> &str {
        text.trim_start_matches(TEXT_MARKER_DELIM)
    }

    /// The string tree's key for `text` under `namespace`, and the length
    /// in chars of the marker at its head: the rule of
    /// [`Self::keyed_tokens`], in chars. A partitioned request keys under
    /// its namespace marker, whose [`TEXT_MARKER_LEN`] chars never count
    /// toward the match ratio. An unpartitioned request keys under its text
    /// stripped of marker material, marker length 0.
    #[inline]
    pub fn keyed_text(namespace: Option<Self>, text: &str) -> (Cow<'_, str>, usize) {
        match namespace {
            Some(namespace) => (Cow::Owned(namespace.prefixed_text(text)), TEXT_MARKER_LEN),
            None => (Cow::Borrowed(Self::unpartitioned_text(text)), 0),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/cache_namespace/tests.rs"]
mod tests;
