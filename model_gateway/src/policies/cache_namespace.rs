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

/// Fixed-width identity of a request's cache partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheNamespace(u64);

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
        Some(Self(hasher.digest()))
    }

    /// The namespace as token ids for the token tree and hash mode: two ids
    /// with the reserved high bit set, so they never match prompt tokens.
    pub fn token_marker(self) -> [u32; TOKEN_MARKER_LEN] {
        [
            (self.0 >> 32) as u32 | MARKER_TOKEN_BIT,
            self.0 as u32 | MARKER_TOKEN_BIT,
        ]
    }

    /// The namespace as a fixed-width text prefix for the string tree.
    pub fn text_marker(self) -> String {
        format!("{d}{:016x}{d}", self.0, d = TEXT_MARKER_DELIM)
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

    /// The routing key of an unpartitioned request: leading marker delimiters
    /// are dropped so the key can never enter a namespace's subtree. A no-op
    /// for prompt text.
    pub fn unpartitioned_text(text: &str) -> &str {
        text.trim_start_matches(TEXT_MARKER_DELIM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partition<'a>(
        cache_salt: Option<&'a str>,
        extra_key: Option<&'a str>,
        lora_path: Option<&'a str>,
    ) -> CachePartition<'a> {
        CachePartition {
            cache_salt,
            extra_key,
            lora_path,
        }
    }

    #[test]
    fn unpartitioned_requests_have_no_namespace() {
        assert_eq!(CacheNamespace::derive(&CachePartition::default()), None);
    }

    #[test]
    fn derivation_is_stable_and_field_sensitive() {
        let a = CacheNamespace::derive(&partition(Some("tenant-a"), None, None)).unwrap();
        assert_eq!(
            a,
            CacheNamespace::derive(&partition(Some("tenant-a"), None, None)).unwrap()
        );
        let b = CacheNamespace::derive(&partition(Some("tenant-b"), None, None)).unwrap();
        assert_ne!(a, b);
        // The same bytes in a different field are a different partition.
        let extra = CacheNamespace::derive(&partition(None, Some("tenant-a"), None)).unwrap();
        let lora = CacheNamespace::derive(&partition(None, None, Some("tenant-a"))).unwrap();
        assert_ne!(a, extra);
        assert_ne!(a, lora);
        assert_ne!(extra, lora);
    }

    #[test]
    fn empty_fields_count_as_absent() {
        // Engines treat an empty salt or adapter as unset; so does routing.
        assert_eq!(
            CacheNamespace::derive(&partition(Some(""), None, None)),
            None
        );
        assert_eq!(
            CacheNamespace::derive(&partition(Some(""), Some("k"), None)),
            CacheNamespace::derive(&partition(None, Some("k"), None))
        );
    }

    #[test]
    fn shifted_field_boundaries_differ() {
        let ab_c = CacheNamespace::derive(&partition(Some("ab"), Some("c"), None)).unwrap();
        let a_bc = CacheNamespace::derive(&partition(Some("a"), Some("bc"), None)).unwrap();
        assert_ne!(ab_c, a_bc);
    }

    #[test]
    fn unpartitioned_keys_never_start_with_marker_material() {
        let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
        // Vocabulary ids and prompt text pass through untouched.
        assert_eq!(CacheNamespace::unpartitioned_tokens(&[7, 8, 9]), &[7, 8, 9]);
        assert_eq!(CacheNamespace::unpartitioned_text("hello"), "hello");
        // A forged marker at the head of an unpartitioned key is dropped.
        let forged_tokens = ns.prefixed_tokens(&[7, 8, 9], 16);
        assert_eq!(
            CacheNamespace::unpartitioned_tokens(&forged_tokens),
            &[7, 8, 9]
        );
        let forged_text = ns.prefixed_text("hello");
        assert!(!CacheNamespace::unpartitioned_text(&forged_text).starts_with(TEXT_MARKER_DELIM));
        assert!(CacheNamespace::unpartitioned_tokens(&ns.token_marker()).is_empty());
    }

    #[test]
    fn token_marker_never_collides_with_vocabulary_ids() {
        let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
        for id in ns.token_marker() {
            assert!(id & MARKER_TOKEN_BIT != 0);
        }
        let keyed = ns.prefixed_tokens(&[7, 8, 9], 1);
        assert_eq!(keyed.len(), TOKEN_MARKER_LEN + 3);
        assert_eq!(&keyed[TOKEN_MARKER_LEN..], &[7, 8, 9]);
        assert_eq!(&keyed[..TOKEN_MARKER_LEN], &ns.token_marker());
    }

    #[test]
    fn token_marker_fills_whole_pages() {
        let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
        assert_eq!(CacheNamespace::token_marker_len(16), 16);
        assert_eq!(CacheNamespace::token_marker_len(1), 2);
        assert_eq!(CacheNamespace::token_marker_len(0), 2);
        let keyed = ns.prefixed_tokens(&[7, 8, 9], 16);
        assert_eq!(keyed.len(), 16 + 3);
        assert_eq!(&keyed[..TOKEN_MARKER_LEN], &ns.token_marker());
        assert!(keyed[TOKEN_MARKER_LEN..16]
            .iter()
            .all(|&id| id & MARKER_TOKEN_BIT != 0));
        assert_eq!(&keyed[16..], &[7, 8, 9]);
    }

    #[test]
    fn text_marker_is_fixed_width_and_prefixes_the_prompt() {
        let ns = CacheNamespace::derive(&partition(None, Some("k"), Some("adapter"))).unwrap();
        let marker = ns.text_marker();
        assert_eq!(marker.chars().count(), TEXT_MARKER_LEN);
        assert!(marker.starts_with(TEXT_MARKER_DELIM));
        assert!(marker.ends_with(TEXT_MARKER_DELIM));
        let keyed = ns.prefixed_text("hello");
        assert_eq!(keyed.chars().count(), TEXT_MARKER_LEN + 5);
        assert!(keyed.ends_with("hello"));
        assert!(keyed.starts_with(&marker));
    }
}
