//! The wire hash scheme, owned by the service.
//!
//! Every hash that crosses a process boundary is defined HERE: the
//! content hash the gateway queries with and the bridge converts
//! events with, and the placement chain every publisher synthesizes.
//! The implementation is deliberately independent of `kv_index` — the
//! golden-vector tests pin it against values captured from the
//! production gateway crate, so agreement is proven, not inherited.
//! This module is the service's only source of the scheme.
//!
//! Scheme v1 (`HASH_SCHEME_V1`):
//! - content hash: XXH3-64, seed 1337, over the block's token ids as
//!   little-endian u32 bytes; only FULL blocks are hashed.
//! - chain hash: `XXH3-64(prev_le_bytes || current_le_bytes)`, seed
//!   1337, over 16 bytes.
//! - position 0 rule: a chain's first seq hash IS its content hash
//!   (no parent to chain from).
//!
//! Keyspace messages carry `hash_scheme`; 0 (unset) means v1 for
//! backward compatibility with in-flight publishers.

use crate::{ContentHash, SequenceHash};

/// The only scheme in existence. Proto `hash_scheme = 0` means this.
pub const HASH_SCHEME_V1: u32 = 1;

const SEED: u64 = 1337;

/// Is `scheme` one this build can serve? (0 = unset = v1.)
pub fn scheme_supported(scheme: u32) -> bool {
    scheme == 0 || scheme == HASH_SCHEME_V1
}

/// Content hash of one full block of token ids.
pub fn content_hash(token_ids: &[u32]) -> ContentHash {
    use std::hash::Hasher;
    let mut hasher = xxhash_rust::xxh3::Xxh3::with_seed(SEED);
    for &t in token_ids {
        hasher.write(&t.to_le_bytes());
    }
    ContentHash(hasher.finish())
}

/// Rolling chain hash: `XXH3(prev || current)` over LE bytes.
pub fn chain_prefix_hash(prev: SequenceHash, current: ContentHash) -> SequenceHash {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&prev.0.to_le_bytes());
    bytes[8..].copy_from_slice(&current.0.to_le_bytes());
    SequenceHash(xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, SEED))
}

/// Query-path chunking: one content hash per FULL block; the partial
/// trailing chunk is discarded (backends only cache full blocks).
pub fn request_content_hashes(tokens: &[u32], block_size: usize) -> Vec<ContentHash> {
    if block_size == 0 {
        return Vec::new();
    }
    tokens
        .chunks(block_size)
        .filter(|chunk| chunk.len() == block_size)
        .map(content_hash)
        .collect()
}

/// Deterministic placement chain: content hashes -> position-chained
/// (seq, content) pairs, identical across every publisher for
/// identical prefixes (the placement feed's idempotence contract).
pub fn placement_chain(content_hashes: &[ContentHash]) -> Vec<(SequenceHash, ContentHash)> {
    let mut out = Vec::with_capacity(content_hashes.len());
    let mut prev = SequenceHash(0);
    for (i, &content_hash) in content_hashes.iter().enumerate() {
        let seq_hash = if i == 0 {
            SequenceHash(content_hash.0)
        } else {
            chain_prefix_hash(prev, content_hash)
        };
        out.push((seq_hash, content_hash));
        prev = seq_hash;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scheme gate backs the server's reject-to-empty guard: only the
    /// versions this build can serve pass; a future/unknown scheme is
    /// refused so a scheme-mismatched hash never silently matches nothing
    /// against a v1 keyspace ("fail loudly, don't match silently").
    #[test]
    fn scheme_gate_admits_only_known_versions() {
        assert!(scheme_supported(0), "0 = unset defaults to v1");
        assert!(scheme_supported(HASH_SCHEME_V1));
        assert!(!scheme_supported(2), "an unknown/future scheme is refused");
        assert!(!scheme_supported(u32::MAX));
    }

    /// The scheme must agree with the production gateway crate on
    /// every path — proven directly, not assumed.
    #[test]
    fn agrees_with_kv_index_on_random_inputs() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..64 {
            let block: Vec<u32> = (0..256).map(|_| (next() & 0x3FFFF) as u32).collect();
            assert_eq!(
                content_hash(&block).0,
                kv_index::compute_content_hash(&block).0
            );
            let prev = next();
            let cur = next();
            assert_eq!(
                chain_prefix_hash(SequenceHash(prev), ContentHash(cur)).0,
                kv_index::chain_prefix_hash(
                    kv_index::SequenceHash(prev),
                    kv_index::ContentHash(cur)
                )
                .0
            );
        }
        let tokens: Vec<u32> = (0..1000).collect();
        assert_eq!(
            request_content_hashes(&tokens, 256)
                .iter()
                .map(|h| h.0)
                .collect::<Vec<_>>(),
            kv_index::compute_request_content_hashes(&tokens, 256)
                .iter()
                .map(|h| h.0)
                .collect::<Vec<_>>()
        );
    }

    /// Golden vectors: captured from `kv_index` output and PINNED as
    /// constants. If these ever fail, the wire scheme changed and
    /// every deployed publisher/index disagrees — that is a
    /// wire-compatibility break, not a refactor.
    #[test]
    fn golden_vectors() {
        // content_hash over [0, 1, ..., 255]
        let ascending: Vec<u32> = (0..256).collect();
        assert_eq!(content_hash(&ascending).0, GOLDEN_CONTENT_ASCENDING);
        // content_hash over 256 zeros
        assert_eq!(content_hash(&[0u32; 256]).0, GOLDEN_CONTENT_ZEROS);
        // content_hash of the empty slice (never on the wire, pinned anyway)
        assert_eq!(content_hash(&[]).0, GOLDEN_CONTENT_EMPTY);
        // chain step from prev=0
        assert_eq!(
            chain_prefix_hash(SequenceHash(0), ContentHash(GOLDEN_CONTENT_ASCENDING)).0,
            GOLDEN_CHAIN_FROM_ZERO
        );
        // three-block placement chain over blocks [0..256), [256..512), [512..768)
        let tokens: Vec<u32> = (0..768).collect();
        let chain = placement_chain(&request_content_hashes(&tokens, 256));
        assert_eq!(chain.len(), 3);
        // position-0 rule
        assert_eq!(chain[0].0 .0, chain[0].1 .0);
        let seqs: Vec<u64> = chain.iter().map(|b| b.0 .0).collect();
        assert_eq!(seqs, GOLDEN_PLACEMENT_SEQS);
    }

    const GOLDEN_CONTENT_ASCENDING: u64 = 0xbb79c83a0cdfbb76;
    const GOLDEN_CONTENT_ZEROS: u64 = 0x5672308b9a027dcd;
    const GOLDEN_CONTENT_EMPTY: u64 = 0xd1f3e77444430ab9;
    const GOLDEN_CHAIN_FROM_ZERO: u64 = 0x02639359a1a2c74b;
    const GOLDEN_PLACEMENT_SEQS: [u64; 3] =
        [0xbb79c83a0cdfbb76, 0xa85c3e553a06b20d, 0x086e95e44d3314d7];

    /// Capture helper the constants above came from (kept for
    /// provenance; run with --ignored --nocapture to re-derive).
    #[test]
    #[ignore = "provenance tool, not a test"]
    fn print_golden_capture() {
        let ascending: Vec<u32> = (0..256).collect();
        println!(
            "GOLDEN_CONTENT_ASCENDING: {:#x}",
            kv_index::compute_content_hash(&ascending).0
        );
        println!(
            "GOLDEN_CONTENT_ZEROS: {:#x}",
            kv_index::compute_content_hash(&[0u32; 256]).0
        );
        println!(
            "GOLDEN_CONTENT_EMPTY: {:#x}",
            kv_index::compute_content_hash(&[]).0
        );
        println!(
            "GOLDEN_CHAIN_FROM_ZERO: {:#x}",
            kv_index::chain_prefix_hash(
                kv_index::SequenceHash(0),
                kv_index::ContentHash(kv_index::compute_content_hash(&ascending).0)
            )
            .0
        );
        let tokens: Vec<u32> = (0..768).collect();
        let hashes = kv_index::compute_request_content_hashes(&tokens, 256);
        let mut prev = kv_index::SequenceHash(0);
        for (i, &h) in hashes.iter().enumerate() {
            let seq = if i == 0 {
                kv_index::SequenceHash(h.0)
            } else {
                kv_index::chain_prefix_hash(prev, h)
            };
            println!("GOLDEN_PLACEMENT_SEQS[{i}]: {:#x}", seq.0);
            prev = seq;
        }
    }
}
