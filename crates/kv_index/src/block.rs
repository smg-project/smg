//! Blocks and their hashes: the vocabulary every index in this crate shares.
//!
//! An engine names a cached block by a position-aware hash (its chain hash,
//! [`SequenceHash`]) and publishes the block's token ids; the router hashes
//! the tokens into a position-independent [`ContentHash`] (XXH3) and chains
//! content hashes into its own prefix hashes ([`chain_prefix_hash`]). A
//! request is the sequence of content hashes of its full blocks
//! ([`compute_request_content_hashes`]); an index scores each worker with the
//! length of the request prefix it holds ([`OverlapScores`]).

use std::fmt;

use rustc_hash::FxHashMap;

/// Seed for XXH3 hashing.
pub const XXH3_SEED: u64 = 1337;

/// Position-independent content hash of tokens within a single block.
/// Computed via XXH3-64 from token IDs. Same tokens always produce the same hash
/// regardless of their position in the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ContentHash(pub u64);

/// Position-aware block hash from backend (sequence hash).
/// Matches the `block_hash` field in KvBlock proto (i64, bitwise reinterpreted as u64).
/// Different from ContentHash because it encodes the full prefix history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SequenceHash(pub u64);

impl From<i64> for SequenceHash {
    fn from(value: i64) -> Self {
        Self(value as u64)
    }
}

impl From<u64> for SequenceHash {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Internal worker identifier used in [`OverlapScores`].
///
/// Consumers map worker URLs to this type via [`ChainIndex::worker_id`](crate::ChainIndex::worker_id).
pub type WorkerId = u32;

/// A block from a store event, carrying both hash representations.
#[derive(Debug, Clone, Copy)]
pub struct StoredBlock {
    /// Position-aware hash from the backend proto (`block_hash` field).
    pub seq_hash: SequenceHash,
    /// Position-independent hash computed from token IDs via XXH3.
    pub content_hash: ContentHash,
}

/// Error returned by [`ChainIndex::apply_stored`](crate::ChainIndex::apply_stored) when the event cannot be applied.
#[derive(Debug)]
pub enum ApplyError {
    /// Worker has no entries in the index — cannot resolve parent block.
    WorkerNotTracked,
    /// The specified `parent_seq_hash` was not found in this worker's reverse lookup.
    ParentBlockNotFound,
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerNotTracked => write!(f, "worker not tracked in index"),
            Self::ParentBlockNotFound => write!(f, "parent block hash not found for worker"),
        }
    }
}

impl std::error::Error for ApplyError {}

/// Error returned by [`ChainIndex::intern_worker`](crate::ChainIndex::intern_worker) when the u32 worker-id
/// space is exhausted. Ids are assigned monotonically and never recycled, so this
/// can only be reached after `u32::MAX + 1` distinct worker URLs have been interned
/// over the indexer's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerIdExhausted;

impl fmt::Display for WorkerIdExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worker id space exhausted (u32::MAX workers interned)")
    }
}

impl std::error::Error for WorkerIdExhausted {}

/// Overlap scores: how many consecutive blocks each worker has cached.
///
/// Keys are internal `u32` worker IDs. Use [`ChainIndex::worker_id`](crate::ChainIndex::worker_id) to
/// map a worker URL to its internal ID for lookups. A worker's total block count
/// is available separately through [`ChainIndex::worker_block_count`](crate::ChainIndex::worker_block_count); it
/// is not collected per lookup, since no routing decision reads it there.
#[derive(Debug, Default)]
pub struct OverlapScores {
    /// internal_worker_id → number of matching prefix blocks (depth in indexer)
    pub scores: FxHashMap<u32, u32>,
}

/// One-shot over a stack buffer for blocks up to 256 tokens: the streaming
/// hasher derives a seeded secret per instance before it sees a byte, which
/// outweighs hashing a 16-token block. Same digest either way (the test
/// below pins that).
pub fn compute_content_hash(token_ids: &[u32]) -> ContentHash {
    const STACK_BYTES: usize = 1024;
    let len = token_ids.len() * 4;
    if len <= STACK_BYTES {
        let mut bytes = [0u8; STACK_BYTES];
        let (slots, _) = bytes[..len].as_chunks_mut::<4>();
        for (slot, token) in slots.iter_mut().zip(token_ids) {
            *slot = token.to_le_bytes();
        }
        return ContentHash(xxhash_rust::xxh3::xxh3_64_with_seed(
            &bytes[..len],
            XXH3_SEED,
        ));
    }
    let bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    ContentHash(xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED))
}

/// Rolling prefix hash over content hashes: `XXH3(prev || current)`, the
/// same chaining the chain index computes internally. Exported so
/// out-of-process publishers (the radix index service's placement feed)
/// can synthesize byte-identical position chains for identical prefixes.
///
/// Note the base case: position 0's `SequenceHash` is the bare
/// `ContentHash` value (`SequenceHash(c0.0)`), NOT
/// `chain_prefix_hash(SequenceHash(0), c0)`. Callers must seed with the
/// first content hash and chain from position 1, or the whole chain
/// silently diverges from the indexer's (zero prefix matches, no error):
///
/// ```
/// use kv_index::{chain_prefix_hash, ContentHash, SequenceHash};
/// let contents = [ContentHash(11), ContentHash(22), ContentHash(33)];
/// let mut chain = vec![SequenceHash(contents[0].0)];
/// for &c in &contents[1..] {
///     let prev = *chain.last().unwrap();
///     chain.push(chain_prefix_hash(prev, c));
/// }
/// assert_eq!(chain.len(), 3);
/// ```
pub fn chain_prefix_hash(prev: SequenceHash, current: ContentHash) -> SequenceHash {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&prev.0.to_le_bytes());
    bytes[8..].copy_from_slice(&current.0.to_le_bytes());
    SequenceHash(xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED))
}

/// Chunk request tokens by block size and compute a [`ContentHash`] per full block.
///
/// This is the entry point for the **query path**: given a request's token IDs and
/// the backend's block size, produce the content-hash sequence that
/// [`ChainIndex::find_matches`](crate::ChainIndex::find_matches) expects.
///
/// Partial trailing chunks (fewer tokens than `block_size`) are discarded because
/// backends only cache full blocks.
///
/// Returns an empty `Vec` if `block_size` is 0.
pub fn compute_request_content_hashes(tokens: &[u32], block_size: usize) -> Vec<ContentHash> {
    if block_size == 0 {
        tracing::warn!("compute_request_content_hashes called with block_size=0, returning empty");
        return Vec::new();
    }
    tokens
        .chunks(block_size)
        .filter(|chunk| chunk.len() == block_size)
        .map(compute_content_hash)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The streaming hasher this function used before; the one-shot path must
    /// stay bit-identical because workers report hashes computed the old way.
    fn streaming_content_hash(token_ids: &[u32]) -> ContentHash {
        use std::hash::Hasher;
        let mut hasher = xxhash_rust::xxh3::Xxh3::with_seed(XXH3_SEED);
        for &t in token_ids {
            hasher.write(&t.to_le_bytes());
        }
        ContentHash(hasher.finish())
    }

    #[test]
    fn content_hash_matches_streaming_hasher() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Every length across the stack/heap boundary (256 tokens), including
        // the short-input, mid-input and long-input XXH3 regimes.
        for len in 0..=300usize {
            let tokens: Vec<u32> = (0..len).map(|_| (next() % 160_000) as u32).collect();
            assert_eq!(
                compute_content_hash(&tokens),
                streaming_content_hash(&tokens),
                "len={len}"
            );
        }
        assert_eq!(
            compute_content_hash(&[]),
            streaming_content_hash(&[]),
            "empty block"
        );
    }

    #[test]
    fn content_hash_of_no_tokens_is_stable() {
        assert_eq!(compute_content_hash(&[]), compute_content_hash(&[]));
    }

    #[test]
    fn content_hash_tells_single_tokens_apart() {
        assert_ne!(compute_content_hash(&[42]), compute_content_hash(&[43]));
    }

    #[test]
    fn request_content_hashes_chunk_full_blocks() {
        let hashes = compute_request_content_hashes(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 4);
        assert_eq!(hashes.len(), 2, "a partial last block is not a block");
        assert_eq!(hashes[0], compute_content_hash(&[1, 2, 3, 4]));
        assert_eq!(hashes[1], compute_content_hash(&[5, 6, 7, 8]));
    }

    #[test]
    fn request_content_hashes_with_zero_block_size_are_empty() {
        assert!(compute_request_content_hashes(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn a_prefix_hash_depends_on_both_its_inputs() {
        let a = chain_prefix_hash(SequenceHash(1), ContentHash(2));
        assert_ne!(a, chain_prefix_hash(SequenceHash(1), ContentHash(3)));
        assert_ne!(a, chain_prefix_hash(SequenceHash(2), ContentHash(2)));
        assert_eq!(a, chain_prefix_hash(SequenceHash(1), ContentHash(2)));
    }
}
