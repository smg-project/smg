//! Which engine (rank) of a grouped data-parallel worker a sticky key is
//! pinned to under `--grouped-dp-sticky` (see
//! `request_execution::attempt_load_guards`).
//!
//! Hash placement (`hash`: `xxh3(key) % dp_size`) needs no memory and gives
//! a key the same rank in every gateway process, across restarts and
//! replicas.

use xxhash_rust::xxh3::xxh3_64;

/// The rank hash placement gives `key` among `size` ranks: the key's xxh3,
/// a hash fixed by its specification, so the same key maps to the same rank
/// across requests, gateway restarts, replicas and builds (std's
/// `DefaultHasher` may change between Rust releases).
pub(crate) fn hash_rank(key: &str, size: usize) -> usize {
    (xxh3_64(key.as_bytes()) % size as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key's hash rank is its xxh3 modulo the ranks, pinned here by value:
    /// a key keeps its rank in every process and build, and a change of hash
    /// would move every live key to another engine's cold cache.
    #[test]
    fn a_keys_hash_rank_is_fixed_by_its_xxh3() {
        for (key, size, rank) in [
            ("conv-0", 8, 4),
            ("conv-1", 8, 5),
            ("conv-2", 8, 7),
            ("conv-2", 4, 3),
            ("session-a", 3, 1),
            ("session-b", 4, 2),
        ] {
            assert_eq!(hash_rank(key, size), rank, "{key} of {size}");
        }
    }
}
