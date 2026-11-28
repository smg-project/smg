//! Which engine (rank) of a grouped data-parallel worker a sticky key is
//! pinned to under `--grouped-dp-sticky` (see
//! `request_execution::pin_grouped_dp_rank`).
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
#[path = "../../../../../tests/unit/routers/grpc/common/stages/grouped_dp/tests.rs"]
mod tests;
