//! Stream chunking and batching primitives.
//!
//! [`chunk_value`] splits an oversized value into a sequence of
//! `StreamEntry`s; [`build_stream_batches`] packs them into
//! `StreamBatch` messages respecting per-batch count and byte caps.
//! [`next_generation`] hands out monotonic tags so the receiver's
//! [`ChunkAssembler`](crate::transport::chunk_assembler::ChunkAssembler)
//! can group chunks of the same value.
//!
//! These are pure data-shape functions: no `MeshKV` state, no peer
//! routing, no envelope construction. Higher-level composition lives
//! in [`crate::transport::sync_stream`].

use std::sync::{
    atomic::{AtomicU64, Ordering},
    LazyLock,
};

use bytes::Bytes;

use crate::service::gossip::{StreamBatch, StreamEntry};

/// Monotonic generation counter for stream-batch values. Seeded from
/// wall-clock nanos on first use so a process restart starts at a
/// value that almost certainly exceeds any generation emitted before
/// the restart (receivers holding stale assembler state won't accept
/// a smaller tag). Subsequent calls are pure atomic increments, so
/// NTP steps or clock drift mid-process can't rewind the counter.
static STREAM_GENERATION: LazyLock<AtomicU64> = LazyLock::new(|| {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    AtomicU64::new(seed)
});

/// Returns the next monotonic generation tag for a stream-batch value.
/// Call this once per published value — every chunk of that value
/// shares the tag so the receiver's assembler can group them, and
/// concurrent publishes to the same key get distinct tags so stale
/// fragments don't mix into a fresh value.
pub fn next_generation() -> u64 {
    STREAM_GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// Split a stream value into one or more `StreamEntry`s.
///
/// Values with `value.len() <= max_chunk_bytes` produce a single entry
/// with `total_chunks = 1`. Larger values split into
/// `ceil(len / max_chunk_bytes)` chunks of at most `max_chunk_bytes`
/// bytes each. Chunks are emitted in index order 0..N. `generation` is
/// a per-publish monotonic tag chosen by the caller.
pub fn chunk_value(
    key: String,
    generation: u64,
    value: Bytes,
    max_chunk_bytes: usize,
) -> Vec<StreamEntry> {
    assert!(max_chunk_bytes > 0, "max_chunk_bytes must be non-zero");

    if value.len() <= max_chunk_bytes {
        return vec![StreamEntry {
            key,
            generation,
            chunk_index: 0,
            total_chunks: 1,
            data: value,
        }];
    }

    debug_assert!(
        !key.starts_with("tree:page:"),
        "tree:page:* values must be bounded by max_tree_repair_page_bytes \
         and never enter the multi-chunk split path"
    );

    let total_chunks = value.len().div_ceil(max_chunk_bytes);
    let mut entries = Vec::with_capacity(total_chunks);
    for index in 0..total_chunks {
        let start = index * max_chunk_bytes;
        let end = (start + max_chunk_bytes).min(value.len());
        entries.push(StreamEntry {
            key: key.clone(),
            generation,
            chunk_index: index as u32,
            total_chunks: total_chunks as u32,
            data: value.slice(start..end),
        });
    }
    entries
}

/// Pack `StreamEntry`s into one or more `StreamBatch`es, flushing
/// the current batch whenever adding the next entry would exceed
/// either `max_chunks_per_batch` (count) or `max_bytes_per_batch`
/// (total payload bytes). An entry that on its own exceeds the byte
/// cap still goes out as a single-entry batch — dropping would
/// strand downstream chunks undeliverable (at-most-once; buffer is
/// already drained so the sender has nothing to retry with).
pub fn build_stream_batches(
    entries: Vec<StreamEntry>,
    max_chunks_per_batch: usize,
    max_bytes_per_batch: usize,
) -> Vec<StreamBatch> {
    debug_assert!(
        max_chunks_per_batch > 0,
        "max_chunks_per_batch must be non-zero; callers should pass \
         DEFAULT_MAX_CHUNKS_PER_BATCH or another positive cap"
    );
    debug_assert!(
        max_bytes_per_batch > 0,
        "max_bytes_per_batch must be non-zero; callers should pass \
         MAX_STREAM_CHUNK_BYTES or another positive cap"
    );
    if max_chunks_per_batch == 0 || max_bytes_per_batch == 0 || entries.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current: Vec<StreamEntry> = Vec::new();
    let mut current_bytes: usize = 0;
    for entry in entries {
        let entry_bytes = entry.data.len();
        let at_count_cap = current.len() >= max_chunks_per_batch;
        let would_exceed_bytes =
            !current.is_empty() && current_bytes.saturating_add(entry_bytes) > max_bytes_per_batch;
        if at_count_cap || would_exceed_bytes {
            out.push(StreamBatch {
                entries: std::mem::take(&mut current),
            });
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(entry_bytes);
        current.push(entry);
    }
    if !current.is_empty() {
        out.push(StreamBatch { entries: current });
    }
    out
}

#[cfg(test)]
#[path = "../../tests/unit/transport/chunking/tests.rs"]
mod tests;
