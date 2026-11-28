//! Receiver-side chunk reassembly for oversized stream entries.
//!
//! Stream values that exceed the gRPC message size budget are split into
//! bounded chunks on the sender and reassembled here on the receiver.
//! Tree repair pages do NOT use this path — they are emitted already
//! bounded (≤ max_tree_repair_page_bytes) and ride the single-message
//! fast path.
//!
//! Semantics are at-most-once: if a chunk is lost, the partial assembly
//! is GC'd after `timeout`, and the application regenerates the value
//! on its next retry cycle. No retries or watermarks.

use std::{
    cmp::Ordering,
    mem::size_of,
    time::{Duration, Instant},
};

use bytes::Bytes;
use dashmap::DashMap;

/// Max concurrent in-flight assemblies. Prevents a peer from flooding
/// the map with partial assemblies for unique keys that never complete.
pub const DEFAULT_MAX_CONCURRENT_ASSEMBLIES: usize = 20;

/// Max total bytes held across all in-flight assemblies. Caps the
/// receiver-side memory independently of any sender-side payload choice.
pub const DEFAULT_MAX_ASSEMBLER_BYTES: usize = 512 * 1024 * 1024;

/// Hard cap on the chunk count advertised by a single chunk header.
/// AssemblyState allocates a `chunks` vector sized to `total`, so an
/// unvalidated peer-supplied `total = u32::MAX` would trigger a
/// multi-GB allocation before the byte-cap enforcer can react.
pub const MAX_TOTAL_CHUNKS: u32 = 1024;

/// Composite key scoping an assembly by sender peer + application key.
/// Two peers chunking the same key in flight don't collide.
type AssemblyKey = (String, String);

pub struct ChunkAssembler {
    assemblies: DashMap<AssemblyKey, AssemblyState>,
    max_concurrent: usize,
    max_bytes: usize,
}

struct AssemblyState {
    generation: u64,
    /// Count of distinct chunk indices recorded so far. Incremented only
    /// when a previously-empty slot is filled, so repeated receives of
    /// the same (generation, index) don't over-count. Completion is
    /// `received_count == chunks.len()`.
    received_count: u32,
    chunks: Vec<Option<Bytes>>,
    created_at: Instant,
}

impl AssemblyState {
    fn new(generation: u64, total: u32) -> Self {
        let n = total as usize;
        Self {
            generation,
            received_count: 0,
            chunks: vec![None; n],
            created_at: Instant::now(),
        }
    }

    fn is_complete(&self) -> bool {
        self.received_count as usize == self.chunks.len()
    }

    /// Total receiver-side memory footprint of this assembly — chunk
    /// payloads plus the per-slot overhead of the `chunks` vector.
    /// Included in the byte cap so the cap bounds real memory, not
    /// just payload bytes.
    fn bytes_held(&self) -> usize {
        let payload: usize = self.chunks.iter().flatten().map(|c| c.len()).sum();
        let overhead = self.chunks.len() * size_of::<Option<Bytes>>();
        payload + overhead
    }

    /// Return the assembled chunks as a fragmented buffer: each chunk
    /// stays in its own `Bytes` handle (no contiguous memcpy of the
    /// full payload).
    fn assemble(self) -> Vec<Bytes> {
        self.chunks.into_iter().flatten().collect()
    }
}

impl ChunkAssembler {
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_MAX_CONCURRENT_ASSEMBLIES,
            DEFAULT_MAX_ASSEMBLER_BYTES,
        )
    }

    pub fn with_limits(max_concurrent: usize, max_bytes: usize) -> Self {
        Self {
            assemblies: DashMap::new(),
            max_concurrent,
            max_bytes,
        }
    }

    /// Record an incoming chunk. Returns `Some(assembled)` once all chunks
    /// for the current generation have arrived; returns `None` otherwise.
    ///
    /// Assemblies are scoped by `(peer_id, key)` so two senders streaming
    /// chunked values under the same key don't collide through the
    /// generation compare. A single node-wide assembler serves all
    /// inbound streams; the peer-scope prevents one peer's newer-
    /// generation reset from wiping another peer's in-flight partial.
    ///
    /// Generation handling is a three-way compare: a newer generation
    /// resets the state (older partials discarded), an older generation
    /// is dropped (newer state kept), equal continues recording.
    /// Malformed chunks are dropped silently: `total == 0`,
    /// `total > MAX_TOTAL_CHUNKS`, or `index >= total`.
    pub fn receive_chunk(
        &self,
        peer_id: &str,
        key: &str,
        generation: u64,
        index: u32,
        total: u32,
        data: Bytes,
    ) -> Option<Vec<Bytes>> {
        if total == 0 || total > MAX_TOTAL_CHUNKS || index >= total {
            return None;
        }

        let asm_key: AssemblyKey = (peer_id.to_string(), key.to_string());

        // Record the chunk under the shard lock. Assembly happens outside
        // the guard so the lock is not held during the allocation+copy of
        // the full reassembled buffer.
        let completed = {
            let mut entry = self
                .assemblies
                .entry(asm_key.clone())
                .or_insert_with(|| AssemblyState::new(generation, total));

            match generation.cmp(&entry.generation) {
                Ordering::Greater => {
                    *entry = AssemblyState::new(generation, total);
                }
                Ordering::Less => {
                    return None;
                }
                Ordering::Equal => {
                    if entry.chunks.len() != total as usize {
                        return None;
                    }
                }
            }

            if entry.chunks[index as usize].is_none() {
                entry.received_count += 1;
            }
            entry.chunks[index as usize] = Some(data);

            entry.is_complete()
        };

        if !completed {
            self.enforce_bounds();
            return None;
        }

        // Atomically take ownership only if the state is still our
        // generation and still complete. If another thread has moved on
        // (newer generation reset the state after our chunk landed), we
        // return None — the newer generation is what matters now.
        self.assemblies
            .remove_if(&asm_key, |_, state| {
                state.generation == generation && state.is_complete()
            })
            .map(|(_, state)| state.assemble())
    }

    /// Evict partials until the receiver-side memory bounds are satisfied.
    /// Over the concurrent-assembly cap: drop the oldest (by created_at).
    /// Over the byte cap: drop the largest (by total bytes held). Checked
    /// after each non-completing receive; the completing path removes its
    /// own entry before returning, so no enforcement is needed there.
    fn enforce_bounds(&self) {
        // Cap checks and eviction target only incomplete assemblies;
        // transiently-complete entries awaiting remove_if must not count.
        while self.incomplete_count() > self.max_concurrent {
            let Some((k, gen)) = self.oldest_incomplete() else {
                break;
            };
            self.try_evict_stale(&k, gen);
        }
        while self.incomplete_bytes() > self.max_bytes {
            let Some((k, gen)) = self.largest_incomplete() else {
                break;
            };
            self.try_evict_stale(&k, gen);
        }
    }

    fn incomplete_count(&self) -> usize {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .count()
    }

    fn incomplete_bytes(&self) -> usize {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .map(|e| e.value().bytes_held())
            .sum()
    }

    fn try_evict_stale(&self, key: &AssemblyKey, gen: u64) {
        self.assemblies
            .remove_if(key, |_, s| s.generation == gen && !s.is_complete());
    }

    fn oldest_incomplete(&self) -> Option<(AssemblyKey, u64)> {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .min_by_key(|e| e.value().created_at)
            .map(|e| (e.key().clone(), e.value().generation))
    }

    fn largest_incomplete(&self) -> Option<(AssemblyKey, u64)> {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .max_by_key(|e| e.value().bytes_held())
            .map(|e| (e.key().clone(), e.value().generation))
    }

    /// Drop partial assemblies older than `timeout`. Collect-then-remove
    /// to avoid holding DashMap shard locks during mutation, and each
    /// removal re-checks the timeout so a concurrent receive that reset
    /// the entry to a new generation (fresh created_at) is spared.
    /// Complete assemblies are also skipped — they belong to an in-flight
    /// receive_chunk that is about to extract them via remove_if.
    pub fn gc(&self, timeout: Duration) {
        let expired = self.collect_expired(timeout);
        self.remove_stale(&expired, timeout);
    }

    fn collect_expired(&self, timeout: Duration) -> Vec<AssemblyKey> {
        self.assemblies
            .iter()
            .filter(|e| e.value().created_at.elapsed() >= timeout && !e.value().is_complete())
            .map(|e| e.key().clone())
            .collect()
    }

    /// Invalidate any pending partial assembly for `(peer_id, key)`.
    /// The single-chunk fast path calls this so a fresh value replaces
    /// leftover fragments from an older multi-chunk publish under the
    /// same key, instead of waiting for the 30 s GC to clean them up.
    /// No-op if no entry exists.
    pub fn drop_pending(&self, peer_id: &str, key: &str) {
        let asm_key: AssemblyKey = (peer_id.to_string(), key.to_string());
        self.assemblies.remove(&asm_key);
    }

    fn remove_stale(&self, keys: &[AssemblyKey], timeout: Duration) {
        for key in keys {
            self.assemblies.remove_if(key, |_, state| {
                state.created_at.elapsed() >= timeout && !state.is_complete()
            });
        }
    }

    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.assemblies.len()
    }

    #[cfg(test)]
    pub(crate) fn total_bytes(&self) -> usize {
        self.assemblies.iter().map(|e| e.value().bytes_held()).sum()
    }
}

impl Default for ChunkAssembler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transport/chunk_assembler/tests.rs"]
mod tests;
