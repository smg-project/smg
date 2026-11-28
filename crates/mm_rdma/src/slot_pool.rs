//! Pure slot bookkeeping over the arena's raw memory: the free-list, the leased
//! (slot_key -> slot) map, and the framed memcpy-into-slot. Deliberately holds NO
//! NIXL handle so the lease / reclaim / reuse policy (where the TTL-vs-READ race
//! lives) is unit-testable without a registered agent or any RDMA hardware.

use std::{
    collections::VecDeque,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use parking_lot::Mutex;

use crate::{FRAME_OVERHEAD, GEN_BYTES};

/// A leased slot, kept until the worker's READ-notif (tagged with the slot key)
/// or the TTL sweep returns it to the free list.
pub(crate) struct OccSlot {
    pub(crate) slot: u32,
    pub(crate) at: Instant,
}

pub(crate) struct SlotPool {
    /// Raw base address of the leaked arena allocation (usize => Send+Sync).
    base: usize,
    slot_bytes: usize,
    /// Available slot indices. A FIFO queue (not a LIFO stack) so reuse is
    /// round-robin across all slots: this maximizes the time before any freed slot
    /// is re-leased, widening the safety window against a slow worker READ still
    /// referencing a slot the reaper just reclaimed (a LIFO stack would hammer the
    /// same few slots and shorten that window).
    free: Mutex<VecDeque<u32>>,
    /// Leased slots keyed by the caller's slot key (free-on-notif + TTL reclaim).
    occupied: DashMap<i64, OccSlot>,
    /// Monotonic per-lease generation stamp (see [`crate::GEN_BYTES`]). Starts at 1
    /// so the zero-initialized arena (a never-written slot reads gen 0) can never
    /// match a real descriptor's gen.
    gen: AtomicU64,
}

impl SlotPool {
    pub(crate) fn new(base: usize, slot_bytes: usize, n_slots: usize) -> SlotPool {
        SlotPool {
            base,
            slot_bytes,
            free: Mutex::new((0..n_slots as u32).collect()),
            occupied: DashMap::new(),
            gen: AtomicU64::new(1),
        }
    }

    /// Lease a free slot for `slot_key`, frame `bytes` as `[gen][payload][gen]` into
    /// it, and record the lease at `now`. Returns `(slot, slot_addr, gen)`, or `None`
    /// if the framed image exceeds `slot_bytes` or no slot is free (caller -> inline).
    /// Recording the lease here (rather than in the caller) keeps the slot and its TTL
    /// timestamp atomic w.r.t. the write. `gen` is shipped in the descriptor and
    /// re-checked by the worker against the slot's stamps to reject a recycled/torn READ.
    pub(crate) fn lease_and_write(
        &self,
        slot_key: i64,
        bytes: &[u8],
        now: Instant,
    ) -> Option<(u32, u64, u64)> {
        // Reserve room for the generation stamps bracketing the payload.
        if bytes.len() + FRAME_OVERHEAD > self.slot_bytes {
            return None;
        }
        let slot = self.free.lock().pop_front()?;
        let addr = self.base + slot as usize * self.slot_bytes;
        // A fresh, monotonically increasing stamp for THIS lease; a later lease of the
        // same slot gets a strictly larger gen. Relaxed is fine: we only need
        // uniqueness, and the cross-process happens-before is the descriptor ship.
        let gen = self.gen.fetch_add(1, Ordering::Relaxed);
        let gen_le = gen.to_le_bytes();
        // SAFETY: `slot` is exclusively leased (popped from `free`, returned only via
        // free_slot_key); [addr, addr + FRAME_OVERHEAD + len) is within the arena
        // (len + FRAME_OVERHEAD <= slot_bytes) and disjoint from every other leased
        // slot; the worker reads it only after we ship the descriptor, and the TTL
        // keeps it leased past the worker's max READ window. The arena is raw, leaked
        // memory (no aliasing typed reference). So there is no data race and no
        // overlap. Write the header gen FIRST and the trailer gen LAST so a racing
        // reuse is visible at one stamp or the other (seqlock framing).
        #[expect(
            unsafe_code,
            reason = "seqlock gen-stamp write into the pre-registered NIXL slot arena"
        )]
        unsafe {
            let p = addr as *mut u8;
            std::ptr::copy_nonoverlapping(gen_le.as_ptr(), p, GEN_BYTES);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(GEN_BYTES), bytes.len());
            std::ptr::copy_nonoverlapping(
                gen_le.as_ptr(),
                p.add(GEN_BYTES + bytes.len()),
                GEN_BYTES,
            );
        }
        self.occupied.insert(slot_key, OccSlot { slot, at: now });
        Some((slot, addr as u64, gen))
    }

    /// Return the slot leased for `slot_key` to the free list (idempotent).
    pub(crate) fn free_slot_key(&self, slot_key: i64) {
        if let Some((_key, occ)) = self.occupied.remove(&slot_key) {
            self.free.lock().push_back(occ.slot);
        }
    }

    /// Reclaim every slot whose lease is older than `ttl` as of `now` (the lost-notif
    /// net). Returns the count reclaimed. `now`/`ttl` are parameters so the reclaim
    /// policy is testable without sleeping.
    pub(crate) fn reap_stale(&self, now: Instant, ttl: Duration) -> usize {
        let stale: Vec<i64> = self
            .occupied
            .iter()
            // Saturate rather than `duration_since` (which can panic if `now` is
            // ever observed before the lease stamp, e.g. cross-core clock skew).
            .filter(|e| now.saturating_duration_since(e.value().at) >= ttl)
            .map(|e| *e.key())
            .collect();
        let n = stale.len();
        for key in stale {
            self.free_slot_key(key);
        }
        n
    }
}

#[cfg(test)]
#[path = "../tests/unit/slot_pool/tests.rs"]
mod tests;
