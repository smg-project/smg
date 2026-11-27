//! Packed slot accounting for the priority scheduler.
//!
//! Per-class in-flight counts are packed into a single `AtomicU64` (four
//! `u16` lanes) so an admission check can both verify availability and
//! claim a slot in a single compare-and-swap. This keeps the reservation
//! guard race-free under concurrent admissions to different classes — a
//! property that a per-class counter array cannot give us cheaply.
//!
//! Lane layout (low to high u16): `[Bulk, Default, Interactive, System]`,
//! indexed by [`Class`] cast to `usize`.

use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};

use super::Class;

/// Decode a packed `u64` into per-class in-flight counts, lane order
/// `[Bulk, Default, Interactive, System]`.
pub(super) fn unpack(packed: u64) -> [u16; 4] {
    [
        (packed & 0xFFFF) as u16,
        ((packed >> 16) & 0xFFFF) as u16,
        ((packed >> 32) & 0xFFFF) as u16,
        ((packed >> 48) & 0xFFFF) as u16,
    ]
}

/// Encode per-class in-flight counts into a single `u64`.
pub(super) fn pack(counts: [u16; 4]) -> u64 {
    u64::from(counts[0])
        | (u64::from(counts[1]) << 16)
        | (u64::from(counts[2]) << 32)
        | (u64::from(counts[3]) << 48)
}

/// How many slots a `class` admission may consume right now, given the
/// current per-class counts, reservations, and total capacity.
///
/// Higher-priority classes hold their reservations: a `Bulk` admission
/// cannot consume a slot that `System` has reserved but not yet used.
/// Once a higher class has actually consumed its reservation, the hold
/// collapses 1:1 and the slot returns to the shared pool.
///
/// A class's own reservation never shrinks its own headroom.
pub(super) fn slots_available_to(
    counts: [u16; 4],
    reserved: [u16; 4],
    capacity: u16,
    class: Class,
) -> u16 {
    let used_total: u16 = counts.iter().copied().fold(0, u16::saturating_add);
    let unavailable_to_us: u16 = Class::ALL
        .iter()
        .filter(|&&c| c > class)
        .map(|&c| reserved[c as usize].saturating_sub(counts[c as usize]))
        .fold(0, u16::saturating_add);
    capacity
        .saturating_sub(used_total)
        .saturating_sub(unavailable_to_us)
}

/// Slot pool: capacity, per-class reservations, and packed per-class
/// in-flight counts. The admission middleware acquires a slot here at
/// admission and releases it from the response-body wrapper.
pub struct SlotPool {
    capacity: AtomicU16,
    inflight_packed: AtomicU64,
    reserved: [AtomicU16; 4],
}

impl SlotPool {
    /// Build a pool with the given capacity and per-class reservations
    /// (lane order `[Bulk, Default, Interactive, System]`).
    pub fn new(capacity: u16, reserved: [u16; 4]) -> Self {
        Self {
            capacity: AtomicU16::new(capacity),
            inflight_packed: AtomicU64::new(0),
            reserved: reserved.map(AtomicU16::new),
        }
    }

    /// Current in-flight count for a class. Single atomic load + a few
    /// shifts; safe to call from any thread.
    pub fn inflight(&self, class: Class) -> u16 {
        unpack(self.inflight_packed.load(Ordering::Acquire))[class as usize]
    }

    /// Current total backend capacity. Read by `try_acquire` on every
    /// admission attempt.
    pub fn capacity(&self) -> u16 {
        self.capacity.load(Ordering::Acquire)
    }

    /// Update the total backend capacity. Returns the previous value.
    /// Called by the scheduler when the WorkerCapacity watch channel
    /// signals a change.
    pub fn set_capacity(&self, new_capacity: u16) -> u16 {
        self.capacity.swap(new_capacity, Ordering::AcqRel)
    }

    /// Reservation held by `class`.
    pub fn reserved(&self, class: Class) -> u16 {
        self.reserved[class as usize].load(Ordering::Acquire)
    }

    /// Update the reservation held by `class`. Used by the scheduler's
    /// capacity-shrink path to scale reservations down proportionally
    /// when `Σ reserved` would otherwise exceed the new capacity.
    pub fn set_reserved(&self, class: Class, new_reserved: u16) {
        self.reserved[class as usize].store(new_reserved, Ordering::Release);
    }

    fn snapshot_reserved(&self) -> [u16; 4] {
        [
            self.reserved[0].load(Ordering::Relaxed),
            self.reserved[1].load(Ordering::Relaxed),
            self.reserved[2].load(Ordering::Relaxed),
            self.reserved[3].load(Ordering::Relaxed),
        ]
    }

    /// Reservation-aware acquire. Returns `true` after a successful CAS;
    /// `false` if `class` has no headroom under the reservation guard.
    /// Bounded loop — re-reads on each failed CAS and exits as soon as
    /// the slot is genuinely full.
    ///
    /// `capacity` and the reservation snapshot are re-read on every
    /// iteration so a live `set_capacity` / `set_reserved` from the
    /// dispatcher can't be missed mid-retry. Otherwise a long-running
    /// contention loop could commit against the stale pre-shrink
    /// ceiling.
    pub fn try_acquire(&self, class: Class) -> bool {
        let lane = class as usize;
        let mut cur = self.inflight_packed.load(Ordering::Acquire);
        loop {
            let capacity = self.capacity.load(Ordering::Acquire);
            let reserved = self.snapshot_reserved();
            let mut counts = unpack(cur);
            if slots_available_to(counts, reserved, capacity, class) == 0 {
                return false;
            }
            counts[lane] = counts[lane].saturating_add(1);
            let new = pack(counts);
            match self.inflight_packed.compare_exchange_weak(
                cur,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Acquire ignoring the reservation guard — only the total-capacity
    /// ceiling applies. Used by the dispatcher's starvation override so
    /// a starved low-class waiter can consume a reserved-but-unused slot
    /// rather than wait indefinitely.
    ///
    /// As with `try_acquire`, `capacity` is re-read every iteration so
    /// the loop honors live `set_capacity` updates mid-retry.
    pub fn try_acquire_ignoring_reservations(&self, class: Class) -> bool {
        let lane = class as usize;
        let mut cur = self.inflight_packed.load(Ordering::Acquire);
        loop {
            let capacity = self.capacity.load(Ordering::Acquire);
            let mut counts = unpack(cur);
            let used_total: u16 = counts.iter().copied().fold(0, u16::saturating_add);
            if used_total >= capacity {
                return false;
            }
            counts[lane] = counts[lane].saturating_add(1);
            match self.inflight_packed.compare_exchange_weak(
                cur,
                pack(counts),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Release a previously acquired slot. Saturates at zero so a stray
    /// extra release cannot underflow the lane (defensive — production
    /// callers acquire and release in `SchedulerPermit::Drop`).
    ///
    /// Returns early without issuing a CAS when the lane is already
    /// zero. An unchanged-value atomic write would still trigger cache-
    /// line invalidation across cores via the coherence protocol, so
    /// the short-circuit matters on hot release paths.
    pub fn release(&self, class: Class) {
        let lane = class as usize;
        let mut cur = self.inflight_packed.load(Ordering::Acquire);
        loop {
            let counts = unpack(cur);
            if counts[lane] == 0 {
                return;
            }
            let mut new_counts = counts;
            new_counts[lane] -= 1;
            match self.inflight_packed.compare_exchange_weak(
                cur,
                pack(new_counts),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/middleware/scheduler/slots/tests.rs"]
mod tests;
