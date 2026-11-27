use std::sync::Arc;

use super::*;
use crate::middleware::scheduler::Class;

// ── pack/unpack pure-function round trip ─────────────────────────

#[test]
fn test_pack_unpack_round_trip() {
    let samples: [[u16; 4]; 6] = [
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [u16::MAX, 0, u16::MAX, 0],
        [0, u16::MAX, 0, u16::MAX],
        [10_000, 20_000, 30_000, 40_000],
        [u16::MAX; 4],
    ];
    for s in samples {
        assert_eq!(unpack(pack(s)), s);
    }
}

// ── slots_available_to: reservation honored across classes ───────

#[test]
fn test_slots_available_to_subtracts_higher_class_reservation() {
    // Capacity 128, System holds a 32-slot reservation (lane 3), no
    // inflight. A Bulk admission may use at most 128 - 32 = 96.
    let counts = [0, 0, 0, 0];
    let reserved = [0, 0, 0, 32];
    assert_eq!(slots_available_to(counts, reserved, 128, Class::Bulk), 96);
    // The reserving class itself gets the full capacity — its own
    // reservation is "available to" it.
    assert_eq!(
        slots_available_to(counts, reserved, 128, Class::System),
        128
    );
}

#[test]
fn test_slots_available_to_ignores_same_class_reservation() {
    // A class's own reservation doesn't shrink its own headroom.
    let counts = [0, 0, 0, 0];
    let reserved = [32, 0, 0, 0];
    assert_eq!(slots_available_to(counts, reserved, 128, Class::Bulk), 128);
}

#[test]
fn test_slots_available_to_consumed_reservation_releases_headroom() {
    // Once a higher class is already consuming its reservation, the
    // reservation hold collapses 1:1 — Bulk regains those slots.
    let counts = [0, 0, 0, 32];
    let reserved = [0, 0, 0, 32];
    // used_total = 32, unavailable_to_bulk = max(0, 32-32) = 0
    // → 128 - 32 - 0 = 96
    assert_eq!(slots_available_to(counts, reserved, 128, Class::Bulk), 96);
}

#[test]
fn test_slots_available_to_returns_zero_when_used_meets_capacity() {
    let counts = [10, 10, 10, 10];
    let reserved = [0, 0, 0, 0];
    assert_eq!(slots_available_to(counts, reserved, 40, Class::Default), 0);
}

// ── try_acquire_slot / release_slot ──────────────────────────────

#[test]
fn test_try_acquire_increments_lane_then_release_decrements() {
    let pool = SlotPool::new(8, [0, 0, 0, 0]);
    assert!(pool.try_acquire(Class::Default));
    assert_eq!(pool.inflight(Class::Default), 1);
    pool.release(Class::Default);
    assert_eq!(pool.inflight(Class::Default), 0);
}

#[test]
fn test_try_acquire_respects_higher_class_reservation() {
    // Capacity 4, System reserves 2. Bulk can take at most 2.
    let pool = SlotPool::new(4, [0, 0, 0, 2]);
    assert!(pool.try_acquire(Class::Bulk));
    assert!(pool.try_acquire(Class::Bulk));
    assert!(
        !pool.try_acquire(Class::Bulk),
        "Bulk must not consume System's reservation"
    );
    // System itself still has room.
    assert!(pool.try_acquire(Class::System));
    assert!(pool.try_acquire(Class::System));
    assert!(!pool.try_acquire(Class::System), "now truly full");
}

#[test]
fn test_try_acquire_returns_false_when_full_without_spinning() {
    let pool = SlotPool::new(1, [0, 0, 0, 0]);
    assert!(pool.try_acquire(Class::Default));
    // Second call must return promptly, not spin forever.
    assert!(!pool.try_acquire(Class::Default));
}

#[test]
fn test_try_acquire_ignoring_reservations_bypasses_guard() {
    // Capacity 4, System reserves 4 (locks Bulk out under the guard).
    let pool = SlotPool::new(4, [0, 0, 0, 4]);
    assert!(!pool.try_acquire(Class::Bulk), "guarded acquire is blocked");
    assert!(
        pool.try_acquire_ignoring_reservations(Class::Bulk),
        "starvation-override path bypasses the reservation guard"
    );
}

#[test]
fn test_try_acquire_honors_live_capacity_shrink() {
    // Pre-fill, then shrink below the current inflight count. Any
    // subsequent try_acquire must refuse on first iteration because
    // the loop re-reads capacity (this would have committed against
    // the old larger ceiling if capacity were cached pre-loop).
    let pool = SlotPool::new(100, [0, 0, 0, 0]);
    for _ in 0..50 {
        assert!(pool.try_acquire(Class::Default));
    }
    pool.set_capacity(40); // used_total (50) > new capacity (40)
    assert!(
        !pool.try_acquire(Class::Default),
        "must refuse against the new (lower) capacity"
    );
}

#[test]
fn test_try_acquire_ignoring_reservations_honors_live_capacity_shrink() {
    let pool = SlotPool::new(100, [0, 0, 0, 0]);
    for _ in 0..50 {
        assert!(pool.try_acquire_ignoring_reservations(Class::Bulk));
    }
    pool.set_capacity(40);
    assert!(
        !pool.try_acquire_ignoring_reservations(Class::Bulk),
        "starvation-override acquire must also honor the new capacity"
    );
}

#[test]
fn test_release_saturates_at_zero() {
    // Defensive: an extra release shouldn't underflow the lane.
    let pool = SlotPool::new(1, [0, 0, 0, 0]);
    pool.release(Class::Default);
    assert_eq!(pool.inflight(Class::Default), 0);
}

// ── concurrent contention: no over- or under-admission ───────────

#[test]
fn test_concurrent_admission_admits_exactly_capacity() {
    // 200 threads racing to acquire one Default slot against a pool
    // sized for 64. The CAS loop must guarantee exactly 64 successes
    // — no over-admission (race) or under-admission (giving up too
    // early). Uses std::thread rather than tokio so the test
    // exercises real OS-level concurrency without a runtime.
    use std::thread;

    let pool = Arc::new(SlotPool::new(64, [0, 0, 0, 0]));
    let handles: Vec<_> = (0..200)
        .map(|_| {
            let pool = pool.clone();
            thread::spawn(move || pool.try_acquire(Class::Default))
        })
        .collect();
    let successes = handles
        .into_iter()
        .filter_map(|h| h.join().ok())
        .filter(|admitted| *admitted)
        .count();
    assert_eq!(successes, 64);
    assert_eq!(pool.inflight(Class::Default), 64);
}
