use std::{sync::Arc, time::Duration};

use super::*;
use crate::middleware::scheduler::{ClassConfig, PrioritySchedulerYaml};

fn default_settings() -> SchedulerSettings {
    SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None).unwrap()
}

fn rid(s: &str) -> RequestId {
    RequestId(s.to_string())
}

#[test]
fn test_new_succeeds_when_reservations_fit() {
    // Built-in defaults at capacity 256: desired Σ = 186 (System 32,
    // Interactive 128, Default ceil(0.10*256)=26) ≤ 256.
    let s = default_settings();
    assert!(PriorityScheduler::new(&s, 256).is_ok());
}

#[test]
fn test_new_rejects_when_reservations_exceed_capacity() {
    // Capacity 100: desired is System 32, Interactive min(128,100)=100,
    // Default 10 — Σ 142 > 100, so construction rejects (caller falls
    // back to legacy).
    let s = default_settings();
    let result = PriorityScheduler::new(&s, 100);
    assert!(matches!(
        result,
        Err(SchedulerInitError::ReservationsExceedCapacity {
            reserved: 142,
            capacity: 100
        })
    ));
}

#[test]
fn test_acquire_returns_permit_when_slot_available() {
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    let permit = scheduler.acquire_inflight(Class::Default, rid("req-1"));
    assert!(permit.is_some());
}

#[test]
fn test_acquire_returns_none_when_reservation_guard_blocks() {
    // Capacity 200, reservations 180 (System 32, Interactive 128, Default
    // ceil(0.10*200)=20). Bulk is the lowest class so it gets capacity -
    // 180 = 20 slots before the guard refuses further admissions.
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 200).unwrap();
    let mut held = Vec::new();
    for i in 0..20 {
        held.push(
            scheduler
                .acquire_inflight(Class::Bulk, rid(&format!("req-{i}")))
                .expect("under guard"),
        );
    }
    assert!(scheduler
        .acquire_inflight(Class::Bulk, rid("overflow"))
        .is_none());
}

#[tokio::test]
async fn test_permit_drop_releases_slot_and_notifies_dispatcher() {
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    let permit = scheduler
        .acquire_inflight(Class::Default, rid("req-1"))
        .expect("admitted");
    assert_eq!(scheduler.slot_pool.inflight(Class::Default), 1);

    let notified = scheduler.release_notify.notified();
    drop(permit);

    assert_eq!(scheduler.slot_pool.inflight(Class::Default), 0);
    tokio::time::timeout(Duration::from_millis(100), notified)
        .await
        .expect("release_notify fires on drop");
}

#[tokio::test]
async fn test_registry_inserts_on_acquire_and_removes_on_drop() {
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    let id = rid("req-1");
    let permit = scheduler
        .acquire_inflight(Class::Default, id.clone())
        .expect("admitted");
    assert!(scheduler.inflight_registry.read().contains_key(&id));
    drop(permit);
    assert!(!scheduler.inflight_registry.read().contains_key(&id));
}

// ── admit ────────────────────────────────────────────────────────

/// Build settings with zero reservations on every class (so we can
/// run admit tests against small capacities without tripping the
/// reserved-vs-capacity guard) and an override on one class's
/// queue_size + queue_timeout.
fn settings_with(class: Class, queue_size: u32, queue_timeout_secs: u64) -> SchedulerSettings {
    let mut classes = HashMap::new();
    for c in Class::ALL {
        let mut cfg = ClassConfig::default_for(c);
        cfg.reserved_floor = 0;
        cfg.reserved_per_slot = 0.0;
        if c == class {
            cfg.queue_size = queue_size;
            cfg.queue_timeout_secs = queue_timeout_secs;
        }
        classes.insert(c, cfg);
    }
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: HashMap::new(),
    };
    SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap()
}

#[tokio::test]
async fn test_admit_fast_path_when_slot_available() {
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    let outcome = scheduler
        .admit(Class::Default, rid("req-1"), CancellationToken::new())
        .await;
    assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
}

#[tokio::test]
async fn test_admit_rejects_when_queue_full() {
    // Capacity 1 (slot held), Default queue_size=1 (one waiter pre-stuffed
    // directly into the queue): the next admit takes the slow path and
    // hits a full queue.
    let s = settings_with(Class::Default, 1, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .expect("admitted directly");

    let (queued_tx, _queued_rx) = oneshot::channel();
    let queued = Waiter::new(
        Class::Default,
        CancellationToken::new(),
        rid("queued"),
        queued_tx,
    );
    scheduler.class_queues[Class::Default as usize]
        .try_enqueue(queued)
        .expect("queue had room for one");

    let outcome = scheduler
        .admit(Class::Default, rid("w2"), CancellationToken::new())
        .await;
    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::QueueFull)
    ));
}

#[tokio::test(start_paused = true)]
async fn test_admit_rejects_on_queue_timeout() {
    // Capacity 1 forces enqueue; queue_timeout=1s; never release → QueueTimeout.
    let s = settings_with(Class::Default, 16, 1);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .expect("admitted directly");

    let admit_future = scheduler.admit(Class::Default, rid("w1"), CancellationToken::new());
    let outcome = admit_future.await;
    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
    ));
}

#[tokio::test]
async fn test_admit_timeout_kicks_dispatcher_to_reap_stale_head() {
    // After admit timeout marks the queued waiter cancelled, the
    // dispatcher must be nudged so drop_cancelled_head runs
    // promptly — otherwise a queue_size=1 queue would reject the
    // next admit as QueueFull until the next periodic tick.
    let s = settings_with(Class::Default, 1, 1); // queue_size=1, timeout=1s
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    scheduler.spawn_dispatcher(dummy_capacity_watch(1));
    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();

    let outcome = scheduler
        .admit(Class::Default, rid("w1"), CancellationToken::new())
        .await;
    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
    ));

    // Let the dispatcher's notify_one wakeup run drop_cancelled_head.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        scheduler.class_queues[Class::Default as usize].depth(),
        0,
        "dispatcher should reap the cancelled head promptly"
    );
}

#[tokio::test(start_paused = true)]
async fn test_admit_timeout_marks_waiter_for_gc() {
    // After QueueTimeout fires, the Waiter still sits in the FIFO
    // until something reaps it. The cancel token on the Waiter must
    // be fired so a future drop_cancelled_head call evicts it,
    // preventing false QueueFull on later admissions.
    let s = settings_with(Class::Default, 16, 1);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .expect("admitted directly");

    let outcome = scheduler
        .admit(Class::Default, rid("w1"), CancellationToken::new())
        .await;
    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
    ));

    // The waiter is still in the queue (one entry), but cancelled.
    assert_eq!(scheduler.class_queues[Class::Default as usize].depth(), 1);
    scheduler.class_queues[Class::Default as usize].drop_cancelled_head();
    assert_eq!(
        scheduler.class_queues[Class::Default as usize].depth(),
        0,
        "drop_cancelled_head must reap the timed-out waiter"
    );
}

#[tokio::test]
async fn test_admit_rejects_on_client_cancel() {
    // Pre-cancel the token before admit is called; the fast path fails
    // (slot held), the slow path enqueues, then the cancel arm of select!
    // fires immediately.
    let s = settings_with(Class::Default, 16, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .expect("admitted directly");

    let cancel = CancellationToken::new();
    cancel.cancel();
    let outcome = scheduler.admit(Class::Default, rid("w1"), cancel).await;
    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::ClientCancelled)
    ));
}

// ── wake_next_waiter / dispatcher ───────────────────────────────

fn enqueue_waiter(
    scheduler: &Arc<PriorityScheduler>,
    class: Class,
) -> oneshot::Receiver<SchedulerPermit> {
    let (tx, rx) = oneshot::channel();
    let waiter = Waiter::new(class, CancellationToken::new(), rid("queued"), tx);
    scheduler.class_queues[class as usize]
        .try_enqueue(waiter)
        .expect("queue had room");
    rx
}

#[tokio::test]
async fn test_send_to_head_skips_waiters_with_closed_receivers() {
    // Stuff a queue with one waiter whose receiver is dropped (admit
    // has already timed out / been cancelled) followed by a live
    // waiter. wake_next_waiter should admit the live waiter without
    // touching the inflight registry for the dead one.
    let s = settings_with(Class::Default, 8, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();

    // Dead waiter — drop the receiver immediately.
    let (dead_tx, dead_rx) = oneshot::channel::<SchedulerPermit>();
    drop(dead_rx);
    let dead = Waiter::new(
        Class::Default,
        CancellationToken::new(),
        rid("dead"),
        dead_tx,
    );
    scheduler.class_queues[Class::Default as usize]
        .try_enqueue(dead)
        .unwrap();

    // Live waiter.
    let (live_tx, mut live_rx) = oneshot::channel::<SchedulerPermit>();
    let live = Waiter::new(
        Class::Default,
        CancellationToken::new(),
        rid("live"),
        live_tx,
    );
    scheduler.class_queues[Class::Default as usize]
        .try_enqueue(live)
        .unwrap();

    // Release the slot.
    drop(_held);
    assert!(scheduler.wake_next_waiter(), "live waiter admitted");

    let permit = live_rx.try_recv().expect("live permit delivered");
    assert_eq!(permit.handle().request_id().0, "live");
    // Registry should hold only the live request — never the dead one.
    let registry = scheduler.inflight_registry.read();
    assert!(registry.contains_key(&rid("live")));
    assert!(!registry.contains_key(&rid("dead")));
}

#[test]
fn test_wake_next_waiter_returns_false_when_no_slot_available() {
    let s = settings_with(Class::Default, 8, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();
    let _rx = enqueue_waiter(&scheduler, Class::Default);
    assert!(!scheduler.wake_next_waiter(), "no slot to give");
}

#[tokio::test]
async fn test_wake_next_waiter_admits_queued_waiter() {
    let s = settings_with(Class::Default, 8, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();
    let mut rx = enqueue_waiter(&scheduler, Class::Default);
    drop(held);
    assert!(scheduler.wake_next_waiter());
    let permit = rx.try_recv().expect("permit delivered");
    assert_eq!(permit.handle().class(), Class::Default);
}

#[tokio::test]
async fn test_wake_next_waiter_honors_priority_order() {
    // Interactive beats Bulk when both are queued and a slot frees.
    let s = settings_with(Class::Bulk, 8, 60); // Bulk queue=8 (interactive defaults too)
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();

    let mut bulk_rx = enqueue_waiter(&scheduler, Class::Bulk);
    let mut interactive_rx = enqueue_waiter(&scheduler, Class::Interactive);

    drop(held);
    assert!(scheduler.wake_next_waiter());
    // Interactive should be served first; bulk still waiting.
    assert!(interactive_rx.try_recv().is_ok());
    assert!(bulk_rx.try_recv().is_err());
}

#[tokio::test]
async fn test_dispatcher_periodic_tick_promotes_stale_waiter_without_release() {
    // No slot release ever fires. The dispatcher's periodic tick
    // must wake on its own, observe that the Bulk head has aged
    // past its starvation_threshold, and admit via the override.
    let mut classes = HashMap::new();
    for c in Class::ALL {
        let mut cfg = ClassConfig::default_for(c);
        cfg.reserved_floor = 0;
        cfg.reserved_per_slot = 0.0;
        // Tight thresholds so the test runs fast.
        cfg.starvation_threshold_secs = 1;
        if c == Class::Bulk {
            cfg.queue_size = 4;
        }
        // Interactive reserves the entire capacity, locking Bulk out
        // of the normal-priority admission path.
        if c == Class::Interactive {
            cfg.reserved_floor = 1;
        }
        classes.insert(c, cfg);
    }
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: HashMap::new(),
    };
    let settings =
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
    let scheduler = PriorityScheduler::new(&settings, 1).unwrap();
    scheduler.spawn_dispatcher(dummy_capacity_watch(1));

    // Admit a Bulk request. No slot release ever fires; the only way
    // for it to be admitted is the dispatcher's periodic tick + the
    // starvation override.
    let outcome = tokio::time::timeout(
        Duration::from_secs(3),
        scheduler.admit(Class::Bulk, rid("bulk"), CancellationToken::new()),
    )
    .await
    .expect("admit completes within timeout");
    assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
}

#[tokio::test]
async fn test_starvation_override_promotes_stale_bulk_head() {
    // Bulk starvation_threshold=0s (anything older than 0 wins), with
    // capacity entirely reserved by Interactive — without the
    // override Bulk could never advance.
    let mut classes = HashMap::new();
    for c in Class::ALL {
        let mut cfg = ClassConfig::default_for(c);
        cfg.reserved_floor = 0;
        cfg.reserved_per_slot = 0.0;
        // Set tiny starvation threshold for Bulk so an immediate
        // enqueue is "stale" by the time wake fires.
        if c == Class::Bulk {
            cfg.starvation_threshold_secs = 1;
            cfg.queue_size = 4;
        }
        // Interactive holds the full capacity in reservation.
        if c == Class::Interactive {
            cfg.reserved_floor = 1;
        }
        classes.insert(c, cfg);
    }
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: HashMap::new(),
    };
    let settings =
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();

    let scheduler = PriorityScheduler::new(&settings, 1).unwrap();
    let mut bulk_rx = enqueue_waiter(&scheduler, Class::Bulk);

    // Sleep past the starvation threshold so head_age > threshold.
    tokio::time::sleep(Duration::from_millis(1100)).await;

    // No regular path admits Bulk under the Interactive reservation,
    // but the starvation override should.
    assert!(scheduler.wake_next_waiter(), "starvation override fires");
    assert!(bulk_rx.try_recv().is_ok());
}

fn dummy_capacity_watch(initial: u16) -> watch::Receiver<u16> {
    // Construct a watch receiver whose sender is intentionally kept
    // alive for the test's duration. Leaking is fine — these tests
    // run in isolation.
    let (tx, rx) = watch::channel(initial);
    std::mem::forget(tx);
    rx
}

#[tokio::test]
async fn test_spawn_dispatcher_admits_queued_waiter_on_release() {
    let s = settings_with(Class::Default, 8, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    scheduler.spawn_dispatcher(dummy_capacity_watch(1));

    let held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();

    // Kick off an admit that must go to the slow path.
    let scheduler_for_admit = Arc::clone(&scheduler);
    let admit_future = async move {
        scheduler_for_admit
            .admit(Class::Default, rid("queued"), CancellationToken::new())
            .await
    };

    // Race: release the slot, then await admit. The dispatcher
    // should observe the release and admit the queued waiter.
    let (admit_outcome, ()) = tokio::join!(admit_future, async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(held);
    });

    assert!(matches!(admit_outcome, AdmitOutcome::Admitted(_)));
}

// ── apply_new_capacity / capacity watch ─────────────────────────

#[test]
fn test_apply_new_capacity_grow_fires_release_notify() {
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    // Notify already has whatever permits prior tests fired; reset by
    // taking a fresh notified() before the apply.
    let notified = scheduler.release_notify.notified();
    scheduler.apply_new_capacity(512);
    assert_eq!(scheduler.slot_pool.capacity(), 512);
    // Grow path must have signaled the dispatcher.
    tokio::pin!(notified);
    let polled = futures::FutureExt::now_or_never(notified);
    assert!(polled.is_some(), "release_notify fires on capacity grow");
}

#[test]
fn test_apply_new_capacity_shrink_clamps_reservations_proportionally() {
    // Built-in defaults at capacity 80: desired is System 32, Interactive
    // 80 (the 128 floor capped at capacity), Default ceil(0.10*80)=8 —
    // Σ 120 > 80. Every class keeps 80/120 of its reservation, rounded
    // down (System 21, Interactive 53, Default 5, Bulk 0), and the
    // remaining slot is shared.
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    scheduler.apply_new_capacity(80);
    assert_eq!(scheduler.slot_pool.capacity(), 80);
    let new_total: u32 = Class::ALL
        .iter()
        .map(|c| u32::from(scheduler.slot_pool.reserved(*c)))
        .sum();
    assert!(
        new_total < 80,
        "clamped reservations ({new_total}) must leave room"
    );
    assert_eq!(scheduler.slot_pool.reserved(Class::System), 21);
    assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 53);
    assert_eq!(scheduler.slot_pool.reserved(Class::Default), 5);
    assert_eq!(scheduler.slot_pool.reserved(Class::Bulk), 0);
}

#[test]
fn test_apply_new_capacity_restores_reservations_after_recovery() {
    // Reservations are recomputed from the floors + shares on every
    // capacity change, so a shrink-then-recover fully restores them (no
    // one-way degrade), and the share tracks a growing fleet.
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    // cap 256: System 32, Interactive max(128, 64)=128, Default ceil(25.6)=26.
    assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 128);
    assert_eq!(scheduler.slot_pool.reserved(Class::Default), 26);

    // Shrink to 80: desired Σ 120 > 80 → the proportional clamp (System
    // 21, Interactive 53, Default 5, Bulk 0; one slot shared).
    scheduler.apply_new_capacity(80);
    assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 53);
    assert_eq!(scheduler.slot_pool.reserved(Class::Default), 5);

    // Recover to 256 → fully restored, not stuck at the clamped values.
    scheduler.apply_new_capacity(256);
    assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 128);
    assert_eq!(scheduler.slot_pool.reserved(Class::Default), 26);

    // Grow past Interactive's floor: at 800 the 0.25 share wins (200 > 128)
    // and Default's 0.10 share is 80 — reservations track the larger fleet.
    scheduler.apply_new_capacity(800);
    assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 200);
    assert_eq!(scheduler.slot_pool.reserved(Class::Default), 80);
}

#[test]
fn test_apply_new_capacity_no_op_when_value_unchanged() {
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    let before = scheduler.slot_pool.reserved(Class::Interactive);
    scheduler.apply_new_capacity(256);
    assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), before);
}

// ── reservation math helpers ──────────────────────────────────────

#[test]
fn test_desired_reservations_floor_then_share() {
    // Indexed by `class as usize`: [Bulk, Default, Interactive, System].
    let floor = [0u16, 0, 128, 32];
    let per_slot = [0.0f64, 0.10, 0.25, 0.0];
    // cap 256: floor wins for Interactive (128 > 64); Default share = 26.
    let d = desired_reservations(floor, &per_slot, 256);
    assert_eq!(d[Class::System as usize], 32);
    assert_eq!(d[Class::Interactive as usize], 128);
    assert_eq!(d[Class::Default as usize], 26);
    assert_eq!(d[Class::Bulk as usize], 0);
    // cap 800: Interactive share (200) overtakes its floor; Default = 80.
    let d = desired_reservations(floor, &per_slot, 800);
    assert_eq!(d[Class::Interactive as usize], 200);
    assert_eq!(d[Class::Default as usize], 80);
    // Each class is capped at capacity (floor 128 capped to 10).
    let d = desired_reservations(floor, &per_slot, 10);
    assert_eq!(d[Class::Interactive as usize], 10);
}

#[test]
fn test_clamp_reservations_proportional() {
    // [Bulk, Default, Interactive, System], Σ 185 > 100: every class
    // keeps 100/185 of its reservation, rounded down; 2 slots are shared.
    let out = clamp_reservations_to_capacity([5, 20, 128, 32], 100);
    assert_eq!(out[Class::System as usize], 17);
    assert_eq!(out[Class::Interactive as usize], 69);
    assert_eq!(out[Class::Default as usize], 10);
    assert_eq!(out[Class::Bulk as usize], 2);
    assert_eq!(out.iter().map(|&r| u32::from(r)).sum::<u32>(), 98);
    // Fits already: every class keeps its full desired value.
    assert_eq!(
        clamp_reservations_to_capacity([1, 2, 3, 4], 100),
        [1, 2, 3, 4]
    );
}

#[test]
fn test_clamp_on_small_pools_leaves_every_class_headroom() {
    // The built-in floors on a pool of 4 (the issue's bed before its
    // second worker was counted): desired is System 4, Interactive 4
    // (both capped at capacity), Default ceil(0.4)=1, Bulk 0 — Σ 9. The
    // old priority-ordered fill reserved all four for System, which
    // carries no traffic, and every other class was admitted only by the
    // starvation override. Now System and Interactive keep 4/9 of their
    // reservation (1 each), 2 slots are shared, and a request of any
    // class finds a slot on an idle pool.
    let floor = [0u16, 0, 128, 32];
    let per_slot = [0.0f64, 0.10, 0.25, 0.0];
    for (capacity, expected, shared) in [
        (4u16, [0u16, 0, 1, 1], 2u16),
        (16, [0, 0, 7, 7], 2),
        (8, [0, 0, 3, 3], 2),
    ] {
        let desired = desired_reservations(floor, &per_slot, capacity);
        let out = clamp_reservations_to_capacity(desired, capacity);
        assert_eq!(out, expected, "capacity {capacity}");
        let reserved: u16 = out.iter().sum();
        assert_eq!(capacity - reserved, shared, "capacity {capacity}");
        for class in Class::ALL {
            assert!(
                super::super::slots::slots_available_to([0; 4], out, capacity, class) >= 1,
                "capacity {capacity}: {class:?} has headroom on an idle pool"
            );
        }
    }
    // A pool large enough for the floors + shares is not clamped at all:
    // 1024 → System 32, Interactive 256, Default 103, Bulk 0 (Σ 391).
    let desired = desired_reservations(floor, &per_slot, 1024);
    assert_eq!(desired, [0, 103, 256, 32]);
    assert_eq!(clamp_reservations_to_capacity(desired, 1024), desired);
}

#[test]
fn test_clamp_leaves_one_slot_unreserved_when_the_scaled_shares_fill_the_pool() {
    // [Bulk, Default, Interactive, System] = [0, 2, 4, 4] at 5: every
    // share divides exactly ([0, 1, 2, 2] = 5), which would reserve the
    // whole pool and lock Bulk out until the starvation override; one
    // slot comes back from the largest reservation (System on the tie).
    assert_eq!(
        clamp_reservations_to_capacity([0, 2, 4, 4], 5),
        [0, 1, 2, 1]
    );
    // A pool of one slot: nothing is reserved, the slot is shared.
    assert_eq!(
        clamp_reservations_to_capacity([0, 1, 8, 8], 1),
        [0, 0, 0, 0]
    );
    // Capacity zero stays zero everywhere.
    assert_eq!(
        clamp_reservations_to_capacity([0, 1, 8, 8], 0),
        [0, 0, 0, 0]
    );
}

#[test]
fn test_one_request_of_any_class_is_admitted_on_a_small_idle_pool() {
    // Built at 1024 (the legacy fallback before the workers report), then
    // the fleet reports 4 slots: one request of each class is admitted at
    // once on the idle pool, and the pool holds four at a time.
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 1024).unwrap();
    scheduler.apply_new_capacity(4);
    for class in Class::ALL {
        let permit = scheduler.acquire_inflight(class, rid("one"));
        assert!(
            permit.is_some(),
            "{class:?} is admitted on an idle pool of 4"
        );
        drop(permit);
    }
    let held: Vec<_> = (0..2)
        .map(|i| scheduler.acquire_inflight(Class::Bulk, rid(&format!("b{i}"))))
        .collect();
    assert!(
        held.iter().all(Option::is_some),
        "the 2 shared slots serve Bulk"
    );
    assert!(
        scheduler.acquire_inflight(Class::Bulk, rid("b2")).is_none(),
        "the slots reserved for Interactive and System stay theirs"
    );
    assert!(
        scheduler
            .acquire_inflight(Class::Interactive, rid("i0"))
            .is_some(),
        "Interactive still has its reserved slot"
    );
}

#[test]
fn test_small_pool_gives_interactive_more_headroom_than_bulk_without_locking_bulk_out() {
    // Two workers of four slots (the issue's bed): capacity 8, desired
    // Σ 17 → System 3, Interactive 3, Default 0, Bulk 0, 2 shared.
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 1024).unwrap();
    scheduler.apply_new_capacity(8);
    let reserved = Class::ALL.map(|c| scheduler.slot_pool.reserved(c));
    let room = |class: Class| super::super::slots::slots_available_to([0; 4], reserved, 8, class);
    assert_eq!(room(Class::Bulk), 2, "Bulk is not locked out");
    assert_eq!(room(Class::Default), 2);
    assert_eq!(
        room(Class::Interactive),
        5,
        "Interactive has the shorter wait"
    );
    assert_eq!(room(Class::System), 8);
}

#[tokio::test]
async fn test_dropped_queued_admission_is_recorded_as_client_cancelled() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };

    use crate::observability::metrics::test_support::render_with_recorder;

    let rendered = render_with_recorder(|| {
        // One slot, held; the next Bulk admission queues. The client goes
        // away: axum drops the middleware's future and the queued admit
        // with it, without firing its cancel token.
        let s = settings_with(Class::Bulk, 8, 30);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Bulk, rid("held"))
            .expect("the slot");
        let admit = scheduler.admit(Class::Bulk, rid("queued"), CancellationToken::new());
        let mut admit = Box::pin(admit);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(
            matches!(admit.as_mut().poll(&mut cx), Poll::Pending),
            "the admission waits in the queue"
        );
        assert_eq!(scheduler.class_queues[Class::Bulk as usize].depth(), 1);
        drop(admit);
        // The abandoned waiter is marked for the queue's head GC.
        scheduler.class_queues[Class::Bulk as usize].drop_cancelled_head();
        assert_eq!(scheduler.class_queues[Class::Bulk as usize].depth(), 0);
    });
    assert!(
        rendered
            .contains("smg_scheduler_admit_total{class=\"bulk\",outcome=\"client_cancelled\"} 1"),
        "the abandoned wait is an admission outcome: {rendered}"
    );
    assert!(
        rendered.contains("smg_scheduler_queue_wait_seconds_count{class=\"bulk\"} 1"),
        "the abandoned wait is recorded: {rendered}"
    );
}

#[tokio::test]
async fn test_dispatcher_keeps_serving_after_capacity_watch_sender_drops() {
    // After the WorkerCapacity sender is dropped, the dispatcher must
    // continue handling release events from release_notify. The earlier
    // implementation hot-looped on the closed watch arm; this test
    // exercises the post-drop path end-to-end.
    let s = settings_with(Class::Default, 8, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let (capacity_tx, capacity_rx) = watch::channel(1u16);
    scheduler.spawn_dispatcher(capacity_rx);

    drop(capacity_tx); // close the watch — the failing branch trigger.
                       // Yield so the dispatcher observes the drop and disables that arm.
    tokio::time::sleep(Duration::from_millis(10)).await;

    let held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();
    let scheduler_for_admit = Arc::clone(&scheduler);
    let admit_future = async move {
        scheduler_for_admit
            .admit(Class::Default, rid("queued"), CancellationToken::new())
            .await
    };
    let (outcome, ()) = tokio::join!(admit_future, async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(held);
    });
    assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
}

#[tokio::test]
async fn test_capacity_watch_grow_drains_queue() {
    // Capacity 1 (slot held), queue has one waiter. Grow capacity to
    // 2 via the watch — dispatcher should drain the queued waiter.
    let s = settings_with(Class::Default, 8, 60);
    let scheduler = PriorityScheduler::new(&s, 1).unwrap();
    let (tx, rx) = watch::channel(1u16);
    scheduler.spawn_dispatcher(rx);

    let _held = scheduler
        .acquire_inflight(Class::Default, rid("held"))
        .unwrap();

    let scheduler_for_admit = Arc::clone(&scheduler);
    let admit_future = async move {
        scheduler_for_admit
            .admit(Class::Default, rid("queued"), CancellationToken::new())
            .await
    };

    let (outcome, ()) = tokio::join!(admit_future, async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        tx.send(2).unwrap();
    });

    assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
    assert_eq!(scheduler.slot_pool.capacity(), 2);
}

#[tokio::test]
async fn test_permit_keeps_scheduler_alive_via_arc() {
    let s = default_settings();
    let scheduler = PriorityScheduler::new(&s, 256).unwrap();
    let weak = Arc::downgrade(&scheduler);
    let permit = scheduler
        .acquire_inflight(Class::Default, rid("req-1"))
        .expect("admitted");
    drop(scheduler);
    // Permit holds a strong ref, so weak still upgrades.
    assert!(weak.upgrade().is_some());
    drop(permit);
    // All strong refs gone now.
    assert!(weak.upgrade().is_none());
}

// ── M3: preemption ────────────────────────────────────────────────

/// All classes reserved=0 (so acquire is purely capacity-bound),
/// short queue timeout (so declined-preempt enqueues resolve fast),
/// can_preempt left at defaults (System/Interactive true, others false).
fn preempt_settings() -> SchedulerSettings {
    let mut classes = HashMap::new();
    for c in Class::ALL {
        let mut cfg = ClassConfig::default_for(c);
        cfg.reserved_floor = 0;
        cfg.reserved_per_slot = 0.0;
        cfg.queue_size = 8;
        cfg.queue_timeout_secs = 1;
        classes.insert(c, cfg);
    }
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: HashMap::new(),
    };
    SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap()
}

#[test]
fn test_find_victim_empty_registry_returns_none() {
    let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
    assert!(sched.find_preemption_victim(Class::Interactive).is_none());
}

#[test]
fn test_find_victim_returns_lower_class_pre_ttft() {
    let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
    let _bulk = sched.acquire_inflight(Class::Bulk, rid("b")).unwrap();
    let v = sched
        .find_preemption_victim(Class::Interactive)
        .expect("victim");
    assert_eq!(v.class(), Class::Bulk);
}

#[test]
fn test_find_victim_prefers_lowest_class() {
    let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
    let _def = sched.acquire_inflight(Class::Default, rid("d")).unwrap();
    let _bulk = sched.acquire_inflight(Class::Bulk, rid("b")).unwrap();
    // Candidate Interactive: Default and Bulk both qualify; Bulk (lowest) wins.
    let v = sched
        .find_preemption_victim(Class::Interactive)
        .expect("victim");
    assert_eq!(v.class(), Class::Bulk);
}

#[test]
fn test_find_victim_skips_post_ttft() {
    let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
    let _def = sched.acquire_inflight(Class::Default, rid("d")).unwrap();
    let bulk = sched.acquire_inflight(Class::Bulk, rid("b")).unwrap();
    bulk.handle().try_mark_first_byte(5); // Bulk no longer preemptible
    let v = sched
        .find_preemption_victim(Class::Interactive)
        .expect("victim");
    assert_eq!(
        v.class(),
        Class::Default,
        "Bulk past TTFT is skipped; Default chosen"
    );
}

#[test]
fn test_find_victim_none_for_lowest_class() {
    let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
    let _def = sched.acquire_inflight(Class::Default, rid("d")).unwrap();
    // Bulk is the lowest class — nothing is strictly lower to preempt.
    assert!(sched.find_preemption_victim(Class::Bulk).is_none());
}

#[test]
fn test_permit_try_mark_first_byte_locks_out_preempt() {
    let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
    let permit = sched.acquire_inflight(Class::Default, rid("x")).unwrap();
    assert!(permit.try_mark_first_byte());
    assert!(
        !permit.handle().try_mark_preempted(),
        "preempt must lose after TTFT is marked"
    );
}

#[test]
fn test_permit_cancel_token_reflects_handle_cancel() {
    let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
    let permit = sched.acquire_inflight(Class::Bulk, rid("x")).unwrap();
    let tok = permit.cancel_token();
    assert!(!tok.is_cancelled());
    assert!(permit.handle().try_mark_preempted());
    permit.handle().cancel();
    assert!(tok.is_cancelled());
}

#[tokio::test]
async fn test_preempt_admits_by_cancelling_lower_class() {
    // Capacity 1, full with a pre-TTFT Bulk request. An Interactive
    // admission (can_preempt) cancels it; a watcher drops the victim's
    // permit on cancel (mimicking the handler unwinding), freeing the
    // slot for the Interactive waiter.
    let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
    let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
    let victim_cancel = victim.cancel_token();

    let sched_admit = Arc::clone(&sched);
    let admit_fut = async move {
        sched_admit
            .admit(Class::Interactive, rid("vip"), CancellationToken::new())
            .await
    };
    let dropper = async move {
        victim_cancel.cancelled().await;
        drop(victim); // handler unwinds → slot frees
    };
    let (outcome, ()) = tokio::join!(admit_fut, dropper);
    assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
}

#[tokio::test]
async fn test_preempt_skipped_when_caller_already_cancelled() {
    // A preempt-capable admission whose own client has already
    // disconnected must not cancel a lower-class victim: it returns
    // ClientCancelled and leaves the victim untouched.
    let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
    let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
    let victim_cancel = victim.cancel_token();

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let outcome = sched.admit(Class::Interactive, rid("vip"), cancelled).await;

    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::ClientCancelled)
    ));
    assert!(
        !victim_cancel.is_cancelled(),
        "victim must not be preempted"
    );
    assert_eq!(sched.inflight_for_test(Class::Bulk), 1);
}

#[test]
fn test_class_pressure_takes_worse_of_queue_and_slot() {
    // preempt_settings reserves 0 for every class, so higher_reserved=0
    // and slot pressure is inflight/capacity.
    let sched = PriorityScheduler::new(&preempt_settings(), 100).unwrap();
    // Queue pressure (8/10) dominates slot pressure (10/100).
    assert!((sched.class_pressure(Class::Default, 10, 8, 10, 100) - 0.8).abs() < 1e-9);
    // Slot pressure (50/100) dominates queue pressure (1/10).
    assert!((sched.class_pressure(Class::Default, 50, 1, 10, 100) - 0.5).abs() < 1e-9);
    // Clamped to 1.0 when oversubscribed.
    assert!((sched.class_pressure(Class::Default, 200, 100, 10, 100) - 1.0).abs() < 1e-9);
    // No queue limit and no inflight → zero, no div-by-zero.
    assert_eq!(sched.class_pressure(Class::Bulk, 0, 0, 0, 100), 0.0);
}

#[tokio::test(start_paused = true)]
async fn test_preempt_declined_when_only_victim_past_ttft() {
    // The single lower-class inflight has already emitted its first
    // byte, so it is not a valid victim. The Interactive admission must
    // NOT cancel it; it falls through to enqueue and times out.
    let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
    let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
    victim.handle().try_mark_first_byte(5);
    let victim_cancel = victim.cancel_token();

    let outcome = sched
        .admit(Class::Interactive, rid("vip"), CancellationToken::new())
        .await;
    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
    ));
    assert!(
        !victim_cancel.is_cancelled(),
        "a post-TTFT request must never be cancelled"
    );
}

#[tokio::test(start_paused = true)]
async fn test_non_preempting_class_does_not_cancel() {
    // Default has can_preempt=false; even with a Bulk victim available
    // it must not attempt preemption.
    let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
    let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
    let victim_cancel = victim.cancel_token();

    let outcome = sched
        .admit(Class::Default, rid("plain"), CancellationToken::new())
        .await;
    assert!(matches!(
        outcome,
        AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
    ));
    assert!(
        !victim_cancel.is_cancelled(),
        "can_preempt=false class must not cancel anyone"
    );
}
