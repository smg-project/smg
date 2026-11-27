use std::sync::atomic::Ordering;

use super::*;

fn handle() -> InflightHandle {
    InflightHandle::new(Class::Default, RequestId("test".to_string()))
}

#[test]
fn test_try_mark_preempted_succeeds_then_fails() {
    let handle = handle();
    assert!(
        handle.try_mark_preempted(),
        "first preempt CAS should succeed"
    );
    assert!(
        !handle.try_mark_preempted(),
        "second preempt CAS should fail (slot already taken)"
    );
}

#[test]
fn test_try_mark_preempted_stores_sentinel() {
    let handle = handle();
    handle.try_mark_preempted();
    assert_eq!(
        handle.first_byte_at_raw(Ordering::Acquire),
        u64::MAX,
        "preempt sentinel is u64::MAX"
    );
}

#[test]
fn test_try_mark_first_byte_succeeds_then_fails() {
    let handle = handle();
    assert!(
        handle.try_mark_first_byte(42),
        "first TTFT CAS should succeed"
    );
    assert!(
        !handle.try_mark_first_byte(99),
        "second TTFT CAS should fail (slot already taken)"
    );
}

#[test]
fn test_ttft_loses_race_against_preempt() {
    let handle = handle();
    assert!(handle.try_mark_preempted());
    assert!(
        !handle.try_mark_first_byte(5),
        "TTFT must lose if preempt already won"
    );
}

#[test]
fn test_preempt_loses_race_against_ttft() {
    let handle = handle();
    assert!(handle.try_mark_first_byte(5));
    assert!(
        !handle.try_mark_preempted(),
        "preempt must lose if TTFT already won"
    );
}

#[test]
fn test_try_mark_first_byte_clamps_to_avoid_sentinel_collision() {
    // u64::MAX is the preempt sentinel; a TTFT measurement that happens
    // to equal that value would falsely appear as "preempted" to any
    // reader. Clamp to u64::MAX - 1.
    let handle = handle();
    assert!(handle.try_mark_first_byte(u64::MAX));
    assert_eq!(handle.first_byte_at_raw(Ordering::Acquire), u64::MAX - 1);
}

#[test]
fn test_initial_state_is_zero() {
    let handle = handle();
    assert_eq!(handle.first_byte_at_raw(Ordering::Acquire), 0);
}

#[test]
fn test_ttft_at_zero_ms_does_not_collide_with_unset_sentinel() {
    // Regression: now_ms == 0 (TTFT in the same millisecond as
    // admission) must not write 0 into first_byte_at. If it did, a
    // subsequent try_mark_preempted would see the "unset" sentinel
    // and succeed, breaking mutual exclusion.
    let handle = handle();
    assert!(handle.try_mark_first_byte(0), "first call must succeed");
    assert!(
        handle.first_byte_at_raw(Ordering::Acquire) != 0,
        "stored value must not collide with the unset sentinel"
    );
    assert!(
        !handle.try_mark_preempted(),
        "preempt must lose because TTFT already won"
    );
}
