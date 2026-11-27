use std::time::{Duration, Instant};

use super::*;

fn limits(tpm: u32, rpm: u32) -> ScopeLimits {
    ScopeLimits {
        tokens_per_minute: tpm,
        requests_per_minute: rpm,
    }
}

#[test]
fn replacement_refills_from_the_old_buckets_final_refill() {
    let old = Arc::new(ScopeBucket::new(limits(1200, 120)));
    old.debit(&mut old.lock_and_refill(), 300.0, 30.0);

    let replacement = old.with_limits(limits(600, 60));
    let cutover = old.state.lock().last_refill;
    let mut state = replacement.state.lock();
    assert_eq!(state.last_refill, cutover);

    let tokens = state.available_tokens;
    let requests = state.available_requests;
    replacement.refill_at(&mut state, cutover + Duration::from_secs(1));
    assert_eq!(state.available_tokens, (tokens + 10.0).min(600.0));
    assert_eq!(state.available_requests, (requests + 1.0).min(60.0));
}

#[test]
fn a_settlement_through_a_replaced_bucket_lands_on_its_successor() {
    let started = Instant::now();
    let old = Arc::new(ScopeBucket::new(limits(200, 10)));
    {
        let mut state = old.lock_and_refill();
        old.debit(&mut state, 100.0, 1.0);
    }
    // The handle a settlement resolved before the replacement.
    let stale = Arc::clone(&old);
    let successor = old.with_limits(limits(300, 10));
    assert!(old
        .successor
        .get()
        .is_some_and(|named| Arc::ptr_eq(named, &successor)));

    // Forty tokens fewer than reserved: credited to the successor, which
    // carried the hundred, never to the bucket nothing serves any more.
    stale.settle_live(-40.0);
    assert!(refilled_from(
        successor.lock().available_tokens,
        240.0,
        300,
        started
    ));
    assert!(refilled_from(
        old.lock().available_tokens,
        100.0,
        200,
        started
    ));
}

/// `tokens` is `base` plus at most the refill a `tpm` bucket earns since
/// `started`: the counters refill on every lock, so a test that pauses
/// reads a little above the exact figure and never below it.
fn refilled_from(tokens: f64, base: f64, tpm: u32, started: Instant) -> bool {
    let earned = f64::from(tpm) / 60.0 * started.elapsed().as_secs_f64();
    tokens >= base - 1e-6 && tokens <= base + earned + 1e-6
}

#[test]
fn a_settlement_waiting_on_a_replaced_bucket_follows_the_successor_named_under_its_lock() {
    let started = Instant::now();
    let old = Arc::new(ScopeBucket::new(limits(200, 10)));
    let successor = Arc::new(ScopeBucket::new(limits(300, 10)));
    for bucket in [&old, &successor] {
        let mut state = bucket.lock_and_refill();
        bucket.debit(&mut state, 100.0, 1.0);
    }
    // The replacement's turn: the old bucket's lock held while the
    // successor is named, as `with_limits` holds it.
    let held = old.lock();
    let stale = Arc::clone(&old);
    let settled = std::thread::spawn(move || stale.settle_live(-40.0));
    // The settlement waits on the lock (the pause only lets it get
    // there first; the outcome does not depend on it).
    std::thread::sleep(Duration::from_millis(50));
    old.succeeded_by(Arc::clone(&successor));
    drop(held);
    settled.join().expect("the settlement thread finishes");

    // Credited to the successor named under the lock it waited on,
    // never to the bucket nothing serves any more.
    assert!(refilled_from(
        successor.lock().available_tokens,
        240.0,
        300,
        started
    ));
    assert!(refilled_from(
        old.lock().available_tokens,
        100.0,
        200,
        started
    ));
}

#[test]
fn admits_within_capacity() {
    let bucket = ScopeBucket::new(limits(600, 10));
    let mut guard = bucket.lock();
    bucket.refill(&mut guard);
    let dry = bucket.dry_run(&guard, 100.0, 1.0);
    assert!(dry.allowed);
    bucket.debit(&mut guard, 100.0, 1.0);
    assert_eq!(guard.available_tokens, 500.0);
    assert_eq!(guard.available_requests, 9.0);
}

#[test]
fn denies_over_capacity_with_retry_after() {
    let bucket = ScopeBucket::new(limits(60, 10));
    let mut guard = bucket.lock();
    bucket.refill(&mut guard);
    bucket.debit(&mut guard, 60.0, 0.0);
    let dry = bucket.dry_run(&guard, 30.0, 0.0);
    assert!(!dry.allowed);
    // 60 tokens/min = 1 token/sec; need 30 more => ~30s.
    assert_eq!(dry.retry_after_secs, 30);
}

#[test]
fn ask_above_bucket_capacity_is_never_retryable() {
    let bucket = ScopeBucket::new(limits(60, 10));
    let mut guard = bucket.lock();
    bucket.refill(&mut guard);
    // 100 tokens can never fit in a 60-token bucket -- refill() clamps
    // at capacity, so no wait time would ever make this admissible.
    let dry = bucket.dry_run(&guard, 100.0, 0.0);
    assert!(!dry.allowed);
    assert_eq!(dry.retry_after_secs, u64::MAX);
}

#[tokio::test]
async fn refill_restores_capacity_over_time() {
    let bucket = ScopeBucket::new(limits(600, 10));
    {
        let mut guard = bucket.lock();
        bucket.refill(&mut guard);
        bucket.debit(&mut guard, 600.0, 0.0);
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let mut guard = bucket.lock();
    bucket.refill(&mut guard);
    // Lower bound proves refill happened; upper bound is capacity
    // (not a tight timing-derived number) so scheduling jitter can't
    // flake this.
    assert!(guard.available_tokens > 0.5 && guard.available_tokens <= 600.0);
}

#[test]
fn a_refill_stamped_before_the_last_one_credits_nothing_and_keeps_the_stamp() {
    let bucket = ScopeBucket::new(limits(600, 10));
    let mut guard = bucket.lock();
    bucket.refill(&mut guard);
    bucket.debit(&mut guard, 300.0, 5.0);
    let last = guard.last_refill;

    // A clock read taken ahead of the lock, behind a later refill.
    let earlier = last.checked_sub(Duration::from_secs(10)).unwrap_or(last);
    bucket.refill_at(&mut guard, earlier);
    assert_eq!(guard.available_tokens, 300.0);
    assert_eq!(guard.available_requests, 5.0);
    assert_eq!(guard.last_refill, last);

    // The next refill credits from the kept stamp: 10 s at 10 tokens/s.
    bucket.refill_at(&mut guard, last + Duration::from_secs(10));
    assert_eq!(guard.available_tokens, 400.0);
    assert_eq!(guard.last_refill, last + Duration::from_secs(10));
}

#[test]
fn settle_can_push_negative_debt_but_not_above_capacity() {
    let bucket = ScopeBucket::new(limits(600, 10));
    let mut guard = bucket.lock();
    bucket.refill(&mut guard);
    bucket.debit(&mut guard, 100.0, 1.0);
    // Actual usage exceeded the reservation by 50 tokens.
    bucket.settle_tokens(&mut guard, 50.0);
    assert_eq!(guard.available_tokens, 450.0);

    // A refund (negative delta) is capped at capacity.
    bucket.settle_tokens(&mut guard, -1000.0);
    assert_eq!(guard.available_tokens, 600.0);
}

#[tokio::test]
async fn lock_touches_last_touched_ns() {
    let bucket = ScopeBucket::new(limits(600, 10));
    assert_eq!(bucket.last_touched_ns(), 0);

    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(bucket.lock());

    let touched = bucket.last_touched_ns();
    assert!(touched > 0);

    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(bucket.lock());

    assert!(bucket.last_touched_ns() > touched);
}

#[tokio::test]
async fn idle_at_counts_from_the_last_lock_and_is_full_is_not_a_touch() {
    let bucket = ScopeBucket::new(limits(600, 10));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(bucket.idle_at(Instant::now()) >= Duration::from_millis(50));

    assert!(bucket.is_full());
    assert!(bucket.idle_at(Instant::now()) >= Duration::from_millis(50));

    let before_touch = Instant::now();
    drop(bucket.lock());
    assert!(bucket.idle_at(Instant::now()) < Duration::from_millis(50));
    // Touched after `before_touch`: idle as of then reads zero.
    assert_eq!(bucket.idle_at(before_touch), Duration::ZERO);
}

#[test]
fn is_full_only_at_capacity_on_both_counters() {
    let bucket = ScopeBucket::new(limits(600, 10));
    assert!(bucket.is_full());

    // 1 token/request short of capacity takes ~100ms/~6s to refill.
    let mut guard = bucket.lock();
    bucket.debit(&mut guard, 1.0, 0.0);
    drop(guard);
    assert!(!bucket.is_full());

    let bucket = ScopeBucket::new(limits(600, 10));
    let mut guard = bucket.lock();
    bucket.debit(&mut guard, 0.0, 1.0);
    drop(guard);
    assert!(!bucket.is_full());
}

#[test]
fn a_bucket_in_debt_is_not_full() {
    let bucket = ScopeBucket::new(limits(600, 10));
    let mut guard = bucket.lock();
    // Usage beyond the whole capacity: a full refill window later the
    // bucket is still short, so it must not read as full.
    bucket.settle_tokens(&mut guard, 1200.0);
    drop(guard);
    assert!(!bucket.is_full());
}
