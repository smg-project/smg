use super::*;

/// With refill, admission spends rate budget and a completion frees a
/// concurrency slot only: after the burst a request needs both a free
/// slot and refilled budget.
#[tokio::test(start_paused = true)]
async fn test_token_bucket_basic() {
    let bucket = TokenBucket::new(10, 5);

    assert!(bucket.try_acquire(5.0).is_ok());
    assert!(bucket.try_acquire(5.0).is_ok());

    // Burst spent and every slot taken.
    assert!(bucket.try_acquire(1.0).is_err());

    // A completion frees a slot but hands no budget back.
    bucket.return_tokens(1.0);
    assert!(bucket.try_acquire(1.0).is_err());

    // 1.5 tokens refilled: the free slot can be taken, once.
    tokio::time::advance(Duration::from_millis(300)).await;
    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(bucket.try_acquire(1.0).is_err());
}

/// Refill replaces spent budget but never mints concurrency: with every
/// slot taken, waiting adds budget and still admits nothing until a
/// request completes.
#[tokio::test(start_paused = true)]
async fn refill_cannot_push_inflight_above_capacity() {
    let bucket = TokenBucket::new(2, 1000);
    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(bucket.try_acquire(1.0).is_ok());

    tokio::time::advance(Duration::from_millis(100)).await;
    assert_eq!(
        bucket.available_tokens(),
        2.0,
        "the budget refills to capacity while the slots are held"
    );
    assert!(bucket.try_acquire(1.0).is_err(), "no slot is free");

    bucket.return_tokens(1.0);
    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(bucket.try_acquire(1.0).is_err());
}

/// A burst of `capacity` admissions followed by their completions leaves
/// the budget empty: the sustained rate is the refill, not the turnover.
#[tokio::test(start_paused = true)]
async fn completions_do_not_replenish_the_rate_budget() {
    let bucket = TokenBucket::new(4, 2);
    for _ in 0..4 {
        assert!(bucket.try_acquire(1.0).is_ok());
    }
    for _ in 0..4 {
        bucket.return_tokens(1.0);
    }
    assert!(
        bucket.try_acquire(1.0).is_err(),
        "budget spent; completions hand none back"
    );

    // 1.2 tokens at 2/s: exactly one more admission.
    tokio::time::advance(Duration::from_millis(600)).await;
    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(bucket.try_acquire(1.0).is_err());
}

/// `try_take` spends budget without holding a slot: the per-second check
/// of a caller that bounds concurrency itself.
#[tokio::test(start_paused = true)]
async fn try_take_spends_budget_without_a_slot() {
    let bucket = TokenBucket::new(2, 2);
    assert!(bucket.try_take(1.0).is_ok());
    assert!(bucket.try_take(1.0).is_ok());
    assert!(bucket.try_take(1.0).is_err(), "budget spent");

    // No slot was taken: once 1.2 tokens refill, one admission fits.
    tokio::time::advance(Duration::from_millis(600)).await;
    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(
        bucket.try_acquire(1.0).is_err(),
        "1.2 tokens: one admission"
    );
}

#[tokio::test(start_paused = true)]
async fn test_token_bucket_refill() {
    let bucket = TokenBucket::new(10, 10);

    assert!(bucket.try_acquire(10.0).is_ok());

    tokio::time::advance(Duration::from_millis(500)).await;

    assert_eq!(
        bucket.available_tokens(),
        5.0,
        "half a second at ten tokens per second"
    );
}

#[tokio::test(start_paused = true)]
async fn test_token_bucket_zero_refill_rate() {
    // With refill_rate=0, tokens should only come back via return_tokens()
    let bucket = TokenBucket::new(2, 0);

    // Acquire both tokens
    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(bucket.try_acquire(1.0).is_ok());

    // No more tokens available
    assert!(bucket.try_acquire(1.0).is_err());

    // refill_rate=0 should NOT refill automatically even after waiting
    tokio::time::advance(Duration::from_millis(500)).await;
    assert!(bucket.try_acquire(1.0).is_err());

    bucket.return_tokens(1.0);
    assert!(bucket.try_acquire(1.0).is_ok());

    // No more tokens again
    assert!(bucket.try_acquire(1.0).is_err());
}

#[tokio::test(start_paused = true)]
async fn test_token_bucket_zero_refill_with_notify() {
    // Test that acquire wakes up when tokens are returned
    let bucket = Arc::new(TokenBucket::new(1, 0));

    // Acquire the only token
    assert!(bucket.try_acquire(1.0).is_ok());

    let bucket_clone = bucket.clone();

    // Spawn a task that will return the token after a delay
    #[expect(
        clippy::disallowed_methods,
        reason = "Test helper: short-lived task that completes before test ends"
    )]
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        bucket_clone.return_tokens(1.0);
    });

    // This should wait and then succeed when token is returned
    let result = bucket.acquire_timeout(1.0, Duration::from_secs(1)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_return_tokens_sync() {
    // Test that sync return works correctly
    let bucket = TokenBucket::new(2, 0);

    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(bucket.try_acquire(1.0).is_ok());
    assert!(bucket.try_acquire(1.0).is_err());

    // Use sync return
    bucket.return_tokens_sync(1.0);
    assert!(bucket.try_acquire(1.0).is_ok());
}

/// A waiter task for one token, joined once it is in the queue. Its
/// timeout is a guard: on the paused clock it fires only when every task
/// is idle, never because the test thread was descheduled on a loaded host.
async fn registered_waiter(
    bucket: &Arc<TokenBucket>,
    expected_waiters: usize,
) -> tokio::task::JoinHandle<Result<(), Elapsed>> {
    let waiter_bucket = bucket.clone();
    #[expect(
        clippy::disallowed_methods,
        reason = "Test helper: waiter tasks are joined or aborted before the test ends"
    )]
    let handle = tokio::spawn(async move {
        waiter_bucket
            .acquire_timeout(1.0, Duration::from_secs(5))
            .await
    });
    while bucket.waiter_count() < expected_waiters {
        tokio::task::yield_now().await;
    }
    handle
}

#[tokio::test(start_paused = true)]
async fn test_fifo_waiters_served_in_arrival_order() {
    let bucket = Arc::new(TokenBucket::new(1, 0));
    assert!(bucket.try_acquire(1.0).is_ok());

    let first = registered_waiter(&bucket, 1).await;
    let second = registered_waiter(&bucket, 2).await;
    let third = registered_waiter(&bucket, 3).await;

    bucket.return_tokens(1.0);
    first.await.expect("first waiter").expect("first grant");
    assert!(!second.is_finished());
    assert!(!third.is_finished());

    bucket.return_tokens(1.0);
    second.await.expect("second waiter").expect("second grant");
    assert!(!third.is_finished());

    bucket.return_tokens(1.0);
    third.await.expect("third waiter").expect("third grant");
    assert_eq!(bucket.available_tokens(), 0.0);
}

#[tokio::test(start_paused = true)]
async fn test_try_acquire_cannot_barge_past_waiters() {
    let bucket = Arc::new(TokenBucket::new(1, 0));
    assert!(bucket.try_acquire(1.0).is_ok());

    let waiter = registered_waiter(&bucket, 1).await;

    bucket.return_tokens(1.0);
    // The returned token is already granted to the waiter.
    assert!(bucket.try_acquire(1.0).is_err());
    waiter.await.expect("waiter").expect("grant");
}

#[tokio::test(start_paused = true)]
async fn test_cancelled_waiter_leaves_queue_and_returns_grant() {
    let bucket = Arc::new(TokenBucket::new(1, 0));
    assert!(bucket.try_acquire(1.0).is_ok());

    let cancelled = registered_waiter(&bucket, 1).await;
    let survivor = registered_waiter(&bucket, 2).await;

    cancelled.abort();
    let _ = cancelled.await;
    assert_eq!(bucket.waiter_count(), 1);

    // The survivor is now first in line.
    bucket.return_tokens(1.0);
    survivor.await.expect("survivor").expect("grant");
}

#[tokio::test(start_paused = true)]
async fn test_timed_out_waiter_does_not_leak_tokens() {
    let bucket = Arc::new(TokenBucket::new(1, 0));
    assert!(bucket.try_acquire(1.0).is_ok());

    let result = bucket.acquire_timeout(1.0, Duration::from_millis(20)).await;
    assert!(result.is_err());
    assert_eq!(bucket.waiter_count(), 0);

    bucket.return_tokens(1.0);
    assert_eq!(bucket.available_tokens(), 1.0);
}

/// With refill enabled and no other bucket traffic, the front waiter's
/// refill timer must drive grants for the whole queue, in FIFO order.
#[tokio::test(start_paused = true)]
async fn test_refill_serves_fifo_waiters_without_other_traffic() {
    let bucket = Arc::new(TokenBucket::new(2, 50));
    // Spend the burst without holding slots, so only the budget gates.
    bucket.try_take(2.0).unwrap();

    let first = registered_waiter(&bucket, 1).await;
    let second = registered_waiter(&bucket, 2).await;

    first.await.expect("first waiter").expect("first grant");
    second.await.expect("second waiter").expect("second grant");
}

/// A front waiter held back by the concurrency bound parks without a
/// timer; the return that frees the slot must wake it to arm the refill
/// timer, instead of leaving it parked until its own timeout.
#[tokio::test(start_paused = true)]
async fn test_return_wakes_slot_blocked_front_to_arm_refill_timer() {
    let bucket = Arc::new(TokenBucket::new(1, 1));
    bucket.try_acquire(1.0).unwrap();

    let waiter_bucket = Arc::clone(&bucket);
    #[expect(
        clippy::disallowed_methods,
        reason = "Test helper: the waiter task is joined before the test ends"
    )]
    let waiter = tokio::spawn(async move {
        waiter_bucket
            .acquire_timeout(1.0, Duration::from_millis(2500))
            .await
    });
    while bucket.waiter_count() < 1 {
        tokio::task::yield_now().await;
    }

    // The slot frees after 200 ms; the budget (1/s) covers the waiter
    // a second after its arrival, well inside its timeout.
    tokio::time::advance(Duration::from_millis(200)).await;
    bucket.return_tokens(1.0);
    waiter
        .await
        .expect("waiter")
        .expect("grant via the refill timer armed on return");
}

/// When the front waiter times out, the next waiter takes over the
/// refill timer instead of waiting forever for a grant.
#[tokio::test(start_paused = true)]
async fn test_cancelled_front_waiter_hands_refill_timer_to_next() {
    let bucket = Arc::new(TokenBucket::new(1, 20));
    bucket.try_take(1.0).unwrap();

    let front = registered_waiter(&bucket, 1).await;
    let survivor = registered_waiter(&bucket, 2).await;

    front.abort();
    let _ = front.await;
    assert_eq!(bucket.waiter_count(), 1);

    survivor.await.expect("survivor").expect("grant via refill");
}

#[tokio::test(start_paused = true)]
async fn test_no_lost_wakeup_under_tight_release_acquire_loop() {
    let bucket = Arc::new(TokenBucket::new(1, 0));

    for _ in 0..200 {
        assert!(bucket.try_acquire(1.0).is_ok());
        let waiter = registered_waiter(&bucket, 1).await;
        bucket.return_tokens(1.0);
        waiter
            .await
            .expect("waiter task")
            .expect("waiter must be granted, not time out");
        // Release the grant the waiter still holds.
        bucket.return_tokens(1.0);
    }
    assert_eq!(bucket.available_tokens(), 1.0);
}
