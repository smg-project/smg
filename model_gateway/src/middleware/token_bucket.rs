use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::{sync::Notify, time::error::Elapsed};
use tracing::{debug, trace};

/// Token bucket for admission control.
///
/// With `refill_rate == 0` it is a concurrency semaphore: a request takes a
/// token at admission and `return_tokens` hands it back on completion, so at
/// most `capacity` requests are in flight.
///
/// With `refill_rate > 0` it is a request-rate limiter with the same
/// concurrency bound: admission spends a token that only the refill replaces
/// (`refill_rate` per second, bursting up to `capacity`), while
/// `return_tokens` frees the concurrency slot alone. A completed request can
/// therefore not raise the admission rate, and the refill can not push the
/// outstanding count above `capacity`. `try_take` spends rate budget without
/// a slot, for callers that bound concurrency elsewhere.
///
/// This implementation provides:
/// - Smooth rate limiting with configurable refill rate
/// - Burst capacity handling
/// - FIFO waiter handoff: returned (and refilled) tokens are granted directly
///   to the oldest waiter, so waiters are served in arrival order, wakeups
///   cannot be lost, and new arrivals cannot barge past the queue
/// - Sync token return for Drop handlers (via `return_tokens_sync`)
///
/// Every waiter parks on its own `Notify` and is woken exactly when granted.
/// With `refill_rate>0` only the front waiter additionally arms a timer for
/// the instant the refill covers its request; it cascades grants to the rest.
/// There is deliberately no wake-all or periodic polling: with thousands of
/// queued waiters those turn every token return into a thundering herd on the
/// bucket mutex, starving the tokio runtime (observed as multi-second stalls
/// of unrelated endpoints like `GET /workers` under generation bursts).
///
/// Uses `parking_lot::Mutex` for sync-compatible locking (no async required).
#[derive(Clone)]
pub struct TokenBucket {
    inner: Arc<Mutex<TokenBucketInner>>,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

struct TokenBucketInner {
    /// Admission budget: free slots without refill, rate budget with it.
    tokens: f64,
    /// Tokens taken and not yet returned (the outstanding permits).
    inflight: f64,
    last_refill: Instant,
    next_waiter_id: u64,
    /// FIFO waiters.
    waiters: VecDeque<Waiter>,
    /// Granted-but-uncollected waiter ids; their tokens are already
    /// deducted from the pool.
    granted: HashSet<u64>,
}

struct Waiter {
    id: u64,
    tokens: f64,
    notify: Arc<Notify>,
}

/// Removes a cancelled waiter; returns an uncollected grant to the pool.
struct FifoWaiterGuard<'a> {
    bucket: &'a TokenBucket,
    id: u64,
    tokens: f64,
    armed: bool,
}

impl Drop for FifoWaiterGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut inner = self.bucket.inner.lock();
        if inner.granted.remove(&self.id) {
            // Un-admission, not completion: the slot and the budget both
            // come back.
            inner.inflight = (inner.inflight - self.tokens).max(0.0);
            inner.tokens = (inner.tokens + self.tokens).min(self.bucket.capacity);
            self.bucket.grant_waiters_locked(&mut inner);
            // A partial return shrinks the front's deficit without granting
            // it; re-arm its refill timer at the new, earlier instant.
            self.bucket.wake_front_locked(&inner);
        } else if let Some(pos) = inner.waiters.iter().position(|w| w.id == self.id) {
            inner.waiters.remove(pos);
            // The removal may have made the next waiter affordable, or (with
            // refill) promoted a new front that must arm its refill timer.
            self.bucket.grant_waiters_locked(&mut inner);
            self.bucket.wake_front_locked(&inner);
        }
    }
}

impl TokenBucket {
    /// Create a new token bucket
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of tokens (burst capacity)
    /// * `refill_rate` - Tokens added per second (0 for pure concurrency limiting)
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        let capacity = capacity as f64;
        // Allow refill_rate=0 for pure concurrency limiting (semaphore behavior)
        // When refill_rate=0, tokens are only returned via return_tokens()
        let refill_rate = refill_rate as f64;

        Self {
            inner: Arc::new(Mutex::new(TokenBucketInner {
                tokens: capacity,
                inflight: 0.0,
                last_refill: Instant::now(),
                next_waiter_id: 0,
                waiters: VecDeque::new(),
                granted: HashSet::new(),
            })),
            capacity,
            refill_rate,
        }
    }

    fn refill_locked(&self, inner: &mut TokenBucketInner) {
        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        inner.tokens = (inner.tokens + elapsed * self.refill_rate).min(self.capacity);
        inner.last_refill = now;
    }

    /// True when `tokens` more can be admitted: enough budget and room under
    /// the concurrency bound. Without refill the budget is the free slots, so
    /// the bound is implied; with refill it is what keeps the refill from
    /// minting permits above `capacity`.
    fn admits_locked(&self, inner: &TokenBucketInner, tokens: f64) -> bool {
        inner.tokens >= tokens && inner.inflight + tokens <= self.capacity
    }

    /// Hand tokens to FIFO waiters, oldest first. When grants promote a new
    /// front and the bucket refills over time, the new front is woken so it
    /// can arm its refill timer.
    fn grant_waiters_locked(&self, inner: &mut TokenBucketInner) {
        let mut granted_any = false;
        loop {
            match inner.waiters.front() {
                Some(front) if self.admits_locked(inner, front.tokens) => {}
                _ => break,
            }
            let Some(waiter) = inner.waiters.pop_front() else {
                break;
            };
            inner.tokens -= waiter.tokens;
            inner.inflight += waiter.tokens;
            inner.granted.insert(waiter.id);
            waiter.notify.notify_one();
            granted_any = true;
        }
        if granted_any {
            self.wake_front_locked(inner);
        }
    }

    /// With refill enabled, wake the front waiter so it (re)arms the timer
    /// for the instant the refill covers its request. No-op without refill:
    /// grants only happen on token return, which notifies grantees directly.
    fn wake_front_locked(&self, inner: &TokenBucketInner) {
        if self.refill_rate > 0.0 {
            if let Some(front) = inner.waiters.front() {
                front.notify.notify_one();
            }
        }
    }

    /// Try to acquire tokens immediately.
    ///
    /// Returns `Ok(())` if tokens were acquired, `Err(())` if insufficient tokens
    /// or FIFO waiters are queued ahead.
    #[expect(
        clippy::result_unit_err,
        reason = "Try-acquire pattern: callers only need success/failure, a custom error type adds no information"
    )]
    pub fn try_acquire(&self, tokens: f64) -> Result<(), ()> {
        self.try_acquire_sync(tokens)
    }

    /// Sync version of try_acquire (for internal use).
    fn try_acquire_sync(&self, tokens: f64) -> Result<(), ()> {
        debug_assert!(
            tokens.is_finite() && tokens >= 0.0,
            "token amount must be non-negative and finite, got {tokens}"
        );
        let mut inner = self.inner.lock();

        self.refill_locked(&mut inner);
        self.grant_waiters_locked(&mut inner);

        trace!(
            "Token bucket: {} tokens available, requesting {}",
            inner.tokens,
            tokens
        );

        if inner.waiters.is_empty() && self.admits_locked(&inner, tokens) {
            inner.tokens -= tokens;
            inner.inflight += tokens;
            debug!(
                "Token bucket: acquired {} tokens, {} remaining",
                tokens, inner.tokens
            );
            Ok(())
        } else {
            Err(())
        }
    }

    /// Spend `tokens` of rate budget without taking a concurrency slot, for
    /// a caller whose concurrency is bounded elsewhere (the priority
    /// scheduler's per-second check). Nothing is returned later: only the
    /// refill replaces what was spent. Fails when the budget is short or
    /// FIFO waiters are queued ahead.
    #[expect(
        clippy::result_unit_err,
        reason = "Try-acquire pattern: callers only need success/failure, a custom error type adds no information"
    )]
    pub fn try_take(&self, tokens: f64) -> Result<(), ()> {
        debug_assert!(
            tokens.is_finite() && tokens >= 0.0,
            "token amount must be non-negative and finite, got {tokens}"
        );
        let mut inner = self.inner.lock();
        self.refill_locked(&mut inner);
        self.grant_waiters_locked(&mut inner);
        if inner.waiters.is_empty() && inner.tokens >= tokens {
            inner.tokens -= tokens;
            Ok(())
        } else {
            Err(())
        }
    }

    /// Acquire tokens, waiting in FIFO order (indefinitely) for tokens to be
    /// returned via `return_tokens()` or, with `refill_rate>0`, refilled over
    /// time. Use `acquire_timeout()` to set an appropriate timeout; a
    /// timed-out or cancelled waiter leaves the queue and returns any
    /// uncollected grant.
    pub async fn acquire(&self, tokens: f64) -> Result<(), Elapsed> {
        let (id, notify) = {
            let mut inner = self.inner.lock();
            self.refill_locked(&mut inner);
            if inner.waiters.is_empty() && self.admits_locked(&inner, tokens) {
                inner.tokens -= tokens;
                inner.inflight += tokens;
                return Ok(());
            }
            let id = inner.next_waiter_id;
            inner.next_waiter_id += 1;
            let notify = Arc::new(Notify::new());
            inner.waiters.push_back(Waiter {
                id,
                tokens,
                notify: Arc::clone(&notify),
            });
            (id, notify)
        };

        debug!("Token bucket: waiting in FIFO queue for {} tokens", tokens);

        let mut guard = FifoWaiterGuard {
            bucket: self,
            id,
            tokens,
            armed: true,
        };
        loop {
            let refill_wait = {
                let mut inner = self.inner.lock();
                if self.refill_rate > 0.0 {
                    self.refill_locked(&mut inner);
                    self.grant_waiters_locked(&mut inner);
                }
                if inner.granted.remove(&id) {
                    guard.armed = false;
                    return Ok(());
                }
                // Only the front waiter sleeps until the refill covers its
                // request; everyone else waits for a direct grant, and so
                // does a front held back by the concurrency bound (the
                // return that frees a slot wakes it to arm the timer).
                match inner.waiters.front() {
                    Some(front)
                        if self.refill_rate > 0.0
                            && front.id == id
                            && inner.inflight + front.tokens <= self.capacity =>
                    {
                        let deficit = (front.tokens - inner.tokens).max(0.0);
                        Some(Duration::from_secs_f64(deficit / self.refill_rate))
                    }
                    _ => None,
                }
            };
            match refill_wait {
                // Wake at the computed deadline; if float rounding leaves a
                // hair of deficit, the next iteration re-arms for the residue.
                Some(wait) => {
                    let _ = tokio::time::timeout(wait, notify.notified()).await;
                }
                // notify_one on grant stores a permit, so a grant between the
                // check above and this await still wakes us.
                None => notify.notified().await,
            }
        }
    }

    /// Acquire tokens with custom timeout.
    pub async fn acquire_timeout(&self, tokens: f64, timeout: Duration) -> Result<(), Elapsed> {
        tokio::time::timeout(timeout, self.acquire(tokens)).await?
    }

    /// Return tokens to the bucket (sync version): the request completed and
    /// its concurrency slot is free again. Without refill the slot is the
    /// budget, so it is reusable at once; with refill the budget only comes
    /// back over time, so a completion cannot raise the admission rate.
    ///
    /// This is safe to call from sync contexts (e.g., Drop handlers).
    /// Uses `parking_lot::Mutex` which never blocks indefinitely.
    pub fn return_tokens_sync(&self, tokens: f64) {
        debug_assert!(
            tokens.is_finite() && tokens >= 0.0,
            "token amount must be non-negative and finite, got {tokens}"
        );
        let mut inner = self.inner.lock();
        inner.inflight = (inner.inflight - tokens).max(0.0);
        if self.refill_rate == 0.0 {
            inner.tokens = (inner.tokens + tokens).min(self.capacity);
        }
        self.grant_waiters_locked(&mut inner);
        // A partial return shrinks the front's deficit without granting it;
        // re-arm its refill timer at the new, earlier instant.
        self.wake_front_locked(&inner);
        debug!(
            "Token bucket: returned {} tokens, {} available",
            tokens, inner.tokens
        );
    }

    /// Return tokens to the bucket.
    pub fn return_tokens(&self, tokens: f64) {
        self.return_tokens_sync(tokens);
    }

    /// Get current available tokens (for monitoring).
    pub fn available_tokens(&self) -> f64 {
        let mut inner = self.inner.lock();
        self.refill_locked(&mut inner);
        inner.tokens
    }

    #[cfg(test)]
    pub(crate) fn waiter_count(&self) -> usize {
        self.inner.lock().waiters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With refill, admission spends rate budget and a completion frees a
    /// concurrency slot only: after the burst a request needs both a free
    /// slot and refilled budget.
    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5);

        assert!(bucket.try_acquire(5.0).is_ok());
        assert!(bucket.try_acquire(5.0).is_ok());

        // Burst spent and every slot taken.
        assert!(bucket.try_acquire(1.0).is_err());

        // A completion frees a slot but hands no budget back.
        bucket.return_tokens(1.0);
        assert!(bucket.try_acquire(1.0).is_err());

        // ~1.5 tokens refilled: the free slot can be taken, once.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(bucket.try_acquire(1.0).is_ok());
        assert!(bucket.try_acquire(1.0).is_err());
    }

    /// Refill replaces spent budget but never mints concurrency: with every
    /// slot taken, waiting adds budget and still admits nothing until a
    /// request completes.
    #[tokio::test]
    async fn refill_cannot_push_inflight_above_capacity() {
        let bucket = TokenBucket::new(2, 1000);
        assert!(bucket.try_acquire(1.0).is_ok());
        assert!(bucket.try_acquire(1.0).is_ok());

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            bucket.available_tokens() >= 1.0,
            "budget refills while the slots are held"
        );
        assert!(bucket.try_acquire(1.0).is_err(), "no slot is free");

        bucket.return_tokens(1.0);
        assert!(bucket.try_acquire(1.0).is_ok());
        assert!(bucket.try_acquire(1.0).is_err());
    }

    /// A burst of `capacity` admissions followed by their completions leaves
    /// the budget empty: the sustained rate is the refill, not the turnover.
    #[tokio::test]
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

        // ~1.2 tokens at 2/s: exactly one more admission.
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(bucket.try_acquire(1.0).is_ok());
        assert!(bucket.try_acquire(1.0).is_err());
    }

    /// `try_take` spends budget without holding a slot: the per-second check
    /// of a caller that bounds concurrency itself.
    #[tokio::test]
    async fn try_take_spends_budget_without_a_slot() {
        let bucket = TokenBucket::new(2, 2);
        assert!(bucket.try_take(1.0).is_ok());
        assert!(bucket.try_take(1.0).is_ok());
        assert!(bucket.try_take(1.0).is_err(), "budget spent");

        // No slot was taken: once the budget refills, an admission fits.
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(bucket.try_acquire(1.0).is_ok());
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10);

        assert!(bucket.try_acquire(10.0).is_ok());

        tokio::time::sleep(Duration::from_millis(500)).await;

        let available = bucket.available_tokens();
        assert!((4.0..=6.0).contains(&available));
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_rate() {
        // With refill_rate=0, tokens should only come back via return_tokens()
        let bucket = TokenBucket::new(2, 0);

        // Acquire both tokens
        assert!(bucket.try_acquire(1.0).is_ok());
        assert!(bucket.try_acquire(1.0).is_ok());

        // No more tokens available
        assert!(bucket.try_acquire(1.0).is_err());

        // refill_rate=0 should NOT refill automatically even after waiting
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(bucket.try_acquire(1.0).is_err());

        bucket.return_tokens(1.0);
        assert!(bucket.try_acquire(1.0).is_ok());

        // No more tokens again
        assert!(bucket.try_acquire(1.0).is_err());
    }

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
    async fn test_try_acquire_cannot_barge_past_waiters() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        assert!(bucket.try_acquire(1.0).is_ok());

        let waiter = registered_waiter(&bucket, 1).await;

        bucket.return_tokens(1.0);
        // The returned token is already granted to the waiter.
        assert!(bucket.try_acquire(1.0).is_err());
        waiter.await.expect("waiter").expect("grant");
    }

    #[tokio::test]
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

    #[tokio::test]
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
    #[tokio::test]
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
    #[tokio::test]
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
        // about a second after its arrival, well inside its timeout.
        tokio::time::sleep(Duration::from_millis(200)).await;
        bucket.return_tokens(1.0);
        waiter
            .await
            .expect("waiter")
            .expect("grant via the refill timer armed on return");
    }

    /// When the front waiter times out, the next waiter takes over the
    /// refill timer instead of waiting forever for a grant.
    #[tokio::test]
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

    #[tokio::test]
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
}
