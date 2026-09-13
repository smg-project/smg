use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::{sync::Notify, time::error::Elapsed};
use tracing::{debug, trace};

/// Token bucket for rate limiting.
///
/// This implementation provides:
/// - Smooth rate limiting with configurable refill rate
/// - Burst capacity handling
/// - FIFO waiter handoff when `refill_rate=0` (pure concurrency limiting):
///   returned tokens are granted directly to the oldest waiter, so waiters
///   are served in arrival order, wakeups cannot be lost, and new arrivals
///   cannot barge past the queue
/// - Sync token return for Drop handlers (via `return_tokens_sync`)
///
/// Uses `parking_lot::Mutex` for sync-compatible locking (no async required).
#[derive(Clone)]
pub struct TokenBucket {
    inner: Arc<Mutex<TokenBucketInner>>,
    notify: Arc<Notify>,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

struct TokenBucketInner {
    tokens: f64,
    last_refill: Instant,
    next_waiter_id: u64,
    /// FIFO waiters (refill_rate=0 acquire path only).
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
            inner.tokens = (inner.tokens + self.tokens).min(self.bucket.capacity);
            TokenBucket::grant_waiters_locked(&mut inner);
        } else if let Some(pos) = inner.waiters.iter().position(|w| w.id == self.id) {
            inner.waiters.remove(pos);
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
                last_refill: Instant::now(),
                next_waiter_id: 0,
                waiters: VecDeque::new(),
                granted: HashSet::new(),
            })),
            notify: Arc::new(Notify::new()),
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

    /// Hand tokens to FIFO waiters, oldest first.
    fn grant_waiters_locked(inner: &mut TokenBucketInner) {
        loop {
            match inner.waiters.front() {
                Some(front) if inner.tokens >= front.tokens => {}
                _ => break,
            }
            let Some(waiter) = inner.waiters.pop_front() else {
                break;
            };
            inner.tokens -= waiter.tokens;
            inner.granted.insert(waiter.id);
            waiter.notify.notify_one();
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
        Self::grant_waiters_locked(&mut inner);

        trace!(
            "Token bucket: {} tokens available, requesting {}",
            inner.tokens,
            tokens
        );

        if inner.waiters.is_empty() && inner.tokens >= tokens {
            inner.tokens -= tokens;
            debug!(
                "Token bucket: acquired {} tokens, {} remaining",
                tokens, inner.tokens
            );
            Ok(())
        } else {
            Err(())
        }
    }

    /// Acquire tokens, waiting if necessary.
    ///
    /// When `refill_rate=0`, waits in FIFO order (indefinitely) for tokens to
    /// be returned via `return_tokens()`. Use `acquire_timeout()` to set an
    /// appropriate timeout; a timed-out or cancelled waiter leaves the queue
    /// and returns any uncollected grant.
    pub async fn acquire(&self, tokens: f64) -> Result<(), Elapsed> {
        // When refill_rate=0 (pure concurrency limiting), tokens only come back
        // via return_tokens(), which hands them to the oldest waiter directly.
        if self.refill_rate == 0.0 {
            return self.acquire_fifo(tokens).await;
        }

        if self.try_acquire(tokens).is_ok() {
            return Ok(());
        }

        let wait_time = {
            let inner = self.inner.lock();
            let tokens_needed = tokens - inner.tokens;
            let wait_secs = (tokens_needed / self.refill_rate).max(0.0);
            Duration::from_secs_f64(wait_secs)
        };

        debug!(
            "Token bucket: waiting {:?} for {} tokens",
            wait_time, tokens
        );

        tokio::time::timeout(wait_time, async {
            loop {
                if self.try_acquire(tokens).is_ok() {
                    return;
                }
                tokio::select! {
                    () = self.notify.notified() => {},
                    () = tokio::time::sleep(Duration::from_millis(10)) => {},
                }
            }
        })
        .await?;

        Ok(())
    }

    async fn acquire_fifo(&self, tokens: f64) -> Result<(), Elapsed> {
        let (id, notify) = {
            let mut inner = self.inner.lock();
            if inner.waiters.is_empty() && inner.tokens >= tokens {
                inner.tokens -= tokens;
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

        debug!(
            "Token bucket: waiting in FIFO queue for {} tokens (refill_rate=0)",
            tokens
        );

        let mut guard = FifoWaiterGuard {
            bucket: self,
            id,
            tokens,
            armed: true,
        };
        loop {
            {
                let mut inner = self.inner.lock();
                if inner.granted.remove(&id) {
                    guard.armed = false;
                    return Ok(());
                }
            }
            // notify_one on grant stores a permit, so a grant between the
            // check above and this await still wakes us.
            notify.notified().await;
        }
    }

    /// Acquire tokens with custom timeout.
    pub async fn acquire_timeout(&self, tokens: f64, timeout: Duration) -> Result<(), Elapsed> {
        tokio::time::timeout(timeout, self.acquire(tokens)).await?
    }

    /// Return tokens to the bucket (sync version).
    ///
    /// This is safe to call from sync contexts (e.g., Drop handlers).
    /// Uses `parking_lot::Mutex` which never blocks indefinitely.
    pub fn return_tokens_sync(&self, tokens: f64) {
        debug_assert!(
            tokens.is_finite() && tokens >= 0.0,
            "token amount must be non-negative and finite, got {tokens}"
        );
        {
            let mut inner = self.inner.lock();
            inner.tokens = (inner.tokens + tokens).min(self.capacity);
            Self::grant_waiters_locked(&mut inner);
            debug!(
                "Token bucket: returned {} tokens, {} available",
                tokens, inner.tokens
            );
        } // Release lock before notify
        self.notify.notify_waiters();
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

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5);

        assert!(bucket.try_acquire(5.0).is_ok());
        assert!(bucket.try_acquire(5.0).is_ok());

        assert!(bucket.try_acquire(1.0).is_err());

        tokio::time::sleep(Duration::from_millis(300)).await;

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
