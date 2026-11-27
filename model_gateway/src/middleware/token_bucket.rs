#[cfg(not(test))]
use std::time::Instant;
use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use parking_lot::Mutex;
#[cfg(test)]
use tokio::time::Instant;
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
    /// When the budget was last refilled: the system clock in the gateway, and
    /// under test tokio's clock, the paused clock the tests advance, so the
    /// refill arithmetic is tested by advancing time, not by sleeping on it.
    /// tokio's `Instant` stays out of non-test builds: wherever tokio's
    /// `test-util` is on (the bench binary, the integration tests), its `now()`
    /// checks a flag on every call, which would skew the admission benches.
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
        // Without refill the budget moves with the returns alone: nothing to
        // add, and no clock to read for it.
        if self.refill_rate <= 0.0 {
            return;
        }
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
#[path = "../../tests/unit/middleware/token_bucket/tests.rs"]
mod tests;
