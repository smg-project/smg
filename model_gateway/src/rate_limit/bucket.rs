//! Per-scope token/request bucket for the local rate-limit backend:
//! continuous-refill float counters (same math as
//! `middleware::token_bucket::TokenBucket`), but two independent counters
//! per scope and a reserve/settle API allowing temporary debt — a
//! completion can use more tokens than were reserved for it.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

use parking_lot::{Mutex, MutexGuard};

use super::policy::ScopeLimits;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScopeDryRun {
    pub allowed: bool,
    /// Seconds until enough capacity would be available. `0` when
    /// `allowed`.
    pub retry_after_secs: u64,
}

pub(crate) struct BucketState {
    available_tokens: f64,
    available_requests: f64,
    last_refill: Instant,
}

pub(crate) struct ScopeBucket {
    state: Mutex<BucketState>,
    limits: ScopeLimits,
    token_capacity: f64,
    token_refill_per_sec: f64,
    request_capacity: f64,
    request_refill_per_sec: f64,
    created_at: Instant,
    /// Nanoseconds since `created_at`, updated on every lock without
    /// taking `state`'s lock — lets the backend's idle sweep read it
    /// without contending with the hot path.
    last_touched_ns: AtomicU64,
    /// The bucket that replaced this one at new limits, once one has: a
    /// reservation that holds this bucket settles on the live one through
    /// it, so a change of limits forgives none of its completion tokens.
    successor: OnceLock<Arc<ScopeBucket>>,
}

impl ScopeBucket {
    pub fn new(limits: ScopeLimits) -> Self {
        let token_capacity = f64::from(limits.tokens_per_minute);
        let request_capacity = f64::from(limits.requests_per_minute);
        Self {
            state: Mutex::new(BucketState {
                available_tokens: token_capacity,
                available_requests: request_capacity,
                last_refill: Instant::now(),
            }),
            limits,
            token_capacity,
            token_refill_per_sec: token_capacity / 60.0,
            request_capacity,
            request_refill_per_sec: request_capacity / 60.0,
            created_at: Instant::now(),
            last_touched_ns: AtomicU64::new(0),
            successor: OnceLock::new(),
        }
    }

    /// The limits the bucket was built with.
    pub fn limits(&self) -> ScopeLimits {
        self.limits
    }

    /// Names the bucket that replaced this one. The first name stands.
    fn succeeded_by(&self, successor: Arc<ScopeBucket>) {
        let _first_name_stands = self.successor.set(successor);
    }

    /// Settles `delta` tokens on the bucket in force for this one's scope.
    /// The successor is read under the lock, never before it: a replacement
    /// names its successor under the old bucket's lock, so a settlement
    /// that waited on that lock finds the successor and follows it instead
    /// of debiting a bucket whose spending was already carried over.
    /// Refilled first, or a later deferred refill can overshoot capacity
    /// and clamp, erasing the delta.
    pub fn settle_live(self: &Arc<Self>, delta: f64) {
        let mut bucket = Arc::clone(self);
        loop {
            let mut state = bucket.lock_and_refill();
            let Some(next) = bucket.successor.get().map(Arc::clone) else {
                bucket.settle_tokens(&mut state, delta);
                return;
            };
            drop(state);
            bucket = next;
        }
    }

    /// A bucket at `limits` that carries what this one has used: each
    /// counter is refilled to now, the amount spent from its capacity is
    /// read, and the new counter starts at the new capacity less that
    /// amount. A raised limit opens headroom at once. A lowered one starts
    /// at most empty: the new counter goes no deeper below zero than the
    /// old one had gone (settlement debt, carried in full), so the scope is
    /// admitted again within one refill window instead of paying the old
    /// window's spending off at the new rate. The new counters start
    /// refilling from the old bucket's final refill timestamp. The new
    /// bucket starts untouched. It is named this bucket's successor
    /// under this bucket's lock, so a settlement that takes the lock next
    /// finds it and follows it (`settle_live`): no debit lands on a bucket
    /// after its spending was carried over.
    pub fn with_limits(self: &Arc<Self>, limits: ScopeLimits) -> Arc<Self> {
        let replacement = Arc::new(Self::new(limits));
        let state = self.lock_and_refill();
        let used_tokens = self.token_capacity - state.available_tokens;
        let used_requests = self.request_capacity - state.available_requests;
        let token_debt = state.available_tokens.min(0.0);
        let request_debt = state.available_requests.min(0.0);
        {
            let mut new_state = replacement.state.lock();
            new_state.available_tokens = (replacement.token_capacity - used_tokens).max(token_debt);
            new_state.available_requests =
                (replacement.request_capacity - used_requests).max(request_debt);
            new_state.last_refill = state.last_refill;
        }
        self.succeeded_by(Arc::clone(&replacement));
        drop(state);
        replacement
    }

    /// The tests' lock: the touch and the guard, the refill left to the
    /// caller.
    #[cfg(test)]
    pub fn lock(&self) -> MutexGuard<'_, BucketState> {
        self.touch_at(Instant::now());
        self.state.lock()
    }

    /// The hot path's lock: the bucket refilled to now under it, the touch
    /// and the refill stamped from one clock read taken under the lock, so
    /// a reader that waited behind a later one never stamps an earlier time.
    /// `dry_run`, `debit` and `settle_tokens` follow under the guard.
    pub fn lock_and_refill(&self) -> MutexGuard<'_, BucketState> {
        let mut state = self.state.lock();
        let now = Instant::now();
        self.touch_at(now);
        self.refill_at(&mut state, now);
        state
    }

    fn touch_at(&self, now: Instant) {
        let now_ns = u64::try_from(now.saturating_duration_since(self.created_at).as_nanos())
            .unwrap_or(u64::MAX);
        self.last_touched_ns.store(now_ns, Ordering::Relaxed);
    }

    /// Nanoseconds since `created_at` as of the last lock.
    pub fn last_touched_ns(&self) -> u64 {
        self.last_touched_ns.load(Ordering::Relaxed)
    }

    /// How long the bucket had gone without a lock (or since creation,
    /// if never locked) as of `now`; zero if it was touched after `now`.
    pub fn idle_at(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.created_at)
            .saturating_sub(Duration::from_nanos(self.last_touched_ns()))
    }

    /// Whether both counters, refilled to now, are at capacity — the state
    /// a freshly created bucket starts in, so dropping this one and
    /// recreating it later loses nothing. A bucket in settlement debt is
    /// not full until the refill has paid the debt off.
    ///
    /// Takes `state`'s lock directly rather than through `lock()`, so
    /// asking doesn't count as a touch.
    pub fn is_full(&self) -> bool {
        let mut state = self.state.lock();
        self.refill(&mut state);
        state.available_tokens >= self.token_capacity
            && state.available_requests >= self.request_capacity
    }

    /// Refill both counters based on elapsed time since the last refill.
    /// Must be called under the lock before `dry_run`/`debit`
    /// ([`Self::lock_and_refill`] does both).
    pub fn refill(&self, state: &mut BucketState) {
        self.refill_at(state, Instant::now());
    }

    /// A `now` behind the last refill credits nothing and leaves the stamp
    /// where it is: moving it back would pay the same stretch of time out
    /// twice at the next refill.
    fn refill_at(&self, state: &mut BucketState, now: Instant) {
        let Some(elapsed) = now.checked_duration_since(state.last_refill) else {
            return;
        };
        let elapsed = elapsed.as_secs_f64();
        state.available_tokens =
            (state.available_tokens + elapsed * self.token_refill_per_sec).min(self.token_capacity);
        state.available_requests = (state.available_requests
            + elapsed * self.request_refill_per_sec)
            .min(self.request_capacity);
        state.last_refill = now;
    }

    /// Non-mutating: would `tokens`/`requests` be admitted right now?
    pub fn dry_run(&self, state: &BucketState, tokens: f64, requests: f64) -> ScopeDryRun {
        let allowed = state.available_tokens >= tokens && state.available_requests >= requests;
        if allowed {
            return ScopeDryRun {
                allowed: true,
                retry_after_secs: 0,
            };
        }
        if tokens > self.token_capacity || requests > self.request_capacity {
            // refill() clamps at capacity, so an ask above it can never
            // become admissible no matter how long the caller waits.
            return ScopeDryRun {
                allowed: false,
                retry_after_secs: u64::MAX,
            };
        }
        let token_wait = Self::wait_secs(state.available_tokens, tokens, self.token_refill_per_sec);
        let request_wait = Self::wait_secs(
            state.available_requests,
            requests,
            self.request_refill_per_sec,
        );
        ScopeDryRun {
            allowed: false,
            retry_after_secs: token_wait.max(request_wait),
        }
    }

    fn wait_secs(available: f64, needed: f64, refill_per_sec: f64) -> u64 {
        if available >= needed {
            return 0;
        }
        if refill_per_sec <= 0.0 {
            return u64::MAX;
        }
        (((needed - available) / refill_per_sec).ceil() as u64).max(1)
    }

    /// Caller must have already confirmed admission via `dry_run` under
    /// the same lock acquisition.
    #[expect(
        clippy::unused_self,
        reason = "API symmetry with refill/dry_run/settle_tokens, which do need self"
    )]
    pub fn debit(&self, state: &mut BucketState, tokens: f64, requests: f64) {
        state.available_tokens -= tokens;
        state.available_requests -= requests;
    }

    /// Signed delta to the token counter only — requests aren't trued up.
    /// Can push tokens negative (debt) but never above capacity.
    pub fn settle_tokens(&self, state: &mut BucketState, delta: f64) {
        state.available_tokens = (state.available_tokens - delta).min(self.token_capacity);
    }
}

#[cfg(test)]
mod tests {
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
}
