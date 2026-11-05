//! In-flight request bookkeeping for the priority scheduler.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use smg_auth::RequestId;
use tokio_util::sync::CancellationToken;

use super::Class;

/// Tracks a single admitted request from admission until its response body
/// finishes draining (or the scheduler preempts it).
///
/// The central race resolver is `first_byte_at`: it encodes three states
/// in one `AtomicU64`, using a sentinel for "preempted" so that the
/// admission side and the body-wrapping side can resolve their race with a
/// single compare-and-swap each.
///
/// State encoding for `first_byte_at`:
///
/// | Value         | Meaning                                                   |
/// |---------------|-----------------------------------------------------------|
/// | `0`           | No first-byte emitted yet and not preempted (initial)     |
/// | `u64::MAX`    | Scheduler preempted this request before TTFT (sentinel)   |
/// | anything else | Millis-since-admit of the first response byte (TTFT)      |
///
/// `try_mark_first_byte` clamps `now_ms` to `[1, u64::MAX - 1]` so a
/// TTFT measurement can never collide with either the unset state or
/// the preempt sentinel.
#[derive(Debug)]
pub struct InflightHandle {
    class: Class,
    request_id: RequestId,
    /// When this request was admitted. Used as the preemption victim
    /// tiebreaker (prefer the most-recently-admitted within a class, to
    /// waste the least upstream work) and as the epoch for TTFT.
    admitted_at: Instant,
    first_byte_at: AtomicU64,
    /// Scheduler-owned cancellation token, fired by `try_mark_preempted`'s
    /// caller when this request is chosen as a preemption victim. Handed
    /// to the handler via [`super::SchedulerPermit::cancel_token`] (which
    /// the admission middleware inserts into request extensions) so the
    /// handler can `select!` against it and unwind. Distinct from the
    /// client-disconnect token `admit` waits on during queueing.
    cancel: CancellationToken,
}

impl InflightHandle {
    /// Sentinel stored in `first_byte_at` once the scheduler has selected
    /// this request for preemption.
    const PREEMPTED_SENTINEL: u64 = u64::MAX;

    /// Construct a handle for a freshly admitted request. The scheduler
    /// inserts the resulting `Arc<Self>` into its inflight registry, keyed
    /// by `request_id`, so per-class iteration (preemption, metrics)
    /// can find it.
    pub fn new(class: Class, request_id: RequestId) -> Self {
        Self {
            class,
            request_id,
            admitted_at: Instant::now(),
            first_byte_at: AtomicU64::new(0),
            cancel: CancellationToken::new(),
        }
    }

    /// Class this admission consumed a slot under.
    pub fn class(&self) -> Class {
        self.class
    }

    /// Request id this handle tracks (registry key).
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// When this request was admitted (preemption tiebreaker + TTFT epoch).
    pub fn admitted_at(&self) -> Instant {
        self.admitted_at
    }

    /// Clone of the scheduler-owned cancel token for this request.
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Fire the scheduler-owned cancel token. Called only after a
    /// successful [`Self::try_mark_preempted`], so a handler is never asked to
    /// unwind a request that has already started streaming output.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Whether this request is still a valid preemption victim: it has
    /// neither emitted its first byte nor already been preempted (both
    /// move `first_byte_at` off `0`). A single `Acquire` load.
    pub fn is_preemptible(&self) -> bool {
        self.first_byte_at.load(Ordering::Acquire) == 0
    }

    /// Attempt to mark this request as preempted. Returns `true` on the
    /// first successful CAS; subsequent calls (or any call after TTFT was
    /// marked) return `false`.
    ///
    /// The caller fires the cancellation token only after a `true` return,
    /// so preemption never races with a response that already started
    /// streaming bytes to the client.
    pub fn try_mark_preempted(&self) -> bool {
        self.first_byte_at
            .compare_exchange(
                0,
                Self::PREEMPTED_SENTINEL,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Attempt to mark the first response byte at `now_ms` millis-since-
    /// admit. Returns `true` on the first successful CAS; subsequent calls
    /// (or any call after preemption was marked) return `false`.
    ///
    /// `now_ms` is clamped to `[1, PREEMPTED_SENTINEL - 1]`. The upper
    /// bound avoids colliding with the preempt sentinel; the lower bound
    /// avoids colliding with the "unset" state (`0`), which would let a
    /// subsequent `try_mark_preempted` succeed even though TTFT has
    /// already happened.
    pub fn try_mark_first_byte(&self, now_ms: u64) -> bool {
        let value = now_ms.clamp(1, Self::PREEMPTED_SENTINEL - 1);
        self.first_byte_at
            .compare_exchange(0, value, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Test-only raw read of the underlying atomic. Production callers
    /// should compare against the sentinel via the type's API rather than
    /// reading the raw value.
    #[cfg(test)]
    fn first_byte_at_raw(&self, ord: Ordering) -> u64 {
        self.first_byte_at.load(ord)
    }
}

#[cfg(test)]
mod tests {
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
}
