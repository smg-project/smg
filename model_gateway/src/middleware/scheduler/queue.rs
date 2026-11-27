//! Per-class waiter queue used by [`super::engine`] to hold admission
//! candidates while their class is at capacity.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_auth::RequestId;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::{engine::SchedulerPermit, Class};

/// A request waiting for admission in a class queue.
///
/// Carries the class lane to admit under, the wait-start instant (used
/// by the dispatcher's starvation override), the request id (carried
/// so admit can build the inflight handle after dequeue), the
/// cancellation token the client owns (lets the queue GC walkaway
/// clients without admitting them), and the oneshot the dispatcher
/// uses to hand back the [`SchedulerPermit`] when this waiter is
/// admitted. Rejections (queue full, queue timeout, client cancelled)
/// are returned synchronously from `admit` and never go through the
/// channel, so this carries only the success case.
#[derive(Debug)]
pub struct Waiter {
    pub class: Class,
    pub queued_at: Instant,
    pub cancel: CancellationToken,
    pub request_id: RequestId,
    pub permit_tx: oneshot::Sender<SchedulerPermit>,
}

impl Waiter {
    pub fn new(
        class: Class,
        cancel: CancellationToken,
        request_id: RequestId,
        permit_tx: oneshot::Sender<SchedulerPermit>,
    ) -> Self {
        Self {
            class,
            queued_at: Instant::now(),
            cancel,
            request_id,
            permit_tx,
        }
    }
}

/// Per-class waiter queue.
///
/// Implemented as a trait so the v1 FIFO impl can be swapped for a
/// per-tenant fair-queue implementation without touching the scheduler.
/// All methods are sync; the queue is a contention point on slot release
/// and admission, so it uses `parking_lot::Mutex` rather than an async
/// mutex.
pub trait ClassQueue: Send + Sync {
    /// Append a waiter. Returns `Err(waiter)` when the queue is at its
    /// configured limit so the caller can convert the rejection into a
    /// 429 response.
    fn try_enqueue(&self, waiter: Waiter) -> Result<(), Waiter>;

    /// Pop the next waiter, or `None` when empty.
    fn pop_eligible(&self) -> Option<Waiter>;

    /// Wall-clock age of the head waiter (used by the dispatcher's
    /// starvation override). `None` when empty.
    fn head_age(&self) -> Option<Duration>;

    /// Current depth, including waiters whose cancel token has fired.
    fn depth(&self) -> usize;

    /// Configured maximum depth (the queue-size limit), for metrics.
    fn capacity(&self) -> usize;

    /// Drain any leading run of waiters whose cancel token has fired
    /// (clients that walked away while queued). One call removes every
    /// consecutive cancelled head in a single lock acquisition; the
    /// first non-cancelled or empty position stops the scan.
    fn drop_cancelled_head(&self);
}

/// First-in, first-out per-class queue with a fixed maximum depth.
pub struct FifoClassQueue {
    waiters: Mutex<VecDeque<Waiter>>,
    max: usize,
}

impl FifoClassQueue {
    pub fn new(max: usize) -> Self {
        Self {
            waiters: Mutex::new(VecDeque::with_capacity(max.min(64))),
            max,
        }
    }
}

impl ClassQueue for FifoClassQueue {
    fn try_enqueue(&self, waiter: Waiter) -> Result<(), Waiter> {
        let mut guard = self.waiters.lock();
        if guard.len() >= self.max {
            return Err(waiter);
        }
        guard.push_back(waiter);
        Ok(())
    }

    fn pop_eligible(&self) -> Option<Waiter> {
        self.waiters.lock().pop_front()
    }

    fn head_age(&self) -> Option<Duration> {
        self.waiters.lock().front().map(|w| w.queued_at.elapsed())
    }

    fn depth(&self) -> usize {
        self.waiters.lock().len()
    }

    fn capacity(&self) -> usize {
        self.max
    }

    fn drop_cancelled_head(&self) {
        let mut guard = self.waiters.lock();
        while guard.front().is_some_and(|w| w.cancel.is_cancelled()) {
            guard.pop_front();
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/middleware/scheduler/queue/tests.rs"]
mod tests;
