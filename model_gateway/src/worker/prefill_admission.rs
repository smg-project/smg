use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::sync::Notify;

use super::{Worker, WorkerLoadGuard};
use crate::observability::metrics::Metrics;

const HEAD_RECHECK_INTERVAL: Duration = Duration::from_secs(1);

/// The result of one capacity-aware Prefill worker selection attempt.
///
/// The callback passed to [`PrefillAdmission::admit`] must first return
/// [`Self::Unavailable`] when no healthy, model-matching worker exists. It
/// must then return [`Self::AtCapacity`] without running its routing policy
/// when no remaining candidate has capacity. This matters for policies such
/// as cache-aware routing whose selection method commits routing state.
pub enum PrefillAdmissionAttempt<T> {
    Selected(PrefillSelection<T>),
    AtCapacity,
    Unavailable,
}

/// A selected worker and the caller-owned value associated with it.
///
/// Callers cannot construct this type directly. Use
/// [`PrefillSelectionContext::select`] so the capacity check and selection
/// stay under the admission lock.
pub struct PrefillSelection<T> {
    worker: Arc<dyn Worker>,
    value: T,
}

/// Capacity view supplied to a worker selection attempt.
pub struct PrefillSelectionContext<'a> {
    max_inflight_requests_per_worker: usize,
    selection_allowed: bool,
    _lock: &'a PrefillAdmissionState,
}

impl PrefillSelectionContext<'_> {
    /// Return whether this Router can start another Prefill request on `worker`.
    /// This returns `false` for a new request when an older request is queued;
    /// the caller can still detect `Unavailable`, but cannot run its policy or
    /// bypass the queue.
    pub fn has_capacity(&self, worker: &Arc<dyn Worker>) -> bool {
        self.selection_allowed && worker.load() < self.max_inflight_requests_per_worker
    }

    /// Commit a worker selected from candidates for which
    /// [`Self::has_capacity`] returned `true`.
    pub fn select<T>(&self, worker: Arc<dyn Worker>, value: T) -> PrefillAdmissionAttempt<T> {
        if self.has_capacity(&worker) {
            PrefillAdmissionAttempt::Selected(PrefillSelection { worker, value })
        } else {
            PrefillAdmissionAttempt::AtCapacity
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillAdmissionRejection {
    QueueFull,
    QueueTimeout,
    Unavailable,
}

/// A successful Prefill admission.
#[must_use = "dropping the result releases the Prefill reservation"]
pub struct AdmittedPrefill<T> {
    pub selected: T,
    pub reservation: PrefillReservation,
}

/// One Router-local Prefill slot on a specific worker.
///
/// Dropping this value releases both the admission count and the worker load,
/// then wakes the head of the Router-wide FIFO queue.
#[must_use = "the reservation must be held until the Prefill phase ends"]
pub struct PrefillReservation {
    inner: Arc<PrefillAdmissionInner>,
    worker: Arc<dyn Worker>,
    load_guard: Option<WorkerLoadGuard>,
}

impl Drop for PrefillReservation {
    fn drop(&mut self) {
        let Some(load_guard) = self.load_guard.take() else {
            return;
        };

        let next = self.inner.release(&self.worker, load_guard);
        if let Some(waiter) = next {
            waiter.notify.notify_one();
        }
    }
}

/// The Prefill load held for one client request.
///
/// With admission enabled this is a queue reservation bounded by the per-worker
/// limit. With admission disabled the load is still tracked, but not bounded.
/// Batch sub-requests share one reservation and take one unit of unbounded load
/// each, which keeps worker load accounting the same as before admission.
pub enum PrefillLoadGuard {
    Admission {
        _reservation: Arc<PrefillReservation>,
    },
    Unbounded {
        _guard: WorkerLoadGuard,
    },
}

impl PrefillLoadGuard {
    /// Acquire another guard for the same request.
    ///
    /// This is explicit instead of `Clone` because the unbounded variant
    /// increments externally visible worker load.
    pub(crate) fn replicate(&self) -> Self {
        match self {
            Self::Admission { _reservation } => Self::Admission {
                _reservation: Arc::clone(_reservation),
            },
            Self::Unbounded { _guard } => Self::Unbounded {
                _guard: _guard.replicate(),
            },
        }
    }

    /// One guard per backend sub-request of the same client request.
    pub(crate) fn replicate_to(self, count: usize) -> Vec<Self> {
        let mut guards: Vec<Self> = (1..count).map(|_| self.replicate()).collect();
        guards.push(self);
        guards
    }
}

/// A candidate-selection error that can also mean "every eligible Prefill
/// worker is at its limit", which makes the request wait instead of fail.
pub trait PrefillCandidateError {
    fn is_at_capacity(&self) -> bool;
}

impl<E: PrefillCandidateError + ?Sized> PrefillCandidateError for Box<E> {
    fn is_at_capacity(&self) -> bool {
        (**self).is_at_capacity()
    }
}

/// Why [`acquire_prefill`] could not reserve Prefill capacity.
pub enum PrefillAcquireError<E> {
    /// Candidate selection found no worker to wait for. With admission
    /// enabled this is the verdict of the last attempt at the queue head.
    Candidate(E),
    /// The request could not be queued or waited too long.
    Rejected(PrefillAdmissionRejection),
}

/// Select a Prefill worker for one client request and hold its load.
///
/// With admission enabled the selection runs at the head of the FIFO queue and
/// under the admission lock, so `select` must filter candidates with
/// [`PrefillSelectionContext::has_capacity`] before it invokes a stateful
/// routing policy. With admission disabled `select` receives `None` and runs
/// once.
pub async fn acquire_prefill<C, E, S, P>(
    admission: Option<&PrefillAdmission>,
    routing_key: Option<&str>,
    prefill_of: P,
    mut select: S,
) -> Result<(C, PrefillLoadGuard), PrefillAcquireError<E>>
where
    S: for<'a> FnMut(Option<&PrefillSelectionContext<'a>>) -> Result<C, E> + Send,
    P: Fn(&C) -> &Arc<dyn Worker> + Send + Sync,
    E: PrefillCandidateError + Send,
{
    let Some(admission) = admission else {
        let candidate = select(None).map_err(PrefillAcquireError::Candidate)?;
        let guard = PrefillLoadGuard::Unbounded {
            _guard: WorkerLoadGuard::with_key(Arc::clone(prefill_of(&candidate)), routing_key),
        };
        return Ok((candidate, guard));
    };

    // The selection verdict is kept so an unavailable fleet answers with the
    // leg's own reason (a shed, a missing leg) rather than a generic one.
    let mut verdict = None;
    let admitted = admission
        .admit(routing_key, |capacity| match select(Some(capacity)) {
            Ok(candidate) => capacity.select(Arc::clone(prefill_of(&candidate)), candidate),
            Err(error) if error.is_at_capacity() => PrefillAdmissionAttempt::AtCapacity,
            Err(error) => {
                verdict = Some(error);
                PrefillAdmissionAttempt::Unavailable
            }
        })
        .await;

    match (admitted, verdict) {
        (Ok(admitted), _) => Ok((
            admitted.selected,
            PrefillLoadGuard::Admission {
                _reservation: Arc::new(admitted.reservation),
            },
        )),
        (Err(PrefillAdmissionRejection::Unavailable), Some(error)) => {
            Err(PrefillAcquireError::Candidate(error))
        }
        (Err(rejection), _) => Err(PrefillAcquireError::Rejected(rejection)),
    }
}

/// Router-local, strict FIFO admission for the Prefill phase of PD requests.
///
/// Waiting requests do not hold a worker. Only the queue head receives a
/// capacity-positive view and can select a worker. The callback runs while the
/// admission lock is held, so candidate filtering, a stateful routing-policy
/// call, and reservation are serialized with every other Prefill admission in
/// this Router process.
#[derive(Clone)]
pub struct PrefillAdmission {
    inner: Arc<PrefillAdmissionInner>,
}

impl std::fmt::Debug for PrefillAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillAdmission")
            .field(
                "max_inflight_requests_per_worker",
                &self.inner.max_inflight_requests_per_worker,
            )
            .field("queue_size", &self.inner.queue_size)
            .field("queue_timeout", &self.inner.queue_timeout)
            .finish_non_exhaustive()
    }
}

struct PrefillAdmissionInner {
    max_inflight_requests_per_worker: usize,
    queue_size: usize,
    queue_timeout: Duration,
    state: Mutex<PrefillAdmissionState>,
}

struct PrefillAdmissionState {
    waiters: VecDeque<Arc<PrefillWaiter>>,
}

struct PrefillWaiter {
    notify: Notify,
}

impl PrefillAdmission {
    pub fn new(
        max_inflight_requests_per_worker: usize,
        queue_size: usize,
        queue_timeout: Duration,
    ) -> Self {
        assert!(
            max_inflight_requests_per_worker > 0,
            "Prefill admission requires a positive per-worker limit"
        );
        assert!(
            queue_size == 0 || !queue_timeout.is_zero(),
            "Prefill admission requires a positive timeout when its queue is enabled"
        );

        Self {
            inner: Arc::new(PrefillAdmissionInner {
                max_inflight_requests_per_worker,
                queue_size,
                queue_timeout,
                state: Mutex::new(PrefillAdmissionState {
                    waiters: VecDeque::with_capacity(queue_size.min(64)),
                }),
            }),
        }
    }

    /// Admit one client request and select its Prefill worker.
    ///
    /// The callback may run more than once while a request waits. On each run
    /// it must refresh the worker set and health state. It must filter
    /// candidates with [`PrefillSelectionContext::has_capacity`] before it
    /// invokes a stateful routing policy, then return the result through
    /// [`PrefillSelectionContext::select`]. When an older request is already
    /// queued, the callback runs only to detect `Unavailable`; capacity is
    /// hidden so the new request cannot invoke its policy or bypass the queue.
    pub async fn admit<T, F>(
        &self,
        routing_key: Option<&str>,
        mut attempt: F,
    ) -> Result<AdmittedPrefill<T>, PrefillAdmissionRejection>
    where
        F: for<'a> FnMut(&PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<T> + Send,
    {
        let (waiter, queued_at) = {
            let mut state = self.inner.state.lock();
            let selection_allowed = state.waiters.is_empty();

            match self.try_admit_locked(&state, routing_key, selection_allowed, &mut attempt) {
                LockedAttempt::Selected(admitted) => return Ok(admitted),
                LockedAttempt::Unavailable => {
                    Metrics::record_pd_prefill_admission_rejection("unavailable");
                    return Err(PrefillAdmissionRejection::Unavailable);
                }
                LockedAttempt::AtCapacity => {}
            }

            if state.waiters.len() >= self.inner.queue_size {
                Metrics::record_pd_prefill_admission_rejection("queue_full");
                return Err(PrefillAdmissionRejection::QueueFull);
            }

            let waiter = Arc::new(PrefillWaiter {
                notify: Notify::new(),
            });
            let queued_at = Instant::now();
            state.waiters.push_back(Arc::clone(&waiter));
            Metrics::set_pd_prefill_admission_queued(state.waiters.len());
            (waiter, queued_at)
        };

        let mut ticket = QueueTicket {
            inner: Arc::clone(&self.inner),
            waiter,
            queued: true,
            queued_at,
        };
        let timeout = tokio::time::sleep(self.inner.queue_timeout);
        let deadline = timeout.deadline();
        tokio::pin!(timeout);

        loop {
            let waiter = Arc::clone(&ticket.waiter);
            let is_head = {
                let state = self.inner.state.lock();
                ticket.is_head(&state)
            };
            let recheck = tokio::time::sleep(HEAD_RECHECK_INTERVAL);
            tokio::pin!(recheck);
            tokio::select! {
                biased;
                () = &mut timeout => {
                    Metrics::record_pd_prefill_admission_rejection("queue_timeout");
                    return Err(PrefillAdmissionRejection::QueueTimeout);
                }
                () = waiter.notify.notified() => {}
                // Open circuit breakers become HalfOpen on their next state
                // check and do not emit a registry event. Poll only the queue
                // head so that recovery can make progress without waking the
                // rest of the FIFO queue.
                () = &mut recheck, if is_head => {}
            }

            if tokio::time::Instant::now() >= deadline {
                Metrics::record_pd_prefill_admission_rejection("queue_timeout");
                return Err(PrefillAdmissionRejection::QueueTimeout);
            }

            let decision = {
                let mut state = self.inner.state.lock();
                if ticket.is_head(&state) {
                    match self.try_admit_locked(&state, routing_key, true, &mut attempt) {
                        LockedAttempt::Selected(admitted) => {
                            let next = ticket.remove_head(&mut state);
                            Some(QueuedAttempt::Selected(admitted, next))
                        }
                        LockedAttempt::Unavailable => {
                            let next = ticket.remove_head(&mut state);
                            Some(QueuedAttempt::Unavailable(next))
                        }
                        LockedAttempt::AtCapacity => None,
                    }
                } else {
                    None
                }
            };

            match decision {
                Some(QueuedAttempt::Selected(admitted, next)) => {
                    if let Some(waiter) = next {
                        waiter.notify.notify_one();
                    }
                    return Ok(admitted);
                }
                Some(QueuedAttempt::Unavailable(next)) => {
                    if let Some(waiter) = next {
                        waiter.notify.notify_one();
                    }
                    Metrics::record_pd_prefill_admission_rejection("unavailable");
                    return Err(PrefillAdmissionRejection::Unavailable);
                }
                None => {}
            }
        }
    }

    /// Wake the queue head after worker registration, removal, replacement, or
    /// health state changes can affect its selection result.
    pub fn notify_capacity_changed(&self) {
        if let Some(waiter) = self.inner.head_waiter() {
            waiter.notify.notify_one();
        }
    }

    pub fn queued_requests(&self) -> usize {
        self.inner.state.lock().waiters.len()
    }

    fn try_admit_locked<T, F>(
        &self,
        state: &PrefillAdmissionState,
        routing_key: Option<&str>,
        selection_allowed: bool,
        attempt: &mut F,
    ) -> LockedAttempt<T>
    where
        F: for<'a> FnMut(&PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<T>,
    {
        let selection = attempt(&PrefillSelectionContext {
            max_inflight_requests_per_worker: self.inner.max_inflight_requests_per_worker,
            selection_allowed,
            _lock: state,
        });

        match selection {
            PrefillAdmissionAttempt::Selected(PrefillSelection { worker, value }) => {
                let Some(load_guard) = WorkerLoadGuard::try_new_with_key(
                    Arc::clone(&worker),
                    routing_key,
                    self.inner.max_inflight_requests_per_worker,
                ) else {
                    return LockedAttempt::AtCapacity;
                };
                Metrics::set_pd_prefill_admission_inflight(worker.url(), worker.load());

                LockedAttempt::Selected(AdmittedPrefill {
                    selected: value,
                    reservation: PrefillReservation {
                        inner: Arc::clone(&self.inner),
                        worker,
                        load_guard: Some(load_guard),
                    },
                })
            }
            PrefillAdmissionAttempt::AtCapacity => LockedAttempt::AtCapacity,
            PrefillAdmissionAttempt::Unavailable => LockedAttempt::Unavailable,
        }
    }
}

enum LockedAttempt<T> {
    Selected(AdmittedPrefill<T>),
    AtCapacity,
    Unavailable,
}

enum QueuedAttempt<T> {
    Selected(AdmittedPrefill<T>, Option<Arc<PrefillWaiter>>),
    Unavailable(Option<Arc<PrefillWaiter>>),
}

impl PrefillAdmissionInner {
    fn head_waiter(&self) -> Option<Arc<PrefillWaiter>> {
        self.state.lock().waiters.front().cloned()
    }

    fn release(
        &self,
        worker: &Arc<dyn Worker>,
        load_guard: WorkerLoadGuard,
    ) -> Option<Arc<PrefillWaiter>> {
        let state = self.state.lock();
        drop(load_guard);
        Metrics::set_pd_prefill_admission_inflight(worker.url(), worker.load());

        state.waiters.front().cloned()
    }
}

struct QueueTicket {
    inner: Arc<PrefillAdmissionInner>,
    waiter: Arc<PrefillWaiter>,
    queued: bool,
    queued_at: Instant,
}

impl QueueTicket {
    fn is_head(&self, state: &PrefillAdmissionState) -> bool {
        state
            .waiters
            .front()
            .is_some_and(|head| Arc::ptr_eq(head, &self.waiter))
    }

    fn remove_head(&mut self, state: &mut PrefillAdmissionState) -> Option<Arc<PrefillWaiter>> {
        let _ = state.waiters.pop_front();
        self.queued = false;
        Metrics::set_pd_prefill_admission_queued(state.waiters.len());
        Metrics::record_pd_prefill_admission_wait(self.queued_at.elapsed());
        state.waiters.front().cloned()
    }
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }

        let next = {
            let mut state = self.inner.state.lock();
            let Some(position) = state
                .waiters
                .iter()
                .position(|waiter| Arc::ptr_eq(waiter, &self.waiter))
            else {
                return;
            };
            let was_head = position == 0;
            let _ = state.waiters.remove(position);
            self.queued = false;
            Metrics::set_pd_prefill_admission_queued(state.waiters.len());
            Metrics::record_pd_prefill_admission_wait(self.queued_at.elapsed());
            if was_head {
                state.waiters.front().cloned()
            } else {
                None
            }
        };

        if let Some(waiter) = next {
            waiter.notify.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };

    use openai_protocol::worker::WorkerStatus;
    use tokio::sync::{mpsc, oneshot};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, CircuitBreakerConfig, WorkerType};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill)
                .build(),
        )
    }

    async fn wait_for_queued(admission: &PrefillAdmission, expected: usize) {
        for _ in 0..1_000 {
            if admission.queued_requests() == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "queue depth did not reach {expected}; actual={}",
            admission.queued_requests()
        );
    }

    #[tokio::test]
    async fn one_client_request_uses_one_worker_slot() {
        let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let worker = worker("http://prefill-one");

        let admitted = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), "selected")
            })
            .await
            .unwrap();

        assert_eq!(admitted.selected, "selected");
        assert_eq!(worker.load(), 1);
        drop(admitted.reservation);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn zero_size_queue_rejects_at_capacity() {
        let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let worker = worker("http://prefill-no-queue");
        let first = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();

        let result = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await;
        assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
        drop(first);
    }

    #[tokio::test]
    async fn full_queue_rejects_without_bypassing_waiter() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let worker = worker("http://prefill-full-queue");
        let first = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is aborted and joined before the test ends"
        )]
        let waiter = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;

        let policy_called = Arc::new(AtomicBool::new(false));
        let result = admission
            .admit(None, {
                let policy_called = Arc::clone(&policy_called);
                let worker = Arc::clone(&worker);
                move |capacity| {
                    if !capacity.has_capacity(&worker) {
                        return PrefillAdmissionAttempt::AtCapacity;
                    }
                    policy_called.store(true, Ordering::Relaxed);
                    capacity.select(Arc::clone(&worker), ())
                }
            })
            .await;
        assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
        assert!(!policy_called.load(Ordering::Relaxed));

        waiter.abort();
        assert!(matches!(
            waiter.await,
            Err(error) if error.is_cancelled()
        ));
        drop(first);
    }

    #[tokio::test(start_paused = true)]
    async fn queued_request_times_out_and_removes_its_ticket() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
        let worker = worker("http://prefill-timeout");
        let first = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let waiter = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;

        tokio::time::advance(Duration::from_secs(5)).await;
        assert!(matches!(
            waiter.await.unwrap(),
            Err(PrefillAdmissionRejection::QueueTimeout)
        ));
        assert_eq!(admission.queued_requests(), 0);
        drop(first);
    }

    #[tokio::test]
    async fn released_worker_load_is_visible_to_the_next_selection() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let worker = worker("http://prefill-release-order");
        let first = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let waiter = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                admission
                    .admit(None, move |capacity| {
                        if !capacity.has_capacity(&worker) {
                            return PrefillAdmissionAttempt::AtCapacity;
                        }
                        assert_eq!(worker.load(), 0);
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;

        drop(first);
        let admitted = waiter.await.unwrap().unwrap();
        drop(admitted);
    }

    #[tokio::test]
    async fn queued_requests_are_admitted_in_strict_fifo_order() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-fifo");
        let initial = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();
        let (acquired_tx, mut acquired_rx) = mpsc::unbounded_channel();
        let (release_first_tx, release_first_rx) = oneshot::channel();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let first = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let acquired_tx = acquired_tx.clone();
            async move {
                let admitted = admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
                    .unwrap();
                acquired_tx.send(1).unwrap();
                let _ = release_first_rx.await;
                drop(admitted);
            }
        });
        wait_for_queued(&admission, 1).await;

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let second = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                let admitted = admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
                    .unwrap();
                acquired_tx.send(2).unwrap();
                drop(admitted);
            }
        });
        wait_for_queued(&admission, 2).await;

        drop(initial);
        assert_eq!(acquired_rx.recv().await, Some(1));
        assert!(acquired_rx.try_recv().is_err());
        release_first_tx.send(()).unwrap();
        assert_eq!(acquired_rx.recv().await, Some(2));

        first.await.unwrap();
        second.await.unwrap();
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn admitted_head_wakes_next_head_while_capacity_remains() {
        let admission = PrefillAdmission::new(2, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-drain");
        let available = Arc::new(AtomicBool::new(false));
        let (acquired_tx, mut acquired_rx) = mpsc::unbounded_channel();
        let (release_first_tx, release_first_rx) = oneshot::channel();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let first = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let available = Arc::clone(&available);
            let acquired_tx = acquired_tx.clone();
            async move {
                let admitted = admission
                    .admit(None, move |capacity| {
                        if available.load(Ordering::Relaxed) {
                            capacity.select(Arc::clone(&worker), ())
                        } else {
                            PrefillAdmissionAttempt::AtCapacity
                        }
                    })
                    .await
                    .unwrap();
                acquired_tx.send(1).unwrap();
                let _ = release_first_rx.await;
                drop(admitted);
            }
        });
        wait_for_queued(&admission, 1).await;

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let second = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let available = Arc::clone(&available);
            async move {
                let admitted = admission
                    .admit(None, move |capacity| {
                        if available.load(Ordering::Relaxed) {
                            capacity.select(Arc::clone(&worker), ())
                        } else {
                            PrefillAdmissionAttempt::AtCapacity
                        }
                    })
                    .await
                    .unwrap();
                acquired_tx.send(2).unwrap();
                drop(admitted);
            }
        });
        wait_for_queued(&admission, 2).await;

        available.store(true, Ordering::Relaxed);
        admission.notify_capacity_changed();
        assert_eq!(acquired_rx.recv().await, Some(1));
        assert_eq!(acquired_rx.recv().await, Some(2));
        release_first_tx.send(()).unwrap();

        first.await.unwrap();
        second.await.unwrap();
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn queue_head_rechecks_circuit_breaker_recovery_without_event() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
        let full = worker("http://prefill-full");
        let occupied = admission
            .admit(None, {
                let full = Arc::clone(&full);
                move |capacity| capacity.select(Arc::clone(&full), ())
            })
            .await
            .unwrap();
        let recovering: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://prefill-recovering")
                .worker_type(WorkerType::Prefill)
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold: 1,
                    timeout_duration: Duration::from_millis(250),
                    ..Default::default()
                })
                .build(),
        );
        recovering.set_status(WorkerStatus::Ready);
        recovering.record_circuit_breaker_outcome(false);
        assert!(!recovering.is_available());

        let admitted = tokio::time::timeout(
            Duration::from_secs(3),
            admission.admit(None, {
                let full = Arc::clone(&full);
                let recovering = Arc::clone(&recovering);
                move |capacity| {
                    if recovering.is_available() {
                        capacity.select(Arc::clone(&recovering), ())
                    } else {
                        capacity.select(Arc::clone(&full), ())
                    }
                }
            }),
        )
        .await
        .expect("queue head should recheck a silently recovered worker")
        .unwrap();

        assert_eq!(full.load(), 1);
        assert_eq!(recovering.load(), 1);
        drop(admitted);
        drop(occupied);
        assert_eq!(full.load(), 0);
        assert_eq!(recovering.load(), 0);
    }

    #[tokio::test]
    async fn cancelled_waiters_are_removed_without_blocking_fifo() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-cancel-head");
        let initial = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is aborted and joined before the test ends"
        )]
        let cancelled = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is aborted and joined before the test ends"
        )]
        let middle = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 2).await;
        middle.abort();
        assert!(matches!(
            middle.await,
            Err(error) if error.is_cancelled()
        ));
        wait_for_queued(&admission, 1).await;

        let attempts = Arc::new(AtomicUsize::new(0));
        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let next = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let attempts = Arc::clone(&attempts);
            async move {
                admission
                    .admit(None, move |capacity| {
                        attempts.fetch_add(1, Ordering::Relaxed);
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 2).await;

        cancelled.abort();
        assert!(matches!(
            cancelled.await,
            Err(error) if error.is_cancelled()
        ));
        wait_for_queued(&admission, 1).await;
        for _ in 0..1_000 {
            if attempts.load(Ordering::Relaxed) > 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(attempts.load(Ordering::Relaxed) > 0);

        drop(initial);
        let admitted = next.await.unwrap().unwrap();
        drop(admitted);
    }

    #[tokio::test]
    async fn queued_request_selects_again_without_worker_binding() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let first_worker = worker("http://prefill-a");
        let second_worker = worker("http://prefill-b");
        let first = admission
            .admit(None, {
                let worker = Arc::clone(&first_worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();
        let second = admission
            .admit(None, {
                let worker = Arc::clone(&second_worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();
        let policy_calls = Arc::new(AtomicUsize::new(0));

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let waiter = tokio::spawn({
            let admission = admission.clone();
            let first_worker = Arc::clone(&first_worker);
            let second_worker = Arc::clone(&second_worker);
            let policy_calls = Arc::clone(&policy_calls);
            async move {
                admission
                    .admit(None, move |capacity| {
                        let selected = if capacity.has_capacity(&first_worker) {
                            Some(Arc::clone(&first_worker))
                        } else if capacity.has_capacity(&second_worker) {
                            Some(Arc::clone(&second_worker))
                        } else {
                            None
                        };
                        let Some(selected) = selected else {
                            return PrefillAdmissionAttempt::AtCapacity;
                        };
                        policy_calls.fetch_add(1, Ordering::Relaxed);
                        let url = selected.url().to_owned();
                        capacity.select(selected, url)
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;
        assert_eq!(policy_calls.load(Ordering::Relaxed), 0);

        drop(second);
        let admitted = waiter.await.unwrap().unwrap();
        assert_eq!(admitted.selected, "http://prefill-b");
        assert_eq!(policy_calls.load(Ordering::Relaxed), 1);
        drop(admitted);
        drop(first);
    }

    #[tokio::test]
    async fn unavailable_head_exposes_the_next_waiter() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-unavailable-head");
        let unavailable = Arc::new(AtomicBool::new(false));

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let first = tokio::spawn({
            let admission = admission.clone();
            let unavailable = Arc::clone(&unavailable);
            async move {
                admission
                    .admit(None, move |_| {
                        if unavailable.load(Ordering::Relaxed) {
                            PrefillAdmissionAttempt::<()>::Unavailable
                        } else {
                            PrefillAdmissionAttempt::AtCapacity
                        }
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let second = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 2).await;

        unavailable.store(true, Ordering::Relaxed);
        admission.notify_capacity_changed();
        assert!(matches!(
            first.await.unwrap(),
            Err(PrefillAdmissionRejection::Unavailable)
        ));
        let admitted = second.await.unwrap().unwrap();
        drop(admitted);
    }

    #[tokio::test]
    async fn unavailable_is_not_queued() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-unavailable");
        let occupied = admission
            .admit(None, {
                let worker = Arc::clone(&worker);
                move |capacity| capacity.select(Arc::clone(&worker), ())
            })
            .await
            .unwrap();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is aborted and joined before the test ends"
        )]
        let queued = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            async move {
                admission
                    .admit(None, move |capacity| {
                        capacity.select(Arc::clone(&worker), ())
                    })
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;

        let result = admission
            .admit(None, |_| PrefillAdmissionAttempt::<()>::Unavailable)
            .await;

        assert!(matches!(
            result,
            Err(PrefillAdmissionRejection::Unavailable)
        ));
        assert_eq!(admission.queued_requests(), 1);

        queued.abort();
        assert!(matches!(
            queued.await,
            Err(error) if error.is_cancelled()
        ));
        drop(occupied);
    }
}
