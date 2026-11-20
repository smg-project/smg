use std::{
    collections::VecDeque,
    future::{ready, Future},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::identity::basic::{Worker, WorkerLoadGuard};
use tokio::{sync::Notify, time::Instant as TokioInstant};

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
    /// The attempt took no worker and the next one may: it runs again at
    /// once. A queued request keeps its place and its deadline. The caller
    /// bounds how often it returns this.
    Again,
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
    /// This returns `false` for a new request when an older request is queued.
    /// The caller can still detect `Unavailable`.
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

impl PrefillAdmissionRejection {
    fn record(self) -> Self {
        let reason = match self {
            Self::QueueFull => "queue_full",
            Self::QueueTimeout => "queue_timeout",
            Self::Unavailable => "unavailable",
        };
        Metrics::record_pd_prefill_admission_rejection(reason);
        self
    }
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
    fn unbounded(worker: &Arc<dyn Worker>, routing_key: Option<&str>) -> Self {
        Self::Unbounded {
            _guard: WorkerLoadGuard::with_key(Arc::clone(worker), routing_key),
        }
    }

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

    /// Whether [`acquire_decided`] decides again at once after this error,
    /// in the request's place and within its deadline. `again` counts the
    /// times the request has done so: the error bounds it.
    fn decides_again(&self, _again: usize) -> bool {
        false
    }
}

impl<E: PrefillCandidateError + ?Sized> PrefillCandidateError for Box<E> {
    fn is_at_capacity(&self) -> bool {
        (**self).is_at_capacity()
    }

    fn decides_again(&self, again: usize) -> bool {
        (**self).decides_again(again)
    }
}

/// Why [`acquire_prefill`] or [`acquire_decided`] could not reserve Prefill capacity.
#[derive(Debug)]
pub enum PrefillAcquireError<E> {
    /// The probe or selection found no worker to wait for.
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
        let guard = PrefillLoadGuard::unbounded(prefill_of(&candidate), routing_key);
        return Ok((candidate, guard));
    };

    // The selection verdict is kept so an unavailable fleet answers with the
    // role's own reason (a shed, a missing role) rather than a generic one.
    let mut verdict = None;
    let admitted = admission
        .admit(routing_key, |capacity| {
            candidate_attempt(select(Some(capacity)), capacity, &prefill_of, &mut verdict)
        })
        .await;

    finish_acquire(admitted.map_err(PrefillAcquireError::Rejected), verdict)
}

/// [`acquire_prefill`] for an attempt decided outside the admission lock.
///
/// With admission enabled, a capacity error from either half makes the
/// request wait if the queue has room. An error that
/// [decides again](PrefillCandidateError::decides_again) runs both halves
/// again at once, with admission or without. Other errors end the attempt.
/// With admission disabled, the probe is skipped, a successful decision is
/// committed with `None`, and the selected worker's load is held without a limit.
pub async fn acquire_decided<D, C, E, Probe, Decide, Fut, Commit, P>(
    admission: Option<&PrefillAdmission>,
    routing_key: Option<&str>,
    prefill_of: P,
    probe: impl FnMut() -> Probe + Send,
    mut decide: Decide,
    mut commit: Commit,
) -> Result<(C, PrefillLoadGuard), PrefillAcquireError<E>>
where
    Probe: Future<Output = Option<E>> + Send,
    Decide: FnMut() -> Fut + Send,
    Fut: Future<Output = Result<D, E>> + Send,
    D: Send,
    Commit: for<'a> FnMut(D, Option<&PrefillSelectionContext<'a>>) -> Result<C, E> + Send,
    P: Fn(&C) -> &Arc<dyn Worker> + Send + Sync,
    E: PrefillCandidateError + Send,
{
    let mut again = 0;
    let Some(admission) = admission else {
        loop {
            let decision = decide().await.map_err(PrefillAcquireError::Candidate)?;
            match commit(decision, None) {
                Ok(candidate) => {
                    let guard = PrefillLoadGuard::unbounded(prefill_of(&candidate), routing_key);
                    return Ok((candidate, guard));
                }
                Err(error) if error.decides_again(again) => again += 1,
                Err(error) => return Err(PrefillAcquireError::Candidate(error)),
            }
        }
    };

    let mut verdict = None;
    let admitted = admission
        .admit_decided(
            routing_key,
            probe,
            || {
                let decision = decide();
                async { Some(decision.await) }
            },
            |decision, capacity| match decision
                .and_then(|decision| commit(decision, Some(capacity)))
            {
                Err(error) if error.decides_again(again) => {
                    again += 1;
                    PrefillAdmissionAttempt::Again
                }
                committed => candidate_attempt(committed, capacity, &prefill_of, &mut verdict),
            },
        )
        .await;

    finish_acquire(admitted, verdict)
}

fn candidate_attempt<C, E: PrefillCandidateError>(
    candidate: Result<C, E>,
    capacity: &PrefillSelectionContext<'_>,
    prefill_of: &impl Fn(&C) -> &Arc<dyn Worker>,
    verdict: &mut Option<E>,
) -> PrefillAdmissionAttempt<C> {
    match candidate {
        Ok(candidate) => capacity.select(Arc::clone(prefill_of(&candidate)), candidate),
        Err(error) if error.is_at_capacity() => PrefillAdmissionAttempt::AtCapacity,
        Err(error) => {
            *verdict = Some(error);
            PrefillAdmissionAttempt::Unavailable
        }
    }
}

fn finish_acquire<C, E>(
    admitted: Result<AdmittedPrefill<C>, PrefillAcquireError<E>>,
    verdict: Option<E>,
) -> Result<(C, PrefillLoadGuard), PrefillAcquireError<E>> {
    match (admitted, verdict) {
        (Ok(admitted), _) => Ok((
            admitted.selected,
            PrefillLoadGuard::Admission {
                _reservation: Arc::new(admitted.reservation),
            },
        )),
        (
            Err(PrefillAcquireError::Rejected(PrefillAdmissionRejection::Unavailable)),
            Some(error),
        ) => Err(PrefillAcquireError::Candidate(error)),
        (Err(error), _) => Err(error),
    }
}

/// Router-local, strict FIFO admission for the Prefill phase of PD requests.
///
/// Waiting requests hold no worker. While the queue has a head, only that
/// request sees capacity. Committing a selection and reserving its load both
/// run under the admission lock.
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
        let mut commit = |(), capacity: &PrefillSelectionContext<'_>| attempt(capacity);
        let ticket = {
            let mut state = self.inner.state.lock();
            let selection_allowed = state.waiters.is_empty();

            loop {
                match self.try_admit_locked(
                    &state,
                    routing_key,
                    selection_allowed,
                    Some(()),
                    &mut commit,
                ) {
                    LockedAttempt::Selected(admitted) => return Ok(admitted),
                    LockedAttempt::Unavailable => {
                        return Err(PrefillAdmissionRejection::Unavailable.record());
                    }
                    LockedAttempt::AtCapacity => break self.enqueue_locked(&mut state)?,
                    LockedAttempt::Again => {}
                }
            }
        };

        self.wait_for_admission(routing_key, ticket, || ready(Some(())), commit)
            .await
    }

    /// Admission for an attempt whose decision is made outside the admission lock.
    ///
    /// If another request is waiting, run `probe` outside the lock. A verdict
    /// ends the request. Otherwise join behind any remaining waiters if the
    /// queue has room.
    ///
    /// If no request is waiting, decide outside the lock and commit under it.
    /// The commit sees no capacity if another request is now queued.
    /// `Selected` admits the request, `Unavailable` refuses it, and
    /// `AtCapacity` queues it if there is room. `Again` decides again at
    /// once, with no place taken in the queue.
    ///
    /// A queued request waits for a notification, the head's one-second
    /// recheck, or the queue deadline. Only the head decides, outside the
    /// lock and within that deadline, then commits under the lock. `Selected`
    /// and `Unavailable` leave the queue and wake the next head.
    /// `AtCapacity` keeps the head in place to decide again at its next wake.
    /// `Again` keeps it in place to decide again at once, within the same
    /// deadline.
    ///
    /// A `None` decision returns `Unavailable` without calling `commit`.
    /// The commit takes ownership of the decision and is responsible for
    /// giving it up if it cannot select it.
    pub async fn admit_decided<D, T, E, Probe, Decide, Fut, Commit>(
        &self,
        routing_key: Option<&str>,
        mut probe: impl FnMut() -> Probe + Send,
        mut decide: Decide,
        mut commit: Commit,
    ) -> Result<AdmittedPrefill<T>, PrefillAcquireError<E>>
    where
        Probe: Future<Output = Option<E>> + Send,
        Decide: FnMut() -> Fut + Send,
        Fut: Future<Output = Option<D>> + Send,
        D: Send,
        Commit: for<'a> FnMut(D, &PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<T> + Send,
    {
        let ticket = if self.queued_requests() > 0 {
            if let Some(verdict) = probe().await {
                // Counted as every other request with no worker to wait for.
                let _ = PrefillAdmissionRejection::Unavailable.record();
                return Err(PrefillAcquireError::Candidate(verdict));
            }
            let mut state = self.inner.state.lock();
            if state.waiters.is_empty() {
                None
            } else {
                Some(
                    self.enqueue_locked(&mut state)
                        .map_err(PrefillAcquireError::Rejected)?,
                )
            }
        } else {
            None
        };

        let ticket = if let Some(ticket) = ticket {
            ticket
        } else {
            loop {
                let decision = decide().await;
                let mut state = self.inner.state.lock();
                let selection_allowed = state.waiters.is_empty();

                match self.try_admit_locked(
                    &state,
                    routing_key,
                    selection_allowed,
                    decision,
                    &mut commit,
                ) {
                    LockedAttempt::Selected(admitted) => return Ok(admitted),
                    LockedAttempt::Unavailable => {
                        return Err(PrefillAcquireError::Rejected(
                            PrefillAdmissionRejection::Unavailable.record(),
                        ));
                    }
                    LockedAttempt::AtCapacity => {
                        break self
                            .enqueue_locked(&mut state)
                            .map_err(PrefillAcquireError::Rejected)?;
                    }
                    LockedAttempt::Again => {}
                }
            }
        };

        self.wait_for_admission(routing_key, ticket, decide, commit)
            .await
            .map_err(PrefillAcquireError::Rejected)
    }

    fn enqueue_locked(
        &self,
        state: &mut PrefillAdmissionState,
    ) -> Result<QueueTicket, PrefillAdmissionRejection> {
        if state.waiters.len() >= self.inner.queue_size {
            return Err(PrefillAdmissionRejection::QueueFull.record());
        }

        let waiter = Arc::new(PrefillWaiter {
            notify: Notify::new(),
        });
        let queued_at = Instant::now();
        state.waiters.push_back(Arc::clone(&waiter));
        Metrics::set_pd_prefill_admission_queued(state.waiters.len());
        Ok(QueueTicket {
            inner: Arc::clone(&self.inner),
            waiter,
            queued: true,
            queued_at,
        })
    }

    async fn wait_for_admission<D, T, Decide, Fut, Commit>(
        &self,
        routing_key: Option<&str>,
        mut ticket: QueueTicket,
        mut decide: Decide,
        mut commit: Commit,
    ) -> Result<AdmittedPrefill<T>, PrefillAdmissionRejection>
    where
        Decide: FnMut() -> Fut + Send,
        Fut: Future<Output = Option<D>> + Send,
        D: Send,
        Commit: for<'a> FnMut(D, &PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<T> + Send,
    {
        let timeout = tokio::time::sleep(self.inner.queue_timeout);
        let deadline = timeout.deadline();
        tokio::pin!(timeout);

        let mut again = false;
        loop {
            let is_head = ticket.is_head(&self.inner.state.lock());
            let recheck = tokio::time::sleep(HEAD_RECHECK_INTERVAL);
            tokio::pin!(recheck);
            tokio::select! {
                biased;
                () = &mut timeout => return Err(PrefillAdmissionRejection::QueueTimeout.record()),
                // A head that decides again awaits no wake.
                () = ready(()), if again => again = false,
                () = ticket.waiter.notify.notified() => {}
                // Open circuit breakers become HalfOpen on their next state
                // check and do not emit a registry event. Poll only the queue
                // head so that recovery can make progress without waking the
                // rest of the FIFO queue.
                () = &mut recheck, if is_head => {}
            }

            // Once at the head, this ticket stays there until the request removes it.
            if !is_head && !ticket.is_head(&self.inner.state.lock()) {
                continue;
            }
            let decision = tokio::select! {
                biased;
                () = &mut timeout => return Err(PrefillAdmissionRejection::QueueTimeout.record()),
                decision = async { decide().await } => decision,
            };
            if TokioInstant::now() >= deadline {
                return Err(PrefillAdmissionRejection::QueueTimeout.record());
            }

            let (result, next) = {
                let mut state = self.inner.state.lock();
                let result =
                    match self.try_admit_locked(&state, routing_key, true, decision, &mut commit) {
                        LockedAttempt::Selected(admitted) => Ok(admitted),
                        LockedAttempt::Unavailable => Err(PrefillAdmissionRejection::Unavailable),
                        LockedAttempt::AtCapacity => continue,
                        LockedAttempt::Again => {
                            again = true;
                            continue;
                        }
                    };
                (result, ticket.remove_head(&mut state))
            };

            if let Some(waiter) = next {
                waiter.notify.notify_one();
            }
            return result.map_err(PrefillAdmissionRejection::record);
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

    /// Read a worker's load against the limit without taking the admission
    /// lock. The load can change before the commit checks it again.
    pub fn reads_full(&self, worker: &Arc<dyn Worker>) -> bool {
        worker.load() >= self.inner.max_inflight_requests_per_worker
    }

    fn try_admit_locked<D, T, Commit>(
        &self,
        state: &PrefillAdmissionState,
        routing_key: Option<&str>,
        selection_allowed: bool,
        decision: Option<D>,
        commit: &mut Commit,
    ) -> LockedAttempt<T>
    where
        Commit: for<'a> FnMut(D, &PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<T>,
    {
        let Some(decision) = decision else {
            return LockedAttempt::Unavailable;
        };
        let selection = commit(
            decision,
            &PrefillSelectionContext {
                max_inflight_requests_per_worker: self.inner.max_inflight_requests_per_worker,
                selection_allowed,
                _lock: state,
            },
        );

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
            PrefillAdmissionAttempt::Again => LockedAttempt::Again,
        }
    }
}

enum LockedAttempt<T> {
    Selected(AdmittedPrefill<T>),
    AtCapacity,
    Unavailable,
    Again,
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
        debug_assert!(
            self.is_head(state),
            "only the head removes itself from the front"
        );
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

/// Admission helpers for tests elsewhere in the crate.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use smg_worker::identity::basic::Worker;

    use super::{PrefillAdmissionAttempt, PrefillSelectionContext};

    /// The attempt a test makes when it wants exactly `worker`: it is selected,
    /// with a unit payload, whenever it has capacity.
    pub(crate) fn select_worker(
        worker: &Arc<dyn Worker>,
    ) -> impl for<'a> FnMut(&PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<()> + Send
    {
        let worker = Arc::clone(worker);
        move |capacity| capacity.select(Arc::clone(&worker), ())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::{pending, ready, Ready},
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        },
        thread,
    };

    use futures::poll;
    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{health::circuit_breaker::CircuitBreakerConfig, identity::basic::WorkerType};
    use tokio::{
        runtime::Handle,
        sync::{mpsc, oneshot},
        task::JoinHandle,
    };

    use super::{test_support::select_worker, *};
    use crate::{
        observability::metrics::test_support::assert_metric,
        routers::grpc::backend_client::BasicWorkerBuilder,
    };

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill)
                .build(),
        )
    }

    /// A task that attempts `worker` through `admission` with the plain
    /// select-this-worker attempt; the test joins it, and aborts it first where
    /// it must never be admitted.
    #[expect(
        clippy::disallowed_methods,
        reason = "the test joins (and where it must, aborts) the task before it ends"
    )]
    fn plain_attempt(
        admission: &PrefillAdmission,
        worker: &Arc<dyn Worker>,
    ) -> JoinHandle<Result<AdmittedPrefill<()>, PrefillAdmissionRejection>> {
        let admission = admission.clone();
        let worker = Arc::clone(worker);
        tokio::spawn(async move { admission.admit(None, select_worker(&worker)).await })
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
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        let result = admission.admit(None, select_worker(&worker)).await;
        assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
        drop(first);
    }

    #[tokio::test]
    async fn full_queue_rejects_without_bypassing_waiter() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let worker = worker("http://prefill-full-queue");
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        let waiter = plain_attempt(&admission, &worker);
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
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        let waiter = plain_attempt(&admission, &worker);
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
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

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
        let initial = admission.admit(None, select_worker(&worker)).await.unwrap();
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
                let admitted = admission.admit(None, select_worker(&worker)).await.unwrap();
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
                let admitted = admission.admit(None, select_worker(&worker)).await.unwrap();
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
        let occupied = admission.admit(None, select_worker(&full)).await.unwrap();
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
        let initial = admission.admit(None, select_worker(&worker)).await.unwrap();

        let cancelled = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 1).await;

        let middle = plain_attempt(&admission, &worker);
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
            .admit(None, select_worker(&first_worker))
            .await
            .unwrap();
        let second = admission
            .admit(None, select_worker(&second_worker))
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

        let second = plain_attempt(&admission, &worker);
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
        let occupied = admission.admit(None, select_worker(&worker)).await.unwrap();

        let queued = plain_attempt(&admission, &worker);
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

    #[tokio::test]
    async fn a_decided_attempt_is_decided_with_no_lock_held() {
        let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
        let worker = worker("http://prefill-decide-unlocked");
        let runtime = Handle::current();
        let (result_tx, result_rx) = oneshot::channel();

        // A separate thread lets the timeout fire even if the decision blocks on the lock.
        let attempt = thread::spawn(move || {
            let result = runtime.block_on(admission.admit_decided(
                None,
                no_probe,
                || ready(Some(admission.queued_requests())),
                |queued, capacity| capacity.select(Arc::clone(&worker), queued),
            ));
            let _ = result_tx.send(result);
        });

        let admitted = tokio::time::timeout(Duration::from_secs(2), result_rx)
            .await
            .expect("decision must finish before the timeout")
            .expect("admission thread returns a result")
            .expect("attempt is admitted");
        assert_eq!(admitted.selected, 0);
        attempt.join().expect("admission thread finishes");
    }

    #[tokio::test]
    async fn a_decided_attempt_commits_under_the_lock_and_takes_one_slot() {
        struct Decision(Arc<dyn Worker>);

        let recorder = PrometheusBuilder::new().build_recorder();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
        let worker = worker("http://prefill-decide-locked-commit");

        let admitted = admission
            .admit_decided(
                Some("request"),
                no_probe,
                || ready(Some(Decision(Arc::clone(&worker)))),
                |Decision(decided), capacity| {
                    assert!(admission.inner.state.try_lock().is_none());
                    assert_eq!(decided.load(), 0);
                    capacity.select(decided, ())
                },
            )
            .await
            .expect("attempt is admitted");

        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_inflight("request"), 1);
        assert_metric(
            &recorder.handle().render(),
            "smg_pd_prefill_admission_inflight",
            &[r#"worker="http://prefill-decide-locked-commit""#],
            "1",
        );
        drop(admitted);
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_inflight("request"), 0);
        assert_metric(
            &recorder.handle().render(),
            "smg_pd_prefill_admission_inflight",
            &[r#"worker="http://prefill-decide-locked-commit""#],
            "0",
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_request_behind_a_waiter_decides_nothing_until_it_is_the_head() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
        let worker = worker("http://prefill-decide-at-head");
        let occupied = admission
            .admit(None, select_worker(&worker))
            .await
            .expect("slot is occupied");
        let first = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 1).await;

        let decisions = AtomicUsize::new(0);
        let second = admission.admit_decided(
            None,
            no_probe,
            || ready(Some(decisions.fetch_add(1, Ordering::Relaxed) + 1)),
            |decision, capacity| capacity.select(Arc::clone(&worker), decision),
        );
        tokio::pin!(second);
        assert!(poll!(second.as_mut()).is_pending());
        wait_for_queued(&admission, 2).await;
        assert_eq!(decisions.load(Ordering::Relaxed), 0);

        // Even a notification to the second waiter cannot let it decide yet.
        admission
            .inner
            .state
            .lock()
            .waiters
            .back()
            .expect("second waiter is queued")
            .notify
            .notify_one();
        assert!(poll!(second.as_mut()).is_pending());
        assert_eq!(decisions.load(Ordering::Relaxed), 0);

        drop(occupied);
        drop(
            first
                .await
                .expect("first waiter joins")
                .expect("first is admitted"),
        );
        let admitted = second.await.expect("second is admitted");
        assert_eq!(admitted.selected, 1);
        assert_eq!(decisions.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_waiter_that_queues_during_a_decision_keeps_its_place() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
        let busy = worker("http://prefill-decide-busy");
        let free = worker("http://prefill-decide-free");
        let occupied = admission
            .admit(None, select_worker(&busy))
            .await
            .expect("slot is occupied");

        let finish_decision = Notify::new();
        let commits = AtomicUsize::new(0);
        let decided = admission.admit_decided(
            None,
            no_probe,
            || async {
                finish_decision.notified().await;
                Some(Arc::clone(&free))
            },
            |worker, capacity| {
                commits.fetch_add(1, Ordering::Relaxed);
                capacity.select(worker, ())
            },
        );
        tokio::pin!(decided);
        assert!(poll!(decided.as_mut()).is_pending());
        tokio::time::sleep(Duration::from_secs(6)).await;
        assert!(poll!(decided.as_mut()).is_pending());
        assert_eq!(admission.queued_requests(), 0);
        let older = plain_attempt(&admission, &busy);
        wait_for_queued(&admission, 1).await;

        finish_decision.notify_one();
        assert!(poll!(decided.as_mut()).is_pending());
        assert_eq!(free.load(), 0);
        assert_eq!(commits.load(Ordering::Relaxed), 1);
        wait_for_queued(&admission, 2).await;

        drop(occupied);
        let _older = older
            .await
            .expect("older waiter joins")
            .expect("older is admitted");
        finish_decision.notify_one();
        let _decided = decided.await.expect("decided request is admitted next");
        assert_eq!(free.load(), 1);
        assert_eq!(commits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn a_queued_head_decides_again_at_each_wake() {
        let admission = PrefillAdmission::new(1, 3, Duration::from_secs(5));
        let other = worker("http://prefill-decide-next");
        let worker = worker("http://prefill-decide-again");
        let occupied = admission
            .admit(None, select_worker(&worker))
            .await
            .expect("slot is occupied");
        let first = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 1).await;

        let decisions = AtomicUsize::new(0);
        let second = admission.admit_decided(
            None,
            no_probe,
            || ready(Some(decisions.fetch_add(1, Ordering::Relaxed) + 1)),
            |decision, capacity| {
                assert!(admission.inner.state.try_lock().is_none());
                if decision < 3 {
                    PrefillAdmissionAttempt::AtCapacity
                } else {
                    capacity.select(Arc::clone(&worker), decision)
                }
            },
        );
        tokio::pin!(second);
        assert!(poll!(second.as_mut()).is_pending());
        let third = plain_attempt(&admission, &other);
        wait_for_queued(&admission, 3).await;

        drop(occupied);
        let first = first
            .await
            .expect("first waiter joins")
            .expect("first is admitted");
        assert!(poll!(second.as_mut()).is_pending());
        assert_eq!(decisions.load(Ordering::Relaxed), 1);

        drop(first);
        assert!(poll!(second.as_mut()).is_pending());
        assert_eq!(decisions.load(Ordering::Relaxed), 2);
        tokio::time::advance(Duration::from_millis(999)).await;
        assert!(poll!(second.as_mut()).is_pending());
        assert_eq!(decisions.load(Ordering::Relaxed), 2);
        tokio::time::advance(Duration::from_millis(2)).await;

        let admitted = tokio::time::timeout(Duration::from_millis(100), second)
            .await
            .expect("the head rechecks after one second")
            .expect("second is admitted at the recheck");
        assert_eq!(admitted.selected, 3);
        assert_eq!(decisions.load(Ordering::Relaxed), 3);
        assert_eq!(admission.queued_requests(), 1);
        let _third = tokio::time::timeout(Duration::from_millis(500), third)
            .await
            .expect("third waiter is woken before the periodic recheck")
            .expect("third waiter joins")
            .expect("third is admitted");
        assert_eq!(worker.load(), 1);
        drop(admitted);
    }

    #[tokio::test(start_paused = true)]
    async fn a_decision_that_names_no_candidate_is_unavailable_and_wakes_the_next_head() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
        let worker = worker("http://prefill-decide-unavailable");
        let occupied = admission
            .admit(None, select_worker(&worker))
            .await
            .expect("slot is occupied");
        let mut has_candidate = true;
        let commits = AtomicUsize::new(0);
        let head = admission.admit_decided(
            None,
            no_probe,
            || ready(std::mem::take(&mut has_candidate).then_some(())),
            |(), capacity| {
                commits.fetch_add(1, Ordering::Relaxed);
                capacity.select(Arc::clone(&worker), ())
            },
        );
        tokio::pin!(head);
        assert!(poll!(head.as_mut()).is_pending());
        let next = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 2).await;

        drop(occupied);
        assert!(matches!(
            head.await,
            Err(PrefillAcquireError::Rejected(
                PrefillAdmissionRejection::Unavailable
            ))
        ));
        assert_eq!(commits.load(Ordering::Relaxed), 1);
        assert_metric(
            &recorder.handle().render(),
            "smg_pd_prefill_admission_rejections_total",
            &[r#"reason="unavailable""#],
            "1",
        );
        let _next = tokio::time::timeout(Duration::from_millis(500), next)
            .await
            .expect("next waiter is woken before the periodic recheck")
            .expect("next waiter joins")
            .expect("next is admitted");
        assert_eq!(admission.queued_requests(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_decided_attempt_meets_a_full_queue_and_a_queue_timeout() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
        let worker = worker("http://prefill-decide-timeout");
        let _occupied = admission
            .admit(None, select_worker(&worker))
            .await
            .expect("slot is occupied");
        let decisions = AtomicUsize::new(0);
        let head = admission.admit_decided(
            None,
            no_probe,
            || async {
                if decisions.fetch_add(1, Ordering::Relaxed) == 0 {
                    Some(())
                } else {
                    pending().await
                }
            },
            |(), capacity| capacity.select(Arc::clone(&worker), ()),
        );
        tokio::pin!(head);
        assert!(poll!(head.as_mut()).is_pending());
        wait_for_queued(&admission, 1).await;
        assert!(recorder
            .handle()
            .render()
            .lines()
            .any(|line| line == "smg_pd_prefill_admission_queued 1"));

        let refused = admission
            .admit_decided(
                None,
                no_probe,
                || {
                    assert_eq!(
                        admission.queued_requests(),
                        0,
                        "a full queue refuses before deciding"
                    );
                    ready(Some(()))
                },
                |(), capacity| capacity.select(Arc::clone(&worker), ()),
            )
            .await;
        assert!(matches!(
            refused,
            Err(PrefillAcquireError::Rejected(
                PrefillAdmissionRejection::QueueFull
            ))
        ));

        tokio::time::advance(Duration::from_secs(3)).await;
        admission.notify_capacity_changed();
        assert!(poll!(head.as_mut()).is_pending());
        assert_eq!(decisions.load(Ordering::Relaxed), 2);
        tokio::time::advance(Duration::from_secs(2)).await;
        let result = tokio::time::timeout(Duration::from_millis(100), head)
            .await
            .expect("queued decision must finish by the queue deadline");
        assert!(matches!(
            result,
            Err(PrefillAcquireError::Rejected(
                PrefillAdmissionRejection::QueueTimeout
            ))
        ));
        assert_eq!(admission.queued_requests(), 0);
        let rendered = recorder.handle().render();
        for expected in [
            "smg_pd_prefill_admission_queued 0",
            r#"smg_pd_prefill_admission_rejections_total{reason="queue_full"} 1"#,
            r#"smg_pd_prefill_admission_rejections_total{reason="queue_timeout"} 1"#,
        ] {
            assert!(rendered.lines().any(|line| line == expected), "{rendered}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_cancelled_head_in_its_decision_leaves_the_queue() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
        let worker = worker("http://prefill-decide-cancelled");
        let occupied = admission
            .admit(None, select_worker(&worker))
            .await
            .expect("slot is occupied");
        let deciding = Arc::new(Notify::new());

        #[expect(
            clippy::disallowed_methods,
            reason = "the waiter task is aborted and joined"
        )]
        let cancelled = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let deciding = Arc::clone(&deciding);
            async move {
                let mut first = true;
                admission
                    .admit_decided(
                        None,
                        no_probe,
                        || {
                            let first = std::mem::take(&mut first);
                            let deciding = &deciding;
                            async move {
                                if first {
                                    Some(())
                                } else {
                                    deciding.notify_one();
                                    pending().await
                                }
                            }
                        },
                        |(), capacity| capacity.select(Arc::clone(&worker), ()),
                    )
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;
        let next = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 2).await;

        drop(occupied);
        tokio::time::timeout(Duration::from_millis(500), deciding.notified())
            .await
            .expect("head is waiting on its decision");
        assert_eq!(admission.queued_requests(), 2);
        assert_eq!(worker.load(), 0);
        cancelled.abort();
        assert!(matches!(cancelled.await, Err(error) if error.is_cancelled()));

        let _next = tokio::time::timeout(Duration::from_millis(500), next)
            .await
            .expect("next waiter is woken before the periodic recheck")
            .expect("next waiter joins")
            .expect("next is admitted");
        assert_eq!(admission.queued_requests(), 0);
    }

    #[derive(Debug, PartialEq, Eq)]
    enum CandidateError {
        AtCapacity(&'static str),
        Unavailable,
        /// Decides again twice, then ends the attempt.
        Again,
    }

    impl PrefillCandidateError for CandidateError {
        fn is_at_capacity(&self) -> bool {
            matches!(self, Self::AtCapacity(_))
        }

        fn decides_again(&self, again: usize) -> bool {
            matches!(self, Self::Again) && again < 2
        }
    }

    fn no_probe() -> Ready<Option<CandidateError>> {
        ready(None)
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "only the head removes itself from the front")]
    fn only_the_head_can_remove_itself_from_the_front() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
        let (_head, mut later) = {
            let mut state = admission.inner.state.lock();
            (
                admission.enqueue_locked(&mut state).expect("the head"),
                admission
                    .enqueue_locked(&mut state)
                    .expect("the next waiter"),
            )
        };
        let mut state = admission.inner.state.lock();
        let _ = later.remove_head(&mut state);
    }

    #[tokio::test]
    async fn acquire_decided_without_admission_decides_and_commits_once() {
        let worker = worker("http://prefill-acquire-decided-unbounded");
        let occupied = PrefillAdmission::new(1, 0, Duration::ZERO)
            .admit(None, select_worker(&worker))
            .await
            .expect("slot is occupied");
        let (mut decisions, mut commits) = (0, 0);
        let result = acquire_decided(
            None,
            Some("request"),
            |worker| worker,
            no_probe,
            || {
                decisions += 1;
                ready(Ok::<_, CandidateError>(Arc::clone(&worker)))
            },
            |decision, capacity| {
                commits += 1;
                assert!(capacity.is_none());
                Ok(decision)
            },
        )
        .await;

        let Ok((candidate, guard)) = result else {
            panic!("unbounded attempt holds load even on an occupied worker");
        };
        assert!(Arc::ptr_eq(&candidate, &worker));
        assert!(matches!(guard, PrefillLoadGuard::Unbounded { .. }));
        assert_eq!((decisions, commits), (1, 1));
        assert_eq!(worker.load(), 2);
        assert_eq!(worker.routing_key_load(), 1);
        drop(guard);
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_load(), 0);
        drop(occupied);
    }

    #[tokio::test]
    async fn acquire_decided_without_admission_returns_the_candidate_error_from_either_step() {
        for decision_fails in [true, false] {
            let result = acquire_decided(
                None,
                None,
                |worker: &Arc<dyn Worker>| worker,
                no_probe,
                || {
                    ready(if decision_fails {
                        Err(CandidateError::Unavailable)
                    } else {
                        Ok(())
                    })
                },
                |(), capacity| {
                    assert!(capacity.is_none());
                    assert!(!decision_fails);
                    Err(CandidateError::AtCapacity("commit"))
                },
            )
            .await;
            let expected = if decision_fails {
                CandidateError::Unavailable
            } else {
                CandidateError::AtCapacity("commit")
            };
            assert!(
                matches!(result, Err(PrefillAcquireError::Candidate(error)) if error == expected)
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn acquire_decided_keeps_the_last_verdict_of_an_unavailable_attempt() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
        let worker = worker("http://prefill-acquire-decided-verdict");
        let mut decisions = [
            Ok(Arc::clone(&worker)),
            Err(CandidateError::AtCapacity("decision")),
            Err(CandidateError::Unavailable),
        ]
        .into_iter();
        let commits = AtomicUsize::new(0);
        let attempt = acquire_decided(
            Some(&admission),
            None,
            |worker: &Arc<dyn Worker>| worker,
            no_probe,
            || ready(decisions.next().expect("one decision per wake")),
            |_, capacity| {
                assert!(capacity.is_some());
                commits.fetch_add(1, Ordering::Relaxed);
                Err(CandidateError::AtCapacity("commit"))
            },
        );
        tokio::pin!(attempt);
        assert!(poll!(attempt.as_mut()).is_pending());
        wait_for_queued(&admission, 1).await;
        admission.notify_capacity_changed();
        assert!(poll!(attempt.as_mut()).is_pending());
        assert_eq!(admission.queued_requests(), 1);
        admission.notify_capacity_changed();
        assert!(matches!(
            attempt.await,
            Err(PrefillAcquireError::Candidate(CandidateError::Unavailable))
        ));
        assert_eq!(admission.queued_requests(), 0);
        assert_eq!(worker.load(), 0);
        assert_eq!(commits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_head_that_decides_again_keeps_its_place_and_its_deadline() {
        for released in [true, false] {
            let recorder = PrometheusBuilder::new().build_recorder();
            let _recorder = metrics::set_default_local_recorder(&recorder);
            let admission = PrefillAdmission::new(1, 2, Duration::from_secs(5));
            let worker = worker("http://prefill-decides-again");
            let occupied = admission
                .admit(None, select_worker(&worker))
                .await
                .expect("slot is occupied");
            let queued_at = TokioInstant::now();
            let commits = AtomicUsize::new(0);
            let head = acquire_decided(
                Some(&admission),
                None,
                |worker: &Arc<dyn Worker>| worker,
                no_probe,
                || ready(Ok(())),
                |(), _| match commits.fetch_add(1, Ordering::Relaxed) {
                    0 => Err(CandidateError::AtCapacity("joins the queue")),
                    1 | 2 => Err(CandidateError::Again),
                    _ => Ok(Arc::clone(&worker)),
                },
            );
            tokio::pin!(head);
            assert!(poll!(head.as_mut()).is_pending());
            let behind = admission.admit(None, select_worker(&worker));
            tokio::pin!(behind);
            assert!(poll!(behind.as_mut()).is_pending());

            // One wake, four seconds into the wait: both refusals, then no room.
            tokio::time::advance(Duration::from_secs(4)).await;
            assert!(poll!(head.as_mut()).is_pending());
            assert_eq!(commits.load(Ordering::Relaxed), 4);
            assert_eq!(admission.queued_requests(), 2);

            if released {
                drop(occupied);
                assert!(poll!(behind.as_mut()).is_pending());
                assert!(head.await.is_ok());
            } else {
                assert!(matches!(
                    head.await,
                    Err(PrefillAcquireError::Rejected(
                        PrefillAdmissionRejection::QueueTimeout
                    ))
                ));
                assert_eq!(queued_at.elapsed(), Duration::from_secs(5));
            }
            assert!(!recorder
                .handle()
                .render()
                .contains(r#"reason="unavailable""#));
        }
    }

    #[tokio::test]
    async fn an_attempt_decides_again_as_often_as_its_error_allows() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
        for admission in [Some(&admission), None] {
            let decisions = AtomicUsize::new(0);
            let result = acquire_decided(
                admission,
                None,
                |worker: &Arc<dyn Worker>| worker,
                no_probe,
                || {
                    decisions.fetch_add(1, Ordering::Relaxed);
                    ready(Ok(()))
                },
                |(), _| Err(CandidateError::Again),
            )
            .await;
            assert!(matches!(
                result,
                Err(PrefillAcquireError::Candidate(CandidateError::Again))
            ));
            assert_eq!(decisions.load(Ordering::Relaxed), 3);
        }
        assert_metric(
            &recorder.handle().render(),
            "smg_pd_prefill_admission_rejections_total",
            &[r#"reason="unavailable""#],
            "1",
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_probes_verdict_leaves_before_the_queue() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
        let worker = worker("http://prefill-probe");
        let _occupied = admission
            .admit(None, select_worker(&worker))
            .await
            .expect("slot is occupied");
        let head = admission.admit(None, select_worker(&worker));
        tokio::pin!(head);
        assert!(poll!(head.as_mut()).is_pending());

        let mut probes = 0;
        let result = acquire_decided(
            Some(&admission),
            None,
            |worker: &Arc<dyn Worker>| worker,
            || {
                probes += 1;
                assert!(admission.inner.state.try_lock().is_some());
                ready(Some(CandidateError::Unavailable))
            },
            || ready(Ok(())),
            |(), _| Ok(Arc::clone(&worker)),
        )
        .await;
        assert!(matches!(
            result,
            Err(PrefillAcquireError::Candidate(CandidateError::Unavailable))
        ));
        assert_eq!(probes, 1);
        assert_metric(
            &recorder.handle().render(),
            "smg_pd_prefill_admission_rejections_total",
            &[r#"reason="unavailable""#],
            "1",
        );
        assert_eq!(admission.queued_requests(), 1);
        assert_eq!(worker.load(), 1);
    }

    #[tokio::test]
    async fn acquire_decided_returns_commit_errors_and_holds_the_selected_slot() {
        let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
        let worker = worker("http://prefill-acquire-decided-bounded");
        let result = acquire_decided(
            Some(&admission),
            Some("request"),
            |worker| worker,
            no_probe,
            || ready(Ok(Arc::clone(&worker))),
            |_, _| Err(CandidateError::Unavailable),
        )
        .await;
        assert!(matches!(
            result,
            Err(PrefillAcquireError::Candidate(CandidateError::Unavailable))
        ));

        let result = acquire_decided(
            Some(&admission),
            Some("request"),
            |worker| worker,
            no_probe,
            || ready(Ok::<_, CandidateError>(Arc::clone(&worker))),
            |decision, capacity| {
                assert!(capacity
                    .expect("commit gets capacity")
                    .has_capacity(&decision));
                Ok(decision)
            },
        )
        .await;
        let Ok((candidate, guard)) = result else {
            panic!("admission holds the selected slot");
        };
        assert!(Arc::ptr_eq(&candidate, &worker));
        assert!(matches!(guard, PrefillLoadGuard::Admission { .. }));
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_load(), 1);
        drop(guard);
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[tokio::test]
    async fn an_admitted_prefill_reports_completion_when_its_reservation_drops() {
        use smg_worker::identity::basic::{RequestCompletionSink, Worker, WorkerType};

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;
        #[derive(Debug, Default)]
        struct CompletionSpy(std::sync::Mutex<Vec<String>>);
        impl RequestCompletionSink for CompletionSpy {
            fn request_completed(&self, worker: &dyn Worker) {
                self.0.lock().unwrap().push(worker.url().to_string());
            }
        }

        let spy = Arc::new(CompletionSpy::default());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill-admitted")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        worker.set_completion_sink(Some(spy.clone() as Arc<dyn RequestCompletionSink>));
        let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let admitted = admission
            .admit(None, select_worker(&worker))
            .await
            .expect("admitted");
        assert_eq!(worker.load(), 1);
        assert!(spy.0.lock().unwrap().is_empty());
        drop(admitted);
        assert_eq!(worker.load(), 0);
        assert_eq!(
            spy.0.lock().unwrap().as_slice(),
            ["grpc://prefill-admitted"]
        );
    }
}
