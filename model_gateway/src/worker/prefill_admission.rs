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
    // role's own reason (a shed, a missing role) rather than a generic one.
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

/// How many times a newcomer with nobody waiting decides and commits
/// before it queues: the proposal and one re-decision after a commit miss.
const NEWCOMER_ATTEMPTS: usize = 2;

/// The per-worker capacity window as a decision reads it outside the
/// admission lock: an atomic read of the worker's load against the limit.
/// The commit under the lock re-validates it ([`PrefillSelectionContext`]).
pub struct CapacityWindow {
    max_inflight_requests_per_worker: usize,
}

impl CapacityWindow {
    /// A window nothing bounds: admission is off.
    pub fn unbounded() -> Self {
        Self {
            max_inflight_requests_per_worker: usize::MAX,
        }
    }

    pub fn has_capacity(&self, worker: &Arc<dyn Worker>) -> bool {
        worker.load() < self.max_inflight_requests_per_worker
    }
}

/// What a decision made outside the admission lock proposes.
pub enum PrefillProposal<P> {
    /// A proposal the commit under the lock re-validates.
    Proposed(P),
    /// No candidate with capacity: wait for a slot.
    AtCapacity,
    /// No healthy, model-matching worker.
    Unavailable,
}

/// [`acquire_prefill`] with the decision made outside the admission lock:
/// `decide` runs with no lock held, only when this request may select (a
/// newcomer with nobody waiting, or the queue head), and `commit` runs under
/// the lock to re-validate the proposal and reserve. A decision books
/// nothing; only the commit reserves, so a proposal the window refuses costs
/// one more decision and no state. `available` answers whether the request's
/// model has a healthy prefill worker at all: it runs once, for a newcomer
/// behind a non-empty queue, in place of a decision (no policy state moves
/// for a request that may not select), and its error is that request's
/// verdict.
pub async fn acquire_prefill_decided<C, E, P, S, A, D, Pf>(
    admission: Option<&PrefillAdmission>,
    routing_key: Option<&str>,
    prefill_of: Pf,
    mut available: A,
    mut decide: D,
    mut commit: S,
) -> Result<(C, PrefillLoadGuard), PrefillAcquireError<E>>
where
    A: FnMut() -> Result<(), E> + Send,
    D: FnMut(&CapacityWindow) -> Result<P, E> + Send,
    S: for<'a> FnMut(Option<&PrefillSelectionContext<'a>>, P) -> Result<C, E> + Send,
    Pf: Fn(&C) -> &Arc<dyn Worker> + Send + Sync,
    E: PrefillCandidateError + Send,
    P: Send,
{
    let Some(admission) = admission else {
        let proposal =
            decide(&CapacityWindow::unbounded()).map_err(PrefillAcquireError::Candidate)?;
        let candidate = commit(None, proposal).map_err(PrefillAcquireError::Candidate)?;
        let guard = PrefillLoadGuard::Unbounded {
            _guard: WorkerLoadGuard::with_key(Arc::clone(prefill_of(&candidate)), routing_key),
        };
        return Ok((candidate, guard));
    };

    // Both closures may record the verdict, and both must stay `Send`.
    let verdict: Mutex<Option<E>> = Mutex::new(None);
    let record = |error: E| *verdict.lock() = Some(error);
    let admitted = admission
        .admit_decided(
            routing_key,
            || match available() {
                Ok(()) => true,
                Err(error) => {
                    record(error);
                    false
                }
            },
            |window| match decide(window) {
                Ok(proposal) => PrefillProposal::Proposed(proposal),
                Err(error) if error.is_at_capacity() => PrefillProposal::AtCapacity,
                Err(error) => {
                    record(error);
                    PrefillProposal::Unavailable
                }
            },
            |capacity, proposal| match commit(Some(capacity), proposal) {
                Ok(candidate) => capacity.select(Arc::clone(prefill_of(&candidate)), candidate),
                Err(error) if error.is_at_capacity() => PrefillAdmissionAttempt::AtCapacity,
                Err(error) => {
                    record(error);
                    PrefillAdmissionAttempt::Unavailable
                }
            },
        )
        .await;

    let verdict = verdict.into_inner();
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
/// this Router process. [`Self::admit_decided`] keeps the same order with the
/// selection made outside the lock and only the commit under it.
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
    /// Counts releases, so a request that decided outside the lock can
    /// tell at its queue entry whether a slot freed unseen meanwhile.
    releases: u64,
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
                    releases: 0,
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

    /// [`Self::admit`] with the decision outside the lock.
    ///
    /// The invariant stays "only the queue head, or a newcomer with nobody
    /// waiting, selects": that request reads the capacity window with no lock
    /// held, decides, then takes the lock to re-validate and reserve. A
    /// newcomer whose proposal the commit refuses decides once more (its
    /// closures know the miss) and then joins the queue; a newcomer that
    /// finds others waiting asks `available` once (the caller's cheap check,
    /// as [`Self::admit`] hides capacity from a request that may not select)
    /// and joins it; the request an entry makes head decides at once when a
    /// slot may have freed unseen (a release since its read, or no read at
    /// all); the head whose proposal is refused waits for the next wake. A
    /// decision books nothing, so the serialized section is the commit alone.
    pub async fn admit_decided<P, T, A, D, C>(
        &self,
        routing_key: Option<&str>,
        mut available: A,
        mut decide: D,
        mut commit: C,
    ) -> Result<AdmittedPrefill<T>, PrefillAdmissionRejection>
    where
        A: FnMut() -> bool + Send,
        D: FnMut(&CapacityWindow) -> PrefillProposal<P> + Send,
        C: for<'a> FnMut(&PrefillSelectionContext<'a>, P) -> PrefillAdmissionAttempt<T> + Send,
        P: Send,
    {
        let window = CapacityWindow {
            max_inflight_requests_per_worker: self.inner.max_inflight_requests_per_worker,
        };
        // A newcomer decides at once when nobody waits: whether anyone does
        // is read under the lock, the decision is made outside it, and the
        // commit under it may still take a slot, since this request is older
        // than anyone who queued meanwhile. The release count read with it
        // tells the queue entry whether a slot freed unseen in between.
        let (newcomer_allowed, releases_seen) = {
            let state = self.inner.state.lock();
            (state.waiters.is_empty(), state.releases)
        };
        if newcomer_allowed {
            // A newcomer's proposal can lose its slot to another newcomer
            // that committed first, the one commit miss of this path; it
            // decides once more before it queues, so two requests that read
            // the same window do not turn one free slot into a queue.
            for _ in 0..NEWCOMER_ATTEMPTS {
                match decide(&window) {
                    PrefillProposal::Unavailable => {
                        Metrics::record_pd_prefill_admission_rejection("unavailable");
                        return Err(PrefillAdmissionRejection::Unavailable);
                    }
                    PrefillProposal::AtCapacity => break,
                    PrefillProposal::Proposed(proposal) => {
                        let state = self.inner.state.lock();
                        match self.commit_locked(&state, routing_key, &mut commit, proposal) {
                            LockedAttempt::Selected(admitted) => return Ok(admitted),
                            LockedAttempt::Unavailable => {
                                Metrics::record_pd_prefill_admission_rejection("unavailable");
                                return Err(PrefillAdmissionRejection::Unavailable);
                            }
                            LockedAttempt::AtCapacity => {}
                        }
                    }
                }
            }
        } else if !available() {
            // Older requests wait, so this one may not select; as under
            // [`Self::admit`], a model with no healthy worker is refused
            // before it takes a place in the queue, by the caller's cheap
            // check rather than a decision whose proposal would go unused.
            Metrics::record_pd_prefill_admission_rejection("unavailable");
            return Err(PrefillAdmissionRejection::Unavailable);
        }

        let (waiter, queued_at, decide_now) = {
            let mut state = self.inner.state.lock();
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
            // The entry's new head decides at once when a slot may have
            // freed unseen: a release since its read found no waiter to
            // wake, or it never read the window (it arrived behind a queue
            // that drained meanwhile). Otherwise the next release finds it
            // at the front, and it waits.
            let decide_now =
                state.waiters.len() == 1 && (!newcomer_allowed || state.releases != releases_seen);
            (waiter, queued_at, decide_now)
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

        // Under [`Self::admit`] the attempt and the entry share one lock
        // hold; here the entry's new head may owe itself a first decision.
        let mut first_pass = decide_now;
        loop {
            let waiter = Arc::clone(&ticket.waiter);
            let is_head = {
                let state = self.inner.state.lock();
                ticket.is_head(&state)
            };
            if !(first_pass && is_head) {
                let recheck = tokio::time::sleep(HEAD_RECHECK_INTERVAL);
                tokio::pin!(recheck);
                tokio::select! {
                    biased;
                    () = &mut timeout => {
                        Metrics::record_pd_prefill_admission_rejection("queue_timeout");
                        return Err(PrefillAdmissionRejection::QueueTimeout);
                    }
                    () = waiter.notify.notified() => {}
                    () = &mut recheck, if is_head => {}
                }

                if tokio::time::Instant::now() >= deadline {
                    Metrics::record_pd_prefill_admission_rejection("queue_timeout");
                    return Err(PrefillAdmissionRejection::QueueTimeout);
                }
            }
            first_pass = false;

            let is_head = {
                let state = self.inner.state.lock();
                ticket.is_head(&state)
            };
            if !is_head {
                continue;
            }
            // The head decides with no lock held; nobody else may select
            // meanwhile, and only the head itself leaves the head position.
            let proposal = match decide(&window) {
                PrefillProposal::Proposed(proposal) => proposal,
                PrefillProposal::AtCapacity => continue,
                PrefillProposal::Unavailable => {
                    let next = {
                        let mut state = self.inner.state.lock();
                        ticket.remove_head(&mut state)
                    };
                    if let Some(waiter) = next {
                        waiter.notify.notify_one();
                    }
                    Metrics::record_pd_prefill_admission_rejection("unavailable");
                    return Err(PrefillAdmissionRejection::Unavailable);
                }
            };
            let decision = {
                let mut state = self.inner.state.lock();
                match self.commit_locked(&state, routing_key, &mut commit, proposal) {
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

    /// The commit of a proposal under the lock: re-validate through `commit`
    /// (which answers through [`PrefillSelectionContext::select`]) and
    /// reserve the slot, as [`Self::try_admit_locked`] does for a selection
    /// made under the lock.
    fn commit_locked<P, T, C>(
        &self,
        state: &PrefillAdmissionState,
        routing_key: Option<&str>,
        commit: &mut C,
        proposal: P,
    ) -> LockedAttempt<T>
    where
        C: for<'a> FnMut(&PrefillSelectionContext<'a>, P) -> PrefillAdmissionAttempt<T>,
    {
        let mut proposal = Some(proposal);
        self.try_admit_locked(
            state,
            routing_key,
            true,
            &mut |capacity| match proposal.take() {
                Some(proposal) => commit(capacity, proposal),
                None => PrefillAdmissionAttempt::AtCapacity,
            },
        )
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
        let mut state = self.state.lock();
        state.releases = state.releases.wrapping_add(1);
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
    use tokio::{
        sync::{mpsc, oneshot},
        task::JoinHandle,
    };

    use super::*;
    use crate::worker::{
        test_support::select_worker, BasicWorkerBuilder, CircuitBreakerConfig, WorkerType,
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
    async fn an_admitted_prefill_reports_completion_when_its_reservation_drops() {
        use crate::worker::{BasicWorkerBuilder, RequestCompletionSink, Worker, WorkerType};
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

    /// A request through the decided path: a decision that proposes `worker`
    /// each time it runs (counted in `decisions`) and a commit that
    /// re-validates the window, as the engine's commit does.
    #[expect(
        clippy::disallowed_methods,
        reason = "the test joins the task before it ends"
    )]
    fn decided_attempt(
        admission: &PrefillAdmission,
        worker: &Arc<dyn Worker>,
        decisions: &Arc<AtomicUsize>,
    ) -> JoinHandle<Result<AdmittedPrefill<()>, PrefillAdmissionRejection>> {
        let admission = admission.clone();
        let worker = Arc::clone(worker);
        let decisions = Arc::clone(decisions);
        tokio::spawn(async move {
            admission
                .admit_decided(
                    None,
                    || true,
                    |_window| {
                        decisions.fetch_add(1, Ordering::Relaxed);
                        PrefillProposal::Proposed(())
                    },
                    |capacity, ()| capacity.select(Arc::clone(&worker), ()),
                )
                .await
        })
    }

    #[tokio::test]
    async fn a_decided_newcomer_with_nobody_waiting_decides_once_and_reserves() {
        let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-newcomer");
        let decisions = AtomicUsize::new(0);
        let commits = AtomicUsize::new(0);

        let admitted = admission
            .admit_decided(
                None,
                || true,
                |window| {
                    decisions.fetch_add(1, Ordering::Relaxed);
                    assert!(window.has_capacity(&worker));
                    PrefillProposal::Proposed("proposed")
                },
                |capacity, proposal| {
                    commits.fetch_add(1, Ordering::Relaxed);
                    capacity.select(Arc::clone(&worker), proposal)
                },
            )
            .await
            .unwrap();

        assert_eq!(admitted.selected, "proposed");
        assert_eq!(decisions.load(Ordering::Relaxed), 1);
        assert_eq!(commits.load(Ordering::Relaxed), 1);
        assert_eq!(worker.load(), 1);
        drop(admitted.reservation);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn a_decided_proposal_the_window_refuses_at_the_commit_waits_as_the_head() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-miss");
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        // The decision proposes the full worker (a stale read would); the
        // commit's re-validation refuses it twice (the proposal and its one
        // re-decision), and the request waits as the head instead of taking
        // a second slot; nothing was released since its read, so the entry
        // owes no first pass.
        let decisions = Arc::new(AtomicUsize::new(0));
        let waiter = decided_attempt(&admission, &worker, &decisions);
        wait_for_queued(&admission, 1).await;
        assert_eq!(worker.load(), 1);
        assert_eq!(decisions.load(Ordering::Relaxed), 2);

        drop(first);
        let admitted = waiter.await.unwrap().unwrap();
        assert!(decisions.load(Ordering::Relaxed) >= 3);
        assert_eq!(worker.load(), 1);
        assert_eq!(admission.queued_requests(), 0);
        drop(admitted);
    }

    #[tokio::test]
    async fn only_the_decided_head_decides_while_others_wait() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-head-only");
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        let head_decisions = Arc::new(AtomicUsize::new(0));
        let head = decided_attempt(&admission, &worker, &head_decisions);
        wait_for_queued(&admission, 1).await;
        let second_decisions = Arc::new(AtomicUsize::new(0));
        let second = decided_attempt(&admission, &worker, &second_decisions);
        wait_for_queued(&admission, 2).await;

        // The second request found a waiter, so it checked availability and
        // never decided; the head decided twice on arrival (the proposal and
        // its re-decision) and decides again only when it is woken.
        assert_eq!(head_decisions.load(Ordering::Relaxed), 2);
        assert_eq!(second_decisions.load(Ordering::Relaxed), 0);
        assert_eq!(worker.load(), 1);

        drop(first);
        let admitted_head = head.await.unwrap().unwrap();
        assert_eq!(worker.load(), 1);
        drop(admitted_head);
        let admitted_second = second.await.unwrap().unwrap();
        assert!(second_decisions.load(Ordering::Relaxed) >= 1);
        assert_eq!(worker.load(), 1);
        drop(admitted_second);
    }

    #[tokio::test(start_paused = true)]
    async fn an_unavailable_decision_rejects_the_newcomer_without_queueing() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        // The newcomer's own decision answers: one decision, no queue entry,
        // and no wait on the recheck interval for a head to answer instead.
        let decisions = AtomicUsize::new(0);
        let started = tokio::time::Instant::now();
        let result = admission
            .admit_decided(
                None,
                || true,
                |_window| {
                    decisions.fetch_add(1, Ordering::Relaxed);
                    PrefillProposal::<()>::Unavailable
                },
                |_capacity, ()| PrefillAdmissionAttempt::<()>::Unavailable,
            )
            .await;
        assert!(matches!(
            result,
            Err(PrefillAdmissionRejection::Unavailable)
        ));
        assert_eq!(decisions.load(Ordering::Relaxed), 1);
        assert_eq!(tokio::time::Instant::now(), started);
        assert_eq!(admission.queued_requests(), 0);
    }

    #[derive(Debug, PartialEq)]
    struct Refused;

    impl PrefillCandidateError for Refused {
        fn is_at_capacity(&self) -> bool {
            false
        }
    }

    fn same(worker: &Arc<dyn Worker>) -> &Arc<dyn Worker> {
        worker
    }

    #[tokio::test]
    async fn acquire_decided_without_admission_commits_once_under_an_unbounded_guard() {
        let worker = worker("http://prefill-decided-unbounded");
        let Ok((selected, guard)) = acquire_prefill_decided::<_, Refused, _, _, _, _, _>(
            None,
            None,
            same,
            || panic!("the unbounded path asks no availability"),
            |window| {
                assert!(window.has_capacity(&worker));
                Ok(())
            },
            |context, ()| {
                assert!(context.is_none());
                Ok(Arc::clone(&worker))
            },
        )
        .await
        else {
            panic!("the unbounded path admits");
        };
        assert_eq!(selected.url(), worker.url());
        assert_eq!(worker.load(), 1);
        drop(guard);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn acquire_decided_reports_the_candidate_error_of_an_unavailable_decision() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let result = acquire_prefill_decided::<Arc<dyn Worker>, Refused, (), _, _, _, _>(
            Some(&admission),
            None,
            same,
            || Ok(()),
            |_window| Err(Refused),
            |_context, ()| panic!("a refused decision never commits"),
        )
        .await;
        assert!(matches!(
            result,
            Err(PrefillAcquireError::Candidate(Refused))
        ));
        assert_eq!(admission.queued_requests(), 0);
    }

    #[tokio::test]
    async fn acquire_decided_reports_the_candidate_error_of_an_unavailable_model_behind_a_queue() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-unavailable-behind-acquire");
        let occupied = admission.admit(None, select_worker(&worker)).await.unwrap();
        let queued = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 1).await;

        // Behind a queue the role's own reason answers, through the
        // availability check, with no decision made.
        let result = acquire_prefill_decided::<Arc<dyn Worker>, Refused, (), _, _, _, _>(
            Some(&admission),
            None,
            same,
            || Err(Refused),
            |_window| panic!("a request behind a queue never decides"),
            |_context, ()| panic!("a request behind a queue never commits"),
        )
        .await;
        assert!(matches!(
            result,
            Err(PrefillAcquireError::Candidate(Refused))
        ));
        assert_eq!(admission.queued_requests(), 1);

        queued.abort();
        assert!(matches!(
            queued.await,
            Err(error) if error.is_cancelled()
        ));
        drop(occupied);
    }

    async fn wait_for_count(counter: &AtomicUsize, expected: usize) {
        for _ in 0..10_000 {
            if counter.load(Ordering::Relaxed) >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "count did not reach {expected}; actual={}",
            counter.load(Ordering::Relaxed)
        );
    }

    #[tokio::test]
    async fn a_decided_newcomer_at_capacity_queues_after_one_decision() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-at-capacity");
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        // A decision that finds no candidate with capacity costs no commit
        // and no second decision: the newcomer queues at once, and with no
        // release since its read the entry owes no first pass.
        let decisions = Arc::new(AtomicUsize::new(0));
        let commits = Arc::new(AtomicUsize::new(0));
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins the task before it ends"
        )]
        let waiter = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let decisions = Arc::clone(&decisions);
            let commits = Arc::clone(&commits);
            async move {
                admission
                    .admit_decided(
                        None,
                        || true,
                        |window| {
                            decisions.fetch_add(1, Ordering::Relaxed);
                            if window.has_capacity(&worker) {
                                PrefillProposal::Proposed(())
                            } else {
                                PrefillProposal::AtCapacity
                            }
                        },
                        |capacity, ()| {
                            commits.fetch_add(1, Ordering::Relaxed);
                            capacity.select(Arc::clone(&worker), ())
                        },
                    )
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;
        assert_eq!(decisions.load(Ordering::Relaxed), 1);
        assert_eq!(commits.load(Ordering::Relaxed), 0);

        drop(first);
        let admitted = waiter.await.unwrap().unwrap();
        assert_eq!(decisions.load(Ordering::Relaxed), 2);
        assert_eq!(commits.load(Ordering::Relaxed), 1);
        assert_eq!(worker.load(), 1);
        drop(admitted);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn a_head_whose_proposal_is_refused_keeps_its_place_and_waits() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-head-refused");
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        // The head's decisions in order: at arrival two proposals the full
        // worker's commit refuses (the second is the newcomer's re-decision);
        // at the release's wake a decision that reads no capacity; at the
        // next wake a proposal the commit refuses (the slot went elsewhere);
        // at the next a proposal that commits. Through all of them the head
        // keeps its place and the waiter behind it never decides.
        let decisions = Arc::new(AtomicUsize::new(0));
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins the task before it ends"
        )]
        let head = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let decisions = Arc::clone(&decisions);
            async move {
                admission
                    .admit_decided(
                        None,
                        || true,
                        |_window| {
                            let decision = decisions.fetch_add(1, Ordering::Relaxed);
                            if decision == 2 {
                                PrefillProposal::AtCapacity
                            } else {
                                PrefillProposal::Proposed(decision)
                            }
                        },
                        |capacity, decision| {
                            if decision == 3 {
                                PrefillAdmissionAttempt::AtCapacity
                            } else {
                                capacity.select(Arc::clone(&worker), decision)
                            }
                        },
                    )
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;
        let second_decisions = Arc::new(AtomicUsize::new(0));
        let second = decided_attempt(&admission, &worker, &second_decisions);
        wait_for_queued(&admission, 2).await;
        assert_eq!(
            decisions.load(Ordering::Relaxed),
            2,
            "the arrival's two decisions"
        );

        drop(first);
        wait_for_count(&decisions, 3).await;
        assert_eq!(
            admission.queued_requests(),
            2,
            "no capacity read: the head waits"
        );
        assert_eq!(worker.load(), 0);

        admission.notify_capacity_changed();
        wait_for_count(&decisions, 4).await;
        assert_eq!(
            admission.queued_requests(),
            2,
            "proposal refused: the head waits"
        );
        assert_eq!(worker.load(), 0);
        assert_eq!(second_decisions.load(Ordering::Relaxed), 0);

        admission.notify_capacity_changed();
        let admitted = head.await.unwrap().unwrap();
        assert_eq!(admitted.selected, 4, "the fifth decision committed");
        assert_eq!(worker.load(), 1);
        assert_eq!(admission.queued_requests(), 1);
        drop(admitted);
        let admitted_second = second.await.unwrap().unwrap();
        assert_eq!(worker.load(), 1);
        drop(admitted_second);
    }

    #[tokio::test]
    async fn the_commit_re_validates_a_window_the_decision_read_outside_the_lock() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-revalidated");

        // The decision reads a free slot; before its commit a dispatch
        // outside this queue fills the worker (the race a decision outside
        // the lock allows). The commit refuses the proposal, the re-decision
        // reads no capacity, and the request queues; the slot's release
        // admits it.
        let decisions = Arc::new(AtomicUsize::new(0));
        let commits = Arc::new(AtomicUsize::new(0));
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins the task before it ends"
        )]
        let waiter = tokio::spawn({
            let admission = admission.clone();
            let worker = Arc::clone(&worker);
            let decisions = Arc::clone(&decisions);
            let commits = Arc::clone(&commits);
            async move {
                admission
                    .admit_decided(
                        None,
                        || true,
                        |window| {
                            let decision = decisions.fetch_add(1, Ordering::Relaxed);
                            if decision == 0 {
                                assert!(window.has_capacity(&worker));
                                worker.increment_load();
                                PrefillProposal::Proposed(())
                            } else if window.has_capacity(&worker) {
                                PrefillProposal::Proposed(())
                            } else {
                                PrefillProposal::AtCapacity
                            }
                        },
                        |capacity, ()| {
                            commits.fetch_add(1, Ordering::Relaxed);
                            capacity.select(Arc::clone(&worker), ())
                        },
                    )
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;
        assert_eq!(decisions.load(Ordering::Relaxed), 2);
        assert_eq!(commits.load(Ordering::Relaxed), 1, "the refused commit");
        assert_eq!(worker.load(), 1, "the outside dispatch alone");

        worker.decrement_load();
        admission.notify_capacity_changed();
        let admitted = waiter.await.unwrap().unwrap();
        assert_eq!(commits.load(Ordering::Relaxed), 2);
        assert_eq!(worker.load(), 1, "the reservation");
        drop(admitted);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn a_decided_newcomer_behind_a_waiter_is_refused_when_the_queue_is_full() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-queue-full");
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();
        let waiter = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 1).await;

        // A waiter is queued, so the newcomer checks availability once and
        // never decides; the queue has no room, so it takes no ticket either.
        let checks = AtomicUsize::new(0);
        let decisions = AtomicUsize::new(0);
        let result = admission
            .admit_decided(
                None,
                || {
                    checks.fetch_add(1, Ordering::Relaxed);
                    true
                },
                |_window| {
                    decisions.fetch_add(1, Ordering::Relaxed);
                    PrefillProposal::Proposed(())
                },
                |capacity, ()| capacity.select(Arc::clone(&worker), ()),
            )
            .await;
        assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
        assert_eq!(checks.load(Ordering::Relaxed), 1);
        assert_eq!(decisions.load(Ordering::Relaxed), 0);
        assert_eq!(admission.queued_requests(), 1);

        waiter.abort();
        assert!(matches!(
            waiter.await,
            Err(error) if error.is_cancelled()
        ));
        drop(first);
    }

    #[tokio::test]
    async fn a_decided_newcomer_with_no_queue_is_refused_after_its_two_decisions() {
        let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-no-queue");
        let first = admission.admit(None, select_worker(&worker)).await.unwrap();

        // Nobody waits, so the newcomer decides; the full worker refuses
        // both commits, and with no queue to join it is refused.
        let decisions = AtomicUsize::new(0);
        let commits = AtomicUsize::new(0);
        let result = admission
            .admit_decided(
                None,
                || true,
                |_window| {
                    decisions.fetch_add(1, Ordering::Relaxed);
                    PrefillProposal::Proposed(())
                },
                |capacity, ()| {
                    commits.fetch_add(1, Ordering::Relaxed);
                    capacity.select(Arc::clone(&worker), ())
                },
            )
            .await;
        assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
        assert_eq!(decisions.load(Ordering::Relaxed), NEWCOMER_ATTEMPTS);
        assert_eq!(commits.load(Ordering::Relaxed), NEWCOMER_ATTEMPTS);
        assert_eq!(admission.queued_requests(), 0);
        drop(first);
    }

    #[tokio::test]
    async fn a_decided_unavailable_request_behind_a_waiter_is_refused_not_queued() {
        let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let worker = worker("http://prefill-decided-unavailable-behind");
        let occupied = admission.admit(None, select_worker(&worker)).await.unwrap();
        let queued = plain_attempt(&admission, &worker);
        wait_for_queued(&admission, 1).await;

        // A waiter is queued, so the newcomer may not select and does not
        // decide; the availability check runs once, and a model with no
        // healthy prefill worker is refused `Unavailable` instead of taking
        // a place in the queue.
        let checks = AtomicUsize::new(0);
        let result = admission
            .admit_decided(
                None,
                || {
                    checks.fetch_add(1, Ordering::Relaxed);
                    false
                },
                |_window| -> PrefillProposal<()> {
                    panic!("a request behind a queue never decides")
                },
                |_capacity, ()| PrefillAdmissionAttempt::<()>::Unavailable,
            )
            .await;

        assert!(matches!(
            result,
            Err(PrefillAdmissionRejection::Unavailable)
        ));
        assert_eq!(checks.load(Ordering::Relaxed), 1);
        assert_eq!(admission.queued_requests(), 1);

        queued.abort();
        assert!(matches!(
            queued.await,
            Err(error) if error.is_cancelled()
        ));
        drop(occupied);
    }

    #[tokio::test(start_paused = true)]
    async fn a_slot_freed_before_the_queue_entry_admits_the_new_head_at_once() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(10));
        let worker = worker("http://prefill-decided-freed-before-entry");
        let occupied = admission.admit(None, select_worker(&worker)).await.unwrap();

        // The newcomer's decision reads the full worker; the slot is
        // released while that decision is in flight, after the read and
        // before the queue entry, so the release finds no waiter to wake.
        // The entry's new head must not sleep until the recheck interval.
        let mut occupied = Some(occupied);
        let decisions = AtomicUsize::new(0);
        let started = tokio::time::Instant::now();
        let admitted = admission
            .admit_decided(
                None,
                || true,
                |window| {
                    if decisions.fetch_add(1, Ordering::Relaxed) == 0 {
                        assert!(!window.has_capacity(&worker));
                        drop(occupied.take());
                        PrefillProposal::AtCapacity
                    } else {
                        assert!(window.has_capacity(&worker));
                        PrefillProposal::Proposed(())
                    }
                },
                |capacity, ()| capacity.select(Arc::clone(&worker), ()),
            )
            .await
            .unwrap();

        assert_eq!(decisions.load(Ordering::Relaxed), 2);
        assert_eq!(
            tokio::time::Instant::now(),
            started,
            "admitted without a wait on the recheck interval"
        );
        assert_eq!(admission.queued_requests(), 0);
        assert_eq!(worker.load(), 1);
        drop(admitted);
    }

    #[tokio::test(start_paused = true)]
    async fn a_new_head_behind_a_drained_queue_decides_at_once_and_waits_when_the_window_is_full() {
        let admission = PrefillAdmission::new(1, 1, Duration::from_secs(10));
        let worker = worker("http://prefill-decided-drained-queue-full-window");
        let occupied = admission.admit(None, select_worker(&worker)).await.unwrap();

        // A plain request waits behind the occupied slot, so the newcomer
        // finds others waiting and reads no window; that waiter leaves while
        // the newcomer asks whether its model is available, so its entry makes
        // it the lone head and it decides at once. The slot stays occupied:
        // the first decision finds the window full, and the head must wait
        // for the next wake or the recheck instead of deciding again at once.
        let mut queued = Box::pin(admission.admit(None, select_worker(&worker)));
        assert!(matches!(
            futures::poll!(queued.as_mut()),
            std::task::Poll::Pending
        ));
        assert_eq!(admission.queued_requests(), 1);

        let mut queued = Some(queued);
        let mut occupied = Some(occupied);
        let decisions = AtomicUsize::new(0);
        let started = tokio::time::Instant::now();
        let admitted = admission
            .admit_decided(
                None,
                || {
                    drop(queued.take());
                    true
                },
                |window| {
                    if decisions.fetch_add(1, Ordering::Relaxed) == 0 {
                        assert_eq!(
                            tokio::time::Instant::now(),
                            started,
                            "the lone new head decides at once"
                        );
                        assert!(!window.has_capacity(&worker));
                        PrefillProposal::AtCapacity
                    } else {
                        assert!(
                            tokio::time::Instant::now() >= started + HEAD_RECHECK_INTERVAL,
                            "a refused first pass waits for the next wake"
                        );
                        drop(occupied.take());
                        PrefillProposal::Proposed(())
                    }
                },
                |capacity, ()| capacity.select(Arc::clone(&worker), ()),
            )
            .await
            .unwrap();

        assert_eq!(decisions.load(Ordering::Relaxed), 2);
        assert_eq!(admission.queued_requests(), 0);
        assert_eq!(worker.load(), 1);
        drop(admitted);
    }
}
