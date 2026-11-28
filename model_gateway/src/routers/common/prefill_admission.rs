use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::identity::basic::{Worker, WorkerLoadGuard};
use tokio::sync::Notify;

use crate::{
    observability::metrics::Metrics,
    routers::common::placement::{Placed, RequestEnd, Unsent},
};

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

/// The Prefill load one client request holds from its selection to its
/// prefill's send: the one form of that load which is not sent. The worker
/// is counted from the selection on, under the route's own key, and
/// Grand Central is told nothing yet.
///
/// Its `sent` is the only way to the load a route holds through its
/// Prefill phase ([`PrefillLoadGuard`]), so a route says the send where it
/// hands the prefill's request to its client. This load has no replica: a
/// request that goes out as several says its send once, ahead of them. A
/// route that sends no prefill drops it: the worker is counted no more,
/// and what the answer left with the load is let go behind that, which
/// Grand Central hears as a release.
#[must_use = "the load is sent where the prefill goes out, or dropped where it never does"]
pub struct PrefillAhead {
    /// The worker's count and, with admission, its slot. Released first.
    load: PrefillLoadGuard,
    /// What the prefill's answer owes its send ([`Placed::unsent`]).
    /// `None` of an answer the classic policies gave.
    unsent: Option<Unsent>,
}

impl PrefillAhead {
    /// The prefill's request goes out now: the load as the request holds
    /// it through its Prefill phase. An answer of the classic policies
    /// hands its load on as it stands, by the one test of what the answer
    /// left. For an answer of Grand Central the engine hears the
    /// dispatch, the load's guard keeps the engine's side to the load's
    /// release, and from here on it counts the worker under the routing
    /// value the engine names ([`Unsent::sent`]).
    #[inline]
    pub(crate) fn sent(mut self) -> PrefillLoadGuard {
        // The engine's arm alone writes the load again: the load of a
        // classic answer goes on as it stands.
        if let Some(unsent) = self.unsent {
            self.load = self.load.sent_by(unsent);
        }
        self.load
    }

    /// Asks `refusal` ahead of this load's send, where that send says
    /// anything: an answer of Grand Central tells the engine the dispatch
    /// there ([`Self::sent`]). For a request that goes out as several,
    /// where what refuses one of the several refuses the send: refused
    /// here, the request drops this load as it was acquired, which the
    /// engine hears as a release. A classic answer's send says nothing, so
    /// it asks nothing here, and the several make their own refusals behind
    /// it. The load is asked where it lies, by reference: handed through an
    /// answer ahead of its send, it is carried to another place on a
    /// classic answer's path too.
    #[inline]
    pub(crate) fn ahead_of_its_send<E>(
        &self,
        refusal: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        if self.unsent.is_some() {
            refusal()
        } else {
            Ok(())
        }
    }

    /// A load of the classic policies' answer, for a case that needs one
    /// not yet sent.
    #[cfg(test)]
    pub(crate) fn classic(load: PrefillLoadGuard) -> Self {
        Self { load, unsent: None }
    }
}

/// The Prefill load held for one client request, from its prefill's send
/// on (`PrefillAhead::sent`).
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
    /// This load, sent for an answer of Grand Central: the load's guard
    /// is handed to what the answer left ([`Unsent::sent`]) and keeps the
    /// engine's side from here on, so the load's release is the end of the
    /// prefill's side. With admission that guard is the reservation's own,
    /// made under the admission lock, and this handle is the reservation's
    /// one: a load that is not sent has no replica. Out of line: the load
    /// of a classic answer never comes here.
    #[inline(never)]
    fn sent_by(self, unsent: Unsent) -> Self {
        match self {
            Self::Unbounded { _guard } => Self::Unbounded {
                _guard: unsent.sent(_guard),
            },
            Self::Admission { mut _reservation } => {
                let held = Arc::get_mut(&mut _reservation);
                debug_assert!(held.is_some(), "a load that is not sent has one handle");
                if let Some(held) = held {
                    if let Some(guard) = held.load_guard.take() {
                        held.load_guard = Some(unsent.sent(guard));
                    }
                }
                Self::Admission { _reservation }
            }
        }
    }

    /// The handle on the end of the prefill this load counts
    /// ([`RequestEnd::of`]), asked of the worker guard the load holds: its
    /// own, or its reservation's. A pair route tells the prefill's end
    /// through it where it records the prefill's status, and keeps no
    /// handle past this load: the prefill's end waits for none.
    ///
    /// A replica asks the same side. It shares the reservation, or it is a
    /// guard of its own that keeps the first one's handle
    /// ([`Self::replicate`]). A later report writes over an earlier one, but
    /// no good answer over a failure. A load whose guard is gone holds no side
    /// and answers the empty handle.
    #[inline]
    pub(crate) fn end(&self) -> RequestEnd {
        match self {
            Self::Admission { _reservation } => _reservation
                .load_guard
                .as_ref()
                .map_or_else(RequestEnd::default, RequestEnd::of),
            Self::Unbounded { _guard } => RequestEnd::of(_guard),
        }
    }

    /// Acquire another guard for the same request.
    ///
    /// This is explicit instead of `Clone` because the unbounded variant
    /// increments externally visible worker load.
    #[inline]
    pub(crate) fn replicate(&self) -> Self {
        match self {
            Self::Admission { _reservation } => Self::Admission {
                _reservation: Arc::clone(_reservation),
            },
            // A guard that keeps nothing is replicated as it stands: a load
            // the classic policies placed pays the one test of that word.
            Self::Unbounded { _guard } if _guard.kept().is_none() => Self::Unbounded {
                _guard: _guard.replicate(),
            },
            // The replica keeps the handle of the guard it is made of: a
            // child that holds it tells the request's one prefill side.
            Self::Unbounded { _guard } => Self::Unbounded {
                _guard: RequestEnd::kept_by_replica_of(_guard),
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

/// Why `acquire_prefill` could not reserve Prefill capacity.
pub enum PrefillAcquireError<E> {
    /// Candidate selection found no worker to wait for. With admission
    /// enabled this is the verdict of the last attempt at the queue head.
    Candidate(E),
    /// The request could not be queued or waited too long.
    Rejected(PrefillAdmissionRejection),
}

/// The load of a prefill selected with admission disabled, held ahead of
/// its send: what [`acquire_prefill`] does with what its selection answered
/// there. The worker `prefill_of` names is counted under `routing_key` by a
/// plain guard, with no test of the answer, and what `hand_on` says the
/// answer owes its send goes with that load. The decision bench's pair
/// route calls this too, so it makes the calls a pair route makes.
#[inline]
pub(crate) fn hold_unbounded<C, A>(
    candidate: C,
    routing_key: Option<&str>,
    prefill_of: impl Fn(&C) -> &Placed,
    hand_on: impl FnOnce(C) -> (A, Option<Unsent>),
) -> (A, PrefillAhead) {
    let prefill = Arc::clone(prefill_of(&candidate).worker());
    let load = PrefillLoadGuard::Unbounded {
        _guard: WorkerLoadGuard::with_key(prefill, routing_key),
    };
    let (handed_on, unsent) = hand_on(candidate);
    (handed_on, PrefillAhead { load, unsent })
}

/// Select a Prefill worker for one client request and hold its load ahead
/// of the prefill's send.
///
/// With admission enabled the selection runs at the head of the FIFO queue and
/// under the admission lock, so `select` must filter candidates with
/// [`PrefillSelectionContext::has_capacity`] before it invokes a stateful
/// routing policy. With admission disabled `select` receives `None` and runs
/// once.
///
/// `prefill_of` names the prefill of what `select` answers, as the placement
/// answers a worker ([`Placed`]). The load held is a plain guard of that
/// worker under `routing_key`, made with no test of the answer: here with
/// admission disabled, and by the admission itself behind the selection
/// with it enabled ([`PrefillAdmission::admit`] holds the answer as a value
/// it does not read). So the worker counts the request under the route's
/// key up to the prefill's send, even where Grand Central names a
/// routing value of its own.
///
/// `hand_on` takes what `select` answered as the route keeps it: the
/// prefill's worker goes on behind its load, and what its answer owes its
/// send ([`Placed::unsent`]) is handed back here and goes with the load
/// ([`PrefillAhead`]). The route says the send where it sends the prefill
/// ([`PrefillAhead::sent`]).
pub(crate) async fn acquire_prefill<C, A, E, S, P, H>(
    admission: Option<&PrefillAdmission>,
    routing_key: Option<&str>,
    prefill_of: P,
    mut select: S,
    hand_on: H,
) -> Result<(A, PrefillAhead), PrefillAcquireError<E>>
where
    S: for<'a> FnMut(Option<&PrefillSelectionContext<'a>>) -> Result<C, E> + Send,
    P: Fn(&C) -> &Placed + Send + Sync,
    H: FnOnce(C) -> (A, Option<Unsent>) + Send,
    E: PrefillCandidateError + Send,
{
    let Some(admission) = admission else {
        let candidate = select(None).map_err(PrefillAcquireError::Candidate)?;
        return Ok(hold_unbounded(candidate, routing_key, prefill_of, hand_on));
    };

    // The selection verdict is kept so an unavailable fleet answers with the
    // role's own reason (a shed, a missing role) rather than a generic one.
    let mut verdict = None;
    let admitted = admission
        .admit(routing_key, |capacity| match select(Some(capacity)) {
            Ok(candidate) => {
                let prefill = Arc::clone(prefill_of(&candidate).worker());
                capacity.select(prefill, candidate)
            }
            Err(error) if error.is_at_capacity() => PrefillAdmissionAttempt::AtCapacity,
            Err(error) => {
                verdict = Some(error);
                PrefillAdmissionAttempt::Unavailable
            }
        })
        .await;

    match (admitted, verdict) {
        (Ok(admitted), _) => {
            let load = PrefillLoadGuard::Admission {
                _reservation: Arc::new(admitted.reservation),
            };
            let (handed_on, unsent) = hand_on(admitted.selected);
            Ok((handed_on, PrefillAhead { load, unsent }))
        }
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

/// Admission helpers for tests elsewhere in the crate.
#[cfg(test)]
#[path = "../../../tests/unit/routers/common/prefill_admission/test_support.rs"]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/prefill_admission/tests.rs"]
mod tests;
