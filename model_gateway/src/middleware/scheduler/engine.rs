//! Priority-aware admission scheduler engine.
//!
//! Owns the [`SlotPool`], per-class [`ClassQueue`]s, and the inflight
//! registry. Construction validates that the configured reservations
//! fit under the live backend capacity; [`PriorityScheduler::admit`] is
//! the admission (a fast path, at most one preemption, then the class
//! queue), and the dispatcher task applies capacity changes and drains
//! the queues on every release, capacity growth and starvation tick.

use std::{
    cmp::Reverse,
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use smg_auth::RequestId;
use thiserror::Error;
use tokio::sync::{oneshot, watch, Notify};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::{
    inflight::InflightHandle,
    queue::{ClassQueue, FifoClassQueue, Waiter},
    slots::SlotPool,
    Class, ClassRuntimeConfig, SchedulerSettings,
};
use crate::{middleware::edge_body::Permit, worker::WorkerCapacity};

/// Max time to wait, after firing a preemption cancel, for the victim's slot
/// to free before falling back to enqueue.
const PREEMPTION_WAIT_BUDGET: Duration = Duration::from_millis(50);

/// Poll interval while waiting for a preempted slot to free.
const PREEMPTION_POLL_INTERVAL: Duration = Duration::from_millis(2);

/// Construction-time failures for [`PriorityScheduler::new`].
///
/// Capacity-vs-reserved is the only invariant the scheduler can check —
/// per-field validation already happened in
/// [`SchedulerSettings::from_cli_and_yaml`].
#[derive(Debug, Error, PartialEq)]
pub enum SchedulerInitError {
    #[error("sum of class reservations ({reserved}) exceeds capacity ({capacity})")]
    ReservationsExceedCapacity { reserved: u32, capacity: u16 },
}

/// Outcome of an [`PriorityScheduler::admit`] call.
pub enum AdmitOutcome {
    Admitted(SchedulerPermit),
    Rejected(RejectionReason),
}

/// Why an admission was rejected. Maps directly to the HTTP response
/// status in the admission middleware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// Per-class queue is at its configured limit. → 429.
    QueueFull,
    /// Queued waiter aged past `queue_timeout`. → 503 + Retry-After.
    QueueTimeout,
    /// Scheduler cancelled this inflight to admit a higher-priority
    /// waiter. → 503 + Retry-After.
    Preempted,
    /// The caller's cancellation token fired before admission completed
    /// (typically the HTTP client disconnected). Never serialized — the
    /// client is already gone.
    ClientCancelled,
}

/// Priority-aware admission scheduler.
///
/// Construction wires up the slot pool, per-class queues, and an empty
/// inflight registry; [`Self::admit`] admits, the dispatcher
/// ([`Self::spawn_dispatcher`]) drains the queues and applies capacity
/// changes, and [`Self::acquire_inflight`] is the fast path on its own,
/// which the tests also exercise against a real slot pool.
pub struct PriorityScheduler {
    slot_pool: SlotPool,
    class_queues: [Arc<dyn ClassQueue>; 4],
    inflight_registry: RwLock<HashMap<RequestId, Arc<InflightHandle>>>,
    /// Arc so the dispatcher task can await on it without holding a strong
    /// reference to the scheduler. On scheduler `Drop` we fire `notify_one`
    /// so the dispatcher wakes, observes the failed `Weak::upgrade`, and
    /// exits cleanly.
    pub(super) release_notify: Arc<Notify>,
    class_config: [ClassRuntimeConfig; 4],
    /// Per-class reservation floors (immutable baseline). The live
    /// `slot_pool` reservations are recomputed from the floors + shares on
    /// every capacity change, so the effective reservation tracks capacity
    /// and a recovery restores it instead of eroding permanently.
    reserved_floor: [u16; 4],
    /// Per-class reservation shares of capacity (immutable baseline).
    /// Effective = `max(floor, ceil(share × capacity))`.
    reserved_per_slot: [f64; 4],
}

impl Drop for PriorityScheduler {
    fn drop(&mut self) {
        // Kick the dispatcher one last time so it can observe the Weak
        // upgrade failure and exit instead of awaiting forever.
        self.release_notify.notify_one();
    }
}

impl PriorityScheduler {
    /// Build a scheduler against the given settings and live backend
    /// capacity. Refuses if the configured reservation floors + shares, at
    /// the initial capacity, sum to more than the available capacity (a
    /// too-small fleet or misconfiguration — the caller falls back to legacy
    /// admission). Runtime capacity dips are handled gracefully by the
    /// proportional clamp in `apply_new_capacity`, not by rejection.
    pub fn new(
        settings: &SchedulerSettings,
        capacity: u16,
    ) -> Result<Arc<Self>, SchedulerInitError> {
        let reserved_floor = Class::ALL.map(|c| settings.class_config(c).reserved_floor);
        let reserved_per_slot = Class::ALL.map(|c| settings.class_config(c).reserved_per_slot);

        let desired = desired_reservations(reserved_floor, &reserved_per_slot, capacity);
        let total: u32 = desired.iter().map(|&r| u32::from(r)).sum();
        if total > u32::from(capacity) {
            return Err(SchedulerInitError::ReservationsExceedCapacity {
                reserved: total,
                capacity,
            });
        }

        let class_queues: [Arc<dyn ClassQueue>; 4] = Class::ALL.map(|c| queue_for(settings, c));
        let class_config: [ClassRuntimeConfig; 4] =
            Class::ALL.map(|c| ClassRuntimeConfig::from_class_config(settings.class_config(c)));

        Ok(Arc::new(Self {
            slot_pool: SlotPool::new(capacity, desired),
            class_queues,
            inflight_registry: RwLock::new(HashMap::new()),
            release_notify: Arc::new(Notify::new()),
            class_config,
            reserved_floor,
            reserved_per_slot,
        }))
    }

    /// Try to acquire a slot under `class` for the given request id.
    /// Returns `Some(permit)` on success and `None` if the slot pool
    /// refuses (e.g. capacity exhausted or reservation guard blocks the
    /// class). [`Self::admit`]'s fast path; its preempt and queue paths
    /// layer on top.
    pub fn acquire_inflight(
        self: &Arc<Self>,
        class: Class,
        request_id: RequestId,
    ) -> Option<SchedulerPermit> {
        if !self.slot_pool.try_acquire(class) {
            return None;
        }
        Some(self.register_inflight(class, request_id))
    }

    /// Register a handle in the registry and wrap it in a permit.
    /// Caller already acquired a slot via the pool.
    fn register_inflight(self: &Arc<Self>, class: Class, request_id: RequestId) -> SchedulerPermit {
        let handle = Arc::new(InflightHandle::new(class, request_id));
        self.inflight_registry
            .write()
            .insert(handle.request_id().clone(), Arc::clone(&handle));
        SchedulerPermit {
            scheduler: Arc::clone(self),
            handle,
        }
    }

    /// Admit a request under `class`. Tries the fast path first; if the
    /// slot pool refuses, enqueues a waiter and awaits the dispatcher
    /// (or a queue timeout, or a client-side cancel).
    ///
    /// The `cancel` token is monitored for the duration of the wait —
    /// fires it to short-circuit a queued admission when the client
    /// disconnects.
    pub async fn admit(
        self: &Arc<Self>,
        class: Class,
        request_id: RequestId,
        cancel: CancellationToken,
    ) -> AdmitOutcome {
        // Fast path: slot available, admit synchronously.
        if let Some(permit) = self.acquire_inflight(class, request_id.clone()) {
            return AdmitOutcome::Admitted(permit);
        }

        // Preemption path: if this class may preempt, try to cancel one
        // lower-class pre-TTFT request and claim its slot. At most one
        // preemption per admission (no cascading); if the victim's body
        // doesn't wind down within the budget we fall through to enqueue
        // — the cancel still fired, so the slot frees shortly regardless.
        if self.class_config[class as usize].can_preempt {
            // Don't cancel a victim on behalf of a caller whose client has
            // already disconnected — the queued path treats a fired `cancel`
            // as ClientCancelled, and the preempt path must match.
            if cancel.is_cancelled() {
                return AdmitOutcome::Rejected(RejectionReason::ClientCancelled);
            }
            if let Some(victim) = self.find_preemption_victim(class) {
                // CAS the victim pre-TTFT lock. Fire its cancel only on a
                // win, so we never unwind a request that already streamed.
                //
                // The mark is irreversible (no rollback — a CAS back to 0
                // could race a now-arriving first byte). So a victim that
                // doesn't unwind within the budget is still preempted: its
                // first data frame is truncated by the edge's body wrapper. A
                // victim unwinds earlier only where its handler runs its
                // upstream work under the `PreemptionGuard` (every inference
                // handler in `server.rs` does, whichever router serves it, as
                // does the HTTP router's streaming relay); elsewhere the slot
                // frees when the body ends.
                //
                // The freed slot is not reserved for this preemptor — the
                // dispatcher (on the victim's release) or a concurrent
                // fast-path admit may take it first, in which case we simply
                // fall through to enqueue. A successful mark therefore does
                // not guarantee this preemptor benefits.
                if victim.try_mark_preempted() {
                    info!(
                        victim_id = %victim.request_id().0,
                        victim_class = ?victim.class(),
                        preemptor_class = ?class,
                        "scheduler: preempting pre-TTFT request to admit higher class"
                    );
                    victim.cancel();
                    super::metrics::record_preemption(victim.class(), class);
                    if let Some(permit) = self
                        .wait_for_slot(class, request_id.clone(), &cancel, PREEMPTION_WAIT_BUDGET)
                        .await
                    {
                        return AdmitOutcome::Admitted(permit);
                    }
                    // Client went away while we waited — don't enqueue work
                    // nobody is waiting for.
                    if cancel.is_cancelled() {
                        return AdmitOutcome::Rejected(RejectionReason::ClientCancelled);
                    }
                    warn!(
                        preemptor_class = ?class,
                        budget_ms = PREEMPTION_WAIT_BUDGET.as_millis() as u64,
                        "scheduler: preempted slot did not free within budget; enqueueing"
                    );
                    // Slot not free in time — fall through to enqueue.
                }
            }
        }

        // Slow path: enqueue and wait. The waiter holds a child cancel
        // token of the caller's `cancel` so the queue's
        // `drop_cancelled_head` GC sees the cancellation regardless of
        // which select arm fires below — client cancel propagates via
        // the parent, queue timeout fires the child explicitly.
        let (tx, rx) = oneshot::channel::<SchedulerPermit>();
        let waiter_cancel = cancel.child_token();
        let waiter = Waiter::new(class, waiter_cancel.clone(), request_id, tx);
        if self.class_queues[class as usize]
            .try_enqueue(waiter)
            .is_err()
        {
            return AdmitOutcome::Rejected(RejectionReason::QueueFull);
        }
        let enqueued_at = Instant::now();
        // The client may go away while we are queued: nothing fires `cancel`
        // (the middleware cannot see a disconnect before the handler runs),
        // the response future is dropped and this future with it. The guard
        // records that exit (the wait, the `client_cancelled` outcome) and
        // reaps the waiter; without it an abandoned wait left no trace in
        // any counter, only in the queue-depth gauge until the dispatcher
        // found the closed receiver.
        let mut queued = QueuedWait {
            class,
            enqueued_at,
            waiter_cancel: waiter_cancel.clone(),
            release_notify: Arc::clone(&self.release_notify),
            outcome_reached: false,
        };

        // Lost-wakeup guard: a slot may have been released after our
        // fast-path try_acquire failed but before try_enqueue returned.
        // That `notify_one` already fired and the dispatcher already
        // drained — without an extra nudge here, our newly-queued waiter
        // would sit behind a now-free slot until the next unrelated
        // release event. Re-kick the dispatcher to re-evaluate.
        self.release_notify.notify_one();

        let timeout = self.class_config[class as usize].queue_timeout;
        let outcome = tokio::select! {
            result = rx => match result {
                Ok(permit) => AdmitOutcome::Admitted(permit),
                // The dispatcher dropped our sender without admitting us.
                // Treat as a cancellation rather than a timeout — the
                // dispatcher only drops if it knows we no longer need a slot.
                Err(_) => AdmitOutcome::Rejected(RejectionReason::ClientCancelled),
            },
            () = tokio::time::sleep(timeout) => AdmitOutcome::Rejected(RejectionReason::QueueTimeout),
            () = cancel.cancelled() => AdmitOutcome::Rejected(RejectionReason::ClientCancelled),
        };
        queued.outcome_reached = true;
        super::metrics::record_queue_wait(class, enqueued_at.elapsed());
        // The guard's drop marks the waiter cancelled and kicks the
        // dispatcher, on this path as on the abandoned one.
        drop(queued);
        outcome
    }

    /// Remove a handle from the registry, release its slot, and notify
    /// the dispatcher. Called from [`SchedulerPermit`]'s `Drop`.
    fn release_inflight(&self, handle: &InflightHandle) {
        self.inflight_registry.write().remove(handle.request_id());
        self.slot_pool.release(handle.class());
        self.release_notify.notify_one();
    }

    /// Test-only in-flight count for a class. Lets sibling-module tests
    /// (e.g. `body.rs`) assert slot release without reaching the private
    /// `slot_pool` field.
    #[cfg(test)]
    pub(crate) fn inflight_for_test(&self, class: Class) -> u16 {
        self.slot_pool.inflight(class)
    }

    /// Find the best preemption victim for an admission of `waiter_class`,
    /// or `None` if there is no eligible victim.
    ///
    /// Eligible = a strictly-lower-class inflight request that is still
    /// pre-TTFT (`is_preemptible`). Among those, prefer the **lowest
    /// class** (cheapest to cancel) and, within that class, the
    /// **most-recently-admitted** request (least upstream work wasted).
    /// `Reverse(class)` makes the lowest class sort highest under
    /// `max_by_key`; `admitted_at` then breaks ties toward the newest.
    ///
    /// Read-locks the registry only; never mutates. Callers gate this on
    /// `class_config[class].can_preempt`, so it runs only on the
    /// contention path for a preempt-capable class — never the hot path.
    fn find_preemption_victim(&self, waiter_class: Class) -> Option<Arc<InflightHandle>> {
        self.inflight_registry
            .read()
            .values()
            .filter(|h| h.class() < waiter_class && h.is_preemptible())
            .max_by_key(|h| (Reverse(h.class()), h.admitted_at()))
            .cloned()
    }

    /// Poll the slot pool for up to `budget` for a slot to free under
    /// `class`, returning a permit if one is acquired. Used only after
    /// firing a preemption cancel, to grab the victim's slot as its body
    /// winds down. Polls (rather than sharing `release_notify` with the
    /// dispatcher, which would let the dispatcher steal the single
    /// `notify_one` wakeup); this is the contention path, not the hot
    /// path, and the poll is bounded.
    async fn wait_for_slot(
        self: &Arc<Self>,
        class: Class,
        request_id: RequestId,
        cancel: &CancellationToken,
        budget: Duration,
    ) -> Option<SchedulerPermit> {
        let deadline = Instant::now() + budget;
        loop {
            // Acquire the slot and register in one step, consuming
            // `request_id` only on success — so a failed poll never clones it
            // (this loop runs up to budget/poll-interval times per
            // preemption).
            if self.slot_pool.try_acquire(class) {
                return Some(self.register_inflight(class, request_id));
            }
            if Instant::now() >= deadline {
                return None;
            }
            // Abort the wait if the caller's client disconnects — no point
            // holding the victim's freed slot for a request nobody wants.
            tokio::select! {
                () = tokio::time::sleep(PREEMPTION_POLL_INTERVAL) => {}
                () = cancel.cancelled() => return None,
            }
        }
    }

    /// Try to admit one queued waiter. Returns `true` if a waiter was
    /// successfully admitted (caller should call again to drain), `false`
    /// if nothing was admittable this pass.
    ///
    /// Honors two policies in order:
    /// 1. **Starvation override** — scans Bulk → Default → Interactive for
    ///    a head waiter that has aged past its class's
    ///    `starvation_threshold`. The first such waiter is admitted via
    ///    `try_acquire_ignoring_reservations`, bypassing the reservation
    ///    guard so a starved low-class waiter can take a slot that a
    ///    higher class has reserved-but-not-used.
    /// 2. **Normal priority** — System → Interactive → Default → Bulk.
    ///    Each class's queue is drained one waiter at a time so the
    ///    caller's outer loop can interleave drains across classes.
    pub fn wake_next_waiter(self: &Arc<Self>) -> bool {
        // Starvation override — lowest priority first (the ones most at
        // risk of starving).
        for class in [Class::Bulk, Class::Default, Class::Interactive] {
            let idx = class as usize;
            self.class_queues[idx].drop_cancelled_head();
            let Some(head_age) = self.class_queues[idx].head_age() else {
                continue;
            };
            if head_age <= self.class_config[idx].starvation_threshold {
                continue;
            }
            if self.slot_pool.try_acquire_ignoring_reservations(class) {
                if self.send_to_head(class) {
                    super::metrics::record_starvation_promotion(class);
                    return true;
                }
                // Acquired a slot but the head was gone (racy cancel).
                // Release and fall through to normal priority.
                self.slot_pool.release(class);
            }
        }

        // Normal priority — highest class first.
        for class in [
            Class::System,
            Class::Interactive,
            Class::Default,
            Class::Bulk,
        ] {
            let idx = class as usize;
            self.class_queues[idx].drop_cancelled_head();
            if self.class_queues[idx].depth() == 0 {
                continue;
            }
            if self.slot_pool.try_acquire(class) {
                if self.send_to_head(class) {
                    return true;
                }
                self.slot_pool.release(class);
            }
        }

        false
    }

    /// Pop the head waiter for `class` and deliver a permit through its
    /// oneshot. Drains waiters whose receivers have already been
    /// dropped (admit timed out / client cancelled between pop and send)
    /// without performing the wasted `register_inflight` + immediate
    /// release for each. Returns `false` if the queue is exhausted
    /// without finding a live waiter.
    ///
    /// The caller must have already acquired a slot under `class`.
    fn send_to_head(self: &Arc<Self>, class: Class) -> bool {
        loop {
            let Some(Waiter {
                request_id,
                permit_tx,
                ..
            }) = self.class_queues[class as usize].pop_eligible()
            else {
                return false;
            };
            if permit_tx.is_closed() {
                // Receiver gone — skip the registry write and the
                // matched permit-drop release. Try the next waiter
                // using the same slot we already acquired.
                continue;
            }
            let permit = self.register_inflight(class, request_id);
            // If the receiver was dropped between is_closed() above and
            // send below (unlikely race window), the permit goes out of
            // scope and its Drop releases the slot back; the caller's
            // outer loop will try again.
            let _ = permit_tx.send(permit);
            return true;
        }
    }

    /// Spawn the dispatcher background task. `select!`s over two
    /// signals:
    ///
    /// - `release_notify` — slot was released (or a waiter enqueued
    ///   via Drop-fire on scheduler shutdown). Drain queued waiters
    ///   until none can be admitted.
    /// - `capacity_watch.changed()` — backend capacity changed. Apply
    ///   the new value (scaling reservations down if they no longer
    ///   fit) and kick the drain.
    ///
    /// Holds only a `Weak<Self>` so the task does not keep the
    /// scheduler alive past its last external strong reference. Drop
    /// on the scheduler fires `release_notify`, which lets the
    /// dispatcher observe the failed upgrade and exit.
    pub fn spawn_dispatcher(self: &Arc<Self>, capacity_watch: watch::Receiver<u16>) {
        self.spawn_dispatcher_inner(capacity_watch, None);
    }

    /// Spawn the dispatcher while retaining the capacity tracker that owns
    /// the watch sender. The dispatcher is the tracker consumer, so its task
    /// owns the tracker for exactly as long as dynamic capacity is needed.
    pub(crate) fn spawn_dispatcher_retaining_capacity(
        self: &Arc<Self>,
        capacity_watch: watch::Receiver<u16>,
        worker_capacity: Arc<WorkerCapacity>,
    ) {
        self.spawn_dispatcher_inner(capacity_watch, Some(worker_capacity));
    }

    fn spawn_dispatcher_inner(
        self: &Arc<Self>,
        capacity_watch: watch::Receiver<u16>,
        capacity_owner: Option<Arc<WorkerCapacity>>,
    ) {
        let weak = Arc::downgrade(self);
        let notify = Arc::clone(&self.release_notify);
        // Periodic tick so the starvation override can fire even when no
        // release events arrive. Set to the smallest per-class
        // starvation_threshold so a head waiter never ages past its
        // threshold by more than ~one tick before the dispatcher checks.
        let tick_period = Class::ALL
            .iter()
            .map(|c| self.class_config[*c as usize].starvation_threshold)
            .min()
            .unwrap_or(Duration::from_secs(60));
        #[expect(
            clippy::disallowed_methods,
            reason = "dispatcher loop holds only a Weak<Self> and exits when the scheduler is dropped (Drop fires release_notify)"
        )]
        tokio::spawn(async move {
            // Keep the sender and its worker-event task alive until this
            // dispatcher exits. Existing receiver-only callers pass None.
            let capacity_owner = capacity_owner;
            // `Option<watch::Receiver>` so we can drop the receiver once the
            // upstream sender is closed. A bare `Receiver` whose sender is
            // gone makes `changed()` resolve `Err` immediately on every
            // poll — the `select!` would otherwise pick that arm forever
            // and burn CPU.
            let mut capacity_watch = Some(capacity_watch);
            let mut tick = tokio::time::interval(tick_period);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                // This read keeps an optional tracker owned by the task for
                // every iteration, without extending Scheduler's lifetime.
                let _ = capacity_owner.as_ref();
                let new_capacity = match capacity_watch.as_mut() {
                    Some(rx) => tokio::select! {
                        () = notify.notified() => None,
                        _ = tick.tick() => None,
                        result = rx.changed() => match result {
                            Ok(()) => Some(*rx.borrow()),
                            Err(_) => {
                                // Upstream sender dropped. Stop watching;
                                // the dispatcher keeps serving release
                                // events via the other arm.
                                capacity_watch = None;
                                continue;
                            }
                        },
                    },
                    None => {
                        tokio::select! {
                            () = notify.notified() => {}
                            _ = tick.tick() => {}
                        }
                        None
                    }
                };
                let Some(scheduler) = weak.upgrade() else {
                    break;
                };
                match new_capacity {
                    Some(new_cap) => scheduler.apply_new_capacity(new_cap),
                    None => while scheduler.wake_next_waiter() {},
                }
            }
        });
    }

    /// Spawn the metrics sampler: every `interval`, refresh the point-in-time
    /// capacity / autoscaling gauges from the slot pool and queues. Keeping
    /// these off the admission path means the hot path only does cheap
    /// counter increments. Holds only a `Weak<Self>`; exits within one
    /// interval of the scheduler being dropped.
    pub fn spawn_sampler(self: &Arc<Self>, interval: Duration) {
        let weak = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "sampler holds only a Weak<Self> and exits when the scheduler is dropped"
        )]
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                let Some(scheduler) = weak.upgrade() else {
                    break;
                };
                scheduler.sample_metrics();
            }
        });
    }

    /// Refresh the capacity / autoscaling gauges. Reads the slot pool and
    /// queues under their own locks; never touches the inflight registry.
    fn sample_metrics(&self) {
        let capacity = self.slot_pool.capacity();
        let mut total_inflight: u32 = 0;
        for class in Class::ALL {
            let inflight = self.slot_pool.inflight(class);
            total_inflight += u32::from(inflight);
            let depth = self.class_queues[class as usize].depth();
            let limit = self.class_queues[class as usize].capacity();
            super::metrics::set_inflight(class, inflight);
            super::metrics::set_queue_depth(class, depth);
            super::metrics::set_queue_size_limit(class, limit);
            super::metrics::set_class_capacity_pressure(
                class,
                self.class_pressure(class, inflight, depth, limit, capacity),
            );
        }
        let utilization = if capacity == 0 {
            0.0
        } else {
            f64::from(total_inflight) / f64::from(capacity)
        };
        super::metrics::set_utilization(utilization);
    }

    /// Normalized 0.0–1.0 pressure for `class`: the worse of queue pressure
    /// (`depth / limit`) and slot pressure (`max(0, inflight − reserved)`
    /// over the capacity not reserved by higher classes).
    fn class_pressure(
        &self,
        class: Class,
        inflight: u16,
        depth: usize,
        limit: usize,
        capacity: u16,
    ) -> f64 {
        let queue_pressure = if limit == 0 {
            0.0
        } else {
            depth as f64 / limit as f64
        };
        let higher_reserved: u32 = Class::ALL
            .iter()
            .filter(|&&c| c > class)
            .map(|&c| u32::from(self.slot_pool.reserved(c)))
            .sum();
        let headroom = u32::from(capacity).saturating_sub(higher_reserved).max(1);
        let overflow =
            u32::from(inflight).saturating_sub(u32::from(self.slot_pool.reserved(class)));
        let slot_pressure = f64::from(overflow) / f64::from(headroom);
        queue_pressure.max(slot_pressure).min(1.0)
    }

    /// Apply a new backend capacity from the WorkerCapacity watch.
    ///
    /// Recomputes the live per-class reservations from the immutable floors +
    /// shares (`desired_reservations`), so the effective reservation tracks
    /// capacity and a recovery restores it automatically. When the desired
    /// sum exceeds the new capacity (a pool smaller than the floors, or a
    /// runtime dip below what the floors + shares want), the proportional
    /// clamp scales every class's reservation by the same fraction and
    /// leaves the rounding remainder unreserved — runtime never rejects, and
    /// no class is locked out of an idle pool. On grow, also fires
    /// `release_notify` so the dispatcher re-evaluates queued waiters under
    /// the larger ceiling.
    ///
    /// The capacity and the per-class reservations are separate atomics, so
    /// they cannot be published together. We order the two writes so a
    /// concurrent fast-path `try_acquire` never sees the larger capacity
    /// paired with the old, understated reservations (which would let a
    /// low-priority admission slip into space a higher class is about to
    /// reserve, and that request would hold the slot past the restore). The
    /// rule: publish the more-restrictive value first. The transient is then
    /// always conservative — `slots_available_to` saturates, so a brief
    /// `Σ reserved > capacity` only under-admits, never over-admits.
    fn apply_new_capacity(&self, new_capacity: u16) {
        let old = self.slot_pool.capacity();
        if new_capacity == old {
            return;
        }

        // Desired reservations (floor + share) at the new capacity, then the
        // proportional clamp if they no longer fit. Pure function of
        // (immutable baseline, new capacity), so it is idempotent and both
        // directional.
        let desired =
            desired_reservations(self.reserved_floor, &self.reserved_per_slot, new_capacity);
        let desired_total: u32 = desired.iter().map(|&r| u32::from(r)).sum();
        let new_reserved = if desired_total > u32::from(new_capacity) {
            let clamped = clamp_reservations_to_capacity(desired, new_capacity);
            let reserved_total: u32 = clamped.iter().map(|&r| u32::from(r)).sum();
            warn!(
                new_capacity,
                desired_total_reserved = desired_total,
                reserved_system = clamped[Class::System as usize],
                reserved_interactive = clamped[Class::Interactive as usize],
                reserved_default = clamped[Class::Default as usize],
                reserved_bulk = clamped[Class::Bulk as usize],
                unreserved = u32::from(new_capacity).saturating_sub(reserved_total),
                "scheduler: capacity is below the configured reservations; each class keeps the \
                 same fraction of its reservation and the rest of the pool is shared by every class"
            );
            clamped
        } else {
            desired
        };

        if new_capacity > old {
            // Grow: raise the reservations before exposing the larger
            // capacity, then wake the dispatcher to drain under the new ceiling.
            for class in Class::ALL {
                self.slot_pool
                    .set_reserved(class, new_reserved[class as usize]);
            }
            self.slot_pool.set_capacity(new_capacity);
            self.release_notify.notify_one();
        } else {
            // Shrink: drop the capacity before lowering the reservations.
            self.slot_pool.set_capacity(new_capacity);
            for class in Class::ALL {
                self.slot_pool
                    .set_reserved(class, new_reserved[class as usize]);
            }
        }
    }
}

/// A queued admission's exit guard. On every exit it marks the waiter
/// cancelled so the queue's `drop_cancelled_head` reaps it (harmless if the
/// waiter was already popped: the token has no other readers) and kicks the
/// dispatcher so head GC runs promptly (without this, a `queue_size = 1`
/// queue could reject the *next* admit as QueueFull for up to one periodic
/// tick even though the head waiter is gone). When no outcome was reached,
/// the future was dropped while queued (the client disconnected): the wait
/// and the `client_cancelled` outcome are recorded here, since the caller
/// never sees a result to record.
struct QueuedWait {
    class: Class,
    enqueued_at: Instant,
    waiter_cancel: CancellationToken,
    release_notify: Arc<Notify>,
    outcome_reached: bool,
}

impl Drop for QueuedWait {
    fn drop(&mut self) {
        if !self.outcome_reached {
            super::metrics::record_queue_wait(self.class, self.enqueued_at.elapsed());
            super::metrics::record_admit(self.class, super::metrics::outcome::CLIENT_CANCELLED);
        }
        self.waiter_cancel.cancel();
        self.release_notify.notify_one();
    }
}

/// Desired per-class reservation at `capacity`:
/// `max(floor, ceil(per_slot × capacity))`, capped per class at `capacity`.
/// The per-class result always fits the pool, but `Σ` may exceed `capacity`
/// (resolved by [`clamp_reservations_to_capacity`]).
fn desired_reservations(floor: [u16; 4], per_slot: &[f64; 4], capacity: u16) -> [u16; 4] {
    let cap = u32::from(capacity);
    Class::ALL.map(|class| {
        let i = class as usize;
        // per_slot is validated finite & >= 0 in SchedulerSettings; the guard
        // and the saturating `as u32` keep a pathological value harmless.
        let share = (per_slot[i] * f64::from(capacity)).ceil();
        let share = if share.is_finite() && share >= 0.0 {
            share as u32
        } else {
            0
        };
        u32::from(floor[i]).max(share).min(cap) as u16
    })
}

/// Clamp desired reservations to fit `capacity`: every class keeps the same
/// fraction of what it asked for (`desired × capacity / Σ desired`, rounded
/// down), so no class is reserved whole at the expense of the others. A
/// priority-ordered fill would hand a pool smaller than the floors to the
/// highest class entirely: with the built-in floors, a pool of four slots
/// was reserved for `system`, which carries no traffic, and every other
/// class was admitted only by the starvation override. The rounding
/// remainder is left unreserved and shared by every class; when the scaled
/// reservations still fill the pool exactly, one slot is taken back from the
/// largest reservation (the highest class on a tie) so a class without a
/// reservation is never locked out of an idle pool. Desired reservations
/// that already fit are returned as they are.
fn clamp_reservations_to_capacity(desired: [u16; 4], capacity: u16) -> [u16; 4] {
    let desired_total: u64 = desired.iter().map(|&r| u64::from(r)).sum();
    if desired_total <= u64::from(capacity) {
        return desired;
    }
    // desired_total > capacity >= 0, so the division is defined and every
    // scaled value is below its desired value and within u16.
    let mut out = desired.map(|r| (u64::from(r) * u64::from(capacity) / desired_total) as u16);
    let reserved_total: u32 = out.iter().map(|&r| u32::from(r)).sum();
    if capacity > 0 && reserved_total >= u32::from(capacity) {
        // `max_by_key` returns the last maximum: in lane order that is the
        // highest class on a tie.
        if let Some(largest) = Class::ALL
            .iter()
            .copied()
            .max_by_key(|&class| out[class as usize])
        {
            out[largest as usize] = out[largest as usize].saturating_sub(1);
        }
    }
    out
}

/// Build the per-class queue with its configured fixed depth limit.
fn queue_for(settings: &SchedulerSettings, class: Class) -> Arc<dyn ClassQueue> {
    Arc::new(FifoClassQueue::new(
        settings.class_config(class).queue_size as usize,
    ))
}

/// RAII handle on one admitted request. Holding a permit keeps the slot
/// reserved; dropping it returns the slot, removes the handle from the
/// inflight registry, and notifies the dispatcher.
pub struct SchedulerPermit {
    scheduler: Arc<PriorityScheduler>,
    handle: Arc<InflightHandle>,
}

impl SchedulerPermit {
    /// Borrow the underlying inflight handle (the admission layer reads
    /// its preemptibility when it hands the permit to the response body).
    pub fn handle(&self) -> &Arc<InflightHandle> {
        &self.handle
    }

    /// Mark the first response byte. Called by the edge's body wrapper (the
    /// `Permit` below) on the first data frame. Returns `false` if the
    /// scheduler already won the preemption CAS — the body wrapper treats
    /// that as "preempted, end the stream." The TTFT value is `admitted_at`
    /// elapsed in millis, clamped by the handle to `[1, u64::MAX - 1]`.
    pub fn try_mark_first_byte(&self) -> bool {
        let now_ms = self.handle.admitted_at().elapsed().as_millis() as u64;
        self.handle.try_mark_first_byte(now_ms)
    }

    /// Clone of the scheduler-owned cancel token for this request. The
    /// admission middleware inserts this into request extensions so the
    /// handler can `select!` against it; the scheduler fires it on
    /// preemption.
    pub fn cancel_token(&self) -> CancellationToken {
        self.handle.cancel_token()
    }
}

impl std::fmt::Debug for SchedulerPermit {
    // Hand-rolled to avoid recursing into Arc<PriorityScheduler>, which
    // doesn't itself derive Debug (it contains atomics and a Notify).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerPermit")
            .field("class", &self.handle.class())
            .field("request_id", self.handle.request_id())
            .finish()
    }
}

impl Drop for SchedulerPermit {
    fn drop(&mut self) {
        self.scheduler.release_inflight(&self.handle);
    }
}

/// On the response body: the first data frame marks TTFT, or ends the body
/// without it when the scheduler preempted the request first; the drop
/// releases the slot.
impl Permit for SchedulerPermit {
    fn first_data_frame(&mut self) -> bool {
        self.try_mark_first_byte()
    }
}

#[cfg(test)]
mod tests;
