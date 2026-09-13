//! Priority-aware admission scheduler engine.
//!
//! Owns the [`SlotPool`], per-class [`ClassQueue`]s, and the inflight
//! registry. Construction validates that the configured reservations
//! fit under the live backend capacity; runtime admission lives in
//! follow-on commits.

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
use crate::worker::WorkerCapacity;

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
/// inflight registry. Admission / dispatch / capacity-watch land in
/// follow-on commits; this commit exposes the read-only constructor
/// plus an internal `acquire_inflight` so [`SchedulerPermit`] can be
/// exercised against a real slot pool from tests.
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
    /// priority-ordered clamp in `apply_new_capacity`, not by rejection.
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
    /// class). The admission middleware's fast path calls this directly;
    /// the queue / preempt paths layer on top in follow-on commits.
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
                // first data frame is truncated by `SchedulerGuardBody`. For
                // the victim to actually unwind, its handler must honor the
                // cancel token (wired into the long-running handlers in M4.5
                // / #1577); until that lands the flag must stay off.
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
        super::metrics::record_queue_wait(class, enqueued_at.elapsed());

        // Mark the waiter cancelled on any exit path so the queue's
        // `drop_cancelled_head` reaps it. Harmless if the waiter was
        // already popped (Admitted path) — the token has no other readers.
        waiter_cancel.cancel();
        // Kick the dispatcher so head GC runs promptly. Without this,
        // a `queue_size = 1` queue could reject the *next* admit as
        // QueueFull for up to one full periodic tick (5s on defaults)
        // even though the head waiter has already timed out / been
        // cancelled.
        self.release_notify.notify_one();
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
    /// sum exceeds the new capacity (a runtime dip below what the floors +
    /// shares want), the priority-ordered clamp keeps the highest classes
    /// whole and yields the lowest — runtime never rejects. On grow, also
    /// fires `release_notify` so the dispatcher re-evaluates queued waiters
    /// under the larger ceiling.
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
        // priority-ordered clamp if they no longer fit. Pure function of
        // (immutable baseline, new capacity), so it is idempotent and both
        // directional.
        let desired =
            desired_reservations(self.reserved_floor, &self.reserved_per_slot, new_capacity);
        let desired_total: u32 = desired.iter().map(|&r| u32::from(r)).sum();
        let new_reserved = if desired_total > u32::from(new_capacity) {
            warn!(
                desired_total_reserved = desired_total,
                new_capacity, "scheduler: reservations clamped to capacity (priority order)"
            );
            clamp_reservations_to_capacity(desired, new_capacity)
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

/// Clamp desired reservations to fit `capacity`, priority-ordered: fill
/// System → Bulk, each taking `min(desired, remaining)`. The highest classes
/// keep their seats under an extreme shrink and the lowest yield first. The
/// result always sums to `<= capacity`.
fn clamp_reservations_to_capacity(desired: [u16; 4], capacity: u16) -> [u16; 4] {
    let mut remaining = capacity;
    let mut out = [0u16; 4];
    for class in [
        Class::System,
        Class::Interactive,
        Class::Default,
        Class::Bulk,
    ] {
        let give = desired[class as usize].min(remaining);
        out[class as usize] = give;
        remaining -= give;
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
    /// Borrow the underlying inflight handle (for TTFT marking and
    /// preemption coordination in follow-on commits).
    pub fn handle(&self) -> &Arc<InflightHandle> {
        &self.handle
    }

    /// Mark the first response byte. Called by [`super::body::SchedulerGuardBody`]
    /// on the first data frame. Returns `false` if the scheduler already
    /// won the preemption CAS — the body wrapper treats that as
    /// "preempted, end the stream." The TTFT value is `admitted_at`
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

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use super::*;
    use crate::middleware::scheduler::{ClassConfig, PrioritySchedulerYaml};

    fn default_settings() -> SchedulerSettings {
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None).unwrap()
    }

    fn rid(s: &str) -> RequestId {
        RequestId(s.to_string())
    }

    #[test]
    fn test_new_succeeds_when_reservations_fit() {
        // Built-in defaults at capacity 256: desired Σ = 186 (System 32,
        // Interactive 128, Default ceil(0.10*256)=26) ≤ 256.
        let s = default_settings();
        assert!(PriorityScheduler::new(&s, 256).is_ok());
    }

    #[test]
    fn test_new_rejects_when_reservations_exceed_capacity() {
        // Capacity 100: desired is System 32, Interactive min(128,100)=100,
        // Default 10 — Σ 142 > 100, so construction rejects (caller falls
        // back to legacy).
        let s = default_settings();
        let result = PriorityScheduler::new(&s, 100);
        assert!(matches!(
            result,
            Err(SchedulerInitError::ReservationsExceedCapacity {
                reserved: 142,
                capacity: 100
            })
        ));
    }

    #[test]
    fn test_acquire_returns_permit_when_slot_available() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let permit = scheduler.acquire_inflight(Class::Default, rid("req-1"));
        assert!(permit.is_some());
    }

    #[test]
    fn test_acquire_returns_none_when_reservation_guard_blocks() {
        // Capacity 200, reservations 180 (System 32, Interactive 128, Default
        // ceil(0.10*200)=20). Bulk is the lowest class so it gets capacity -
        // 180 = 20 slots before the guard refuses further admissions.
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 200).unwrap();
        let mut held = Vec::new();
        for i in 0..20 {
            held.push(
                scheduler
                    .acquire_inflight(Class::Bulk, rid(&format!("req-{i}")))
                    .expect("under guard"),
            );
        }
        assert!(scheduler
            .acquire_inflight(Class::Bulk, rid("overflow"))
            .is_none());
    }

    #[tokio::test]
    async fn test_permit_drop_releases_slot_and_notifies_dispatcher() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let permit = scheduler
            .acquire_inflight(Class::Default, rid("req-1"))
            .expect("admitted");
        assert_eq!(scheduler.slot_pool.inflight(Class::Default), 1);

        let notified = scheduler.release_notify.notified();
        drop(permit);

        assert_eq!(scheduler.slot_pool.inflight(Class::Default), 0);
        tokio::time::timeout(Duration::from_millis(100), notified)
            .await
            .expect("release_notify fires on drop");
    }

    #[tokio::test]
    async fn test_registry_inserts_on_acquire_and_removes_on_drop() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let id = rid("req-1");
        let permit = scheduler
            .acquire_inflight(Class::Default, id.clone())
            .expect("admitted");
        assert!(scheduler.inflight_registry.read().contains_key(&id));
        drop(permit);
        assert!(!scheduler.inflight_registry.read().contains_key(&id));
    }

    // ── admit ────────────────────────────────────────────────────────

    /// Build settings with zero reservations on every class (so we can
    /// run admit tests against small capacities without tripping the
    /// reserved-vs-capacity guard) and an override on one class's
    /// queue_size + queue_timeout.
    fn settings_with(class: Class, queue_size: u32, queue_timeout_secs: u64) -> SchedulerSettings {
        use std::collections::HashMap as StdMap;

        let mut classes = StdMap::new();
        for c in Class::ALL {
            let mut cfg = ClassConfig::default_for(c);
            cfg.reserved_floor = 0;
            cfg.reserved_per_slot = 0.0;
            if c == class {
                cfg.queue_size = queue_size;
                cfg.queue_timeout_secs = queue_timeout_secs;
            }
            classes.insert(c, cfg);
        }
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: StdMap::new(),
        };
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap()
    }

    #[tokio::test]
    async fn test_admit_fast_path_when_slot_available() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let outcome = scheduler
            .admit(Class::Default, rid("req-1"), CancellationToken::new())
            .await;
        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
    }

    #[tokio::test]
    async fn test_admit_rejects_when_queue_full() {
        // Capacity 1 (slot held), Default queue_size=1 (one waiter pre-stuffed
        // directly into the queue): the next admit takes the slow path and
        // hits a full queue.
        let s = settings_with(Class::Default, 1, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let (queued_tx, _queued_rx) = oneshot::channel();
        let queued = Waiter::new(
            Class::Default,
            CancellationToken::new(),
            rid("queued"),
            queued_tx,
        );
        scheduler.class_queues[Class::Default as usize]
            .try_enqueue(queued)
            .expect("queue had room for one");

        let outcome = scheduler
            .admit(Class::Default, rid("w2"), CancellationToken::new())
            .await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueFull)
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn test_admit_rejects_on_queue_timeout() {
        // Capacity 1 forces enqueue; queue_timeout=1s; never release → QueueTimeout.
        let s = settings_with(Class::Default, 16, 1);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let admit_future = scheduler.admit(Class::Default, rid("w1"), CancellationToken::new());
        let outcome = admit_future.await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
        ));
    }

    #[tokio::test]
    async fn test_admit_timeout_kicks_dispatcher_to_reap_stale_head() {
        // After admit timeout marks the queued waiter cancelled, the
        // dispatcher must be nudged so drop_cancelled_head runs
        // promptly — otherwise a queue_size=1 queue would reject the
        // next admit as QueueFull until the next periodic tick.
        let s = settings_with(Class::Default, 1, 1); // queue_size=1, timeout=1s
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        scheduler.spawn_dispatcher(dummy_capacity_watch(1));
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        let outcome = scheduler
            .admit(Class::Default, rid("w1"), CancellationToken::new())
            .await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
        ));

        // Let the dispatcher's notify_one wakeup run drop_cancelled_head.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            scheduler.class_queues[Class::Default as usize].depth(),
            0,
            "dispatcher should reap the cancelled head promptly"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_admit_timeout_marks_waiter_for_gc() {
        // After QueueTimeout fires, the Waiter still sits in the FIFO
        // until something reaps it. The cancel token on the Waiter must
        // be fired so a future drop_cancelled_head call evicts it,
        // preventing false QueueFull on later admissions.
        let s = settings_with(Class::Default, 16, 1);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let outcome = scheduler
            .admit(Class::Default, rid("w1"), CancellationToken::new())
            .await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
        ));

        // The waiter is still in the queue (one entry), but cancelled.
        assert_eq!(scheduler.class_queues[Class::Default as usize].depth(), 1);
        scheduler.class_queues[Class::Default as usize].drop_cancelled_head();
        assert_eq!(
            scheduler.class_queues[Class::Default as usize].depth(),
            0,
            "drop_cancelled_head must reap the timed-out waiter"
        );
    }

    #[tokio::test]
    async fn test_admit_rejects_on_client_cancel() {
        // Pre-cancel the token before admit is called; the fast path fails
        // (slot held), the slow path enqueues, then the cancel arm of select!
        // fires immediately.
        let s = settings_with(Class::Default, 16, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let cancel = CancellationToken::new();
        cancel.cancel();
        let outcome = scheduler.admit(Class::Default, rid("w1"), cancel).await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::ClientCancelled)
        ));
    }

    // ── wake_next_waiter / dispatcher ───────────────────────────────

    fn enqueue_waiter(
        scheduler: &Arc<PriorityScheduler>,
        class: Class,
    ) -> oneshot::Receiver<SchedulerPermit> {
        let (tx, rx) = oneshot::channel();
        let waiter = Waiter::new(class, CancellationToken::new(), rid("queued"), tx);
        scheduler.class_queues[class as usize]
            .try_enqueue(waiter)
            .expect("queue had room");
        rx
    }

    #[tokio::test]
    async fn test_send_to_head_skips_waiters_with_closed_receivers() {
        // Stuff a queue with one waiter whose receiver is dropped (admit
        // has already timed out / been cancelled) followed by a live
        // waiter. wake_next_waiter should admit the live waiter without
        // touching the inflight registry for the dead one.
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        // Dead waiter — drop the receiver immediately.
        let (dead_tx, dead_rx) = oneshot::channel::<SchedulerPermit>();
        drop(dead_rx);
        let dead = Waiter::new(
            Class::Default,
            CancellationToken::new(),
            rid("dead"),
            dead_tx,
        );
        scheduler.class_queues[Class::Default as usize]
            .try_enqueue(dead)
            .unwrap();

        // Live waiter.
        let (live_tx, mut live_rx) = oneshot::channel::<SchedulerPermit>();
        let live = Waiter::new(
            Class::Default,
            CancellationToken::new(),
            rid("live"),
            live_tx,
        );
        scheduler.class_queues[Class::Default as usize]
            .try_enqueue(live)
            .unwrap();

        // Release the slot.
        drop(_held);
        assert!(scheduler.wake_next_waiter(), "live waiter admitted");

        let permit = live_rx.try_recv().expect("live permit delivered");
        assert_eq!(permit.handle().request_id().0, "live");
        // Registry should hold only the live request — never the dead one.
        let registry = scheduler.inflight_registry.read();
        assert!(registry.contains_key(&rid("live")));
        assert!(!registry.contains_key(&rid("dead")));
    }

    #[test]
    fn test_wake_next_waiter_returns_false_when_no_slot_available() {
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();
        let _rx = enqueue_waiter(&scheduler, Class::Default);
        assert!(!scheduler.wake_next_waiter(), "no slot to give");
    }

    #[tokio::test]
    async fn test_wake_next_waiter_admits_queued_waiter() {
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();
        let mut rx = enqueue_waiter(&scheduler, Class::Default);
        drop(held);
        assert!(scheduler.wake_next_waiter());
        let permit = rx.try_recv().expect("permit delivered");
        assert_eq!(permit.handle().class(), Class::Default);
    }

    #[tokio::test]
    async fn test_wake_next_waiter_honors_priority_order() {
        // Interactive beats Bulk when both are queued and a slot frees.
        let s = settings_with(Class::Bulk, 8, 60); // Bulk queue=8 (interactive defaults too)
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        let mut bulk_rx = enqueue_waiter(&scheduler, Class::Bulk);
        let mut interactive_rx = enqueue_waiter(&scheduler, Class::Interactive);

        drop(held);
        assert!(scheduler.wake_next_waiter());
        // Interactive should be served first; bulk still waiting.
        assert!(interactive_rx.try_recv().is_ok());
        assert!(bulk_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_dispatcher_periodic_tick_promotes_stale_waiter_without_release() {
        // No slot release ever fires. The dispatcher's periodic tick
        // must wake on its own, observe that the Bulk head has aged
        // past its starvation_threshold, and admit via the override.
        use std::collections::HashMap as StdMap;
        let mut classes = StdMap::new();
        for c in Class::ALL {
            let mut cfg = ClassConfig::default_for(c);
            cfg.reserved_floor = 0;
            cfg.reserved_per_slot = 0.0;
            // Tight thresholds so the test runs fast.
            cfg.starvation_threshold_secs = 1;
            if c == Class::Bulk {
                cfg.queue_size = 4;
            }
            // Interactive reserves the entire capacity, locking Bulk out
            // of the normal-priority admission path.
            if c == Class::Interactive {
                cfg.reserved_floor = 1;
            }
            classes.insert(c, cfg);
        }
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: StdMap::new(),
        };
        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
        let scheduler = PriorityScheduler::new(&settings, 1).unwrap();
        scheduler.spawn_dispatcher(dummy_capacity_watch(1));

        // Admit a Bulk request. No slot release ever fires; the only way
        // for it to be admitted is the dispatcher's periodic tick + the
        // starvation override.
        let outcome = tokio::time::timeout(
            Duration::from_secs(3),
            scheduler.admit(Class::Bulk, rid("bulk"), CancellationToken::new()),
        )
        .await
        .expect("admit completes within timeout");
        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
    }

    #[tokio::test]
    async fn test_starvation_override_promotes_stale_bulk_head() {
        // Bulk starvation_threshold=0s (anything older than 0 wins), with
        // capacity entirely reserved by Interactive — without the
        // override Bulk could never advance.
        use std::collections::HashMap as StdMap;
        let mut classes = StdMap::new();
        for c in Class::ALL {
            let mut cfg = ClassConfig::default_for(c);
            cfg.reserved_floor = 0;
            cfg.reserved_per_slot = 0.0;
            // Set tiny starvation threshold for Bulk so an immediate
            // enqueue is "stale" by the time wake fires.
            if c == Class::Bulk {
                cfg.starvation_threshold_secs = 1;
                cfg.queue_size = 4;
            }
            // Interactive holds the full capacity in reservation.
            if c == Class::Interactive {
                cfg.reserved_floor = 1;
            }
            classes.insert(c, cfg);
        }
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: StdMap::new(),
        };
        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();

        let scheduler = PriorityScheduler::new(&settings, 1).unwrap();
        let mut bulk_rx = enqueue_waiter(&scheduler, Class::Bulk);

        // Sleep past the starvation threshold so head_age > threshold.
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // No regular path admits Bulk under the Interactive reservation,
        // but the starvation override should.
        assert!(scheduler.wake_next_waiter(), "starvation override fires");
        assert!(bulk_rx.try_recv().is_ok());
    }

    fn dummy_capacity_watch(initial: u16) -> watch::Receiver<u16> {
        // Construct a watch receiver whose sender is intentionally kept
        // alive for the test's duration. Leaking is fine — these tests
        // run in isolation.
        let (tx, rx) = watch::channel(initial);
        std::mem::forget(tx);
        rx
    }

    #[tokio::test]
    async fn test_spawn_dispatcher_admits_queued_waiter_on_release() {
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        scheduler.spawn_dispatcher(dummy_capacity_watch(1));

        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        // Kick off an admit that must go to the slow path.
        let scheduler_for_admit = Arc::clone(&scheduler);
        let admit_future = async move {
            scheduler_for_admit
                .admit(Class::Default, rid("queued"), CancellationToken::new())
                .await
        };

        // Race: release the slot, then await admit. The dispatcher
        // should observe the release and admit the queued waiter.
        let (admit_outcome, ()) = tokio::join!(admit_future, async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(held);
        });

        assert!(matches!(admit_outcome, AdmitOutcome::Admitted(_)));
    }

    // ── apply_new_capacity / capacity watch ─────────────────────────

    #[test]
    fn test_apply_new_capacity_grow_fires_release_notify() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        // Notify already has whatever permits prior tests fired; reset by
        // taking a fresh notified() before the apply.
        let notified = scheduler.release_notify.notified();
        scheduler.apply_new_capacity(512);
        assert_eq!(scheduler.slot_pool.capacity(), 512);
        // Grow path must have signaled the dispatcher.
        tokio::pin!(notified);
        let polled = futures::FutureExt::now_or_never(notified);
        assert!(polled.is_some(), "release_notify fires on capacity grow");
    }

    #[test]
    fn test_apply_new_capacity_shrink_clamps_reservations_priority_ordered() {
        // Built-in defaults at capacity 80: desired is System 32, Interactive
        // 128 (floor), Default ceil(0.10*80)=8 — Σ 168 > 80. The
        // priority-ordered clamp keeps System whole (32), gives Interactive
        // the remainder (48), and yields Default and Bulk.
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        scheduler.apply_new_capacity(80);
        assert_eq!(scheduler.slot_pool.capacity(), 80);
        let new_total: u32 = Class::ALL
            .iter()
            .map(|c| u32::from(scheduler.slot_pool.reserved(*c)))
            .sum();
        assert!(
            new_total <= 80,
            "clamped reservations ({new_total}) must fit"
        );
        assert_eq!(scheduler.slot_pool.reserved(Class::System), 32);
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 48);
        assert_eq!(scheduler.slot_pool.reserved(Class::Default), 0);
        assert_eq!(scheduler.slot_pool.reserved(Class::Bulk), 0);
    }

    #[test]
    fn test_apply_new_capacity_restores_reservations_after_recovery() {
        // Reservations are recomputed from the floors + shares on every
        // capacity change, so a shrink-then-recover fully restores them (no
        // one-way degrade), and the share tracks a growing fleet.
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        // cap 256: System 32, Interactive max(128, 64)=128, Default ceil(25.6)=26.
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 128);
        assert_eq!(scheduler.slot_pool.reserved(Class::Default), 26);

        // Shrink to 80: desired Σ 168 > 80 → priority clamp (System 32, the
        // remaining 48 to Interactive, Default/Bulk yield).
        scheduler.apply_new_capacity(80);
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 48);
        assert_eq!(scheduler.slot_pool.reserved(Class::Default), 0);

        // Recover to 256 → fully restored, not stuck at the clamped values.
        scheduler.apply_new_capacity(256);
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 128);
        assert_eq!(scheduler.slot_pool.reserved(Class::Default), 26);

        // Grow past Interactive's floor: at 800 the 0.25 share wins (200 > 128)
        // and Default's 0.10 share is 80 — reservations track the larger fleet.
        scheduler.apply_new_capacity(800);
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 200);
        assert_eq!(scheduler.slot_pool.reserved(Class::Default), 80);
    }

    #[test]
    fn test_apply_new_capacity_no_op_when_value_unchanged() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let before = scheduler.slot_pool.reserved(Class::Interactive);
        scheduler.apply_new_capacity(256);
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), before);
    }

    // ── reservation math helpers ──────────────────────────────────────

    #[test]
    fn test_desired_reservations_floor_then_share() {
        // Indexed by `class as usize`: [Bulk, Default, Interactive, System].
        let floor = [0u16, 0, 128, 32];
        let per_slot = [0.0f64, 0.10, 0.25, 0.0];
        // cap 256: floor wins for Interactive (128 > 64); Default share = 26.
        let d = desired_reservations(floor, &per_slot, 256);
        assert_eq!(d[Class::System as usize], 32);
        assert_eq!(d[Class::Interactive as usize], 128);
        assert_eq!(d[Class::Default as usize], 26);
        assert_eq!(d[Class::Bulk as usize], 0);
        // cap 800: Interactive share (200) overtakes its floor; Default = 80.
        let d = desired_reservations(floor, &per_slot, 800);
        assert_eq!(d[Class::Interactive as usize], 200);
        assert_eq!(d[Class::Default as usize], 80);
        // Each class is capped at capacity (floor 128 capped to 10).
        let d = desired_reservations(floor, &per_slot, 10);
        assert_eq!(d[Class::Interactive as usize], 10);
    }

    #[test]
    fn test_clamp_reservations_priority_ordered() {
        // [Bulk, Default, Interactive, System], Σ 185 > 100.
        let out = clamp_reservations_to_capacity([5, 20, 128, 32], 100);
        // System keeps its 32, Interactive takes the remaining 68, rest yield.
        assert_eq!(out[Class::System as usize], 32);
        assert_eq!(out[Class::Interactive as usize], 68);
        assert_eq!(out[Class::Default as usize], 0);
        assert_eq!(out[Class::Bulk as usize], 0);
        assert_eq!(out.iter().map(|&r| u32::from(r)).sum::<u32>(), 100);
        // Fits already: every class keeps its full desired value.
        assert_eq!(
            clamp_reservations_to_capacity([1, 2, 3, 4], 100),
            [1, 2, 3, 4]
        );
    }

    #[tokio::test]
    async fn test_dispatcher_keeps_serving_after_capacity_watch_sender_drops() {
        // After the WorkerCapacity sender is dropped, the dispatcher must
        // continue handling release events from release_notify. The earlier
        // implementation hot-looped on the closed watch arm; this test
        // exercises the post-drop path end-to-end.
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let (capacity_tx, capacity_rx) = watch::channel(1u16);
        scheduler.spawn_dispatcher(capacity_rx);

        drop(capacity_tx); // close the watch — the failing branch trigger.
                           // Yield so the dispatcher observes the drop and disables that arm.
        tokio::time::sleep(Duration::from_millis(10)).await;

        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();
        let scheduler_for_admit = Arc::clone(&scheduler);
        let admit_future = async move {
            scheduler_for_admit
                .admit(Class::Default, rid("queued"), CancellationToken::new())
                .await
        };
        let (outcome, ()) = tokio::join!(admit_future, async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(held);
        });
        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
    }

    #[tokio::test]
    async fn test_capacity_watch_grow_drains_queue() {
        // Capacity 1 (slot held), queue has one waiter. Grow capacity to
        // 2 via the watch — dispatcher should drain the queued waiter.
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let (tx, rx) = watch::channel(1u16);
        scheduler.spawn_dispatcher(rx);

        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        let scheduler_for_admit = Arc::clone(&scheduler);
        let admit_future = async move {
            scheduler_for_admit
                .admit(Class::Default, rid("queued"), CancellationToken::new())
                .await
        };

        let (outcome, ()) = tokio::join!(admit_future, async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tx.send(2).unwrap();
        });

        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
        assert_eq!(scheduler.slot_pool.capacity(), 2);
    }

    #[tokio::test]
    async fn test_permit_keeps_scheduler_alive_via_arc() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let weak = Arc::downgrade(&scheduler);
        let permit = scheduler
            .acquire_inflight(Class::Default, rid("req-1"))
            .expect("admitted");
        drop(scheduler);
        // Permit holds a strong ref, so weak still upgrades.
        assert!(weak.upgrade().is_some());
        drop(permit);
        // All strong refs gone now.
        assert!(weak.upgrade().is_none());
    }

    // ── M3: preemption ────────────────────────────────────────────────

    /// All classes reserved=0 (so acquire is purely capacity-bound),
    /// short queue timeout (so declined-preempt enqueues resolve fast),
    /// can_preempt left at defaults (System/Interactive true, others false).
    fn preempt_settings() -> SchedulerSettings {
        use std::collections::HashMap as StdMap;
        let mut classes = StdMap::new();
        for c in Class::ALL {
            let mut cfg = ClassConfig::default_for(c);
            cfg.reserved_floor = 0;
            cfg.reserved_per_slot = 0.0;
            cfg.queue_size = 8;
            cfg.queue_timeout_secs = 1;
            classes.insert(c, cfg);
        }
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: StdMap::new(),
        };
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap()
    }

    #[test]
    fn test_find_victim_empty_registry_returns_none() {
        let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
        assert!(sched.find_preemption_victim(Class::Interactive).is_none());
    }

    #[test]
    fn test_find_victim_returns_lower_class_pre_ttft() {
        let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
        let _bulk = sched.acquire_inflight(Class::Bulk, rid("b")).unwrap();
        let v = sched
            .find_preemption_victim(Class::Interactive)
            .expect("victim");
        assert_eq!(v.class(), Class::Bulk);
    }

    #[test]
    fn test_find_victim_prefers_lowest_class() {
        let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
        let _def = sched.acquire_inflight(Class::Default, rid("d")).unwrap();
        let _bulk = sched.acquire_inflight(Class::Bulk, rid("b")).unwrap();
        // Candidate Interactive: Default and Bulk both qualify; Bulk (lowest) wins.
        let v = sched
            .find_preemption_victim(Class::Interactive)
            .expect("victim");
        assert_eq!(v.class(), Class::Bulk);
    }

    #[test]
    fn test_find_victim_skips_post_ttft() {
        let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
        let _def = sched.acquire_inflight(Class::Default, rid("d")).unwrap();
        let bulk = sched.acquire_inflight(Class::Bulk, rid("b")).unwrap();
        bulk.handle().try_mark_first_byte(5); // Bulk no longer preemptible
        let v = sched
            .find_preemption_victim(Class::Interactive)
            .expect("victim");
        assert_eq!(
            v.class(),
            Class::Default,
            "Bulk past TTFT is skipped; Default chosen"
        );
    }

    #[test]
    fn test_find_victim_none_for_lowest_class() {
        let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
        let _def = sched.acquire_inflight(Class::Default, rid("d")).unwrap();
        // Bulk is the lowest class — nothing is strictly lower to preempt.
        assert!(sched.find_preemption_victim(Class::Bulk).is_none());
    }

    #[test]
    fn test_permit_try_mark_first_byte_locks_out_preempt() {
        let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
        let permit = sched.acquire_inflight(Class::Default, rid("x")).unwrap();
        assert!(permit.try_mark_first_byte());
        assert!(
            !permit.handle().try_mark_preempted(),
            "preempt must lose after TTFT is marked"
        );
    }

    #[test]
    fn test_permit_cancel_token_reflects_handle_cancel() {
        let sched = PriorityScheduler::new(&preempt_settings(), 8).unwrap();
        let permit = sched.acquire_inflight(Class::Bulk, rid("x")).unwrap();
        let tok = permit.cancel_token();
        assert!(!tok.is_cancelled());
        assert!(permit.handle().try_mark_preempted());
        permit.handle().cancel();
        assert!(tok.is_cancelled());
    }

    #[tokio::test]
    async fn test_preempt_admits_by_cancelling_lower_class() {
        // Capacity 1, full with a pre-TTFT Bulk request. An Interactive
        // admission (can_preempt) cancels it; a watcher drops the victim's
        // permit on cancel (mimicking the handler unwinding), freeing the
        // slot for the Interactive waiter.
        let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
        let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
        let victim_cancel = victim.cancel_token();

        let sched_admit = Arc::clone(&sched);
        let admit_fut = async move {
            sched_admit
                .admit(Class::Interactive, rid("vip"), CancellationToken::new())
                .await
        };
        let dropper = async move {
            victim_cancel.cancelled().await;
            drop(victim); // handler unwinds → slot frees
        };
        let (outcome, ()) = tokio::join!(admit_fut, dropper);
        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
    }

    #[tokio::test]
    async fn test_preempt_skipped_when_caller_already_cancelled() {
        // A preempt-capable admission whose own client has already
        // disconnected must not cancel a lower-class victim: it returns
        // ClientCancelled and leaves the victim untouched.
        let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
        let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
        let victim_cancel = victim.cancel_token();

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let outcome = sched.admit(Class::Interactive, rid("vip"), cancelled).await;

        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::ClientCancelled)
        ));
        assert!(
            !victim_cancel.is_cancelled(),
            "victim must not be preempted"
        );
        assert_eq!(sched.inflight_for_test(Class::Bulk), 1);
    }

    #[test]
    fn test_class_pressure_takes_worse_of_queue_and_slot() {
        // preempt_settings reserves 0 for every class, so higher_reserved=0
        // and slot pressure is inflight/capacity.
        let sched = PriorityScheduler::new(&preempt_settings(), 100).unwrap();
        // Queue pressure (8/10) dominates slot pressure (10/100).
        assert!((sched.class_pressure(Class::Default, 10, 8, 10, 100) - 0.8).abs() < 1e-9);
        // Slot pressure (50/100) dominates queue pressure (1/10).
        assert!((sched.class_pressure(Class::Default, 50, 1, 10, 100) - 0.5).abs() < 1e-9);
        // Clamped to 1.0 when oversubscribed.
        assert!((sched.class_pressure(Class::Default, 200, 100, 10, 100) - 1.0).abs() < 1e-9);
        // No queue limit and no inflight → zero, no div-by-zero.
        assert_eq!(sched.class_pressure(Class::Bulk, 0, 0, 0, 100), 0.0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_preempt_declined_when_only_victim_past_ttft() {
        // The single lower-class inflight has already emitted its first
        // byte, so it is not a valid victim. The Interactive admission must
        // NOT cancel it; it falls through to enqueue and times out.
        let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
        let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
        victim.handle().try_mark_first_byte(5);
        let victim_cancel = victim.cancel_token();

        let outcome = sched
            .admit(Class::Interactive, rid("vip"), CancellationToken::new())
            .await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
        ));
        assert!(
            !victim_cancel.is_cancelled(),
            "a post-TTFT request must never be cancelled"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_non_preempting_class_does_not_cancel() {
        // Default has can_preempt=false; even with a Bulk victim available
        // it must not attempt preemption.
        let sched = PriorityScheduler::new(&preempt_settings(), 1).unwrap();
        let victim = sched.acquire_inflight(Class::Bulk, rid("victim")).unwrap();
        let victim_cancel = victim.cancel_token();

        let outcome = sched
            .admit(Class::Default, rid("plain"), CancellationToken::new())
            .await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
        ));
        assert!(
            !victim_cancel.is_cancelled(),
            "can_preempt=false class must not cancel anyone"
        );
    }
}
