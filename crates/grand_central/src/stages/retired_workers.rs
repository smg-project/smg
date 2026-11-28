//! A stage's retired workers: the ids a lifecycle pruned a short while ago,
//! kept out of a listing that a view older than the pruning would bring back.
//!
//! A stage that learns its members from a decision's view holds one: the ring
//! of `consistent_hashing` and the scorer of `bucket`.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;

use crate::engine::view::WorkerId;

/// How long a pruned id stays fenced against relearning from a view older
/// than the pruning: longer than any gap between a request's capture of the
/// pool and its decision (the stage holds no await between the two), shorter
/// than a worker's re-registration, after which a new handle at a reused
/// address may carry the same id and joins from its view as any other. The
/// feed naming the id (added, replacing, in a snapshot) lifts it at once.
pub const RETIRED_WORKER_TTL: Duration = Duration::from_secs(10);

/// The ids a lifecycle pruned within [`RETIRED_WORKER_TTL`], fenced against
/// relearning from a view older than the pruning. A stage that learns its
/// members from a decision's view holds one. It keeps pruned workers out of a
/// listing and is no kin of [`crate::service::elector::Fence`], which holds the
/// term a process leads under.
#[derive(Debug, Default)]
pub(crate) struct RetiredWorkers {
    /// The retirements within [`RETIRED_WORKER_TTL`], oldest first. Those of an
    /// id the feed named since stay until they lapse and lift nothing.
    retired: VecDeque<(WorkerId, Instant)>,
    /// Each fenced id with its last retirement, the one whose lapse lifts
    /// the fence.
    fenced: FxHashMap<WorkerId, Instant>,
}

impl RetiredWorkers {
    /// Whether no id holds a fence.
    pub(crate) fn is_empty(&self) -> bool {
        self.fenced.is_empty()
    }

    /// Whether `id` holds a fence the last [`RetiredWorkers::expire`] left
    /// standing.
    pub(crate) fn holds(&self, id: WorkerId) -> bool {
        self.fenced.contains_key(&id)
    }

    /// Whether `id` was pruned within [`RETIRED_WORKER_TTL`] of `now` and is
    /// fenced against relearning.
    pub(crate) fn is_retired(&self, id: WorkerId, now: Instant) -> bool {
        self.fenced
            .get(&id)
            .is_some_and(|&at| now.saturating_duration_since(at) < RETIRED_WORKER_TTL)
    }

    /// Lift the fence on `id`: the feed named it as a member again.
    pub(crate) fn admit(&mut self, id: WorkerId) {
        // Its retirements stay in the list: none is its last any more, so
        // their lapses lift nothing.
        self.fenced.remove(&id);
    }

    /// Forget the retirements older than [`RETIRED_WORKER_TTL`] at `now`.
    pub(crate) fn expire(&mut self, now: Instant) {
        while let Some(&(old, at)) = self.retired.front() {
            if now.saturating_duration_since(at) < RETIRED_WORKER_TTL {
                break;
            }
            self.retired.pop_front();
            // The fence lifts with the id's last retirement. The map says
            // whether this entry is it: no pass over the list for a lapse.
            if self.fenced.get(&old) == Some(&at) {
                self.fenced.remove(&old);
            }
        }
    }

    /// Fence `id` for [`RETIRED_WORKER_TTL`] from `now`: a lifecycle pruned it.
    pub(crate) fn retire(&mut self, id: WorkerId, now: Instant) {
        self.expire(now);
        let last = self.fenced.entry(id).or_insert(now);
        *last = (*last).max(now);
        // Two callers can read their clocks in one order and get here in
        // the other. The list stays oldest first all the same, so a
        // retirement that lapsed never waits behind a later one.
        let behind = self.retired.partition_point(|&(_, at)| at <= now);
        self.retired.insert(behind, (id, now));
    }
}
