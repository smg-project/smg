//! The gateway as the policy controller's host in process: one decision an
//! attempt, from the ask to the request's end.
//!
//! An attempt has two halves. The ask ([`Host::ask`]) states what the
//! attempt knows of its request and gets a decision through the
//! controller's handle. The commit ([`Host::commit`]) walks the decision's
//! targets under the gateway's own locks and takes the first it can send
//! to. [`Host::place`] is their loop for the regular shapes. The store
//! builds the candidates: the gateway sends none and keeps no view of the
//! rows.
//!
//! The ask is the one function that may wait. In process the door decides
//! on the calling thread and the ask is ready at once. A controller at an
//! address is asked over its session, behind the same call.
//!
//! A router speaks to [`Placed`] and [`Held`], never to the controller's
//! handle. [`Placed`] is the attempt's worker with its decision's lease:
//! the router says the dispatch at the send, then the request's end, or a
//! release before any send. One dropped with a report owed says it, as the
//! handle does. [`Held`] carries the worker crate's load guard beside it,
//! so one value travels with a response to its end.
//!
//! The gateway's own check of a target is what it alone knows: the worker's
//! status as its probes read it, and its circuit breaker. The overload and
//! stall vetoes are the controller's, read from its rows.

use std::{
    fmt,
    future::{ready, Ready},
    sync::Arc,
    time::Instant,
};

use rand::RngExt;
use smg_policy_controller::{
    decision::{Decision, ShedReason, Targets},
    door::{Door, Refused, Request},
    handle::Handle,
    lease::{Outcome, ReleaseReason, Side, TransportClass},
    store::{GatewayId, MemoryStore, WorkersStore},
    view::{Constraints, RequestFacts, WorkerId},
};
use smg_worker::identity::basic::{Worker, WorkerLoadGuard};

use crate::routers::common::snapshot::WorkerRegistry;

/// The gateway of this process, as the controller's store names it. One
/// gateway decides in process, so the id tells no two apart.
const GATEWAY: GatewayId = GatewayId(1);

/// The asks of one attempt: the first, and two more that name the targets
/// the gateway could not send to.
const ASKS: usize = 3;

/// What an attempt states of its request.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ask<'a> {
    /// The request's id.
    pub id: &'a [u8],
    /// The model the request is for, as the router resolved it.
    pub model: &'a str,
    /// Whether the answer is streamed.
    pub streaming: bool,
    /// The rows of the workers the request's failed attempts were sent to.
    pub tried: &'a [WorkerId],
}

impl<'a> Ask<'a> {
    /// The request as the controller reads it, with the targets the gateway
    /// `missed` at this attempt's earlier commits. The one place a fact of
    /// a request is stated: a later step adds its fact here.
    ///
    /// Each call draws the decision's seed anew. The seed is the one
    /// randomness a stage gets: under a constant seed a policy that draws
    /// picks the same worker of an unchanged pool every time.
    fn request(&self, missed: &'a [WorkerId]) -> Request<'a> {
        Request {
            id: self.id,
            facts: RequestFacts {
                model: self.model,
                streaming: self.streaming,
                attempt: u8::try_from(self.tried.len()).unwrap_or(u8::MAX),
                tried: self.tried,
                seed: rand::rng().random(),
                ..RequestFacts::default()
            },
            constraints: Constraints {
                excluded: missed,
                ..Constraints::default()
            },
        }
    }
}

/// Why an attempt has no worker.
#[derive(Debug, PartialEq)]
pub(crate) enum Unplaced {
    /// The controller shed the request.
    Shed {
        /// Why.
        reason: ShedReason,
        /// The stage that left no candidate.
        emptied_by: &'static str,
        /// The seconds after which a retry may find room.
        retry_after_secs: u16,
    },
    /// The controller queued the request. No stage of this build issues a
    /// ticket, and a host that meets one refuses the request.
    Queued,
    /// The door made no decision.
    Refused(Refused),
    /// The gateway could send to no target of its three asks.
    NoTargetHolds,
}

/// A decision that picked: its targets, best first, and its lease, not yet
/// committed.
pub(crate) struct Asked {
    targets: Targets,
    handle: Handle<Arc<Door>>,
}

/// The policy controller's host in this process: the door over the
/// controller's store, and the registry the gateway's workers stand in.
pub(crate) struct Host {
    door: Arc<Door>,
    registry: Arc<WorkerRegistry>,
}

impl Host {
    /// The host over the controller's `store`, for the workers of
    /// `registry`.
    pub(crate) fn new(store: Arc<MemoryStore>, registry: Arc<WorkerRegistry>) -> Self {
        Self {
            door: Arc::new(Door::new(store)),
            registry,
        }
    }

    /// The gateway's worker of a target: the one function that resolves a
    /// row of the store to a worker. `None` where the row is gone, or the
    /// registry holds no worker at its key.
    ///
    /// This body reads the row whole and looks its key up as a URL. A URL is
    /// not one life of a worker, and the read clones the row: the host
    /// resolves by the row's id, through the registry's projection of the
    /// store, before a router calls it.
    fn worker(&self, id: WorkerId) -> Option<Arc<dyn Worker>> {
        let row = WorkersStore::get(&**self.door.store(), id)?;
        self.registry.get_by_url(&row.facts.key)
    }

    /// The ask: one decision for `ask` at `now` through the controller's
    /// handle, naming the targets the gateway `missed` at this attempt's
    /// earlier commits.
    ///
    /// The one function of the host that may wait. The door of this process
    /// decides on the calling thread, so the answer is ready.
    pub(crate) fn ask(
        &self,
        ask: &Ask<'_>,
        missed: &[WorkerId],
        now: Instant,
    ) -> Ready<Result<Asked, Unplaced>> {
        let decided = Handle::decide(Arc::clone(&self.door), GATEWAY, &ask.request(missed), now);
        ready(decided.map_err(Unplaced::Refused).and_then(|handle| {
            let decision = *handle.decision();
            match decision {
                Decision::Pick(pick) => Ok(Asked {
                    targets: pick.targets,
                    handle,
                }),
                Decision::Shed(shed) => Err(Unplaced::Shed {
                    reason: shed.reason,
                    emptied_by: shed.emptied_by,
                    retry_after_secs: shed.retry_after_secs,
                }),
                Decision::Queue(_) => Err(Unplaced::Queued),
            }
        }))
    }

    /// The commit: walks the targets of `asked`, best first, and takes the
    /// first the gateway can send to. It never waits, so a caller may hold
    /// a lock of its own across it.
    ///
    /// Where no target holds, the decision's lease is given up by name at
    /// `now` and its targets join `missed`: the next ask names them.
    pub(crate) fn commit(
        &self,
        asked: Asked,
        missed: &mut Vec<WorkerId>,
        now: Instant,
    ) -> Option<Placed> {
        let Asked {
            targets,
            mut handle,
        } = asked;
        for (index, target) in targets.iter().enumerate() {
            let sendable = self
                .worker(target.id)
                .filter(|worker| worker.is_healthy() && worker.circuit_breaker_can_execute());
            if let Some(worker) = sendable {
                handle.take(index);
                return Some(Placed {
                    worker,
                    id: target.id,
                    handle,
                });
            }
        }
        handle.released(ReleaseReason::CommitFailed, now);
        missed.extend(targets.iter().map(|target| target.id));
        None
    }

    /// A regular attempt's worker at `now`: the ask and the commit, three
    /// asks at most. After the third the attempt has no worker.
    pub(crate) async fn place(&self, ask: &Ask<'_>, now: Instant) -> Result<Placed, Unplaced> {
        let mut missed: Vec<WorkerId> = Vec::new();
        for _ in 0..ASKS {
            let asked = self.ask(ask, &missed, now).await?;
            if let Some(placed) = self.commit(asked, &mut missed, now) {
                return Ok(placed);
            }
        }
        Err(Unplaced::NoTargetHolds)
    }
}

/// A request's end, as its router read it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct End {
    /// The status the worker answered with, 0 where none was read.
    pub status: u16,
    /// What became of the transport.
    pub class: TransportClass,
    /// The time to the first token, in microseconds.
    pub ttft_micros: u32,
    /// The tokens the worker read.
    pub tokens_in: u32,
    /// The tokens the worker wrote.
    pub tokens_out: u32,
    /// The time from the send to the end, in microseconds.
    pub duration_micros: u32,
}

/// An attempt's worker, with its decision's lease held to the request's
/// end.
pub(crate) struct Placed {
    worker: Arc<dyn Worker>,
    id: WorkerId,
    handle: Handle<Arc<Door>>,
}

impl Placed {
    /// The worker the request goes to.
    pub(crate) fn worker(&self) -> &Arc<dyn Worker> {
        &self.worker
    }

    /// The store's row of that worker: what a retry names as tried.
    pub(crate) fn id(&self) -> WorkerId {
        self.id
    }

    /// The request was sent to the worker at `now`. Said once.
    pub(crate) fn dispatched(&mut self, now: Instant) {
        self.handle.dispatched(now);
    }

    /// The request ended with `end`, as the router read it at `now`. Said
    /// once.
    pub(crate) fn completed(&mut self, end: End, now: Instant) {
        let outcome = Outcome {
            side: Side::Single,
            served: self.id,
            status: end.status,
            class: end.class,
            ttft_micros: end.ttft_micros,
            tokens_in: end.tokens_in,
            tokens_out: end.tokens_out,
            duration_micros: end.duration_micros,
        };
        self.handle.completed(&outcome, now);
    }

    /// The request is given up under `reason` at `now`: before a send the
    /// lease is released, and after one it completes as cancelled.
    pub(crate) fn released(&mut self, reason: ReleaseReason, now: Instant) {
        self.handle.released(reason, now);
    }
}

impl fmt::Debug for Placed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Placed")
            .field("worker", &self.worker.url())
            .field("id", &self.id)
            .field("handle", &self.handle)
            .finish()
    }
}

/// What a response carries of its worker to its end: the attempt's
/// placement, and the worker crate's load guard beside it. The guard gains
/// no field: the worker crate does not name the policy crate.
pub(crate) struct Held {
    /// Dropped first: a report still owed is said while the load stands.
    pub placed: Placed,
    /// The worker's load, given back at the drop.
    #[expect(dead_code, reason = "held for its drop alone")]
    pub guard: WorkerLoadGuard,
}

#[cfg(test)]
mod tests;
