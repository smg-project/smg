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
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::{
    http::{header::RETRY_AFTER, HeaderValue},
    response::Response,
};
use rand::RngExt;
use smg_policy_controller::{
    decision::{Decision, Lifted, ShedReason, Target, Targets, Trace},
    door::{Door, Refused, Request},
    handle::Handle,
    lease::{Outcome, ReleaseReason, Side, TransportClass},
    store::{GatewayId, MemoryStore, WorkersStore},
    view::{Constraints, HashKind, RequestFacts, WorkerId},
};
use smg_worker::identity::basic::{Worker, WorkerLoadGuard};

use crate::{
    middleware::SHED_RETRY_AFTER_SECS,
    observability::{
        cache_trace,
        metrics::{Metrics, PolicyDecision},
    },
    routers::{
        common::{
            overload::WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE,
            placement::no_available_workers_because, retry::mark_non_retryable,
            snapshot::WorkerRegistry,
        },
        error,
    },
};

mod hashes;

pub(crate) use hashes::Hashes;
#[cfg_attr(
    not(test),
    expect(unused_imports, reason = "the prompt is part of the host's interface")
)]
pub(crate) use hashes::Prompt;

/// The gateway of this process, as the controller's store names it: the
/// crate's own name for the one gateway that decides in its own process.
const GATEWAY: GatewayId = GatewayId::LOCAL;

/// The asks of one attempt: the first, and two more that name the targets
/// the gateway could not send to.
const ASKS: usize = 3;

/// The crate's word, in a shed's `emptied_by`, for a pool no stage emptied:
/// it held no candidate before any stage ran.
const NO_STAGE: &str = "none";

fn outcome(decision: &Decision) -> &'static str {
    match decision {
        Decision::Pick(_) => "pick",
        Decision::Shed(_) => "shed",
        Decision::Queue(_) => "queue",
    }
}

/// The controller's sheds made ahead of every profile, each by its reason
/// and the stage it names: a request no binding matches, an affinity column
/// that does not fit the rows, rows not written for a profile's scope, a
/// gateway that shares a wire with no row of the pool, and a subset hint of
/// none.
///
/// A copy of the policy crate's own words, kept until a decision says which
/// profile made it. It mirrors the sheds `Engine::decide` and
/// `Engine::decide_pair` make on the bare trace, and the two the door makes
/// in their place. A shed the crate adds ahead of a profile is counted under
/// the first profile until it joins this table. So is a shed of the encode
/// entry, which this host does not ask. A plugin stage registered under one
/// of these words is read as no profile.
const AHEAD_OF_A_PROFILE: [(ShedReason, &str); 5] = [
    (ShedReason::PolicyDeclined, "binding"),
    (ShedReason::PolicyDeclined, "affinity-column"),
    (ShedReason::PolicyDeclined, "written-rows"),
    (ShedReason::NoEndpoints, "wires"),
    (ShedReason::NoEndpoints, "subset-hint"),
];

/// The profile that made `decision`, `None` where none did. A shed made
/// ahead of every profile carries the default trace, whose profile index is
/// the first profile's: it is counted and traced under none.
fn profile_of(decision: &Decision) -> Option<u16> {
    match decision {
        Decision::Shed(shed) if AHEAD_OF_A_PROFILE.contains(&(shed.reason, shed.emptied_by)) => {
            None
        }
        _ => Some(decision.trace().profile),
    }
}

/// The next ask's process-local number, in eight bytes. The gateway assigns
/// it independently of any request id supplied by a client.
pub(crate) fn mark() -> [u8; 8] {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed).to_be_bytes()
}

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
    /// The attempt's prompt hashes, where it has them. The stage that holds
    /// the tokens fills this once per attempt, before its first ask.
    pub hashes: Option<&'a Hashes>,
}

impl<'a> Ask<'a> {
    /// The request as the controller reads it, with the targets the gateway
    /// `missed` at this attempt's earlier commits. The one place a fact of
    /// a request is stated: a later step adds its fact here. Each ask
    /// borrows the attempt's hashes.
    ///
    /// Each call draws the decision's seed anew. The seed is the one
    /// randomness a stage gets: under a constant seed a policy that draws
    /// picks the same worker of an unchanged pool every time.
    fn request(&self, missed: &'a [WorkerId]) -> Request<'a> {
        let mut facts = RequestFacts {
            model: self.model,
            streaming: self.streaming,
            attempt: u8::try_from(self.tried.len()).unwrap_or(u8::MAX),
            tried: self.tried,
            seed: rand::rng().random(),
            ..RequestFacts::default()
        };
        if let Some(hashes) = self.hashes {
            facts.block_hashes = &hashes.blocks;
            facts.hash_kind = HashKind::Token;
            facts.block_size = hashes.block_size;
            facts.namespace_seed = hashes.namespace_seed;
            facts.prompt_tokens = hashes.prompt_tokens;
        }
        Request {
            id: self.id,
            facts,
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

impl Unplaced {
    /// The answer for a request of `model` that has no worker: the status,
    /// the code and the retry hint, one function for every router.
    ///
    /// | Why | The answer |
    /// |---|---|
    /// | The pool is saturated, or the controller queued the request | 503 `worker_overload_protection_shed`, with a retry hint |
    /// | No endpoint is left, or the gateway can send to no target | 503 `no_available_workers`, with a retry hint |
    /// | A queue is at its limit | 429 `admission_queue_full`, with the admission queue's retry hint |
    /// | The decision's budget passed | 503 `admission_queue_timeout`, with the same hint |
    /// | The picker declined every candidate | 503 `policy_declined` |
    /// | The door made no decision | 503 `policy_controller_unavailable`, with a retry hint |
    ///
    /// No answer is retried in process: a second ask a moment later reads
    /// the same rows. A route that asks for a side of a pair refuses by name
    /// before it asks, so that refusal of the door has no row of its own.
    pub(crate) fn answer(&self, model: &str) -> Response {
        let mut response = match *self {
            Self::Shed {
                reason: ShedReason::Saturated,
                retry_after_secs,
                ..
            } => hinted(
                error::service_unavailable(
                    WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE,
                    format!("All workers for model '{model}' are overloaded"),
                ),
                u32::from(retry_after_secs),
            ),
            Self::Queued => hinted(
                error::service_unavailable(
                    WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE,
                    format!("The workers for model '{model}' take no request now"),
                ),
                0,
            ),
            Self::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by,
                ..
            } => no_available_workers_because(
                format!("No available workers for model '{model}'"),
                left_no_worker(emptied_by),
            ),
            Self::NoTargetHolds => no_available_workers_because(
                format!("No available workers for model '{model}'"),
                "the gateway could send to no target the policy controller picked",
            ),
            Self::Shed {
                reason: ShedReason::FlowLimit,
                ..
            } => hinted(
                error::too_many_requests(
                    "admission_queue_full",
                    "a queue of the policy controller is at its limit",
                ),
                SHED_RETRY_AFTER_SECS,
            ),
            Self::Shed {
                reason: ShedReason::Deadline,
                ..
            } => hinted(
                error::service_unavailable(
                    "admission_queue_timeout",
                    "the decision's budget passed before a worker was picked",
                ),
                SHED_RETRY_AFTER_SECS,
            ),
            Self::Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by,
                ..
            } => error::service_unavailable(
                "policy_declined",
                format!(
                    "The policy's {emptied_by} stage declined every worker for model '{model}'"
                ),
            ),
            Self::Refused(_) => hinted(
                error::service_unavailable(
                    "policy_controller_unavailable",
                    "The policy controller made no decision for this request",
                ),
                0,
            ),
        };
        mark_non_retryable(&mut response);
        response
    }
}

/// Why the controller left no worker, for the refusal's line: the stage
/// that emptied the pool, or that the pool held no candidate before any
/// stage ran.
fn left_no_worker(emptied_by: &str) -> String {
    if emptied_by == NO_STAGE {
        "the policy controller holds no candidate for the model".to_owned()
    } else {
        format!("the policy controller's {emptied_by} stage left no worker")
    }
}

/// `response` with a retry hint of `secs` seconds, one at the least.
fn hinted(mut response: Response, secs: u32) -> Response {
    response
        .headers_mut()
        .insert(RETRY_AFTER, HeaderValue::from(secs.max(1)));
    response
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

    fn profile_name(&self, trace: &Trace) -> Option<Arc<str>> {
        self.door.store().with_world(|world| {
            let spec = world.spec();
            if spec.generation() != trace.generation {
                return None;
            }
            let profile = spec.engine().profiles().get(usize::from(trace.profile))?;
            Some(Arc::from(profile.name.as_str()))
        })
    }

    fn trace_decision(&self, decision: &Decision, worker: Option<&Arc<dyn Worker>>) {
        cache_trace::decided(|| {
            let trace = decision.trace();
            let shed = match decision {
                Decision::Shed(shed) => {
                    let reason = match shed.reason {
                        ShedReason::Saturated => "saturated",
                        ShedReason::FlowLimit => "flow_limit",
                        ShedReason::NoEndpoints => "no_endpoints",
                        ShedReason::PolicyDeclined => "policy_declined",
                        ShedReason::Deadline => "deadline",
                    };
                    Some((reason, shed.emptied_by))
                }
                Decision::Pick(_) | Decision::Queue(_) => None,
            };
            cache_trace::Decided {
                profile: profile_of(decision).and_then(|_| self.profile_name(trace)),
                outcome: outcome(decision),
                generation: trace.generation,
                survivors: trace.survivors.get(),
                top: trace
                    .top
                    .iter()
                    .take(usize::from(trace.top_len))
                    .map(|&(id, score)| (self.worker(id), score))
                    .collect(),
                seed: trace.seed,
                lifted: match trace.lifted {
                    Lifted::None => "none",
                    Lifted::Overload => "overload",
                    Lifted::Liveness => "liveness",
                },
                shed,
                worker: worker.cloned(),
            }
        });
    }

    /// The ask: one decision for `ask` at `now` through the controller's
    /// handle, naming the targets the gateway `missed` at this attempt's
    /// earlier commits.
    ///
    /// The one function of the host that may await. The door of this
    /// process decides on the calling thread, under the store's own locks,
    /// and awaits nothing.
    #[expect(
        clippy::unused_async,
        reason = "a controller at an address is asked behind this call, over its session"
    )]
    pub(crate) async fn ask(
        &self,
        ask: &Ask<'_>,
        missed: &[WorkerId],
        now: Instant,
    ) -> Result<Asked, Unplaced> {
        let request = ask.request(missed);
        let started = Instant::now();
        let decided = Handle::decide(Arc::clone(&self.door), GATEWAY, &request, now);
        let seconds = started.elapsed().as_secs_f64();
        let decision = decided.as_ref().ok().map(Handle::decision);
        let trace = decision.map(Decision::trace);
        let generation = trace.map_or(0, |trace| trace.generation);
        let profile = decision.and_then(profile_of);
        Metrics::record_policy_decision(
            ask.model,
            PolicyDecision {
                generation,
                profile,
                outcome: decision.map_or("refused", outcome),
                mode: "embedded",
            },
            || trace.and_then(|trace| self.profile_name(trace)),
            seconds,
        );
        let handle = decided.map_err(Unplaced::Refused)?;
        let decision = *handle.decision();
        match decision {
            Decision::Pick(pick) => Ok(Asked {
                targets: pick.targets,
                handle,
            }),
            Decision::Shed(shed) => {
                self.trace_decision(&decision, None);
                Err(Unplaced::Shed {
                    reason: shed.reason,
                    emptied_by: shed.emptied_by,
                    retry_after_secs: shed.retry_after_secs,
                })
            }
            Decision::Queue(_) => Err(Unplaced::Queued),
        }
    }

    /// The commit: walks the targets of `asked`, best first, and takes the
    /// first the gateway can send to. It awaits nothing. It reads each
    /// target's row under the store's membership lock, which a publication
    /// holds through a rebuild: until the host resolves a target by its id,
    /// a caller holds no lock of its own across it.
    ///
    /// Where no target holds, the decision's lease is given up by name at
    /// `now` and its targets join `missed`: the next ask names them.
    pub(crate) fn commit(
        &self,
        asked: Asked,
        missed: &mut Vec<WorkerId>,
        now: Instant,
    ) -> Option<Placed> {
        self.commit_where(asked, |_| true, missed, now)
    }

    /// The commit of a route with a rule of its own: as [`Host::commit`],
    /// and a target whose worker `holds` refuses is passed over like one
    /// the gateway cannot send to.
    ///
    /// The rule is the route's and the controller does not read it. Where
    /// few workers pass it, an attempt's asks may end with no worker while
    /// one that passes stands. A demand the controller keeps ahead of its
    /// stages takes the rule's place once the request can state it.
    pub(crate) fn commit_where(
        &self,
        asked: Asked,
        holds: impl Fn(&dyn Worker) -> bool,
        missed: &mut Vec<WorkerId>,
        now: Instant,
    ) -> Option<Placed> {
        let Asked {
            targets,
            mut handle,
        } = asked;
        for (index, target) in targets.iter().enumerate() {
            let sendable = self.worker(target.id).filter(|worker| {
                worker.is_healthy()
                    && worker.circuit_breaker_can_execute()
                    && holds(worker.as_ref())
            });
            if let Some(worker) = sendable {
                handle.take(index);
                self.trace_decision(handle.decision(), Some(&worker));
                return Some(Placed {
                    worker,
                    target: *target,
                    handle,
                });
            }
        }
        handle.released(ReleaseReason::CommitFailed, now);
        missed.extend(targets.iter().map(|target| target.id));
        self.trace_decision(handle.decision(), None);
        None
    }

    /// A regular attempt's worker at `now`: the ask and the commit, three
    /// asks at most. After the third the attempt has no worker.
    pub(crate) async fn place(&self, ask: &Ask<'_>, now: Instant) -> Result<Placed, Unplaced> {
        self.place_where(ask, |_| true, now).await
    }

    /// A regular attempt's worker at `now` on a route with a rule of its
    /// own: as [`Host::place`], over [`Host::commit_where`].
    pub(crate) async fn place_where(
        &self,
        ask: &Ask<'_>,
        holds: impl Fn(&dyn Worker) -> bool + Send + Sync,
        now: Instant,
    ) -> Result<Placed, Unplaced> {
        let mut missed: Vec<WorkerId> = Vec::new();
        for _ in 0..ASKS {
            let asked = self.ask(ask, &missed, now).await?;
            if let Some(placed) = self.commit_where(asked, &holds, &mut missed, now) {
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
    target: Target,
    handle: Handle<Arc<Door>>,
}

impl Placed {
    /// The worker the request goes to.
    pub(crate) fn worker(&self) -> &Arc<dyn Worker> {
        &self.worker
    }

    /// The store's row of that worker: what a retry names as tried.
    pub(crate) fn id(&self) -> WorkerId {
        self.target.id
    }

    /// The data-parallel rank the decision named for the worker: what a
    /// dispatch that pins a rank sends.
    pub(crate) fn rank(&self) -> u16 {
        self.target.dp_rank
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
            served: self.target.id,
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
            .field("target", &self.target)
            .field("handle", &self.handle)
            .finish()
    }
}

/// What a response carries of its worker to its end: the attempt's
/// placement, and the worker crate's load guard beside it. The guard gains
/// no field: the worker crate does not name the policy crate.
pub(crate) struct Held {
    /// Dropped first: a report still owed is said while the load stands.
    placed: Placed,
    /// The load of the placement's worker, given back at the drop.
    #[expect(dead_code, reason = "held for its drop alone")]
    guard: WorkerLoadGuard,
}

impl Held {
    /// The placement beside the load guard `guard` makes of its worker. The
    /// guard is asked of the placement's own worker, so the two do not name
    /// two workers, and neither is dropped ahead of its turn.
    pub(crate) fn new(
        placed: Placed,
        guard: impl FnOnce(&Arc<dyn Worker>) -> WorkerLoadGuard,
    ) -> Self {
        let guard = guard(placed.worker());
        Self { placed, guard }
    }

    /// The placement, for the reports its router says.
    pub(crate) fn placed(&mut self) -> &mut Placed {
        &mut self.placed
    }
}

#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;
