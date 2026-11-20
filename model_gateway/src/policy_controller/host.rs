//! The gateway as the policy controller's host in process: one decision an
//! attempt, from the ask to the request's end.
//!
//! An attempt has two halves. The ask ([`Host::ask`]) states what the
//! attempt knows of its request and gets a decision through the
//! controller's handle. The commit takes the first target the gateway
//! can send to. [`Host::place`] does this for a regular attempt. [`Host::place_pair`]
//! resolves both workers of each target in the ask so the commit can run
//! under the prefill admission lock. The store builds the candidates: the
//! gateway sends none and keeps no view of the rows.
//!
//! The door in this process decides on the calling thread. A controller
//! at an address is asked over the gateway's link, behind the same call
//! ([`Host::linked`]): the decision is that controller's, its lease is
//! booked there, and the handle's reports go on the link's session. Where
//! that controller gives an ask no decision, the door of this process
//! decides in its place, over the last rows the controller told. Such a
//! gateway keeps serving while its controller is away. A pair can wait
//! for prefill admission and asks again when the queue wakes it.
//!
//! A router speaks to [`Placed`] and [`Held`], never to the controller's
//! handle. [`Placed`] holds the workers and the decision's lease. Each
//! side reports its dispatch, then its end or a release. The handle says
//! any reports still owed at the drop. [`Held`] carries a regular
//! worker's load guard with the placement to the response's end. The
//! regular HTTP router holds its placement over the body that carries the
//! guard.
//!
//! The gateway's own check of a target is the worker's status and its
//! circuit breaker. In process the status is what the gateway's probes
//! read. Under a controller's address the gateway probes nothing: the
//! status is that controller's verdict, copied from the row. The overload
//! and stall vetoes are the controller's, read from its rows.

use std::{
    cell::Cell,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::{
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use parking_lot::Mutex;
use rand::RngExt;
use smg_policy_controller::{
    decision::{Decision, Lifted, Pair, ShadowKind, ShedReason, Targets, Trace, MAX_TARGETS},
    door::{Door, Maker, Refused, Request, Swept},
    handle::Handle,
    lease::{Outcome, ReleaseReason, Side, TransportClass},
    spec::DecodePick,
    store::{GatewayId, MemoryStore, World},
    view::{Constraints, Identity, Phase, RequestFacts, WorkerId},
    worker_host::Objects,
    AppliedSpec, DerivedKey, KeySource, MediaCounts, WireModes,
};
use smg_worker::identity::basic::{Worker, WorkerLoadGuard, WorkerType};

use super::{follower::Linked, refused};
use crate::{
    middleware::{
        ADMISSION_QUEUE_FULL_CODE, ADMISSION_QUEUE_FULL_STATUS, ADMISSION_QUEUE_TIMEOUT_CODE,
        ADMISSION_QUEUE_TIMEOUT_STATUS, SHED_RETRY_AFTER_SECS,
    },
    observability::{
        cache_trace,
        metrics::{Metrics, PolicyDecision},
    },
    routers::{
        common::{
            overload::{routable_but_for_veto, WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE},
            placement::{self, no_available_workers_because},
            prefill_admission::{
                acquire_decided, PrefillAcquireError, PrefillAdmission, PrefillAdmissionRejection,
                PrefillCandidateError, PrefillLoadGuard, PrefillSelectionContext,
            },
            retry::mark_non_retryable,
            snapshot::{ModelWorkerSnapshot, WorkerRegistry},
        },
        error, prefill_queue_full, prefill_queue_timeout,
    },
};

mod hashes;

pub(crate) use hashes::{Hashes, Prompt};

/// The gateway of this process, as the controller's store names it: the
/// crate's own name for the one gateway that decides in its own process.
const GATEWAY: GatewayId = GatewayId::LOCAL;

/// The asks of one attempt: the first, and two more that name the targets
/// the gateway could not send to.
const ASKS: usize = 3;

/// The time an ask over the link gives the controller to decide in, in
/// microseconds, stated in each request. A request's own budget of 0 means
/// no bound at the door and a spent request over a link, so the host states
/// one. It is fixed until the policy document names it.
pub(crate) const DECIDE_BUDGET_MICROS: u32 = 50_000;

/// The shape of request a gateway under a controller's address does not
/// decide yet, as its refusal names it: the host asks no pair over its
/// link.
const PAIR_UNDER_AN_ADDRESS: &str = "a pair under a controller's address";

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

/// The profile `decision` was made under, `None` where it was made ahead of
/// every profile: its trace says so itself, and it is counted and traced
/// under none.
fn profile_of(decision: &Decision) -> Option<u16> {
    decision.trace().made_under()
}

/// The next ask's process-local number, in eight bytes. The gateway assigns
/// it independently of any request id supplied by a client.
pub(crate) fn mark() -> [u8; 8] {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed).to_be_bytes()
}

/// What an attempt states of its request.
///
/// A stage builds its ask over [`Ask::of`] and names the facts it holds:
/// `Ask { streaming, tried, ..Ask::of(id, model) }`. A fact the host learns
/// to read later is then stated by the stages that hold it, and breaks no
/// other.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ask<'a> {
    /// The request's id.
    pub id: &'a [u8],
    /// The model the request is for, as the router resolved it.
    pub model: &'a str,
    /// Whether the answer is streamed.
    pub streaming: bool,
    /// The request's headers. The sources of its routing key are read from
    /// them, where its binding names a header.
    pub headers: Option<&'a HeaderMap>,
    /// The request's own id, the `rid` of its body: a source of its routing
    /// key, where its binding names it.
    pub rid: Option<&'a str>,
    /// A regular attempt is single. A pair is asked at its prefill.
    pub phase: Phase,
    /// The route asks a pair's decode again behind its prefill's answer
    /// ([`Placed::decode_again`]). Where the document in force picks the
    /// decode after the prefill ([`DecodePick::AfterPrefill`]), such an ask
    /// states the later decode: its decision books the prefill alone and
    /// names the decode's targets. A route that never asks again states
    /// nothing here, and keeps one decision under every document.
    pub decode_again: bool,
    /// The size of the prompt in tokens. Zero uses the count from hashing if available.
    pub prompt_tokens: u32,
    /// The media the request carries, by count. Zero where it carries
    /// none. A stage that places by media reads it.
    pub media: MediaCounts,
    /// The wires the route dials. A route that dials one states it, and
    /// the controller places among the rows that serve it. None stated
    /// leaves every row in.
    pub wires: WireModes,
    /// The rows of the workers the request's failed attempts were sent to.
    pub tried: &'a [WorkerId],
    /// The retry number, `0` for the first attempt. `None` counts the rows
    /// of `tried`, one a failed attempt. A stage whose failed attempt names
    /// more rows than one states the number: a pair names two.
    pub attempt: Option<u8>,
    /// The attempt's prompt hashes, where it has them. The stage that holds
    /// the tokens fills this once per attempt, before its first ask.
    pub hashes: Option<&'a Hashes>,
}

impl<'a> Ask<'a> {
    /// An ask for `model` under the request's `id` that states no other
    /// fact: a single request of no stated prompt size, not streamed, no
    /// input of a routing key, no media, no stated wire, nothing tried, no
    /// stated attempt, no hashes, no second ask of a decode.
    pub(crate) const fn of(id: &'a [u8], model: &'a str) -> Self {
        Self {
            id,
            model,
            streaming: false,
            headers: None,
            rid: None,
            phase: Phase::Single,
            decode_again: false,
            prompt_tokens: 0,
            media: MediaCounts::NONE,
            wires: WireModes::NONE,
            tried: &[],
            attempt: None,
            hashes: None,
        }
    }

    /// The request's two keys by the sources its binding names in `spec`.
    /// Its routing key is the first session source that holds a value, with
    /// the source it came from
    /// ([`KeySources::derive`](smg_policy_controller::KeySources::derive)):
    /// `None` where the binding names no source, or none of them holds a
    /// value. Its client's key is read where it has no routing key
    /// ([`KeySources::derive_client`](smg_policy_controller::KeySources::derive_client)),
    /// and is `0` where it carries none.
    ///
    /// The routing key borrows the document. The door hands the ask the
    /// world it decides under, and the keys are derived there
    /// ([`Handle::decide_under`]).
    fn keys<'s>(&self, spec: &'s AppliedSpec) -> (Option<DerivedKey<'s>>, u64)
    where
        'a: 's,
    {
        let Some(sources) = spec.key_sources_for(self.model, self.phase) else {
            return (None, 0);
        };
        let headers = self.headers;
        let header = |name: &str| headers?.get(name).map(HeaderValue::as_bytes);
        let derived = sources.derive(header, self.rid);
        // A client's key places where no session's does: it is read for
        // such a request alone.
        let client_key = match derived {
            Some(_) => 0,
            None => sources.derive_client(header),
        };
        (derived, client_key)
    }

    /// The request as the controller reads it, with its routing `key`, its
    /// client's key and the targets the gateway `missed` at this attempt's
    /// earlier asks or commits. The one place a fact of a request is
    /// stated: a later step adds its fact here. Each ask borrows the
    /// attempt's hashes.
    ///
    /// Each call draws the decision's seed anew. The seed is the one
    /// randomness a stage gets: under a constant seed a policy that draws
    /// picks the same worker of an unchanged pool every time.
    fn request<'s>(&self, key: Identity<'s>, client_key: u64, missed: &'s [WorkerId]) -> Request<'s>
    where
        'a: 's,
    {
        let mut facts = RequestFacts {
            model: self.model,
            streaming: self.streaming,
            session_key: key,
            phase: self.phase,
            prompt_tokens: self.prompt_tokens,
            media: self.media,
            client_key,
            attempt: self
                .attempt
                .unwrap_or_else(|| u8::try_from(self.tried.len()).unwrap_or(u8::MAX)),
            tried: self.tried,
            seed: rand::rng().random(),
            ..RequestFacts::default()
        };
        if let Some(hashes) = self.hashes {
            facts.block_hashes = &hashes.blocks;
            facts.hash_kind = hashes.kind;
            facts.block_size = hashes.block_size;
            facts.namespace_seed = hashes.namespace_seed;
            if facts.prompt_tokens == 0 {
                facts.prompt_tokens = hashes.prompt_tokens;
            }
        }
        Request {
            id: self.id,
            facts,
            constraints: Constraints {
                excluded: missed,
                wire_modes: self.wires,
                ..Constraints::default()
            },
        }
    }

    /// Whether this ask states the later decode under `spec`: its route
    /// asks the decode again, and the document picks the decode after the
    /// prefill.
    fn decode_later(&self, spec: &AppliedSpec) -> bool {
        self.decode_again && spec.decode_pick() == DecodePick::AfterPrefill
    }
}

/// One ask of an attempt, as the door takes it up: its request is made from
/// the world the decision is made under, so its keys are derived by the
/// sources of the document that decides.
struct Made<'r, 'a> {
    ask: &'r Ask<'a>,
    missed: &'r [WorkerId],
    /// Where the key of the last making came from, under the label the
    /// gateway counts it by. The door may make the request a second time,
    /// behind a lease the store refused: the host counts after the
    /// decision, once.
    source: Cell<Option<&'static str>>,
}

impl Maker for Made<'_, '_> {
    fn model(&self) -> &str {
        self.ask.model
    }

    fn make<'w>(&'w self, world: &'w World) -> Request<'w> {
        let (derived, client_key) = self.ask.keys(world.spec());
        self.source
            .set(derived.map(|derived| counted_as(&derived.source)));
        let key = derived.map_or(Identity::NONE, |derived| derived.key);
        let mut request = self.ask.request(key, client_key, self.missed);
        // The document's word, like the keys: read from the generation
        // that decides.
        request.constraints.decode_later = self.ask.decode_later(world.spec());
        request
    }
}

/// The label the gateway counts a key's source by.
fn counted_as(source: &KeySource<'_>) -> &'static str {
    match source {
        KeySource::Header(_) => "header",
        KeySource::Rid => "rid",
    }
}

/// Why an attempt has no worker.
#[derive(Debug, PartialEq)]
pub(crate) enum Unplaced {
    /// The controller shed the request.
    Shed {
        /// Why.
        reason: ShedReason,
        /// The stage that left no candidate, or what of the policy
        /// declined.
        emptied_by: &'static str,
        /// The seconds after which a retry may find room.
        retry_after_secs: u16,
    },
    /// The controller queued the request.
    Queued,
    /// The door made no decision.
    Refused(Refused),
    /// The controller at the gateway's address gave no decision, and the
    /// door of this process had no fleet to decide over: no snapshot was
    /// taken yet.
    Unanswered(NoAnswer),
    /// The controller is picked and does not decide this shape of request
    /// yet. The shape is named as the refusal says it.
    Undecided(&'static str),
    /// The gateway could send to no target of its three asks.
    NoTargetHolds,
    /// A pair's ask found no endpoint, and by membership no prefill pairs
    /// with a decode: each pool's pairing keys and the components the two
    /// pools disagree on, as the store's pair classes hold them.
    NoCompatiblePair {
        /// The prefill pool's pairing keys.
        prefill: Vec<String>,
        /// The decode pool's pairing keys.
        decode: Vec<String>,
        /// The components the pools disagree on.
        mismatches: Vec<String>,
    },
    /// A pair can be sent, but its prefill has no admission room.
    NoRoom,
    /// The prefill admission queue refused the request.
    Admission(PrefillAdmissionRejection),
}

impl PrefillCandidateError for Unplaced {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Self::NoRoom)
    }

    /// A decision with no target the gateway can send to is decided again
    /// with those targets named, three passes at most at one wake.
    fn decides_again(&self, again: usize) -> bool {
        matches!(self, Self::NoTargetHolds) && again + 1 < ASKS
    }
}

/// Why a controller at an address gave an attempt no decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoAnswer {
    /// No link stands: before the first dial, or from a link's end to the
    /// next dial.
    NoLink,
    /// The link's decide stream holds no credit. Nothing was sent.
    NoCredit,
    /// The controller left the request to the gateway.
    Busy,
    /// No answer came inside the wait.
    Late,
    /// An ask under the request's id is in flight already. Nothing was sent.
    InFlight,
    /// The link ended under the ask.
    Ended,
    /// The request could not be sent, or its answer could not be read.
    Unsent,
}

impl NoAnswer {
    /// The reason, as the gateway's metrics label it.
    fn reason(self) -> &'static str {
        match self {
            Self::NoLink => "no_link",
            Self::NoCredit => "no_credit",
            Self::Busy => "busy",
            Self::Late => "late",
            Self::InFlight => "in_flight",
            Self::Ended => "ended",
            Self::Unsent => "unsent",
        }
    }
}

/// Who decided an ask.
#[derive(Clone, Copy, PartialEq, Eq)]
enum By {
    /// The door of this process, for a gateway with no controller's
    /// address.
    Door,
    /// The controller at the gateway's address.
    Controller,
    /// The door of this process, in the place of a controller that gave
    /// the ask no decision.
    DoorInstead,
}

impl By {
    /// The `mode` label of the decision metrics.
    fn mode(self) -> &'static str {
        match self {
            Self::Door => "embedded",
            Self::Controller => "remote",
            Self::DoorInstead => "local",
        }
    }

    /// Whether the document this process holds decided: its profiles have
    /// names here.
    fn here(self) -> bool {
        self != Self::Controller
    }
}

impl Unplaced {
    /// The answer for a request of `model` that has no worker: the status,
    /// the code and the retry hint, one function for every router.
    ///
    /// | Why | The answer |
    /// |---|---|
    /// | The pool is saturated, admission found no room, or the controller queued the request | 503 `worker_overload_protection_shed`, with a retry hint |
    /// | No endpoint is left, or the gateway can send to no target | 503 `no_available_workers`, with a retry hint |
    /// | A pair's pools hold workers and no prefill pairs with a decode | 503 `no_compatible_pd_pair`, naming what the pools disagree on |
    /// | A controller queue is at its limit | 429 `admission_queue_full`, with a retry hint |
    /// | The decision's budget passed | 503 `admission_queue_timeout`, with the same hint |
    /// | The prefill admission queue is full or timed out | 429 `pd_prefill_queue_full` or `pd_prefill_queue_timeout` |
    /// | The policy declined the request | 503 `policy_declined` |
    /// | The door made no decision, or a controller at an address gave none before its first snapshot | 503 `policy_controller_unavailable`, with a retry hint |
    /// | The controller does not decide the request's shape yet | 501 `not_decided_by_policy_controller` |
    ///
    /// A declined request reads the same message for every cause. Its
    /// refusal line names where the policy declined.
    ///
    /// These answers end the routing attempt.
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
            Self::NoRoom => hinted(
                error::service_unavailable(
                    WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE,
                    format!("No prefill worker for model '{model}' has admission room"),
                ),
                SHED_RETRY_AFTER_SECS,
            ),
            Self::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by,
                ..
            } => no_available_workers_because(
                format!("No available workers for model '{model}'"),
                left_no_worker(emptied_by),
            ),
            Self::NoTargetHolds | Self::Admission(PrefillAdmissionRejection::Unavailable) => {
                no_available_workers_because(
                    format!("No available workers for model '{model}'"),
                    "the gateway could send to no target the policy controller picked",
                )
            }
            Self::NoCompatiblePair {
                ref prefill,
                ref decode,
                ref mismatches,
            } => placement::no_compatible_pair(model, mismatches, prefill, decode),
            Self::Admission(PrefillAdmissionRejection::QueueFull) => prefill_queue_full(),
            Self::Admission(PrefillAdmissionRejection::QueueTimeout) => prefill_queue_timeout(),
            Self::Shed {
                reason: ShedReason::FlowLimit,
                ..
            } => hinted(
                error::create_error(
                    ADMISSION_QUEUE_FULL_STATUS,
                    ADMISSION_QUEUE_FULL_CODE,
                    "a queue of the policy controller is at its limit",
                ),
                SHED_RETRY_AFTER_SECS,
            ),
            Self::Shed {
                reason: ShedReason::Deadline,
                ..
            } => hinted(
                error::create_error(
                    ADMISSION_QUEUE_TIMEOUT_STATUS,
                    ADMISSION_QUEUE_TIMEOUT_CODE,
                    "the decision's budget passed before a worker was picked",
                ),
                SHED_RETRY_AFTER_SECS,
            ),
            Self::Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by,
                ..
            } => return placement::policy_declined(model, emptied_by),
            Self::Refused(_) => hinted(
                error::service_unavailable(
                    "policy_controller_unavailable",
                    "The policy controller made no decision for this request",
                ),
                0,
            ),
            Self::Unanswered(_) => hinted(
                error::service_unavailable(
                    "policy_controller_unavailable",
                    "The policy controller gave no decision for this request",
                ),
                0,
            ),
            Self::Undecided(shape) => refused(shape),
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

/// A pick awaiting commit, with its targets and lease.
pub(crate) struct Asked {
    targets: Targets,
    /// The prefill and decode resolved for each pair target.
    pairs: [[Option<Arc<dyn Worker>>; 2]; MAX_TARGETS],
    handle: Handle<Arc<Door>>,
    by: By,
}

/// A routing key copied out of the world it was derived under, for an ask
/// that waits: a world held across the wait would keep a publication's rows
/// from their next use.
struct Key {
    hash: u64,
    plain: Option<Box<str>>,
}

impl Key {
    fn of(key: Identity<'_>) -> Self {
        Self {
            hash: key.hash,
            plain: key.plain.map(Box::from),
        }
    }

    fn identity(&self) -> Identity<'_> {
        Identity {
            hash: self.hash,
            plain: self.plain.as_deref(),
        }
    }
}

/// The policy controller's host in this process: the door over the
/// controller's store, and the holder of the gateway's worker objects.
pub(crate) struct Host {
    door: Arc<Door>,
    workers: Workers,
    /// The link to the controller that decides, where the gateway was given
    /// a controller's address. `None`: the door decides in this process.
    linked: Option<Arc<Linked>>,
}

/// Where the host finds the worker of a target's row.
enum Workers {
    /// The object the feed's holder laid for the row.
    Held(Arc<Objects<ModelWorkerSnapshot>>),
    /// The registry's worker at the row's key, for a bed that registers its
    /// workers by hand. A key is not one life of a worker.
    #[cfg(test)]
    AtKey(Arc<WorkerRegistry>),
}

impl Host {
    /// The host over the controller's `store` and the holder of its worker
    /// `objects`. The door decides in this process.
    pub(crate) fn new(store: Arc<MemoryStore>, objects: Arc<Objects<ModelWorkerSnapshot>>) -> Self {
        Self {
            door: Arc::new(Door::new(store)),
            workers: Workers::Held(objects),
            linked: None,
        }
    }

    /// The host of a gateway under a controller's address: each decision is
    /// that controller's, asked over the link the follower holds. `store`
    /// holds the controller's rows as its watch tells them, and the commit
    /// finds a decision's targets among the `objects` the feed's holder
    /// laid for those rows. An ask the controller leaves with no decision
    /// is decided at the door over the same store.
    pub(crate) fn linked(
        store: Arc<MemoryStore>,
        objects: Arc<Objects<ModelWorkerSnapshot>>,
        linked: Arc<Linked>,
    ) -> Self {
        Self {
            linked: Some(linked),
            ..Self::new(store, objects)
        }
    }

    /// One pass of the door's clock at `now`: the call of the host's timer
    /// ([`upkeep`](super::upkeep)), about once a second.
    pub(crate) fn sweep(&self, now: Instant) -> Swept {
        self.door.sweep(now)
    }

    /// How many shadows said `kind` beside a single decision of the
    /// binding at position `binding`: the door's count, which the timer
    /// shows as a series.
    pub(crate) fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64 {
        self.door.shadowed(binding, kind)
    }

    /// How many decisions panicked, the shadows that did beside a live
    /// decision that stood among them: the door's count, shown the same way.
    pub(crate) fn faults(&self) -> u64 {
        self.door.faults()
    }

    /// An attempt's prompt hashes for `model`, under the generation in force.
    /// The stage holding the tokens calls this once before the first ask.
    /// Each ask borrows the same result through [`Ask::hashes`].
    pub(crate) fn hashes(&self, model: &str, prompt: &Prompt<'_>) -> Hashes {
        Hashes::of(self.door.store(), model, prompt)
    }

    /// The host of a bed that registers its workers by hand in `registry`.
    #[cfg(test)]
    pub(crate) fn at_keys(store: Arc<MemoryStore>, registry: Arc<WorkerRegistry>) -> Self {
        Self {
            door: Arc::new(Door::new(store)),
            workers: Workers::AtKey(registry),
            linked: None,
        }
    }

    /// The gateway's worker of a target, found by the row's id: the one
    /// function that resolves a row of the store to a worker. `None` where
    /// the holder has laid no object for the row or the object has left the
    /// registry. So a row the store shows ahead of its object, as a
    /// replacement inside its own write, takes no request.
    fn worker(&self, id: WorkerId) -> Option<Arc<dyn Worker>> {
        match &self.workers {
            Workers::Held(objects) => objects.registry().get(&objects.object(id)?),
            #[cfg(test)]
            Workers::AtKey(registry) => {
                use smg_policy_controller::store::WorkersStore;
                let row = WorkersStore::get(&**self.door.store(), id)?;
                registry.get_by_url(&row.facts.key)
            }
        }
    }

    /// The registry the gateway's workers stand in: the holder's, or a
    /// bed's own.
    fn registry(&self) -> &WorkerRegistry {
        match &self.workers {
            Workers::Held(objects) => objects.registry(),
            #[cfg(test)]
            Workers::AtKey(registry) => registry,
        }
    }

    /// The name of the profile `trace` names in the document this process
    /// holds. It is asked for a decision of this process's door alone
    /// ([`By::here`]): a controller at an address decides by its own
    /// document, which this process holds no copy of.
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

    /// Counts a pair's second ask of its decode for `model`, as an ask
    /// counts its decision: under the generation and the profile of a
    /// call that `picked`, and under none for a call that shed or was
    /// refused, whose decision the handle does not hand on.
    fn count_decode_call(
        &self,
        model: &str,
        picked: Option<&Decision>,
        outcome: &'static str,
        seconds: f64,
    ) {
        let trace = picked.map(Decision::trace);
        Metrics::record_policy_decision(
            model,
            PolicyDecision {
                generation: trace.map_or(0, |trace| trace.generation),
                profile: picked.and_then(profile_of),
                outcome,
                mode: "embedded",
            },
            || trace.and_then(|trace| self.profile_name(trace)),
            seconds,
        );
    }

    /// Why no pair can be made for `ask`, where it asks for one on the wires
    /// it states and the store's pair classes refuse that side of its
    /// fleet. The classes judge from membership alone, as the classic pair
    /// index does, so a fleet that pairs nothing reads the classic's answer
    /// and not a missing endpoint.
    ///
    /// An ask that states no wire does not say which side it draws from. It
    /// keeps the controller's shed.
    fn unpaired(&self, ask: &Ask<'_>) -> Option<Unplaced> {
        if ask.phase == Phase::Single || ask.wires.is_empty() {
            return None;
        }
        self.door.store().with_world(|world| {
            let refusal = world.pairs(ask.model)?.refusal(ask.wires)?;
            Some(Unplaced::NoCompatiblePair {
                prefill: refusal.prefill.clone(),
                decode: refusal.decode.clone(),
                mismatches: refusal.mismatches.clone(),
            })
        })
    }

    fn trace_decision(&self, decision: &Decision, worker: Option<&Arc<dyn Worker>>, by: By) {
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
                profile: profile_of(decision)
                    .filter(|_| by.here())
                    .and_then(|_| self.profile_name(trace)),
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
    /// Resolves both workers of each pair before admission can lock the
    /// queue. The door of this process decides on the calling thread,
    /// under the store's own locks, and awaits nothing. A controller at an
    /// address is asked over the link, and the ask waits for its answer.
    /// Where it gives none, the door of this process decides at the same
    /// `now`, once the store holds a fleet that controller told. A decision
    /// the controller gives late is the link's to release.
    ///
    /// An attempt that keeps one list across its asks asks first with the
    /// list empty: this ask counts as its first then.
    pub(crate) async fn ask(
        &self,
        ask: &Ask<'_>,
        missed: &[WorkerId],
        now: Instant,
    ) -> Result<Asked, Unplaced> {
        self.ask_as(ask, missed, missed.is_empty(), now).await
    }

    /// [`Self::ask`] for an attempt that says itself whether this is its
    /// `first` ask, the one that counts its key's source. A pair's attempt
    /// decides again at each wake of the admission queue, from a list a
    /// wait on room leaves empty: its list does not tell its first ask.
    async fn ask_as(
        &self,
        ask: &Ask<'_>,
        missed: &[WorkerId],
        first: bool,
        now: Instant,
    ) -> Result<Asked, Unplaced> {
        let (decided, source, by) = match &self.linked {
            None => {
                let (decided, source) = self.decide(ask, missed, now, By::Door);
                (decided, source, By::Door)
            }
            Some(linked) => {
                // A pair is not asked over the link yet, as a regular
                // request is, and it is not decided here in the
                // controller's place.
                if ask.phase != Phase::Single {
                    return Err(Unplaced::Undecided(PAIR_UNDER_AN_ADDRESS));
                }
                let started = Instant::now();
                // The remote ask is boxed at its one call. By value its
                // future, with the request it sends, would lie in the future
                // of every selection stage, the local path's too.
                let (asked, source) = Box::pin(self.asked(linked, ask, missed, now)).await;
                self.count(ask, asked.as_ref(), By::Controller, started);
                match asked {
                    Err(Unplaced::Unanswered(why)) => {
                        Metrics::record_policy_unanswered(why.reason());
                        // The door of this process decides in the
                        // controller's place, over the last rows told.
                        // Before the first snapshot it has none.
                        if linked.fleet() {
                            let instead = By::DoorInstead;
                            let (decided, source) = self.decide(ask, missed, now, instead);
                            (decided, source, instead)
                        } else {
                            (Err(Unplaced::Unanswered(why)), source, By::Controller)
                        }
                    }
                    asked => (asked, source, By::Controller),
                }
            }
        };
        // A key's source counts once an attempt, at its first ask, as a
        // keyed selection of the gateway's own policies counts it: an ask
        // again, behind a missed commit or at a pair's later wake, is the
        // same attempt's.
        if let Some(source) = source.filter(|_| first) {
            Metrics::record_routing_key_source(source);
        }
        let handle = decided?;
        let decision = *handle.decision();
        match decision {
            Decision::Pick(pick) => {
                let mut pairs: [_; MAX_TARGETS] = Default::default();
                for (target, workers) in pick.targets.iter().zip(&mut pairs) {
                    if let Some(pair) = target.pair {
                        *workers = [self.worker(target.id), self.worker(pair.decode_id)];
                    }
                }
                Ok(Asked {
                    targets: pick.targets,
                    pairs,
                    handle,
                    by,
                })
            }
            Decision::Shed(shed) => {
                self.trace_decision(&decision, None, by);
                let unpaired = (shed.reason == ShedReason::NoEndpoints)
                    .then(|| self.unpaired(ask))
                    .flatten();
                Err(unpaired.unwrap_or(Unplaced::Shed {
                    reason: shed.reason,
                    emptied_by: shed.emptied_by,
                    retry_after_secs: shed.retry_after_secs,
                }))
            }
            Decision::Queue(_) => Err(Unplaced::Queued),
        }
    }

    /// One decision of the door of this process for `ask`, counted as `by`
    /// names it, with the label of the source its key came from.
    fn decide(
        &self,
        ask: &Ask<'_>,
        missed: &[WorkerId],
        now: Instant,
        by: By,
    ) -> (Result<Handle<Arc<Door>>, Unplaced>, Option<&'static str>) {
        let started = Instant::now();
        let made = Made {
            ask,
            missed,
            source: Cell::new(None),
        };
        let decided = Handle::decide_under(Arc::clone(&self.door), GATEWAY, &made, now)
            .map_err(Unplaced::Refused);
        self.count(ask, decided.as_ref(), by, started);
        (decided, made.source.get())
    }

    /// Counts one ask's end as `by` names it, with its time since `started`.
    fn count(
        &self,
        ask: &Ask<'_>,
        decided: Result<&Handle<Arc<Door>>, &Unplaced>,
        by: By,
        started: Instant,
    ) {
        let seconds = started.elapsed().as_secs_f64();
        let decision = decided.ok().map(Handle::decision);
        let trace = decision.map(Decision::trace);
        let generation = trace.map_or(0, |trace| trace.generation);
        // A controller's profile has no name here, so its decision counts
        // under none and its counter keeps one handle.
        let profile = decision.filter(|_| by.here()).and_then(profile_of);
        let undecided = match decided {
            Err(Unplaced::Unanswered(_)) => "unanswered",
            _ => "refused",
        };
        Metrics::record_policy_decision(
            ask.model,
            PolicyDecision {
                generation,
                profile,
                outcome: decision.map_or(undecided, outcome),
                mode: by.mode(),
            },
            || trace.and_then(|trace| self.profile_name(trace)),
            seconds,
        );
    }

    /// The controller's decision for `ask` over the gateway's link, with
    /// the label of the source its key came from.
    ///
    /// The request's key is derived by the sources of the document this
    /// process holds, as the door derives it, and copied out of the world
    /// before the wait. The controller decides by its own document.
    async fn asked(
        &self,
        linked: &Linked,
        ask: &Ask<'_>,
        missed: &[WorkerId],
        now: Instant,
    ) -> (Result<Handle<Arc<Door>>, Unplaced>, Option<&'static str>) {
        let (key, client_key, source) = self.door.store().with_world(|world| {
            let (derived, client_key) = ask.keys(world.spec());
            (
                Key::of(derived.map_or(Identity::NONE, |derived| derived.key)),
                client_key,
                derived.map(|derived| counted_as(&derived.source)),
            )
        });
        let mut request = ask.request(key.identity(), client_key, missed);
        request.facts.budget_micros = DECIDE_BUDGET_MICROS;
        let decided = linked
            .asked(&request, now)
            .await
            .map_err(Unplaced::Unanswered);
        (decided, source)
    }

    /// The commit: walks the targets of `asked`, best first, and takes the
    /// first the gateway can send to. It awaits nothing and takes the
    /// holder's read lock once for each target it looks up.
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
            by,
            ..
        } = asked;
        for (index, target) in targets.iter().enumerate() {
            let sendable = self
                .worker(target.id)
                .filter(|worker| routable_but_for_veto(worker.as_ref()) && holds(worker.as_ref()));
            if let Some(worker) = sendable {
                handle.take(index);
                self.trace_decision(handle.decision(), Some(&worker), by);
                return Some(Placed {
                    worker,
                    id: target.id,
                    rank: target.dp_rank,
                    pair: None,
                    handle,
                });
            }
        }
        handle.released(ReleaseReason::CommitFailed, now);
        missed.extend(targets.iter().map(|target| target.id));
        self.trace_decision(handle.decision(), None, by);
        None
    }

    /// Takes the first sendable pair whose prefill has room. The admission
    /// lock can be held here, so every lookup happens in the ask.
    pub(crate) fn commit_pair(
        &self,
        asked: Asked,
        room: Option<&PrefillSelectionContext<'_>>,
        missed: &mut Vec<WorkerId>,
        now: Instant,
    ) -> Result<Placed, Unplaced> {
        self.commit_pair_where(asked, |_| true, room, missed, now)
    }

    /// The pair's commit on a route with a rule of its own: as
    /// [`Host::commit_pair`], and a target either of whose workers `holds`
    /// refuses is passed over like one the gateway cannot send to. The
    /// rule is the route's, as [`Host::commit_where`] says of a regular one.
    #[expect(clippy::unused_self, reason = "the commit uses workers from the ask")]
    pub(crate) fn commit_pair_where(
        &self,
        asked: Asked,
        holds: impl Fn(&dyn Worker) -> bool,
        room: Option<&PrefillSelectionContext<'_>>,
        missed: &mut Vec<WorkerId>,
        now: Instant,
    ) -> Result<Placed, Unplaced> {
        let Asked {
            targets,
            pairs,
            mut handle,
            ..
        } = asked;
        let mut failed = [[None; 2]; MAX_TARGETS];
        let mut no_room = false;
        for (index, (target, [prefill, decode])) in targets.iter().zip(pairs).enumerate() {
            let Some(pair) = target.pair else {
                continue;
            };
            let sendable = |worker: &Arc<dyn Worker>| {
                routable_but_for_veto(worker.as_ref()) && holds(worker.as_ref())
            };
            let prefill = prefill.filter(sendable);
            let decode = decode.filter(sendable);
            let (Some(worker), Some(decode)) = (&prefill, decode.as_ref()) else {
                failed[index] = [
                    prefill.is_none().then_some(target.id),
                    decode.is_none().then_some(pair.decode_id),
                ];
                continue;
            };
            if room.is_some_and(|room| !room.has_capacity(worker)) {
                no_room = true;
                continue;
            }
            handle.take(index);
            return Ok(Placed {
                worker: Arc::clone(worker),
                id: target.id,
                rank: target.dp_rank,
                pair: Some(Box::new(PlacedPair {
                    decode: Arc::clone(decode),
                    side: pair,
                    targets,
                    index,
                })),
                handle,
            });
        }
        handle.released(ReleaseReason::CommitFailed, now);
        if no_room {
            return Err(Unplaced::NoRoom);
        }
        missed.extend(failed.into_iter().flatten().flatten());
        Err(Unplaced::NoTargetHolds)
    }

    /// Places a pair and takes its prefill load guard. A commit that can
    /// send to no target decides again in place, three passes at most at
    /// one wake: a queued request keeps its place and its deadline. A pass
    /// asks up to three times while every prefill it names reads full
    /// ([`Self::decide_pair`]), so one wake asks nine times at most.
    /// Admission can decide again at each wake.
    pub(crate) async fn place_pair(
        &self,
        ask: &Ask<'_>,
        admission: Option<&PrefillAdmission>,
        routing_key: Option<&str>,
    ) -> Result<(Placed, PrefillLoadGuard), Unplaced> {
        self.place_pair_where(ask, |_| true, admission, routing_key)
            .await
    }

    /// Places a pair on a route with a rule of its own: as
    /// [`Host::place_pair`], over [`Host::commit_pair_where`]. The pass
    /// that reads which prefills are full and the probe of a request
    /// behind a waiter count a worker the rule refuses as one the gateway
    /// cannot send to.
    pub(crate) async fn place_pair_where(
        &self,
        ask: &Ask<'_>,
        holds: impl Fn(&dyn Worker) -> bool + Send + Sync,
        admission: Option<&PrefillAdmission>,
        routing_key: Option<&str>,
    ) -> Result<(Placed, PrefillLoadGuard), Unplaced> {
        let holds = &holds;
        let missed = Mutex::new(Vec::new());
        // The attempt's first ask, whichever half makes it: each wake's
        // decision starts from the attempt's list, which stays empty
        // across a wait on room.
        let first = AtomicBool::new(true);
        acquire_decided(
            admission,
            routing_key,
            Placed::worker,
            || self.probe_pair(ask, holds, &missed, &first),
            || self.decide_pair(ask, holds, admission, missed.lock().clone(), &first),
            |asked, room| {
                self.commit_pair_where(asked, holds, room, &mut missed.lock(), Instant::now())
            },
        )
        .await
        .map_err(|error| match error {
            PrefillAcquireError::Candidate(verdict) => verdict,
            PrefillAcquireError::Rejected(rejection) => Unplaced::Admission(rejection),
        })
    }

    /// One pass of a pair's decision: full prefills are excluded for up to
    /// two more asks of the pass. The next wake can use them again if load
    /// has fallen.
    async fn decide_pair(
        &self,
        ask: &Ask<'_>,
        holds: &(impl Fn(&dyn Worker) -> bool + Sync),
        admission: Option<&PrefillAdmission>,
        mut missed: Vec<WorkerId>,
        first: &AtomicBool,
    ) -> Result<Asked, Unplaced> {
        for _ in 0..ASKS {
            let first = first.swap(false, Ordering::Relaxed);
            let mut asked = self.ask_as(ask, &missed, first, Instant::now()).await?;
            let Some(admission) = admission else {
                return Ok(asked);
            };
            let before = missed.len();
            for (target, [prefill, decode]) in asked.targets.iter().zip(&asked.pairs) {
                let (Some(prefill), Some(decode)) = (prefill, decode) else {
                    continue;
                };
                let sendable = |worker: &Arc<dyn Worker>| {
                    routable_but_for_veto(worker.as_ref()) && holds(worker.as_ref())
                };
                if sendable(prefill) && sendable(decode) {
                    if !admission.reads_full(prefill) {
                        return Ok(asked);
                    }
                    missed.push(target.id);
                }
            }
            if missed.len() == before {
                return Ok(asked);
            }
            asked
                .handle
                .released(ReleaseReason::CommitFailed, Instant::now());
        }
        Err(Unplaced::NoRoom)
    }

    /// A request behind a waiter asks only if a role has no sendable worker.
    /// A pick is released before the request joins the queue.
    async fn probe_pair(
        &self,
        ask: &Ask<'_>,
        holds: &(impl Fn(&dyn Worker) -> bool + Sync),
        missed: &Mutex<Vec<WorkerId>>,
        first: &AtomicBool,
    ) -> Option<Unplaced> {
        let workers = self.registry().generation(ask.model);
        let (mut prefill, mut decode) = (false, false);
        for worker in workers.iter() {
            let found = match worker.worker_type() {
                WorkerType::Prefill => &mut prefill,
                WorkerType::Decode => &mut decode,
                WorkerType::Regular | WorkerType::Encode => continue,
            };
            *found = *found || (routable_but_for_veto(worker.as_ref()) && holds(worker.as_ref()));
            if prefill && decode {
                return None;
            }
        }
        let missed = missed.lock().clone();
        let first = first.swap(false, Ordering::Relaxed);
        match self.ask_as(ask, &missed, first, Instant::now()).await {
            Ok(mut asked) => {
                asked
                    .handle
                    .released(ReleaseReason::CommitFailed, Instant::now());
                None
            }
            Err(verdict) => Some(verdict),
        }
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
    /// The response status, 0 where none was read.
    pub status: u16,
    /// What became of the transport.
    pub class: TransportClass,
    /// The time to the first token or data frame, in microseconds.
    pub ttft_micros: u32,
    /// The tokens the worker read.
    pub tokens_in: u32,
    /// The tokens the worker wrote.
    pub tokens_out: u32,
    /// The time from the send to the end, in microseconds.
    pub duration_micros: u32,
}

/// What became of the transport, for the status a request ended with.
/// `answered` says the worker gave that status itself.
pub(crate) fn transport_class(status: StatusCode, answered: bool) -> TransportClass {
    match status {
        status if status < StatusCode::BAD_REQUEST => TransportClass::Ok,
        StatusCode::GATEWAY_TIMEOUT => TransportClass::Timeout,
        status if status.is_client_error() => TransportClass::ClientError,
        _ if answered => TransportClass::ServerError,
        _ => TransportClass::ConnectFailed,
    }
}

impl End {
    /// An HTTP answer with timing from the send and any first data frame.
    fn http(
        status: StatusCode,
        class: TransportClass,
        sent: Instant,
        first: Option<Instant>,
        now: Instant,
    ) -> Self {
        Self {
            status: status.as_u16(),
            class,
            ttft_micros: first.map_or(0, |first| elapsed_micros(first, sent)),
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: elapsed_micros(now, sent),
        }
    }

    fn outcome(self, side: Side, served: WorkerId) -> Outcome {
        Outcome {
            side,
            served,
            status: self.status,
            class: self.class,
            ttft_micros: self.ttft_micros,
            tokens_in: self.tokens_in,
            tokens_out: self.tokens_out,
            duration_micros: self.duration_micros,
        }
    }
}

/// The time since `sent` in whole microseconds, saturating at `u32::MAX`.
fn elapsed_micros(now: Instant, sent: Instant) -> u32 {
    u32::try_from(now.saturating_duration_since(sent).as_micros()).unwrap_or(u32::MAX)
}

/// An attempt's workers, with its decision's lease held to the request's
/// end.
pub(crate) struct Placed {
    worker: Arc<dyn Worker>,
    /// The store's row of the worker, from the target taken.
    id: WorkerId,
    /// The rank that target names for the worker.
    rank: u16,
    /// A pair's own part, behind one pointer: a regular placement rides in
    /// its router's request context, whose size a case pins.
    pair: Option<Box<PlacedPair>>,
    handle: Handle<Arc<Door>>,
}

/// What a placed pair holds beside its prefill: the decode worker with the
/// decode side of the target taken, and the pick's targets with that
/// target's index, for the next decode of the same prefill.
struct PlacedPair {
    decode: Arc<dyn Worker>,
    side: Pair,
    targets: Targets,
    index: usize,
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

    /// The data-parallel rank the decision named for the worker: what a
    /// dispatch that pins a rank sends.
    pub(crate) fn rank(&self) -> u16 {
        self.rank
    }

    /// The size the placement's lease states, for a case that reads it.
    #[cfg(test)]
    pub(crate) fn lease_tokens(&self) -> Option<u32> {
        self.handle.lease().map(|leased| leased.lease.tokens)
    }

    /// The id of the placement's lease, for a case that reads the lease's
    /// record at its store.
    #[cfg(test)]
    pub(crate) fn lease(&self) -> Option<smg_policy_controller::lease::LeaseId> {
        self.handle.lease().map(|leased| leased.lease.id)
    }

    /// The request was sent to the worker at `now`. Said once.
    pub(crate) fn dispatched(&mut self, now: Instant) {
        self.handle.dispatched(now);
    }

    /// The request ended with `end`, as the router read it at `now`. Said
    /// once.
    pub(crate) fn completed(&mut self, end: End, now: Instant) {
        let side = self
            .handle
            .lease()
            .map_or(Side::Single, |leased| leased.lease.side);
        let outcome = end.outcome(side, self.id());
        self.handle.completed(&outcome, now);
    }

    /// The request is given up under `reason` at `now`: before a send the
    /// lease is released, and after one it completes as cancelled.
    pub(crate) fn released(&mut self, reason: ReleaseReason, now: Instant) {
        self.handle.released(reason, now);
    }

    /// The decode worker of a pair. A regular attempt has none.
    pub(crate) fn decode(&self) -> Option<&Arc<dyn Worker>> {
        self.pair.as_ref().map(|pair| &pair.decode)
    }

    /// The store's row of the decode worker.
    pub(crate) fn decode_id(&self) -> Option<WorkerId> {
        self.pair.as_ref().map(|pair| pair.side.decode_id)
    }

    /// The rank the decision named for the decode worker.
    pub(crate) fn decode_rank(&self) -> Option<u16> {
        self.pair.as_ref().map(|pair| pair.side.decode_dp_rank)
    }

    /// The decode request was sent at `now`. Said once.
    pub(crate) fn decode_dispatched(&mut self, now: Instant) {
        self.handle.dispatched_decode(now);
    }

    /// The decode request ended with `end`. Said once.
    pub(crate) fn decode_completed(&mut self, end: End, now: Instant) {
        if let Some(worker) = self.decode_id() {
            self.handle
                .completed(&end.outcome(Side::Decode, worker), now);
        }
    }

    /// The decode side is given up under `reason`.
    pub(crate) fn decode_released(&mut self, reason: ReleaseReason, now: Instant) {
        self.handle.released_decode(reason, now);
    }

    /// After the prefill is sent, take the next sendable decode of that
    /// prefill. Returns `false` when the pick holds none or the decode has
    /// already been sent or released.
    pub(crate) fn next_decode(&mut self, host: &Host) -> bool {
        let Some(placed) = self.pair.as_deref_mut() else {
            return false;
        };
        for (index, target) in placed.targets.iter().enumerate().skip(placed.index + 1) {
            if target.id != self.id {
                continue;
            }
            let Some(pair) = target.pair else {
                continue;
            };
            let Some(worker) = host
                .worker(pair.decode_id)
                .filter(|worker| routable_but_for_veto(worker.as_ref()))
            else {
                continue;
            };
            if self.handle.take_decode(index).is_some() {
                placed.decode = worker;
                placed.side = pair;
                placed.index = index;
                return true;
            }
        }
        false
    }

    /// Whether the pair's decode stands chosen and booked nowhere: its ask
    /// stated the later decode, and the decode's dispatch is still owed.
    /// The route asks it again behind its prefill's answer
    /// ([`Self::decode_again`]), or sends the first pick's.
    pub(crate) fn decode_waits(&self) -> bool {
        self.handle.decode_chosen()
    }

    /// Asks the pair's decode again at `now`, on the rows of that instant:
    /// behind the prefill's answer, and ahead of the prefill's end. `ask`
    /// states the request as its first ask did. A pick moves the placement
    /// to the first of its decodes the gateway can send to and `holds`
    /// admits: the route's rule, as at the first commit. Returns the decode
    /// the call moved the placement to, and `None` where the placement
    /// keeps the decode it named.
    ///
    /// The call never fails its request. A decode that does not stand
    /// chosen takes no call. A call the door sheds or refuses leaves the
    /// decode the first pick named, still chosen. Where the call picked
    /// and the gateway can send to none of its decodes, the decode side's
    /// lease is given up: the first pick's decode is then sent unbooked.
    pub(crate) fn decode_again(
        &mut self,
        host: &Host,
        ask: &Ask<'_>,
        holds: impl Fn(&dyn Worker) -> bool,
        now: Instant,
    ) -> Option<&Arc<dyn Worker>> {
        let placed = self.pair.as_deref_mut()?;
        if !self.handle.decode_chosen() {
            return None;
        }
        let made = Made {
            ask,
            missed: &[],
            source: Cell::new(None),
        };
        let handle = &mut self.handle;
        let started = Instant::now();
        // The call's request is made as the first ask's was: from the
        // world the door decides it under, by the sources of that document.
        let decided = handle.decide_decode_under(&made, now).copied();
        let seconds = started.elapsed().as_secs_f64();
        // A pick issued the side. A shed and a refusal left it chosen: the
        // handle answers the first pick behind a shed, so the side tells
        // the two apart.
        let (outcome, picked) = match decided {
            Err(_) => ("refused", None),
            Ok(_) if handle.decode_chosen() => ("shed", None),
            Ok(decision) => (outcome(&decision), Some(decision)),
        };
        host.count_decode_call(ask.model, picked.as_ref(), outcome, seconds);
        let Some(Decision::Pick(pick)) = picked else {
            return None;
        };
        for (index, target) in pick.targets.iter().enumerate() {
            let Some(pair) = target.pair.filter(|_| target.id == self.id) else {
                continue;
            };
            let sendable = |worker: &Arc<dyn Worker>| {
                routable_but_for_veto(worker.as_ref()) && holds(worker.as_ref())
            };
            let Some(worker) = host.worker(pair.decode_id).filter(sendable) else {
                continue;
            };
            // The call took its first target. A later one is taken by name.
            if index == 0 || self.handle.take_decode(index).is_some() {
                placed.decode = worker;
                placed.side = pair;
                placed.targets = pick.targets;
                placed.index = index;
                return Some(&placed.decode);
            }
        }
        self.handle
            .released_decode(ReleaseReason::CommitFailed, now);
        None
    }
}

impl fmt::Debug for Placed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Placed")
            .field("worker", &self.worker.url())
            .field("id", &self.id)
            .field("rank", &self.rank)
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
    /// The load of the worker the response is read from, given back at the
    /// drop.
    #[expect(dead_code, reason = "held for its drop alone")]
    guard: WorkerLoadGuard,
}

impl Held {
    /// The placement beside the load guard `guard` makes of the worker its
    /// response is read from: a single's worker, a pair's decode. The guard
    /// is asked of the placement's own worker, so the two do not name two
    /// workers, and neither is dropped ahead of its turn.
    ///
    /// A pair's prefill holds its load by the admission guard
    /// [`Host::place_pair`] hands back apart. Its router gives that one up
    /// when the prefill phase ends, ahead of the response.
    pub(crate) fn new(
        placed: Placed,
        guard: impl FnOnce(&Arc<dyn Worker>) -> WorkerLoadGuard,
    ) -> Self {
        let guard = guard(placed.decode().unwrap_or_else(|| placed.worker()));
        Self { placed, guard }
    }

    /// The placement, for the reports its router says.
    pub(crate) fn placed(&mut self) -> &mut Placed {
        &mut self.placed
    }
}

mod ride;
pub(crate) use ride::{BehindAnswer, Sent};
#[cfg(test)]
mod decode_again_tests;
#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;
