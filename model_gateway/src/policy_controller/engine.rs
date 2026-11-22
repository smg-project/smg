//! The gateway's policy engine, the policy controller in process: one decision
//! an attempt, from the ask to the request's end.
//!
//! An attempt has two halves. The ask ([`PolicyEngine::ask`]) states what the
//! attempt knows of its request and gets a decision through the
//! controller's handle. The commit takes the first target the gateway
//! can send to. [`PolicyEngine::place`] does this for a regular attempt.
//! [`PolicyEngine::place_pair`] resolves both workers of each target in the ask
//! so the commit can run under the prefill admission lock.
//!
//! The object stands over one of two engines ([`Decider`]). A gateway that
//! follows a controller asks the engine over the store that follows it:
//! the store builds the candidates, and the gateway sends none and keeps no
//! view of the rows. A gateway that decides in its own process asks the
//! engine of this process, over the list its registry publishes for the
//! ask's route ([`Ask::pool`]) and the worker objects on it: no store.
//!
//! Either engine decides on the calling thread. A pair can wait for
//! prefill admission and asks again when the queue wakes it.
//!
//! A router speaks to [`Placed`] and [`Held`], never to the controller's
//! handle. [`Placed`] holds the workers and the decision's lease. Each
//! side reports its dispatch, then its end or a release. The handle says
//! any reports still owed at the drop. A placement the engine of this
//! process made under a document no stage of which takes a word of a
//! request holds no lease and no handle: nothing is owed of it, and each
//! report says nothing. [`Held`] carries a regular
//! worker's load guard with the placement to the response's end. The
//! regular HTTP router holds its placement over the body that carries the
//! guard.
//!
//! The gateway's own check of a target is what it alone knows: the worker's
//! status as its probes read it, and its circuit breaker. The overload and
//! stall vetoes are the controller's, read from its rows.

use std::{
    borrow::Cow,
    cell::Cell,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use arc_swap::ArcSwap;
use axum::{
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use parking_lot::Mutex;
use rand::RngExt;
use rustc_hash::FxHashMap;
use smg_policy_controller::{
    document::{AppliedSpec, DecodePick, DerivedKey, KeySource},
    engine::{
        worker_id, worker_rank, AlonePick, Constraints, DecideOutcome, Decided, Decision,
        DecisionEngine, DecisionRefused, EncodeItems, EncodePlaced, HashSeeds, HeldDecision,
        Identity, LiftedVeto, MediaCounts, Pair, Phase, Plain, PolicyEngine as OwnEngine,
        PoolMembers, Request, RequestFacts, RequestMaker, ShadowKind, ShedReason,
        StoreBackedEngine, SweepOutcome, Target, Targets, Trace, WireModes, WorkerId, WorkerLists,
        MAX_TARGETS,
    },
    store::{pairing::Refusal, GatewayId, MemoryStore, Said},
    tracking::{DecisionGuard, Outcome, ReleaseReason, Side, TransportClass},
    worker_host::Objects,
};
use smg_worker::{
    health::overload::STAGE_SELECTION,
    identity::basic::{Worker, WorkerLoadGuard, WorkerType},
    load::state::{LoadReceiver, LoadSnapshot},
};

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
            snapshot::{ModelWorkerSnapshot, RoutingPool, WorkerRegistry, WorkerRegistryExt},
        },
        error, prefill_queue_full, prefill_queue_timeout,
    },
};

mod hashes;
mod loads;

pub(crate) use smg_policy_controller::engine::PromptHashes as Hashes;
// What an attempt states of its prompt: what its hashes are made of
// ([`PolicyEngine::hashes`]), and what the engine of this process is lent
// ([`Ask::prompt`]). It is the policy crate's own type for a lent prompt,
// so an ask that is printed shows the prompt's sizes alone.
pub(crate) use smg_policy_controller::engine::PromptLent as Prompt;

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

/// The profile `decision` was made under, `None` where it was made ahead of
/// every profile: its trace says so itself, and it is counted and traced
/// under none.
fn profile_of(decision: &Decision) -> Option<u16> {
    decision.trace().made_under()
}

/// The labels a decision made in this process is counted under: those of
/// its trace, and `refused` with no profile where the engine made none.
fn labels(decision: Option<&Decision>) -> PolicyDecision {
    PolicyDecision {
        generation: decision.map_or(0, |decision| decision.trace().generation),
        profile: decision.and_then(profile_of),
        outcome: decision.map_or("refused", outcome),
        mode: "embedded",
    }
}

/// The next ask's process-local number, in eight bytes. The gateway assigns
/// it independently of any request id supplied by a client.
///
/// A thread counts its own asks under a number it takes once: the high 24
/// bits name the thread and the low 40 count its asks. So no word is
/// written by two asks of different threads.
pub(crate) fn mark() -> [u8; 8] {
    static THREADS: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static NEXT: Cell<u64> = Cell::new(THREADS.fetch_add(1, Ordering::Relaxed) << 40);
    }
    NEXT.with(|next| {
        let mark = next.get();
        next.set(mark.wrapping_add(1));
        mark.to_be_bytes()
    })
}

/// What an attempt states of its request.
///
/// A stage builds its ask over [`Ask::of`] and names the facts it holds:
/// `Ask { streaming, tried, ..Ask::of(id, model) }`. A fact the policy engine
/// learns to read later is then stated by the stages that hold it, and breaks
/// no other.
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
    /// The attempt's prompt as its route holds it, where it holds one: lent
    /// to the engine of this process for its record of the prefixes it
    /// sent, and to nothing else. No fact of the request holds it. A
    /// followed controller's engine reads nothing here.
    pub prompt: Option<&'a Prompt<'a>>,
    /// The pool of the route that asks: the registry's list a regular
    /// attempt is placed over, where the policy engine decides in this
    /// process. The route states the pool its classic arm reads. An ask that
    /// states none is decided over no worker there. A followed controller's
    /// engine reads its own rows and nothing here.
    pub pool: Option<RoutingPool>,
    /// The workers the request may take, where its route narrowed them
    /// ahead of the ask by the test its classic arm asks of a candidate: a
    /// part of the pool's list, in the list's order. The engine of this
    /// process decides over them alone and keeps nothing of the list. A
    /// route that narrows nothing states none, and its ask is decided over
    /// the pool. A followed controller's engine reads nothing here.
    pub among: Option<&'a Arc<[Arc<dyn Worker>]>>,
    /// The snapshot of the model's workers the route read for this attempt,
    /// where it read one for a rule of its own. The engine of this process
    /// decides the attempt's first ask over the pool's list in it, so a
    /// request reads the registry once. A route that read none states none,
    /// and the ask reads the registry. A followed controller's engine reads
    /// nothing here.
    pub snapshot: Option<&'a Arc<ModelWorkerSnapshot>>,
}

impl<'a> Ask<'a> {
    /// An ask for `model` under the request's `id` that states no other
    /// fact: a single request of no stated prompt size, not streamed, no
    /// input of a routing key, no media, no stated wire, nothing tried, no
    /// stated attempt, no hashes, no prompt lent, no second ask of a decode,
    /// no pool, no narrowed list and no snapshot.
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
            prompt: None,
            pool: None,
            among: None,
            snapshot: None,
        }
    }

    /// The request's two keys by the sources its binding names in `spec`.
    /// Its routing key is the first session source that holds a value, with
    /// the source it came from
    /// ([`KeySources::derive`](smg_policy_controller::document::KeySources::derive)):
    /// `None` where the binding names no source, or none of them holds a
    /// value. Its client's key is read where it has no routing key
    /// ([`KeySources::derive_client`](smg_policy_controller::document::KeySources::derive_client)),
    /// and is `0` where it carries none.
    ///
    /// The routing key borrows the document. The door hands the ask the
    /// world it decides under, and the keys are derived there
    /// ([`DecisionGuard::decide_under`]).
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
            hashes.write_to(&mut facts);
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
    /// The list the ask is decided over, where the engine of this process
    /// decides it: the policy engine's value of the ask's pool.
    members: Option<&'r PoolMembers>,
    /// The load monitor's snapshot for this ask, where the engine of this
    /// process reads load reports: lent once an ask ([`loads::Loads`]).
    loads: Option<&'r Arc<LoadSnapshot>>,
    /// Where the key of the last making came from, under the label the
    /// gateway counts it by. The door may make the request a second time,
    /// behind a lease the store refused: the policy engine counts after the
    /// decision, once.
    source: Cell<Option<&'static str>>,
}

impl RequestMaker for Made<'_, '_> {
    fn model(&self) -> &str {
        self.ask.model
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        let (derived, client_key) = self.ask.keys(spec);
        self.source.set(derived.map(|derived| match derived.source {
            KeySource::Header(_) => "header",
            KeySource::Rid => "rid",
        }));
        let key = derived.map_or(Identity::NONE, |derived| derived.key);
        let mut request = self.ask.request(key, client_key, self.missed);
        // The document's word, like the keys: read from the generation
        // that decides.
        request.constraints.decode_later = self.ask.decode_later(spec);
        request
    }

    // An ask states no subset hint, no hard demand and no lifted veto, and
    // its host writes its standing vetoes into no mask: a first attempt
    // that names nothing tried or missed and asks no decode again is plain.
    // The size is the one the request made would state.
    fn plain(&self) -> Option<Plain> {
        let ask = self.ask;
        let first = ask.tried.is_empty() && self.missed.is_empty() && ask.attempt.unwrap_or(0) == 0;
        (first && !ask.decode_again).then(|| Plain {
            phase: ask.phase,
            wires: ask.wires,
            prompt_tokens: ask.hashes.map_or(ask.prompt_tokens, |hashes| {
                hashes.prompt_tokens_or(ask.prompt_tokens)
            }),
        })
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.members.map_or(WorkerLists::None, WorkerLists::Single)
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads
    }

    // The prompt its route stated on the ask, where it stated one.
    fn prompt(&self) -> Option<Prompt<'_>> {
        self.ask.prompt.copied()
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
    Refused(DecisionRefused),
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
    /// | The door made no decision | 503 `policy_controller_unavailable`, with a retry hint |
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
    handle: DecisionGuard<Arc<Decider>>,
}

/// The engine a gateway's asks reach.
pub(crate) enum Decider {
    /// The engine over the policy store that follows a controller.
    Followed(StoreBackedEngine),
    /// The engine of this process, over the gateway's own worker objects.
    Own(Arc<OwnEngine>),
}

impl Decider {
    /// The engine over the policy store, where one stands.
    fn followed(&self) -> Option<&StoreBackedEngine> {
        match self {
            Self::Followed(door) => Some(door),
            Self::Own(_) => None,
        }
    }

    /// The policy store, where the engine decides over one.
    fn store(&self) -> Option<&Arc<MemoryStore>> {
        self.followed().map(StoreBackedEngine::store)
    }
}

// Each call goes to the engine that stands, by one match.
impl DecisionEngine for Decider {
    #[inline]
    fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<Decided, DecisionRefused> {
        match self {
            Self::Followed(door) => DecisionEngine::decide_under(door, gateway, maker, now),
            Self::Own(engine) => engine.decide_under(gateway, maker, now),
        }
    }

    #[inline]
    fn decide_decode_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        match self {
            Self::Followed(door) => {
                DecisionEngine::decide_decode_under(door, gateway, maker, held, now)
            }
            Self::Own(engine) => engine.decide_decode_under(gateway, maker, held, now),
        }
    }

    fn decide_encode_pair(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        match self {
            Self::Followed(door) => {
                DecisionEngine::decide_encode_pair(door, gateway, request, items, placed, now)
            }
            Self::Own(engine) => engine.decide_encode_pair(gateway, request, items, placed, now),
        }
    }

    #[inline]
    fn say(&self, gateway: GatewayId, held: &HeldDecision<'_>, said: Said, now: Instant) {
        match self {
            Self::Followed(door) => DecisionEngine::say(door, gateway, held, said, now),
            Self::Own(engine) => engine.say(gateway, held, said, now),
        }
    }

    fn sweep(&self, now: Instant) -> SweepOutcome {
        match self {
            Self::Followed(door) => DecisionEngine::sweep(door, now),
            Self::Own(engine) => engine.sweep(now),
        }
    }

    fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R {
        match self {
            Self::Followed(door) => door.with_spec(read),
            Self::Own(engine) => engine.with_spec(read),
        }
    }

    fn prompt_hashes(
        &self,
        model: &str,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
    ) -> Hashes {
        match self {
            Self::Followed(door) => door.prompt_hashes(model, tokens, text, seeds),
            Self::Own(engine) => engine.prompt_hashes(model, tokens, text, seeds),
        }
    }

    fn pair_refusal(&self, model: &str, wires: WireModes) -> Option<Refusal> {
        match self {
            Self::Followed(door) => door.pair_refusal(model, wires),
            Self::Own(engine) => engine.pair_refusal(model, wires),
        }
    }

    fn faults(&self) -> u64 {
        match self {
            Self::Followed(door) => DecisionEngine::faults(door),
            Self::Own(engine) => engine.faults(),
        }
    }

    fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64 {
        match self {
            Self::Followed(door) => DecisionEngine::shadowed(door, binding, kind),
            Self::Own(engine) => engine.shadowed(binding, kind),
        }
    }
}

/// The gateway's policy engine: the engine its asks reach, and where the
/// worker of a target is found.
pub(crate) struct PolicyEngine {
    door: Arc<Decider>,
    workers: Workers,
    /// The load monitor's feed, where the engine of this process decides
    /// under a document that reads load reports
    /// ([`Self::read_loads`]).
    loads: loads::Loads,
    /// The memberships whose hash rings stand, where the engine of this
    /// process places by a ring.
    standing: Standing,
}

/// The memberships whose hash rings stand, by model: for each model the
/// engine of this process places by a ring, the registry's value of the
/// last membership whose rings the fill built ([`PolicyEngine::fill_rings`]).
///
/// A membership change publishes another list, and that list's ring is
/// built off the path of a request. An attempt that comes ahead of it is
/// decided over the membership kept here, by its ring, and takes a worker
/// of the list in force ([`stood_before`]): a key keeps its worker across
/// the change, a key whose worker left goes to the next member of the ring,
/// and a worker that joined takes no keyed request until its list's ring
/// stands.
///
/// The fill alone writes, a whole map a time. A request reads with no lock,
/// and only where the list in force has no ring yet.
struct Standing {
    /// Whether the document places a single request of some model by a
    /// ring, read once where the object is made. `false` under a followed
    /// controller and under a document with no ring: nothing is kept and no
    /// request reads the map.
    kept: bool,
    memberships: ArcSwap<FxHashMap<Box<str>, Arc<ModelWorkerSnapshot>>>,
}

impl Standing {
    fn new(kept: bool) -> Self {
        Self {
            kept,
            memberships: ArcSwap::default(),
        }
    }

    /// Keeps `membership`, whose rings the fill has built, as `model`'s. A
    /// membership with no worker keeps none: the model's next list waits
    /// for a ring of its own.
    fn keep(&self, model: &str, membership: &Arc<ModelWorkerSnapshot>) {
        self.memberships.rcu(|memberships| {
            let mut memberships = FxHashMap::clone(memberships);
            if membership.is_empty() {
                memberships.remove(model);
            } else {
                memberships.insert(model.into(), Arc::clone(membership));
            }
            memberships
        });
    }
}

/// Where the policy engine finds the worker of a target.
enum Workers {
    /// The object the feed's holder laid for the target's row.
    Held(Arc<Objects<ModelWorkerSnapshot>>),
    /// The registry's worker at the row's key, for a bed that registers its
    /// workers by hand. A key is not one life of a worker.
    #[cfg(test)]
    AtKey(Arc<WorkerRegistry>),
    /// The registry's own lists, where the engine of this process decides
    /// over them: a target names its worker by its place in the list its
    /// ask was made over.
    Listed(Arc<WorkerRegistry>),
}

impl PolicyEngine {
    /// The policy engine over the controller's `store` and the holder of its
    /// worker `objects`.
    pub(crate) fn new(store: Arc<MemoryStore>, objects: Arc<Objects<ModelWorkerSnapshot>>) -> Self {
        Self {
            door: Arc::new(Decider::Followed(StoreBackedEngine::new(store))),
            workers: Workers::Held(objects),
            loads: loads::Loads::default(),
            standing: Standing::new(false),
        }
    }

    /// The policy engine of a gateway that decides in its own process:
    /// `engine` decides over the lists `registry` publishes, and no store
    /// stands between them.
    pub(crate) fn own(engine: Arc<OwnEngine>, registry: Arc<WorkerRegistry>) -> Self {
        let rings = engine.spec().engine().needs_for_phase(Phase::Single).ring;
        Self {
            door: Arc::new(Decider::Own(engine)),
            workers: Workers::Listed(registry),
            loads: loads::Loads::default(),
            standing: Standing::new(rings),
        }
    }

    /// Hands the policy engine the load monitor's feed, at the gateway's
    /// start: the one feed the classic policies read. The engine of this
    /// process reads its workers' reports there, one look an ask, where
    /// its document reads load reports. Under any other document, and
    /// where the gateway follows a controller, the feed is not kept: the
    /// controller's rows carry the reports there.
    pub(crate) fn read_loads(&self, feed: LoadReceiver) {
        if let Decider::Own(engine) = &*self.door {
            if engine.spec().needs().loads {
                self.loads.read(feed);
            }
        }
    }

    /// Whether the engine was handed the monitor's feed and kept it.
    #[cfg(test)]
    pub(crate) fn reads_loads(&self) -> bool {
        self.loads.reads_a_feed()
    }

    /// One pass of the door's clock at `now`: the call of the policy engine's
    /// timer ([`upkeep`](super::upkeep)), about once a second.
    pub(crate) fn sweep(&self, now: Instant) -> SweepOutcome {
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

    /// Whether the engine of this process decides: `false` where the asks
    /// reach the engine over the store that follows a controller.
    pub(crate) fn in_process(&self) -> bool {
        matches!(*self.door, Decider::Own(_))
    }

    /// Whether the placement of a single request reads the request's body
    /// under the document in force: some stage that may decide one reads
    /// the prompt. The document's needs say it, as the engine folded them
    /// over its bindings when the document was applied: nothing is worked
    /// out here.
    ///
    /// A stage reads the prompt as its block hashes, which a route makes
    /// from the body's token ids or text, or as its size, which the route
    /// counts where it makes the hashes (a stage that places by the size
    /// declares it, as the bucket stage does). A valid tokens hint states
    /// the ids in a header, and with them their count, so `hinted` lifts
    /// the read: it is asked only where a stage reads the hashes or the
    /// size. The hint's count then stands for the size, 512 ids at most,
    /// as a hint lifts every classic policy's read of the body. The text
    /// itself, which no stage of this build reads, is in the body alone.
    ///
    /// A key source that is the request's own id reads the body too. The
    /// routing settings name one under the sticky override alone, and the
    /// route holds the body for that setting by its own name.
    pub(crate) fn reads_body(&self, hinted: impl FnOnce() -> bool) -> bool {
        let needs = self
            .door
            .with_spec(|spec| spec.engine().needs_for_phase(Phase::Single));
        let prompt = needs.token_hashes || needs.text_hashes || needs.prompt_size;
        needs.text || (prompt && !hinted())
    }

    /// An attempt's prompt hashes for `model`, under the generation in force.
    /// The stage holding the tokens calls this once before the first ask.
    /// Each ask borrows the same result through [`Ask::hashes`].
    pub(crate) fn hashes(&self, model: &str, prompt: &Prompt<'_>) -> Hashes {
        hashes::hashed(prompt, |tokens, text, seeds| {
            self.door.prompt_hashes(model, tokens, text, seeds)
        })
    }

    /// Fills the hash ring of each list a regular request of `model` is
    /// placed over ([`RING_POOLS`]), where the engine of this process
    /// places it by a ring ([`OwnEngine::fill_ring`]), and says whether the
    /// engine of this process places the gateway's requests. `false` under
    /// a followed controller, whose engine reads its own rows: nothing is
    /// filled.
    ///
    /// For the task that hears the registry's membership events, off the
    /// path of a request: a fill hashes every worker's positions and sorts
    /// them, and no request builds a ring or waits for one. The lists are
    /// the ones the registry publishes for `model` now, each with what the
    /// engine keeps of it in the snapshot's cell, as a request finds it.
    /// A list that holds no worker takes no ring. Under a document that
    /// places the model by no ring nothing is made of its lists here.
    ///
    /// The membership whose rings were built is kept as the model's
    /// ([`Standing`]): the attempts that come ahead of the next
    /// membership's rings are decided over it.
    pub(crate) fn fill_rings(&self, model: &str) -> bool {
        let (Workers::Listed(registry), Decider::Own(engine)) = (&self.workers, &*self.door) else {
            return false;
        };
        if engine.reads_ring(model) {
            let snapshot = registry.get_routing_snapshot(model);
            for pool in RING_POOLS {
                if !snapshot.pool(pool).is_empty() {
                    engine.fill_ring(model, members_of(engine, &snapshot, pool));
                }
            }
            self.standing.keep(model, &snapshot);
        }
        true
    }

    /// Whether the engine's ring stands for each list a regular request of
    /// `model` is placed over, where the document places it by a ring: the
    /// requests of `model` walk the ring of the membership that stands. For
    /// a caller that changed the membership and waits for the placements to
    /// follow, as a test bed does.
    #[cfg(any(test, feature = "test-util"))]
    pub(crate) fn rings_stand(&self, model: &str) -> bool {
        let (Workers::Listed(registry), Decider::Own(engine)) = (&self.workers, &*self.door) else {
            return true;
        };
        if !engine.reads_ring(model) {
            return true;
        }
        let snapshot = registry.get_routing_snapshot(model);
        RING_POOLS.into_iter().all(|pool| {
            let members = snapshot.pool_members_cell(pool).get();
            snapshot.pool(pool).is_empty()
                || members.is_some_and(|members| members.ring().get().is_some())
        })
    }

    /// The policy engine of a bed that registers its workers by hand in
    /// `registry`.
    #[cfg(test)]
    pub(crate) fn at_keys(store: Arc<MemoryStore>, registry: Arc<WorkerRegistry>) -> Self {
        Self {
            door: Arc::new(Decider::Followed(StoreBackedEngine::new(store))),
            workers: Workers::AtKey(registry),
            loads: loads::Loads::default(),
            standing: Standing::new(false),
        }
    }

    /// The gateway's worker of a target, found by the row's id: the one
    /// function that resolves a row of the store to a worker. `None` where
    /// the holder has laid no object for the row or the object has left the
    /// registry. So a row the store shows ahead of its object, as a
    /// replacement inside its own write, takes no request. `None` too where
    /// the engine of this process decided: its targets name a worker by its
    /// place in the list its ask was made over ([`Self::place_listed`]).
    fn worker(&self, id: WorkerId) -> Option<Arc<dyn Worker>> {
        match &self.workers {
            Workers::Held(objects) => objects.registry().get(&objects.object(id)?),
            #[cfg(test)]
            Workers::AtKey(registry) => {
                use smg_policy_controller::store::WorkersStore;
                let row = WorkersStore::get(&**self.door.store()?, id)?;
                registry.get_by_url(&row.facts.key)
            }
            Workers::Listed(_) => None,
        }
    }

    /// The registry the gateway's workers stand in: the holder's, a bed's
    /// own, or the one whose lists the engine decides over.
    fn registry(&self) -> &WorkerRegistry {
        match &self.workers {
            Workers::Held(objects) => objects.registry(),
            #[cfg(test)]
            Workers::AtKey(registry) => registry,
            Workers::Listed(registry) => registry,
        }
    }

    /// The name of the profile `decision` was made under, for the counter
    /// that asks it the first time it sees the profile.
    fn named(&self, decision: Option<&Decision>) -> Option<Arc<str>> {
        self.profile_name(decision?.trace())
    }

    fn profile_name(&self, trace: &Trace) -> Option<Arc<str>> {
        self.profile_named(trace.generation, trace.profile)
    }

    /// The name of the profile at `profile` of the document of
    /// `generation`: `None` where another generation is in force.
    fn profile_named(&self, generation: u64, profile: u16) -> Option<Arc<str>> {
        self.door.with_spec(|spec| {
            if spec.generation() != generation {
                return None;
            }
            let profile = spec.engine().profiles().get(usize::from(profile))?;
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
        let refusal = self.door.pair_refusal(ask.model, ask.wires)?;
        Some(Unplaced::NoCompatiblePair {
            prefill: refusal.prefill,
            decode: refusal.decode,
            mismatches: refusal.mismatches,
        })
    }

    fn trace_decision(&self, decision: &Decision, worker: Option<&Arc<dyn Worker>>) {
        cache_trace::decided(|| self.traced(decision, worker));
    }

    /// The cache trace's line of a place the engine of this process
    /// answered ([`Decided::Alone`]), with `worker` the worker at the
    /// place: `taken`, or passed over by its route's rule. The decision the
    /// line shows is written here, where the process captures the trace:
    /// nothing else reads one of a place.
    fn trace_place(&self, pick: AlonePick, worker: &Arc<dyn Worker>, taken: bool) {
        cache_trace::decided(|| {
            let id = worker_id(worker.as_ref());
            let decision = pick.decision(id, worker_rank(worker.as_ref()));
            self.traced(&decision, taken.then_some(worker))
        });
    }

    /// What the cache trace shows of `decision` and of the `worker` taken
    /// for it.
    fn traced(
        &self,
        decision: &Decision,
        worker: Option<&Arc<dyn Worker>>,
    ) -> cache_trace::Decided {
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
                LiftedVeto::None => "none",
                LiftedVeto::Overload => "overload",
                LiftedVeto::Liveness => "liveness",
            },
            shed,
            worker: worker.cloned(),
        }
    }

    /// The ask: one decision for `ask` at `now` through the controller's
    /// handle, naming the targets the gateway `missed` at this attempt's
    /// earlier commits.
    ///
    /// Resolves both workers of each pair before admission can lock the
    /// queue. The door in this process decides on the calling thread.
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
    ///
    /// The engine of this process is handed no list here. A regular attempt
    /// reaches it over the registry's list by [`Self::place_listed`].
    #[expect(
        clippy::unused_async,
        reason = "a controller at an address is asked behind this call, over its session"
    )]
    async fn ask_as(
        &self,
        ask: &Ask<'_>,
        missed: &[WorkerId],
        first: bool,
        now: Instant,
    ) -> Result<Asked, Unplaced> {
        let made = Made {
            ask,
            missed,
            members: None,
            loads: None,
            source: Cell::new(None),
        };
        let started = Instant::now();
        let decided = DecisionGuard::decide_under(Arc::clone(&self.door), GATEWAY, &made, now);
        let seconds = started.elapsed().as_secs_f64();
        // A key's source counts once an attempt, at its first ask, as a
        // keyed selection of the gateway's own policies counts it: an ask
        // again, behind a missed commit or at a pair's later wake, is the
        // same attempt's.
        if let Some(source) = made.source.get().filter(|_| first) {
            Metrics::record_routing_key_source(source);
        }
        let decision = decided.as_ref().ok().map(DecisionGuard::decision);
        Metrics::record_policy_decision(
            ask.model,
            labels(decision),
            || self.named(decision),
            seconds,
        );
        let handle = decided.map_err(Unplaced::Refused)?;
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
                })
            }
            Decision::Shed(shed) => {
                self.trace_decision(&decision, None);
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

    /// The commit of a route with a rule of its own: as
    /// [`PolicyEngine::commit`], and a target whose worker `holds` refuses is
    /// passed over like one the gateway cannot send to.
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
            ..
        } = asked;
        for (index, target) in targets.iter().enumerate() {
            let sendable = self
                .worker(target.id)
                .filter(|worker| routable_but_for_veto(worker.as_ref()) && holds(worker.as_ref()));
            if let Some(worker) = sendable {
                handle.take(index);
                self.trace_decision(handle.decision(), Some(&worker));
                return Some(Placed {
                    worker,
                    named: Named::of(target),
                    pair: None,
                    handle: Some(handle),
                });
            }
        }
        handle.released(ReleaseReason::CommitFailed, now);
        missed.extend(targets.iter().map(|target| target.id));
        self.trace_decision(handle.decision(), None);
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
    /// [`PolicyEngine::commit_pair`], and a target either of whose workers
    /// `holds` refuses is passed over like one the gateway cannot send to. The
    /// rule is the route's, as [`PolicyEngine::commit_where`] says of a regular
    /// one.
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
                named: Named::of(target),
                pair: Some(Box::new(PlacedPair {
                    kept: None,
                    decode: Arc::clone(decode),
                    side: pair,
                    targets,
                    index,
                })),
                handle: Some(handle),
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
    /// [`PolicyEngine::place_pair`], over [`PolicyEngine::commit_pair_where`].
    /// The pass that reads which prefills are full and the probe of a request
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
    /// own: as [`PolicyEngine::place`], over [`PolicyEngine::commit_where`].
    /// Where the engine of this process decides, the ask and the commit are
    /// one pass over the registry's list ([`Self::place_listed`]).
    pub(crate) async fn place_where(
        &self,
        ask: &Ask<'_>,
        holds: impl Fn(&dyn Worker) -> bool + Send + Sync,
        now: Instant,
    ) -> Result<Placed, Unplaced> {
        {
            // The attempt's first decision is dated at the instant handed.
            // A later ask of the attempt reads the clock.
            let handed = Cell::new(Some(now));
            let clock = || handed.take().unwrap_or_else(Instant::now);
            if let Some(placed) = self.place_here(ask, &holds, clock) {
                return placed;
            }
        }
        self.place_followed(ask, &holds, now).await
    }

    /// [`Self::place_where`] for a route that hands a clock and no instant.
    /// `clock` is read where the attempt's decision is dated, and not at
    /// all where the engine of this process answers a place: a request
    /// nothing is dated for reads no clock.
    pub(crate) async fn place_at(
        &self,
        ask: &Ask<'_>,
        holds: impl Fn(&dyn Worker) -> bool + Send + Sync,
        clock: impl Fn() -> Instant + Send + Sync,
    ) -> Result<Placed, Unplaced> {
        if let Some(placed) = self.place_here(ask, &holds, &clock) {
            return placed;
        }
        self.place_followed(ask, &holds, clock()).await
    }

    /// A regular attempt's worker with no wait, where the engine of this
    /// process decides: [`Self::place_at`] as one pass on the calling
    /// thread ([`Self::place_listed`]), for a route that asks while it
    /// holds a borrow no wait may hold. `None` under a followed
    /// controller's engine, whose ask waits.
    pub(crate) fn place_here(
        &self,
        ask: &Ask<'_>,
        holds: impl Fn(&dyn Worker) -> bool,
        clock: impl Fn() -> Instant,
    ) -> Option<Result<Placed, Unplaced>> {
        let (Workers::Listed(registry), Decider::Own(engine)) = (&self.workers, &*self.door) else {
            return None;
        };
        Some(self.place_listed(registry, engine, ask, &holds, &clock))
    }

    /// The ask and the commit of a followed controller's engine, three
    /// asks at most.
    async fn place_followed(
        &self,
        ask: &Ask<'_>,
        holds: &(impl Fn(&dyn Worker) -> bool + Send + Sync),
        now: Instant,
    ) -> Result<Placed, Unplaced> {
        let mut missed: Vec<WorkerId> = Vec::new();
        for _ in 0..ASKS {
            let asked = self.ask(ask, &missed, now).await?;
            if let Some(placed) = self.commit_where(asked, holds, &mut missed, now) {
                return Ok(placed);
            }
        }
        Err(Unplaced::NoTargetHolds)
    }

    /// A regular attempt's worker where `engine`, the engine of this
    /// process, decides over the lists `registry` publishes: the ask and
    /// the commit of [`PolicyEngine::place_where`] in one pass on the
    /// calling thread, three asks at most.
    ///
    /// Each ask reads the list the registry publishes for the ask's pool
    /// once: the membership a classic arm reads. An attempt's first ask
    /// reads it in the snapshot its route read, where the ask states one
    /// ([`Ask::snapshot`]), so such a request reads the registry once. What
    /// the engine keeps of the list stands in the snapshot's cell beside
    /// it, made by the list's first ask. An ask that states no pool is
    /// decided over no worker. An ask that states the workers it may take
    /// ([`Ask::among`]) is decided over them alone, and the registry is not
    /// read for it.
    ///
    /// The engine answers one of two things ([`Decided`]). A place is a
    /// worker of the list and nothing else: no stage of the document takes
    /// a word of the request, so the placement holds no guard, names no id
    /// and reads no clock, and its decision is counted and not timed. A
    /// whole decision is dated by one read of `clock`, where the engine
    /// dates it, and timed from that instant. Its targets are walked best
    /// first: a target's worker is the one at the target's place in the
    /// list, and the first the gateway can send to is taken on one read of
    /// the worker's state. It rides in a guard where
    /// it is leased, and where it is not, nothing is owed of it. Where no
    /// target holds, the lease is given up by name and the targets join the
    /// next ask's missed.
    ///
    /// The route's rule (`holds`) is asked of a worker picked. The first
    /// time it refuses one, the attempt's later asks are decided over the
    /// workers of the list the rule holds, and over no other: the engine
    /// picks among what the route can send to, as a classic pass narrows
    /// its candidates ahead of its policy. Such a list is a part of its
    /// pool ([`PoolMembers::part`]): nothing is kept of it, and the
    /// requests narrowed to one part take its workers in turn.
    ///
    /// Under a document that places by a hash ring, an attempt's first ask
    /// can come ahead of the ring of the list in force, which is built off
    /// the path of a request. It is then decided over the last membership
    /// whose rings stand, and a target's worker is the one the list in
    /// force holds at that worker's address ([`stood_before`]). Where no
    /// target holds, the next ask is decided over the list in force. An ask
    /// over a part of a pool, or over the workers on its wires, is decided
    /// over that list as it is.
    fn place_listed(
        &self,
        registry: &WorkerRegistry,
        engine: &OwnEngine,
        ask: &Ask<'_>,
        holds: &impl Fn(&dyn Worker) -> bool,
        clock: &impl Fn() -> Instant,
    ) -> Result<Placed, Unplaced> {
        let mut missed: Vec<WorkerId> = Vec::new();
        // The workers the route's rule holds, from its first refusal of a
        // worker picked.
        let mut ruled: Option<Arc<[Arc<dyn Worker>]>> = None;
        for asked in 0..ASKS {
            let first = asked == 0;
            let among = ruled.as_ref().or(ask.among);
            // The model's workers as the registry lists them: the snapshot
            // the route read, where the attempt's first ask states one,
            // else one read here.
            let listed = || match ask.snapshot.filter(|_| first) {
                Some(stated) => Cow::Borrowed(stated),
                None => Cow::Owned(registry.get_routing_snapshot(ask.model)),
            };
            let snapshot = match among {
                Some(_) => None,
                None => ask.pool.map(|pool| (listed(), pool)),
            };
            let kept = snapshot
                .as_ref()
                .map(|(snapshot, pool)| members_of(engine, snapshot, *pool));
            // A list its route or its route's rule narrowed is a part of
            // its pool, and the engine's for this ask alone: its turns are
            // its set's own.
            let narrowed = among.map(PoolMembers::part);
            let whole = narrowed.as_ref().or(kept);
            // An ask held to wires the list's workers do not all serve (a
            // retry pinned to its retained plan's wire, over a pool of two
            // wires) is made over a list of its own, the workers on those
            // wires, as the classic arm makes one. Nothing is kept of it.
            let dialed = whole
                .filter(|pool| !ask.wires.is_empty() && !ask.wires.contains(pool.wires()))
                .map(|pool| {
                    let dialed: Arc<[Arc<dyn Worker>]> = pool
                        .workers()
                        .iter()
                        .filter(|worker| {
                            WireModes::from(*worker.connection_mode()).intersects(ask.wires)
                        })
                        .cloned()
                        .collect();
                    PoolMembers::part(&dialed)
                });
            let members = dialed.as_ref().or(whole);
            let list = members.map(PoolMembers::workers);
            // The monitor's snapshot for this ask, where the document reads
            // load reports: lent once, and every record of the decision is
            // read in it. The ask's instant was read before it.
            let lent = self.loads.lend();
            // The first ask over the registry's own list is decided over
            // the membership before it, where this list's ring is not
            // built yet.
            let stood = match (kept, &snapshot) {
                (Some(in_force), Some((_, pool))) if first && dialed.is_none() => {
                    stood_before(&self.standing, ask.model, *pool, in_force)
                }
                _ => None,
            };
            let members = stood.as_ref().and_then(Stood::members).or(members);
            // The worker at a place of the list the ask is decided over,
            // as the list in force holds it.
            let at = |place: u32| match &stood {
                Some(stood) => stood.worker(place),
                None => list?.get(place as usize),
            };
            let made = Made {
                ask,
                missed: &missed,
                members,
                loads: lent.snapshot(),
                source: Cell::new(None),
            };
            // The decision's instant, where the engine dated one: a place
            // reads no clock.
            let dated = Cell::new(None);
            let decided = engine.decide_at(&made, || {
                let now = clock();
                dated.set(Some(now));
                now
            });
            // A key's source counts once an attempt, at its first ask, as
            // an ask of a followed controller counts it.
            if let Some(source) = made.source.get().filter(|_| first) {
                Metrics::record_routing_key_source(source);
            }
            let (decided, later) = match decided {
                Ok(Decided::Alone(pick)) => {
                    Metrics::count_policy_decision(
                        ask.model,
                        PolicyDecision {
                            generation: pick.generation,
                            profile: Some(pick.profile),
                            outcome: "pick",
                            mode: "embedded",
                        },
                        || self.profile_named(pick.generation, pick.profile),
                    );
                    // The engine picked the place in this list, on the
                    // read of the worker that found it eligible: the
                    // worker is read no second time.
                    let Some(worker) = at(pick.place) else {
                        continue;
                    };
                    if holds(worker.as_ref()) {
                        self.trace_place(pick, worker, true);
                        return Ok(Placed {
                            worker: Arc::clone(worker),
                            named: Named::ByWorker,
                            pair: None,
                            handle: None,
                        });
                    }
                    self.trace_place(pick, worker, false);
                    match list.filter(|_| ruled.is_none()) {
                        Some(list) => ruled = Some(held_by(list, holds)?),
                        // A rule that refuses a worker it held.
                        None => missed.push(worker_id(worker.as_ref())),
                    }
                    continue;
                }
                Ok(Decided::Whole(decided, later)) => (decided, later),
                Err(refused) => {
                    // A refusal ahead of the decision's instant ran no
                    // stage: it is counted and not timed.
                    match dated.get() {
                        Some(at) => Metrics::record_policy_decision(
                            ask.model,
                            labels(None),
                            || None,
                            at.elapsed().as_secs_f64(),
                        ),
                        None => Metrics::count_policy_decision(ask.model, labels(None), || None),
                    }
                    return Err(Unplaced::Refused(refused));
                }
            };
            let now = dated.get().unwrap_or_else(clock);
            let seconds = now.elapsed().as_secs_f64();
            Metrics::record_policy_decision(
                ask.model,
                labels(Some(&decided.decision)),
                || self.named(Some(&decided.decision)),
                seconds,
            );
            // A decision is leased where a stage takes a word of its
            // request, and rides in its guard from here: a walk that ends
            // early gives the lease up at the guard's drop. One that is
            // not leased owes the engine nothing, and no guard is made.
            let mut handle = None;
            let decision = if decided.lease.is_some() {
                let door = Arc::clone(&self.door);
                handle
                    .insert(DecisionGuard::holding(door, GATEWAY, decided, later, now))
                    .decision()
            } else {
                &decided.decision
            };
            let mut refused = false;
            let taken = match decision {
                Decision::Pick(pick) => {
                    let taken = pick.targets.iter().enumerate().find_map(|(index, target)| {
                        let worker = at(target.position)?;
                        // Ready and let through by its circuit breaker, as
                        // a commit over a store's rows asks
                        // (`routable_but_for_veto`), on one read.
                        let state = worker.routing_state();
                        if !(state.healthy && state.can_execute) {
                            return None;
                        }
                        if !holds(worker.as_ref()) {
                            refused = true;
                            return None;
                        }
                        Some((index, Arc::clone(worker), Named::of(target)))
                    });
                    if taken.is_none() {
                        missed.extend(pick.targets.iter().map(|target| target.id));
                    }
                    taken
                }
                Decision::Shed(shed) => {
                    let verdict = Unplaced::Shed {
                        reason: shed.reason,
                        emptied_by: shed.emptied_by,
                        retry_after_secs: shed.retry_after_secs,
                    };
                    let unpaired = (shed.reason == ShedReason::NoEndpoints)
                        .then(|| self.unpaired(ask))
                        .flatten();
                    self.trace_decision(decision, None);
                    return Err(unpaired.unwrap_or(verdict));
                }
                Decision::Queue(_) => return Err(Unplaced::Queued),
            };
            let Some((index, worker, named)) = taken else {
                self.trace_decision(decision, None);
                if let Some(handle) = &mut handle {
                    handle.released(ReleaseReason::CommitFailed, now);
                }
                if let Some(list) = list.filter(|_| refused && ruled.is_none()) {
                    ruled = Some(held_by(list, holds)?);
                }
                continue;
            };
            self.trace_decision(decision, Some(&worker));
            // A pick the engine's rule for a pool whose every worker is
            // vetoed made by lifting a veto: counted as the classic
            // placement counts its own steer of such a pool.
            match decision.trace().lifted {
                LiftedVeto::None => {}
                LiftedVeto::Overload => Metrics::record_worker_overload_fallback(STAGE_SELECTION),
                LiftedVeto::Liveness => Metrics::record_worker_liveness_fallback(STAGE_SELECTION),
            }
            if let Some(handle) = &mut handle {
                handle.take(index);
            }
            return Ok(Placed {
                worker,
                named,
                pair: None,
                handle,
            });
        }
        Err(Unplaced::NoTargetHolds)
    }
}

/// The workers of `list` a route's rule `holds`, as a list of their own in
/// the list's order: what an attempt's later asks are decided over once the
/// rule refused a worker picked. A rule that holds no worker of the list
/// leaves the attempt none.
fn held_by(
    list: &[Arc<dyn Worker>],
    holds: &impl Fn(&dyn Worker) -> bool,
) -> Result<Arc<[Arc<dyn Worker>]>, Unplaced> {
    let held: Arc<[Arc<dyn Worker>]> = list
        .iter()
        .filter(|worker| holds(worker.as_ref()))
        .cloned()
        .collect();
    if held.is_empty() {
        return Err(Unplaced::NoTargetHolds);
    }
    Ok(held)
}

/// The pools the engine of this process places a regular request over: the
/// lists whose hash ring the gateway fills ([`PolicyEngine::fill_rings`]).
const RING_POOLS: [RoutingPool; 2] = [RoutingPool::HttpRegular, RoutingPool::GrpcPipelineRegular];

/// What `engine` keeps of `pool`'s list in `snapshot`, the registry's value
/// of one membership: made at the first call for the pool and kept in the
/// snapshot's cell beside the list, for as long as the list stands. The one
/// read of the cell, for a request and for the fill of the list's ring.
fn members_of<'s>(
    engine: &OwnEngine,
    snapshot: &'s ModelWorkerSnapshot,
    pool: RoutingPool,
) -> &'s PoolMembers {
    let list = snapshot.pool(pool);
    snapshot
        .pool_members_cell(pool)
        .get_or_init(|| Box::new(engine.pool_members(list)))
}

/// The membership an attempt's first ask is decided over in the place of
/// `in_force`, what the engine keeps of `pool`'s list of `model` as the
/// registry publishes it: the last one whose rings the fill built
/// ([`Standing`]). `None` where the ask is decided over the list in force:
/// the document places by no ring, the list's own ring stands, or no ring
/// of the model stands yet, as at its first requests.
///
/// One load of the kept map and one handle of the membership, and only for
/// an ask that finds its list's ring not built.
fn stood_before<'f>(
    standing: &Standing,
    model: &str,
    pool: RoutingPool,
    in_force: &'f PoolMembers,
) -> Option<Stood<'f>> {
    if !standing.kept || in_force.ring().get().is_some() {
        return None;
    }
    let membership = Arc::clone(standing.memberships.load().get(model)?);
    let stood = Stood {
        membership,
        pool,
        in_force: in_force.workers(),
    };
    // A membership whose list of this pool took no ring decides nothing.
    stood.members()?.ring().get()?;
    Some(stood)
}

/// A membership whose rings stand, and the list in force that came after
/// it: what an ask ahead of that list's ring is decided over
/// ([`stood_before`]).
struct Stood<'f> {
    membership: Arc<ModelWorkerSnapshot>,
    pool: RoutingPool,
    in_force: &'f [Arc<dyn Worker>],
}

impl<'f> Stood<'f> {
    /// What the engine keeps of the membership's list: the members the ask
    /// is decided over, by their ring and their records.
    fn members(&self) -> Option<&PoolMembers> {
        let members = self.membership.pool_members_cell(self.pool).get()?;
        Some(members)
    }

    /// The worker of the list in force at the address of the membership's
    /// worker at `place`: that worker where it stayed, the one registered
    /// at its address where it was replaced, and `None` where it left. A
    /// worker's id is its address, so the engine knows the two by one id.
    ///
    /// One pass over the list in force by the object, and a second by the
    /// address where the object is not on it.
    fn worker(&self, place: u32) -> Option<&'f Arc<dyn Worker>> {
        let stood = self.members()?.workers().get(place as usize)?;
        let held = self.in_force.iter().find(|held| Arc::ptr_eq(held, stood));
        held.or_else(|| self.in_force.iter().find(|held| held.url() == stood.url()))
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
/// end where the decision is leased.
pub(crate) struct Placed {
    worker: Arc<dyn Worker>,
    /// The worker's id and its rank, as the placement names them.
    named: Named,
    /// A pair's own part, behind one pointer: a regular placement rides in
    /// its router's request context, whose size a case pins.
    pair: Option<Box<PlacedPair>>,
    /// The guard of the decision's lease: the words of the request are
    /// said through it. `None` where the engine of this process leased
    /// nothing, under a document no stage of which takes a word of a
    /// request: nothing is owed of the placement then, and each report
    /// below says nothing. A pair's placement holds one.
    handle: Option<DecisionGuard<Arc<Decider>>>,
}

/// How a placement names its worker's id and data-parallel rank.
#[derive(Clone, Copy, Debug)]
enum Named {
    /// As the target taken names them: the worker's row and the rank the
    /// decision picked.
    ByTarget(WorkerId, u16),
    /// By nothing kept. A place the engine of this process answered names
    /// neither: each is read of the worker where a route asks it, a retry
    /// for the id and a dispatch that pins a rank for the rank.
    ByWorker,
}

impl Named {
    /// The id and the rank `target` names.
    const fn of(target: &Target) -> Self {
        Self::ByTarget(target.id, target.dp_rank)
    }
}

/// A route's rule for a decode its second commit may take, as a kept ask
/// holds it.
type DecodeRule = dyn Fn(&dyn Worker) -> bool + Send + Sync;

/// What a route keeps of a pair's first ask for the second ask of its
/// decode ([`Placed::keep_ask`]), where the two are stages apart and the
/// first ask's borrows end between them: the policy engine that decides, the
/// ask's facts, owned, and the route's own rule at the second commit.
pub(crate) struct KeptAsk {
    policy_engine: Arc<PolicyEngine>,
    id: Box<[u8]>,
    model: Box<str>,
    streaming: bool,
    headers: Option<HeaderMap>,
    rid: Option<Box<str>>,
    prompt_tokens: u32,
    media: MediaCounts,
    wires: WireModes,
    tried: Box<[WorkerId]>,
    attempt: Option<u8>,
    hashes: Option<Hashes>,
    holds: Box<DecodeRule>,
}

impl KeptAsk {
    /// The facts of `ask`, owned, for a second ask by `policy_engine`, with no
    /// hashes: the ask borrows the attempt's, and they move in behind its
    /// last use ([`Self::with_hashes`]). `holds` is the route's rule for a
    /// decode the second commit may take.
    pub(crate) fn of(
        policy_engine: &Arc<PolicyEngine>,
        ask: &Ask<'_>,
        holds: impl Fn(&dyn Worker) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            policy_engine: Arc::clone(policy_engine),
            id: ask.id.into(),
            model: ask.model.into(),
            streaming: ask.streaming,
            headers: ask.headers.cloned(),
            rid: ask.rid.map(Into::into),
            prompt_tokens: ask.prompt_tokens,
            media: ask.media,
            wires: ask.wires,
            tried: ask.tried.into(),
            attempt: ask.attempt,
            hashes: None,
            holds: Box::new(holds),
        }
    }

    /// The attempt's `hashes`, moved in: no ask reads them between the two
    /// decisions.
    pub(crate) fn with_hashes(mut self, hashes: Hashes) -> Self {
        self.hashes = Some(hashes);
        self
    }

    /// The pair's ask again, as its first ask stated it.
    fn ask(&self) -> Ask<'_> {
        Ask {
            id: &self.id,
            model: &self.model,
            streaming: self.streaming,
            headers: self.headers.as_ref(),
            rid: self.rid.as_deref(),
            phase: Phase::Prefill,
            decode_again: true,
            prompt_tokens: self.prompt_tokens,
            media: self.media,
            wires: self.wires,
            tried: &self.tried,
            attempt: self.attempt,
            hashes: self.hashes.as_ref(),
            // A prompt is its route's to lend, for as long as the route
            // holds it: a kept ask owns its facts, and lends none.
            prompt: None,
            // A pair is placed over its family's two lists, and no pool
            // of a regular route.
            pool: None,
            among: None,
            snapshot: None,
        }
    }
}

/// What a placed pair holds beside its prefill: the decode worker with the
/// decode side of the target taken, and the pick's targets with that
/// target's index, for the next decode of the same prefill.
struct PlacedPair {
    /// The first ask as its route kept it for the second ask of the decode.
    kept: Option<Box<KeptAsk>>,
    decode: Arc<dyn Worker>,
    side: Pair,
    targets: Targets,
    index: usize,
}

impl Placed {
    /// A placement that is `worker` and nothing else: it holds no
    /// decision's guard, so nothing is owed of it ([`Self::owes`]). What a
    /// placement by the classic policies hands back.
    #[inline]
    pub(crate) fn of_worker(worker: Arc<dyn Worker>) -> Self {
        Self {
            worker,
            named: Named::ByWorker,
            pair: None,
            handle: None,
        }
    }

    /// The worker the request goes to.
    #[inline]
    pub(crate) fn worker(&self) -> &Arc<dyn Worker> {
        &self.worker
    }

    /// Whether the placement is more to its route than its worker: a word
    /// of the request is owed to its engine (it holds its decision's
    /// guard), or it is a pair's. One that owes nothing is its worker and
    /// nothing else: each report below says nothing.
    #[inline]
    pub(crate) fn owes(&self) -> bool {
        self.handle.is_some() || self.pair.is_some()
    }

    /// The worker alone, of a placement its route keeps nothing else of.
    #[inline]
    pub(crate) fn into_worker(self) -> Arc<dyn Worker> {
        self.worker
    }

    /// The worker's id: the store's row of it, and what a retry names as
    /// tried. A place kept none, and the id is read of the worker here.
    pub(crate) fn id(&self) -> WorkerId {
        match self.named {
            Named::ByTarget(id, _) => id,
            Named::ByWorker => worker_id(self.worker.as_ref()),
        }
    }

    /// The data-parallel rank the decision named for the worker: what a
    /// dispatch that pins a rank sends. A place kept none, and the rank is
    /// the one the worker's registration states.
    pub(crate) fn rank(&self) -> u16 {
        match self.named {
            Named::ByTarget(_, rank) => rank,
            Named::ByWorker => worker_rank(self.worker.as_ref()),
        }
    }

    /// The placement's lease, for a case that reads it: `None` where the
    /// placement holds no guard.
    #[cfg(test)]
    pub(crate) fn lease(&self) -> Option<&smg_policy_controller::engine::Leased> {
        self.handle.as_ref()?.lease()
    }

    /// The size the placement's lease states, for a case that reads it.
    #[cfg(test)]
    pub(crate) fn lease_tokens(&self) -> Option<u32> {
        self.lease().map(|leased| leased.lease.tokens)
    }

    /// The request was sent to the worker at `now`. Said once.
    pub(crate) fn dispatched(&mut self, now: Instant) {
        if let Some(handle) = &mut self.handle {
            handle.dispatched(now);
        }
    }

    /// The request ended with `end`, as the router read it at `now`. Said
    /// once.
    pub(crate) fn completed(&mut self, end: End, now: Instant) {
        if self.handle.is_none() {
            return;
        }
        let served = self.id();
        if let Some(handle) = &mut self.handle {
            let side = handle
                .lease()
                .map_or(Side::Single, |leased| leased.lease.side);
            handle.completed(&end.outcome(side, served), now);
        }
    }

    /// The request is given up under `reason` at `now`: before a send the
    /// lease is released, and after one it completes as cancelled.
    pub(crate) fn released(&mut self, reason: ReleaseReason, now: Instant) {
        if let Some(handle) = &mut self.handle {
            handle.released(reason, now);
        }
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
        if let Some(handle) = &mut self.handle {
            handle.dispatched_decode(now);
        }
    }

    /// The decode request ended with `end`. Said once.
    pub(crate) fn decode_completed(&mut self, end: End, now: Instant) {
        if let (Some(worker), Some(handle)) = (self.decode_id(), &mut self.handle) {
            handle.completed(&end.outcome(Side::Decode, worker), now);
        }
    }

    /// The decode side is given up under `reason`.
    pub(crate) fn decode_released(&mut self, reason: ReleaseReason, now: Instant) {
        if let Some(handle) = &mut self.handle {
            handle.released_decode(reason, now);
        }
    }

    /// After the prefill is sent, take the next sendable decode of that
    /// prefill. Returns `false` when the pick holds none or the decode has
    /// already been sent or released.
    pub(crate) fn next_decode(&mut self, policy_engine: &PolicyEngine) -> bool {
        let prefill = self.id();
        let (Some(placed), Some(handle)) = (self.pair.as_deref_mut(), &mut self.handle) else {
            return false;
        };
        for (index, target) in placed.targets.iter().enumerate().skip(placed.index + 1) {
            if target.id != prefill {
                continue;
            }
            let Some(pair) = target.pair else {
                continue;
            };
            let Some(worker) = policy_engine
                .worker(pair.decode_id)
                .filter(|worker| routable_but_for_veto(worker.as_ref()))
            else {
                continue;
            };
            if handle.take_decode(index).is_some() {
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
        self.handle
            .as_ref()
            .is_some_and(DecisionGuard::decode_chosen)
    }

    /// Keeps `kept` for the second ask of the pair's decode
    /// ([`Self::decode_again_kept`]): for a route whose second ask is stages
    /// behind its first. A placement whose decode does not wait keeps
    /// nothing.
    pub(crate) fn keep_ask(&mut self, kept: KeptAsk) {
        let waits = self.decode_waits();
        if let Some(pair) = self.pair.as_deref_mut().filter(|_| waits) {
            pair.kept = Some(Box::new(kept));
        }
    }

    /// The pair keeps the decode its first decision named, and says that
    /// decode's dispatch at `now`: a decode that stood chosen is booked
    /// there. The ask kept for a second one is let go, and
    /// [`Self::decode_again_kept`] asks nothing. For a route that knows the
    /// decode cannot move, at its commit or behind it: a decode nobody asks
    /// again gains nothing by standing unbooked while its prefill runs.
    pub(crate) fn keep_first_decode(&mut self, now: Instant) {
        if let Some(pair) = self.pair.as_deref_mut() {
            pair.kept = None;
        }
        self.decode_dispatched(now);
    }

    /// [`Self::decode_again`] with the ask and the rule the route kept
    /// ([`Self::keep_ask`]), and with `holds` beside that rule: what the
    /// route knows at the call alone. The kept ask is spent. A placement
    /// that kept none asks nothing.
    pub(crate) fn decode_again_kept(
        &mut self,
        holds: impl Fn(&Arc<dyn Worker>) -> bool,
        now: Instant,
    ) -> Option<&Arc<dyn Worker>> {
        let kept = self.pair.as_deref_mut()?.kept.take()?;
        let ask = kept.ask();
        let rule = |worker: &Arc<dyn Worker>| (kept.holds)(worker.as_ref()) && holds(worker);
        self.decode_again(&kept.policy_engine, &ask, rule, now)
    }

    /// Asks the pair's decode again at `now`, on the rows of that instant:
    /// behind the prefill's answer, and ahead of the prefill's end. `ask`
    /// states the request as its first ask did. A pick moves the placement
    /// to the first of its decodes the gateway can send to and `holds`
    /// admits: the route's rule, as at the first commit. Returns the decode
    /// the call moved the placement to, and `None` where the placement
    /// keeps the decode it named.
    ///
    /// `holds` is asked of one decode at a time, best first, and of none
    /// behind the one the placement takes. So a rule may take hold of what
    /// the route needs on the worker it admits, as a claim at its gate, and
    /// give it up when it is asked of the next.
    ///
    /// The call never fails its request. A decode that does not stand
    /// chosen takes no call. A call the door sheds or refuses leaves the
    /// decode the first pick named, still chosen. Where the call picked
    /// and the gateway can send to none of its decodes, the decode side's
    /// lease is given up: the first pick's decode is then sent unbooked.
    pub(crate) fn decode_again(
        &mut self,
        policy_engine: &PolicyEngine,
        ask: &Ask<'_>,
        holds: impl Fn(&Arc<dyn Worker>) -> bool,
        now: Instant,
    ) -> Option<&Arc<dyn Worker>> {
        let prefill = self.id();
        let placed = self.pair.as_deref_mut()?;
        let handle = self.handle.as_mut()?;
        if !handle.decode_chosen() {
            return None;
        }
        let made = Made {
            ask,
            missed: &[],
            // A pair's decode call hands no list of a regular pool.
            members: None,
            loads: None,
            source: Cell::new(None),
        };
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
        policy_engine.count_decode_call(ask.model, picked.as_ref(), outcome, seconds);
        let Some(Decision::Pick(pick)) = picked else {
            return None;
        };
        for (index, target) in pick.targets.iter().enumerate() {
            let Some(pair) = target.pair.filter(|_| target.id == prefill) else {
                continue;
            };
            let sendable =
                |worker: &Arc<dyn Worker>| routable_but_for_veto(worker.as_ref()) && holds(worker);
            let Some(worker) = policy_engine.worker(pair.decode_id).filter(sendable) else {
                continue;
            };
            // The call took its first target. A later one is taken by name.
            if index == 0 || handle.take_decode(index).is_some() {
                placed.decode = worker;
                placed.side = pair;
                placed.targets = pick.targets;
                placed.index = index;
                return Some(&placed.decode);
            }
        }
        handle.released_decode(ReleaseReason::CommitFailed, now);
        None
    }
}

impl fmt::Debug for Placed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Placed")
            .field("worker", &self.worker.url())
            .field("named", &self.named)
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
    guard: WorkerLoadGuard,
}

impl Held {
    /// The placement beside the load guard `guard` makes of the worker its
    /// response is read from: a single's worker, a pair's decode. The guard
    /// is asked of the placement's own worker, so the two do not name two
    /// workers, and neither is dropped ahead of its turn.
    ///
    /// A pair's prefill holds its load by the admission guard
    /// [`PolicyEngine::place_pair`] hands back apart. Its router gives that one
    /// up when the prefill phase ends, ahead of the response.
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

    /// [`Placed::decode_again_kept`] for the held placement. A decode the
    /// pair moves to takes the load guard with it, under the first guard's
    /// key: the response is read from that worker now, and the first one's
    /// load is given back.
    pub(crate) fn decode_again(
        &mut self,
        holds: impl Fn(&Arc<dyn Worker>) -> bool,
        now: Instant,
    ) -> Option<Arc<dyn Worker>> {
        let moved = self.placed.decode_again_kept(holds, now).cloned()?;
        self.guard = self.guard.onto(Arc::clone(&moved));
        Some(moved)
    }
}

mod ride;
pub(crate) use ride::{BehindAnswer, Sent};
#[cfg(test)]
mod decode_again_tests;
#[cfg(test)]
mod own_tests;
#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;
