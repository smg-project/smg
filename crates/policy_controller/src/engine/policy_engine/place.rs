//! One call for one request ([`PolicyEngine::place`]): what a caller states
//! of the request, the request's facts made of that in one place, and the
//! worker the request goes to.
//!
//! A caller fills a [`PlaceRequest`] from what it already holds: the model,
//! its worker lists, the prompt, the headers. It names no policy, derives
//! nothing and looks inside no decision. The engine makes the request of it
//! ([`Made`]), under the document that decides:
//!
//! - the routing value and the client's value come from the sources the
//!   document's binding names, read in the caller's headers and in the
//!   request's own id ([`AppliedSpec::key_sources_for`]);
//! - the seed is one no other request of the thread was given
//!   ([`fresh_seed`]);
//! - the prompt's blocks are hashed under its cache partition, at the first
//!   making, and never for a request a place answers
//!   ([`RequestMaker::plain`]): such a request is not made at all.
//!
//! The engine then decides, and takes the first target of the decision that
//! can be sent to ([`ready`]). The caller is answered a [`Placement`], which
//! it keeps with the request, or the reason there is none ([`Unplaced`]).

use std::{
    cell::{Cell, OnceCell},
    collections::hash_map::RandomState,
    fmt,
    hash::BuildHasher,
    sync::Arc,
    thread,
    time::Instant,
};

use smg_worker::identity::basic::Worker;

use super::{PolicyEngine, PoolMembers};
use crate::{
    document::spec::{AppliedSpec, DecodePick},
    engine::{
        decision::{Decision, Pick, ShedReason},
        decision_engine::{
            DecideOutcome, Decided, DecisionEngine, DecisionRefused, Plain, Request, RequestMaker,
            WorkerLists,
        },
        demands::Demands,
        framework::EncodeItems,
        prompt_hashes::{prompt_tokens_of, HashSeeds, PromptHashes},
        records::worker_id,
        view::{
            Constraints, Header, Identity, MediaCounts, Phase, RequestFacts, WireModes, WorkerId,
        },
    },
    store::GatewayId,
    tracking::{lease::ReleaseReason, DecisionGuard, Placement},
    Draw,
};

/// A request's headers, as its caller holds them.
///
/// The engine reads the headers the document names and no other: the
/// sources of a request's routing value and of its client's value, and the
/// headers its view shows a stage ([`AppliedSpec::view_headers`]).
pub trait HeaderRead {
    /// The value of the header `name`, where the request carries it. `name`
    /// is lowercase, as a document writes it.
    fn header(&self, name: &str) -> Option<&[u8]>;
}

/// A request that carries no header.
impl HeaderRead for () {
    fn header(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

/// Headers as pairs of a lowercase name and a value. Where a name stands
/// twice, the first pair is the header.
impl<const N: usize> HeaderRead for [(&str, &str); N] {
    fn header(&self, name: &str) -> Option<&[u8]> {
        let (_, value) = self.iter().find(|(held, _)| *held == name)?;
        Some(value.as_bytes())
    }
}

/// A request's headers as an HTTP server holds them. The map keeps its
/// names lowercase, and where a name stands twice the first value is the
/// header. A value that is not text is handed as its bytes: a source that
/// reads it passes it over.
impl HeaderRead for http::HeaderMap {
    fn header(&self, name: &str) -> Option<&[u8]> {
        self.get(name).map(http::HeaderValue::as_bytes)
    }
}

/// A request's prompt, as its caller holds it: what the stages that place a
/// prompt on the worker that holds its beginning read.
#[derive(Clone, Copy)]
pub struct Prompt<'a> {
    /// The prompt's token ids, where the caller tokenized it.
    pub tokens: Option<&'a [u32]>,
    /// The prompt's text, where the caller holds no token ids. Token ids
    /// win where both are stated.
    pub text: Option<&'a str>,
    /// The two seeds of the prompt's cache partition, and
    /// [`HashSeeds::PLAIN`] for a request with none. The caller's own rule
    /// keeps two partitions apart.
    pub seeds: HashSeeds,
}

impl Prompt<'_> {
    /// A request that states no prompt.
    pub const NONE: Self = Self {
        tokens: None,
        text: None,
        seeds: HashSeeds::PLAIN,
    };
}

/// Sizes alone: a client's tokens and text reach no log line.
impl fmt::Debug for Prompt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Prompt")
            .field("tokens", &self.tokens.map(<[u32]>::len))
            .field("text", &self.text.map(str::len))
            .field("seeds", &self.seeds)
            .finish()
    }
}

/// What a caller states of one request.
///
/// Every field is the caller's own reading of its request. A caller names
/// the fields it holds over [`PlaceRequest::of`], which states nothing
/// else:
/// `PlaceRequest { streaming: true, tried: &tried, ..PlaceRequest::of(model, lists) }`.
pub struct PlaceRequest<'a> {
    /// The model the request is for, as its caller resolved it.
    pub model: &'a str,
    /// The workers the request may be sent to, as the caller holds them
    /// for this request: one list, or a pair's two with their pair
    /// classes.
    pub lists: WorkerLists<'a>,
    /// The request's prompt.
    pub prompt: Prompt<'a>,
    /// The request's headers.
    pub headers: &'a dyn HeaderRead,
    /// The request's own id, where it has one: a source of its routing
    /// value where the document names it.
    pub request_id: Option<&'a str>,
    /// The tenant the request is of.
    pub tenant: Option<&'a str>,
    /// The request's priority band.
    pub priority: i32,
    /// Whether the answer is streamed.
    pub streaming: bool,
    /// The most tokens the answer may hold. Zero states none.
    pub max_output_tokens: u32,
    /// The media the request carries, by count.
    pub media: MediaCounts,
    /// What the request demands of the worker that takes it.
    pub demands: Demands,
    /// The items of an encode request, and `None` for every other.
    pub encode: Option<EncodeItems<'a>>,
    /// The retry number, `0` for the first attempt. It is stated and not
    /// counted: a failed attempt of a pair names two workers as tried.
    pub attempt: u8,
    /// The workers the request's failed attempts were sent to, a pair's
    /// two sides each by its own id.
    pub tried: &'a [WorkerId],
    /// The time left for the decision, in microseconds. Zero leaves it
    /// unbounded.
    pub budget_micros: u32,
    /// Whether the caller asks a pair's decode again behind its prefill's
    /// answer. Where the document picks the decode after the prefill, such
    /// a request's first decision books the prefill alone. A caller that
    /// sends both sides at once states nothing here and keeps one decision
    /// under every document.
    pub decode_again: bool,
}

impl<'a> PlaceRequest<'a> {
    /// A first attempt for `model` over `lists` that states no other fact:
    /// no prompt, no header, no id, no tenant, no media, no demand, no
    /// encode item, nothing tried, no budget and no second ask of a decode.
    #[must_use]
    pub fn of(model: &'a str, lists: WorkerLists<'a>) -> Self {
        Self {
            model,
            lists,
            prompt: Prompt::NONE,
            headers: &(),
            request_id: None,
            tenant: None,
            priority: 0,
            streaming: false,
            max_output_tokens: 0,
            media: MediaCounts::NONE,
            demands: Demands::default(),
            encode: None,
            attempt: 0,
            tried: &[],
            budget_micros: 0,
            decode_again: false,
        }
    }

    /// The phase the request is decided at: a pair is asked at its
    /// prefill, and every other request is single.
    const fn phase(&self) -> Phase {
        match self.lists {
            WorkerLists::Pair(_) => Phase::Prefill,
            WorkerLists::None | WorkerLists::Single(_) => Phase::Single,
        }
    }
}

/// A seed no other request of this thread was given: the one randomness a
/// stage gets ([`RequestFacts::seed`]). Under a constant seed a policy that
/// draws picks the same worker of an unchanged pool every time.
///
/// Each thread walks a draw sequence of its own ([`Draw`]). Its start is
/// the standard library's process-random hash of the thread's id, so two
/// threads and two runs of a process start apart.
fn fresh_seed() -> u64 {
    thread_local! {
        static NEXT: Cell<u64> = Cell::new(RandomState::new().hash_one(thread::current().id()));
    }
    NEXT.with(|next| {
        let mut draw = Draw::new(next.get());
        let seed = draw.next_u64();
        next.set(draw.state());
        seed
    })
}

/// One request as the engine makes it of what its caller stated: the one
/// place a fact of a request is made.
pub(super) struct Made<'r, 'a> {
    /// The engine that decides. Its one document names the sources of the
    /// request's routing value.
    engine: &'a PolicyEngine,
    request: &'r PlaceRequest<'a>,
    /// The targets earlier decisions of this call named that could not be
    /// sent to.
    missed: &'r [WorkerId],
    /// The prompt's block hashes, made at the request's first making and
    /// borrowed by each one after it.
    hashes: OnceCell<PromptHashes>,
    /// The headers of the document's view that the request carries, made
    /// at the request's first making like the hashes.
    headers: OnceCell<Vec<Header<'a>>>,
    /// The request's routing value in the clear, as the last making
    /// derived it ([`Made::key`]).
    key: Cell<Option<&'a str>>,
}

impl<'r, 'a> Made<'r, 'a> {
    /// `request` for a decision of `engine` that leaves `missed` out.
    pub(super) const fn new(
        engine: &'a PolicyEngine,
        request: &'r PlaceRequest<'a>,
        missed: &'r [WorkerId],
    ) -> Self {
        Self {
            engine,
            request,
            missed,
            hashes: OnceCell::new(),
            headers: OnceCell::new(),
            key: Cell::new(None),
        }
    }

    /// The request's routing value in the clear, where a making derived
    /// one that leaves so: what the request's placement names. None ahead
    /// of a making, and for a request a place answers, which is not made.
    pub(super) fn key(&self) -> Option<&'a str> {
        self.key.get()
    }

    /// The request's routing value and its client's value, by the sources
    /// its binding names in the engine's document. The routing value is
    /// the first source that holds one; none where the binding names no
    /// source or none holds a value. The client's value is read only where
    /// the request has no routing value: it places where no session does.
    fn keys(&self) -> (Identity<'a>, u64) {
        let request = self.request;
        let spec = self.engine.spec();
        let Some(sources) = spec.key_sources_for(request.model, request.phase()) else {
            return (Identity::NONE, 0);
        };
        let headers = request.headers;
        let header = move |name: &str| headers.header(name);
        match sources.derive(header, request.request_id) {
            Some(derived) => (derived.key, 0),
            None => (Identity::NONE, sources.derive_client(header)),
        }
    }
}

impl RequestMaker for Made<'_, '_> {
    fn model(&self) -> &str {
        self.request.model
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        let request = self.request;
        // The engine decides under its one document, the one it hands here.
        debug_assert!(std::ptr::eq(spec, self.engine.spec()));
        let (session_key, client_key) = self.keys();
        self.key.set(session_key.plain);
        // The headers the document's view shows a stage, each named by its
        // place in that list. A value that is not text is passed over.
        let headers = self.headers.get_or_init(|| {
            let read = request.headers;
            let stated = |(name, at): (&str, u8)| {
                let value = std::str::from_utf8(read.header(name)?).ok()?;
                Some(Header { name: at, value })
            };
            // A view lists 256 names at most: the range ends where they do.
            let listed = spec.view_headers().zip(0..=u8::MAX);
            listed.filter_map(stated).collect()
        });
        let mut facts = RequestFacts {
            model: request.model,
            phase: request.phase(),
            tenant: request.tenant.map_or(Identity::NONE, Identity::plain),
            priority: request.priority,
            streaming: request.streaming,
            max_output_tokens: request.max_output_tokens,
            // A key that leaves in the clear is the one the placement names,
            // and its caller counts the request under it on its worker: the
            // count a capped sessions rule reads off the list's records.
            session_counted: session_key.plain.is_some(),
            session_key,
            client_key,
            attempt: request.attempt,
            tried: request.tried,
            budget_micros: request.budget_micros,
            headers: headers.as_slice(),
            media: request.media,
            seed: fresh_seed(),
            ..RequestFacts::default()
        };
        let (tokens, text) = (request.prompt.tokens, request.prompt.text);
        if tokens.is_some() || text.is_some() {
            let hashes = self.hashes.get_or_init(|| {
                self.engine
                    .prompt_hashes(request.model, tokens, text, request.prompt.seeds)
            });
            hashes.write_to(&mut facts);
        }
        // The document's word, like the keys: read from the generation
        // that decides.
        let later = request.decode_again && spec.decode_pick() == DecodePick::AfterPrefill;
        Request {
            id: request.request_id.map_or(&[], str::as_bytes),
            facts,
            constraints: Constraints {
                excluded: self.missed,
                decode_later: later,
                demands: request.demands,
                ..Constraints::default()
            },
        }
    }

    // A first attempt that names nothing tried or missed, demands nothing,
    // carries no encode item and asks no decode again is plain. Its size
    // is the one the request made would state.
    fn plain(&self) -> Option<Plain> {
        let request = self.request;
        let first = request.attempt == 0 && request.tried.is_empty() && self.missed.is_empty();
        let asks_more =
            !request.demands.is_none() || request.encode.is_some() || request.decode_again;
        (first && !asks_more).then(|| Plain {
            phase: request.phase(),
            wires: WireModes::NONE,
            prompt_tokens: prompt_tokens_of(request.prompt.tokens, request.prompt.text),
        })
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.request.lists
    }
}

/// Why a request has no worker. The engine says why, and its caller answers
/// its client by it: a status and its words are the caller's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unplaced {
    /// The engine shed the request. A caller answers by `reason`: the
    /// model's workers are overloaded ([`ShedReason::Saturated`]), it has
    /// no worker for the request ([`ShedReason::NoEndpoints`]), a queue is
    /// at its limit ([`ShedReason::FlowLimit`]), the decision's time
    /// passed ([`ShedReason::Deadline`]), or the policy declined the
    /// request ([`ShedReason::PolicyDeclined`]).
    Shed {
        /// Why.
        reason: ShedReason,
        /// The stage that left no worker, or what of the request the
        /// engine does not decide: a caller's refusal line names it.
        emptied_by: &'static str,
        /// The seconds after which a retry may find room: a caller's
        /// retry hint.
        retry_after_secs: u16,
    },
    /// The engine queued the request. A caller that holds no request
    /// answers as for overloaded workers.
    Queued,
    /// The decision named workers and none of them can be sent to now:
    /// none is ready and let through by its circuit breaker. A caller
    /// answers that it has no worker for the request.
    NoWorkers,
    /// The engine made no decision, for the reason it carries. A caller
    /// answers that it cannot place the request now.
    Refused(DecisionRefused),
}

/// Whether `worker` can be sent to now: ready, and let through by its
/// circuit breaker, on one read of its state. What a caller alone knows of
/// a worker at the instant of a send. The overload veto and the liveness
/// veto are the decision's own, and a pick that lifted one stands.
pub(super) fn ready(worker: &dyn Worker) -> bool {
    let state = worker.routing_state();
    state.healthy && state.can_execute
}

/// What a request with a pair's two lists is declined under: this call
/// places a single request.
const PAIR_LISTS: &str = "pair-lists";

/// What a request that carries encode items is declined under.
const ENCODE_ITEMS: &str = "encode-items";

/// The refusal of a request this call does not place, by name: the shed a
/// decision answers for what no stage of it runs.
const fn declined(stated: &'static str) -> Unplaced {
    Unplaced::Shed {
        reason: ShedReason::PolicyDeclined,
        emptied_by: stated,
        retry_after_secs: 1,
    }
}

/// The engine's one decision for `made` ([`PolicyEngine::decide_at`]), and
/// the instant it was dated at. A place dates nothing: no clock is read
/// for it. Every walk of this module decides here, and nowhere else.
#[inline]
pub(super) fn decide(
    engine: &PolicyEngine,
    made: &Made<'_, '_>,
) -> Result<(Decided, Option<Instant>), DecisionRefused> {
    let dated = Cell::new(None);
    let decided = engine.decide_at(made, || {
        let now = Instant::now();
        dated.set(Some(now));
        now
    })?;
    Ok((decided, dated.get()))
}

/// The pick of a whole decision, or why the decision placed nothing: its
/// shed, with the reason and the stage that left nobody, or its queue.
pub(super) fn picked(decision: &Decision) -> Result<&Pick, Unplaced> {
    match decision {
        Decision::Pick(pick) => Ok(pick),
        Decision::Shed(shed) => Err(Unplaced::Shed {
            reason: shed.reason,
            emptied_by: shed.emptied_by,
            retry_after_secs: shed.retry_after_secs,
        }),
        Decision::Queue(_) => Err(Unplaced::Queued),
    }
}

/// Gives up a whole decision none of whose targets can be sent to. A lease
/// its pick was issued is released by name, and the stages that take a
/// release hear it. A decision that is not leased owes nothing.
pub(super) fn give_up(
    engine: &'static PolicyEngine,
    decided: DecideOutcome,
    later: bool,
    now: Instant,
) {
    if decided.lease.is_some() {
        DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now)
            .released(ReleaseReason::CommitFailed, now);
    }
}

/// One decision for a single request over `members`, the list its caller
/// handed, and the first of its targets that can be sent to.
fn single<'a>(
    engine: &'static PolicyEngine,
    request: &PlaceRequest<'a>,
    members: Option<&PoolMembers>,
) -> Result<Placement<'a>, Unplaced> {
    let made = Made::new(engine, request, &[]);
    let (decided, dated) = decide(engine, &made).map_err(Unplaced::Refused)?;
    let at = |place: u32| members?.workers().get(place as usize);
    let mut placement = match decided {
        // The engine picked the place on the read of the worker that
        // found it eligible: the worker is read no second time.
        Decided::Alone(pick) => {
            let worker = at(pick.place).ok_or(Unplaced::NoWorkers)?;
            // A caller of this call states no worker id of its own: where
            // its list's records stand, they name the worker by its
            // address, as the placement of a place does.
            let stated = members.and_then(|list| list.records.get());
            debug_assert!(
                stated.is_none_or(|rows| rows.ids()[pick.place as usize] == worker_id(&**worker)),
                "a caller of `place` states no worker id of its own"
            );
            Placement::alone(Arc::clone(worker), engine.label_at(pick.profile))
        }
        Decided::Whole(decided, later) => {
            let now = dated.unwrap_or_else(Instant::now);
            let pick = picked(&decided.decision)?;
            let taken = pick.targets.iter().enumerate().find_map(|(index, target)| {
                let worker = at(target.position)?;
                ready(worker.as_ref()).then(|| (index, Arc::clone(worker)))
            });
            let Some((index, worker)) = taken else {
                give_up(engine, decided, later, now);
                return Err(Unplaced::NoWorkers);
            };
            Placement::taken(engine, decided, later, now, index, worker, None)
                .ok_or(Unplaced::NoWorkers)?
        }
    };
    placement.routing_key = made.key();
    Ok(placement)
}

impl PolicyEngine {
    /// Places one request: decides it, and answers the worker it goes to
    /// as a [`Placement`] its caller keeps with the request.
    ///
    /// The engine makes the request of what `request` states, decides it
    /// once, and takes the first target of the decision that can be sent
    /// to: ready, and let through by its circuit breaker. A request a
    /// place answers is placed on the worker at that place, which the pick
    /// itself read. The placement names the request's routing value where
    /// the document derived one that leaves in the clear, and none for a
    /// request a place answers: no fact of such a request is made.
    ///
    /// Not placed by this call yet, and declined by name
    /// ([`ShedReason::PolicyDeclined`]): a request with a pair's two lists
    /// (`pair-lists`) and a request that carries encode items
    /// (`encode-items`).
    ///
    /// Its caller counts each request it sends under the placement's
    /// routing value on the request's worker, as that worker's load guard
    /// does. The decision is told that the count stands
    /// ([`RequestFacts::session_counted`]), so a sessions rule with an
    /// in-flight cap keeps a session on its worker below the cap. A stage
    /// is shown the headers of the document's view that the request
    /// carries ([`AppliedSpec::view_headers`]), each named by its place in
    /// that list.
    ///
    /// Its caller still asks the engine's value of the list it lends
    /// ([`PolicyEngine::members_of`]). It lends no load snapshot: a
    /// decision that reads load reports reads the newest snapshot the
    /// engine's own task keeps at hand ([`PolicyEngine::run`]), of the
    /// channel its host handed ([`PolicyEngine::set_loads`]), and no report
    /// until its host has done both.
    ///
    /// The overload test is the engine's. The decision leaves out a worker
    /// over its overload thresholds (the `not-overloaded` filter), and the
    /// binding's own rule answers a list whose every worker is over them.
    /// This call asks it of no worker again, so a caller that asks this
    /// call for every request keeps no overload test of its own.
    ///
    /// A program that asks this call states no worker id of its own
    /// ([`FixedFacts::id`](crate::engine::FixedFacts::id)): a placement
    /// names its worker by the fingerprint of the worker's address
    /// ([`Placement::id`]), the id a next attempt states as tried. A debug
    /// build fails a request placed on a worker its list's records name
    /// otherwise.
    ///
    /// # Errors
    ///
    /// [`Unplaced`]: why the request has no worker. Where the decision
    /// named workers and none can be sent to, a lease it was issued is
    /// given up at once, and the stages that take a release hear it.
    pub fn place<'a>(&'static self, request: &PlaceRequest<'a>) -> Result<Placement<'a>, Unplaced> {
        if request.encode.is_some() {
            return Err(declined(ENCODE_ITEMS));
        }
        match request.lists {
            WorkerLists::Pair(_) => Err(declined(PAIR_LISTS)),
            WorkerLists::Single(members) => single(self, request, Some(members)),
            WorkerLists::None => single(self, request, None),
        }
    }
}

#[cfg(test)]
mod tests;
