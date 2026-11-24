//! What a host's decisions stand on: the engine that makes a whole decision
//! and hears what its host says of the decision's lease.
//!
//! The engine decides over the worker objects its host lends and holds no
//! store. A [`DecisionGuard`](crate::tracking::DecisionGuard) holds a
//! decision of it, and a host's own object asks it, through
//! [`DecisionEngine`].
//!
//! **What a guard lends.** A guard lends its engine one view of the
//! decision it holds, with each word and with a pair's decode call
//! ([`HeldDecision`]): the pick, the lease as issued, the target its host
//! took and that target's place in the pick. An engine with a file of its
//! leases finds its record by the lease's id and reads nothing else. An
//! engine with none needs no table to find one: every fact a stage is told
//! of a request is in the view.
//!
//! **What an engine holds to**, whatever it decides over:
//!
//! - One clock. `now` is its host's reading of the clock a decision runs
//!   on. The request's clock value and the age of each sample a decision
//!   reads are counted from one origin.
//! - No count beside the workers' own. What the requests on their way book
//!   on a worker is read where its host counts them.
//! - A rule the document states is run or refused by name. An engine that
//!   is handed no overload verdict, no pins or no list for a rule the
//!   document states does not start under that document.
//! - No pick waits on another's. State every pick shares (the session
//!   table, the dispatch credit, a pool's turn) moves under an atomic or a
//!   lock no decision holds across its stages. One wait is a list's: the
//!   decisions that meet at a list's first ask wait for the one that fills
//!   its cell (`ListCell`).

use std::time::Instant;
#[cfg(feature = "in-process")]
use std::{borrow::Cow, fmt, sync::Arc};

#[cfg(feature = "in-process")]
use smg_worker::load::state::LoadSnapshot;

use super::{
    decision::{Decision, Pick, ShadowKind, Survivors, Target, Targets, Trace},
    prompt_hashes::{HashSeeds, PromptHashes},
    sticky_sessions::SessionVerdict,
    view::{Constraints, Phase, RequestFacts, Session, SessionCountedUnder, WireModes, WorkerId},
};
use crate::{
    document::AppliedSpec,
    store::GatewayId,
    tracking::lease::{Lease, Said, Side},
};

/// A request its host makes under the document a decision is made under.
/// What the host reads of the applied document to state the request, the
/// sources of its routing key for one ([`AppliedSpec::key_sources_for`]),
/// is then of the same generation as the engine that decides.
///
/// A request its host stated ahead is its own maker ([`Request`]).
///
/// The wire contract's `SnapshotRequest` is another thing: the operator's
/// call that reads the store's worker rows.
pub trait RequestMaker {
    /// The model of the request. An engine that keeps a seat for each
    /// model takes it by this, ahead of the load: the request
    /// [`RequestMaker::make`] makes names this model.
    fn model(&self) -> &str;

    /// The request, made under the document its decision is made under. An
    /// engine that keeps a seat makes it inside its hold of the seat: a
    /// maker that waits holds up the seat's other requests, and one that
    /// asks the engine again can wait on the seat it holds.
    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w>;

    /// What the request states where it is plain ([`Plain`]): `None` for
    /// every other request, and from a maker that does not say.
    ///
    /// An engine that decides over its host's lists asks this ahead of
    /// [`RequestMaker::make`], at its entry that issues a pick's lease. A
    /// place one pick of a picker decides reads no fact of a request, so a
    /// plain request of such a place is decided with none made: its routing
    /// key is not derived and no seed is drawn for it. The answer is cheap:
    /// it reads the document nowhere. That engine's entry that issues no
    /// lease never asks: it makes every request.
    ///
    /// A request held to places of its list
    /// ([`Constraints::places`](super::view::Constraints::places)) is not
    /// plain: that pick reads no constraint. A maker that states places
    /// answers `None` here.
    fn plain(&self) -> Option<Plain> {
        None
    }

    /// The length of the request's text in characters, as the request made
    /// would state it
    /// ([`RequestFacts::prompt_chars`](super::view::RequestFacts::prompt_chars)):
    /// zero for a request of token ids or of no text, and from a maker
    /// that does not say.
    ///
    /// An engine that decides over its host's lists asks this where it
    /// issues the lease of a plain request's pick, which states the
    /// length, and nowhere else. So [`RequestMaker::plain`] stays cheap,
    /// and a text is passed over once a request: here for a pick that is
    /// leased ahead of the request's making, and at the making otherwise.
    fn prompt_chars(&self) -> u32 {
        0
    }

    /// The length of the request's text in bytes, as the request made
    /// would state it
    /// ([`RequestFacts::prompt_bytes`](super::view::RequestFacts::prompt_bytes)):
    /// zero for a request of token ids or of no text, and from a maker
    /// that does not say. Asked where [`RequestMaker::prompt_chars`] is,
    /// for the same lease. It costs no pass over the text.
    fn prompt_bytes(&self) -> u32 {
        0
    }

    /// Where a worker's object counts the requests of the request's
    /// session key ([`SessionCountedUnder`]). The trait's answer is the
    /// key's own text, where the caller's own guard counts each request it
    /// sends. A host that serves no request counts by the decisions it
    /// holds on file, and answers the session's number.
    ///
    /// An engine that decides over its host's lists asks this for a
    /// document with a stage that reads the count
    /// ([`Needs::session_in_flight`](crate::document::spec::Needs::session_in_flight)),
    /// and reads each worker's count under the name it gives
    /// ([`SessionCountedUnder::name`]). It asks under no other document.
    fn session_counted_under(&self) -> SessionCountedUnder {
        SessionCountedUnder::KeyText
    }

    /// The workers the request may be sent to, where its host holds them:
    /// the lists the decision's targets name by position
    /// ([`Target::position`], [`Pair::decode_position`](super::decision::Pair::decode_position)).
    /// An engine with no store decides over these and refuses a request
    /// that hands none. An engine that reads its own rows asks for none,
    /// and a host of such an engine states none.
    #[cfg(feature = "in-process")]
    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::None
    }

    /// The load reports of the host's workers, as the host holds them for
    /// this decision: its load monitor's snapshot, borrowed once. An engine
    /// that decides over its host's lists reads each worker's report there,
    /// and every record of one decision in the one snapshot. A host hands
    /// it where the document reads reports
    /// ([`Needs::loads`](crate::document::spec::Needs::loads)) or a named
    /// signal
    /// ([`Needs::signal_cells`](crate::document::spec::Needs::signal_cells)),
    /// whose values the same reports state. Where it hands none, a decision
    /// reads the newest snapshot the engine's own task keeps at hand
    /// ([`PolicyEngine::run`](super::policy_engine::PolicyEngine::run));
    /// under no snapshot a record carries no report and a signal reads no
    /// value. A host whose document holds a stage that keeps load reports
    /// ([`Lifecycle::keeps_loads`](crate::stages::Lifecycle::keeps_loads))
    /// hands none: the task tells such a stage a report ahead of the
    /// snapshot it keeps, and a snapshot handed here is outside that order.
    #[cfg(feature = "in-process")]
    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        None
    }

    /// The entry of the request's prompt in the host's token tree of sent
    /// prefixes ([`CacheIndexes::token_tree`]), as its host names it
    /// ([`TokenKey`]): `None` from a host that holds no token ids of the
    /// prompt, and from a maker that does not say. `page` is the tree's
    /// page in tokens, which the marker of a cache partition fills whole.
    ///
    /// Asked at most once a decision, and only where the host lends the
    /// request's place such a tree: never by a place one pick decides, and
    /// never by an engine that reads its own rows.
    ///
    /// [`CacheIndexes::token_tree`]: crate::engine::CacheIndexes::token_tree
    #[cfg(feature = "in-process")]
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
        let _ = page;
        None
    }

    /// Whether the request's caller sends it to `target`, a target of the
    /// decision's pick. A decision that stores where its prompt went, in
    /// the host's token tree ([`Self::token_key`]) or in the engine's
    /// record of sent prefixes, asks it of the pick's targets, best first,
    /// and stores the prompt for the first one answered `true`: for none
    /// where each is answered `false`, so neither names a worker the
    /// request was not sent to. A pair's decode decided again asks it of
    /// each target, and its maker answers for the target's decode alone:
    /// the prefill a target names was sent already.
    ///
    /// It is asked behind the pick and ahead of the decision's lease,
    /// inside the decision: an answer asks the engine nothing. No other
    /// decision asks it.
    ///
    /// `true` from a maker that does not say: its caller sends to the
    /// first target of a decision. A maker whose caller may send to a
    /// later target answers it, or the first target is named for a prompt
    /// that went to another.
    #[cfg(feature = "in-process")]
    fn sends_to(&self, target: &Target) -> bool {
        let _ = target;
        true
    }
}

impl RequestMaker for Request<'_> {
    fn model(&self) -> &str {
        self.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        *self
    }

    fn plain(&self) -> Option<Plain> {
        let (facts, cx) = (&self.facts, &self.constraints);
        let plain = facts.tried.is_empty()
            && cx.excluded.is_empty()
            && cx.subset_hint.is_none()
            && cx.demands.is_none()
            && !cx.overload_lifted
            && !cx.decode_later
            && cx.stalled.is_empty()
            && cx.places.is_none();
        plain.then_some(Plain {
            phase: facts.phase,
            wires: cx.wire_modes,
            prompt_tokens: facts.prompt_tokens,
        })
    }

    fn prompt_chars(&self) -> u32 {
        self.facts.prompt_chars
    }

    fn prompt_bytes(&self) -> u32 {
        self.facts.prompt_bytes
    }
}

/// What a plain request states ([`RequestMaker::plain`]): all that a pick
/// which reads no fact of a request still needs of it.
///
/// A request is plain where it is a first attempt that asks nothing beyond
/// a worker of its list: it names no worker tried or just missed, states no
/// subset hint, no hard demand, no later decode and no places of its list,
/// lifts no veto, and no standing veto of its host masks a row.
///
/// A lease issued for a plain request that was never made states no
/// session: no rule of a place picked by its picker alone reads one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plain {
    /// The request's phase
    /// ([`RequestFacts::phase`](super::view::RequestFacts::phase)).
    pub phase: Phase,
    /// The wires its host dials for it
    /// ([`Constraints::wire_modes`](super::view::Constraints::wire_modes)).
    pub wires: WireModes,
    /// The size of its prompt in tokens, as the request made would state it
    /// ([`RequestFacts::prompt_tokens`](super::view::RequestFacts::prompt_tokens)):
    /// a lease issued for it states the size. Its text's two lengths are
    /// asked apart, where that lease is issued
    /// ([`RequestMaker::prompt_chars`], [`RequestMaker::prompt_bytes`]).
    pub prompt_tokens: u32,
}

/// The workers a request may be sent to, as its host lends them for one
/// decision: each list with the cell the host keeps beside it
/// ([`LentList`](super::list_cell::LentList)). The engine asks a cell for
/// its own value of the list inside the decision, where it decides over
/// the list, under the document it decides under: a host holds no value of
/// a list and asks the engine for none. A plain single request that one
/// pick does not decide lends its list again behind its making, since a
/// maker may name its list while it makes the request: a cell the pick
/// asked is not asked again.
/// A list is the host's own: the host finds a target's worker at the
/// target's position in the list it lent, with no lookup by id.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug, Default)]
pub enum WorkerLists<'a> {
    /// The host hands no list.
    #[default]
    None,
    /// A single request's list, and the one list of a request of another
    /// phase that stands alone.
    Single(super::list_cell::LentList<'a>),
    /// A pair's two lists, its prefills and its decodes, with the pair
    /// classes of their workers and the host's reads of its admission.
    Pair(super::policy_engine::pair::PairLists<'a>),
    /// An encode request's three lists: its pair's two and its encoders,
    /// with what the encode side reads of the workers beside their records
    /// ([`PolicyEngine::decide_encode_at`](super::policy_engine::PolicyEngine::decide_encode_at)).
    /// The engine's other entries take no items: through them the three
    /// are the pair's two, for a request with no item and for a decode
    /// asked again. Such a decision does not read whether a decode takes
    /// images, so the encode entry declines a request that states the
    /// later decode where its decode is held to the ones that take images:
    /// by its demand, or by the preference of references beside an image
    /// or a video.
    Encode(super::policy_engine::encode::EncodeLists<'a>),
}

/// The entry of a request's prompt in its host's token tree of sent
/// prefixes, named by the request's host ([`RequestMaker::token_key`]).
///
/// The naming is the host's, as it is for a prompt's hashes
/// ([`HashSeeds`]). A host that partitions its workers' caches hands the
/// prompt's ids behind a marker of the partition, so two partitions part
/// within the marker, and says how long the marker is. The engine keeps two
/// partitions apart exactly as well as these ids do.
///
/// It is never a field of a request's facts: no stage, trace or kept
/// sample holds a client's tokens. Its `Debug` prints sizes alone.
#[cfg(feature = "in-process")]
pub struct TokenKey<'a> {
    /// The ids the tree is matched and written by: the prompt's token ids,
    /// behind the partition's marker where the request has one.
    pub tokens: Cow<'a, [u32]>,
    /// How many of the leading ids are the marker and no part of the
    /// prompt: a match's length has them taken off. Zero with no partition.
    pub marker: usize,
}

#[cfg(feature = "in-process")]
impl fmt::Debug for TokenKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenKey")
            .field("tokens", &self.tokens.len())
            .field("marker", &self.marker)
            .finish()
    }
}

/// What a guard holds of its decision, lent to its engine for one word
/// ([`DecisionEngine::say`]) or for a pair's decode call
/// ([`DecisionEngine::decide_decode_under`]). It is read before the word
/// moves the guard.
///
/// An engine that keeps no file of its leases tells its stages of a request
/// from this alone, by one rule: a side's later word is told under the
/// lease that side was issued under, field for field, but for the two a
/// dispatch moves, its `target_index` and the worker it books. The lease
/// here is a single's or the prefill side's. The decode side's differs
/// from it in its side, in the worker it books (the decode of the pick's
/// first target where the side was issued with its prefill, the decode of
/// the target taken where it stood chosen), in its session
/// ([`HeldDecode::session`]) and, behind a decode call, in its three sizes
/// ([`HeldDecode::tokens`], [`HeldDecode::chars`] and
/// [`HeldDecode::bytes`]). A decode issued later is dated at its
/// issue, the decode call's instant or its dispatch's, and its later words
/// bear the first decision's instant, which is this lease's
/// ([`Lifecycle::on_decision_issued`](crate::stages::Lifecycle::on_decision_issued)).
#[derive(Clone, Copy, Debug)]
pub struct HeldDecision<'a> {
    /// The pick the guard holds: the first decision's, or a decode call's
    /// once one picked.
    pub decision: &'a Decision,
    /// The lease as its decision issued it: a single's, or the prefill
    /// side's of a pair.
    pub lease: &'a Lease,
    /// The place in the pick of the target taken. An end and a release
    /// carry no index of their own. Behind a decode call that picked, it is
    /// the place in that call's pick, for the prefill's later words too:
    /// the guard keeps one index for both sides, and the worker a side
    /// books stands.
    pub index: u8,
    /// The target its host took: the pick's first, or the one a take moved
    /// to. A pair's decode is in it.
    pub target: &'a Target,
    /// A pair's decode side, read under a pair's lease alone
    /// ([`Lease::side`] is [`Side::Prefill`]). A single's lease has none:
    /// its handle lends no session, the lease's own size and a side that
    /// never stood chosen, and no reader takes them.
    pub decode: HeldDecode,
}

/// What a guard holds of a pair's decode side ([`HeldDecision::decode`]):
/// what that side's lease does not share with the prefill side's, and
/// where the side stands as the word finds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldDecode {
    /// The pair's session under the decode phase
    /// ([`DecideOutcome::decode_session`]): a value of the decode side's
    /// lease that the prefill side's does not give.
    pub session: Session,
    /// The prompt size the side's words state ([`Lease::tokens`]): the size
    /// the side was issued under. That is the first decision's where the
    /// side was issued with its prefill or goes out with no call, and a
    /// decode call's own once one picked the side.
    pub tokens: u32,
    /// The characters of the text the side's words state
    /// ([`Lease::chars`]), by the rule of [`Self::tokens`]: the first
    /// decision's, and a decode call's own once one picked the side.
    pub chars: u32,
    /// The bytes of that text ([`Lease::bytes`]), by the same rule.
    pub bytes: u32,
    /// The side stood chosen until this word: its request stated the later
    /// decode, no decode call picked it, and nothing was issued for it.
    /// The side's dispatch said under this is its issue too, dated at the
    /// word's instant.
    pub chosen: bool,
}

/// What the one entry answers its host ([`DecisionEngine::decide_under`]).
///
/// The answer is written where its host reads it and matched at once: it
/// is not kept, so the two forms differ in size freely. A whole decision in
/// a box would be an allocation a decision.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "matched where it is returned; a boxed decision would allocate at each one"
)]
pub enum Decided {
    /// A pick of which nothing is owed ([`AlonePick`]): the host sends to
    /// the worker at the place and tells the engine no more.
    Alone(AlonePick),
    /// A decision as a guard holds it
    /// ([`DecisionGuard::holding`](crate::tracking::DecisionGuard::holding)),
    /// and whether the request its maker made stated the later decode
    /// ([`Constraints::decode_later`](super::view::Constraints::decode_later)).
    Whole(DecideOutcome, bool),
}

impl Decided {
    /// The whole decision, for a host that handed no single list of its
    /// own. No place answers such a host ([`AlonePick`]), so `None` there
    /// is its engine's fault.
    #[must_use]
    pub fn whole(self) -> Option<(DecideOutcome, bool)> {
        match self {
            Self::Whole(decided, later) => Some((decided, later)),
            Self::Alone(_) => None,
        }
    }
}

/// A plain single request's pick, where the document decides the request's
/// place by one pick of its picker and no stage of the document takes a
/// word on a lease ([`Lifecycle::hears`](crate::stages::Lifecycle::hears)).
///
/// The worker is the one at `place` of the list its host handed
/// (`WorkerLists::Single`): the picker's turn landed on it and read it
/// eligible at this call. No request was made for the pick, no lease is
/// issued, no id is taken and nothing is dated. The host sends to the
/// worker and owes the engine no word of the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlonePick {
    /// The worker's place in the list.
    pub place: u32,
    /// The profile whose picker picked, by its position in the document:
    /// with `generation`, what a host counts the decision under.
    pub profile: u16,
    /// The document's generation.
    pub generation: u64,
}

impl AlonePick {
    /// The pick as a decision that names its one worker, by `id` and
    /// `dp_rank`: what a lease is issued for where a stage takes a word,
    /// and what a host's trace of the pick shows. The trace is a drawn
    /// pick's: no pass over the pool ran, so it counts no survivor, and no
    /// seed was drawn for it.
    #[must_use]
    pub fn decision(self, id: WorkerId, dp_rank: u16) -> Decision {
        let mut targets = Targets::new();
        targets.push(Target {
            position: self.place,
            id,
            dp_rank,
            pair: None,
        });
        let mut trace = Trace {
            profile: self.profile,
            survivors: Survivors::NOT_COUNTED,
            top_len: 1,
            generation: self.generation,
            ..Trace::default()
        };
        trace.top[0] = (id, 0.0);
        Decision::Pick(Pick {
            targets,
            trace,
            session_verdict: SessionVerdict::None,
        })
    }
}

/// An engine a host asks for whole decisions and tells of each lease.
pub trait DecisionEngine {
    /// One decision for `gateway`'s request at `now`, on the calling
    /// thread: the binding, its rules, the plan and the profile, for the
    /// request `maker` makes under the document in force.
    ///
    /// The answer is whole ([`Decided::Whole`]) or a place alone
    /// ([`Decided::Alone`]). A whole pick comes with its lease where a
    /// stage of the document takes a word on one, issued before the answer
    /// leaves: the stages that take the issue have heard of the request,
    /// and its booking stands on the first target. The flag beside it says
    /// whether the request the maker made stated the later decode
    /// ([`Constraints::decode_later`](super::view::Constraints::decode_later)).
    /// A place answers only a maker that hands a single list
    /// (`RequestMaker::workers`) and says its request plain
    /// ([`RequestMaker::plain`]).
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the request.
    fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<Decided, DecisionRefused>;

    /// [`Self::decide_under`] for a request its host stated ahead.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the request.
    fn decide(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        // A request stated ahead hands no list: no place answers it.
        let decided = self.decide_under(gateway, request, now)?;
        let (decided, _) = decided.whole().ok_or(DecisionRefused::Faulted)?;
        Ok(decided)
    }

    /// A pair's decode, decided again behind its prefill's dispatch: among
    /// the decodes of the prefill the host holds, as they stand at `now`.
    /// `held` is the view of the guard that holds the first decision: the
    /// pair's lease as issued, and the target its host took, which names
    /// the held prefill. The guard asks only where its prefill is
    /// dispatched and not ended and its decode stands chosen. A pick
    /// issues the decode side under the lease's id.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the call.
    fn decide_decode_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused>;

    /// [`Self::decide_decode_under`] for a request its host stated ahead.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the call.
    fn decide_decode(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        self.decide_decode_under(gateway, request, held, now)
    }

    /// What the host says of a lease at `now`: a side's dispatch, a side's
    /// end, or a release. `held` is the guard's view of the decision at the
    /// word. The word names its side. Whoever applies the word tells the
    /// stages of the engine that issued the lease, once. A decode that
    /// stood chosen ([`HeldDecode::chosen`]) goes out with no
    /// decode call: its dispatch tells the side's issue first.
    fn say(&self, gateway: GatewayId, held: &HeldDecision<'_>, said: Said, now: Instant);

    /// One pass of the engine's clock at `now`, about once a second: what
    /// lives past its end is ended, and what was told of it is said.
    fn sweep(&self, now: Instant);

    /// Reads the document in force. A host names a profile, reads a
    /// binding's key sources and a document's own rules here, under the
    /// generation a decision made at this instant would run.
    fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R;

    /// The block hashes of a request's prompt for `model`, as the
    /// generation in force reads them: its token blocks, its text blocks,
    /// or both, under `seeds`.
    fn prompt_hashes(
        &self,
        model: &str,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
    ) -> PromptHashes;

    /// How many decisions panicked, a shadow's beside a live decision that
    /// stood among them.
    fn faults(&self) -> u64;

    /// How many shadows said `kind` beside a single decision of the
    /// binding at position `binding`.
    fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64;
}

/// A request and its constraints as the host holds them. A door reads
/// these facts in process. A link carries them to the controller.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// The request's id as the host states it: its lease keeps the id's
    /// fingerprint.
    pub id: &'a [u8],
    /// The request as the engine reads it. The deciding view supplies the
    /// clock value, affinity, named rows, lease counts and the key's count
    /// under a rule with an in-flight cap. A budget of 0 leaves a decision
    /// in process unbounded. Over a link it says the request's time is spent.
    pub facts: RequestFacts<'a>,
    /// What the host alone knows.
    pub constraints: Constraints<'a>,
}

/// A decision as its host takes it.
#[derive(Debug)]
pub struct DecideOutcome {
    /// The engine's decision.
    pub decision: Decision,
    /// The lease issued for a pick, with the target index it books. For
    /// a pair this is the prefill side. `None` for a decision that
    /// picks no worker.
    pub lease: Option<Leased>,
    /// A pair's session under the decode phase: the decode side's lease
    /// carries it, and it cannot be told from the prefill side's. A guard
    /// lends it back with each word
    /// ([`HeldDecision`]).
    /// [`Session::NONE`] for a single, for a pair with no key, and from an
    /// engine whose file of leases holds the decode side whole.
    pub decode_session: Session,
}

impl DecideOutcome {
    /// Whether the lease is a pair's prefill side. The decode side's
    /// lease shares its id and dispatch span.
    #[must_use]
    pub fn pair(&self) -> bool {
        self.lease
            .is_some_and(|leased| leased.lease.side == Side::Prefill)
    }
}

/// A pick's lease as its host receives it, with its dispatch deadline.
#[derive(Clone, Copy, Debug)]
pub struct Leased {
    /// The lease issued for the pick.
    pub lease: Lease,
    /// When its dispatch span ends on the host's clock. The store ends
    /// leases whose dispatch span has passed and whose dispatch is still
    /// owed.
    pub due: Instant,
}

/// Why a door has no decision for a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionRefused {
    /// The request is a decode or encode side sent on its own. A pair is
    /// asked at its prefill, and that one request decides both sides.
    PairSide,
    /// The store filed no lease for the pick, at the second try too: its
    /// term mints no id, or another hand raised it under both tries.
    Unleased,
    /// The decision panicked. It cost this request alone: the next one
    /// decides on new buffers.
    Faulted,
    /// The store takes no decode side under the lease a decode call
    /// named, for the reason it carries.
    Decode(DecodeRefused),
}

/// Why a call could not issue the decode side of a pair's lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeRefused {
    /// No lease of the gateway under that id.
    NoLease,
    /// The lease was filed without a chosen decode side.
    NotChosen,
    /// The prefill side is waiting for dispatch or has ended.
    PrefillNotHeld,
    /// The chosen side was already issued or dispatched, even if it has
    /// ended, or its host gave it up, or it took as many calls as one
    /// lease takes.
    DecodeStands,
}
