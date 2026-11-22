//! What a host's decisions stand on: the engine that makes a whole decision
//! and hears what its host says of the decision's lease.
//!
//! Two engines are one to a host. [`StoreBackedEngine`] decides over the
//! policy store's rows: a controller's own, and a gateway's that follows a
//! controller. The engine of a gateway's own process decides over the
//! gateway's worker objects and holds no store. A
//! [`DecisionGuard`](crate::tracking::DecisionGuard) holds a decision of
//! either, and a host's own object asks either, through [`DecisionEngine`].
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
//!   lock no decision holds across its stages.

#[cfg(feature = "in-process")]
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "in-process")]
use smg_worker::load::state::LoadSnapshot;

use super::{
    decision::{Decision, Pick, ShadowKind, Survivors, Target, Targets, Trace},
    framework::{EncodeItems, EncodePlaced},
    prompt_hashes::{HashSeeds, PromptHashes},
    sticky_sessions::SessionVerdict,
    store_backed_engine::{
        self, DecideOutcome, DecisionRefused, Request, StoreBackedEngine, SweepOutcome,
    },
    view::{Phase, Session, WireModes, WorkerId},
};
use crate::{
    document::AppliedSpec,
    store::{pairing::Refusal, GatewayId, Said},
    tracking::lease::Lease,
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
    /// [`RequestMaker::make`]. A place one pick of a picker decides reads
    /// no fact of a request but its seed, where the picker draws
    /// ([`RequestMaker::seed`]): a plain request of such a place is decided
    /// with none made, and its routing key is not derived. The answer is
    /// cheap: it reads the document nowhere.
    fn plain(&self) -> Option<Plain> {
        None
    }

    /// The request's seed
    /// ([`RequestFacts::seed`](super::view::RequestFacts::seed)), for a
    /// pick that reads nothing else of the request. An engine that decides
    /// over its host's lists asks it ahead of [`RequestMaker::make`], where
    /// the place's picker picks by a draw
    /// ([`PicksAlone::ByDraw`](super::framework::PicksAlone::ByDraw)), and
    /// at no other pick: a rotation's request draws none.
    ///
    /// A maker that says it states a seed drawn for the request, as the
    /// request it makes states one: under a constant seed a picker that
    /// draws names the same worker of an unchanged pool every time. A
    /// request stated ahead says its own, so its pick alone is the one its
    /// whole decision would draw first. `None` from a maker that does not
    /// say: its request is then made, and the whole decision draws by the
    /// seed that request states.
    fn seed(&self) -> Option<u64> {
        None
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
    /// ([`Needs::loads`](crate::document::spec::Needs::loads)); under no
    /// snapshot a record carries no report.
    #[cfg(feature = "in-process")]
    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        None
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
            && cx.stalled.is_empty();
        plain.then_some(Plain {
            phase: facts.phase,
            wires: cx.wire_modes,
            prompt_tokens: facts.prompt_tokens,
        })
    }

    fn seed(&self) -> Option<u64> {
        Some(self.facts.seed)
    }
}

/// What a plain request states ([`RequestMaker::plain`]): all that a pick
/// which reads no fact of a request still needs of it.
///
/// A request is plain where it is a first attempt that asks nothing beyond
/// a worker of its list: it names no worker tried or just missed, states no
/// subset hint, no hard demand and no later decode, lifts no veto, and no
/// standing veto of its host masks a row.
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
    /// a lease issued for it states the size.
    pub prompt_tokens: u32,
}

/// The workers a request may be sent to, as its host holds them for one
/// decision. A list is the host's own, read once for the decision: the host
/// finds a target's worker at the target's position in the list it handed,
/// with no lookup by id.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug, Default)]
pub enum WorkerLists<'a> {
    /// The host hands no list.
    #[default]
    None,
    /// A single request's list.
    Single(&'a super::policy_engine::PoolMembers),
    /// A pair's two lists, its prefills and its decodes, with the pair
    /// classes of their workers and the host's reads of its admission.
    Pair(super::policy_engine::pair::PairLists<'a>),
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
/// the target taken where it stood chosen) and in its session
/// ([`Self::decode_session`]). A decode issued later is dated at its
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
    /// A pair's session under the decode phase
    /// ([`DecideOutcome::decode_session`]): the one value of the decode
    /// side's lease that the prefill side's does not give.
    pub decode_session: Session,
    /// The place in the pick of the target taken. An end and a release
    /// carry no index of their own. Behind a decode call that picked, it is
    /// the place in that call's pick, for the prefill's later words too:
    /// the guard keeps one index for both sides, and the worker a side
    /// books stands.
    pub index: u8,
    /// The target its host took: the pick's first, or the one a take moved
    /// to. A pair's decode is in it.
    pub target: &'a Target,
    /// A pair's decode side stood chosen until this word: its request
    /// stated the later decode, no decode call picked it, and nothing was
    /// issued for it. The side's dispatch said under this is its issue
    /// too, dated at the word's instant.
    pub decode_chosen: bool,
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
/// ([`WorkerLists::Single`]): the picker's turn or its draw landed on it
/// and read it eligible at this call. No request was made for the pick, no
/// lease is issued, no id is taken and nothing is dated. The host sends to
/// the worker and owes the engine no word of the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlonePick {
    /// The worker's place in the list.
    pub place: u32,
    /// The profile whose picker picked, by its position in the document:
    /// with `generation`, what a host counts the decision under.
    pub profile: u16,
    /// The document's generation.
    pub generation: u64,
    /// The seed the picker drew by ([`RequestMaker::seed`]): the pick
    /// replays from it. Zero under a picker that draws nothing, for which
    /// none was asked.
    pub seed: u64,
}

impl AlonePick {
    /// The pick as a decision that names its one worker, by `id` and
    /// `dp_rank`: what a lease is issued for where a stage takes a word,
    /// and what a host's trace of the pick shows. The trace is a drawn
    /// pick's: no pass over the pool ran, so it counts no survivor, and it
    /// states the seed the pick drew by.
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
            seed: self.seed,
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
    /// ([`RequestMaker::workers`]) and says its request plain
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

    /// One decision for a pair with the encode `items` of its request: the
    /// pair, and a worker for each item, written to `placed`. `placed` is
    /// left empty where no decision is made.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the request.
    fn decide_encode_pair(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused>;

    /// What the host says of a lease at `now`: a side's dispatch, a side's
    /// end, or a release. `held` is the guard's view of the decision at the
    /// word. The word names its side. Whoever applies the word tells the
    /// stages of the engine that issued the lease, once. A decode that
    /// stood chosen ([`HeldDecision::decode_chosen`]) goes out with no
    /// decode call: its dispatch tells the side's issue first.
    fn say(&self, gateway: GatewayId, held: &HeldDecision<'_>, said: Said, now: Instant);

    /// One pass of the engine's clock at `now`, about once a second: what
    /// lives past its end is ended, and what was told of it is said.
    fn sweep(&self, now: Instant) -> SweepOutcome;

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

    /// Why no prefill of `model` pairs with a decode of it, among the
    /// workers on the side of `wires`. `None` where a pair can be made,
    /// and where a pool holds no worker on that side.
    fn pair_refusal(&self, model: &str, wires: WireModes) -> Option<Refusal>;

    /// How many decisions panicked, a shadow's beside a live decision that
    /// stood among them.
    fn faults(&self) -> u64;

    /// How many shadows said `kind` beside a single decision of the
    /// binding at position `binding`.
    fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64;
}

impl DecisionEngine for StoreBackedEngine {
    #[inline]
    fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<Decided, DecisionRefused> {
        let (decided, later) = self.decide_under_stated(gateway, maker, now)?;
        Ok(Decided::Whole(decided, later))
    }

    // A request stated ahead of the load pays nothing for a maker's form.
    #[inline]
    fn decide(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide(self, gateway, request, now)
    }

    // The file of the store's leases holds the prefill a lease booked.
    #[inline]
    fn decide_decode_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_decode_under(self, gateway, maker, held.lease.id, now)
    }

    #[inline]
    fn decide_decode(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_decode(self, gateway, request, held.lease.id, now)
    }

    #[inline]
    fn decide_encode_pair(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_encode_pair(self, gateway, request, items, placed, now)
    }

    // The file finds the lease's record by its id, and holds whatever
    // else the view lends. A completion is also written on the served
    // worker's row, for the health precedence.
    #[inline]
    fn say(&self, gateway: GatewayId, held: &HeldDecision<'_>, said: Said, now: Instant) {
        let lease = held.lease.id;
        match said {
            Said::Completed(outcome) => {
                store_backed_engine::completed(self.store(), gateway, &outcome, now, |said| {
                    self.report(gateway, lease, said);
                });
            }
            Said::Dispatched(..) | Said::Released(..) => self.report(gateway, lease, said),
        }
    }

    fn sweep(&self, now: Instant) -> SweepOutcome {
        Self::sweep(self, now)
    }

    fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R {
        self.store().with_snapshot(|world| read(world.spec()))
    }

    fn prompt_hashes(
        &self,
        model: &str,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
    ) -> PromptHashes {
        self.store().prompt_hashes(model, tokens, text, seeds)
    }

    fn pair_refusal(&self, model: &str, wires: WireModes) -> Option<Refusal> {
        self.store()
            .with_snapshot(|world| world.pairs(model)?.refusal(wires).cloned())
    }

    fn faults(&self) -> u64 {
        Self::faults(self)
    }

    fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64 {
        Self::shadowed(self, binding, kind)
    }
}
