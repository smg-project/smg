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

use std::time::Instant;

use super::{
    decision::{Decision, ShadowKind, Target},
    framework::{EncodeItems, EncodePlaced},
    prompt_hashes::{HashSeeds, PromptHashes},
    store_backed_engine::{
        self, DecideOutcome, DecisionRefused, Request, StoreBackedEngine, SweepOutcome,
    },
    view::{Phase, Session, WireModes},
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
    /// no fact of a request, so a plain request of such a place is decided
    /// with none made: its routing key is not derived and no seed is drawn
    /// for it. The answer is cheap: it reads the document nowhere.
    fn plain(&self) -> Option<Plain> {
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

/// An engine a host asks for whole decisions and tells of each lease.
pub trait DecisionEngine {
    /// One decision for `gateway`'s request at `now`, on the calling
    /// thread: the binding, its rules, the plan and the profile, for the
    /// request `maker` makes under the document in force. A pick comes
    /// with its lease, issued before the answer leaves: the stages have
    /// heard of the request, and its booking stands on the first target.
    ///
    /// The flag beside the decision says whether the request the maker
    /// made stated the later decode
    /// ([`Constraints::decode_later`](super::view::Constraints::decode_later)).
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the request.
    fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<(DecideOutcome, bool), DecisionRefused>;

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
        let (decided, _) = self.decide_under(gateway, request, now)?;
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
    ) -> Result<(DecideOutcome, bool), DecisionRefused> {
        self.decide_under_stated(gateway, maker, now)
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
