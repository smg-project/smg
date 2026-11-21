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
//! **What a guard hands back.** A guard holds the lease as its decision
//! issued it and the target its host took, and hands both with each word
//! ([`DecisionEngine::say`], [`DecisionEngine::decide_decode_under`]). An
//! engine with a file of its leases finds its record by the lease's id. An
//! engine with none needs no table to find one: every fact a stage is told
//! of a request is in the lease and the target.
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
    decision::{ShadowKind, Target},
    framework::{EncodeItems, EncodePlaced},
    prompt_hashes::{HashSeeds, PromptHashes},
    store_backed_engine::{
        self, DecideOutcome, DecisionRefused, Request, StoreBackedEngine, SweepOutcome,
    },
    view::WireModes,
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

    /// The workers the request may be sent to, where its host holds them:
    /// the lists the decision's targets name by position
    /// ([`Target::position`], [`Pair::decode_position`](super::decision::Pair::decode_position)).
    /// An engine with no store decides over these and refuses a request
    /// that hands none. An engine that reads its own rows asks for none,
    /// and a host of such an engine states none.
    #[cfg(feature = "in-process")]
    fn workers(&self) -> HostWorkers<'_> {
        HostWorkers::None
    }
}

impl RequestMaker for Request<'_> {
    fn model(&self) -> &str {
        self.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        *self
    }
}

/// The workers a request may be sent to, as its host holds them for one
/// decision. A list is the host's own, read once for the decision: the host
/// finds a target's worker at the target's position in the list it handed,
/// with no lookup by id.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug, Default)]
pub enum HostWorkers<'a> {
    /// The host hands no list.
    #[default]
    None,
    /// A single request's list.
    Single(&'a super::policy_engine::PoolMembers),
    /// A pair's two lists: its prefills and its decodes.
    Pair {
        prefill: &'a super::policy_engine::PoolMembers,
        decode: &'a super::policy_engine::PoolMembers,
    },
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
    /// `lease` is the pair's lease as its first decision issued it and
    /// `prefill` the target its host took: the guard that holds the first
    /// decision hands both. A pick issues the decode side under the
    /// lease's id.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the call.
    fn decide_decode_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        lease: &Lease,
        prefill: &Target,
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
        lease: &Lease,
        prefill: &Target,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        self.decide_decode_under(gateway, request, lease, prefill, now)
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
    /// end, or a release. `lease` is the lease as its decision issued it,
    /// and `target` the target its host took, a pair's decode among it. The
    /// word names its side. Whoever applies the word tells the stages of
    /// the engine that issued the lease, once.
    fn say(&self, gateway: GatewayId, lease: &Lease, target: &Target, said: Said, now: Instant);

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
        lease: &Lease,
        _: &Target,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_decode_under(self, gateway, maker, lease.id, now)
    }

    #[inline]
    fn decide_decode(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        lease: &Lease,
        _: &Target,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_decode(self, gateway, request, lease.id, now)
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

    // The file finds the lease's record by its id. A completion is also
    // written on the served worker's row, for the health precedence.
    #[inline]
    fn say(&self, gateway: GatewayId, lease: &Lease, _: &Target, said: Said, now: Instant) {
        match said {
            Said::Completed(outcome) => {
                store_backed_engine::completed(self.store(), gateway, &outcome, now, |said| {
                    self.report(gateway, lease.id, said);
                });
            }
            Said::Dispatched(..) | Said::Released(..) => self.report(gateway, lease.id, said),
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
