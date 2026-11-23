//! An encode request's decision over three lists of records written for
//! it: its pair and the encoder of each of its media items, in the one call
//! the engine decides them in
//! ([`Engine::decide_encode_pair`](crate::engine::Engine::decide_encode_pair)).
//!
//! No affinity column and no signal cell stands beside the encoders, and
//! no ring is lent for them: an encode profile's stage that places by a
//! ring places by its own.
//!
//! **The entries.** A host asks through
//! [`PolicyEngine::decide_encode_at`]: the three lists' decision, with the
//! pair's lease a pick is issued. No lease is issued for an encoder.

use std::{sync::Arc, time::Instant};

use smg_worker::load::state::LoadSnapshot;

use super::{
    pair::{issued, not_narrowed_by, stamp, Buffers, PairLists, BUFFERS},
    single::{self, PLACES},
    PolicyEngine, PoolMembers,
};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{shed, Decision, ShedReason, WIRES},
        decision_engine::{DecideOutcome, DecisionRefused, Request, RequestMaker, WorkerLists},
        framework::{EncodeItems, EncodePlaced, EncodePool, EncoderFacts, PairPools},
        records::{DecisionClock, Live},
        view::Phase,
    },
};

#[cfg(test)]
mod tests;

/// What a request that states the later decode and asks images of its
/// decode is declined under: a decode asked again takes no items, so
/// nothing there holds it to a decode that takes images.
const DECODE_LATER: &str = "decode-later";

/// An encode request's three lists as their host holds them for one
/// decision: the pair's two lists, the encoders, and what the encode side
/// reads of the workers beside their records. A decision names each placed
/// key's encoder by its place in the encoder list, so the host finds it
/// with no lookup by id.
///
/// What a worker takes is its host's word. The engine holds every encoder
/// and every decode to the `vision` its host said with the list
/// ([`FixedFacts::vision`](crate::engine::view::FixedFacts::vision)) and
/// applies no rule of a runtime's own: a host that holds only some of its
/// decodes to a label says of the others that they take images.
#[derive(Clone, Copy, Debug)]
pub struct EncodeLists<'a> {
    /// The pair's two lists, with their pair classes and the host's reads
    /// of its admission.
    pub pair: PairLists<'a>,
    /// The encoders.
    pub encoders: &'a PoolMembers,
    /// What the encode side reads of each encoder beside its record, by its
    /// place in the encoder list
    /// ([`EncodeClasses::encoder_facts`](crate::store::EncodeClasses::encoder_facts)).
    pub facts: &'a [EncoderFacts],
    /// The encode class each prefill names, one entry a prefill, by its
    /// place in the prefill list
    /// ([`EncodeClasses::prefill_class`](crate::store::EncodeClasses::prefill_class)).
    pub prefill_classes: &'a [u32],
}

impl EncodeLists<'_> {
    /// One decision for an encode request at `now`: its pair, and the
    /// encoder of each key of `items`, which `placed` takes. One record is
    /// written for each worker of the three lists, in its list's order, as
    /// the workers stand at the call, with `loads` as a pair's decision
    /// reads them; the pair's records are stamped as a pair's
    /// ([`PairLists::decide`]). Whether an encoder or a decode takes images
    /// is on its record as its host said when it handed the list
    /// ([`PolicyEngine::pool_members_with_fixed_facts`](super::PolicyEngine::pool_members_with_fixed_facts)).
    ///
    /// `request` is the pair's, as its host made it for the prefill phase.
    /// This call's instant is written into its facts: `now` on `clock`.
    ///
    /// A pair's lists are narrowed by no demand, no subset hint and no
    /// places: a request that states one is shed or declined as its pair's
    /// own decision answers it, and no key is placed. One side of the pair
    /// alone is no pair, and the engine says so behind the encoders: a
    /// request that no encoder is ready for is shed under the encode side's
    /// stage. For a request with items, facts or classes of another length
    /// than their list are declined by the engine under the column's name:
    /// a request with none reads neither.
    ///
    /// Lists whose pair is held to places ([`PairLists::prefill_places`],
    /// [`PairLists::decode_places`]) are declined under the places' name,
    /// and no key is placed: the engine decides an encode request's pair
    /// over the workers it admits for the items and reads no place of the
    /// lists.
    ///
    /// No lease is issued here and no stage is told of one: the engine's
    /// entry issues the pair's
    /// ([`PolicyEngine::decide_encode_at`](super::PolicyEngine::decide_encode_at)).
    /// A request held to wires is not declined here: that entry tests the
    /// lists' wires ahead of this call.
    #[expect(
        clippy::too_many_arguments,
        reason = "a pair's inputs and the encode side's"
    )]
    pub fn decide(
        &self,
        spec: &AppliedSpec,
        loads: Option<&Arc<LoadSnapshot>>,
        clock: &DecisionClock,
        now: Instant,
        request: &mut Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
    ) -> Decision {
        let pair = &self.pair;
        if not_narrowed_by(&request.constraints).is_some() {
            placed.clear();
            return pair.decide(spec, loads, clock, now, request, None);
        }
        // One value for the three lists' records, as a pair's decision
        // takes one for its two.
        let live = Live::at(spec.engine(), clock, now, loads, None);
        request.facts.now_micros = live.now_micros();
        // The engine's pass decides the pair over the prefills and the
        // decodes it admits for the items: a place of the lists as their
        // host holds them names no row there. The call's instant is
        // written first, as every answer of this entry leaves it.
        if pair.prefill_places.is_some() || pair.decode_places.is_some() {
            placed.clear();
            let (seed, generation) = (request.facts.seed, spec.generation());
            return shed(ShedReason::PolicyDeclined, PLACES, seed, generation);
        }
        let (facts, cx) = (&request.facts, &request.constraints);
        let mut decide = |buffers: &mut Buffers| {
            let [prefills, decodes] = &mut buffers.rows;
            let (workers, records) = (pair.prefill.workers(), pair.prefill.records(spec, clock));
            let (prefills, pool) = records.write(workers, &live, prefills);
            let (workers, records) = (pair.decode.workers(), pair.decode.records(spec, clock));
            let (decodes, _) = records.write(workers, &live, decodes);
            stamp(prefills, decodes, pair.classes, pair.admission);
            let (workers, records) = (self.encoders.workers(), self.encoders.records(spec, clock));
            let (encoders, _) = records.write(workers, &live, &mut buffers.encoders);
            let pools = PairPools {
                prefills,
                decodes,
                table: pair.classes.table(),
                ..PairPools::default()
            };
            let encode = EncodePool {
                encoders,
                facts: self.facts,
                prefill_classes: self.prefill_classes,
                ..EncodePool::default()
            };
            let (engine, scratch) = (spec.engine(), &mut buffers.scratch);
            engine.decide_encode_pair(
                facts, &pool, pools, encode, items, cx, cx, cx, scratch, placed,
            )
        };
        BUFFERS.with(|buffers| match buffers.try_borrow_mut() {
            Ok(mut buffers) => decide(&mut buffers),
            // A stage asked for a decision from inside its own, on this
            // thread: that decision takes buffers of its own.
            Err(_) => decide(&mut Buffers::default()),
        })
    }
}

impl PolicyEngine {
    /// One encode request's decision over its host's three lists
    /// ([`WorkerLists::Encode`]), as its host takes it: the pair, with the
    /// lease its pick is issued as a pair's first decision is, and the
    /// encoders of `items`, which `placed` takes. With the answer comes
    /// whether the request made stated the later decode: what
    /// a guard holds the answer by. `clock` is read once, where the
    /// decision is dated.
    ///
    /// The decision reads the load reports a pair's first decision reads:
    /// the snapshot the request gives for this call, or the newest the
    /// engine's own task keeps at hand
    /// ([`RequestMaker::loads`]).
    ///
    /// No lease is issued for an encoder and no stage is told of one. A
    /// request held to wires is declined under the wires' name where a
    /// worker of a list it reads is dialed on a wire it does not state:
    /// the pair's two lists, and the encoders for a request with items.
    ///
    /// A request that states the later decode and asks images of its
    /// decode ([`EncodeItems::decode_images`]) is declined under
    /// `decode-later`: a decode asked again takes no items, so nothing
    /// there holds it to a decode that takes images.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused::PairSide`] for a request of any phase but
    /// `Prefill`, and for a maker that hands other lists than an encode
    /// request's three. [`DecisionRefused::Faulted`] where the decision
    /// panicked. `placed` is left empty where no decision is made.
    pub fn decide_encode_at(
        &self,
        maker: &impl RequestMaker,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        clock: impl FnOnce() -> Instant,
    ) -> Result<(DecideOutcome, bool), DecisionRefused> {
        self.over_three(maker, placed, |lists, request, loads, placed| {
            let later = request.constraints.decode_later;
            let decided = decided(self, lists, request, items, loads, placed, clock());
            (decided, later)
        })
    }

    /// [`Self::decide_encode_at`] with nothing its host hears of later: no
    /// lease is issued, no id is taken and no stage is told. For a host
    /// that files a pick's lease itself, as [`Self::decide_without_lease`]
    /// is for a single request and for a pair. The wires' test stands in
    /// it.
    ///
    /// # Errors
    ///
    /// As [`Self::decide_encode_at`].
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the controller's service calls it once it decides by this engine"
        )
    )]
    pub(crate) fn decide_encode_without_lease(
        &self,
        maker: &impl RequestMaker,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<Decision, DecisionRefused> {
        self.over_three(maker, placed, |lists, request, loads, placed| {
            decide(self, lists, request, items, loads, placed, now)
        })
    }

    /// The request `maker` makes, decided by `decide` over the three lists
    /// the maker hands, with the load reports a pair's first decision
    /// reads ([`Self::loads_read`]), under the engine's catch. A request of
    /// another phase than `Prefill`, or other lists, is a side sent on its
    /// own. `placed` is left empty where no decision is made.
    fn over_three<R>(
        &self,
        maker: &impl RequestMaker,
        placed: &mut EncodePlaced,
        decide: impl FnOnce(
            &EncodeLists<'_>,
            &mut Request<'_>,
            Option<&Arc<LoadSnapshot>>,
            &mut EncodePlaced,
        ) -> R,
    ) -> Result<R, DecisionRefused> {
        let decided = self.caught(|| {
            let mut request = self.made(maker);
            let asked = (request.facts.phase, maker.workers());
            let (Phase::Prefill, WorkerLists::Encode(lists)) = asked else {
                return Err(DecisionRefused::PairSide);
            };
            let read = self.loads_read(maker);
            Ok(decide(&lists, &mut request, read.snapshot(), placed))
        });
        // A refusal, and a decision that panicked, leave no key placed.
        if decided.is_err() {
            placed.clear();
        }
        decided
    }
}

/// One decision of an encode request by `own` over the `lists` its host
/// handed, and nothing its host hears of later: no lease is issued, no id
/// is taken and no stage is told. The lists' own decision
/// ([`EncodeLists::decide`]) reads the reports in `loads`, as a pair's
/// first decision reads them ([`super::pair::decide`]).
///
/// A request held to wires is its host's to hand lists for. Where a worker
/// of a list the request reads is dialed on a wire it does not state, the
/// request is declined under the wires' name, as a pair is, and no key is
/// placed. A request with items reads the three lists, and one with none
/// the pair's two: it takes no encoder.
///
/// A request that states the later decode and asks images of its decode is
/// declined under [`DECODE_LATER`], and no key is placed: the entries that
/// decide a decode again take no items
/// ([`WorkerLists::Encode`]). The preference of a request with image
/// references for a decode that takes images
/// ([`EncodeItems::references`]) is no demand: that request is decided.
#[inline]
pub(super) fn decide(
    own: &PolicyEngine,
    lists: &EncodeLists<'_>,
    request: &mut Request<'_>,
    items: EncodeItems<'_>,
    loads: Option<&Arc<LoadSnapshot>>,
    placed: &mut EncodePlaced,
    now: Instant,
) -> Decision {
    let (wires, pair) = (request.constraints.wire_modes, &lists.pair);
    if !wires.is_empty() {
        let dialed = pair.prefill.wires().union(pair.decode.wires());
        let dialed = if items.keys.is_empty() {
            dialed
        } else {
            dialed.union(lists.encoders.wires())
        };
        if !wires.contains(dialed) {
            placed.clear();
            return single::declined(own, request, WIRES);
        }
    }
    if items.decode_images && request.constraints.decode_later {
        placed.clear();
        return single::declined(own, request, DECODE_LATER);
    }
    lists.decide(&own.spec, loads, &own.clock, now, request, items, placed)
}

/// One decision of an encode request by `own`, the engine of a gateway's
/// own process, over the `lists` its host handed, as the host takes it: the
/// decision ([`decide`]) with the lease a pick issued. A pick issues its
/// pair's two sides as a pair's first decision does ([`issued`]), and
/// nothing for an encoder: no stage is told of one. A decision that picks
/// no pair issues nothing.
pub(super) fn decided(
    own: &PolicyEngine,
    lists: &EncodeLists<'_>,
    request: &mut Request<'_>,
    items: EncodeItems<'_>,
    loads: Option<&Arc<LoadSnapshot>>,
    placed: &mut EncodePlaced,
    now: Instant,
) -> DecideOutcome {
    let decision = decide(own, lists, request, items, loads, placed, now);
    issued(own, request, decision, None, now)
}
