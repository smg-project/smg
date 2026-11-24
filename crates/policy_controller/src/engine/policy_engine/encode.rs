//! An encode request's decision over three lists of records written for
//! it: its pair and the encoder of each of its media items, in the one call
//! the engine decides them in
//! ([`Engine::decide_encode_pair`](crate::engine::Engine::decide_encode_pair)).
//!
//! No affinity column stands beside the encoders, and no ring is lent for
//! them: an encode profile's stage that places by a ring places by its
//! own.
//!
//! **Named signals.** Under a document a stage of which reads a named
//! signal, each of the three sides whose profile reads one is lent its own
//! list's signal cells, of the snapshot the decision holds
//! ([`signals::three`]). The decode asked again is a pair's
//! ([`PairLists::decide`]) and is lent as one.
//!
//! **The entries.** A host asks through
//! [`PolicyEngine::decide_encode_at`]: the three lists' decision, with the
//! pair's lease a pick is issued. No lease is issued for an encoder.

use std::{sync::Arc, time::Instant};

use smg_worker::load::state::LoadSnapshot;

use super::{
    pair::{issued, not_decided, stamp, tried_engines, Buffers, PairLists, BUFFERS},
    signals, single, Loaded, PolicyEngine, PoolMembers,
};
use crate::engine::{
    decision::{Decision, WIRES},
    decision_engine::{DecideOutcome, DecisionRefused, Request, RequestMaker, WorkerLists},
    framework::{EncodeItems, EncodePlaced, EncodePool, EncoderFacts, PairPools},
    list_cell::LentList,
    records::{HeldCells, Live},
    view::{Constraints, Phase, RequestFacts},
};

#[cfg(test)]
mod tests;

/// What a request that states the later decode is declined under where its
/// decode is held to the ones that take images, by its demand or by
/// references beside an image or a video: a decode asked again takes no
/// items, so nothing there holds it to a decode that takes images.
const DECODE_LATER: &str = "decode-later";

/// An encode request's three lists as their host lends them for one
/// decision: the pair's two lists, the encoders with their cell, and what
/// the encode side reads of the workers beside their records. A decision names each placed
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
    pub encoders: LentList<'a>,
    /// The places of the encoder list the request may take, ascending,
    /// each once; `None` for every place. The request's own fact and no
    /// veto of a worker, stated as the pair's are
    /// ([`PairLists::prefill_places`]): an item is placed on an encoder at
    /// one of them under any document, and a prefill that only other
    /// encoders hand to is not the pair's. A place past the list's end
    /// names no encoder.
    pub encoders_places: Option<&'a [u32]>,
    /// The places of the encoder list a standing veto of the host masks,
    /// ascending, each once; empty for none ([`Constraints::stalled`]). The
    /// encode side reads them as a single request's list reads its own. A
    /// place past the list's end names no encoder and is passed by without
    /// a word: a host's standing veto may name a place of an older list.
    pub encoders_stalled: &'a [u32],
    /// The places of [`Self::encoders_stalled`] a hold of the host stands
    /// on, which no rule lifts ([`Constraints::held`]). A place past the
    /// list's end is passed by here too.
    pub encoders_held: &'a [u32],
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
    /// is on its record as its host lends it with the list
    /// ([`LentList::facts`]).
    ///
    /// `engine` decides it: under its document, on its clock, and over its
    /// own value of each of the three lists, which it asks of the list's
    /// cell.
    ///
    /// `request` is the pair's, as its host made it for the prefill phase.
    /// This call's instant is written into its facts: `now` on the engine's
    /// clock.
    ///
    /// A request that states no budget is decided with the widest one
    /// written into its facts too, as a pair's is ([`PairLists::decide`]).
    ///
    /// These lists are narrowed by no demand and no subset hint, and a
    /// request's own places and its own mask name places of no one list: a
    /// request that states one of these is shed or declined as its pair's
    /// records answer it, and no key is placed. A pair's own request is
    /// held to a demand a side at a time ([`PairLists::decide`]), and an
    /// encode request is not. One side of the pair
    /// alone is no pair, and the engine says so behind the encoders: a
    /// request that no encoder is ready for is shed under the encode side's
    /// stage. For a request with items, facts or classes of another length
    /// than their list are declined by the engine under the column's name:
    /// a request with none reads neither.
    ///
    /// Each list's host may hold the request to places of that list: the
    /// pair's two ([`PairLists::prefill_places`],
    /// [`PairLists::decode_places`]) and the encoders'
    /// ([`Self::encoders_places`]). Each side of the pair is decided among
    /// its own, numbered with the workers the engine admits for the items,
    /// and each item's encoder is one at the encoders' places. A list
    /// whose places name workers and none that takes the request, though a
    /// worker outside them does, sheds it under the places' name, and no
    /// key is placed: the request is held off every worker that takes it.
    /// Places that name no worker of a list are no such case. Such places
    /// of a side shed the request as they shed its pair alone, and such
    /// places of the encoders shed it as a list with no open encoder does.
    ///
    /// Each side of the pair reads the places its host's standing vetoes
    /// mask and hold beside that side's list
    /// ([`PairLists::prefill_stalled`], [`PairLists::decode_stalled`]),
    /// numbered the same way. A side whose host states none reads no mask.
    /// A side whose profile shows its rows to a plugin is declined beside
    /// a mask, under the stage `written-rows`.
    ///
    /// The encoders read their own list's vetoes the same way
    /// ([`Self::encoders_stalled`], [`Self::encoders_held`]). A stage of
    /// the encode profile reads the two numbered with the rows it is
    /// handed: by the list where the engine asks which encoders are open,
    /// and with the encoders offered where a key is placed. The mask is
    /// the health filter's to read, as a single request's is: under a
    /// profile that lists it, a masked encoder is turned away as one that
    /// is not ready is. It takes no key, a prefill whose class's encoders
    /// are all masked is not admitted, and with every encoder masked the
    /// request is shed as with none ready. No rule of the encode side
    /// lifts a veto, so a hold answers as a mask does. A profile that
    /// lists no health filter reads no mask. No mask is written into an
    /// encoder's record: an encode profile that shows its rows to a plugin
    /// is declined for a request with items, under the stage
    /// `encode-scope`, with a veto as without.
    ///
    /// No lease is issued here and no stage is told of one: the engine's
    /// entry issues the pair's
    /// ([`PolicyEngine::decide_encode_at`](super::PolicyEngine::decide_encode_at)).
    /// A request held to wires is not declined here: that entry tests the
    /// lists' wires ahead of this call. The decision itself leaves out an
    /// encoder dialed on none of them.
    pub fn decide(
        &self,
        engine: &PolicyEngine,
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
    ) -> Decision {
        let engine = engine.loaded();
        let sides = self.asked(engine);
        self.decide_over(engine, sides, None, loads, now, request, items, placed)
    }

    /// The engine's own value of each of the pair's two lists, asked of
    /// the list's cell: the prefills', the decodes'.
    fn asked(&self, engine: Loaded<'_, '_>) -> (&PoolMembers, &PoolMembers) {
        let pair = &self.pair;
        (
            engine.members_of(pair.prefill),
            engine.members_of(pair.decode),
        )
    }

    /// [`Self::decide`] over the engine's values of the pair's two lists,
    /// which its caller asked of their cells, and of the encoders' where
    /// its caller read it. The encoders' cell is else asked here, behind
    /// every answer that reads no row of an encoder: a decision fills no
    /// cell it does not read.
    #[expect(
        clippy::too_many_arguments,
        reason = "a pair's inputs and the encode side's"
    )]
    fn decide_over(
        &self,
        engine: Loaded<'_, '_>,
        (prefill, decode): (&PoolMembers, &PoolMembers),
        encoders: Option<&PoolMembers>,
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
    ) -> Decision {
        let (spec, clock) = (&engine.document.spec, &engine.clock);
        let pair = &self.pair;
        // A hard demand is among what these lists do not decide: the pair
        // sheds it by its records, and holds no side to it for this entry.
        if not_decided(&request.constraints).is_some() {
            placed.clear();
            return pair.decide_unjudged(engine, (prefill, decode), loads, now, request);
        }
        // One value for the three lists' records, as a pair's decision
        // takes one for its two.
        let live = Live::at(spec.engine(), clock, now, loads, None);
        request.facts.now_micros = live.now_micros();
        request.facts.budget_micros = super::budget::shown(request.facts.budget_micros);
        // Each side reads the places and the mask its host states beside
        // its list. The engine's pass decides the pair over the prefills
        // and the decodes it admits for the items, and numbers both with
        // them.
        let (facts, cx) = (&request.facts, &request.constraints);
        let [prefill_cx, decode_cx] = pair.sides_of(cx);
        // The encoders read their own list's places and vetoes: the
        // request states none of its own here.
        let encode_cx = Constraints {
            places: self.encoders_places,
            stalled: self.encoders_stalled,
            held: self.encoders_held,
            ..*cx
        };
        let encoders = encoders.unwrap_or_else(|| engine.members_of(self.encoders));
        // A retry leaves every worker of the engines it tried, each of the
        // three lists by its own engines, as a pair's retry does over its
        // two. The pair's pass and the encoders' profile read one list.
        let tried = if facts.tried.is_empty() {
            None
        } else {
            tried_engines(engine, facts.tried, &[prefill, decode, encoders])
        };
        let widened;
        let facts = match &tried {
            Some(tried) => {
                widened = RequestFacts { tried, ..*facts };
                &widened
            }
            None => facts,
        };
        let mut decide = |buffers: &mut Buffers| {
            let [prefills, decodes] = &mut buffers.rows;
            let (workers, records) = (prefill.workers(), prefill.records(spec, clock));
            let (prefills, pool) = records.write(workers, &live, prefills);
            let (workers, records) = (decode.workers(), decode.records(spec, clock));
            let (decodes, _) = records.write(workers, &live, decodes);
            stamp(prefills, decodes, pair.classes, pair.admission);
            let (workers, records) = (encoders.workers(), encoders.records(spec, clock));
            let (rows, _) = records.write(workers, &live, &mut buffers.encoders);
            // Under a document a stage of which reads a named signal, each
            // of the three sides that reads one is lent its list's cells
            // for this call.
            let lent = spec.needs().signal_cells.then(|| {
                let members = [prefill, decode, encoders];
                signals::three(members, spec, clock, &live, facts.model)
            });
            let [prefill_signal_cells, decode_signal_cells, signal_cells] = match &lent {
                Some(lent) => lent
                    .each_ref()
                    .map(|side| side.as_ref().and_then(HeldCells::cells)),
                None => [None; 3],
            };
            let pools = PairPools {
                prefills,
                decodes,
                prefill_signal_cells,
                decode_signal_cells,
                table: pair.classes.table(),
                first: pair.classes.first(),
                ..PairPools::default()
            };
            let encode = EncodePool {
                encoders: rows,
                signal_cells,
                facts: self.facts,
                prefill_classes: self.prefill_classes,
                ..EncodePool::default()
            };
            let (engine, scratch) = (spec.engine(), &mut buffers.scratch);
            engine.decide_encode_pair(
                facts,
                &pool,
                pools,
                encode,
                items,
                &prefill_cx,
                &decode_cx,
                &encode_cx,
                scratch,
                placed,
            )
        };
        let decision = BUFFERS.with(|buffers| match buffers.try_borrow_mut() {
            Ok(mut buffers) => decide(&mut buffers),
            // A stage asked for a decision from inside its own, on this
            // thread: that decision takes buffers of its own.
            Err(_) => decide(&mut Buffers::default()),
        });
        // The thread's list goes back for its next retry.
        if let Some(tried) = tried {
            single::TRIED.set(tried);
        }
        decision
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
    /// worker of its pair's two lists is dialed on a wire it does not
    /// state. Its items are placed on the encoders dialed on a wire it
    /// states, and a request with items is declined under that name too
    /// where the list holds encoders and none of them is. A request with
    /// no item reads no encoder's wire.
    ///
    /// A request that states the later decode is declined under
    /// `decode-later` where it asks images of its decode
    /// ([`EncodeItems::decode_images`]), and where it carries references
    /// beside an image or a video ([`EncodeItems::references`]), which
    /// prefer a decode that takes images: a decode asked again takes no
    /// items, so nothing there holds it to such a decode. The decline reads
    /// the request's own facts and no list: it stands also where no decode
    /// of the lists takes images.
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
        self.loaded().decide_encode_at(maker, items, placed, clock)
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
        let own = self.loaded();
        own.over_three(maker, placed, |lists, request, loads, placed| {
            decide(own, lists, request, items, loads, placed, now)
        })
    }
}

impl Loaded<'_, '_> {
    /// [`PolicyEngine::decide_encode_at`] under the document this value
    /// holds: for a caller that reads that document further behind the
    /// decision, as the call that places a request does.
    pub(super) fn decide_encode_at(
        self,
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

    /// The request `maker` makes, decided by `decide` over the three lists
    /// the maker hands, with the load reports a pair's first decision
    /// reads ([`PolicyEngine::loads_read`]), under the engine's catch. A request of
    /// another phase than `Prefill`, or other lists, is a side sent on its
    /// own. `placed` is left empty where no decision is made.
    fn over_three<R>(
        self,
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
/// A request held to wires is its host's to hand a pair's lists for. Where
/// a worker of the pair's two lists is dialed on a wire the request does
/// not state, the request is declined under the wires' name, as a pair is,
/// and no key is placed.
///
/// The encoders are narrowed here, by the decision. An encoder is named by
/// its place in its list, and what its host says of it and of each prefill
/// is laid by that place ([`EncodeLists::facts`],
/// [`EncodeLists::prefill_classes`]), so the host has no narrowed list to
/// hand. A request with items takes each item's encoder among the encoders
/// dialed on a wire it states, and a prefill that only other encoders hand
/// to is not its pair's. Where the list holds encoders and none is dialed
/// on a stated wire, the request is declined under the wires' name too. A
/// request with no item reads the pair's two lists and no encoder's wire:
/// it takes no encoder.
///
/// A request that states the later decode is declined under
/// [`DECODE_LATER`], and no key is placed, where its decode is held to the
/// ones that take images: by its demand ([`EncodeItems::decode_images`]),
/// or by the preference of references beside an image or a video
/// ([`EncodeItems::references`]). The entries that decide a decode again
/// take no items ([`WorkerLists::Encode`]). The decline reads the request's
/// own facts and no list, so it stands also where no decode takes images.
#[inline]
pub(super) fn decide(
    own: Loaded<'_, '_>,
    lists: &EncodeLists<'_>,
    request: &mut Request<'_>,
    items: EncodeItems<'_>,
    loads: Option<&Arc<LoadSnapshot>>,
    placed: &mut EncodePlaced,
    now: Instant,
) -> Decision {
    // The engine's own value of each of the pair's lists, asked of the
    // list's cell once. The encoders' is asked where it is read: here
    // for the wires of a request with items whose pair's lists are on its
    // wires, and else where their rows are written.
    let sides = lists.asked(own);
    let mut encoders = None;
    let wires = request.constraints.wire_modes;
    if !wires.is_empty() {
        let (prefill, decode) = sides;
        // The pair's two lists are their host's to hand on the request's
        // wires: every worker of both is dialed on one.
        let mut on_wires = wires.contains(prefill.wires().union(decode.wires()));
        if on_wires && !items.keys.is_empty() {
            // The encoders are held to less: the decision leaves out each
            // one dialed on no stated wire, so the list answers where one
            // of its encoders is dialed on one. A list of no encoder is
            // the encode side's own gap.
            let asked = own.members_of(lists.encoders);
            let theirs = asked.wires();
            on_wires = theirs.is_empty() || theirs.intersects(wires);
            encoders = Some(asked);
        }
        if !on_wires {
            placed.clear();
            return single::declined(own, request, WIRES);
        }
    }
    let images = items.holds_decode_to_images(request.facts.media.holds_pixels());
    if images && request.constraints.decode_later {
        placed.clear();
        return single::declined(own, request, DECODE_LATER);
    }
    lists.decide_over(own, sides, encoders, loads, now, request, items, placed)
}

/// One decision of an encode request by `own`, the engine of a gateway's
/// own process, over the `lists` its host handed, as the host takes it: the
/// decision ([`decide`]) with the lease a pick issued. A pick issues its
/// pair's two sides as a pair's first decision does ([`issued`]), and
/// nothing for an encoder: no stage is told of one. A decision that picks
/// no pair issues nothing.
pub(super) fn decided(
    own: Loaded<'_, '_>,
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
