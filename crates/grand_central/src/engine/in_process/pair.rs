//! A pair's decision and the decode asked again, over two lists of records
//! written for one decision.
//!
//! The engine of a gateway's own process keeps no file of leases. A pair's
//! note is what the request's guard holds ([`HeldDecision`]): the pick, the
//! prefill side's lease as issued, the target its host took and what each
//! side still owes. The functions here make a pair's two decisions over
//! the records of its host's two lists, and tell the document's stages each
//! word of a pair's lease, a side at a time.
//!
//! **The lists.** A host lends its two lists for one decision
//! ([`PairLists`]): each list with its cell, the pair class of every
//! worker, and its own reads of its admission. The engine asks each side's
//! cell for its own value of the list, once a decision. A pair's first
//! decision writes one record for each worker of the prefill list and of
//! the decode list, each in its list's order, as the workers stand at that
//! call ([`PoolRecords::write`](crate::engine::records::PoolRecords::write)).
//! The decode decided again writes the decodes' records and, of the
//! prefills', the held prefill's alone where its list names its place
//! ([`PoolRecords::read`](crate::engine::records::PoolRecords::read)): it
//! reads no other prefill's worker. It is handed the decode its pair's
//! first decision named: a decode profile that keeps its pick answers that
//! decode while its filters keep it
//! ([`Engine::decide_decode`](crate::engine::framework::Engine::decide_decode)).
//!
//! **The records.** [`stamp`] writes on them what a pair reads beside a
//! single: each record's pair class, and its side's admission bit as the
//! host reads it at that instant ([`PairAdmission`]). The stock pair
//! document filters by both bits (`has-capacity`, `has-room`).
//!
//! **A hard demand** ([`Demands`]). A pair's request that states one is
//! held to it a side at a time: each side to the places of its list whose
//! worker meets the demand as that side's role is held to it
//! ([`Demands::held_by`]), among the places its host states for the side,
//! by a single request's own rule ([`single::held_in`]). A side the demand
//! leaves nobody sheds the pair with no endpoint, under the stage of the
//! demand that took that side's last worker, the prefills' first. The
//! decode asked again judges its decodes alone. An encode request's pair
//! is held by this rule too ([`PairLists::sides_meeting`]), and its
//! encoders by the engine's encode pass ([`super::encode`]).
//!
//! **The words.** A side is told its issue once: with the prefill's
//! ([`issue_caught`]), or, where its decode is sent later, at the decode
//! call ([`issue_decode`]) or at the dispatch that sends the decode with no
//! call ([`lease_events::pair`]). A decode call whose placement can send to
//! none of its decodes gives the side up by name, and the next call, or the
//! dispatch of the first decode, issues it again
//! ([`Placement::decode_again`](crate::tracking::Placement::decode_again)).
//! A side's dispatch and its end are told under that
//! side's lease. It names the worker the side booked at its issue until
//! the side's dispatch, and the worker it went to from there. A side its
//! sessions rule placed alone and keeps from the profile is told to no
//! stage ahead of its end, each side by its own word
//! ([`lease_events::stages_hear`]). The decode asked again that names the
//! decode its pair holds keeps that pair's word of the side: the decision
//! that put the request there stands
//! ([`Targets::decode_placed_as`](crate::engine::decision::Targets::decode_placed_as)).
//!
//! **The answer.** The engine asks through [`decided`], for a pair's first
//! decision and for its decode asked again: the decision ([`decide`]), with
//! the lease a pick issued.
//!
//! No function here waits for a lock. A decision's records are written into
//! buffers of the calling thread, so it allocates nothing once they have
//! grown to the lists. The one exception stands under a snapshot of load
//! reports: one decision converts a list's reports, at most once a
//! published snapshot a list. It allocates the converted rows, under the
//! list's claim, which is tried and never waited for
//! ([`PoolRecords::write`](crate::engine::records::PoolRecords::write)). A
//! stage takes what it takes of its own state, where it scores a record by
//! its report and where it hears a word.
//!
//! **Who holds the prompt.** A side whose phase reads affinity is handed
//! its own list's column for the decision ([`overlap::column`]): what
//! each worker of the side holds of the request, by the host's index of
//! cache events and by the record of sent prefixes, read at the side's
//! phase. The record is read by the side's own profile's word of it: for
//! every worker where that profile reads sent prefixes alone.
//! Behind a pick its profile is told of, each side's record keeps where
//! the prompt went ([`overlap::record_by_column`]). A pair's first pick
//! states what its prefill holds of the request, by the prefills' column
//! ([`overlap::state`]): a host observes its cache series from it. Nothing
//! is stated of the decode. The decode asked again
//! reads the decodes' column alone. The two sides write into the model's
//! one record under its one bound: a pair's request spends a placement a
//! block for each side, where a single's spends one.
//!
//! **Named signals.** Under a document a stage of which reads a named
//! signal, a side whose profile reads one is lent its own list's signal
//! cells, of the snapshot the decision holds ([`signals::sides`]). They are
//! made by the conversion above. A decision that was not the one to
//! convert writes its own into a buffer of the calling thread, and where
//! both sides do, the second side's are allocated.
//!
//! **A side's shadow.** A pair's binding may name a shadow profile a side.
//! A pair's first decision made through the entry that issues its lease
//! ([`decided`]) decides each such side once more by that profile
//! ([`side_shadows`]), inside the calling thread's buffers and behind the
//! live decision, which it changes nothing of. The decode decided again
//! through that entry decides its decode side once more
//! ([`later_shadow`]). What each said is counted by its side, a decode
//! decided again apart, ahead of the issue
//! ([`InProcess::shadow_counts`]). A document that names no shadow pays
//! the test of that word at the entry and one test of the room in the
//! decision, and no call. No other entry asks a side's shadow yet: not
//! the decision alone ([`decide`]), nor an encode request's first
//! decision. An encode request's decode decided again comes through this
//! entry: its decode side is asked as a pair's is and counted under
//! [`ShadowSide::DecodeAgain`], with no count of its first decision
//! beside it.

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

use std::{
    cell::{Cell, RefCell},
    fmt,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot};

use super::{overlap, signals, single, with_shadow, InProcess, Loaded, PoolMembers};
use crate::{
    engine::{
        decision::{shed, Decision, Pick, ShadowKind, ShedReason, Target, SUBSET_HINT, WIRES},
        decision_engine::{DecideOutcome, HeldDecision, Leased, Request},
        demands::{Demands, UnmetDemand},
        framework::{Engine, PairPools, PairSide, Scratch, ShadowRows, ShedOver},
        lease_events,
        list_cell::LentList,
        pair_classes::PartnerClasses,
        records::{vetoes, HeldCells, Live, PoolRecords},
        shadow::{budget_left, ShadowSide, Shadowed, SHADOW_FLOOR},
        view::{
            AffinityFacts, Candidate, Constraints, Feed, Phase, PoolFacts, RequestFacts, Role,
            Session, SessionCountedUnder, WorkerId,
        },
    },
    stages::Lifecycle,
    tracking::lease::{Lease, LeaseId, LeaseKind, Side, DISPATCH_TTL},
};

/// The filter a pair's shed names where the admission limit alone holds
/// back a prefill with an open partner, or the one prefill the pair's
/// default took.
const HAS_CAPACITY: &str = "has-capacity";

/// Whether `decision` sheds a pair at the prefill admission limit: the
/// capacity bit its host stated ([`PairAdmission::has_capacity`]) alone
/// left the prefills with an open partner nobody, or the one prefill the
/// pair's default took by load is at its limit. Such a request is not
/// refused. Its host makes it wait for a prefill's room and decides it
/// again.
#[must_use]
pub fn prefills_full(decision: &Decision) -> bool {
    matches!(decision, Decision::Shed(shed) if shed.emptied_by == HAS_CAPACITY)
}

/// What the host of a pair reads of its own admission at a decision's
/// instant, a worker at a time. Each answer is a live read of the host's
/// own gate: nothing of it is kept.
pub trait PairAdmission: fmt::Debug {
    /// Whether the prefill at place `prefill` of the prefill list can take
    /// one more request by the host's prefill limit.
    fn has_capacity(&self, prefill: usize) -> bool;

    /// Whether the decode at place `decode` of the decode list has a room
    /// that nothing waiting at its gate is in line for.
    fn has_room(&self, decode: usize) -> bool;
}

/// A pair's two lists as their host lends them for one decision: each list
/// with its cell, the pair class of every worker of the two, and the
/// host's reads of its own admission. A decision names a prefill by
/// its place in the prefill list and a decode by its place in the decode
/// list, so the host finds both workers with no lookup by id.
///
/// A host that holds the request to a part of a pool states the places of
/// that side's list it may take ([`Self::prefill_places`],
/// [`Self::decode_places`]). The lists stay the pools' own, with their
/// records and their pair classes: nothing is made for a part of a list.
///
/// A host whose worker objects do not carry its own standing vetoes states
/// them beside each list too, by place ([`Self::prefill_stalled`],
/// [`Self::prefill_held`], [`Self::decode_stalled`],
/// [`Self::decode_held`]): a place of one list says nothing of the other.
/// A side whose profile shows its rows to a plugin reads them in the rows
/// written for the decision.
#[derive(Clone, Copy, Debug)]
pub struct PairLists<'a> {
    /// The prefills.
    pub prefill: LentList<'a>,
    /// The decodes.
    pub decode: LentList<'a>,
    /// The pair class of each worker by its place in its list, and the
    /// table of the classes that pair.
    pub classes: &'a PartnerClasses,
    /// The host's reads of its prefill limit and of its decode rooms.
    pub admission: &'a dyn PairAdmission,
    /// The places of the prefill list the request may take, ascending,
    /// each once; `None` for every place. The request's own fact and no
    /// veto of a worker: the prefill side's set opens with them under any
    /// document ([`Constraints::places`]).
    pub prefill_places: Option<&'a [u32]>,
    /// The places of the decode list the request may take, stated as the
    /// prefills' are. The two stand apart: a host may hold one side alone.
    /// The decode asked again behind a held prefill is held to them too,
    /// and its held prefill is found wherever it stands in its list.
    pub decode_places: Option<&'a [u32]>,
    /// The places of the prefill list a standing veto of the host masks,
    /// ascending, each once; empty for none ([`Constraints::stalled`]). The
    /// prefill side reads them as a single request's list reads its own.
    pub prefill_stalled: &'a [u32],
    /// The places of [`Self::prefill_stalled`] a hold of the host stands
    /// on, which no rule lifts ([`Constraints::held`]).
    pub prefill_held: &'a [u32],
    /// The places of the decode list a standing veto of the host masks,
    /// stated as the prefills' are. The decode asked again behind a held
    /// prefill reads them too, and its held prefill is found whatever
    /// masks it by then.
    pub decode_stalled: &'a [u32],
    /// The places of [`Self::decode_stalled`] a hold of the host stands on.
    pub decode_held: &'a [u32],
}

/// What a host states of one side's list beside the list: the places the
/// request may take, and the places its standing vetoes mask and hold.
/// Where the request states a hard demand, the places are those of them
/// whose worker meets it ([`Self::meeting`]), and a side the demand leaves
/// nobody names the demand that took its last worker.
#[derive(Clone, Copy, Debug)]
struct Beside<'a> {
    places: Option<&'a [u32]>,
    stalled: &'a [u32],
    held: &'a [u32],
    unmet: Option<UnmetDemand>,
}

impl<'a> Beside<'a> {
    /// Whether the side states nothing: the request's constraints are then
    /// the side's as they stand.
    const fn states_nothing(&self) -> bool {
        self.places.is_none() && self.stalled.is_empty() && self.held.is_empty()
    }

    /// The side held to the places of its list whose worker meets
    /// `demands`, among the places it states, by a single request's own
    /// rule ([`single::held_in`]): they are written into `kept`, a list
    /// the calling thread keeps for the side. Where every worker the side
    /// may take meets the demands it stands as it was stated, and where
    /// none does it names the demand that took the last one. Out of line:
    /// its code stays off the path of a pair that states no demand.
    #[inline(never)]
    fn meeting(self, records: &PoolRecords, demands: Demands, kept: &'a mut Vec<u32>) -> Self {
        let stated = Constraints {
            demands,
            places: self.places,
            ..Constraints::default()
        };
        match single::held_in(records, stated, kept) {
            Ok(held) => Self {
                places: held.places,
                ..self
            },
            Err(unmet) => Self {
                unmet: Some(unmet),
                ..self
            },
        }
    }

    /// The request's constraints with this side's places and mask.
    fn of(self, cx: &Constraints<'a>) -> Constraints<'a> {
        Constraints {
            places: self.places,
            stalled: self.stalled,
            held: self.held,
            ..*cx
        }
    }

    /// The side as its decision of `phase` reads it over `rows`, the
    /// side's records written for that decision. Where the side's profile
    /// shows its rows to a plugin, the mask is written into them and the
    /// side states none ([`vetoes::written`]): a plugin is shown a record's
    /// own fields, and the engine declines such a profile over a mask
    /// beside its rows. A side under no veto pays one test.
    fn written(self, engine: &Engine, model: &str, phase: Phase, rows: &mut [Candidate]) -> Self {
        if self.stalled.is_empty() {
            return self;
        }
        let beside = Constraints {
            stalled: self.stalled,
            held: self.held,
            ..Constraints::default()
        };
        let read = vetoes::written(engine, model, phase, rows, &beside);
        Self {
            stalled: read.stalled,
            held: read.held,
            ..self
        }
    }
}

impl<'a> PairLists<'a> {
    /// The request's constraints as each side reads them: the prefills',
    /// then the decodes'. A side reads the places and the mask its host
    /// states beside its list and no other: a request that states places
    /// or a mask of its own is declined ahead of this ([`not_decided`]).
    /// An encode request's decision hands them to the engine, which
    /// numbers both with the workers it admits for the items.
    pub(super) fn sides_of(&self, cx: &Constraints<'a>) -> [Constraints<'a>; 2] {
        let side = |places, stalled, held| {
            let side = Beside {
                places,
                stalled,
                held,
                unmet: None,
            };
            side.of(cx)
        };
        [
            side(self.prefill_places, self.prefill_stalled, self.prefill_held),
            side(self.decode_places, self.decode_stalled, self.decode_held),
        ]
    }

    /// [`Self::sides_of`] for a request that states a hard demand: each
    /// side held to the places of its list whose worker meets `demands` as
    /// that side's role is held to it, among the places its host states
    /// ([`Beside::meeting`]). `kept` takes the places, the prefills' first.
    /// `Err` names the demand that left a side nobody, the prefills'
    /// first, and that side. An encode request's decision judges its
    /// pair's two lists here, so a side is held one way under a pair's
    /// decision and under its.
    pub(super) fn sides_meeting<'s>(
        &self,
        cx: &Constraints<'s>,
        demands: Demands,
        (prefill, decode): (&PoolRecords, &PoolRecords),
        [prefill_kept, decode_kept]: [&'s mut Vec<u32>; 2],
    ) -> Result<[Constraints<'s>; 2], (UnmetDemand, PairSide)>
    where
        'a: 's,
    {
        let side = |places, stalled, held| Beside {
            places,
            stalled,
            held,
            unmet: None,
        };
        let prefills = side(self.prefill_places, self.prefill_stalled, self.prefill_held);
        let prefills = prefills.meeting(prefill, demands.held_by(Role::Prefill), prefill_kept);
        let decodes = side(self.decode_places, self.decode_stalled, self.decode_held);
        let decodes = decodes.meeting(decode, demands.held_by(Role::Decode), decode_kept);
        match (prefills.unmet, decodes.unmet) {
            (Some(unmet), _) => Err((unmet, PairSide::Prefill)),
            (None, Some(unmet)) => Err((unmet, PairSide::Decode)),
            (None, None) => Ok([prefills.of(cx), decodes.of(cx)]),
        }
    }

    /// One decision for a pair at `now`, over records written for it from
    /// both lists' worker objects and from `loads`, the host's snapshot of
    /// its workers' reports, with each record's pair class and its side's
    /// admission bit as the host reads it at this call. With no `held`
    /// target it is the pair's first decision: the engine's pair entry.
    /// With one, the target the pair's guard holds, it is the decode
    /// decided again behind that target's prefill: the engine's decode
    /// entry over the prefill's partners, handed the decode the target
    /// names, which a profile that keeps its pick answers again.
    ///
    /// `engine` decides it: under its document, on its clock, and over its
    /// own value of each list, which it asks of the list's cell.
    ///
    /// `request` is the pair's, as its host made it for the prefill phase.
    /// This call's instant is written into its facts: `now` on the engine's
    /// clock, the clock the records' sample instants stand on.
    ///
    /// A side whose profile reads a named signal reads each worker's value
    /// in `loads`, by the worker's place in that side's list: its stages
    /// find their list's signal cells in their facts
    /// ([`RequestFacts::signal_cells`]). The decode decided again reads
    /// the values of this call's snapshot. Cells on `request` are read by
    /// neither side, and under no snapshot every worker reads no value.
    ///
    /// A request that states a hard demand ([`Constraints::demands`]) is
    /// decided with each side held to the places of its list whose worker
    /// meets the demand as that side's role is held to it
    /// ([`Demands::held_by`]), among the places stated for the side here.
    /// The decode decided again judges its decodes alone. A side the
    /// demand leaves nobody sheds the pair with no endpoint under the
    /// stage of the demand that took that side's last worker, the
    /// prefills' first. A request that states places of its own beside a
    /// demand is declined under the places' name.
    ///
    /// Prefills with an open partner that are all at their admission limit
    /// shed a first decision, which reads as a request that waits
    /// ([`prefills_full`]).
    ///
    /// The call writes the widest budget into the facts of a request that
    /// states none: a decision in process is unbounded
    /// ([`Request::facts`]), and a stage reads the number as the time left.
    pub fn decide(
        &self,
        engine: &InProcess,
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
        held: Option<&Target>,
    ) -> Decision {
        let document = engine.in_force();
        let engine = engine.under(&document);
        let sides = (
            engine.members_of(self.prefill),
            engine.members_of(self.decode),
        );
        self.decide_over(engine, sides, loads, now, request, held)
    }

    /// [`Self::decide`] over the engine's values of the two lists, which
    /// its caller asked of their cells.
    pub(super) fn decide_over(
        &self,
        engine: Loaded<'_, '_>,
        sides: (&PoolMembers, &PoolMembers),
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
        held: Option<&Target>,
    ) -> Decision {
        let counted = SessionCountedUnder::default;
        self.decide_sides(
            engine,
            sides,
            loads,
            now,
            request,
            held,
            |_| true,
            counted,
            None,
        )
    }

    /// [`Self::decide_over`], and where the request's caller sends it:
    /// `sends_to` says whether it sends to a target of the pick
    /// ([`RequestMaker::sends_to`]), and each side's record of sent
    /// prefixes is written for the first target it takes. `counted` says
    /// where a worker's object counts the requests of the request's
    /// session key ([`RequestMaker::session_counted_under`]): it is asked
    /// under a document that reads the count and under no other, and the
    /// entry above, which holds no maker, answers the key's own text.
    ///
    /// `said` is the room an entry lends for its sides' shadows. A
    /// decision lent one is made out of line and leaves there what its
    /// shadows said beside it, ahead of the records of sent prefixes,
    /// which keep the live pick: a first decision each side's
    /// ([`decide_shadowed`]), and the decode decided again its decode
    /// side's ([`decide_decode_shadowed`]). A decision lent none asks no
    /// shadow and pays one test of its room.
    ///
    /// A first decision's pick states what its prefill holds of the request
    /// ([`overlap::state`]), where the request's own profile reads who
    /// holds the prompt at the prefill phase. The decode decided again
    /// states nothing.
    ///
    /// [`RequestMaker::sends_to`]: crate::engine::RequestMaker::sends_to
    /// [`RequestMaker::session_counted_under`]: crate::engine::RequestMaker::session_counted_under
    #[expect(
        clippy::too_many_arguments,
        reason = "a pair's decision over its two lists, where its caller sends it, and its shadows' room"
    )]
    fn decide_sides(
        &self,
        engine: Loaded<'_, '_>,
        (prefill, decode): (&PoolMembers, &PoolMembers),
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
        held: Option<&Target>,
        sends_to: impl Fn(&Target) -> bool,
        counted: impl FnOnce() -> SessionCountedUnder,
        said: Option<&mut SidesSaid>,
    ) -> Decision {
        let (spec, clock) = (&engine.document.spec, &engine.clock);
        // The engine as the fill of a side's column takes it: the name
        // stands for the document's engine inside the decision below.
        let loaded = engine;
        // The name each side's records read the count of the key's
        // requests under, on a worker's object: the one its side's session
        // is counted under, as the request's maker states it. One text
        // names both sides where the key's own text is the name, and a
        // session's number is each side's own. Named under a document that
        // reads the count and under no other.
        let mut names;
        let [prefill_key, decode_key] = if spec.engine().needs().session_in_flight {
            names = [[0; Session::TEXT_LEN]; 2];
            let (counted, [prefill, decode]) = (counted(), &mut names);
            let side = |phase| RequestFacts {
                phase,
                ..request.facts
            };
            [
                counted.name(&side(Phase::Prefill), prefill),
                counted.name(&side(Phase::Decode), decode),
            ]
        } else {
            [None; 2]
        };
        // One value for both lists' records: this call's instant on the
        // clock and the host's snapshot, as the host holds it.
        let live = Live::at(spec.engine(), clock, now, loads, prefill_key);
        request.facts.now_micros = live.now_micros();
        request.facts.budget_micros = super::budget::shown(request.facts.budget_micros);
        let (facts, cx) = (&request.facts, &request.constraints);
        // A retry leaves every worker of the engines it tried, each list by
        // its own engines: where a list holds several workers of one
        // engine, the facts name them all, as a single request's do. A
        // first attempt runs nine instructions for it: the test of its
        // empty list, and behind its decision the test of a list to hand
        // back.
        let tried = if facts.tried.is_empty() {
            None
        } else {
            tried_engines(engine, facts.tried, &[prefill, decode])
        };
        let widened;
        let facts = match &tried {
            Some(tried) => {
                widened = RequestFacts { tried, ..*facts };
                &widened
            }
            None => facts,
        };
        let sides = [
            Beside {
                places: self.prefill_places,
                stalled: self.prefill_stalled,
                held: self.prefill_held,
                unmet: None,
            },
            Beside {
                places: self.decode_places,
                stalled: self.decode_stalled,
                held: self.decode_held,
                unmet: None,
            },
        ];
        let decide = |buffers: &mut Buffers| {
            let [prefills, decodes] = &mut buffers.rows;
            let (workers, prefill_records) = (prefill.workers(), prefill.records(spec, clock));
            let (decode_workers, decode_records) = (decode.workers(), decode.records(spec, clock));
            // The rows are filled with no read of a worker object where
            // both lists are wholly available and the host states no veto
            // beside either. The decode asked again writes one prefill: it
            // takes the prefills' kept answer and asks no prefill.
            let live = live.rested_where(spec.engine(), |word| {
                self.prefill_stalled.is_empty()
                    && self.decode_stalled.is_empty()
                    && match held {
                        Some(_) => prefill_records.stood(word),
                        None => prefill_records.stands(workers, word),
                    }
                    && decode_records.stands(decode_workers, word)
            });
            // The decodes' records read the key's count under their own
            // name, and are written at rest where the prefills' are.
            let decode_live = live.counted_under(decode_key);
            // The decode asked again reads one prefill, the held one.
            let (prefills, prefill_facts) = match held {
                Some(held) => held_prefill(prefill_records, workers, &live, held, prefills),
                None => prefill_records.write(workers, &live, prefills),
            };
            let (decodes, decode_facts) =
                decode_records.write(decode_workers, &decode_live, decodes);
            // A host that counts the request on its first decode
            // ([`Constraints::counted_on_decode`]) is answered over that
            // decode's record without the request. The decode is read at
            // its first place, and looked for where the list moved.
            if let Some(first) = held
                .filter(|_| cx.counted_on_decode)
                .and_then(|held| held.pair)
            {
                let is_first = |row: &Candidate| row.id == first.decode_id;
                let place = Some(first.decode_position as usize)
                    .filter(|&place| decodes.get(place).is_some_and(is_first))
                    .or_else(|| decodes.iter().position(is_first));
                if let Some(row) = place.and_then(|place| decodes.get_mut(place)) {
                    row.in_flight = row.in_flight.saturating_sub(1);
                    // The host's count of the request there is under its
                    // key too: a capped rule reads the key's other
                    // requests, as a first decision does.
                    row.session_in_flight = row.session_in_flight.saturating_sub(1);
                }
            }
            stamp(prefills, decodes, self.classes, self.admission);
            let (engine, scratch) = (spec.engine(), &mut buffers.scratch);
            let [prefill_side, decode_side] = sides;
            // A hard demand the request states is judged here, a side at a
            // time, by what each side's role is held to. The records'
            // decision is handed the places and no demand. The decode
            // asked again judges no prefill: the held one ran already.
            let spent;
            let demanded = !cx.demands.is_none();
            let (cx, prefill_side, decode_side) = if demanded {
                let [prefill_kept, decode_kept] = &mut buffers.meeting;
                let demands = cx.demands;
                let prefill_side = match held {
                    None => {
                        let demands = demands.held_by(Role::Prefill);
                        prefill_side.meeting(prefill_records, demands, prefill_kept)
                    }
                    Some(_) => prefill_side,
                };
                let demands = demands.held_by(Role::Decode);
                let decode_side = decode_side.meeting(decode_records, demands, decode_kept);
                spent = Constraints {
                    demands: Demands::default(),
                    ..*cx
                };
                (&spent, prefill_side, decode_side)
            } else {
                (cx, prefill_side, decode_side)
            };
            // A side whose profile shows its rows is handed its mask in
            // them. The decode asked again reads no mask of the prefills.
            let decode_side = decode_side.written(engine, facts.model, Phase::Decode, decodes);
            let prefill_side = match held {
                None => prefill_side.written(engine, facts.model, Phase::Prefill, prefills),
                Some(_) => prefill_side,
            };
            // Under a document a stage of which reads a named signal, each
            // side that reads one is lent its list's cells for this call.
            let lent = spec.needs().signal_cells.then(|| {
                let first = held.is_none();
                signals::sides([prefill, decode], spec, clock, &live, facts.model, first)
            });
            let [prefill_signal_cells, decode_signal_cells] = match &lent {
                Some(lent) => lent
                    .each_ref()
                    .map(|side| side.as_ref().and_then(HeldCells::cells)),
                None => [None; 2],
            };
            // A side whose phase reads who holds the prompt is handed its
            // own list's column. The decode asked again reads the decodes'
            // alone: its prefill ran already.
            let first = held.is_none().then_some((prefill, prefill_records));
            let columns = &mut buffers.columns;
            let read = side_columns(loaded, facts, first, (decode, decode_records), columns);
            let [prefill_column, decode_column] = &buffers.columns;
            let pools = PairPools {
                prefills,
                decodes,
                prefill_affinity: prefill_column.as_slice(),
                decode_affinity: decode_column.as_slice(),
                prefill_signal_cells,
                decode_signal_cells,
                table: self.classes.table(),
                first: self.classes.first(),
            };
            let mut decision = match (held, said) {
                (None, None) => {
                    let sides = (cx, [prefill_side, decode_side]);
                    decide_records(engine, facts, sides, &prefill_facts, pools, scratch)
                }
                // An entry that lends room for them is left what each
                // side's shadow said beside the pair.
                (None, Some(said)) => {
                    let (sides, pool) = ((cx, [prefill_side, decode_side]), &prefill_facts);
                    decide_shadowed(loaded, facts, sides, pool, pools, scratch, said)
                }
                (Some(held), None) => {
                    let side = (cx, decode_side);
                    decide_decode_records(engine, facts, side, &decode_facts, pools, held, scratch)
                }
                // The decode's shadow alone: the prefill ran already.
                (Some(held), Some(said)) => {
                    let (side, pool) = ((cx, decode_side), &decode_facts);
                    decide_decode_shadowed(loaded, facts, side, pool, pools, held, scratch, said)
                }
            };
            // A pair states what its prefill holds of the request: its first
            // target is that prefill, and the prefills' column is the one
            // this decision was given. Stated at a first decision alone,
            // where the request's own profile reads who holds the prompt at
            // the prefill phase: the decode asked again has no prefill to
            // state of, and a shadow's rows say nothing of the live pick.
            if held.is_none()
                && engine.needs_for_phase(Phase::Prefill).affinity
                && engine.profile_reads_affinity(facts.model, Phase::Prefill)
            {
                let column = Feed::Column(prefill_column);
                overlap::state(&mut decision, column, facts, spec.block_size());
            }
            // The decode the pair holds, named again: the decision that
            // put the request there stands, and its word of the side with
            // it, ahead of the record's write. The pick is turned where it
            // stands: built beside it, it was copied once more.
            if let (Decision::Pick(pick), Some(held)) =
                (&mut decision, held.and_then(|held| held.pair))
            {
                pick.targets.decode_placed_as(&held);
            }
            record_sides(
                &decision,
                facts,
                (prefill, decode),
                read,
                &buffers.columns,
                &sends_to,
            );
            decision
        };
        let decision = BUFFERS.with(|buffers| match buffers.try_borrow_mut() {
            Ok(mut buffers) => decide(&mut buffers),
            // A stage asked for a decision from inside its own, on this
            // thread: that decision takes buffers of its own, and its word
            // of a shed is kept for its call ([`said_apart`]).
            Err(_) => {
                let mut own = Buffers::default();
                let decision = decide(&mut own);
                said_apart(&own.scratch);
                decision
            }
        });
        // The thread's list goes back for its next retry.
        if let Some(tried) = tried {
            single::TRIED.set(tried);
        }
        decision
    }
}

/// A pair's two columns of who holds the prompt, the prefills' first, and
/// what each fill read of the record of sent prefixes. A side whose phase
/// reads affinity is handed its own list's column, read at the side's
/// phase and by the side's own profile's word of the record
/// ([`overlap::column`]), which the list's table read. A side whose phase
/// reads none pays one test and keeps an empty column. `prefill` is `None`
/// for the decode asked again, which fills no column of the prefills.
///
/// A pair's decision and an encode request's fill their columns here, so
/// the two read one way.
// In line at its two callers: as a call it costs a pair's decision whose
// sides read no affinity 25 instructions, counted with Callgrind.
#[inline(always)]
pub(super) fn side_columns(
    engine: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    prefill: Option<(&PoolMembers, &PoolRecords)>,
    decode: (&PoolMembers, &PoolRecords),
    columns: &mut [Vec<AffinityFacts>; 2],
) -> [Option<overlap::PrefixRead>; 2] {
    // The host's indexes of cache events and the records of sent
    // prefixes, which a side's column is read from.
    let (kept, spec) = (&engine.affinity, &engine.document.spec);
    let fill = |(side, records): (&PoolMembers, &PoolRecords), phase, column: &mut Vec<_>| {
        column.clear();
        if !spec.engine().needs_for_phase(phase).affinity {
            return None;
        }
        let (workers, ids) = (side.workers(), records.ids());
        overlap::column(
            kept,
            side.affinity(),
            spec,
            workers,
            ids,
            phase,
            facts,
            column,
        )
    };
    let [prefill_column, decode_column] = columns;
    let prefill_read = match prefill {
        Some(side) => fill(side, Phase::Prefill, prefill_column),
        None => {
            prefill_column.clear();
            None
        }
    };
    [prefill_read, fill(decode, Phase::Decode, decode_column)]
}

/// Behind a pick, each side's record of sent prefixes keeps where the
/// prompt went ([`overlap::record_by_column`]), where that side's profile
/// is told of the side ([`lease_events::stages_hear`]): the first target
/// the request's caller sends to (`sends_to`, [`overlap::sent_to`]), and
/// nobody where it sends to none. Each side asks its own word: a sessions
/// rule that keeps what it places from the profile may have placed one
/// side alone and left the other to its stages, and the side it placed
/// writes nothing. The prefill's worker is written at the prefill phase
/// and the decode's at the decode phase, each where its side's fill read
/// the record (`read`, as [`side_columns`] returned it, and `columns` as
/// it filled them). A pair neither side of which read its record asks
/// nothing of the pick.
#[inline]
pub(super) fn record_sides(
    decision: &Decision,
    facts: &RequestFacts<'_>,
    (prefill, decode): (&PoolMembers, &PoolMembers),
    [prefill_read, decode_read]: [Option<overlap::PrefixRead>; 2],
    [prefill_column, decode_column]: &[Vec<AffinityFacts>; 2],
    sends_to: impl Fn(&Target) -> bool,
) {
    let told = |side| lease_events::stages_hear(decision, side);
    let prefill_read = prefill_read.filter(|_| told(Side::Prefill));
    let decode_read = decode_read.filter(|_| told(Side::Decode));
    if prefill_read.is_none() && decode_read.is_none() {
        return;
    }
    let Some(sent) = overlap::sent_to(decision, sends_to) else {
        return;
    };
    if let Some(read) = prefill_read {
        let (pool, column) = (prefill.affinity(), prefill_column.as_slice());
        overlap::record_by_column(pool, Phase::Prefill, facts, read, column, sent.id);
    }
    if let (Some(read), Some(pair)) = (decode_read, sent.pair) {
        let (pool, column) = (decode.affinity(), decode_column.as_slice());
        overlap::record_by_column(pool, Phase::Decode, facts, read, column, pair.decode_id);
    }
}

/// The workers of the engines a retry tried: `tried` as its host named
/// them, and behind them every other worker of an engine of one of
/// `lists` that holds a tried worker, each list read by its own engines
/// (`PoolRecords::tried_engines`). A pair hands its two lists and an
/// encode request its three. `None` where no list holds several workers
/// of one engine: the request's own list stands.
///
/// The workers stand in the calling thread's own list ([`single::TRIED`]),
/// which the caller puts back behind its decision: a retry allocates
/// nothing once the list has grown. Out of line and cold: a first attempt
/// runs none of it.
#[cold]
#[inline(never)]
pub(super) fn tried_engines(
    engine: Loaded<'_, '_>,
    tried: &[WorkerId],
    lists: &[&PoolMembers],
) -> Option<Vec<WorkerId>> {
    let (spec, clock) = (&engine.document.spec, &engine.clock);
    let mut ranked = lists
        .iter()
        .map(|list| list.records(spec, clock))
        .filter(|records| records.has_ranks());
    let first = ranked.next()?;
    let mut whole = single::TRIED.take();
    first.tried_engines(tried, &mut whole);
    for records in ranked {
        records.more_tried_engines(tried, &mut whole);
    }
    Some(whole)
}

/// The record of the prefill `held` names, written alone into `rows` for
/// the decode asked again, which reads no other prefill and no fact of the
/// prefills' list ([`PoolRecords::read`]). No other prefill's worker is
/// read: a circuit breaker that such a read moves from open to half-open
/// at its timeout waits for the next first decision. Where the list names
/// no place of the held prefill, because it left or because an id stands
/// twice in the list ([`PoolRecords::place_of`]), it is looked for in the
/// list's whole write. Out of line: its code stays off the path of a
/// pair's first decision.
#[inline(never)]
fn held_prefill<'a, 'f>(
    records: &'f PoolRecords,
    workers: &[Arc<dyn Worker>],
    live: &Live<'f>,
    held: &Target,
    rows: &'a mut Vec<Candidate>,
) -> (&'a mut [Candidate], PoolFacts<'f>) {
    let place = records.place_of(held.id);
    match place.and_then(|place| records.read(workers, live, place)) {
        Some(row) => {
            rows.clear();
            rows.push(row);
            (rows.as_mut_slice(), PoolFacts::default())
        }
        None => records.write(workers, live, rows),
    }
}

/// What one thread keeps for its pair decisions: each list's records, the
/// prefills' first, and the engine's scratch. An encode request's decision
/// keeps its encoders' records here too ([`super::encode`]): a pair's
/// leaves them as they were. A request that states a hard demand keeps
/// each side's places that meet it here, the prefills' first, a pair's and
/// an encode request's alike: such a request allocates nothing once each
/// list has grown to its side.
/// A side whose phase reads who holds the prompt keeps its list's column
/// here, the prefills' first.
#[derive(Debug, Default)]
pub(super) struct Buffers {
    pub(super) rows: [Vec<Candidate>; 2],
    pub(super) encoders: Vec<Candidate>,
    pub(super) scratch: Scratch,
    pub(super) meeting: [Vec<u32>; 2],
    pub(super) columns: [Vec<AffinityFacts>; 2],
}

thread_local! {
    /// The calling thread's own buffers, so a pair's decision allocates
    /// nothing after a thread's first decisions over lists of a size.
    pub(super) static BUFFERS: RefCell<Buffers> = RefCell::new(Buffers::default());

    /// The word of the list it shed over that the thread's last decision
    /// on buffers of its own left ([`said_apart`]).
    static SAID_APART: Cell<Option<ShedOver>> = const { Cell::new(None) };
}

/// Keeps the word of a decision made on buffers of its own, with `scratch`:
/// the list it shed its request over as saturated, where it says one
/// ([`Scratch::shed_over`]). A stage asked for that decision from inside
/// another of this thread, which holds the thread's buffers, and the
/// scratch goes with the decision: its call reads the word here
/// ([`shed_over`]). Out of line: a decision on the thread's buffers runs
/// none of it.
#[cold]
#[inline(never)]
pub(super) fn said_apart(scratch: &Scratch) {
    SAID_APART.set(scratch.shed_over());
}

/// The list the calling thread's last pair's decision or encode decision
/// shed its request over as saturated, by that decision's own word
/// ([`Scratch::shed_over`]). The call that asked for the decision reads it
/// behind that decision's shed as saturated and behind nothing else, ahead
/// of anything else it asks on this thread ([`place`](super::place)): a
/// request answered ahead of its decision leaves the word of the decision
/// before.
pub(super) fn shed_over() -> Option<ShedOver> {
    BUFFERS.with(|buffers| match buffers.try_borrow() {
        Ok(buffers) => buffers.scratch.shed_over(),
        // The call stands inside a decision of this thread, which holds
        // the buffers: its own decision was made on buffers of its own.
        Err(_) => SAID_APART.get(),
    })
}

/// Writes on the two lists' records what a pair's decision reads beside a
/// single's: each record's pair class, a prefill's capacity bit and a
/// decode's room bit, as `admission` reads them now, each by the record's
/// place in its list ([`Candidate::position`]): the decode asked again
/// hands its one prefill's record. A record past the classes takes class
/// `0`, which pairs with nobody.
pub(super) fn stamp(
    prefills: &mut [Candidate],
    decodes: &mut [Candidate],
    classes: &PartnerClasses,
    admission: &dyn PairAdmission,
) {
    let class = |classes: &[u32], at: usize| classes.get(at).copied().unwrap_or(0);
    for row in prefills.iter_mut() {
        let at = row.position as usize;
        row.pair_class = class(classes.prefills(), at);
        row.has_capacity = admission.has_capacity(at);
    }
    for row in decodes.iter_mut() {
        let at = row.position as usize;
        row.pair_class = class(classes.decodes(), at);
        row.has_room = admission.has_room(at);
    }
}

/// How a pair's request is shed where it states what a pair's records are
/// not narrowed by, and the stage it is shed under: a subset hint, a hard
/// demand, or places of its list. A pair's records keep no demand fact. A
/// pair's own decision judges a demand ahead of them, a side at a time
/// ([`PairLists::decide`]), and hands them none, and so does an encode
/// request's. A demand that reaches them was judged by no decision: it is
/// shed, and the request is never placed past its demand. A hint of some
/// records is declined: no stage here takes a part of a list. So are
/// places the request states itself ([`single::PLACES`]): one set cannot
/// hold two lists, and each side's places are its lists' to state
/// ([`PairLists::prefill_places`], [`PairLists::decode_places`]).
fn not_narrowed_by(cx: &Constraints<'_>) -> Option<(ShedReason, &'static str)> {
    match cx.subset_hint {
        Some(0) => Some((ShedReason::NoEndpoints, SUBSET_HINT)),
        Some(_) => Some((ShedReason::PolicyDeclined, SUBSET_HINT)),
        None if !cx.demands.is_none() => Some((ShedReason::NoEndpoints, WIRES)),
        None if cx.places.is_some() => Some((ShedReason::PolicyDeclined, single::PLACES)),
        None => None,
    }
}

/// How a pair's own decision answers a request it is not narrowed by
/// ([`not_narrowed_by`]) or one that states a mask of its own
/// ([`Constraints::stalled`], [`Constraints::held`]). Such a mask is
/// declined under the places' name ([`single::PLACES`]): it names places
/// of one list, and each side's are stated beside its list
/// ([`PairLists::prefill_stalled`], [`PairLists::decode_stalled`]). What
/// a pair is not narrowed by is answered first. An encode request is the
/// pair's too: its three lists' decision asks this ahead of its own pass,
/// so the same request is answered alike at its first decision and at its
/// decode asked again.
pub(super) fn not_decided(cx: &Constraints<'_>) -> Option<(ShedReason, &'static str)> {
    let masked = !cx.stalled.is_empty() || !cx.held.is_empty();
    let declined = (ShedReason::PolicyDeclined, single::PLACES);
    not_narrowed_by(cx).or_else(|| masked.then_some(declined))
}

/// The request's session under the decode phase: each side of a pair keeps
/// its own.
fn decode_session(facts: &RequestFacts<'_>) -> Session {
    RequestFacts {
        phase: Phase::Decode,
        ..*facts
    }
    .session()
}

/// A pair's decision over the records of its two lists, stamped
/// ([`stamp`]): the engine's pair entry. `pool` holds the prefill list's
/// facts. The request's constraints are read by both sides, each with
/// the places its list states and the places its host's vetoes mask and
/// hold in it (`sides`: the prefills', the decodes'). A side whose mask
/// was written into its records states none here ([`Beside::written`]).
///
/// One side alone is no pair: where either list holds no record the engine
/// answers over no candidate. A hard demand that reaches the records, or a
/// subset hint of none, is shed, and a hint of some records is declined,
/// each under a stage's name ([`not_decided`]). A side the request's
/// demand left nobody ([`Beside::meeting`]) sheds the pair with no
/// endpoint under that demand's stage, the prefills' first, behind what
/// the records decline by name. The refusal over no candidate names the
/// side of the list that holds no record, the prefills' first, and a side
/// its demand left nobody is the side its refusal names
/// ([`Shed::side`](crate::engine::decision::Shed::side)).
fn decide_records(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    (cx, sides): (&Constraints<'_>, [Beside<'_>; 2]),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    scratch: &mut Scratch,
) -> Decision {
    if pools.prefills.is_empty() || pools.decodes.is_empty() {
        let prefills = !pools.prefills.is_empty();
        return engine.decide_over_an_empty_list(facts, pool, prefills, cx, scratch);
    }
    if let Some((reason, stage)) = not_decided(cx) {
        return shed(reason, stage, facts.seed, pool.generation);
    }
    let [prefills, decodes] = sides;
    // A side the request's demand left nobody: the prefills' first.
    if let Some(unmet) = prefills.unmet.or(decodes.unmet) {
        let stage = unmet.stage();
        let side = if prefills.unmet.is_some() {
            PairSide::Prefill
        } else {
            PairSide::Decode
        };
        return shed(ShedReason::NoEndpoints, stage, facts.seed, pool.generation).of_side(side);
    }
    if prefills.states_nothing() && decodes.states_nothing() {
        return engine.decide_pair(facts, pool, pools, cx, cx, scratch);
    }
    // Each side's set opens with its own places, under its own mask.
    let (prefill_cx, decode_cx) = (prefills.of(cx), decodes.of(cx));
    engine.decide_pair(facts, pool, pools, &prefill_cx, &decode_cx, scratch)
}

/// What each side's shadow said beside a pair's first decision, the
/// prefill's then the decode's ([`side_shadows`]): `None` for a side that
/// was due none, and for one whose shadow panicked. Beside a decode
/// decided again the decode's place holds its shadow's word
/// ([`later_shadow`]) and the prefill's holds none. The room an entry
/// lends its decision for those words: an entry that counts no side's
/// shadow lends none, and its decision asks none.
type SidesSaid = [Option<Shadowed>; 2];

/// A pair's first decision by `own` over the records of its two lists
/// ([`decide_records`]), with what each side's shadow said beside it left
/// in `said` ([`side_shadows`]): the shadows run on the scratch as that
/// decision left it, ahead of anything else that writes it.
///
/// Out of line: the path of a decision lent no room holds no part of it.
/// The shadows read the live decision by reference, and a decision lent
/// so inside that path is copied once more on its way out, for every
/// pair.
#[inline(never)]
fn decide_shadowed(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    sides: (&Constraints<'_>, [Beside<'_>; 2]),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    scratch: &mut Scratch,
    said: &mut SidesSaid,
) -> Decision {
    let engine = own.document.spec.engine();
    let decision = decide_records(engine, facts, sides, pool, pools, scratch);
    *said = side_shadows(own, facts, sides, pool, pools, &decision, scratch);
    decision
}

/// What each side's shadow said beside `live`, a pair's first decision
/// over the records of its two lists ([`decide_records`]), which read the
/// request's constraints with each side's places and mask (`sides`).
///
/// A side's shadow is due where the pair's binding names one for it that
/// is free to run ([`Engine::shadows_side`]) and a profile made the live
/// decision. A request refused ahead of the pair's profiles has no shadow,
/// as a single's has none: no profile decided it
/// ([`Trace::made_under`](crate::engine::decision::Trace::made_under)).
///
/// A side that is due is decided once more by its shadow profile
/// ([`Engine::shadow_side`]): over the records the live decision read and
/// the scratch it left, under that side's own places and mask. The run
/// writes the scratch and the shadow profile's stages. It writes no record
/// of sent prefixes, issues nothing and tells no stage.
///
/// A side that is due and not asked is said as skipped
/// ([`ShadowKind::Skipped`]): beside a pair that was shed, under less than
/// [`SHADOW_FLOOR`] of the request's stated budget, and where the engine's
/// run says none. That run says none where the side's shadow profile would
/// be handed another column than the side's pool carries
/// ([`Engine::side_apart`]), beside a pair the pair's default made or a
/// weighing prefill profile's lift alone placed, and beside a side its
/// binding's sessions rule or its picker's assignment placed. Beside a
/// pair whose decode side the binding's all-vetoed rule steered, the
/// decode's shadow is put to that rule and compared.
/// So the kinds a side counts add up to the decisions its pair's profiles
/// made.
///
/// Each side's run stands under the engine's catch
/// ([`InProcess::caught`]): a shadow that panics costs its request
/// nothing, is counted with the engine's faults and says nothing, and the
/// other side is asked all the same.
fn side_shadows(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    (cx, sides): (&Constraints<'_>, [Beside<'_>; 2]),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    live: &Decision,
    scratch: &mut Scratch,
) -> SidesSaid {
    let engine = own.document.spec.engine();
    let made = live.trace().made_under();
    let word = made.and_then(|_| engine.shadow_for(facts.model, Phase::Prefill));
    let Some((at, ShadowRows::Sides)) = word else {
        return [None; 2];
    };
    let [prefills, decodes] = sides;
    let (prefill_cx, decode_cx) = (prefills.of(cx), decodes.of(cx));
    let left = budget_left(facts);
    PairSide::BOTH.map(|side| {
        if !engine.shadows_side(at, side) {
            return None;
        }
        let said = |kind, first| {
            Some(Shadowed {
                binding: at,
                kind,
                first,
            })
        };
        // A live pick and a budget over the floor: else the side is due
        // and not asked.
        let Some(pick) = live.pick().filter(|_| left >= SHADOW_FLOOR) else {
            return said(ShadowKind::Skipped, None);
        };
        let ran = own.caught(|| {
            let (cx, decode_cx) = (&prefill_cx, &decode_cx);
            Ok(engine.shadow_side(at, side, facts, pool, pools, cx, decode_cx, pick, scratch))
        });
        match ran {
            Ok(Some(shadow)) => {
                let (kind, first) = beside(side, pick, &shadow);
                said(kind, first)
            }
            // The engine's run says none for a profile of another column,
            // beside a pair the default made or the weighing lift alone
            // placed, and beside a side its rule or its picker's
            // assignment placed.
            Ok(None) => said(ShadowKind::Skipped, None),
            // The shadow panicked: counted where it was caught.
            Err(_) => None,
        }
    })
}

/// How `shadow`, the decision of a side's shadow, stands beside the live
/// `pick` of its pair, and the worker the shadow took. A pair's first
/// target names its prefill, with the decode beside it, at a first
/// decision and at a decode decided again: a side is compared by its own
/// row of that target.
fn beside(side: PairSide, pick: &Pick, shadow: &Decision) -> (ShadowKind, Option<WorkerId>) {
    let first = shadow.pick().and_then(|pick| pick.targets.first());
    let taken = pick.targets.first().and_then(|target| match side {
        PairSide::Prefill => Some(target.position),
        PairSide::Decode => target.pair.map(|pair| pair.decode_position),
    });
    let kind = match first {
        Some(first) if Some(first.position) == taken => ShadowKind::Agreed,
        Some(_) => ShadowKind::OtherTarget,
        None => ShadowKind::OtherOutcome,
    };
    (kind, first.map(|first| first.id))
}

/// The decode decided again by `own` over the records of the two lists
/// ([`decide_decode_records`]), with what the decode side's shadow said
/// beside it left in the decode's place of `said` ([`later_shadow`]).
///
/// Out of line, as a first decision that is lent room is
/// ([`decide_shadowed`]): the path of a call lent no room holds no part
/// of it.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the decode call's inputs, beside the room its shadow's word is left in"
)]
fn decide_decode_shadowed(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    side: (&Constraints<'_>, Beside<'_>),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    held: &Target,
    scratch: &mut Scratch,
    said: &mut SidesSaid,
) -> Decision {
    let engine = own.document.spec.engine();
    let decision = decide_decode_records(engine, facts, side, pool, pools, held, scratch);
    let again = (held, &decision);
    said[PairSide::Decode.index()] = later_shadow(own, facts, side, pool, pools, again, scratch);
    decision
}

/// What the decode side's shadow said beside the decode decided again,
/// `again`: the target the pair holds and the live decision behind its
/// prefill ([`decide_decode_records`]), which read the request's
/// constraints with the decode list's places and mask (`side`).
///
/// The shadow is due where the pair's binding names one for its decode
/// side that is free to run ([`Engine::shadows_side`]) and the call was
/// decided under the decode profile's name
/// ([`Trace::made_under`](crate::engine::decision::Trace::made_under)): a
/// call refused ahead of it has no shadow.
///
/// A shadow that is due is asked once ([`Engine::shadow_decode_later`]):
/// over the partners of the held prefill the live call read, under the
/// decode list's places and mask, on the call's own draw sequence. It is
/// handed the decode the pair's first decision named, as the live call is
/// ([`first_decode`]): a shadow whose picker takes turns keeps that decode
/// and takes no turn, as its live side takes none. It is
/// said as skipped ([`ShadowKind::Skipped`]) beside a call that picked no
/// decode, under less than [`SHADOW_FLOOR`] of the request's stated
/// budget, and where the engine's run says none, as a side's is beside a
/// pair's first decision ([`side_shadows`]). The run stands under the
/// engine's catch ([`InProcess::caught`]): a shadow that panics costs
/// its call nothing and is counted with the engine's faults.
fn later_shadow(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    (cx, side): (&Constraints<'_>, Beside<'_>),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    (held, live): (&Target, &Decision),
    scratch: &mut Scratch,
) -> Option<Shadowed> {
    let engine = own.document.spec.engine();
    let made = live.trace().made_under();
    let word = made.and_then(|_| engine.shadow_for(facts.model, Phase::Prefill));
    let Some((at, ShadowRows::Sides)) = word else {
        return None;
    };
    if !engine.shadows_side(at, PairSide::Decode) {
        return None;
    }
    let said = |kind, first| {
        Some(Shadowed {
            binding: at,
            kind,
            first,
        })
    };
    let Some(pick) = live.pick().filter(|_| budget_left(facts) >= SHADOW_FLOOR) else {
        return said(ShadowKind::Skipped, None);
    };
    // The constraints of the live call. The run states the decode phase
    // itself, where it takes the decodes' side of the pools.
    let decode_cx = side.of(cx);
    let first = first_decode(held);
    let ran = own.caught(|| {
        let cx = &decode_cx;
        Ok(engine.shadow_decode_later(at, facts, pool, pools, held.id, first, cx, scratch))
    });
    match ran {
        Ok(Some(shadow)) => {
            let (kind, first) = beside(PairSide::Decode, pick, &shadow);
            said(kind, first)
        }
        Ok(None) => said(ShadowKind::Skipped, None),
        // The shadow panicked: counted where it was caught.
        Err(_) => None,
    }
}

/// What a pair's first pick ([`PairLists::decide`]) issues, for a first
/// decision of `own`: the prefill side's lease under `id`, which is the
/// answer's lease, and the request's session under the decode phase, which
/// its guard keeps for the decode side's words ([`first_sides`]). The
/// stages hear the prefill's issue, and the decode's behind it unless the
/// request sends its decode `later` ([`first_told`], [`tell_first`]): such
/// a side stands chosen and is told at its own issue ([`issue_decode`],
/// [`lease_events::pair`]).
///
/// The two issues are told under one catch of their own
/// ([`InProcess::word_caught`]). A stage that panics at a side's issue
/// is held and counted, as at a later word of the lease, and the stages
/// behind it do not hear that issue, nor the decode side's where the
/// prefill side's panicked. The pick stands: the leases are made ahead of
/// the catch, so the decision answers its lease all the same, and its guard
/// holds the pair as issued and tells each side's dispatch and its end. A
/// stage that heard an issue then hears that side end.
///
/// A panic in the decision itself is not held here: it costs the request
/// its decision ([`InProcess::caught`]).
///
/// `None` for a decision that picks no pair: it issues nothing.
// In line in the one issue of a pair: as a call it costs a pair's first
// decision about 44 instructions, counted with Callgrind.
#[inline(always)]
fn issue_caught(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    id: LeaseId,
    later: bool,
) -> Option<(Lease, Session)> {
    let (lease, decode, session) = first_sides(facts, decision, id)?;
    let engine = own.document.spec.engine();
    // Read ahead of the catch, which holds the stages' calls alone: with
    // the two reads inside it the catch stood as a call of its own, 46
    // instructions a pair, counted with Callgrind.
    let told = first_told(decision, later);
    own.word_caught(|| tell_first(engine, &lease, &decode, decision, told));
    Some((lease, session))
}

/// The two sides of a pair's first pick, and no word to a stage: the
/// prefill side's lease under `id`, the decode side's as it is told with
/// the pair, and the request's session under the decode phase.
///
/// `None` for a decision that picks no pair.
#[inline]
fn first_sides(
    facts: &RequestFacts<'_>,
    decision: &Decision,
    id: LeaseId,
) -> Option<(Lease, Lease, Session)> {
    let pick = decision.pick()?;
    let first = pick.targets.first()?;
    let session = decode_session(facts);
    let lease = Lease {
        id,
        kind: LeaseKind::Credit,
        side: Side::Prefill,
        target_index: 0,
        booked: first.id,
        generation: pick.trace.generation,
        session: facts.session(),
        session_verdict: pick.session_verdict,
        issued_micros: facts.now_micros,
        tokens: facts.prompt_tokens,
        chars: facts.prompt_chars,
        bytes: facts.prompt_bytes,
    };
    let decode = lease_events::decode_as_issued(&lease, first, session)?;
    Some((lease, decode, session))
}

/// Whether the stages hear the issue of each side of a pair's first pick,
/// the prefill side's first: the decode side's is told with the pair
/// unless the request sends its decode `later`. A side its sessions rule
/// keeps from the profile is issued too, and no stage hears its issue, as
/// of a single ([`lease_events::stages_hear`]): each side by its own word.
#[inline]
fn first_told(decision: &Decision, later: bool) -> [bool; 2] {
    [
        lease_events::stages_hear(decision, Side::Prefill),
        !later && lease_events::stages_hear(decision, Side::Decode),
    ]
}

/// The issue of a pair's first pick told to the stages of `engine`: the
/// prefill side's `lease`, then the `decode` side's, each where `told`
/// says its stages hear it ([`first_told`]).
#[inline]
fn tell_first(
    engine: &Engine,
    lease: &Lease,
    decode: &Lease,
    decision: &Decision,
    [prefill_told, decode_told]: [bool; 2],
) {
    if prefill_told {
        engine.on_decision_issued(lease, decision);
    }
    if decode_told {
        engine.on_decision_issued(decode, decision);
    }
}

/// The decode asked again over the records of the two lists, stamped
/// ([`stamp`]): the engine's decode entry over the partners of the prefill
/// of `held`, the target the pair holds, handed the decode that target
/// names. `pool` holds the decode list's facts. The decodes are held to
/// the decode list's places and read under its mask (`side`). The prefill
/// list's mask is not read: the held prefill ran already.
///
/// A prefill list with no record holds no held prefill: the entry answers
/// over no list, is handed no decode and names the prefill that left,
/// whatever the request states. A decode list with no record is read as
/// it stands. The entry finds the held prefill in its list and refuses
/// the call for the partners it lacks.
///
/// The held prefill is looked for before what the request states is
/// judged: a call whose prefill is in no list of records reads
/// `prefill-gone` from the engine's entry too, whatever it states. Behind
/// a prefill that stands in its list, decodes the request's demand left
/// nobody of shed the call under that demand's stage.
fn decide_decode_records(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    (cx, side): (&Constraints<'_>, Beside<'_>),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    held: &Target,
    scratch: &mut Scratch,
) -> Decision {
    let prefill = held.id;
    let facts = RequestFacts {
        phase: Phase::Decode,
        ..*facts
    };
    if pools.prefills.is_empty() {
        let none = PoolFacts {
            generation: pool.generation,
            ..PoolFacts::default()
        };
        let no_lists = PairPools::default();
        return engine.decide_decode(&facts, &none, no_lists, prefill, None, cx, scratch);
    }
    let listed = || pools.prefills.iter().any(|row| row.id == prefill);
    if let Some((reason, stage)) = not_decided(cx).filter(|_| listed()) {
        return shed(reason, stage, facts.seed, pool.generation);
    }
    // The decodes the request's demand left nobody of.
    if let Some(unmet) = side.unmet.filter(|_| listed()) {
        let stage = unmet.stage();
        return shed(ShedReason::NoEndpoints, stage, facts.seed, pool.generation)
            .of_side(PairSide::Decode);
    }
    let first = first_decode(held);
    if side.states_nothing() {
        return engine.decide_decode(&facts, pool, pools, prefill, first, cx, scratch);
    }
    let decode_cx = side.of(cx);
    engine.decide_decode(&facts, pool, pools, prefill, first, &decode_cx, scratch)
}

/// The decode the pair's first decision named, of `held`, the target the
/// pair holds: its worker and its rank, as the decode decided again is
/// handed them ([`decide_decode_records`]) and its shadow beside it
/// ([`later_shadow`]).
fn first_decode(held: &Target) -> Option<(WorkerId, u16)> {
    held.pair.map(|pair| (pair.decode_id, pair.decode_dp_rank))
}

/// What a pick of the decode decided again ([`PairLists::decide`] behind a
/// held prefill) issues: the decode side's lease, under the id and the
/// generation of `prefill`, the prefill side's lease as issued. The stages
/// hear the side's issue dated `now_micros`, the instant of the call. The
/// lease handed back keeps the instant of the pair's first decision, as
/// every later word of the side does. A decode its sessions rule keeps
/// from the profile is issued too, and no stage hears its issue: the call
/// asks its own pick's word of the decode side
/// ([`lease_events::stages_hear`]).
///
/// `None` for a decision that picks no pair.
pub(super) fn issue_decode(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    prefill: &Lease,
    now_micros: u64,
) -> Option<Lease> {
    let decode = decode_side(facts, decision, prefill)?;
    if lease_events::stages_hear(decision, Side::Decode) {
        let issued = Lease {
            issued_micros: now_micros,
            ..decode
        };
        engine.on_decision_issued(&issued, decision);
    }
    Some(decode)
}

/// The decode side's lease for a pick of the decode decided again, as
/// [`issue_decode`] hands it back, and no word to a stage: under the id and
/// the generation of `prefill`, with the three sizes the call states: its
/// token count and its text's two lengths. A call may state another text
/// than the first decision did, and its side reads the call's.
///
/// `None` for a decision that picks no pair.
fn decode_side(facts: &RequestFacts<'_>, decision: &Decision, prefill: &Lease) -> Option<Lease> {
    let first = decision.pick()?.targets.first()?;
    Some(Lease {
        tokens: facts.prompt_tokens,
        chars: facts.prompt_chars,
        bytes: facts.prompt_bytes,
        ..lease_events::decode_as_issued(prefill, first, decode_session(facts))?
    })
}

/// [`issue_decode`] for the decode call of `own`, told to `engine`'s
/// stages and dated at the call, under a catch of its own
/// ([`InProcess::word_caught`]). A stage that panics at the side's
/// issue is held and counted, as at a later word of the lease, and the
/// stages behind it do not hear the issue. The pick stands: the call
/// answers the side's lease all the same, so its guard holds the side as
/// issued and tells its dispatch and its end. A stage that heard the issue
/// then hears the side end.
///
/// A panic in the decision itself is not held here: it costs the call its
/// decision, as a first decision's does.
fn issue_decode_caught(
    own: Loaded<'_, '_>,
    engine: &Engine,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    prefill: &Lease,
) -> Option<Lease> {
    let told = own.word_caught(|| issue_decode(engine, facts, decision, prefill, facts.now_micros));
    told.unwrap_or_else(|| decode_side(facts, decision, prefill))
}

/// One decision of a pair by `own` over the `lists` its host handed, and
/// nothing its host hears of later: no lease is issued, no id is taken and
/// no stage is told. With no `target` it is the pair's first decision. With
/// the target the pair holds it is the decode decided again behind that
/// target's prefill ([`PairLists::decide`]). Both lists' records read their
/// workers' reports in `loads`, as a single's do.
///
/// A request held to wires is its host's to hand lists for. Where a worker
/// of either list is dialed on a wire the request does not state, the
/// request is declined under the wires' name, ahead of every test the
/// lists' own decision makes.
///
/// `sends_to` says whether the request's caller sends it to a target of
/// the pick ([`RequestMaker::sends_to`]): each side's record of sent
/// prefixes is written for the first target it takes. A decode decided
/// again is asked the same of its decode, and writes the decodes' record
/// for the first decode it takes. The entry its first decision wrote, for
/// the decode chosen then, stands beside it.
///
/// `counted` says where a worker's object counts the requests of the
/// request's session key ([`RequestMaker::session_counted_under`]). It is
/// asked under a document one of whose stages reads that count, and each
/// side's records then read it under the name of the side's own session.
///
/// No side's shadow is asked here, at either decision: no room is lent
/// for what one would say, and nobody would count it.
///
/// [`RequestMaker::sends_to`]: crate::engine::RequestMaker::sends_to
/// [`RequestMaker::session_counted_under`]: crate::engine::RequestMaker::session_counted_under
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "a pair's decision as its host takes it, and its maker's two words"
)]
pub(super) fn decide(
    own: Loaded<'_, '_>,
    lists: &PairLists<'_>,
    request: &mut Request<'_>,
    target: Option<&Target>,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    sends_to: impl Fn(&Target) -> bool,
    counted: impl FnOnce() -> SessionCountedUnder,
) -> Decision {
    decide_with(
        own, lists, request, target, loads, now, sends_to, counted, None,
    )
}

/// [`decide`], lent the room its caller has for the sides' shadows
/// ([`SidesSaid`]): a first decision lent one leaves there what each
/// side's shadow said beside it, and a decode decided again what its
/// decode side's said, for the caller to count. A request declined for
/// its wires has no shadow: no profile decided it.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "a pair's decision as its entries ask it, and its shadows' room"
)]
fn decide_with(
    own: Loaded<'_, '_>,
    lists: &PairLists<'_>,
    request: &mut Request<'_>,
    target: Option<&Target>,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    sends_to: impl Fn(&Target) -> bool,
    counted: impl FnOnce() -> SessionCountedUnder,
    said: Option<&mut SidesSaid>,
) -> Decision {
    // The engine's own value of each list, asked of the list's cell once.
    let sides = (own.members_of(lists.prefill), own.members_of(lists.decode));
    let wires = request.constraints.wire_modes;
    let dialed = sides.0.wires().union(sides.1.wires());
    if !wires.is_empty() && !wires.contains(dialed) {
        return single::declined(own, request, WIRES);
    }
    lists.decide_sides(
        own, sides, loads, now, request, target, sends_to, counted, said,
    )
}

/// One decision of a pair by `own`, the engine of a gateway's own process,
/// over the `lists` its host handed, as the host takes it: the pair's
/// decision alone ([`decide`]), the wires' test in it, with the lease a
/// pick issued ([`issued`]). With no `held` decision it is the pair's first.
/// With the view of the guard that holds the first it is the decode asked
/// again behind the prefill that view names. The decision reads the
/// reports in `loads`: the snapshot the request gives for this call, or
/// the newest the engine's own task keeps at hand. `sends_to` and `counted`
/// are handed to [`decide`] as they stand.
///
/// Under a document that names a shadow, a pair's decision is lent room
/// for what its shadows say: a first decision's two sides
/// ([`side_shadows`]), and the decode asked again its decode side's
/// ([`later_shadow`]). Each word is counted in its own table
/// ([`count_sides`]). The count stands ahead of the issue: an outcome
/// held across the count is copied once more for every pair, and a stage
/// that panics at that issue is held ([`issue_caught`],
/// [`issue_decode_caught`]), so the decision stands and its shadows are
/// counted in either order. The decision and the issue are the ones of a
/// document that names no shadow, which pays the test of that word here
/// and one test of the room in the decision.
#[expect(
    clippy::too_many_arguments,
    reason = "a pair's decision as its host takes it, and its maker's two words"
)]
pub(super) fn decided(
    own: Loaded<'_, '_>,
    lists: &PairLists<'_>,
    request: &mut Request<'_>,
    held: Option<&HeldDecision<'_>>,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    sends_to: impl Fn(&Target) -> bool,
    counted: impl FnOnce() -> SessionCountedUnder,
) -> DecideOutcome {
    let target = held.map(|held| held.target);
    let shadowed = own.document.any_shadow;
    let mut said: SidesSaid = [None; 2];
    let room = shadowed.then_some(&mut said);
    let decision = decide_with(
        own, lists, request, target, loads, now, sends_to, counted, room,
    );
    if shadowed {
        count_sides(own, said, held.is_some());
    }
    issued(own, request, decision, held, now)
}

/// What each shadow said beside a pair's decision by `own` ([`decided`]),
/// counted in its own table ([`with_shadow::count`]): a first decision's
/// under its side, and the decode decided `again` apart from the decode
/// its pair named first ([`ShadowSide::DecodeAgain`]).
///
/// Out of line: the path of a document that names no shadow holds no part
/// of it.
#[inline(never)]
fn count_sides(own: Loaded<'_, '_>, said: SidesSaid, again: bool) {
    for (side, said) in PairSide::BOTH.into_iter().zip(said) {
        if let Some(said) = said {
            let asked = if again {
                ShadowSide::DecodeAgain
            } else {
                ShadowSide::of_pair(side)
            };
            with_shadow::count(own, said, asked);
        }
    }
}

/// A pair's `decision` as its host takes it, with the lease its pick is
/// issued: the one issue of a pair, for its own decision and for an encode
/// request's ([`super::encode::decided`]). With no `held` decision the pick
/// is a pair's first, issued under the engine's next id
/// ([`issue_caught`]). With the view of the guard that holds the first,
/// the pick issues the decode side under the held lease, dated at this
/// call ([`issue_decode_caught`]). Either issue is told under a catch of
/// its own: a stage that panics at it costs the call nothing. A lease's
/// dispatch is due as a single's. A decision that picks no pair issues
/// nothing. Behind a held lease of an earlier document than the one that
/// decides, the side's issue is told as each later word of that lease
/// is: to the stages that stood under that document, and to no stage built
/// since ([`InProcess::apply`]).
///
/// The pick of a decode asked again decided no prefill. It keeps the held
/// pick's word of who placed the prefill side
/// ([`Targets::placed_as`](crate::engine::decision::Targets::placed_as)):
/// that word is on no lease, so the pick alone states it to the guard that
/// takes this pick and to its host. The word is written where the answer
/// stands, behind the decode's issue: a stage that reads
/// [`Pick::placed_by`](crate::engine::decision::Pick::placed_by) at that
/// issue reads the stages, as the engine's decode entry states of a
/// prefill it did not place.
// In line at its two callers: as a call it costs a pair's decision 33
// instructions, counted with Callgrind.
#[inline(always)]
pub(super) fn issued(
    own: Loaded<'_, '_>,
    request: &Request<'_>,
    decision: Decision,
    held: Option<&HeldDecision<'_>>,
    now: Instant,
) -> DecideOutcome {
    let (facts, later) = (&request.facts, request.constraints.decode_later);
    let issued = match held {
        // A shed writes no word that every pick shares.
        None => decision.pick().and_then(|_| {
            let id = LeaseId(own.leases.0.fetch_add(1, Ordering::Relaxed) + 1);
            issue_caught(own, facts, &decision, id, later)
        }),
        Some(held) => {
            // Behind a held lease of an earlier document the side's issue
            // is told as its later words are: to the stages that stood
            // under that document.
            let document = own.document;
            let hearing = if held.lease.generation == document.spec.generation() {
                document.spec.engine()
            } else {
                document.stood_at(held.lease.generation)
            };
            issue_decode_caught(own, hearing, facts, &decision, held.lease)
                .map(|decode| (decode, decode.session))
        }
    };
    let mut outcome = DecideOutcome {
        decision,
        lease: issued.map(|(lease, _)| Leased {
            lease,
            due: now + DISPATCH_TTL,
        }),
        decode_session: issued.map_or(Session::NONE, |(_, session)| session),
    };
    // The prefill side's word of the pair's first decision, kept where the
    // answer stands: a write into the decision ahead of its issue costs
    // every pair a copy of its pick. The decode's issue reads the decode
    // side's word, which the pick's pairs state.
    let first = held.and_then(|held| held.decision.pick());
    if let (Decision::Pick(pick), Some(first)) = (&mut outcome.decision, first) {
        pick.targets.placed_as(&first.targets);
    }
    outcome
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/pair/tests/mod.rs"]
mod tests;
