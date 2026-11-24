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
//! decode asked again judges its decodes alone. An encode request's three
//! lists are held to no demand: such a request is shed under the wires'
//! name.
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
//! the side's dispatch, and the worker it went to from there.
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
//! its own list's column for the decision ([`overlap::column`]): what each
//! worker of the side holds of the request, by the host's index of cache
//! events and by the record of sent prefixes, read at the side's phase.
//! Behind a pick its profile is told of, each side's record keeps where
//! the prompt went ([`overlap::record_prefix`]). The decode asked again
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

use std::{
    cell::RefCell,
    fmt,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot};

use super::{overlap, signals, single, Loaded, PolicyEngine, PoolMembers};
use crate::{
    engine::{
        decision::{shed, Decision, ShedReason, Target, SUBSET_HINT, WIRES},
        decision_engine::{DecideOutcome, HeldDecision, Leased, Request},
        demands::{Demands, UnmetDemand},
        framework::{Engine, PairPools, Scratch},
        lease_events,
        list_cell::LentList,
        pair_classes::PartnerClasses,
        records::{vetoes, HeldCells, Live, PoolRecords},
        view::{
            AffinityFacts, Candidate, Constraints, Phase, PoolFacts, RequestFacts, Role, Session,
            WorkerId,
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
        engine: &PolicyEngine,
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
        held: Option<&Target>,
    ) -> Decision {
        let engine = engine.loaded();
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
        self.decide_sides(engine, sides, loads, now, request, held, true, |_| true)
    }

    /// A pair's first decision with a hard demand its request states left
    /// to the records' decision, which sheds it ([`not_decided`]): what an
    /// encode request's entry asks for a request its three lists do not
    /// decide. Those lists are held to no demand.
    pub(super) fn decide_unjudged(
        &self,
        engine: Loaded<'_, '_>,
        sides: (&PoolMembers, &PoolMembers),
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
    ) -> Decision {
        self.decide_sides(engine, sides, loads, now, request, None, false, |_| true)
    }

    /// [`Self::decide_over`], and whether a hard demand the request states
    /// is `judged` over the two lists: each side is then held to the
    /// places whose worker meets it, and the records' decision is handed
    /// none. `sends_to` says whether the request's caller sends it to a
    /// target of the pick ([`RequestMaker::sends_to`]): each side's record
    /// of sent prefixes is written for the first target it takes.
    ///
    /// [`RequestMaker::sends_to`]: crate::engine::RequestMaker::sends_to
    #[expect(
        clippy::too_many_arguments,
        reason = "a pair's decision over its two lists, and how it reads a demand"
    )]
    fn decide_sides(
        &self,
        engine: Loaded<'_, '_>,
        (prefill, decode): (&PoolMembers, &PoolMembers),
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        request: &mut Request<'_>,
        held: Option<&Target>,
        judged: bool,
        sends_to: impl Fn(&Target) -> bool,
    ) -> Decision {
        let (spec, clock) = (&engine.document.spec, &engine.clock);
        // The host's indexes of cache events and the records of sent
        // prefixes, which a side's column is read from.
        let kept = &engine.affinity;
        // One value for both lists' records: this call's instant on the
        // clock and the host's snapshot, as the host holds it. No session
        // key's text is handed to it: a pair's records carry no count of a
        // key's requests.
        let live = Live::at(spec.engine(), clock, now, loads, None);
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
            // The decode asked again reads one prefill, the held one.
            let (prefills, prefill_facts) = match held {
                Some(held) => held_prefill(prefill_records, workers, &live, held, prefills),
                None => prefill_records.write(workers, &live, prefills),
            };
            let (workers, decode_records) = (decode.workers(), decode.records(spec, clock));
            let (decodes, decode_facts) = decode_records.write(workers, &live, decodes);
            stamp(prefills, decodes, self.classes, self.admission);
            let (engine, scratch) = (spec.engine(), &mut buffers.scratch);
            let [prefill_side, decode_side] = sides;
            // A hard demand the request states is judged here, a side at a
            // time, by what each side's role is held to. The records'
            // decision is handed the places and no demand. The decode
            // asked again judges no prefill: the held one ran already.
            let spent;
            let (cx, prefill_side, decode_side) = if judged && !cx.demands.is_none() {
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
            // own list's column, read at the side's phase. The decode
            // asked again reads the decodes' alone: its prefill ran
            // already. A side whose phase reads none pays this one test.
            let [prefill_column, decode_column] = &mut buffers.columns;
            let fill = |side: &PoolMembers, records: &PoolRecords, phase, column| {
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
            let reads = |phase| engine.needs_for_phase(phase).affinity;
            prefill_column.clear();
            let prefill_read = (held.is_none() && reads(Phase::Prefill))
                .then(|| fill(prefill, prefill_records, Phase::Prefill, prefill_column))
                .flatten();
            decode_column.clear();
            let decode_read = reads(Phase::Decode)
                .then(|| fill(decode, decode_records, Phase::Decode, decode_column))
                .flatten();
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
            let decision = match held {
                None => {
                    let sides = (cx, [prefill_side, decode_side]);
                    decide_records(engine, facts, sides, &prefill_facts, pools, scratch)
                }
                Some(held) => {
                    let side = (cx, decode_side);
                    decide_decode_records(engine, facts, side, &decode_facts, pools, held, scratch)
                }
            };
            // Behind a pick its profile is told of, each side's record of
            // sent prefixes keeps where the prompt went: the first target
            // the request's caller sends to (`sends_to`), and nobody
            // where it sends to none. A pair neither side of which read
            // its record asks nothing of the pick.
            let read = prefill_read.is_some() || decode_read.is_some();
            let sent = read.then(|| overlap::sent_target(&decision, &sends_to));
            if let Some(sent) = sent.flatten() {
                if let Some(read) = prefill_read {
                    let pool = prefill.affinity();
                    overlap::record_prefix(pool, Phase::Prefill, facts, read, sent.id);
                }
                if let (Some(read), Some(pair)) = (decode_read, sent.pair) {
                    let pool = decode.affinity();
                    overlap::record_prefix(pool, Phase::Decode, facts, read, pair.decode_id);
                }
            }
            decision
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
/// leaves them as they were. A pair that states a hard demand keeps each
/// side's places that meet it here, the prefills' first: such a request
/// allocates nothing once each list has grown to its side.
/// A side whose phase reads who holds the prompt keeps its list's column
/// here, the prefills' first.
#[derive(Debug, Default)]
pub(super) struct Buffers {
    pub(super) rows: [Vec<Candidate>; 2],
    pub(super) encoders: Vec<Candidate>,
    pub(super) scratch: Scratch,
    meeting: [Vec<u32>; 2],
    columns: [Vec<AffinityFacts>; 2],
}

thread_local! {
    /// The calling thread's own buffers, so a pair's decision allocates
    /// nothing after a thread's first decisions over lists of a size.
    pub(super) static BUFFERS: RefCell<Buffers> = RefCell::new(Buffers::default());
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
/// ([`PairLists::decide`]), and hands them none. A demand that reaches
/// them is an encode request's, whose three lists are held to none: it is
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
/// the records decline by name.
fn decide_records(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    (cx, sides): (&Constraints<'_>, [Beside<'_>; 2]),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    scratch: &mut Scratch,
) -> Decision {
    if pools.prefills.is_empty() || pools.decodes.is_empty() {
        let none = PoolFacts {
            generation: pool.generation,
            ..PoolFacts::default()
        };
        return engine.decide_pair(facts, &none, PairPools::default(), cx, cx, scratch);
    }
    if let Some((reason, stage)) = not_decided(cx) {
        return shed(reason, stage, facts.seed, pool.generation);
    }
    let [prefills, decodes] = sides;
    // A side the request's demand left nobody: the prefills' first.
    if let Some(unmet) = prefills.unmet.or(decodes.unmet) {
        let stage = unmet.stage();
        return shed(ShedReason::NoEndpoints, stage, facts.seed, pool.generation);
    }
    if prefills.states_nothing() && decodes.states_nothing() {
        return engine.decide_pair(facts, pool, pools, cx, cx, scratch);
    }
    // Each side's set opens with its own places, under its own mask.
    let (prefill_cx, decode_cx) = (prefills.of(cx), decodes.of(cx));
    engine.decide_pair(facts, pool, pools, &prefill_cx, &decode_cx, scratch)
}

/// What a pair's first pick ([`PairLists::decide`]) issues, for a first
/// decision of `own`: the prefill side's lease under `id`, which is the
/// answer's lease, and the request's session under the decode phase, which
/// its guard keeps for the decode side's words ([`first_sides`]). The
/// stages hear the prefill's issue, and the decode's behind it unless the
/// request sends its decode `later` ([`tell_first`]): such a side stands
/// chosen and is told at its own issue ([`issue_decode`],
/// [`lease_events::pair`]).
///
/// The two issues are told under one catch of their own
/// ([`PolicyEngine::word_caught`]). A stage that panics at a side's issue
/// is held and counted, as at a later word of the lease, and the stages
/// behind it do not hear that issue, nor the decode side's where the
/// prefill side's panicked. The pick stands: the leases are made ahead of
/// the catch, so the decision answers its lease all the same, and its guard
/// holds the pair as issued and tells each side's dispatch and its end. A
/// stage that heard an issue then hears that side end.
///
/// A panic in the decision itself is not held here: it costs the request
/// its decision ([`PolicyEngine::caught`]).
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
    own.word_caught(|| tell_first(engine, &lease, &decode, decision, later));
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

/// The issue of a pair's first pick told to the stages of `engine`: the
/// prefill side's `lease`, then the `decode` side's unless the request
/// sends its decode `later`.
#[inline]
fn tell_first(engine: &Engine, lease: &Lease, decode: &Lease, decision: &Decision, later: bool) {
    engine.on_decision_issued(lease, decision);
    if !later {
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
        return shed(ShedReason::NoEndpoints, stage, facts.seed, pool.generation);
    }
    let first = held.pair.map(|pair| (pair.decode_id, pair.decode_dp_rank));
    if side.states_nothing() {
        return engine.decide_decode(&facts, pool, pools, prefill, first, cx, scratch);
    }
    let decode_cx = side.of(cx);
    engine.decide_decode(&facts, pool, pools, prefill, first, &decode_cx, scratch)
}

/// What a pick of the decode decided again ([`PairLists::decide`] behind a
/// held prefill) issues: the decode side's lease, under the id and the
/// generation of `prefill`, the prefill side's lease as issued. The stages
/// hear the side's issue dated `now_micros`, the instant of the call. The
/// lease handed back keeps the instant of the pair's first decision, as
/// every later word of the side does.
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
    let issued = Lease {
        issued_micros: now_micros,
        ..decode
    };
    engine.on_decision_issued(&issued, decision);
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

/// [`issue_decode`] for the decode call of `own`, dated at the call, under
/// a catch of its own ([`PolicyEngine::word_caught`]). A stage that panics
/// at the side's issue is held and counted, as at a later word of the
/// lease, and the stages behind it do not hear the issue. The pick stands:
/// the call answers the side's lease all the same, so its guard holds the
/// side as issued and tells its dispatch and its end. A stage that heard
/// the issue then hears the side end.
///
/// A panic in the decision itself is not held here: it costs the call its
/// decision, as a first decision's does.
fn issue_decode_caught(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    prefill: &Lease,
) -> Option<Lease> {
    let engine = own.document.spec.engine();
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
/// [`RequestMaker::sends_to`]: crate::engine::RequestMaker::sends_to
#[inline]
pub(super) fn decide(
    own: Loaded<'_, '_>,
    lists: &PairLists<'_>,
    request: &mut Request<'_>,
    target: Option<&Target>,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    sends_to: impl Fn(&Target) -> bool,
) -> Decision {
    // The engine's own value of each list, asked of the list's cell once.
    let sides = (own.members_of(lists.prefill), own.members_of(lists.decode));
    let wires = request.constraints.wire_modes;
    let dialed = sides.0.wires().union(sides.1.wires());
    if !wires.is_empty() && !wires.contains(dialed) {
        return single::declined(own, request, WIRES);
    }
    lists.decide_sides(own, sides, loads, now, request, target, true, sends_to)
}

/// One decision of a pair by `own`, the engine of a gateway's own process,
/// over the `lists` its host handed, as the host takes it: the pair's
/// decision alone ([`decide`]), the wires' test in it, with the lease a
/// pick issued ([`issued`]). With no `held` decision it is the pair's first.
/// With the view of the guard that holds the first it is the decode asked
/// again behind the prefill that view names. The decision reads the
/// reports in `loads`: the snapshot the request gives for this call, or
/// the newest the engine's own task keeps at hand. `sends_to` is handed
/// to [`decide`] as it stands.
pub(super) fn decided(
    own: Loaded<'_, '_>,
    lists: &PairLists<'_>,
    request: &mut Request<'_>,
    held: Option<&HeldDecision<'_>>,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    sends_to: impl Fn(&Target) -> bool,
) -> DecideOutcome {
    let target = held.map(|held| held.target);
    let decision = decide(own, lists, request, target, loads, now, sends_to);
    issued(own, request, decision, held, now)
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
/// nothing.
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
        Some(held) => issue_decode_caught(own, facts, &decision, held.lease)
            .map(|decode| (decode, decode.session)),
    };
    DecideOutcome {
        decision,
        lease: issued.map(|(lease, _)| Leased {
            lease,
            due: now + DISPATCH_TTL,
        }),
        decode_session: issued.map_or(Session::NONE, |(_, session)| session),
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeSet, sync::Arc, time::Duration};

    use parking_lot::Mutex;
    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        load::state::{one_rank_load, TestFeed},
        test_support::no_health_check,
        worker::{ConnectionMode, WorkerLoadResponse, WorkerSpec, WorkerStatus, WorkerType},
    };

    use super::*;
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{
            decision::{Pair, Pick, Target, Targets},
            decision_engine::{DecisionEngine, HeldDecode, RequestMaker, WorkerLists},
            demands::DemandFacts,
            framework::{Filter, Picker, TieBand},
            list_cell::tests::PublishedList,
            prompt_hashes::{HashSeeds, PromptHashes},
            records::worker_id,
            sticky_sessions::{DecodeVerdict, SessionVerdict},
            view::{FixedFacts, Identity, WireModes},
        },
        policies::consistent_hashing::InOrder,
        stages::{Best, HasCapacity, StageRegistry},
        store::GatewayId,
        tracking::{
            lease::{Dispatch, Outcome, ReleaseReason, Said, TransportClass},
            DecisionGuard,
        },
    };

    mod vetoed;

    const MODEL: &str = "model-a";

    /// One word a stage heard of a lease: what was said, the side, the
    /// worker the lease books, the index it names, its session, its issue
    /// instant, and the worker the word names beside the lease (the pick's
    /// first target at an issue, the target at a dispatch, the worker that
    /// served at a completion).
    type Word = (&'static str, Side, u64, u8, Session, u64, u64);

    thread_local! {
        /// The prompt size each word a stage heard on this thread stated,
        /// beside the word and its side.
        static SIZES: RefCell<Vec<(&'static str, Side, u32)>> = const { RefCell::new(Vec::new()) };
    }

    /// The sizes the stages heard on this thread since the last read.
    fn sizes() -> Vec<(&'static str, Side, u32)> {
        SIZES.with_borrow_mut(std::mem::take)
    }

    thread_local! {
        /// The side of each word a stage heard on this thread, and what
        /// its lease stated of its request's text: its characters, its
        /// bytes and the size made of them ([`Lease::prompt_size`]).
        static LENGTHS: RefCell<Vec<(Side, [u32; 3])>> = const { RefCell::new(Vec::new()) };
    }

    /// A picker that picks as [`Best`] and keeps every word it hears.
    #[derive(Debug)]
    struct Hears(Arc<Mutex<Vec<Word>>>);

    impl Hears {
        fn heard(&self, word: &'static str, lease: &Lease, named: u64) {
            self.0.lock().push((
                word,
                lease.side,
                lease.booked.0,
                lease.target_index,
                lease.session,
                lease.issued_micros,
                named,
            ));
            SIZES.with_borrow_mut(|sizes| sizes.push((word, lease.side, lease.tokens)));
            let stated = [lease.chars, lease.bytes, lease.prompt_size()];
            LENGTHS.with_borrow_mut(|lengths| lengths.push((lease.side, stated)));
        }
    }

    impl Lifecycle for Hears {
        fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
            let first = decision.pick().and_then(|pick| pick.targets.first());
            self.heard("issued", lease, first.map_or(0, |first| first.id.0));
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
            self.heard("dispatched", lease, target.id.0);
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            self.heard("completed", lease, outcome.served.0);
        }

        fn on_released(&self, lease: &Lease, _: ReleaseReason) {
            self.heard("released", lease, 0);
        }
    }

    impl Picker for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            rows: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Best.pick(facts, rows, kept, scores, band, out);
        }
    }

    /// The stock pair document over round robin on both sides, its decode
    /// side picking through a stage that keeps what it hears: the document
    /// as applied, and the words.
    fn hearing() -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
        hearing_with(None)
    }

    /// A picker of a case's own: its name in the catalog and the stage it
    /// makes.
    type OwnPicker = (&'static str, fn() -> Box<dyn Picker>);

    /// [`hearing`], its prefill side picking through `prefill` where a case
    /// names one, which is added to the catalog.
    fn hearing_with(prefill: Option<OwnPicker>) -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
        let heard = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&heard);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("hears", move || Box::new(Hears(Arc::clone(&kept))))
            .expect("a free picker name");
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        document.generation = 3;
        let decode = document.profiles.get_mut("decode");
        decode.expect("the decode profile").pick = "hears".into();
        if let Some((name, stage)) = prefill {
            catalog.picker(name, stage).expect("a free picker name");
            let side = document.profiles.get_mut("prefill");
            side.expect("the prefill profile").pick = name.into();
        }
        let digest = document.digest().expect("a digest");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        (spec.expect("a valid document"), heard)
    }

    /// The host's admission as two tests of a place: the prefills', then
    /// the decodes'.
    #[derive(Debug)]
    struct Gates(fn(usize) -> bool, fn(usize) -> bool);

    impl PairAdmission for Gates {
        fn has_capacity(&self, prefill: usize) -> bool {
            (self.0)(prefill)
        }

        fn has_room(&self, decode: usize) -> bool {
            (self.1)(decode)
        }
    }

    /// Every prefill has capacity and every decode a room.
    const OPEN: Gates = Gates(|_| true, |_| true);

    /// A ready record of `role` at `position` of its list.
    fn row(position: u32, id: u64, role: Role) -> Candidate {
        Candidate::ready(position, WorkerId(id), role)
    }

    /// Prefills 1.., and decodes 101.., each in its list's order.
    fn lists(prefills: u64, decodes: u64) -> (Vec<Candidate>, Vec<Candidate>) {
        let prefills = (0..prefills).map(|at| row(at as u32, at + 1, Role::Prefill));
        let decodes = (0..decodes).map(|at| row(at as u32, at + 101, Role::Decode));
        (prefills.collect(), decodes.collect())
    }

    fn facts() -> RequestFacts<'static> {
        RequestFacts {
            model: MODEL,
            phase: Phase::Prefill,
            prompt_tokens: 40,
            now_micros: 7_000,
            seed: 11,
            ..RequestFacts::default()
        }
    }

    /// A session key, so each side's session is one a case can tell.
    fn keyed() -> RequestFacts<'static> {
        RequestFacts {
            session_key: Identity {
                hash: 77,
                plain: None,
            },
            ..facts()
        }
    }

    fn pool(rows: &[Candidate]) -> PoolFacts<'static> {
        PoolFacts {
            pool_size: rows.len() as u32,
            generation: 3,
            ..PoolFacts::default()
        }
    }

    fn pools<'a>(
        prefills: &'a [Candidate],
        decodes: &'a [Candidate],
        classes: &'a PartnerClasses,
    ) -> PairPools<'a> {
        PairPools {
            table: classes.table(),
            ..PairPools::of(prefills, decodes)
        }
    }

    /// The pair `decision` picks first: its prefill and its decode.
    fn picked(decision: &Decision) -> (u64, u64) {
        let pick = decision.pick().expect("a pick");
        let first = pick.targets.first().expect("a target");
        (first.id.0, first.pair.expect("a pair").decode_id.0)
    }

    /// Every worker pairs with every worker.
    fn every(prefills: usize, decodes: u32) -> PartnerClasses {
        let all: Vec<u32> = (0..decodes).collect();
        PartnerClasses::number(decodes as usize, (0..prefills).map(|_| all.as_slice()))
    }

    /// What a host states beside one side's list where it holds the side
    /// to `places` and no veto of its own stands on a place.
    fn unvetoed(places: Option<&[u32]>) -> Beside<'_> {
        Beside {
            places,
            stalled: &[],
            held: &[],
            unmet: None,
        }
    }

    /// The pair decided now over the two lists' records.
    fn decide_over(
        spec: &AppliedSpec,
        facts: &RequestFacts<'_>,
        prefills: &[Candidate],
        decodes: &[Candidate],
        classes: &PartnerClasses,
    ) -> Decision {
        let free = Constraints::default();
        decide_records(
            spec.engine(),
            facts,
            (&free, [unvetoed(None); 2]),
            &pool(prefills),
            pools(prefills, decodes, classes),
            &mut Scratch::default(),
        )
    }

    #[test]
    fn the_capacity_filters_name_is_the_one_a_full_list_is_read_by() {
        assert_eq!(HasCapacity.name(), HAS_CAPACITY);
    }

    #[test]
    fn a_stamp_writes_each_records_class_and_its_sides_admission_bit() {
        let (mut prefills, mut decodes) = lists(3, 2);
        let classes = PartnerClasses::number(2, [&[0u32][..], &[1][..]]);
        let gates = Gates(|at| at != 1, |at| at == 1);
        stamp(&mut prefills, &mut decodes, &classes, &gates);
        let class = |rows: &[Candidate]| rows.iter().map(|row| row.pair_class).collect::<Vec<_>>();
        // The third prefill stands past the classes: it pairs with nobody.
        assert_eq!(class(&prefills), [1, 2, 0]);
        assert_eq!(class(&decodes), [1, 2]);
        let capacity: Vec<bool> = prefills.iter().map(|row| row.has_capacity).collect();
        assert_eq!(capacity, [true, false, true]);
        let room: Vec<bool> = decodes.iter().map(|row| row.has_room).collect();
        assert_eq!(room, [false, true]);
    }

    /// Two groups that never pair: whichever prefill the rotation lands
    /// on, its decode is one of its own group.
    #[test]
    fn a_pair_is_decided_among_the_partners_the_classes_name() {
        let (spec, _) = hearing();
        let (mut prefills, mut decodes) = lists(2, 4);
        let classes = PartnerClasses::number(4, [&[0u32, 1][..], &[2, 3][..]]);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let mut seen = [false; 2];
        for seed in 0..16 {
            let facts = RequestFacts { seed, ..facts() };
            let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
            let (prefill, decode) = picked(&decision);
            let group = if decode <= 102 { 1 } else { 2 };
            assert_eq!(group, prefill, "prefill {prefill} with decode {decode}");
            seen[prefill as usize - 1] = true;
        }
        assert_eq!(seen, [true; 2], "the rotation reaches both prefills");
    }

    /// The capacity bit is the host's own read: a full prefill takes no
    /// request while another has room, and a list of full prefills is not
    /// a refusal but a request that waits.
    #[test]
    fn a_full_prefill_is_passed_over_and_a_list_of_full_prefills_reads_as_full() {
        let (spec, _) = hearing();
        let (mut prefills, mut decodes) = lists(2, 1);
        let classes = every(2, 1);
        stamp(
            &mut prefills,
            &mut decodes,
            &classes,
            &Gates(|at| at == 1, |_| true),
        );
        for seed in 0..4 {
            let facts = RequestFacts { seed, ..facts() };
            let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
            assert_eq!(picked(&decision), (2, 101));
            assert!(!prefills_full(&decision));
        }
        stamp(
            &mut prefills,
            &mut decodes,
            &classes,
            &Gates(|_| false, |_| true),
        );
        let decision = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
        assert!(prefills_full(&decision), "{decision:?}");
    }

    /// The room bit is soft: a decode with no room is passed over while
    /// another has one, and where none has the list stands and the
    /// dispatch waits at its pick's gate.
    #[test]
    fn a_decode_with_no_room_is_passed_over_and_stands_where_none_has_one() {
        let (spec, _) = hearing();
        let (mut prefills, mut decodes) = lists(1, 2);
        let classes = every(1, 2);
        stamp(
            &mut prefills,
            &mut decodes,
            &classes,
            &Gates(|_| true, |at| at == 1),
        );
        for seed in 0..4 {
            let facts = RequestFacts { seed, ..facts() };
            let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
            assert_eq!(picked(&decision), (1, 102));
        }
        stamp(
            &mut prefills,
            &mut decodes,
            &classes,
            &Gates(|_| true, |_| false),
        );
        let none_has = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
        assert_eq!(picked(&none_has).0, 1, "the list stands: {none_has:?}");
    }

    #[test]
    fn a_request_that_states_what_a_pair_is_not_narrowed_by_is_shed_under_that_stage() {
        let (spec, _) = hearing();
        let (mut prefills, mut decodes) = lists(1, 1);
        let classes = every(1, 1);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let shed_under = |cx: &Constraints<'_>| {
            let decision = decide_records(
                spec.engine(),
                &facts(),
                (cx, [unvetoed(None); 2]),
                &pool(&prefills),
                pools(&prefills, &decodes, &classes),
                &mut Scratch::default(),
            );
            match decision {
                Decision::Shed(shed) => (shed.reason, shed.emptied_by),
                other => panic!("not shed: {other:?}"),
            }
        };
        let demanded = Constraints {
            demands: Demands {
                vision: true,
                ..Demands::default()
            },
            ..Constraints::default()
        };
        assert_eq!(shed_under(&demanded), (ShedReason::NoEndpoints, WIRES));
        let none = Constraints {
            subset_hint: Some(0),
            ..Constraints::default()
        };
        assert_eq!(shed_under(&none), (ShedReason::NoEndpoints, SUBSET_HINT));
        let some = Constraints {
            subset_hint: Some(2),
            ..Constraints::default()
        };
        let declined = (ShedReason::PolicyDeclined, SUBSET_HINT);
        assert_eq!(shed_under(&some), declined);
        let held = Constraints {
            places: Some(&[0]),
            ..Constraints::default()
        };
        let declined = (ShedReason::PolicyDeclined, single::PLACES);
        assert_eq!(shed_under(&held), declined);
    }

    #[test]
    fn one_side_alone_is_no_pair() {
        let (spec, _) = hearing();
        let (mut prefills, mut decodes) = lists(2, 0);
        let classes = every(2, 0);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let decision = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
        // A pair's first decision over no candidate is emptied by no stage.
        let none = matches!(decision, Decision::Shed(shed) if shed.emptied_by == "none");
        assert!(none, "{decision:?}");
        assert!(!prefills_full(&decision));
    }

    /// One prefill and its decodes, the pick over them, and the words the
    /// document's stages hear.
    struct Decided {
        spec: AppliedSpec,
        heard: Arc<Mutex<Vec<Word>>>,
        decision: Decision,
    }

    impl Decided {
        /// The words heard since the last call.
        fn words(&self) -> Vec<Word> {
            std::mem::take(&mut *self.heard.lock())
        }
    }

    /// A pair decided over one prefill (worker 1) and `decodes` decodes
    /// (workers 101..), every pair open.
    fn decided(decodes: u64) -> Decided {
        let (spec, heard) = hearing();
        let (mut prefills, mut rows) = lists(1, decodes);
        let classes = every(1, decodes as u32);
        stamp(&mut prefills, &mut rows, &classes, &OPEN);
        let decision = decide_over(&spec, &facts(), &prefills, &rows, &classes);
        Decided {
            spec,
            heard,
            decision,
        }
    }

    #[test]
    fn a_pairs_issue_tells_each_side_once_under_its_own_lease() {
        let pair = decided(1);
        let facts = keyed();
        let issued = issue(
            pair.spec.engine(),
            &facts,
            &pair.decision,
            LeaseId(9),
            false,
        );
        let (lease, session) = issued.expect("a pair's lease");
        assert_eq!(
            (lease.id, lease.side, lease.booked, lease.target_index),
            (LeaseId(9), Side::Prefill, WorkerId(1), 0)
        );
        assert_eq!(
            (lease.generation, lease.tokens, lease.issued_micros),
            (3, 40, 7_000)
        );
        assert_eq!(lease.session, facts.session());
        assert_ne!(session, lease.session, "each side keeps its own session");
        assert_eq!(
            pair.words(),
            [
                ("issued", Side::Prefill, 1, 0, lease.session, 7_000, 1),
                ("issued", Side::Decode, 101, 0, session, 7_000, 1),
            ]
        );
    }

    #[test]
    fn a_decode_sent_later_is_not_told_with_its_prefill() {
        let pair = decided(1);
        let issued = issue(
            pair.spec.engine(),
            &keyed(),
            &pair.decision,
            LeaseId(9),
            true,
        );
        let (lease, _) = issued.expect("a pair's lease");
        assert_eq!(
            pair.words(),
            [("issued", Side::Prefill, 1, 0, lease.session, 7_000, 1)]
        );
    }

    #[test]
    fn a_decision_that_picks_no_pair_issues_nothing() {
        let pair = decided(1);
        let none = shed(ShedReason::NoEndpoints, WIRES, 0, 3);
        assert!(issue(pair.spec.engine(), &facts(), &none, LeaseId(1), false).is_none());
        assert!(pair.words().is_empty());
    }

    /// The second ask runs over the records of its own instant: the held
    /// prefill's partners, with the room each has then. Its pick names the
    /// held prefill, and its issue is told dated at the call under the
    /// lease's own id.
    #[test]
    fn the_decode_asked_again_is_decided_over_the_held_prefills_partners() {
        let pair = decided(2);
        let facts = keyed();
        let issued = issue(pair.spec.engine(), &facts, &pair.decision, LeaseId(9), true);
        let (lease, session) = issued.expect("a pair's lease");
        pair.words();
        // At the call the first decode has no room and the second has one.
        let (mut prefills, mut decodes) = lists(1, 2);
        let classes = every(1, 2);
        stamp(
            &mut prefills,
            &mut decodes,
            &classes,
            &Gates(|_| true, |at| at == 1),
        );
        let again = decide_decode_records(
            pair.spec.engine(),
            &facts,
            (&Constraints::default(), unvetoed(None)),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(1)),
            &mut Scratch::default(),
        );
        assert_eq!(picked(&again), (1, 102));
        let decode = issue_decode(pair.spec.engine(), &facts, &again, &lease, 9_500);
        let decode = decode.expect("the decode side's lease");
        assert_eq!(
            (decode.id, decode.side, decode.booked, decode.session),
            (LeaseId(9), Side::Decode, WorkerId(102), session)
        );
        assert_eq!(
            decode.issued_micros, 7_000,
            "the lease keeps its first instant"
        );
        assert_eq!(
            pair.words(),
            [("issued", Side::Decode, 102, 0, session, 9_500, 1)]
        );
    }

    #[test]
    fn a_prefill_that_left_its_list_sheds_the_second_ask() {
        let pair = decided(1);
        let (mut prefills, mut decodes) = lists(1, 1);
        let classes = every(1, 1);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let again = decide_decode_records(
            pair.spec.engine(),
            &facts(),
            (&Constraints::default(), unvetoed(None)),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(5)),
            &mut Scratch::default(),
        );
        match again {
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "prefill-gone"),
            other => panic!("not shed: {other:?}"),
        }
    }

    /// A ready worker of `worker_type` at `url`.
    fn worker(url: String, worker_type: WorkerType) -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::from_spec(WorkerSpec {
            worker_type,
            ..WorkerSpec::new(url)
        })
        .health_config(no_health_check())
        .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    }

    /// A held target that names its prefill alone: no decode of a first
    /// decision is handed with it.
    fn prefill_alone(prefill: WorkerId) -> Target {
        Target {
            id: prefill,
            ..Target::default()
        }
    }

    /// A host's two lists of ready workers as it publishes them, each with
    /// its cell, under the engine of the stock pair document.
    struct Host {
        engine: PolicyEngine,
        heard: Arc<Mutex<Vec<Word>>>,
        prefills: Arc<[Arc<dyn Worker>]>,
        decodes: Arc<[Arc<dyn Worker>]>,
        lists: [PublishedList; 2],
    }

    /// A pair's request stated ahead, with the lists its host hands for it
    /// and its snapshot of the workers' reports, where it lends one.
    struct Asks<'a>(Request<'a>, PairLists<'a>, Option<Arc<LoadSnapshot>>);

    impl RequestMaker for Asks<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.0
        }

        fn workers(&self) -> WorkerLists<'_> {
            WorkerLists::Pair(self.1)
        }

        fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
            self.2.as_ref()
        }
    }

    impl Host {
        fn of(prefills: usize, decodes: usize) -> Self {
            Self::under(hearing(), prefills, decodes)
        }

        /// [`Self::of`] under the document `spec`, with the words its
        /// stages keep.
        fn under(
            (spec, heard): (AppliedSpec, Arc<Mutex<Vec<Word>>>),
            prefills: usize,
            decodes: usize,
        ) -> Self {
            let side = |name: &str, worker_type, count| -> Arc<[Arc<dyn Worker>]> {
                let url = |at: usize| format!("grpc://{name}-{at}:9000");
                (0..count).map(|at| worker(url(at), worker_type)).collect()
            };
            let prefills = side("prefill", WorkerType::Prefill, prefills);
            let decodes = side("decode", WorkerType::Decode, decodes);
            Self::over((spec, heard), prefills, decodes)
        }

        /// [`Self::under`] over the two lists as their host built them.
        fn over(
            (spec, heard): (AppliedSpec, Arc<Mutex<Vec<Word>>>),
            prefills: Arc<[Arc<dyn Worker>]>,
            decodes: Arc<[Arc<dyn Worker>]>,
        ) -> Self {
            let engine = PolicyEngine::new(spec);
            let lists = [&prefills, &decodes].map(|list| PublishedList::new(Arc::clone(list)));
            Self {
                engine,
                heard,
                prefills,
                decodes,
                lists,
            }
        }

        /// [`Self::of`] under a document whose profiles list no filter: a
        /// valid document that weighs no worker's health.
        fn unfiltered(prefills: usize, decodes: usize) -> Self {
            let mut host = Self::of(prefills, decodes);
            let mut document =
                PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
            document.generation = 3;
            for profile in document.profiles.values_mut() {
                profile.filter.clear();
            }
            let spec = AppliedSpec::validate(&document).expect("a valid document");
            // The lists stand: each cell makes the new engine's value at
            // its first ask.
            host.engine = PolicyEngine::new(spec);
            host
        }

        /// The host's keyed request for a pair over its lists, every pair
        /// open, its decode sent `later` or with its prefill.
        fn asks<'a>(&'a self, classes: &'a PartnerClasses, later: bool) -> Asks<'a> {
            let [prefill, decode] = &self.lists;
            let lists = PairLists {
                prefill: prefill.lent(),
                decode: decode.lent(),
                classes,
                admission: &OPEN,
                prefill_places: None,
                decode_places: None,
                prefill_stalled: &[],
                prefill_held: &[],
                decode_stalled: &[],
                decode_held: &[],
            };
            let constraints = Constraints {
                decode_later: later,
                ..Constraints::default()
            };
            let request = Request {
                id: b"",
                facts: keyed(),
                constraints,
            };
            Asks(request, lists, None)
        }

        /// What the stages heard since the last call: each word, its side
        /// and the worker its lease books.
        fn words(&self) -> Vec<(&'static str, Side, WorkerId, u64)> {
            let heard = std::mem::take(&mut *self.heard.lock());
            let word = |word: Word| (word.0, word.1, WorkerId(word.2), word.5);
            heard.into_iter().map(word).collect()
        }

        /// One decision over the lists as they stand now, every pair open,
        /// drawing by `seed`: the pair's first, or the decode's again
        /// behind the `held` prefill, with no decode handed.
        fn decide(
            &self,
            admission: &dyn PairAdmission,
            seed: u64,
            held: Option<WorkerId>,
        ) -> Decision {
            let held = held.map(prefill_alone);
            self.decide_held(admission, seed, held.as_ref())
        }

        /// [`Self::decide`] behind the `held` target: its prefill, and the
        /// decode it names.
        fn decide_held(
            &self,
            admission: &dyn PairAdmission,
            seed: u64,
            held: Option<&Target>,
        ) -> Decision {
            self.decide_excluding(admission, seed, held, &[])
        }

        /// [`Self::decide_held`] for a request whose host names `excluded`,
        /// the workers it just failed to reach.
        fn decide_excluding(
            &self,
            admission: &dyn PairAdmission,
            seed: u64,
            held: Option<&Target>,
            excluded: &[WorkerId],
        ) -> Decision {
            let constraints = Constraints {
                excluded,
                ..Constraints::default()
            };
            self.decide_among(admission, seed, held, constraints, [None, None])
        }

        /// [`Self::decide_held`] for a request that states `constraints`
        /// and that its host holds to `places`: those of the prefill list
        /// and those of the decode list it may take.
        fn decide_among(
            &self,
            admission: &dyn PairAdmission,
            seed: u64,
            held: Option<&Target>,
            constraints: Constraints<'_>,
            places: [Option<&[u32]>; 2],
        ) -> Decision {
            let unmasked = [[&[][..]; 2]; 2];
            self.decide_vetoed(admission, (seed, held), constraints, places, unmasked)
        }

        /// [`Self::decide_among`] for a host whose standing vetoes mask
        /// `vetoes`: for the prefill list and then for the decode list,
        /// the places masked and those of them a hold stands on.
        fn decide_vetoed(
            &self,
            admission: &dyn PairAdmission,
            (seed, held): (u64, Option<&Target>),
            constraints: Constraints<'_>,
            places: [Option<&[u32]>; 2],
            vetoes: [[&[u32]; 2]; 2],
        ) -> Decision {
            let [prefill, decode] = &self.lists;
            let classes = every(self.prefills.len(), self.decodes.len() as u32);
            let [[prefill_stalled, prefill_held], [decode_stalled, decode_held]] = vetoes;
            let lists = PairLists {
                prefill: prefill.lent(),
                decode: decode.lent(),
                classes: &classes,
                admission,
                prefill_places: places[0],
                decode_places: places[1],
                prefill_stalled,
                prefill_held,
                decode_stalled,
                decode_held,
            };
            let mut request = Request {
                id: b"",
                facts: RequestFacts { seed, ..facts() },
                constraints,
            };
            let (engine, now) = (&self.engine, Instant::now());
            let decision = lists.decide(engine, None, now, &mut request, held);
            assert_eq!(request.facts.now_micros, engine.clock().micros(now));
            decision
        }

        /// The places of the pair `decision` picks first, in the prefill
        /// list and in the decode list. Each names its worker's own id.
        fn picked(&self, decision: &Decision) -> (usize, usize) {
            let pick = decision.pick().expect("a pick");
            let first = pick.targets.first().expect("a target");
            let pair = first.pair.expect("a pair");
            let (prefill, decode) = (first.position as usize, pair.decode_position as usize);
            assert_eq!(worker_id(self.prefills[prefill].as_ref()), first.id);
            assert_eq!(worker_id(self.decodes[decode].as_ref()), pair.decode_id);
            (prefill, decode)
        }

        /// [`Self::asks`] drawing by `seed`, with the snapshot of the
        /// workers' reports its host lends for the call.
        fn asks_by<'a>(
            &'a self,
            classes: &'a PartnerClasses,
            (seed, later): (u64, bool),
            loads: Option<Arc<LoadSnapshot>>,
        ) -> Asks<'a> {
            let mut stated = self.asks(classes, later);
            stated.0.facts = RequestFacts { seed, ..facts() };
            stated.2 = loads;
            stated
        }

        /// The places of the pair `stated` is first decided to, by the
        /// engine's entry for a whole decision.
        fn first(&self, stated: &Asks<'_>) -> (usize, usize) {
            let decided = self
                .engine
                .decide_under(GatewayId::LOCAL, stated, Instant::now());
            let (outcome, _) = decided
                .expect("an answer")
                .whole()
                .expect("a whole decision");
            self.picked(&outcome.decision)
        }
    }

    /// A placement says a pair's words as a guard driven by hand says
    /// them, each side's under its own lease and the index of the target
    /// taken. It writes into each end the worker that served, from that
    /// target, whatever its host wrote there. Its two workers are the
    /// lists' own objects at the target's places. An encode side's send
    /// and end say nothing. A pair told such an end behind its two sends
    /// still owes both sides. A pair whose host said those two alone and
    /// let go is released whole, as a guard dropped unsent.
    #[test]
    fn a_placement_says_a_pairs_words_and_names_the_workers_that_served() {
        let host: &'static Host = Box::leak(Box::new(Host::of(1, 2)));
        let classes = every(1, 2);
        let asks = host.asks(&classes, false);
        let whole = || {
            let now = Instant::now();
            let answer = host.engine.decide_under(GatewayId::LOCAL, &asks, now);
            let (decided, later) = answer
                .expect("an answer")
                .whole()
                .expect("a whole decision");
            let taken = target_at(&decided.decision, 1);
            (decided, later, now, taken, taken.pair.expect("a pair"))
        };
        // What the stages heard since the last call: each word with its
        // side and its lease's index, and the worker each completion names.
        let words = || {
            let heard = std::mem::take(&mut *host.heard.lock());
            let told: Vec<_> = heard.iter().map(|word| (word.0, word.1, word.3)).collect();
            let ended = heard.iter().filter(|word| word.0 == "completed");
            let served: Vec<_> = ended.map(|word| (word.1, WorkerId(word.6))).collect();
            (told, served)
        };

        let (decided, later, now, taken, pair) = whole();
        let profile = decided.decision.pick().expect("a pick").trace.profile;
        assert_ne!(profile, 0, "the document's second profile picked");
        let prefill = &host.prefills[taken.position as usize];
        let decode = &host.decodes[pair.decode_position as usize];
        let placement = crate::tracking::Placement::taken(
            host.engine.loaded(),
            decided,
            later,
            now,
            1,
            Arc::clone(prefill),
            Some(Arc::clone(decode)),
        );
        let mut placement = placement.expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, prefill));
        assert!(Arc::ptr_eq(
            placement.decode.as_ref().expect("a decode"),
            decode
        ));
        let ids = (placement.id(), placement.decode_id());
        assert_eq!(ids, (taken.id, Some(pair.decode_id)));
        assert_eq!(placement.profile, "prefill");
        placement.sent(Side::Prefill);
        placement.sent(Side::Decode);
        placement.ended(Side::Encode, outcome(Side::Encode, WorkerId(77)));
        // Its host names no worker and another side in each outcome.
        placement.ended(Side::Prefill, outcome(Side::Single, WorkerId(0)));
        assert!(placement.owes_events(), "the decode side has not ended");
        placement.ended(Side::Decode, outcome(Side::Single, WorkerId(0)));
        assert!(!placement.owes_events());
        drop(placement);
        let (decided, later, now, _, _) = whole();
        let unsent = crate::tracking::Placement::taken(
            host.engine.loaded(),
            decided,
            later,
            now,
            1,
            Arc::clone(prefill),
            Some(Arc::clone(decode)),
        );
        let mut unsent = unsent.expect("a placement");
        unsent.sent(Side::Encode);
        unsent.ended(Side::Encode, outcome(Side::Encode, WorkerId(77)));
        drop(unsent);
        let (through_placements, served) = words();
        let named = [(Side::Prefill, taken.id), (Side::Decode, pair.decode_id)];
        assert_eq!(served, named);

        let (decided, later, now, _, _) = whole();
        let mut guard = DecisionGuard::holding(&host.engine, GatewayId::LOCAL, decided, later, now);
        let taken = guard.take(1).expect("a target");
        let decode = taken.pair.expect("a pair").decode_id;
        guard.dispatched(now);
        guard.dispatched_decode(now);
        guard.completed(&outcome(Side::Prefill, taken.id), now);
        guard.completed(&outcome(Side::Decode, decode), now);
        drop(guard);
        let (decided, later, now, _, _) = whole();
        let mut guard = DecisionGuard::holding(&host.engine, GatewayId::LOCAL, decided, later, now);
        assert!(guard.take(1).is_some());
        drop(guard);
        assert_eq!(through_placements, words().0);
        let told: Vec<_> = through_placements.iter().map(|word| word.0).collect();
        // Each word is told of both sides: the pair sent and ended, then
        // the pair let go.
        let sent_and_ended = ["issued", "dispatched", "completed"].map(|word| [word; 2]);
        let let_go = ["issued", "released"].map(|word| [word; 2]);
        assert_eq!(told, [sent_and_ended.concat(), let_go.concat()].concat());
        assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 0));
    }

    /// A decision names each worker by its place in the list its host
    /// handed and by its id, and reads the workers and the host's admission
    /// as they stand at the call.
    #[test]
    fn a_pair_is_decided_over_the_hosts_two_lists_as_they_stand_at_the_call() {
        let host = Host::of(2, 3);
        let mut seen = [false; 2];
        for seed in 0..16 {
            seen[host.picked(&host.decide(&OPEN, seed, None)).0] = true;
        }
        assert_eq!(seen, [true; 2], "the rotation reaches both prefills");
        // A prefill that stopped being ready and a decode with no room: the
        // next decision reads both where the host keeps them.
        host.prefills[0].set_status(WorkerStatus::NotReady);
        let one_room = Gates(|_| true, |at| at == 2);
        assert_eq!(host.picked(&host.decide(&one_room, 1, None)), (1, 2));
        // The one prefill left is at its limit: the request waits.
        let full = Gates(|_| false, |_| true);
        assert!(prefills_full(&host.decide(&full, 1, None)));
    }

    /// A pair its host holds to places is decided over the pools' own lists,
    /// and each side opens with its own places: no prefill and no decode
    /// outside them is picked, under the stock document and under one whose
    /// profiles list no filter. One side may be held alone. A side with no
    /// place kept picks no pair. The decode asked again is held to the
    /// decodes' places of its own call.
    #[test]
    fn a_pair_held_to_places_is_handed_no_other_place_on_either_side() {
        for host in [Host::of(3, 4), Host::unfiltered(3, 4)] {
            // Both sides held: every pick is of the places, and the turns
            // reach each of them.
            let mut seen = ([false; 3], [false; 4]);
            for seed in 0..64 {
                let both = [Some(&[1][..]), Some(&[0, 3][..])];
                let picked = host.picked(&host.decide_among(
                    &OPEN,
                    seed,
                    None,
                    Constraints::default(),
                    both,
                ));
                seen.0[picked.0] = true;
                seen.1[picked.1] = true;
            }
            assert_eq!(seen, ([false, true, false], [true, false, false, true]));
            for seed in 0..32 {
                let prefills = [Some(&[0, 2][..]), None];
                let picked = host.picked(&host.decide_among(
                    &OPEN,
                    seed,
                    None,
                    Constraints::default(),
                    prefills,
                ));
                assert_ne!(picked.0, 1, "seed {seed}");
                let decodes = [None, Some(&[2][..])];
                let picked = host.picked(&host.decide_among(
                    &OPEN,
                    seed,
                    None,
                    Constraints::default(),
                    decodes,
                ));
                assert_eq!(picked.1, 2, "seed {seed}");
            }
            // No decode kept: no pair is picked, whatever is ready.
            let none = [None, Some(&[][..])];
            assert!(host
                .decide_among(&OPEN, 1, None, Constraints::default(), none)
                .pick()
                .is_none());
            // The decode asked again behind a sent prefill.
            let (sent, _) = host.picked(&host.decide(&OPEN, 3, None));
            let held = prefill_alone(worker_id(host.prefills[sent].as_ref()));
            for seed in 0..16 {
                let decodes = [None, Some(&[1][..])];
                let again =
                    host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), decodes);
                assert_eq!(host.picked(&again), (sent, 1), "seed {seed}");
            }
        }
    }

    /// A decode among the decode places is handed only with the prefill it
    /// partners. Two prefills, each with its own decode, both decode places
    /// stated and the prefill list held to the second: the pair is the
    /// second prefill with the second decode at every turn. So it is where
    /// a place of either side stands past its list's end, which names no
    /// row. With the prefill list free and the first decode not ready, the
    /// second decode place alone opens a prefill: the same pair. The
    /// gateway's classic router picks a pair's decode among the picked
    /// prefill's own partners that are open for the request
    /// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
    /// `:1389-1394`, `:1548-1553`, at main `b8c15d40a`).
    #[test]
    fn a_decode_among_the_places_is_handed_only_with_the_prefill_it_partners() {
        let (spec, _) = hearing();
        let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
        let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
        let free = Constraints::default();
        let decide = |seed: u64, first_ready: bool, places: [Option<&[u32]>; 2]| {
            let (mut prefills, mut decodes) = lists(2, 2);
            stamp(&mut prefills, &mut decodes, &classes, &OPEN);
            decodes[0].health.healthy = first_ready;
            let facts = RequestFacts { seed, ..facts() };
            decide_records(
                spec.engine(),
                &facts,
                (&free, places.map(unvetoed)),
                &pool(&prefills),
                pools(&prefills, &decodes, &classes),
                &mut Scratch::default(),
            )
        };
        for seed in 0..24 {
            let held = [Some(&[1][..]), Some(&[0, 1][..])];
            assert_eq!(picked(&decide(seed, true, held)), (2, 102), "seed {seed}");
            let past = [Some(&[1, 7][..]), Some(&[0, 1, 9][..])];
            assert_eq!(picked(&decide(seed, true, past)), (2, 102), "seed {seed}");
            let second = [None, Some(&[0, 1][..])];
            assert_eq!(
                picked(&decide(seed, false, second)),
                (2, 102),
                "seed {seed}"
            );
        }
    }

    /// The decode asked again is held to the decode places that partner its
    /// held prefill. Two prefills, each with its own decode, and the second
    /// prefill held, which is not the first of its list: among both decode
    /// places, and at the second alone, its decode is the second at every
    /// turn. At the first place alone the places and the prefill's partners
    /// meet in no decode, and the call is shed under the partners' name. So
    /// it is over no place at all. The gateway's classic router asks no
    /// decode again: it selects a request's pair whole, once, the decode
    /// among the picked prefill's own partners
    /// (`model_gateway/src/routers/common/placement.rs:1227-1275`,
    /// `:1548-1553`, at main `b8c15d40a`).
    #[test]
    fn the_decode_asked_again_is_held_to_the_places_that_partner_its_prefill() {
        let pair = decided(2);
        let (mut prefills, mut decodes) = lists(2, 2);
        let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
        let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let free = Constraints::default();
        let again = |seed: u64, places: &[u32]| {
            let facts = RequestFacts { seed, ..facts() };
            decide_decode_records(
                pair.spec.engine(),
                &facts,
                (&free, unvetoed(Some(places))),
                &pool(&decodes),
                pools(&prefills, &decodes, &classes),
                &prefill_alone(WorkerId(2)),
                &mut Scratch::default(),
            )
        };
        for seed in 0..24 {
            assert_eq!(picked(&again(seed, &[0, 1])), (2, 102), "seed {seed}");
            assert_eq!(picked(&again(seed, &[1])), (2, 102), "seed {seed}");
        }
        let nobody = (ShedReason::NoEndpoints, "partners");
        assert_eq!(shed_for(&again(1, &[0])), nobody);
        assert_eq!(shed_for(&again(1, &[])), nobody);
    }

    /// A side's places stand beside what the request states, and hold the
    /// other side to nothing. Three prefills and four decodes, every pair
    /// open. With the prefill list held to its second place the turns reach
    /// every decode, and with the decode list held to its fourth they reach
    /// every prefill. A decode its host just failed to reach steps aside
    /// among the decode places while another of them is left. A prefill
    /// list held to no place picks no pair, as a pair with no prefill picks
    /// none. Behind the third prefill the
    /// decode asked again reads the decodes' places alone: the prefills'
    /// places, which leave that prefill out, are not read, and decode
    /// places that keep none pick none. The gateway's classic router steps
    /// aside from the engines a request tried on each side of a pair, among
    /// the workers its request's rule admits
    /// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
    /// `:1492-1494`, `:1568-1570`, at main `b8c15d40a`).
    #[test]
    fn a_sides_places_stand_beside_the_requests_word_and_hold_the_other_side_to_nothing() {
        let host = Host::of(3, 4);
        let free = Constraints::default;
        let mut decodes = [false; 4];
        for seed in 0..64 {
            let prefill = [Some(&[1][..]), None];
            let picked = host.picked(&host.decide_among(&OPEN, seed, None, free(), prefill));
            assert_eq!(picked.0, 1, "seed {seed}");
            decodes[picked.1] = true;
        }
        assert_eq!(decodes, [true; 4]);
        let mut prefills = [false; 3];
        for seed in 0..64 {
            let decode = [None, Some(&[3][..])];
            let picked = host.picked(&host.decide_among(&OPEN, seed, None, free(), decode));
            assert_eq!(picked.1, 3, "seed {seed}");
            prefills[picked.0] = true;
        }
        assert_eq!(prefills, [true; 3]);
        let tried = [worker_id(host.decodes[0].as_ref())];
        for seed in 0..16 {
            let excluding = Constraints {
                excluded: &tried,
                ..Constraints::default()
            };
            let among = [None, Some(&[0, 3][..])];
            let picked = host.picked(&host.decide_among(&OPEN, seed, None, excluding, among));
            assert_eq!(picked.1, 3, "seed {seed}");
        }
        let none = [Some(&[][..]), None];
        let unheld = host.decide_among(&OPEN, 1, None, free(), none);
        assert_eq!(shed_for(&unheld), (ShedReason::NoEndpoints, "none"));
        let held = prefill_alone(worker_id(host.prefills[2].as_ref()));
        for seed in 0..16 {
            let other = [Some(&[0][..]), Some(&[1][..])];
            let again = host.decide_among(&OPEN, seed, Some(&held), free(), other);
            assert_eq!(host.picked(&again), (2, 1), "seed {seed}");
        }
        let none = [None, Some(&[][..])];
        let again = host.decide_among(&OPEN, 1, Some(&held), free(), none);
        assert_eq!(shed_for(&again), (ShedReason::NoEndpoints, "partners"));
    }

    /// The decode asked again keeps the decode its first decision named
    /// while that decode stands among the decodes' places of the call, and
    /// picks among the places where it does not: no decode outside them is
    /// handed, kept or not. One prefill and three decodes under round
    /// robin, which keeps its pick. The gateway's classic router asks no
    /// decode again: it selects a request's pair whole, once
    /// (`model_gateway/src/routers/common/placement.rs:1227-1275`, at main
    /// `b8c15d40a`).
    #[test]
    fn the_decode_asked_again_keeps_its_first_decode_among_the_places() {
        let host = Host::under(presets("round_robin"), 1, 3);
        for seed in 0..12 {
            let held = first_target(&host.decide(&OPEN, seed, None));
            let named = held.pair.expect("a decode").decode_position;
            let other = (named + 1) % 3;
            // A host states its places ascending.
            let (with, without) = ([named.min(other), named.max(other)], [other]);
            let among = [None, Some(&with[..])];
            let again = host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), among);
            assert_eq!(host.picked(&again).1, named as usize, "seed {seed}");
            let outside = [None, Some(&without[..])];
            let again =
                host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), outside);
            assert_eq!(host.picked(&again).1, other as usize, "seed {seed}");
        }
    }

    /// The decode asked again reads both lists at its own call, and keeps
    /// the prefill that was sent.
    #[test]
    fn the_decode_asked_again_reads_the_hosts_lists_of_its_own_instant() {
        let host = Host::of(2, 2);
        let (sent, chosen) = host.picked(&host.decide(&OPEN, 3, None));
        let held = worker_id(host.prefills[sent].as_ref());
        // The decode the first pick named stops being ready behind it.
        host.decodes[chosen].set_status(WorkerStatus::NotReady);
        let again = host.decide(&OPEN, 4, Some(held));
        assert_eq!(host.picked(&again), (sent, 1 - chosen));
        // A prefill no list holds any more sheds the call.
        match host.decide(&OPEN, 5, Some(WorkerId(held.0 ^ 1))) {
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "prefill-gone"),
            other => panic!("not shed: {other:?}"),
        }
    }

    /// The decode asked again reads one prefill, the held one: its host's
    /// prefill limit is asked of that prefill's place alone.
    #[test]
    fn the_decode_asked_again_asks_the_limit_of_its_held_prefill_alone() {
        /// A host with no limit that keeps the places its prefill limit
        /// was asked of.
        #[derive(Debug, Default)]
        struct Asked(RefCell<Vec<usize>>);

        impl PairAdmission for Asked {
            fn has_capacity(&self, prefill: usize) -> bool {
                self.0.borrow_mut().push(prefill);
                true
            }

            fn has_room(&self, _: usize) -> bool {
                true
            }
        }

        let (host, asked) = (Host::of(3, 2), Asked::default());
        let held = worker_id(host.prefills[1].as_ref());
        let again = host.decide(&asked, 4, Some(held));
        assert_eq!(host.picked(&again).0, 1);
        assert_eq!(asked.0.take(), [1]);
        // The thread's buffer holds the one prefill record written.
        let written = BUFFERS.with(|buffers| buffers.borrow().rows[0].len());
        assert_eq!(written, 1);
    }

    /// A decode asked again where its host lists no decode is refused for
    /// the partners it lacks: the held prefill stands in its list. So it is
    /// behind a target that names the decode of its first decision. A
    /// prefill that left is still named where a list holds no record.
    ///
    /// The classic router places both sides of a pair in one call
    /// (`placement.rs:1227`) and decides no decode again, so this answer
    /// is the engine's own.
    #[test]
    fn a_decode_asked_again_over_no_decode_is_not_shed_as_a_prefill_that_left() {
        let shed_under = |host: &Host, held: &Target| match host.decide_held(&OPEN, 4, Some(held)) {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        };
        let host = Host::of(1, 0);
        let held = worker_id(host.prefills[0].as_ref());
        let no_partner = (ShedReason::NoEndpoints, "partners");
        assert_eq!(shed_under(&host, &prefill_alone(held)), no_partner);
        let first = first_target(&Host::of(1, 1).decide(&OPEN, 3, None));
        assert_eq!((first.id, first.pair.is_some()), (held, true));
        assert_eq!(shed_under(&host, &first), no_partner);
        let gone = (ShedReason::NoEndpoints, "prefill-gone");
        let left = prefill_alone(WorkerId(held.0 ^ 1));
        assert_eq!(shed_under(&host, &left), gone);
        assert_eq!(shed_under(&Host::of(0, 1), &prefill_alone(held)), gone);
    }

    /// Over no decode, as over two lists with records, a request that
    /// states a demand is shed under the demand's stage. Over no prefill
    /// the call is shed for the prefill that left, before a demand is read.
    /// The classic router decides no decode again (`placement.rs:1227`):
    /// this answer is the engine's own too.
    #[test]
    fn a_decode_asked_again_over_no_decode_sheds_a_stated_demand_under_its_stage() {
        let pair = decided(1);
        let demanded = Constraints {
            demands: Demands {
                vision: true,
                ..Demands::default()
            },
            ..Constraints::default()
        };
        let shed_under = |prefills: u8, decodes: u8| {
            let classes = every(usize::from(prefills), u32::from(decodes));
            let (mut prefills, mut decodes) = lists(u64::from(prefills), u64::from(decodes));
            stamp(&mut prefills, &mut decodes, &classes, &OPEN);
            let again = decide_decode_records(
                pair.spec.engine(),
                &facts(),
                (&demanded, unvetoed(None)),
                &pool(&decodes),
                pools(&prefills, &decodes, &classes),
                &prefill_alone(WorkerId(1)),
                &mut Scratch::default(),
            );
            match again {
                Decision::Shed(shed) => (shed.reason, shed.emptied_by),
                other => panic!("not shed: {other:?}"),
            }
        };
        assert_eq!(shed_under(1, 0), (ShedReason::NoEndpoints, WIRES));
        assert_eq!(shed_under(0, 1), (ShedReason::NoEndpoints, "prefill-gone"));
    }

    /// [`Host::of`] whose host says what each worker states for a request's
    /// demands: `stated` answers for a side, the prefills' `0`, and a place
    /// of that side's list.
    fn stating(
        prefills: usize,
        decodes: usize,
        stated: impl Fn(usize, usize) -> FixedFacts,
    ) -> Host {
        let mut host = Host::of(prefills, decodes);
        let said = |side: usize, list: &Arc<[Arc<dyn Worker>]>| {
            let mut places = 0..;
            PublishedList::with_facts(Arc::clone(list), |_| {
                stated(side, places.next().expect("a place"))
            })
        };
        host.lists = [said(0, &host.prefills), said(1, &host.decodes)];
        host
    }

    /// A pair's constraints that state `demands` and nothing else.
    fn demanding(demands: Demands) -> Constraints<'static> {
        Constraints {
            demands,
            ..Constraints::default()
        }
    }

    /// The places of each list that twelve first decisions of `host` name
    /// for a request that states `cx`, each list held to `places`.
    fn places_named(
        host: &Host,
        cx: Constraints<'_>,
        places: [Option<&[u32]>; 2],
    ) -> (BTreeSet<usize>, BTreeSet<usize>) {
        let picks = (0..12).map(|seed| {
            let decision = host.decide_among(&OPEN, seed, None, cx, places);
            host.picked(&decision)
        });
        picks.unzip()
    }

    /// What a decision that sheds is shed for, and under which stage.
    fn shed_of(decision: &Decision) -> (ShedReason, &'static str) {
        match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        }
    }

    /// A pair that states a hard demand is decided over its two lists,
    /// each side held to the workers that meet the demand: every pick
    /// names one of them on each side, and each of them takes its turn.
    /// With no demand the other workers take theirs.
    ///
    /// The classic router holds both of a pair's pools to the request's
    /// own rule (`model_gateway/src/routers/common/placement.rs:1227`, the
    /// rule's read at `:1262-1266`; its cases at `:3272` and `:3324`, at
    /// main `b8c15d40a`).
    #[test]
    fn a_pair_that_states_a_demand_is_decided_among_the_workers_of_each_side_that_meet_it() {
        // The second prefill takes images, and so do the first and the
        // third decode.
        let host = stating(2, 3, |side, at| FixedFacts {
            vision: if side == 0 { at == 1 } else { at != 1 },
            ..FixedFacts::default()
        });
        let pixels = demanding(Demands {
            vision: true,
            ..Demands::default()
        });
        let seeing = (BTreeSet::from([1]), BTreeSet::from([0, 2]));
        assert_eq!(places_named(&host, pixels, [None, None]), seeing);
        let all = (BTreeSet::from([0, 1]), BTreeSet::from([0, 1, 2]));
        assert_eq!(
            places_named(&host, Constraints::default(), [None, None]),
            all
        );
    }

    /// A prefill is held to neither the completion's bound nor the way a
    /// constraint was decided ([`Demands::held_by`]): it runs the input,
    /// and the decode beside it holds the budget and generates the tokens.
    /// So a prefill whose window is under the completion's bound stays a
    /// candidate while a decode under it leaves, and the input's bound
    /// holds both sides.
    #[test]
    fn a_pairs_prefill_is_held_to_neither_the_completions_bound_nor_a_constraints_way() {
        // Every worker states a window of 4,096 but the second decode,
        // which states 16,384. The first decode alone states a way to
        // decide a constraint.
        let host = stating(2, 2, |side, at| FixedFacts {
            demands: DemandFacts {
                window: if (side, at) == (1, 1) { 16_384 } else { 4_096 },
                decoding: if (side, at) == (1, 0) { 3 } else { 0 },
                ..DemandFacts::default()
            },
            ..FixedFacts::default()
        });
        let both = BTreeSet::from([0, 1]);
        let long = demanding(Demands {
            output_window: 8_000,
            ..Demands::default()
        });
        let wide = (both.clone(), BTreeSet::from([1]));
        assert_eq!(places_named(&host, long, [None, None]), wide);
        let decided = demanding(Demands {
            decoding: 3,
            ..Demands::default()
        });
        let alike = (both, BTreeSet::from([0]));
        assert_eq!(places_named(&host, decided, [None, None]), alike);
        // The input's bound is every worker's: no prefill holds it.
        let prompt = demanding(Demands {
            input_window: 8_000,
            ..Demands::default()
        });
        let decision = host.decide_among(&OPEN, 0, None, prompt, [None, None]);
        let narrow = (ShedReason::NoEndpoints, "context-window");
        assert_eq!(shed_of(&decision), narrow);
    }

    /// A side the demand leaves nobody sheds the pair with no endpoint
    /// under the stage of the demand that took that side's last worker, as
    /// a single request's list is shed. Where each side is left nobody, the
    /// prefills' demand is the one named.
    #[test]
    fn a_side_a_demand_leaves_nobody_sheds_the_pair_under_that_demands_stage() {
        // No prefill takes images. Every decode does, and each states a
        // window of 4,096.
        let host = stating(2, 2, |side, _| FixedFacts {
            vision: side == 1,
            demands: DemandFacts {
                window: if side == 1 { 4_096 } else { 0 },
                ..DemandFacts::default()
            },
            ..FixedFacts::default()
        });
        let shed_under = |demands: Demands| {
            let decision = host.decide_among(&OPEN, 0, None, demanding(demands), [None, None]);
            shed_of(&decision)
        };
        let pixels = Demands {
            vision: true,
            ..Demands::default()
        };
        assert_eq!(shed_under(pixels), (ShedReason::NoEndpoints, "vision"));
        let long = Demands {
            output_window: 8_000,
            ..Demands::default()
        };
        let narrow = (ShedReason::NoEndpoints, "context-window");
        assert_eq!(shed_under(long), narrow);
        // Both sides are left nobody, each by another demand.
        let each = Demands {
            vision: true,
            ..long
        };
        assert_eq!(shed_under(each), (ShedReason::NoEndpoints, "vision"));
    }

    /// A demand is judged among the places a pair's host states for a
    /// side: the side is held to those of them whose worker meets it, and
    /// a side none of whose places meets it sheds the pair, whatever a
    /// worker outside them states.
    #[test]
    fn a_pairs_demand_is_judged_among_the_places_its_host_states() {
        // The second and the third worker of each list take images.
        let host = stating(3, 3, |_, at| FixedFacts {
            vision: at != 0,
            ..FixedFacts::default()
        });
        let pixels = demanding(Demands {
            vision: true,
            ..Demands::default()
        });
        let held = places_named(&host, pixels, [Some(&[0, 1]), Some(&[1, 2])]);
        assert_eq!(held, (BTreeSet::from([1]), BTreeSet::from([1, 2])));
        let blind = host.decide_among(&OPEN, 0, None, pixels, [None, Some(&[0])]);
        assert_eq!(shed_of(&blind), (ShedReason::NoEndpoints, "vision"));
    }

    /// A pair's request that states both a hard demand and places of its
    /// own is declined under the places' name, whether its sides meet the
    /// demand or not: a pair's places are its lists' to state, and that
    /// answer comes ahead of the demand's.
    #[test]
    fn a_pair_that_states_a_demand_and_places_of_its_own_is_declined_under_the_places() {
        let pixels = demanding(Demands {
            vision: true,
            ..Demands::default()
        });
        let stated = Constraints {
            places: Some(&[0]),
            ..pixels
        };
        let declined = (ShedReason::PolicyDeclined, single::PLACES);
        // Every worker takes images, then none does.
        for sees in [true, false] {
            let host = stating(1, 1, |_, _| FixedFacts {
                vision: sees,
                ..FixedFacts::default()
            });
            let decision = host.decide_among(&OPEN, 0, None, stated, [None, None]);
            assert_eq!(shed_of(&decision), declined, "takes images: {sees}");
        }
    }

    /// The decode asked again behind a held prefill is held to the decodes
    /// that meet the demand, and judges no prefill: the held one ran
    /// already, whatever it states. Where no decode meets the demand the
    /// call is shed under that demand's stage.
    ///
    /// The classic router decides no decode again (`placement.rs:1227`, at
    /// main `b8c15d40a`): this answer is the engine's own.
    #[test]
    fn a_decode_asked_again_is_held_to_the_decodes_that_meet_the_demand() {
        let pixels = demanding(Demands {
            vision: true,
            ..Demands::default()
        });
        // No prefill takes images, and the second decode alone does.
        let host = stating(1, 2, |side, at| FixedFacts {
            vision: (side, at) == (1, 1),
            ..FixedFacts::default()
        });
        let held = prefill_alone(worker_id(host.prefills[0].as_ref()));
        for seed in 0..6 {
            let again = host.decide_among(&OPEN, seed, Some(&held), pixels, [None, None]);
            assert_eq!(host.picked(&again), (0, 1));
        }
        // The pair's first decision judges its prefills.
        let first = host.decide_among(&OPEN, 0, None, pixels, [None, None]);
        assert_eq!(shed_of(&first), (ShedReason::NoEndpoints, "vision"));
        // No decode takes images.
        let blind = stating(1, 2, |_, _| FixedFacts::default());
        let held = prefill_alone(worker_id(blind.prefills[0].as_ref()));
        let again = blind.decide_among(&OPEN, 0, Some(&held), pixels, [None, None]);
        assert_eq!(shed_of(&again), (ShedReason::NoEndpoints, "vision"));
    }

    /// Under round robin on both sides a pair's first decision takes the
    /// decode rotation's turn, and the decode decided again is handed the
    /// decode that decision named and keeps it. So requests that come one
    /// at a time reach both decodes evenly, and each second answer is its
    /// first. Handed the prefill alone, every second answer was the other
    /// decode. A first decode that stopped being ready is left for another.
    #[test]
    fn the_decode_asked_again_keeps_its_first_decode_under_round_robin() {
        let host = Host::under(presets("round_robin"), 1, 2);
        let mut sent = [0; 2];
        for seed in 0..8 {
            let held = first_target(&host.decide(&OPEN, seed, None));
            let named = held.pair.expect("a decode").decode_position as usize;
            let again = host.decide_held(&OPEN, seed, Some(&held));
            assert_eq!(host.picked(&again).1, named, "seed {seed}");
            sent[named] += 1;
        }
        assert_eq!(sent, [4; 2]);
        let held = first_target(&host.decide(&OPEN, 8, None));
        let named = held.pair.expect("a decode").decode_position as usize;
        let alone = host.decide(&OPEN, 8, Some(held.id));
        assert_eq!(host.picked(&alone).1, 1 - named, "no decode handed");
        let held = first_target(&host.decide(&OPEN, 9, None));
        let named = held.pair.expect("a decode").decode_position as usize;
        host.decodes[named].set_status(WorkerStatus::NotReady);
        let again = host.decide_held(&OPEN, 9, Some(&held));
        assert_eq!(host.picked(&again).1, 1 - named);
    }

    /// The decode asked again through the guard is handed the decode of the
    /// target the guard holds. Under round robin each second answer is its
    /// first, and requests one at a time reach both decodes evenly.
    #[test]
    fn the_guards_decode_call_keeps_its_first_decode() {
        let host = Host::under(presets("round_robin"), 1, 2);
        let classes = every(1, 2);
        let mut sent = [0; 2];
        for _ in 0..4 {
            let (asks, now) = (host.asks(&classes, true), Instant::now());
            let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
            let mut guard = guard.expect("a decision");
            let named = host.picked(guard.decision()).1;
            guard.dispatched(now);
            let again = *guard.decide_decode_under(&asks, now).expect("a decision");
            assert!(!guard.decode_chosen(), "the call picked");
            assert_eq!(host.picked(&again).1, named);
            sent[named] += 1;
        }
        assert_eq!(sent, [2; 2]);
    }

    /// A host that takes another target of its pick holds that target, and
    /// the guard hands it to the decode asked again. Under round robin the
    /// answer's first decode is then the taken target's: for a target taken
    /// before its prefill went out, and for a target's decode taken behind
    /// the prefill. The pick's third target is taken: its decode is neither
    /// the first target's nor the one the rotation's next turn names.
    #[test]
    fn the_guards_decode_call_keeps_the_decode_its_host_took() {
        let host = Host::under(presets("round_robin"), 1, 3);
        let classes = every(1, 3);
        for behind_the_prefill in [false, true] {
            let (asks, now) = (host.asks(&classes, true), Instant::now());
            let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
            let mut guard = guard.expect("a decision");
            let first = host.picked(guard.decision()).1;
            let taken = if behind_the_prefill {
                guard.dispatched(now);
                guard.take_decode(2)
            } else {
                let taken = guard.take(2);
                guard.dispatched(now);
                taken
            };
            let taken = taken.and_then(|target| target.pair);
            let taken = taken.expect("a third target").decode_position as usize;
            assert_ne!(taken, first, "another decode than the first target's");
            let again = *guard.decide_decode_under(&asks, now).expect("a decision");
            assert_eq!(host.picked(&again).1, taken, "{behind_the_prefill}");
        }
    }

    /// A decode that is one rank of a data-parallel engine is kept by its
    /// worker and its rank. Over an engine's two ranks each second answer is
    /// the rank its first named, and both ranks are named.
    #[test]
    fn a_ranked_decode_is_kept_by_its_rank() {
        let rank = |rank: usize| -> Arc<dyn Worker> {
            let spec = WorkerSpec {
                worker_type: WorkerType::Decode,
                ..WorkerSpec::new("grpc://decode:9000".to_string())
            };
            let worker = <BasicWorkerBuilder>::from_spec(spec)
                .dp_config(rank, 2)
                .health_config(no_health_check())
                .build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker)
        };
        let prefill = worker("grpc://prefill:9000".to_string(), WorkerType::Prefill);
        let ranks = Arc::from(vec![rank(0), rank(1)]);
        let host = Host::over(presets("round_robin"), Arc::from(vec![prefill]), ranks);
        let mut named_ranks = [0; 2];
        for seed in 0..4 {
            let held = first_target(&host.decide(&OPEN, seed, None));
            let named = held.pair.expect("a decode");
            let again = first_target(&host.decide_held(&OPEN, seed, Some(&held)));
            assert_eq!(again.pair, Some(named), "seed {seed}");
            named_ranks[usize::from(named.decode_dp_rank)] += 1;
        }
        assert_eq!(named_ranks, [2; 2], "both ranks are named");
    }

    /// The presets' own filters stand ahead of a kept decode. A first decode
    /// the request excludes is left for another decode while one is kept,
    /// and is handed again where it is the only decode.
    #[test]
    fn an_excluded_first_decode_is_kept_only_where_it_is_alone() {
        for (decodes, kept) in [(2, false), (1, true)] {
            let host = Host::under(presets("round_robin"), 1, decodes);
            let held = first_target(&host.decide(&OPEN, 0, None));
            let first = held.pair.expect("a decode");
            let excluded = [first.decode_id];
            let again = host.decide_excluding(&OPEN, 0, Some(&held), &excluded);
            let named = first.decode_position as usize;
            assert_eq!(host.picked(&again).1 == named, kept, "{decodes}");
        }
    }

    /// `passthrough` takes no turn, so its decode asked again is decided
    /// again, whatever the call is handed. A pair's first decision names
    /// the first decode of the list that has a room. A decode ahead of that
    /// one has its room by the second call, and the answer is that decode.
    #[test]
    fn the_decode_asked_again_under_passthrough_is_the_lists_first() {
        let host = Host::under(presets("passthrough"), 1, 3);
        let no_room_ahead = Gates(|_| true, |at| at != 0);
        let held = first_target(&host.decide(&no_room_ahead, 0, None));
        assert_eq!(held.pair.expect("a decode").decode_position, 1);
        let again = host.decide_held(&OPEN, 0, Some(&held));
        assert_eq!(host.picked(&again).1, 0);
        let alone = host.decide(&OPEN, 0, Some(held.id));
        assert_eq!(host.picked(&alone).1, 0, "no decode handed");
    }

    /// Both sides of a pair take the hash ring their caller lends, each
    /// over its own list. The gateway hands a pair its model's one ring
    /// where either side's policy reads it
    /// (`model_gateway/src/routers/common/placement.rs`, lines 1574 to 1585
    /// at 882be6f4f), and each side's policy places by its two rules
    /// (`model_gateway/src/policies/consistent_hashing.rs`, lines 99 to
    /// 140). A side takes the ring's worker only where its own list holds
    /// an eligible one: the walk passes the other side's workers, and a
    /// worker of its own that is not ready. A value no worker of the side
    /// takes goes by its position over that side's eligible workers.
    #[test]
    fn each_side_of_a_pair_places_on_the_ring_its_caller_lends() {
        let document = PolicySpec::pair_presets("consistent_hashing", "consistent_hashing");
        let spec = AppliedSpec::validate(&document.expect("the pair presets"));
        let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
        let lent = |workers: &str| {
            let address = |at: &str| format!("grpc://{at}:9000");
            InOrder(workers.split(' ').map(address).collect())
        };
        // The pair a keyed request is placed on where it stands at
        // `position` of the ring lent.
        let placed = |ring: &InOrder, position: u64| {
            let [prefill, decode] = &host.lists;
            let classes = every(3, 3);
            let lists = PairLists {
                prefill: prefill.lent(),
                decode: decode.lent(),
                classes: &classes,
                admission: &OPEN,
                prefill_places: None,
                decode_places: None,
                prefill_stalled: &[],
                prefill_held: &[],
                decode_stalled: &[],
                decode_held: &[],
            };
            let positions = [position];
            let facts = RequestFacts {
                ring: Some(ring),
                ring_positions: &positions,
                ..keyed()
            };
            let mut request = Request {
                id: b"",
                facts,
                constraints: Constraints::default(),
            };
            let now = Instant::now();
            host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
        };
        // The model's one ring holds both lists' workers: each side takes
        // the first worker of its own list that the walk reaches.
        let model = lent("decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2");
        assert_eq!(placed(&model, 0), (2, 1));
        assert_eq!(placed(&model, 3), (0, 2));
        // A worker that is not ready is passed over on its side.
        host.prefills[2].set_status(WorkerStatus::NotReady);
        host.decodes[1].set_status(WorkerStatus::NotReady);
        assert_eq!(placed(&model, 0), (0, 0));
        // A ring that reaches no eligible worker of a side: the value goes
        // by its position over that side's eligible workers, two here.
        let no_prefill = lent("decode-2 decode-0 prefill-2");
        assert_eq!(placed(&no_prefill, 3), (1, 2));
        let no_decode = lent("prefill-1 decode-1");
        assert_eq!(placed(&no_decode, 4), (1, 0));
    }

    /// Under the prefix form each side of a pair is placed on the lent ring
    /// too, from the first place its caller states whose owner is within
    /// the bound: the first eligible worker of its own list that the walk
    /// reaches there, as the gateway's own policy takes it from the pair's
    /// one ring (`model_gateway/src/policies/prefix_hash.rs`, lines 291 to
    /// 315 at 882be6f4f). Every worker here is idle: the first place is
    /// the one taken.
    #[test]
    fn each_side_of_a_pair_under_the_prefix_form_places_on_the_lent_ring() {
        let document = PolicySpec::pair_presets("prefix_hash", "prefix_hash");
        let spec = AppliedSpec::validate(&document.expect("the pair presets"));
        let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
        let address = |at: &str| format!("grpc://{at}:9000");
        let model = "decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2";
        let ring = InOrder(model.split(' ').map(address).collect());
        let placed = |position: u64| {
            let classes = every(3, 3);
            let positions = [position];
            // The host's own request over its two lists, every pair open.
            let Asks(mut request, lists, _) = host.asks(&classes, false);
            request.facts = RequestFacts {
                ring: Some(&ring),
                ring_positions: &positions,
                ..keyed()
            };
            let now = Instant::now();
            host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
        };
        assert_eq!(placed(0), (2, 1));
        assert_eq!(placed(3), (0, 2));
    }

    /// The stock presets of a pair as a document in force, round robin for
    /// its prefills and `decode` for its decodes, with the list its stages'
    /// words would be kept in.
    fn presets(decode: &str) -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
        let document = PolicySpec::pair_presets("round_robin", decode);
        let mut document = document.expect("the pair presets");
        document.generation = 3;
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        (spec, Arc::default())
    }

    /// The first target a decision names.
    fn first_target(decision: &Decision) -> Target {
        let first = decision.pick().and_then(|pick| pick.targets.first());
        *first.expect("a target")
    }

    fn dispatched(side: Side, index: u8) -> Said {
        let dispatch = Dispatch {
            target_index: index,
            side,
            since_decision_micros: 12,
        };
        Said::Dispatched(dispatch, Instant::now())
    }

    fn outcome(side: Side, served: WorkerId) -> Outcome {
        Outcome {
            side,
            served,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        }
    }

    fn completed(side: Side, served: u64) -> Said {
        Said::Completed(outcome(side, WorkerId(served)))
    }

    /// The pick's target at `index`.
    fn target_at(decision: &Decision, index: usize) -> Target {
        let pick = decision.pick().expect("a pick");
        *pick.targets.as_slice().get(index).expect("a target")
    }

    /// A pick of one prefill (worker 1) with two decodes (workers 101 and
    /// 102): the first target and an alternate, as the pair entry orders
    /// them.
    fn two_decodes() -> Decision {
        let mut targets = Targets::new();
        for (at, decode) in [(0, 101), (1, 102)] {
            targets.push(Target {
                position: 0,
                id: WorkerId(1),
                dp_rank: 0,
                pair: Some(Pair {
                    decode_position: at,
                    decode_id: WorkerId(decode),
                    ..Pair::default()
                }),
            });
        }
        Decision::Pick(Pick {
            targets,
            ..Pick::default()
        })
    }

    /// A pair's first pick issued as a first decision issues it
    /// ([`first_sides`], [`tell_first`]), with no engine above the stages
    /// and so no catch.
    fn issue(
        engine: &Engine,
        facts: &RequestFacts<'_>,
        decision: &Decision,
        id: LeaseId,
        later: bool,
    ) -> Option<(Lease, Session)> {
        let (lease, decode, session) = first_sides(facts, decision, id)?;
        tell_first(engine, &lease, &decode, decision, later);
        Some((lease, session))
    }

    /// One word of a held pair told to the stages, the clock reading 9,900
    /// and every worker still listed.
    fn tell(engine: &Engine, held: &HeldDecision<'_>, word: Said) {
        lease_events::pair(engine, held, word, |_| 9_900);
    }

    /// Each word of a pair, both sides issued with the prefill, the host
    /// taking the alternate decode. A side's dispatch names the worker it
    /// booked at its issue and the worker it goes to, and its end names
    /// the worker it went to.
    #[test]
    fn each_word_of_a_pair_is_told_under_its_sides_lease() {
        let pair = decided(1);
        let engine = pair.spec.engine();
        let decision = two_decodes();
        let issued = issue(engine, &keyed(), &decision, LeaseId(9), false);
        let (lease, session) = issued.expect("a pair's lease");
        pair.words();
        let own = lease.session;
        let taken = target_at(&decision, 1);
        let held = HeldDecision {
            decision: &decision,
            lease: &lease,
            index: 1,
            target: &taken,
            decode: HeldDecode {
                session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: false,
            },
        };
        tell(engine, &held, dispatched(Side::Prefill, 1));
        tell(engine, &held, dispatched(Side::Decode, 1));
        assert_eq!(
            pair.words(),
            [
                ("dispatched", Side::Prefill, 1, 1, own, 7_000, 1),
                ("dispatched", Side::Decode, 101, 1, session, 7_000, 102),
            ]
        );
        tell(engine, &held, completed(Side::Prefill, 1));
        tell(engine, &held, completed(Side::Decode, 102));
        assert_eq!(
            pair.words(),
            [
                ("completed", Side::Prefill, 1, 0, own, 7_000, 1),
                ("completed", Side::Decode, 102, 1, session, 7_000, 102),
            ]
        );
        // A side a pair's lease does not hold is told to nobody.
        tell(engine, &held, completed(Side::Encode, 1));
        tell(engine, &held, dispatched(Side::Single, 0));
        assert!(pair.words().is_empty());
    }

    /// Each side's lease states its own side's session verdict. The
    /// prefill's is the pick's. The decode's is the one the pair of the
    /// pick's first target keeps, whatever the prefill's is, and a claim
    /// names that pair's decode.
    #[test]
    fn each_side_of_a_pair_states_its_own_session_verdict() {
        let pair = decided(1);
        let engine = pair.spec.engine();
        let stated = |decode_verdict| {
            let mut targets = Targets::new();
            targets.push(Target {
                id: WorkerId(1),
                pair: Some(Pair {
                    decode_id: WorkerId(101),
                    decode_verdict,
                    ..Pair::default()
                }),
                ..Target::default()
            });
            let decision = Decision::Pick(Pick {
                targets,
                session_verdict: SessionVerdict::Pinned,
                ..Pick::default()
            });
            let issued = issue(engine, &keyed(), &decision, LeaseId(9), true);
            let (prefill, _) = issued.expect("a pair's lease");
            let decode = issue_decode(engine, &keyed(), &decision, &prefill, 9_900);
            let decode = decode.expect("the decode side's lease");
            (prefill.session_verdict, decode.session_verdict)
        };
        let claimed = SessionVerdict::Claimed(WorkerId(101));
        for (kept, whole) in [
            (DecodeVerdict::NONE, SessionVerdict::None),
            (DecodeVerdict::PINNED, SessionVerdict::Pinned),
            (DecodeVerdict::RESPILL, SessionVerdict::Respill),
            (DecodeVerdict::CLAIMED, claimed),
        ] {
            assert_eq!(stated(kept), (SessionVerdict::Pinned, whole));
        }
    }

    #[test]
    fn a_release_ends_the_side_it_names_and_with_none_both() {
        let pair = decided(1);
        let engine = pair.spec.engine();
        let issued = issue(engine, &keyed(), &pair.decision, LeaseId(9), false);
        let (lease, session) = issued.expect("a pair's lease");
        pair.words();
        let own = lease.session;
        let taken = target_at(&pair.decision, 0);
        let held = HeldDecision {
            decision: &pair.decision,
            lease: &lease,
            index: 0,
            target: &taken,
            decode: HeldDecode {
                session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: false,
            },
        };
        let prefill = ("released", Side::Prefill, 1, 0, own, 7_000, 0);
        let decode = ("released", Side::Decode, 101, 0, session, 7_000, 0);
        let gone = ReleaseReason::ClientGone;
        tell(
            engine,
            &held,
            Said::Released(None, ReleaseReason::CommitFailed),
        );
        assert_eq!(pair.words(), [prefill, decode]);
        tell(engine, &held, Said::Released(Some(Side::Decode), gone));
        assert_eq!(pair.words(), [decode]);
        tell(engine, &held, Said::Released(Some(Side::Prefill), gone));
        assert_eq!(pair.words(), [prefill]);
    }

    /// A decode that stood chosen goes out as its pick's with no call. Its
    /// dispatch is its issue too: the stages hear the issue first, dated at
    /// the dispatch, and both words name the decode it goes to.
    #[test]
    fn a_chosen_decode_sent_with_no_call_is_told_issued_at_its_dispatch() {
        let pair = decided(1);
        let engine = pair.spec.engine();
        let decision = two_decodes();
        let issued = issue(engine, &keyed(), &decision, LeaseId(9), true);
        let (lease, session) = issued.expect("a pair's lease");
        pair.words();
        let taken = target_at(&decision, 1);
        let held = HeldDecision {
            decision: &decision,
            lease: &lease,
            index: 1,
            target: &taken,
            decode: HeldDecode {
                session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: true,
            },
        };
        tell(engine, &held, dispatched(Side::Decode, 1));
        assert_eq!(
            pair.words(),
            [
                ("issued", Side::Decode, 102, 1, session, 9_900, 1),
                ("dispatched", Side::Decode, 102, 1, session, 7_000, 102),
            ]
        );
    }

    /// A decode that stands chosen was issued to no stage, so its end with
    /// its prefill tells the prefill's alone.
    #[test]
    fn a_decode_that_stands_chosen_ends_with_its_prefill_and_is_told_to_nobody() {
        let pair = decided(1);
        let engine = pair.spec.engine();
        let issued = issue(engine, &keyed(), &pair.decision, LeaseId(9), true);
        let (lease, session) = issued.expect("a pair's lease");
        pair.words();
        let own = lease.session;
        let taken = target_at(&pair.decision, 0);
        let held = HeldDecision {
            decision: &pair.decision,
            lease: &lease,
            index: 0,
            target: &taken,
            decode: HeldDecode {
                session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: true,
            },
        };
        tell(
            engine,
            &held,
            Said::Released(None, ReleaseReason::Abandoned),
        );
        tell(
            engine,
            &held,
            Said::Released(Some(Side::Decode), ReleaseReason::Abandoned),
        );
        tell(engine, &held, completed(Side::Decode, 101));
        assert_eq!(
            pair.words(),
            [("released", Side::Prefill, 1, 0, own, 7_000, 0)]
        );
    }

    /// The engine decides a pair whole for the guard that asks. A pick is
    /// issued with both sides told, and each side's dispatch and end reach
    /// the stages from what the guard lends. A request held to a wire no
    /// worker of the lists is dialed on is declined under the wires' name.
    #[test]
    fn the_engine_decides_a_pair_whole_and_hears_each_sides_words_from_its_guard() {
        let host = Host::of(1, 1);
        let classes = every(1, 1);
        let (asks, now) = (host.asks(&classes, false), Instant::now());
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
            .expect("a decision");
        let first = target_at(guard.decision(), 0);
        let decode = first.pair.expect("a pair").decode_id;
        let at = guard.lease().expect("a lease").lease.issued_micros;
        guard.dispatched(now);
        guard.dispatched_decode(now);
        guard.completed(&outcome(Side::Prefill, first.id), now);
        guard.completed(&outcome(Side::Decode, decode), now);
        drop(guard);
        assert_eq!(
            host.words(),
            [
                ("issued", Side::Prefill, first.id, at),
                ("issued", Side::Decode, decode, at),
                ("dispatched", Side::Prefill, first.id, at),
                ("dispatched", Side::Decode, decode, at),
                ("completed", Side::Prefill, first.id, at),
                ("completed", Side::Decode, decode, at),
            ]
        );
        let mut undialed = host.asks(&classes, false);
        undialed.0.constraints.wire_modes = WireModes::ZMQ;
        let decided = host.engine.decide_under(GatewayId::LOCAL, &undialed, now);
        match decided
            .expect("an answer")
            .whole()
            .expect("a whole decision")
            .0
            .decision
        {
            Decision::Shed(shed) => assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::PolicyDeclined, WIRES)
            ),
            other => panic!("not shed: {other:?}"),
        }
        assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 0));
    }

    /// A pair's decision alone ([`decide`]) is the decision its leasing form
    /// makes ([`decided`]): of two engines made alike, the one asked for the
    /// decision alone names, call by call, the pair the other issues a lease
    /// for, and behind a held target the decode the other issues. It takes
    /// no lease id and tells no stage. A request held to a wire no worker
    /// of the lists is dialed on is declined under the wires' name by the
    /// decision alone, and the leasing form answers that decline with no
    /// lease, takes no id for it and tells no stage.
    #[test]
    fn a_pairs_decision_alone_is_its_leasing_forms_and_takes_no_id_and_tells_no_stage() {
        let (alone, leasing) = (Host::of(2, 2), Host::of(2, 2));
        let classes = every(2, 2);
        let now = Instant::now();
        for _ in 0..4 {
            let mut asked = alone.asks(&classes, false);
            let mut leased = leasing.asks(&classes, false);
            let decision = decide(
                alone.engine.loaded(),
                &asked.1,
                &mut asked.0,
                None,
                None,
                now,
                |_| true,
            );
            let outcome = super::decided(
                leasing.engine.loaded(),
                &leased.1,
                &mut leased.0,
                None,
                None,
                now,
                |_| true,
            );
            assert!(outcome.lease.is_some(), "the leasing form issues its pick");
            assert_eq!(alone.picked(&decision), leasing.picked(&outcome.decision));
        }
        // A pair whose decode is sent later, and that decode decided again
        // behind the target each engine's first pick names.
        let mut asked = alone.asks(&classes, true);
        let mut leased = leasing.asks(&classes, true);
        let first = decide(
            alone.engine.loaded(),
            &asked.1,
            &mut asked.0,
            None,
            None,
            now,
            |_| true,
        );
        let outcome = super::decided(
            leasing.engine.loaded(),
            &leased.1,
            &mut leased.0,
            None,
            None,
            now,
            |_| true,
        );
        assert_eq!(alone.picked(&first), leasing.picked(&outcome.decision));
        let (ours, theirs) = (target_at(&first, 0), target_at(&outcome.decision, 0));
        let lease = outcome.lease.expect("the pair's lease").lease;
        let held = HeldDecision {
            decision: &outcome.decision,
            lease: &lease,
            index: 0,
            target: &theirs,
            decode: HeldDecode {
                session: outcome.decode_session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: true,
            },
        };
        let call = now + Duration::from_millis(1);
        let again = decide(
            alone.engine.loaded(),
            &asked.1,
            &mut asked.0,
            Some(&ours),
            None,
            call,
            |_| true,
        );
        let issued = super::decided(
            leasing.engine.loaded(),
            &leased.1,
            &mut leased.0,
            Some(&held),
            None,
            call,
            |_| true,
        );
        assert!(issued.lease.is_some(), "the leasing form issues the decode");
        assert_eq!(alone.picked(&again), leasing.picked(&issued.decision));
        assert_eq!(alone.picked(&again).0, alone.picked(&first).0);
        // A request held to a wire no worker of the lists is dialed on.
        let mut undialed = alone.asks(&classes, false);
        undialed.0.constraints.wire_modes = WireModes::ZMQ;
        match decide(
            alone.engine.loaded(),
            &undialed.1,
            &mut undialed.0,
            None,
            None,
            now,
            |_| true,
        ) {
            Decision::Shed(shed) => assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::PolicyDeclined, WIRES)
            ),
            other => panic!("not shed: {other:?}"),
        }
        // The leasing form answers that request the same decline, first and
        // behind the held decision, with no lease and no session.
        let mut refused = leasing.asks(&classes, true);
        refused.0.constraints.wire_modes = WireModes::ZMQ;
        for behind in [None, Some(&held)] {
            let declined = super::decided(
                leasing.engine.loaded(),
                &refused.1,
                &mut refused.0,
                behind,
                None,
                call,
                |_| true,
            );
            assert!(matches!(&declined.decision, Decision::Shed(shed) if shed.emptied_by == WIRES));
            assert!(declined.lease.is_none());
            assert_eq!(declined.decode_session, Session::NONE);
        }
        // The leasing form took an id a pair and told each side's issue, and
        // neither for a decline: the decision alone took none and told nobody.
        assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 5);
        assert_eq!(leasing.words().len(), 10);
        assert_eq!(alone.engine.leases.0.load(Ordering::Relaxed), 0);
        assert!(alone.words().is_empty(), "no stage heard a word");
    }

    /// The whole decision the leasing entry answers a host's request.
    fn leased(host: &Host, maker: &impl RequestMaker, now: Instant) -> DecideOutcome {
        let decided = host.engine.decide_under(GatewayId::LOCAL, maker, now);
        decided.expect("an answer").whole().expect("no place").0
    }

    /// The targets a decision names, in their order: none for a shed.
    fn named(decision: &Decision) -> &[Target] {
        decision.pick().map_or(&[], |pick| pick.targets.as_slice())
    }

    /// What a decision that picks nothing is shed for, and under which
    /// stage's name.
    fn shed_for(decision: &Decision) -> (ShedReason, &'static str) {
        match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        }
    }

    /// The entry that issues no lease decides a pair as the leasing entry
    /// does ([`PolicyEngine::decide_without_lease`]). Of two engines made
    /// alike it names, call by call, the targets the other issues a lease
    /// for, and it takes no lease id and tells no stage. The decode asked
    /// again behind the target a first decision names
    /// ([`PolicyEngine::decide_decode_without_lease`]) is the decode the
    /// leasing entry answers a guard, over the lists as they stand at the
    /// call: the decode the first decision named is not ready by then.
    #[test]
    fn a_pair_without_a_lease_is_decided_as_the_leasing_entry_decides_it() {
        let (bare, leasing) = (Host::of(2, 3), Host::of(2, 3));
        let classes = every(2, 3);
        let now = Instant::now();
        for call in 0..6 {
            let asks = bare.asks(&classes, false);
            let decision = bare.engine.decide_without_lease(&asks, now);
            let (decision, _) = decision.expect("a decision");
            let outcome = leased(&leasing, &leasing.asks(&classes, false), now);
            assert!(outcome.lease.is_some(), "the leasing entry issues its pick");
            assert!(!named(&decision).is_empty(), "call {call}: a pick");
            assert_eq!(named(&decision), named(&outcome.decision), "call {call}");
        }
        // A pair whose decode is sent later, then that decode decided again
        // behind the target each engine's first decision names.
        let asks = bare.asks(&classes, true);
        let first = bare.engine.decide_without_lease(&asks, now);
        let (first, _) = first.expect("a decision");
        let theirs = leasing.asks(&classes, true);
        let outcome = leased(&leasing, &theirs, now);
        assert_eq!(named(&first), named(&outcome.decision));
        let (prefill, chosen) = bare.picked(&first);
        for host in [&bare, &leasing] {
            host.decodes[chosen].set_status(WorkerStatus::NotReady);
        }
        let (ours, target) = (target_at(&first, 0), target_at(&outcome.decision, 0));
        let lease = outcome.lease.expect("the pair's lease").lease;
        let held = HeldDecision {
            decision: &outcome.decision,
            lease: &lease,
            index: 0,
            target: &target,
            decode: HeldDecode {
                session: outcome.decode_session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: true,
            },
        };
        let call = now + Duration::from_millis(1);
        let again = bare.engine.decide_decode_without_lease(&asks, &ours, call);
        let again = again.expect("a decision");
        let issued = leasing
            .engine
            .decide_decode_under(GatewayId::LOCAL, &theirs, &held, call);
        let issued = issued.expect("a decision");
        assert!(
            issued.lease.is_some(),
            "the leasing entry issues the decode"
        );
        assert_eq!(named(&again), named(&issued.decision));
        let (kept, decode) = bare.picked(&again);
        assert_eq!(kept, prefill, "the held prefill");
        assert_ne!(decode, chosen, "a decode that is ready at the call");
        // The leasing entry took an id a pair and told each side's issue.
        // The other took none and told nobody.
        assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 7);
        assert_eq!(leasing.words().len(), 14);
        assert_eq!(bare.engine.leases.0.load(Ordering::Relaxed), 0);
        assert!(bare.words().is_empty(), "no stage heard a word");
    }

    /// The entry that issues no lease keeps the test of a request's wires
    /// that a pair's decision makes ([`decide`]), at the pair's first
    /// decision and at its decode's behind a held target. The prefills
    /// here are dialed on one wire and the decodes on another. A request
    /// that states one of the two is declined under the wires' name: a
    /// worker of the other list is dialed on a wire it does not state. A
    /// request that states both is decided, and so is one that states a
    /// third beside them and one that states none. The leasing entry
    /// answers each request alike.
    #[test]
    fn a_pair_without_a_lease_keeps_the_test_of_its_wires() {
        let dialed = |name: &str, worker_type, mode| -> Arc<[Arc<dyn Worker>]> {
            let spec = |at: usize| WorkerSpec {
                worker_type,
                connection_mode: mode,
                ..WorkerSpec::new(format!("http://{name}-{at}:9000"))
            };
            let ready = |at| {
                let worker = <BasicWorkerBuilder>::from_spec(spec(at))
                    .health_config(no_health_check())
                    .build();
                worker.set_status(WorkerStatus::Ready);
                Arc::new(worker) as Arc<dyn Worker>
            };
            (0..2).map(ready).collect()
        };
        let host = Host::over(
            hearing(),
            dialed("prefill", WorkerType::Prefill, ConnectionMode::Http),
            dialed("decode", WorkerType::Decode, ConnectionMode::Grpc),
        );
        let [prefills, decodes] = &host.lists;
        let wires = |list: &PublishedList| host.engine.members_of(list.lent()).wires();
        assert_eq!(
            (wires(prefills), wires(decodes)),
            (WireModes::HTTP, WireModes::GRPC)
        );
        let classes = every(2, 2);
        let now = Instant::now();
        let stating = |wires: WireModes| {
            let mut asks = host.asks(&classes, true);
            asks.0.constraints.wire_modes = wires;
            asks
        };
        let both = WireModes::HTTP.union(WireModes::GRPC);
        let first = host.engine.decide_without_lease(&stating(both), now);
        let held = target_at(&first.expect("a decision").0, 0);
        for wires in [WireModes::NONE, both, both.union(WireModes::ZMQ)] {
            let asks = stating(wires);
            let decision = host.engine.decide_without_lease(&asks, now);
            let (decision, _) = decision.expect("a decision");
            assert!(decision.pick().is_some(), "{wires:?}: {decision:?}");
            let again = host.engine.decide_decode_without_lease(&asks, &held, now);
            let again = again.expect("a decision");
            assert!(again.pick().is_some(), "{wires:?} again: {again:?}");
            assert!(leased(&host, &asks, now).decision.pick().is_some());
        }
        let declined = (ShedReason::PolicyDeclined, WIRES);
        for wires in [WireModes::HTTP, WireModes::GRPC, WireModes::ZMQ] {
            let asks = stating(wires);
            let decision = host.engine.decide_without_lease(&asks, now);
            assert_eq!(shed_for(&decision.expect("a decision").0), declined);
            let again = host.engine.decide_decode_without_lease(&asks, &held, now);
            assert_eq!(shed_for(&again.expect("a decision")), declined);
            assert_eq!(shed_for(&leased(&host, &asks, now).decision), declined);
        }
    }

    /// A request that hands no pair's lists is decided over no worker by
    /// the entries that issue no lease, as the leasing entries answer it: a
    /// pair's prefill, and its decode asked again behind a held target.
    #[test]
    fn a_pair_without_a_lease_that_hands_no_lists_is_decided_over_no_worker() {
        let host = Host::of(1, 1);
        let classes = every(1, 1);
        let now = Instant::now();
        let asks = host.asks(&classes, true);
        let outcome = leased(&host, &asks, now);
        let target = target_at(&outcome.decision, 0);
        let lease = outcome.lease.expect("the pair's lease").lease;
        let held = HeldDecision {
            decision: &outcome.decision,
            lease: &lease,
            index: 0,
            target: &target,
            decode: HeldDecode {
                session: outcome.decode_session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: true,
            },
        };
        // The request by itself is a maker that hands no list.
        let alone: &Request<'_> = &asks.0;
        let none = (ShedReason::NoEndpoints, "none");
        let prefill = host.engine.decide_without_lease(alone, now);
        assert_eq!(shed_for(&prefill.expect("a decision").0), none);
        assert_eq!(shed_for(&leased(&host, alone, now).decision), none);
        let again = host.engine.decide_decode_without_lease(alone, &target, now);
        assert_eq!(shed_for(&again.expect("a decision")), none);
        let issued = host
            .engine
            .decide_decode_under(GatewayId::LOCAL, alone, &held, now);
        let issued = issued.expect("an answer");
        assert_eq!(shed_for(&issued.decision), none);
        assert!(issued.lease.is_none(), "a shed issues nothing");
    }

    /// A guard lends its pair's session of the decode phase with the decode
    /// side ([`HeldDecode::session`]), and every word of that side is told
    /// under it: the session the side's issue stated, which is not the
    /// prefill side's.
    #[test]
    fn a_guards_decode_side_is_told_under_the_session_of_its_issue() {
        let host = Host::of(1, 1);
        let classes = every(1, 1);
        let (asks, now) = (host.asks(&classes, false), Instant::now());
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
            .expect("a decision");
        let decode = target_at(guard.decision(), 0).pair.expect("a pair");
        guard.dispatched(now);
        guard.dispatched_decode(now);
        guard.completed(&outcome(Side::Decode, decode.decode_id), now);
        drop(guard);
        let heard = std::mem::take(&mut *host.heard.lock());
        let told = |side: Side| -> Vec<(&'static str, Session)> {
            let of_side = heard.iter().filter(|word| word.1 == side);
            of_side.map(|word| (word.0, word.4)).collect()
        };
        let own = asks.0.facts.session();
        let decode_side = told(Side::Decode);
        let issued = decode_side.first().expect("the decode's issue").1;
        assert_ne!(issued, own, "each side keeps its own session");
        assert_eq!(
            decode_side,
            [
                ("issued", issued),
                ("dispatched", issued),
                ("completed", issued),
            ]
        );
        assert!(told(Side::Prefill).iter().all(|word| word.1 == own));
    }

    /// A decode sent later stands chosen in its guard. Behind the prefill's
    /// dispatch the engine decides it again over the host's lists of that
    /// call. The side is issued then under the pair's lease, dated at the
    /// call, and its later words bear the first decision's instant.
    #[test]
    fn the_engine_decides_a_chosen_decode_again_for_its_guard() {
        let host = Host::of(1, 2);
        let classes = every(1, 2);
        let (asks, now) = (host.asks(&classes, true), Instant::now());
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
            .expect("a decision");
        assert!(guard.decode_chosen());
        let first = target_at(guard.decision(), 0);
        let chosen = host.picked(guard.decision()).1;
        let at = guard.lease().expect("a lease").lease.issued_micros;
        guard.dispatched(now);
        // The decode the first pick named stops being ready behind it.
        host.decodes[chosen].set_status(WorkerStatus::NotReady);
        let call = now + Duration::from_millis(1);
        let again = *guard.decide_decode_under(&asks, call).expect("a decision");
        assert_eq!(host.picked(&again), (0, 1 - chosen));
        assert!(!guard.decode_chosen(), "the call's pick issued the side");
        let decode = target_at(&again, 0).pair.expect("a pair").decode_id;
        guard.dispatched_decode(call);
        guard.completed(&outcome(Side::Prefill, first.id), call);
        guard.completed(&outcome(Side::Decode, decode), call);
        drop(guard);
        assert_eq!(
            host.words(),
            [
                ("issued", Side::Prefill, first.id, at),
                ("dispatched", Side::Prefill, first.id, at),
                ("issued", Side::Decode, decode, at + 1_000),
                ("dispatched", Side::Decode, decode, at),
                ("completed", Side::Prefill, first.id, at),
                ("completed", Side::Decode, decode, at),
            ]
        );
    }

    /// A picker that picks as [`Best`] and panics at a decode side's issue.
    #[derive(Debug)]
    struct StopsAtDecode;

    impl Lifecycle for StopsAtDecode {
        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            assert!(lease.side != Side::Decode, "this stage panics there");
        }
    }

    impl Picker for StopsAtDecode {
        fn name(&self) -> &'static str {
            "stops-at-decode"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            rows: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Best.pick(facts, rows, kept, scores, band, out);
        }
    }

    /// [`hearing`] with a prefill side that picks through [`StopsAtDecode`].
    /// The decode profile's stage, which keeps what it hears, is told each
    /// word ahead of it.
    fn stopping() -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
        hearing_with(Some(("stops-at-decode", || Box::new(StopsAtDecode))))
    }

    /// A stage that panics at the issue of a decode decided again is held
    /// and counted with the words a stage did not hear. The call answers
    /// its pick and its guard holds the side as issued, so the stage that
    /// heard the issue ahead of the panic hears the side's dispatch and its
    /// end. Before, the call answered a fault, its guard kept the first
    /// decode as chosen, and no word ended the issue that stage had heard.
    ///
    /// The gateway's own routing picks a pair's decode with its prefill and
    /// has no stage that hears an issue
    /// (`model_gateway/src/routers/common/placement.rs:1227`).
    #[test]
    fn a_stage_that_panics_at_a_decode_calls_issue_costs_the_call_nothing() {
        let host = Host::under(stopping(), 1, 2);
        let classes = every(1, 2);
        let (asks, now) = (host.asks(&classes, true), Instant::now());
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
            .expect("a decision");
        assert!(guard.decode_chosen());
        let first = target_at(guard.decision(), 0);
        let at = guard.lease().expect("a lease").lease.issued_micros;
        guard.dispatched(now);
        let call = now + Duration::from_millis(1);
        let again = *guard.decide_decode_under(&asks, call).expect("a decision");
        let decode = target_at(&again, 0).pair.expect("a pair").decode_id;
        assert!(!guard.decode_chosen(), "the call's pick issued the side");
        guard.dispatched_decode(call);
        guard.completed(&outcome(Side::Prefill, first.id), call);
        guard.completed(&outcome(Side::Decode, decode), call);
        drop(guard);
        assert_eq!(
            host.words(),
            [
                ("issued", Side::Prefill, first.id, at),
                ("dispatched", Side::Prefill, first.id, at),
                ("issued", Side::Decode, decode, at + 1_000),
                ("dispatched", Side::Decode, decode, at),
                ("completed", Side::Prefill, first.id, at),
                ("completed", Side::Decode, decode, at),
            ]
        );
        // The held panic alone is counted, and no later word of the side.
        assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
    }

    /// A stage that panics at an issue of a pair's first decision is held
    /// and counted with the words a stage did not hear. The decision
    /// answers its pick and its lease, so the guard holds both sides as
    /// issued: the stage that heard each issue ahead of the panic hears
    /// each side's dispatch and its end. Before, the request was answered
    /// a fault and no word ended the two issues that stage had heard.
    ///
    /// The classic pass tells a policy nothing where a pick is issued: it
    /// asks each side's policy for one pick
    /// (`model_gateway/src/routers/common/placement.rs:2758`, `:2820`).
    #[test]
    fn a_stage_that_panics_at_a_pairs_first_issue_costs_the_request_nothing() {
        let host = Host::under(stopping(), 1, 2);
        let classes = every(1, 2);
        // The decode is issued with its prefill: its issue is the one the
        // stage panics at, behind the prefill side's.
        let (asks, now) = (host.asks(&classes, false), Instant::now());
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
            .expect("a decision");
        assert!(!guard.decode_chosen(), "the pick issued both sides");
        let first = target_at(guard.decision(), 0);
        let decode = first.pair.expect("a pair").decode_id;
        let at = guard.lease().expect("a lease").lease.issued_micros;
        guard.dispatched(now);
        guard.dispatched_decode(now);
        guard.completed(&outcome(Side::Prefill, first.id), now);
        guard.completed(&outcome(Side::Decode, decode), now);
        drop(guard);
        assert_eq!(
            host.words(),
            [
                ("issued", Side::Prefill, first.id, at),
                ("issued", Side::Decode, decode, at),
                ("dispatched", Side::Prefill, first.id, at),
                ("dispatched", Side::Decode, decode, at),
                ("completed", Side::Prefill, first.id, at),
                ("completed", Side::Decode, decode, at),
            ]
        );
        // The held panic alone is counted: no decision faulted.
        assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
    }

    /// The prompt size each word of a pair states to a stage, through a
    /// guard of `door`: the prefill side's words, then the decode side's.
    /// The first decision is `first`'s. Behind the prefill's dispatch the
    /// decode is decided again by `again`, or with none goes out as the
    /// first decision left it. The decode side is then sent and ends, or is
    /// `released` with no dispatch. The guard is dropped with its prefill
    /// sent, so that side ends as cancelled.
    fn sizes_by_side<D: DecisionEngine, M: RequestMaker>(
        door: &D,
        gateway: GatewayId,
        first: &M,
        again: Option<&M>,
        released: bool,
    ) -> [Vec<(&'static str, u32)>; 2] {
        // Words an earlier guard said on this thread are not this one's.
        SIZES.with_borrow_mut(Vec::clear);
        let now = Instant::now();
        let mut guard = DecisionGuard::decide_under(door, gateway, first, now).expect("a decision");
        guard.dispatched(now);
        if let Some(again) = again {
            let picked = guard.decide_decode_under(again, now);
            assert!(picked.is_ok_and(|decision| decision.pick().is_some()));
        }
        if released {
            guard.released_decode(ReleaseReason::ClientGone, now);
        } else {
            let decode = target_at(guard.decision(), 0).pair.expect("a pair");
            guard.dispatched_decode(now);
            guard.completed(&outcome(Side::Decode, decode.decode_id), now);
        }
        drop(guard);
        let heard = sizes();
        let of = |side: Side| {
            let words = heard.iter().filter(|word| word.1 == side);
            words.map(|&(word, _, size)| (word, size)).collect()
        };
        [of(Side::Prefill), of(Side::Decode)]
    }

    /// A decode call may state another prompt size than the first decision
    /// did. The side is issued under the call's size, and its dispatch, its
    /// end and its release state that size too: a stage that books a size
    /// at the issue reads the same size at every later word of the side.
    /// The prefill side states the first decision's size behind the call
    /// as before it. A decode that goes out with no call states the first
    /// decision's size, issued at its own dispatch or with its prefill.
    #[test]
    fn a_decode_calls_side_states_the_calls_size_at_every_word() {
        let host = Host::of(1, 2);
        let classes = every(1, 2);
        let first = host.asks(&classes, true);
        let stated = first.0.facts.prompt_tokens;
        let mut again = host.asks(&classes, true);
        again.0.facts.prompt_tokens = stated + 7;
        let (door, local, called) = (&host.engine, GatewayId::LOCAL, stated + 7);
        let own = [
            ("issued", stated),
            ("dispatched", stated),
            ("completed", stated),
        ];
        let [prefill, decode] = sizes_by_side(door, local, &first, Some(&again), false);
        assert_eq!(prefill, own);
        assert_eq!(
            decode,
            [
                ("issued", called),
                ("dispatched", called),
                ("completed", called),
            ]
        );
        let [_, decode] = sizes_by_side(door, local, &first, Some(&again), true);
        assert_eq!(decode, [("issued", called), ("released", called)]);
        let [_, decode] = sizes_by_side(door, local, &first, None, false);
        assert_eq!(decode, own);
        let with = host.asks(&classes, false);
        let [_, decode] = sizes_by_side(door, local, &with, None, false);
        assert_eq!(decode, own);
    }

    /// A decode call may state another text than the first decision did.
    /// The side is issued under the call's three sizes, and its dispatch,
    /// its end and its release state them too: a stage that books a
    /// request of text by its characters reads one length at every word of
    /// the side, and a plugin is shown the size the call's own request
    /// showed. Neither request here states a token count: the first is a
    /// text of 400 characters in 800 bytes, the call's one of 900 in 1,800.
    /// The prefill side states the first decision's lengths behind the
    /// call as before it, and so does a decode that goes out with no call.
    #[test]
    fn a_decode_calls_side_states_the_calls_text_lengths_at_every_word() {
        let host = Host::of(1, 2);
        let classes = every(1, 2);
        let of_text = |chars, bytes| {
            let mut asks = host.asks(&classes, true);
            let facts = &mut asks.0.facts;
            (facts.prompt_tokens, facts.prompt_chars, facts.prompt_bytes) = (0, chars, bytes);
            asks
        };
        let (first, again) = (of_text(400, 800), of_text(900, 1_800));
        let (door, local) = (&host.engine, GatewayId::LOCAL);
        // What each word of a side stated: the prefill's, then the decode's.
        let heard = |again, released| {
            LENGTHS.with_borrow_mut(Vec::clear);
            let _ = sizes_by_side(door, local, &first, again, released);
            let heard = LENGTHS.with_borrow_mut(std::mem::take);
            let of = |side| {
                let words = heard.iter().filter(|word| word.0 == side);
                words.map(|word| word.1).collect::<Vec<_>>()
            };
            [of(Side::Prefill), of(Side::Decode)]
        };
        let (own, called) = ([400, 800, 200], [900, 1_800, 450]);
        let [prefill, decode] = heard(Some(&again), false);
        assert_eq!(prefill, [own; 3], "issued, sent and ended");
        assert_eq!(decode, [called; 3], "issued, sent and ended");
        let [_, decode] = heard(Some(&again), true);
        assert_eq!(decode, [called; 2], "issued and released");
        let [_, decode] = heard(None, false);
        assert_eq!(decode, [own; 3], "with no call");
    }

    /// Four prefills and four decodes under the pair document of the two
    /// presets, every pair open.
    fn of_presets(prefill: &str, decode: &str) -> (Host, PartnerClasses) {
        let mut document = PolicySpec::pair_presets(prefill, decode).expect("presets");
        document.generation = 3;
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        (Host::under((spec, Arc::default()), 4, 4), every(4, 4))
    }

    /// Those workers' reports: every worker of a side waits on a long queue
    /// but the one at the place named for its side.
    fn reports(prefill: usize, decode: usize) -> Vec<(String, WorkerLoadResponse)> {
        let side = |side: &'static str, idle: usize| {
            (0..4).map(move |at| {
                let queued = if at == idle { 0 } else { 90_000 };
                let url = format!("grpc://{side}-{at}:9000");
                (url, one_rank_load(queued, 0.1, 1_000.0))
            })
        };
        let both = side("prefill", prefill).chain(side("decode", decode));
        both.collect()
    }

    /// A snapshot of those reports, as a request gives one.
    fn queues(prefill: usize, decode: usize) -> Arc<LoadSnapshot> {
        LoadSnapshot::from_loads_for_test(reports(prefill, decode))
    }

    /// The engine's own task, turned once: it takes what stands ready for
    /// it, here the newest snapshot on the channel its host handed.
    async fn turned(engine: &PolicyEngine) {
        let step = Duration::from_millis(1);
        assert!(tokio::time::timeout(step, engine.run()).await.is_err());
    }

    /// Each side of a pair that scores by load reads its workers' reports in
    /// the snapshot its host hands for the call: the prefill and the decode
    /// whose queue is empty are picked at every seed. The decode decided
    /// again behind its prefill reads the snapshot of its own call, and
    /// keeps the prefill that was sent whatever that snapshot says of the
    /// prefills. Handed no snapshot, the same engine reads no report: no
    /// worker here holds a request in flight, so each scores alike and the
    /// picks spread over both lists. The gateway's classic router reads the
    /// reports where a side's policy has workers to choose between: each
    /// side of a pair asks its policy
    /// (`model_gateway/src/routers/common/placement.rs:1521` and `:1583`),
    /// and `least_load` reads its snapshot and each worker's report in it
    /// (`model_gateway/src/policies/least_load.rs:918` and `:450`, at main
    /// `ddd8a44cb`), behind its answer for a side's one healthy worker
    /// (`:1119-1120`). A decode pool all over its overload thresholds is not
    /// asked of its policy there: its pick is the least loaded partner
    /// (`placement.rs:1575-1578`). No worker of this case is over one.
    #[test]
    fn a_pair_that_scores_by_load_reads_the_reports_its_host_hands() {
        let (host, classes) = of_presets("least_load", "least_load");
        let mut unread = [[0_u32; 4]; 2];
        for seed in 0..64 {
            let fed = host.asks_by(&classes, (seed, false), Some(queues(2, 1)));
            assert_eq!(host.first(&fed), (2, 1), "seed {seed}");
            let (prefill, decode) = host.first(&host.asks_by(&classes, (seed, false), None));
            unread[0][prefill] += 1;
            unread[1][decode] += 1;
        }
        assert!(
            unread.iter().flatten().all(|&picks| picks > 0),
            "{unread:?}"
        );
        // The decode decided again: in its call's snapshot another decode
        // has the empty queue, and another prefill too. The prefill that
        // was sent stays the pair's.
        for seed in 0..16 {
            let now = Instant::now();
            let sent = host.asks_by(&classes, (seed, true), Some(queues(2, 1)));
            let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &sent, now)
                .expect("a decision");
            assert_eq!(host.picked(guard.decision()), (2, 1), "seed {seed}");
            guard.dispatched(now);
            let call = host.asks_by(&classes, (seed, true), Some(queues(0, 3)));
            let again = *guard.decide_decode_under(&call, now).expect("a decision");
            assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
        }
    }

    /// The snapshot reaches a side that scores by load beside a side that
    /// reads no report. With round robin prefills the decode whose queue is
    /// empty is picked at every seed, and with round robin decodes the
    /// prefill whose queue is empty.
    #[test]
    fn a_side_that_scores_by_load_reads_the_reports_beside_a_side_that_reads_none() {
        for (prefill, decode) in [("round_robin", "least_load"), ("least_load", "round_robin")] {
            let (host, classes) = of_presets(prefill, decode);
            for seed in 0..64 {
                let fed = host.asks_by(&classes, (seed, false), Some(queues(2, 1)));
                let picked = host.first(&fed);
                if decode == "least_load" {
                    assert_eq!(picked.1, 1, "the decode at seed {seed}");
                } else {
                    assert_eq!(picked.0, 2, "the prefill at seed {seed}");
                }
            }
        }
    }

    /// A pair whose request gives no snapshot reads the newest the engine's
    /// own task keeps at hand, as a single request does. At its first
    /// decision and at its decode decided again, the worker whose queue is
    /// empty in the reports the load monitor published last is picked at
    /// every seed, by the entries that issue a lease and by those that
    /// issue none. Until the task has read the channel the engine holds no
    /// snapshot, and the picks spread. A snapshot the request gives is read
    /// in place of the engine's.
    #[tokio::test(start_paused = true)]
    async fn a_pair_whose_request_gives_no_snapshot_reads_the_engines_newest() {
        let (host, classes) = of_presets("least_load", "least_load");
        let mut channel = None;
        let feed = TestFeed::attached_to(|loads| channel = loads);
        // The load monitor's side: the reports published again.
        let publish = |prefill, decode| {
            let published = reports(prefill, decode);
            feed.publish(&published.into_iter().collect());
        };
        publish(2, 1);
        host.engine.set_loads(channel.expect("the feed's channel"));
        let bare = |seed, later| host.asks_by(&classes, (seed, later), None);
        let mut unread = [[0_u32; 4]; 2];
        for seed in 0..64 {
            let (prefill, decode) = host.first(&bare(seed, false));
            unread[0][prefill] += 1;
            unread[1][decode] += 1;
        }
        assert!(
            unread.iter().flatten().all(|&picks| picks > 0),
            "{unread:?}"
        );
        turned(&host.engine).await;
        for seed in 0..32 {
            assert_eq!(host.first(&bare(seed, false)), (2, 1), "seed {seed}");
            let given = host.asks_by(&classes, (seed, false), Some(queues(0, 3)));
            assert_eq!(host.first(&given), (0, 3), "seed {seed}");
        }
        // The decode decided again reads what the monitor published since
        // the first decision, and keeps the prefill that was sent.
        for seed in 0..16 {
            publish(2, 1);
            turned(&host.engine).await;
            let now = Instant::now();
            let sent = bare(seed, true);
            let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &sent, now)
                .expect("a decision");
            assert_eq!(host.picked(guard.decision()), (2, 1), "seed {seed}");
            guard.dispatched(now);
            let unleased = host.engine.decide_without_lease(&sent, now);
            let (unleased, _) = unleased.expect("a decision");
            assert_eq!(host.picked(&unleased), (2, 1), "seed {seed}");
            publish(0, 3);
            turned(&host.engine).await;
            let call = bare(seed, true);
            let again = *guard.decide_decode_under(&call, now).expect("a decision");
            assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
            let held = target_at(&unleased, 0);
            let again = host.engine.decide_decode_without_lease(&call, &held, now);
            assert_eq!(
                host.picked(&again.expect("a decision")),
                (2, 3),
                "seed {seed}"
            );
        }
    }

    /// The entries that issue no lease read the reports as the leasing
    /// entries do ([`PolicyEngine::decide_without_lease`],
    /// [`PolicyEngine::decide_decode_without_lease`]). At every seed a
    /// pair's first decision picks the two workers whose queues are empty
    /// in the snapshot its request gives. Its decode decided again picks
    /// the decode whose queue is empty in the snapshot of its own call,
    /// behind the prefill the first decision named.
    #[test]
    fn a_pair_decided_without_a_lease_reads_the_reports_its_host_hands() {
        let (host, classes) = of_presets("least_load", "least_load");
        let now = Instant::now();
        for seed in 0..32 {
            let asks = host.asks_by(&classes, (seed, true), Some(queues(2, 1)));
            let first = host.engine.decide_without_lease(&asks, now);
            let (first, _) = first.expect("a decision");
            assert_eq!(host.picked(&first), (2, 1), "seed {seed}");
            let held = target_at(&first, 0);
            let call = host.asks_by(&classes, (seed, true), Some(queues(0, 3)));
            let again = host.engine.decide_decode_without_lease(&call, &held, now);
            let again = again.expect("a decision");
            assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
        }
    }

    /// A pair that exists among the places is found. Six prefills, each
    /// with its own decode, and the decode list held to the sixth decode's
    /// place: the sixth prefill is picked at every turn. A pick among all
    /// six names four of them a turn and would miss it on some. The
    /// gateway's classic router does the same by the request's own rule: a
    /// prefill is a candidate only where one of its partners is open for
    /// the request (`model_gateway/src/routers/common/placement.rs:1301-1306`,
    /// `:1389-1394`, `:1423-1437`, at main `c738c8d3b`).
    #[test]
    fn a_prefill_whose_partners_stand_outside_the_decode_places_is_no_candidate() {
        let (spec, _) = hearing();
        let (mut prefills, mut decodes) = lists(6, 6);
        let own: Vec<[u32; 1]> = (0..6).map(|at| [at]).collect();
        let classes = PartnerClasses::number(6, own.iter().map(|one| one.as_slice()));
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let free = Constraints::default();
        for seed in 0..24 {
            let facts = RequestFacts { seed, ..facts() };
            let decision = decide_records(
                spec.engine(),
                &facts,
                (&free, [unvetoed(None), unvetoed(Some(&[5]))]),
                &pool(&prefills),
                pools(&prefills, &decodes, &classes),
                &mut Scratch::default(),
            );
            assert_eq!(picked(&decision), (6, 106), "seed {seed}");
        }
    }

    /// What a pair's shed says is said of the prefills the request may take.
    /// The one prefill place holds a worker that is not ready, the prefills
    /// outside the places are ready, and every decode is over its
    /// thresholds: the shed names the prefill side's stage, not the decodes'
    /// veto, which stands against prefills the request may not take. The
    /// gateway's classic router names the prefill role there: no prefill
    /// its request's rule admits is available
    /// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
    /// `:1342-1346`, `:1395-1413`, at main `c738c8d3b`).
    #[test]
    fn a_pair_with_no_ready_prefill_among_its_places_is_shed_at_the_prefill_stage() {
        let (spec, _) = hearing();
        let (mut prefills, mut decodes) = lists(3, 2);
        let classes = every(3, 2);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        prefills[0].health.healthy = false;
        for decode in &mut decodes {
            decode.health.overloaded = true;
        }
        let free = Constraints::default();
        let decision = decide_records(
            spec.engine(),
            &facts(),
            (&free, [unvetoed(Some(&[0])), unvetoed(None)]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        );
        match decision {
            Decision::Shed(shed) => assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::NoEndpoints, "healthy")
            ),
            other => panic!("not shed: {other:?}"),
        }
    }

    /// The decode asked again looks its held prefill up before it judges
    /// what the call states. A prefill that is in no list reads
    /// `prefill-gone` under a demand, a subset hint, places of the request's
    /// own, and decode places of the list. With the prefill listed, the
    /// first three read their own stage and the fourth is decided among its
    /// places. The gateway's classic router asks no decode again: it selects
    /// a request's pair whole, once
    /// (`model_gateway/src/routers/common/placement.rs:1227-1275`, at main
    /// `c738c8d3b`).
    #[test]
    fn a_decode_asked_for_a_prefill_that_left_reads_prefill_gone_whatever_it_states() {
        let pair = decided(2);
        let (mut prefills, mut decodes) = lists(1, 2);
        let classes = every(1, 2);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let again = |held: u64, cx: &Constraints<'_>, places: Option<&[u32]>| {
            decide_decode_records(
                pair.spec.engine(),
                &facts(),
                (cx, unvetoed(places)),
                &pool(&decodes),
                pools(&prefills, &decodes, &classes),
                &prefill_alone(WorkerId(held)),
                &mut Scratch::default(),
            )
        };
        let shed_of = |decision: Decision| match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        };
        let free = Constraints::default();
        let demanded = Constraints {
            demands: Demands {
                vision: true,
                ..Demands::default()
            },
            ..free
        };
        let hinted = Constraints {
            subset_hint: Some(2),
            ..free
        };
        let held_to = Constraints {
            places: Some(&[0]),
            ..free
        };
        let gone = (ShedReason::NoEndpoints, "prefill-gone");
        for (cx, own) in [
            (&demanded, (ShedReason::NoEndpoints, WIRES)),
            (&hinted, (ShedReason::PolicyDeclined, SUBSET_HINT)),
            (&held_to, (ShedReason::PolicyDeclined, single::PLACES)),
        ] {
            assert_eq!(shed_of(again(5, cx, None)), gone);
            assert_eq!(shed_of(again(1, cx, None)), own);
        }
        assert_eq!(shed_of(again(5, &free, Some(&[1]))), gone);
        assert_eq!(picked(&again(1, &free, Some(&[1]))), (1, 102));
    }

    /// The all-vetoed rule keeps to a side's places. Two prefills, each
    /// with its own decode, and the decode list held to the second decode's
    /// place, which is over its thresholds. Under a binding that steers the
    /// pair is the second prefill with that decode, whatever stands outside
    /// the places: a first decode over its thresholds too opens no prefill,
    /// and a free one does not keep the pool from the rule. Under a binding
    /// that sheds both are refused as saturated. Where the one place holds
    /// a decode that is not ready, a decode outside it that is over its
    /// thresholds or doubted takes nothing: the rule looks at every row of
    /// the side and the pair is refused at the place's own stage, under
    /// either word. The gateway's classic router counts only the decodes
    /// its request's rule admits when it lifts a decode pool, and pairs
    /// among those
    /// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
    /// `:1315-1321`, `:1341`, `:1389-1394`, `:1575-1578`, at main
    /// `ebbcd9c6d`).
    #[test]
    fn the_all_vetoed_rule_keeps_to_a_sides_places() {
        use crate::{document::spec::OverloadSpec, engine::decision::LiftedVeto};
        let spec_of = |shed: bool| {
            let presets = PolicySpec::pair_presets("passthrough", "passthrough");
            let mut document = presets.expect("the pair presets");
            document.generation = 3;
            document.bindings[0].overload = Some(OverloadSpec {
                shed_when_all_vetoed: shed,
                ..OverloadSpec::default()
            });
            AppliedSpec::validate(&document).expect("a valid document")
        };
        let (steers, sheds) = (spec_of(false), spec_of(true));
        let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
        let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
        let free = Constraints::default();
        // The two lists as `health` leaves the decodes, decided with the
        // decode list held to its second place.
        let decide = |spec: &AppliedSpec, health: &dyn Fn(&mut [Candidate])| {
            let (mut prefills, mut decodes) = lists(2, 2);
            stamp(&mut prefills, &mut decodes, &classes, &OPEN);
            health(&mut decodes);
            decide_records(
                spec.engine(),
                &facts(),
                (&free, [unvetoed(None), unvetoed(Some(&[1]))]),
                &pool(&prefills),
                pools(&prefills, &decodes, &classes),
                &mut Scratch::default(),
            )
        };
        let shed_of = |decision: Decision| match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        };
        for first_over in [true, false] {
            let over = |decodes: &mut [Candidate]| {
                decodes[0].health.overloaded = first_over;
                decodes[1].health.overloaded = true;
            };
            let steered = decide(&steers, &over);
            assert_eq!(picked(&steered), (2, 102), "first over {first_over}");
            assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
            assert_eq!(
                shed_of(decide(&sheds, &over)),
                (ShedReason::Saturated, "not-overloaded"),
                "first over {first_over}"
            );
        }
        for doubted in [false, true] {
            let outside = |decodes: &mut [Candidate]| {
                decodes[0].health.overloaded = !doubted;
                decodes[0].health.stalled = doubted;
                decodes[1].health.healthy = false;
            };
            for spec in [&steers, &sheds] {
                assert_eq!(
                    shed_of(decide(spec, &outside)),
                    (ShedReason::NoEndpoints, "healthy"),
                    "doubted {doubted}"
                );
            }
        }
    }

    /// The lifted rule opens no prefill for a decode outside the decode
    /// places. Six prefills, each with its own decode, every decode over
    /// its thresholds, and the decode list held to the sixth decode's
    /// place. Under a binding that steers the pair is the sixth prefill
    /// with that decode at every turn, the overload veto lifted. Were the
    /// other five decodes to open their prefills too, the prefills' pick,
    /// which names four of the six a turn, would miss the sixth on some
    /// turns and no pair would be found. The gateway's classic router lifts
    /// a decode pool over the workers its request's rule admits, and takes
    /// a prefill as a candidate only where one of its partners is open
    /// among those
    /// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
    /// `:1315-1321`, `:1341-1345`, `:1389-1394`, `:1423-1437`, at main
    /// `9f303447f`).
    #[test]
    fn the_lifted_rule_opens_no_prefill_for_a_decode_outside_the_decode_places() {
        use crate::{document::spec::OverloadSpec, engine::decision::LiftedVeto};
        let presets = PolicySpec::pair_presets("round_robin", "round_robin");
        let mut document = presets.expect("the pair presets");
        document.generation = 3;
        document.bindings[0].overload = Some(OverloadSpec {
            shed_when_all_vetoed: false,
            ..OverloadSpec::default()
        });
        let steers = AppliedSpec::validate(&document).expect("a valid document");
        let (mut prefills, mut decodes) = lists(6, 6);
        let own: Vec<[u32; 1]> = (0..6).map(|at| [at]).collect();
        let classes = PartnerClasses::number(6, own.iter().map(|one| one.as_slice()));
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        for decode in &mut decodes {
            decode.health.overloaded = true;
        }
        let free = Constraints::default();
        for seed in 0..24 {
            let facts = RequestFacts { seed, ..facts() };
            let decision = decide_records(
                steers.engine(),
                &facts,
                (&free, [unvetoed(None), unvetoed(Some(&[5]))]),
                &pool(&prefills),
                pools(&prefills, &decodes, &classes),
                &mut Scratch::default(),
            );
            assert_eq!(picked(&decision), (6, 106), "seed {seed}");
            assert_eq!(decision.trace().lifted, LiftedVeto::Overload, "seed {seed}");
        }
    }

    /// The later call picks a decode behind one prefill.
    /// Its filter scope opens once with a pool size of two and two decode rows.
    #[test]
    fn a_decode_call_shows_its_profile_the_decode_pools_facts() {
        use crate::engine::framework::DecisionScope;

        #[derive(Debug)]
        struct ObservePool(Arc<Mutex<Vec<(u32, usize)>>>);

        impl Filter for ObservePool {
            fn name(&self) -> &'static str {
                "observe-pool"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self)
            }
        }

        impl DecisionScope for ObservePool {
            fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], pool: &PoolFacts<'_>) {
                self.0.lock().push((pool.pool_size, rows.len()));
            }

            fn close(&self) {}
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let opens = Arc::clone(&seen);
        let mut catalog = StageRegistry::builtin();
        catalog
            .filter("observe-pool", move || {
                Box::new(ObservePool(Arc::clone(&opens)))
            })
            .expect("a free filter name");
        let mut document =
            PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
        document
            .profiles
            .get_mut("decode")
            .expect("the decode profile")
            .filter
            .push("observe-pool".into());
        let spec = AppliedSpec::validate_in(&document, &catalog, &[]).expect("a valid document");
        let host = Host::under((spec, Arc::default()), 1, 2);
        let classes = every(1, 2);
        let asks = host.asks(&classes, true);
        let now = Instant::now();
        let mut guard = DecisionGuard::decide_under(&host.engine, Default::default(), &asks, now)
            .expect("a pair decision");
        guard.dispatched(now);
        seen.lock().clear();

        let later = guard
            .decide_decode_under(&asks, now)
            .expect("a decode decision");
        assert!(later.pick().is_some(), "{later:?}");
        assert!(!guard.decode_chosen(), "the later call issued its pick");
        assert_eq!(seen.lock().as_slice(), [(2, 2)]);
    }

    /// A shed at the later call leaves the original decode chosen.
    /// Only the prefill has a booking until dispatch books the decode.
    #[test]
    fn a_shed_at_the_call_leaves_the_side_chosen() {
        let host = Host::of(1, 1);
        let classes = every(1, 1);
        let mut asks = host.asks(&classes, true);
        let now = Instant::now();
        let local = Default::default();
        let mut guard =
            DecisionGuard::decide_under(&host.engine, local, &asks, now).expect("a pair decision");
        let original = *guard.decision();
        let target = first_target(&original);
        let decode = target.pair.expect("the chosen decode").decode_id;
        let before = *guard.lease().expect("the prefill lease");
        guard.dispatched(now);

        let call = now + Duration::from_millis(1);
        for (hint, reason) in [
            (0, ShedReason::NoEndpoints),
            (1, ShedReason::PolicyDeclined),
        ] {
            asks.0.constraints.subset_hint = Some(hint);

            // The raw answer exposes the shed and its lease field.
            let held = HeldDecision {
                decision: guard.decision(),
                lease: &before.lease,
                index: 0,
                target: guard.target(),
                decode: HeldDecode {
                    session: decode_session(&asks.0.facts),
                    tokens: before.lease.tokens,
                    chars: before.lease.chars,
                    bytes: before.lease.bytes,
                    chosen: guard.decode_chosen(),
                },
            };
            let answer = host
                .engine
                .decide_decode_under(local, &asks, &held, call)
                .expect("a decode decision");
            assert_eq!(
                shed_for(&answer.decision),
                (reason, "subset-hint"),
                "hint {hint}"
            );
            assert!(answer.lease.is_none(), "hint {hint}");

            // The guard returns the pick it still holds.
            let standing = guard
                .decide_decode_under(&asks, call)
                .expect("the held pick");
            assert_eq!(standing, &original);
            assert_eq!(guard.target(), &target);
            assert!(guard.decode_chosen());
        }

        let after = guard.lease().expect("the held prefill lease");
        assert_eq!((after.lease, after.due), (before.lease, before.due));
        let at = before.lease.issued_micros;
        assert_eq!(
            host.words(),
            [
                ("issued", Side::Prefill, target.id, at),
                ("dispatched", Side::Prefill, target.id, at),
            ]
        );

        guard.dispatched_decode(call);
        assert!(!guard.decode_chosen());
        assert_eq!(
            host.words(),
            [
                ("issued", Side::Decode, decode, at + 1_000),
                ("dispatched", Side::Decode, decode, at),
            ]
        );
    }

    /// The later decode call makes its request from the document it decides under.
    /// It issues a decode lease and records that document's generation in the trace.
    #[test]
    fn a_decode_calls_request_is_made_from_the_document_the_call_decides_under() {
        struct FromDocument<'a> {
            asks: Asks<'a>,
            document: &'a AppliedSpec,
        }

        impl RequestMaker for FromDocument<'_> {
            fn model(&self) -> &str {
                self.asks.model()
            }

            fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
                assert!(
                    std::ptr::eq(spec, self.document),
                    "the maker receives the deciding document"
                );
                let mut request = self.asks.make(spec);
                request.facts.seed = spec.generation();
                request
            }

            fn workers(&self) -> WorkerLists<'_> {
                self.asks.workers()
            }
        }

        let host = Host::of(1, 1);
        let classes = every(1, 1);
        let asks = host.asks(&classes, true);
        let first = leased(&host, &asks, Instant::now());
        let target = first_target(&first.decision);
        let lease = first.lease.expect("the prefill lease").lease;
        let held = HeldDecision {
            decision: &first.decision,
            lease: &lease,
            index: 0,
            target: &target,
            decode: HeldDecode {
                session: first.decode_session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: true,
            },
        };
        let local = Default::default();
        host.engine
            .say(local, &held, dispatched(Side::Prefill, 0), Instant::now());

        let maker = FromDocument {
            asks,
            document: host.engine.spec(),
        };
        let answer = host
            .engine
            .decide_decode_under(local, &maker, &held, Instant::now())
            .expect("a decode decision");
        assert_eq!(
            answer.lease.expect("a decode lease").lease.side,
            Side::Decode
        );
        let trace = answer.decision.trace();
        assert_eq!(
            trace.seed,
            maker.document.generation(),
            "the call uses the seed its maker set"
        );
        assert_eq!(
            trace.generation, trace.seed,
            "the trace names the generation handed to the maker"
        );
    }

    /// A caller held to the decodes it reaches (`decode_places`) is paired
    /// by the table of the full lists. The first prefill, a partner of
    /// every decode, takes each decode its caller reaches and no other.
    /// The second, a partner of the one decode both callers reach, takes
    /// that one. The decode records are written whole, each with its class.
    #[test]
    fn each_gateway_pairs_its_reachable_decodes_by_the_full_lists_table() {
        let all = [0_u32, 1, 2, 3, 4, 5, 6, 7, 8];
        let shared = [1];
        let host = Host::of(2, all.len());
        let classes = PartnerClasses::number(all.len(), [&all[..], &shared[..]]);
        let full_records: Vec<_> = host
            .decodes
            .iter()
            .map(|worker| worker_id(worker.as_ref()))
            .zip(classes.decodes().iter().copied())
            .collect();
        let now = Instant::now();

        let partners = |prefill: usize, places: Option<&[u32]>| {
            let other = worker_id(host.prefills[1 - prefill].as_ref());
            let mut excluded = vec![other];
            let mut drawn = Vec::new();
            loop {
                let mut ask = host.asks(&classes, false);
                ask.1.decode_places = places;
                ask.0.constraints.excluded = &excluded;
                let (decision, _) = host
                    .engine
                    .decide_without_lease(&ask, now)
                    .expect("a pair decision");
                let (chosen, decode) = host.picked(&decision);
                assert_eq!(chosen, prefill, "decode places {places:?}");

                // A soft exclusion allows a repeat once every partner has been drawn.
                let id = full_records[decode].0;
                if excluded.contains(&id) {
                    break;
                }
                excluded.push(id);
                drawn.push(decode as u32);
            }

            BUFFERS.with_borrow(|buffers| {
                let written: Vec<_> = buffers.rows[1]
                    .iter()
                    .map(|row| (row.id, row.pair_class))
                    .collect();
                assert_eq!(written, full_records, "decode classes for {places:?}");
            });
            drawn.sort_unstable();
            drawn
        };

        assert_eq!(partners(0, None), all);
        assert_eq!(partners(1, None), shared);

        // Place 1 is shared. Each other place is local to one gateway.
        let gateway_a = [0, 1, 3, 5, 7];
        let gateway_b = [1, 2, 4, 6, 8];
        for places in [gateway_a, gateway_b] {
            assert_eq!(partners(0, Some(&places)), places, "reachable {places:?}");
            assert_eq!(partners(1, Some(&places)), shared, "reachable {places:?}");
        }
    }

    /// One side's list engine after engine, one engine a number of
    /// `shape`: that many ranks of one data-parallel engine in their
    /// order, or a worker that registered no rank for a `1`. Beside it,
    /// each worker's engine by its place in the list.
    fn ranked(
        name: &str,
        worker_type: WorkerType,
        shape: &[usize],
    ) -> (Arc<[Arc<dyn Worker>]>, Vec<usize>) {
        let (mut list, mut engine_of) = (Vec::new(), Vec::new());
        for (engine, &size) in shape.iter().enumerate() {
            for rank in 0..size {
                let built = <BasicWorkerBuilder>::from_spec(WorkerSpec {
                    worker_type,
                    ..WorkerSpec::new(format!("grpc://{name}-{engine}:9000"))
                });
                let built = match size {
                    1 => built,
                    _ => built.dp_config(rank, size),
                };
                let worker = built.health_config(no_health_check()).build();
                worker.set_status(WorkerStatus::Ready);
                list.push(Arc::new(worker) as Arc<dyn Worker>);
                engine_of.push(engine);
            }
        }
        (list.into(), engine_of)
    }

    /// A pair's retry leaves every rank of the engines it tried, a side at
    /// a time: a tried prefill takes its engine's other rank out of the
    /// prefills and leaves the decodes as they were, a tried decode does
    /// the same on its side, and both together move both sides. Named the
    /// tried rank alone, each side kept its engine's other rank. That is
    /// the gateway's classic answer for a pair: each side steps aside from
    /// the engines the request tried
    /// (`model_gateway/src/routers/common/placement.rs:1486-1493` and
    /// `:1562-1569` at main `75d279287`).
    #[test]
    fn a_pairs_retry_leaves_every_rank_of_the_engines_it_tried() {
        use std::collections::BTreeSet;

        // On each side two ranks of one engine, then two of another.
        let (prefills, prefill_engine) = ranked("prefill", WorkerType::Prefill, &[2, 2]);
        let (decodes, decode_engine) = ranked("decode", WorkerType::Decode, &[2, 2]);
        let classes = every(prefills.len(), decodes.len() as u32);
        let tried_prefill = worker_id(prefills[1].as_ref());
        let tried_decode = worker_id(decodes[0].as_ref());
        let host = Host::over(hearing(), prefills, decodes);
        // The engines a retry's targets stand on, a side each, over the
        // turns of both lists.
        let landed = |tried: &[WorkerId]| {
            let (mut on_prefills, mut on_decodes) = (BTreeSet::new(), BTreeSet::new());
            for seed in 0..32 {
                let mut asks = host.asks(&classes, false);
                asks.0.facts.seed = seed;
                asks.0.facts.attempt = 1;
                asks.0.facts.tried = tried;
                let outcome = leased(&host, &asks, Instant::now());
                for target in named(&outcome.decision) {
                    let pair = target.pair.expect("a pair");
                    on_prefills.insert(prefill_engine[target.position as usize]);
                    on_decodes.insert(decode_engine[pair.decode_position as usize]);
                }
            }
            (on_prefills, on_decodes)
        };
        let (both, second) = (BTreeSet::from([0, 1]), BTreeSet::from([1]));
        assert_eq!(landed(&[]), (both.clone(), both.clone()), "nobody tried");
        assert_eq!(
            landed(&[tried_prefill]),
            (second.clone(), both.clone()),
            "the tried prefill's engine steps aside, and no decode does"
        );
        assert_eq!(
            landed(&[tried_decode]),
            (both, second.clone()),
            "the tried decode's engine steps aside, and no prefill does"
        );
        assert_eq!(
            landed(&[tried_prefill, tried_decode]),
            (second.clone(), second),
            "a tried pair moves on both sides"
        );
    }

    /// The decode asked again behind a held prefill leaves every rank of
    /// the engine its retry tried, as the pair's first decision does.
    #[test]
    fn the_decode_asked_again_leaves_every_rank_of_the_engine_it_tried() {
        use std::collections::BTreeSet;

        let (prefills, _) = ranked("prefill", WorkerType::Prefill, &[1]);
        let (decodes, decode_engine) = ranked("decode", WorkerType::Decode, &[2, 2]);
        let classes = every(prefills.len(), decodes.len() as u32);
        let held = prefill_alone(worker_id(prefills[0].as_ref()));
        // The first engine's second rank was tried.
        let tried = [worker_id(decodes[1].as_ref())];
        let host = Host::over(hearing(), prefills, decodes);
        let landed = |tried: &[WorkerId]| {
            let mut on_decodes = BTreeSet::new();
            for seed in 0..32 {
                let Asks(mut request, lists, _) = host.asks(&classes, true);
                request.facts.seed = seed;
                request.facts.attempt = 1;
                request.facts.tried = tried;
                let now = Instant::now();
                let again = lists.decide(&host.engine, None, now, &mut request, Some(&held));
                for target in named(&again) {
                    let pair = target.pair.expect("a pair");
                    on_decodes.insert(decode_engine[pair.decode_position as usize]);
                }
            }
            on_decodes
        };
        assert_eq!(landed(&[]), BTreeSet::from([0, 1]), "nobody tried");
        assert_eq!(
            landed(&tried),
            BTreeSet::from([1]),
            "the tried decode's engine"
        );
    }

    /// A host of four prefills and four decodes under the stock policies
    /// `prefill` and `decode` of a pair's two sides.
    fn host_under(prefill: &str, decode: &str) -> Host {
        let document = PolicySpec::pair_presets(prefill, decode).expect("the pair presets");
        host_of(document, 4, 4)
    }

    /// A host of `prefills` and `decodes` workers under `document`.
    fn host_of(mut document: PolicySpec, prefills: usize, decodes: usize) -> Host {
        document.generation = 3;
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        Host::under((spec, Arc::default()), prefills, decodes)
    }

    /// A prompt of four whole blocks as its host hashes it, its tokens
    /// counted from `first`.
    fn prompt(host: &Host, first: u32) -> PromptHashes {
        let tokens: Vec<u32> = (first..first + 64).collect();
        let spec = host.engine.spec();
        let seeds = HashSeeds::PLAIN;
        let hashes = PromptHashes::under_document(spec, Some(16), Some(&tokens), None, seeds);
        let mut read = RequestFacts::default();
        hashes.write_to(&mut read);
        assert_eq!(read.block_hashes.len(), 4, "four hashed blocks");
        hashes
    }

    /// One decision of a pair for the prompt `hashes`, drawing by `seed`:
    /// the pair's first, or the decode's again behind `held`.
    fn decide_prompt(
        host: &Host,
        hashes: &PromptHashes,
        seed: u64,
        held: Option<&Target>,
    ) -> Decision {
        let classes = every(host.prefills.len(), host.decodes.len() as u32);
        let Asks(mut request, lists, _) = host.asks(&classes, false);
        request.facts = RequestFacts { seed, ..facts() };
        hashes.write_to(&mut request.facts);
        lists.decide(&host.engine, None, Instant::now(), &mut request, held)
    }

    /// Under `cache_aware` on both sides, a prompt sent to a pair goes to
    /// that pair again under any draw: each side reads its own list's
    /// column of who was sent the prompt, and its pick is written behind
    /// it. Prompts nobody was sent are placed by their draws, on more than
    /// one pair.
    #[test]
    fn a_prompt_sent_to_a_pair_goes_to_that_pair_again() {
        let host = host_under("cache_aware", "cache_aware");
        let sent = prompt(&host, 1_000);
        let first = host.picked(&decide_prompt(&host, &sent, 1, None));
        for seed in 2..34 {
            let again = host.picked(&decide_prompt(&host, &sent, seed, None));
            assert_eq!(again, first, "seed {seed}");
        }
        let fresh: BTreeSet<(usize, usize)> = (0..32_u32)
            .map(|at| {
                let unsent = prompt(&host, 100_000 + 1_000 * at);
                host.picked(&decide_prompt(&host, &unsent, u64::from(at), None))
            })
            .collect();
        assert!(fresh.len() > 1, "{fresh:?}");
    }

    /// The decode asked again behind a held prefill reads the decodes'
    /// column: it goes to the decode its prompt was sent to, under any
    /// draw.
    #[test]
    fn the_decode_asked_again_goes_to_the_decode_its_prompt_was_sent_to() {
        let host = host_under("cache_aware", "cache_aware");
        let sent = prompt(&host, 1_000);
        let (prefill, decode) = host.picked(&decide_prompt(&host, &sent, 1, None));
        let held = prefill_alone(worker_id(host.prefills[prefill].as_ref()));
        for seed in 2..34 {
            let again = decide_prompt(&host, &sent, seed, Some(&held));
            assert_eq!(host.picked(&again), (prefill, decode), "seed {seed}");
        }
    }

    /// A side reads its own list's column and no other's: with
    /// `cache_aware` on the prefills and `round_robin` on the decodes, a
    /// prompt's prefill stays and its decodes take their turns.
    #[test]
    fn a_side_that_reads_no_affinity_keeps_its_own_rule() {
        let host = host_under("cache_aware", "round_robin");
        let sent = prompt(&host, 1_000);
        let (prefill, _) = host.picked(&decide_prompt(&host, &sent, 1, None));
        let mut decodes = BTreeSet::new();
        for seed in 2..10 {
            let again = host.picked(&decide_prompt(&host, &sent, seed, None));
            assert_eq!(again.0, prefill, "seed {seed}");
            decodes.insert(again.1);
        }
        assert_eq!(decodes.len(), 4, "{decodes:?}");
    }

    /// Asserts that under `host` a prompt's decode stays while its prefills
    /// take their turns, and that the decode asked again behind a held
    /// prefill is the prompt's too.
    fn the_decode_stays_and_the_prefills_turn(host: &Host) {
        let sent = prompt(host, 1_000);
        let (prefill, decode) = host.picked(&decide_prompt(host, &sent, 1, None));
        let held = prefill_alone(worker_id(host.prefills[prefill].as_ref()));
        let mut prefills = BTreeSet::from([prefill]);
        for seed in 2..10 {
            let again = host.picked(&decide_prompt(host, &sent, seed, None));
            assert_eq!(again.1, decode, "seed {seed}");
            prefills.insert(again.0);
            let behind = decide_prompt(host, &sent, seed, Some(&held));
            assert_eq!(host.picked(&behind), (prefill, decode), "seed {seed}");
        }
        assert_eq!(prefills.len(), 4, "{prefills:?}");
    }

    /// The mirror of the case above, where the decode alone reads: with
    /// `round_robin` on the prefills and `cache_aware` on the decodes, a
    /// prompt's decode stays and its prefills take their turns. The
    /// decode's pick is written and none for the prefill.
    #[test]
    fn a_decode_that_reads_alone_keeps_its_prompts_decode() {
        the_decode_stays_and_the_prefills_turn(&host_under("round_robin", "cache_aware"));
    }

    /// A binding of the decode phase alone listed ahead of the pair's takes
    /// a decode request asked alone and leaves the pair its decode side.
    /// The decodes' list keeps the record that side reads where the first
    /// binding reads none, so a prompt's decode stays there too.
    #[test]
    fn a_decode_binding_listed_ahead_leaves_the_pair_its_record() {
        let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
        let mut document = pair.expect("the pair presets");
        let stock = PolicySpec::preset("round_robin").expect("a preset");
        let mut ahead = stock.bindings[0].clone();
        ahead.matcher.phase = vec![Phase::Decode];
        ahead.profile = Some("ahead".into());
        document.bindings.insert(0, ahead);
        let turns = stock.profiles.into_values().next();
        let turns = turns.expect("the preset's profile");
        document.profiles.insert("ahead".into(), turns);
        the_decode_stays_and_the_prefills_turn(&host_of(document, 4, 4));
    }

    /// Each side's column is its own list's, whatever the lists' sizes:
    /// over two prefills and four decodes a prompt sent to a pair goes to
    /// that pair again under any draw.
    #[test]
    fn a_prompt_stays_with_its_pair_over_lists_of_two_sizes() {
        let pair = PolicySpec::pair_presets("cache_aware", "cache_aware");
        let host = host_of(pair.expect("the pair presets"), 2, 4);
        let sent = prompt(&host, 1_000);
        let first = host.picked(&decide_prompt(&host, &sent, 1, None));
        for seed in 2..34 {
            let again = host.picked(&decide_prompt(&host, &sent, seed, None));
            assert_eq!(again, first, "seed {seed}");
        }
    }
}
