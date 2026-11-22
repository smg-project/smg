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
//! **The lists.** A host hands its two lists for one decision
//! ([`PairLists`]): the engine's value of each, the pair class of every
//! worker, and its own reads of its admission. A decision writes one
//! record for each worker of the prefill list and of the decode list, each
//! in its list's order, as the workers stand at that call
//! ([`PoolRecords::write`](crate::engine::records::PoolRecords::write)).
//!
//! **The records.** [`stamp`] writes on them what a pair reads beside a
//! single: each record's pair class, and its side's admission bit as the
//! host reads it at that instant ([`PairAdmission`]). The stock pair
//! document filters by both bits (`has-capacity`, `has-room`).
//!
//! **The words.** A side is told its issue once: with the prefill's
//! ([`issue`]), or, where the request sends its decode later, at the decode
//! call ([`issue_decode`]) or at the dispatch that sends the decode with no
//! call ([`lease_events::pair`]). A side's dispatch and its end are told under that
//! side's lease. It names the worker the side booked at its issue until
//! the side's dispatch, and the worker it went to from there.
//!
//! **The answer.** The engine asks through [`decided`], for a pair's first
//! decision and for its decode asked again: the decision, with the lease a
//! pick issued.
//!
//! No function here takes a lock. A decision's records are written into
//! buffers of the calling thread, so it allocates nothing once they have
//! grown to the lists. A stage that hears a word takes what it takes of
//! its own state.

use std::{
    cell::RefCell,
    fmt,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::load::state::LoadSnapshot;

use super::{single, PolicyEngine, PoolMembers};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{shed, Decision, ShedReason, SUBSET_HINT, WIRES},
        decision_engine::{DecideOutcome, HeldDecision, Leased, Request},
        framework::{Engine, PairPools, Scratch},
        lease_events,
        pair_classes::PartnerClasses,
        records::{DecisionClock, Live},
        view::{Candidate, Constraints, Phase, PoolFacts, RequestFacts, Session, WorkerId},
    },
    stages::Lifecycle,
    store::leases::DISPATCH_TTL,
    tracking::lease::{Lease, LeaseId, LeaseKind, Side},
};

/// The filter that empties a pair's prefill list where every prefill the
/// request could take is at its admission limit.
const HAS_CAPACITY: &str = "has-capacity";

/// Whether `decision` sheds a pair because no prefill it could take has
/// admission capacity: its prefill list was emptied by the capacity bit its
/// host stated ([`PairAdmission::has_capacity`]). Such a request is not
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

/// A pair's two lists as their host holds them for one decision: the
/// engine's value of each list, the pair class of every worker of the two,
/// and the host's reads of its own admission. A decision names a prefill by
/// its place in the prefill list and a decode by its place in the decode
/// list, so the host finds both workers with no lookup by id.
#[derive(Clone, Copy, Debug)]
pub struct PairLists<'a> {
    /// The prefills.
    pub prefill: &'a PoolMembers,
    /// The decodes.
    pub decode: &'a PoolMembers,
    /// The pair class of each worker by its place in its list, and the
    /// table of the classes that pair.
    pub classes: &'a PartnerClasses,
    /// The host's reads of its prefill limit and of its decode rooms.
    pub admission: &'a dyn PairAdmission,
}

impl PairLists<'_> {
    /// One decision for a pair at `now`, over records written for it from
    /// both lists' worker objects and from `loads`, the host's snapshot of
    /// its workers' reports, with each record's pair class and its side's
    /// admission bit as the host reads it at this call. With no `held`
    /// prefill it is the pair's first decision: the engine's pair entry.
    /// With one it is the decode decided again behind that prefill's
    /// dispatch: the engine's decode entry over its partners.
    ///
    /// `request` is the pair's, as its host made it for the prefill phase.
    /// This call's instant is written into its facts: `now` on `clock`, the
    /// clock the records' sample instants stand on.
    ///
    /// Prefills that are all at their admission limit shed a first
    /// decision, which reads as a request that waits ([`prefills_full`]).
    pub fn decide(
        &self,
        spec: &AppliedSpec,
        loads: Option<&Arc<LoadSnapshot>>,
        clock: &DecisionClock,
        now: Instant,
        request: &mut Request<'_>,
        held: Option<WorkerId>,
    ) -> Decision {
        // One value for both lists' records: this call's instant on the
        // clock and the host's snapshot, as the host holds it. No session
        // key's text is handed to it: a pair's records carry no count of a
        // key's requests.
        let live = Live::at(spec.engine(), clock, now, loads, None);
        request.facts.now_micros = live.now_micros();
        let (facts, cx) = (&request.facts, &request.constraints);
        let decide = |buffers: &mut Buffers| {
            let [prefills, decodes] = &mut buffers.rows;
            let (workers, records) = (self.prefill.workers(), self.prefill.records(spec, clock));
            let (prefills, prefill_facts) = records.write(workers, &live, prefills);
            let (workers, records) = (self.decode.workers(), self.decode.records(spec, clock));
            let (decodes, decode_facts) = records.write(workers, &live, decodes);
            stamp(prefills, decodes, self.classes, self.admission);
            let pools = PairPools {
                prefills,
                decodes,
                table: self.classes.table(),
                ..PairPools::default()
            };
            let (engine, scratch) = (spec.engine(), &mut buffers.scratch);
            match held {
                None => decide_records(engine, facts, cx, &prefill_facts, pools, scratch),
                Some(held) => {
                    decide_decode_records(engine, facts, cx, &decode_facts, pools, held, scratch)
                }
            }
        };
        BUFFERS.with(|buffers| match buffers.try_borrow_mut() {
            Ok(mut buffers) => decide(&mut buffers),
            // A stage asked for a decision from inside its own, on this
            // thread: that decision takes buffers of its own.
            Err(_) => decide(&mut Buffers::default()),
        })
    }
}

/// What one thread keeps for its pair decisions: each list's records, the
/// prefills' first, and the engine's scratch.
#[derive(Debug, Default)]
struct Buffers {
    rows: [Vec<Candidate>; 2],
    scratch: Scratch,
}

thread_local! {
    /// The calling thread's own buffers, so a pair's decision allocates
    /// nothing after a thread's first decisions over lists of a size.
    static BUFFERS: RefCell<Buffers> = RefCell::new(Buffers::default());
}

/// Writes on the two lists' records what a pair's decision reads beside a
/// single's: each record's pair class, a prefill's capacity bit and a
/// decode's room bit, as `admission` reads them now. A record past the
/// classes takes class `0`, which pairs with nobody.
pub(super) fn stamp(
    prefills: &mut [Candidate],
    decodes: &mut [Candidate],
    classes: &PartnerClasses,
    admission: &dyn PairAdmission,
) {
    let class = |classes: &[u32], at: usize| classes.get(at).copied().unwrap_or(0);
    for (at, row) in prefills.iter_mut().enumerate() {
        row.pair_class = class(classes.prefills(), at);
        row.has_capacity = admission.has_capacity(at);
    }
    for (at, row) in decodes.iter_mut().enumerate() {
        row.pair_class = class(classes.decodes(), at);
        row.has_room = admission.has_room(at);
    }
}

/// How a pair's request is shed where it states what a pair's lists are not
/// narrowed by, and the stage it is shed under: a subset hint, a hard
/// demand, or places of its list. A pair's records keep no demand fact, so
/// such a request is shed and never placed past its demand. A hint of some
/// records is declined: no stage here takes a part of a list. So are
/// places ([`single::PLACES`]): a pair's sides do not read them.
fn not_narrowed_by(cx: &Constraints<'_>) -> Option<(ShedReason, &'static str)> {
    match cx.subset_hint {
        Some(0) => Some((ShedReason::NoEndpoints, SUBSET_HINT)),
        Some(_) => Some((ShedReason::PolicyDeclined, SUBSET_HINT)),
        None if !cx.demands.is_none() => Some((ShedReason::NoEndpoints, WIRES)),
        None if cx.places.is_some() => Some((ShedReason::PolicyDeclined, single::PLACES)),
        None => None,
    }
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
/// facts. The request's constraints are read by both sides.
///
/// One side alone is no pair: where either list holds no record the engine
/// answers over no candidate. A request that states a hard demand or a
/// subset hint of none is shed under the stage's name the engine over a
/// store sheds it by, and a hint of some records is declined under it.
fn decide_records(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
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
    if let Some((reason, stage)) = not_narrowed_by(cx) {
        return shed(reason, stage, facts.seed, pool.generation);
    }
    engine.decide_pair(facts, pool, pools, cx, cx, scratch)
}

/// What a pair's first pick ([`PairLists::decide`]) issues: the prefill
/// side's lease under `id`, which is the answer's lease, and the request's
/// session under the decode phase, which its guard keeps for the decode
/// side's words. The stages hear the prefill's issue, and the decode's
/// behind it unless the request sends its decode `later`: such a side
/// stands chosen and is told at its own issue ([`issue_decode`],
/// [`lease_events::pair`]).
///
/// `None` for a decision that picks no pair: it issues nothing.
pub(super) fn issue(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    id: LeaseId,
    later: bool,
) -> Option<(Lease, Session)> {
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
    };
    let decode = lease_events::decode_as_issued(&lease, first, session)?;
    engine.on_decision_issued(&lease, decision);
    if !later {
        engine.on_decision_issued(&decode, decision);
    }
    Some((lease, session))
}

/// The decode asked again over the records of the two lists, stamped
/// ([`stamp`]): the engine's decode entry over the partners of the held
/// `prefill`. `pool` holds the decode list's facts. The entry is handed no
/// first decode here, so it decides again under every profile.
fn decide_decode_records(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    prefill: WorkerId,
    scratch: &mut Scratch,
) -> Decision {
    let facts = RequestFacts {
        phase: Phase::Decode,
        ..*facts
    };
    if pools.prefills.is_empty() || pools.decodes.is_empty() {
        let none = PoolFacts {
            generation: pool.generation,
            ..PoolFacts::default()
        };
        let no_lists = PairPools::default();
        return engine.decide_decode(&facts, &none, no_lists, prefill, None, cx, scratch);
    }
    if let Some((reason, stage)) = not_narrowed_by(cx) {
        return shed(reason, stage, facts.seed, pool.generation);
    }
    engine.decide_decode(&facts, pool, pools, prefill, None, cx, scratch)
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
    let first = decision.pick()?.targets.first()?;
    let decode = Lease {
        tokens: facts.prompt_tokens,
        ..lease_events::decode_as_issued(prefill, first, decode_session(facts))?
    };
    let issued = Lease {
        issued_micros: now_micros,
        ..decode
    };
    engine.on_decision_issued(&issued, decision);
    Some(decode)
}

/// One decision of a pair by `own`, the engine of a gateway's own process,
/// over the `lists` its host handed, as the host takes it. With no `held`
/// decision it is the pair's first, and a pick is issued under the engine's
/// next id ([`issue`]). With the view of the guard that holds the first it
/// is the decode asked again behind the prefill that view names, and a pick
/// issues the decode side under the held lease, dated at this call
/// ([`issue_decode`]). A lease's dispatch is due as a single's.
///
/// A request held to wires is its host's to hand lists for. Where a worker
/// of either list is dialed on a wire the request does not state, the
/// request is declined under the wires' name, as a single is.
pub(super) fn decided(
    own: &PolicyEngine,
    lists: &PairLists<'_>,
    request: &mut Request<'_>,
    held: Option<&HeldDecision<'_>>,
    now: Instant,
) -> DecideOutcome {
    let wires = request.constraints.wire_modes;
    let dialed = lists.prefill.wires().union(lists.decode.wires());
    if !wires.is_empty() && !wires.contains(dialed) {
        return DecideOutcome {
            decision: single::declined(own, request, WIRES),
            lease: None,
            decode_session: Session::NONE,
        };
    }
    let (spec, later) = (&own.spec, request.constraints.decode_later);
    let prefill = held.map(|held| held.target.id);
    let decision = lists.decide(spec, None, &own.clock, now, request, prefill);
    let (engine, facts) = (spec.engine(), &request.facts);
    let issued = match held {
        // A shed writes no word that every pick shares.
        None => decision.pick().and_then(|_| {
            let id = LeaseId(own.leases.0.fetch_add(1, Ordering::Relaxed) + 1);
            issue(engine, facts, &decision, id, later)
        }),
        Some(held) => issue_decode(engine, facts, &decision, held.lease, facts.now_micros)
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
    use std::{sync::Arc, time::Duration};

    use parking_lot::Mutex;
    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        test_support::no_health_check,
        worker::{WorkerSpec, WorkerStatus, WorkerType},
    };

    use super::*;
    use crate::{
        document::spec::PolicySpec,
        engine::{
            decision::{Pair, Pick, Target, Targets},
            decision_engine::{DecisionEngine, RequestMaker, WorkerLists},
            demands::Demands,
            framework::{Filter, Picker, TieBand},
            records::worker_id,
            view::{Health, Identity, Role, WireModes},
        },
        stages::{Best, HasCapacity, StageRegistry},
        store::{Dispatch, GatewayId, Said},
        tracking::{
            lease::{Outcome, ReleaseReason, TransportClass},
            DecisionGuard,
        },
    };

    const MODEL: &str = "model-a";

    /// One word a stage heard of a lease: what was said, the side, the
    /// worker the lease books, the index it names, its session, its issue
    /// instant, and the worker the word names beside the lease (the pick's
    /// first target at an issue, the target at a dispatch, the worker that
    /// served at a completion).
    type Word = (&'static str, Side, u64, u8, Session, u64, u64);

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
        Candidate {
            position,
            id: WorkerId(id),
            role,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            has_capacity: true,
            has_room: true,
            ..Candidate::default()
        }
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
            prefills,
            decodes,
            table: classes.table(),
            ..PairPools::default()
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

    /// The pair decided now over the two lists' records.
    fn decide_over(
        spec: &AppliedSpec,
        facts: &RequestFacts<'_>,
        prefills: &[Candidate],
        decodes: &[Candidate],
        classes: &PartnerClasses,
    ) -> Decision {
        decide_records(
            spec.engine(),
            facts,
            &Constraints::default(),
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
                cx,
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
        assert!(matches!(decision, Decision::Shed(_)), "{decision:?}");
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
            &Constraints::default(),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            WorkerId(1),
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
            &Constraints::default(),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            WorkerId(5),
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

    /// A host's two lists of ready workers with the engine's value of each,
    /// under the engine of the stock pair document and its clock.
    struct Host {
        engine: PolicyEngine,
        heard: Arc<Mutex<Vec<Word>>>,
        clock: DecisionClock,
        prefills: Arc<[Arc<dyn Worker>]>,
        decodes: Arc<[Arc<dyn Worker>]>,
        members: [PoolMembers; 2],
    }

    /// A pair's request stated ahead, with the lists its host hands for it.
    struct Asks<'a>(Request<'a>, PairLists<'a>);

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
    }

    impl Host {
        fn of(prefills: usize, decodes: usize) -> Self {
            let side = |name: &str, worker_type, count| -> Arc<[Arc<dyn Worker>]> {
                let url = |at: usize| format!("grpc://{name}-{at}:9000");
                (0..count).map(|at| worker(url(at), worker_type)).collect()
            };
            let (spec, heard) = hearing();
            let engine = PolicyEngine::new(spec);
            let prefills = side("prefill", WorkerType::Prefill, prefills);
            let decodes = side("decode", WorkerType::Decode, decodes);
            let members = [
                engine.pool_members(&prefills),
                engine.pool_members(&decodes),
            ];
            Self {
                engine,
                heard,
                clock: DecisionClock::new(),
                prefills,
                decodes,
                members,
            }
        }

        /// The host's keyed request for a pair over its lists, every pair
        /// open, its decode sent `later` or with its prefill.
        fn asks<'a>(&'a self, classes: &'a PartnerClasses, later: bool) -> Asks<'a> {
            let [prefill, decode] = &self.members;
            let lists = PairLists {
                prefill,
                decode,
                classes,
                admission: &OPEN,
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
            Asks(request, lists)
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
        /// behind the `held` prefill.
        fn decide(
            &self,
            admission: &dyn PairAdmission,
            seed: u64,
            held: Option<WorkerId>,
        ) -> Decision {
            let [prefill, decode] = &self.members;
            let classes = every(self.prefills.len(), self.decodes.len() as u32);
            let lists = PairLists {
                prefill,
                decode,
                classes: &classes,
                admission,
            };
            let mut request = Request {
                id: b"",
                facts: RequestFacts { seed, ..facts() },
                constraints: Constraints::default(),
            };
            let (spec, now) = (self.engine.spec(), Instant::now());
            let decision = lists.decide(spec, None, &self.clock, now, &mut request, held);
            assert_eq!(request.facts.now_micros, self.clock.micros(now));
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
                    decode_dp_rank: 0,
                }),
            });
        }
        Decision::Pick(Pick {
            targets,
            ..Pick::default()
        })
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
            decode_session: session,
            index: 1,
            target: &taken,
            decode_chosen: false,
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
            decode_session: session,
            index: 0,
            target: &taken,
            decode_chosen: false,
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
            decode_session: session,
            index: 1,
            target: &taken,
            decode_chosen: true,
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
            decode_session: session,
            index: 0,
            target: &taken,
            decode_chosen: true,
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
}
