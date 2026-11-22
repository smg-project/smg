//! A single request's whole decision in a gateway's own process.
//!
//! **The short path, in front** ([`alone`]). Where the document decides the
//! request's place by one pick of its picker
//! ([`Place::Alone`](crate::engine::Place::Alone)), the decision is that
//! pick: the picker's own rule over the list, with one read of each worker
//! the rule asks. A rotation reads the worker its turn lands on, and a scan
//! reads the list in order up to the first eligible worker. It writes no
//! record and takes no lock. It answers a plain request ([`Plain`]) over a
//! list where the rule finds an eligible worker, and it reads no fact of a
//! request: the engine asks it before the request is made, so such a
//! request is never made.
//!
//! **A key's rows, between** ([`keys::decide`]). A request that is made
//! under a sessions rule is decided by the row its key names: its key's
//! pin, which is one worker's record. It answers where that row says what
//! the whole pass would, and nothing where it does not.
//!
//! **The whole pass, behind them** ([`decide`]). Every other request is made
//! and put to the engine's whole decision over records written for it from
//! the gateway's own data ([`records::decide`]): a retry that names the
//! workers it tried, a list the picker's turns find no eligible worker of,
//! and a place whose profile scores. The binding's rule for a pool every
//! worker of which is vetoed, and the test of a tried worker, so have one
//! home.
//!
//! **What is refused by name.** The list a host hands is its request's own:
//! every worker of it is dialed on a wire the request states. A list that is
//! not, a subset hint and a hard demand are declined under the stage that
//! would have run them.
//!
//! **A request held to places of its list**
//! ([`Constraints::places`](crate::engine::view::Constraints::places)) is
//! made and decided over the list's own records: it is not plain, the
//! keyed decision leaves it to the whole pass, and that pass opens with
//! the places. A pair that states places is declined ([`PLACES`]).

use std::{
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::{load::state::LoadSnapshot, runtime::RoutingState};

use super::{keys, overlap, PolicyEngine, PoolMembers};
use crate::{
    engine::{
        decision::{Decision, Shed, ShedReason, Trace, SUBSET_HINT, WIRES},
        decision_engine::{AlonePick, DecideOutcome, Leased, Plain, Request},
        records,
        sticky_sessions::SessionVerdict,
        view::{Phase, Session},
    },
    stages::Lifecycle,
    tracking::lease::{EventKind, EventSet, Lease, LeaseId, LeaseKind, Side, DISPATCH_TTL},
};

/// The stage a request that states a hard demand is declined under: no
/// stage of this engine reads a worker's facts for one.
const DEMANDS: &str = "demands";

/// The stage a pair held to places is declined under
/// ([`Constraints::places`](crate::engine::view::Constraints::places)):
/// one set of places cannot hold a pair's two lists.
pub(super) const PLACES: &str = "places";

/// The crate's word, in a shed's `emptied_by`, for a pool no stage emptied.
const NO_STAGE: &str = "none";

/// The short path: the place in `members`, the list its host handed, of
/// the worker picked for a plain single request of `model`, and the profile
/// whose picker picked it ([`AlonePick`]). `None` where the request takes
/// the whole pass ([`decide`]): it is no single request, its place is not
/// decided by one pick of a picker, a worker of the list is not dialed on a
/// wire it states, or the picker's rule finds no eligible worker.
///
/// It ends in the place picked. It reads no fact of a request, so none is
/// made ahead of it, and nothing a host is handed of a pick is built here.
/// Under a document that takes no word on a lease the place is the whole
/// answer. Under any other the pick is noted ([`noted`]).
pub(super) fn alone(
    own: &PolicyEngine,
    model: &str,
    plain: Plain,
    members: &PoolMembers,
) -> Option<AlonePick> {
    if plain.phase != Phase::Single {
        return None;
    }
    // The whole pass declines such a list by name.
    if !plain.wires.is_empty() && !plain.wires.contains(members.wires()) {
        return None;
    }
    let profile = members.alone(&own.spec, model)?;
    let position = own.pick(profile, members, |state: RoutingState| state.eligible())?;
    Some(AlonePick {
        place: u32::try_from(position).ok()?,
        profile: profile.0,
        generation: own.spec.generation(),
    })
}

/// The short path's pick under a document that takes a word on a lease,
/// as its host takes it: the decision that names the worker at the place
/// picked, alone ([`AlonePick::decision`]), and its lease, issued here as a
/// whole pass's is ([`issued`]). The lease states the size `plain` states
/// and no session, and is dated at `now` on the engine's clock. The issue
/// is told where a stage takes it. `None` where the list holds no worker at
/// the place.
pub(super) fn noted(
    own: &PolicyEngine,
    pick: AlonePick,
    members: &PoolMembers,
    plain: Plain,
    now: Instant,
) -> Option<DecideOutcome> {
    let position = pick.place as usize;
    let worker = members.workers().get(position)?;
    // The list's records name each worker once a list: a pick reads the id
    // there and hashes no address.
    let id = *members.records(&own.spec, &own.clock).ids().get(position)?;
    let dp_rank = records::worker_rank(worker.as_ref());
    let lease = Lease {
        id: LeaseId(own.leases.0.fetch_add(1, Ordering::Relaxed) + 1),
        kind: LeaseKind::Credit,
        side: Side::Single,
        target_index: 0,
        booked: id,
        generation: pick.generation,
        session: Session::NONE,
        session_verdict: SessionVerdict::None,
        issued_micros: own.clock.micros(now),
        tokens: plain.prompt_tokens,
    };
    let decided = DecideOutcome {
        decision: pick.decision(id, dp_rank),
        lease: Some(Leased {
            lease,
            due: now + DISPATCH_TTL,
        }),
        decode_session: Session::NONE,
    };
    if own.hears.contains(EventKind::Issued) {
        own.spec
            .engine()
            .on_decision_issued(&lease, &decided.decision);
    }
    Some(decided)
}

/// One made request's decision over `members`, the list its host handed,
/// at `now`: by the rows its key names where they answer
/// ([`keys::decide`]), and else the engine's whole decision over records
/// written for the request from the list's workers. Each worker's report
/// is read from `loads`, the host's load snapshot for this decision. A
/// request that states what no stage here runs is declined under that
/// stage's name.
pub(super) fn decide(
    own: &PolicyEngine,
    request: &mut Request<'_>,
    members: &PoolMembers,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
) -> Decision {
    let spec = &own.spec;
    let cx = request.constraints;
    if cx.subset_hint.is_some() {
        return declined(own, request, SUBSET_HINT);
    }
    if !cx.demands.is_none() {
        return declined(own, request, DEMANDS);
    }
    if !cx.wire_modes.is_empty() && !cx.wire_modes.contains(members.wires()) {
        return declined(own, request, WIRES);
    }
    let records = members.records(spec, &own.clock);
    let facts = &mut request.facts;
    // One value for every record of the decision: its instant, the host's
    // snapshot as the host holds it, and the key's text for a stage that
    // counts the key's requests on a worker.
    let key = facts.session_key.plain;
    let live = records::Live::at(spec.engine(), &own.clock, now, loads, key);
    let (engine, workers) = (spec.engine(), members.workers());
    // A keyed request is decided by the rows its key names, where they say
    // what the whole pass would: its key's pin.
    if let Some(keyed) = keys::decide(engine, records, workers, &live, facts, &cx) {
        return keyed;
    }
    // A profile whose stages read who holds the prompt is named its rows
    // from the gateway's own data, at the decision's instant the keyed
    // entry wrote, and its pick is recorded behind it; any other passes
    // straight to the whole pass.
    overlap::decide(
        &own.affinity,
        members.affinity(),
        spec,
        workers,
        records.ids(),
        facts,
        |facts| records::decide(engine, records, workers, &live, facts, &cx),
    )
}

/// The answer for a request whose host handed no list: a pool of no worker,
/// which no stage emptied.
pub(super) fn over_none(own: &PolicyEngine, request: &Request<'_>) -> Decision {
    shed(own, request, ShedReason::NoEndpoints, NO_STAGE)
}

/// The answer for a request that states what no stage of this engine runs,
/// under the name of the stage that would have run it. Cold: [`decide`]'s
/// tests are then predicted to pass. Without it, one test more in front
/// of the whole pass kept the pass from being inlined, 43 instructions a
/// decision.
#[cold]
pub(super) fn declined(own: &PolicyEngine, request: &Request<'_>, stage: &'static str) -> Decision {
    shed(own, request, ShedReason::PolicyDeclined, stage)
}

fn shed(
    own: &PolicyEngine,
    request: &Request<'_>,
    reason: ShedReason,
    emptied_by: &'static str,
) -> Decision {
    Decision::Shed(Shed {
        reason,
        emptied_by,
        retry_after_secs: 1,
        wait_allowed: false,
        trace: Trace {
            seed: request.facts.seed,
            generation: own.spec.generation(),
            ..Trace::default()
        },
    })
}

/// The decision as its host takes it. Under a document that takes a word
/// on a lease a pick is issued before the answer leaves: its lease takes
/// the next id, books the pick's first target and is dated at the
/// decision's instant, and the stages that take the issue hear of it here,
/// on the calling thread and inside the decision's own catch. From the
/// answer the lease is its guard's: nothing of it is filed.
///
/// Under a document that takes no word a pick has no lease: no id is
/// taken, and its guard owes the engine nothing.
pub(super) fn issued(
    own: &PolicyEngine,
    decision: Decision,
    request: &Request<'_>,
    now: Instant,
) -> DecideOutcome {
    let facts = &request.facts;
    let leased = own.hears.intersects(EventSet::LEASE);
    let noted = decision.pick().filter(|_| leased);
    let lease = noted.and_then(|pick| {
        let first = pick.targets.first()?;
        Some(Lease {
            id: LeaseId(own.leases.0.fetch_add(1, Ordering::Relaxed) + 1),
            kind: LeaseKind::Credit,
            side: Side::Single,
            target_index: 0,
            booked: first.id,
            generation: pick.trace.generation,
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: facts.now_micros,
            tokens: facts.prompt_tokens,
        })
    });
    let issue = own.hears.contains(EventKind::Issued);
    if let Some(lease) = lease.as_ref().filter(|_| issue) {
        own.spec.engine().on_decision_issued(lease, &decision);
    }
    DecideOutcome {
        decision,
        lease: lease.map(|lease| Leased {
            lease,
            due: now + DISPATCH_TTL,
        }),
        decode_session: Session::NONE,
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, sync::Arc, time::Duration};

    use parking_lot::Mutex;
    use smg_worker::{
        identity::basic::Worker, load::state::one_rank_load, test_support::ready_worker,
        worker::WorkerStatus,
    };

    use super::*;
    use crate::{
        document::spec::{AppliedSpec, Assignment, PolicySpec},
        engine::{
            decision::{LiftedVeto, Target, Targets, MAX_TARGETS},
            decision_engine::{
                Decided, DecisionEngine, DecisionRefused, RequestMaker, WorkerLists,
            },
            framework::{Picker, Place, PoolWord, TieBand},
            prompt_hashes::HashSeeds,
            records::worker_id,
            view::{Candidate, Constraints, Identity, RequestFacts, WireModes, WorkerId},
        },
        policies::random::Random,
        stages::registry::StageRegistry,
        store::GatewayId,
        tracking::{
            lease::{Membership, Outcome, ReleaseReason, TransportClass},
            DecisionGuard,
        },
    };

    const MODEL: &str = "model-a";

    fn preset(name: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
    }

    /// The preset `name` with no filter in any of its profiles: a valid
    /// document that weighs no worker's health.
    fn unfiltered(name: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
    }

    fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
        (0..workers)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect()
    }

    fn request<'a>() -> Request<'a> {
        Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed: 7,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        }
    }

    /// A request stated ahead, with the list its host hands for it, and how
    /// many times the engine made it.
    struct Asks<'a> {
        request: Request<'a>,
        members: Option<&'a PoolMembers>,
        loads: Option<Arc<LoadSnapshot>>,
        made: Cell<u32>,
    }

    impl<'a> Asks<'a> {
        fn of(request: Request<'a>, members: Option<&'a PoolMembers>) -> Self {
            Self {
                request,
                members,
                loads: None,
                made: Cell::new(0),
            }
        }
    }

    impl RequestMaker for Asks<'_> {
        fn model(&self) -> &str {
            self.request.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.made.set(self.made.get() + 1);
            self.request
        }

        fn plain(&self) -> Option<Plain> {
            self.request.plain()
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.members.map_or(WorkerLists::None, WorkerLists::Single)
        }

        fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
            self.loads.as_ref()
        }
    }

    /// What the one entry answers a request stated ahead.
    fn answered(engine: &PolicyEngine, asks: &Asks<'_>) -> Decided {
        let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
        decided.expect("an answer")
    }

    /// The whole decision the entry answers: no place.
    fn decided(engine: &PolicyEngine, asks: &Asks<'_>) -> DecideOutcome {
        answered(engine, asks).whole().expect("a whole decision").0
    }

    /// The place the entry answers: no whole decision.
    fn placed(engine: &PolicyEngine, asks: &Asks<'_>) -> AlonePick {
        match answered(engine, asks) {
            Decided::Alone(pick) => pick,
            Decided::Whole(decided, _) => panic!("a place, not {decided:?}"),
        }
    }

    /// Round robin's plain single request is one turn of the list's
    /// rotation, answered as the place in the list the turn lands on: no
    /// pass over the pool ran, no request was made, and no lease took an
    /// id, since no stage of the document takes a word of one. The decision
    /// a host writes of the place names that worker alone.
    #[test]
    fn round_robin_is_one_turn_of_the_list_answered_as_a_place() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = engine.pool_members(&list);
        let mut sized = request();
        sized.facts.prompt_tokens = 321;
        sized.facts.session_key = Identity::plain("a-key");
        let asks = Asks::of(sized, Some(&members));
        let mut seen = [0_u32; 4];
        for _ in 0..8 {
            let pick = placed(&engine, &asks);
            assert_eq!(pick.generation, 1);
            seen[pick.place as usize] += 1;
            let id = worker_id(list[pick.place as usize].as_ref());
            let decision = pick.decision(id, 3);
            let named = decision.pick().expect("a pick");
            assert_eq!(named.targets.len(), 1);
            assert!(named.trace.drew_first(), "no pass over the pool ran");
            assert_eq!(named.trace.seed, 0, "no seed was drawn");
            let target = named.targets.first().expect("the target");
            assert_eq!(
                (target.position, target.id, target.dp_rank),
                (pick.place, id, 3)
            );
        }
        assert_eq!(seen, [2; 4], "every worker takes its turn");
        assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
    }

    /// Passthrough's plain single request is answered as a place: the first
    /// worker of the list its host's test admits. That is the gateway's
    /// classic rule, the position of the first worker whose routing state
    /// the selection admits (`model_gateway/src/policies/passthrough.rs:48`
    /// at main `882be6f4f`). No pass over the pool ran, no request was made
    /// and no lease took an id. A list that admits no worker is the whole
    /// decision's: the request is made once.
    #[test]
    fn passthrough_is_the_first_admitted_worker_answered_as_a_place() {
        let engine = preset("passthrough");
        for workers in [1, 2, 8, 64] {
            let list = pool(workers);
            let members = engine.pool_members(&list);
            let asks = Asks::of(request(), Some(&members));
            for refused in 0..workers.min(12) {
                // Classic's rule, asked of the same worker objects.
                let classic = list.iter().position(|w| w.routing_state().eligible());
                assert_eq!(classic, Some(refused));
                let place = placed(&engine, &asks).place;
                assert_eq!(place as usize, refused, "{workers} workers");
                // The first is refused by its status alone, the others by overload.
                list[refused].set_overloaded(refused > 0);
                list[0].set_status(WorkerStatus::NotReady);
            }
            assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
            if workers <= 12 {
                // The whole decision's rule, not the first worker: a worker
                // that is not ready is never named, and the overload veto
                // is lifted off the workers it alone refuses.
                match decided(&engine, &asks).decision {
                    Decision::Shed(shed) => assert_eq!((workers, shed.emptied_by), (1, "healthy")),
                    Decision::Pick(pick) => {
                        assert_eq!(pick.trace.lifted, LiftedVeto::Overload);
                        assert!(pick.targets.iter().all(|target| target.position > 0));
                    }
                    Decision::Queue(_) => panic!("no queue is planned"),
                }
                assert_eq!(asks.made.get(), 1, "no worker is admitted: made once");
            }
        }
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
    }

    /// Under a document a stage of which takes a word on a lease, a plain
    /// request of a place one pick decides is still never made, and its
    /// pick is noted: a whole decision of one target under a lease that
    /// takes the next id, states the size the plain word states and no
    /// session, and is told to the stage that takes the issue.
    #[test]
    fn a_place_picked_under_a_document_that_takes_a_word_is_noted() {
        let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
        let place = engine.place(MODEL, Phase::Single);
        assert!(matches!(place, Place::Alone(_)), "{place:?}");
        let list = pool(4);
        let members = engine.pool_members(&list);
        let mut sized = request();
        sized.facts.prompt_tokens = 321;
        sized.facts.session_key = Identity::plain("a-key");
        let asks = Asks::of(sized, Some(&members));
        let (mut seen, mut last) = ([0_u32; 4], 0);
        for _ in 0..8 {
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            assert_eq!(pick.targets.len(), 1);
            assert!(pick.trace.drew_first(), "no pass over the pool ran");
            assert_eq!(pick.trace.seed, 0, "no seed was drawn");
            let target = pick.targets.first().expect("the target");
            let worker = &list[target.position as usize];
            assert_eq!(target.id, worker_id(worker.as_ref()));
            seen[target.position as usize] += 1;
            let lease = decided.lease.expect("a lease").lease;
            assert_eq!(
                (lease.side, lease.booked, lease.target_index),
                (Side::Single, target.id, 0)
            );
            assert_eq!((lease.tokens, lease.session), (321, Session::NONE));
            assert_eq!(lease.session_verdict, SessionVerdict::None);
            assert!(lease.id.0 > last, "each pick takes the next id");
            last = lease.id.0;
            assert_eq!(decided.decode_session, Session::NONE);
            assert_eq!(heard.lock().last(), Some(&("issued", target.id, 0)));
        }
        assert_eq!(seen, [2; 4], "every worker takes its turn");
        assert_eq!(heard.lock().len(), 8, "one issue a pick");
        assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
    }

    /// A request that is not plain, and a plain one the turn finds no
    /// worker for, is made once and put to the whole pass. Under a document
    /// that takes no word its pick has no lease either.
    #[test]
    fn a_request_the_one_pick_does_not_decide_is_made_once() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = engine.pool_members(&list);

        // A retry is not plain: it is made, once a decision.
        let tried = [worker_id(list[0].as_ref())];
        let mut retry = request();
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let whole = decided(&engine, &asks);
        assert!(whole.decision.pick().is_some());
        assert!(whole.lease.is_none(), "no stage takes a word of it");
        assert_eq!(asks.made.get(), 1);
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");

        // A plain request whose turns land on no eligible worker is made
        // for the whole pass, which answers by the binding's own rule.
        for worker in list.iter() {
            worker.set_overloaded(true);
        }
        let asks = Asks::of(request(), Some(&members));
        let _ = decided(&engine, &asks);
        assert_eq!(asks.made.get(), 1);
    }

    /// The entry reads its host's clock where a decision is dated and
    /// nowhere else: a place reads none, and a retry's whole pass, a profile
    /// that scores and a place noted for a stage that takes a word read it
    /// once each.
    #[test]
    fn a_place_reads_no_clock_and_a_dated_decision_reads_it_once() {
        let list = pool(4);
        let tried = [worker_id(list[0].as_ref())];
        let read = |engine: &PolicyEngine, retried: bool| {
            let members = engine.pool_members(&list);
            let mut stated = request();
            if retried {
                stated.facts.tried = &tried;
            }
            let asks = Asks::of(stated, Some(&members));
            let reads = Cell::new(0_u32);
            let decided = engine.decide_at(&asks, || {
                reads.set(reads.get() + 1);
                Instant::now()
            });
            let place = matches!(decided.expect("an answer"), Decided::Alone(_));
            (place, reads.get())
        };
        assert_eq!(read(&preset("round_robin"), false), (true, 0));
        assert_eq!(read(&preset("round_robin"), true), (false, 1));
        assert_eq!(read(&preset("least_load"), false), (false, 1));
        let (noting, _) = hearing_as(false, true, EventSet::LEASE);
        assert_eq!(read(&noting, false), (false, 1));
    }

    /// A place whose profile scores is never picked ahead of its request:
    /// the request is made for every decision, plain or not.
    #[test]
    fn a_place_that_scores_makes_its_request() {
        let engine = preset("least_load");
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        for made in 1..=3 {
            assert!(decided(&engine, &asks).decision.pick().is_some());
            assert_eq!(asks.made.get(), made);
        }
    }

    /// A place that scores by load reads each worker's report in the
    /// snapshot its host hands for the decision: the worker with the
    /// shortest queue is picked at every seed. Handed no snapshot, the same
    /// engine reads no report, and its picks spread over the list.
    #[test]
    fn a_place_that_scores_by_load_reads_the_reports_its_host_hands() {
        let engine = preset("least_load");
        let list = pool(4);
        let members = engine.pool_members(&list);
        let reports = (0..4)
            .map(|at| {
                let queued = if at == 2 { 0 } else { 90_000 };
                let url = format!("http://worker-{at}:8000");
                (url, one_rank_load(queued, 0.1, 1_000.0))
            })
            .collect();
        let picked_at = |loads: Option<Arc<LoadSnapshot>>, seed: u64| {
            let mut request = request();
            request.facts.seed = seed;
            let asks = Asks {
                loads,
                ..Asks::of(request, Some(&members))
            };
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        let snapshot = LoadSnapshot::from_loads_for_test(reports);
        for seed in 0..32 {
            assert_eq!(
                picked_at(Some(Arc::clone(&snapshot)), seed),
                2,
                "seed {seed}"
            );
        }
        let mut unread = [0_u32; 4];
        for seed in 0..64 {
            unread[picked_at(None, seed) as usize] += 1;
        }
        assert!(unread.iter().all(|&picks| picks > 0), "{unread:?}");
    }

    /// A request stated ahead is plain where it is a first attempt that
    /// asks nothing beyond a worker of its list, and says what a pick with
    /// no request made needs: its phase, its wires and its size. Each thing
    /// it states beyond that makes it a request the whole pass reads.
    #[test]
    fn a_request_stated_ahead_says_whether_it_is_plain() {
        let mut stated = request();
        stated.facts.prompt_tokens = 9;
        stated.constraints.wire_modes = WireModes::GRPC;
        assert_eq!(
            stated.plain(),
            Some(Plain {
                phase: Phase::Single,
                wires: WireModes::GRPC,
                prompt_tokens: 9,
            })
        );
        let tried = [WorkerId(1)];
        let rows = [0_u32];
        let mut retried = request();
        retried.facts.tried = &tried;
        let mut missed = request();
        missed.constraints.excluded = &tried;
        let mut hinted = request();
        hinted.constraints.subset_hint = Some(2);
        let mut lifted = request();
        lifted.constraints.overload_lifted = true;
        let mut later = request();
        later.constraints.decode_later = true;
        let mut masked = request();
        masked.constraints.stalled = &rows;
        let mut held = request();
        held.constraints.places = Some(&rows);
        for request in [retried, missed, hinted, lifted, later, masked, held] {
            assert_eq!(request.plain(), None);
        }
    }

    /// What the one turn does not decide goes to the whole pass over
    /// records: a retry never meets the worker it tried, and a pool whose
    /// every worker is overloaded is answered by the binding's own rule.
    #[test]
    fn a_retry_and_a_pool_no_turn_admits_are_put_to_the_whole_pass() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = engine.pool_members(&list);
        let tried = [worker_id(list[0].as_ref())];
        let mut retry = request();
        retry.facts.tried = &tried;
        retry.facts.attempt = 1;
        let asks = Asks::of(retry, Some(&members));
        for _ in 0..8 {
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            assert!(pick.targets.iter().all(|target| target.id != tried[0]));
        }
        for worker in list.iter() {
            worker.set_overloaded(true);
        }
        let asks = Asks::of(request(), Some(&members));
        match decided(&engine, &asks).decision {
            Decision::Pick(pick) => assert_eq!(pick.trace.lifted, LiftedVeto::Overload),
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "not-overloaded"),
            Decision::Queue(_) => panic!("no queue is planned"),
        }
    }

    /// A request that hands no list is decided over no worker. One that
    /// states what no stage here runs is declined under that stage's name,
    /// and a side sent on its own has no decision.
    #[test]
    fn what_no_stage_runs_is_answered_by_name() {
        let engine = preset("round_robin");
        let list = pool(2);
        let members = engine.pool_members(&list);
        let shed = |request: Request<'_>, members: Option<&PoolMembers>| {
            let asks = Asks::of(request, members);
            match decided(&engine, &asks).decision {
                Decision::Shed(shed) => (shed.reason, shed.emptied_by),
                other => panic!("a shed, not {other:?}"),
            }
        };
        assert_eq!(shed(request(), None), (ShedReason::NoEndpoints, NO_STAGE));
        let mut hinted = request();
        hinted.constraints.subset_hint = Some(1);
        assert_eq!(
            shed(hinted, Some(&members)),
            (ShedReason::PolicyDeclined, SUBSET_HINT)
        );
        // The list's workers are dialed over HTTP.
        let mut dialed = request();
        dialed.constraints.wire_modes = WireModes::GRPC;
        assert_eq!(
            shed(dialed, Some(&members)),
            (ShedReason::PolicyDeclined, WIRES)
        );
        dialed.constraints.wire_modes = WireModes::HTTP.union(WireModes::GRPC);
        let asks = Asks::of(dialed, Some(&members));
        assert!(matches!(answered(&engine, &asks), Decided::Alone(_)));
        let mut side = request();
        side.facts.phase = Phase::Decode;
        let asks = Asks::of(side, Some(&members));
        assert_eq!(
            engine
                .decide_under(GatewayId::LOCAL, &asks, Instant::now())
                .err(),
            Some(DecisionRefused::PairSide)
        );
    }

    /// A request its host holds to places of its list is made and decided
    /// over the list's own records: every target stands at one of the
    /// places, and each of them takes requests. Under a rotation, two draws
    /// and a profile that scores, each as it ships and with no filter, over
    /// a list under the draw's floor and one over it.
    #[test]
    fn a_request_held_to_places_is_handed_no_other_place_under_any_document() {
        for name in ["round_robin", "random", "power_of_two", "least_load"] {
            for bare in [false, true] {
                let engine = if bare { unfiltered(name) } else { preset(name) };
                for workers in [8, 32] {
                    let list = pool(workers);
                    let members = engine.pool_members(&list);
                    let mut asks = Asks::of(request(), Some(&members));
                    asks.request.constraints.places = Some(&[3, 6]);
                    let mut first = [0_u32; 8];
                    for seed in 0..32 {
                        asks.request.facts.seed = seed;
                        let decided = decided(&engine, &asks);
                        let pick = decided.decision.pick().expect("a pick");
                        let mut places = pick.targets.iter().map(|target| target.position);
                        assert!(places.all(|place| place == 3 || place == 6), "{name}");
                        let lead = pick.targets.first().expect("a target").position;
                        first[lead as usize] += 1;
                    }
                    assert!(first[3] > 0 && first[6] > 0, "{name}: {first:?}");
                    assert_eq!(first[3] + first[6], 32, "{name}: {first:?}");
                    assert_eq!(asks.made.get(), 32, "{name}: such a request is made");
                }
            }
        }
    }

    /// A keyed request held to places is handed the next members of the
    /// stock ring. Held to every member but its key's owner, it is handed
    /// the members a request held to nothing is named behind the owner, in
    /// that order: as many targets, and never the owner.
    #[test]
    fn a_keyed_request_held_to_places_takes_the_rings_next_members() {
        for engine in [
            preset("consistent_hashing"),
            unfiltered("consistent_hashing"),
        ] {
            let list = pool(16);
            let members = engine.pool_members(&list);
            let places = |decided: &DecideOutcome| -> Vec<u32> {
                let pick = decided.decision.pick().expect("a pick");
                pick.targets.iter().map(|target| target.position).collect()
            };
            for key in 1..=40 {
                let mut keyed = request();
                keyed.facts.seed = key;
                keyed.facts.session_key = Identity::hashed(key);
                let whole = places(&decided(&engine, &Asks::of(keyed, Some(&members))));
                assert_eq!(whole.len(), MAX_TARGETS, "key {key}");
                let others: Vec<u32> = (0..16).filter(|&place| place != whole[0]).collect();
                let mut held = Asks::of(keyed, Some(&members));
                held.request.constraints.places = Some(&others);
                let part = places(&decided(&engine, &held));
                assert_eq!(part.len(), MAX_TARGETS, "key {key}");
                assert_eq!(part[..MAX_TARGETS - 1], whole[1..], "key {key}");
                assert!(!part.contains(&whole[0]), "key {key}: the owner is handed");
            }
        }
    }

    /// The default policy holds a request to its places ahead of its
    /// prompt's holder. A prompt is sent to a worker, and the same prompt
    /// held to places that worker is outside of is handed none but them,
    /// as a request with no prompt is: over a list under the draw's floor
    /// and one over it.
    #[test]
    fn a_prompts_holder_outside_the_places_is_not_handed() {
        for workers in [8_u32, 32] {
            let engine = preset("cache_aware");
            let list = pool(workers as usize);
            let members = engine.pool_members(&list);
            let tokens: Vec<u32> = (0..1_024).collect();
            let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
            let sent = |places: Option<&[u32]>, seed: u64, prompted: bool| {
                let mut asked = request();
                asked.facts.seed = seed;
                asked.constraints.places = places;
                if prompted {
                    hashes.write_to(&mut asked.facts);
                }
                let decided = decided(&engine, &Asks::of(asked, Some(&members)));
                let pick = decided.decision.pick().expect("a pick");
                let places = pick.targets.iter().map(|target| target.position);
                places.collect::<Vec<u32>>()
            };
            let holder = sent(None, 0, true)[0];
            let others: Vec<u32> = (0..workers)
                .filter(|&place| place != holder)
                .take(2)
                .collect();
            for seed in 1..32 {
                for prompted in [true, false] {
                    let named = sent(Some(&others), seed, prompted);
                    assert!(
                        named.iter().all(|place| others.contains(place)),
                        "{workers} workers, seed {seed}: {named:?}"
                    );
                }
            }
        }
    }

    /// The rule for a list whose every worker is refused steers a held
    /// request among its places. Held to a worker over its thresholds, it
    /// is handed that worker with the veto lifted, beside free workers
    /// outside its places.
    #[test]
    fn a_held_request_over_its_thresholds_everywhere_is_steered_among_its_places() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = engine.pool_members(&list);
        list[1].set_overloaded(true);
        let mut held = request();
        held.constraints.places = Some(&[1]);
        let decided = decided(&engine, &Asks::of(held, Some(&members)));
        let pick = decided.decision.pick().expect("a pick");
        assert_eq!(pick.trace.lifted, LiftedVeto::Overload);
        assert!(pick.targets.iter().all(|target| target.position == 1));
    }

    /// The default policy's single request is named the worker its prompt
    /// was sent to. The whole pass reads the engine's record of sent
    /// prefixes, which the pick before it wrote: a prompt goes back to its
    /// worker whatever the request's seed, other prompts are not held to
    /// that worker, and a worker told gone holds no prompt.
    #[test]
    fn the_default_policy_sends_a_prompt_back_to_the_worker_it_was_sent_to() {
        let engine = preset("cache_aware");
        let list = pool(8);
        let members = engine.pool_members(&list);
        let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
        let sent = |tokens: &[u32], seed: u64| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            hashes.write_to(&mut asked.facts);
            let asks = Asks::of(asked, Some(&members));
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        let asked = prompt(0);
        let first = sent(&asked, 0);
        for seed in 1..32 {
            assert_eq!(sent(&asked, seed), first, "seed {seed}");
        }
        // Sixteen other prompts are placed by their seeds, over idle workers.
        let others: Vec<u32> = (1..=16)
            .map(|at| sent(&prompt(at * 100_000), u64::from(at)))
            .collect();
        assert!(others.iter().any(|&position| position != others[0]));
        // With every worker told gone before a request, its seed places it.
        let moved = (0..32).any(|seed| {
            for worker in list.iter() {
                engine.forget(worker.url());
            }
            sent(&asked, seed) != first
        });
        assert!(moved, "a forgotten worker holds no prompt");
        assert_eq!(engine.affinity().left_out(), 0);
    }

    /// The default policy under a sessions rule. A key's first request is
    /// placed by the whole pass, and its dispatch makes the key's pin. The
    /// key's next request follows that pin by one worker's record: no pass
    /// over the list runs and nobody is named its prompt's holder, so the
    /// prompt of a pinned request is recorded for no worker, as a gateway's
    /// own sticky map keeps a pinned request from its policy. A request of
    /// that prompt that carries no key is then placed by the policy, and
    /// not held to the pin's worker. A pinned worker that is not ready
    /// gives the key back to the whole pass.
    #[test]
    fn the_default_policy_follows_a_sessions_pin_by_one_record() {
        // The stock sessions rule over the default policy's profile, a
        // new key placed by the profile's own stages.
        let engine = {
            let mut document = PolicySpec::preset("manual").expect("a built-in preset");
            let default = PolicySpec::preset("cache_aware").expect("a built-in preset");
            document.profiles = default.profiles;
            let binding = &mut document.bindings[0];
            binding.profile = Some("cache_aware".into());
            let rule = binding.sessions.as_mut().expect("the stock rule");
            rule.assignment = Assignment::Delegate;
            document.generation = 1;
            PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
        };
        let list = pool(8);
        let members = engine.pool_members(&list);
        let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
        // One request of `tokens` under the session key `key`, with no key
        // where it is `0`, sent to its first target and served there: the
        // place it went to, what the decision did with its key, and
        // whether no pass over the list ran.
        let sent = |tokens: &[u32], key: u64, seed: u64| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            asked.facts.session_key = Identity::hashed(key);
            hashes.write_to(&mut asked.facts);
            let asks = Asks::of(asked, Some(&members));
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
            let pick = *guard.decision().pick().expect("a pick");
            let target = guard.take(0).expect("the first target");
            guard.dispatched(now);
            guard.completed(
                &Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
                now,
            );
            (
                target.position,
                pick.session_verdict,
                pick.trace.drew_first(),
            )
        };
        let fresh = |key: u32| prompt(key * 100_000 + 50_000);
        let (mut pins, mut passed) = (Vec::new(), Vec::new());
        for key in 1..=6_u32 {
            // The key's first request: the whole pass places it by its
            // seed, over idle workers, and claims it.
            let seed = u64::from(key);
            let (pin, verdict, unpassed) = sent(&prompt(key * 100_000), seed, seed);
            assert!(matches!(verdict, SessionVerdict::Claimed(_)), "key {key}");
            assert!(!unpassed, "key {key}: the whole pass places a new key");
            // Its next request, of a prompt nobody was sent: the pin.
            let (went, verdict, unpassed) = sent(&fresh(key), seed, 1);
            assert_eq!((went, verdict), (pin, SessionVerdict::Pinned), "key {key}");
            if !unpassed {
                passed.push(key);
            }
            pins.push(pin);
        }
        assert!(
            pins.iter().any(|pin| *pin != pins[0]),
            "the keys are pinned to more workers than one: {pins:?}"
        );
        // The pinned requests' prompts were recorded for nobody: a request
        // of one that carries no key is placed by its seed, over idle
        // workers, and not held to the pin's worker.
        let elsewhere = (1..=6_u32)
            .filter(|&key| sent(&fresh(key), 0, u64::from(key) + 40).0 != pins[key as usize - 1])
            .count();
        assert!(
            elsewhere > 0,
            "each of six prompts went back to its pin's worker: {pins:?}"
        );
        // Each pin was followed by its one worker's record.
        assert!(
            passed.is_empty(),
            "a pass over the list ran for the pinned requests of keys {passed:?}"
        );
        // A pin whose worker is not ready is the whole pass's to place anew.
        list[pins[0] as usize].set_status(WorkerStatus::NotReady);
        let (moved, _, unpassed) = sent(&prompt(100_000), 1, 3);
        assert!(!unpassed, "the whole pass places the key anew");
        assert_ne!(moved, pins[0]);
    }

    /// What a stage heard of a lease: the word, the worker the lease told
    /// books (the target's own at a dispatch), and the lease's index.
    type Heard = Arc<Mutex<Vec<(&'static str, WorkerId, u8)>>>;

    /// The `random` picker, keeping what it hears of the words it names
    /// (`takes`). `panics` makes its pick panic. `alone` makes it place a
    /// request over a pool by itself, by the pool's own turn.
    #[derive(Debug)]
    struct Hears {
        heard: Heard,
        panics: bool,
        alone: bool,
        takes: EventSet,
    }

    impl Lifecycle for Hears {
        fn hears(&self) -> EventSet {
            self.takes
        }

        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            let heard = ("issued", lease.booked, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
            let heard = ("dispatched", target.id, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_completed(&self, lease: &Lease, _: &Outcome) {
            let heard = ("completed", lease.booked, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_released(&self, lease: &Lease, _: ReleaseReason) {
            let heard = ("released", lease.booked, lease.target_index);
            self.heard.lock().push(heard);
        }

        fn on_membership(&self, change: &Membership<'_>) {
            assert!(!self.panics, "this picker panics");
            let heard = match *change {
                Membership::Added(id) => ("joined", id, 0),
                Membership::Removed(id) => ("left", id, 0),
                _ => return,
            };
            self.heard.lock().push(heard);
        }
    }

    impl Picker for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        // A picker that places by itself refuses by no report.
        fn refuses_by_reports(&self) -> bool {
            !self.alone
        }

        fn picks_alone(&self) -> bool {
            self.alone
        }

        fn pick_alone(
            &self,
            pool: &PoolWord<'_>,
            keep: &mut dyn FnMut(usize) -> bool,
        ) -> Option<usize> {
            let turn = pool.word.fetch_add(1, Ordering::Relaxed) as usize % pool.rows.max(1);
            keep(turn).then_some(turn)
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            assert!(!self.panics, "this picker panics");
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The engine of the `random` preset with [`Hears`] in its pick's place,
    /// taking every word of a lease, and what it heard.
    fn hearing(panics: bool) -> (PolicyEngine, Heard) {
        hearing_as(panics, false, EventSet::LEASE)
    }

    /// [`hearing`] with a picker that places by itself where `alone`, and
    /// that names the words `takes`.
    fn hearing_as(panics: bool, alone: bool, takes: EventSet) -> (PolicyEngine, Heard) {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let engine = picked_by("hears", move || {
            Box::new(Hears {
                heard: Arc::clone(&kept),
                panics,
                alone,
                takes,
            })
        });
        (engine, heard)
    }

    /// The engine of the `random` preset with the picker `make` builds, a
    /// stage of the host's own build under `name`, in its pick's place.
    fn picked_by(
        name: &'static str,
        make: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
    ) -> PolicyEngine {
        let mut stages = StageRegistry::builtin();
        let added = stages.picker(name, make);
        added.expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a built-in preset");
        document.generation = 1;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = name.into();
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        PolicyEngine::new(spec.expect("the stages hold the picker"))
    }

    /// [`Hears`] as a stage of a host's own build that says it needs the
    /// lifecycle's calls and names no word of its own.
    #[derive(Debug)]
    struct Declares(Hears);

    impl Lifecycle for Declares {
        fn needs_lifecycle(&self) -> bool {
            true
        }

        fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
            self.0.on_decision_issued(lease, decision);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
            self.0.on_dispatched(lease, target, side, since);
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            self.0.on_completed(lease, outcome);
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            self.0.on_released(lease, reason);
        }
    }

    impl Picker for Declares {
        fn name(&self) -> &'static str {
            "declares"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.0.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The words a stage of `engine` put into `heard` of two requests: one
    /// sent to its first target and completed, and one given up unsent.
    fn told(engine: &PolicyEngine, heard: &Heard) -> Vec<&'static str> {
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        let now = Instant::now();
        let mut handle =
            DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let first = handle.take(0).expect("the first target");
        handle.dispatched(now);
        handle.completed(
            &Outcome {
                side: Side::Single,
                served: first.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        drop(handle);
        let unsent =
            DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        drop(unsent);
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
        let words = heard.lock().iter().map(|heard| heard.0).collect();
        words
    }

    /// A stage hears a single's words from what the request's guard lends,
    /// with no lease on file: the issue on the pick's first target, the
    /// dispatch on the target taken under the index the word names, the
    /// end under the lease moved there, and a release under the lease as
    /// issued.
    #[test]
    fn the_stages_hear_a_singles_words_from_what_its_guard_lends() {
        let (engine, heard) = hearing(false);
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        let now = Instant::now();
        let mut handle =
            DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let first = handle.take(0).expect("the first target");
        let taken = handle.take(1).expect("an alternate");
        handle.dispatched(now);
        handle.completed(
            &Outcome {
                side: Side::Single,
                served: taken.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        drop(handle);
        let unsent =
            DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let given_up = unsent
            .decision()
            .pick()
            .and_then(|pick| pick.targets.first().copied())
            .expect("the first target");
        drop(unsent);
        assert_eq!(
            *heard.lock(),
            [
                ("issued", first.id, 0),
                ("dispatched", taken.id, 1),
                ("completed", taken.id, 1),
                ("issued", given_up.id, 0),
                ("released", given_up.id, 0),
            ]
        );
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// A word of a lease is told where a stage takes it and to nobody
    /// where none does. One request is sent and completed and one is given
    /// up unsent: a document whose one stage takes one kind of word hears
    /// that kind of the two leases and no other. A completion and a release
    /// are told apart.
    #[test]
    fn a_word_no_stage_takes_is_told_to_nobody() {
        let told_of = |kind: EventKind| {
            let (engine, heard) = hearing_as(false, false, EventSet::NONE.with(kind));
            told(&engine, &heard)
        };
        assert_eq!(told_of(EventKind::Issued), ["issued", "issued"]);
        assert_eq!(told_of(EventKind::Dispatched), ["dispatched"]);
        assert_eq!(told_of(EventKind::Completed), ["completed"]);
        assert_eq!(told_of(EventKind::Released), ["released"]);
    }

    /// A stage of a host's own build that says it needs the lifecycle's
    /// calls and names no word of its own is told every word on a lease:
    /// the issue, the dispatch and the completion of a request sent, and
    /// the issue and the release of one given up.
    #[test]
    fn a_stage_that_needs_the_calls_and_names_no_word_is_told_each() {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let engine = picked_by("declares", move || {
            Box::new(Declares(Hears {
                heard: Arc::clone(&kept),
                panics: false,
                alone: false,
                takes: EventSet::NONE,
            }))
        });
        assert_eq!(engine.hears, EventSet::LEASE);
        assert_eq!(
            told(&engine, &heard),
            ["issued", "dispatched", "completed", "issued", "released"]
        );
    }

    /// A decision that panics costs its request alone and is counted.
    #[test]
    fn a_decision_that_panics_is_refused_and_counted() {
        let (engine, heard) = hearing(true);
        let list = pool(2);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        for faults in 1..=2 {
            assert_eq!(
                engine
                    .decide_under(GatewayId::LOCAL, &asks, Instant::now())
                    .err(),
                Some(DecisionRefused::Faulted)
            );
            assert_eq!(engine.faults(), faults);
        }
        assert!(heard.lock().is_empty());
    }

    /// A worker that joins or leaves is told to the document's stages under
    /// the id a decision names it by, in the order its host says the two. A
    /// stage that panics at the word is held and counted: the call returns.
    #[test]
    fn a_worker_that_joins_or_leaves_is_told_to_the_documents_stages() {
        let list = pool(2);
        let id = |at: usize| worker_id(list[at].as_ref());
        let (engine, heard) = hearing(false);
        engine.joined(list[0].url());
        engine.forget(list[1].url());
        assert_eq!(*heard.lock(), [("joined", id(0), 0), ("left", id(1), 0)]);
        assert_eq!(engine.unheard(), 0);
        let (engine, heard) = hearing(true);
        engine.joined(list[0].url());
        engine.forget(list[1].url());
        assert!(heard.lock().is_empty());
        assert_eq!(engine.unheard(), 2);
    }

    /// One request of `asks` decided at `now`, sent to its first target and
    /// served there: that target.
    fn served(engine: &PolicyEngine, asks: &Asks<'_>, now: Instant) -> Target {
        let mut guard =
            DecisionGuard::decide_under(engine, GatewayId::LOCAL, asks, now).expect("a decision");
        let target = guard.take(0).expect("the first target");
        guard.dispatched(now);
        let outcome = Outcome {
            side: Side::Single,
            served: target.id,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        guard.completed(&outcome, now);
        target
    }

    /// A worker said gone and back at its address serves its sessions
    /// again, however long it was gone and with no join said: the session
    /// table is not told a leave, as the classic policies' map is not. A
    /// key is pinned on a worker that leaves, and is served by the worker
    /// that stayed for twice the time the table holds an absent worker's
    /// pins, a quarter of that time apart, by the engine's own number. Back
    /// in the list, its first worker serves it again.
    #[test]
    fn a_worker_said_gone_and_back_at_its_address_serves_its_sessions_again() {
        let (engine, both) = (preset("manual"), pool(2));
        let start = Instant::now();
        let absence = Duration::from_micros(engine.spec.engine().pins_absence_micros());
        // The key's request over `list` in quarter `turn`: its worker.
        let sent = |list: &Arc<[Arc<dyn Worker>]>, turn: u32| {
            let members = engine.pool_members(list);
            let mut keyed = request();
            keyed.facts.session_key = Identity::hashed(1);
            let asks = Asks::of(keyed, Some(&members));
            served(&engine, &asks, start + absence / 4 * turn).id
        };
        let first = sent(&both, 0);
        let at = usize::from(worker_id(both[1].as_ref()) == first);
        let (gone, other) = (&both[at], Arc::from([Arc::clone(&both[1 - at])]));
        engine.forget(gone.url());
        for turn in 1..=8 {
            assert_ne!(sent(&other, turn), first);
        }
        assert_eq!(sent(&both, 9), first, "it has its key again");
    }
}
