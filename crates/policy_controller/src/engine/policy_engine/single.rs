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
//! **A hard demand** ([`Demands`](crate::engine::Demands)). A request that
//! states one is held to the places of its list whose worker meets it, by
//! what the list's host said of each worker when the list's records were
//! made and by the rank the worker's object registered
//! ([`PoolRecords::meeting`]), among the places its host holds it to.
//! It is then decided as a request held to places is, so a key's pin on
//! a worker that does not meet the demand is not followed. A list the
//! demands leave nobody is shed with no endpoint, under the stage of the
//! demand that took its last worker. A pair that states a demand is
//! shed.
//!
//! **What is refused by name.** The list a host hands is its request's own:
//! every worker of it is dialed on a wire the request states. A list that is
//! not, and a subset hint, are declined under the stage that would have run
//! them.
//!
//! **A request held to places of its list**
//! ([`Constraints::places`](crate::engine::view::Constraints::places)) is
//! made and decided over the list's own records: it is not plain, the
//! keyed decision leaves it to the whole pass, and that pass opens with
//! the places. A pair whose request states places is declined ([`PLACES`]).

use std::{
    cell::Cell,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot, runtime::RoutingState};

use super::{keys, overlap, PolicyEngine, PoolMembers};
use crate::{
    engine::{
        decision::{Decision, Shed, ShedReason, Trace, SUBSET_HINT, WIRES},
        decision_engine::{AlonePick, DecideOutcome, Leased, Plain, Request},
        demands::UnmetDemand,
        framework::Engine,
        records::{self, PoolRecords},
        sticky_sessions::SessionVerdict,
        view::{Constraints, Phase, RequestFacts, Session},
    },
    stages::Lifecycle,
    tracking::lease::{EventKind, EventSet, Lease, LeaseId, LeaseKind, Side, DISPATCH_TTL},
};

thread_local! {
    /// The calling thread's own list of the places that meet a request's
    /// demands, taken for one decision and put back behind it: such a
    /// request allocates nothing once the list has grown to its pool.
    static MEETING: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
}

/// The stage a pair whose request states places is declined under
/// ([`Constraints::places`](crate::engine::view::Constraints::places)):
/// one set of places cannot hold a pair's two lists. An encode request's
/// lists whose pair is held to places are declined under it too.
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

/// How a made request's whole decision runs over its pool's records. The
/// caller of [`decide`] states the form where it is written, so [`decide`]
/// holds no test of it. One form runs the binding's live profile alone
/// ([`WithoutShadow`]): one call of [`records::decide`] and nothing else.
/// The other runs the binding's shadow profile behind it
/// ([`super::with_shadow`]).
pub(super) trait WholeDecision {
    /// The engine's whole decision over `records`, the records of
    /// `workers`, for the request `facts` state.
    fn decide(
        self,
        engine: &Engine,
        records: &PoolRecords,
        workers: &[Arc<dyn Worker>],
        live: &records::Live<'_>,
        facts: &mut RequestFacts<'_>,
        cx: &Constraints<'_>,
    ) -> Decision;
}

/// The whole decision by the live profile alone.
pub(super) struct WithoutShadow;

impl WholeDecision for WithoutShadow {
    #[inline(always)]
    fn decide(
        self,
        engine: &Engine,
        records: &PoolRecords,
        workers: &[Arc<dyn Worker>],
        live: &records::Live<'_>,
        facts: &mut RequestFacts<'_>,
        cx: &Constraints<'_>,
    ) -> Decision {
        records::decide(engine, records, workers, live, facts, cx)
    }
}

/// One made request's decision over `members`, the list its host handed,
/// at `now`: by the rows its key names where they answer
/// ([`keys::decide`]), and else the engine's whole decision over records
/// written for the request from the list's workers, run as `whole` states
/// ([`WholeDecision`]). Each worker's report is read from `loads`, the
/// host's load snapshot for this decision. A request that states what no
/// stage here runs is declined under that stage's name.
///
/// A request that states a hard demand is decided under its constraints
/// held to the workers that meet it ([`held`]). A list the demands leave
/// nobody is shed with no endpoint under the stage of the demand that took
/// its last worker ([`UnmetDemand::stage`]), as a pool its request's
/// demands emptied is shed over a store's rows.
pub(super) fn decide(
    own: &PolicyEngine,
    request: &mut Request<'_>,
    members: &PoolMembers,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    whole: impl WholeDecision,
) -> Decision {
    let spec = &own.spec;
    let cx = request.constraints;
    if cx.subset_hint.is_some() {
        return declined(own, request, SUBSET_HINT);
    }
    if !cx.wire_modes.is_empty() && !cx.wire_modes.contains(members.wires()) {
        return declined(own, request, WIRES);
    }
    let records = members.records(spec, &own.clock);
    // The calling thread's list of places, where the request states a
    // demand: lent for this decision and given back behind it. The rest of
    // the decision stands once and takes its constraints by reference.
    let mut meeting = None;
    let narrowed;
    let cx = if cx.demands.is_none() {
        &cx
    } else {
        match held(records, cx, &mut meeting) {
            Ok(held) => {
                narrowed = held;
                &narrowed
            }
            Err(unmet) => {
                MEETING.set(meeting.unwrap_or_default());
                return shed(own, request, ShedReason::NoEndpoints, unmet.stage());
            }
        }
    };
    let facts = &mut request.facts;
    // One value for every record of the decision: its instant, the host's
    // snapshot as the host holds it, the key's text for a stage that
    // counts the key's requests on a worker, and the list's own word, the
    // one a pick alone turns ([`PolicyEngine::pick`]): a draw of this
    // decision takes the list's next turn.
    let key = facts.session_key.plain;
    let live =
        records::Live::at(spec.engine(), &own.clock, now, loads, key).turning(&members.rotation.0);
    let (engine, workers) = (spec.engine(), members.workers());
    // A keyed request is decided by the rows its key names, where they say
    // what the whole pass would: its key's pin.
    let decision = match keys::decide(engine, records, workers, &live, facts, cx) {
        Some(keyed) => keyed,
        // A profile whose stages read who holds the prompt is named its
        // rows from the gateway's own data, at the decision's instant the
        // keyed entry wrote, and its pick is recorded behind it; any other
        // passes straight to the whole pass.
        None => overlap::decide(
            &own.affinity,
            members.affinity(),
            spec,
            workers,
            records.ids(),
            facts,
            |facts| whole.decide(engine, records, workers, &live, facts, cx),
        ),
    };
    if let Some(meeting) = meeting {
        MEETING.set(meeting);
    }
    decision
}

/// The constraints a request that states a hard demand is decided under:
/// its own, held to the places of the list whose worker meets its demands,
/// among the places its host holds it to ([`PoolRecords::meeting`]). They
/// are the decision's places ([`Constraints::places`]), so the request is
/// answered no other worker. Where every worker the request may take meets
/// the demands, it is held to no more than its host said. Where none does,
/// the answer is the demand that took the last worker.
///
/// The places stand in `meeting`, the calling thread's own list, taken
/// here: its caller puts it back ([`MEETING`]). Out of line: its code
/// stays off the path of a request that states no demand.
#[inline(never)]
fn held<'a>(
    records: &PoolRecords,
    cx: Constraints<'a>,
    meeting: &'a mut Option<Vec<u32>>,
) -> Result<Constraints<'a>, UnmetDemand> {
    let kept = meeting.insert(MEETING.take());
    match records.meeting(cx.demands, cx.places, kept) {
        None => Ok(cx),
        Some(unmet) if kept.is_empty() => Err(unmet),
        Some(_) => Ok(Constraints {
            places: Some(kept),
            ..cx
        }),
    }
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
pub(crate) mod tests {
    use std::{cell::Cell, collections::BTreeSet, sync::Arc, time::Duration};

    use parking_lot::Mutex;
    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        load::state::one_rank_load,
        model_card::ModelCard,
        test_support::{no_health_check, ready_worker},
        worker::WorkerStatus,
    };

    use super::*;
    use crate::{
        document::spec::{
            AppliedSpec, Assignment, PolicySpec, ScoreStage, ShadowSpec, WarmupSliceParams,
        },
        engine::{
            decision::{LiftedVeto, PlacedBy, Target, Targets, MAX_TARGETS},
            decision_engine::{
                Decided, DecisionEngine, DecisionRefused, RequestMaker, WorkerLists,
            },
            demands::{DemandFacts, Demands},
            framework::{Picker, Place, PoolWord, TieBand, DRAW_FROM},
            prompt_hashes::HashSeeds,
            records::worker_id,
            view::{
                Candidate, Constraints, FixedFacts, Identity, RequestFacts, WireModes, WorkerId,
            },
        },
        policies::random::Random,
        stages::registry::StageRegistry,
        store::GatewayId,
        tracking::{
            lease::{Membership, Outcome, ReleaseReason, TransportClass},
            DecisionGuard,
        },
    };

    pub(crate) const MODEL: &str = "model-a";

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

    pub(crate) fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
        (0..workers)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect()
    }

    pub(crate) fn request<'a>() -> Request<'a> {
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
    pub(crate) struct Asks<'a> {
        pub(crate) request: Request<'a>,
        pub(crate) members: Option<&'a PoolMembers>,
        pub(crate) loads: Option<Arc<LoadSnapshot>>,
        pub(crate) made: Cell<u32>,
    }

    impl<'a> Asks<'a> {
        pub(crate) fn of(request: Request<'a>, members: Option<&'a PoolMembers>) -> Self {
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
    pub(crate) fn answered(engine: &PolicyEngine, asks: &Asks<'_>) -> Decided {
        let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
        decided.expect("an answer")
    }

    /// The whole decision the entry answers: no place.
    pub(crate) fn decided(engine: &PolicyEngine, asks: &Asks<'_>) -> DecideOutcome {
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

    /// A program that keeps ids of its own states each worker's with its
    /// list's fixed facts. The list's records carry those ids whole, and
    /// a record's key stays the low half of its address's fingerprint. A
    /// whole decision names every target by the stated id, and passes
    /// over the worker a retry names by it. A list handed with nothing
    /// said names each worker by its address's fingerprint.
    #[test]
    fn a_list_names_each_worker_by_the_id_its_host_states() {
        use crate::engine::view::FixedFacts;

        // Ids past 32 bits: no half of one stands for it.
        let wide = |at: u64| WorkerId((1 << 40) + at);
        let engine = preset("round_robin");
        let list = pool(4);
        let place = |worker: &Arc<dyn Worker>| list.iter().position(|one| Arc::ptr_eq(one, worker));
        let members = engine.pool_members_with_fixed_facts(&list, |worker| FixedFacts {
            id: place(worker).map(|at| wide(at as u64)),
            ..FixedFacts::default()
        });
        let records = members.records(&engine.spec, &engine.clock);
        let stated: Vec<WorkerId> = (0..4).map(wide).collect();
        assert_eq!(records.ids(), stated);
        for (row, worker) in records.fixed().iter().zip(list.iter()) {
            assert_eq!(row.key, worker_id(worker.as_ref()).0 as u32);
        }

        let tried = [stated[1]];
        let mut retry = request();
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let mut seen = [false; 4];
        for _ in 0..12 {
            let whole = decided(&engine, &asks);
            let pick = whole.decision.pick().expect("a pick");
            for target in pick.targets.iter() {
                assert_eq!(target.id, stated[target.position as usize]);
                assert_ne!(target.id, tried[0], "the tried one is no alternate");
            }
            let target = pick.targets.first().expect("a target");
            seen[target.position as usize] = true;
        }
        assert_eq!(seen, [true, false, true, true], "the tried one is passed");

        let unsaid = engine.pool_members(&list);
        let prints: Vec<WorkerId> = list
            .iter()
            .map(|worker| worker_id(worker.as_ref()))
            .collect();
        let records = unsaid.records(&engine.spec, &engine.clock);
        assert_eq!(records.ids(), prints);
    }

    /// Zero is no worker's id: it reads as absent wherever an id travels,
    /// and a session's table keeps no pin on it. A program that states
    /// zero for a worker states none: the list's records name the worker
    /// by its address's fingerprint, and a session placed on that worker
    /// keeps its pin.
    #[test]
    fn a_list_names_a_worker_by_its_address_where_its_host_states_zero() {
        use crate::engine::view::FixedFacts;

        let engine = preset("manual");
        let list = pool(4);
        let members = engine.pool_members_with_fixed_facts(&list, |_| FixedFacts {
            id: Some(WorkerId(0)),
            ..FixedFacts::default()
        });
        let prints: Vec<WorkerId> = list
            .iter()
            .map(|worker| worker_id(worker.as_ref()))
            .collect();
        let records = members.records(&engine.spec, &engine.clock);
        assert_eq!(records.ids(), prints);

        // The stock sessions rule: a key's first request is placed and
        // its dispatch pins the key, and the key's next request follows
        // the pin.
        let mut keyed = request();
        keyed.facts.session_key = Identity::hashed(41);
        let asks = Asks::of(keyed, Some(&members));
        let placed = served(&engine, &asks, Instant::now());
        assert_eq!(placed.id, prints[placed.position as usize]);
        let followed = decided(&engine, &asks).decision;
        let pick = followed.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        let again = pick.targets.first().expect("a target");
        assert_eq!(again.id, placed.id);
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

    /// Round robin under a sessions rule that leaves a new session to the
    /// profile (`delegate`) turns one rotation for a request with no routing
    /// key and for a new session: in turn they take the list's workers one
    /// after the other, under the draw's floor, at it and above it. The
    /// gateway's classic override asks its one policy for both
    /// (`model_gateway/src/policies/round_robin.rs:142-167`): for a request
    /// with no key at `model_gateway/src/policies/registry.rs:471` and for a
    /// new session at `:581`.
    #[test]
    fn a_plain_request_and_a_new_session_share_one_rotation_under_delegate() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let manual = PolicySpec::preset("manual").expect("a built-in preset");
        document.view.headers.clone_from(&manual.view.headers);
        let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
        rule.assignment = Assignment::Delegate;
        document.bindings[0].sessions = Some(rule);
        let keys = ["key-a", "key-b", "key-c", "key-d", "key-e", "key-f"];
        for workers in [DRAW_FROM - 1, DRAW_FROM, 64] {
            let spec = AppliedSpec::validate(&document).expect("the document validates");
            let engine = PolicyEngine::new(spec);
            let list = pool(workers);
            let members = engine.pool_members(&list);
            let mut places = Vec::new();
            for turn in 0..12_usize {
                let mut ask = request();
                if turn % 2 == 1 {
                    ask.facts.session_key = Identity::plain(keys[turn / 2]);
                }
                let asks = Asks::of(ask, Some(&members));
                let decided = decided(&engine, &asks);
                let pick = decided.decision.pick().expect("a pick");
                places.push(pick.targets.first().expect("the target").position as usize);
            }
            let in_turn: Vec<usize> = (0..12).map(|turn| (places[0] + turn) % workers).collect();
            assert_eq!(places, in_turn, "{workers} workers");
        }
    }

    /// A draw turns the list's own word, the one a pick alone turns. Two
    /// lists of one model and one size so turn a rotation each: asked in
    /// turn, a request with no routing key and a new session at each, they
    /// take each list's workers one after the other. And over one list a
    /// pick alone and a new session's whole decision share one rotation.
    /// The gateway's classic policy keeps its rotation by the list it is
    /// handed whole (`model_gateway/src/policies/round_robin.rs:95-99`,
    /// `:151`).
    #[test]
    fn each_list_turns_its_own_rotation_for_every_request() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let manual = PolicySpec::preset("manual").expect("a built-in preset");
        document.view.headers.clone_from(&manual.view.headers);
        let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
        rule.assignment = Assignment::Delegate;
        document.bindings[0].sessions = Some(rule);
        let keys: Vec<String> = (0..24).map(|key| format!("key-{key}")).collect();
        let mut keys = keys.iter();
        for workers in [DRAW_FROM, 64] {
            let spec = AppliedSpec::validate(&document).expect("the document validates");
            let engine = PolicyEngine::new(spec);
            let Place::Alone(profile) = engine.place(MODEL, Phase::Single) else {
                panic!("one pick of round robin decides the place");
            };
            let (one, other) = (pool(workers), pool(workers));
            let lists = [engine.pool_members(&one), engine.pool_members(&other)];
            let mut whole = |members: &PoolMembers, keyed: bool| {
                let mut ask = request();
                if keyed {
                    ask.facts.session_key = Identity::plain(keys.next().expect("a key"));
                }
                let asks = Asks::of(ask, Some(members));
                let decided = decided(&engine, &asks);
                let pick = decided.decision.pick().expect("a pick");
                pick.targets.first().expect("the target").position as usize
            };
            let mut places = [Vec::new(), Vec::new()];
            for turn in 0..8 {
                for (list, members) in lists.iter().enumerate() {
                    places[list].push(whole(members, turn % 2 == 1));
                }
            }
            for places in &places {
                let in_turn: Vec<usize> = (0..8).map(|turn| (places[0] + turn) % workers).collect();
                assert_eq!(*places, in_turn, "{workers} workers, two lists");
            }
            // The first list's rotation goes on, a pick alone and a new
            // session's whole decision in turn.
            let mut shared = Vec::new();
            for turn in 0..8 {
                shared.push(if turn % 2 == 0 {
                    let alone = engine.pick(profile, &lists[0], RoutingState::eligible);
                    alone.expect("a worker")
                } else {
                    whole(&lists[0], true)
                });
            }
            let in_turn: Vec<usize> = (8..16)
                .map(|turn| (places[0][0] + turn) % workers)
                .collect();
            assert_eq!(shared, in_turn, "{workers} workers, one list");
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

    /// A request stated ahead by a maker that states no plain request
    /// ([`RequestMaker::plain`]'s own answer), as a host that hands a
    /// decision's targets to another process states none: the leasing
    /// entry then makes the request and decides it whole.
    struct Whole<'a>(Asks<'a>);

    impl RequestMaker for Whole<'_> {
        fn model(&self) -> &str {
            self.0.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            self.0.make(spec)
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.0.workers()
        }

        fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
            self.0.loads()
        }
    }

    /// The entry for a host that files leases itself answers the decision
    /// alone. Under a document whose stages take every word of a lease and
    /// whose picker places a plain request by itself, the entry still makes
    /// each request once and decides it whole: its pick names its
    /// alternates, where the leasing entry answers the one place. No id is
    /// taken and no stage hears an issue: the host tells the stages, under
    /// the lease it files.
    #[test]
    fn a_decision_without_a_lease_takes_no_id_and_tells_no_stage() {
        let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
        let place = engine.place(MODEL, Phase::Single);
        assert!(matches!(place, Place::Alone(_)), "{place:?}");
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        assert!(asks.plain().is_some(), "a plain request");
        let mut last = None;
        for _ in 0..8 {
            let decision = engine.decide_without_lease(&asks, Instant::now());
            let (decision, _) = decision.expect("a decision");
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.targets.len(), 4, "the pick and its alternates");
            for target in pick.targets.iter() {
                let worker = &list[target.position as usize];
                assert_eq!(target.id, worker_id(worker.as_ref()));
            }
            last = Some(decision);
        }
        assert_eq!(asks.made.get(), 8, "each request is made once");
        assert!(heard.lock().is_empty(), "no stage heard an issue");
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");

        // The host files the pick's lease and tells the stages under it.
        let decision = last.expect("a decision");
        let pick = decision.pick().expect("a pick");
        let first = *pick.targets.first().expect("the target");
        let lease = Lease {
            id: LeaseId(77),
            kind: LeaseKind::Credit,
            side: Side::Single,
            target_index: 0,
            booked: first.id,
            generation: pick.trace.generation,
            session: Session::NONE,
            session_verdict: SessionVerdict::None,
            issued_micros: 0,
            tokens: 0,
        };
        engine.spec().engine().on_decision_issued(&lease, &decision);
        assert_eq!(heard.lock().as_slice(), [("issued", first.id, 0)]);
    }

    /// The entry makes the leasing entry's whole decision of the same
    /// request, under every stock policy: two engines made alike, the
    /// leasing one asked by a maker that states no plain request, name the
    /// same workers in the same order, request after request, by the
    /// request's seed and by a rotation's turns. With a load snapshot lent
    /// both read it: a policy that scores by load hands the worker with the
    /// shortest queue at every seed.
    #[test]
    fn a_decision_without_a_lease_is_the_leasing_entrys_whole_decision() {
        use crate::document::spec::PRESETS;

        let places = |decision: &Decision| -> Vec<u32> {
            let targets = decision.pick().map(|pick| pick.targets.iter());
            targets.map_or_else(Vec::new, |targets| {
                targets.map(|target| target.position).collect()
            })
        };
        let reports = (0..8)
            .map(|at| {
                let queued = if at == 5 { 0 } else { 90_000 };
                let url = format!("http://worker-{at}:8000");
                (url, one_rank_load(queued, 0.1, 1_000.0))
            })
            .collect();
        let snapshot = LoadSnapshot::from_loads_for_test(reports);
        for policy in PRESETS {
            for loads in [None, Some(Arc::clone(&snapshot))] {
                let (leasing, bare) = (preset(policy), preset(policy));
                let list = pool(8);
                let (theirs, ours) = (leasing.pool_members(&list), bare.pool_members(&list));
                for seed in 0..16 {
                    let asked = |members| {
                        let mut seeded = request();
                        seeded.facts.seed = seed;
                        Asks {
                            loads: loads.clone(),
                            ..Asks::of(seeded, Some(members))
                        }
                    };
                    let whole = Whole(asked(&theirs));
                    let decided = leasing.decide_under(GatewayId::LOCAL, &whole, Instant::now());
                    let decided = decided.expect("an answer").whole();
                    let whole = places(&decided.expect("a whole decision").0.decision);
                    assert!(!whole.is_empty(), "{policy}, seed {seed}: a pick");
                    let alone = bare.decide_without_lease(&asked(&ours), Instant::now());
                    let alone = places(&alone.expect("a decision").0);
                    assert_eq!(alone, whole, "{policy}, seed {seed}");
                    if policy == "least_load" && loads.is_some() {
                        assert_eq!(alone[0], 5, "seed {seed}: the shortest queue");
                    }
                }
            }
        }
    }

    /// The entry reads no pins of record, as the one entry reads none: a
    /// request that lends a record naming another worker is decided as one
    /// that lends none, and the record is never asked.
    #[test]
    fn a_decision_without_a_lease_reads_no_pins_of_record() {
        use std::sync::OnceLock;

        use crate::engine::view::PinSource;

        /// Pins of record: one worker for every key, and whether it was
        /// asked.
        #[derive(Debug)]
        struct Elsewhere(WorkerId, OnceLock<()>);

        impl PinSource for Elsewhere {
            fn pinned(&self, _: Session, _: Option<u64>) -> Option<WorkerId> {
                let _ = self.1.set(());
                Some(self.0)
            }
        }

        let (engine, twin) = (preset("manual"), preset("manual"));
        let list = pool(16);
        let (members, others) = (engine.pool_members(&list), twin.pool_members(&list));
        let keyed = || {
            let mut keyed = request();
            keyed.facts.seed = 41;
            keyed.facts.session_key = Identity::hashed(41);
            keyed
        };
        let named = |engine: &PolicyEngine, asks: &Asks<'_>| {
            let decision = engine.decide_without_lease(asks, Instant::now());
            let (decision, _) = decision.expect("a decision");
            let first = decision.pick().and_then(|pick| pick.targets.first());
            first.map(|target| target.id).expect("a pick")
        };
        let plain = named(&twin, &Asks::of(keyed(), Some(&others)));
        let mut ids = list.iter().map(|worker| worker_id(worker.as_ref()));
        let other = ids.find(|id| *id != plain).expect("another worker");
        let record = Elsewhere(other, OnceLock::new());
        let mut lent = keyed();
        lent.constraints.pins = Some(&record);
        assert_eq!(named(&engine, &Asks::of(lent, Some(&members))), plain);
        assert!(record.1.get().is_none(), "the record is not asked");
    }

    /// A request that hands no list of its kind is decided over no worker,
    /// as the one entry decides it: a single with no list, a pair's prefill
    /// with none, and a prefill over a single's list. A decode or an encode
    /// request sent on its own is refused.
    #[test]
    fn a_decision_without_a_lease_over_no_list_names_no_worker() {
        let engine = preset("round_robin");
        let list = pool(2);
        let members = engine.pool_members(&list);
        let stating = |phase, members| {
            let mut stated = request();
            stated.facts.phase = phase;
            Asks::of(stated, members)
        };
        let none = (ShedReason::NoEndpoints, "none");
        for (phase, members) in [
            (Phase::Single, None),
            (Phase::Prefill, None),
            (Phase::Prefill, Some(&members)),
        ] {
            let listed = members.is_some();
            let shed = engine.decide_without_lease(&stating(phase, members), Instant::now());
            let Ok((Decision::Shed(shed), None)) = shed else {
                panic!("{phase:?}, {listed}: a shed, not {shed:?}");
            };
            assert_eq!((shed.reason, shed.emptied_by), none, "{phase:?}, {listed}");
            let whole = Whole(stating(phase, members));
            let leased = engine.decide_under(GatewayId::LOCAL, &whole, Instant::now());
            let leased = leased.expect("an answer").whole().expect("no place").0;
            let Decision::Shed(theirs) = leased.decision else {
                panic!("{phase:?}, {listed}: the one entry picks");
            };
            assert_eq!((theirs.reason, theirs.emptied_by), none);
        }
        for phase in [Phase::Decode, Phase::Encode] {
            let asks = stating(phase, None);
            let refused = engine.decide_without_lease(&asks, Instant::now());
            assert_eq!(refused.err(), Some(DecisionRefused::PairSide), "{phase:?}");
        }
    }

    /// A request its sessions rule places alone says so on its pick and
    /// leaves no prompt in the record, in three forms. A pin followed by its
    /// one worker's record. A pin the whole pass follows, under a binding
    /// with a shadow. And a key's first request under a rule whose own
    /// assignment places it, where no stage of the profile is asked. A
    /// gateway's sticky map hands none of the three to its policy, so the
    /// policy's record never sees the prompt
    /// (`model_gateway/src/policies/registry.rs:540-546` for a pin under
    /// its cap, `:589-590` for the map's own assignment). A request of that
    /// prompt that carries no key is then placed by the policy, and not
    /// held to the session's worker. The first form's pin is followed ahead
    /// of every write of the record, so that form holds the word alone: the
    /// two others hold the record, and run again under a profile that reads
    /// the column whole, where the record's other write stands. No recorded
    /// sheet runs this.
    #[test]
    fn a_request_its_sessions_rule_places_alone_records_no_prompt() {
        for (form, shadowed, assignment, whole) in [
            ("a pin by one record", false, Assignment::Delegate, false),
            ("a shadow", true, Assignment::Delegate, false),
            ("the rule's assignment", false, Assignment::Random, false),
            ("a shadow, the column", true, Assignment::Delegate, true),
            ("the rule's, the column", true, Assignment::Random, true),
        ] {
            let engine = {
                let mut document = PolicySpec::preset("manual").expect("a built-in preset");
                let default = PolicySpec::preset("cache_aware").expect("a built-in preset");
                document.profiles = default.profiles;
                let binding = &mut document.bindings[0];
                binding.profile = Some("cache_aware".into());
                let rule = binding.sessions.as_mut().expect("the stock rule");
                rule.assignment = assignment;
                if shadowed {
                    let twin = document.profiles["cache_aware"].clone();
                    document.profiles.insert("twin".into(), twin);
                    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
                }
                if whole {
                    // A second warm-up stage that rules by another window:
                    // the live profile is named no row and reads the column
                    // whole, and its list keeps the record for the shadow.
                    let profile = document.profiles.get_mut("cache_aware");
                    profile.expect("the preset's profile").score.push(
                        ScoreStage::WarmupSlice(WarmupSliceParams {
                            weight: 1.0,
                            secs: Some(7),
                            ..WarmupSliceParams::default()
                        })
                        .into(),
                    );
                }
                document.generation = 1;
                let applied = AppliedSpec::validate(&document).expect("the document validates");
                let cap = applied.engine().affinity_rows_cap(MODEL, Phase::Single);
                assert_eq!(cap == 0, whole, "{form}: the form of the record's write");
                let record = applied.engine().pool_record_for(MODEL, Phase::Single);
                assert!(record.is_some(), "{form}: the list keeps a record");
                PolicyEngine::new(applied)
            };
            let list = pool(8);
            let members = engine.pool_members(&list);
            let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
            // One request sent to its first target and served there: the
            // place it went to, and who placed it.
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
                (target.position, pick.placed_by(), pick.trace.drew_first())
            };
            let mut held = 0;
            for key in 1..=6_u32 {
                let seed = u64::from(key);
                // The key's first request, then one of a prompt nobody was
                // sent: the rule alone places the one this form is about.
                let (first, by, _) = sent(&prompt(key * 100_000), seed, seed);
                let fresh = prompt(key * 100_000 + 50_000);
                let asked = if assignment == Assignment::Delegate {
                    assert_eq!(by, PlacedBy::Stages, "{form}: the stages place a new key");
                    let (went, by, unpassed) = sent(&fresh, seed, 1);
                    assert_eq!((went, by), (first, PlacedBy::Rule), "{form}: key {key}");
                    // The whole pass follows the pin under a shadow alone.
                    assert_eq!(unpassed, !shadowed, "{form}: key {key}");
                    fresh
                } else {
                    assert_eq!(by, PlacedBy::Rule, "{form}: key {key}");
                    prompt(key * 100_000)
                };
                // The same prompt with no key: the policy places it.
                let (went, by, _) = sent(&asked, 0, seed + 40);
                assert_eq!(by, PlacedBy::Stages, "{form}: no key");
                held += u32::from(went == first);
            }
            assert!(
                held < 6,
                "{form}: each of six prompts went back to its session's worker"
            );
        }
    }

    /// A request with no key is never its sessions rule's, also where the
    /// profile's picker leaves the placement to the binding: the binding's
    /// placing assignment places such a request, under a rule and under
    /// none, and its pick keeps the stages' word. The rule's word is for
    /// its key alone, which the same assignment places.
    #[test]
    fn a_request_with_no_key_is_never_its_sessions_rules() {
        for ruled in [true, false] {
            let engine = {
                let mut document = PolicySpec::preset("manual").expect("a built-in preset");
                if !ruled {
                    document.bindings[0].sessions = None;
                }
                document.generation = 1;
                let applied = AppliedSpec::validate(&document).expect("the document validates");
                PolicyEngine::new(applied)
            };
            let list = pool(8);
            let members = engine.pool_members(&list);
            let placed_by = |key: u64| {
                let mut asked = request();
                asked.facts.seed = 7;
                asked.facts.session_key = Identity::hashed(key);
                let asks = Asks::of(asked, Some(&members));
                let now = Instant::now();
                let guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                    .expect("a decision");
                guard.decision().pick().expect("a pick").placed_by()
            };
            assert_eq!(placed_by(0), PlacedBy::Stages, "no key, ruled: {ruled}");
            // A key is the rule's where a rule stands, and no one's else.
            let keyed = if ruled {
                PlacedBy::Rule
            } else {
                PlacedBy::Stages
            };
            assert_eq!(placed_by(9), keyed, "a key, ruled: {ruled}");
        }
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

    /// A placement says a single's words as a guard driven by hand says
    /// them, and names the list's own worker at the place of the target
    /// taken. Four requests, each once through a placement and once
    /// through a guard: one sent to an alternate and ended, one sent and
    /// dropped, one dropped unsent, and one whose pick has no target at the
    /// index taken, which makes no placement and gives its lease up.
    #[test]
    fn a_placement_says_a_singles_words_as_its_guard_says_them() {
        let (engine, heard) = hearing(false);
        let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        let ended = |side: Side, served: WorkerId| Outcome {
            side,
            served,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        let whole = || {
            let answer = answered(engine, &asks).whole();
            let (decided, later) = answer.expect("a whole decision");
            (decided, later, Instant::now())
        };
        // One request as a placement with the target at `index` taken, and
        // that target.
        let taken_at = |index: usize| {
            let (decided, later, now) = whole();
            let targets = decided.decision.pick().expect("a pick").targets;
            let taken = targets.as_slice().get(index).copied();
            let worker = Arc::clone(&list[taken.map_or(0, |taken| taken.position as usize)]);
            let placement =
                crate::tracking::Placement::taken(engine, decided, later, now, index, worker, None);
            (placement, taken)
        };
        let by_hand = || {
            let (decided, later, now) = whole();
            (
                DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now),
                now,
            )
        };
        // What the stage heard since the last call: each word with its
        // worker, and each word with its lease's index.
        let words = || {
            let heard = std::mem::take(&mut *heard.lock());
            let told: Vec<_> = heard.iter().map(|word| (word.0, word.2)).collect();
            (heard, told)
        };

        // Sent to an alternate and ended. Its host wrote another side and no
        // worker into the outcome.
        let (placement, alternate) = taken_at(1);
        let mut placement = placement.expect("a placement");
        let alternate = alternate.expect("a target");
        let at = alternate.position as usize;
        assert!(Arc::ptr_eq(&placement.worker, &list[at]));
        assert_eq!(
            (placement.id(), placement.decode_id()),
            (alternate.id, None)
        );
        assert!(placement.decode.is_none() && placement.routing_key.is_none());
        placement.routing_key = Some("a-key");
        let mut placement = placement.kept();
        assert_eq!(
            placement.routing_key, None,
            "a kept placement borrows nothing"
        );
        assert!(placement.owes_events());
        placement.sent(Side::Single);
        placement.sent(Side::Single);
        assert!(placement.owes_events(), "sent, and not ended");
        placement.ended(Side::Single, ended(Side::Decode, WorkerId(0)));
        assert!(!placement.owes_events());
        drop(placement);
        // Sent and dropped: a cancelled end.
        let mut placement = taken_at(0).0.expect("a placement");
        placement.sent(Side::Single);
        drop(placement);
        // Dropped unsent: a release.
        drop(taken_at(0).0.expect("a placement"));
        // No target at the index taken: no placement, and the lease is given
        // up.
        assert!(taken_at(9).0.is_none());
        let (heard_of_placements, through_placements) = words();

        let (mut guard, now) = by_hand();
        let taken = guard.take(1).expect("a target");
        guard.dispatched(now);
        guard.completed(&ended(Side::Single, taken.id), now);
        drop(guard);
        let (mut guard, now) = by_hand();
        guard.dispatched(now);
        drop(guard);
        drop(by_hand());
        let (mut guard, _) = by_hand();
        assert!(guard.take(9).is_none());
        drop(guard);
        assert_eq!(through_placements, words().1);
        assert_eq!(
            through_placements,
            [
                ("issued", 0),
                ("dispatched", 1),
                ("completed", 1),
                ("issued", 0),
                ("dispatched", 0),
                ("completed", 0),
                ("issued", 0),
                ("released", 0),
                ("issued", 0),
                ("released", 0),
            ]
        );
        // The dispatch names the target taken, and the end is told under
        // the lease moved there.
        let on_the_alternate = [heard_of_placements[1].1, heard_of_placements[2].1];
        assert_eq!(on_the_alternate, [alternate.id; 2]);
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// [`Hears`] as a stage that also keeps the outcome of each end it is
    /// told.
    #[derive(Debug)]
    struct Ends(Hears, Arc<Mutex<Vec<Outcome>>>);

    impl Lifecycle for Ends {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
            self.0.on_decision_issued(lease, decision);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
            self.0.on_dispatched(lease, target, side, since);
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            self.0.on_completed(lease, outcome);
            self.1.lock().push(*outcome);
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            self.0.on_released(lease, reason);
        }
    }

    impl Picker for Ends {
        fn name(&self) -> &'static str {
            "ends"
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

    /// An engine whose one stage is [`Ends`], with what the stage hears
    /// and the outcomes it keeps.
    fn hearing_ends() -> (&'static PolicyEngine, Heard, Arc<Mutex<Vec<Outcome>>>) {
        let (heard, ends) = (Heard::default(), Arc::new(Mutex::new(Vec::new())));
        let kept = (Arc::clone(&heard), Arc::clone(&ends));
        let engine = picked_by("ends", move || {
            let hears = Hears {
                heard: Arc::clone(&kept.0),
                panics: false,
                alone: false,
                takes: EventSet::LEASE,
            };
            Box::new(Ends(hears, Arc::clone(&kept.1)))
        });
        (Box::leak(Box::new(engine)), heard, ends)
    }

    /// A single request's placement goes out under any side its host names
    /// but an encode side, as it ends under any. A host that names its one
    /// call a decode side or a prefill side says the request's dispatch:
    /// the stage counts the request as sent while it runs, and a drop
    /// behind the send is an end of class cancelled, not a release.
    #[test]
    fn a_single_requests_placement_is_sent_under_the_side_its_host_names() {
        let (engine, heard, ends) = hearing_ends();
        let list = pool(4);
        let members = engine.pool_members(&list);
        let words = || std::mem::take(&mut *heard.lock());
        for side in [Side::Decode, Side::Prefill, Side::Single] {
            let asks = Asks::of(request(), Some(&members));
            let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
            let first = decided
                .decision
                .pick()
                .and_then(|pick| pick.targets.first().copied());
            let first = first.expect("a target");
            let worker = Arc::clone(&list[first.position as usize]);
            let now = Instant::now();
            let placement =
                crate::tracking::Placement::taken(engine, decided, later, now, 0, worker, None);
            let mut placement = placement.expect("a placement");
            assert_eq!(words(), [("issued", first.id, 0)]);
            placement.sent(side);
            assert_eq!(words(), [("dispatched", first.id, 0)], "{side:?}");
            drop(placement);
            assert_eq!(words(), [("completed", first.id, 0)], "{side:?}");
            let cancelled = std::mem::take(&mut *ends.lock());
            let classes: Vec<TransportClass> = cancelled.iter().map(|end| end.class).collect();
            assert_eq!(classes, [TransportClass::Cancelled], "{side:?}");
        }
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// An encode side's end says nothing through a placement, as its send
    /// says nothing: no lease stands for such a side. A single request's
    /// placement is told one before the request is sent and one behind its
    /// send. No stage hears a word of either, and the request is still
    /// owed its end. Its own end behind them is said once, under the
    /// request's side and the worker of the target taken, whatever its
    /// host wrote there.
    #[test]
    fn an_encode_sides_end_through_a_placement_says_nothing() {
        let (heard, ends) = (Heard::default(), Arc::new(Mutex::new(Vec::new())));
        let kept = (Arc::clone(&heard), Arc::clone(&ends));
        let engine = picked_by("ends", move || {
            let hears = Hears {
                heard: Arc::clone(&kept.0),
                panics: false,
                alone: false,
                takes: EventSet::LEASE,
            };
            Box::new(Ends(hears, Arc::clone(&kept.1)))
        });
        let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first().copied());
        let first = first.expect("a target");
        let worker = Arc::clone(&list[first.position as usize]);
        let now = Instant::now();
        let placement =
            crate::tracking::Placement::taken(engine, decided, later, now, 0, worker, None);
        let mut placement = placement.expect("a placement");
        let ended = |side: Side| Outcome {
            side,
            served: WorkerId(77),
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        let words = || std::mem::take(&mut *heard.lock());
        assert_eq!(words(), [("issued", first.id, 0)]);
        // Unsent: no dispatch is said for it, and no end.
        placement.ended(Side::Encode, ended(Side::Encode));
        assert!(placement.owes_events(), "the request has not ended");
        assert!(words().is_empty(), "an encode side's end is no word");
        placement.sent(Side::Single);
        assert_eq!(words(), [("dispatched", first.id, 0)]);
        // Sent: the request's lease stands.
        placement.ended(Side::Encode, ended(Side::Encode));
        assert!(placement.owes_events(), "the request has not ended");
        assert!(words().is_empty(), "an encode side's end is no word");
        assert!(ends.lock().is_empty());
        // The request's own end. Its host wrote another side there.
        placement.ended(Side::Single, ended(Side::Decode));
        assert!(!placement.owes_events());
        assert_eq!(words(), [("completed", first.id, 0)]);
        let told = Outcome {
            side: Side::Single,
            served: first.id,
            ..ended(Side::Decode)
        };
        assert_eq!(*ends.lock(), [told]);
        drop(placement);
        assert!(words().is_empty(), "the drop says nothing more");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// A placement under no lease owes nothing and says nothing: a place's,
    /// and a whole decision's under a document no stage of which takes a
    /// word. It names the list's own worker, counts under the profile that
    /// picked, and takes its worker's id when it is asked, and no lease's.
    /// The profile that picks is the document's second: a spare one of
    /// another name stands ahead of it.
    #[test]
    fn a_placement_under_no_lease_owes_nothing_and_says_nothing() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        document.generation = 1;
        let spare = document.profiles["round_robin"].clone();
        document.profiles.insert("a_spare".into(), spare);
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
        let list = pool(4);
        let members = engine.pool_members(&list);
        let ended = Outcome {
            side: Side::Single,
            served: WorkerId(0),
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };

        let pick = placed(engine, &Asks::of(request(), Some(&members)));
        assert_eq!(pick.profile, 1, "the document's second profile picked");
        let worker = &list[pick.place as usize];
        let mut placement = crate::tracking::Placement::alone(Arc::clone(worker), pick);
        assert!(!placement.owes_events());
        placement.sent(Side::Single);
        placement.ended(Side::Single, ended);
        assert!(Arc::ptr_eq(&placement.worker, worker));
        assert_eq!(placement.id(), worker_id(worker.as_ref()));
        assert_eq!(
            (placement.decode_id(), placement.profile),
            (None, pick.profile)
        );
        assert!(placement.decode.is_none() && placement.routing_key.is_none());

        // A retry is decided whole, and under this document with no lease.
        let tried = [worker_id(list[0].as_ref())];
        let mut retry = request();
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let picked = decided.decision.pick().expect("a pick");
        let first = picked.targets.first().expect("a target");
        let (profile, target) = (picked.trace.profile, *first);
        assert_eq!(profile, 1, "the document's second profile picked");
        let worker = &list[target.position as usize];
        let now = Instant::now();
        let placement = crate::tracking::Placement::taken(
            engine,
            decided,
            later,
            now,
            0,
            Arc::clone(worker),
            None,
        );
        let mut placement = placement.expect("a placement");
        assert!(!placement.owes_events());
        placement.sent(Side::Single);
        placement.ended(Side::Single, ended);
        assert!(Arc::ptr_eq(&placement.worker, worker));
        assert_eq!((placement.id(), placement.profile), (target.id, profile));
        assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// [`Hears`] as a stage that keeps the span each dispatch states since
    /// its decision.
    #[derive(Debug)]
    struct Spans(Hears, Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for Spans {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, since: u32) {
            self.1.lock().push(since);
        }
    }

    impl Picker for Spans {
        fn name(&self) -> &'static str {
            "spans"
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

    /// A placement dates its lease at the instant its host hands, the
    /// decision's, and reads the clock itself at a send: the span the
    /// dispatch states counts from the one to the other.
    #[test]
    fn a_placements_dispatch_counts_from_the_decisions_instant() {
        let spans = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&spans);
        let engine = picked_by("spans", move || {
            let hears = Hears {
                heard: Heard::default(),
                panics: false,
                alone: false,
                takes: EventSet::LEASE,
            };
            Box::new(Spans(hears, Arc::clone(&kept)))
        });
        let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first());
        let worker = Arc::clone(&list[first.expect("a target").position as usize]);
        // The decision stood a second before its request went out.
        let second = Duration::from_secs(1);
        let decided_at = Instant::now().checked_sub(second);
        let decided_at = decided_at.expect("an earlier instant");
        let placement =
            crate::tracking::Placement::taken(engine, decided, later, decided_at, 0, worker, None);
        let mut placement = placement.expect("a placement");
        placement.sent(Side::Single);
        drop(placement);
        let stated = spans.lock().clone();
        assert_eq!(stated.len(), 1, "one dispatch");
        assert!(stated[0] >= 1_000_000, "{stated:?}");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    /// A caller ends its request with an outcome that names no side and no
    /// worker ([`Outcome::reported`]), and adds what it measured. The
    /// placement writes the worker of the target it took, and what the
    /// caller measured stands.
    #[test]
    fn a_placement_ends_under_an_outcome_that_names_no_worker() {
        let (engine, _, ends) = hearing_ends();
        let list = pool(4);
        let members = engine.pool_members(&list);
        let asks = Asks::of(request(), Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let picked = decided.decision.pick().expect("a pick");
        let first = *picked.targets.first().expect("a target");
        let worker = Arc::clone(&list[first.position as usize]);
        let now = Instant::now();
        let placement =
            crate::tracking::Placement::taken(engine, decided, later, now, 0, worker, None);
        let mut placement = placement.expect("a placement");
        let measured = Outcome {
            tokens_out: 34,
            duration_micros: 5_000,
            ..Outcome::reported(200, TransportClass::Ok)
        };
        assert_eq!(measured.served, WorkerId(0), "no worker named");
        placement.ended(Side::Single, measured);
        let told = Outcome {
            served: first.id,
            ..measured
        };
        assert_eq!(*ends.lock(), [told]);
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

    /// The places sixteen decisions of one request that states `demands`
    /// name over `members`, each as a place or as a whole decision's first
    /// target, or what the request is shed for.
    fn places_named(
        engine: &PolicyEngine,
        members: &PoolMembers,
        demands: Demands,
    ) -> Result<BTreeSet<u32>, (ShedReason, &'static str)> {
        places_named_among(engine, members, demands, None)
    }

    /// [`places_named`], for a request its host holds to `among`.
    fn places_named_among(
        engine: &PolicyEngine,
        members: &PoolMembers,
        demands: Demands,
        among: Option<&[u32]>,
    ) -> Result<BTreeSet<u32>, (ShedReason, &'static str)> {
        let mut stated = request();
        stated.constraints.demands = demands;
        stated.constraints.places = among;
        let asks = Asks::of(stated, Some(members));
        let mut places = BTreeSet::new();
        for _ in 0..16 {
            let place = match answered(engine, &asks) {
                Decided::Alone(pick) => pick.place,
                Decided::Whole(decided, _) => match decided.decision {
                    Decision::Pick(pick) => pick.targets.first().expect("a target").position,
                    Decision::Shed(shed) => return Err((shed.reason, shed.emptied_by)),
                    Decision::Queue(_) => panic!("a queue ticket"),
                },
            };
            places.insert(place);
        }
        Ok(places)
    }

    /// A request that states a hard demand is decided among the workers of
    /// its list that meet it, by what the list's host said of each worker
    /// when it made the list's value, once a worker and at no decision:
    /// every pick lands on one of them, and each of them takes its turn. A
    /// demand no worker meets sheds the list with no endpoint, under that
    /// demand's stage. Of a list its host said nothing of, no worker takes
    /// images and each holds a request of any size.
    #[test]
    fn a_request_that_states_a_demand_is_decided_among_the_workers_that_meet_it() {
        let engine = preset("round_robin");
        let list = pool(4);
        // The second and the fourth worker take images; the fourth states
        // a window, each of the others a runtime.
        let (sees, bounded) = (Arc::clone(&list[1]), Arc::clone(&list[3]));
        let answers = Cell::new(0);
        let said = engine.pool_members_with_fixed_facts(&list, |worker| {
            answers.set(answers.get() + 1);
            let bounded = Arc::ptr_eq(worker, &bounded);
            // Each fact behind the rest of the type: a further fact of a
            // worker adds its field.
            let seeing = FixedFacts {
                vision: bounded || Arc::ptr_eq(worker, &sees),
                ..FixedFacts::default()
            };
            FixedFacts {
                demands: DemandFacts {
                    window: if bounded { 4_096 } else { 0 },
                    runtime: if bounded { 0 } else { 7 },
                    ..DemandFacts::default()
                },
                ..seeing
            }
        });
        let silent = engine.pool_members(&list);
        let asked = |members, demands| places_named(&engine, members, demands);
        let pixels = Demands {
            vision: true,
            ..Demands::default()
        };
        let long = Demands {
            input_window: 5_000,
            ..Demands::default()
        };
        assert_eq!(asked(&said, pixels), Ok(BTreeSet::from([1, 3])));
        assert_eq!(asked(&said, long), Ok(BTreeSet::from([0, 1, 2])));
        let both = Demands {
            vision: true,
            ..long
        };
        assert_eq!(asked(&said, both), Ok(BTreeSet::from([1])));
        let stays = Demands {
            runtime: 7,
            ..Demands::default()
        };
        assert_eq!(asked(&said, stays), Ok(BTreeSet::from([0, 1, 2])));
        // Held to places by its host too: the places that meet the demand,
        // every place where each meets it, and the demand's stage where
        // none does. The classic router's tests of a retry are one test of
        // a candidate too
        // (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:779-782`
        // at main `f55a0a6bf`).
        let among = |places| places_named_among(&engine, &said, pixels, Some(places));
        assert_eq!(among(&[0, 1, 2]), Ok(BTreeSet::from([1])));
        assert_eq!(among(&[1, 3]), Ok(BTreeSet::from([1, 3])));
        let blind = Err((ShedReason::NoEndpoints, "vision"));
        assert_eq!(among(&[0, 2]), blind);
        // Nobody meets both: the second worker alone takes images and is
        // too small, and the others take none. The list is shed under the
        // later of the two demands in a host's order, which is neither
        // the first worker's failure nor the last one's.
        let narrow = engine.pool_members_with_fixed_facts(&list, |worker| {
            let small = Arc::ptr_eq(worker, &sees);
            let seeing = FixedFacts {
                vision: small,
                ..FixedFacts::default()
            };
            FixedFacts {
                demands: DemandFacts {
                    window: if small { 4_096 } else { 0 },
                    ..DemandFacts::default()
                },
                ..seeing
            }
        });
        let too_small = Err((ShedReason::NoEndpoints, "context-window"));
        assert_eq!(asked(&narrow, both), too_small);
        // Nothing said of the list: no worker takes images. The classic
        // router reads a worker with no label as taking them
        // (`model_gateway/src/routers/grpc/multimodal/capability.rs:98-105`
        // at main `c22acc3b9`): a host that follows it says so of each such
        // worker when it makes the list's value.
        assert_eq!(asked(&silent, long), Ok(BTreeSet::from([0, 1, 2, 3])));
        assert_eq!(asked(&silent, pixels), blind);
        assert_eq!(answers.get(), 4, "once a worker, and at no decision");
    }

    /// Over a worker list a worker's rank is read off its object, whatever
    /// the list's host says of the worker. A request no rank can be pinned
    /// for is decided among the workers that registered none, a group of
    /// one left out with the rest, and a list of ranked workers alone is
    /// shed with no endpoint under the rank's stage. That is the gateway's
    /// classic rule: it filters its pool to the workers with no rank before
    /// it selects one, and refuses where none is left
    /// (`model_gateway/src/routers/http/router.rs:1237-1247` at main
    /// `ddd8a44cb`). A request that says nothing of ranks takes any worker.
    #[test]
    fn a_request_no_rank_can_be_pinned_for_stays_off_a_worker_whose_object_registered_one() {
        let engine = preset("round_robin");
        // The object of one rank of a data-parallel group of `size`.
        let ranked = |at: usize, rank: usize, size: usize| -> Arc<dyn Worker> {
            let worker = <BasicWorkerBuilder>::new(format!("http://ranked-{at}:8000"))
                .model(ModelCard::new(MODEL))
                .dp_config(rank, size)
                .health_config(no_health_check())
                .build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker)
        };
        let plain = pool(2);
        // The second worker is a group of one, the fourth a rank of two.
        let list: Arc<[Arc<dyn Worker>]> = Arc::from(vec![
            Arc::clone(&plain[0]),
            ranked(1, 0, 1),
            Arc::clone(&plain[1]),
            ranked(3, 1, 2),
        ]);
        let unpinned = Demands {
            no_rank: true,
            ..Demands::default()
        };
        // A list nothing is said of, and one whose host says the opposite
        // of each worker's rank: both read the objects.
        let silent = engine.pool_members(&list);
        let said = engine.pool_members_with_fixed_facts(&list, |worker| FixedFacts {
            demands: DemandFacts {
                ranked: !worker.is_dp_aware(),
                ..DemandFacts::default()
            },
            ..FixedFacts::default()
        });
        for members in [&silent, &said] {
            let kept = places_named(&engine, members, unpinned);
            assert_eq!(kept, Ok(BTreeSet::from([0, 2])));
            let any = places_named(&engine, members, Demands::default());
            assert_eq!(any, Ok(BTreeSet::from([0, 1, 2, 3])));
        }
        let ranks: Arc<[Arc<dyn Worker>]> = Arc::from(vec![ranked(0, 0, 2), ranked(1, 1, 2)]);
        let ranks = engine.pool_members(&ranks);
        let shed = places_named(&engine, &ranks, unpinned);
        assert_eq!(shed, Err((ShedReason::NoEndpoints, "rank")));
    }

    /// Over a list long enough for the passes that draw, a request that
    /// states a demand is answered a worker that meets it at every
    /// decision, whatever the picker: those passes step aside for a request
    /// held to places.
    #[test]
    fn a_demand_holds_under_a_pick_that_draws_over_a_long_list() {
        for name in [
            "random",
            "power_of_two",
            "round_robin",
            "consistent_hashing",
        ] {
            let engine = preset(name);
            let list = pool(32);
            // Every fourth worker takes images.
            let sees: Vec<_> = list.iter().step_by(4).cloned().collect();
            let members = engine.pool_members_with_fixed_facts(&list, |worker| FixedFacts {
                vision: sees.iter().any(|seeing| Arc::ptr_eq(seeing, worker)),
                ..FixedFacts::default()
            });
            let (mut landed, mut firsts) = (BTreeSet::new(), BTreeSet::new());
            for seed in 0..64 {
                let mut stated = request();
                stated.facts.seed = seed;
                // The ring places a request by its key.
                if name == "consistent_hashing" {
                    stated.facts.session_key = Identity::hashed(seed + 1);
                }
                stated.constraints.demands = Demands {
                    vision: true,
                    ..Demands::default()
                };
                let asks = Asks::of(stated, Some(&members));
                let outcome = decided(&engine, &asks);
                let pick = outcome.decision.pick().expect("a pick");
                firsts.insert(pick.targets.first().expect("a target").position);
                let places = pick.targets.iter().map(|target| target.position);
                landed.extend(places);
            }
            // Every target takes images, the alternates too, and the first
            // targets are more than one worker.
            let seeing = landed.iter().all(|place| place % 4 == 0);
            assert!(seeing, "{name}: {landed:?} take images");
            assert!(firsts.len() > 1, "{name}: {firsts:?}");
        }
    }

    /// Where a binding of the request's phase pins a key, a request that
    /// states a hard demand is decided among the workers that meet it, as
    /// a request held to places is: a pin on a worker that meets the
    /// demand is followed, and a pin on one that does not is not. That is
    /// the gateway's classic answer: it narrows its candidates by the
    /// request's own test before its policy runs
    /// (`model_gateway/src/routers/common/placement.rs:508-509` at main
    /// `f55a0a6bf`), and its sticky policy looks a key's pinned workers up
    /// among the workers it is handed and picks another of them where none
    /// is there (`model_gateway/src/policies/manual.rs:303-308`).
    #[test]
    fn a_demand_under_a_pinned_key_is_decided_among_the_workers_that_meet_it() {
        // The stock sessions rule: its binding pins a key.
        let engine = preset("manual");
        let list = pool(8);
        // Every other worker takes images.
        let sees: Vec<_> = list.iter().step_by(2).cloned().collect();
        let members = engine.pool_members_with_fixed_facts(&list, |worker| FixedFacts {
            vision: sees.iter().any(|seeing| Arc::ptr_eq(seeing, worker)),
            ..FixedFacts::default()
        });
        // One request of `key` that states `demands`, served by its first
        // target: that target's place.
        let sent = |key: u64, demands: Demands| {
            let mut keyed = request();
            keyed.facts.seed = key;
            keyed.facts.session_key = Identity::hashed(key);
            keyed.constraints.demands = demands;
            let asks = Asks::of(keyed, Some(&members));
            served(&engine, &asks, Instant::now()).position
        };
        let pixels = Demands {
            vision: true,
            ..Demands::default()
        };
        let (mut followed, mut left) = (0, 0);
        for key in 1..=16 {
            // The key's first request states nothing and is pinned.
            let pin = sent(key, Demands::default());
            assert_eq!(sent(key, Demands::default()), pin, "key {key}: its pin");
            let went = sent(key, pixels);
            assert_eq!(went % 2, 0, "key {key}: place {went} takes images");
            if pin % 2 == 0 {
                assert_eq!(went, pin, "key {key}: its pin meets the demand");
                followed += 1;
            } else {
                left += 1;
            }
        }
        assert!(
            followed > 0 && left > 0,
            "{followed} pins followed, {left} left"
        );
    }
}
