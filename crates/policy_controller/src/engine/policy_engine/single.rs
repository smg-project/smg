//! A single request's whole decision in a gateway's own process.
//!
//! **The short path, in front** ([`alone`]). Where the document decides the
//! request's place by one pick of its picker
//! ([`Place::Alone`](crate::engine::Place::Alone)), the decision is that
//! pick: one turn of the list's own rotation and one read of the worker the
//! turn lands on. It writes no record and takes no lock. It answers a plain
//! request ([`Plain`]) over a list that holds an eligible worker where the
//! picker's turns land, and it reads no fact of a request: the engine asks
//! it before the request is made, so such a request is never made.
//!
//! **A key's rows, between** ([`keys::decide`]). A request that is made,
//! under a sessions rule or a ring, is decided by the rows its key names:
//! its key's pin, which is one worker's record, or its key's owner on the
//! list's ring and the members behind it. It answers where those rows say
//! what the whole pass would, and nothing where they do not.
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

use std::{sync::atomic::Ordering, time::Instant};

use smg_worker::runtime::RoutingState;

use super::{keys, overlap, PolicyEngine, PoolMembers};
use crate::{
    engine::{
        decision::{Decision, Shed, ShedReason, Target, Trace},
        decision_engine::{AlonePick, HeldDecision, Plain},
        framework::Engine,
        records,
        sticky_sessions::SessionVerdict,
        store_backed_engine::{DecideOutcome, Leased, Request, SUBSET_HINT, WIRES},
        view::{Phase, Session},
    },
    stages::Lifecycle,
    store::{leases::DISPATCH_TTL, Said},
    tracking::lease::{EventKind, EventSet, Lease, LeaseId, LeaseKind, Side},
};

/// The stage a request that states a hard demand is declined under: no
/// stage of this engine reads a worker's facts for one.
const DEMANDS: &str = "demands";

/// The crate's word, in a shed's `emptied_by`, for a pool no stage emptied.
const NO_STAGE: &str = "none";

/// The short path: the place in `members`, the list its host handed, of
/// the worker picked for a plain single request of `model`, and the profile
/// whose picker picked it ([`AlonePick`]). `None` where the request takes
/// the whole pass ([`decide`]): it is no single request, its place is not
/// decided by one pick of a picker, a worker of the list is not dialed on a
/// wire it states, or the picker's turns land on no eligible worker.
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
/// written for the request from the list's workers. A request that states
/// what no stage here runs is declined under that stage's name. The list is
/// borrowed as long as the request: its ring is lent into the request's
/// facts.
pub(super) fn decide<'r>(
    own: &PolicyEngine,
    request: &mut Request<'r>,
    members: &'r PoolMembers,
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
    // No load snapshot and no session key's text are handed to this call
    // yet: a record carries no report and no count of a key's requests.
    let live = records::Live::at(spec.engine(), &own.clock, now, None, None);
    let facts = &mut request.facts;
    let (engine, workers) = (spec.engine(), members.workers());
    // A keyed request is decided by the rows its key names, where they say
    // what the whole pass would: its key's pin, or its key's owner on the
    // list's ring and the members behind it.
    if let Some(keyed) = keys::decide(engine, members.ring(), records, workers, &live, facts, &cx) {
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
/// under the name of the stage that would have run it.
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

/// Tells `engine`'s stages one word of a single's lease, from what the
/// request's guard lends back, as a file of leases tells it: a dispatch
/// under the lease as issued with the index the word names, on the target
/// taken, by its id and rank; an end under the lease moved to that target;
/// a release under the lease as issued.
pub(super) fn said(engine: &Engine, held: &HeldDecision<'_>, said: Said) {
    match said {
        Said::Dispatched(dispatch, _) => {
            let lease = Lease {
                target_index: dispatch.target_index,
                ..*held.lease
            };
            let target = Target {
                id: held.target.id,
                dp_rank: held.target.dp_rank,
                ..Target::default()
            };
            engine.on_dispatched(
                &lease,
                &target,
                dispatch.side,
                dispatch.since_decision_micros,
            );
        }
        Said::Completed(outcome) => {
            let lease = Lease {
                target_index: held.index,
                booked: held.target.id,
                ..*held.lease
            };
            engine.on_completed(&lease, &outcome);
        }
        Said::Released(_, reason) => engine.on_released(held.lease, reason),
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, sync::Arc};

    use parking_lot::Mutex;
    use smg_worker::{identity::basic::Worker, test_support::ready_worker};

    use super::*;
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{
            decision::{LiftedVeto, Targets},
            decision_engine::{Decided, DecisionEngine, RequestMaker, WorkerLists},
            framework::{Picker, Place, PoolWord, TieBand},
            prompt_hashes::HashSeeds,
            records::worker_id,
            store_backed_engine::DecisionRefused,
            view::{Candidate, Constraints, Identity, RequestFacts, WireModes, WorkerId},
        },
        policies::random::Random,
        stages::registry::StageRegistry,
        store::GatewayId,
        tracking::{
            lease::{Outcome, ReleaseReason, TransportClass},
            DecisionGuard,
        },
    };

    const MODEL: &str = "model-a";

    fn preset(name: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
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
        made: Cell<u32>,
    }

    impl<'a> Asks<'a> {
        fn of(request: Request<'a>, members: Option<&'a PoolMembers>) -> Self {
            Self {
                request,
                members,
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
        for request in [retried, missed, hinted, lifted, later, masked] {
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
        let mut stages = StageRegistry::builtin();
        let added = stages.picker("hears", move || {
            Box::new(Hears {
                heard: Arc::clone(&kept),
                panics,
                alone,
                takes,
            })
        });
        added.expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a built-in preset");
        document.generation = 1;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "hears".into();
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        (
            PolicyEngine::new(spec.expect("the stages hold the picker")),
            heard,
        )
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
        let told = |kind: EventKind| {
            let (engine, heard) = hearing_as(false, false, EventSet::NONE.with(kind));
            let list = pool(4);
            let members = engine.pool_members(&list);
            let asks = Asks::of(request(), Some(&members));
            let now = Instant::now();
            let mut handle = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
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
            let unsent = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
            drop(unsent);
            assert_eq!((engine.faults(), engine.unheard()), (0, 0));
            let words: Vec<&'static str> = heard.lock().iter().map(|heard| heard.0).collect();
            words
        };
        assert_eq!(told(EventKind::Issued), ["issued", "issued"]);
        assert_eq!(told(EventKind::Dispatched), ["dispatched"]);
        assert_eq!(told(EventKind::Completed), ["completed"]);
        assert_eq!(told(EventKind::Released), ["released"]);
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
}
