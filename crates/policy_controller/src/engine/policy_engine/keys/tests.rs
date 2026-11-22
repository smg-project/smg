//! The cases of a keyed request's decision in a gateway's own process: each
//! asked through the engine's own entry, over a list of worker objects, and
//! held to the whole decision of a twin engine over every worker's record.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::{identity::basic::Worker, test_support::ready_worker, worker::WorkerStatus};

use crate::{
    document::spec::{
        AppliedSpec, Assignment, KeySourceSpec, PolicySpec, Recovery, RoutingKeySpec, SessionsSpec,
    },
    engine::{
        decision::{Decision, MAX_TARGETS},
        decision_engine::{DecisionEngine, RequestMaker, WorkerLists},
        policy_engine::{PolicyEngine, PoolMembers},
        records::{self, Live},
        sticky_sessions::SessionVerdict,
        store_backed_engine::{DecideOutcome, Request},
        view::{fingerprint, Constraints, Identity, RequestFacts, WorkerId},
    },
    store::GatewayId,
    tracking::{
        lease::{Outcome, Side, TransportClass},
        DecisionGuard,
    },
};

const MODEL: &str = "model-a";

/// The engine of a stock document.
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

/// A request stated ahead, with the list its host hands for it.
struct Asks<'a> {
    request: Request<'a>,
    members: &'a PoolMembers,
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members)
    }
}

/// A request of the session key `key`, with no key where it is `0`, over
/// `members`.
fn keyed(key: u64, members: &PoolMembers) -> Asks<'_> {
    Asks {
        request: Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed: key,
                session_key: Identity::hashed(key),
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        },
        members,
    }
}

fn decided(engine: &PolicyEngine, asks: &Asks<'_>) -> DecideOutcome {
    let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
    let whole = decided.expect("an answer").whole();
    whole.expect("a whole decision").0
}

/// The whole decision of `asks`' request over every worker's record, by
/// `twin`: an engine of the same document that is lent no ring and learns
/// its own.
fn whole(twin: &PolicyEngine, asks: &Asks<'_>) -> Decision {
    let spec = twin.spec();
    let members = twin.pool_members(asks.members.workers());
    let records = members.records(spec, twin.clock());
    let live = Live::at(spec.engine(), twin.clock(), Instant::now(), None, None);
    let mut facts = asks.request.facts;
    let cx = asks.request.constraints;
    records::decide(
        spec.engine(),
        records,
        members.workers(),
        &live,
        &mut facts,
        &cx,
    )
}

/// The workers a decision names, in its order, by place and by id.
fn targets(decision: &Decision) -> Vec<(u32, WorkerId)> {
    let pick = decision.pick().expect("a pick");
    pick.targets
        .iter()
        .map(|target| (target.position, target.id))
        .collect()
}

fn served(worker: WorkerId) -> Outcome {
    Outcome {
        side: Side::Single,
        served: worker,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    }
}

/// Under the stock ring a keyed request goes to its key's owner and the
/// members behind it with no pass over the list: the workers the whole
/// decision of a twin engine names over every worker's record, in its
/// order, while every worker is ready and with one worker in four not
/// ready. The list's ring is the one the gateway filled, off the path of a
/// request. A request with no key is the whole pass's.
#[test]
fn a_rings_keyed_request_goes_to_its_keys_owner_with_no_pass_over_the_list() {
    let (engine, twin) = (preset("consistent_hashing"), preset("consistent_hashing"));
    let list = pool(64);
    let members = engine.pool_members(&list);
    assert!(
        engine.fill_ring(MODEL, &members),
        "the stock ring reads one"
    );
    let ring = members.ring().get().expect("the list's ring");
    assert_eq!(ring.len(), 64);
    for round in 0..2 {
        let mut by_walk = 0;
        for key in 1..=200 {
            let asks = keyed(key, &members);
            let decision = decided(&engine, &asks).decision;
            assert_eq!(
                targets(&decision),
                targets(&whole(&twin, &asks)),
                "key {key}"
            );
            assert_eq!(targets(&decision).len(), MAX_TARGETS);
            let trace = &decision.pick().expect("a pick").trace;
            by_walk += u32::from(trace.drew_first());
        }
        // A walk that meets eight workers not ready is the whole pass's.
        let least = if round == 0 { 200 } else { 150 };
        assert!(
            by_walk >= least,
            "round {round}: {by_walk} of 200 by the walk"
        );
        for worker in list.iter().step_by(4) {
            worker.set_status(WorkerStatus::NotReady);
        }
    }
    let keyless = decided(&engine, &keyed(0, &members)).decision;
    let trace = &keyless.pick().expect("a pick").trace;
    assert!(!trace.drew_first(), "a pass over the list ran");
}

/// No decision builds a list's ring and none waits for one. Until the
/// gateway has filled it, a keyed request under the stock ring is placed as
/// one that carries no key: by the profile's draw over the whole list, so
/// one key is handed more workers than one. Once the ring is filled the key
/// keeps its owner. Under a document that reads no ring a list takes none.
#[test]
fn a_keyed_request_ahead_of_its_lists_ring_is_drawn_and_builds_none() {
    let engine = preset("consistent_hashing");
    let list = pool(64);
    let members = engine.pool_members(&list);
    let mut asks = keyed(7, &members);
    let first = |decision: &Decision| targets(decision)[0];
    let mut drawn = Vec::new();
    // Each request draws by a seed of its own, as a gateway's requests do.
    for seed in 0..24 {
        asks.request.facts.seed = seed;
        let decision = decided(&engine, &asks).decision;
        assert_eq!(targets(&decision).len(), MAX_TARGETS, "placed, not refused");
        let trace = &decision.pick().expect("a pick").trace;
        assert!(!trace.drew_first(), "a pass over the list placed it");
        if !drawn.contains(&first(&decision)) {
            drawn.push(first(&decision));
        }
    }
    assert!(drawn.len() > 1, "one key, one worker: no draw ran");
    assert!(members.ring().get().is_none(), "a decision built the ring");

    assert!(engine.fill_ring(MODEL, &members));
    let owner = first(&decided(&engine, &asks).decision);
    for seed in 24..48 {
        asks.request.facts.seed = seed;
        assert_eq!(first(&decided(&engine, &asks).decision), owner);
    }

    let plain = preset("round_robin");
    let members = plain.pool_members(&list);
    assert!(!plain.fill_ring(MODEL, &members));
    assert!(members.ring().get().is_none(), "no ring is built");
}

/// Under the stock sessions rule a key's first request is placed by the
/// whole pass, and its dispatch makes the key's pin in the engine's session
/// table. The key's next request follows that pin by one worker's record,
/// with no pass over the list. A pinned worker that is not ready gives the
/// key back to the whole pass, which places it anew. No ring is built for a
/// document that reads none.
#[test]
fn a_sessions_pin_is_made_at_the_dispatch_and_followed_with_no_pass_over_the_list() {
    let engine = preset("manual");
    let list = pool(16);
    let members = engine.pool_members(&list);
    let now = Instant::now();
    let asks = keyed(41, &members);
    let mut first =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let placed = *first.decision().pick().expect("a pick");
    assert!(matches!(placed.session_verdict, SessionVerdict::Claimed(_)));
    assert!(
        !placed.trace.drew_first(),
        "the whole pass places a new key"
    );
    let pinned = first.take(0).expect("the first target");
    first.dispatched(now);
    first.completed(&served(pinned.id), now);
    drop(first);

    for _ in 0..3 {
        let followed = decided(&engine, &asks).decision;
        let pick = followed.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        assert_eq!(targets(&followed), [(pinned.position, pinned.id)]);
        assert!(pick.trace.drew_first(), "no pass over the list ran");
    }
    // Another key holds no pin: the whole pass places it.
    let other = decided(&engine, &keyed(42, &members)).decision;
    assert!(!other.pick().expect("a pick").trace.drew_first());

    list[pinned.position as usize].set_status(WorkerStatus::NotReady);
    let moved = decided(&engine, &asks).decision;
    let pick = moved.pick().expect("a pick");
    assert!(
        !pick.trace.drew_first(),
        "the whole pass places the key anew"
    );
    assert!(pick.targets.iter().all(|target| target.id != pinned.id));
    assert!(members.ring().get().is_none(), "no ring is built");
}

/// The engine's own pass drops the session keys past their end, off the
/// path of a request: a pin made at a dispatch is held through the rule's
/// idle time and gone after it.
#[test]
fn the_engines_pass_drops_the_session_keys_past_their_end() {
    let engine = preset("manual");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let now = Instant::now();
    let asks = keyed(51, &members);
    let mut first =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let pinned = first.take(0).expect("the first target");
    first.dispatched(now);
    first.completed(&served(pinned.id), now);
    drop(first);
    let sessions = engine.spec().engine().sessions();
    assert_eq!(sessions.len(), 1);
    // The stock rule forgets a key four hours after its last request.
    engine.sweep(now + Duration::from_secs(14_000));
    assert_eq!(sessions.len(), 1, "inside the idle time");
    engine.sweep(now + Duration::from_secs(14_500));
    assert_eq!(sessions.len(), 0, "past its end");
}

/// The engine of the document the routing settings mean for round robin
/// under the sticky override: the policy's own stages, and on its binding
/// the session rule of two requests of a key a worker, with the key's
/// sources under `routing_key` and the key's header in the view.
fn sticky_override() -> PolicyEngine {
    const HEADER: &str = "x-smg-routing-key";
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 1;
    document.view.headers = vec![HEADER.to_owned()];
    let binding = document.bindings.first_mut().expect("the preset's binding");
    binding.routing_key = Some(RoutingKeySpec {
        sources: vec![
            KeySourceSpec {
                header: Some(HEADER.to_owned()),
                ..KeySourceSpec::default()
            },
            KeySourceSpec {
                rid: true,
                ..KeySourceSpec::default()
            },
        ],
        strip_lineage: true,
    });
    binding.sessions = Some(SessionsSpec {
        sticky_header: String::new(),
        headers: Vec::new(),
        ttl_s: 14_400,
        sweep_s: 60,
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        issue_session_id: false,
        recovery: Recovery::Latest,
    });
    PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
}

/// A request of the session key `key` as its host states it, over
/// `members`.
fn session<'a>(key: Identity<'a>, members: &'a PoolMembers) -> Asks<'a> {
    Asks {
        request: Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed: 7,
                session_key: key,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        },
        members,
    }
}

/// Under the sticky override's rule a worker carries two requests of one
/// key, and the count is the worker object's own: what the load guard of
/// each dispatch writes under the key's text. The engine reads it there for
/// a request whose key comes with its text. One request of the key leaves
/// the pin standing. At two, the stages place the next request on another
/// worker, and its commit would front that worker. Another key's requests
/// on the worker count for nothing. A key that comes as its fingerprint
/// alone has no text to read a count by, and keeps its pin at any count:
/// the document lists the key's header in its view so that it has one.
#[test]
fn a_capped_rule_reads_a_keys_requests_on_the_worker_object() {
    let engine = sticky_override();
    assert!(engine.spec().engine().needs().session_in_flight);
    let list = pool(4);
    let members = engine.pool_members(&list);
    let now = Instant::now();
    let asks = session(Identity::plain("conv"), &members);
    let mut first =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let pinned = first.take(0).expect("the first target");
    first.dispatched(now);
    first.completed(&served(pinned.id), now);
    drop(first);
    let worker = &list[pinned.position as usize];
    let follows = |asks: &Asks<'_>| {
        let followed = decided(&engine, asks).decision;
        let verdict = followed.pick().expect("a pick").session_verdict;
        (targets(&followed), verdict)
    };
    let on_its_pin = (vec![(pinned.position, pinned.id)], SessionVerdict::Pinned);

    for _ in 0..3 {
        worker.increment_routing_key_load("another");
    }
    worker.increment_routing_key_load("conv");
    assert_eq!(follows(&asks), on_its_pin, "one request of the key");

    worker.increment_routing_key_load("conv");
    let (moved, verdict) = follows(&asks);
    assert_eq!(verdict, SessionVerdict::Respill);
    assert!(
        moved.iter().all(|(_, id)| *id != pinned.id),
        "two requests of the key: {moved:?}"
    );

    // The same key with no text: no count is read for it.
    let hashed = Identity::hashed(fingerprint(b"conv"));
    assert_eq!(follows(&session(hashed, &members)), on_its_pin);

    // The key's requests answered, it is its pin's again.
    worker.decrement_routing_key_load("conv");
    worker.decrement_routing_key_load("conv");
    assert_eq!(follows(&asks), on_its_pin);
}
