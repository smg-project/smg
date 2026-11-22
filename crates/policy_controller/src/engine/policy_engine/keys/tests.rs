//! The cases of a keyed request's decision in a gateway's own process: each
//! asked through the engine's own entry, over a list of worker objects.

use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use smg_worker::{identity::basic::Worker, test_support::ready_worker, worker::WorkerStatus};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::Decision,
        decision_engine::{DecideOutcome, DecisionEngine, Request, RequestMaker, WorkerLists},
        policy_engine::{PolicyEngine, PoolMembers},
        records,
        sticky_sessions::SessionVerdict,
        view::{Constraints, Identity, PinSource, RequestFacts, Session, WorkerId},
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

/// Under either stock ring document a keyed request over a list of worker
/// objects is placed by the ring stage's own ring, and its host builds
/// nothing for it: a key names the same workers in the same order whatever
/// its request's seed, and the keys of a list do not all go to one worker.
///
/// The gateway's own policies keep a key on one worker too, the first
/// eligible one their ring reaches from the key
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 99 to 115 at
/// 882be6f4f, and `prefix_hash.rs`, lines 291 to 315). The stage's own ring
/// is another ring than theirs: which worker a key takes here is not the
/// worker it takes there.
#[test]
fn a_rings_keyed_request_keeps_its_keys_workers_whatever_its_seed() {
    for ring in ["consistent_hashing", "prefix_hash"] {
        let engine = preset(ring);
        let list = pool(8);
        let members = engine.pool_members(&list);
        let named = |key: u64, seed: u64| {
            let mut asks = keyed(key, &members);
            asks.request.facts.seed = seed;
            targets(&decided(&engine, &asks).decision)
        };
        let mut owners = Vec::new();
        for key in 1..=32 {
            let first = named(key, 0);
            for seed in 1..8 {
                assert_eq!(named(key, seed), first, "{ring}: key {key} at seed {seed}");
            }
            owners.push(first[0]);
        }
        owners.sort_unstable();
        owners.dedup();
        assert!(owners.len() > 1, "{ring}: every key went to one worker");
    }
}

/// A worker that leaves a list stays a member of the ring stage's own ring
/// until its host says it is gone ([`PolicyEngine::forget`]): a list
/// published without it tells that ring nothing. It changes no answer:
/// over the list without that worker every key names the workers an engine
/// that never saw the worker names.
///
/// The gateway's own policy walks a ring kept from another membership past
/// a worker its list holds no more
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 99 to 121 at
/// 882be6f4f): no other key moves there either.
#[test]
fn a_worker_that_left_its_list_stays_in_the_stages_ring_and_moves_no_other_key() {
    for ring in ["consistent_hashing", "prefix_hash"] {
        let (seen, fresh) = (preset(ring), preset(ring));
        let named = |engine: &PolicyEngine, list: &Arc<[Arc<dyn Worker>]>, key: u64| {
            let members = engine.pool_members(list);
            targets(&decided(engine, &keyed(key, &members)).decision)
        };
        // The stage's ring learns all eight workers.
        let whole = pool(8);
        for key in 1..=64 {
            named(&seen, &whole, key);
        }
        // The fourth leaves: the list of the seven others.
        let others = whole.iter().enumerate().filter(|&(at, _)| at != 3);
        let rest: Arc<[Arc<dyn Worker>]> = others.map(|(_, worker)| Arc::clone(worker)).collect();
        for key in 1..=64 {
            let (stale, never) = (named(&seen, &rest, key), named(&fresh, &rest, key));
            assert_eq!(stale, never, "{ring}: key {key}");
        }
    }
}

/// Under the stock sessions rule a key's first request is placed by the
/// whole pass, and its dispatch makes the key's pin in the engine's session
/// table. The key's next request follows that pin by one worker's record,
/// with no pass over the list. A pinned worker that is not ready gives the
/// key back to the whole pass, which places it anew.
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
    // The stock rule forgets a key at its first sweep, one a minute, four
    // hours after its last request.
    engine.sweep(now + Duration::from_secs(14_000));
    assert_eq!(sessions.len(), 1, "inside the idle time");
    engine.sweep(now + Duration::from_secs(14_500));
    assert_eq!(sessions.len(), 0, "past its end");
}

/// A key used again after its idle time and ahead of its rule's next sweep
/// follows its pin, and one used at that sweep is placed anew. The gateway's
/// sticky map ends an idle entry at an eviction tick alone, and every use
/// renews it (`model_gateway/src/policies/manual.rs:395`, `:234`).
#[test]
fn a_key_idle_past_its_time_keeps_its_worker_until_the_sweep() {
    // Idle after a minute, swept every hour of the engine's clock.
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    document.generation = 1;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the stock rule");
    (rule.ttl_s, rule.sweep_s) = (60, 3_600);
    let engine = PolicyEngine::new(AppliedSpec::validate(&document).expect("it validates"));
    let list = pool(4);
    let members = engine.pool_members(&list);
    let (asks, now) = (keyed(61, &members), Instant::now());
    // A request of the key, dispatched and served: what its rule did, and where.
    let served_at = |secs: u64| {
        let at = now + Duration::from_secs(secs);
        let decided = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, at);
        let mut decided = decided.expect("a decision");
        let verdict = decided.decision().pick().expect("a pick").session_verdict;
        let target = decided.take(0).expect("the first target");
        decided.dispatched(at);
        decided.completed(&served(target.id), at);
        (verdict, target.id)
    };
    let (placed, worker) = served_at(0);
    assert!(matches!(placed, SessionVerdict::Claimed(_)));
    assert_eq!(served_at(61), (SessionVerdict::Pinned, worker));
    assert!(matches!(served_at(3_600).0, SessionVerdict::Claimed(_)));
}

/// Pins of record: one worker for every key, and whether it was asked.
#[derive(Debug)]
struct Elsewhere(WorkerId, OnceLock<()>);

impl PinSource for Elsewhere {
    fn pinned(&self, _: Session, _: Option<u64>) -> Option<WorkerId> {
        let _ = self.1.set(());
        Some(self.0)
    }
}

/// This engine reads no pins of record: a request that lends a record naming
/// another worker is decided as one that lends none, where the whole pass
/// places its key and where its pin is followed. The record is never asked.
#[test]
fn a_request_that_lends_pins_of_record_is_decided_as_one_that_lends_none() {
    let (engine, twin) = (preset("manual"), preset("manual"));
    let list = pool(16);
    let (members, others) = (engine.pool_members(&list), twin.pool_members(&list));
    // The whole pass places the key, then the key's pin is followed.
    let twice = |engine: &PolicyEngine, asks: &Asks<'_>| {
        [(); 2].map(|()| targets(&decided(engine, asks).decision))
    };
    let plain = twice(&twin, &keyed(41, &others));
    let mut ids = list
        .iter()
        .map(|worker| records::worker_id(worker.as_ref()));
    let other = ids.find(|id| *id != plain[0][0].1).expect("another worker");
    let record = Elsewhere(other, OnceLock::new());
    let mut lent = keyed(41, &members);
    lent.request.constraints.pins = Some(&record);
    assert_eq!(twice(&engine, &lent), plain);
    assert!(record.1.get().is_none(), "the record is not asked");
}
