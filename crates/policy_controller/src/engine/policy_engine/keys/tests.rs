//! The cases of a keyed request's decision in a gateway's own process: each
//! asked through the engine's own entry, over a list of worker objects. A
//! ring's are held to the host's own lookup on the ring it hands
//! ([`HashRing::find_healthy_url`]), and to the whole decision of a twin
//! engine over every worker's record, lent the same ring.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::{identity::basic::Worker, test_support::ready_worker, worker::WorkerStatus};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, MAX_TARGETS},
        decision_engine::{DecisionEngine, MembershipRing, RequestMaker, WorkerLists},
        policy_engine::{PolicyEngine, PoolMembers},
        records::{self, Live},
        sticky_sessions::SessionVerdict,
        store_backed_engine::{DecideOutcome, Request},
        view::{Constraints, Identity, RequestFacts, WorkerId},
    },
    policies::consistent_hashing::{HashRing, RingPlaces},
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

/// A request stated ahead, with the list its host hands for it and the
/// hash ring its host holds for the list: the ring, and whether it is of
/// the list's own membership.
struct Asks<'a> {
    request: Request<'a>,
    members: &'a PoolMembers,
    ring: Option<(&'a Arc<HashRing>, bool)>,
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

    fn ring(&self) -> Option<MembershipRing<'_>> {
        self.ring.map(|(ring, of_this_membership)| MembershipRing {
            ring,
            of_this_membership,
        })
    }
}

/// The text of the session key `key`, which a host's ring places it by.
fn text(key: u64) -> String {
    format!("session-{key}")
}

/// A request of the session key `key`, with no key where it is `0`, over
/// `members`: its host states where the key's text stands on its ring, and
/// hands no ring.
fn keyed(key: u64, members: &PoolMembers) -> Asks<'_> {
    let ring_position = match key {
        0 => 0,
        key => HashRing::hash_position(&text(key)),
    };
    Asks {
        request: Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed: key,
                session_key: Identity::hashed(key),
                ring_position,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        },
        members,
        ring: None,
    }
}

/// The address the host's own lookup answers for `key` on `ring`, among
/// the workers of `list` its classic policy would take.
fn looked_up<'a>(ring: &'a HashRing, list: &[Arc<dyn Worker>], key: u64) -> Option<&'a str> {
    ring.find_healthy_url(&text(key), |address| {
        let mut here = list.iter().filter(|worker| worker.url() == address);
        here.any(|worker| worker.is_healthy_and_eligible())
    })
}

fn decided(engine: &PolicyEngine, asks: &Asks<'_>) -> DecideOutcome {
    let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
    let whole = decided.expect("an answer").whole();
    whole.expect("a whole decision").0
}

/// The whole decision of `asks`' request over every worker's record, by
/// `twin`: an engine of the same document, lent the ring `asks` hands as
/// the list reads it.
fn whole(twin: &PolicyEngine, asks: &Asks<'_>) -> Decision {
    let spec = twin.spec();
    let members = twin.pool_members(asks.members.workers());
    let records = members.records(spec, twin.clock());
    let live = Live::at(spec.engine(), twin.clock(), Instant::now(), None, None);
    let places = asks
        .ring
        .map(|(ring, _)| RingPlaces::new(ring, records.ids()));
    let lent = places.as_ref().map(RingPlaces::list_ring);
    let mut facts = RequestFacts {
        ring: lent.as_ref(),
        ..asks.request.facts
    };
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

/// Under the stock ring a keyed request goes to its key's owner on its
/// host's ring and the members behind it, with no pass over the list: the
/// owner is the worker the host's own lookup answers for the key's text,
/// and the workers named are those the whole decision of a twin engine
/// names over every worker's record, in its order. So while every worker is
/// ready, and with one worker in four not ready. The list's table of the
/// ring is made once, by its first decision. A request with no key is the
/// whole pass's.
#[test]
fn a_rings_keyed_request_goes_to_its_keys_owner_with_no_pass_over_the_list() {
    let (engine, twin) = (preset("consistent_hashing"), preset("consistent_hashing"));
    let list = pool(64);
    let members = engine.pool_members(&list);
    let ring = Arc::new(HashRing::new(list.iter().map(|worker| worker.url())));
    assert!(members.ring_places().get().is_none(), "no table yet");
    for round in 0..2 {
        let mut by_walk = 0;
        for key in 1..=200 {
            let asks = Asks {
                ring: Some((&ring, true)),
                ..keyed(key, &members)
            };
            let decision = decided(&engine, &asks).decision;
            assert_eq!(
                targets(&decision),
                targets(&whole(&twin, &asks)),
                "key {key}"
            );
            assert_eq!(targets(&decision).len(), MAX_TARGETS);
            let owner = targets(&decision)[0].0 as usize;
            assert_eq!(
                Some(list[owner].url()),
                looked_up(&ring, &list, key),
                "key {key}: the host's lookup names the owner"
            );
            let trace = &decision.pick().expect("a pick").trace;
            by_walk += u32::from(trace.drew_first());
        }
        // A walk that meets eight workers not ready is the whole pass's.
        let least = if round == 0 { 200 } else { 150 };
        assert!(
            by_walk >= least,
            "round {round}: {by_walk} of 200 by the walk"
        );
        let places = members.ring_places().get().expect("the list's table");
        assert!(places.is_of(&ring), "made once, of the ring handed");
        for worker in list.iter().step_by(4) {
            worker.set_status(WorkerStatus::NotReady);
        }
    }
    let keyless = decided(&engine, &keyed(0, &members)).decision;
    let trace = &keyless.pick().expect("a pick").trace;
    assert!(!trace.drew_first(), "a pass over the list ran");
}

/// No decision builds a ring and none waits for one. While its host holds
/// no ring for the list, a keyed request under the stock ring is placed by
/// the host's rule for a request with no ring: the worker at its key's
/// position modulo the workers it may take, in the list's order, so one
/// key keeps one worker. A ring of another membership is walked with no
/// table made, and the ring of the list's own membership through a table
/// made once. Under a document that reads no ring, the ring handed is not
/// read and no table is made.
#[test]
fn a_keyed_request_is_placed_as_its_host_places_it_ahead_of_the_ring_and_on_it() {
    let engine = preset("consistent_hashing");
    let list = pool(64);
    let members = engine.pool_members(&list);
    let mut asks = keyed(7, &members);
    let position = asks.request.facts.ring_position;
    let first = |decision: &Decision| targets(decision)[0].0 as usize;
    // No ring handed: the key's position over the list, at every seed.
    for seed in 0..24 {
        asks.request.facts.seed = seed;
        let decision = decided(&engine, &asks).decision;
        let trace = &decision.pick().expect("a pick").trace;
        assert!(!trace.drew_first(), "a pass over the list placed it");
        assert_eq!(first(&decision) as u64, position % 64, "seed {seed}");
    }
    // A worker that is not ready takes no key: the position is taken over
    // the workers that are.
    list[3].set_status(WorkerStatus::NotReady);
    let ready: Vec<usize> = (0..64).filter(|&place| place != 3).collect();
    let decision = decided(&engine, &asks).decision;
    assert_eq!(first(&decision), ready[(position % 63) as usize]);
    list[3].set_status(WorkerStatus::Ready);
    assert!(members.ring_places().get().is_none(), "no table was made");

    // The ring of an earlier membership, which the first four workers had
    // not joined: the key's owner on it, by the host's lookup, with no
    // table made.
    let earlier = Arc::new(HashRing::new(list[4..].iter().map(|worker| worker.url())));
    for key in 1..=200 {
        let asks = Asks {
            ring: Some((&earlier, false)),
            ..keyed(key, &members)
        };
        let decision = decided(&engine, &asks).decision;
        assert_eq!(
            Some(list[first(&decision)].url()),
            looked_up(&earlier, &list, key),
            "key {key}"
        );
        assert!(first(&decision) >= 4, "a worker that joined takes no key");
    }
    assert!(members.ring_places().get().is_none(), "no table was made");

    // The ring of the list's own membership: the owner by the host's
    // lookup, through the table the first decision makes.
    let own = Arc::new(HashRing::new(list.iter().map(|worker| worker.url())));
    let mut joined = 0;
    for key in 1..=200 {
        let asks = Asks {
            ring: Some((&own, true)),
            ..keyed(key, &members)
        };
        let decision = decided(&engine, &asks).decision;
        assert_eq!(
            Some(list[first(&decision)].url()),
            looked_up(&own, &list, key),
            "key {key}"
        );
        joined += u32::from(first(&decision) < 4);
        let places = members.ring_places().get().expect("the list's table");
        assert!(places.is_of(&own));
    }
    assert!(
        joined > 0,
        "the workers that joined take keys on their ring"
    );
    // A ring of another membership handed behind it is still read as one.
    let asks = Asks {
        ring: Some((&earlier, false)),
        ..keyed(7, &members)
    };
    let decision = decided(&engine, &asks).decision;
    assert_eq!(
        Some(list[first(&decision)].url()),
        looked_up(&earlier, &list, 7)
    );

    let scored = preset("least_load");
    let members = scored.pool_members(&list);
    let asks = Asks {
        ring: Some((&own, true)),
        ..keyed(7, &members)
    };
    assert!(decided(&scored, &asks).decision.pick().is_some());
    assert!(
        members.ring_places().get().is_none(),
        "a list no ring stage decides makes no table"
    );
}

/// Under the stock sessions rule a key's first request is placed by the
/// whole pass, and its dispatch makes the key's pin in the engine's session
/// table. The key's next request follows that pin by one worker's record,
/// with no pass over the list. A pinned worker that is not ready gives the
/// key back to the whole pass, which places it anew. No table of a ring is
/// made for a document that reads none.
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
    assert!(
        members.ring_places().get().is_none(),
        "no table of a ring is made"
    );
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
