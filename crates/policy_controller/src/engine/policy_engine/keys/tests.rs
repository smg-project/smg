//! The cases of a keyed request's decision in a gateway's own process: each
//! asked through the engine's own entry, over a list of worker objects.

use std::{
    cell::OnceCell,
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::{
    identity::basic::Worker,
    test_support::{ready_worker, worker_with_model_aliases},
    worker::{WorkerStatus, WorkerType},
};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::Decision,
        decision_engine::{DecideOutcome, DecisionEngine, Request, RequestMaker, WorkerLists},
        list_cell::tests::PublishedList,
        policy_engine::PolicyEngine,
        records,
        sticky_sessions::{shards_for_this_host, SessionVerdict},
        view::{Constraints, Identity, Phase, RequestFacts, Session, WorkerId},
    },
    policies::consistent_hashing::{InOrder, LentRing},
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
    members: &'a PublishedList,
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members.lent())
    }
}

/// A request of the session key `key`, with no key where it is `0`, over
/// `members`.
fn keyed(key: u64, members: &PublishedList) -> Asks<'_> {
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

/// Under the stock ring a keyed request over a list of worker objects goes
/// to the workers its caller's ring reaches from where the caller says the
/// key stands, in the ring's order: each worker the ring names by address
/// is found in the list. A worker that is not ready passes the key on. No
/// pass over the list runs: the decision writes the records of the workers
/// its key reaches and of no other.
///
/// The gateway's own policy takes the first eligible worker its ring
/// reaches from the key (`model_gateway/src/policies/consistent_hashing.rs`,
/// lines 99 to 115 at 882be6f4f): where that ring is the one lent, the
/// first worker named here.
#[test]
fn a_rings_keyed_request_goes_to_the_workers_its_callers_ring_reaches() {
    let engine = preset("consistent_hashing");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [5, 2, 7, 0, 3, 6, 1, 4];
    let ring = InOrder(order.iter().map(|&at| list[at].url().to_owned()).collect());
    let places = |asks: &Asks<'_>| -> Vec<u32> {
        let decision = decided(&engine, asks).decision;
        let trace = &decision.pick().expect("a pick").trace;
        assert!(trace.drew_first(), "a pass over the list ran");
        targets(&decision).iter().map(|&(place, _)| place).collect()
    };
    let mut asks = keyed(9, &members);
    asks.request.facts.ring = Some(&ring);
    // Two workers along the ring's order.
    asks.request.facts.ring_positions = &[10];
    assert_eq!(places(&asks), [7, 0, 3, 6]);
    list[7].set_status(WorkerStatus::NotReady);
    list[3].set_status(WorkerStatus::NotReady);
    assert_eq!(places(&asks), [0, 6, 1, 4]);
}

/// A caller with no ring of the model's workers yet lends an empty one, and
/// the key goes to the ready worker at its position modulo their count, in
/// the list's order: what that caller's own policy answers with no ring
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 124 to 140 at
/// 882be6f4f).
#[test]
fn a_key_an_empty_lent_ring_places_goes_to_the_ready_worker_at_its_position() {
    let engine = preset("consistent_hashing");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let ring = InOrder(Vec::new());
    let mut asks = keyed(9, &members);
    asks.request.facts.ring = Some(&ring);
    asks.request.facts.ring_positions = &[13];
    let first = |asks: &Asks<'_>| targets(&decided(&engine, asks).decision)[0].0;
    assert_eq!(first(&asks), 5);
    // Seven are ready, and the seventh of them stands at place 7.
    list[2].set_status(WorkerStatus::NotReady);
    assert_eq!(first(&asks), 7);
}

/// A request as a caller that keeps a ring makes it: its keys read from
/// its headers by the document's sources, with where the key stands on the
/// ring the caller lends.
struct Derives<'a> {
    headers: &'a [(&'a str, &'a [u8])],
    ring: &'a InOrder,
    members: &'a PublishedList,
    /// Where the key stands, found while the request is made and kept
    /// here: the request's facts borrow it for as long as it stands.
    position: OnceCell<u64>,
}

impl RequestMaker for Derives<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        let sources = spec.key_sources_for(MODEL, Phase::Single);
        let header = |name: &str| {
            let named = self.headers.iter().find(|(listed, _)| *listed == name);
            named.map(|(_, value)| *value)
        };
        let keys = sources
            .expect("the ring's sources")
            .derive_keys(header, None, Some(self.ring));
        let position = keys
            .ring_position
            .map(|at| self.position.get_or_init(|| at));
        Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                session_key: keys.session.map_or(Identity::NONE, |key| key.key),
                client_key: keys.client_key,
                ring: Some(self.ring),
                ring_positions: position.map_or(&[], std::slice::from_ref),
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        }
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members.lent())
    }
}

/// A request's headers place it on the ring its caller lends: under the
/// stock ring its routing key, and with none its client's address, is read
/// by the document's sources with where its text stands on that ring, and
/// the request goes to the workers the ring reaches from there. A request
/// that carries neither states no place, and is placed all the same.
///
/// The gateway's own policy takes the two in that order, behind a worker
/// the request names, and draws for a request with neither
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 189 to 241 at
/// 882be6f4f).
#[test]
fn a_requests_headers_place_it_on_the_ring_its_caller_lends() {
    let engine = preset("consistent_hashing");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let ring = InOrder(order.iter().map(|&at| list[at].url().to_owned()).collect());
    // The four workers the ring reaches from where `text` stands on it.
    let reached = |text: &str| -> Vec<u32> {
        let from = (ring.position(text) % 8) as usize;
        let around = order.iter().cycle().skip(from).take(4);
        around.map(|&at| at as u32).collect()
    };
    let placed = |headers: &[(&str, &[u8])]| -> Vec<u32> {
        let asks = Derives {
            headers,
            ring: &ring,
            members: &members,
            position: OnceCell::new(),
        };
        let decided = engine.decide_under(GatewayId::LOCAL, &asks, Instant::now());
        let decision = decided
            .expect("an answer")
            .whole()
            .expect("whole")
            .0
            .decision;
        targets(&decision).iter().map(|&(place, _)| place).collect()
    };
    let keyed: [(&str, &[u8]); 2] = [
        ("x-forwarded-for", b"10.0.0.1"),
        ("x-smg-routing-key", b"conv-1"),
    ];
    assert_eq!(placed(&keyed), reached("conv-1"));
    assert_eq!(placed(&keyed[..1]), reached("10.0.0.1"));
    assert!(
        !placed(&[]).is_empty(),
        "no key: placed over the whole list"
    );
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
        let members = PublishedList::new(Arc::clone(&list));
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

/// A key keeps its worker over that worker's overload thresholds among
/// regular workers alone. Over a list of prefill workers the owner over its
/// thresholds passes the key to the next worker its ring reaches: on the
/// ring its caller lends, where the key's rows alone are read, and on the
/// stage's own ring, where the whole decision answers.
///
/// The gateway's own placement lifts the overload veto for a key where
/// every candidate is a regular worker (`select_pass` in
/// `model_gateway/src/routers/common/placement.rs`).
#[test]
fn a_key_over_prefill_workers_passes_an_owner_over_its_thresholds() {
    let engine = preset("consistent_hashing");
    let list: Arc<[Arc<dyn Worker>]> = (0..8)
        .map(|at| {
            let url = format!("http://prefill-{at}:8000");
            let worker = worker_with_model_aliases(&url, MODEL, &[], WorkerType::Prefill);
            worker.set_status(WorkerStatus::Ready);
            worker
        })
        .collect();
    let members = PublishedList::new(Arc::clone(&list));
    let ring = InOrder(list.iter().map(|worker| worker.url().to_owned()).collect());
    let places = |lent: bool| -> Vec<u32> {
        let mut asks = keyed(9, &members);
        if lent {
            asks.request.facts.ring = Some(&ring);
            asks.request.facts.ring_positions = &[0];
        }
        let decision = decided(&engine, &asks).decision;
        let trace = &decision.pick().expect("a pick").trace;
        assert!(!lent || trace.drew_first(), "a pass over the list ran");
        targets(&decision).iter().map(|&(place, _)| place).collect()
    };
    for lent in [true, false] {
        let free = places(lent);
        let owner = &list[free[0] as usize];
        owner.set_overloaded(true);
        assert_eq!(places(lent)[0], free[1], "ring lent: {lent}");
        owner.set_overloaded(false);
    }
}

/// A list of decode workers and a list of encode workers pass the key on as
/// a list of prefill workers does: every role but the regular one counts,
/// on the ring the caller lends and on the stage's own.
///
/// The gateway's own placement lifts the overload veto for a key only
/// where every candidate's type is the regular one (`select_pass`, at main
/// `450084124`, `model_gateway/src/routers/common/placement.rs:1337`).
#[test]
fn a_key_over_decode_or_encode_workers_passes_an_owner_over_its_thresholds() {
    for role in [WorkerType::Decode, WorkerType::Encode] {
        let engine = preset("consistent_hashing");
        let list: Arc<[Arc<dyn Worker>]> = (0..8)
            .map(|at| {
                let url = format!("http://other-{at}:8000");
                let worker = worker_with_model_aliases(&url, MODEL, &[], role);
                worker.set_status(WorkerStatus::Ready);
                worker
            })
            .collect();
        let members = PublishedList::new(Arc::clone(&list));
        let ring = InOrder(list.iter().map(|worker| worker.url().to_owned()).collect());
        let places = |lent: bool| -> Vec<u32> {
            let mut asks = keyed(9, &members);
            if lent {
                asks.request.facts.ring = Some(&ring);
                asks.request.facts.ring_positions = &[0];
            }
            let decision = decided(&engine, &asks).decision;
            let trace = &decision.pick().expect("a pick").trace;
            assert!(!lent || trace.drew_first(), "a pass over the list ran");
            targets(&decision).iter().map(|&(place, _)| place).collect()
        };
        for lent in [true, false] {
            let free = places(lent);
            let owner = &list[free[0] as usize];
            owner.set_overloaded(true);
            assert_eq!(places(lent)[0], free[1], "{role:?}, ring lent: {lent}");
            owner.set_overloaded(false);
        }
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
fn a_worker_that_left_its_list_moves_no_other_key() {
    for ring in ["consistent_hashing", "prefix_hash"] {
        let (seen, fresh) = (preset(ring), preset(ring));
        let named = |engine: &PolicyEngine, list: &Arc<[Arc<dyn Worker>]>, key: u64| {
            let members = PublishedList::new(Arc::clone(list));
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
    let members = PublishedList::new(Arc::clone(&list));
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
    let members = PublishedList::new(Arc::clone(&list));
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
    let members = PublishedList::new(Arc::clone(&list));
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

/// A document is handed the engine's table ahead of its engine's first
/// decision: the table that engine was made with is let go, and a pin it
/// held would go with it. A debug build refuses a document whose engine
/// already holds a key.
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "ahead of the hand-over of its session table")
)]
fn a_document_whose_engine_holds_a_key_is_not_handed_a_table() {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let made = spec.engine().sessions();
    made.dispatched(Session::from_raw(71), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(made.len(), 1);
    let engine = PolicyEngine::new(spec);
    // A build with no debug check lets the key go with the table.
    assert_eq!(engine.sessions.len(), 0);
}

/// The session table is the engine's own, one for its life: the document's
/// engine reads and writes it through a pointer, so a pin can outlive the
/// document it was made under. The table has the host's shards whatever
/// the document rules, where a document with no sessions rule makes itself
/// a table of one shard, and it takes the document's two times. The third
/// document's rule states an idle time of its own: the stock rule's four
/// hours are the absence time a new table has, and would hide a time that
/// was never handed.
#[test]
fn the_session_table_is_the_engines_own_under_every_document() {
    for (name, ruled, idle_s) in [
        ("manual", true, None),
        ("random", false, None),
        ("manual", true, Some(600)),
    ] {
        let document = || {
            let mut document = PolicySpec::preset(name).expect("a built-in preset");
            document.generation = 1;
            if let Some(idle_s) = idle_s {
                let rule = document.bindings[0].sessions.as_mut();
                rule.expect("the stock rule").ttl_s = idle_s;
            }
            AppliedSpec::validate(&document).expect("the document validates")
        };
        let alone = document();
        let made = alone.engine().sessions();
        assert_eq!(made.shards() == 1, !ruled, "{name}");
        assert_eq!(made.sweep_micros() != 0, ruled, "{name}");
        if let Some(idle_s) = idle_s {
            let stated = u64::from(idle_s) * 1_000_000;
            assert_eq!(made.absence_micros(), stated, "the rule's own idle time");
        }
        let engine = PolicyEngine::new(document());
        let own = &*engine.sessions;
        assert!(
            std::ptr::eq(engine.spec().engine().sessions(), own),
            "{name}: the document's engine reads the engine's table"
        );
        assert_eq!(own.shards(), shards_for_this_host(), "{name}");
        assert_eq!(
            (own.sweep_micros(), own.absence_micros()),
            (made.sweep_micros(), made.absence_micros()),
            "{name}: the document's two times"
        );
    }
}

/// The owner of a key leaves and a new worker takes its place in the
/// list, two thousand times. A key whose owner left goes to another worker
/// while the old list still stands. Behind the last change the keys reach
/// all four workers, and each goes where an engine that saw none of the
/// changes sends it.
#[test]
fn keyed_decisions_reach_all_four_live_workers_after_two_thousand_replacements() {
    let first = |engine: &PolicyEngine, members: &PublishedList, key: u64| {
        let mut asks = keyed(key, members);
        asks.request.facts.seed = 1;
        let named = targets(&decided(engine, &asks).decision);
        let &(place, id) = named.first().expect("a first target");
        (place as usize, id)
    };

    let engine = preset("consistent_hashing");
    let mut live = pool(4);
    let mut members = PublishedList::new(Arc::clone(&live));
    for key in 1..=2_000 {
        let (slot, departed) = first(&engine, &members, key);
        assert_eq!(
            departed,
            records::worker_id(live[slot].as_ref()),
            "key {key}"
        );
        engine.forget(live[slot].url());

        // This list still contains the worker that owned the key.
        assert_ne!(
            first(&engine, &members, key).1,
            departed,
            "key {key} still picks its departed owner"
        );

        let next = ready_worker(&format!("http://replacement-{key}:8000"), MODEL);
        engine.joined(next.url());
        Arc::make_mut(&mut live)[slot] = next;
        members = PublishedList::new(Arc::clone(&live));

        let (place, id) = first(&engine, &members, 77);
        assert_eq!(
            id,
            records::worker_id(live[place].as_ref()),
            "key 77 after replacement {key}"
        );
    }

    let fresh = preset("consistent_hashing");
    let final_members = PublishedList::new(Arc::clone(&live));
    let mut reached = [false; 4];
    for key in 1..=500 {
        let (place, id) = first(&engine, &members, key);
        assert_eq!(id, records::worker_id(live[place].as_ref()), "key {key}");
        assert_eq!((place, id), first(&fresh, &final_members, key), "key {key}");
        reached[place] = true;
    }
    assert_eq!(reached, [true; 4], "each live worker owns a key");
}
