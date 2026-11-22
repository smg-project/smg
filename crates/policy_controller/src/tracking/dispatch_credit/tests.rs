use std::{sync::Arc, time::Instant};

use parking_lot::Mutex;
use serde_json::json;

use crate::{
    document::{AppliedSpec, PolicySpec, SpecError, API_VERSION},
    engine::{
        decision, decision_engine::Request, framework, store_backed_engine::StoreBackedEngine,
        Candidate, Constraints, Engine, Filter, LoadRow, Pair, PoolFacts, RequestFacts, Scorer,
        Scratch, Target, TieBand, WorkerId,
    },
    stages::{pickers, Lifecycle, StageRegistry},
    store::{
        Dispatch, GatewayId, LeaseState, LeasesStore, MemoryStore, Said, StaleGeneration,
        WorkerFacts, WorkerStatus, WorkersStore,
    },
    tracking::{
        decision_guard::DecisionGuard, CreditedDispatches, DispatchCounts, DispatchCredit, Lease,
        Membership, Side,
    },
};

const POOL: &str = "work";
const GATEWAY: GatewayId = GatewayId(37);
const START: u64 = 91_300;
const DELAY: u32 = 47;
const TOKENS: u32 = 317;
const NEXT_GENERATION: u64 = 43;

fn one(tokens: u32) -> CreditedDispatches {
    CreditedDispatches {
        requests: 1,
        tokens: u64::from(tokens),
        ..CreditedDispatches::default()
    }
}

#[derive(Debug)]
struct AvoidSent(Arc<DispatchCredit>);

impl Lifecycle for AvoidSent {}

impl Scorer for AvoidSent {
    fn name(&self) -> &'static str {
        "avoid-sent"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let sent = self.0.pass(facts);
        for (&row, score) in kept.iter().zip(out) {
            *score = if sent.of(&candidates[row as usize]).requests == 0 {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }
}

type Readers = Arc<Mutex<Vec<Arc<DispatchCredit>>>>;

fn catalog() -> (StageRegistry, Readers) {
    let readers = Readers::default();
    let kept = Arc::clone(&readers);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("avoid-sent", move |build| {
            let sent = build.dispatch_credit();
            kept.lock().push(Arc::clone(&sent));
            Ok(Box::new(AvoidSent(sent)))
        })
        .expect("a free name");
    (catalog, readers)
}

fn document(limit: Option<u32>) -> PolicySpec {
    serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": {
            "work": { "sent_ttl_ms": limit, "score": ["avoid-sent"], "pick": "best" }
        }
    }))
    .expect("a document")
}

fn built(limit: Option<u32>) -> (Engine, Arc<DispatchCredit>) {
    let (catalog, readers) = catalog();
    let engine = document(limit)
        .validate_in(&catalog, &[])
        .expect("the profile validates");
    let sent = Arc::clone(&readers.lock()[0]);
    (engine, sent)
}

fn row(id: u64, dp_rank: u16) -> Candidate {
    Candidate {
        id: WorkerId(id),
        dp_rank,
        ..Candidate::default()
    }
}

fn target(row: &Candidate) -> Target {
    Target {
        id: row.id,
        dp_rank: row.dp_rank,
        ..Target::default()
    }
}

fn send(engine: &Engine, row: &Candidate, issued_micros: u64, tokens: u32) {
    engine.on_dispatched(
        &Lease {
            issued_micros,
            tokens,
            ..Lease::default()
        },
        &target(row),
        Side::Single,
        DELAY,
    );
}

fn clock(now_micros: u64) -> RequestFacts<'static> {
    RequestFacts {
        now_micros,
        ..RequestFacts::default()
    }
}

#[test]
fn a_registered_scorer_reads_the_requests_and_tokens_sent_to_each_row() {
    let (engine, sent) = built(None);
    let a = row(101, 3);
    let b = row(103, a.dp_rank);
    send(&engine, &a, START, TOKENS);

    let facts = clock(START + 700);
    let reads = sent.pass(&facts);
    assert_eq!(reads.of(&a), one(TOKENS));
    assert_eq!(reads.of(&b), CreditedDispatches::default());
    assert_eq!(sent.read(&facts, &a), one(TOKENS));
}

#[test]
fn a_pass_keeps_one_tally_map_and_a_later_pass_reads_new_rows() {
    let (engine, sent) = built(None);
    let a = row(107, 3);
    let facts = clock(START + 700);
    let pass = sent.pass(&facts);
    send(&engine, &a, START, TOKENS);
    assert_eq!(pass.of(&a), CreditedDispatches::default());
    assert_eq!(sent.pass(&facts).of(&a), one(TOKENS));
}

#[test]
fn a_send_with_no_stated_size_counts_only_as_a_request() {
    let (engine, sent) = built(None);
    let a = row(109, 5);
    send(&engine, &a, START, 0);
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            unsized_requests: 1,
            ..one(0)
        }
    );
}

#[test]
fn a_send_of_no_stated_size_leaves_the_count_with_its_release() {
    let (engine, sent) = built(None);
    let mut a = row(179, 5);
    send(&engine, &a, START, 0);
    send(&engine, &a, START + 9, 0);
    send(&engine, &a, START + 20, TOKENS);
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 3,
            tokens: u64::from(TOKENS),
            unsized_requests: 2,
        }
    );
    // A report that reflects the first two sends, each of no stated size.
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + 9 + u64::from(DELAY);
    assert_eq!(sent.read(&clock(0), &a), one(TOKENS));
}

#[test]
fn a_retired_batch_takes_its_sends_of_no_stated_size_with_it() {
    let (engine, sent) = built(None);
    let a = row(181, 5);
    for _ in 0..300 {
        send(&engine, &a, START, 0);
    }
    for _ in 300..8201 {
        send(&engine, &a, START + 1, 13);
    }
    // The 256 oldest sends went, each of no stated size.
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 7945,
            tokens: 7901 * 13,
            unsized_requests: 44,
        }
    );
}

#[test]
fn a_busy_rows_tally_retires_its_oldest_batch() {
    let (engine, sent) = built(None);
    let a = row(111, 3);
    for _ in 0..8037 {
        send(&engine, &a, START, 13);
    }
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 8037,
            tokens: 8037 * 13,
            ..CreditedDispatches::default()
        }
    );
    for _ in 8037..8201 {
        send(&engine, &a, START + 1, 19);
    }
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 7945,
            tokens: 7781 * 13 + 164 * 19,
            ..CreditedDispatches::default()
        }
    );
}

#[test]
fn the_row_report_ends_sends_at_its_sample_and_earlier() {
    let (engine, sent) = built(None);
    let mut a = row(113, 3);
    let at = START + u64::from(DELAY);
    send(&engine, &a, START, TOKENS);
    a.load_sampled_at_micros = at;
    let facts = clock(at + 400);
    assert_eq!(sent.read(&facts, &a), one(TOKENS));

    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = at - 1;
    assert_eq!(sent.read(&facts, &a), one(TOKENS));
    send(&engine, &a, START + 7, 293);
    a.load_sampled_at_micros = at;
    assert_eq!(sent.read(&facts, &a), one(293));
    a.load_sampled_at_micros = at + 8;
    assert_eq!(sent.read(&facts, &a), CreditedDispatches::default());
}

#[test]
fn the_limit_ends_sends_at_its_cutoff_and_earlier() {
    let (engine, sent) = built(Some(17));
    let mut a = row(127, 3);
    let at = START + u64::from(DELAY);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = at - 1;
    send(&engine, &a, START - 2, 281);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 4, 331);

    assert_eq!(sent.read(&clock(at + 17_000), &a), one(331));
    a.load_sampled_at_micros = at + 4;
    assert_eq!(
        sent.read(&clock(at + 17_001), &a),
        CreditedDispatches::default()
    );
}

#[test]
fn the_two_number_read_says_the_requests_and_stated_tokens_of_the_whole_read() {
    let (engine, sent) = built(None);
    let a = row(331, 3);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 1, 0);
    send(&engine, &a, START + 2, 293);

    let facts = clock(START + 700);
    let reads = sent.pass(&facts);
    let whole = reads.of(&a);
    assert_eq!(whole.unsized_requests, 1);
    let counts = DispatchCounts {
        requests: 3,
        tokens: u64::from(TOKENS) + 293,
    };
    assert_eq!(reads.counts(&a), counts);
    assert_eq!(
        (whole.requests, whole.tokens),
        (counts.requests, counts.tokens)
    );
    assert_eq!(
        reads.counts(&row(337, a.dp_rank)),
        DispatchCounts::default()
    );
}

#[test]
fn the_two_number_read_releases_the_sends_a_report_reflects() {
    let (engine, sent) = built(None);
    let mut a = row(347, 3);
    let at = START + u64::from(DELAY);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 7, 293);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = at;

    // No other read ran first: this one releases the send the report
    // reflects, and the whole read finds it gone.
    let facts = clock(at + 400);
    let later = DispatchCounts {
        requests: 1,
        tokens: 293,
    };
    assert_eq!(sent.pass(&facts).counts(&a), later);
    assert_eq!(sent.read(&facts, &a), one(293));
}

#[test]
fn a_release_leaves_a_later_send_of_no_stated_size_in_the_read_that_ran_it() {
    let (engine, sent) = built(None);
    let mut a = row(353, 5);
    send(&engine, &a, START, 0);
    send(&engine, &a, START + 9, 0);
    // A report that reflects the first send: the read that releases it
    // still counts the second, and counts it as of no stated size.
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY);
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 1,
            tokens: 0,
            unsized_requests: 1,
        }
    );
}

#[test]
fn the_two_number_read_counts_no_send_past_the_time_limit() {
    let (engine, sent) = built(Some(17));
    let a = row(349, 3);
    let at = START + u64::from(DELAY);
    send(&engine, &a, START, TOKENS);

    let held = DispatchCounts {
        requests: 1,
        tokens: u64::from(TOKENS),
    };
    assert_eq!(sent.pass(&clock(at + 16_999)).counts(&a), held);
    assert_eq!(
        sent.pass(&clock(at + 17_000)).counts(&a),
        DispatchCounts::default()
    );
}

#[test]
fn an_unset_limit_uses_the_report_alone() {
    let (engine, sent) = built(None);
    let mut a = row(131, 3);
    let facts = clock(START + 7_300_000);
    send(&engine, &a, START, TOKENS);
    assert_eq!(sent.read(&facts, &a), one(TOKENS));
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY);
    assert_eq!(sent.read(&facts, &a), CreditedDispatches::default());
}

#[test]
fn a_clock_at_zero_or_before_the_limit_does_not_age_sends() {
    let (engine, sent) = built(Some(19));
    let a = row(137, 3);
    send(&engine, &a, 71, TOKENS);
    assert_eq!(sent.read(&clock(0), &a), one(TOKENS));
    assert_eq!(sent.read(&clock(271), &a), one(TOKENS));
}

#[test]
fn removing_a_worker_clears_its_ranks_sends() {
    let (engine, sent) = built(None);
    let a = row(139, 3);
    let other_rank = row(a.id.0, 7);
    let b = row(149, 3);
    for candidate in [&a, &other_rank, &b] {
        send(&engine, candidate, START, TOKENS);
    }
    engine.on_membership(&Membership::Removed(a.id));
    let facts = clock(0);
    let reads = sent.pass(&facts);
    assert_eq!(reads.of(&a), CreditedDispatches::default());
    assert_eq!(reads.of(&other_rank), CreditedDispatches::default());
    assert_eq!(reads.of(&b), one(TOKENS));
}

fn validated(document: &PolicySpec, catalog: &StageRegistry) -> AppliedSpec {
    let digest = document.digest().expect("a digest");
    AppliedSpec::validate_digested(document, digest, catalog, &[]).expect("the policy validates")
}

fn apply(store: &MemoryStore, document: &PolicySpec, catalog: &StageRegistry) {
    store
        .apply(validated(document, catalog))
        .expect("a newer generation");
}

fn next_document(limit: Option<u32>) -> PolicySpec {
    let mut next = document(limit);
    next.generation = NEXT_GENERATION;
    next
}

fn door() -> (StoreBackedEngine, StageRegistry, Readers) {
    let (catalog, readers) = catalog();
    let store = Arc::new(MemoryStore::new());
    for name in ["http://first:8000", "http://second:8000"] {
        let id = store
            .upsert(WorkerFacts {
                key: name.to_owned(),
                engine: name.to_owned(),
                incarnation: 1,
                models: vec![POOL.to_owned()],
                dp_size: 1,
                ..WorkerFacts::default()
            })
            .id();
        let revision = WorkersStore::get(&*store, id).expect("the row").revision;
        store.set_status(id, WorkerStatus::Ready, revision);
    }
    store.publish_now();
    apply(&store, &document(None), &catalog);
    (StoreBackedEngine::new(store), catalog, readers)
}

fn decide<'a>(
    door: &'a StoreBackedEngine,
    id: &[u8],
    now: Instant,
) -> DecisionGuard<&'a StoreBackedEngine> {
    DecisionGuard::decide(
        door,
        GATEWAY,
        &Request {
            id,
            facts: RequestFacts {
                model: POOL,
                prompt_tokens: TOKENS,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        },
        now,
    )
    .expect("a decision")
}

fn chosen(decision: &DecisionGuard<&StoreBackedEngine>) -> Candidate {
    let target = decision
        .decision()
        .pick()
        .and_then(|pick| pick.targets.first())
        .expect("a pick");
    row(target.id.0, target.dp_rank)
}

#[test]
fn a_handle_dispatch_changes_the_next_in_process_pick() {
    let (door, _, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"first", now);
    let sent_to = chosen(&first);
    let sent = Arc::clone(&readers.lock()[0]);
    assert_eq!(
        sent.read(&clock(0), &sent_to),
        CreditedDispatches::default()
    );
    first.dispatched(now);
    let second = decide(&door, b"second", now);
    assert_ne!(chosen(&second).id, sent_to.id);
    assert_eq!(sent.read(&clock(0), &sent_to), one(TOKENS));
}

#[test]
fn a_dispatch_of_an_older_lease_reaches_the_standing_generation() {
    let (door, catalog, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"first", now);
    let sent_to = chosen(&first);
    let mut next = document(None);
    next.generation = 7;
    apply(door.store(), &next, &catalog);

    first.dispatched(now);
    let readers = readers.lock();
    assert_eq!(readers[1].read(&clock(0), &sent_to), one(TOKENS));
    assert_eq!(
        readers[0].read(&clock(0), &sent_to),
        CreditedDispatches::default()
    );
}

#[test]
fn a_decision_reads_the_sends_carried_to_its_profile() {
    let (door, catalog, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"choose", now);
    let sent_to = chosen(&first);
    first.dispatched(now);

    apply(door.store(), &next_document(None), &catalog);
    let next = decide(&door, b"choose", now);
    assert_eq!(readers.lock()[1].read(&clock(0), &sent_to), one(TOKENS));
    assert_ne!(chosen(&next).id, sent_to.id);
}

#[test]
fn a_profile_under_a_new_name_starts_with_no_sends() {
    let (door, catalog, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"before", now);
    let sent_to = chosen(&first);
    first.dispatched(now);

    let mut next = next_document(None);
    let profile = next.profiles.remove(POOL).expect("the standing profile");
    next.profiles.insert("new".into(), profile);
    apply(door.store(), &next, &catalog);
    let readers = readers.lock();
    assert_eq!(
        readers[1].read(&clock(0), &sent_to),
        CreditedDispatches::default()
    );
    assert_eq!(readers[0].read(&clock(0), &sent_to), one(TOKENS));
}

#[test]
fn the_new_profiles_limit_applies_to_carried_sends() {
    let (door, catalog, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"before", now);
    let sent_to = chosen(&first);
    let issued = first.lease().expect("the pick's lease").lease.issued_micros;
    first.dispatched(now);

    let limit = 29;
    apply(door.store(), &next_document(Some(limit)), &catalog);
    let edge = issued + u64::from(limit) * 1000;
    let readers = readers.lock();
    assert_eq!(readers[1].read(&clock(edge - 1), &sent_to), one(TOKENS));
    assert_eq!(readers[0].read(&clock(edge), &sent_to), one(TOKENS));
    assert_eq!(
        readers[1].read(&clock(edge), &sent_to),
        CreditedDispatches::default()
    );
}

#[test]
fn a_refused_generation_takes_no_sends_from_the_standing_profile() {
    let (door, catalog, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"before", now);
    let sent_to = chosen(&first);
    first.dispatched(now);

    let stale = document(None);
    assert_eq!(
        door.store().apply(validated(&stale, &catalog)),
        Err(StaleGeneration {
            standing: stale.generation,
            offered: stale.generation,
        })
    );
    let readers = readers.lock();
    assert_eq!(
        readers[1].read(&clock(0), &sent_to),
        CreditedDispatches::default()
    );
    assert_eq!(readers[0].read(&clock(0), &sent_to), one(TOKENS));
}

#[test]
fn a_decision_holding_the_standing_world_reads_the_same_carried_entry() {
    let (door, catalog, _) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"choose", now);
    let mut second = decide(&door, b"choose", now);
    let sent_to = chosen(&first);
    assert_eq!(chosen(&second).id, sent_to.id);
    first.dispatched(now);

    let store = door.store();
    store.deciding(|world| {
        let standing = &world.spec().engine().sent_credits[0];
        apply(store, &next_document(None), &catalog);
        second.dispatched(now);

        let both = CreditedDispatches {
            requests: 2,
            tokens: 2 * u64::from(TOKENS),
            ..CreditedDispatches::default()
        };
        assert_eq!(standing.read(&clock(0), &sent_to), both);
        store.deciding(|next| {
            let carried = &next.spec().engine().sent_credits[0];
            assert_eq!(carried.read(&clock(0), &sent_to), both);
        });
    });
}

#[test]
fn a_late_dispatch_adds_no_entry_for_a_row_that_left() {
    for resize in [false, true] {
        let (door, _, readers) = door();
        let now = Instant::now();
        let mut first = decide(&door, b"before", now);
        let sent_to = chosen(&first);
        let lease = first.lease().expect("the pick's lease").lease;
        let store = door.store();
        let worker = WorkersStore::get(&**store, sent_to.id).expect("the worker");
        if resize {
            let facts = WorkerFacts {
                dp_rank: 6,
                dp_size: 9,
                ..worker.facts
            };
            assert_eq!(store.upsert(facts).id(), sent_to.id);
            store.publish_now();
        } else {
            store.remove(sent_to.id, worker.revision).expect("removal");
        }
        first.dispatched(now);

        let record = LeasesStore::get(&**store, GATEWAY, lease.id).expect("the filed lease");
        assert!(matches!(record.state, LeaseState::Dispatched { .. }));
        let tally = readers.lock()[0].credits.tally.load();
        assert!(!tally.contains_key(&(sent_to.id, sent_to.dp_rank)));
    }
}

#[test]
fn a_generation_with_no_readers_leaves_sends_behind() {
    let (door, catalog, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"before", now);
    let sent_to = chosen(&first);
    first.dispatched(now);

    let mut next: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": NEXT_GENERATION,
        "profiles": { (POOL): { "pick": "random" } }
    }))
    .expect("a document");
    apply(door.store(), &next, &catalog);
    door.store().deciding(|world| {
        let engine = world.spec().engine();
        assert!(engine.sent_credits.is_empty());
        engine.dispatched_sent(
            &Lease::default(),
            &Target::default(),
            Side::Single,
            0,
            || {
                panic!("a dispatch with no readers checked membership");
            },
        );
    });

    next = next_document(None);
    next.generation += 5;
    apply(door.store(), &next, &catalog);
    let readers = readers.lock();
    assert_eq!(readers.len(), 2);
    assert_eq!(readers[0].read(&clock(0), &sent_to), one(TOKENS));
    assert_eq!(
        readers[1].read(&clock(0), &sent_to),
        CreditedDispatches::default()
    );
}

#[test]
fn a_second_dispatch_report_of_the_same_side_counts_nothing() {
    let (door, _, readers) = door();
    let now = Instant::now();
    let mut first = decide(&door, b"first", now);
    let sent_to = chosen(&first);
    let lease = first.lease().expect("the pick's lease").lease;
    first.dispatched(now);
    door.report(
        GATEWAY,
        lease.id,
        Said::Dispatched(
            Dispatch {
                target_index: lease.target_index,
                side: lease.side,
                since_decision_micros: DELAY,
            },
            now,
        ),
    );
    assert_eq!(readers.lock()[0].read(&clock(0), &sent_to), one(TOKENS));
}

#[test]
fn each_side_of_a_pair_credits_its_own_rank() {
    let (engine, sent) = built(None);
    let prefill = row(151, 3);
    let decode = row(157, 7);
    let target = Target {
        pair: Some(Pair {
            decode_id: decode.id,
            decode_dp_rank: decode.dp_rank,
            ..Pair::default()
        }),
        ..target(&prefill)
    };
    let mut lease = Lease {
        issued_micros: START,
        tokens: TOKENS,
        ..Lease::default()
    };
    let facts = clock(0);
    engine.on_dispatched(&lease, &target, Side::Decode, DELAY);
    assert_eq!(sent.read(&facts, &decode), one(TOKENS));
    assert_eq!(sent.read(&facts, &prefill), CreditedDispatches::default());
    lease.tokens = 349;
    engine.on_dispatched(&lease, &target, Side::Prefill, DELAY);
    assert_eq!(sent.read(&facts, &prefill), one(349));
    assert_eq!(sent.read(&facts, &decode), one(TOKENS));
    assert_eq!(
        sent.read(&facts, &row(decode.id.0, 3)),
        CreditedDispatches::default()
    );
}

#[test]
fn a_profiles_limit_does_not_end_another_profiles_sends() {
    let (catalog, readers) = catalog();
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "profiles": {
            "brief": { "sent_ttl_ms": 23, "score": ["avoid-sent"], "pick": "best" },
            "long": { "sent_ttl_ms": 3719, "score": ["avoid-sent"], "pick": "best" }
        },
        "bindings": [{ "match": { "model": "*" }, "profiles": { "single": "brief" } }]
    }))
    .expect("a document");
    let engine = document.validate_in(&catalog, &[]).expect("validates");
    let a = row(163, 3);
    send(&engine, &a, START, TOKENS);
    let facts = clock(START + 29_000);
    let readers = readers.lock();
    assert_eq!(readers[0].read(&facts, &a), CreditedDispatches::default());
    assert_eq!(readers[1].read(&facts, &a), one(TOKENS));
}

#[test]
fn two_scorers_in_one_profile_share_one_tally() {
    let (catalog, readers) = catalog();
    let mut document = document(None);
    document
        .profiles
        .get_mut("work")
        .expect("the profile")
        .score
        .push("avoid-sent".into());
    let engine = document.validate_in(&catalog, &[]).expect("validates");
    let readers = readers.lock();
    assert!(Arc::ptr_eq(&readers[0], &readers[1]));
    let a = row(167, 3);
    send(&engine, &a, START, TOKENS);
    assert_eq!(readers[1].read(&clock(0), &a), one(TOKENS));
}

/// A profile of `least-load` under `name`, with the registered reader
/// beside it where `reader` says so.
fn load_document(name: &str, limit: Option<u32>, reader: bool) -> PolicySpec {
    let score = if reader {
        json!(["least-load", "avoid-sent"])
    } else {
        json!(["least-load"])
    };
    serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": { (name): { "sent_ttl_ms": limit, "score": score, "pick": "best" } }
    }))
    .expect("a document")
}

fn load_engine(name: &str, limit: Option<u32>) -> Engine {
    load_document(name, limit, false)
        .validate()
        .expect("the profile validates")
}

/// Two rows with equal reports, sampled ahead of every send of a case.
fn reporting() -> [Candidate; 2] {
    [(0, 191), (1, 193)].map(|(position, id)| Candidate {
        position,
        load: Some(LoadRow::default()),
        load_sampled_at_micros: START - 1,
        ..row(id, 3)
    })
}

/// The scores of `rows` in a decision at `now_micros`, in the rows' order.
fn scores(engine: &Engine, rows: &[Candidate; 2], now_micros: u64) -> [f32; 2] {
    let pool = PoolFacts {
        pool_size: 2,
        ..PoolFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &clock(now_micros),
        &pool,
        rows,
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("the rows route");
    let top = &pick.trace.top[..usize::from(pick.trace.top_len)];
    rows.each_ref().map(|row| {
        let scored = top.iter().find(|(id, _)| *id == row.id);
        scored.expect("each row is scored").1
    })
}

#[test]
fn least_load_reads_its_profiles_count_and_keeps_none_of_its_own() {
    let engine = load_engine(POOL, None);
    assert_eq!(engine.sent_credits.len(), 1);
    let rows = reporting();
    let lease = Lease {
        issued_micros: START,
        tokens: TOKENS,
        ..Lease::default()
    };
    // The counts alone hear this send: a stage with a tally of its own
    // would score the two rows alike.
    engine.dispatched_sent(&lease, &target(&rows[0]), Side::Single, DELAY, || true);
    let [sent_to, other] = scores(&engine, &rows, START + 700);
    assert!(sent_to < other, "{sent_to} against {other}");
}

#[test]
fn least_load_and_a_reader_share_one_count_and_a_send_is_counted_once() {
    let (catalog, readers) = catalog();
    let engine = load_document(POOL, None, true)
        .validate_in(&catalog, &[])
        .expect("the profile validates");
    assert_eq!(engine.sent_credits.len(), 1);
    assert!(Arc::ptr_eq(&engine.sent_credits[0], &readers.lock()[0]));
    let rows = reporting();
    send(&engine, &rows[0], START, TOKENS);
    assert_eq!(
        engine.sent_credits[0].read(&clock(0), &rows[0]),
        one(TOKENS)
    );
}

#[test]
fn least_load_prices_a_send_of_no_stated_size_at_its_mean() {
    let mut document = load_document(POOL, None, false);
    document.profiles.get_mut(POOL).expect("the profile").score =
        vec![
            serde_json::from_value(json!({ "least-load": { "mean_prefill_tokens": 701 } }))
                .expect("a stage"),
        ];
    let engine = document.validate().expect("the profile validates");
    let rows = reporting();
    send(&engine, &rows[0], START, 0);
    send(&engine, &rows[1], START, 701);
    let sent = &engine.sent_credits[0];
    assert_eq!(
        sent.read(&clock(0), &rows[0]),
        CreditedDispatches {
            unsized_requests: 1,
            ..one(0)
        }
    );
    let [unsized_send, stated] = scores(&engine, &rows, START + 700);
    assert_eq!(unsized_send, stated);
    send(&engine, &rows[1], START, 1);
    let [unsized_send, stated] = scores(&engine, &rows, START + 700);
    assert!(stated < unsized_send, "{stated} against {unsized_send}");
}

#[test]
fn the_profiles_limit_ends_least_loads_credit() {
    let limited = load_engine(POOL, Some(17));
    let unlimited = load_engine(POOL, None);
    let rows = reporting();
    for engine in [&limited, &unlimited] {
        send(engine, &rows[0], START, TOKENS);
    }
    let ends = START + u64::from(DELAY) + 17_000;
    let [sent_to, other] = scores(&limited, &rows, ends - 1);
    assert!(sent_to < other, "{sent_to} against {other}");
    let [sent_to, other] = scores(&limited, &rows, ends);
    assert_eq!(sent_to, other);
    // With no limit the credit stands until a report reflects the send.
    let [sent_to, other] = scores(&unlimited, &rows, ends + 7_300_000);
    assert!(sent_to < other, "{sent_to} against {other}");
    let reflected = rows.map(|row| Candidate {
        load_sampled_at_micros: START + u64::from(DELAY),
        ..row
    });
    let [sent_to, other] = scores(&unlimited, &reflected, ends + 7_300_000);
    assert_eq!(sent_to, other);
}

#[test]
fn a_new_documents_least_load_keeps_the_sends_of_the_standing_one() {
    let standing = load_engine(POOL, None);
    let rows = reporting();
    send(&standing, &rows[0], START, TOKENS);
    let next = load_engine(POOL, None);
    next.succeed(&standing);
    let [sent_to, other] = scores(&next, &rows, START + 700);
    assert!(sent_to < other, "{sent_to} against {other}");
}

#[test]
fn least_load_under_a_new_profile_name_starts_with_no_sends() {
    let standing = load_engine(POOL, None);
    let rows = reporting();
    send(&standing, &rows[0], START, TOKENS);
    let next = load_engine("renamed", None);
    next.succeed(&standing);
    let [sent_to, other] = scores(&next, &rows, START + 700);
    assert_eq!(sent_to, other);
}

#[test]
fn presence_is_asked_only_where_a_send_enters_a_rank() {
    let (engine, sent) = built(None);
    let a = row(197, 3);
    let lease = Lease {
        issued_micros: START,
        tokens: TOKENS,
        ..Lease::default()
    };
    // A rank the count does not know is asked for, and a row that left
    // enters none.
    engine.dispatched_sent(&lease, &target(&a), Side::Single, DELAY, || false);
    assert!(!sent.credits.tally.load().contains_key(&(a.id, a.dp_rank)));
    engine.dispatched_sent(&lease, &target(&a), Side::Single, DELAY, || true);
    // A rank it knows takes the send unasked.
    engine.dispatched_sent(&lease, &target(&a), Side::Single, DELAY, || {
        panic!("a known rank was asked for")
    });
    assert_eq!(sent.read(&clock(0), &a).requests, 2);
}

#[test]
fn a_limit_is_taken_where_least_load_reads_the_sends() {
    assert_eq!(load_engine(POOL, Some(71)).sent_credits.len(), 1);
}

#[test]
fn a_zero_limit_is_refused_at_the_profiles_field() {
    let (catalog, _) = catalog();
    assert_eq!(
        document(Some(0)).validate_in(&catalog, &[]).err(),
        Some(SpecError::BadField {
            path: "profiles.work.sent_ttl_ms".into(),
            value: "0".into(),
            reason: "above zero",
        })
    );
}

#[test]
fn a_limit_without_a_reader_is_refused_at_the_profiles_field() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document
        .profiles
        .get_mut("random")
        .expect("the profile")
        .sent_ttl_ms = Some(71);
    assert_eq!(
        document.validate().err(),
        Some(SpecError::BadField {
            path: "profiles.random.sent_ttl_ms".into(),
            value: "71".into(),
            reason: "a limit any stage of this profile reads",
        })
    );
}

#[test]
fn a_document_with_no_reader_keeps_no_send_tallies() {
    let engine = PolicySpec::preset("random")
        .expect("a preset")
        .validate()
        .expect("a valid document");
    assert!(engine.sent_credits.is_empty());
    send(&engine, &row(173, 3), START, TOKENS);
    assert!(engine.sent_credits.is_empty());
}

#[test]
fn an_unset_limit_leaves_a_documents_canonical_form_and_digest_intact() {
    let canonical = format!(
        r#"{{"apiVersion":"{API_VERSION}","generation":0,"profiles":{{"work":{{"filter":["healthy"],"score":["least-requests"],"pick":"best"}}}},"bindings":[]}}"#
    );
    let document: PolicySpec = serde_json::from_str(&canonical).expect("an existing document");
    assert_eq!(
        serde_json::to_string(&document).expect("the form"),
        canonical
    );
    assert_eq!(
        document.digest().expect("a digest").to_string(),
        "sha256:5d615ed114ca79046283cd72596dd3428baa2ec110170a5bb72ea4bc525b1ade"
    );
}

/// Keeps the rows its profile sent fewer than `most` requests.
#[derive(Debug)]
struct SentUnder {
    sent: Arc<DispatchCredit>,
    most: u32,
}

impl Filter for SentUnder {
    fn name(&self) -> &'static str {
        "sent-under"
    }

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        self.sent.read(facts, candidate).requests < self.most
    }
}

/// A registered filter holds its profile's count of sends, as a registered
/// scorer does. The count is made where the filter's factory asks for it,
/// ahead of any scorer, so the profile's limit has its reader.
#[test]
fn a_registered_filter_reads_the_sends_of_its_profile() {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Most {
        requests: u32,
    }

    let readers = Readers::default();
    let kept = Arc::clone(&readers);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("sent-under", move |build| {
            let Most { requests } = build.params()?;
            let sent = build.dispatch_credit();
            kept.lock().push(Arc::clone(&sent));
            Ok(Box::new(SentUnder {
                sent,
                most: requests,
            }))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": {
            "work": {
                "sent_ttl_ms": 24_001,
                "filter": [{ "sent-under": { "requests": 2 } }],
                "pick": "random"
            }
        }
    }))
    .expect("a document");
    let engine = document
        .validate_in(&catalog, &[])
        .expect("the filter reads the sends the limit ages");
    assert_eq!(engine.sent_credits.len(), 1);

    let a = row(127, 3);
    let b = row(131, a.dp_rank);
    let facts = clock(START + 700);
    let filter = &engine.profiles()[0].filters[0];
    let keeps = |row: &Candidate| filter.keep(&facts, row, &Constraints::default());
    send(&engine, &a, START, TOKENS);
    assert!(keeps(&a) && keeps(&b), "one send is under two");
    send(&engine, &a, START, TOKENS);
    assert!(!keeps(&a), "two sends are not under two");
    assert!(keeps(&b), "another row's sends are its own");
    // The count the factory was handed is the one the engine feeds.
    assert_eq!(
        readers.lock()[0].read(&facts, &a),
        CreditedDispatches {
            requests: 2,
            tokens: 2 * u64::from(TOKENS),
            ..CreditedDispatches::default()
        }
    );
}

/// Picks the kept row its profile sent the fewest requests, the first of
/// them at a tie.
#[derive(Debug)]
struct LeastSent {
    sent: Arc<DispatchCredit>,
}

impl Lifecycle for LeastSent {}

impl framework::Picker for LeastSent {
    fn name(&self) -> &'static str {
        "least-sent"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut decision::Targets,
    ) {
        let sent = |row: &&u32| self.sent.read(facts, &candidates[**row as usize]).requests;
        if let Some(&row) = kept.iter().min_by_key(sent) {
            out.push(pickers::target_of(&candidates[row as usize]));
        }
    }
}

/// A registered picker holds its profile's count of sends, as a registered
/// filter does. A profile whose only reader of sends is its picker keeps
/// its limit: the picker is built ahead of the limit's check. The count its
/// factory was handed is the profile's own, the one the engine feeds.
#[test]
fn a_registered_picker_reads_the_sends_of_its_profile() {
    let readers = Readers::default();
    let kept = Arc::clone(&readers);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker_with("least-sent", move |build| {
            let sent = build.dispatch_credit();
            kept.lock().push(Arc::clone(&sent));
            Ok(Box::new(LeastSent { sent }))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": {
            "work": {
                "sent_ttl_ms": 24_001,
                "filter": ["healthy"],
                "score": ["least-requests"],
                "pick": "least-sent"
            }
        }
    }))
    .expect("a document");
    let engine = document
        .validate_in(&catalog, &[])
        .expect("the picker reads the sends the limit ages");
    assert_eq!(engine.sent_credits.len(), 1);

    let a = row(127, 3);
    let facts = clock(START + 700);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START, TOKENS);
    assert_eq!(
        readers.lock()[0].read(&facts, &a),
        CreditedDispatches {
            requests: 2,
            tokens: 2 * u64::from(TOKENS),
            ..CreditedDispatches::default()
        }
    );
}
