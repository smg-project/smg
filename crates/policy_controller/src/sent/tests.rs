use std::{sync::Arc, time::Instant};

use parking_lot::Mutex;
use serde_json::json;

use crate::{
    door::{Door, Request},
    handle::Handle,
    store::{
        Dispatch, GatewayId, LeaseState, LeasesStore, MemoryStore, Said, StaleGeneration,
        WorkerFacts, WorkerStatus, WorkersStore,
    },
    AppliedSpec, Candidate, Catalog, Constraints, Engine, Lease, Lifecycle, LoadRow, Membership,
    Pair, PolicySpec, RequestFacts, Scorer, Sent, SentCredit, Side, SpecError, Target, TieBand,
    WorkerId, API_VERSION,
};

const POOL: &str = "work";
const GATEWAY: GatewayId = GatewayId(37);
const START: u64 = 91_300;
const DELAY: u32 = 47;
const TOKENS: u32 = 317;
const NEXT_GENERATION: u64 = 43;

fn one(tokens: u32) -> Sent {
    Sent {
        requests: 1,
        tokens: u64::from(tokens),
    }
}

#[derive(Debug)]
struct AvoidSent(Arc<SentCredit>);

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
        let sent = self.0.reads(facts);
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

type Readers = Arc<Mutex<Vec<Arc<SentCredit>>>>;

fn catalog() -> (Catalog, Readers) {
    let readers = Readers::default();
    let kept = Arc::clone(&readers);
    let mut catalog = Catalog::builtin();
    catalog
        .scorer_with("avoid-sent", move |build| {
            let sent = build.sent();
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

fn built(limit: Option<u32>) -> (Engine, Arc<SentCredit>) {
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
    let reads = sent.reads(&facts);
    assert_eq!(reads.of(&a), one(TOKENS));
    assert_eq!(reads.of(&b), Sent::default());
    assert_eq!(sent.read(&facts, &a), one(TOKENS));
}

#[test]
fn a_pass_keeps_one_tally_map_and_a_later_pass_reads_new_rows() {
    let (engine, sent) = built(None);
    let a = row(107, 3);
    let facts = clock(START + 700);
    let pass = sent.reads(&facts);
    send(&engine, &a, START, TOKENS);
    assert_eq!(pass.of(&a), Sent::default());
    assert_eq!(sent.reads(&facts).of(&a), one(TOKENS));
}

#[test]
fn a_send_with_no_stated_size_counts_only_as_a_request() {
    let (engine, sent) = built(None);
    let a = row(109, 5);
    send(&engine, &a, START, 0);
    assert_eq!(sent.read(&clock(0), &a), one(0));
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
        Sent {
            requests: 8037,
            tokens: 8037 * 13,
        }
    );
    for _ in 8037..8201 {
        send(&engine, &a, START + 1, 19);
    }
    assert_eq!(
        sent.read(&clock(0), &a),
        Sent {
            requests: 7945,
            tokens: 7781 * 13 + 164 * 19,
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
    assert_eq!(sent.read(&facts, &a), Sent::default());
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
    assert_eq!(sent.read(&clock(at + 17_001), &a), Sent::default());
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
    assert_eq!(sent.read(&facts, &a), Sent::default());
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
    let reads = sent.reads(&facts);
    assert_eq!(reads.of(&a), Sent::default());
    assert_eq!(reads.of(&other_rank), Sent::default());
    assert_eq!(reads.of(&b), one(TOKENS));
}

fn validated(document: &PolicySpec, catalog: &Catalog) -> AppliedSpec {
    let digest = document.digest().expect("a digest");
    AppliedSpec::validate_digested(document, digest, catalog, &[]).expect("the policy validates")
}

fn apply(store: &MemoryStore, document: &PolicySpec, catalog: &Catalog) {
    store
        .apply(validated(document, catalog))
        .expect("a newer generation");
}

fn next_document(limit: Option<u32>) -> PolicySpec {
    let mut next = document(limit);
    next.generation = NEXT_GENERATION;
    next
}

fn door() -> (Door, Catalog, Readers) {
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
    (Door::new(store), catalog, readers)
}

fn decide<'a>(door: &'a Door, id: &[u8], now: Instant) -> Handle<&'a Door> {
    Handle::decide(
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

fn chosen(decision: &Handle<&Door>) -> Candidate {
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
    assert_eq!(sent.read(&clock(0), &sent_to), Sent::default());
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
    assert_eq!(readers[0].read(&clock(0), &sent_to), Sent::default());
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
    assert_eq!(readers[1].read(&clock(0), &sent_to), Sent::default());
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
    assert_eq!(readers[1].read(&clock(edge), &sent_to), Sent::default());
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
    assert_eq!(readers[1].read(&clock(0), &sent_to), Sent::default());
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

        let both = Sent {
            requests: 2,
            tokens: 2 * u64::from(TOKENS),
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
    assert_eq!(readers[1].read(&clock(0), &sent_to), Sent::default());
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
    assert_eq!(sent.read(&facts, &prefill), Sent::default());
    lease.tokens = 349;
    engine.on_dispatched(&lease, &target, Side::Prefill, DELAY);
    assert_eq!(sent.read(&facts, &prefill), one(349));
    assert_eq!(sent.read(&facts, &decode), one(TOKENS));
    assert_eq!(sent.read(&facts, &row(decode.id.0, 3)), Sent::default());
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
    assert_eq!(readers[0].read(&facts, &a), Sent::default());
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
