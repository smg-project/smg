//! The records written from worker objects: what a record states at a
//! decision, where each fact was read, and the engine's decision over them.

use std::{cell::Cell, time::Duration};

use smg_worker::{
    identity::builder::BasicWorkerBuilder,
    load::state::one_rank_load,
    test_support::{no_health_check, ready_worker},
    worker::{WorkerSpec, WorkerType},
};

use super::*;
use crate::{
    document::spec::{Assignment, PickSpec, PolicySpec},
    engine::{
        decision::Decision,
        framework::DRAW_FROM,
        view::{AffinityHolder, AffinityRows, Phase},
    },
};

const MODEL: &str = "model-a";

fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

fn fleet(workers: usize) -> Vec<Arc<dyn Worker>> {
    (0..workers)
        .map(|at| ready_worker(&url(at), MODEL))
        .collect()
}

fn applied_of(mut document: PolicySpec) -> AppliedSpec {
    document.generation = 1;
    AppliedSpec::validate(&document).expect("the document validates")
}

fn applied(preset: &str) -> AppliedSpec {
    applied_of(PolicySpec::preset(preset).expect("a built-in preset"))
}

/// A snapshot that holds one report a worker of `queued`, by position:
/// that many queued tokens, sampled `ago` before now.
fn reports(queued: &[i32], ago: Duration) -> (Arc<LoadSnapshot>, Instant) {
    let sampled_at = Instant::now()
        .checked_sub(ago)
        .expect("the host's clock reaches that far back");
    let loads = queued
        .iter()
        .enumerate()
        .map(|(at, &tokens)| {
            let mut report = one_rank_load(tokens, 0.1, 1_000.0);
            report.sampled_at = Some(sampled_at);
            (url(at), report)
        })
        .collect();
    (LoadSnapshot::from_loads_for_test(loads), sampled_at)
}

fn single() -> RequestFacts<'static> {
    RequestFacts {
        model: MODEL,
        phase: Phase::Single,
        prompt_tokens: 64,
        seed: 7,
        ..RequestFacts::default()
    }
}

fn picked(decision: &Decision) -> (u32, WorkerId) {
    let pick = decision.pick().expect("a pick");
    let first = pick.targets.first().expect("a target");
    (first.position, first.id)
}

/// Every target of a decision, by position and id; none for a shed.
fn targets(decision: &Decision) -> Vec<(u32, WorkerId)> {
    decision.pick().map_or_else(Vec::new, |pick| {
        pick.targets
            .iter()
            .map(|target| (target.position, target.id))
            .collect()
    })
}

#[test]
fn a_workers_id_is_the_fingerprint_of_its_address_whichever_object_stands_there() {
    let first = ready_worker("http://worker-a:8000", MODEL);
    let again = ready_worker("http://worker-a:8000", MODEL);
    let other = ready_worker("http://worker-b:8000", MODEL);
    assert_eq!(worker_id(first.as_ref()), worker_id(again.as_ref()));
    assert_ne!(worker_id(first.as_ref()), worker_id(other.as_ref()));
    assert_eq!(
        worker_id(first.as_ref()),
        WorkerId(fingerprint(b"http://worker-a:8000"))
    );
}

#[test]
fn a_record_states_its_workers_registration_by_its_place_in_the_list() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let prefill = <BasicWorkerBuilder>::from_spec(WorkerSpec {
        worker_type: WorkerType::Prefill,
        dp_rank: Some(1),
        dp_size: Some(2),
        ..WorkerSpec::new("grpc://prefill:9000".to_owned())
    })
    .health_config(no_health_check())
    .build();
    let workers: Vec<Arc<dyn Worker>> = vec![ready_worker(&url(0), MODEL), Arc::new(prefill)];
    let pool = PoolRecords::new(&workers, &spec, &clock);
    assert_eq!(pool.len(), 2);
    assert_eq!(
        pool.ids(),
        [
            worker_id(workers[0].as_ref()),
            worker_id(workers[1].as_ref())
        ]
    );
    assert_eq!(pool.generation(), 1);

    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
    let mut rows = Vec::new();
    let (rows, facts) = pool.write(&workers, &live, &mut rows);
    assert_eq!(facts.pool_size, 2);
    assert_eq!(facts.generation, 1);
    assert_eq!((rows[0].position, rows[1].position), (0, 1));
    assert_eq!(rows[0].id, pool.ids()[0]);
    assert_eq!(rows[0].key, pool.ids()[0].0 as u32);
    assert_eq!(
        (rows[0].role, rows[0].wire_modes),
        (Role::Single, WireModes::HTTP)
    );
    assert_eq!((rows[0].dp_rank, rows[0].dp_size), (0, 1));
    assert_eq!(rows[1].role, Role::Prefill);
    assert_eq!((rows[1].dp_rank, rows[1].dp_size), (1, 2));
    // Left for a caller that holds them: open, and no class.
    assert!(rows.iter().all(|row| row.has_capacity && row.has_room));
    assert!(rows.iter().all(|row| row.pair_class == 0));
    // The fixed half is the record with every live field at rest.
    assert_eq!(pool.fixed().len(), 2);
    assert_eq!(pool.fixed()[1].id, rows[1].id);
    assert!(!pool.fixed()[1].health.healthy);
}

#[test]
fn a_record_states_what_its_worker_object_says_at_the_decision() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(4);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let live = || Live::at(spec.engine(), &clock, Instant::now(), None, None);

    let (written, _) = pool.write(&workers, &live(), &mut rows);
    assert!(written.iter().all(|row| row.health.routable()));
    assert!(written.iter().all(|row| row.in_flight == 0));

    // Each fact moves on its worker object, and the next write reads it:
    // nothing of the first write is kept.
    workers[0].set_status(WorkerStatus::NotReady);
    workers[1].set_overloaded(true);
    workers[2].increment_load();
    workers[2].increment_load();
    workers[3].set_status(WorkerStatus::Draining);
    let (written, _) = pool.write(&workers, &live(), &mut rows);
    assert!(!written[0].health.healthy && !written[0].health.draining);
    assert!(written[1].health.overloaded && written[1].health.routable_ignoring_overload());
    assert_eq!(written[2].in_flight, 2);
    assert!(written[2].health.routable());
    assert!(!written[3].health.healthy && written[3].health.draining);
    // One worker read alone is the record the whole write holds of it.
    let alone = pool
        .read(&workers, &live(), 2)
        .expect("a worker of the list");
    assert_eq!((alone.position, alone.in_flight), (2, 2));
    assert_eq!(alone.id, written[2].id);
    assert!(pool.read(&workers, &live(), 4).is_none());

    workers[0].set_status(WorkerStatus::Ready);
    workers[1].set_overloaded(false);
    workers[2].decrement_load();
    let (written, _) = pool.write(&workers, &live(), &mut rows);
    assert!(written[0].health.routable() && written[1].health.routable());
    assert_eq!(written[2].in_flight, 1);
}

#[test]
fn a_record_carries_its_workers_report_dated_on_the_decisions_clock() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(3);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let now = Instant::now();

    // No snapshot, and a snapshot that holds no report of a worker: the
    // record carries none, and the pool does not say that every worker
    // reported.
    let live = Live::at(spec.engine(), &clock, now, None, None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert!(written.iter().all(|row| row.load.is_none()));
    assert!(!facts.reported);
    let (two, _) = reports(&[100, 200], Duration::from_millis(40));
    let live = Live::at(spec.engine(), &clock, now, Some(&two), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert!(written[0].load.is_some() && written[2].load.is_none());
    assert_eq!(written[2].load_sampled_at_micros, 0);
    assert!(!facts.reported);

    let (all, sampled_at) = reports(&[100, 200, 300], Duration::from_millis(40));
    let now = Instant::now();
    let live = Live::at(spec.engine(), &clock, now, Some(&all), None);
    assert_eq!(live.now_micros(), clock.micros(now));
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert!(facts.reported);
    for (row, queued) in written.iter().zip([100, 200, 300]) {
        let load = row.load.expect("a report");
        assert_eq!(load.waiting_uncached_tokens, queued);
        assert_eq!(load.source, SignalSource::Poll);
        // The sample's instant and the decision's stand on one origin.
        assert_eq!(row.load_sampled_at_micros, clock.micros(sampled_at));
        assert!(row.load_sampled_at_micros > 0);
        assert!(row.load_sampled_at_micros < live.now_micros());
        // The report's age is the difference of the two, in milliseconds.
        let age = (live.now_micros() - row.load_sampled_at_micros) / 1_000;
        assert_eq!(u64::from(load.report_age_ms), age);
        assert!((40..10_000).contains(&load.report_age_ms));
    }
}

#[test]
fn a_session_count_is_read_only_for_a_document_that_reads_it() {
    let clock = DecisionClock::new();
    let workers = fleet(2);
    workers[1].increment_routing_key_load("key-a");
    workers[1].increment_routing_key_load("key-a");
    workers[1].increment_routing_key_load("key-b");
    let mut rows = Vec::new();

    // Least load reads neither count: no record carries one, with a key or
    // with none.
    let plain = applied("least_load");
    let pool = PoolRecords::new(&workers, &plain, &clock);
    let live = Live::at(plain.engine(), &clock, Instant::now(), None, Some("key-a"));
    let (written, _) = pool.write(&workers, &live, &mut rows);
    assert!(written
        .iter()
        .all(|row| row.session_in_flight == 0 && row.active_sessions == 0));

    // A sessions rule that places a new key on the worker with the fewest
    // sessions reads the count of keys on each worker.
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rules = 0;
    for binding in &mut document.bindings {
        if let Some(rule) = &mut binding.sessions {
            rule.assignment = Assignment::MinGroup;
            rules += 1;
        }
    }
    assert!(rules > 0, "the manual preset states a sessions rule");
    let grouped = applied_of(document);
    assert!(grouped.engine().needs().active_sessions);
    let pool = PoolRecords::new(&workers, &grouped, &clock);
    let live = Live::at(
        grouped.engine(),
        &clock,
        Instant::now(),
        None,
        Some("key-a"),
    );
    let (written, _) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        (written[0].active_sessions, written[1].active_sessions),
        (0, 2)
    );
}

#[test]
fn every_pool_reads_one_membership_epoch_that_a_new_pool_moves() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let mut rows = Vec::new();
    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);

    let first = PoolRecords::new(&workers, &spec, &clock);
    let (_, facts) = first.write(&workers, &live, &mut rows);
    assert_eq!(facts.membership_epoch, 1);

    // A second list is a membership change of the engine: the first pool
    // reads the new epoch too, so a stage that keeps what it learned of the
    // pools by the epoch learns each again once.
    let others = fleet(3);
    let second = PoolRecords::new(&others, &spec, &clock);
    let (_, facts) = first.write(&workers, &live, &mut rows);
    assert_eq!(facts.membership_epoch, 2);
    let (_, facts) = second.write(&others, &live, &mut rows);
    assert_eq!(facts.membership_epoch, 2);
}

#[test]
fn the_ranks_of_one_engine_are_listed_together_and_a_plain_pool_lists_none() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let mut rows = Vec::new();
    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);

    let plain = fleet(3);
    let pool = PoolRecords::new(&plain, &spec, &clock);
    let (_, facts) = pool.write(&plain, &live, &mut rows);
    assert!(facts.engines.is_empty());

    // Two ranks of one engine, registered apart and out of their ranks'
    // order, around a worker that stands alone.
    let rank = |rank: usize| -> Arc<dyn Worker> {
        Arc::new(
            <BasicWorkerBuilder>::from_spec(WorkerSpec {
                dp_base_url: Some("grpc://engine:9000".to_owned()),
                dp_rank: Some(rank),
                dp_size: Some(2),
                ..WorkerSpec::new(format!("grpc://engine:9000@{rank}"))
            })
            .health_config(no_health_check())
            .build(),
        )
    };
    let ranked = vec![rank(1), ready_worker(&url(0), MODEL), rank(0)];
    let pool = PoolRecords::new(&ranked, &spec, &clock);
    let (_, facts) = pool.write(&ranked, &live, &mut rows);
    assert_eq!(facts.engines.len(), 2);
    assert_eq!(facts.engines.rows(0), [2, 0]);
    assert_eq!(facts.engines.rows(1), [1]);
}

#[test]
fn least_load_decides_over_the_records_by_the_reports_they_carry() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(4);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let cx = Constraints::default();
    let decide_now = |loads: &Arc<LoadSnapshot>| {
        let live = Live::at(spec.engine(), &clock, Instant::now(), Some(loads), None);
        let mut facts = single();
        let decision = decide(spec.engine(), &pool, &workers, &live, &mut facts, &cx);
        assert_eq!(facts.now_micros, live.now_micros());
        decision
    };

    // The worker with the least queued work takes the request.
    let (loads, _) = reports(&[9_000, 8_000, 50, 7_000], Duration::from_millis(5));
    let decision = decide_now(&loads);
    assert_eq!(picked(&decision), (2, worker_id(workers[2].as_ref())));

    // It leaves the pool: the next decision reads that on its object and
    // takes the next least.
    workers[2].set_status(WorkerStatus::NotReady);
    assert_eq!(picked(&decide_now(&loads)).0, 3);

    // No worker passes: the engine sheds, as it does over any records.
    for worker in &workers {
        worker.set_status(WorkerStatus::NotReady);
    }
    assert!(matches!(decide_now(&loads), Decision::Shed(_)));
}

/// A fleet of `workers` whose reports and states differ worker by worker:
/// queued work that falls and rises over the list, one worker in seven not
/// ready, one in eleven overloaded, and a few requests in flight.
fn mixed_fleet(workers: usize) -> (Vec<Arc<dyn Worker>>, Arc<LoadSnapshot>) {
    let fleet = fleet(workers);
    for (at, worker) in fleet.iter().enumerate() {
        if at % 7 == 3 {
            worker.set_status(WorkerStatus::NotReady);
        }
        if at % 11 == 5 {
            worker.set_overloaded(true);
        }
        for _ in 0..at % 4 {
            worker.increment_load();
        }
    }
    let queued: Vec<i32> = (0..workers)
        .map(|at| ((at * 37 + 11) % 97) as i32 * 40)
        .collect();
    let (loads, _) = reports(&queued, Duration::from_millis(5));
    (fleet, loads)
}

/// Asserts that, for `seeds` seeds, the decision that writes a worker's
/// record when its draw asks about the worker ([`decide`]) hands out the
/// targets of the decision over every worker's record written first, under
/// two engines of `document` asked the same requests in the same order.
/// Returns how many of the decisions were a draw's.
fn assert_drawn_as_written(
    document: &PolicySpec,
    workers: &[Arc<dyn Worker>],
    loads: &Arc<LoadSnapshot>,
    named: Option<AffinityRows<'_>>,
    seeds: u64,
) -> usize {
    let (drawing, writing) = (applied_of(document.clone()), applied_of(document.clone()));
    let clock = DecisionClock::new();
    let drawn_pool = PoolRecords::new(workers, &drawing, &clock);
    let written_pool = PoolRecords::new(workers, &writing, &clock);
    let cx = Constraints::default();
    let mut rows = Vec::new();
    let mut scratch = Scratch::default();
    let mut draws = 0;
    for seed in 0..seeds {
        let now = Instant::now();
        let request = RequestFacts {
            seed: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            named,
            ..single()
        };

        let live = Live::at(drawing.engine(), &clock, now, Some(loads), None);
        let mut facts = request;
        let unsaid = drawn_pool.facts(&clock, false);
        draws += usize::from(
            drawing
                .engine()
                .drawn_for(&facts, &unsaid, drawn_pool.len())
                .is_some(),
        );
        let drawn = decide(
            drawing.engine(),
            &drawn_pool,
            workers,
            &live,
            &mut facts,
            &cx,
        );

        let live = Live::at(writing.engine(), &clock, now, Some(loads), None);
        let mut facts = request;
        facts.now_micros = live.now_micros();
        let (written, pool_facts) = written_pool.write(workers, &live, &mut rows);
        let whole = writing
            .engine()
            .decide(&facts, &pool_facts, written, &cx, &mut scratch);

        assert_eq!(
            targets(&drawn),
            targets(&whole),
            "seed {seed}: the draw before the records and the whole write hand out the same targets"
        );
        assert_eq!(
            matches!(drawn, Decision::Shed(_)),
            matches!(whole, Decision::Shed(_))
        );
    }
    draws
}

#[test]
fn a_draw_before_the_records_hands_out_the_whole_writes_targets() {
    let (workers, loads) = mixed_fleet(64);
    let preset = |name: &str| PolicySpec::preset(name).expect("a built-in preset");
    let mut count_first = preset("power_of_two");
    for profile in count_first.profiles.values_mut() {
        profile.pick = PickSpec::from("power-of-two-count-first");
    }
    for (name, document) in [
        ("power_of_two", preset("power_of_two")),
        ("power-of-two-count-first", count_first),
        ("random", preset("random")),
        ("passthrough", preset("passthrough")),
        ("round_robin", preset("round_robin")),
    ] {
        let draws = assert_drawn_as_written(&document, &workers, &loads, None, 400);
        assert_eq!(draws, 400, "{name}: every decision is its draw's");
    }

    // The sampled best draws for a request that names rows, and the rows
    // named join the rows drawn: their records are written too.
    let holders = [
        AffinityHolder {
            row: 9,
            overlap_blocks: 12,
            predicted_blocks: 0,
        },
        AffinityHolder {
            row: 40,
            overlap_blocks: 3,
            predicted_blocks: 0,
        },
    ];
    let named = AffinityRows {
        holders: &holders,
        warming: &[17],
        ..AffinityRows::default()
    };
    let document = preset("cache_aware");
    let draws = assert_drawn_as_written(&document, &workers, &loads, Some(named), 400);
    assert_eq!(
        draws, 400,
        "cache_aware: a request that names rows is drawn for"
    );
    // A request that names none is the whole pass's, and decides the same.
    assert_eq!(
        assert_drawn_as_written(&document, &workers, &loads, None, 50),
        0
    );
}

#[test]
fn a_draw_that_keeps_nobody_widens_to_every_record() {
    // One worker in sixty-four is ready: nearly every draw of two lands on
    // refused workers alone, and the whole pass finds the one.
    let (workers, loads) = mixed_fleet(64);
    for (at, worker) in workers.iter().enumerate() {
        if at != 29 {
            worker.set_status(WorkerStatus::NotReady);
        }
    }
    let document = PolicySpec::preset("power_of_two").expect("a built-in preset");
    assert_drawn_as_written(&document, &workers, &loads, None, 200);

    let spec = applied_of(document);
    let clock = DecisionClock::new();
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let mut facts = single();
    let decision = decide(
        spec.engine(),
        &pool,
        &workers,
        &live,
        &mut facts,
        &Constraints::default(),
    );
    assert_eq!(picked(&decision).0, 29);
}

#[test]
fn a_pick_of_two_writes_the_records_of_the_two_workers_it_draws() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(512);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
    let cx = Constraints::default();
    let mut facts = single();
    facts.now_micros = live.now_micros();
    let unsaid = pool.facts(&clock, false);
    let drawn = spec
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .expect("a pick of two over a large pool is its draw's");
    let written = Cell::new(0);
    let mut write = |fixed: &Candidate, row: &mut Candidate| {
        written.set(written.get() + 1);
        *row = *fixed;
        live.write(workers[fixed.position as usize].as_ref(), row);
    };
    let mut rows = vec![Candidate::default(); pool.len()];
    let decision = spec.engine().decide_drawn(
        drawn,
        &facts,
        &unsaid,
        pool.fixed(),
        &mut rows,
        &mut write,
        &cx,
        &mut Scratch::default(),
    );
    assert_eq!(decision.pick().expect("a pick").targets.len(), 2);
    assert_eq!(written.get(), 2, "two records of 512");

    // Under the draw's floor every record is written: the pool is small.
    let few = fleet(DRAW_FROM - 1);
    let pool = PoolRecords::new(&few, &spec, &clock);
    let unsaid = pool.facts(&clock, false);
    assert!(spec
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .is_none());

    // A profile that scores every worker is no draw's.
    let scores = applied("least_load");
    let pool = PoolRecords::new(&workers, &scores, &clock);
    let unsaid = pool.facts(&clock, false);
    assert!(scores
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .is_none());
}
