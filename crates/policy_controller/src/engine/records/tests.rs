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
        view::{fingerprint, AffinityHolder, AffinityRows, Phase},
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
    // A removal names a worker by its address alone: the same id.
    assert_eq!(
        WorkerId::from_address("http://worker-a:8000"),
        worker_id(first.as_ref())
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

/// The rows a decision writes borrow their buffer alone, and the pool's
/// facts what its live facts borrow: the rows stand after the key text
/// handed to the live facts is gone. A build that ties the two fails here.
#[test]
fn the_rows_written_outlive_what_the_live_facts_borrow() {
    let (spec, clock) = (applied("round_robin"), DecisionClock::new());
    let workers: Vec<Arc<dyn Worker>> = vec![ready_worker(&url(0), MODEL)];
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let written = {
        let key = String::from("key-a");
        let live = Live::at(spec.engine(), &clock, Instant::now(), None, Some(&key));
        pool.write(&workers, &live, &mut rows).0
    };
    assert_eq!(written.len(), 1);
    // The method is one function value for a call of any lifetime: a bound
    // between its lifetimes would fix them where it is named.
    type Write = for<'a> fn(
        &'a PoolRecords,
        &[Arc<dyn Worker>],
        &Live<'a>,
        &'a mut Vec<Candidate>,
    ) -> (&'a mut [Candidate], PoolFacts<'a>);
    let _: Write = PoolRecords::write;
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

/// A list builds its column of demand facts where a worker of it states
/// one, and whether a worker registered a data-parallel rank is no such
/// fact. A list of rank workers that states nothing builds no column, as a
/// list of plain workers builds none, and each of its workers meets a
/// request that demands nothing of them. A list with one worker that
/// states a runtime builds the column.
#[test]
fn a_list_of_rank_workers_builds_no_demand_column() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
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
    let ranked = vec![rank(0), rank(1)];
    assert!(ranked.iter().all(|worker| worker.is_dp_aware()));
    let pool = PoolRecords::new(&ranked, &spec, &clock);
    assert!(pool.demand_facts.is_none());
    let mut kept = Vec::new();
    assert_eq!(pool.meeting(Demands::default(), None, &mut kept), None);
    assert_eq!(kept, [0, 1]);
    assert!(PoolRecords::new(&fleet(2), &spec, &clock)
        .demand_facts
        .is_none());

    let stated = PoolRecords::with_fixed_facts(&ranked, &spec, &clock, |worker| FixedFacts {
        demands: DemandFacts {
            runtime: u32::from(worker.url().ends_with("@1")),
            ..DemandFacts::default()
        },
        ..FixedFacts::default()
    });
    let column = stated.demand_facts.as_deref().expect("a column");
    assert_eq!(
        column.iter().map(|facts| facts.runtime).collect::<Vec<_>>(),
        [0, 1]
    );
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

/// A stage tells which records are ranks of one engine by a field of the
/// record: each states its engine's place among the list's engines, in the
/// order each engine's first row stands in the list. Every record a decision
/// is handed states it, written with the pool, from a converted snapshot or
/// alone, and it is the place the list's engines are listed at.
#[test]
fn each_record_states_its_engines_place_among_the_lists_engines() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let mut rows = Vec::new();
    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
    let places = |rows: &[Candidate]| -> Vec<u32> { rows.iter().map(|row| row.engine).collect() };

    // A pool with no engine of several rows lists none, and each record
    // states a place of its own.
    let plain = fleet(3);
    let pool = PoolRecords::new(&plain, &spec, &clock);
    assert_eq!(places(pool.fixed()), [0, 1, 2]);
    let (written, facts) = pool.write(&plain, &live, &mut rows);
    assert_eq!(places(written), [0, 1, 2]);
    assert!(facts.engines.is_empty());

    // Two engines of two ranks each, registered apart and out of their
    // ranks' order, around a worker that stands alone.
    let rank = |engine: &str, rank: usize| -> Arc<dyn Worker> {
        Arc::new(
            <BasicWorkerBuilder>::from_spec(WorkerSpec {
                dp_base_url: Some(format!("grpc://{engine}:9000")),
                dp_rank: Some(rank),
                dp_size: Some(2),
                ..WorkerSpec::new(format!("grpc://{engine}:9000@{rank}"))
            })
            .health_config(no_health_check())
            .build(),
        )
    };
    let ranked = vec![
        rank("b", 1),
        rank("a", 0),
        ready_worker(&url(0), MODEL),
        rank("a", 1),
        rank("b", 0),
    ];
    let stated = [0, 1, 2, 1, 0];
    let pool = PoolRecords::new(&ranked, &spec, &clock);
    assert_eq!(places(pool.fixed()), stated);
    let (written, facts) = pool.write(&ranked, &live, &mut rows);
    assert_eq!(places(written), stated);
    // A record's place is the place its engine is listed at.
    assert_eq!(facts.engines.len(), 3);
    for row in written {
        let listed = facts.engines.rows(row.engine as usize);
        assert!(listed.contains(&row.position), "row {}", row.position);
    }
    // One record read alone states it too.
    for (position, engine) in (0u32..).zip(stated) {
        let row = pool.read(&ranked, &live, position);
        assert_eq!(row.map(|row| row.engine), Some(engine));
    }
    // And so do the rows converted under a load snapshot.
    let (loads, _) = reports(&[10, 20, 30, 40, 50], Duration::from_millis(5));
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let (written, _) = pool.write(&ranked, &live, &mut rows);
    assert_eq!(places(written), stated);
    assert_eq!(pool.conversions(), 1);
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
/// `converted` says whether the drawing side's list holds its snapshot's
/// rows before the draws: a draw then copies a worker's report from them,
/// and looks the report up itself where the list holds none. Returns how
/// many of the decisions were a draw's.
fn assert_drawn_as_written(
    document: &PolicySpec,
    workers: &[Arc<dyn Worker>],
    loads: &Arc<LoadSnapshot>,
    named: Option<AffinityRows<'_>>,
    seeds: u64,
    converted: bool,
) -> usize {
    let (drawing, writing) = (applied_of(document.clone()), applied_of(document.clone()));
    let clock = DecisionClock::new();
    let drawn_pool = PoolRecords::new(workers, &drawing, &clock);
    let written_pool = PoolRecords::new(workers, &writing, &clock);
    let cx = Constraints::default();
    let mut rows = Vec::new();
    if converted {
        let live = Live::at(drawing.engine(), &clock, Instant::now(), Some(loads), None);
        let _ = drawn_pool.write(workers, &live, &mut rows);
    }
    assert_eq!(held(&drawn_pool).is_some(), converted);
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
        let unsaid = drawn_pool.facts(&live, false);
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
    // Where every decision was a draw's, the list was converted one time, ahead
    // of them or by them: the targets above are the whole write's either side.
    if draws as u64 == seeds {
        assert_eq!(drawn_pool.conversions(), 1);
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
        for converted in [false, true] {
            let draws = assert_drawn_as_written(&document, &workers, &loads, None, 400, converted);
            assert_eq!(draws, 400, "{name}: every decision is its draw's");
        }
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
    for converted in [false, true] {
        let draws =
            assert_drawn_as_written(&document, &workers, &loads, Some(named), 400, converted);
        assert_eq!(
            draws, 400,
            "cache_aware: a request that names rows is drawn for"
        );
    }
    // A request that names none is the whole pass's, and decides the same.
    assert_eq!(
        assert_drawn_as_written(&document, &workers, &loads, None, 50, false),
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
    assert_drawn_as_written(&document, &workers, &loads, None, 200, false);
    assert_drawn_as_written(&document, &workers, &loads, None, 200, true);

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
    let unsaid = pool.facts(&live, false);
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
    let unsaid = pool.facts(&live, false);
    assert!(spec
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .is_none());

    // A profile that scores every worker is no draw's.
    let scores = applied("least_load");
    let pool = PoolRecords::new(&workers, &scores, &clock);
    let unsaid = pool.facts(&live, false);
    assert!(scores
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .is_none());
}

/// The rows the list holds of one load snapshot, where it holds any.
fn held(pool: &PoolRecords) -> Option<Arc<Reported>> {
    pool.reported.load_full()
}

/// A record as its every field reads: two records of one worker at one
/// instant are the same text.
fn text(row: &Candidate) -> String {
    format!("{row:?}")
}

#[test]
fn a_lists_reports_are_converted_once_a_snapshot_by_the_decision_that_holds_it() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(3);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);
    assert!(held(&pool).is_none() && pool.conversions() == 0);

    // A decision that holds no snapshot converts nothing.
    let live = Live::at(spec.engine(), &clock, at(0), None, None);
    let _ = pool.write(&workers, &live, &mut rows);
    assert!(held(&pool).is_none() && pool.conversions() == 0);

    // The first decision under a snapshot converts the list's reports, and
    // the list holds them under the snapshot's name.
    let (first, _) = reports(&[100, 200, 300], Duration::from_millis(40));
    let live = Live::at(spec.engine(), &clock, at(1), Some(&first), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert!(facts.reported);
    assert_eq!(
        written[1].load.expect("a report").waiting_uncached_tokens,
        200
    );
    let converted = held(&pool).expect("the rows of the first snapshot");
    assert!(converted.is_of(&first));
    assert_eq!(pool.conversions(), 1);

    // The decisions behind it under the same snapshot read those rows: no
    // report is converted again, and the rows are the same value.
    for millis in 2..6 {
        let live = Live::at(spec.engine(), &clock, at(millis), Some(&first), None);
        let (written, _) = pool.write(&workers, &live, &mut rows);
        assert_eq!(
            written[2].load.expect("a report").waiting_uncached_tokens,
            300
        );
    }
    assert_eq!(pool.conversions(), 1);
    assert!(Arc::ptr_eq(
        &converted,
        &held(&pool).expect("the same rows")
    ));

    // The next snapshot replaces the rows whole: one conversion more.
    let (second, _) = reports(&[900, 800], Duration::from_millis(10));
    let live = Live::at(spec.engine(), &clock, at(10), Some(&second), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        900
    );
    assert!(written[2].load.is_none() && !facts.reported);
    let replaced = held(&pool).expect("the rows of the second snapshot");
    assert!(replaced.is_of(&second) && !replaced.is_of(&first));
    assert_eq!(pool.conversions(), 2);

    // The rows name their snapshot and do not hold it: the first snapshot
    // is gone with its last holder.
    let name = Arc::downgrade(&first);
    drop(first);
    assert!(name.upgrade().is_none());
    drop(second);
    assert_eq!(replaced.snapshot.strong_count(), 0);
}

/// A document that declares the two token counts a report offers by name,
/// and reads no load report. Where `read`, a stage of its one profile reads
/// both counts. Where not, no stage reads either.
fn by_signals(read: bool) -> AppliedSpec {
    let room = r#",{"has-room":{"used":{"signal.kv_tokens_used":1},
                    "capacity":"signal.kv_tokens_capacity"}}"#;
    let document = format!(
        r#"{{"apiVersion":"smg.policy/v1",
            "signals":{{"kv_tokens_capacity":{{"unit":"tokens"}},
                       "kv_tokens_used":{{"unit":"tokens"}}}},
            "profiles":{{"p":{{"filter":["healthy"{}],"pick":"random"}}}}}}"#,
        if read { room } else { "" },
    );
    applied_of(serde_json::from_str(&document).expect("a well-formed document"))
}

/// A snapshot in which each worker of `used`, by position, uses that many
/// of a thousand tokens, sampled now.
fn counts(used: &[i32]) -> (Arc<LoadSnapshot>, Instant) {
    let sampled_at = Instant::now();
    let loads = used
        .iter()
        .enumerate()
        .map(|(at, &used)| {
            let mut report = one_rank_load(0, 0.1, 1_000.0);
            report.sampled_at = Some(sampled_at);
            for rank in &mut report.loads {
                rank.num_used_tokens = used;
                rank.max_total_num_tokens = 1_000;
            }
            (url(at), report)
        })
        .collect();
    (LoadSnapshot::from_loads_for_test(loads), sampled_at)
}

/// Each row's capacity and what is used of it, as `cells` hold them for
/// three rows under [`by_signals`]: the capacity's column stands first.
fn counted(cells: &SignalCells) -> Vec<Option<(f64, f64)>> {
    (0..3)
        .map(|row| {
            let row = cells.row(row).expect("a row of cells");
            (!row[0].is_none()).then(|| (row[0].value, row[1].value))
        })
        .collect()
}

/// A conversion of a list's reports makes the list's signal cells under a
/// document a stage of which reads a named signal: a cell a worker a
/// signal, each value at its worker's place under its column, dated on the
/// decision's clock. A decision is lent those cells under their snapshot.
/// One that holds another snapshot and is not the one to convert is lent
/// its own snapshot's, in the thread's buffer, which takes them back. One
/// that holds no snapshot is lent none. Under a document that declares the
/// signals and reads neither, a conversion makes no cell.
#[test]
fn a_conversion_makes_the_lists_signal_cells_under_a_document_that_reads_one() {
    let spec = by_signals(true);
    let clock = DecisionClock::new();
    let workers = fleet(3);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);

    // A decision that writes every record converts: the third worker has
    // no report, and its cells hold no value.
    let (first, sampled_at) = counts(&[100, 200]);
    let live = Live::at(spec.engine(), &clock, at(5), Some(&first), None);
    let _ = pool.write(&workers, &live, &mut Vec::new());
    let converted = held(&pool).expect("the rows of the first snapshot");
    assert!(converted.cells.fits(3) && converted.cells.signals() == 2);
    let stated = [Some((1_000.0, 100.0)), Some((1_000.0, 200.0)), None];
    assert_eq!(counted(&converted.cells), stated);
    let dated = converted.cells.cells().iter().take(4);
    assert!(dated
        .map(|cell| cell.sampled_at_micros)
        .eq([clock.micros(sampled_at); 4]));
    let lent = pool.cells(&workers, &live).expect("held cells");
    let own = ptr::eq(lent.cells().expect("cells"), &converted.cells);
    assert!(own, "the cells the conversion left");

    // A decision that began earlier and holds another snapshot leaves the
    // list's rows in place, and reads its own snapshot's values. They
    // stand in the thread's buffer, which has its room back afterwards.
    let (second, _) = counts(&[700, 800, 900]);
    let earlier = Live::at(spec.engine(), &clock, at(1), Some(&second), None);
    let lent = pool.cells(&workers, &earlier).expect("held cells");
    let all = [700.0, 800.0, 900.0].map(|used| Some((1_000.0, used)));
    assert_eq!(counted(lent.cells().expect("cells")), all);
    assert!(held(&pool).is_some_and(|rows| rows.is_of(&first)));
    assert_eq!(CELLS.with(Cell::take).cells_mut().capacity(), 0);
    drop(lent);
    assert!(CELLS.with(Cell::take).cells_mut().capacity() >= 6);
    // A later one that holds it converts, and is lent what it converted.
    let later = Live::at(spec.engine(), &clock, at(9), Some(&second), None);
    let lent = pool.cells(&workers, &later).expect("held cells");
    assert_eq!(counted(lent.cells().expect("cells")), all);
    assert!(held(&pool).is_some_and(|rows| rows.is_of(&second)));
    assert_eq!(pool.conversions(), 2);
    // Under no snapshot a decision is lent no cells.
    let bare = Live::at(spec.engine(), &clock, at(10), None, None);
    assert!(pool.cells(&workers, &bare).is_none());

    // A document that declares the two signals and reads neither: the
    // conversion makes no cell.
    let unread = by_signals(false);
    let pool = PoolRecords::new(&workers, &unread, &clock);
    let live = Live::at(unread.engine(), &clock, at(5), Some(&first), None);
    let _ = pool.write(&workers, &live, &mut Vec::new());
    let converted = held(&pool).expect("the rows of the first snapshot");
    assert!(converted.cells.cells().is_empty(), "no cell is made");
}

#[test]
fn a_decision_that_holds_another_snapshot_reads_its_own_reports() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);
    let (old, _) = reports(&[100, 200], Duration::from_millis(40));
    let (new, _) = reports(&[900], Duration::from_millis(10));

    // The list holds the new snapshot's rows, converted by a decision that
    // began at the tenth millisecond.
    let live = Live::at(spec.engine(), &clock, at(10), Some(&new), None);
    let _ = pool.write(&workers, &live, &mut rows);
    let converted = held(&pool).expect("the rows of the new snapshot");
    assert!(converted.is_of(&new));

    // A decision that began before it still holds the old snapshot. It
    // reads the old snapshot's reports, every one, never the list's rows,
    // and it leaves the rows in place: its snapshot is no newer.
    let live = Live::at(spec.engine(), &clock, at(9), Some(&old), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        100
    );
    assert_eq!(
        written[1].load.expect("a report").waiting_uncached_tokens,
        200
    );
    assert!(facts.reported, "the old snapshot holds a report of each");
    assert!(Arc::ptr_eq(
        &converted,
        &held(&pool).expect("the same rows")
    ));
    assert_eq!(pool.conversions(), 1);
    // One record read alone under the old snapshot is the old report too.
    let alone = pool.read(&workers, &live, 1).expect("a worker");
    assert_eq!(text(&alone), text(&written[1]));

    // The decision under the new snapshot still reads the new rows.
    let live = Live::at(spec.engine(), &clock, at(11), Some(&new), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        900
    );
    assert!(written[1].load.is_none() && !facts.reported);
    assert_eq!(pool.conversions(), 1);
}

#[test]
fn a_decision_that_meets_a_conversion_under_way_converts_nothing_and_waits_for_nothing() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let (loads, _) = reports(&[100, 200], Duration::from_millis(40));

    // Another decision holds the list's claim: it is converting.
    let claim = pool.converting.try_lock().expect("the claim is free");
    assert!(
        pool.converting.try_lock().is_none(),
        "one decision holds it"
    );
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    // This one wrote its own records from its own snapshot, and left
    // nothing behind.
    assert_eq!(
        written[1].load.expect("a report").waiting_uncached_tokens,
        200
    );
    assert!(facts.reported);
    assert!(held(&pool).is_none() && pool.conversions() == 0);

    // The claim is given back at its drop, and the next decision converts.
    drop(claim);
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let _ = pool.write(&workers, &live, &mut rows);
    assert!(held(&pool).is_some_and(|converted| converted.is_of(&loads)));
    assert_eq!(pool.conversions(), 1);
    assert!(pool.converting.try_lock().is_some(), "given back");
}

/// One draw of `spec`'s engine over `pool` under `loads`, by `seed`.
fn drawn(
    spec: &AppliedSpec,
    clock: &DecisionClock,
    pool: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    loads: &Arc<LoadSnapshot>,
    seed: u64,
) -> Decision {
    let live = Live::at(spec.engine(), clock, Instant::now(), Some(loads), None);
    let mut facts = RequestFacts { seed, ..single() };
    let cx = Constraints::default();
    decide(spec.engine(), pool, workers, &live, &mut facts, &cx)
}

/// Draws convert a list's reports once they looked up as many as it holds: a
/// pick of two looks up two a decision, so the 32nd pays the last of the 64 and
/// converts. Later draws copy the list's rows, and a new snapshot counts again.
#[test]
fn draws_convert_a_lists_reports_once_they_looked_up_a_lists_length() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(64);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let queued: Vec<i32> = (0..64).map(|at| at * 10).collect();
    for snapshot in 1..=2 {
        let (loads, _) = reports(&queued, Duration::from_millis(5));
        for seed in 1..=31 {
            let decision = drawn(&spec, &clock, &pool, &workers, &loads, seed);
            assert_eq!(decision.pick().expect("a pick").targets.len(), 2);
            assert_eq!(pool.conversions(), snapshot - 1, "decision {seed}");
        }
        let _ = drawn(&spec, &clock, &pool, &workers, &loads, 32);
        assert_eq!(pool.conversions(), snapshot, "the draw that pays the last");
        assert!(held(&pool).is_some_and(|rows| rows.is_of(&loads)));
        for seed in 33..=200 {
            let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
        }
        assert_eq!(pool.conversions(), snapshot, "the rows are copied");
    }
}

/// A snapshot published after one that is gone starts the list's count again,
/// wherever it stands in memory: the list names the snapshot it counted under,
/// so 62 look-ups under one and four under each of 64 later convert nothing.
#[test]
fn a_later_snapshot_does_not_inherit_a_lists_count() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(64);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let queued: Vec<i32> = (0..64).map(|at| at * 10).collect();
    let (first, _) = reports(&queued, Duration::from_millis(5));
    for seed in 1..=31 {
        let _ = drawn(&spec, &clock, &pool, &workers, &first, seed);
    }
    // One more decision would convert. The snapshot goes, its place stays named.
    let place = Arc::as_ptr(&first).addr();
    drop(first);
    for later in 0..64 {
        let (loads, _) = reports(&queued, Duration::from_millis(5));
        if later == 0 {
            assert_ne!(Arc::as_ptr(&loads).addr(), place, "the place is named");
        }
        for seed in [100 + later, 200 + later] {
            let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
            assert_eq!(pool.conversions(), 0, "snapshot {later}");
        }
    }
}

/// A draw that finds the list's claim taken counts nothing and waits for
/// nothing: the count starts with the first draw that takes the claim.
#[test]
fn a_draw_that_finds_the_claim_taken_counts_nothing() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(64);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let queued: Vec<i32> = (0..64).map(|at| at * 10).collect();
    let (loads, _) = reports(&queued, Duration::from_millis(5));
    let claim = pool.converting.try_lock().expect("the claim is free");
    for seed in 1..=40 {
        let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
    }
    assert_eq!(claim.records, 0, "80 look-ups, none counted");
    drop(claim);
    for seed in 1..=31 {
        let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
    }
    assert_eq!(pool.conversions(), 0, "62 counted");
    let _ = drawn(&spec, &clock, &pool, &workers, &loads, 32);
    assert_eq!(pool.conversions(), 1);
}

#[test]
fn what_a_worker_object_says_and_a_reports_age_are_read_at_every_decision() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let (loads, _) = reports(&[100, 200], Duration::from_millis(40));
    let mut rows = Vec::new();
    let now = Instant::now();
    let live = Live::at(spec.engine(), &clock, now, Some(&loads), None);
    let (written, _) = pool.write(&workers, &live, &mut rows);
    let age = written[0].load.expect("a report").report_age_ms;
    assert!(written[0].health.routable() && written[1].in_flight == 0);

    // The list holds the snapshot's rows now. A worker leaves, another
    // takes a request, and a quarter of a second passes: the next decision
    // under the same snapshot reads all three, and converts nothing.
    workers[0].set_status(WorkerStatus::NotReady);
    workers[1].increment_load();
    let later = now + Duration::from_millis(250);
    let live = Live::at(spec.engine(), &clock, later, Some(&loads), None);
    let (written, _) = pool.write(&workers, &live, &mut rows);
    assert!(!written[0].health.healthy);
    assert_eq!(written[1].in_flight, 1);
    assert_eq!(written[0].load.expect("a report").report_age_ms, age + 250);
    assert_eq!(pool.conversions(), 1);

    // The rows the list holds carry none of it: no word of a worker object
    // and no age.
    let converted = held(&pool).expect("the snapshot's rows");
    for row in &converted.rows {
        assert!(!row.health.healthy && row.in_flight == 0);
        assert_eq!(row.load.expect("a report").report_age_ms, 0);
    }

    // One record read alone is the whole write's record, from the list's
    // rows and from a list that holds none.
    let alone = pool.read(&workers, &live, 1).expect("a worker");
    assert_eq!(text(&alone), text(&written[1]));
    let cold = PoolRecords::new(&workers, &spec, &clock);
    let alone = cold.read(&workers, &live, 1).expect("a worker");
    assert_eq!(text(&alone), text(&written[1]));
    assert!(
        held(&cold).is_none() && cold.conversions() == 0,
        "a read of one record converts no other worker's report"
    );
}

/// An ask is lent the snapshot current when it reads the feed, which is
/// behind the instant the ask was made: a report in it can be sampled
/// after that instant. Such a report is of age zero in a record found by
/// look-up, in the whole write and in a record read from the list's rows:
/// never an age that wrapped, which a stage would read as a stale report.
#[test]
fn a_report_sampled_after_the_decisions_instant_is_of_age_zero() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let now = Instant::now();
    let sampled_at = now + Duration::from_millis(30);
    let loads = (0..2)
        .map(|at| {
            let mut report = one_rank_load(100, 0.1, 1_000.0);
            report.sampled_at = Some(sampled_at);
            (url(at), report)
        })
        .collect();
    let loads = LoadSnapshot::from_loads_for_test(loads);
    let live = Live::at(spec.engine(), &clock, now, Some(&loads), None);

    let found = pool.read(&workers, &live, 1).expect("a worker");
    assert!(held(&pool).is_none(), "one read converts no list");
    assert!(found.load_sampled_at_micros > live.now_micros());
    assert_eq!(found.load.expect("a report").report_age_ms, 0);

    let mut rows = Vec::new();
    let (written, _) = pool.write(&workers, &live, &mut rows);
    for row in written {
        assert_eq!(row.load.expect("a report").report_age_ms, 0);
    }
    assert!(held(&pool).is_some());
    let copied = pool.read(&workers, &live, 0).expect("a worker");
    assert_eq!(copied.load.expect("a report").report_age_ms, 0);
}

#[test]
fn a_workers_place_is_found_by_its_id_and_by_no_id_two_rows_carry() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(5);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    for (place, worker) in (0u32..).zip(&workers) {
        assert_eq!(pool.place_of(worker_id(worker.as_ref())), Some(place));
    }
    let stranger = ready_worker("http://stranger:8000", MODEL);
    assert_eq!(pool.place_of(worker_id(stranger.as_ref())), None);

    // One address listed twice: a pin names neither row, and no id of the
    // list has a place.
    let twice = vec![
        ready_worker(&url(0), MODEL),
        ready_worker(&url(1), MODEL),
        ready_worker(&url(0), MODEL),
    ];
    let pool = PoolRecords::new(&twice, &spec, &clock);
    for worker in &twice {
        assert_eq!(pool.place_of(worker_id(worker.as_ref())), None);
    }
}

/// A retry's tried workers are widened to their engines: the workers as
/// they were named, and behind them every other rank of an engine one of
/// them is a rank of, each once, in the engine's ranks' order. A worker
/// that stands alone and an id the list does not hold add nothing. A
/// program's own ids come back as it stated them. Where two rows carry one
/// id nothing is added, and a list of workers that each stand alone says
/// it holds no ranks.
#[test]
fn a_tried_worker_names_every_rank_of_its_engine_once() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    // One rank of the three-rank engine at `engine`.
    let rank = |engine: &str, rank: usize| -> Arc<dyn Worker> {
        Arc::new(
            <BasicWorkerBuilder>::new(engine.to_owned())
                .dp_config(rank, 3)
                .health_config(no_health_check())
                .build(),
        )
    };
    let (a, b) = ("grpc://engine-a:9000", "grpc://engine-b:9000");
    // Two engines' ranks out of their order, around a worker alone.
    let workers = vec![
        rank(a, 2),
        rank(b, 0),
        ready_worker(&url(0), MODEL),
        rank(a, 0),
        rank(b, 1),
        rank(a, 1),
        rank(b, 2),
    ];
    let pool = PoolRecords::new(&workers, &spec, &clock);
    assert!(pool.has_ranks());
    let id = |place: usize| worker_id(workers[place].as_ref());
    // What the list held before is not kept.
    let mut whole = vec![WorkerId(9)];
    pool.tried_engines(&[id(5)], &mut whole);
    assert_eq!(whole, [id(5), id(3), id(0)], "the tried rank first");
    pool.tried_engines(&[id(1)], &mut whole);
    assert_eq!(whole, [id(1), id(4), id(6)], "the second engine's own");
    pool.tried_engines(&[id(2)], &mut whole);
    assert_eq!(whole, [id(2)], "a worker alone names itself");
    // A further list of the same decision adds its own engines' ranks to
    // what the list holds, and takes nothing out of it.
    let mut whole = vec![WorkerId(9), id(5)];
    pool.more_tried_engines(&[id(5), id(1)], &mut whole);
    assert_eq!(
        whole,
        [WorkerId(9), id(5), id(3), id(0), id(1), id(4), id(6)]
    );

    let stranger = WorkerId(77);
    pool.tried_engines(&[id(0), id(2), stranger, id(3), id(4)], &mut whole);
    let both = [id(0), id(2), stranger, id(3), id(4), id(5), id(1), id(6)];
    assert_eq!(whole, both, "each rank once, behind the workers named");

    let mut next = 100;
    let stated = PoolRecords::with_fixed_facts(&workers, &spec, &clock, |_| {
        next += 1;
        FixedFacts {
            id: Some(WorkerId(next)),
            ..FixedFacts::default()
        }
    });
    stated.tried_engines(&[WorkerId(106)], &mut whole);
    assert_eq!(whole, [WorkerId(106), WorkerId(104), WorkerId(101)]);

    // One rank listed twice: no id of the list has a place.
    let twice = vec![rank(a, 0), rank(a, 1), rank(a, 0)];
    let pool = PoolRecords::new(&twice, &spec, &clock);
    let named = [worker_id(twice[1].as_ref())];
    pool.tried_engines(&named, &mut whole);
    assert_eq!(whole, named);

    let alone = fleet(3);
    let pool = PoolRecords::new(&alone, &spec, &clock);
    assert!(!pool.has_ranks());
    let named = [worker_id(alone[1].as_ref())];
    pool.tried_engines(&named, &mut whole);
    assert_eq!(whole, named);
}

/// A report's age stops at the far end: a decision one millisecond past
/// the largest age a `u32` holds reads `u32::MAX` milliseconds, never an
/// age that wrapped to a young report.
#[test]
fn a_record_caps_its_report_age_at_u32_max() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(1);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let (loads, sampled_at) = reports(&[100], Duration::ZERO);
    let now = sampled_at + Duration::from_millis(u64::from(u32::MAX) + 1);
    let live = Live::at(spec.engine(), &clock, now, Some(&loads), None);

    let mut rows = Vec::new();
    let (written, _) = pool.write(&workers, &live, &mut rows);
    assert_eq!(written[0].load.expect("a report").report_age_ms, u32::MAX);
}

/// Two ranks of one engine report at two instants, and one decision ages
/// each from its own sample. A rank that has not reported reads no load
/// beside a rank that has.
#[test]
fn two_ranks_are_each_aged_from_their_own_sample() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let base = url(0);
    let workers = [0, 1].map(|rank| -> Arc<dyn Worker> {
        Arc::new(
            <BasicWorkerBuilder>::new(base.clone())
                .dp_config(rank, 2)
                .health_config(no_health_check())
                .build(),
        )
    });
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let ages_at = |now, reports| {
        let loads = LoadSnapshot::from_loads_for_test(reports);
        let live = Live::at(spec.engine(), &clock, now, Some(&loads), None);
        let mut rows = Vec::new();
        let (written, _) = pool.write(&workers, &live, &mut rows);
        [0, 1].map(|rank| {
            let row = written.iter().find(|row| row.dp_rank == rank);
            let row = row.expect("the rank's record");
            row.load.map(|load| load.report_age_ms)
        })
    };

    let first_at = Instant::now();
    let mut first = one_rank_load(0, 0.1, 1_000.0);
    first.sampled_at = Some(first_at);
    let mut reports = vec![(workers[0].url().to_owned(), first)];
    assert_eq!(
        ages_at(first_at + Duration::from_millis(2), reports.clone()),
        [Some(2), None]
    );

    let second_at = first_at + Duration::from_millis(20);
    let mut second = one_rank_load(0, 0.1, 1_000.0);
    second.loads[0].dp_rank = 1;
    second.sampled_at = Some(second_at);
    reports.push((workers[1].url().to_owned(), second));
    assert_eq!(
        ages_at(second_at + Duration::from_millis(5), reports),
        [Some(25), Some(5)]
    );
}
