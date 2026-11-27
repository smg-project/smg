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
    document::spec::{
        Assignment, Capacity, FilterSpec, FilterStage, HasRoomParams, PickSpec, PolicySpec,
    },
    engine::{
        decision::Decision,
        framework::{DRAW_FROM, WRITTEN_PAIR_FROM},
        sticky_sessions::SessionVerdict,
        view::{fingerprint, AffinityHolder, AffinityRows, Phase},
    },
};

mod claims;
mod conversion;
mod draws;
mod standing_rows;

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

/// Under a document no stage of which reads a count, a list whose every
/// worker publishes available is written with no read of a worker object.
/// A veto on one worker is read by the next decision of every list that
/// holds the worker, which reads every object again, and the lists stand
/// again once the veto is gone.
#[test]
fn a_wholly_available_list_is_written_with_no_read_of_a_worker_object() {
    let spec = applied("round_robin");
    let clock = DecisionClock::new();
    let workers = fleet(4);
    // Two lists hold the last three workers: a veto on one reaches both.
    let lists = [&workers[..], &workers[1..]];
    let pools = lists.map(|list| PoolRecords::new(list, &spec, &clock));
    // What the object alone knows: a row written with no read of it
    // carries no count.
    workers[2].increment_load();
    let mut rows = Vec::new();
    let mut read = |at: usize| {
        let live = Live::at(spec.engine(), &clock, Instant::now(), None, None)
            .rested_where(spec.engine(), |word| pools[at].stands(lists[at], word));
        let (written, _) = pools[at].write(lists[at], &live, &mut rows);
        let routable = written.iter().filter(|row| row.health.routable());
        let in_flight = written.iter().map(|row| row.in_flight);
        (routable.count(), in_flight.sum::<u32>())
    };
    assert_eq!([read(0), read(1)], [(4, 0), (3, 0)], "no object is read");
    assert!(workers[3].set_overloaded(true));
    assert_eq!([read(0), read(1)], [(3, 1), (2, 1)], "every object is read");
    assert!(workers[3].set_overloaded(false));
    assert_eq!([read(0), read(1)], [(4, 0), (3, 0)], "both lists stand");
    // The word moved and another decision holds the list's claim: this one
    // asks no worker, waits for nothing and reads every object.
    assert!(workers[0].set_overloaded(true));
    assert!(workers[0].set_overloaded(false));
    let claim = pools[0].asking.lock();
    assert_eq!(read(0), (4, 1));
    drop(claim);
    assert_eq!(read(0), (4, 0));
    // A worker that publishes no state answers as a worker type of a host's
    // own does: its list never stands, and every object of it is read.
    let unpublished = <BasicWorkerBuilder>::new(url(4))
        .health_config(no_health_check())
        .build();
    let mixed: [Arc<dyn Worker>; 2] = [Arc::clone(&workers[2]), Arc::new(unpublished)];
    let pool = PoolRecords::new(&mixed, &spec, &clock);
    for _ in 0..2 {
        let live = Live::at(spec.engine(), &clock, Instant::now(), None, None)
            .rested_where(spec.engine(), |word| pool.stands(&mixed, word));
        let (written, _) = pool.write(&mixed, &live, &mut rows);
        assert_eq!(written[0].in_flight, 1);
    }
    // A document a stage of which reads the count reads every object, and
    // so does one whose sessions rule places a new key on the worker with
    // the fewest requests: the rule scores the rows beside every profile.
    let mut keyed = PolicySpec::preset("manual").expect("a built-in preset");
    for profile in keyed.profiles.values_mut() {
        profile.pick = PickSpec::from("round-robin");
    }
    for binding in &mut keyed.bindings {
        let rule = binding.sessions.as_mut().expect("the preset's rule");
        rule.assignment = Assignment::MinLoad;
    }
    for spec in [applied("least_load"), applied_of(keyed)] {
        let pool = PoolRecords::new(&workers, &spec, &clock);
        let live = Live::at(spec.engine(), &clock, Instant::now(), None, None)
            .rested_where(spec.engine(), |word| pool.stands(&workers, word));
        let (written, _) = pool.write(&workers, &live, &mut rows);
        assert_eq!(written[2].in_flight, 1);
    }
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

/// A pool's number is the membership epoch its making moved to: the first
/// pool made on a clock is number one, and no two pools of one clock take
/// one number. A stage that keeps what it learned of a pool keeps it by
/// that number and the epoch. The number is read on pools that list an
/// engine of two ranks: a pool that lists no engine hands a stage an empty
/// list, which states no pool.
#[test]
fn a_pool_is_numbered_by_the_epoch_its_making_moves_to() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let mut rows = Vec::new();
    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
    // The two ranks of one engine.
    let rank = |rank: usize| -> Arc<dyn Worker> {
        Arc::new(
            <BasicWorkerBuilder>::new("grpc://engine:9000".to_owned())
                .dp_config(rank, 2)
                .health_config(no_health_check())
                .build(),
        )
    };
    let ranked = vec![rank(0), rank(1)];

    let first = PoolRecords::new(&ranked, &spec, &clock);
    let (_, facts) = first.write(&ranked, &live, &mut rows);
    assert_eq!((facts.engines.pool(), facts.membership_epoch), (1, 1));

    let second = PoolRecords::new(&ranked, &spec, &clock);
    let (_, facts) = second.write(&ranked, &live, &mut rows);
    assert_eq!((facts.engines.pool(), facts.membership_epoch), (2, 2));
    // The first pool keeps its number under the epoch the second moved.
    let (_, facts) = first.write(&ranked, &live, &mut rows);
    assert_eq!((facts.engines.pool(), facts.membership_epoch), (1, 2));
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
/// id nothing is added. Two lists of one decision each add the ranks of
/// their own engines behind the workers named. A list of workers that each
/// stand alone says it holds no ranks: alone it hands no list back, and
/// beside a list with ranks it is passed over.
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
    // The one operation over the lists of a decision. What the list it is
    // handed held before is not kept.
    let of = |lists: &[&PoolRecords], tried: &[WorkerId]| {
        PoolRecords::tried_engines(lists.iter().copied(), tried, || vec![WorkerId(9)])
            .expect("a list with ranks")
    };
    let first = of(&[&pool], &[id(5)]);
    assert_eq!(first, [id(5), id(3), id(0)], "the tried rank first");
    let second = of(&[&pool], &[id(1)]);
    assert_eq!(second, [id(1), id(4), id(6)], "the second engine's own");
    let single = of(&[&pool], &[id(2)]);
    assert_eq!(single, [id(2)], "a worker alone names itself");
    // One list's part adds its own engines' ranks to what the list holds,
    // and takes nothing out of it.
    let mut whole = vec![WorkerId(9), id(5)];
    pool.add_engines_of(&[id(5), id(1)], &mut whole);
    assert_eq!(
        whole,
        [WorkerId(9), id(5), id(3), id(0), id(1), id(4), id(6)]
    );

    let stranger = WorkerId(77);
    let whole = of(&[&pool], &[id(0), id(2), stranger, id(3), id(4)]);
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
    let whole = of(&[&stated], &[WorkerId(106)]);
    assert_eq!(whole, [WorkerId(106), WorkerId(104), WorkerId(101)]);

    // Two lists of one decision, as a pair hands them: the workers named
    // stand once at the front, each list adds the ranks of its own
    // engines, and the second takes nothing out of what the first wrote.
    let whole = of(&[&stated, &pool], &[WorkerId(106), id(1)]);
    let pair = [
        WorkerId(106),
        id(1),
        WorkerId(104),
        WorkerId(101),
        id(4),
        id(6),
    ];
    assert_eq!(whole, pair, "each list by its own engines");

    // One rank listed twice: no id of the list has a place.
    let twice = vec![rank(a, 0), rank(a, 1), rank(a, 0)];
    let doubled = PoolRecords::new(&twice, &spec, &clock);
    let named = [worker_id(twice[1].as_ref())];
    assert_eq!(of(&[&doubled], &named), named);

    // A list of workers that each stand alone holds no ranks. Alone it
    // hands no list back, and the list to write is not asked for. Beside
    // a list with ranks it is passed over.
    let alone = fleet(3);
    let plain = PoolRecords::new(&alone, &spec, &clock);
    assert!(!plain.has_ranks());
    let named = [worker_id(alone[1].as_ref())];
    let asked = Cell::new(false);
    let none = PoolRecords::tried_engines([&plain], &named, || {
        asked.set(true);
        Vec::new()
    });
    assert_eq!(none, None, "the request's own list stands");
    assert!(!asked.get(), "no list to write is asked for");
    let whole = of(&[&plain, &pool], &[id(5)]);
    assert_eq!(whole, [id(5), id(3), id(0)], "the list with ranks is read");
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
