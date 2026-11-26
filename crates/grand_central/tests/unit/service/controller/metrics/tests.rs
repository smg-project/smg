use std::{sync::Arc, time::Duration};

use super::{
    Generations, LoadTotals, Metrics, StoreStats, MEMBERS_OUTSIDE_LIST, POLICY_FAULTS_HELP,
    POLICY_FAULTS_TOTAL, REPLICATION_LAG_SECONDS, REPLICATION_RESTORES_ASKED_TOTAL,
    REPLICATION_SYNCED, SPEC_APPLY_TOTAL, SPEC_GENERATION, START_TIME_SECONDS, STATE,
};
use crate::{
    service::controller::{
        decision_slots::DecideStats,
        readiness::{Sample, State},
    },
    store::{EngineFaults, OutsideLists},
};

fn line(rendered: &str, series: &str, label: &str) -> Option<String> {
    rendered
        .lines()
        .find(|l| l.starts_with(series) && l.contains(label))
        .map(str::to_owned)
}

#[test]
fn the_state_gauge_follows_the_transitions_on_the_controllers_own_recorder() {
    let metrics = Metrics::new();
    let rendered = metrics.render(
        || State::Starting,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert_eq!(
        line(&rendered, STATE, "state=\"starting\"").as_deref(),
        Some("smg_grand_central_state{state=\"starting\"} 1")
    );
    assert_eq!(
        line(&rendered, STATE, "state=\"leader\"").as_deref(),
        Some("smg_grand_central_state{state=\"leader\"} 0")
    );
    assert!(rendered.contains("# HELP smg_grand_central_state "));

    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert_eq!(
        line(&rendered, STATE, "state=\"starting\"").as_deref(),
        Some("smg_grand_central_state{state=\"starting\"} 0")
    );
    assert_eq!(
        line(&rendered, STATE, "state=\"leader\"").as_deref(),
        Some("smg_grand_central_state{state=\"leader\"} 1")
    );

    let other = Metrics::new();
    assert_eq!(
        line(
            &other.render(
                || State::Starting,
                &LoadTotals::default(),
                &DecideStats::default(),
                StoreStats::default
            ),
            STATE,
            "state=\"leader\""
        )
        .as_deref(),
        Some("smg_grand_central_state{state=\"leader\"} 0"),
        "a second controller starts from its own recorder"
    );
}

#[test]
fn two_scrapes_across_a_change_of_role_each_render_one_state() {
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc,
        },
        thread,
        time::Instant,
    };

    let metrics = Arc::new(Metrics::new());
    let stats = LoadTotals::default();
    let inside = Arc::new(AtomicBool::new(false));
    let (sampling, sampled) = mpsc::channel();
    // The first scrape stays inside its sample while the second begins,
    // and the role changes between the two. It runs on a thread the
    // case does not wait for without a bound.
    let first = thread::spawn({
        let (metrics, inside) = (Arc::clone(&metrics), Arc::clone(&inside));
        move || {
            metrics.render(
                || {
                    inside.store(true, Ordering::SeqCst);
                    sampling.send(()).expect("the case listens");
                    thread::sleep(Duration::from_millis(100));
                    inside.store(false, Ordering::SeqCst);
                    State::Leader
                },
                &stats,
                &DecideStats::default(),
                StoreStats::default,
            )
        }
    });
    // A scrape that blocks before its sample fails the case by name.
    sampled
        .recv_timeout(Duration::from_secs(20))
        .expect("the first scrape samples within 20 s");
    let mut overlapped = false;
    let second = metrics.render(
        || {
            overlapped = inside.load(Ordering::SeqCst);
            State::Follower
        },
        &stats,
        &DecideStats::default(),
        StoreStats::default,
    );
    let waited = Instant::now();
    while !first.is_finished() {
        assert!(
            waited.elapsed() < Duration::from_secs(20),
            "the first scrape did not end in 20 s"
        );
        thread::sleep(Duration::from_millis(2));
    }
    let first = first.join().expect("the first scrape ends");
    assert!(!overlapped, "the second scrape sampled inside the first");
    for (rendered, own) in [(&first, State::Leader), (&second, State::Follower)] {
        for state in State::ALL {
            let label = format!("state=\"{}\"", state.as_str());
            let value = u8::from(state == own);
            assert_eq!(
                line(rendered, STATE, &label),
                Some(format!("{STATE}{{{label}}} {value}")),
                "the scrape that sampled {}",
                own.as_str()
            );
        }
    }
}

/// Every load series renders the number the load monitor's one read
/// states for it, each mode and each reason under its own label, and
/// the overload flips render the registry's count of its edges. Every
/// number of the read is another, so a series that took a neighbour's
/// would read wrong.
#[test]
fn the_load_counters_render_the_monitors_counts() {
    let metrics = Metrics::new();
    let rendered = metrics.render(
        || State::Starting,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert!(rendered.contains("smg_grand_central_load_polls_total{mode=\"poll\"} 0"));
    assert!(rendered.contains("smg_grand_central_load_rows_total 0"));
    assert!(rendered.contains("smg_grand_central_overload_flips_total 0"));

    let mut totals = LoadTotals {
        overload_flips: 9,
        ..LoadTotals::default()
    };
    let polled = &mut totals.polls;
    (polled.polls, polled.fallbacks, polled.suppressed) = (3, 5, 2);
    (polled.scrapes, polled.rows, polled.values_dropped) = (11, 7, 4);
    let failed = &mut polled.poll_failures;
    (failed.transport, failed.answer, failed.deadline) = (21, 22, 23);
    (failed.no_load, failed.unsupported, failed.client) = (24, 25, 26);
    let failed = &mut polled.scrape_failures;
    (failed.transport, failed.answer, failed.deadline) = (31, 32, 33);
    (failed.no_load, failed.unsupported, failed.client) = (34, 35, 36);
    let rendered = metrics.render(
        || State::Starting,
        &totals,
        &DecideStats::default(),
        StoreStats::default,
    );
    let mut lines = vec![
        "smg_grand_central_load_polls_total{mode=\"poll\"} 3".to_string(),
        "smg_grand_central_load_polls_total{mode=\"fallback\"} 5".to_string(),
        "smg_grand_central_load_polls_total{mode=\"suppressed\"} 2".to_string(),
        "smg_grand_central_load_scrapes_total 11".to_string(),
        "smg_grand_central_load_rows_total 7".to_string(),
        "smg_grand_central_load_values_dropped_total 4".to_string(),
        "smg_grand_central_overload_flips_total 9".to_string(),
    ];
    let reasons = [
        "transport",
        "answer",
        "deadline",
        "no_load",
        "unsupported",
        "client",
    ];
    for (at, reason) in (1..).zip(reasons) {
        lines.push(format!(
            "smg_grand_central_load_poll_failures_total{{reason=\"{reason}\"}} {}",
            20 + at
        ));
        lines.push(format!(
            "smg_grand_central_load_scrape_failures_total{{reason=\"{reason}\"}} {}",
            30 + at
        ));
    }
    for line in lines {
        assert!(rendered.contains(&line), "{line}\n{rendered}");
    }
}

#[test]
fn the_decide_counters_render_the_deciders_and_the_streams_counts() {
    use std::sync::atomic::Ordering;

    let metrics = Metrics::new();
    let counts = DecideStats::default();
    let rendered = metrics.render(
        || State::Starting,
        &LoadTotals::default(),
        &counts,
        StoreStats::default,
    );
    // Every series stands at zero from the first scrape, by its labels.
    for (series, labels) in [
        (
            "decide_busy_total",
            &DecideStats::BUSY.map(|reason| format!("{{reason=\"{reason}\"}}"))[..],
        ),
        (
            "decide_no_decision_total",
            &[
                "fault",
                "pair_side",
                "prompt_text",
                "decider_gone",
                "lease_ids_spent",
                "document_awaited",
            ]
            .map(|cause| format!("{{cause=\"{cause}\"}}"))[..],
        ),
        ("decide_hashes_cut_total", &[String::new()][..]),
        (
            "decide_answers_dropped_total",
            &["queue_full", "stream_gone"].map(|cause| format!("{{cause=\"{cause}\"}}"))[..],
        ),
        (
            "decide_asked_again_total",
            &["none", "ended"].map(|lease| format!("{{earlier_lease=\"{lease}\"}}"))[..],
        ),
        ("decide_candidates_undialed_total", &[String::new()][..]),
        ("decide_budget_outlasted_total", &[String::new()][..]),
    ] {
        for labels in labels {
            let line = format!("smg_grand_central_{series}{labels} 0");
            assert!(rendered.contains(&line), "{line}\n{rendered}");
        }
    }

    // Each count under its own series and label, none under another's.
    let all = counts.busy.iter().chain([
        &counts.faults,
        &counts.pair_sides,
        &counts.texts,
        &counts.gone,
        &counts.dropped_full,
        &counts.dropped_gone,
        &counts.outlasted,
        &counts.unleased,
        &counts.awaited,
        &counts.again[0],
        &counts.again[1],
    ]);
    for (count, value) in all.zip(1..) {
        count.store(value, Ordering::Relaxed);
    }
    // The two counts kept in parts are read as the sums of their
    // deciders' parts.
    let (first, fourth) = (counts.part(0), counts.part(3));
    first.undialed.store(14, Ordering::Relaxed);
    fourth.undialed.store(17, Ordering::Relaxed);
    first.hashes_cut.store(20, Ordering::Relaxed);
    fourth.hashes_cut.store(22, Ordering::Relaxed);
    let rendered = metrics.render(
        || State::Starting,
        &LoadTotals::default(),
        &counts,
        StoreStats::default,
    );
    for line in [
        "decide_busy_total{reason=\"unspecified\"} 1",
        "decide_busy_total{reason=\"queue_depth\"} 2",
        "decide_busy_total{reason=\"no_credit\"} 3",
        "decide_busy_total{reason=\"budget_gone_on_arrival\"} 4",
        "decide_busy_total{reason=\"not_leader\"} 5",
        "decide_busy_total{reason=\"decode_no_lease\"} 6",
        "decide_busy_total{reason=\"decode_not_chosen\"} 7",
        "decide_busy_total{reason=\"decode_prefill_not_held\"} 8",
        "decide_busy_total{reason=\"decode_stands\"} 9",
        "decide_no_decision_total{cause=\"fault\"} 10",
        "decide_no_decision_total{cause=\"pair_side\"} 11",
        "decide_no_decision_total{cause=\"prompt_text\"} 12",
        "decide_no_decision_total{cause=\"decider_gone\"} 13",
        "decide_answers_dropped_total{cause=\"queue_full\"} 14",
        "decide_answers_dropped_total{cause=\"stream_gone\"} 15",
        "decide_budget_outlasted_total 16",
        "decide_no_decision_total{cause=\"lease_ids_spent\"} 17",
        "decide_no_decision_total{cause=\"document_awaited\"} 18",
        "decide_asked_again_total{earlier_lease=\"none\"} 19",
        "decide_asked_again_total{earlier_lease=\"ended\"} 20",
        "decide_candidates_undialed_total 31",
        "decide_hashes_cut_total 42",
    ] {
        let line = format!("smg_grand_central_{line}");
        assert!(rendered.contains(&line), "{line}\n{rendered}");
    }
}

#[test]
fn the_stores_counts_render_by_list_from_each_scrapes_own_read() {
    let metrics = Metrics::new();
    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert_eq!(
        line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"engines\"").as_deref(),
        Some("smg_grand_central_members_outside_list{list=\"engines\"} 0")
    );
    assert!(rendered.contains("smg_grand_central_lifecycle_faults_total 0"));
    assert!(rendered.contains("# HELP smg_grand_central_members_outside_list "));

    let counts = StoreStats {
        outside: OutsideLists {
            engines: 2,
            zones: 5,
        },
        lifecycle_faults: 4,
        engine_faults: EngineFaults::default(),
    };
    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        || counts,
    );
    assert_eq!(
        line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"engines\"").as_deref(),
        Some("smg_grand_central_members_outside_list{list=\"engines\"} 2")
    );
    assert_eq!(
        line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"zones\"").as_deref(),
        Some("smg_grand_central_members_outside_list{list=\"zones\"} 5")
    );
    assert!(rendered.contains("smg_grand_central_lifecycle_faults_total 4"));

    // The count outside a list falls once a generation names the member.
    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert_eq!(
        line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"zones\"").as_deref(),
        Some("smg_grand_central_members_outside_list{list=\"zones\"} 0")
    );
}

/// The panics the store's engine held render as the engine's
/// family, a series a kind, each at zero from the first scrape. Each is
/// set to its count as its scrape's read answers it, under its own kind.
/// The store's counts never fall. Were a read to answer less, the series
/// would stay where it stood: a scraper reads a fall as a restart.
#[test]
fn the_panics_the_stores_engine_held_render_by_kind_and_never_fall() {
    let metrics = Metrics::new();
    let scraped = |held: [u64; 3]| {
        let [decisions, events, upkeep] = held;
        let counts = StoreStats {
            engine_faults: EngineFaults {
                decisions,
                events,
                upkeep,
            },
            ..StoreStats::default()
        };
        let rendered = metrics.render(
            || State::Leader,
            &LoadTotals::default(),
            &DecideStats::default(),
            || counts,
        );
        assert_eq!(POLICY_FAULTS_TOTAL, crate::engine::POLICY_FAULTS_TOTAL);
        let help = format!("# HELP smg_policy_faults_total {POLICY_FAULTS_HELP}\n");
        assert!(rendered.contains(&help), "{rendered}");
        assert!(rendered.contains("# TYPE smg_policy_faults_total counter"));
        ["decision", "event", "upkeep"].map(|kind| {
            let label = format!("kind=\"{kind}\"");
            let series = line(&rendered, POLICY_FAULTS_TOTAL, &label).expect("the kind's series");
            assert!(series.starts_with(&format!("smg_policy_faults_total{{{label}}} ")));
            let count = series.rsplit(' ').next().expect("the series' value");
            count.parse::<u64>().expect("a whole count")
        })
    };
    assert_eq!(scraped([0, 0, 0]), [0, 0, 0]);
    assert_eq!(scraped([1, 2, 3]), [1, 2, 3]);
    assert_eq!(
        scraped([1, 2, 3]),
        [1, 2, 3],
        "a second scrape adds nothing"
    );
    assert_eq!(scraped([4, 2, 5]), [4, 2, 5]);
    assert_eq!(scraped([0, 0, 0]), [4, 2, 5]);
    // The older series are not this family's: neither moved.
    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert!(rendered.contains("smg_grand_central_lifecycle_faults_total 0"));
    let fault = "smg_grand_central_decide_no_decision_total{cause=\"fault\"} 0";
    assert!(rendered.contains(fault), "{rendered}");
}

#[test]
fn the_start_time_is_set_once_at_construction() {
    let metrics = Metrics::new();
    let rendered = metrics.render(
        || State::Starting,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    let start = rendered
        .lines()
        .find(|l| l.starts_with(START_TIME_SECONDS) && !l.starts_with('#'))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse::<f64>().ok())
        .expect("the start time line");
    assert!(start > 1.7e9, "seconds since the epoch, read {start}");
}

/// A controller with no spec source renders no spec series. Once one is
/// bound, every render reads the generation and the apply counter by
/// result as they stand at that render.
#[test]
fn the_spec_series_render_once_a_source_is_bound() {
    use crate::{
        service::controller::generations::{
            tests::{with_line, LEAST_LOAD, RANDOM},
            Format,
        },
        store::MemoryStore,
    };

    let metrics = Metrics::new();
    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert!(!rendered.contains("smg_policy_spec"), "{rendered}");

    let generations = Arc::new(Generations::new(
        Arc::new(MemoryStore::new()),
        Arc::new(crate::stages::StageRegistry::builtin()),
    ));
    metrics.spec(Arc::clone(&generations));
    let series = |generation: u64, applied: u64, rejected: u64| {
        let rendered = metrics.render(
            || State::Leader,
            &LoadTotals::default(),
            &DecideStats::default(),
            StoreStats::default,
        );
        for line in [
            format!("{SPEC_GENERATION} {generation}"),
            format!("{SPEC_APPLY_TOTAL}{{result=\"applied\"}} {applied}"),
            format!("{SPEC_APPLY_TOTAL}{{result=\"rejected\"}} {rejected}"),
        ] {
            assert!(
                rendered.lines().any(|rendered| rendered == line),
                "no line {line:?} in {rendered}"
            );
        }
    };
    series(0, 0, 0);
    generations
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    series(1, 1, 0);
    let refused = generations
        .apply(with_line(RANDOM, 1), Format::Yaml)
        .expect_err("a changed document under the standing line");
    generations.reject(refused);
    series(1, 1, 1);
    generations
        .apply(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    series(2, 2, 1);
}

#[test]
fn a_scrape_tells_the_standbys_sync_and_its_lag_from_one_sample() {
    let metrics = Metrics::new();
    let scrape = |state, lag, synced| {
        let rendered = metrics.render(
            || Sample {
                state,
                lag,
                synced,
                restores: 0,
            },
            &LoadTotals::default(),
            &DecideStats::default(),
            StoreStats::default,
        );
        (
            line(&rendered, REPLICATION_SYNCED, ""),
            line(&rendered, REPLICATION_LAG_SECONDS, ""),
        )
    };
    let read = |synced: &str, lag: &str| {
        (
            Some(format!("smg_grand_central_replication_synced {synced}")),
            Some(format!("smg_grand_central_replication_lag_seconds {lag}")),
        )
    };
    // A follower a second and a half behind its leader, and synced.
    assert_eq!(
        scrape(State::Follower, Some(Duration::from_millis(1500)), true),
        read("1", "1.5")
    );
    // Its stream fell silent: the lag grows and the sync is gone.
    assert_eq!(
        scrape(State::Syncing, Some(Duration::from_secs(40)), false),
        read("0", "40")
    );
    // A leader follows nobody: both read zero, as before any frame.
    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert_eq!(
        (
            line(&rendered, REPLICATION_SYNCED, ""),
            line(&rendered, REPLICATION_LAG_SECONDS, "")
        ),
        read("0", "0")
    );
    assert!(rendered.contains("# HELP smg_grand_central_replication_synced "));
    assert!(rendered.contains("# HELP smg_grand_central_replication_lag_seconds "));
}

#[test]
fn a_scrape_tells_the_restores_this_replica_asked_of_its_leader() {
    let metrics = Metrics::new();
    let scrape = |restores| {
        let rendered = metrics.render(
            || Sample {
                restores,
                ..Sample::from(State::Syncing)
            },
            &LoadTotals::default(),
            &DecideStats::default(),
            StoreStats::default,
        );
        line(&rendered, REPLICATION_RESTORES_ASKED_TOTAL, "")
    };
    let read = |asked: u64| {
        Some(format!(
            "smg_grand_central_replication_restores_asked_total {asked}"
        ))
    };
    // The series is there before any ask.
    assert_eq!(scrape(0), read(0));
    // A scrape tells the standby's own count: it is set, never added.
    assert_eq!(scrape(2), read(2));
    assert_eq!(scrape(2), read(2));
    assert_eq!(scrape(3), read(3));
    let rendered = metrics.render(
        || State::Leader,
        &LoadTotals::default(),
        &DecideStats::default(),
        StoreStats::default,
    );
    assert!(rendered.contains("# TYPE smg_grand_central_replication_restores_asked_total counter"));
}
