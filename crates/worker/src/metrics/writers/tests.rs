use metrics_exporter_prometheus::PrometheusBuilder;

use super::{
    series::{WorkerSeries, WorkerSeriesRecorder},
    *,
};
use crate::{
    test_support::{assert_metric, gone, held, render_with_recorder, sorted_series},
    worker::{SchedulerLoadSnapshot, WorkerLoadResponse},
};

/// Core engine gauges share these labels for the snapshot fixtures: a
/// worker that is on neither side of a pair, whose engine reports no role.
const CORE_LABELS: [&str; 4] = [
    "dp_rank=\"2\"",
    "model=\"m\"",
    "worker=\"http://w:1\"",
    "role=\"null\"",
];

#[test]
fn record_engine_load_sets_core_gauges_per_dp_rank() {
    let response = WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 2,
            num_running_reqs: 7,
            num_waiting_reqs: 3,
            token_usage: 0.5,
            gen_throughput: 42.0,
            cache_hit_rate: 0.25,
            ..Default::default()
        }],
        ..Default::default()
    };

    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load(held("http://w:1").0.as_ref(), "m", "null", &response);
    });

    // The exporter renders labels in insertion order, so assert on the
    // metric line's components rather than a fixed label ordering.
    assert_metric(&rendered, "smg_engine_running_requests", &CORE_LABELS, "7");
    assert_metric(&rendered, "smg_engine_waiting_requests", &CORE_LABELS, "3");
    assert_metric(&rendered, "smg_engine_gen_throughput", &CORE_LABELS, "42");
    // PD gauges absent when no disagg section was reported.
    assert!(
        !rendered.contains("smg_engine_pd_"),
        "PD gauges must not appear without a disagg section; rendered:\n{rendered}"
    );
}

/// The capacity behind the usage ratio is a series of its own when the
/// engine reports it, and absent when the record carries the ratio alone.
#[test]
fn record_engine_load_sets_the_kv_capacity_when_the_engine_reports_it() {
    let mut response = WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 2,
            token_usage: 0.5,
            max_total_num_tokens: 262_144,
            ..Default::default()
        }],
        ..Default::default()
    };
    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load(held("http://w:1").0.as_ref(), "m", "null", &response);
    });
    assert_metric(
        &rendered,
        "smg_engine_kv_capacity_tokens",
        &CORE_LABELS,
        "262144",
    );
    assert_metric(&rendered, "smg_engine_token_usage", &CORE_LABELS, "0.5");

    response.loads[0].max_total_num_tokens = 0;
    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load(held("http://w:1").0.as_ref(), "m", "null", &response);
    });
    assert!(
        !rendered.contains("smg_engine_kv_capacity_tokens"),
        "a record without the capacity sets no series; rendered:\n{rendered}"
    );
    assert_metric(&rendered, "smg_engine_token_usage", &CORE_LABELS, "0.5");
}

/// A model's block size is one gauge, set from what the gateway knows, and
/// 0 once it knows none.
#[test]
fn the_kv_block_size_of_a_model_is_a_gauge() {
    let rendered = render_with_recorder(|| {
        Metrics::set_kv_block_size("m", 16);
        Metrics::set_kv_block_size("m", 64);
    });
    assert_metric(
        &rendered,
        "smg_kv_block_size_tokens",
        &["model=\"m\""],
        "64",
    );
    let rendered = render_with_recorder(|| {
        Metrics::set_kv_block_size("m", 64);
        Metrics::set_kv_block_size("m", 0);
    });
    assert_metric(&rendered, "smg_kv_block_size_tokens", &["model=\"m\""], "0");
}

#[test]
fn record_engine_load_sets_pd_gauges_when_disagg_present() {
    let response = WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 0,
            disagg_mode: Some("prefill".to_string()),
            kv_transfer_latency_ms: Some(3.5),
            kv_transfer_speed_gb_s: Some(12.0),
            prefill_queue_reqs: Some(9),
            decode_queue_reqs: Some(4),
            ..Default::default()
        }],
        ..Default::default()
    };

    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load(held("http://w:1").0.as_ref(), "m", "null", &response);
    });

    let pd_labels = ["role=\"prefill\"", "worker=\"http://w:1\"", "dp_rank=\"0\""];
    assert_metric(
        &rendered,
        "smg_engine_pd_kv_transfer_latency_ms",
        &pd_labels,
        "3.5",
    );
    assert_metric(
        &rendered,
        "smg_engine_pd_prefill_queue_reqs",
        &pd_labels,
        "9",
    );
    assert_metric(
        &rendered,
        "smg_engine_pd_decode_queue_reqs",
        &pd_labels,
        "4",
    );
}

/// The `smg_engine_*` set recorded through kept handles renders exactly as
/// it does with a key built per record: the same series with the same
/// values, a capacity or PD series only where a load carried the figure,
/// and a rank's own record (the pushed shape) moving that rank's values
/// alone.
#[test]
fn the_engine_load_set_renders_the_same_through_kept_handles() {
    let report = |running: i32, hit_rate: f64| WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 2,
        loads: vec![
            SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: running,
                num_waiting_reqs: 1,
                token_usage: 0.5,
                gen_throughput: 42.0,
                cache_hit_rate: hit_rate,
                max_total_num_tokens: 262_144,
                disagg_mode: Some("prefill".to_string()),
                kv_transfer_latency_ms: Some(3.5),
                prefill_queue_reqs: Some(9),
                ..Default::default()
            },
            SchedulerLoadSnapshot {
                dp_rank: 1,
                num_running_reqs: running + 1,
                cache_hit_rate: hit_rate,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let records = || {
        let (first, second) = (held("http://w:1").0, held("http://w:2").0);
        Metrics::record_engine_load(first.as_ref(), "m", "null", &report(7, 0.25));
        Metrics::record_engine_rank_load(first.as_ref(), "m", "null", &report(8, 0.75).loads[0]);
        Metrics::record_engine_load(second.as_ref(), "m2", "null", &report(1, 0.5));
    };
    let built = render_with_recorder(|| {
        handles::force_on_this_thread(Some(false));
        records();
        handles::reset();
    });
    let kept = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        records();
        handles::reset();
    });
    assert_eq!(sorted_series(&kept), sorted_series(&built));
    for expected in [
        "smg_engine_running_requests{worker=\"http://w:1\",model=\"m\",dp_rank=\"0\",role=\"null\"} 8",
        "smg_engine_running_requests{worker=\"http://w:1\",model=\"m\",dp_rank=\"1\",role=\"null\"} 8",
        "smg_engine_cache_hit_rate{worker=\"http://w:1\",model=\"m\",dp_rank=\"0\",role=\"null\"} 0.75",
        "smg_engine_cache_hit_rate{worker=\"http://w:1\",model=\"m\",dp_rank=\"1\",role=\"null\"} 0.25",
        "smg_engine_kv_capacity_tokens{worker=\"http://w:1\",model=\"m\",dp_rank=\"0\",role=\"null\"} 262144",
        "smg_engine_pd_kv_transfer_latency_ms{worker=\"http://w:1\",role=\"prefill\",dp_rank=\"0\"} 3.5",
        "smg_engine_pd_prefill_queue_reqs{worker=\"http://w:1\",role=\"prefill\",dp_rank=\"0\"} 9",
        "smg_engine_running_requests{worker=\"http://w:2\",model=\"m2\",dp_rank=\"1\",role=\"null\"} 2",
    ] {
        assert!(
            kept.lines().any(|line| line == expected),
            "missing {expected} in:\n{kept}"
        );
    }
    assert!(
        !kept.contains(
            "smg_engine_kv_capacity_tokens{worker=\"http://w:1\",model=\"m\",dp_rank=\"1\""
        ),
        "a rank whose loads carry no capacity has no capacity series:\n{kept}"
    );
    assert!(
        !kept.contains("smg_engine_pd_kv_transfer_speed_gb_s"),
        "a figure no load carried has no series:\n{kept}"
    );
}

/// A worker's load series carry the role it is registered in, whether or
/// not its engine reports one, so a pool's load splits by role for every
/// engine. The pair series keep the role the engine reports.
#[test]
fn a_load_series_carries_its_workers_registered_role() {
    let load = |disagg_mode: Option<&str>| WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 0,
            num_running_reqs: 5,
            max_total_num_tokens: 1_024,
            disagg_mode: disagg_mode.map(str::to_string),
            prefill_queue_reqs: disagg_mode.map(|_| 3),
            ..Default::default()
        }],
        ..Default::default()
    };
    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load(held("http://d:1").0.as_ref(), "m", "decode", &load(None));
        Metrics::record_engine_load(
            held("http://p:1").0.as_ref(),
            "m",
            "prefill",
            &load(Some("prefill")),
        );
        Metrics::record_engine_load(held("http://r:1").0.as_ref(), "m", "null", &load(None));
    });
    for (worker, role) in [
        ("http://d:1", "decode"),
        ("http://p:1", "prefill"),
        ("http://r:1", "null"),
    ] {
        for (name, value) in [
            ("smg_engine_running_requests", "5"),
            ("smg_engine_kv_capacity_tokens", "1024"),
        ] {
            let expected = format!(
                "{name}{{worker=\"{worker}\",model=\"m\",dp_rank=\"0\",role=\"{role}\"}} {value}"
            );
            assert!(
                rendered.lines().any(|line| line == expected),
                "missing {expected} in:\n{rendered}"
            );
        }
    }
    assert_eq!(
        rendered
            .lines()
            .filter(|line| line.starts_with("smg_engine_running_requests{"))
            .count(),
        3,
        "one series a worker, under one role:\n{rendered}"
    );
    // The engine that reports its role has its pair series under it, and an
    // engine that reports none has no pair series.
    assert_eq!(
        rendered
            .lines()
            .filter(|line| line.starts_with("smg_engine_pd_prefill_queue_reqs{"))
            .collect::<Vec<_>>(),
        ["smg_engine_pd_prefill_queue_reqs{worker=\"http://p:1\",role=\"prefill\",dp_rank=\"0\"} 3"],
        "{rendered}"
    );
}

/// The registered role of each worker type: the two sides of a pair by
/// name, and the engines' own `null` for a worker on neither.
#[test]
fn the_registered_role_is_the_side_of_a_pair_or_null() {
    use crate::worker::WorkerType;
    assert_eq!(registered_engine_role(WorkerType::Prefill), "prefill");
    assert_eq!(registered_engine_role(WorkerType::Decode), "decode");
    assert_eq!(registered_engine_role(WorkerType::Regular), "null");
    assert_eq!(registered_engine_role(WorkerType::Encode), "null");
}

#[test]
fn test_intern_string_returns_same_arc() {
    let s1 = intern_string("test_model");
    let s2 = intern_string("test_model");

    // Should return the same Arc (pointer equality)
    assert!(Arc::ptr_eq(&s1, &s2));
    assert_eq!(&*s1, "test_model");
}

#[test]
fn test_intern_string_different_strings() {
    let s1 = intern_string("model_a");
    let s2 = intern_string("model_b");

    // Different strings should have different Arcs
    assert!(!Arc::ptr_eq(&s1, &s2));
    assert_eq!(&*s1, "model_a");
    assert_eq!(&*s2, "model_b");
}

#[test]
fn test_intern_string_empty() {
    let s1 = intern_string("");
    let s2 = intern_string("");

    assert!(Arc::ptr_eq(&s1, &s2));
    assert_eq!(&*s1, "");
}

/// A rank's label is its decimal text, from the table and past it: a
/// rank's series keep one name whichever way the label is built.
#[test]
fn a_ranks_label_is_its_decimal_text_in_the_table_and_past_it() {
    for rank in [0, 1, 15, 16, 1000, -1] {
        assert_eq!(&*rank_label(rank), rank.to_string());
    }
}

#[test]
fn test_interner_size_grows() {
    let initial_size = interner_size();

    // Intern some unique strings
    let unique = format!("unique_test_string_{initial_size}");
    intern_string(&unique);

    assert!(interner_size() > initial_size);
}

/// The kept worker counters follow the worker set: a removal drops the
/// address's series and the handles kept to them, so a write after the
/// removal lands nowhere, and a re-registration starts the counter over
/// instead of feeding the handle built while the address was retired.
#[test]
fn kept_worker_counters_follow_the_workers_registrations() {
    let workers = Arc::new(WorkerSeries::with_exporter_quantiles());
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    // The rendered `smg_worker_requests_total` of each worker, in order.
    let counts = || {
        let mut rendered = String::new();
        workers.render_into(&mut rendered);
        ["http://w1", "http://w2"].map(|worker| {
            rendered
                .lines()
                .find(|line| {
                    line.starts_with("smg_worker_requests_total{")
                        && line.contains(&format!("worker=\"{worker}\""))
                })
                .map(|line| line.rsplit(' ').next().unwrap_or_default().to_owned())
        })
    };
    let (w1, w2) = (held("http://w1").0, held("http://w2").0);
    handles::force_on_this_thread(Some(true));
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_worker_request(w1.as_ref());
        Metrics::record_worker_request(w1.as_ref());
        Metrics::record_worker_request(w2.as_ref());
        assert_eq!(counts(), [Some("2".to_owned()), Some("1".to_owned())]);

        workers.retire("http://w1", gone("http://w1"));
        handles::worker_set_changed();
        // A send racing the removal: its write is dropped, not kept.
        Metrics::record_worker_request(w1.as_ref());
        assert_eq!(counts(), [None, Some("1".to_owned())]);

        workers.activate("http://w1");
        handles::worker_set_changed();
        Metrics::record_worker_request(w1.as_ref());
        Metrics::record_worker_request(w1.as_ref());
        Metrics::record_worker_request(w2.as_ref());
        assert_eq!(counts(), [Some("2".to_owned()), Some("2".to_owned())]);
    });
    handles::reset();
}

/// With handles kept, the per-request worker gauges set the recorder's
/// own series (the last value written shows), and a retirement drops the
/// kept handle: a set racing the removal feeds nothing, a re-registration
/// gets a fresh handle.
#[test]
fn kept_worker_gauges_set_the_recorders_series() {
    let workers = Arc::new(WorkerSeries::with_exporter_quantiles());
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    workers.activate("http://w1");
    // The rendered value of one per-worker series.
    let value = |series: &str, worker: &str| -> Option<String> {
        let mut rendered = String::new();
        workers.render_into(&mut rendered);
        rendered
            .lines()
            .find(|line| {
                line.starts_with(&format!("{series}{{"))
                    && line.contains(&format!("worker=\"{worker}\""))
            })
            .map(|line| line.rsplit(' ').next().unwrap_or_default().to_owned())
    };
    let w1 = WorkerHandles::new("http://w1");
    handles::force_on_this_thread(Some(true));
    metrics::with_local_recorder(&recorder, || {
        Metrics::set_worker_requests_active(&w1, 3);
        Metrics::set_worker_requests_active(&w1, 2);
        Metrics::set_worker_cb_consecutive_failures(&w1, 1);
        Metrics::set_worker_cb_consecutive_successes(&w1, 4);
        Metrics::set_worker_routing_keys_active(&w1, 5);
        assert_eq!(
            value("smg_worker_requests_active", "http://w1").as_deref(),
            Some("2")
        );
        assert_eq!(
            value("smg_worker_cb_consecutive_failures", "http://w1").as_deref(),
            Some("1")
        );
        assert_eq!(
            value("smg_worker_cb_consecutive_successes", "http://w1").as_deref(),
            Some("4")
        );
        assert_eq!(
            value("smg_worker_routing_keys_active", "http://w1").as_deref(),
            Some("5")
        );

        workers.retire("http://w1", gone("http://w1"));
        handles::worker_set_changed();
        Metrics::set_worker_requests_active(&w1, 9);
        assert_eq!(value("smg_worker_requests_active", "http://w1"), None);

        workers.activate("http://w1");
        handles::worker_set_changed();
        Metrics::set_worker_requests_active(&w1, 1);
        assert_eq!(
            value("smg_worker_requests_active", "http://w1").as_deref(),
            Some("1")
        );
    });
    handles::reset();
}

/// A removed worker a handle still holds keeps its address retired through
/// the upkeep, so its late write lands nowhere; the address is forgotten
/// once the handle is gone.
#[test]
fn a_held_removed_worker_keeps_its_address_retired_through_upkeep() {
    let workers = Arc::new(WorkerSeries::with_exporter_quantiles());
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    let listed = || {
        let mut rendered = String::new();
        workers.render_into(&mut rendered);
        rendered.lines().any(|line| {
            line.starts_with("smg_worker_requests_total{") && line.contains("worker=\"http://w1\"")
        })
    };
    let w1 = held("http://w1").0;
    handles::force_on_this_thread(Some(true));
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_worker_request(w1.as_ref());
        assert!(listed());
        let (worker, witness) = held("http://w1");
        workers.retire("http://w1", witness);
        handles::worker_set_changed();
        workers.run_upkeep();
        // A write from a handle that still holds the removed worker lands nowhere.
        Metrics::record_worker_request(w1.as_ref());
        assert!(!listed());
        assert_eq!(workers.retired_len(), 1);
        drop(worker);
        workers.run_upkeep();
        assert_eq!(workers.retired_len(), 0);
    });
    handles::reset();
}

/// A worker's handles are its own: another worker object at its address
/// builds its handles and does not write through the first one's.
#[test]
fn a_workers_handles_are_not_shared_with_another_worker_at_its_address() {
    let (first, second) = (held("http://w1").0, held("http://w1").0);
    handles::force_on_this_thread(Some(true));
    let under_first = render_with_recorder(|| Metrics::record_worker_request(first.as_ref()));
    let under_second = render_with_recorder(|| Metrics::record_worker_request(second.as_ref()));
    handles::reset();
    for rendered in [under_first, under_second] {
        assert!(
            rendered.contains("smg_worker_requests_total{"),
            "a record that went through another worker's handle: {rendered}"
        );
    }
}

/// The registry's hooks are what move the worker-set generation the kept
/// counters are checked against: a removal, and the registration the store
/// reports a kept no-op handle to. A first registration is one only after
/// the upkeep forgot a retired address.
#[test]
fn a_removal_or_the_registration_after_it_drops_the_kept_worker_counters() {
    let store = WorkerSeries::with_exporter_quantiles();
    assert!(!store.activate("http://first"), "a first registration");
    store.retire("http://w", gone("http://w"));
    assert!(store.activate("http://w"), "the end of a retirement");
    store.retire("http://w", gone("http://w"));
    store.run_upkeep();
    assert!(store.activate("http://first"), "the upkeep forgot one");
    assert!(!store.activate("http://first"), "the registration after it");

    // The hooks, on the process's store. The witness is held, so no other
    // test's upkeep forgets the address before it registers again.
    series::installed_or_default();
    let (_removed, witness) = held("http://gone");
    let worker = WorkerHandles::new("http://w");
    let keeps = || {
        handles::Kept::current(&worker.kept)
            .requests
            .get()
            .is_some()
    };
    let _ = handles::Kept::current(&worker.kept)
        .requests
        .set(metrics::Counter::noop());
    Metrics::remove_worker_metrics("http://gone", witness);
    assert!(!keeps(), "a removal");
    let _ = handles::Kept::current(&worker.kept)
        .requests
        .set(metrics::Counter::noop());
    Metrics::worker_registered("http://gone");
    assert!(!keeps(), "the registration after it");
    // Another test's removal may move the generation at any moment, hence
    // the tries.
    let kept = (0..100).any(|fresh| {
        let _ = handles::Kept::current(&worker.kept)
            .requests
            .set(metrics::Counter::noop());
        Metrics::worker_registered(&format!("http://fresh-{fresh}"));
        keeps()
    });
    assert!(kept, "a first registration");
}

/// The breaker's outcome counter is a per-worker series: through kept
/// handles it counts on the recorder's own series, a retirement drops the
/// kept handle (an outcome racing the removal feeds nothing), and a
/// re-registration starts the counter over instead of feeding the handle
/// built while the address was retired.
#[test]
fn kept_worker_outcome_counters_follow_the_workers_registrations() {
    let workers = Arc::new(WorkerSeries::with_exporter_quantiles());
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    workers.activate("http://w1");
    // The rendered value of one worker's outcome series.
    let value = |worker: &str, outcome: &str| -> Option<String> {
        let mut rendered = String::new();
        workers.render_into(&mut rendered);
        rendered
            .lines()
            .find(|line| {
                line.starts_with("smg_worker_cb_outcomes_total{")
                    && line.contains(&format!("worker=\"{worker}\""))
                    && line.contains(&format!("outcome=\"{outcome}\""))
            })
            .map(|line| line.rsplit(' ').next().unwrap_or_default().to_owned())
    };
    let w1 = WorkerHandles::new("http://w1");
    handles::force_on_this_thread(Some(true));
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_worker_cb_outcome(&w1, "success");
        Metrics::record_worker_cb_outcome(&w1, "success");
        Metrics::record_worker_cb_outcome(&w1, "failure");
        assert_eq!(value("http://w1", "success").as_deref(), Some("2"));
        assert_eq!(value("http://w1", "failure").as_deref(), Some("1"));

        workers.retire("http://w1", gone("http://w1"));
        handles::worker_set_changed();
        Metrics::record_worker_cb_outcome(&w1, "success");
        assert_eq!(value("http://w1", "success"), None);

        workers.activate("http://w1");
        handles::worker_set_changed();
        Metrics::record_worker_cb_outcome(&w1, "success");
        assert_eq!(value("http://w1", "success").as_deref(), Some("1"));
    });
    handles::reset();
}

/// The KV-event set a batch moves, recorded through kept handles, renders
/// exactly as it does with a key built per record: the same series with
/// the same values, each worker's own.
#[test]
fn the_kv_event_set_renders_the_same_through_kept_handles() {
    let records = || {
        for worker in ["grpc://w:1", "grpc://w:2"] {
            Metrics::record_kv_event_blocks(worker, "stored", 3);
            Metrics::record_kv_event_blocks(worker, "stored", 2);
            Metrics::record_kv_event_blocks(worker, "removed", 1);
            Metrics::record_kv_event_batch(worker, "applied");
            Metrics::record_kv_event_batch(worker, "applied");
            Metrics::record_kv_event_batch(worker, "load_only");
            Metrics::record_kv_event_apply(worker, 0.001);
            Metrics::record_kv_event_lag(worker, 0.25);
            Metrics::set_kv_index_blocks(worker, 9);
            Metrics::set_kv_index_blocks(worker, 4);
        }
    };
    let built = render_with_recorder(|| {
        handles::force_on_this_thread(Some(false));
        records();
        handles::reset();
    });
    let kept = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        records();
        handles::reset();
    });
    assert_eq!(sorted_series(&kept), sorted_series(&built));
    for expected in [
        "smg_kv_event_blocks_total{worker=\"grpc://w:1\",op=\"stored\"} 5",
        "smg_kv_event_blocks_total{worker=\"grpc://w:2\",op=\"removed\"} 1",
        "smg_kv_event_batches_total{worker=\"grpc://w:1\",disposition=\"applied\"} 2",
        "smg_kv_event_batches_total{worker=\"grpc://w:2\",disposition=\"load_only\"} 1",
        "smg_kv_event_apply_seconds_count{worker=\"grpc://w:2\"} 1",
        "smg_kv_event_lag_seconds_count{worker=\"grpc://w:1\"} 1",
        "smg_kv_index_blocks{worker=\"grpc://w:1\"} 4",
    ] {
        assert!(
            kept.lines().any(|line| line == expected),
            "missing {expected} in:\n{kept}"
        );
    }
}

/// The KV-event set is per-worker series: through kept handles a batch's
/// records land on the recorder's own series, a retirement drops the kept
/// handles (a batch racing the removal feeds nothing), and a
/// re-registration starts the counter over instead of feeding the handles
/// built while the address was retired.
#[test]
fn kept_kv_event_handles_follow_the_workers_registrations() {
    let workers = Arc::new(WorkerSeries::with_exporter_quantiles());
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    workers.activate("grpc://w1");
    // The rendered value of the worker's one series of a name.
    let value = |name: &str| -> Option<String> {
        let mut rendered = String::new();
        workers.render_into(&mut rendered);
        rendered
            .lines()
            .find(|line| line.starts_with(&format!("{name}{{")))
            .map(|line| line.rsplit(' ').next().unwrap_or_default().to_owned())
    };
    let batch = |blocks: usize| {
        Metrics::record_kv_event_batch("grpc://w1", "applied");
        Metrics::set_kv_index_blocks("grpc://w1", blocks);
    };
    handles::force_on_this_thread(Some(true));
    metrics::with_local_recorder(&recorder, || {
        batch(3);
        batch(5);
        assert_eq!(value("smg_kv_event_batches_total").as_deref(), Some("2"));
        assert_eq!(value("smg_kv_index_blocks").as_deref(), Some("5"));

        workers.retire("grpc://w1", gone("grpc://w1"));
        handles::worker_set_changed();
        batch(7);
        assert_eq!(value("smg_kv_event_batches_total"), None);
        assert_eq!(value("smg_kv_index_blocks"), None);

        workers.activate("grpc://w1");
        handles::worker_set_changed();
        batch(2);
        assert_eq!(value("smg_kv_event_batches_total").as_deref(), Some("1"));
        assert_eq!(value("smg_kv_index_blocks").as_deref(), Some("2"));
    });
    handles::reset();
}
