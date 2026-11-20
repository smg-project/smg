mod engine_connection;
mod running_window_label;

use std::{thread, time::Duration};

use metrics_exporter_prometheus::PrometheusBuilder;

use super::*;
use crate::{
    health::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        overload::OverloadThresholds,
    },
    identity::metadata::DecodingLabels,
    test_support::{no_health_check, BasicWorkerBuilder, RefusedAddress},
    worker::{HealthCheckConfig, WorkerSpec},
};

/// The `smg_worker_cb_state` sample for the test worker's URL in a
/// rendered scrape.
fn cb_state_gauge(rendered: &str) -> Option<f64> {
    rendered
        .lines()
        .find_map(|line| line.strip_prefix("smg_worker_cb_state{worker=\"http://worker:8000\"} "))
        .map(|value| value.parse().unwrap())
}

/// A same-URL replacement is built on a fresh breaker, and neither its
/// build nor its adoption of the live breaker writes the URL's state gauge:
/// an open breaker reads open through both.
#[test]
fn building_and_adopting_leave_a_live_breakers_state_gauge_as_it_stood() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..CircuitBreakerConfig::default()
        };
        let old = BasicWorkerBuilder::new("http://worker:8000")
            .circuit_breaker_config(config.clone())
            .build();
        old.record_circuit_breaker_outcome(false);
        assert_eq!(old.circuit_breaker_state(), CircuitState::Open);
        assert_eq!(cb_state_gauge(&handle.render()), Some(1.0));

        let new = BasicWorkerBuilder::new("http://worker:8000")
            .circuit_breaker_config(config)
            .build();
        assert_eq!(cb_state_gauge(&handle.render()), Some(1.0));

        assert!(new.inherit_shared_state_from(&old));
        assert_eq!(new.circuit_breaker_state(), CircuitState::Open);
        assert_eq!(cb_state_gauge(&handle.render()), Some(1.0));
    });
}

#[test]
fn rediscovered_worker_keeps_its_own_kv_engine_state() {
    let old = BasicWorkerBuilder::new("grpc://worker:8080")
        .kv_engine_id("old-engine")
        .build();
    old.set_kv_engine_id_confirmed(false);
    let new = BasicWorkerBuilder::new("grpc://worker:8080")
        .kv_engine_id("new-engine")
        .build();

    assert!(new.inherit_shared_state_from(&old));
    assert_eq!(new.kv_engine_id().as_deref(), Some("new-engine"));
    assert!(new.kv_engine_id_confirmed());

    // A late probe for the old worker must not alter the new discovery.
    old.refresh_kv_engine_id(Some("late-engine".to_string()));
    old.set_kv_engine_id_confirmed(false);
    assert_eq!(new.kv_engine_id().as_deref(), Some("new-engine"));
    assert!(new.kv_engine_id_confirmed());
}

#[test]
fn test_worker_type_display() {
    assert_eq!(WorkerType::Regular.to_string(), "regular");
    assert_eq!(WorkerType::Prefill.to_string(), "prefill");
    assert_eq!(WorkerType::Decode.to_string(), "decode");
}

#[test]
fn test_worker_type_equality() {
    assert_eq!(WorkerType::Regular, WorkerType::Regular);
    assert_ne!(WorkerType::Regular, WorkerType::Decode);
    assert_eq!(WorkerType::Prefill, WorkerType::Prefill);
}

#[test]
fn test_worker_type_clone() {
    let original = WorkerType::Prefill;
    let cloned = original;
    assert_eq!(original, cloned);
}

#[test]
fn test_health_config_default() {
    use crate::worker::HealthCheckConfig;
    let config = HealthCheckConfig::default();
    assert_eq!(config.timeout_secs, 30);
    assert_eq!(config.check_interval_secs, 60);
    assert_eq!(config.failure_threshold, 3);
    assert_eq!(config.success_threshold, 2);
    assert!(!config.disable_health_check);
}

#[test]
fn test_health_config_custom() {
    use crate::worker::HealthCheckConfig;
    let config = HealthCheckConfig {
        timeout_secs: 10,
        check_interval_secs: 60,
        failure_threshold: 5,
        success_threshold: 3,
        disable_health_check: true,
        drain_settle_secs: 5,
    };
    assert_eq!(config.timeout_secs, 10);
    assert_eq!(config.check_interval_secs, 60);
    assert_eq!(config.failure_threshold, 5);
    assert_eq!(config.success_threshold, 3);
    assert!(config.disable_health_check);
}

#[test]
fn test_basic_worker_creation() {
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    assert_eq!(worker.url(), "http://test:8080");
    assert_eq!(worker.worker_type(), &WorkerType::Regular);
    assert!(worker.is_healthy());
    assert_eq!(worker.load(), 0);
    assert_eq!(worker.processed_requests(), 0);
}

#[test]
fn test_worker_with_labels() {
    let mut labels = HashMap::new();
    labels.insert("env".to_string(), "prod".to_string());
    labels.insert("zone".to_string(), "us-west".to_string());

    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .labels(labels.clone())
        .build();

    assert_eq!(worker.metadata().spec.labels, labels);
}

#[test]
fn test_max_running_requests_parses_from_label() {
    use crate::test_support::BasicWorkerBuilder;
    let mut labels = HashMap::new();
    labels.insert("max_running_requests".to_string(), "256".to_string());
    let worker = BasicWorkerBuilder::new("http://w:9000")
        .labels(labels)
        .build();
    assert_eq!(worker.max_running_requests(), Some(256));
}

#[test]
fn test_max_running_requests_returns_none_when_label_missing() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://w:9000").build();
    assert_eq!(worker.max_running_requests(), None);
}

#[test]
fn test_max_running_requests_returns_none_on_bad_value() {
    use crate::test_support::BasicWorkerBuilder;
    let mut labels = HashMap::new();
    labels.insert(
        "max_running_requests".to_string(),
        "not-a-number".to_string(),
    );
    let worker = BasicWorkerBuilder::new("http://w:9000")
        .labels(labels)
        .build();
    assert_eq!(worker.max_running_requests(), None);
}

#[test]
fn test_max_running_requests_zero_treated_as_none() {
    use crate::test_support::BasicWorkerBuilder;
    let mut labels = HashMap::new();
    labels.insert("max_running_requests".to_string(), "0".to_string());
    let worker = BasicWorkerBuilder::new("http://w:9000")
        .labels(labels)
        .build();
    // Zero is meaningless for capacity; treat as "not reported".
    assert_eq!(worker.max_running_requests(), None);
}

#[test]
fn test_max_running_requests_reads_the_max_num_seqs_spelling() {
    use crate::test_support::BasicWorkerBuilder;
    // Some engines advertise the same scheduler window as `max_num_seqs`;
    // without this such a worker looks like a non-reporter to both
    // fleet capacity and PD admission.
    let mut labels = HashMap::new();
    labels.insert("max_num_seqs".to_string(), "16".to_string());
    let worker = BasicWorkerBuilder::new("grpc://w:30000")
        .labels(labels)
        .build();
    assert_eq!(worker.max_running_requests(), Some(16));
}

#[test]
fn test_max_running_requests_prefers_max_num_seqs_when_both_are_reported() {
    use crate::test_support::BasicWorkerBuilder;
    // The servicer's `running_window` resolves in the same order, so a
    // worker reporting both spellings cannot have its discovery label
    // disagree with its GetLoads report.
    let mut labels = HashMap::new();
    labels.insert("max_running_requests".to_string(), "256".to_string());
    labels.insert("max_num_seqs".to_string(), "16".to_string());
    let worker = BasicWorkerBuilder::new("grpc://w:30000")
        .labels(labels)
        .build();
    assert_eq!(worker.max_running_requests(), Some(16));
}

#[test]
fn test_pd_admission_claims_are_atomic_and_bounded_by_the_window() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("grpc://w:30000").build();

    assert!(worker.try_admit_pd(2, 3));
    assert_eq!(worker.pd_admitted(), 2);
    // All-or-nothing: a refusal must leave the claim untouched.
    assert!(!worker.try_admit_pd(2, 3));
    assert_eq!(worker.pd_admitted(), 2);
    assert!(worker.try_admit_pd(1, 3));
    assert_eq!(worker.pd_admitted(), 3);

    worker.release_pd(3);
    assert_eq!(worker.pd_admitted(), 0);
    // Releases saturate rather than wrapping to usize::MAX.
    worker.release_pd(1);
    assert_eq!(worker.pd_admitted(), 0);
}

#[test]
fn test_pd_admission_never_overshoots_the_window_under_real_parallelism() {
    use std::{sync::Arc, thread};

    use crate::test_support::BasicWorkerBuilder;
    // The whole point of claiming rather than reading: threads racing for
    // the last rooms must not all win.
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://w:30000").build());
    let window = 8;
    let threads: Vec<_> = (0..32)
        .map(|_| {
            let worker = Arc::clone(&worker);
            thread::spawn(move || worker.try_admit_pd(1, window))
        })
        .collect();

    let admitted = threads
        .into_iter()
        .map(|handle| handle.join().expect("claim thread"))
        .filter(|claimed| *claimed)
        .count();

    assert_eq!(admitted, window, "exactly the window's worth may claim");
    assert_eq!(worker.pd_admitted(), window);
}

#[test]
fn test_pd_admission_queue_is_arrival_ordered_and_closes_over_departures() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("grpc://w:30000").build();

    let (first, second, third) = (
        worker.pd_enqueue(1),
        worker.pd_enqueue(1),
        worker.pd_enqueue(1),
    );
    assert!(worker.pd_may_claim(first), "the oldest place is next");
    assert!(!worker.pd_may_claim(second));
    assert!(!worker.pd_may_claim(third));
    assert!(
        worker.pd_wake(second).is_some(),
        "a queued place has its notifier"
    );

    // A dispatch that stops waiting from the middle leaves no hole.
    worker.pd_dequeue(second);
    assert!(worker.pd_may_claim(first));
    assert!(
        worker.pd_wake(second).is_none(),
        "a place given back has none"
    );
    worker.pd_dequeue(first);
    assert!(worker.pd_may_claim(third), "the places behind move up");
    assert!(
        !worker.pd_may_claim(second),
        "a place given back never comes back"
    );

    // Leaving twice, or with a place never handed out, changes nothing.
    worker.pd_dequeue(second);
    worker.pd_dequeue(u64::MAX);
    assert!(worker.pd_may_claim(third));

    worker.pd_dequeue(third);
    assert!(
        worker.pd_may_claim(worker.pd_enqueue(1)),
        "with nobody waiting a newcomer is next at once"
    );
}

#[test]
fn test_pd_admission_queue_lets_a_narrower_place_pass_a_wider_older_one() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("grpc://w:30000").build();

    // Arrival order, with the rooms each needs: 4, 1, 2, 1, 3.
    let wide = worker.pd_enqueue(4);
    let narrow = worker.pd_enqueue(1);
    let middle = worker.pd_enqueue(2);
    let narrow_too = worker.pd_enqueue(1);
    let wider = worker.pd_enqueue(3);
    assert!(worker.pd_may_claim(wide), "the front always may");
    assert!(
        worker.pd_may_claim(narrow),
        "it needs fewer rooms than everyone ahead of it"
    );
    assert!(
        !worker.pd_may_claim(middle),
        "the 1-room place ahead goes first"
    );
    assert!(
        !worker.pd_may_claim(narrow_too),
        "its equal ahead goes first"
    );
    assert!(!worker.pd_may_claim(wider));

    // The narrow one is served: the 2-room place now needs fewer than the
    // front, and the other 1-room place fewer than both ahead of it.
    worker.pd_dequeue(narrow);
    assert!(worker.pd_may_claim(wide));
    assert!(worker.pd_may_claim(middle));
    assert!(worker.pd_may_claim(narrow_too));
    assert!(!worker.pd_may_claim(wider), "a 1-room place is still ahead");

    // The wide front leaves (shed, or its rooms freed together).
    worker.pd_dequeue(wide);
    assert!(worker.pd_may_claim(middle));
    assert!(worker.pd_may_claim(narrow_too));
    assert!(!worker.pd_may_claim(wider));

    worker.pd_dequeue(middle);
    worker.pd_dequeue(narrow_too);
    assert!(worker.pd_may_claim(wider), "last in line, and alone");
}

#[test]
fn test_is_realtime_capable_true_from_label() {
    use crate::test_support::BasicWorkerBuilder;
    let mut labels = HashMap::new();
    labels.insert("realtime".to_string(), "true".to_string());
    let worker = BasicWorkerBuilder::new("http://w:9000")
        .labels(labels)
        .build();
    assert!(worker.is_realtime_capable());
}

#[test]
fn test_is_realtime_capable_false_when_label_missing() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://w:9000").build();
    assert!(!worker.is_realtime_capable());
}

#[test]
fn test_is_realtime_capable_false_on_non_true_value() {
    use crate::test_support::BasicWorkerBuilder;
    let mut labels = HashMap::new();
    labels.insert("realtime".to_string(), "1".to_string());
    let worker = BasicWorkerBuilder::new("http://w:9000")
        .labels(labels)
        .build();
    // Only the exact string "true" enables realtime.
    assert!(!worker.is_realtime_capable());
}

#[test]
fn test_worker_with_health_config() {
    use crate::worker::HealthCheckConfig;
    let custom_config = HealthCheckConfig {
        timeout_secs: 15,
        check_interval_secs: 45,
        failure_threshold: 4,
        success_threshold: 2,
        disable_health_check: false,
        drain_settle_secs: 5,
    };

    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .health_config(custom_config.clone())
        .health_endpoint("/custom-health")
        .build();

    assert_eq!(worker.metadata().health_config.timeout_secs, 15);
    assert_eq!(worker.metadata().health_config.check_interval_secs, 45);
    assert_eq!(worker.metadata().health_endpoint, "/custom-health");
}

#[test]
fn test_worker_url() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://worker1:8080")
        .worker_type(WorkerType::Regular)
        .build();
    assert_eq!(worker.url(), "http://worker1:8080");
}

#[test]
fn test_worker_type_getter() {
    use crate::test_support::BasicWorkerBuilder;
    let regular = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .build();
    assert_eq!(regular.worker_type(), &WorkerType::Regular);

    let prefill = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Prefill)
        .bootstrap_port(Some(9090))
        .build();
    assert_eq!(prefill.worker_type(), &WorkerType::Prefill);
    assert_eq!(prefill.bootstrap_port(), Some(9090));

    let decode = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Decode)
        .build();
    assert_eq!(decode.worker_type(), &WorkerType::Decode);
}

#[test]
fn test_health_status() {
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    assert!(worker.is_healthy());
    assert_eq!(worker.status(), WorkerStatus::Ready);

    worker.set_status(WorkerStatus::NotReady);
    assert!(!worker.is_healthy());
    assert_eq!(worker.status(), WorkerStatus::NotReady);

    worker.set_status(WorkerStatus::Ready);
    assert!(worker.is_healthy());
    assert_eq!(worker.status(), WorkerStatus::Ready);
}

#[test]
fn test_pending_worker_not_routable() {
    // Default health config: health checks enabled → starts Pending
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .build();

    assert_eq!(worker.status(), WorkerStatus::Pending);
    assert!(!worker.is_healthy()); // Pending is not routable
    assert!(!worker.is_available()); // Pending is not available
}

#[test]
fn test_load_counter_operations() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .build();

    assert_eq!(worker.load(), 0);

    worker.increment_load();
    assert_eq!(worker.load(), 1);

    worker.increment_load();
    worker.increment_load();
    assert_eq!(worker.load(), 3);

    worker.decrement_load();
    assert_eq!(worker.load(), 2);

    worker.decrement_load();
    worker.decrement_load();
    assert_eq!(worker.load(), 0);

    worker.decrement_load();
    assert_eq!(worker.load(), 0);
}

#[test]
fn test_processed_counter() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .build();

    assert_eq!(worker.processed_requests(), 0);

    for i in 1..=100 {
        worker.increment_processed();
        assert_eq!(worker.processed_requests(), i);
    }
}

#[tokio::test]
async fn test_concurrent_load_increments() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = Arc::new(
        BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build(),
    );

    let mut handles = vec![];

    for _ in 0..100 {
        let worker_clone = Arc::clone(&worker);
        #[expect(
            clippy::disallowed_methods,
            reason = "Test helper: short-lived tasks joined before test ends"
        )]
        let handle = tokio::spawn(async move {
            worker_clone.increment_load();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    assert_eq!(worker.load(), 100);
}

#[tokio::test]
async fn test_concurrent_load_decrements() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = Arc::new(
        BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build(),
    );

    for _ in 0..100 {
        worker.increment_load();
    }
    assert_eq!(worker.load(), 100);

    let mut handles = vec![];

    for _ in 0..100 {
        let worker_clone = Arc::clone(&worker);
        #[expect(
            clippy::disallowed_methods,
            reason = "Test helper: short-lived tasks joined before test ends"
        )]
        let handle = tokio::spawn(async move {
            worker_clone.decrement_load();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn test_concurrent_health_updates() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = Arc::new(
        BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build(),
    );

    let mut handles = vec![];

    for i in 0..100 {
        let worker_clone = Arc::clone(&worker);
        #[expect(
            clippy::disallowed_methods,
            reason = "Test helper: short-lived tasks joined before test ends"
        )]
        let handle = tokio::spawn(async move {
            let status = if i % 2 == 0 {
                WorkerStatus::Ready
            } else {
                WorkerStatus::NotReady
            };
            worker_clone.set_status(status);
            time::sleep(Duration::from_micros(10)).await;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

#[test]
fn test_create_regular_worker() {
    use crate::test_support::BasicWorkerBuilder;
    let worker: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://regular:8080")
            .worker_type(WorkerType::Regular)
            .build(),
    );
    assert_eq!(worker.url(), "http://regular:8080");
    assert_eq!(worker.worker_type(), &WorkerType::Regular);
}

#[test]
fn test_create_prefill_worker() {
    use crate::test_support::BasicWorkerBuilder;
    let worker1: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://prefill:8080")
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(9090))
            .build(),
    );
    assert_eq!(worker1.url(), "http://prefill:8080");
    assert_eq!(worker1.worker_type(), &WorkerType::Prefill);
    assert_eq!(worker1.bootstrap_port(), Some(9090));

    let worker2: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://prefill:8080")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    assert_eq!(worker2.worker_type(), &WorkerType::Prefill);
    assert_eq!(worker2.bootstrap_port(), None);
}

#[test]
fn test_create_decode_worker() {
    use crate::test_support::BasicWorkerBuilder;
    let worker: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://decode:8080")
            .worker_type(WorkerType::Decode)
            .build(),
    );
    assert_eq!(worker.url(), "http://decode:8080");
    assert_eq!(worker.worker_type(), &WorkerType::Decode);
}

#[tokio::test]
async fn test_check_health_async() {
    use crate::test_support::BasicWorkerBuilder;
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .build();

    // Health check should fail since there's no actual server
    let result = worker.check_health_async().await;
    assert!(result.is_err());
}

/// The load counter's increment, a million times: the count is exact,
/// and the rate is printed. The rate is machine-bound (a loaded or slow
/// machine reaches a fraction of a quiet one's), so the unit suite
/// asserts no absolute floor; `SMG_TEST_LOAD_COUNTER_FLOOR=<ops/s>` turns
/// the printed measurement into an assertion for a run on a known machine.
#[test]
#[expect(clippy::print_stderr)]
fn test_load_counter_performance() {
    use std::time::Instant;

    use crate::test_support::BasicWorkerBuilder;

    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .build();
    let iterations = 1_000_000;

    let start = Instant::now();
    for _ in 0..iterations {
        worker.increment_load();
    }
    let duration = start.elapsed();
    assert_eq!(worker.load(), iterations);

    let ops_per_sec = iterations as f64 / duration.as_secs_f64();
    eprintln!("Load counter operations per second: {ops_per_sec:.0}");

    // An asked-for floor is an assertion; an unreadable or malformed one
    // is an error, not a silently skipped check: only an absent variable
    // means no floor was asked for.
    match std::env::var("SMG_TEST_LOAD_COUNTER_FLOOR") {
        Err(std::env::VarError::NotPresent) => {}
        Err(error) => panic!("SMG_TEST_LOAD_COUNTER_FLOOR is unreadable: {error}"),
        Ok(floor) => {
            let floor: f64 = match floor.parse::<f64>() {
                Ok(floor) if floor.is_finite() => floor,
                _ => panic!("SMG_TEST_LOAD_COUNTER_FLOOR {floor:?} is not a finite number"),
            };
            assert!(
                ops_per_sec > floor,
                "load counter: {ops_per_sec:.0} ops/s, under the asked-for floor of {floor:.0}"
            );
        }
    }
}

#[test]
fn test_dp_aware_worker_creation() {
    let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
        .dp_config(2, 4)
        .worker_type(WorkerType::Regular)
        .build();

    assert_eq!(dp_worker.url(), "http://worker1:8080@2");
    assert_eq!(dp_worker.base_url(), "http://worker1:8080");
    assert!(dp_worker.is_dp_aware());
    assert_eq!(dp_worker.dp_rank(), Some(2));
    assert_eq!(dp_worker.dp_size(), Some(4));
    assert_eq!(dp_worker.worker_type(), &WorkerType::Regular);
}

#[test]
fn test_dp_aware_worker_creation_prefill() {
    let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
        .dp_config(1, 2)
        .worker_type(WorkerType::Prefill)
        .build();

    assert_eq!(dp_worker.url(), "http://worker1:8080@1");
    assert!(dp_worker.is_dp_aware());
    assert_eq!(dp_worker.worker_type(), &WorkerType::Prefill);
}

#[test]
fn test_dp_aware_worker_creation_decode() {
    let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
        .dp_config(0, 4)
        .worker_type(WorkerType::Decode)
        .build();

    assert_eq!(dp_worker.url(), "http://worker1:8080@0");
    assert!(dp_worker.is_dp_aware());
    assert_eq!(dp_worker.worker_type(), &WorkerType::Decode);
}

#[test]
fn test_dp_aware_prepare_request() {
    let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
        .dp_config(3, 8)
        .worker_type(WorkerType::Regular)
        .build();

    let original_req = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 100
    });

    let prepared_req = dp_worker.prepare_request(original_req).unwrap();

    assert_eq!(prepared_req["prompt"], "Hello");
    assert_eq!(prepared_req["max_tokens"], 100);
    assert_eq!(prepared_req["data_parallel_rank"], 3);
}

#[test]
fn test_dp_aware_prepare_request_invalid() {
    let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
        .dp_config(0, 4)
        .worker_type(WorkerType::Regular)
        .build();

    // Non-object JSON should fail
    let invalid_req = serde_json::json!("not an object");
    let result = dp_worker.prepare_request(invalid_req);

    assert!(result.is_err());
    match result.unwrap_err() {
        WorkerError::InvalidConfiguration { message } => {
            assert!(message.contains("JSON object"));
        }
        _ => panic!("Expected InvalidConfiguration error"),
    }
}

#[test]
fn test_dp_aware_endpoint_url() {
    let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
        .dp_config(1, 4)
        .worker_type(WorkerType::Regular)
        .build();

    assert_eq!(
        dp_worker.endpoint_url("/generate"),
        "http://worker1:8080/generate"
    );
    assert_eq!(
        dp_worker.endpoint_url("/health"),
        "http://worker1:8080/health"
    );
}

#[test]
fn test_dp_aware_worker_delegated_methods() {
    let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
        .dp_config(0, 2)
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    assert!(dp_worker.is_healthy());
    dp_worker.set_status(WorkerStatus::NotReady);
    assert!(!dp_worker.is_healthy());

    assert_eq!(dp_worker.load(), 0);
    dp_worker.increment_load();
    assert_eq!(dp_worker.load(), 1);
    dp_worker.decrement_load();
    assert_eq!(dp_worker.load(), 0);

    assert_eq!(dp_worker.processed_requests(), 0);
    dp_worker.increment_processed();
    assert_eq!(dp_worker.processed_requests(), 1);
}

#[test]
fn test_worker_circuit_breaker() {
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    assert!(worker.is_available());
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);

    worker.record_outcome(503);
    worker.record_outcome(503);

    assert!(worker.is_available());

    worker.record_outcome(503);
    worker.record_outcome(503);
    worker.record_outcome(503);

    assert!(!worker.is_available());
    assert!(worker.is_healthy());
    assert!(!worker.circuit_breaker_can_execute());
}

#[test]
fn test_capacity_pushback_never_trips_circuit_breaker() {
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    // A storm of 429 capacity pushback must not open the breaker...
    for _ in 0..20 {
        worker.record_outcome(429);
    }
    assert!(worker.is_available());
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);

    // ...while genuine failures still do.
    for _ in 0..5 {
        worker.record_outcome(500);
    }
    assert!(!worker.is_available());
}

#[test]
fn test_capacity_pushback_does_not_close_half_open_breaker() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        success_threshold: 1,
        timeout_duration: Duration::from_millis(50),
        window_duration: Duration::from_secs(60),
    };
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .circuit_breaker_config(config)
        .health_config(no_health_check())
        .build();

    worker.record_outcome(500);
    worker.record_outcome(500);
    // The trip as stored: on a loaded machine the 50 ms timeout can pass
    // before this line, and `is_available` would already start the trial.
    assert_eq!(
        worker.circuit_breaker.load().peek_state(),
        CircuitState::Open
    );
    thread::sleep(Duration::from_millis(80));
    assert!(worker.is_available());
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);

    // 429 records no sample: the breaker must stay half-open, not close.
    worker.record_outcome(429);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);

    worker.record_outcome(200);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

#[test]
fn engine_rejections_do_not_open_the_circuit_breaker() {
    // The case never waits for the breaker's timeout, so the timeout is one
    // it cannot outlive: a slow host reads the open state all the same.
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        success_threshold: 1,
        timeout_duration: Duration::from_secs(60),
        window_duration: Duration::from_secs(60),
    };
    let worker = BasicWorkerBuilder::new("http://test:8080")
        .worker_type(WorkerType::Regular)
        .circuit_breaker_config(config)
        .health_config(no_health_check())
        .build();

    // A 400 is the client's fault: any number of them leaves the breaker closed.
    for _ in 0..10 {
        worker.record_outcome(400);
    }
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    assert!(worker.is_available());

    worker.record_outcome(500);
    worker.record_outcome(500);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    assert!(!worker.is_available());
}

#[test]
fn test_worker_with_circuit_breaker_config() {
    let build = |timeout_duration| {
        BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration,
                window_duration: Duration::from_secs(60),
            })
            .health_config(no_health_check())
            .build()
    };

    // The trip, on a breaker whose timeout the case cannot outlive: a slow
    // host between two lines reads the open state all the same.
    let worker = build(Duration::from_secs(60));
    worker.record_outcome(503);
    assert!(worker.is_available());
    worker.record_outcome(503);
    assert!(!worker.is_available());

    // The timeout's move to half-open, on a breaker with a short one, read
    // only after the sleep, however long the host made it.
    let worker = build(Duration::from_millis(100));
    worker.record_outcome(503);
    worker.record_outcome(503);
    thread::sleep(Duration::from_millis(150));

    assert!(worker.is_available());
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);

    worker.record_outcome(200);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

#[test]
fn test_dp_aware_worker_circuit_breaker() {
    let dp_worker = BasicWorkerBuilder::new("http://worker:8080")
        .dp_config(0, 2)
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    assert!(dp_worker.is_available());

    for _ in 0..5 {
        dp_worker.record_outcome(503);
    }

    assert!(!dp_worker.is_available());
    assert_eq!(dp_worker.circuit_breaker_state(), CircuitState::Open);
}

#[tokio::test]
async fn test_mixed_worker_types() {
    let hc = no_health_check();
    let regular: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://regular:8080")
            .worker_type(WorkerType::Regular)
            .health_config(hc.clone())
            .build(),
    );
    let prefill: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://prefill:8080")
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(9090))
            .health_config(hc.clone())
            .build(),
    );
    let decode: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://decode:8080")
            .worker_type(WorkerType::Decode)
            .health_config(hc.clone())
            .build(),
    );
    let dp_aware_regular: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://dp:8080")
            .dp_config(0, 2)
            .worker_type(WorkerType::Regular)
            .health_config(hc.clone())
            .api_key("test_api_key")
            .build(),
    );
    let dp_aware_prefill: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://dp-prefill:8080")
            .dp_config(1, 2)
            .worker_type(WorkerType::Prefill)
            .health_config(hc.clone())
            .api_key("test_api_key")
            .build(),
    );
    let dp_aware_decode: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://dp-decode:8080")
            .dp_config(0, 4)
            .worker_type(WorkerType::Decode)
            .health_config(hc.clone())
            .api_key("test_api_key")
            .build(),
    );

    let workers: Vec<Box<dyn Worker>> = vec![
        regular,
        prefill,
        decode,
        dp_aware_regular,
        dp_aware_prefill,
        dp_aware_decode,
    ];

    for worker in &workers {
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 0);
    }

    assert!(!workers[0].is_dp_aware());
    assert!(!workers[1].is_dp_aware());
    assert!(!workers[2].is_dp_aware());
    assert!(workers[3].is_dp_aware());
    assert!(workers[4].is_dp_aware());
    assert!(workers[5].is_dp_aware());

    assert_eq!(workers[0].worker_type(), &WorkerType::Regular);
    assert_eq!(workers[1].worker_type(), &WorkerType::Prefill);
    assert_eq!(workers[2].worker_type(), &WorkerType::Decode);
    assert_eq!(workers[3].worker_type(), &WorkerType::Regular);
    assert_eq!(workers[4].worker_type(), &WorkerType::Prefill);
    assert_eq!(workers[5].worker_type(), &WorkerType::Decode);
}

// === Phase 1.3: WorkerMetadata model methods tests ===

#[test]
fn test_worker_metadata_empty_models_accepts_all() {
    let metadata = WorkerMetadata {
        pd_pairing: PdPairing::derive(&WorkerSpec::new("http://test:8080")),
        spec: Arc::new(WorkerSpec::new("http://test:8080")),
        redacted_url: "http://test:8080".to_string(),
        decoding: DecodingLabels::default(),
        health_config: HealthCheckConfig::default(),
        health_endpoint: "/health".to_string(),
        zmq_startup_timeout: Duration::from_secs(600),
        zmq_request_timeout: None,
        overload: OverloadThresholds::default(),
        http2: false,
        discovered_labels: Arc::default(),
    };

    // Empty models list should accept any model
    assert!(metadata.supports_model("any-model"));
    assert!(metadata.supports_model("gpt-4"));
    assert!(metadata.supports_model("llama-3.1"));
}

#[test]
fn test_worker_metadata_find_model() {
    use super::ModelCard;

    let model1 = ModelCard::new("meta-llama/Llama-3.1-8B")
        .with_alias("llama-3.1-8b")
        .with_alias("llama3.1");
    let model2 = ModelCard::new("gpt-4o");

    let mut spec = WorkerSpec::new("http://test:8080");
    spec.models = WorkerModels::from(vec![model1, model2]);
    let metadata = WorkerMetadata {
        pd_pairing: PdPairing::derive(&spec),
        spec: Arc::new(spec),
        redacted_url: "http://test:8080".to_string(),
        decoding: DecodingLabels::default(),
        health_config: HealthCheckConfig::default(),
        health_endpoint: "/health".to_string(),
        zmq_startup_timeout: Duration::from_secs(600),
        zmq_request_timeout: None,
        overload: OverloadThresholds::default(),
        http2: false,
        discovered_labels: Arc::default(),
    };

    // Find by primary ID
    assert!(metadata.find_model("meta-llama/Llama-3.1-8B").is_some());
    assert!(metadata.find_model("gpt-4o").is_some());

    // Find by alias
    assert!(metadata.find_model("llama-3.1-8b").is_some());
    assert!(metadata.find_model("llama3.1").is_some());

    // Not found
    assert!(metadata.find_model("unknown-model").is_none());
}

#[test]
fn test_lazy_discovered_models_override_wildcard() {
    let worker = BasicWorkerBuilder::new("http://test:8080").build();

    // Wildcard worker starts with no models listed, but accepts any model
    assert!(worker.models().is_empty());
    assert!(worker.supports_model("gpt-4o-mini")); // wildcard accepts anything

    // Simulate lazy discovery via set_models
    let discovered = vec![
        ModelCard::new("gpt-4o-mini"),
        ModelCard::new("text-embedding-3-small"),
    ];
    worker.set_models(discovered);

    let ids: Vec<String> = worker.models().into_iter().map(|m| m.id).collect();
    assert_eq!(ids, vec!["gpt-4o-mini", "text-embedding-3-small"]);
    assert!(worker.supports_model("gpt-4o-mini"));
    assert!(worker.supports_model("text-embedding-3-small"));
    assert!(!worker.supports_model("non-existent-model"));
}

#[test]
fn replacement_adopts_the_backend_client_only_on_a_matching_transport() {
    // The cell object itself is the connection: sharing it is what keeps a
    // replaced ZMQ worker talking to the engine that already handshook.
    let build = |mode: ConnectionMode| {
        BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(mode)
            .health_config(no_health_check())
            .build()
    };

    let old = build(ConnectionMode::Zmq);
    let new = build(ConnectionMode::Zmq);
    assert!(new.inherit_shared_state_from(&old));
    assert!(Arc::ptr_eq(
        &new.engine_connection.load_full(),
        &old.engine_connection.load_full()
    ));

    // A transport change means a different wire protocol — no adoption.
    let retyped = build(ConnectionMode::Http);
    assert!(retyped.inherit_shared_state_from(&old));
    assert!(!Arc::ptr_eq(
        &retyped.engine_connection.load_full(),
        &old.engine_connection.load_full()
    ));
}

#[test]
fn replacement_adopts_the_backend_client_only_at_the_same_base_address() {
    // A host may key a worker by a URL of its own and state the engine's
    // address beside it. A replacement under that key whose address moved
    // dials another engine: the old connection is not its own.
    let build = |engine: &str| {
        let mut spec = WorkerSpec::new("grpc://worker:1");
        spec.connection_mode = ConnectionMode::Grpc;
        spec.dp_base_url = Some(engine.to_string());
        BasicWorkerBuilder::from_spec(spec)
            .health_config(no_health_check())
            .build()
    };

    let old = build("grpc://engine-a:30000");
    let same = build("grpc://engine-a:30000");
    assert!(same.inherit_shared_state_from(&old));
    assert!(Arc::ptr_eq(
        &same.engine_connection.load_full(),
        &old.engine_connection.load_full()
    ));

    let moved = build("grpc://engine-b:30000");
    assert!(moved.inherit_shared_state_from(&old));
    assert!(!Arc::ptr_eq(
        &moved.engine_connection.load_full(),
        &old.engine_connection.load_full()
    ));
}

#[test]
fn output_silence_is_read_only_while_a_stream_that_produced_output_is_open() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://decode:30000").build());
    // Nothing in flight it has shown output on: nothing to judge.
    assert!(worker.output_silence().is_none());
    worker.runtime_state().unwrap().note_output_started();
    worker.runtime_state().unwrap().note_token_progress();
    thread::sleep(Duration::from_millis(20));
    let silence = worker.output_silence().expect("a producing stream is open");
    assert!(silence >= Duration::from_millis(20), "{silence:?}");
    // A token: the silence restarts.
    worker.runtime_state().unwrap().note_token_progress();
    assert!(worker.output_silence().expect("still open") < silence);
    // The stream ended: nothing to judge again, however quiet the worker.
    worker.runtime_state().unwrap().note_output_ended();
    assert!(worker.output_silence().is_none());
    // The count never goes negative.
    worker.runtime_state().unwrap().note_output_ended();
    assert!(worker.output_silence().is_none());
}
/// The `smg_worker_state` samples of the test worker in a rendered
/// scrape: `(state, reason)` and the value.
fn state_series(rendered: &str) -> Vec<((String, String), f64)> {
    rendered
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("smg_worker_state{worker=\"http://worker:8000\",")?;
            let (labels, value) = rest.split_once("} ")?;
            let field = |name: &str| {
                let after = labels.split_once(&format!("{name}=\""))?.1;
                Some(after.split_once('"')?.0.to_owned())
            };
            Some(((field("state")?, field("reason")?), value.parse().ok()?))
        })
        .collect()
}

/// The one `smg_worker_state` series at 1, as `state/reason`; every
/// other series of the worker reads 0.
fn current_state(rendered: &str) -> String {
    let series = state_series(rendered);
    let mut on = series
        .iter()
        .filter(|(_, value)| *value == 1.0)
        .map(|((state, reason), _)| format!("{state}/{reason}"));
    let current = on.next().expect("one series at 1");
    assert_eq!(on.next(), None, "two series at 1: {series:?}");
    assert!(
        series
            .iter()
            .all(|(_, value)| *value == 0.0 || *value == 1.0),
        "{series:?}"
    );
    current
}

/// A worker's routing state is one series at 1 with the reason it is
/// out for, in the order the routing decision reads the facts, and the
/// series it leaves reads 0: the status, the overload veto, the liveness
/// veto, the breaker (a trip, and its timeout's move to half-open on the
/// next read of the breaker).
#[test]
fn a_workers_routing_state_is_one_series_with_the_reason_it_is_out_for() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let worker = BasicWorkerBuilder::new("http://worker:8000")
            .health_config(no_health_check())
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                timeout_duration: Duration::from_millis(50),
                ..CircuitBreakerConfig::default()
            })
            .build();
        assert!(
            state_series(&handle.render()).is_empty(),
            "nothing before the first publish"
        );
        worker.republish_state();
        assert_eq!(current_state(&handle.render()), "available/none");

        worker.set_status(WorkerStatus::NotReady);
        assert_eq!(current_state(&handle.render()), "excluded/not_ready");
        worker.set_status(WorkerStatus::Draining);
        assert_eq!(current_state(&handle.render()), "excluded/draining");
        worker.set_status(WorkerStatus::Pending);
        assert_eq!(current_state(&handle.render()), "excluded/pending");
        worker.set_status(WorkerStatus::Failed);
        assert_eq!(current_state(&handle.render()), "excluded/failed");
        worker.set_status(WorkerStatus::Ready);
        assert_eq!(current_state(&handle.render()), "available/none");

        assert!(worker.set_overloaded(true));
        assert_eq!(current_state(&handle.render()), "excluded/overloaded");
        assert!(!worker.set_overloaded(true), "no edge, no publish");
        assert!(worker.set_overloaded(false));
        assert_eq!(current_state(&handle.render()), "available/none");

        assert!(worker.set_stall(Some(StallReason::Wedged)));
        assert_eq!(current_state(&handle.render()), "excluded/wedged");
        assert!(worker.set_stall(Some(StallReason::Unreachable)));
        assert_eq!(current_state(&handle.render()), "excluded/unreachable");
        assert!(worker.set_stall(None));
        assert_eq!(current_state(&handle.render()), "available/none");

        worker.record_circuit_breaker_outcome(false);
        // The state as stored, not the routing decision's read: on a loaded
        // machine the 50 ms timeout can pass before this line, and the
        // decision's read would already be the move to half-open.
        assert_eq!(
            worker.circuit_breaker.load().peek_state(),
            CircuitState::Open
        );
        assert_eq!(current_state(&handle.render()), "excluded/circuit_open");
        // The breaker's timeout moves it to half-open on the next read,
        // which is the routing decision's: the worker is on trial, and
        // its series says so.
        thread::sleep(Duration::from_millis(60));
        assert!(worker.circuit_breaker_can_execute());
        assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
        assert_eq!(current_state(&handle.render()), "available/none");
        worker.record_circuit_breaker_outcome(true);
        assert_eq!(current_state(&handle.render()), "available/none");
        // Every series the worker was on is still there, at 0: one per
        // reason of the table.
        assert_eq!(
            state_series(&handle.render()).len(),
            ROUTING_STATE_SERIES.len()
        );
    });
}

/// The facts are read in the routing decision's order: a worker that is
/// not ready names its status before its open breaker, and the breaker
/// once the status is ready again. A record that lands on an open
/// breaker past its timeout is ignored, and a record that finds the
/// breaker open publishes. A registration publishes the current pair
/// whether or not it was the one published before.
#[test]
fn the_first_excluding_fact_names_the_state_and_a_registration_republishes_it() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let worker = BasicWorkerBuilder::new("http://worker:8000")
            .health_config(no_health_check())
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                ..CircuitBreakerConfig::default()
            })
            .build();
        worker.set_status(WorkerStatus::NotReady);
        worker.record_circuit_breaker_outcome(false);
        assert_eq!(current_state(&handle.render()), "excluded/not_ready");
        worker.set_status(WorkerStatus::Ready);
        assert_eq!(current_state(&handle.render()), "excluded/circuit_open");

        // A record that lands on an open breaker past its timeout is
        // ignored, as before this series existed: the breaker does not
        // go to half-open and back on the record (no transition counted),
        // and the series stays on circuit_open until the routing
        // decision reads the breaker.
        let late = BasicWorkerBuilder::new("http://worker:8000")
            .health_config(no_health_check())
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                timeout_duration: Duration::from_millis(10),
                ..CircuitBreakerConfig::default()
            })
            .build();
        late.republish_state();
        late.record_circuit_breaker_outcome(false);
        thread::sleep(Duration::from_millis(20));
        late.record_circuit_breaker_outcome(false);
        let rendered = handle.render();
        assert_eq!(current_state(&rendered), "excluded/circuit_open");
        // Neither worker entered a registry, so no counter started at zero:
        // a transition that was never counted has no series, and the two
        // trips have theirs.
        let counted = |from_to: &str| {
            rendered
                .lines()
                .find(|line| {
                    line.starts_with("smg_worker_cb_transitions_total{") && line.contains(from_to)
                })
                .and_then(|line| line.rsplit(' ').next())
        };
        assert_eq!(
            counted("from=\"closed\",to=\"open\""),
            Some("2"),
            "{rendered}"
        );
        assert_eq!(
            counted("from=\"open\",to=\"half_open\""),
            None,
            "{rendered}"
        );
        assert!(
            late.circuit_breaker_can_execute(),
            "the decision's read starts the trial"
        );
        assert_eq!(current_state(&handle.render()), "available/none");
        // The round trip a late failure's compare cannot see: the
        // decision's read moved the breaker to half-open and the series
        // to available between the failure's peek of the breaker (open)
        // and its record, which trips the breaker again. The breaker is
        // open and the series says available; a record that finds the
        // breaker open publishes, and the series says so again.
        late.circuit_breaker.load().record_failure();
        assert_eq!(late.circuit_breaker.load().peek_state(), CircuitState::Open);
        assert_eq!(current_state(&handle.render()), "available/none");
        late.record_circuit_breaker_outcome(false);
        assert_eq!(late.circuit_breaker.load().peek_state(), CircuitState::Open);
        assert_eq!(current_state(&handle.render()), "excluded/circuit_open");

        // The address's series were dropped (a removal) and the worker
        // registers again: the plain publish sees no move, the
        // registration's publish sets the pair again.
        let again = PrometheusBuilder::new().build_recorder();
        let again_handle = again.handle();
        metrics::with_local_recorder(&again, || {
            worker.publish_state();
            assert!(state_series(&again_handle.render()).is_empty());
            worker.republish_state();
            assert_eq!(
                current_state(&again_handle.render()),
                "excluded/circuit_open"
            );
        });
    });
}

/// The routing snapshot and the availability predicate answer from the
/// published state code when it says `available`, and from the facts on
/// any other code. Through every veto and back, both agree with the facts
/// read one by one, and the counters the snapshot carries are live.
#[test]
fn the_routing_reads_follow_every_veto_through_the_published_code() {
    let build = |timeout_duration| {
        BasicWorkerBuilder::new("http://worker:8000")
            .health_config(no_health_check())
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                timeout_duration,
                ..CircuitBreakerConfig::default()
            })
            .build()
    };
    let agree = |worker: &BasicWorker, eligible: bool| {
        let state = worker.routing_state();
        assert_eq!(state.healthy, worker.is_healthy());
        assert_eq!(state.can_execute, worker.circuit_breaker_can_execute());
        assert_eq!(state.load, worker.load());
        assert_eq!(state.processed, worker.processed_requests());
        assert_eq!(state.overloaded, worker.is_overloaded());
        assert_eq!(state.stalled, worker.stall_reason().is_some());
        assert_eq!(state.eligible(), eligible);
        assert_eq!(worker.is_available(), eligible);
    };
    // The vetoes, on a breaker whose timeout the case cannot outlive: a slow
    // host between the trip and its reads finds it open all the same.
    let worker = build(Duration::from_secs(60));
    // Nothing published yet: the facts.
    agree(&worker, true);
    worker.republish_state();
    worker.increment_load();
    worker.increment_processed();
    agree(&worker, true);
    assert_eq!(worker.routing_state().load, 1);
    assert_eq!(worker.routing_state().processed, 1);
    worker.set_status(WorkerStatus::NotReady);
    agree(&worker, false);
    worker.set_status(WorkerStatus::Ready);
    agree(&worker, true);
    assert!(worker.set_overloaded(true));
    agree(&worker, false);
    assert!(worker.set_overloaded(false));
    agree(&worker, true);
    assert!(worker.set_stall(Some(StallReason::Unreachable)));
    agree(&worker, false);
    assert!(worker.set_stall(None));
    agree(&worker, true);
    worker.record_circuit_breaker_outcome(false);
    agree(&worker, false);
    // The breaker's timeout moves it to half-open on the routing read
    // itself, which is then the first to say the worker is back: on a
    // breaker with a short timeout, read only after the sleep.
    let worker = build(Duration::from_millis(50));
    worker.republish_state();
    worker.record_circuit_breaker_outcome(false);
    thread::sleep(Duration::from_millis(60));
    assert!(worker.routing_state().can_execute);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
    agree(&worker, true);
}

/// The code is a shortcut for `available` alone. A veto written past the
/// published code (a worker never registered, so never published; a
/// breaker tripped behind the worker's own record) is read as it is.
#[test]
fn a_veto_the_published_code_does_not_carry_is_still_read() {
    let worker = BasicWorkerBuilder::new("http://worker:8000")
        .health_config(no_health_check())
        .circuit_breaker_config(CircuitBreakerConfig {
            failure_threshold: 1,
            ..CircuitBreakerConfig::default()
        })
        .build();
    // Never published: the status decides.
    worker.runtime.load().set_status(WorkerStatus::NotReady);
    assert!(!worker.routing_state().healthy);
    assert!(!worker.is_available());
    worker.runtime.load().set_status(WorkerStatus::Ready);
    worker.republish_state();
    assert!(worker.routing_state().eligible());
    assert!(worker.is_available());
    // Published as overloaded, then the breaker tripped behind the worker's
    // record: the code names the overload, the read finds the breaker too.
    assert!(worker.set_overloaded(true));
    worker.circuit_breaker.load().record_failure();
    let state = worker.routing_state();
    assert!(state.overloaded);
    assert!(!state.can_execute);
    assert!(!state.eligible_ignoring_overload());
    assert!(!worker.is_available());
}

/// A retired instance asserts no shared state code. A same-URL replacement
/// built with another breaker config keeps a breaker of its own; the retired
/// instance's breaker moves must not publish over the replacement's veto,
/// which the routing reads answer from. They mark the code stale, and the
/// replacement's next read publishes its own facts.
#[test]
fn a_retired_worker_does_not_publish_over_its_replacement() {
    let retired = BasicWorkerBuilder::new("http://worker:8000")
        .health_config(no_health_check())
        .circuit_breaker_config(CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(50),
            ..CircuitBreakerConfig::default()
        })
        .build();
    retired.set_status(WorkerStatus::Ready);
    let replacement = BasicWorkerBuilder::new("http://worker:8000")
        .health_config(no_health_check())
        .circuit_breaker_config(CircuitBreakerConfig {
            failure_threshold: 2,
            ..CircuitBreakerConfig::default()
        })
        .build();
    assert!(replacement.inherit_shared_state_from(&retired));
    retired.abort_background_tasks();

    // The retired instance's own breaker opens; the replacement trips its own
    // and publishes `circuit_open` on the shared runtime.
    retired.record_circuit_breaker_outcome(false);
    replacement.record_circuit_breaker_outcome(false);
    replacement.record_circuit_breaker_outcome(false);
    assert!(!replacement.circuit_breaker_can_execute());
    assert_eq!(
        replacement.runtime.load().published_state(),
        CIRCUIT_OPEN_CODE
    );

    // The retired breaker's timeout moves it to half-open on a routing read, a
    // move that publishes `available` from an instance the registry holds.
    thread::sleep(Duration::from_millis(60));
    assert!(retired.routing_state().can_execute);
    let published = replacement.runtime.load().published_state();
    assert_ne!(
        published, AVAILABLE_CODE,
        "a retired instance does not publish"
    );
    assert_eq!(
        published & !STALE_BIT,
        CIRCUIT_OPEN_CODE,
        "it marks the code stale"
    );
    assert!(!replacement.routing_state().can_execute);
    assert_eq!(
        replacement.runtime.load().published_state(),
        CIRCUIT_OPEN_CODE,
        "the replacement's read published its own facts"
    );
    assert!(!replacement.is_available());
}

/// A same-URL replacement built with the same breaker config shares the
/// retired instance's breaker with its runtime. A request in flight on the
/// retired instance trips that breaker behind the replacement's back: the
/// retired instance asserts no code, it marks the published one stale, and
/// the replacement's next routing read publishes `circuit_open` from the
/// facts instead of answering `available` from the code.
#[test]
fn a_shared_breaker_tripped_through_a_retired_worker_is_read() {
    let config = || CircuitBreakerConfig {
        failure_threshold: 1,
        ..CircuitBreakerConfig::default()
    };
    let retired = BasicWorkerBuilder::new("http://worker:8000")
        .health_config(no_health_check())
        .circuit_breaker_config(config())
        .build();
    retired.set_status(WorkerStatus::Ready);
    let replacement = BasicWorkerBuilder::new("http://worker:8000")
        .health_config(no_health_check())
        .circuit_breaker_config(config())
        .build();
    assert!(replacement.inherit_shared_state_from(&retired));
    retired.abort_background_tasks();
    assert_eq!(replacement.runtime.load().published_state(), AVAILABLE_CODE);

    retired.record_circuit_breaker_outcome(false);
    assert!(
        !replacement.circuit_breaker_can_execute(),
        "the breaker is shared"
    );
    assert_ne!(
        replacement.runtime.load().published_state(),
        AVAILABLE_CODE,
        "a trip through the retired instance leaves no available code behind"
    );
    assert!(!replacement.routing_state().can_execute);
    assert_eq!(
        replacement.runtime.load().published_state(),
        CIRCUIT_OPEN_CODE,
        "the replacement's read published the facts"
    );
    assert!(!replacement.is_available());
}

/// A stall written through the retired instance lands on the shared runtime,
/// and the replacement's identical write is no edge to publish on: the
/// retired instance's stale mark is what gets the stall read and published.
#[test]
fn a_stall_written_through_a_retired_worker_is_read() {
    let retired = BasicWorkerBuilder::new("http://worker:8000")
        .health_config(no_health_check())
        .build();
    retired.set_status(WorkerStatus::Ready);
    let replacement = BasicWorkerBuilder::new("http://worker:8000")
        .health_config(no_health_check())
        .build();
    assert!(replacement.inherit_shared_state_from(&retired));
    retired.abort_background_tasks();
    assert_eq!(replacement.runtime.load().published_state(), AVAILABLE_CODE);

    assert!(retired.set_stall(Some(StallReason::Unreachable)));
    assert!(
        !replacement.set_stall(Some(StallReason::Unreachable)),
        "the shared fact is written already: no edge"
    );
    assert!(!replacement.is_available());
    let stalled = routing_state_series(
        WorkerStatus::Ready,
        false,
        Some(StallReason::Unreachable),
        false,
    ) as u8
        + 1;
    assert_eq!(
        replacement.runtime.load().published_state(),
        stalled,
        "the replacement's read published the stall"
    );
    assert!(replacement.routing_state().stalled);
}

#[tokio::test]
async fn a_failed_http_admin_call_names_the_worker_once_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("{}/?token=s3cret", refused.http_url());
    let text = BasicWorkerBuilder::new(&url)
        .build()
        .flush_cache()
        .await
        .expect_err("nothing listens at the address")
        .to_string();
    assert!(!text.contains("s3cret"), "{text}");
    assert_eq!(
        text.matches(&refused.addr().to_string()).count(),
        1,
        "{text}"
    );
}

#[tracing_test::traced_test]
#[tokio::test]
async fn a_failed_http_health_check_is_logged_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("{}/?token=s3cret", refused.http_url());
    let worker = BasicWorkerBuilder::new(&url).build();

    worker
        .check_health_async()
        .await
        .expect_err("nothing listens at the address");

    assert!(!logs_contain("s3cret"));
    assert!(logs_contain(&format!(
        "HTTP health check failed for {}:",
        refused.http_url()
    )));
}

#[test]
fn a_worker_and_its_metadata_print_no_api_key() {
    let marker = "marble";
    let url = "http://worker:8000";
    let debug = |worker: &BasicWorker| {
        let metadata = worker.metadata();
        [
            format!("{metadata:?}"),
            format!("{metadata:#?}"),
            format!("{worker:?}"),
            format!("{worker:#?}"),
        ]
    };

    let worker = BasicWorkerBuilder::new(url).api_key(marker).build();
    for text in debug(&worker) {
        assert!(!text.contains(marker), "{text}");
        assert!(text.contains("api_key: Some("), "{text}");
        assert!(text.contains(r#""<redacted>""#), "{text}");
    }

    let worker = BasicWorkerBuilder::new(url).build();
    for text in debug(&worker) {
        assert!(text.contains("api_key: None"), "{text}");
    }
}
