use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};

use super::*;

/// Run `f` with a real Prometheus recorder installed thread-locally
/// (the same recorder type production uses, including the canary bucket
/// override) and return the rendered /metrics text plus `f`'s result.
fn with_test_recorder<T>(f: impl FnOnce() -> T) -> (String, T) {
    let recorder = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full(EVENT_LOOP_DELAY_SECONDS.to_string()),
            EVENT_LOOP_DELAY_BUCKETS,
        )
        .unwrap()
        .build_recorder();
    let handle = recorder.handle();
    let result = metrics::with_local_recorder(&recorder, f);
    (handle.render(), result)
}

#[test]
fn canary_measures_drift_and_counts_stall_on_blocked_runtime() {
    // Single-threaded runtime: a task that blocks the thread stalls the
    // event loop, so the canary's 10ms timer must fire late.
    let (rendered, drift) = with_test_recorder(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(async {
            #[expect(
                clippy::disallowed_methods,
                reason = "test needs a concurrent task to block the runtime"
            )]
            tokio::spawn(async {
                // Polled while the canary awaits its timer; blocks the
                // only worker thread well past the 5ms stall threshold.
                std::thread::sleep(Duration::from_millis(100));
            });
            canary_tick().await
        })
    });

    assert!(
        drift >= Duration::from_millis(50),
        "expected >=50ms drift from a 100ms thread block, got {drift:?}"
    );
    assert!(
        rendered.contains("smg_tokio_event_loop_stalls_total 1"),
        "stall counter not incremented; rendered:\n{rendered}"
    );
    assert!(
        rendered.contains("smg_tokio_event_loop_delay_seconds_count 1"),
        "delay histogram not recorded; rendered:\n{rendered}"
    );
}

#[test]
fn sampler_registration_and_sampling_does_not_panic() {
    let (rendered, ()) = with_test_recorder(|| {
        describe();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        // block_on runs this future on the current thread, so the
        // thread-local test recorder sees every sample.
        rt.block_on(async {
            let mut sampler = RuntimeSampler::new();
            sampler.sample(&Handle::current().metrics());
            tokio::time::sleep(Duration::from_millis(20)).await;
            sampler.sample(&Handle::current().metrics());
        });
    });

    assert!(rendered.contains("smg_tokio_workers 2"), "{rendered}");
    assert!(
        rendered.contains("smg_tokio_global_queue_depth"),
        "{rendered}"
    );
    assert!(rendered.contains("smg_tokio_alive_tasks"), "{rendered}");
    assert!(
        rendered.contains(r#"smg_tokio_worker_busy_ratio{worker="0"}"#),
        "{rendered}"
    );
    assert!(
        rendered.contains(r#"smg_tokio_worker_parks_total{worker="1"}"#),
        "{rendered}"
    );
}

#[test]
fn busy_ratio_renders_within_bounds_under_load() {
    let (rendered, ()) = with_test_recorder(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(async {
            let mut sampler = RuntimeSampler::new();
            sampler.sample(&Handle::current().metrics());
            // Busy-spin the only worker so busy_delta approaches elapsed.
            let spin_until = Instant::now() + Duration::from_millis(20);
            while Instant::now() < spin_until {
                tokio::task::yield_now().await;
            }
            sampler.sample(&Handle::current().metrics());
        });
    });

    // The clamp is load-bearing: the first sample's busy delta covers
    // "since runtime start" and may exceed the sampler's elapsed window.
    let ratio_line = rendered
        .lines()
        .find(|line| line.starts_with(r#"smg_tokio_worker_busy_ratio{worker="0"}"#))
        .expect("busy ratio gauge not rendered");
    let ratio: f64 = ratio_line
        .rsplit(' ')
        .next()
        .unwrap()
        .parse()
        .expect("busy ratio value not a float");
    assert!(
        (0.0..=1.0).contains(&ratio),
        "busy ratio out of bounds: {ratio}"
    );
}
