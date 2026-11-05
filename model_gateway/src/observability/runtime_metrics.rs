//! Tokio runtime self-observability: event-loop canary + RuntimeMetrics sampler.
//!
//! Adapted from ai-dynamo/dynamo `lib/runtime/src/metrics/tokio_perf.rs`
//! (Apache-2.0); see "Ported vs dropped" below for how this differs.
//!
//! A single background task (`spawn_observer`) gives the runtime two cheap,
//! always-on probes so starvation is visible on the `/metrics` endpoint:
//!
//! - **Event-loop canary**: sleeps 10ms in a loop and measures wake drift.
//!   Drift is recorded into `EVENT_LOOP_DELAY_SECONDS`; drift above 5ms
//!   increments `smg_tokio_event_loop_stalls_total`. Any stall means every
//!   task on the runtime (accept loops, health checks, streaming responses)
//!   was delayed by at least that long — it distinguishes "runtime starved"
//!   from "backend slow".
//! - **Runtime sampler**: every 1s reads `Handle::current().metrics()` and
//!   exports queue depth, alive tasks, worker count, and per-worker busy
//!   ratio / park counts.
//!
//! `spawn_observer` must be called from within the runtime being measured.
//!
//! # Ported vs dropped (donor: dynamo `tokio_perf.rs`)
//!
//! SMG builds tokio 1.52 without `--cfg tokio_unstable`, so donor metrics
//! backed by unstable `RuntimeMetrics` APIs are dropped:
//!
//! - ported: `global_queue_depth`, `num_alive_tasks` (gauges),
//!   `worker_park_count` (counter, delta-tracked), event-loop delay histogram
//!   and stall counter;
//! - ported with a better source: `worker_busy_ratio` — the donor proxied it
//!   from `worker_mean_poll_time` (unstable); SMG computes the real ratio from
//!   `worker_total_busy_duration` deltas (stable on 64-bit targets);
//! - added: `smg_tokio_workers` (actual worker-thread count, since
//!   misconfigured `TOKIO_WORKER_THREADS` has caused real incidents);
//! - dropped (all `tokio_unstable`-gated on 1.52): `num_blocking_threads`,
//!   `num_idle_blocking_threads`, `blocking_queue_depth`,
//!   `budget_forced_yield_count`, `worker_mean_poll_time`,
//!   `worker_local_queue_depth`, `worker_steal_count`, `worker_overflow_count`;
//! - dropped: the donor's opt-in poll-time histogram
//!   (`DYN_ENABLE_POLL_HISTOGRAM`). On tokio 1.52
//!   `Builder::enable_metrics_poll_time_histogram()` is itself
//!   `tokio_unstable`-only and would require changing the runtime builder in
//!   `main.rs`. Revisit behind an `SMG_ENABLE_POLL_HISTOGRAM` env var once
//!   tokio stabilizes the API (it costs ~2x `Instant::now()` per task poll,
//!   so it must stay opt-in).
//!
//! # How to read the metrics
//!
//! - `smg_tokio_event_loop_stalls_total` increasing → something is blocking
//!   worker threads (sync I/O, CPU-bound work, lock contention) — alert on it.
//! - `smg_tokio_worker_busy_ratio` near 1.0 across workers → runtime
//!   saturated; correlate with `smg_tokio_global_queue_depth` growth.
//! - `rate(smg_tokio_worker_parks_total)` near zero while busy ratio is high →
//!   workers never go idle (sustained overload rather than bursts).

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use tokio::runtime::{Handle, RuntimeMetrics};

use super::metrics::intern_string;

/// Canary sleep interval. Wake drift is measured against this.
const CANARY_INTERVAL: Duration = Duration::from_millis(10);
/// Canary drift above this counts as an event-loop stall.
const STALL_THRESHOLD: Duration = Duration::from_millis(5);
/// How often the runtime sampler reads `RuntimeMetrics`.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// Event-loop delay canary histogram (drift from a 10ms sleep, in seconds).
///
/// `pub(crate)` so `observability::metrics::start_prometheus` can attach
/// [`EVENT_LOOP_DELAY_BUCKETS`]; without explicit buckets the recorder would
/// render this as a summary.
pub(crate) const EVENT_LOOP_DELAY_SECONDS: &str = "smg_tokio_event_loop_delay_seconds";
const EVENT_LOOP_STALLS_TOTAL: &str = "smg_tokio_event_loop_stalls_total";
const GLOBAL_QUEUE_DEPTH: &str = "smg_tokio_global_queue_depth";
const ALIVE_TASKS: &str = "smg_tokio_alive_tasks";
const WORKERS: &str = "smg_tokio_workers";
const WORKER_BUSY_RATIO: &str = "smg_tokio_worker_busy_ratio";
const WORKER_PARKS_TOTAL: &str = "smg_tokio_worker_parks_total";

/// Histogram buckets for [`EVENT_LOOP_DELAY_SECONDS`] (donor values). Drift is
/// expected in the 0-50ms range when healthy; the 1s bucket captures severe
/// stalls.
pub(crate) const EVENT_LOOP_DELAY_BUCKETS: &[f64] =
    &[0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0];

/// Register descriptions. Called once from `observability::metrics::init_metrics`.
pub(crate) fn describe() {
    describe_histogram!(
        EVENT_LOOP_DELAY_SECONDS,
        "Event loop delay canary: wake drift from a 10ms sleep (seconds)"
    );
    describe_counter!(
        EVENT_LOOP_STALLS_TOTAL,
        "Event loop stalls (canary wake drift exceeded 5ms)"
    );
    describe_gauge!(
        GLOBAL_QUEUE_DEPTH,
        "Tasks currently queued in the tokio runtime global queue"
    );
    describe_gauge!(ALIVE_TASKS, "Tasks alive (spawned, not yet completed)");
    describe_gauge!(WORKERS, "Worker threads used by the tokio runtime");
    describe_gauge!(
        WORKER_BUSY_RATIO,
        "Fraction of the last sampling interval each worker spent busy (0.0-1.0; >0.95 = saturated)"
    );
    describe_counter!(
        WORKER_PARKS_TOTAL,
        "Times each worker has parked (gone idle); a near-zero rate under high busy ratio means sustained overload"
    );
}

/// Spawn the combined event-loop canary + runtime sampler.
///
/// Must be called from within the runtime to be observed — the canary
/// measures the wake latency of whichever runtime polls it. The task runs for
/// the lifetime of the process and exits with the runtime.
pub(crate) fn spawn_observer() {
    #[expect(
        clippy::disallowed_methods,
        reason = "runtime observer runs for the lifetime of the process, like the metrics upkeep task"
    )]
    tokio::spawn(async {
        let mut sampler = RuntimeSampler::new();
        let mut next_sample = Instant::now() + SAMPLE_INTERVAL;
        loop {
            canary_tick().await;
            if Instant::now() >= next_sample {
                next_sample = Instant::now() + SAMPLE_INTERVAL;
                sampler.sample(&Handle::current().metrics());
            }
        }
    });
}

/// One canary tick: sleep [`CANARY_INTERVAL`], measure wake drift, record it.
/// Returns the measured drift.
async fn canary_tick() -> Duration {
    let start = Instant::now();
    tokio::time::sleep(CANARY_INTERVAL).await;
    let drift = start.elapsed().saturating_sub(CANARY_INTERVAL);
    histogram!(EVENT_LOOP_DELAY_SECONDS).record(drift.as_secs_f64());
    if drift > STALL_THRESHOLD {
        counter!(EVENT_LOOP_STALLS_TOTAL).increment(1);
    }
    drift
}

/// Per-worker previous samples for the monotonic counters and busy durations.
/// Owned by the single observer task — no locks needed.
struct RuntimeSampler {
    last_sample: Instant,
    prev_busy: Vec<Duration>,
    prev_parks: Vec<u64>,
    /// Interned `"0"`, `"1"`, ... worker labels, built once.
    worker_labels: Vec<Arc<str>>,
}

impl RuntimeSampler {
    fn new() -> Self {
        Self {
            last_sample: Instant::now(),
            prev_busy: Vec::new(),
            prev_parks: Vec::new(),
            worker_labels: Vec::new(),
        }
    }

    fn ensure_workers(&mut self, num_workers: usize) {
        if self.prev_busy.len() < num_workers {
            self.prev_busy.resize(num_workers, Duration::ZERO);
            self.prev_parks.resize(num_workers, 0);
        }
        while self.worker_labels.len() < num_workers {
            let label = self.worker_labels.len().to_string();
            self.worker_labels.push(intern_string(&label));
        }
    }

    fn sample(&mut self, runtime: &RuntimeMetrics) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_sample);
        self.last_sample = now;

        gauge!(GLOBAL_QUEUE_DEPTH).set(runtime.global_queue_depth() as f64);
        gauge!(ALIVE_TASKS).set(runtime.num_alive_tasks() as f64);
        let num_workers = runtime.num_workers();
        gauge!(WORKERS).set(num_workers as f64);

        self.ensure_workers(num_workers);
        for worker in 0..num_workers {
            let label = Arc::clone(&self.worker_labels[worker]);

            // True busy ratio over the elapsed interval. The first sample
            // covers "since runtime start" and is clamped, like the donor's
            // first park delta.
            let busy = runtime.worker_total_busy_duration(worker);
            let busy_delta = busy.saturating_sub(self.prev_busy[worker]);
            self.prev_busy[worker] = busy;
            let ratio = if elapsed.is_zero() {
                0.0
            } else {
                (busy_delta.as_secs_f64() / elapsed.as_secs_f64()).clamp(0.0, 1.0)
            };
            gauge!(WORKER_BUSY_RATIO, "worker" => Arc::clone(&label)).set(ratio);

            // Monotonically increasing total: track deltas so we can use a
            // Counter (tokio reports the absolute count since runtime start).
            let parks = runtime.worker_park_count(worker);
            counter!(WORKER_PARKS_TOTAL, "worker" => label)
                .increment(parks.saturating_sub(self.prev_parks[worker]));
            self.prev_parks[worker] = parks;
        }
    }
}

#[cfg(test)]
mod tests {
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
}
