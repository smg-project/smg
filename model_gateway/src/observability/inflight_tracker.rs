use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, LazyLock, OnceLock,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use serde::Serialize;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    gauge_histogram::{BucketBounds, GaugeHistogramHandle, GaugeHistogramVec},
    metrics::Metrics,
};
use crate::policies::utils::PeriodicTask;

static INFLIGHT_AGE_BOUNDS: BucketBounds<11> =
    BucketBounds::new([30, 60, 180, 300, 600, 1200, 3600, 7200, 14400, 28800, 86400]);
static INFLIGHT_AGE_HISTOGRAM: GaugeHistogramVec<11> =
    GaugeHistogramVec::new("smg_http_inflight_request_age_count", &INFLIGHT_AGE_BOUNDS);
static INFLIGHT_AGE_HANDLE: LazyLock<GaugeHistogramHandle> =
    LazyLock::new(|| INFLIGHT_AGE_HISTOGRAM.register_no_labels());

pub struct InFlightRequestTracker {
    instance_id: Uuid,
    requests: DashMap<u64, Instant>,
    /// Live request count. `DashMap::len()`/`is_empty()` lock every shard, and
    /// the metrics middleware reads the count at request start and end, so it
    /// is kept in one atomic instead.
    count: AtomicUsize,
    next_id: AtomicU64,
    sampler: OnceLock<PeriodicTask>,
    draining: AtomicBool,
    drain_signal: CancellationToken,
    upgraded: AtomicUsize,
    /// Signaled when requests drain to zero during shutdown.
    drain_complete: Notify,
}

impl InFlightRequestTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            instance_id: Uuid::now_v7(),
            requests: DashMap::new(),
            count: AtomicUsize::new(0),
            next_id: AtomicU64::new(0),
            sampler: OnceLock::new(),
            draining: AtomicBool::new(false),
            drain_signal: CancellationToken::new(),
            upgraded: AtomicUsize::new(0),
            drain_complete: Notify::new(),
        })
    }

    pub fn start_sampler(self: &Arc<Self>, interval_secs: u64) {
        let tracker = self.clone();
        let task = PeriodicTask::spawn(interval_secs, "InFlightRequestSampler", move || {
            tracker.sample_and_record();
        });
        #[expect(
            clippy::expect_used,
            reason = "start_sampler is called once at startup; double-init is a fatal bug"
        )]
        self.sampler
            .set(task)
            .expect("start_sampler is called once at startup; double-init is a bug");
    }

    pub fn track(self: &Arc<Self>) -> InFlightGuard {
        self.track_from(Instant::now())
    }

    /// [`Self::track`] for a request that came in at `started` (the edge's
    /// entry stamp): its age is counted from there.
    pub fn track_from(self: &Arc<Self>, started: Instant) -> InFlightGuard {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // Count first: a drain check must never see zero once admission began.
        self.count.fetch_add(1, Ordering::AcqRel);
        self.requests.insert(request_id, started);
        InFlightGuard {
            tracker: self.clone(),
            request_id,
        }
    }

    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Observe requests through response-body completion. To drain a worker,
    /// first stop new routing to it, then retain this snapshot's IDs and poll
    /// until none remain. Later arrivals do not prolong that cohort's drain.
    /// A different instance ID invalidates a previous snapshot.
    /// Upgraded connections outlive their HTTP response and require separate,
    /// protocol-specific drain tracking.
    pub fn snapshot(&self) -> InFlightSnapshot {
        InFlightSnapshot {
            instance_id: self.instance_id,
            request_ids: self.requests.iter().map(|entry| *entry.key()).collect(),
        }
    }

    /// Begin graceful shutdown: mark as draining. Idempotent.
    pub fn begin_drain(&self) {
        // Pair with upgrade admission: either the upgrade sees drain, or
        // shutdown sees the upgrade registered before its drain check.
        self.draining.store(true, Ordering::SeqCst);
        self.drain_signal.cancel();
    }

    /// Returns `true` if drain has been initiated.
    #[inline]
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    pub(crate) async fn draining(&self) {
        self.drain_signal.cancelled().await;
    }

    /// Upgraded sessions keep the shutdown alive through their close frames.
    /// They are tracked separately from requests and worker-drain snapshots.
    pub(crate) fn track_upgrade(self: &Arc<Self>) -> UpgradeGuard {
        self.upgraded.fetch_add(1, Ordering::SeqCst);
        UpgradeGuard(self.clone())
    }

    pub(crate) fn upgraded_sessions(&self) -> usize {
        self.upgraded.load(Ordering::SeqCst)
    }

    pub(crate) fn shutdown_pending(&self) -> bool {
        !self.is_empty() || self.upgraded_sessions() != 0
    }

    pub(crate) async fn wait_for_shutdown(&self, timeout: Duration) -> bool {
        self.wait_until(timeout, || !self.shutdown_pending()).await
    }

    /// Wait until all in-flight requests complete, or until `max_timeout` elapses.
    ///
    /// Returns `true` if all requests drained, `false` if timed out.
    pub async fn wait_for_drain(&self, max_timeout: Duration) -> bool {
        self.wait_until(max_timeout, || self.is_empty()).await
    }

    async fn wait_until(&self, max_timeout: Duration, done: impl Fn() -> bool) -> bool {
        if done() {
            return true;
        }
        tokio::time::timeout(max_timeout, async {
            loop {
                // Create the notified future BEFORE checking the condition
                // to avoid TOCTOU race where a guard drops between check and wait.
                let notified = self.drain_complete.notified();
                if done() {
                    return;
                }
                notified.await;
            }
        })
        .await
        .is_ok()
    }

    pub fn compute_bucket_counts(&self) -> Vec<usize> {
        let ages = self.collect_ages();
        INFLIGHT_AGE_BOUNDS.compute_counts(&ages)
    }

    fn collect_ages(&self) -> Vec<u64> {
        let now = Instant::now();
        self.requests
            .iter()
            .map(|entry| now.duration_since(*entry.value()).as_secs())
            .collect()
    }

    /// The sampler's tick: the in-flight age histogram from the tracked
    /// requests, and `smg_http_connections_active` from the count. The
    /// gauge's per-event sets (the arrival, the three releases) each load
    /// the count and set the gauge as two steps, so two releases that
    /// interleave as load, load, set, set leave the gauge one above the
    /// count until the next event, and a server that goes idle right then
    /// has no next event; the tick sets the gauge from the count, so a stale
    /// reading lasts one sample interval at most.
    fn sample_and_record(&self) {
        let counts = self.compute_bucket_counts();
        INFLIGHT_AGE_HANDLE.set_counts(&counts);
        Metrics::set_http_connections_active(self.len());
    }
}

/// Opaque process-local identities, unrelated to client request IDs or content.
#[derive(Serialize)]
pub struct InFlightSnapshot {
    pub instance_id: Uuid,
    pub request_ids: Vec<u64>,
}

pub(crate) struct UpgradeGuard(Arc<InFlightRequestTracker>);

impl Drop for UpgradeGuard {
    fn drop(&mut self) {
        if self.0.upgraded.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.drain_complete.notify_waiters();
        }
    }
}

pub struct InFlightGuard {
    tracker: Arc<InFlightRequestTracker>,
    request_id: u64,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.tracker.requests.remove(&self.request_id);
        let was = self.tracker.count.fetch_sub(1, Ordering::AcqRel);
        // Notify drain waiters when we hit zero. `wait_for_drain()` re-checks
        // the condition after waking, so spurious wakeups are harmless.
        // We unconditionally notify (without checking is_draining()) to avoid a
        // race where the last guard drops before begin_drain()'s store
        // is visible, which would skip the notification entirely.
        if was == 1 {
            self.tracker.drain_complete.notify_waiters();
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/inflight_tracker/tests.rs"]
mod tests;
