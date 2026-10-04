use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, LazyLock, OnceLock,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio::sync::Notify;

use super::gauge_histogram::{BucketBounds, GaugeHistogramHandle, GaugeHistogramVec};
use crate::policies::utils::PeriodicTask;

static INFLIGHT_AGE_BOUNDS: BucketBounds<11> =
    BucketBounds::new([30, 60, 180, 300, 600, 1200, 3600, 7200, 14400, 28800, 86400]);
static INFLIGHT_AGE_HISTOGRAM: GaugeHistogramVec<11> =
    GaugeHistogramVec::new("smg_http_inflight_request_age_count", &INFLIGHT_AGE_BOUNDS);
static INFLIGHT_AGE_HANDLE: LazyLock<GaugeHistogramHandle> =
    LazyLock::new(|| INFLIGHT_AGE_HISTOGRAM.register_no_labels());

pub struct InFlightRequestTracker {
    requests: DashMap<u64, Instant>,
    /// Live request count. `DashMap::len()`/`is_empty()` lock every shard, and
    /// the metrics middleware reads the count at request start and end, so it
    /// is kept in one atomic instead.
    count: AtomicUsize,
    next_id: AtomicU64,
    sampler: OnceLock<PeriodicTask>,
    /// Monotonic flag: false → true. Never reset.
    draining: AtomicBool,
    /// Signaled when requests drain to zero during shutdown.
    drain_complete: Notify,
}

impl InFlightRequestTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            requests: DashMap::new(),
            count: AtomicUsize::new(0),
            next_id: AtomicU64::new(0),
            sampler: OnceLock::new(),
            draining: AtomicBool::new(false),
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
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // Count first: a drain check must never see zero once admission began.
        self.count.fetch_add(1, Ordering::AcqRel);
        self.requests.insert(request_id, Instant::now());
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

    /// Begin graceful shutdown: mark as draining. Idempotent.
    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::Release);
    }

    /// Returns `true` if drain has been initiated.
    #[inline]
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }

    /// Wait until all in-flight requests complete, or until `max_timeout` elapses.
    ///
    /// Returns `true` if all requests drained, `false` if timed out.
    pub async fn wait_for_drain(&self, max_timeout: Duration) -> bool {
        if self.is_empty() {
            return true;
        }
        tokio::time::timeout(max_timeout, async {
            loop {
                // Create the notified future BEFORE checking the condition
                // to avoid TOCTOU race where a guard drops between check and wait.
                let notified = self.drain_complete.notified();
                if self.is_empty() {
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

    fn sample_and_record(&self) {
        let counts = self.compute_bucket_counts();
        INFLIGHT_AGE_HANDLE.set_counts(&counts);
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
        // race where the last guard drops before begin_drain()'s Release store
        // is visible, which would skip the notification entirely.
        if was == 1 {
            self.tracker.drain_complete.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    impl InFlightRequestTracker {
        fn insert_with_time(&self, request_id: u64, start_time: Instant) {
            self.requests.insert(request_id, start_time);
        }
    }

    #[test]
    fn test_track_and_drop() {
        let tracker = InFlightRequestTracker::new();
        {
            let _guard1 = tracker.track();
            let _guard2 = tracker.track();
            assert_eq!(tracker.len(), 2);
        }
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_guard_auto_deregister() {
        let tracker = InFlightRequestTracker::new();
        let guard = tracker.track();
        assert_eq!(tracker.len(), 1);
        drop(guard);
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_request_age_tracking() {
        let tracker = InFlightRequestTracker::new();
        let _guard = tracker.track();
        std::thread::sleep(Duration::from_millis(100));

        let entry = tracker.requests.iter().next().unwrap();
        let age = entry.value().elapsed();
        assert!(age >= Duration::from_millis(100));
    }

    #[test]
    fn test_collect_ages_empty() {
        let tracker = InFlightRequestTracker::new();
        let ages = tracker.collect_ages();
        assert!(ages.is_empty());
    }

    #[test]
    fn test_collect_ages() {
        let tracker = InFlightRequestTracker::new();
        let now = Instant::now();

        tracker.insert_with_time(1, now);
        tracker.insert_with_time(2, now - Duration::from_secs(45));
        tracker.insert_with_time(3, now - Duration::from_secs(100));

        let ages = tracker.collect_ages();
        assert_eq!(ages.len(), 3);
        // Ages should be approximately 0, 45, 100 (order may vary due to DashMap)
        let mut sorted_ages = ages.clone();
        sorted_ages.sort_unstable();
        assert!(sorted_ages[0] <= 1); // ~0s
        assert!((44..=46).contains(&sorted_ages[1])); // ~45s
        assert!((99..=101).contains(&sorted_ages[2])); // ~100s
    }

    #[test]
    fn test_concurrent_tracking() {
        use std::thread;

        let tracker = InFlightRequestTracker::new();
        let mut handles = vec![];

        for _ in 0..10 {
            let t = tracker.clone();
            handles.push(thread::spawn(move || {
                (0..100).map(|_| t.track()).collect::<Vec<_>>()
            }));
        }

        let all_guards: Vec<_> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        assert_eq!(tracker.len(), 1000);
        drop(all_guards);
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_unique_ids() {
        let tracker = InFlightRequestTracker::new();
        let g1 = tracker.track();
        let g2 = tracker.track();
        let g3 = tracker.track();

        assert_ne!(g1.request_id, g2.request_id);
        assert_ne!(g2.request_id, g3.request_id);
        assert_eq!(tracker.len(), 3);
    }

    #[test]
    fn test_draining_flag() {
        let tracker = InFlightRequestTracker::new();
        assert!(!tracker.is_draining());
        tracker.begin_drain();
        assert!(tracker.is_draining());
        // Idempotent
        tracker.begin_drain();
        assert!(tracker.is_draining());
    }

    #[tokio::test]
    async fn test_wait_for_drain_already_empty() {
        let tracker = InFlightRequestTracker::new();
        tracker.begin_drain();
        let drained = tracker.wait_for_drain(Duration::from_secs(1)).await;
        assert!(drained);
    }

    #[tokio::test]
    async fn test_wait_for_drain_completes_when_guards_drop() {
        let tracker = InFlightRequestTracker::new();
        let guard1 = tracker.track();
        let guard2 = tracker.track();
        assert_eq!(tracker.len(), 2);

        tracker.begin_drain();

        let tracker_clone = tracker.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "test needs a concurrent task to exercise drain notification"
        )]
        let drain_handle =
            tokio::spawn(async move { tracker_clone.wait_for_drain(Duration::from_secs(5)).await });

        // Give the drain task a moment to start waiting
        tokio::time::sleep(Duration::from_millis(10)).await;

        drop(guard1);
        assert_eq!(tracker.len(), 1);

        drop(guard2);
        assert_eq!(tracker.len(), 0);

        let drained = drain_handle.await.unwrap();
        assert!(drained);
    }

    #[tokio::test]
    async fn test_wait_for_drain_times_out() {
        let tracker = InFlightRequestTracker::new();
        let _guard = tracker.track();

        tracker.begin_drain();

        let drained = tracker.wait_for_drain(Duration::from_millis(50)).await;
        assert!(!drained);
        assert_eq!(tracker.len(), 1);
    }
}
