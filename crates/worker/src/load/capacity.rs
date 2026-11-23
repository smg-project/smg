//! Aggregate backend capacity tracking.

#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{
    atomic::{AtomicU16, AtomicU8, Ordering},
    Arc, Weak,
};

use futures::FutureExt as _;
use tokio::sync::{broadcast, watch};

use crate::{
    identity::basic::Worker,
    registry::{event::WorkerEvent, Generation, WorkerRegistry},
};

/// Upper bound on events folded into one recompute: one awaited event plus
/// `MAX_COALESCED_EVENTS - 1` immediately ready events.
const MAX_COALESCED_EVENTS: usize = 1024;

/// Which precedence tier produced the most recently computed capacity value.
/// Exposed as a Prometheus gauge so operators can debug capacity decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CapacitySource {
    /// Tier 4 — no workers known yet (or unhealthy fleet). Using legacy
    /// `--max-concurrent-requests` value as a fallback.
    LegacyFallback = 0,
    /// Tier 1 — operator pinned via `--worker-capacity-override`.
    Override = 1,
    /// Tier 2 — every healthy worker reported `max_running_requests`. Sum
    /// across the fleet.
    WorkerReported = 2,
    /// Tier 3 — at least one worker did not report; reporters contribute
    /// their reported value, non-reporters contribute `slots_per_worker`.
    /// Also used when no worker reports but workers exist (pure tier 3).
    Mixed = 3,
}

impl CapacitySource {
    pub fn from_u8(raw: u8) -> Self {
        match raw {
            1 => Self::Override,
            2 => Self::WorkerReported,
            3 => Self::Mixed,
            _ => Self::LegacyFallback,
        }
    }
}

/// Configuration for the WorkerCapacity tracker.
///
/// Built once at gateway startup from CLI flags. None of the fields
/// change at runtime (use a future `ArcSwap<Settings>` if hot-reload
/// is ever needed).
#[derive(Debug, Clone)]
pub struct CapacityTrackerSettings {
    /// Tier 1 override. `Some(n)` with `n > 0` pins capacity to `n`
    /// regardless of the fleet. `None` (or originally-zero) means
    /// "derive from workers."
    pub override_capacity: Option<u16>,
    /// Per-worker slot count used for tier 3 (mixed) and pure tier-3
    /// (workers known, none report).
    pub slots_per_worker: u16,
    /// Tier 4 fallback when no healthy workers are known.
    /// Typically sourced from the existing `--max-concurrent-requests` flag.
    pub legacy_max_concurrent_requests: u16,
}

impl CapacityTrackerSettings {
    /// Constructor that maps an `i32` override (0 = derive) to the
    /// internal `Option<u16>` representation. Matches the shape of
    /// the `--worker-capacity-override` CLI flag.
    ///
    /// Values that don't fit in `u16` (negative or > 65 535) fall back
    /// to "derive from workers" *and* log a warning so operators notice
    /// that their override was ignored.
    pub fn with_override(raw: i32) -> Self {
        let override_capacity = u16::try_from(raw).ok().filter(|n| *n > 0);
        if override_capacity.is_none() && raw != 0 {
            tracing::warn!(
                value = raw,
                max = u16::MAX,
                "worker-capacity-override out of u16 range; falling back to dynamic capacity"
            );
        }
        Self {
            override_capacity,
            ..Self::default()
        }
    }
}

impl Default for CapacityTrackerSettings {
    fn default() -> Self {
        Self {
            override_capacity: None,
            slots_per_worker: 64,
            legacy_max_concurrent_requests: 1024,
        }
    }
}

/// Tracks aggregate backend capacity derived from a `WorkerRegistry`.
///
/// One supervised tokio task subscribes to worker lifecycle events and
/// recomputes capacity by the 4-tier precedence (see `recompute`).
/// Consumers read the current value via `current()` or react to changes
/// via `watch()`.
///
/// **Lifecycle.** The struct itself holds no `Arc` to the registry or
/// to its own settings; the event task holds `Weak` references and
/// upgrades them per iteration. When the caller drops their last
/// `Arc<WorkerCapacity>`, the next event in the task triggers an exit
/// (or the task exits on `RecvError::Closed` when the registry itself
/// drops). This avoids the obvious Arc cycle.
pub struct WorkerCapacity {
    capacity: AtomicU16,
    source: AtomicU8,
    watch_tx: watch::Sender<u16>,
    #[cfg(test)]
    recompute_count: AtomicUsize,
    #[cfg(test)]
    event_loop_ready: tokio::sync::Notify,
    #[cfg(test)]
    panic_before_recompute: AtomicBool,
}

impl WorkerCapacity {
    /// Synchronous current capacity. Hot-path safe — single atomic load.
    #[inline]
    pub fn current(&self) -> u16 {
        self.capacity.load(Ordering::Acquire)
    }

    /// Which tier produced the current capacity.
    pub fn source(&self) -> CapacitySource {
        CapacitySource::from_u8(self.source.load(Ordering::Acquire))
    }

    /// Receiver for reacting to capacity changes. Cheap to clone.
    pub fn watch(&self) -> watch::Receiver<u16> {
        self.watch_tx.subscribe()
    }

    /// Test-only constructor that bypasses the event task. Used in unit tests
    /// that only need to exercise the read API.
    #[cfg(test)]
    pub(crate) fn for_test_with_value(capacity: u16, source: CapacitySource) -> Arc<Self> {
        let (tx, _rx) = watch::channel(capacity);
        Arc::new(Self {
            capacity: AtomicU16::new(capacity),
            source: AtomicU8::new(source as u8),
            watch_tx: tx,
            recompute_count: AtomicUsize::new(0),
            event_loop_ready: tokio::sync::Notify::new(),
            panic_before_recompute: AtomicBool::new(false),
        })
    }

    /// Construct a `WorkerCapacity`, compute the initial value
    /// synchronously, and spawn the supervised event-loop task.
    ///
    /// The task holds only `Weak` references to the registry and tracker
    /// (and an owned copy of the settings). When the caller drops their
    /// last `Arc<WorkerCapacity>` or the registry is dropped, the task
    /// exits on the next iteration — no Arc cycle.
    pub fn spawn<G: Generation>(
        registry: Arc<WorkerRegistry<G>>,
        settings: CapacityTrackerSettings,
    ) -> Arc<Self> {
        // Subscribe before taking the initial snapshot so mutations during
        // construction remain buffered for the event loop.
        let mut events = registry.subscribe_events();

        // Initial compute: synchronous over the current registry state,
        // so callers see a valid `current()` immediately after spawn.
        let workers = healthy_workers(&registry);
        let (initial_capacity, initial_source) = recompute(&settings, &workers);

        let (watch_tx, _initial_rx) = watch::channel(initial_capacity);

        let this = Arc::new(Self {
            capacity: AtomicU16::new(initial_capacity),
            source: AtomicU8::new(initial_source as u8),
            watch_tx,
            #[cfg(test)]
            recompute_count: AtomicUsize::new(0),
            #[cfg(test)]
            event_loop_ready: tokio::sync::Notify::new(),
            #[cfg(test)]
            panic_before_recompute: AtomicBool::new(false),
        });

        // Supervised loop: any panic in the inner future restarts the task
        // after a 1s backoff. Graceful exit (Ok) breaks out.
        let weak_tracker = Arc::downgrade(&this);
        let weak_registry = Arc::downgrade(&registry);
        let task_settings = settings;
        #[expect(
            clippy::disallowed_methods,
            reason = "supervised long-lived task: panics are caught and restarted; holds Weak refs so it cannot keep the tracker or registry alive"
        )]
        tokio::spawn(async move {
            let mut resync = false;
            loop {
                let result = std::panic::AssertUnwindSafe(run_event_loop(
                    weak_tracker.clone(),
                    weak_registry.clone(),
                    task_settings.clone(),
                    &mut events,
                    resync,
                ))
                .catch_unwind()
                .await;
                match result {
                    Ok(()) => break,
                    Err(payload) => {
                        let msg = payload
                            .downcast_ref::<&str>()
                            .copied()
                            .map(String::from)
                            .or_else(|| payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "(non-string panic)".into());
                        tracing::error!(
                            panic.message = %msg,
                            "WorkerCapacity event task panicked; restarting in 1s"
                        );
                        resync = true;
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            }
        });

        this
    }
}

/// Filter the registry to healthy workers only. Allocates a Vec so we
/// don't hold the registry's internal locks while iterating.
fn healthy_workers<G: Generation>(registry: &WorkerRegistry<G>) -> Vec<Arc<dyn Worker>> {
    registry
        .get_all()
        .into_iter()
        .filter(|w| w.is_healthy())
        .collect()
}

async fn run_event_loop<G: Generation>(
    tracker: Weak<WorkerCapacity>,
    registry: Weak<WorkerRegistry<G>>,
    settings: CapacityTrackerSettings,
    events: &mut broadcast::Receiver<WorkerEvent>,
    resync: bool,
) {
    use tokio::sync::broadcast::error::{RecvError, TryRecvError};

    #[cfg(test)]
    if let Some(t) = tracker.upgrade() {
        t.event_loop_ready.notify_one();
    }

    let mut wait_for_event = !resync;
    loop {
        if wait_for_event {
            // Wait for a mutation before taking another snapshot. The initial
            // value was already computed synchronously by `spawn`.
            match events.recv().await {
                Ok(_event) => {}
                Err(RecvError::Lagged(n)) => {
                    tracing::warn!(
                        skipped = n,
                        "WorkerCapacity event task lagged; recomputing from snapshot"
                    );
                }
                Err(RecvError::Closed) => {
                    tracing::info!("WorkerCapacity event task exiting: registry dropped");
                    break;
                }
            }

            // Coalesce a bounded number of the immediately ready events.
            let mut events_closed = false;
            for _ in 1..MAX_COALESCED_EVENTS {
                match events.try_recv() {
                    Ok(_event) => {}
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Lagged(n)) => {
                        tracing::warn!(
                            skipped = n,
                            "WorkerCapacity event task lagged; recomputing from snapshot"
                        );
                    }
                    Err(TryRecvError::Closed) => {
                        events_closed = true;
                        break;
                    }
                }
            }
            if events_closed {
                tracing::info!("WorkerCapacity event task exiting: registry dropped");
                break;
            }
        }
        wait_for_event = true;

        // Upgrade weak refs briefly to do the recompute. If either is gone,
        // exit cleanly. Holding only Weak means the task itself cannot keep
        // the tracker or registry alive.
        let Some(t) = tracker.upgrade() else {
            tracing::info!("WorkerCapacity event task exiting: tracker dropped");
            break;
        };
        let Some(r) = registry.upgrade() else {
            tracing::info!("WorkerCapacity event task exiting: registry dropped");
            break;
        };

        #[cfg(test)]
        assert!(
            !t.panic_before_recompute.swap(false, Ordering::Relaxed),
            "test panic before capacity recompute"
        );

        let workers = healthy_workers(&r);
        #[cfg(test)]
        t.recompute_count.fetch_add(1, Ordering::Relaxed);
        let (new_capacity, new_source) = recompute(&settings, &workers);

        let old_capacity = t.capacity.swap(new_capacity, Ordering::AcqRel);
        let old_source_raw = t.source.swap(new_source as u8, Ordering::AcqRel);
        let new_source_raw = new_source as u8;

        let capacity_changed = new_capacity != old_capacity;
        let source_changed = new_source_raw != old_source_raw;

        if capacity_changed {
            // send() returns Err if there are no subscribers; we don't care.
            // The stored watch value is still updated, so late subscribers
            // see the latest.
            let _ = t.watch_tx.send(new_capacity);
        }

        if capacity_changed || source_changed {
            tracing::info!(
                old = old_capacity,
                new = new_capacity,
                old_source = ?CapacitySource::from_u8(old_source_raw),
                new_source = ?new_source,
                "backend capacity updated"
            );
        }

        // Drop the strong refs before awaiting so the task does not keep
        // the tracker or registry alive while it sleeps.
        drop(t);
        drop(r);
    }
}

/// Compute capacity and source tier from settings + a snapshot of healthy workers.
///
/// Pure function: no I/O, no atomics. Easily unit-testable.
/// Callers are responsible for filtering `workers` to only healthy ones.
pub(super) fn recompute(
    settings: &CapacityTrackerSettings,
    workers: &[Arc<dyn Worker>],
) -> (u16, CapacitySource) {
    // Tier 1: operator override always wins.
    if let Some(n) = settings.override_capacity {
        if n > 0 {
            return (n, CapacitySource::Override);
        }
    }

    let total_workers = workers.len();
    if total_workers == 0 {
        return (
            settings.legacy_max_concurrent_requests,
            CapacitySource::LegacyFallback,
        );
    }

    let mut sum_reported: u32 = 0;
    let mut non_reporters: usize = 0;
    for w in workers {
        match w.max_running_requests() {
            Some(n) => sum_reported = sum_reported.saturating_add(u32::from(n)),
            None => non_reporters += 1,
        }
    }

    if non_reporters == 0 {
        // Tier 2: every worker reported.
        let capped = sum_reported.min(u32::from(u16::MAX)) as u16;
        return (capped, CapacitySource::WorkerReported);
    }

    // Tier 3 (Mixed): reporters contribute reported, non-reporters contribute
    // slots_per_worker. Also covers "zero reporters" since non_reporters > 0 here.
    let from_non_reporters =
        (non_reporters as u32).saturating_mul(u32::from(settings.slots_per_worker));
    let total = sum_reported.saturating_add(from_non_reporters);
    let capped = total.min(u32::from(u16::MAX)) as u16;
    (capped, CapacitySource::Mixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_source_round_trips_through_u8() {
        for src in [
            CapacitySource::LegacyFallback,
            CapacitySource::Override,
            CapacitySource::WorkerReported,
            CapacitySource::Mixed,
        ] {
            let raw = src as u8;
            assert_eq!(CapacitySource::from_u8(raw), src);
        }
    }

    #[test]
    fn test_capacity_source_unknown_u8_decodes_to_legacy() {
        assert_eq!(CapacitySource::from_u8(99), CapacitySource::LegacyFallback);
    }

    #[test]
    fn test_settings_default_has_sensible_values() {
        let s = CapacityTrackerSettings::default();
        assert_eq!(s.override_capacity, None);
        assert_eq!(s.slots_per_worker, 64);
        assert_eq!(s.legacy_max_concurrent_requests, 1024);
    }

    #[test]
    fn test_settings_override_zero_treated_as_none() {
        let s = CapacityTrackerSettings::with_override(0);
        assert!(
            s.override_capacity.is_none(),
            "0 is the sentinel for 'derive'"
        );
    }

    #[test]
    fn test_settings_override_nonzero_preserved() {
        let s = CapacityTrackerSettings::with_override(2048);
        assert_eq!(s.override_capacity, Some(2048));
    }

    use std::collections::HashMap;

    use crate::test_support::BasicWorkerBuilder;

    fn worker_with_capacity(url: &str, reported: Option<u16>) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        if let Some(n) = reported {
            labels.insert("max_running_requests".to_string(), n.to_string());
        }
        Arc::new(BasicWorkerBuilder::new(url).labels(labels).build())
    }

    #[test]
    fn test_recompute_tier1_override_wins_even_with_workers() {
        let settings = CapacityTrackerSettings {
            override_capacity: Some(512),
            ..Default::default()
        };
        let workers = vec![worker_with_capacity("http://w1", Some(256))];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 512);
        assert_eq!(source, CapacitySource::Override);
    }

    #[test]
    fn test_recompute_tier4_no_workers_uses_legacy_fallback() {
        let settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: 256,
            ..Default::default()
        };
        let (capacity, source) = recompute(&settings, &[]);
        assert_eq!(capacity, 256);
        assert_eq!(source, CapacitySource::LegacyFallback);
    }

    #[test]
    fn test_recompute_tier2_all_workers_report() {
        let settings = CapacityTrackerSettings::default();
        let workers = vec![
            worker_with_capacity("http://w1", Some(256)),
            worker_with_capacity("http://w2", Some(128)),
        ];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 384);
        assert_eq!(source, CapacitySource::WorkerReported);
    }

    #[test]
    fn test_recompute_tier3_mixed_reporters_and_non_reporters() {
        let settings = CapacityTrackerSettings {
            slots_per_worker: 64,
            ..Default::default()
        };
        let workers = vec![
            worker_with_capacity("http://w1", Some(256)),
            worker_with_capacity("http://w2", None),
        ];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 256 + 64);
        assert_eq!(source, CapacitySource::Mixed);
    }

    #[test]
    fn test_recompute_tier3_zero_reporters_uses_worker_count() {
        let settings = CapacityTrackerSettings {
            slots_per_worker: 64,
            ..Default::default()
        };
        let workers = vec![
            worker_with_capacity("http://w1", None),
            worker_with_capacity("http://w2", None),
            worker_with_capacity("http://w3", None),
        ];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 3 * 64);
        assert_eq!(source, CapacitySource::Mixed);
    }

    #[test]
    fn test_recompute_saturates_at_u16_max() {
        let settings = CapacityTrackerSettings::default();
        // 1000 workers × 100 slots = 100_000, exceeds u16::MAX (65_535).
        let workers: Vec<_> = (0..1000)
            .map(|i| worker_with_capacity(&format!("http://w{i}"), Some(100)))
            .collect();
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, u16::MAX);
        assert_eq!(source, CapacitySource::WorkerReported);
    }

    #[test]
    fn test_worker_capacity_initial_value_visible_via_current() {
        let tracker = WorkerCapacity::for_test_with_value(777, CapacitySource::Override);
        assert_eq!(tracker.current(), 777);
        assert_eq!(tracker.source(), CapacitySource::Override);
    }

    #[test]
    fn test_worker_capacity_watch_returns_current_value() {
        let tracker = WorkerCapacity::for_test_with_value(123, CapacitySource::Mixed);
        let rx = tracker.watch();
        assert_eq!(*rx.borrow(), 123);
    }

    use std::time::Duration;

    use tokio::time::timeout;

    use crate::{test_support::WorkerRegistry, worker::WorkerStatus};

    #[tokio::test]
    async fn test_event_task_recomputes_on_registered() {
        let registry = Arc::new(WorkerRegistry::new());
        let settings = CapacityTrackerSettings {
            slots_per_worker: 64,
            legacy_max_concurrent_requests: 0,
            ..Default::default()
        };
        let tracker = WorkerCapacity::spawn(registry.clone(), settings);

        // Initial: 0 workers → tier 4 fallback (0).
        assert_eq!(tracker.current(), 0);

        let mut rx = tracker.watch();
        rx.borrow_and_update();

        // Register a worker reporting 256 capacity.
        let mut labels = HashMap::new();
        labels.insert("max_running_requests".to_string(), "256".to_string());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .labels(labels)
                .build(),
        );
        let worker_id = registry.register(worker).expect("registered");
        registry.transition_status(&worker_id, WorkerStatus::Ready);

        // Wait for the watch channel to update.
        let changed = timeout(Duration::from_secs(2), rx.changed()).await;
        assert!(changed.is_ok(), "watch channel did not signal a change");
        assert_eq!(*rx.borrow(), 256);
        assert_eq!(tracker.source(), CapacitySource::WorkerReported);
    }

    #[tokio::test]
    async fn test_event_task_coalesces_ready_event_burst() {
        let registry = Arc::new(WorkerRegistry::new());
        let settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: 0,
            ..Default::default()
        };
        let worker = worker_with_capacity("http://w1", Some(7));
        let worker_id = registry.register(worker).expect("registered");
        let tracker = WorkerCapacity::spawn(registry.clone(), settings);
        let mut rx = tracker.watch();
        rx.borrow_and_update();

        tracker.event_loop_ready.notified().await;
        let startup_recomputes = tracker.recompute_count.load(Ordering::Relaxed);

        for i in 0..1000 {
            let status = if i % 2 == 0 {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            };
            registry.transition_status(&worker_id, status);
        }

        timeout(Duration::from_secs(2), rx.changed())
            .await
            .expect("capacity did not update after event burst")
            .expect("capacity watch closed");
        tokio::task::yield_now().await;

        assert_eq!(tracker.current(), 7);
        assert_eq!(
            (
                startup_recomputes,
                tracker.recompute_count.load(Ordering::Relaxed)
            ),
            (0, 1),
            "the initial snapshot must not be repeated, and a ready burst must be coalesced"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_event_task_resyncs_after_panic_consumes_event() {
        let registry = Arc::new(WorkerRegistry::new());
        let settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: 0,
            ..Default::default()
        };
        let worker = worker_with_capacity("http://w1", Some(7));
        worker.set_status(WorkerStatus::Ready);
        let worker_id = registry.register(worker).expect("registered");
        let tracker = WorkerCapacity::spawn(registry.clone(), settings);
        let mut rx = tracker.watch();
        rx.borrow_and_update();
        assert_eq!(tracker.current(), 7);

        tracker.event_loop_ready.notified().await;
        tracker
            .panic_before_recompute
            .store(true, Ordering::Relaxed);
        registry.transition_status(&worker_id, WorkerStatus::NotReady);

        timeout(Duration::from_secs(2), rx.changed())
            .await
            .expect("capacity stayed stale after the event task restarted")
            .expect("capacity watch closed");
        assert_eq!(tracker.current(), 0);
    }

    #[tokio::test]
    async fn test_event_task_shrinks_capacity_when_worker_goes_unhealthy() {
        // Primary safety property: capacity must shed when a worker stops being
        // healthy, otherwise the gateway over-admits while the backend is hurt.
        let registry = Arc::new(WorkerRegistry::new());
        let settings = CapacityTrackerSettings {
            slots_per_worker: 64,
            legacy_max_concurrent_requests: 0,
            ..Default::default()
        };
        let tracker = WorkerCapacity::spawn(registry.clone(), settings);

        // Register two reporting workers, mark both Ready, wait for both updates.
        let mut rx = tracker.watch();
        rx.borrow_and_update();
        let mk = |url: &str, cap: u16| -> Arc<dyn Worker> {
            let mut labels = HashMap::new();
            labels.insert("max_running_requests".to_string(), cap.to_string());
            Arc::new(BasicWorkerBuilder::new(url).labels(labels).build())
        };
        let id1 = registry.register(mk("http://w1", 256)).expect("registered");
        registry.transition_status(&id1, WorkerStatus::Ready);
        let id2 = registry.register(mk("http://w2", 256)).expect("registered");
        registry.transition_status(&id2, WorkerStatus::Ready);

        // Drain updates until we see 512 (the registrations may coalesce).
        let deadline = Duration::from_secs(2);
        let combined_seen = timeout(deadline, async {
            loop {
                if *rx.borrow_and_update() == 512 {
                    break;
                }
                rx.changed().await.expect("watch not closed");
            }
        })
        .await;
        assert!(
            combined_seen.is_ok(),
            "never observed combined capacity 512"
        );
        assert_eq!(tracker.source(), CapacitySource::WorkerReported);

        // Take one worker unhealthy → capacity must drop.
        registry.transition_status(&id1, WorkerStatus::NotReady);

        let shrunk = timeout(Duration::from_secs(2), async {
            loop {
                rx.changed().await.expect("watch not closed");
                if *rx.borrow() < 512 {
                    break;
                }
            }
        })
        .await;
        assert!(shrunk.is_ok(), "watch did not signal capacity shrink");
        assert_eq!(
            tracker.current(),
            256,
            "only the remaining healthy worker counts"
        );
    }

    #[tokio::test]
    async fn test_spawn_computes_initial_capacity_from_empty_registry() {
        let registry = Arc::new(WorkerRegistry::new());
        let settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: 999,
            ..Default::default()
        };
        let tracker = WorkerCapacity::spawn(registry, settings);

        // Empty registry → tier 4 fallback.
        assert_eq!(tracker.current(), 999);
        assert_eq!(tracker.source(), CapacitySource::LegacyFallback);
    }

    #[tokio::test]
    async fn test_spawn_watch_starts_at_initial_value() {
        let registry = Arc::new(WorkerRegistry::new());
        let settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: 42,
            ..Default::default()
        };
        let tracker = WorkerCapacity::spawn(registry, settings);

        let rx = tracker.watch();
        assert_eq!(*rx.borrow(), 42);
    }
}
