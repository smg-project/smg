//! Aggregate backend capacity tracking.
//!
//! Capacity is a single fleet-wide scalar (summed across every healthy
//! worker, with no model dimension). That is correct only for
//! single-model fleets: with multiple models, an idle model's slots
//! inflate other models' admission budgets (smg-project/smg#2069). The
//! tracker logs a one-time warning when the healthy fleet reports more
//! than one model id.

use std::sync::{
    atomic::{AtomicBool, AtomicU16, AtomicU8, Ordering},
    Arc, Weak,
};

use futures::FutureExt as _;
use tokio::sync::watch;

use super::{registry::WorkerRegistry, Worker};

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
    /// One-shot latch for the multi-model warning (#2069).
    warned_multi_model: AtomicBool,
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
            warned_multi_model: AtomicBool::new(false),
        })
    }

    /// Log once at startup when the healthy fleet serves more than one
    /// model id.
    ///
    /// Capacity is a fleet-wide scalar, so multi-model admission limits are
    /// incorrect (#2069): an idle model's slots mask a saturated model's
    /// queue. Startup-only by design — the check is O(fleet size), which is
    /// not acceptable in the worker-event loop.
    fn warn_once_if_multi_model(&self, workers: &[Arc<dyn Worker>]) {
        if self.warned_multi_model.load(Ordering::Relaxed) {
            return;
        }
        let models = distinct_model_count(workers);
        if models > 1 {
            self.warned_multi_model.store(true, Ordering::Relaxed);
            tracing::warn!(
                models,
                "priority scheduler capacity is fleet-wide: an idle model's slots \
                 inflate other models' admission budgets (#2069). Mitigations: \
                 --worker-capacity-override, or one gateway per model."
            );
        }
    }

    /// Construct a `WorkerCapacity`, compute the initial value
    /// synchronously, and spawn the supervised event-loop task.
    ///
    /// The task holds only `Weak` references to the registry and tracker
    /// (and an owned copy of the settings). When the caller drops their
    /// last `Arc<WorkerCapacity>` or the registry is dropped, the task
    /// exits on the next iteration — no Arc cycle.
    pub fn spawn(registry: Arc<WorkerRegistry>, settings: CapacityTrackerSettings) -> Arc<Self> {
        // Initial compute: synchronous over the current registry state,
        // so callers see a valid `current()` immediately after spawn.
        let workers = healthy_workers(&registry);
        let (initial_capacity, initial_source) = recompute(&settings, &workers);

        let (watch_tx, _initial_rx) = watch::channel(initial_capacity);

        let this = Arc::new(Self {
            capacity: AtomicU16::new(initial_capacity),
            source: AtomicU8::new(initial_source as u8),
            watch_tx,
            warned_multi_model: AtomicBool::new(false),
        });
        this.warn_once_if_multi_model(&workers);

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
            loop {
                let result = std::panic::AssertUnwindSafe(run_event_loop(
                    weak_tracker.clone(),
                    weak_registry.clone(),
                    task_settings.clone(),
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
fn healthy_workers(registry: &WorkerRegistry) -> Vec<Arc<dyn Worker>> {
    registry
        .get_all()
        .into_iter()
        .filter(|w| w.is_healthy())
        .collect()
}

/// Count distinct advertised model ids across the given workers.
///
/// Every model card counts, not just the primary — a worker serving
/// several models must not read as single-model. Used by the multi-model
/// warning: capacity is a fleet-wide scalar, so a fleet reporting more
/// than one model id gets incorrect per-model admission limits (#2069).
pub(super) fn distinct_model_count(workers: &[Arc<dyn Worker>]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for w in workers {
        let cards = w.models();
        if cards.is_empty() {
            // No model cards (e.g. wildcard or label-only worker) — fall back
            // to the primary id, matching WorkerRegistry::worker_model_ids().
            seen.insert(w.model_id().to_string());
        } else {
            seen.extend(cards.into_iter().map(|card| card.id));
        }
    }
    seen.len()
}

async fn run_event_loop(
    tracker: Weak<WorkerCapacity>,
    registry: Weak<WorkerRegistry>,
    settings: CapacityTrackerSettings,
) {
    use tokio::sync::broadcast::error::RecvError;

    // Acquire the receiver up front. If the registry is already gone we
    // have nothing to do; the receiver does not keep the registry alive.
    let mut events = match registry.upgrade() {
        Some(r) => r.subscribe_events(),
        None => return,
    };

    loop {
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

        let workers = healthy_workers(&r);
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

        // Wait for the next worker lifecycle event. `RecvError::Closed`
        // fires when the registry is fully dropped (sender gone).
        match events.recv().await {
            Ok(_event) => continue,
            Err(RecvError::Lagged(n)) => {
                tracing::warn!(
                    skipped = n,
                    "WorkerCapacity event task lagged; recomputing from snapshot"
                );
                continue;
            }
            Err(RecvError::Closed) => {
                tracing::info!("WorkerCapacity event task exiting: registry dropped");
                break;
            }
        }
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

    use crate::worker::BasicWorkerBuilder;

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

    use openai_protocol::worker::WorkerStatus;
    use tokio::time::timeout;

    use crate::worker::WorkerRegistry;

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

    fn worker_with_model(url: &str, model: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(openai_protocol::model_card::ModelCard::new(model))
                .build(),
        )
    }

    #[test]
    fn distinct_model_count_counts_unique_models() {
        let workers = vec![
            worker_with_model("http://w1", "llama-7b"),
            worker_with_model("http://w2", "llama-7b"),
            worker_with_model("http://w3", "llama-70b"),
        ];
        assert_eq!(distinct_model_count(&workers), 2);
    }

    #[test]
    fn distinct_model_count_single_model_fleet() {
        let workers = vec![
            worker_with_model("http://w1", "llama-7b"),
            worker_with_model("http://w2", "llama-7b"),
        ];
        assert_eq!(distinct_model_count(&workers), 1);
    }

    #[test]
    fn distinct_model_count_counts_every_advertised_model_card() {
        // One worker can advertise multiple model cards; counting only the
        // primary id would undercount and suppress the warning.
        let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
            "url": "http://w1",
            "models": [{"id": "llama-7b"}, {"id": "llama-70b"}],
        }))
        .expect("multi-model spec");
        let workers =
            vec![Arc::new(BasicWorkerBuilder::from_spec(spec).build()) as Arc<dyn Worker>];
        assert_eq!(distinct_model_count(&workers), 2);
    }

    #[test]
    fn distinct_model_count_falls_back_to_primary_model_id_when_no_cards() {
        // Label-only workers (no model cards) must still count — matches the
        // WorkerRegistry::worker_model_ids() fallback.
        let with_label = |url: &str, model: &str| {
            let mut labels = HashMap::new();
            labels.insert("model_id".to_string(), model.to_string());
            Arc::new(BasicWorkerBuilder::new(url).labels(labels).build()) as Arc<dyn Worker>
        };
        let workers = vec![
            with_label("http://w1", "llama-7b"),
            with_label("http://w2", "llama-70b"),
        ];
        assert_eq!(distinct_model_count(&workers), 2);
    }

    #[tokio::test]
    async fn spawn_latches_warning_for_multi_model_registry() {
        let registry = Arc::new(WorkerRegistry::new());
        let id1 = registry
            .register(worker_with_model("http://w1", "llama-7b"))
            .expect("registered");
        registry.transition_status(&id1, WorkerStatus::Ready);
        let id2 = registry
            .register(worker_with_model("http://w2", "llama-70b"))
            .expect("registered");
        registry.transition_status(&id2, WorkerStatus::Ready);

        let tracker = WorkerCapacity::spawn(registry, CapacityTrackerSettings::default());
        assert!(tracker.warned_multi_model.load(Ordering::Relaxed));
    }

    #[test]
    fn warn_once_if_multi_model_sets_flag_once() {
        let tracker = WorkerCapacity::for_test_with_value(64, CapacitySource::Mixed);
        let workers = vec![
            worker_with_model("http://w1", "llama-7b"),
            worker_with_model("http://w2", "llama-70b"),
        ];
        assert!(!tracker.warned_multi_model.load(Ordering::Relaxed));
        tracker.warn_once_if_multi_model(&workers);
        assert!(tracker.warned_multi_model.load(Ordering::Relaxed));
        // Second call is a no-op regardless of input.
        tracker.warn_once_if_multi_model(&[]);
        assert!(tracker.warned_multi_model.load(Ordering::Relaxed));
    }

    #[tracing_test::traced_test]
    #[test]
    fn warn_once_if_multi_model_logs_for_multi_model_fleet() {
        let tracker = WorkerCapacity::for_test_with_value(64, CapacitySource::Mixed);
        let workers = vec![
            worker_with_model("http://w1", "llama-7b"),
            worker_with_model("http://w2", "llama-70b"),
        ];
        tracker.warn_once_if_multi_model(&workers);
        assert!(logs_contain("fleet-wide"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn warn_once_if_multi_model_quiet_for_single_model() {
        let tracker = WorkerCapacity::for_test_with_value(64, CapacitySource::Mixed);
        let workers = vec![worker_with_model("http://w1", "llama-7b")];
        tracker.warn_once_if_multi_model(&workers);
        assert!(!logs_contain("fleet-wide"));
    }
}
