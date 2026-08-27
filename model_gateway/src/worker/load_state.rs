//! Shared, immutable worker-load snapshots.
//!
//! `WorkerMonitor`'s group loops publish per-worker load reports and routing
//! hot paths read them. The publication contract:
//!
//! - **Readers** grab the current snapshot as one `Arc` clone and scan
//!   immutable data: no channel lock is held while scanning, and no map is
//!   copied per request.
//! - **Publishers** rebuild by copy-on-write. The copy is shallow — `Arc`
//!   pointer bumps for keys and values, never a deep `WorkerLoadResponse` or
//!   `String` clone — so a full-fleet rebuild is O(fleet) pointer work at
//!   polling cadence (seconds). That is why one global map is used instead of
//!   per-group shards: the per-publish cost that sharding would save is
//!   pointer-width, and a flat map keeps every reader trivial.
//! - **Provenance fences staleness.** Every entry records the exact worker
//!   incarnation that produced it, and a publish admits a report only while
//!   the registry still holds that incarnation as Ready. A poll that raced a
//!   removal, replacement, or readiness flip is dropped here instead of
//!   resurrecting a dead worker's load; an eviction removes only its own
//!   incarnation's entry, so a same-URL replacement never loses (or
//!   inherits) the predecessor's state.
//! - **Evictions batch.** They queue here and the monitor's debounced
//!   flusher applies the whole backlog in a single rebuild, so removing N
//!   workers costs one publish, not N.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
};

use openai_protocol::worker::{WorkerLoadResponse, WorkerStatus};
use parking_lot::Mutex;
use tokio::sync::{watch, Notify};

use super::{registry::WorkerRegistry, Worker};

/// Subscription handle to the published load snapshots. `borrow().clone()`
/// yields an `Arc<LoadSnapshot>`; the watch guard must not be held across a
/// scan.
pub(crate) type LoadReceiver = watch::Receiver<Arc<LoadSnapshot>>;

/// One worker's published load and the incarnation that produced it.
#[derive(Clone)]
struct LoadEntry {
    load: Arc<WorkerLoadResponse>,
    /// The exact worker the report came from, so an eviction removes only
    /// its own incarnation's entry and never a same-URL replacement's fresh
    /// one.
    source: Weak<dyn Worker>,
}

/// An immutable point-in-time view of every published worker load.
#[derive(Default)]
pub(crate) struct LoadSnapshot {
    entries: HashMap<Arc<str>, LoadEntry>,
}

impl std::fmt::Debug for LoadSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadSnapshot")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl LoadSnapshot {
    pub(crate) fn get(&self, url: &str) -> Option<&WorkerLoadResponse> {
        self.entries.get(url).map(|entry| &*entry.load)
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, url: &str) -> bool {
        self.entries.contains_key(url)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Test-only: a snapshot with the given loads and no live sources.
    #[cfg(test)]
    pub(crate) fn from_loads_for_test(loads: Vec<(String, WorkerLoadResponse)>) -> Arc<Self> {
        let entries = loads
            .into_iter()
            .map(|(url, load)| {
                let dangling: Weak<super::worker::BasicWorker> = Weak::new();
                let entry = LoadEntry {
                    load: Arc::new(load),
                    source: dangling,
                };
                (Arc::<str>::from(url.as_str()), entry)
            })
            .collect();
        Arc::new(Self { entries })
    }
}

/// Publication side of the shared load view. Owned by `WorkerMonitor`.
pub(crate) struct LoadState {
    registry: Arc<WorkerRegistry>,
    tx: watch::Sender<Arc<LoadSnapshot>>,
    pending_evictions: Mutex<Vec<Arc<dyn Worker>>>,
    /// Wakes the monitor's debounced eviction flusher.
    eviction_notify: Notify,
    /// Snapshot publications, so tests can assert eviction batching stays
    /// bounded.
    publishes: AtomicU64,
}

impl LoadState {
    pub(crate) fn new(registry: Arc<WorkerRegistry>) -> Self {
        let (tx, _rx) = watch::channel(Arc::new(LoadSnapshot::default()));
        Self {
            registry,
            tx,
            pending_evictions: Mutex::new(Vec::new()),
            eviction_notify: Notify::new(),
            publishes: AtomicU64::new(0),
        }
    }

    pub(crate) fn subscribe(&self) -> LoadReceiver {
        self.tx.subscribe()
    }

    /// The current snapshot — one `Arc` clone, no lock held afterwards.
    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Arc<LoadSnapshot> {
        self.tx.borrow().clone()
    }

    /// Publish one group's polling tick: prune every entry for the group's
    /// URLs, then insert the fresh reports that pass the incarnation fence.
    ///
    /// The fence admits a report only while the registry still holds the
    /// exact worker (`Arc` identity) that produced it, and holds it Ready. A
    /// report from a worker that was removed, replaced, or marked unready
    /// between poll start and publish is dropped; entries that were already
    /// published stay the eviction path's responsibility, which always runs
    /// after the registry mutation and therefore after this fence would have
    /// started failing.
    pub(crate) fn publish_group(
        &self,
        group_urls: &[String],
        fresh: Vec<(Arc<dyn Worker>, Arc<WorkerLoadResponse>)>,
    ) {
        let admitted: Vec<(Arc<str>, LoadEntry)> = fresh
            .into_iter()
            .filter(|(worker, _)| self.is_current_incarnation(worker))
            .map(|(worker, load)| {
                let entry = LoadEntry {
                    load,
                    source: Arc::downgrade(&worker),
                };
                (Arc::<str>::from(worker.url()), entry)
            })
            .collect();

        self.rebuild(move |entries| {
            let mut changed = false;
            for url in group_urls {
                changed |= entries.remove(url.as_str()).is_some();
            }
            for (url, entry) in admitted {
                entries.insert(url, entry);
                changed = true;
            }
            changed
        });
    }

    fn is_current_incarnation(&self, worker: &Arc<dyn Worker>) -> bool {
        self.registry
            .get_by_url(worker.url())
            .is_some_and(|current| {
                Arc::ptr_eq(&current, worker) && current.status() == WorkerStatus::Ready
            })
    }

    /// Queue a worker's eviction and wake the flusher. The snapshot changes
    /// on the next [`Self::apply_pending_evictions`] batch.
    pub(crate) fn enqueue_eviction(&self, worker: Arc<dyn Worker>) {
        self.pending_evictions.lock().push(worker);
        self.eviction_notify.notify_one();
    }

    /// Resolves once at least one eviction has been queued since the last
    /// [`Self::apply_pending_evictions`].
    pub(crate) async fn eviction_wakeup(&self) {
        self.eviction_notify.notified().await;
    }

    /// Apply every queued eviction in one rebuild. Returns the workers whose
    /// own entry was actually removed (the metrics sentinel set): a same-URL
    /// replacement that already republished keeps its fresh entry, and its
    /// series simply continues under the same URL.
    pub(crate) fn apply_pending_evictions(&self) -> Vec<Arc<dyn Worker>> {
        let pending: Vec<Arc<dyn Worker>> = std::mem::take(&mut *self.pending_evictions.lock());
        if pending.is_empty() {
            return Vec::new();
        }

        let mut removed: Vec<Arc<dyn Worker>> = Vec::new();
        self.rebuild(|entries| {
            let mut changed = false;
            for worker in &pending {
                let evict = entries.get(worker.url()).is_some_and(|entry| {
                    match entry.source.upgrade() {
                        // The entry belongs to the evicted incarnation.
                        Some(source) => Arc::ptr_eq(&source, worker),
                        // The producer is gone entirely; nothing live owns
                        // this entry.
                        None => true,
                    }
                });
                if evict {
                    entries.remove(worker.url());
                    removed.push(Arc::clone(worker));
                    changed = true;
                }
            }
            changed
        });
        removed
    }

    /// Drop every published entry and any queued evictions (monitor
    /// shutdown and lag-recovery rebuilds).
    pub(crate) fn clear(&self) {
        self.pending_evictions.lock().clear();
        self.rebuild(|entries| {
            let changed = !entries.is_empty();
            entries.clear();
            changed
        });
    }

    /// Snapshot publications so far. Test observability for the batching
    /// guarantees.
    #[cfg(test)]
    pub(crate) fn publish_count(&self) -> u64 {
        self.publishes.load(Ordering::Relaxed)
    }

    /// Copy-on-write rebuild: clone the entry map shallowly (pointer bumps),
    /// let `mutate` edit it, and publish a fresh `Arc` only when something
    /// changed. Publishers serialize on the watch channel's internal lock;
    /// readers holding older `Arc`s are unaffected.
    fn rebuild(&self, mutate: impl FnOnce(&mut HashMap<Arc<str>, LoadEntry>) -> bool) {
        self.tx.send_if_modified(|snapshot| {
            let mut entries = snapshot.entries.clone();
            let changed = mutate(&mut entries);
            if changed {
                *snapshot = Arc::new(LoadSnapshot { entries });
                self.publishes.fetch_add(1, Ordering::Relaxed);
            }
            changed
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;

    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, SchedulerLoadSnapshot, WorkerType},
    };

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode};

    fn ready_worker(url: &str, model: &str) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .model(ModelCard::new(model))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    }

    /// A load report distinguishable by its waiting-token count.
    fn load(waiting: i32) -> Arc<WorkerLoadResponse> {
        Arc::new(WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_uncached_tokens: waiting,
                ..Default::default()
            }],
            ..Default::default()
        })
    }

    fn waiting_of(snapshot: &LoadSnapshot, url: &str) -> Option<i64> {
        snapshot.get(url).map(|l| l.total_waiting_uncached_tokens())
    }

    fn setup() -> (Arc<WorkerRegistry>, LoadState) {
        let registry = Arc::new(WorkerRegistry::new());
        let state = LoadState::new(Arc::clone(&registry));
        (registry, state)
    }

    #[test]
    fn readers_scan_an_old_snapshot_while_a_new_one_publishes() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&worker)).unwrap();

        state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);
        let held = state.snapshot();

        state.publish_group(&[], vec![(Arc::clone(&worker), load(2))]);

        // The held snapshot is immutable and stays scannable as-is...
        assert_eq!(waiting_of(&held, "http://w1:8080"), Some(1));
        // ...while a fresh grab observes the newer publication.
        assert_eq!(waiting_of(&state.snapshot(), "http://w1:8080"), Some(2));
    }

    #[test]
    fn concurrent_group_updates_are_both_retained() {
        let (registry, state) = setup();
        let w1 = ready_worker("http://w1:8080", "m1");
        let w2 = ready_worker("http://w2:8080", "m2");
        registry.register(Arc::clone(&w1)).unwrap();
        registry.register(Arc::clone(&w2)).unwrap();

        let state = Arc::new(state);
        let barrier = Arc::new(Barrier::new(2));
        let threads: Vec<_> = [w1, w2]
            .into_iter()
            .map(|worker| {
                let state = Arc::clone(&state);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let group_urls = vec![worker.url().to_string()];
                    barrier.wait();
                    for round in 0..100 {
                        state.publish_group(&group_urls, vec![(Arc::clone(&worker), load(round))]);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        // Neither group's stream of publications lost the other's entry.
        let snapshot = state.snapshot();
        assert_eq!(waiting_of(&snapshot, "http://w1:8080"), Some(99));
        assert_eq!(waiting_of(&snapshot, "http://w2:8080"), Some(99));
    }

    #[test]
    fn eviction_removes_the_workers_own_entry() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&worker)).unwrap();
        state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);

        state.enqueue_eviction(Arc::clone(&worker));
        let removed = state.apply_pending_evictions();

        assert_eq!(removed.len(), 1);
        assert!(Arc::ptr_eq(&removed[0], &worker));
        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn stale_poll_cannot_resurrect_a_removed_worker() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        let id = registry.register(Arc::clone(&worker)).unwrap();
        state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);

        registry.remove(&id);
        state.enqueue_eviction(Arc::clone(&worker));
        state.apply_pending_evictions();
        assert!(state.snapshot().is_empty());

        // An in-flight poll that started before the removal lands late: the
        // incarnation fence drops it instead of re-inserting the dead worker
        // (which the pre-snapshot code did, permanently — the group's next
        // tick no longer listed the URL to prune).
        state.publish_group(&[], vec![(Arc::clone(&worker), load(2))]);
        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn unready_worker_reports_are_fenced() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&worker)).unwrap();

        worker.set_status(WorkerStatus::NotReady);
        state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);

        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn same_url_replacement_does_not_inherit_the_old_incarnations_load() {
        let (registry, state) = setup();
        let old = ready_worker("http://w1:8080", "m");
        let old_id = registry.register(Arc::clone(&old)).unwrap();
        state.publish_group(&[], vec![(Arc::clone(&old), load(1))]);

        // Replace: same URL, new incarnation.
        registry.remove(&old_id);
        let new = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&new)).unwrap();

        // The old incarnation's eviction flushes and its entry vanishes —
        // the replacement starts with no inherited load...
        state.enqueue_eviction(Arc::clone(&old));
        state.apply_pending_evictions();
        assert!(state.snapshot().is_empty());

        // ...and once the replacement has republished, a LATE flush of the
        // old incarnation's eviction must not tear down the fresh entry.
        state.publish_group(&[], vec![(Arc::clone(&new), load(2))]);
        state.enqueue_eviction(Arc::clone(&old));
        let removed = state.apply_pending_evictions();
        assert!(
            removed.is_empty(),
            "the replacement's entry is not the old worker's"
        );
        assert_eq!(waiting_of(&state.snapshot(), "http://w1:8080"), Some(2));
    }

    #[test]
    fn a_large_removal_batch_causes_one_rebuild() {
        let (registry, state) = setup();
        let workers: Vec<Arc<dyn Worker>> = (0..50)
            .map(|i| {
                let worker = ready_worker(&format!("http://w{i}:8080"), "m");
                registry.register(Arc::clone(&worker)).unwrap();
                worker
            })
            .collect();
        let fresh = workers
            .iter()
            .map(|w| (Arc::clone(w), load(1)))
            .collect::<Vec<_>>();
        state.publish_group(&[], fresh);
        assert_eq!(state.snapshot().len(), 50);

        for worker in &workers {
            state.enqueue_eviction(Arc::clone(worker));
        }
        let before = state.publish_count();
        let removed = state.apply_pending_evictions();

        assert_eq!(removed.len(), 50);
        assert!(state.snapshot().is_empty());
        assert_eq!(
            state.publish_count(),
            before + 1,
            "a 50-worker eviction batch must be one snapshot rebuild"
        );
    }

    #[test]
    fn clear_drops_entries_and_queued_evictions() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&worker)).unwrap();
        state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);
        state.enqueue_eviction(Arc::clone(&worker));

        state.clear();

        assert!(state.snapshot().is_empty());
        assert!(state.apply_pending_evictions().is_empty());
    }
}
