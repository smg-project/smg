//! Shared, immutable worker-load snapshots.
//!
//! `WorkerMonitor` publishes per-worker load reports here (a group's poll
//! tick, a window of pushed records) and the routing hot paths read them:
//! this is the one feed the load-aware policies score on, next to the
//! dispatch credit they book themselves. The publication contract:
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
//! - **Every report carries its sample instant.** An entry records when the
//!   engine's state behind it was sampled (the report's own instant, else
//!   the publish instant), so a policy that books dispatches locally can
//!   release the ones a report already reflects when it reads the report.

use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
    time::Instant,
};

use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use tokio::sync::{watch, Notify};

#[cfg(any(test, feature = "test-util"))]
use crate::identity::basic::BasicWorker;
use crate::{
    identity::basic::Worker,
    registry::{Generation, WorkerRegistry},
    worker::{WorkerLoadResponse, WorkerStatus},
};

/// Subscription handle to the published load snapshots. `borrow().clone()`
/// yields an `Arc<LoadSnapshot>`; the watch guard must not be held across a
/// scan.
pub type LoadReceiver = watch::Receiver<Arc<LoadSnapshot>>;

/// A worker of an eviction batch's sentinel set, with the load of its own
/// entry where the batch removed it.
pub(crate) type Evicted = (Arc<dyn Worker>, Option<Arc<WorkerLoadResponse>>);

/// A row a publish took away where no fresh row of the same worker stands
/// at the URL behind it, with that worker.
pub(crate) type Replaced = (Arc<dyn Worker>, Arc<WorkerLoadResponse>);

/// One worker's published load and the incarnation that produced it.
#[derive(Clone)]
struct LoadEntry {
    load: Arc<WorkerLoadResponse>,
    /// When the engine's state behind the report was sampled, on this
    /// process's clock: the report's own instant, else the publish instant
    /// (the latest the sample can have been taken).
    sampled_at: Instant,
    /// The exact worker the report came from, so an eviction removes only
    /// its own incarnation's entry and never a same-URL replacement's fresh
    /// one.
    source: Weak<dyn Worker>,
}

/// An immutable point-in-time view of every published worker load.
#[derive(Default)]
pub struct LoadSnapshot {
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
    pub fn get(&self, url: &str) -> Option<&WorkerLoadResponse> {
        self.entries.get(url).map(|entry| &*entry.load)
    }

    /// A worker's report with the instant its sample was taken.
    pub fn report(&self, url: &str) -> Option<(&WorkerLoadResponse, Instant)> {
        self.entries
            .get(url)
            .map(|entry| (&*entry.load, entry.sampled_at))
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

    /// Test-only: a snapshot with the given loads and no live sources, each
    /// report sampled now unless it says otherwise.
    #[cfg(any(test, feature = "test-util"))]
    pub fn from_loads_for_test(loads: Vec<(String, WorkerLoadResponse)>) -> Arc<Self> {
        let published_at = Instant::now();
        let entries = loads
            .into_iter()
            .map(|(url, load)| {
                let dangling: Weak<BasicWorker> = Weak::new();
                let entry = LoadEntry {
                    sampled_at: load.sampled_at.unwrap_or(published_at),
                    load: Arc::new(load),
                    source: dangling,
                };
                (Arc::<str>::from(url.as_str()), entry)
            })
            .collect();
        Arc::new(Self { entries })
    }
}

/// Publisher standing in for the monitor in tests and benchmarks: hands the
/// feed to a reader through `attach` and publishes each map of reports as the
/// whole snapshot.
#[cfg(any(test, feature = "test-util"))]
pub struct TestFeed(watch::Sender<Arc<LoadSnapshot>>);

#[cfg(any(test, feature = "test-util"))]
impl TestFeed {
    pub fn attached_to(attach: impl FnOnce(Option<LoadReceiver>)) -> Self {
        let (tx, rx) = watch::channel(Arc::new(LoadSnapshot::default()));
        attach(Some(rx));
        Self(tx)
    }

    /// Publish `loads` as the current snapshot; a report without a sample
    /// instant is sampled now.
    pub fn publish(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        let loads = loads
            .iter()
            .map(|(url, load)| (url.clone(), load.clone()))
            .collect();
        self.0
            .send_replace(LoadSnapshot::from_loads_for_test(loads));
    }
}

/// One DP rank with the given queued uncached tokens, KV utilization and
/// live throughput; one request is running, so a rate gauge counts.
#[cfg(any(test, feature = "test-util"))]
pub fn one_rank_load(
    num_waiting_uncached_tokens: i32,
    token_usage: f64,
    gen_throughput: f64,
) -> WorkerLoadResponse {
    WorkerLoadResponse {
        timestamp: String::new(),
        dp_rank_count: 1,
        loads: vec![crate::worker::SchedulerLoadSnapshot {
            dp_rank: 0,
            num_running_reqs: 1,
            num_waiting_uncached_tokens,
            token_usage,
            gen_throughput,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// One DP rank reporting a queue depth with no token count for it: the shape
/// of a backend scored from Prometheus gauges, which have no waiting-token
/// equivalent.
#[cfg(any(test, feature = "test-util"))]
pub fn one_rank_load_reqs_only(
    num_waiting_reqs: i32,
    token_usage: f64,
    gen_throughput: f64,
) -> WorkerLoadResponse {
    let mut load = one_rank_load(0, token_usage, gen_throughput);
    load.loads[0].num_waiting_reqs = num_waiting_reqs;
    load
}

/// Publication side of the shared load view. Owned by `WorkerMonitor`.
pub(crate) struct LoadState<G> {
    registry: Arc<WorkerRegistry<G>>,
    tx: watch::Sender<Arc<LoadSnapshot>>,
    pending_evictions: Mutex<Vec<Arc<dyn Worker>>>,
    /// Wakes the monitor's debounced eviction flusher.
    eviction_notify: Notify,
    /// Snapshot publications, so tests can assert eviction batching stays
    /// bounded.
    publishes: AtomicU64,
}

impl<G: Generation> LoadState<G> {
    pub(crate) fn new(registry: Arc<WorkerRegistry<G>>) -> Self {
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
    ///
    /// Returns each row the publish took away where no fresh row of the
    /// same worker stands at the URL behind it, with that worker while it
    /// lives: a row that went under its successor's, and a row pruned with
    /// no fresh one. Whether such a row's worker has left is the caller's
    /// to ask the registry: the read here is by pointer alone.
    #[must_use = "a returned row's series end only where the caller ends them"]
    pub(crate) fn publish_group(
        &self,
        group_urls: &[String],
        fresh: Vec<(Arc<dyn Worker>, Arc<WorkerLoadResponse>)>,
    ) -> Vec<Replaced> {
        let published_at = Instant::now();
        let admitted: Vec<(Arc<str>, LoadEntry)> = fresh
            .into_iter()
            .filter(|(worker, _)| self.is_current_incarnation(worker))
            .map(|(worker, load)| {
                let entry = LoadEntry {
                    sampled_at: load.sampled_at.unwrap_or(published_at),
                    load,
                    source: Arc::downgrade(&worker),
                };
                (Arc::<str>::from(worker.url()), entry)
            })
            .collect();

        let mut replaced = Vec::new();
        let ended = &mut replaced;
        self.rebuild(move |entries| {
            let mut changed = false;
            {
                // A URL with a fresh row keeps its entry for the insert
                // below, which reads whose row it replaces. The fresh rows
                // come in the URLs' order, less the ones that failed, so
                // one cursor finds a URL's. Where the cursor misses, the
                // set of the fresh URLs says whether the rows are out of
                // that order (an engine's ranks are gathered) or the URL
                // has none.
                let mut next = 0;
                let mut out_of_order: Option<FxHashSet<&str>> = None;
                for url in group_urls {
                    if admitted
                        .get(next)
                        .is_some_and(|(fresh, _)| **fresh == **url)
                    {
                        next += 1;
                        continue;
                    }
                    let fresh_urls = out_of_order.get_or_insert_with(|| {
                        admitted.iter().map(|(fresh, _)| &**fresh).collect()
                    });
                    if fresh_urls.contains(url.as_str()) {
                        continue;
                    }
                    if let Some(old) = entries.remove(url.as_str()) {
                        changed = true;
                        ended.extend(old.source.upgrade().map(|worker| (worker, old.load)));
                    }
                }
            }
            for (url, entry) in admitted {
                match entries.entry(url) {
                    Entry::Occupied(mut slot) => {
                        let old = slot.insert(entry);
                        if !Weak::ptr_eq(&old.source, &slot.get().source) {
                            ended.extend(old.source.upgrade().map(|worker| (worker, old.load)));
                        }
                    }
                    Entry::Vacant(slot) => {
                        slot.insert(entry);
                    }
                }
                changed = true;
            }
            changed
        });
        replaced
    }

    /// Whether `worker` is the `Ready` worker the registry holds at its URL
    /// now, by pointer. One test for every place that asks it: the publish
    /// fence and a tick's series, the pushers a reset keeps, the eviction
    /// batch, the leave a status event tells, a held report's flush and
    /// the sentinel's guard.
    pub(crate) fn is_current_incarnation(&self, worker: &Arc<dyn Worker>) -> bool {
        self.registry
            .get_by_url(worker.url())
            .is_some_and(|current| {
                Arc::ptr_eq(&current, worker) && current.status() == WorkerStatus::Ready
            })
    }

    /// The row at `worker`'s URL, where it came from another worker that
    /// is the registry's `Ready` worker now: that worker and its row's
    /// load. Such a worker is a live writer of the engine series that load
    /// carries, under its own models: a poll writes them with the row it
    /// publishes, and a pushed record ahead of its row.
    pub(crate) fn another_live_workers_row(
        &self,
        worker: &Arc<dyn Worker>,
    ) -> Option<(Arc<dyn Worker>, Arc<WorkerLoadResponse>)> {
        let (source, load) = self
            .tx
            .borrow()
            .entries
            .get(worker.url())
            .and_then(|entry| {
                let source = entry.source.upgrade()?;
                Some((source, Arc::clone(&entry.load)))
            })?;
        (!Arc::ptr_eq(&source, worker) && self.is_current_incarnation(&source))
            .then_some((source, load))
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

    /// Apply every queued eviction in one rebuild. Returns the metrics
    /// sentinel set: the workers whose own entry was removed, each with
    /// that entry's load, whose ranks the sentinel covers, and those whose
    /// URL holds another worker's entry, with none. A same-URL replacement
    /// that already republished keeps its fresh entry. The entry of the
    /// worker it replaced went under it, so that worker's series end here
    /// too, but for the ones the replacement writes (the collector's
    /// guard). A worker that is the registry's `Ready` worker again is not
    /// evicted: it left and came back ahead of the batch. Its row stands
    /// until what it publishes next replaces it, and it is not in the
    /// sentinel set.
    pub(crate) fn apply_pending_evictions(&self) -> Vec<Evicted> {
        let pending: Vec<Arc<dyn Worker>> = std::mem::take(&mut *self.pending_evictions.lock());
        if pending.is_empty() {
            return Vec::new();
        }

        let mut ended = Vec::new();
        self.rebuild(|entries| {
            let mut changed = false;
            for worker in &pending {
                // Read under the publishers' lock. A worker that leaves
                // behind this read queues an eviction of its own.
                if self.is_current_incarnation(worker) {
                    continue;
                }
                let own = entries.get(worker.url()).map(|entry| {
                    match entry.source.upgrade() {
                        // The entry belongs to the evicted incarnation.
                        Some(source) => Arc::ptr_eq(&source, worker),
                        // The producer is gone entirely; nothing live owns
                        // this entry.
                        None => true,
                    }
                });
                match own {
                    Some(true) => {
                        let row = entries.remove(worker.url()).map(|entry| entry.load);
                        ended.push((Arc::clone(worker), row));
                        changed = true;
                    }
                    // Another worker's entry stands at the URL: this
                    // worker's own went under it. The entry stays, and
                    // the worker joins the sentinel set.
                    Some(false) => ended.push((Arc::clone(worker), None)),
                    None => {}
                }
            }
            changed
        });
        ended
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

    use super::*;
    use crate::{
        test_support::{ready_worker, Members, WorkerRegistry},
        worker::SchedulerLoadSnapshot,
    };

    /// The load state over the gateway's registry.
    type LoadState = super::LoadState<Members>;

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

        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);
        let held = state.snapshot();

        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(2))]);

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
                        let _ = state
                            .publish_group(&group_urls, vec![(Arc::clone(&worker), load(round))]);
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
        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);

        worker.set_status(WorkerStatus::NotReady);
        state.enqueue_eviction(Arc::clone(&worker));
        let removed = state.apply_pending_evictions();

        assert_eq!(removed.len(), 1);
        assert!(Arc::ptr_eq(&removed[0].0, &worker));
        assert!(removed[0].1.is_some(), "with the row the batch removed");
        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn stale_poll_cannot_resurrect_a_removed_worker() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        let id = registry.register(Arc::clone(&worker)).unwrap();
        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);

        registry.remove(&id);
        state.enqueue_eviction(Arc::clone(&worker));
        state.apply_pending_evictions();
        assert!(state.snapshot().is_empty());

        // An in-flight poll that started before the removal lands late: the
        // incarnation fence drops it instead of re-inserting the dead worker
        // (which the pre-snapshot code did, permanently — the group's next
        // tick no longer listed the URL to prune).
        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(2))]);
        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn unready_worker_reports_are_fenced() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&worker)).unwrap();

        worker.set_status(WorkerStatus::NotReady);
        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);

        assert!(state.snapshot().is_empty());
    }

    #[test]
    fn same_url_replacement_does_not_inherit_the_old_incarnations_load() {
        let (registry, state) = setup();
        let old = ready_worker("http://w1:8080", "m");
        let old_id = registry.register(Arc::clone(&old)).unwrap();
        let _ = state.publish_group(&[], vec![(Arc::clone(&old), load(1))]);

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
        let _ = state.publish_group(&[], vec![(Arc::clone(&new), load(2))]);
        state.enqueue_eviction(Arc::clone(&old));
        let ended = state.apply_pending_evictions();
        assert!(
            ended.len() == 1 && Arc::ptr_eq(&ended[0].0, &old) && ended[0].1.is_none(),
            "the old worker's entry went under the replacement's: its series end"
        );
        assert_eq!(waiting_of(&state.snapshot(), "http://w1:8080"), Some(2));
    }

    /// A group publish checks the incarnation by identity, not by URL: a
    /// late poll of a worker that was replaced under its own URL publishes
    /// nothing, and the replacement's poll publishes its own.
    #[test]
    fn a_late_poll_of_a_replaced_worker_is_fenced_by_its_identity() {
        let (registry, state) = setup();
        let old = ready_worker("http://w1:8080", "m");
        let old_id = registry.register(Arc::clone(&old)).unwrap();
        registry.remove(&old_id);
        let new = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&new)).unwrap();

        let _ = state.publish_group(&[], vec![(Arc::clone(&old), load(1))]);
        assert!(state.snapshot().is_empty(), "the old worker's poll is late");

        let _ = state.publish_group(&[], vec![(Arc::clone(&new), load(2))]);
        assert_eq!(waiting_of(&state.snapshot(), "http://w1:8080"), Some(2));
    }

    /// A publish hands back the row it took from a worker whose fresh row
    /// does not stand at the URL behind it: one a successor's row replaced,
    /// by a publish that prunes or by one that prunes nothing, and one
    /// pruned with no fresh row. A worker's own row, refreshed, is not
    /// handed back.
    #[test]
    fn a_publish_hands_back_the_rows_it_took_from_other_workers() {
        let (registry, state) = setup();
        let urls = ["http://w1:8080".to_string()];
        let old = ready_worker("http://w1:8080", "m");
        let old_id = registry.register(Arc::clone(&old)).unwrap();
        assert!(state
            .publish_group(&urls, vec![(Arc::clone(&old), load(1))])
            .is_empty());
        assert!(
            state
                .publish_group(&urls, vec![(Arc::clone(&old), load(2))])
                .is_empty(),
            "its own row, refreshed"
        );

        registry.remove(&old_id);
        let new = ready_worker("http://w1:8080", "m");
        let new_id = registry.register(Arc::clone(&new)).unwrap();
        let replaced = state.publish_group(&urls, vec![(Arc::clone(&new), load(3))]);
        assert!(replaced.len() == 1 && Arc::ptr_eq(&replaced[0].0, &old));
        assert_eq!(replaced[0].1.total_waiting_uncached_tokens(), 2);

        registry.remove(&new_id);
        let third = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&third)).unwrap();
        let replaced = state.publish_group(&[], vec![(Arc::clone(&third), load(4))]);
        assert!(
            replaced.len() == 1 && Arc::ptr_eq(&replaced[0].0, &new),
            "a publish that prunes nothing overwrote it"
        );

        let pruned = state.publish_group(&urls, Vec::new());
        assert!(
            pruned.len() == 1 && Arc::ptr_eq(&pruned[0].0, &third),
            "pruned with no fresh row"
        );
        assert!(state.snapshot().is_empty());
    }

    /// The fresh rows can come out of the order of the group's URLs: the
    /// ranks of one engine are gathered. Each worker's own row is still its
    /// own, and a URL with no fresh row is still pruned.
    #[test]
    fn fresh_rows_out_of_the_urls_order_hand_back_no_row_of_their_own() {
        let (registry, state) = setup();
        let workers: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| {
                let worker = ready_worker(&format!("http://w{i}:8080"), "m");
                registry.register(Arc::clone(&worker)).unwrap();
                worker
            })
            .collect();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();
        let rows = |order: &[usize]| -> Vec<(Arc<dyn Worker>, Arc<WorkerLoadResponse>)> {
            order
                .iter()
                .map(|&i| (Arc::clone(&workers[i]), load(1)))
                .collect()
        };
        assert!(state.publish_group(&urls, rows(&[0, 1, 2, 3])).is_empty());
        assert!(state.publish_group(&urls, rows(&[2, 0, 3, 1])).is_empty());

        let pruned = state.publish_group(&urls, rows(&[3, 0, 1]));
        assert!(
            pruned.len() == 1 && Arc::ptr_eq(&pruned[0].0, &workers[2]),
            "the one URL with no fresh row"
        );
        for (i, url) in urls.iter().enumerate() {
            let waiting = waiting_of(&state.snapshot(), url);
            assert_eq!(waiting, (i != 2).then_some(1), "{url}");
        }
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
        let _ = state.publish_group(&[], fresh);
        assert_eq!(state.snapshot().len(), 50);

        for worker in &workers {
            worker.set_status(WorkerStatus::NotReady);
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
        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);
        state.enqueue_eviction(Arc::clone(&worker));

        state.clear();

        assert!(state.snapshot().is_empty());
        assert!(state.apply_pending_evictions().is_empty());
    }

    /// The batch evicts no worker that is the registry's `Ready` worker
    /// again: one that left and came back ahead of the batch keeps its row,
    /// and is not in the sentinel set.
    #[test]
    fn the_batch_evicts_no_worker_that_is_ready_again() {
        let (registry, state) = setup();
        let worker = ready_worker("http://w1:8080", "m");
        registry.register(Arc::clone(&worker)).unwrap();
        let _ = state.publish_group(&[], vec![(Arc::clone(&worker), load(1))]);
        worker.set_status(WorkerStatus::NotReady);
        state.enqueue_eviction(Arc::clone(&worker));
        worker.set_status(WorkerStatus::Ready);

        let removed = state.apply_pending_evictions();

        assert!(removed.is_empty(), "the worker is current again");
        assert_eq!(waiting_of(&state.snapshot(), "http://w1:8080"), Some(1));
    }
}
