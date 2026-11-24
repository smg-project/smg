//! A worker's load bookkeeping on the request path: the routing-key load
//! (how many distinct sticky keys are in flight on the worker, and how many
//! requests each) and the RAII guard that holds a request's load, keyed or
//! not, until the request ends.

use std::{
    any::Any,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use dashmap::mapref::entry::Entry;

use crate::{
    identity::{basic::Worker, endpoint::Endpoint},
    metrics::writers::{Metrics, WorkerHandles},
};

pub struct WorkerRoutingKeyLoad {
    /// The handles of the gauge it sets; they carry the worker URL.
    metrics: WorkerHandles,
    /// Built by the first keyed request: the map takes four shards a CPU up
    /// front, and a worker that serves no sticky session never reads it.
    active_routing_keys: OnceLock<dashmap::DashMap<String, usize>>,
    /// How many keys the map holds. The map's own length locks every shard
    /// (four a CPU), and every keyed request reads the count twice.
    active_keys: AtomicUsize,
}

impl WorkerRoutingKeyLoad {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            metrics: WorkerHandles::new(url),
            active_routing_keys: OnceLock::new(),
            active_keys: AtomicUsize::new(0),
        }
    }

    pub fn value(&self) -> usize {
        self.active_keys.load(Ordering::Relaxed)
    }

    /// In-flight requests for one routing key.
    pub fn key_inflight(&self, routing_key: &str) -> usize {
        self.active_routing_keys
            .get()
            .and_then(|keys| keys.get(routing_key))
            .map_or(0, |count| *count)
    }

    pub fn increment(&self, routing_key: &str) {
        // A key already in flight, the common case of a sticky session, is
        // bumped in place; the map's own copy of the key is built only for
        // the key's first request, after the lookup has let its shard go.
        let keys = self.active_routing_keys.get_or_init(dashmap::DashMap::new);
        match keys.get_mut(routing_key) {
            Some(mut count) => *count += 1,
            None => match keys.entry(routing_key.to_string()) {
                Entry::Occupied(mut count) => *count.get_mut() += 1,
                Entry::Vacant(slot) => {
                    // Counted under the key's shard lock, so the key's
                    // removal can never be counted first.
                    self.active_keys.fetch_add(1, Ordering::Relaxed);
                    slot.insert(1);
                }
            },
        }
        self.update_metrics();
    }

    pub fn decrement(&self, routing_key: &str) {
        // One pass over the key's shard, with nothing built for the lookup:
        // the count moves in place and the key leaves the map with its last
        // request. What the count was is kept for the warnings, which are
        // written once the shard is released.
        let mut before = None;
        let removed = self.active_routing_keys.get().and_then(|keys| {
            keys.remove_if_mut(routing_key, |_, count| {
                let was = *count;
                before = Some(was);
                *count = was.saturating_sub(1);
                was == 1
            })
        });
        if removed.is_some() {
            self.active_keys.fetch_sub(1, Ordering::Relaxed);
        }
        match before {
            Some(1..) => {}
            Some(0) => {
                tracing::warn!(
                    worker_url = %Endpoint::redacted_str(self.metrics.url()),
                    routing_key = %routing_key,
                    "Attempted to decrement routing key counter that is already at 0"
                );
            }
            None => {
                tracing::warn!(
                    worker_url = %Endpoint::redacted_str(self.metrics.url()),
                    routing_key = %routing_key,
                    "Attempted to decrement non-existent routing key"
                );
            }
        }
        self.update_metrics();
    }

    fn update_metrics(&self) {
        Metrics::set_worker_routing_keys_active(&self.metrics, self.value());
    }
}

impl fmt::Debug for WorkerRoutingKeyLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerRoutingKeyLoad")
            .field("url", &self.metrics.url())
            .field("active_routing_keys", &self.value())
            .finish()
    }
}

/// RAII guard for worker load management
///
/// Automatically decrements worker load when dropped. A streaming handler
/// returns at once and keeps producing from a task, so it hands the guard
/// to the response body's own stream (the routers' guarded stream), which drops it
/// when the body is dropped: at the stream's end or the client's disconnect.
///
/// A guard can keep one value of its maker's to the request's end
/// ([`WorkerLoadGuard::keeping`]). The guard never reads it: it lets it go
/// where the guard is dropped, behind the worker's load.
pub struct WorkerLoadGuard {
    worker: Arc<dyn Worker>,
    routing_key: Option<String>,
    /// What the guard's maker left with it ([`Self::keeping`]). The last
    /// field: it is dropped behind the load's release and the worker's
    /// handle.
    kept: Option<Box<dyn Any + Send + Sync>>,
}

impl WorkerLoadGuard {
    /// Guard keyed by the caller-resolved effective sticky key (rid-derived
    /// wins over the header), so keyed-load accounting matches selection.
    pub fn with_key(worker: Arc<dyn Worker>, routing_key: Option<&str>) -> Self {
        Self::with_owned_key(worker, routing_key.map(String::from))
    }

    /// The guard's one copy of the key, made by the constructor that resolved
    /// it; the worker's keyed-load bookkeeping borrows it from here on.
    // Always in line: on every request's path, and the call cost more than the body.
    #[inline(always)]
    fn with_owned_key(worker: Arc<dyn Worker>, routing_key: Option<String>) -> Self {
        worker.increment_load();
        Self::from_acquired_load(worker, routing_key)
    }

    pub fn try_new_with_key(
        worker: Arc<dyn Worker>,
        routing_key: Option<&str>,
        max: usize,
    ) -> Option<Self> {
        if !worker.try_increment_load(max) {
            return None;
        }
        Some(Self::from_acquired_load(
            worker,
            routing_key.map(String::from),
        ))
    }

    /// Acquire another load guard for the same worker and routing key.
    ///
    /// This is explicit instead of implementing `Clone` because replication
    /// increments externally visible worker load.
    pub fn replicate(&self) -> Self {
        self.worker.increment_load();
        if let Some(ref key) = self.routing_key {
            self.worker.increment_routing_key_load(key);
        }

        Self {
            worker: Arc::clone(&self.worker),
            routing_key: self.routing_key.clone(),
            kept: None,
        }
    }

    /// A load guard on `worker` under this guard's routing key: for a
    /// request that moves to another worker behind its guard. It raises
    /// `worker`'s load. This guard stands until it is dropped.
    pub fn onto(&self, worker: Arc<dyn Worker>) -> Self {
        Self::with_owned_key(worker, self.routing_key.clone())
    }

    fn from_acquired_load(worker: Arc<dyn Worker>, routing_key: Option<String>) -> Self {
        if let Some(ref key) = routing_key {
            worker.increment_routing_key_load(key);
        }

        Self {
            worker,
            routing_key,
            kept: None,
        }
    }

    /// This guard, keeping `kept` until the guard is dropped: a value whose
    /// own drop tells the guard's maker that the request ended. The guard
    /// never reads it. It is let go behind the worker's load, which the
    /// guard's drop lowers first. A guard made of this one
    /// ([`Self::replicate`], [`Self::onto`]) keeps nothing.
    #[must_use]
    pub fn keeping(mut self, kept: Box<dyn Any + Send + Sync>) -> Self {
        self.kept = Some(kept);
        self
    }

    /// Whether this guard counts `worker`'s load: the same worker object.
    #[must_use]
    pub fn counts(&self, worker: &Arc<dyn Worker>) -> bool {
        Arc::ptr_eq(&self.worker, worker)
    }
}

impl Drop for WorkerLoadGuard {
    // Always in line: on every request's path, and the call cost more than the body.
    #[inline(always)]
    fn drop(&mut self) {
        self.worker.decrement_load();
        if let Some(ref key) = self.routing_key {
            self.worker.decrement_routing_key_load(key);
        }
        // Every request path releases its worker load here, so this is where
        // policies learn that the request ended, whatever the outcome.
        if let Some(sink) = self.worker.completion_sink() {
            sink.request_completed(self.worker.as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use dashmap::DashMap;

    use super::*;
    use crate::{
        identity::basic::RequestCompletionSink, test_support::BasicWorkerBuilder,
        worker::WorkerType,
    };

    #[test]
    fn test_worker_routing_key_load_increment_decrement() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        assert_eq!(load.value(), 0);

        load.increment("key1");
        assert_eq!(load.value(), 1);

        load.increment("key2");
        assert_eq!(load.value(), 2);

        load.increment("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 1);

        load.decrement("key2");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_builds_its_map_on_the_first_key() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        assert_eq!(load.key_inflight("key1"), 0);
        load.decrement("key1");
        assert_eq!(load.value(), 0);
        assert!(load.active_routing_keys.get().is_none());

        load.increment("key1");
        assert_eq!(load.key_inflight("key1"), 1);
        assert!(load.active_routing_keys.get().is_some());
    }

    #[test]
    fn test_worker_routing_key_load_cleanup_on_zero() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key1");
        load.increment("key2");
        load.increment("key3");
        assert_eq!(load.active_routing_keys.get().map_or(0, DashMap::len), 3);

        load.decrement("key1");
        assert_eq!(load.active_routing_keys.get().map_or(0, DashMap::len), 2);

        load.decrement("key2");
        assert_eq!(load.active_routing_keys.get().map_or(0, DashMap::len), 1);

        load.decrement("key3");
        assert_eq!(load.active_routing_keys.get().map_or(0, DashMap::len), 0);
    }

    #[test]
    fn test_worker_routing_key_load_multiple_requests_same_key() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key-1");
        load.increment("key-1");
        load.increment("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 0);
        assert_eq!(load.active_routing_keys.get().map_or(0, DashMap::len), 0);
    }

    #[test]
    fn test_worker_routing_key_load_value_takes_no_shard_lock() {
        use std::{sync::mpsc, thread, time::Duration};

        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        load.increment("key1");

        // A held entry keeps its shard locked, as a request on another
        // thread does while it counts itself in.
        let held = load
            .active_routing_keys
            .get()
            .and_then(|keys| keys.get_mut("key1"));
        let (tx, rx) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| tx.send(load.value()));
            let read = rx.recv_timeout(Duration::from_secs(10));
            drop(held);
            assert_eq!(read, Ok(1));
        });
    }

    #[test]
    fn test_worker_routing_key_load_count_matches_map_after_churn() {
        use std::thread;

        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        load.increment("held");

        // Eight threads count four shared keys in and out, so a key's
        // first request races another's and a removal races a re-insert.
        thread::scope(|scope| {
            for thread_index in 0..8 {
                let load = &load;
                scope.spawn(move || {
                    for round in 0..2_000 {
                        let key = format!("key{}", (round + thread_index) % 4);
                        load.increment(&key);
                        load.decrement(&key);
                    }
                });
            }
        });

        assert_eq!(load.value(), 1);
        assert_eq!(load.active_routing_keys.get().map_or(0, DashMap::len), 1);
    }

    #[test]
    fn test_worker_routing_key_load_decrement_nonexistent() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        load.decrement("nonexistent");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_load_guard_with_routing_key() {
        use crate::test_support::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);

        {
            let _guard = WorkerLoadGuard::with_key(worker.clone(), Some("key-123"));
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.routing_key_load(), 1);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_worker_load_guard_without_routing_key() {
        use crate::test_support::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);

        {
            let _guard = WorkerLoadGuard::with_key(worker.clone(), None);
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.routing_key_load(), 0);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_worker_load_guard_multiple_same_routing_key() {
        use crate::test_support::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let guard1 = WorkerLoadGuard::with_key(worker.clone(), Some("key-123"));
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_load(), 1);

        let guard2 = WorkerLoadGuard::with_key(worker.clone(), Some("key-123"));
        assert_eq!(worker.load(), 2);
        assert_eq!(worker.routing_key_load(), 1);

        drop(guard1);
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_load(), 1);

        drop(guard2);
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_worker_load_guard_respects_atomic_limit() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Prefill)
                .build(),
        );

        let first = WorkerLoadGuard::try_new_with_key(Arc::clone(&worker), None, 1).unwrap();
        assert!(WorkerLoadGuard::try_new_with_key(Arc::clone(&worker), None, 1).is_none());

        drop(first);
        let second = WorkerLoadGuard::try_new_with_key(Arc::clone(&worker), None, 1).unwrap();
        drop(second);
        assert_eq!(worker.load(), 0);
    }

    /// Records which worker URLs reported a completion.
    #[derive(Debug, Default)]
    struct CompletionSpy(std::sync::Mutex<Vec<String>>);

    impl RequestCompletionSink for CompletionSpy {
        fn request_completed(&self, worker: &dyn Worker) {
            self.0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(worker.url().to_string());
        }
    }

    #[test]
    fn load_guard_drop_reports_the_completion_to_the_installed_sink() {
        let spy = Arc::new(CompletionSpy::default());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        worker.set_completion_sink(Some(Arc::clone(&spy) as Arc<dyn RequestCompletionSink>));

        let guard = WorkerLoadGuard::with_key(Arc::clone(&worker), Some("session-a"));
        let twin = guard.replicate();
        assert_eq!(worker.load(), 2);
        assert!(
            spy.0.lock().unwrap().is_empty(),
            "nothing completes while guards live"
        );
        drop(guard);
        assert_eq!(worker.load(), 1);
        assert_eq!(spy.0.lock().unwrap().as_slice(), ["http://w1:8000"]);
        drop(twin);
        assert_eq!(worker.load(), 0);
        assert_eq!(
            spy.0.lock().unwrap().len(),
            2,
            "each held load reports once"
        );

        // Without a sink the guard is exactly what it was.
        worker.set_completion_sink(None);
        drop(WorkerLoadGuard::with_key(Arc::clone(&worker), None));
        assert_eq!(spy.0.lock().unwrap().len(), 2);
    }

    /// Writes down, where it is dropped, its worker's load plus one hundred.
    struct LoadAtDrop(Arc<dyn Worker>, Arc<AtomicUsize>);

    impl Drop for LoadAtDrop {
        fn drop(&mut self) {
            self.1.store(self.0.load() + 100, Ordering::Relaxed);
        }
    }

    /// A guard keeps what its maker left with it until the guard is dropped,
    /// and lets it go behind the worker's load: the value reads a load the
    /// guard no longer holds. A guard made of that one keeps nothing.
    #[test]
    fn a_guard_lets_what_it_keeps_go_behind_its_load() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let seen = Arc::new(AtomicUsize::new(0));
        let kept = LoadAtDrop(Arc::clone(&worker), Arc::clone(&seen));
        let guard = WorkerLoadGuard::with_key(Arc::clone(&worker), None).keeping(Box::new(kept));
        let twin = guard.replicate();
        assert_eq!((worker.load(), seen.load(Ordering::Relaxed)), (2, 0));
        drop(guard);
        assert_eq!(seen.load(Ordering::Relaxed), 101, "behind the load");
        seen.store(0, Ordering::Relaxed);
        drop(twin);
        assert_eq!((worker.load(), seen.load(Ordering::Relaxed)), (0, 0));
    }
}
