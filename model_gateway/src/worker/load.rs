//! A worker's load bookkeeping on the request path: the routing-key load
//! (how many distinct sticky keys are in flight on the worker, and how many
//! requests each) and the RAII guard that holds a request's load, keyed or
//! not, until the request ends.

use std::{fmt, sync::Arc};

// The stream is named only by the doc links of this module.
#[cfg(doc)]
use super::worker::GuardedStream;
use super::worker::Worker;
use crate::{observability::metrics::Metrics, routers::common::header_utils::extract_routing_key};

pub struct WorkerRoutingKeyLoad {
    url: String,
    active_routing_keys: dashmap::DashMap<String, usize>,
}

impl WorkerRoutingKeyLoad {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            active_routing_keys: dashmap::DashMap::new(),
        }
    }

    pub fn value(&self) -> usize {
        self.active_routing_keys.len()
    }

    /// In-flight requests for one routing key.
    pub fn key_inflight(&self, routing_key: &str) -> usize {
        self.active_routing_keys
            .get(routing_key)
            .map_or(0, |count| *count)
    }

    pub fn increment(&self, routing_key: &str) {
        *self
            .active_routing_keys
            .entry(routing_key.to_string())
            .or_insert(0) += 1;
        self.update_metrics();
    }

    pub fn decrement(&self, routing_key: &str) {
        use dashmap::mapref::entry::Entry;

        match self.active_routing_keys.entry(routing_key.to_string()) {
            Entry::Occupied(mut entry) => {
                let counter = entry.get_mut();
                if *counter > 0 {
                    *counter -= 1;
                    if *counter == 0 {
                        entry.remove();
                    }
                } else {
                    tracing::warn!(
                        worker_url = %self.url,
                        routing_key = %routing_key,
                        "Attempted to decrement routing key counter that is already at 0"
                    );
                }
            }
            Entry::Vacant(_) => {
                tracing::warn!(
                    worker_url = %self.url,
                    routing_key = %routing_key,
                    "Attempted to decrement non-existent routing key"
                );
            }
        }
        self.update_metrics();
    }

    fn update_metrics(&self) {
        Metrics::set_worker_routing_keys_active(&self.url, self.value());
    }
}

impl fmt::Debug for WorkerRoutingKeyLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerRoutingKeyLoad")
            .field("url", &self.url)
            .field("active_routing_keys", &self.value())
            .finish()
    }
}

/// RAII guard for worker load management
///
/// Automatically decrements worker load when dropped. A streaming handler
/// returns at once and keeps producing from a task, so it hands the guard
/// to the response body's own stream ([`GuardedStream`]), which drops it
/// when the body is dropped: at the stream's end or the client's disconnect.
pub struct WorkerLoadGuard {
    worker: Arc<dyn Worker>,
    routing_key: Option<String>,
}

impl WorkerLoadGuard {
    pub fn new(worker: Arc<dyn Worker>, headers: Option<&http::HeaderMap>) -> Self {
        let key = extract_routing_key(headers).map(String::from);
        Self::with_key(worker, key.as_deref())
    }

    /// Guard keyed by the caller-resolved effective sticky key (rid-derived
    /// wins over the header), so keyed-load accounting matches selection.
    pub fn with_key(worker: Arc<dyn Worker>, routing_key: Option<&str>) -> Self {
        worker.increment_load();
        Self::from_acquired_load(worker, routing_key)
    }

    pub fn try_new(
        worker: Arc<dyn Worker>,
        headers: Option<&http::HeaderMap>,
        max: usize,
    ) -> Option<Self> {
        let key = extract_routing_key(headers);
        Self::try_new_with_key(worker, key, max)
    }

    pub fn try_new_with_key(
        worker: Arc<dyn Worker>,
        routing_key: Option<&str>,
        max: usize,
    ) -> Option<Self> {
        if !worker.try_increment_load(max) {
            return None;
        }
        Some(Self::from_acquired_load(worker, routing_key))
    }

    /// Acquire another load guard for the same worker and routing key.
    ///
    /// This is explicit instead of implementing `Clone` because replication
    /// increments externally visible worker load.
    pub(crate) fn replicate(&self) -> Self {
        self.worker.increment_load();
        if let Some(ref key) = self.routing_key {
            self.worker.increment_routing_key_load(key);
        }

        Self {
            worker: Arc::clone(&self.worker),
            routing_key: self.routing_key.clone(),
        }
    }

    fn from_acquired_load(worker: Arc<dyn Worker>, routing_key: Option<&str>) -> Self {
        let routing_key = routing_key.map(String::from);
        if let Some(ref key) = routing_key {
            worker.increment_routing_key_load(key);
        }

        Self {
            worker,
            routing_key,
        }
    }
}

impl Drop for WorkerLoadGuard {
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
    use openai_protocol::worker::WorkerType;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, RequestCompletionSink};

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
    fn test_worker_routing_key_load_cleanup_on_zero() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key1");
        load.increment("key2");
        load.increment("key3");
        assert_eq!(load.active_routing_keys.len(), 3);

        load.decrement("key1");
        assert_eq!(load.active_routing_keys.len(), 2);

        load.decrement("key2");
        assert_eq!(load.active_routing_keys.len(), 1);

        load.decrement("key3");
        assert_eq!(load.active_routing_keys.len(), 0);
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
        assert_eq!(load.active_routing_keys.len(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_decrement_nonexistent() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        load.decrement("nonexistent");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_load_guard_with_routing_key() {
        use crate::worker::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), Some(&headers));
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.routing_key_load(), 1);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_worker_load_guard_without_routing_key() {
        use crate::worker::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), None);
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.routing_key_load(), 0);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_worker_load_guard_multiple_same_routing_key() {
        use crate::worker::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        let guard1 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_load(), 1);

        let guard2 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
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

        let first = WorkerLoadGuard::try_new(Arc::clone(&worker), None, 1).unwrap();
        assert!(WorkerLoadGuard::try_new(Arc::clone(&worker), None, 1).is_none());

        drop(first);
        let second = WorkerLoadGuard::try_new(Arc::clone(&worker), None, 1).unwrap();
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
}
