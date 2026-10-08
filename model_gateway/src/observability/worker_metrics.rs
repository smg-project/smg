//! Recorder storage whose lifetime follows registered worker URLs.
//!
//! URL attribution is unchanged: a late event after the same URL is registered
//! again belongs to that URL. Events for absent URLs cannot recreate storage.
//! Request counters use the model from the latest successful mutation among live
//! registry owners; removing that owner restores the latest surviving owner.

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock, Weak,
    },
};

use dashmap::{mapref::entry::Entry, DashMap};
use metrics::{
    Counter, CounterFn, Gauge, GaugeFn, Histogram, HistogramFn, Key, KeyName, Metadata, Recorder,
    SharedString, Unit,
};
use metrics_exporter_prometheus::{
    formatting::{sanitize_metric_name, write_help_line, write_type_line},
    PrometheusHandle, PrometheusRecorder,
};
use parking_lot::RwLock;

static INSTALLED: OnceLock<Weak<WorkerMetrics>> = OnceLock::new();

struct RequestModel {
    owner: Arc<()>,
    model: Arc<str>,
}

struct Scope {
    url: Arc<str>,
    public_label: Arc<str>,
    request_models: RwLock<Vec<RequestModel>>,
    active: Arc<AtomicBool>,
    recorder: PrometheusRecorder,
}

impl Scope {
    /// Resolve ownership using the raw URL, then transform only its public label.
    fn metric_key<'a>(&self, key: &'a Key) -> Cow<'a, Key> {
        if self.public_label == self.url {
            return Cow::Borrowed(key);
        }
        Cow::Owned(Key::from_parts(
            key.name().to_owned(),
            key.labels()
                .map(|label| {
                    if label.key() == "worker" {
                        metrics::Label::new("worker", self.public_label.clone())
                    } else {
                        label.clone()
                    }
                })
                .collect::<Vec<_>>(),
        ))
    }

    fn record_request(&self, value: u64) {
        // Keep the identity stable through registration and increment. A
        // replacement cannot commit between choosing the model and recording.
        let models = self.request_models.read();
        self.record_request_with_models(&models, value);
    }

    fn record_request_with_models(&self, models: &[RequestModel], value: u64) {
        if !self.active.load(Ordering::Acquire) {
            return;
        }
        let Some(current) = models.last() else {
            return;
        };
        let key = Key::from_parts(
            "smg_worker_requests_total",
            vec![
                metrics::Label::new("worker", self.url.clone()),
                metrics::Label::new("model", current.model.clone()),
            ],
        );
        let metadata = Metadata::new(module_path!(), metrics::Level::INFO, Some(module_path!()));
        self.recorder
            .register_counter(self.metric_key(&key).as_ref(), &metadata)
            .increment(value);
    }
}

struct OwnedScope {
    scope: Arc<Scope>,
    owners: usize,
}

pub(crate) struct WorkerMetrics {
    scopes: DashMap<Arc<str>, OwnedScope>,
    descriptions: RwLock<HashMap<String, SharedString>>,
    factory: Box<dyn Fn() -> PrometheusRecorder + Send + Sync>,
}

impl fmt::Debug for WorkerMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerMetrics")
            .field("active_urls", &self.scopes.len())
            .finish_non_exhaustive()
    }
}

/// A successful registry registration owns one lease; replacement preserves it.
/// Dropping the registry also drops its leases, after its final Arc disappears.
pub(crate) struct WorkerMetricsLease {
    metrics: Weak<WorkerMetrics>,
    url: Arc<str>,
    owner: Arc<()>,
}

impl fmt::Debug for WorkerMetricsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerMetricsLease").finish_non_exhaustive()
    }
}

impl WorkerMetricsLease {
    /// The write guard spans the registry insertion, making the model update
    /// and publication one ordered operation for request-counter observers.
    pub(crate) fn with_model_update<T>(&self, model: &str, commit: impl FnOnce() -> T) -> T {
        let Some(metrics) = self.metrics.upgrade() else {
            return commit();
        };
        let Some(scope) = metrics.scope(&self.url) else {
            return commit();
        };
        let mut models = scope.request_models.write();
        models.retain(|entry| !Arc::ptr_eq(&entry.owner, &self.owner));
        models.push(RequestModel {
            owner: self.owner.clone(),
            model: super::metrics::intern_model_label(model),
        });
        let result = commit();
        scope.record_request_with_models(&models, 0);
        result
    }
}

impl Drop for WorkerMetricsLease {
    fn drop(&mut self) {
        let Some(metrics) = self.metrics.upgrade() else {
            return;
        };
        let retired = match metrics.scopes.entry(self.url.clone()) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().owners -= 1;
                let mut models = entry.get().scope.request_models.write();
                models.retain(|model| !Arc::ptr_eq(&model.owner, &self.owner));
                if entry.get().owners == 0 {
                    entry.get().scope.active.store(false, Ordering::Release);
                    drop(models);
                    Some(entry.remove().scope)
                } else {
                    None
                }
            }
            Entry::Vacant(_) => None,
        };
        if let Some(scope) = retired {
            // Drain pending histogram buckets before dropping the distributions.
            // Old returned handles keep empty atomic storage, never this recorder.
            scope.recorder.handle().run_upkeep();
            super::metrics::forget_worker_label(&self.url);
        }
    }
}

impl WorkerMetrics {
    fn acquire(self: &Arc<Self>, url: &str) -> WorkerMetricsLease {
        let scope = match self.scopes.entry(Arc::from(url)) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().owners += 1;
                entry.get().scope.clone()
            }
            Entry::Vacant(entry) => {
                let scope = Arc::new(Scope {
                    url: Arc::from(url),
                    public_label: super::worker_identity::public_worker_label(url),
                    request_models: RwLock::new(Vec::new()),
                    active: Arc::new(AtomicBool::new(true)),
                    recorder: (self.factory)(),
                });
                entry.insert(OwnedScope {
                    scope: scope.clone(),
                    owners: 1,
                });
                scope
            }
        };
        WorkerMetricsLease {
            metrics: Arc::downgrade(self),
            url: scope.url.clone(),
            owner: Arc::new(()),
        }
    }

    fn scope(&self, url: &str) -> Option<Arc<Scope>> {
        self.scopes.get(url).map(|entry| entry.scope.clone())
    }

    fn active_scopes(&self) -> Vec<Arc<Scope>> {
        self.scopes
            .iter()
            .map(|entry| entry.scope.clone())
            .collect()
    }

    fn describe(&self, key: &KeyName, description: SharedString) {
        self.descriptions
            .write()
            .entry(sanitize_metric_name(key.as_str()))
            .or_insert(description);
    }
}

pub(crate) fn acquire_registered_worker(url: &str) -> Option<WorkerMetricsLease> {
    INSTALLED
        .get()?
        .upgrade()
        .map(|metrics| metrics.acquire(url))
}

/// Return true when the installed lifecycle handled the write, including an
/// absent URL. Only standalone recorders fall back to the caller's model.
pub(crate) fn record_registered_request(url: &str, value: u64) -> bool {
    let Some(metrics) = INSTALLED.get().and_then(Weak::upgrade) else {
        return false;
    };
    if let Some(scope) = metrics.scope(url) {
        scope.record_request(value);
    }
    true
}

pub(crate) fn worker_label(url: &str) -> Arc<str> {
    INSTALLED
        .get()
        .and_then(Weak::upgrade)
        .and_then(|metrics| metrics.scope(url))
        .map_or_else(|| Arc::from(url), |scope| scope.url.clone())
}

pub(crate) fn installed() -> bool {
    INSTALLED.get().and_then(Weak::upgrade).is_some()
}

pub(crate) struct WorkerRecorder {
    global: PrometheusRecorder,
    workers: Arc<WorkerMetrics>,
}

impl WorkerRecorder {
    pub(crate) fn new(factory: impl Fn() -> PrometheusRecorder + Send + Sync + 'static) -> Self {
        let global = factory();
        Self {
            global,
            workers: Arc::new(WorkerMetrics {
                scopes: DashMap::new(),
                descriptions: RwLock::new(HashMap::new()),
                factory: Box::new(factory),
            }),
        }
    }

    pub(crate) fn handle(&self) -> MetricsHandle {
        MetricsHandle {
            global: self.global.handle(),
            workers: Some(self.workers.clone()),
        }
    }
}

fn worker_url(key: &Key) -> Option<&str> {
    // Tokio calls its thread index "worker" too; that is a global runtime metric.
    if !key.name().starts_with("smg_") || key.name().starts_with("smg_tokio_") {
        return None;
    }
    key.labels()
        .find(|label| label.key() == "worker")
        .map(|label| label.value())
}

struct ActiveCounter {
    active: Arc<AtomicBool>,
    inner: Counter,
}
impl CounterFn for ActiveCounter {
    fn increment(&self, value: u64) {
        if self.active.load(Ordering::Acquire) {
            self.inner.increment(value);
        }
    }
    fn absolute(&self, value: u64) {
        if self.active.load(Ordering::Acquire) {
            self.inner.absolute(value);
        }
    }
}
struct ActiveGauge {
    active: Arc<AtomicBool>,
    inner: Gauge,
}
impl GaugeFn for ActiveGauge {
    fn increment(&self, value: f64) {
        if self.active.load(Ordering::Acquire) {
            self.inner.increment(value);
        }
    }
    fn decrement(&self, value: f64) {
        if self.active.load(Ordering::Acquire) {
            self.inner.decrement(value);
        }
    }
    fn set(&self, value: f64) {
        if self.active.load(Ordering::Acquire) {
            self.inner.set(value);
        }
    }
}
struct ActiveHistogram {
    active: Arc<AtomicBool>,
    inner: Histogram,
}
impl HistogramFn for ActiveHistogram {
    fn record(&self, value: f64) {
        if self.active.load(Ordering::Acquire) {
            self.inner.record(value);
        }
    }
}

impl Recorder for WorkerRecorder {
    fn describe_counter(&self, key: KeyName, unit: Option<Unit>, desc: SharedString) {
        self.workers.describe(&key, desc.clone());
        self.global.describe_counter(key, unit, desc);
    }
    fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, desc: SharedString) {
        self.workers.describe(&key, desc.clone());
        self.global.describe_gauge(key, unit, desc);
    }
    fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, desc: SharedString) {
        self.workers.describe(&key, desc.clone());
        self.global.describe_histogram(key, unit, desc);
    }
    fn register_counter(&self, key: &Key, metadata: &Metadata<'_>) -> Counter {
        let Some(url) = worker_url(key) else {
            return self.global.register_counter(key, metadata);
        };
        self.workers.scope(url).map_or_else(Counter::noop, |scope| {
            Counter::from_arc(Arc::new(ActiveCounter {
                active: scope.active.clone(),
                inner: scope
                    .recorder
                    .register_counter(scope.metric_key(key).as_ref(), metadata),
            }))
        })
    }
    fn register_gauge(&self, key: &Key, metadata: &Metadata<'_>) -> Gauge {
        let Some(url) = worker_url(key) else {
            return self.global.register_gauge(key, metadata);
        };
        self.workers.scope(url).map_or_else(Gauge::noop, |scope| {
            Gauge::from_arc(Arc::new(ActiveGauge {
                active: scope.active.clone(),
                inner: scope
                    .recorder
                    .register_gauge(scope.metric_key(key).as_ref(), metadata),
            }))
        })
    }
    fn register_histogram(&self, key: &Key, metadata: &Metadata<'_>) -> Histogram {
        let Some(url) = worker_url(key) else {
            return self.global.register_histogram(key, metadata);
        };
        self.workers
            .scope(url)
            .map_or_else(Histogram::noop, |scope| {
                Histogram::from_arc(Arc::new(ActiveHistogram {
                    active: scope.active.clone(),
                    inner: scope
                        .recorder
                        .register_histogram(scope.metric_key(key).as_ref(), metadata),
                }))
            })
    }
}

/// Scrape and upkeep handle for global and registered worker storage.
#[derive(Clone, Debug)]
pub struct MetricsHandle {
    global: PrometheusHandle,
    workers: Option<Arc<WorkerMetrics>>,
}

impl From<PrometheusHandle> for MetricsHandle {
    fn from(global: PrometheusHandle) -> Self {
        Self {
            global,
            workers: None,
        }
    }
}

#[derive(Default)]
struct Family {
    kind: String,
    help: Option<String>,
    samples: String,
}

fn merge_scrape(families: &mut BTreeMap<String, Family>, rendered: &str) {
    for block in rendered.split("\n\n").filter(|block| !block.is_empty()) {
        let Some((name, kind)) = block
            .lines()
            .find_map(|line| line.strip_prefix("# TYPE ")?.split_once(' '))
        else {
            continue;
        };
        let family = families.entry(name.to_owned()).or_default();
        kind.clone_into(&mut family.kind);
        for line in block.lines() {
            if line.starts_with("# HELP ") {
                family.help.get_or_insert_with(|| line.to_owned());
            } else if !line.is_empty() && !line.starts_with('#') {
                family.samples.push_str(line);
                family.samples.push('\n');
            }
        }
    }
}

impl MetricsHandle {
    pub(crate) fn install_lifecycle(&self) {
        if let Some(workers) = &self.workers {
            // A weak pointer cannot independently retain recorder storage.
            let _ = INSTALLED.set(Arc::downgrade(workers));
        }
    }

    pub fn render(&self) -> String {
        let Some(workers) = &self.workers else {
            return self.global.render();
        };
        let mut families = BTreeMap::new();
        merge_scrape(&mut families, &self.global.render());
        for scope in workers.active_scopes() {
            if scope.active.load(Ordering::Acquire) {
                merge_scrape(&mut families, &scope.recorder.handle().render());
            }
        }
        let descriptions = workers.descriptions.read();
        let mut rendered = String::new();
        for (name, family) in families {
            if let Some(help) = family.help {
                rendered.push_str(&help);
                rendered.push('\n');
            } else if let Some(desc) = descriptions.get(&name) {
                write_help_line(&mut rendered, &name, None, None, desc);
            }
            write_type_line(&mut rendered, &name, None, None, &family.kind);
            rendered.push_str(&family.samples);
            rendered.push('\n');
        }
        rendered
    }

    pub fn run_upkeep(&self) {
        self.global.run_upkeep();
        if let Some(workers) = &self.workers {
            for scope in workers.active_scopes() {
                scope.recorder.handle().run_upkeep();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};

    use super::*;

    fn recorder() -> WorkerRecorder {
        WorkerRecorder::new(|| {
            PrometheusBuilder::new()
                .set_buckets_for_metric(
                    Matcher::Full("smg_kv_event_lag_seconds".into()),
                    &[0.001, 0.01],
                )
                .unwrap()
                .build_recorder()
        })
    }

    #[test]
    fn retirement_releases_recorder_and_intern_cache_even_with_old_metric_handles() {
        let recorder = recorder();
        let url = "http://retention-storage-regression";
        let lease = recorder.workers.acquire(url);
        let scope = recorder.workers.scope(url).unwrap();
        let weak_scope = Arc::downgrade(&scope);
        let weak_url = Arc::downgrade(&scope.url);
        let cached = super::super::metrics::intern_string(url);
        let weak_cached = Arc::downgrade(&cached);
        drop(cached);
        drop(scope);
        let (counter, gauge, histogram) = metrics::with_local_recorder(&recorder, || {
            let counter = metrics::counter!("smg_worker_cb_outcomes_total", "worker" => url, "outcome" => "failure");
            let gauge = metrics::gauge!("smg_worker_health", "worker" => url);
            let histogram = metrics::histogram!("smg_kv_event_lag_seconds", "worker" => url);
            counter.increment(1);
            gauge.set(1.0);
            histogram.record(0.001);
            (counter, gauge, histogram)
        });
        // Exercise the separately retained aggregated histogram distribution.
        assert!(recorder
            .handle()
            .render()
            .contains("smg_kv_event_lag_seconds_count{"));
        drop(lease);
        assert!(
            weak_scope.upgrade().is_none(),
            "metric handles retained the whole recorder"
        );
        assert!(
            weak_url.upgrade().is_none(),
            "retirement retained scope URL ownership"
        );
        assert!(
            weak_cached.upgrade().is_none(),
            "intern cache retained the retired URL"
        );
        counter.increment(1);
        gauge.set(1.0);
        histogram.record(10.0);
        assert!(recorder.workers.scopes.is_empty());
        assert!(!recorder.handle().render().contains(url));
    }

    #[test]
    fn quiet_owned_scope_survives_and_final_lease_reclaims_churn_storage() {
        let recorder = recorder();
        let url = "http://retention-quiet-storage";
        let first = recorder.workers.acquire(url);
        let second = recorder.workers.acquire(url);
        metrics::with_local_recorder(&recorder, || {
            metrics::gauge!("smg_worker_health", "worker" => url).set(1.0);
            metrics::histogram!("smg_kv_event_lag_seconds", "worker" => url).record(0.001);
        });
        drop(first);
        for _ in 0..3 {
            recorder.handle().run_upkeep();
        }
        assert!(recorder.handle().render().contains(url));
        drop(second);
        assert!(recorder.workers.scopes.is_empty());
        let mut retired = Vec::new();
        for n in 0..100 {
            let lease = recorder
                .workers
                .acquire(&format!("http://retention-storage-churn-{n}"));
            retired.push(Arc::downgrade(&recorder.workers.scope(&lease.url).unwrap()));
            drop(lease);
        }
        assert!(recorder.workers.scopes.is_empty());
        assert!(retired.iter().all(|weak| weak.upgrade().is_none()));
    }

    #[test]
    fn runtime_thread_worker_labels_remain_global_and_absent_urls_get_no_storage() {
        let recorder = recorder();
        metrics::with_local_recorder(&recorder, || {
            metrics::gauge!("smg_tokio_worker_busy_ratio", "worker" => "0").set(0.5);
            metrics::gauge!("smg_worker_health", "worker" => "http://absent-worker").set(1.0);
        });
        let rendered = recorder.handle().render();
        assert!(rendered.contains("smg_tokio_worker_busy_ratio{worker=\"0\"} 0.5"));
        assert!(!rendered.contains("http://absent-worker"));
        assert!(recorder.workers.scopes.is_empty());
    }

    #[test]
    fn request_recording_waits_for_the_registry_model_commit() {
        use std::{sync::mpsc, thread, time::Duration};

        let recorder = recorder();
        let url = "http://request-model-commit-order";
        let lease = recorder.workers.acquire(url);
        lease.with_model_update("before", || {});
        let scope = recorder.workers.scope(url).unwrap();
        scope.record_request(1);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        thread::scope(|threads| {
            lease.with_model_update("after", || {
                threads.spawn(|| {
                    started_tx.send(()).unwrap();
                    scope.record_request(1);
                    finished_tx.send(()).unwrap();
                });
                started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                assert_eq!(
                    finished_rx.recv_timeout(Duration::from_millis(100)),
                    Err(mpsc::RecvTimeoutError::Timeout),
                    "dispatch crossed an uncommitted model replacement"
                );
            });
            finished_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        });
        let rendered = recorder.handle().render();
        assert!(rendered
            .lines()
            .any(|line| line.starts_with("smg_worker_requests_total{")
                && line.contains("model=\"before\"")
                && line.ends_with(" 1")));
        assert!(rendered
            .lines()
            .any(|line| line.starts_with("smg_worker_requests_total{")
                && line.contains("model=\"after\"")
                && line.ends_with(" 1")));
    }

    #[test]
    fn repeated_replacements_keep_only_live_owner_metadata() {
        let recorder = recorder();
        let url = "http://request-model-owner-storage";
        let first = recorder.workers.acquire(url);
        let second = recorder.workers.acquire(url);
        let scope = recorder.workers.scope(url).unwrap();
        for _ in 0..100 {
            first.with_model_update("first", || {});
            second.with_model_update("second", || {});
        }
        assert_eq!(scope.request_models.read().len(), 2);
        drop(first);
        assert_eq!(scope.request_models.read().len(), 1);
        drop(second);
        assert!(scope.request_models.read().is_empty());
        assert!(recorder.workers.scopes.is_empty());
        scope.record_request(1);
        assert!(!recorder.handle().render().contains(url));
    }

    #[test]
    fn standalone_request_counter_keeps_the_supplied_model() {
        use super::super::metrics::Metrics;

        let flat = PrometheusBuilder::new().build_recorder();
        metrics::with_local_recorder(&flat, || {
            Metrics::initialize_worker_request_series(
                "http://standalone-request-model",
                "standalone-model",
            );
            Metrics::record_worker_request("http://standalone-request-model", "standalone-model");
        });
        let rendered = flat.handle().render();
        assert!(rendered
            .lines()
            .any(|line| line.starts_with("smg_worker_requests_total{")
                && line.contains("model=\"standalone-model\"")
                && line.ends_with(" 1")));
    }

    #[test]
    fn sensitive_worker_labels_are_private_distinct_and_retire_with_their_scope() {
        let recorder = recorder();
        let first_url =
            "https://alice:private-one@example.com:8443/v1?token=secret-query#private-fragment";
        let second_url =
            "https://alice:private-two@example.com:8443/v1?token=secret-query#private-fragment";
        let first = recorder.workers.acquire(first_url);
        let second = recorder.workers.acquire(second_url);
        for url in [first_url, second_url] {
            metrics::with_local_recorder(&recorder, || {
                metrics::counter!("smg_worker_cb_outcomes_total", "worker" => url, "outcome" => "failure").increment(1);
                metrics::gauge!("smg_worker_health", "worker" => url).set(1.0);
                metrics::histogram!("smg_kv_event_lag_seconds", "worker" => url).record(0.001);
            });
        }
        let rendered = recorder.handle().render();
        for secret in [
            "alice",
            "private-one",
            "private-two",
            "secret-query",
            "private-fragment",
        ] {
            assert!(
                !rendered.contains(secret),
                "scrape exposed sensitive worker identity"
            );
        }
        let labels: Vec<_> = rendered
            .lines()
            .filter(|line| line.starts_with("smg_worker_health{"))
            .map(|line| {
                line.split("worker=\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(labels.len(), 2, "distinct authenticated backends merged");
        assert_ne!(labels[0], labels[1]);
        for family in [
            "smg_worker_cb_outcomes_total{",
            "smg_worker_health{",
            "smg_kv_event_lag_seconds_count{",
        ] {
            for label in &labels {
                assert!(
                    rendered.lines().any(|line| line.starts_with(family)
                        && line.contains(&format!("worker=\"{label}\""))),
                    "worker families used different public identities"
                );
            }
        }
        drop(first);
        let remaining = recorder.handle().render();
        assert_eq!(
            remaining
                .lines()
                .filter(|line| line.starts_with("smg_worker_health{"))
                .count(),
            1
        );
        drop(second);
        assert!(recorder.workers.scopes.is_empty());
        assert!(!recorder.handle().render().contains("smg_worker_health{"));
    }

    #[test]
    fn many_active_workers_keep_the_same_samples_and_single_family_metadata() {
        use std::time::Instant;

        use super::super::metrics::{init_metrics, Metrics};

        let scoped = recorder();
        let flat = PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Full("smg_kv_event_lag_seconds".into()),
                &[0.001, 0.01],
            )
            .unwrap()
            .build_recorder();
        let mut leases = Vec::new();
        for n in 0..500 {
            let url = format!("http://retention-active-scale-{n}");
            leases.push(scoped.workers.acquire(&url));
            for recorder in [&scoped as &dyn Recorder, &flat as &dyn Recorder] {
                metrics::with_local_recorder(recorder, || {
                    Metrics::initialize_worker_series(&url, "regular", "http");
                    Metrics::initialize_worker_cb_series(&url);
                    Metrics::set_worker_health(&url, true);
                    Metrics::set_worker_requests_active(&url, 2);
                    Metrics::record_kv_event_lag(&url, 0.001);
                });
            }
        }
        metrics::with_local_recorder(&scoped, init_metrics);
        metrics::with_local_recorder(&flat, init_metrics);
        let started = Instant::now();
        let flat_rendered = flat.handle().render();
        let flat_elapsed = started.elapsed();
        let started = Instant::now();
        let rendered = scoped.handle().render();
        let scoped_elapsed = started.elapsed();
        let samples = |text: &str| {
            text.lines()
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .count()
        };
        assert_eq!(samples(&rendered), samples(&flat_rendered));
        assert_eq!(
            rendered
                .lines()
                .filter(|line| line.starts_with("# TYPE smg_worker_health "))
                .count(),
            1
        );
        tracing::info!(
            workers = 500,
            samples = samples(&rendered),
            bytes = rendered.len(),
            flat_micros = flat_elapsed.as_micros(),
            scoped_micros = scoped_elapsed.as_micros(),
            "worker metrics scrape cost"
        );
        drop(leases);
        assert!(scoped.workers.scopes.is_empty());
    }
}
