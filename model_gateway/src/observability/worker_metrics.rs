//! Per-worker series that leave `/metrics` with their worker.
//!
//! Every family labelled `worker` is keyed by the worker URL, and with
//! pod-based discovery a replaced engine pod comes back under a new address.
//! The Prometheus exporter cannot delete a series (metrics-rs/metrics#653),
//! so each replacement left the old address's full set of series in the
//! scrape for the life of the process, and sums over the per-worker gauges
//! counted dead workers. These series are stored here instead, in a registry
//! of their own. The worker registry retires an address when its registration
//! is removed: the address's series are dropped at once, and writes that
//! arrive after the removal (a health check racing it, a load guard released
//! after it, a stale handle) fall into no-op handles until the address is
//! registered again. The series render after the exporter's output, in the
//! exporter's format and with the exporter's histogram configuration.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Display, Write as _},
    sync::{atomic::Ordering, Arc, OnceLock},
};

use metrics::{
    atomics::AtomicU64, Counter, Gauge, Histogram, HistogramFn, Key, KeyName, Metadata, Recorder,
    SharedString, Unit,
};
use metrics_exporter_prometheus::{
    formatting::{
        sanitize_label_key, sanitize_label_value, sanitize_metric_name, write_help_line,
        write_type_line,
    },
    Distribution, DistributionBuilder,
};
use metrics_util::{
    registry::{Registry, Storage},
    storage::AtomicBucket,
};
use parking_lot::{Mutex, RwLock};
use quanta::Instant;

/// The label that keys a series to a worker; its value is the worker URL.
const WORKER_LABEL: &str = "worker";

/// The store the installed recorder routes per-worker series to.
static INSTALLED: OnceLock<Arc<WorkerSeries>> = OnceLock::new();

/// Install the store the global recorder routes to. The first call wins, as
/// the recorder installation it accompanies does.
pub(crate) fn install(series: Arc<WorkerSeries>) {
    let _ = INSTALLED.set(series);
}

/// The installed store, if the Prometheus recorder has been installed.
pub(crate) fn installed() -> Option<&'static Arc<WorkerSeries>> {
    INSTALLED.get()
}

/// Append the installed store's series to a scrape.
pub(crate) fn render_into(out: &mut String) {
    if let Some(series) = INSTALLED.get() {
        series.render_into(out);
    }
}

/// Fold the installed store's histogram samples, as the exporter's upkeep does.
pub(crate) fn run_upkeep() {
    if let Some(series) = INSTALLED.get() {
        series.run_upkeep();
    }
}

/// The per-worker series of the process, keyed by the worker URL.
pub(crate) struct WorkerSeries {
    /// Addresses whose registration was removed; their writes are dropped
    /// until the address registers again. Handle creation holds the read
    /// side, retirement the write side, so no series is created for an
    /// address between its retirement and the drop of its series.
    retired: RwLock<HashSet<String>>,
    registry: Registry<Key, WorkerStorage>,
    /// Descriptions by sanitized metric name, for the `# HELP` lines.
    descriptions: RwLock<HashMap<String, SharedString>>,
    distribution_builder: DistributionBuilder,
    /// Histogram samples aggregated per series, the exporter's way.
    distributions: Mutex<HashMap<Key, Distribution>>,
}

impl WorkerSeries {
    /// A store whose histograms take the exporter's bucket configuration.
    pub(crate) fn new(distribution_builder: DistributionBuilder) -> Self {
        Self {
            retired: RwLock::new(HashSet::new()),
            registry: Registry::new(WorkerStorage),
            descriptions: RwLock::new(HashMap::new()),
            distribution_builder,
            distributions: Mutex::new(HashMap::new()),
        }
    }

    /// A worker registered at `url`: its series record (again).
    pub(crate) fn activate(&self, url: &str) {
        self.retired.write().remove(url);
    }

    /// The registration at `url` was removed: drop its series, and drop the
    /// writes that follow until the address registers again.
    pub(crate) fn retire(&self, url: &str) {
        let mut retired = self.retired.write();
        retired.insert(url.to_owned());
        self.registry.retain_counters(|key, _| !keyed_to(key, url));
        self.registry.retain_gauges(|key, _| !keyed_to(key, url));
        self.registry
            .retain_histograms(|key, _| !keyed_to(key, url));
        self.distributions
            .lock()
            .retain(|key, _| !keyed_to(key, url));
    }

    fn describe(&self, name: &KeyName, description: &SharedString) {
        self.descriptions
            .write()
            .entry(sanitize_metric_name(name.as_str()))
            .or_insert_with(|| description.clone());
    }

    fn counter(&self, key: &Key, url: &str) -> Counter {
        let retired = self.retired.read();
        if retired.contains(url) {
            return Counter::noop();
        }
        self.registry
            .get_or_create_counter(key, |counter| Counter::from_arc(Arc::clone(counter)))
    }

    fn gauge(&self, key: &Key, url: &str) -> Gauge {
        let retired = self.retired.read();
        if retired.contains(url) {
            return Gauge::noop();
        }
        self.registry
            .get_or_create_gauge(key, |gauge| Gauge::from_arc(Arc::clone(gauge)))
    }

    fn histogram(&self, key: &Key, url: &str) -> Histogram {
        let retired = self.retired.read();
        if retired.contains(url) {
            return Histogram::noop();
        }
        self.registry
            .get_or_create_histogram(key, |samples| Histogram::from_arc(Arc::clone(samples)))
    }

    /// Fold recorded histogram samples into the per-series distributions, as
    /// the exporter's upkeep does, so the sample buffers stay small between
    /// scrapes.
    pub(crate) fn run_upkeep(&self) {
        let mut distributions = self.distributions.lock();
        self.registry.visit_histograms(|key, samples| {
            let distribution = distributions.entry(key.clone()).or_insert_with(|| {
                self.distribution_builder
                    .get_distribution(&sanitize_metric_name(key.name()))
            });
            samples
                .0
                .clear_with(|batch| distribution.record_samples(batch));
        });
    }

    /// Append the stored series to `out` in the exposition format: one
    /// family per metric name, the samples in label order.
    pub(crate) fn render_into(&self, out: &mut String) {
        let descriptions = self.descriptions.read();

        let mut counters: BTreeMap<String, Vec<(Vec<String>, u64)>> = BTreeMap::new();
        self.registry.visit_counters(|key, counter| {
            counters
                .entry(sanitize_metric_name(key.name()))
                .or_default()
                .push((label_pairs(key), counter.load(Ordering::Acquire)));
        });
        for (name, mut series) in counters {
            series.sort_by(|a, b| a.0.cmp(&b.0));
            write_family(out, &name, "counter", descriptions.get(&name));
            for (labels, value) in &series {
                write_sample(out, &name, None, labels, None, value);
            }
            out.push('\n');
        }

        let mut gauges: BTreeMap<String, Vec<(Vec<String>, f64)>> = BTreeMap::new();
        self.registry.visit_gauges(|key, gauge| {
            gauges
                .entry(sanitize_metric_name(key.name()))
                .or_default()
                .push((
                    label_pairs(key),
                    f64::from_bits(gauge.load(Ordering::Acquire)),
                ));
        });
        for (name, mut series) in gauges {
            series.sort_by(|a, b| a.0.cmp(&b.0));
            write_family(out, &name, "gauge", descriptions.get(&name));
            for (labels, value) in &series {
                write_sample(out, &name, None, labels, None, value);
            }
            out.push('\n');
        }

        self.run_upkeep();
        let distributions = self.distributions.lock();
        let mut by_name: BTreeMap<String, Vec<(Vec<String>, &Distribution)>> = BTreeMap::new();
        for (key, distribution) in distributions.iter() {
            by_name
                .entry(sanitize_metric_name(key.name()))
                .or_default()
                .push((label_pairs(key), distribution));
        }
        for (name, mut series) in by_name {
            let kind = self.distribution_builder.get_distribution_type(&name);
            if kind == "native_histogram" {
                // Only the protobuf exposition carries native histograms.
                continue;
            }
            series.sort_by(|a, b| a.0.cmp(&b.0));
            write_family(out, &name, kind, descriptions.get(&name));
            for (labels, distribution) in series {
                let (sum, count) = match distribution {
                    Distribution::Summary(summary, quantiles, sum) => {
                        let snapshot = summary.snapshot(Instant::now());
                        for quantile in quantiles.iter() {
                            let value = snapshot.quantile(quantile.value()).unwrap_or(0.0);
                            let quantile = quantile.value();
                            write_sample(
                                out,
                                &name,
                                None,
                                &labels,
                                Some(("quantile", &quantile)),
                                &value,
                            );
                        }
                        (*sum, u64::try_from(summary.count()).unwrap_or(u64::MAX))
                    }
                    Distribution::Histogram(histogram) => {
                        for (le, count) in histogram.buckets() {
                            write_sample(
                                out,
                                &name,
                                Some("bucket"),
                                &labels,
                                Some(("le", &le)),
                                &count,
                            );
                        }
                        write_sample(
                            out,
                            &name,
                            Some("bucket"),
                            &labels,
                            Some(("le", &"+Inf")),
                            &histogram.count(),
                        );
                        (histogram.sum(), histogram.count())
                    }
                    Distribution::NativeHistogram(_) => continue,
                };
                write_sample(out, &name, Some("sum"), &labels, None, &sum);
                write_sample(out, &name, Some("count"), &labels, None, &count);
            }
            out.push('\n');
        }
    }
}

/// The gateway's recorder: series labelled `worker` go to the store, every
/// other series to the Prometheus exporter.
pub(crate) struct WorkerSeriesRecorder<R> {
    inner: R,
    series: Arc<WorkerSeries>,
}

impl<R> WorkerSeriesRecorder<R> {
    pub(crate) fn new(inner: R, series: Arc<WorkerSeries>) -> Self {
        Self { inner, series }
    }
}

impl<R: Recorder> Recorder for WorkerSeriesRecorder<R> {
    fn describe_counter(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.series.describe(&key, &description);
        self.inner.describe_counter(key, unit, description);
    }

    fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.series.describe(&key, &description);
        self.inner.describe_gauge(key, unit, description);
    }

    fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.series.describe(&key, &description);
        self.inner.describe_histogram(key, unit, description);
    }

    fn register_counter(&self, key: &Key, metadata: &Metadata<'_>) -> Counter {
        match worker_url(key) {
            Some(url) => self.series.counter(key, url),
            None => self.inner.register_counter(key, metadata),
        }
    }

    fn register_gauge(&self, key: &Key, metadata: &Metadata<'_>) -> Gauge {
        match worker_url(key) {
            Some(url) => self.series.gauge(key, url),
            None => self.inner.register_gauge(key, metadata),
        }
    }

    fn register_histogram(&self, key: &Key, metadata: &Metadata<'_>) -> Histogram {
        match worker_url(key) {
            Some(url) => self.series.histogram(key, url),
            None => self.inner.register_histogram(key, metadata),
        }
    }
}

/// Storage of the per-worker registry: the exporter's atomic counters and
/// gauges, and histogram samples stamped with their time of recording so the
/// exporter's rolling summaries age them as the exporter's own.
#[derive(Debug)]
struct WorkerStorage;

impl<K> Storage<K> for WorkerStorage {
    type Counter = Arc<AtomicU64>;
    type Gauge = Arc<AtomicU64>;
    type Histogram = Arc<TimedSamples>;

    fn counter(&self, _: &K) -> Self::Counter {
        Arc::new(AtomicU64::new(0))
    }

    fn gauge(&self, _: &K) -> Self::Gauge {
        Arc::new(AtomicU64::new(0))
    }

    fn histogram(&self, _: &K) -> Self::Histogram {
        Arc::new(TimedSamples(AtomicBucket::new()))
    }
}

/// Histogram samples with their time of recording.
#[derive(Debug)]
struct TimedSamples(AtomicBucket<(f64, Instant)>);

impl HistogramFn for TimedSamples {
    fn record(&self, value: f64) {
        self.0.push((value, Instant::now()));
    }
}

fn worker_url(key: &Key) -> Option<&str> {
    key.labels()
        .find(|label| label.key() == WORKER_LABEL)
        .map(|label| label.value())
}

fn keyed_to(key: &Key, url: &str) -> bool {
    worker_url(key) == Some(url)
}

fn label_pairs(key: &Key) -> Vec<String> {
    key.labels()
        .map(|label| {
            format!(
                "{}=\"{}\"",
                sanitize_label_key(label.key()),
                sanitize_label_value(label.value())
            )
        })
        .collect()
}

fn write_family(out: &mut String, name: &str, kind: &str, description: Option<&SharedString>) {
    if let Some(description) = description {
        write_help_line(out, name, None, None, description);
    }
    write_type_line(out, name, None, None, kind);
}

fn write_sample(
    out: &mut String,
    name: &str,
    suffix: Option<&str>,
    labels: &[String],
    extra: Option<(&str, &dyn Display)>,
    value: &dyn Display,
) {
    out.push_str(name);
    if let Some(suffix) = suffix {
        out.push('_');
        out.push_str(suffix);
    }
    if !labels.is_empty() || extra.is_some() {
        out.push('{');
        out.push_str(&labels.join(","));
        if let Some((key, value)) = extra {
            if !labels.is_empty() {
                out.push(',');
            }
            let _ = write!(out, "{key}=\"{value}\"");
        }
        out.push('}');
    }
    let _ = writeln!(out, " {value}");
}

#[cfg(test)]
mod tests {
    use metrics::{counter, describe_counter, gauge, histogram};
    use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
    use metrics_util::parse_quantiles;

    use super::*;

    const WORKER_A: &str = "grpc://[fd00::a]:50051";
    const WORKER_B: &str = "grpc://[fd00::b]:50051";

    /// A store with the exporter's default quantiles and one bucketed name.
    fn store() -> Arc<WorkerSeries> {
        let overrides = HashMap::from([(
            Matcher::Full(String::from("apply_seconds")),
            vec![0.001, 0.01],
        )]);
        Arc::new(WorkerSeries::new(DistributionBuilder::new(
            parse_quantiles(&[0.5, 1.0]),
            None,
            None,
            None,
            Some(overrides),
            None,
        )))
    }

    /// Run `f` under a recorder over `series` and return the exporter's part
    /// of the scrape and the store's part.
    fn scrape(series: &Arc<WorkerSeries>, f: impl FnOnce()) -> (String, String) {
        let recorder =
            WorkerSeriesRecorder::new(PrometheusBuilder::new().build_recorder(), series.clone());
        let handle: PrometheusHandle = recorder.inner.handle();
        metrics::with_local_recorder(&recorder, f);
        let mut workers = String::new();
        series.render_into(&mut workers);
        (handle.render(), workers)
    }

    fn line<'a>(rendered: &'a str, prefix: &str) -> Option<&'a str> {
        rendered.lines().find(|line| line.starts_with(prefix))
    }

    #[test]
    fn worker_series_go_to_the_store_and_the_rest_to_the_exporter() {
        let series = store();
        let (exporter, workers) = scrape(&series, || {
            describe_counter!("requests_total", "Requests by worker");
            counter!("requests_total", "worker" => WORKER_A, "model" => "m").increment(2);
            gauge!("pool_size", "model" => "m").set(3.0);
        });

        assert!(exporter.contains("pool_size{model=\"m\"} 3"), "{exporter}");
        assert!(!exporter.contains("requests_total"), "{exporter}");
        assert!(!workers.contains("pool_size"), "{workers}");
        assert!(
            workers.contains(
                "# HELP requests_total Requests by worker\n# TYPE requests_total counter\n"
            ),
            "{workers}"
        );
        assert_eq!(
            line(&workers, "requests_total{"),
            Some(format!("requests_total{{worker=\"{WORKER_A}\",model=\"m\"}} 2").as_str()),
            "{workers}"
        );
    }

    #[test]
    fn retiring_a_worker_drops_its_series_and_the_writes_that_follow() {
        let series = store();
        let (_, workers) = scrape(&series, || {
            for worker in [WORKER_A, WORKER_B] {
                gauge!("health", "worker" => worker).set(1.0);
                counter!("transitions_total", "worker" => worker, "to" => "open").increment(1);
                histogram!("apply_seconds", "worker" => worker).record(0.005);
                histogram!("lag_seconds", "worker" => worker).record(0.5);
            }
        });
        assert!(
            workers.contains(WORKER_A) && workers.contains(WORKER_B),
            "{workers}"
        );

        series.retire(WORKER_A);
        let (_, workers) = scrape(&series, || {
            // Writes after the removal: a racing health check, a late outcome.
            gauge!("health", "worker" => WORKER_A).set(0.0);
            counter!("transitions_total", "worker" => WORKER_A, "to" => "open").increment(1);
            histogram!("apply_seconds", "worker" => WORKER_A).record(0.005);
        });
        assert!(!workers.contains(WORKER_A), "{workers}");
        assert_eq!(
            line(&workers, "health{"),
            Some(format!("health{{worker=\"{WORKER_B}\"}} 1").as_str()),
            "{workers}"
        );
        assert!(
            workers.contains(&format!("apply_seconds_count{{worker=\"{WORKER_B}\"}} 1")),
            "{workers}"
        );

        // The address registered again records again, from fresh series.
        series.activate(WORKER_A);
        let (_, workers) = scrape(&series, || {
            counter!("transitions_total", "worker" => WORKER_A, "to" => "open").increment(1);
        });
        assert_eq!(
            line(
                &workers,
                &format!("transitions_total{{worker=\"{WORKER_A}\"")
            ),
            Some(format!("transitions_total{{worker=\"{WORKER_A}\",to=\"open\"}} 1").as_str()),
            "{workers}"
        );
    }

    #[test]
    fn histograms_and_summaries_render_like_the_exporter() {
        let series = store();
        let (_, workers) = scrape(&series, || {
            histogram!("apply_seconds", "worker" => WORKER_A).record(0.005);
            histogram!("apply_seconds", "worker" => WORKER_A).record(0.5);
            histogram!("lag_seconds", "worker" => WORKER_A).record(2.0);
        });

        let expected_histogram = format!(
            "# TYPE apply_seconds histogram\n\
             apply_seconds_bucket{{worker=\"{WORKER_A}\",le=\"0.001\"}} 0\n\
             apply_seconds_bucket{{worker=\"{WORKER_A}\",le=\"0.01\"}} 1\n\
             apply_seconds_bucket{{worker=\"{WORKER_A}\",le=\"+Inf\"}} 2\n\
             apply_seconds_sum{{worker=\"{WORKER_A}\"}} 0.505\n\
             apply_seconds_count{{worker=\"{WORKER_A}\"}} 2\n"
        );
        assert!(workers.contains(&expected_histogram), "{workers}");
        // The summary's quantiles come from a sketch, so the median of one
        // sample is 2 within the sketch's relative error.
        assert!(
            workers.contains("# TYPE lag_seconds summary\n"),
            "{workers}"
        );
        let median = line(
            &workers,
            &format!("lag_seconds{{worker=\"{WORKER_A}\",quantile=\"0.5\"}} "),
        )
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("median missing: {workers}"));
        assert!((median - 2.0).abs() < 0.01, "{workers}");
        for expected in [
            format!("lag_seconds{{worker=\"{WORKER_A}\",quantile=\"1\"}} 2\n"),
            format!("lag_seconds_sum{{worker=\"{WORKER_A}\"}} 2\n"),
            format!("lag_seconds_count{{worker=\"{WORKER_A}\"}} 1\n"),
        ] {
            assert!(workers.contains(&expected), "{expected} missing: {workers}");
        }
    }

    #[test]
    fn label_values_are_escaped() {
        let series = store();
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => "http://w\"1\\", "note" => "a\nb").set(1.0);
        });
        assert_eq!(
            line(&workers, "health{"),
            Some("health{worker=\"http://w\\\"1\\\\\",note=\"a\\nb\"} 1"),
            "{workers}"
        );
    }
}
