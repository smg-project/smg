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
//! registered again, or until nothing holds a worker that served it any
//! more: the registry hands the store a `Weak` of every such worker (an
//! instance replaced at the address as it is replaced, the removed one with
//! the retirement), and the upkeep forgets the addresses whose workers are
//! all gone, so the retired set is bounded by the removed and replaced
//! workers still referenced. The series render after the exporter's output,
//! in the exporter's format and with the exporter's histogram configuration.
//!
//! A series is keyed by the address as registered, and the scrape shows that
//! address masked: a userinfo masked whole, a query or fragment cut (see
//! [`Labels`]). An address with none of them is shown byte for byte.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{Display, Write as _},
    sync::{atomic::Ordering, Arc, OnceLock, Weak},
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

use crate::worker::{endpoint::Endpoint, Worker};

/// The label that keys a series to a worker; its value is the worker URL as
/// registered, and a scrape shows it by [`Labels`].
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

/// The installed store, a default one (the exporter's quantiles, no
/// buckets) installed first when the process has none: what a test of the
/// registry's handoff to the store reads.
#[cfg(test)]
pub(crate) fn installed_or_default() -> &'static Arc<WorkerSeries> {
    INSTALLED.get_or_init(|| {
        Arc::new(WorkerSeries::new(DistributionBuilder::new(
            metrics_util::parse_quantiles(&[0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]),
            None,
            None,
            None,
            None,
            None,
        )))
    })
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

/// What the store keeps of an address besides its series: the workers that
/// served it and may still write for it, and whether its registration was
/// removed.
#[derive(Default)]
struct Address {
    /// The registration was removed: the address's writes are dropped until
    /// it registers again.
    retired: bool,
    /// The workers that served the address and may still write for it from a
    /// handle that outlived the registry's (a load guard on a replaced
    /// instance, a removed worker's late health check), each a `Weak`: the
    /// entry goes with the last of them.
    writers: Vec<Weak<dyn Worker>>,
}

/// What a scrape shows for each value of the `worker` label it has decided:
/// `None` shows the value as it is.
type Shown = HashMap<String, Option<String>>;

/// What a scrape shows for the values of the `worker` label.
///
/// A series is keyed by the address as registered, userinfo and query
/// included: the worker registry keys a worker by that string, and this
/// store's removal and replacement find a worker's series by it. The scrape
/// shows the address as a log line and the routed-worker header do
/// ([`Endpoint::redacted_url`]): an address with a userinfo, a query or a
/// fragment by its masked form, any other address by its own bytes, so its
/// series keep their label. A value that is no address is shown as it is:
/// the runtime's thread index rides the label too.
///
/// Two addresses of one masked form are two workers and stay two label
/// sets: the one decided first holds the masked form alone, a later one the
/// masked form followed by its worker id, or by a number until the registry
/// has given it one. An address is decided when the registry names its id
/// or at the first scrape that finds a series of it, whichever comes first,
/// and the decision stands until the address is retired, with one
/// exception: a value shown as it is keeps its bytes, so a masked address
/// that reads the same gives way to it and is decided again.
#[derive(Default)]
struct Labels {
    /// The decision for each value seen. A scrape holds the map it read
    /// while it writes, so every line of it follows one state of the
    /// decisions: a decision made meanwhile goes to a copy.
    shown: Arc<Shown>,
    /// The address that shows each masked label.
    holders: HashMap<String, String>,
    /// The worker id of each registered address whose label is masked.
    ids: HashMap<String, String>,
}

impl Labels {
    /// The decisions as they stand.
    fn shown(&self) -> Arc<Shown> {
        Arc::clone(&self.shown)
    }

    /// Decide what a scrape shows for `value`, unless it is decided.
    fn decide(&mut self, value: &str) {
        if !self.shown.contains_key(value) {
            self.settle(value);
        }
    }

    /// The registry's id for the worker at `url`. A label decided before it
    /// stands when it is the masked form alone; a numbered one gives way to
    /// the id.
    fn identify(&mut self, url: &str, worker_id: &str) {
        let Some(masked) = masked_form(url) else {
            return;
        };
        self.ids.insert(url.to_owned(), worker_id.to_owned());
        let with_id = format!("{masked} {worker_id}");
        let stands = self.shown.get(url).is_some_and(|shown| {
            shown
                .as_deref()
                .is_some_and(|label| label == masked || label == with_id)
        });
        if !stands {
            self.settle(url);
        }
    }

    /// The address was retired: its label is free for the next address of
    /// its masked form.
    fn forget(&mut self, url: &str) {
        if self.shown.contains_key(url) {
            if let Some(Some(label)) = Arc::make_mut(&mut self.shown).remove(url) {
                self.holders.remove(&label);
            }
        }
        self.ids.remove(url);
    }

    /// Decide `value` beside the decisions that stand.
    fn settle(&mut self, value: &str) {
        let Some(masked) = masked_form(value) else {
            // Shown as it is, and its own bytes come before a masked address
            // that reads the same: that address is decided again.
            let gives_way = self.holders.get(value).cloned();
            Arc::make_mut(&mut self.shown).insert(value.to_owned(), None);
            if let Some(address) = gives_way {
                self.settle(&address);
            }
            return;
        };
        let label = self.unshared(value, &masked);
        let before = Arc::make_mut(&mut self.shown).insert(value.to_owned(), Some(label.clone()));
        if let Some(Some(before)) = before {
            self.holders.remove(&before);
        }
        self.holders.insert(label, value.to_owned());
    }

    /// The first of `masked`, `masked` followed by the worker id of `value`
    /// (once the registry has given it one) and `masked` followed by a
    /// number, that no other value shows.
    fn unshared(&self, value: &str, masked: &str) -> String {
        // Shown by another value: its own bytes when that one is shown as
        // it is, its label when it is masked.
        let taken = |label: &str| {
            self.shown.get(label).is_some_and(Option::is_none)
                || self.holders.get(label).is_some_and(|other| other != value)
        };
        let with_id = self.ids.get(value).map(|id| format!("{masked} {id}"));
        for label in [Some(masked.to_owned()), with_id].into_iter().flatten() {
            if !taken(&label) {
                return label;
            }
        }
        let mut number = 2;
        loop {
            let label = format!("{masked} #{number}");
            if !taken(&label) {
                return label;
            }
            number += 1;
        }
    }
}

/// The masked form of a `worker` label's value, when the scrape must not
/// show the value itself: an address with a userinfo, a query or a fragment.
/// `None` for an address with none of them and for a value that is no
/// address.
fn masked_form(value: &str) -> Option<String> {
    Endpoint::parse_with_rank(value).ok()?;
    let masked = Endpoint::redacted_url(value);
    (masked != value).then_some(masked)
}

/// The per-worker series of the process, keyed by the worker URL.
pub(crate) struct WorkerSeries {
    /// The addresses with a removed registration or a replaced worker (see
    /// [`Address`]); a retired one's writes are dropped until the address
    /// registers again or the upkeep finds every one of its workers gone.
    /// Handle creation holds the read side, retirement the write side, so no
    /// series is created for an address between its retirement and the drop
    /// of its series.
    addresses: RwLock<HashMap<String, Address>>,
    registry: Registry<Key, WorkerStorage>,
    /// Descriptions by sanitized metric name, for the `# HELP` lines.
    descriptions: RwLock<HashMap<String, SharedString>>,
    distribution_builder: DistributionBuilder,
    /// Histogram samples aggregated per series, the exporter's way.
    distributions: Mutex<HashMap<Key, Distribution>>,
    /// What a scrape shows for each value of the worker label. The last
    /// lock taken: nothing takes another of the store's locks under it.
    labels: Mutex<Labels>,
}

impl WorkerSeries {
    /// A store whose histograms take the exporter's bucket configuration.
    pub(crate) fn new(distribution_builder: DistributionBuilder) -> Self {
        Self {
            addresses: RwLock::new(HashMap::new()),
            registry: Registry::new(WorkerStorage),
            descriptions: RwLock::new(HashMap::new()),
            distribution_builder,
            distributions: Mutex::new(HashMap::new()),
            labels: Mutex::new(Labels::default()),
        }
    }

    /// The registry's id for the worker registered at `url`: what tells its
    /// label from another address of the same masked form (see [`Labels`]).
    pub(crate) fn identify(&self, url: &str, worker_id: &str) {
        self.labels.lock().identify(url, worker_id);
    }

    /// A worker registered at `url`: its series record (again). The workers
    /// that served the address before stay known while a handle holds one.
    pub(crate) fn activate(&self, url: &str) {
        let mut addresses = self.addresses.write();
        let Some(address) = addresses.get_mut(url) else {
            return;
        };
        address.retired = false;
        let nothing_to_keep = address.writers.is_empty();
        if nothing_to_keep {
            addresses.remove(url);
        }
    }

    /// `worker`, replaced at `url` by another instance, may still write for
    /// the address from the handles that hold it: the store keeps its `Weak`,
    /// so a later retirement of the address lasts until it is gone too.
    pub(crate) fn replaced(&self, url: &str, worker: Weak<dyn Worker>) {
        let mut addresses = self.addresses.write();
        let address = addresses.entry(url.to_owned()).or_default();
        address.writers.retain(|writer| writer.strong_count() > 0);
        address.writers.push(worker);
    }

    /// The registration at `url` was removed: drop its series, and drop the
    /// writes that follow until the address registers again or `worker`, the
    /// removed one, and the instances replaced at the address before it are
    /// gone from every handle that could still write for them.
    pub(crate) fn retire(&self, url: &str, worker: Weak<dyn Worker>) {
        let mut addresses = self.addresses.write();
        let address = addresses.entry(url.to_owned()).or_default();
        address.retired = true;
        address.writers.push(worker);
        self.registry.retain_counters(|key, _| !keyed_to(key, url));
        self.registry.retain_gauges(|key, _| !keyed_to(key, url));
        self.registry
            .retain_histograms(|key, _| !keyed_to(key, url));
        self.distributions
            .lock()
            .retain(|key, _| !keyed_to(key, url));
        self.labels.lock().forget(url);
    }

    fn describe(&self, name: &KeyName, description: &SharedString) {
        self.descriptions
            .write()
            .entry(sanitize_metric_name(name.as_str()))
            .or_insert_with(|| description.clone());
    }

    fn counter(&self, key: &Key, url: &str) -> Counter {
        let addresses = self.addresses.read();
        if addresses.get(url).is_some_and(|address| address.retired) {
            return Counter::noop();
        }
        self.registry
            .get_or_create_counter(key, |counter| Counter::from_arc(Arc::clone(counter)))
    }

    fn gauge(&self, key: &Key, url: &str) -> Gauge {
        let addresses = self.addresses.read();
        if addresses.get(url).is_some_and(|address| address.retired) {
            return Gauge::noop();
        }
        self.registry
            .get_or_create_gauge(key, |gauge| Gauge::from_arc(Arc::clone(gauge)))
    }

    fn histogram(&self, key: &Key, url: &str) -> Histogram {
        let addresses = self.addresses.read();
        if addresses.get(url).is_some_and(|address| address.retired) {
            return Histogram::noop();
        }
        self.registry
            .get_or_create_histogram(key, |samples| Histogram::from_arc(Arc::clone(samples)))
    }

    /// The retired addresses, for the tests.
    #[cfg(test)]
    pub(crate) fn retired_len(&self) -> usize {
        self.addresses
            .read()
            .values()
            .filter(|address| address.retired)
            .count()
    }

    /// Whether `worker_url` is retired, for the tests.
    #[cfg(test)]
    pub(crate) fn is_retired(&self, worker_url: &str) -> bool {
        self.addresses
            .read()
            .get(worker_url)
            .is_some_and(|address| address.retired)
    }

    /// Fold recorded histogram samples into the per-series distributions, as
    /// the exporter's upkeep does, so the sample buffers stay small between
    /// scrapes, and forget the addresses whose workers are all gone. A write
    /// for an address comes from a handle on a worker that served it (a writer
    /// that keys by the URL alone holds the worker through its write, as the
    /// KV-event subscription task does), so the retired set stays bounded by
    /// the removed and replaced workers still referenced.
    pub(crate) fn run_upkeep(&self) {
        self.addresses.write().retain(|_, address| {
            address.writers.retain(|writer| writer.strong_count() > 0);
            !address.writers.is_empty()
        });
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
    /// family per metric name, the samples in label order. Every line names
    /// its worker by one state of the decisions (see [`Labels`]), so no
    /// label changes within a scrape: a scrape that meets a value with no
    /// decision decides it and is written again.
    pub(crate) fn render_into(&self, out: &mut String) {
        let start = out.len();
        let shown = self.labels.lock().shown();
        let undecided = self.render_by(out, &shown);
        if undecided.is_empty() {
            return;
        }
        // Let go of the map first: a decision copies a map that is held. An
        // address retired since the first writing is decided all the same:
        // one entry, which stays until that address is retired again.
        drop(shown);
        let shown = {
            let mut labels = self.labels.lock();
            for value in &undecided {
                labels.decide(value);
            }
            labels.shown()
        };
        // Written again from its start, by the new decisions. A value first
        // met in this second writing is left to the next scrape.
        out.truncate(start);
        self.render_by(out, &shown);
    }

    /// Write the series whose worker value `shown` holds a decision for, and
    /// return the values it holds none for, in byte order whatever order the
    /// registry visits them in: their series are left out. The upkeep runs
    /// here, so a scrape written again runs it twice: the second run folds
    /// the samples recorded since the first.
    fn render_by(&self, out: &mut String, shown: &Shown) -> BTreeSet<String> {
        let mut undecided = BTreeSet::new();
        let descriptions = self.descriptions.read();

        let mut counters: BTreeMap<String, Vec<(Vec<String>, u64)>> = BTreeMap::new();
        self.registry.visit_counters(|key, counter| {
            let Some(labels) = label_pairs(key, shown, &mut undecided) else {
                return;
            };
            counters
                .entry(sanitize_metric_name(key.name()))
                .or_default()
                .push((labels, counter.load(Ordering::Acquire)));
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
            let Some(labels) = label_pairs(key, shown, &mut undecided) else {
                return;
            };
            gauges
                .entry(sanitize_metric_name(key.name()))
                .or_default()
                .push((labels, f64::from_bits(gauge.load(Ordering::Acquire))));
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
            let Some(labels) = label_pairs(key, shown, &mut undecided) else {
                continue;
            };
            by_name
                .entry(sanitize_metric_name(key.name()))
                .or_default()
                .push((labels, distribution));
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
        undecided
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

/// A series' labels as a scrape writes them: the worker label by what `shown`
/// holds for its value, every other label as it is. `None` when the worker's
/// value has no decision in `shown`: the value joins `undecided`, and the
/// scrape leaves the series out until the value is decided. A scrape never
/// falls back to the value itself.
fn label_pairs(key: &Key, shown: &Shown, undecided: &mut BTreeSet<String>) -> Option<Vec<String>> {
    key.labels()
        .map(|label| {
            let value = label.value();
            let value = if label.key() == WORKER_LABEL {
                let Some(decided) = shown.get(value) else {
                    if !undecided.contains(value) {
                        undecided.insert(value.to_owned());
                    }
                    return None;
                };
                decided.as_deref().unwrap_or(value)
            } else {
                value
            };
            Some(format!(
                "{}=\"{}\"",
                sanitize_label_key(label.key()),
                sanitize_label_value(value)
            ))
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
    use metrics::{counter, describe_counter, gauge, histogram, Label};
    use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
    use metrics_util::parse_quantiles;

    use super::*;
    use crate::worker::{BasicWorker, BasicWorkerBuilder};

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

    /// A worker registered with a credential in its address is named in the
    /// scrape by its masked address: in a gauge, a counter, a bucketed
    /// histogram and a summary.
    #[test]
    fn a_registered_credential_is_not_in_the_scrape() {
        const WITH_A_SECRET: &str = "http://user:s3cret@w:8000/v1?token=s3cret";
        let series = store();
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => WITH_A_SECRET).set(1.0);
            counter!("requests_total", "worker" => WITH_A_SECRET, "model" => "m").increment(1);
            histogram!("apply_seconds", "worker" => WITH_A_SECRET).record(0.005);
            histogram!("lag_seconds", "worker" => WITH_A_SECRET).record(0.5);
        });
        assert!(!workers.contains("s3cret"), "{workers}");
        for family in [
            "health{",
            "requests_total{",
            "apply_seconds_bucket{",
            "lag_seconds{",
        ] {
            let line = line(&workers, family).unwrap_or_else(|| panic!("no {family} line"));
            assert!(line.contains("worker=\"http://***@w:8000/v1\""), "{line}");
        }
    }

    /// A value that is no address is shown as it is: the runtime's thread
    /// index rides the worker label too, and so might any text. An address
    /// with nothing to mask keeps its bytes.
    #[test]
    fn a_value_with_nothing_to_mask_is_shown_as_it_is() {
        let series = store();
        let (_, workers) = scrape(&series, || {
            for value in ["3", "not an address ?", WORKER_A, "http://w:8000/v1"] {
                gauge!("busy_ratio", "worker" => value).set(0.5);
            }
        });
        for value in ["3", "not an address ?", WORKER_A, "http://w:8000/v1"] {
            let label = format!("busy_ratio{{worker=\"{value}\"}} 0.5");
            assert!(workers.contains(&label), "{label} is not in:\n{workers}");
        }
    }

    /// A masked worker's removal drops its series, and its registration
    /// brings them back under the same masked label.
    #[test]
    fn a_masked_worker_leaves_and_returns_under_one_label() {
        const WITH_A_SECRET: &str = "http://user:s3cret@w:8000";
        let health = "health{worker=\"http://***@w:8000\"}";
        let series = store();
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => WITH_A_SECRET).set(1.0);
        });
        assert_eq!(
            line(&workers, "health{"),
            Some(format!("{health} 1").as_str())
        );

        series.retire(WITH_A_SECRET, Weak::<BasicWorker>::new());
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => WITH_A_SECRET).set(0.0);
        });
        assert_eq!(line(&workers, "health{"), None, "{workers}");

        series.activate(WITH_A_SECRET);
        series.identify(WITH_A_SECRET, "id-1");
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => WITH_A_SECRET).set(1.0);
        });
        assert_eq!(
            line(&workers, "health{"),
            Some(format!("{health} 1").as_str())
        );
        assert!(!workers.contains("s3cret"), "{workers}");
    }

    /// Addresses of one masked form are as many workers and label sets: the
    /// first holds the masked form alone, a later one carries its worker id,
    /// or a number until the registry has given it one. A number given back
    /// is the next unidentified address's, and removing the first frees the
    /// masked form for the next address that reads the same.
    #[test]
    fn addresses_of_one_masked_form_are_as_many_label_sets() {
        const FIRST: &str = "https://key-1@api:443";
        const SECOND: &str = "https://key-2@api:443";
        const THIRD: &str = "https://key-3@api:443";
        const FOURTH: &str = "https://key-4@api:443";
        const FIFTH: &str = "https://key-5@api:443";
        let series = store();
        let healths = |workers: &str| -> Vec<String> {
            workers
                .lines()
                .filter(|line| line.starts_with("health{"))
                .map(str::to_string)
                .collect()
        };
        series.identify(FIRST, "id-1");
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => FIRST).set(1.0);
            // Seen before their registration: a builder writes first.
            gauge!("health", "worker" => SECOND).set(0.0);
            gauge!("health", "worker" => THIRD).set(0.5);
        });
        // In label order: a space sorts before the closing quote.
        assert_eq!(
            healths(&workers),
            [
                "health{worker=\"https://***@api:443 #2\"} 0",
                "health{worker=\"https://***@api:443 #3\"} 0.5",
                "health{worker=\"https://***@api:443\"} 1",
            ]
        );

        // The second takes its id and gives its number back: the fourth,
        // unidentified, takes it.
        series.identify(SECOND, "id-2");
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => FOURTH).set(0.25);
        });
        assert_eq!(
            healths(&workers),
            [
                "health{worker=\"https://***@api:443 #2\"} 0.25",
                "health{worker=\"https://***@api:443 #3\"} 0.5",
                "health{worker=\"https://***@api:443 id-2\"} 0",
                "health{worker=\"https://***@api:443\"} 1",
            ]
        );

        // The first is removed: the others keep their labels, and the
        // fifth takes the masked form alone.
        series.retire(FIRST, Weak::<BasicWorker>::new());
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => FIFTH).set(0.75);
        });
        assert_eq!(
            healths(&workers),
            [
                "health{worker=\"https://***@api:443 #2\"} 0.25",
                "health{worker=\"https://***@api:443 #3\"} 0.5",
                "health{worker=\"https://***@api:443 id-2\"} 0",
                "health{worker=\"https://***@api:443\"} 0.75",
            ]
        );
        assert!(!workers.contains("key-"), "{workers}");
    }

    /// A masked label never repeats the text of a value shown as it is: an
    /// address whose label reads as such a value's own bytes is decided
    /// again, and takes a number when its id form is that text.
    #[test]
    fn a_masked_label_never_repeats_another_values_text() {
        const FIRST: &str = "https://key-1@api:443";
        const SECOND: &str = "https://key-2@api:443";
        // No address: shown as it is.
        const TEXT: &str = "https://***@api:443 id-2";
        let series = store();
        series.identify(FIRST, "id-1");
        series.identify(SECOND, "id-2");
        let (_, workers) = scrape(&series, || {
            gauge!("health", "worker" => FIRST).set(1.0);
            gauge!("health", "worker" => SECOND).set(0.0);
            gauge!("health", "worker" => TEXT).set(0.5);
        });
        let healths: Vec<&str> = workers
            .lines()
            .filter(|line| line.starts_with("health{"))
            .collect();
        assert_eq!(
            healths,
            [
                "health{worker=\"https://***@api:443 #2\"} 0",
                "health{worker=\"https://***@api:443 id-2\"} 0.5",
                "health{worker=\"https://***@api:443\"} 1",
            ]
        );
    }

    /// A series whose worker value has no decision is left out and its value
    /// is noted: a scrape never falls back to the value itself.
    #[test]
    fn a_series_with_no_decision_is_left_out() {
        const WITH_A_SECRET: &str = "http://user:s3cret@w:8000";
        let key = Key::from_parts("health", vec![Label::new("worker", WITH_A_SECRET)]);
        let mut undecided = BTreeSet::new();
        assert_eq!(label_pairs(&key, &Shown::new(), &mut undecided), None);
        assert_eq!(undecided, BTreeSet::from([WITH_A_SECRET.to_owned()]));

        let mut labels = Labels::default();
        labels.decide(WITH_A_SECRET);
        let mut undecided = BTreeSet::new();
        assert_eq!(
            label_pairs(&key, &labels.shown(), &mut undecided),
            Some(vec![String::from("worker=\"http://***@w:8000\"")])
        );
        assert_eq!(undecided, BTreeSet::new());
    }

    /// An address shown as it is keeps its label before a masked address that
    /// reads the same, whichever was decided first: the masked one carries
    /// its worker id.
    #[test]
    fn a_plain_address_keeps_its_label_before_a_masked_twin() {
        const PLAIN: &str = "grpc://w:8000";
        const TWIN: &str = "grpc://w:8000?token=s3cret";
        let expected = [
            "health{worker=\"grpc://w:8000 id-2\"} 0",
            "health{worker=\"grpc://w:8000\"} 1",
        ];
        for twin_first in [false, true] {
            let series = store();
            // The first of the two alone, decided at its first scrape.
            let first = if twin_first { TWIN } else { PLAIN };
            let (_, workers) = scrape(&series, || {
                gauge!("health", "worker" => first).set(0.5);
            });
            assert_eq!(
                line(&workers, "health{"),
                Some("health{worker=\"grpc://w:8000\"} 0.5"),
                "alone, either reads as the plain address"
            );
            series.identify(TWIN, "id-2");
            let (_, workers) = scrape(&series, || {
                gauge!("health", "worker" => PLAIN).set(1.0);
                gauge!("health", "worker" => TWIN).set(0.0);
            });
            let healths: Vec<&str> = workers
                .lines()
                .filter(|line| line.starts_with("health{"))
                .collect();
            assert_eq!(healths, expected, "the twin first: {twin_first}");
            assert!(!workers.contains("s3cret"), "{workers}");
        }
    }

    /// A worker replaced at a masked address keeps the address's one series.
    #[test]
    fn a_replacement_at_a_masked_address_keeps_one_series() {
        const WITH_A_SECRET: &str = "http://user:s3cret@w:8000";
        let series = store();
        series.identify(WITH_A_SECRET, "id-1");
        let replaced: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(WITH_A_SECRET).build());
        let (_, workers) = scrape(&series, || {
            counter!("requests_total", "worker" => WITH_A_SECRET, "model" => "m").increment(2);
            series.replaced(WITH_A_SECRET, Arc::downgrade(&replaced));
            series.identify(WITH_A_SECRET, "id-1");
            counter!("requests_total", "worker" => WITH_A_SECRET, "model" => "m").increment(3);
        });
        assert_eq!(
            workers
                .lines()
                .filter(|line| line.starts_with("requests_total{"))
                .collect::<Vec<_>>(),
            ["requests_total{worker=\"http://***@w:8000\",model=\"m\"} 5"]
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

        // A dead witness: the test's late writes race the removal, before any upkeep.
        series.retire(WORKER_A, Weak::<BasicWorker>::new());
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
