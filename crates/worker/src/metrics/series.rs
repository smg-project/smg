//! Per-worker series that leave `/metrics` with their worker, and histogram
//! series that record into fixed cells.
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
//! Histograms record here in one of two ways. A series of a family the bucket
//! configuration renders as a histogram records into fixed cells (one count a
//! bucket and a sum), so it holds the same bytes however many samples it
//! takes; a series of any other family keeps its samples until the upkeep
//! folds them. The per-worker histograms take this road for every host. The
//! bucketed families with no `worker` label stay with the exporter unless the
//! host asks the store to keep them too ([`keep_histograms`]); they then
//! render here as well, and no family is split between the two texts.

use std::{
    borrow::Cow,
    cell::Cell,
    collections::{BTreeMap, HashMap},
    fmt::{Display, Write as _},
    sync::{
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
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

use crate::identity::basic::Worker;

/// The label that keys a series to a worker; its value is the worker URL.
const WORKER_LABEL: &str = "worker";

/// The store the installed recorder routes per-worker series to, and the
/// bucketed histograms a host has it keep.
static INSTALLED: OnceLock<Arc<WorkerSeries>> = OnceLock::new();

/// Install the store the global recorder routes to. The first call wins, as
/// the recorder installation it accompanies does.
pub fn install(series: Arc<WorkerSeries>) {
    let _ = INSTALLED.set(series);
}

/// The installed store, if the Prometheus recorder has been installed.
pub(crate) fn installed() -> Option<&'static Arc<WorkerSeries>> {
    INSTALLED.get()
}

/// The installed store, a default one (the exporter's quantiles, no
/// buckets) installed first when the process has none: what a test of the
/// registry's handoff to the store reads.
#[cfg(any(test, feature = "test-util"))]
pub fn installed_or_default() -> &'static Arc<WorkerSeries> {
    INSTALLED.get_or_init(|| Arc::new(WorkerSeries::with_exporter_quantiles()))
}

#[cfg(any(test, feature = "test-util"))]
impl WorkerSeries {
    /// A store as a test builds one: a copy of the exporter's default
    /// quantiles, spelled here, and no buckets.
    pub(crate) fn with_exporter_quantiles() -> Self {
        Self::new(DistributionBuilder::new(
            metrics_util::parse_quantiles(&[0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]),
            None,
            None,
            None,
            None,
            None,
        ))
    }
}

/// Append the installed store's series to a scrape.
pub fn render_into(out: &mut String) {
    if let Some(series) = INSTALLED.get() {
        series.render_into(out);
    }
}

/// Have the installed store keep the histogram families with no `worker`
/// label too: a series of such a family that registers from this call on
/// records into the store's fixed cells and renders with [`render_into`],
/// not in the exporter's text. It is for a host whose scrape appends
/// [`render_into`] to the exporter's text. A host that renders the
/// exporter's handle alone does not call it, and reads every such family
/// there as before. No family is split between the two texts: the call takes
/// only while no such series has registered with the exporter.
///
/// Returns whether the store keeps those families from here on. `false`
/// leaves every one of them with the exporter: no store is installed, or a
/// series of one, a summary too, registered before the call.
#[must_use]
pub fn keep_histograms() -> bool {
    INSTALLED
        .get()
        .is_some_and(|series| series.keep_histograms())
}

/// Fold the installed store's histogram samples, as the exporter's upkeep does.
pub fn run_upkeep() {
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

/// The per-worker series of the process, keyed by the worker URL.
pub struct WorkerSeries {
    /// The addresses with a removed registration or a replaced worker (see
    /// [`Address`]); a retired one's writes are dropped until the address
    /// registers again or the upkeep finds every one of its workers gone.
    /// Handle creation holds the read side, retirement the write side, so no
    /// series is created for an address between its retirement and the drop
    /// of its series.
    addresses: RwLock<HashMap<String, Address>>,
    /// The upkeep forgot a retired address since the last registration: a
    /// thread may still keep a no-op handle it built for that address.
    forgot_retired: AtomicBool,
    registry: Registry<Key, WorkerStorage>,
    /// Descriptions by sanitized metric name, for the `# HELP` lines.
    descriptions: RwLock<HashMap<String, SharedString>>,
    distribution_builder: Arc<DistributionBuilder>,
    /// Histogram samples aggregated per series, the exporter's way.
    distributions: Mutex<HashMap<Key, Distribution>>,
    /// Whether a family renders as a histogram, by its name as a caller
    /// writes it: the bucket configuration is asked once a name.
    bucketed: RwLock<HashMap<String, bool>>,
    /// Who keeps the histogram families keyed to no worker: [`UNSETTLED`],
    /// [`WITH_EXPORTER`] or [`IN_CELLS`].
    unkeyed_histograms: AtomicU8,
}

/// No histogram series with no `worker` label has registered yet, and the
/// host has not asked the store to keep them.
const UNSETTLED: u8 = 0;
/// Such a series registered with the exporter before the host asked: every
/// family of them stays with the exporter, so none is split between the
/// exporter's text and the store's.
const WITH_EXPORTER: u8 = 1;
/// The host asked first ([`keep_histograms`]): such a family records in the
/// store's cells where the bucket configuration renders it as a histogram.
const IN_CELLS: u8 = 2;

impl WorkerSeries {
    /// A store whose histograms take the exporter's bucket configuration.
    pub fn new(distribution_builder: DistributionBuilder) -> Self {
        let distribution_builder = Arc::new(distribution_builder);
        Self {
            addresses: RwLock::new(HashMap::new()),
            forgot_retired: AtomicBool::new(false),
            registry: Registry::new(WorkerStorage(Arc::clone(&distribution_builder))),
            descriptions: RwLock::new(HashMap::new()),
            distribution_builder,
            distributions: Mutex::new(HashMap::new()),
            bucketed: RwLock::new(HashMap::new()),
            unkeyed_histograms: AtomicU8::new(UNSETTLED),
        }
    }

    /// A worker registered at `url`: its series record (again). The workers
    /// that served the address before stay known while a handle holds one.
    /// Says whether a kept handle may be a no-op one, built while an address
    /// was retired: this address until now, or one the upkeep forgot since
    /// the last registration.
    pub(crate) fn activate(&self, url: &str) -> bool {
        let mut addresses = self.addresses.write();
        let mut noop_kept = self.forgot_retired.swap(false, Ordering::AcqRel);
        let Some(address) = addresses.get_mut(url) else {
            return noop_kept;
        };
        noop_kept |= std::mem::take(&mut address.retired);
        let nothing_to_keep = address.writers.is_empty();
        if nothing_to_keep {
            addresses.remove(url);
        }
        noop_kept
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
    pub fn retire(&self, url: &str, worker: Weak<dyn Worker>) {
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
        self.registry.get_or_create_histogram(key, Samples::handle)
    }

    /// Keep the histogram families with no worker label too, unless a series
    /// of one has registered with the exporter (see [`keep_histograms`]).
    /// Whether the store keeps them.
    fn keep_histograms(&self) -> bool {
        let settled = self.unkeyed_histograms.compare_exchange(
            UNSETTLED,
            IN_CELLS,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        settled.is_ok() || settled == Err(IN_CELLS)
    }

    /// The handle of `key`, a series with no worker label, where the host
    /// asked the store to keep such families and the bucket configuration
    /// renders this one as a histogram: the series records into the store's
    /// fixed cells. `None` for a family that stays with the exporter: every
    /// one until the host asks, every one for good once a series registered
    /// before the host asked, and a family that renders as a summary.
    fn bucketed(&self, key: &Key) -> Option<Histogram> {
        match self.unkeyed_histograms.load(Ordering::Relaxed) {
            IN_CELLS => {}
            WITH_EXPORTER => return None,
            _ => {
                // The first series to register settles it for them all.
                let settled = self.unkeyed_histograms.compare_exchange(
                    UNSETTLED,
                    WITH_EXPORTER,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                if settled != Err(IN_CELLS) {
                    return None;
                }
            }
        }
        let known = self.bucketed.read().get(key.name()).copied();
        let bucketed = known.unwrap_or_else(|| {
            let kind = self
                .distribution_builder
                .get_distribution_type(&sanitize_metric_name(key.name()));
            let bucketed = kind == "histogram";
            self.bucketed
                .write()
                .insert(key.name().to_owned(), bucketed);
            bucketed
        });
        bucketed.then(|| self.registry.get_or_create_histogram(key, Samples::handle))
    }

    /// The retired addresses, for the tests.
    #[cfg(any(test, feature = "test-util"))]
    pub fn retired_len(&self) -> usize {
        self.addresses
            .read()
            .values()
            .filter(|address| address.retired)
            .count()
    }

    /// Whether `worker_url` is retired, for the tests.
    #[cfg(any(test, feature = "test-util"))]
    pub fn is_retired(&self, worker_url: &str) -> bool {
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
    pub fn run_upkeep(&self) {
        self.addresses.write().retain(|_, address| {
            address.writers.retain(|writer| writer.strong_count() > 0);
            let referenced = !address.writers.is_empty();
            if address.retired && !referenced {
                self.forgot_retired.store(true, Ordering::Release);
            }
            referenced
        });
        let mut distributions = self.distributions.lock();
        self.registry.visit_histograms(|key, samples| {
            let Samples::Timed(samples) = samples else {
                // Fixed cells hold no sample to fold.
                return;
            };
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
    pub fn render_into(&self, out: &mut String) {
        let descriptions = self.descriptions.read();

        let mut counters: BTreeMap<String, Vec<(String, u64)>> = BTreeMap::new();
        self.registry.visit_counters(|key, counter| {
            let sample = (label_text(key), counter.load(Ordering::Acquire));
            gather(&mut counters, key.name(), sample);
        });
        for (name, mut series) in counters {
            series.sort_by(|a, b| a.0.cmp(&b.0));
            write_family(out, &name, "counter", descriptions.get(&name));
            for (labels, value) in &series {
                write_sample(out, &name, None, labels, None, value);
            }
            out.push('\n');
        }

        let mut gauges: BTreeMap<String, Vec<(String, f64)>> = BTreeMap::new();
        self.registry.visit_gauges(|key, gauge| {
            let sample = (
                label_text(key),
                f64::from_bits(gauge.load(Ordering::Acquire)),
            );
            gather(&mut gauges, key.name(), sample);
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
        let mut by_name: BTreeMap<String, Vec<(String, Folded<'_>)>> = BTreeMap::new();
        for (key, distribution) in distributions.iter() {
            let series = (label_text(key), Folded::Kept(distribution));
            gather(&mut by_name, key.name(), series);
        }
        self.registry.visit_histograms(|key, samples| {
            if let Samples::Cells(cells) = samples {
                let series = (label_text(key), Folded::Cells(Arc::clone(cells)));
                gather(&mut by_name, key.name(), series);
            }
        });
        for (name, mut series) in by_name {
            let kind = self.distribution_builder.get_distribution_type(&name);
            if kind == "native_histogram" {
                // Only the protobuf exposition carries native histograms.
                continue;
            }
            series.sort_by(|a, b| a.0.cmp(&b.0));
            write_family(out, &name, kind, descriptions.get(&name));
            let mut series = series.into_iter().peekable();
            while let Some((labels, folded)) = series.next() {
                let (sum, count) = match folded {
                    Folded::Cells(cells) => {
                        // The keys that clean to this family and these labels
                        // are one series, as the exporter folds them: their
                        // cells add up. They share the family's bounds.
                        let mut alike = Vec::new();
                        while let Some((_, Folded::Cells(more))) = series.next_if(|(next, more)| {
                            *next == labels && matches!(more, Folded::Cells(_))
                        }) {
                            alike.push(more);
                        }
                        let in_cell = |cell: usize| {
                            cells.count(cell)
                                + alike.iter().map(|more| more.count(cell)).sum::<u64>()
                        };
                        let mut count = 0_u64;
                        for (cell, le) in cells.bounds.iter().enumerate() {
                            count += in_cell(cell);
                            write_sample(
                                out,
                                &name,
                                Some("bucket"),
                                &labels,
                                Some(("le", le)),
                                &count,
                            );
                        }
                        count += in_cell(cells.bounds.len());
                        write_sample(
                            out,
                            &name,
                            Some("bucket"),
                            &labels,
                            Some(("le", &"+Inf")),
                            &count,
                        );
                        let sum = cells.sum() + alike.iter().map(|more| more.sum()).sum::<f64>();
                        (sum, count)
                    }
                    Folded::Kept(Distribution::Summary(summary, quantiles, sum)) => {
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
                    Folded::Kept(Distribution::Histogram(histogram)) => {
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
                    Folded::Kept(Distribution::NativeHistogram(_)) => continue,
                };
                write_sample(out, &name, Some("sum"), &labels, None, &sum);
                write_sample(out, &name, Some("count"), &labels, None, &count);
            }
            out.push('\n');
        }
    }
}

/// The gateway's recorder: series labelled `worker` go to the store, and so
/// do the bucketed histograms with no such label once a host has asked the
/// store to keep them ([`keep_histograms`]); every other series goes to the
/// Prometheus exporter.
pub struct WorkerSeriesRecorder<R> {
    inner: R,
    series: Arc<WorkerSeries>,
}

impl<R> WorkerSeriesRecorder<R> {
    pub fn new(inner: R, series: Arc<WorkerSeries>) -> Self {
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
            None => match self.series.bucketed(key) {
                Some(histogram) => histogram,
                None => self.inner.register_histogram(key, metadata),
            },
        }
    }
}

/// Storage of the store's registry: the exporter's atomic counters and
/// gauges; a histogram series in fixed cells where the bucket configuration
/// renders its family as a histogram, and else its samples stamped with
/// their time of recording so the exporter's rolling summaries age them as
/// the exporter's own.
#[derive(Debug)]
struct WorkerStorage(Arc<DistributionBuilder>);

impl Storage<Key> for WorkerStorage {
    type Counter = Arc<AtomicU64>;
    type Gauge = Arc<AtomicU64>;
    type Histogram = Samples;

    fn counter(&self, _: &Key) -> Self::Counter {
        Arc::new(AtomicU64::new(0))
    }

    fn gauge(&self, _: &Key) -> Self::Gauge {
        Arc::new(AtomicU64::new(0))
    }

    fn histogram(&self, key: &Key) -> Self::Histogram {
        match self.0.get_distribution(&sanitize_metric_name(key.name())) {
            Distribution::Histogram(histogram) => Samples::Cells(Arc::new(BucketCells::new(
                histogram.buckets().into_iter().map(|(bound, _)| bound),
                STRIPES,
            ))),
            Distribution::Summary(..) | Distribution::NativeHistogram(_) => {
                Samples::Timed(Arc::new(TimedSamples(AtomicBucket::new())))
            }
        }
    }
}

/// What a histogram series of the store records into.
#[derive(Clone, Debug)]
enum Samples {
    /// The samples, kept until the upkeep folds them.
    Timed(Arc<TimedSamples>),
    /// One count a bucket and a sum; no sample is kept.
    Cells(Arc<BucketCells>),
}

impl Samples {
    /// A handle that records straight into what the series keeps.
    fn handle(&self) -> Histogram {
        match self {
            Self::Timed(samples) => Histogram::from_arc(Arc::clone(samples)),
            Self::Cells(cells) => Histogram::from_arc(Arc::clone(cells)),
        }
    }
}

/// The registry's storage asks this of its histogram type. No record takes
/// this road: every handle comes from [`Samples::handle`].
impl HistogramFn for Samples {
    fn record(&self, value: f64) {
        match self {
            Self::Timed(samples) => samples.record(value),
            Self::Cells(cells) => cells.record(value),
        }
    }
}

/// A series as a scrape reads it: a distribution the upkeep folded, or the
/// series' cells.
enum Folded<'d> {
    Kept(&'d Distribution),
    Cells(Arc<BucketCells>),
}

/// Histogram samples with their time of recording.
#[derive(Debug)]
struct TimedSamples(AtomicBucket<(f64, Instant)>);

impl HistogramFn for TimedSamples {
    fn record(&self, value: f64) {
        self.0.push((value, Instant::now()));
    }
}

/// The stripes of a series in cells. A thread records into the stripe of its
/// number, so callers on different threads meet on no word until there are
/// more threads than stripes. A power of two.
const STRIPES: usize = 8;

/// The words of a cache line.
const LINE_WORDS: usize = 8;

/// One cache line of a stripe's words. A stripe is whole lines, so two
/// stripes share none.
#[derive(Debug, Default)]
#[repr(align(64))]
struct Line([AtomicU64; LINE_WORDS]);

thread_local! {
    /// The calling thread's number among the threads that recorded into
    /// cells; `usize::MAX` until it takes one.
    static THREAD_NUMBER: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// The threads that took a number.
static THREADS: AtomicUsize = AtomicUsize::new(0);

/// The calling thread's number: how many threads took theirs before it.
fn thread_number() -> usize {
    THREAD_NUMBER
        .try_with(|number| {
            if number.get() == usize::MAX {
                number.set(THREADS.fetch_add(1, Ordering::Relaxed));
            }
            number.get()
        })
        .unwrap_or(0)
}

/// A histogram series in fixed cells. A sample counts in the cell of the
/// first bound, in the order the configuration lists them, that it is no
/// greater than, as the exporter's fold places it; a sample over every bound,
/// and a NaN, counts in the last cell, which only the `+Inf` line and the
/// count read. The cells stand once a stripe, and a scrape adds the stripes
/// up bound by bound.
#[derive(Debug)]
struct BucketCells {
    bounds: Box<[f64]>,
    /// The stripes, one behind the other. The first word of a stripe is the
    /// sum of its samples, as the bits of an `f64`; behind it stand one count
    /// a bound and one more for the samples over every bound.
    lines: Box<[Line]>,
    /// The lines of one stripe.
    stripe_lines: usize,
    /// The number of the last stripe.
    last_stripe: usize,
}

impl BucketCells {
    fn new(bounds: impl Iterator<Item = f64>, stripes: usize) -> Self {
        let bounds: Box<[f64]> = bounds.collect();
        let stripes = stripes.max(1).next_power_of_two();
        let stripe_lines = (bounds.len() + 2).div_ceil(LINE_WORDS);
        Self {
            bounds,
            lines: (0..stripes * stripe_lines)
                .map(|_| Line::default())
                .collect(),
            stripe_lines,
            last_stripe: stripes - 1,
        }
    }

    /// The `word`th word of `stripe`.
    fn word(&self, stripe: usize, word: usize) -> Option<&AtomicU64> {
        self.lines
            .get(stripe * self.stripe_lines + word / LINE_WORDS)?
            .0
            .get(word % LINE_WORDS)
    }

    /// The samples counted in `cell`, over every stripe.
    fn count(&self, cell: usize) -> u64 {
        (0..=self.last_stripe)
            .filter_map(|stripe| self.word(stripe, 1 + cell))
            .map(|count| count.load(Ordering::Relaxed))
            .sum()
    }

    /// The sum of the samples, over every stripe.
    fn sum(&self) -> f64 {
        (0..=self.last_stripe)
            .filter_map(|stripe| self.word(stripe, 0))
            .map(|sum| f64::from_bits(sum.load(Ordering::Relaxed)))
            .fold(0.0, |sum, stripe| sum + stripe)
    }
}

impl HistogramFn for BucketCells {
    fn record(&self, value: f64) {
        let cell = self
            .bounds
            .iter()
            .position(|bound| value <= *bound)
            .unwrap_or(self.bounds.len());
        let stripe = thread_number() & self.last_stripe;
        if let Some(count) = self.word(stripe, 1 + cell) {
            count.fetch_add(1, Ordering::Relaxed);
        }
        let Some(sum) = self.word(stripe, 0) else {
            return;
        };
        let mut seen = sum.load(Ordering::Relaxed);
        loop {
            let next = (f64::from_bits(seen) + value).to_bits();
            match sum.compare_exchange_weak(seen, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(now) => seen = now,
            }
        }
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

/// `text` as `clean` writes it, borrowed when that is `text` itself: `valid`
/// holds for each of its characters at its index. A scrape cleans the names
/// and labels of every series, and nearly all of them are clean already.
fn cleaned<'t>(
    text: &'t str,
    valid: impl Fn(usize, char) -> bool,
    clean: impl FnOnce(&str) -> String,
) -> Cow<'t, str> {
    if text.chars().enumerate().all(|(i, c)| valid(i, c)) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(clean(text))
    }
}

/// Whether the exporter's cleaner keeps `c` at index `i` of a family name.
/// The three rules repeat its own (`metrics_exporter_prometheus::formatting`),
/// which are private to it.
fn metric_name_char(i: usize, c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == ':' || (i != 0 && c.is_ascii_digit())
}

/// The same for a label name.
fn label_key_char(i: usize, c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || (i != 0 && c.is_ascii_digit())
}

/// The same for a label value: every character but the three it escapes.
fn label_value_char(_: usize, c: char) -> bool {
    !matches!(c, '\n' | '"' | '\\')
}

/// Put `sample` with the family `name` cleans to; a name that is clean
/// already is allocated for the family's first sample alone.
fn gather<V>(families: &mut BTreeMap<String, Vec<V>>, name: &str, sample: V) {
    let name = cleaned(name, metric_name_char, sanitize_metric_name);
    match families.get_mut(name.as_ref()) {
        Some(family) => family.push(sample),
        None => {
            families.insert(name.into_owned(), vec![sample]);
        }
    }
}

/// The labels of `key` as a sample line prints them, in one string. A cleaned
/// pair ends with its closing quote and a quote inside a value is escaped, so
/// no pair is the start of another: the samples of a family sort by this
/// text as they would by the list of its pairs.
fn label_text(key: &Key) -> String {
    let pairs = key.labels();
    let mut text = String::with_capacity(
        pairs
            .clone()
            .map(|label| label.key().len() + label.value().len() + 4)
            .sum(),
    );
    for label in pairs {
        if !text.is_empty() {
            text.push(',');
        }
        text.push_str(&cleaned(label.key(), label_key_char, sanitize_label_key));
        text.push_str("=\"");
        text.push_str(&cleaned(
            label.value(),
            label_value_char,
            sanitize_label_value,
        ));
        text.push('"');
    }
    text
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
    labels: &str,
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
        out.push_str(labels);
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
    use crate::identity::basic::BasicWorker;

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

    /// A scrape writes the labels of a series straight from its key and
    /// calls the exporter's cleaners only for a text that needs them: a
    /// clean text is borrowed, and the page must read as those cleaners
    /// write every name, and keep its samples in the order of their label
    /// pairs.
    #[test]
    fn a_scrape_reads_as_the_exporters_cleaners_write_and_order_it() {
        // A clean family name, label name and worker address are borrowed
        // under the rules the scrape applies.
        assert!(matches!(
            cleaned("requests_total", metric_name_char, sanitize_metric_name),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            cleaned(WORKER_LABEL, label_key_char, sanitize_label_key),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            cleaned(WORKER_A, label_value_char, sanitize_label_value),
            Cow::Borrowed(WORKER_A)
        ));

        // Every ASCII character and one beyond: first in a text, later in
        // it, and behind a backslash.
        let texts: Vec<String> = (0..=0x7f_u8)
            .map(char::from)
            .chain(['é'])
            .flat_map(|c| [format!("{c}"), format!("a{c}"), format!("\\{c}")])
            .collect();
        let series = store();
        let (_, workers) = scrape(&series, || {
            gauge!("labels", "worker" => "a").set(1.0);
            for text in &texts {
                counter!(text.clone(), "worker" => "w").increment(1);
                let labels = vec![
                    Label::new("worker", text.clone()),
                    Label::new(text.clone(), "v"),
                ];
                gauge!("labels", labels).set(1.0);
            }
        });

        let mut families: BTreeMap<String, usize> = BTreeMap::new();
        for text in &texts {
            *families.entry(sanitize_metric_name(text)).or_default() += 1;
        }
        let mut expected = String::new();
        for (name, samples) in families {
            expected.push_str(&format!("# TYPE {name} counter\n"));
            expected.push_str(&format!("{name}{{worker=\"w\"}} 1\n").repeat(samples));
            expected.push('\n');
        }
        let mut pairs: Vec<Vec<String>> = texts
            .iter()
            .map(|text| {
                vec![
                    format!("worker=\"{}\"", sanitize_label_value(text)),
                    format!("{}=\"v\"", sanitize_label_key(text)),
                ]
            })
            .collect();
        pairs.push(vec![String::from("worker=\"a\"")]);
        pairs.sort();
        expected.push_str("# TYPE labels gauge\n");
        for pairs in pairs {
            expected.push_str(&format!("labels{{{}}} 1\n", pairs.join(",")));
        }
        expected.push('\n');
        assert_eq!(workers, expected);
    }

    /// The lines of the family `name` in a scrape: its `# HELP`, its `# TYPE`
    /// and its samples.
    fn family<'t>(rendered: &'t str, name: &str) -> Vec<&'t str> {
        rendered
            .lines()
            .filter(|line| {
                let line = line
                    .strip_prefix("# HELP ")
                    .or_else(|| line.strip_prefix("# TYPE "))
                    .unwrap_or(line);
                line.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(['_', '{', ' ']))
            })
            .collect()
    }

    /// What the exporter alone writes for `samples` of `apply_seconds`,
    /// under the store's buckets.
    fn exporter_alone(samples: &[f64]) -> String {
        let recorder = PrometheusBuilder::new()
            .set_buckets_for_metric(Matcher::Full(String::from("apply_seconds")), &[0.001, 0.01])
            .unwrap_or_else(|error| panic!("the buckets: {error}"))
            .build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            metrics::describe_histogram!("apply_seconds", "Time of an apply");
            let series = histogram!("apply_seconds", "model" => "m");
            for sample in samples {
                series.record(*sample);
            }
        });
        handle.render()
    }

    #[test]
    fn a_bucketed_family_reads_in_the_store_as_the_exporter_writes_it() {
        let lists: [&[f64]; 5] = [
            // Under the first bound, on a bound, between two, on the last
            // and over it.
            &[0.0005, 0.001, 0.005, 0.01, 0.5, 2.0],
            &[0.25, f64::NAN, 0.001],
            &[f64::INFINITY, 0.01],
            &[f64::NEG_INFINITY, -1.0],
            &[],
        ];
        for samples in lists {
            let series = store();
            assert!(series.keep_histograms());
            let (exporter, stored) = scrape(&series, || {
                metrics::describe_histogram!("apply_seconds", "Time of an apply");
                let series = histogram!("apply_seconds", "model" => "m");
                for sample in samples {
                    series.record(*sample);
                }
            });
            assert!(!exporter.contains("apply_seconds"), "{exporter}");
            let alone = exporter_alone(samples);
            assert_eq!(
                family(&stored, "apply_seconds"),
                family(&alone, "apply_seconds"),
                "{samples:?}"
            );
            assert_eq!(
                family(&stored, "apply_seconds").len(),
                7,
                "{samples:?}: {stored}"
            );
        }
    }

    #[test]
    fn two_names_that_clean_to_one_family_read_as_one_series() {
        // Sums of powers of two: exact in whatever order they are added.
        let (under, between, over) = (0.000_244_140_625, 0.007_812_5, 0.5);
        let series = store();
        assert!(series.keep_histograms());
        let (exporter, stored) = scrape(&series, || {
            metrics::describe_histogram!("apply_seconds", "Time of an apply");
            histogram!("apply.seconds", "model" => "m").record(under);
            let cleaned = histogram!("apply_seconds", "model" => "m");
            cleaned.record(between);
            cleaned.record(over);
        });
        assert!(!exporter.contains("apply_seconds"), "{exporter}");
        assert_eq!(
            family(&stored, "apply_seconds"),
            family(&exporter_alone(&[under, between, over]), "apply_seconds"),
            "{stored}"
        );
        assert_eq!(
            line(&stored, "apply_seconds_count"),
            Some("apply_seconds_count{model=\"m\"} 3")
        );
    }

    #[test]
    fn two_names_under_one_worker_label_read_as_one_series() {
        let series = store();
        let (_, stored) = scrape(&series, || {
            histogram!("apply.seconds", "worker" => WORKER_A).record(0.007_812_5);
            let cleaned = histogram!("apply_seconds", "worker" => WORKER_A);
            cleaned.record(0.007_812_5);
            cleaned.record(0.5);
        });
        let expected = format!(
            "# TYPE apply_seconds histogram\n\
             apply_seconds_bucket{{worker=\"{WORKER_A}\",le=\"0.001\"}} 0\n\
             apply_seconds_bucket{{worker=\"{WORKER_A}\",le=\"0.01\"}} 2\n\
             apply_seconds_bucket{{worker=\"{WORKER_A}\",le=\"+Inf\"}} 3\n\
             apply_seconds_sum{{worker=\"{WORKER_A}\"}} 0.515625\n\
             apply_seconds_count{{worker=\"{WORKER_A}\"}} 3"
        );
        assert_eq!(
            family(&stored, "apply_seconds"),
            expected.lines().collect::<Vec<_>>(),
            "{stored}"
        );
    }

    #[test]
    fn a_family_with_no_worker_label_stays_with_the_exporter_until_the_host_asks() {
        let series = store();
        let (exporter, stored) = scrape(&series, || {
            let series = histogram!("apply_seconds", "model" => "m");
            for sample in [0.0005, 0.005, 0.5] {
                series.record(sample);
            }
        });
        assert!(!stored.contains("apply_seconds"), "{stored}");
        let count = "apply_seconds_count{model=\"m\"} 3";
        assert!(exporter.lines().any(|line| line == count), "{exporter}");
    }

    #[test]
    fn asking_after_a_series_reached_the_exporter_moves_no_family() {
        let series = store();
        let (exporter, stored) = scrape(&series, || {
            histogram!("apply_seconds", "model" => "m").record(0.005);
            assert!(!series.keep_histograms());
            histogram!("apply_seconds", "model" => "n").record(0.005);
        });
        assert!(!stored.contains("apply_seconds"), "{stored}");
        for count in [
            "apply_seconds_count{model=\"m\"} 1",
            "apply_seconds_count{model=\"n\"} 1",
        ] {
            assert!(exporter.lines().any(|line| line == count), "{exporter}");
        }
    }

    #[test]
    fn a_summary_that_registered_first_leaves_every_family_with_the_exporter() {
        let series = store();
        let (exporter, stored) = scrape(&series, || {
            histogram!("lag_seconds").record(0.005);
            assert!(!series.keep_histograms());
            histogram!("apply_seconds", "model" => "m").record(0.005);
        });
        assert!(!stored.contains("apply_seconds"), "{stored}");
        let count = "apply_seconds_count{model=\"m\"} 1";
        assert!(exporter.lines().any(|line| line == count), "{exporter}");
    }

    #[test]
    fn a_summary_with_no_worker_label_stays_with_the_exporter_when_the_host_asked() {
        let series = store();
        assert!(series.keep_histograms());
        let (exporter, stored) = scrape(&series, || {
            histogram!("lag_seconds").record(0.005);
        });
        assert!(!stored.contains("lag_seconds"), "{stored}");
        let count = "lag_seconds_count 1";
        assert!(exporter.lines().any(|line| line == count), "{exporter}");
    }

    #[test]
    fn the_samples_of_many_threads_add_up_in_a_scrape() {
        // Sums of powers of two are exact in any order: the first is under
        // the first bound, the second under the last, the third over it.
        let samples = [0.000_976_562_5, 0.007_812_5, 0.5];
        // More threads than a series has stripes, so some share one.
        let threads = 3 * STRIPES;
        let series = store();
        assert!(series.keep_histograms());
        let (_, stored) = scrape(&series, || {
            metrics::describe_histogram!("apply_seconds", "Time of an apply");
            let histogram = histogram!("apply_seconds", "model" => "m");
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        for sample in samples {
                            histogram.record(sample);
                        }
                    });
                }
            });
        });
        let alone = exporter_alone(&samples.repeat(threads));
        assert_eq!(
            family(&stored, "apply_seconds"),
            family(&alone, "apply_seconds"),
            "{stored}"
        );
        let count = format!("apply_seconds_count{{model=\"m\"}} {}", 3 * threads);
        assert!(stored.lines().any(|line| line == count), "{stored}");
    }
}
