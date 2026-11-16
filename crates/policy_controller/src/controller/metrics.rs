//! The controller's process series: its state, its start time and the load
//! driver's counts (design 14 names the rest, which arrive with the scopes,
//! the replication and the election). They live on a recorder of the
//! controller's own, not the process global, so a gateway that embeds the
//! controller keeps its recorder and two controllers in one test process
//! keep theirs; the admin listener renders it in the Prometheus text form.
//! The driver's counts are read from its atomics at each render, so the
//! driver carries no recorder handle.

use std::{
    sync::{atomic::Ordering, Arc, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use metrics::{Counter, Gauge, Key, KeyName, Label, Level, Metadata, Recorder};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle, PrometheusRecorder};
use parking_lot::Mutex;

use super::{generations::Generations, readiness::State};
use crate::ingest::loads::{Failures, Stats};

/// `1` for the controller's state (`starting`, `leader`, `follower`,
/// `draining`), `0` for the other three. The admin listener sets it from the
/// readiness at each render, so the role is the fence's at the scrape, and
/// one scrape's sample is one write: two scrapes never mix theirs.
pub const STATE: &str = "smg_policy_controller_state";
/// The controller's start, in seconds since the Unix epoch.
pub const START_TIME_SECONDS: &str = "smg_policy_controller_start_time_seconds";
/// Load polls by the arbitration's decision (`mode`: `poll`, `fallback`,
/// `suppressed`; a suppressed one made no call), each counted once per
/// engine and poll interval.
pub const LOAD_POLLS_TOTAL: &str = "smg_policy_controller_load_polls_total";
/// Load polls that produced no report, by `reason`.
pub const LOAD_POLL_FAILURES_TOTAL: &str = "smg_policy_controller_load_poll_failures_total";
/// Metrics scrapes that produced the engine's sample.
pub const LOAD_SCRAPES_TOTAL: &str = "smg_policy_controller_load_scrapes_total";
/// Metrics scrapes that produced no sample, by `reason`.
pub const LOAD_SCRAPE_FAILURES_TOTAL: &str = "smg_policy_controller_load_scrape_failures_total";
/// Rank rows the load driver wrote.
pub const LOAD_ROWS_TOTAL: &str = "smg_policy_controller_load_rows_total";
/// The generation of the policy spec the controller runs. Rendered by a
/// controller with a spec source, from its first document on.
pub const SPEC_GENERATION: &str = "smg_policy_spec_generation";

/// The controller's series and the handle that renders them.
pub(crate) struct Metrics {
    handle: PrometheusHandle,
    /// One gauge per [`State::ALL`] entry, in that order.
    state: [Gauge; 4],
    loads: LoadCounters,
    /// Held from a scrape's sample of the state to the end of its render:
    /// the four gauges are written from one sample and read back whole.
    scrape: Mutex<()>,
    /// The recorder itself: the spec's series is registered on it when a
    /// spec source is bound, so a controller with none renders none.
    recorder: PrometheusRecorder,
    /// The spec's series and the generations it reads, once bound.
    spec: OnceLock<SpecSeries>,
}

/// The spec's series, set from the generations at each render.
struct SpecSeries {
    generations: Arc<Generations>,
    generation: Gauge,
}

/// The load driver's counts as the recorder's counters, set absolute from
/// its atomics at each render.
struct LoadCounters {
    /// `poll`, `fallback`, `suppressed`.
    polls: [Counter; 3],
    /// In the order of [`Failures::LABELS`].
    poll_failures: Vec<Counter>,
    scrapes: Counter,
    /// In the order of [`Failures::LABELS`].
    scrape_failures: Vec<Counter>,
    rows: Counter,
}

impl Metrics {
    /// A recorder with the two series registered and the start time set.
    ///
    /// No histogram lives here yet, so the recorder needs no upkeep task;
    /// the first histogram brings one.
    pub(crate) fn new() -> Self {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let metadata = Metadata::new(module_path!(), Level::INFO, Some(module_path!()));
        recorder.describe_gauge(
            KeyName::from_const_str(STATE),
            None,
            "1 for the controller's state (starting, leader, follower, draining), 0 for the others"
                .into(),
        );
        recorder.describe_gauge(
            KeyName::from_const_str(START_TIME_SECONDS),
            None,
            "The controller's start, in seconds since the Unix epoch".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_POLLS_TOTAL),
            None,
            "Load polls by the arbitration's decision (poll, fallback, suppressed), one per engine and poll interval".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_POLL_FAILURES_TOTAL),
            None,
            "Load polls that produced no report, by reason".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_SCRAPES_TOTAL),
            None,
            "Metrics scrapes that produced the engine's sample".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_SCRAPE_FAILURES_TOTAL),
            None,
            "Metrics scrapes that produced no sample, by reason".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_ROWS_TOTAL),
            None,
            "Rank rows the load driver wrote".into(),
        );
        let state = State::ALL.map(|state| {
            let key = Key::from_parts(STATE, vec![Label::new("state", state.as_str())]);
            recorder.register_gauge(&key, &metadata)
        });
        let start = recorder.register_gauge(&Key::from_name(START_TIME_SECONDS), &metadata);
        let counter = |name: &'static str, label: &'static str, value: &'static str| {
            recorder.register_counter(
                &Key::from_parts(name, vec![Label::new(label, value)]),
                &metadata,
            )
        };
        let loads = LoadCounters {
            polls: ["poll", "fallback", "suppressed"]
                .map(|mode| counter(LOAD_POLLS_TOTAL, "mode", mode)),
            poll_failures: Failures::LABELS
                .iter()
                .map(|reason| counter(LOAD_POLL_FAILURES_TOTAL, "reason", reason))
                .collect(),
            scrapes: recorder.register_counter(&Key::from_name(LOAD_SCRAPES_TOTAL), &metadata),
            scrape_failures: Failures::LABELS
                .iter()
                .map(|reason| counter(LOAD_SCRAPE_FAILURES_TOTAL, "reason", reason))
                .collect(),
            rows: recorder.register_counter(&Key::from_name(LOAD_ROWS_TOTAL), &metadata),
        };
        start.set(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs_f64(),
        );
        let metrics = Self {
            handle,
            state,
            loads,
            scrape: Mutex::new(()),
            recorder,
            spec: OnceLock::new(),
        };
        metrics.set_state(State::Starting);
        metrics
    }

    /// Renders the generation `generations` runs from here on. Bound once,
    /// when a spec source has applied its first document; a controller with
    /// no source renders no spec series.
    pub(crate) fn spec(&self, generations: Arc<Generations>) {
        self.spec.get_or_init(|| {
            let metadata = Metadata::new(module_path!(), Level::INFO, Some(module_path!()));
            self.recorder.describe_gauge(
                KeyName::from_const_str(SPEC_GENERATION),
                None,
                "The generation of the policy spec the controller runs".into(),
            );
            SpecSeries {
                generation: self
                    .recorder
                    .register_gauge(&Key::from_name(SPEC_GENERATION), &metadata),
                generations,
            }
        });
    }

    /// One scrape's text form: the state sampled once by `state` and its
    /// gauge written, the load driver's counts read, and the render, with
    /// no other scrape between the sample and the render. Two scrapes
    /// across a change of role each render one state, their own.
    pub(crate) fn render(&self, state: impl FnOnce() -> State, loads: &Stats) -> String {
        let _one = self.scrape.lock();
        self.set_state(state());
        self.observe_loads(loads);
        if let Some(spec) = self.spec.get() {
            spec.generation.set(spec.generations.generation() as f64);
        }
        self.handle.render()
    }

    /// Sets the load counters from the driver's atomics.
    fn observe_loads(&self, stats: &Stats) {
        let polls = [&stats.polls, &stats.fallbacks, &stats.suppressed];
        for (counter, count) in self.loads.polls.iter().zip(polls) {
            counter.absolute(count.load(Ordering::Relaxed));
        }
        for (counter, (_, count)) in self
            .loads
            .poll_failures
            .iter()
            .zip(stats.poll_failures.each())
        {
            counter.absolute(count);
        }
        self.loads
            .scrapes
            .absolute(stats.scrapes.load(Ordering::Relaxed));
        for (counter, (_, count)) in self
            .loads
            .scrape_failures
            .iter()
            .zip(stats.scrape_failures.each())
        {
            counter.absolute(count);
        }
        self.loads.rows.absolute(stats.rows.load(Ordering::Relaxed));
    }

    /// Moves the state gauge: `1` on `now`, `0` on the others.
    fn set_state(&self, now: State) {
        for (gauge, state) in self.state.iter().zip(State::ALL) {
            gauge.set(if state == now { 1.0 } else { 0.0 });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use super::{Generations, Metrics, SPEC_GENERATION, START_TIME_SECONDS, STATE};
    use crate::{controller::readiness::State, ingest::loads::Stats};

    fn line(rendered: &str, series: &str, label: &str) -> Option<String> {
        rendered
            .lines()
            .find(|l| l.starts_with(series) && l.contains(label))
            .map(str::to_owned)
    }

    #[test]
    fn the_state_gauge_follows_the_transitions_on_the_controllers_own_recorder() {
        let metrics = Metrics::new();
        let rendered = metrics.render(|| State::Starting, &Stats::default());
        assert_eq!(
            line(&rendered, STATE, "state=\"starting\"").as_deref(),
            Some("smg_policy_controller_state{state=\"starting\"} 1")
        );
        assert_eq!(
            line(&rendered, STATE, "state=\"leader\"").as_deref(),
            Some("smg_policy_controller_state{state=\"leader\"} 0")
        );
        assert!(rendered.contains("# HELP smg_policy_controller_state "));

        let rendered = metrics.render(|| State::Leader, &Stats::default());
        assert_eq!(
            line(&rendered, STATE, "state=\"starting\"").as_deref(),
            Some("smg_policy_controller_state{state=\"starting\"} 0")
        );
        assert_eq!(
            line(&rendered, STATE, "state=\"leader\"").as_deref(),
            Some("smg_policy_controller_state{state=\"leader\"} 1")
        );

        let other = Metrics::new();
        assert_eq!(
            line(
                &other.render(|| State::Starting, &Stats::default()),
                STATE,
                "state=\"leader\""
            )
            .as_deref(),
            Some("smg_policy_controller_state{state=\"leader\"} 0"),
            "a second controller starts from its own recorder"
        );
    }

    #[test]
    fn two_scrapes_across_a_change_of_role_each_render_one_state() {
        use std::{
            sync::{
                atomic::{AtomicBool, Ordering},
                mpsc, Arc,
            },
            thread,
            time::Instant,
        };

        let metrics = Arc::new(Metrics::new());
        let stats = Arc::new(Stats::default());
        let inside = Arc::new(AtomicBool::new(false));
        let (sampling, sampled) = mpsc::channel();
        // The first scrape stays inside its sample while the second begins,
        // and the role changes between the two. It runs on a thread the
        // case does not wait for without a bound.
        let first = thread::spawn({
            let (metrics, stats, inside) = (
                Arc::clone(&metrics),
                Arc::clone(&stats),
                Arc::clone(&inside),
            );
            move || {
                metrics.render(
                    || {
                        inside.store(true, Ordering::SeqCst);
                        sampling.send(()).expect("the case listens");
                        thread::sleep(Duration::from_millis(100));
                        inside.store(false, Ordering::SeqCst);
                        State::Leader
                    },
                    &stats,
                )
            }
        });
        // A scrape that blocks before its sample fails the case by name.
        sampled
            .recv_timeout(Duration::from_secs(20))
            .expect("the first scrape samples within 20 s");
        let mut overlapped = false;
        let second = metrics.render(
            || {
                overlapped = inside.load(Ordering::SeqCst);
                State::Follower
            },
            &stats,
        );
        let waited = Instant::now();
        while !first.is_finished() {
            assert!(
                waited.elapsed() < Duration::from_secs(20),
                "the first scrape did not end in 20 s"
            );
            thread::sleep(Duration::from_millis(2));
        }
        let first = first.join().expect("the first scrape ends");
        assert!(!overlapped, "the second scrape sampled inside the first");
        for (rendered, own) in [(&first, State::Leader), (&second, State::Follower)] {
            for state in State::ALL {
                let label = format!("state=\"{}\"", state.as_str());
                let value = u8::from(state == own);
                assert_eq!(
                    line(rendered, STATE, &label),
                    Some(format!("{STATE}{{{label}}} {value}")),
                    "the scrape that sampled {}",
                    own.as_str()
                );
            }
        }
    }

    #[test]
    fn the_load_counters_render_the_drivers_counts() {
        use std::sync::atomic::Ordering;

        use crate::ingest::loads::ReadError;

        let metrics = Metrics::new();
        let stats = Stats::default();
        let rendered = metrics.render(|| State::Starting, &stats);
        assert!(rendered.contains("smg_policy_controller_load_polls_total{mode=\"poll\"} 0"));
        assert!(rendered.contains("smg_policy_controller_load_rows_total 0"));

        stats.polls.fetch_add(3, Ordering::Relaxed);
        stats.suppressed.fetch_add(2, Ordering::Relaxed);
        stats.rows.fetch_add(7, Ordering::Relaxed);
        stats.scrapes.fetch_add(1, Ordering::Relaxed);
        stats.poll_failures.deadline.fetch_add(1, Ordering::Relaxed);
        let rendered = metrics.render(|| State::Starting, &stats);
        assert!(rendered.contains("smg_policy_controller_load_polls_total{mode=\"poll\"} 3"));
        assert!(rendered.contains("smg_policy_controller_load_polls_total{mode=\"suppressed\"} 2"));
        assert!(rendered
            .contains("smg_policy_controller_load_poll_failures_total{reason=\"deadline\"} 1"));
        assert!(rendered.contains("smg_policy_controller_load_scrapes_total 1"));
        assert!(rendered
            .contains("smg_policy_controller_load_scrape_failures_total{reason=\"no_load\"} 0"));
        assert!(rendered.contains("smg_policy_controller_load_rows_total 7"));
        assert_eq!(
            ReadError::Deadline(Duration::from_secs(3)).label(),
            "deadline"
        );
    }

    #[test]
    fn the_start_time_is_set_once_at_construction() {
        let metrics = Metrics::new();
        let rendered = metrics.render(|| State::Starting, &Stats::default());
        let start = rendered
            .lines()
            .find(|l| l.starts_with(START_TIME_SECONDS) && !l.starts_with('#'))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse::<f64>().ok())
            .expect("the start time line");
        assert!(start > 1.7e9, "seconds since the epoch, read {start}");
    }

    /// A controller with no spec source renders no spec series. Once one is
    /// bound, every render reads the generation that stands at that render.
    #[test]
    fn the_spec_generation_renders_once_a_source_is_bound() {
        use crate::{
            controller::generations::{
                tests::{LEAST_LOAD, RANDOM},
                Format,
            },
            store::MemoryStore,
        };

        let metrics = Metrics::new();
        let rendered = metrics.render(|| State::Leader, &Stats::default());
        assert!(!rendered.contains("smg_policy_spec"), "{rendered}");

        let generations = Arc::new(Generations::new(
            Arc::new(MemoryStore::new()),
            Arc::new(crate::Catalog::builtin()),
        ));
        metrics.spec(Arc::clone(&generations));
        for (document, generation) in [(LEAST_LOAD, 1), (RANDOM, 2)] {
            generations
                .apply(document.into(), Format::Yaml)
                .expect("a valid document");
            let rendered = metrics.render(|| State::Leader, &Stats::default());
            let line = format!("{SPEC_GENERATION} {generation}");
            assert!(
                rendered.lines().any(|rendered| rendered == line),
                "no line {line:?} in {rendered}"
            );
        }
    }
}
