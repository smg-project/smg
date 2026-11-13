//! The controller's process series: its state, its start time and the load
//! driver's counts (design 14 names the rest, which arrive with the scopes,
//! the replication and the election). They live on a recorder of the
//! controller's own, not the process global, so a gateway that embeds the
//! controller keeps its recorder and two controllers in one test process
//! keep theirs; the admin listener renders it in the Prometheus text form.
//! The driver's counts are read from its atomics at each render, so the
//! driver carries no recorder handle.

use std::{
    sync::atomic::Ordering,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use metrics::{Counter, Gauge, Key, KeyName, Label, Level, Metadata, Recorder};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use super::readiness::State;
use crate::ingest::loads::{Failures, Stats};

/// `1` for the controller's state (`starting`, `ready`, `draining`), `0` for
/// the other two.
pub const STATE: &str = "smg_policy_controller_state";
/// The controller's start, in seconds since the Unix epoch.
pub const START_TIME_SECONDS: &str = "smg_policy_controller_start_time_seconds";
/// Load polls by the arbitration's decision (`mode`: `poll`, `fallback`,
/// `suppressed`; a suppressed one made no call), each counted once per
/// engine and poll interval.
pub const LOAD_POLLS_TOTAL: &str = "smg_policy_controller_load_polls_total";
/// Load polls that produced no report, by `reason`; under `no_load`, a poll
/// whose report left a rank without a row.
pub const LOAD_POLL_FAILURES_TOTAL: &str = "smg_policy_controller_load_poll_failures_total";
/// Metrics scrapes that wrote the engine's row: one whose rank took a record
/// of its own while the call was out wrote none and is not counted.
pub const LOAD_SCRAPES_TOTAL: &str = "smg_policy_controller_load_scrapes_total";
/// Metrics scrapes that produced no sample, by `reason`.
pub const LOAD_SCRAPE_FAILURES_TOTAL: &str = "smg_policy_controller_load_scrape_failures_total";
/// Rank rows the load driver wrote.
pub const LOAD_ROWS_TOTAL: &str = "smg_policy_controller_load_rows_total";

/// The controller's series and the handle that renders them.
pub(crate) struct Metrics {
    handle: PrometheusHandle,
    /// One gauge per [`State::ALL`] entry, in that order.
    state: [Gauge; 3],
    loads: LoadCounters,
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
            "1 for the controller's state (starting, ready, draining), 0 for the others".into(),
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
            "Metrics scrapes that wrote the engine's row".into(),
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
        };
        metrics.set_state(State::Starting);
        metrics
    }

    /// The text form, the load driver's counts read first.
    pub(crate) fn render(&self, loads: &Stats) -> String {
        self.observe_loads(loads);
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
    pub(crate) fn set_state(&self, now: State) {
        for (gauge, state) in self.state.iter().zip(State::ALL) {
            gauge.set(if state == now { 1.0 } else { 0.0 });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Metrics, START_TIME_SECONDS, STATE};
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
        let rendered = metrics.render(&Stats::default());
        assert_eq!(
            line(&rendered, STATE, "state=\"starting\"").as_deref(),
            Some("smg_policy_controller_state{state=\"starting\"} 1")
        );
        assert_eq!(
            line(&rendered, STATE, "state=\"ready\"").as_deref(),
            Some("smg_policy_controller_state{state=\"ready\"} 0")
        );
        assert!(rendered.contains("# HELP smg_policy_controller_state "));

        metrics.set_state(State::Ready);
        let rendered = metrics.render(&Stats::default());
        assert_eq!(
            line(&rendered, STATE, "state=\"starting\"").as_deref(),
            Some("smg_policy_controller_state{state=\"starting\"} 0")
        );
        assert_eq!(
            line(&rendered, STATE, "state=\"ready\"").as_deref(),
            Some("smg_policy_controller_state{state=\"ready\"} 1")
        );

        let other = Metrics::new();
        assert_eq!(
            line(&other.render(&Stats::default()), STATE, "state=\"ready\"").as_deref(),
            Some("smg_policy_controller_state{state=\"ready\"} 0"),
            "a second controller starts from its own recorder"
        );
    }

    #[test]
    fn the_load_counters_render_the_drivers_counts() {
        use std::sync::atomic::Ordering;

        use crate::ingest::loads::ReadError;

        let metrics = Metrics::new();
        let stats = Stats::default();
        let rendered = metrics.render(&stats);
        assert!(rendered.contains("smg_policy_controller_load_polls_total{mode=\"poll\"} 0"));
        assert!(rendered.contains("smg_policy_controller_load_rows_total 0"));

        stats.polls.fetch_add(3, Ordering::Relaxed);
        stats.suppressed.fetch_add(2, Ordering::Relaxed);
        stats.rows.fetch_add(7, Ordering::Relaxed);
        stats.scrapes.fetch_add(1, Ordering::Relaxed);
        stats.poll_failures.deadline.fetch_add(1, Ordering::Relaxed);
        let rendered = metrics.render(&stats);
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
        let rendered = metrics.render(&Stats::default());
        let start = rendered
            .lines()
            .find(|l| l.starts_with(START_TIME_SECONDS) && !l.starts_with('#'))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse::<f64>().ok())
            .expect("the start time line");
        assert!(start > 1.7e9, "seconds since the epoch, read {start}");
    }
}
