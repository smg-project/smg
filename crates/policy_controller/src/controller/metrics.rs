//! The controller's process series: its state and its start time (design 14
//! names the rest, which arrive with the scopes, the replication and the
//! election). They live on a recorder of the controller's own, not the
//! process global, so a gateway that embeds the controller keeps its
//! recorder and two controllers in one test process keep theirs; the admin
//! listener renders it in the Prometheus text form.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use metrics::{Gauge, Key, KeyName, Label, Level, Metadata, Recorder};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use super::readiness::State;

/// `1` for the controller's state (`starting`, `ready`, `draining`), `0` for
/// the other two.
pub const STATE: &str = "smg_policy_controller_state";
/// The controller's start, in seconds since the Unix epoch.
pub const START_TIME_SECONDS: &str = "smg_policy_controller_start_time_seconds";

/// The controller's series and the handle that renders them.
pub(crate) struct Metrics {
    handle: PrometheusHandle,
    /// One gauge per [`State::ALL`] entry, in that order.
    state: [Gauge; 3],
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
        let state = State::ALL.map(|state| {
            let key = Key::from_parts(STATE, vec![Label::new("state", state.as_str())]);
            recorder.register_gauge(&key, &metadata)
        });
        let start = recorder.register_gauge(&Key::from_name(START_TIME_SECONDS), &metadata);
        start.set(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs_f64(),
        );
        let metrics = Self { handle, state };
        metrics.set_state(State::Starting);
        metrics
    }

    /// Moves the state gauge: `1` on `now`, `0` on the others.
    pub(crate) fn set_state(&self, now: State) {
        for (gauge, state) in self.state.iter().zip(State::ALL) {
            gauge.set(if state == now { 1.0 } else { 0.0 });
        }
    }

    /// The handle the admin listener renders.
    pub(crate) fn handle(&self) -> PrometheusHandle {
        self.handle.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{Metrics, START_TIME_SECONDS, STATE};
    use crate::controller::readiness::State;

    fn line(rendered: &str, series: &str, label: &str) -> Option<String> {
        rendered
            .lines()
            .find(|l| l.starts_with(series) && l.contains(label))
            .map(str::to_owned)
    }

    #[test]
    fn the_state_gauge_follows_the_transitions_on_the_controllers_own_recorder() {
        let metrics = Metrics::new();
        let rendered = metrics.handle().render();
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
        let rendered = metrics.handle().render();
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
            line(&other.handle().render(), STATE, "state=\"ready\"").as_deref(),
            Some("smg_policy_controller_state{state=\"ready\"} 0"),
            "a second controller starts from its own recorder"
        );
    }

    #[test]
    fn the_start_time_is_set_once_at_construction() {
        let metrics = Metrics::new();
        let rendered = metrics.handle().render();
        let start = rendered
            .lines()
            .find(|l| l.starts_with(START_TIME_SECONDS) && !l.starts_with('#'))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse::<f64>().ok())
            .expect("the start time line");
        assert!(start > 1.7e9, "seconds since the epoch, read {start}");
    }
}
