//! The controller's process series: its state, its start time, the load
//! driver's counts, the decide call's and the store's two counts (design 14
//! names the rest, which arrive with the scopes, the replication and the
//! election). They live on a recorder of the controller's own, not the
//! process global, so a gateway that embeds the controller keeps its
//! recorder and two controllers in one test process keep theirs; the admin
//! listener renders it in the Prometheus text form. The driver's counts and
//! the decide call's are read from their atomics at each render and the
//! store's from the store, so none carries a recorder handle and a decision
//! writes no series.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use metrics::{Counter, Gauge, Key, KeyName, Label, Level, Metadata, Recorder};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle, PrometheusRecorder};
use parking_lot::Mutex;

use super::{
    deciders::Counts,
    generations::Generations,
    readiness::{Sample, State},
};
use crate::{
    ingest::loads::{Failures, Stats},
    store::StoreCounts,
};

/// `1` for the controller's state (`starting`, `leader`, `syncing`,
/// `follower`, `draining`), `0` for the other four. The admin listener sets it from the
/// readiness at each render, so the role is the fence's at the scrape, and
/// one scrape's sample is one write: two scrapes never mix theirs.
pub const STATE: &str = "smg_policy_controller_state";
/// The controller's start, in seconds since the Unix epoch.
pub const START_TIME_SECONDS: &str = "smg_policy_controller_start_time_seconds";
/// `1` while this replica does not lead and its standby is synced with its
/// leader, `0` otherwise: a leader's is `0`.
pub const REPLICATION_SYNCED: &str = "smg_policy_controller_replication_synced";
/// The age in seconds of the last frame this replica applied from its
/// leader. `0` while it leads and before any frame, where
/// [`REPLICATION_SYNCED`] is `0` too.
pub const REPLICATION_LAG_SECONDS: &str = "smg_policy_controller_replication_lag_seconds";
/// How many times this replica asked its leader to be restored from a
/// snapshot: an event of the leader's did not land on its tables or its
/// log. The leader cannot tell such an ask from a place it does not hold,
/// so the count is the standby's.
pub const REPLICATION_RESTORES_ASKED_TOTAL: &str =
    "smg_policy_controller_replication_restores_asked_total";
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
/// Policy spec documents a running source handed over, by `result`:
/// `applied` as a new generation, or `rejected` with the running generation
/// kept. A document that is the standing one counts as neither. Rendered
/// with [`SPEC_GENERATION`].
pub const SPEC_APPLY_TOTAL: &str = "smg_policy_spec_apply_total";
/// Decide requests answered `Busy`, by the `reason` the answer carries
/// (`unspecified`, `queue_depth`, `no_credit`, `budget_gone_on_arrival`,
/// `not_leader`): each is one the gateway decided locally.
pub const DECIDE_BUSY_TOTAL: &str = "smg_policy_controller_decide_busy_total";
/// Decide requests this build had no decision for, by `cause` (`fault`,
/// `pair_side`, `prompt_text`, `decider_gone`, `lease_ids_spent`): the
/// `Busy` answers with no reason, each under what took its decision. A
/// lease's lifecycle call that panics as its decider leaves its work is a
/// `fault` too, and answered no request.
pub const DECIDE_NO_DECISION_TOTAL: &str = "smg_policy_controller_decide_no_decision_total";
/// Decide requests that carried more block hashes than their generation
/// reads of them: decided over the first ones, the rest hashed for nothing.
pub const DECIDE_HASHES_CUT_TOTAL: &str = "smg_policy_controller_decide_hashes_cut_total";
/// Answers no decide stream took, by `cause`: `stream_gone` for a stream
/// that had ended, `queue_full` for one whose queue had no place, which a
/// stream's window rules out.
pub const DECIDE_ANSWERS_DROPPED_TOTAL: &str = "smg_policy_controller_decide_answers_dropped_total";
/// Candidates a gateway's wires took from its decisions: members that
/// share no wire with the gateway that asked.
pub const DECIDE_CANDIDATES_UNDIALED_TOTAL: &str =
    "smg_policy_controller_decide_candidates_undialed_total";
/// Decisions cut at their last read, with under a microsecond of the budget
/// or under a millisecond of the lease's dispatch span left: each is
/// answered `Busy` under `budget_gone_on_arrival`, and counted here apart
/// from a request that arrived late and from a pick whose span ended while
/// it waited to be sent.
pub const DECIDE_BUDGET_OUTLASTED_TOTAL: &str =
    "smg_policy_controller_decide_budget_outlasted_total";
/// Members whose engine family or zone the standing generation's list does
/// not name, by `list` (`engines`, `zones`). Such a member is admitted and
/// reads position 0 in that field. A generation with no list counts none.
pub const MEMBERS_OUTSIDE_LIST: &str = "smg_policy_controller_members_outside_list";
/// Writes to the load table that moved an overload verdict.
pub const OVERLOAD_FLIPS_TOTAL: &str = "smg_policy_controller_overload_flips_total";

/// The controller's series and the handle that renders them.
pub(crate) struct Metrics {
    handle: PrometheusHandle,
    /// One gauge per [`State::ALL`] entry, in that order.
    state: [Gauge; 5],
    /// The standby's sync and its lag, from the same sample as the state.
    synced: Gauge,
    lag: Gauge,
    /// The restores this replica asked of its leader, from that sample too.
    restores: Counter,
    loads: LoadCounters,
    decides: DecideCounters,
    /// The members outside a list: `engines`, then `zones`.
    outside: [Gauge; 2],
    overload_flips: Counter,
    /// Held from a scrape's sample of the state to the end of its render:
    /// the gauges are written from one sample and read back whole.
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
    applied: Counter,
    rejected: Counter,
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

/// The decide call's counts as the recorder's counters, set absolute from
/// the deciders' and the streams' atomics at each render.
struct DecideCounters {
    /// In the order of [`Counts::BUSY`].
    busy: [Counter; 5],
    /// `fault`, `pair_side`, `prompt_text`, `decider_gone`, `lease_ids_spent`.
    no_decision: [Counter; 5],
    hashes_cut: Counter,
    /// `queue_full`, `stream_gone`.
    dropped: [Counter; 2],
    undialed: Counter,
    outlasted: Counter,
}

impl Metrics {
    /// A recorder with the series registered and the start time set.
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
            "1 for the controller's state (starting, leader, syncing, follower, draining), 0 for the others"
                .into(),
        );
        recorder.describe_gauge(
            KeyName::from_const_str(START_TIME_SECONDS),
            None,
            "The controller's start, in seconds since the Unix epoch".into(),
        );
        recorder.describe_gauge(
            KeyName::from_const_str(REPLICATION_SYNCED),
            None,
            "1 while this replica does not lead and its standby is synced with its leader, 0 otherwise".into(),
        );
        recorder.describe_gauge(
            KeyName::from_const_str(REPLICATION_LAG_SECONDS),
            None,
            "Seconds since the last frame applied from the leader, 0 while this replica leads and before any frame".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(REPLICATION_RESTORES_ASKED_TOTAL),
            None,
            "Restores from a snapshot this replica asked of its leader after an event that did not land".into(),
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
        recorder.describe_counter(
            KeyName::from_const_str(DECIDE_BUSY_TOTAL),
            None,
            "Decide requests answered Busy, by the reason the answer carries".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(DECIDE_NO_DECISION_TOTAL),
            None,
            "Decide requests this build had no decision for, by cause".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(DECIDE_HASHES_CUT_TOTAL),
            None,
            "Decide requests that carried more block hashes than their generation reads".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(DECIDE_ANSWERS_DROPPED_TOTAL),
            None,
            "Answers no decide stream took, by cause".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(DECIDE_CANDIDATES_UNDIALED_TOTAL),
            None,
            "Candidates a gateway's wires took from its decisions".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(DECIDE_BUDGET_OUTLASTED_TOTAL),
            None,
            "Decisions cut at their last read with under a microsecond of budget or under a millisecond of the lease's dispatch span left, of the Busy answers under budget_gone_on_arrival"
                .into(),
        );
        recorder.describe_gauge(
            KeyName::from_const_str(MEMBERS_OUTSIDE_LIST),
            None,
            "Members whose engine family or zone the policy document's list does not name, by list (engines, zones)".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(OVERLOAD_FLIPS_TOTAL),
            None,
            "Writes to the load table that moved an overload verdict".into(),
        );
        let state = State::ALL.map(|state| {
            let key = Key::from_parts(STATE, vec![Label::new("state", state.as_str())]);
            recorder.register_gauge(&key, &metadata)
        });
        let start = recorder.register_gauge(&Key::from_name(START_TIME_SECONDS), &metadata);
        let synced = recorder.register_gauge(&Key::from_name(REPLICATION_SYNCED), &metadata);
        let lag = recorder.register_gauge(&Key::from_name(REPLICATION_LAG_SECONDS), &metadata);
        let restores =
            recorder.register_counter(&Key::from_name(REPLICATION_RESTORES_ASKED_TOTAL), &metadata);
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
        let decides = DecideCounters {
            busy: Counts::BUSY.map(|reason| counter(DECIDE_BUSY_TOTAL, "reason", reason)),
            no_decision: [
                "fault",
                "pair_side",
                "prompt_text",
                "decider_gone",
                "lease_ids_spent",
            ]
            .map(|cause| counter(DECIDE_NO_DECISION_TOTAL, "cause", cause)),
            hashes_cut: recorder
                .register_counter(&Key::from_name(DECIDE_HASHES_CUT_TOTAL), &metadata),
            dropped: ["queue_full", "stream_gone"]
                .map(|cause| counter(DECIDE_ANSWERS_DROPPED_TOTAL, "cause", cause)),
            undialed: recorder
                .register_counter(&Key::from_name(DECIDE_CANDIDATES_UNDIALED_TOTAL), &metadata),
            outlasted: recorder
                .register_counter(&Key::from_name(DECIDE_BUDGET_OUTLASTED_TOTAL), &metadata),
        };
        let outside = ["engines", "zones"].map(|list| {
            let key = Key::from_parts(MEMBERS_OUTSIDE_LIST, vec![Label::new("list", list)]);
            recorder.register_gauge(&key, &metadata)
        });
        let overload_flips =
            recorder.register_counter(&Key::from_name(OVERLOAD_FLIPS_TOTAL), &metadata);
        start.set(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs_f64(),
        );
        let metrics = Self {
            handle,
            state,
            synced,
            lag,
            restores,
            loads,
            decides,
            outside,
            overload_flips,
            scrape: Mutex::new(()),
            recorder,
            spec: OnceLock::new(),
        };
        metrics.set_state(State::Starting);
        metrics
    }

    /// Renders the generation `generations` runs and its apply counter by
    /// result from here on. Bound once, when a spec source has applied its
    /// first document; a controller with no source renders no spec series.
    pub(crate) fn spec(&self, generations: Arc<Generations>) {
        self.spec.get_or_init(|| {
            let metadata = Metadata::new(module_path!(), Level::INFO, Some(module_path!()));
            self.recorder.describe_gauge(
                KeyName::from_const_str(SPEC_GENERATION),
                None,
                "The generation of the policy spec the controller runs".into(),
            );
            self.recorder.describe_counter(
                KeyName::from_const_str(SPEC_APPLY_TOTAL),
                None,
                "Policy spec documents a source handed over, by result (applied, rejected)".into(),
            );
            let result = |result: &'static str| {
                self.recorder.register_counter(
                    &Key::from_parts(SPEC_APPLY_TOTAL, vec![Label::new("result", result)]),
                    &metadata,
                )
            };
            SpecSeries {
                generation: self
                    .recorder
                    .register_gauge(&Key::from_name(SPEC_GENERATION), &metadata),
                applied: result("applied"),
                rejected: result("rejected"),
                generations,
            }
        });
    }

    /// One scrape's text form: the state and the standby's lag sampled once
    /// by `sample` and their gauges written, the load driver's counts, the
    /// decide call's and the store's read, and the render, with no other
    /// scrape between the sample and the render. Two scrapes across a change
    /// of role each render one state, their own.
    pub(crate) fn render<S: Into<Sample>>(
        &self,
        sample: impl FnOnce() -> S,
        loads: &Stats,
        decides: &Counts,
        store: impl FnOnce() -> StoreCounts,
    ) -> String {
        let _one = self.scrape.lock();
        let sample = sample().into();
        self.set_state(sample.state);
        self.synced.set(if sample.synced { 1.0 } else { 0.0 });
        self.lag
            .set(sample.lag.map_or(0.0, |lag| lag.as_secs_f64()));
        self.restores.absolute(sample.restores);
        self.observe_loads(loads);
        if let Some(spec) = self.spec.get() {
            let counts = spec.generations.counts();
            spec.generation.set(spec.generations.generation() as f64);
            spec.applied.absolute(counts.applied);
            spec.rejected.absolute(counts.rejected);
        }
        self.observe_decides(decides);
        self.observe_store(store());
        self.handle.render()
    }

    /// Sets the decide counters from the deciders' and the streams' atomics.
    fn observe_decides(&self, counts: &Counts) {
        let read = |count: &AtomicU64| count.load(Ordering::Relaxed);
        for (counter, count) in self.decides.busy.iter().zip(&counts.busy) {
            counter.absolute(read(count));
        }
        let causes = [
            &counts.faults,
            &counts.pair_sides,
            &counts.texts,
            &counts.gone,
            &counts.unleased,
        ];
        for (counter, count) in self.decides.no_decision.iter().zip(causes) {
            counter.absolute(read(count));
        }
        self.decides.hashes_cut.absolute(counts.hashes_cut());
        let dropped = [&counts.dropped_full, &counts.dropped_gone];
        for (counter, count) in self.decides.dropped.iter().zip(dropped) {
            counter.absolute(read(count));
        }
        self.decides.undialed.absolute(counts.undialed());
        self.decides.outlasted.absolute(read(&counts.outlasted));
    }

    /// Sets the store's series from one read of its counts.
    fn observe_store(&self, counts: StoreCounts) {
        let [engines, zones] = &self.outside;
        engines.set(f64::from(counts.outside.engines));
        zones.set(f64::from(counts.outside.zones));
        self.overload_flips.absolute(counts.overload_flips);
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

    use super::{
        Generations, Metrics, StoreCounts, MEMBERS_OUTSIDE_LIST, REPLICATION_LAG_SECONDS,
        REPLICATION_RESTORES_ASKED_TOTAL, REPLICATION_SYNCED, SPEC_APPLY_TOTAL, SPEC_GENERATION,
        START_TIME_SECONDS, STATE,
    };
    use crate::{
        controller::{
            deciders::Counts,
            readiness::{Sample, State},
        },
        ingest::loads::Stats,
        store::OutsideLists,
    };

    fn line(rendered: &str, series: &str, label: &str) -> Option<String> {
        rendered
            .lines()
            .find(|l| l.starts_with(series) && l.contains(label))
            .map(str::to_owned)
    }

    #[test]
    fn the_state_gauge_follows_the_transitions_on_the_controllers_own_recorder() {
        let metrics = Metrics::new();
        let rendered = metrics.render(
            || State::Starting,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
        assert_eq!(
            line(&rendered, STATE, "state=\"starting\"").as_deref(),
            Some("smg_policy_controller_state{state=\"starting\"} 1")
        );
        assert_eq!(
            line(&rendered, STATE, "state=\"leader\"").as_deref(),
            Some("smg_policy_controller_state{state=\"leader\"} 0")
        );
        assert!(rendered.contains("# HELP smg_policy_controller_state "));

        let rendered = metrics.render(
            || State::Leader,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
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
                &other.render(
                    || State::Starting,
                    &Stats::default(),
                    &Counts::default(),
                    StoreCounts::default
                ),
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
                    &Counts::default(),
                    StoreCounts::default,
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
            &Counts::default(),
            StoreCounts::default,
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
        let rendered = metrics.render(
            || State::Starting,
            &stats,
            &Counts::default(),
            StoreCounts::default,
        );
        assert!(rendered.contains("smg_policy_controller_load_polls_total{mode=\"poll\"} 0"));
        assert!(rendered.contains("smg_policy_controller_load_rows_total 0"));

        stats.polls.fetch_add(3, Ordering::Relaxed);
        stats.suppressed.fetch_add(2, Ordering::Relaxed);
        stats.rows.fetch_add(7, Ordering::Relaxed);
        stats.scrapes.fetch_add(1, Ordering::Relaxed);
        stats.poll_failures.deadline.fetch_add(1, Ordering::Relaxed);
        let rendered = metrics.render(
            || State::Starting,
            &stats,
            &Counts::default(),
            StoreCounts::default,
        );
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
    fn the_decide_counters_render_the_deciders_and_the_streams_counts() {
        use std::sync::atomic::Ordering;

        let metrics = Metrics::new();
        let counts = Counts::default();
        let rendered = metrics.render(
            || State::Starting,
            &Stats::default(),
            &counts,
            StoreCounts::default,
        );
        // Every series stands at zero from the first scrape, by its labels.
        for (series, labels) in [
            (
                "decide_busy_total",
                &Counts::BUSY.map(|reason| format!("{{reason=\"{reason}\"}}"))[..],
            ),
            (
                "decide_no_decision_total",
                &[
                    "fault",
                    "pair_side",
                    "prompt_text",
                    "decider_gone",
                    "lease_ids_spent",
                ]
                .map(|cause| format!("{{cause=\"{cause}\"}}"))[..],
            ),
            ("decide_hashes_cut_total", &[String::new()][..]),
            (
                "decide_answers_dropped_total",
                &["queue_full", "stream_gone"].map(|cause| format!("{{cause=\"{cause}\"}}"))[..],
            ),
            ("decide_candidates_undialed_total", &[String::new()][..]),
            ("decide_budget_outlasted_total", &[String::new()][..]),
        ] {
            for labels in labels {
                let line = format!("smg_policy_controller_{series}{labels} 0");
                assert!(rendered.contains(&line), "{line}\n{rendered}");
            }
        }

        // Each count under its own series and label, none under another's.
        let all = counts.busy.iter().chain([
            &counts.faults,
            &counts.pair_sides,
            &counts.texts,
            &counts.gone,
            &counts.dropped_full,
            &counts.dropped_gone,
            &counts.outlasted,
            &counts.unleased,
        ]);
        for (count, value) in all.zip(1..) {
            count.store(value, Ordering::Relaxed);
        }
        // The two counts kept in parts are read as the sums of their
        // deciders' parts.
        let (first, fourth) = (counts.part(0), counts.part(3));
        first.undialed.store(14, Ordering::Relaxed);
        fourth.undialed.store(17, Ordering::Relaxed);
        first.hashes_cut.store(20, Ordering::Relaxed);
        fourth.hashes_cut.store(22, Ordering::Relaxed);
        let rendered = metrics.render(
            || State::Starting,
            &Stats::default(),
            &counts,
            StoreCounts::default,
        );
        for line in [
            "decide_busy_total{reason=\"unspecified\"} 1",
            "decide_busy_total{reason=\"queue_depth\"} 2",
            "decide_busy_total{reason=\"no_credit\"} 3",
            "decide_busy_total{reason=\"budget_gone_on_arrival\"} 4",
            "decide_busy_total{reason=\"not_leader\"} 5",
            "decide_no_decision_total{cause=\"fault\"} 6",
            "decide_no_decision_total{cause=\"pair_side\"} 7",
            "decide_no_decision_total{cause=\"prompt_text\"} 8",
            "decide_no_decision_total{cause=\"decider_gone\"} 9",
            "decide_answers_dropped_total{cause=\"queue_full\"} 10",
            "decide_answers_dropped_total{cause=\"stream_gone\"} 11",
            "decide_budget_outlasted_total 12",
            "decide_no_decision_total{cause=\"lease_ids_spent\"} 13",
            "decide_candidates_undialed_total 31",
            "decide_hashes_cut_total 42",
        ] {
            let line = format!("smg_policy_controller_{line}");
            assert!(rendered.contains(&line), "{line}\n{rendered}");
        }
    }

    #[test]
    fn the_stores_counts_render_by_list_from_each_scrapes_own_read() {
        let metrics = Metrics::new();
        let rendered = metrics.render(
            || State::Leader,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
        assert_eq!(
            line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"engines\"").as_deref(),
            Some("smg_policy_controller_members_outside_list{list=\"engines\"} 0")
        );
        assert!(rendered.contains("smg_policy_controller_overload_flips_total 0"));
        assert!(rendered.contains("# HELP smg_policy_controller_members_outside_list "));

        let counts = StoreCounts {
            outside: OutsideLists {
                engines: 2,
                zones: 5,
            },
            overload_flips: 9,
        };
        let rendered = metrics.render(
            || State::Leader,
            &Stats::default(),
            &Counts::default(),
            || counts,
        );
        assert_eq!(
            line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"engines\"").as_deref(),
            Some("smg_policy_controller_members_outside_list{list=\"engines\"} 2")
        );
        assert_eq!(
            line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"zones\"").as_deref(),
            Some("smg_policy_controller_members_outside_list{list=\"zones\"} 5")
        );
        assert!(rendered.contains("smg_policy_controller_overload_flips_total 9"));

        // The count outside a list falls once a generation names the member.
        let rendered = metrics.render(
            || State::Leader,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
        assert_eq!(
            line(&rendered, MEMBERS_OUTSIDE_LIST, "list=\"zones\"").as_deref(),
            Some("smg_policy_controller_members_outside_list{list=\"zones\"} 0")
        );
    }

    #[test]
    fn the_start_time_is_set_once_at_construction() {
        let metrics = Metrics::new();
        let rendered = metrics.render(
            || State::Starting,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
        let start = rendered
            .lines()
            .find(|l| l.starts_with(START_TIME_SECONDS) && !l.starts_with('#'))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse::<f64>().ok())
            .expect("the start time line");
        assert!(start > 1.7e9, "seconds since the epoch, read {start}");
    }

    /// A controller with no spec source renders no spec series. Once one is
    /// bound, every render reads the generation and the apply counter by
    /// result as they stand at that render.
    #[test]
    fn the_spec_series_render_once_a_source_is_bound() {
        use crate::{
            controller::generations::{
                tests::{with_line, LEAST_LOAD, RANDOM},
                Format,
            },
            store::MemoryStore,
        };

        let metrics = Metrics::new();
        let rendered = metrics.render(
            || State::Leader,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
        assert!(!rendered.contains("smg_policy_spec"), "{rendered}");

        let generations = Arc::new(Generations::new(
            Arc::new(MemoryStore::new()),
            Arc::new(crate::Catalog::builtin()),
        ));
        metrics.spec(Arc::clone(&generations));
        let series = |generation: u64, applied: u64, rejected: u64| {
            let rendered = metrics.render(
                || State::Leader,
                &Stats::default(),
                &Counts::default(),
                StoreCounts::default,
            );
            for line in [
                format!("{SPEC_GENERATION} {generation}"),
                format!("{SPEC_APPLY_TOTAL}{{result=\"applied\"}} {applied}"),
                format!("{SPEC_APPLY_TOTAL}{{result=\"rejected\"}} {rejected}"),
            ] {
                assert!(
                    rendered.lines().any(|rendered| rendered == line),
                    "no line {line:?} in {rendered}"
                );
            }
        };
        series(0, 0, 0);
        generations
            .apply(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
        series(1, 1, 0);
        let refused = generations
            .apply(with_line(RANDOM, 1), Format::Yaml)
            .expect_err("a changed document under the standing line");
        generations.reject(refused);
        series(1, 1, 1);
        generations
            .apply(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        series(2, 2, 1);
    }

    #[test]
    fn a_scrape_tells_the_standbys_sync_and_its_lag_from_one_sample() {
        let metrics = Metrics::new();
        let scrape = |state, lag, synced| {
            let rendered = metrics.render(
                || Sample {
                    state,
                    lag,
                    synced,
                    restores: 0,
                },
                &Stats::default(),
                &Counts::default(),
                StoreCounts::default,
            );
            (
                line(&rendered, REPLICATION_SYNCED, ""),
                line(&rendered, REPLICATION_LAG_SECONDS, ""),
            )
        };
        let read = |synced: &str, lag: &str| {
            (
                Some(format!("smg_policy_controller_replication_synced {synced}")),
                Some(format!(
                    "smg_policy_controller_replication_lag_seconds {lag}"
                )),
            )
        };
        // A follower a second and a half behind its leader, and synced.
        assert_eq!(
            scrape(State::Follower, Some(Duration::from_millis(1500)), true),
            read("1", "1.5")
        );
        // Its stream fell silent: the lag grows and the sync is gone.
        assert_eq!(
            scrape(State::Syncing, Some(Duration::from_secs(40)), false),
            read("0", "40")
        );
        // A leader follows nobody: both read zero, as before any frame.
        let rendered = metrics.render(
            || State::Leader,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
        assert_eq!(
            (
                line(&rendered, REPLICATION_SYNCED, ""),
                line(&rendered, REPLICATION_LAG_SECONDS, "")
            ),
            read("0", "0")
        );
        assert!(rendered.contains("# HELP smg_policy_controller_replication_synced "));
        assert!(rendered.contains("# HELP smg_policy_controller_replication_lag_seconds "));
    }

    #[test]
    fn a_scrape_tells_the_restores_this_replica_asked_of_its_leader() {
        let metrics = Metrics::new();
        let scrape = |restores| {
            let rendered = metrics.render(
                || Sample {
                    restores,
                    ..Sample::from(State::Syncing)
                },
                &Stats::default(),
                &Counts::default(),
                StoreCounts::default,
            );
            line(&rendered, REPLICATION_RESTORES_ASKED_TOTAL, "")
        };
        let read = |asked: u64| {
            Some(format!(
                "smg_policy_controller_replication_restores_asked_total {asked}"
            ))
        };
        // The series is there before any ask.
        assert_eq!(scrape(0), read(0));
        // A scrape tells the standby's own count: it is set, never added.
        assert_eq!(scrape(2), read(2));
        assert_eq!(scrape(2), read(2));
        assert_eq!(scrape(3), read(3));
        let rendered = metrics.render(
            || State::Leader,
            &Stats::default(),
            &Counts::default(),
            StoreCounts::default,
        );
        assert!(rendered
            .contains("# TYPE smg_policy_controller_replication_restores_asked_total counter"));
    }
}
