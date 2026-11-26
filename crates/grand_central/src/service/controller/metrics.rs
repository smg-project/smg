//! The controller's process series: its state, its start time, the load
//! monitor's counts, the decide call's and the store's two counts (design 14
//! names the rest, which arrive with the scopes, the replication and the
//! election), and the engine's family of the panics it held, as a
//! gateway names it. They live on a recorder of the controller's own, not the
//! process global, so a gateway that embeds the controller keeps its
//! recorder and two controllers in one test process keep theirs; the admin
//! listener renders it in the Prometheus text form. The monitor's counts
//! are read from the worker crate at each render, the decide call's from
//! their atomics and the store's from the store, so none carries a recorder
//! handle and a decision writes no series.

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
use smg_worker::load::counts::{FailureTotals, PollTotals};

use super::{
    decision_slots::DecideStats,
    generations::Generations,
    readiness::{Sample, State},
};
use crate::{
    engine::{POLICY_FAULTS_HELP, POLICY_FAULT_KINDS},
    store::StoreStats,
};

/// `1` for the controller's state (`starting`, `leader`, `syncing`,
/// `follower`, `draining`), `0` for the other four. The admin listener sets it from the
/// readiness at each render, so the role is the fence's at the scrape, and
/// one scrape's sample is one write: two scrapes never mix theirs.
pub const STATE: &str = "smg_grand_central_state";
/// The controller's start, in seconds since the Unix epoch.
pub const START_TIME_SECONDS: &str = "smg_grand_central_start_time_seconds";
/// `1` while this replica does not lead and its standby is synced with its
/// leader, `0` otherwise: a leader's is `0`.
pub const REPLICATION_SYNCED: &str = "smg_grand_central_replication_synced";
/// The age in seconds of the last frame that left this replica at its
/// leader's head. `0` while it leads and before any such frame, where
/// [`REPLICATION_SYNCED`] is `0` too.
pub const REPLICATION_LAG_SECONDS: &str = "smg_grand_central_replication_lag_seconds";
/// How many times this replica asked its leader to be restored from a
/// snapshot: an event of the leader's did not land on its tables or its
/// log. The leader cannot tell such an ask from a place it does not hold,
/// so the count is the standby's.
pub const REPLICATION_RESTORES_ASKED_TOTAL: &str =
    "smg_grand_central_replication_restores_asked_total";
/// Load polls by the arbitration's decision (`mode`: `poll`, `fallback`,
/// `suppressed`; a suppressed one made no call), each counted once per
/// worker and poll interval, for a worker on a connection. A worker is one
/// rank of its engine: the count is one an engine and interval but for an
/// engine of several ranks.
pub const LOAD_POLLS_TOTAL: &str = "smg_grand_central_load_polls_total";
/// Load polls over a connection that produced no report, by `reason`.
pub const LOAD_POLL_FAILURES_TOTAL: &str = "smg_grand_central_load_poll_failures_total";
/// Load reads over HTTP that produced the engine's report.
pub const LOAD_SCRAPES_TOTAL: &str = "smg_grand_central_load_scrapes_total";
/// Load reads over HTTP that produced no report, by `reason`.
pub const LOAD_SCRAPE_FAILURES_TOTAL: &str = "smg_grand_central_load_scrape_failures_total";
/// Rank rows of polled answers the load monitor published.
pub const LOAD_ROWS_TOTAL: &str = "smg_grand_central_load_rows_total";
/// Named values the bounds took out of the load monitor's polled answers.
pub const LOAD_VALUES_DROPPED_TOTAL: &str = "smg_grand_central_load_values_dropped_total";
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
/// `not_leader`): each is one the gateway decided locally. A refused
/// decode call counts under a reason of its own (`decode_no_lease`,
/// `decode_not_chosen`, `decode_prefill_not_held`, `decode_stands`): its
/// gateway sends the decode its first pick named.
pub const DECIDE_BUSY_TOTAL: &str = "smg_grand_central_decide_busy_total";
/// Decide requests this build had no decision for, by `cause` (`fault`,
/// `pair_side`, `prompt_text`, `decider_gone`, `lease_ids_spent`,
/// `document_awaited`): the `Busy` answers with no reason, each under what
/// took its decision. `document_awaited` is a new leader's moment: the
/// start's document of a replica under a shared lease had not landed yet.
pub const DECIDE_NO_DECISION_TOTAL: &str = "smg_grand_central_decide_no_decision_total";
/// Decide requests that carried more block hashes than their generation
/// reads of them: decided over the first ones, the rest hashed for nothing.
pub const DECIDE_HASHES_CUT_TOTAL: &str = "smg_grand_central_decide_hashes_cut_total";
/// Answers no decide stream took, by `cause`: `stream_gone` for a stream
/// that had ended, before the answer was made or with it still in the
/// stream's queue, `queue_full` for one whose queue had no place, which a
/// stream's window rules out.
pub const DECIDE_ANSWERS_DROPPED_TOTAL: &str = "smg_grand_central_decide_answers_dropped_total";
/// Decide requests their gateway said it asked again, holding no answer to
/// the first ask, by what became of the lease that ask was issued
/// (`earlier_lease`): `ended` where it still waited for a dispatch, `none`
/// where no lease of the request waited.
pub const DECIDE_ASKED_AGAIN_TOTAL: &str = "smg_grand_central_decide_asked_again_total";
/// Candidates a gateway's wires took from its decisions: members that
/// share no wire with the gateway that asked.
pub const DECIDE_CANDIDATES_UNDIALED_TOTAL: &str =
    "smg_grand_central_decide_candidates_undialed_total";
/// Decisions cut at their last read, with under a microsecond of the budget
/// or under a millisecond of the lease's dispatch span left: each is
/// answered `Busy` under `budget_gone_on_arrival`, and counted here apart
/// from a request that arrived late and from a pick whose span ended while
/// it waited to be sent.
pub const DECIDE_BUDGET_OUTLASTED_TOTAL: &str = "smg_grand_central_decide_budget_outlasted_total";
/// Members whose engine family or zone the standing generation's list does
/// not name, by `list` (`engines`, `zones`). Such a member is admitted and
/// reads position 0 in that field. A generation with no list counts none.
pub const MEMBERS_OUTSIDE_LIST: &str = "smg_grand_central_members_outside_list";
/// Overload verdicts of a worker that changed, either way: the load
/// monitor's own, a lead's end that takes each one off among them.
pub const OVERLOAD_FLIPS_TOTAL: &str = "smg_grand_central_overload_flips_total";
/// Lifecycle calls that panicked as a lease's event was applied (a
/// dispatch, a completion, a release a frame brings, a dispatch
/// deadline's end, a decode call's issue): each is held there, so it ends
/// no thread. A call that unwinds the decision it was made in (the issue
/// of a single request's lease or a pair's, the release of a lease the
/// decision gives up) counts with the requests that got no decision.
pub const LIFECYCLE_FAULTS_TOTAL: &str = "smg_grand_central_lifecycle_faults_total";
/// Panics the store's one engine held under its own catches, by
/// `kind`: `decision` (a whole decision, a shadow's
/// beside a live decision among them), `event` (a stage at a word the
/// engine tells it) and `upkeep` (a step of the engine's upkeep). It is a
/// gateway's family of that name, with one definition in both processes:
/// each page takes the kinds and the description from the engine's
/// constants ([`POLICY_FAULT_KINDS`], [`POLICY_FAULTS_HELP`]). The name
/// stays a literal here, as every series this module declares (the
/// reference's guard reads them), and a case holds it to the engine's. It
/// never starts again at a new document.
///
/// On a controller `upkeep` is a run of the upkeep at the presence sweep's
/// tick, and `event` is a stage's panic at a change of membership: the one
/// word a store says under the engine's catch. A stage's panic at a
/// lease's word is held by the controller: it counts with the requests
/// that got no decision where it unwinds the decision that told the word,
/// and in [`LIFECYCLE_FAULTS_TOTAL`] at every other word, a decode call's
/// issue among them. A live decision's panic also counts as a request
/// with no decision ([`DECIDE_NO_DECISION_TOTAL`], `fault`).
pub const POLICY_FAULTS_TOTAL: &str = "smg_policy_faults_total";

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
    lifecycle_faults: Counter,
    /// The panics the store's engine held: `decision`, `event`, `upkeep`.
    policy_faults: [Counter; 3],
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

/// What the load monitor's polls did and how often an overload verdict of a
/// worker changed: one read of the worker crate's counts, for a render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoadTotals {
    /// The monitor's polls, as its one read answers them.
    pub(crate) polls: PollTotals,
    /// The overload verdicts that changed, as the monitor's registry counts
    /// its edges.
    pub(crate) overload_flips: u64,
}

/// The `reason` labels of a failed poll or read, in the order of
/// [`by_reason`].
const REASONS: [&str; 6] = [
    "transport",
    "answer",
    "deadline",
    "no_load",
    "unsupported",
    "client",
];

/// The failures of `totals`, in the order of [`REASONS`].
fn by_reason(totals: &FailureTotals) -> [u64; 6] {
    [
        totals.transport,
        totals.answer,
        totals.deadline,
        totals.no_load,
        totals.unsupported,
        totals.client,
    ]
}

/// The load monitor's counts as the recorder's counters, set absolute from
/// one read of them at each render.
struct LoadCounters {
    /// `poll`, `fallback`, `suppressed`.
    polls: [Counter; 3],
    /// In the order of [`REASONS`].
    poll_failures: [Counter; 6],
    scrapes: Counter,
    /// In the order of [`REASONS`].
    scrape_failures: [Counter; 6],
    rows: Counter,
    values_dropped: Counter,
}

/// The decide call's counts as the recorder's counters, set absolute from
/// the deciders' and the streams' atomics at each render.
struct DecideCounters {
    /// In the order of [`DecideStats::BUSY`].
    busy: [Counter; 9],
    /// `fault`, `pair_side`, `prompt_text`, `decider_gone`,
    /// `lease_ids_spent`, `document_awaited`.
    no_decision: [Counter; 6],
    hashes_cut: Counter,
    /// `queue_full`, `stream_gone`.
    dropped: [Counter; 2],
    /// `none`, `ended`: in the order of [`DecideStats::again`].
    again: [Counter; 2],
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
            "Seconds since the last frame that left this replica at its leader's head, 0 while this replica leads and before any such frame".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(REPLICATION_RESTORES_ASKED_TOTAL),
            None,
            "Restores from a snapshot this replica asked of its leader after an event that did not land".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_POLLS_TOTAL),
            None,
            "Load polls by the arbitration's decision (poll, fallback, suppressed), one per worker on a connection and poll interval".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_POLL_FAILURES_TOTAL),
            None,
            "Load polls over a connection that produced no report, by reason".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_SCRAPES_TOTAL),
            None,
            "Load reads over HTTP that produced the engine's report".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_SCRAPE_FAILURES_TOTAL),
            None,
            "Load reads over HTTP that produced no report, by reason".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_ROWS_TOTAL),
            None,
            "Rank rows of polled answers the load monitor published".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(LOAD_VALUES_DROPPED_TOTAL),
            None,
            "Named values the bounds took out of the load monitor's polled answers".into(),
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
            KeyName::from_const_str(DECIDE_ASKED_AGAIN_TOTAL),
            None,
            "Decide requests asked again, by what became of the earlier lease".into(),
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
        recorder.describe_counter(
            KeyName::from_const_str(LIFECYCLE_FAULTS_TOTAL),
            None,
            "Lifecycle calls that panicked as a lease's event was applied, each held there".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(POLICY_FAULTS_TOTAL),
            None,
            POLICY_FAULTS_HELP.into(),
        );
        recorder.describe_gauge(
            KeyName::from_const_str(MEMBERS_OUTSIDE_LIST),
            None,
            "Members whose engine family or zone the policy document's list does not name, by list (engines, zones)".into(),
        );
        recorder.describe_counter(
            KeyName::from_const_str(OVERLOAD_FLIPS_TOTAL),
            None,
            "Overload verdicts of a worker that changed, either way".into(),
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
            poll_failures: REASONS
                .map(|reason| counter(LOAD_POLL_FAILURES_TOTAL, "reason", reason)),
            scrapes: recorder.register_counter(&Key::from_name(LOAD_SCRAPES_TOTAL), &metadata),
            scrape_failures: REASONS
                .map(|reason| counter(LOAD_SCRAPE_FAILURES_TOTAL, "reason", reason)),
            rows: recorder.register_counter(&Key::from_name(LOAD_ROWS_TOTAL), &metadata),
            values_dropped: recorder
                .register_counter(&Key::from_name(LOAD_VALUES_DROPPED_TOTAL), &metadata),
        };
        let decides = DecideCounters {
            busy: DecideStats::BUSY.map(|reason| counter(DECIDE_BUSY_TOTAL, "reason", reason)),
            no_decision: [
                "fault",
                "pair_side",
                "prompt_text",
                "decider_gone",
                "lease_ids_spent",
                "document_awaited",
            ]
            .map(|cause| counter(DECIDE_NO_DECISION_TOTAL, "cause", cause)),
            hashes_cut: recorder
                .register_counter(&Key::from_name(DECIDE_HASHES_CUT_TOTAL), &metadata),
            dropped: ["queue_full", "stream_gone"]
                .map(|cause| counter(DECIDE_ANSWERS_DROPPED_TOTAL, "cause", cause)),
            again: ["none", "ended"]
                .map(|lease| counter(DECIDE_ASKED_AGAIN_TOTAL, "earlier_lease", lease)),
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
        let lifecycle_faults =
            recorder.register_counter(&Key::from_name(LIFECYCLE_FAULTS_TOTAL), &metadata);
        let policy_faults =
            POLICY_FAULT_KINDS.map(|kind| counter(POLICY_FAULTS_TOTAL, "kind", kind));
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
            lifecycle_faults,
            policy_faults,
            scrape: Mutex::new(()),
            recorder,
            spec: OnceLock::new(),
        };
        metrics.set_state(State::Starting);
        metrics
    }

    /// Whether the spec series are bound, as a case reads it.
    #[cfg(test)]
    pub(crate) fn spec_bound(&self) -> bool {
        self.spec.get().is_some()
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
    /// by `sample` and their gauges written, the load monitor's counts, the
    /// decide call's and the store's read, and the render, with no other
    /// scrape between the sample and the render. Two scrapes across a change
    /// of role each render one state, their own.
    pub(crate) fn render<S: Into<Sample>>(
        &self,
        sample: impl FnOnce() -> S,
        loads: &LoadTotals,
        decides: &DecideStats,
        store: impl FnOnce() -> StoreStats,
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
            let counts = spec.generations.apply_counts();
            spec.generation.set(spec.generations.generation() as f64);
            spec.applied.absolute(counts.applied);
            spec.rejected.absolute(counts.rejected);
        }
        self.observe_decides(decides);
        self.observe_store(store());
        self.handle.render()
    }

    /// Sets the decide counters from the deciders' and the streams' atomics.
    fn observe_decides(&self, counts: &DecideStats) {
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
            &counts.awaited,
        ];
        for (counter, count) in self.decides.no_decision.iter().zip(causes) {
            counter.absolute(read(count));
        }
        self.decides.hashes_cut.absolute(counts.hashes_cut());
        let dropped = [&counts.dropped_full, &counts.dropped_gone];
        for (counter, count) in self.decides.dropped.iter().zip(dropped) {
            counter.absolute(read(count));
        }
        for (counter, count) in self.decides.again.iter().zip(&counts.again) {
            counter.absolute(read(count));
        }
        self.decides.undialed.absolute(counts.undialed());
        self.decides.outlasted.absolute(read(&counts.outlasted));
    }

    /// Sets the store's series from one read of its counts: the members
    /// outside a list, the lifecycle calls that panicked, and the panics
    /// its engine held, by kind.
    fn observe_store(&self, counts: StoreStats) {
        let [engines, zones] = &self.outside;
        engines.set(f64::from(counts.outside.engines));
        zones.set(f64::from(counts.outside.zones));
        self.lifecycle_faults.absolute(counts.lifecycle_faults);
        // The engine's counts stand for the store's life, whichever
        // document each rose under: each series is set to its count as read.
        let held = counts.engine_faults;
        let read = [held.decisions, held.events, held.upkeep];
        for (counter, count) in self.policy_faults.iter().zip(read) {
            counter.absolute(count);
        }
    }

    /// Sets the load counters and the overload flips from one read of the
    /// load monitor's counts.
    fn observe_loads(&self, totals: &LoadTotals) {
        let (loads, polled) = (&self.loads, &totals.polls);
        let polls = [polled.polls, polled.fallbacks, polled.suppressed];
        for (counter, count) in loads.polls.iter().zip(polls) {
            counter.absolute(count);
        }
        let failures = by_reason(&polled.poll_failures);
        for (counter, count) in loads.poll_failures.iter().zip(failures) {
            counter.absolute(count);
        }
        loads.scrapes.absolute(polled.scrapes);
        let failures = by_reason(&polled.scrape_failures);
        for (counter, count) in loads.scrape_failures.iter().zip(failures) {
            counter.absolute(count);
        }
        loads.rows.absolute(polled.rows);
        loads.values_dropped.absolute(polled.values_dropped);
        self.overload_flips.absolute(totals.overload_flips);
    }

    /// Moves the state gauge: `1` on `now`, `0` on the others.
    fn set_state(&self, now: State) {
        for (gauge, state) in self.state.iter().zip(State::ALL) {
            gauge.set(if state == now { 1.0 } else { 0.0 });
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/metrics/tests.rs"]
mod tests;
