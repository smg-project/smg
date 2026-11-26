//! The `least_load` policy: how it is composed, and its scorer, the expected
//! wait.
//!
//! `power_of_two` and `cache_aware` name the scorer too, by its stage name
//! `least-load`.
//!
//! Most of the scorer's mechanism is the credit of the sends since a worker's
//! last report. That credit is shared code: `RankCredit` among the shared
//! stages, and the sent record in `tracking`.

use std::sync::Arc;

#[cfg(test)]
use crate::stages::scorers::HeldCredit;
use crate::{
    document::spec::{Needs, ProfileSpec},
    engine::{
        decision::Target,
        framework::{ScoreFront, Scorer, TieBand},
        view::{Candidate, LoadRow, RequestFacts, WorkerId},
    },
    policies::standard_filters,
    stages::{
        lifecycle::Lifecycle,
        scorers::{normalize_costs, ReadsLoad, PENDING, TIE_EPSILON_SECS},
    },
    tracking::{
        dispatch_credit::{DispatchCredit, ReportedSends, SentSettings},
        lease::{EventKind, EventSet, Lease, Membership, Side},
    },
};

/// How the `least_load` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    ProfileSpec {
        filter: standard_filters(),
        score: vec!["least-load".into()],
        pick: "best".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// The expected wait: token-work over drain rate plus a convex KV-pressure
/// barrier, the since-report dispatch credit water-filling arrivals between
/// reports. The host's in-flight count floors the report.
#[derive(Debug)]
pub struct LeastLoad {
    pub kv_pressure_weight: f32,
    pub mean_prefill_tokens: u32,
    pub default_throughput: f32,
    /// The count of sends the stage reads: its profile's where a document
    /// built the stage, one of its own elsewhere.
    sent: Arc<DispatchCredit>,
    /// Whether the count is the stage's own. The stage's hooks feed its
    /// own count. The engine feeds a profile's.
    feeds: bool,
}

impl LeastLoad {
    /// The stage over a count of its own, which its hooks feed.
    pub fn new(kv_pressure_weight: f32, mean_prefill_tokens: u32, default_throughput: f32) -> Self {
        Self {
            feeds: true,
            ..Self::of_profile(
                kv_pressure_weight,
                mean_prefill_tokens,
                default_throughput,
                Arc::new(DispatchCredit::new("", SentSettings::default())),
            )
        }
    }

    /// The stage over its profile's count of sends, as a document builds
    /// it: the profile's other readers read the same sends, and the
    /// profile's time limit and report delay hold here too.
    pub(crate) fn of_profile(
        kv_pressure_weight: f32,
        mean_prefill_tokens: u32,
        default_throughput: f32,
        sent: Arc<DispatchCredit>,
    ) -> Self {
        Self {
            kv_pressure_weight,
            mean_prefill_tokens: mean_prefill_tokens.max(1),
            default_throughput: if default_throughput > 0.0 {
                default_throughput
            } else {
                1.0
            },
            sent,
            feeds: false,
        }
    }

    /// Dispatches no report reflects yet on `worker`, over every rank of it,
    /// for tests and dumps.
    pub fn held(&self, worker: WorkerId) -> usize {
        self.sent.credits().held(worker)
    }

    /// Dispatches no report reflects yet on one rank of `worker`.
    pub fn held_at(&self, worker: WorkerId, dp_rank: u16) -> usize {
        self.sent.credits().held_at(worker, dp_rank)
    }

    /// One rank's held work as a decision reads it: token-work, a send of
    /// no stated size at the mean prefill, and requests.
    #[cfg(test)]
    fn held_word_at(&self, worker: WorkerId, dp_rank: u16) -> (u64, u64) {
        self.sent
            .credits()
            .tally
            .load()
            .get(&(worker, dp_rank))
            .map_or((0, 0), |since| {
                let (HeldCredit { tokens, requests }, unsized_held) = since.held_for_test();
                let unsized_held = u64::from(unsized_held);
                (
                    tokens + unsized_held * u64::from(self.mean_prefill_tokens),
                    requests,
                )
            })
    }

    /// A retirement for the rank row, as a dispatch that found the row at
    /// its bound runs it.
    #[cfg(test)]
    fn retire_oldest_of(&self, worker: WorkerId, dp_rank: u16) {
        if let Some(since) = self.sent.credits().tally.load().get(&(worker, dp_rank)) {
            since.retire_for_test();
        }
    }

    /// A reporting candidate's dispatch credit as token-work: the since-report
    /// tokens plus the host's in-flight requests no report or credit accounts
    /// for, at the mean prefill. Read under one tally load; no division here.
    fn credit_work(
        &self,
        report: &LoadRow,
        in_flight: u32,
        credit_tokens: u64,
        credit_requests: u64,
    ) -> f32 {
        let reported = u64::from(report.running) + u64::from(report.queued);
        let unaccounted = u64::from(in_flight).saturating_sub(reported + credit_requests);
        unaccounted as f32 * self.mean_prefill_tokens as f32 + credit_tokens as f32
    }

    /// The expected wait of a reporting candidate once its credit work is
    /// known: the queued and running work over the drain rate, plus the convex
    /// KV-pressure barrier over the running requests' share of the cache (a
    /// full but reclaimable prefix cache is no pressure). Runs after the
    /// tally is released.
    fn reported_cost(&self, r: &LoadRow, credit_work: f32, drain: f32) -> f32 {
        let mean = self.mean_prefill_tokens as f32;
        let queued = if r.waiting_uncached_tokens > 0 {
            r.waiting_uncached_tokens as f32
        } else {
            r.queued as f32 * mean
        };
        let running = r.running as f32 * mean;
        // The active share, at the legacy formula's clamp: a saturated row
        // prices its barrier at 999 times the weight, not a million.
        let k = r.active_kv().clamp(0.0, 0.999);
        (queued + running + credit_work) / drain + self.kv_pressure_weight * k / (1.0 - k)
    }

    /// Costs the rows the first pass left pending, those with no report:
    /// each from the reporting peers' baseline among the rows handed
    /// ([`Self::unreported_cost`]). Out of line: a pool whose every row
    /// reports pays for no second pass.
    #[cold]
    #[inline(never)]
    fn cost_pending(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        nominal: f32,
        fleet_has_loads: bool,
    ) {
        let mut peer_baseline = 0.0;
        if fleet_has_loads {
            peer_baseline = out
                .iter()
                .copied()
                .filter(|cost| *cost != PENDING)
                .fold(f32::INFINITY, f32::min);
            if !peer_baseline.is_finite() {
                peer_baseline = 0.0;
            }
        }
        for (slot, &i) in out.iter_mut().zip(kept) {
            if *slot == PENDING {
                *slot = self.unreported_cost(
                    facts.in_flight(&candidates[i as usize]),
                    nominal,
                    fleet_has_loads,
                    peer_baseline,
                );
            }
        }
    }

    /// The cost of a candidate without a report: the reporting peers' baseline
    /// plus its in-flight work, or the in-flight count alone when no peer
    /// reports.
    fn unreported_cost(
        &self,
        in_flight: u32,
        nominal: f32,
        fleet_has_loads: bool,
        peer_baseline: f32,
    ) -> f32 {
        if fleet_has_loads {
            peer_baseline + in_flight as f32 * self.mean_prefill_tokens as f32 / nominal
        } else {
            in_flight as f32
        }
    }

    /// The mean rate of the kept rows that report one, which a report
    /// without a rate drains at: the default where none reports one.
    ///
    /// A scan first asks whether any row states a rate, and only then
    /// does the sum run: a fleet whose reports state none asks this at
    /// every decision, and its walk keeps no sum. Out of line and not
    /// cold: the pass that never asks it holds none of it, and the pass
    /// that asks it at every decision gets it built for speed.
    #[inline(never)]
    fn mean_rate(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) -> f32 {
        let rate_of = |i: u32| load(&candidates[i as usize]).map_or(0, |r| r.live_rate);
        if !kept.iter().any(|&i| rate_of(i) > 0) {
            return self.default_throughput;
        }
        let (sum, n) = kept
            .iter()
            .map(|&i| rate_of(i))
            .filter(|&rate| rate > 0)
            .fold((0.0f32, 0u32), |(s, n), rate| (s + rate as f32, n + 1));
        if sum.is_finite() {
            sum / n as f32
        } else {
            self.default_throughput
        }
    }

    /// Writes each reporting row's credit work and marks each row without
    /// a report, and says how many report. Built once for each of the
    /// count's two passes ([`DispatchCredit::limited`]): the pass of a count
    /// with no time limit and no report delay reads each row by its sample
    /// alone.
    #[inline(always)]
    fn credit_pass(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        rows: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
        sends: &impl ReportedSends,
    ) -> usize {
        let mean = u64::from(self.mean_prefill_tokens);
        let mut reporting = 0usize;
        for (slot, &i) in out.iter_mut().zip(rows) {
            let c = &candidates[i as usize];
            *slot = match load(c) {
                Some(report) => {
                    let sent = sends.reported(c);
                    let tokens = sent.tokens + u64::from(sent.unsized_requests) * mean;
                    reporting += 1;
                    self.credit_work(report, facts.in_flight(c), tokens, u64::from(sent.requests))
                }
                None => PENDING,
            };
        }
        reporting
    }

    /// The costs of `rows`, spread over their span, and their band: the one
    /// body behind the stage's two entries. A row that needs a fallback
    /// drains at the mean of the kept rows' rates ([`Self::mean_rate`]).
    #[inline(always)]
    fn costs(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        rows: &[u32],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        // One load of the tally: each reporting row's held work is read
        // (released first when its report's sample or the time limit
        // reaches a dispatch) and its credit work written; the divisions
        // and the KV barrier run after the load is gone. A send of no
        // stated size is priced here, at the stage's mean. A row without a
        // report is marked and costed after the peer baseline is known. No
        // allocation.
        let reporting = if self.sent.limited() {
            let sends = self.sent.pass(facts);
            self.credit_pass(facts, candidates, rows, out, load, &sends)
        } else {
            let sends = self.sent.reads_unlimited();
            self.credit_pass(facts, candidates, rows, out, load, &sends)
        };
        let mut mean = None;
        let mut nominal = || *mean.get_or_insert_with(|| self.mean_rate(candidates, kept, load));
        for (slot, &i) in out.iter_mut().zip(rows) {
            if let Some(r) = load(&candidates[i as usize]) {
                let drain = if r.live_rate > 0 {
                    r.live_rate as f32
                } else {
                    nominal()
                };
                *slot = self.reported_cost(r, *slot, drain);
            }
        }
        if reporting < rows.len() {
            let fleet_has_loads = reporting > 0;
            let rate = if fleet_has_loads {
                nominal()
            } else {
                self.default_throughput
            };
            self.cost_pending(facts, candidates, rows, out, rate, fleet_has_loads);
        }
        // The band of a microsecond of expected wait on the normalized scale:
        // exact for one scorer, additive under several.
        let span = normalize_costs(out);
        if span > 0.0 {
            TieBand((TIE_EPSILON_SECS / span).min(1.0))
        } else {
            TieBand::NORMALIZED
        }
    }
}

impl LeastLoad {
    /// Counts one send in the stage's own count. Out of line: the hook of
    /// a stage over its profile's count returns at once.
    #[inline(never)]
    fn feed(&self, lease: &Lease, target: &Target, side: Side, since_decision_micros: u32) {
        self.sent
            .dispatched(lease, target, side, since_decision_micros, || true);
    }
}

impl Lifecycle for LeastLoad {
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        if self.feeds {
            self.feed(lease, target, side, since_decision_micros);
        }
    }

    fn on_membership(&self, change: &Membership<'_>) {
        if self.feeds {
            self.sent.membership(change);
        }
    }

    /// The dispatch, where the stage feeds its own count of sends.
    fn hears(&self) -> EventSet {
        if self.feeds {
            EventSet::NONE.with(EventKind::Dispatched)
        } else {
            EventSet::NONE
        }
    }
}

impl Scorer for LeastLoad {
    fn name(&self) -> &'static str {
        "least-load"
    }

    // Costs by the count in flight and its credits: late ones move the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    /// The rows a rank put in front are costed alone where each holds a
    /// report, and the stage says where one does not
    /// ([`Self::score_front`]).
    fn front_wise(&self) -> bool {
        true
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.score_with(facts, candidates, kept, out, |c| c.load.as_ref())
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        self.score_front_with(facts, candidates, kept, asked, out, |c| c.load.as_ref())
    }

    fn needs(&self) -> Needs {
        Needs {
            loads: true,
            ..Needs::default()
        }
    }
}

impl ReadsLoad for LeastLoad {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        self.costs(facts, candidates, kept, kept, out, load)
    }

    /// As [`Self::score`], with one word more, of the rows asked. A row
    /// without a report is costed from its reporting peers' baseline, the
    /// least cost among them, which is a pass over every kept row: where
    /// one is asked the stage says that it must be asked whole
    /// ([`ScoreFront::Whole`]), before it costs any. A report without a rate
    /// drains at the kept rows' mean rate: where one is asked that mean is
    /// folded over the kept rows as the plain entry folds it, and where
    /// each row asked reports a rate no other row is read. Either way a
    /// row asked costs what it costs among every kept row. Their costs
    /// are spread over their own span and not the pool's, so the front can
    /// tell two of its rows apart where the whole pass cannot, whatever
    /// the rows behind them cost: a difference past the tie's microsecond
    /// stands wherever it is a large enough share of the front's own span
    /// for a profile's sum to hold. A long queue on a row in front widens that
    /// span, as one behind the front widens the pool's. Among every row, a
    /// difference that is a small enough share of the pool's span falls
    /// under the last bit of a profile's sum, and the rows are drawn among.
    fn score_front_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> (TieBand, ScoreFront) {
        for &i in asked {
            if load(&candidates[i as usize]).is_none() {
                return (TieBand::NORMALIZED, ScoreFront::Whole);
            }
        }
        (
            self.costs(facts, candidates, asked, kept, out, load),
            ScoreFront::Unsaid,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/least_load/tests.rs"]
mod tests;
