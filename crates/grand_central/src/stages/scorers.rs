//! The scorers written for no one policy. Every stateful one keeps its state
//! behind the lifecycle alone, and none allocates per decision.
//!
//! The credit of the sends since a worker's last report (`RankCredit`,
//! `Credits`) is in `tracking`. `ResidentWork` scores with it, and so does
//! the scorer of `least_load`, through the sent record there.

use std::{cell::Cell, sync::Arc};

use crate::{
    document::spec::{CostMargin, Needs, Unreported},
    engine::{
        decision::{Decision, Overlap, Target},
        framework::{DecisionScope, ScoreFront, ScoreSpread, Scorer, TieBand},
        view::{Candidate, LoadReportCutoff, LoadRow, PoolFacts, RequestFacts, WorkerId},
        warm_up::WarmUpRule,
    },
    load::signals::{SignalSums, SignalTerm},
    stages::{
        cost_terms::{Coefficients, CostTerm},
        lifecycle::Lifecycle,
    },
    tracking::{
        dispatch_credit::DispatchCredit,
        lease::{EventKind, EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal},
        rank_credit::{Credits, DispatchCounts, HeldCredit, RankCredit, RankTable, SendSize},
    },
};

/// Expected waits within this many seconds of the minimum are one tie: the
/// least-load scorer's own band, mapped onto the normalized scale per
/// decision. A wait is a 32-bit float, so past about 16 s it holds no
/// microsecond of its own: two waits that round together are a tie whatever
/// the band, and two that round apart are not.
pub(crate) const TIE_EPSILON_SECS: f32 = 1e-6;

/// A scorer that can read reports through a decision's cutoff. The bare
/// scorer reads the row directly, and the document wraps it only when it
/// sets a limit.
pub(crate) trait ReadsLoad: Scorer {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand;

    fn score_front_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> (TieBand, ScoreFront) {
        (
            self.score_with(facts, candidates, asked, out, load),
            ScoreFront::Unsaid,
        )
    }
}

#[derive(Debug)]
struct WithLoadReportAgeLimit<S> {
    stage: S,
    max_age_ms: u64,
}

pub(crate) fn with_load_report_age_limit(
    stage: impl ReadsLoad + 'static,
    max_age_ms: Option<u64>,
) -> Box<dyn Scorer> {
    match max_age_ms {
        Some(max_age_ms) => Box::new(WithLoadReportAgeLimit { stage, max_age_ms }),
        None => Box::new(stage),
    }
}

impl<S: ReadsLoad> Lifecycle for WithLoadReportAgeLimit<S> {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.stage.on_decision_issued(lease, decision);
    }

    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.stage
            .on_dispatched(lease, target, side, since_decision_micros);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.stage.on_completed(lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.stage.on_released(lease, reason);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        self.stage.on_membership(change);
    }

    fn on_signal(&self, signal: &Signal<'_>) {
        self.stage.on_signal(signal);
    }

    fn keeps_loads(&self) -> bool {
        self.stage.keeps_loads()
    }

    fn needs_lifecycle(&self) -> bool {
        self.stage.needs_lifecycle()
    }

    fn hears(&self) -> EventSet {
        self.stage.hears()
    }
}

impl<S: ReadsLoad> Scorer for WithLoadReportAgeLimit<S> {
    fn name(&self) -> &'static str {
        self.stage.name()
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let cutoff = LoadReportCutoff::new(facts.now_micros, self.max_age_ms);
        self.stage
            .score_with(facts, candidates, kept, out, |row| cutoff.load(row))
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        let cutoff = LoadReportCutoff::new(facts.now_micros, self.max_age_ms);
        self.stage
            .score_front_with(facts, candidates, kept, asked, out, |row| cutoff.load(row))
    }

    fn row_wise(&self) -> bool {
        self.stage.row_wise()
    }

    fn front_wise(&self) -> bool {
        self.stage.front_wise()
    }

    fn owns(&self) -> bool {
        self.stage.owns()
    }

    // The rows reach the stage as they stand, not through the cutoff: a
    // stage that owns its pool by a row's load would read past the limit.
    // No wrapped stage owns a pool today.
    fn own(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
    ) -> bool {
        self.stage.own(facts, pool, candidates, keep, kept)
    }

    fn spread(&self) -> ScoreSpread {
        self.stage.spread()
    }

    fn needs(&self) -> Needs {
        self.stage.needs()
    }

    fn balances_load_against_affinity(&self) -> bool {
        self.stage.balances_load_against_affinity()
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        self.stage.scope()
    }

    fn refuses_by_reports(&self) -> bool {
        self.stage.refuses_by_reports()
    }

    fn warm_up_rule(&self) -> Option<WarmUpRule> {
        self.stage.warm_up_rule()
    }

    fn states_overlap(&self) -> bool {
        self.stage.states_overlap()
    }

    fn said(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], row: u32) -> Overlap {
        self.stage.said(facts, candidates, row)
    }
}

/// The rows whose least and greatest cost are found by compares
/// ([`normalize_costs`]): the ranks of one engine, the rows of a draw.
const FEW_COSTS: usize = 16;

/// Lower raw costs become higher normalized scores; equal costs score `1.0`.
/// Returns the span of the costs, `0.0` when they are all equal.
pub(crate) fn normalize_costs(out: &mut [f32]) -> f32 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    if out.len() <= FEW_COSTS {
        // A few rows, compared one by one: a cost that is no number moves
        // neither end, as the fold below leaves them.
        for &c in out.iter() {
            if c < lo {
                lo = c;
            }
            if c > hi {
                hi = c;
            }
        }
    } else {
        // Many rows, folded: the minimum and the maximum take any order,
        // so a step of the loop reads several rows.
        for &c in out.iter() {
            lo = lo.min(c);
            hi = hi.max(c);
        }
    }
    let span = hi - lo;
    let spread = span > 0.0 && span.is_finite();
    for o in out.iter_mut() {
        *o = if spread { 1.0 - (*o - lo) / span } else { 1.0 };
    }
    if spread {
        span
    } else {
        0.0
    }
}

/// Fewest requests in flight on the host.
#[derive(Debug, Default)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct LeastRequests;

impl Lifecycle for LeastRequests {}

impl Scorer for LeastRequests {
    fn name(&self) -> &'static str {
        "least-requests"
    }

    // Scores by the count in flight: a late count moves the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (slot, &i) in out.iter_mut().zip(kept) {
            *slot = facts.in_flight(&candidates[i as usize]) as f32;
        }
        normalize_costs(out);
        TieBand::NORMALIZED
    }
}

/// The owner of the request's session key scores `1.0`, every other `0.0`.
#[derive(Debug)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct SessionAffinity;

impl Lifecycle for SessionAffinity {}

impl Scorer for SessionAffinity {
    fn name(&self) -> &'static str {
        "session-affinity"
    }

    // Scores by the session's hit on a row.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = if candidates[i as usize].session_hit {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            headers: true,
            ..Needs::default()
        }
    }
}

/// Fewest session keys with a request in flight on the worker (the host's
/// count): the `min_group` assignment of a sticky session.
#[derive(Debug, Default)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct FewestSessions;

impl Lifecycle for FewestSessions {}

impl Scorer for FewestSessions {
    fn name(&self) -> &'static str {
        "fewest-sessions"
    }

    // Scores by the sessions a row holds: a late count moves the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (slot, &i) in out.iter_mut().zip(kept) {
            *slot = f32::from(candidates[i as usize].active_sessions);
        }
        normalize_costs(out);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            active_sessions: true,
            ..Needs::default()
        }
    }
}

/// The mean prefill a load scorer prices a request of unknown size at when
/// its profile names none.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;

/// The weight of the KV-pressure barrier in a load scorer's expected wait
/// when its profile names none.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f32 = 0.15;

/// The drain rate, in tokens per second, a load scorer assumes when no
/// candidate reports a live one and its profile names none.
pub const DEFAULT_THROUGHPUT: f32 = 2000.0;

/// A cost slot not written yet: a candidate without a report is costed
/// after the reporting peers' baseline is known. Costs are never negative.
pub(crate) const PENDING: f32 = f32::NEG_INFINITY;

/// The resident work, in context tokens, lower better: what a worker's
/// report says its running requests hold, the prompts dispatched to it since
/// that report (the credit the least-load scorer books too), and
/// `request_tokens` for every request it holds: the reported ones, the
/// credited ones, and the ones the host holds beyond both, those priced at
/// the worker's mean context as well. Every row handed is scored: every
/// survivor of a whole pass, or the rows a picker drew. A worker
/// whose report does not count resident tokens (the row's
/// `resident_counted`), or that has no report, scores as the best reporting
/// peer among the rows handed plus its in-flight requests, each at the mean
/// prefill and `request_tokens`; with no row handed counting them, the
/// in-flight count decides.
#[derive(Debug)]
pub struct ResidentWork {
    pub request_tokens: u32,
    pub mean_prefill_tokens: u32,
    credits: Credits,
}

impl ResidentWork {
    pub fn new(request_tokens: u32) -> Self {
        Self {
            request_tokens,
            mean_prefill_tokens: DEFAULT_MEAN_PREFILL_TOKENS,
            credits: Credits::default(),
        }
    }

    /// Dispatches no report reflects yet on one rank of `worker`.
    pub fn held_at(&self, worker: WorkerId, dp_rank: u16) -> usize {
        self.credits.held_at(worker, dp_rank)
    }

    /// The resident work of a candidate whose report counts resident tokens,
    /// with its since-report credit.
    fn reported_cost(
        &self,
        in_flight: u32,
        r: &LoadRow,
        credit_tokens: u64,
        credit_requests: u64,
    ) -> f32 {
        let reported = u64::from(r.running) + u64::from(r.queued);
        let unaccounted = u64::from(in_flight).saturating_sub(reported + credit_requests);
        let resident = r.resident_tokens as f32;
        let mean_context = if reported > 0 {
            resident / reported as f32
        } else {
            self.mean_prefill_tokens as f32
        };
        let requests = reported + credit_requests + unaccounted;
        resident
            + credit_tokens as f32
            + unaccounted as f32 * mean_context
            + self.request_tokens as f32 * requests as f32
    }
}

impl ResidentWork {
    /// Costs the rows the first pass left pending, those with no report
    /// that counts resident tokens: the best reporting peer among the rows
    /// handed plus the row's requests in flight, or the requests alone
    /// where no row reports. Out of line: a pool whose every row reports
    /// pays for no second pass.
    #[cold]
    #[inline(never)]
    fn cost_pending(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        fleet_reports: bool,
    ) {
        let peer_baseline = if fleet_reports {
            out.iter()
                .copied()
                .filter(|cost| *cost != PENDING)
                .fold(f32::INFINITY, f32::min)
        } else {
            0.0
        };
        let in_flight_work = self.mean_prefill_tokens as f32 + self.request_tokens as f32;
        for (slot, &i) in out.iter_mut().zip(kept) {
            if *slot == PENDING {
                let in_flight = facts.in_flight(&candidates[i as usize]) as f32;
                *slot = if fleet_reports {
                    peer_baseline + in_flight * in_flight_work
                } else {
                    in_flight
                };
            }
        }
    }
}

impl ResidentWork {
    /// The cost of `row` by its report and its credit under one load of
    /// the tally, or the mark of a row costed after the reporting peers'
    /// baseline is known. A report releases the dispatches it reflects
    /// whatever the score reads of them: a rank whose report counts no
    /// resident tokens would hold its credit at the bound. The row's
    /// requests in flight are asked for where the row is costed, and of no
    /// row that is marked. Inlined into the row loop of the score: as a
    /// call it read 19 instructions a row more.
    #[inline(always)]
    fn cost_reported(
        &self,
        tally: &RankTable,
        row: &Candidate,
        report: Option<&LoadRow>,
        in_flight: impl FnOnce() -> u32,
        reporting: &mut usize,
    ) -> f32 {
        let Some(r) = report else {
            return PENDING;
        };
        let credit = tally.get(&(row.id, row.dp_rank));
        let HeldCredit { tokens, requests } = credit.map_or(HeldCredit::NONE, |since| {
            // A row that holds no dispatch at or before its sample reads
            // the word alone: the release is a call.
            if since.reaches(row.load_sampled_at_micros) {
                return Self::release(since, row);
            }
            since.held_credit()
        });
        if r.resident_counted {
            *reporting += 1;
            self.reported_cost(in_flight(), r, tokens, requests)
        } else {
            PENDING
        }
    }

    /// Releases the dispatches `row`'s report reflects, and hands back the
    /// held token-work and requests behind the release. Cold and out of
    /// line: a decision reads a row far more often than a report releases
    /// from it.
    #[cold]
    #[inline(never)]
    fn release(since: &RankCredit, row: &Candidate) -> HeldCredit {
        RankCredit::credit_of(since.release_through(row.load_sampled_at_micros))
    }
}

impl ResidentWork {
    /// The rows' costs, and how many of them report: every row by its
    /// report and its credit, under one load of the tally. Out of line:
    /// inlined into the score, the bench's cell `resident_work` ran 103
    /// instructions a decision more at 64 workers and 10 fewer at 8.
    #[inline(never)]
    fn cost_own(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) -> usize {
        let mut reporting = 0usize;
        let tally = self.credits.tally.load();
        for (slot, &i) in out.iter_mut().zip(kept) {
            let c = &candidates[i as usize];
            *slot = self.cost_reported(&tally, c, load(c), || c.in_flight, &mut reporting);
        }
        reporting
    }
}

impl Lifecycle for ResidentWork {
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.credits.dispatched(
            lease,
            target,
            side,
            since_decision_micros,
            || true,
            |since, at_micros, size| {
                // The stage's own tally holds a send of no stated size at
                // the stage's mean.
                let tokens = match size.tokens() {
                    0 => self.mean_prefill_tokens,
                    stated => stated,
                };
                since.record(at_micros, SendSize::stated(tokens));
            },
        );
    }

    fn on_membership(&self, change: &Membership<'_>) {
        self.credits.membership(change);
    }

    /// The dispatch: a credit is counted from it.
    fn hears(&self) -> EventSet {
        EventSet::NONE.with(EventKind::Dispatched)
    }
}

impl Scorer for ResidentWork {
    fn name(&self) -> &'static str {
        "resident-work"
    }

    // Costs by the count in flight and its credits: late ones move the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// The rows a picker drew are costed among themselves, as the
    /// least-load scorer costs them: a row with no report, or whose report
    /// counts no resident tokens, is costed against the rows handed.
    fn row_wise(&self) -> bool {
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

    fn needs(&self) -> Needs {
        Needs {
            loads: true,
            ..Needs::default()
        }
    }
}

impl ReadsLoad for ResidentWork {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        // A row's report is read through `load`: a row past a decision's
        // cutoff has none, so no tally is read for it.
        let reporting = self.cost_own(candidates, kept, out, load);
        if reporting < kept.len() {
            self.cost_pending(facts, candidates, kept, out, reporting > 0);
        }
        // The least-load scorer's band, on a score in tokens: equal work is
        // a tie, drawn by the picker.
        let span = normalize_costs(out);
        if span > 0.0 {
            TieBand((TIE_EPSILON_SECS / span).min(1.0))
        } else {
            TieBand::NORMALIZED
        }
    }
}

/// The sum of a row's named fields times their coefficients. Rows within
/// the cost margin of the least cost share the top score.
#[derive(Debug)]
pub struct LeastCost {
    terms: Coefficients,
    margin: CostMargin,
    request_cost: Option<f64>,
    loads: bool,
}

impl LeastCost {
    pub(crate) const NAME: &str = "least-cost";

    pub(crate) fn new(
        terms: Box<[(CostTerm, f64)]>,
        margin: CostMargin,
        unreported: Option<Unreported>,
    ) -> Self {
        let loads = terms.iter().any(|&(term, _)| term.reads_load());
        let request_cost = match unreported {
            Some(Unreported::OverLeast { request_cost }) => Some(request_cost),
            Some(Unreported::MissingAsZero) | None => None,
        };
        Self {
            terms: Coefficients::of(&terms),
            margin,
            request_cost,
            loads,
        }
    }

    /// Whether a term prices the sends by their tokens: a stage that holds
    /// such a term asks for each request's size
    /// ([`Coefficients::sizes_sends`]).
    fn sizes_sends(&self) -> bool {
        self.terms.sizes_sends()
    }

    /// Closes a row's sum by the profile's sends, in name order
    /// ([`Coefficients::add_sent`]). A pass reads the count first, so it
    /// holds no sum across that read.
    #[inline(always)]
    fn add_sent(&self, sum: &mut f64, sent: DispatchCounts) {
        self.terms.add_sent(sum, sent);
    }

    /// The cost of a row that lacks a value a term names: its report, a
    /// field of it, or a signal's fresh value. Under `over-least` the row
    /// is left open: `sum` is dropped, and the row's requests in flight
    /// price it over the least kept sum ([`Self::price_and_score`]). Under
    /// `missing-as-zero` the term adds nothing and `sum` stands. The count
    /// is read only where it prices the row.
    #[inline(always)]
    fn incomplete(&self, sum: f64, in_flight: impl FnOnce() -> u32) -> RowCost {
        self.request_cost.map_or(RowCost::Sum(sum), |price| {
            RowCost::OverLeast(f64::from(in_flight()) * price)
        })
    }

    /// Prices the rows left open over the least kept sum, then scores the
    /// costs between their ends.
    #[inline(always)]
    fn price_and_score(
        &self,
        out: &mut [f32],
        costs: &mut [RowCost],
        mut bounds: CostBounds,
        pending: bool,
    ) -> TieBand {
        if pending {
            let baseline = if bounds.least.is_finite() {
                bounds.least
            } else {
                0.0
            };
            for cost in costs.iter_mut() {
                if let RowCost::OverLeast(price) = *cost {
                    let sum = baseline + price;
                    bounds.include(sum);
                    *cost = RowCost::Sum(sum);
                }
            }
        }
        write_cost_scores(out, self.margin, costs, bounds)
    }

    /// The cost of a row with its report and count as the decision reads them.
    #[inline(always)]
    fn cost_of<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        load: Option<&LoadRow>,
        in_flight: u32,
    ) -> RowCost {
        let sum = 0.0 + f64::from(in_flight) * self.terms.in_flight;
        let missing = || self.incomplete(sum, || in_flight);
        match load {
            Some(report) => self
                .terms
                .add_report::<RESIDENT, WAITING, KV_OR_RUNNING>(
                    sum,
                    report,
                    self.request_cost.is_some(),
                )
                .map_or_else(missing, RowCost::Sum),
            None if self.loads => missing(),
            None => RowCost::Sum(sum),
        }
    }

    #[inline(always)]
    fn sum_rows<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        costs: &mut Vec<RowCost>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
        counts: impl Fn(&Candidate) -> u32,
    ) -> (CostBounds, bool) {
        let mut bounds = CostBounds::EMPTY;
        let mut pending = false;
        costs.extend(kept.iter().map(|&i| {
            let row = &candidates[i as usize];
            let cost = self.cost_of::<RESIDENT, WAITING, KV_OR_RUNNING>(load(row), counts(row));
            match cost {
                RowCost::Sum(sum) => bounds.include(sum),
                RowCost::OverLeast(_) => pending = true,
            }
            cost
        }));
        (bounds, pending)
    }

    /// A document names which optional values the walk tests.
    #[inline(always)]
    fn walk<const KV_OR_RUNNING: bool>(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        costs: &mut Vec<RowCost>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
        counts: impl Fn(&Candidate) -> u32,
    ) -> (CostBounds, bool) {
        let resident = self.terms.resident_tokens != 0.0;
        let waiting = self.terms.waiting_uncached_tokens != 0.0;
        match (resident, waiting) {
            (false, false) => {
                self.sum_rows::<false, false, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
            (false, true) => {
                self.sum_rows::<false, true, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
            (true, false) => {
                self.sum_rows::<true, false, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
            (true, true) => {
                self.sum_rows::<true, true, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
        }
    }

    fn collect_costs(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        costs: &mut Vec<RowCost>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) -> (CostBounds, bool) {
        // Every row carries its count, and the walk reads it where it
        // costs the row.
        let counts = |row: &Candidate| facts.in_flight(row);
        if self.terms.names_kv_or_running() {
            self.walk::<true>(candidates, kept, costs, load, counts)
        } else {
            self.walk::<false>(candidates, kept, costs, load, counts)
        }
    }
}

impl Lifecycle for LeastCost {}

impl Scorer for LeastCost {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    // Costs by reports and the count in flight: late ones move the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn needs(&self) -> Needs {
        Needs {
            loads: self.loads,
            ..Needs::default()
        }
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
}

impl ReadsLoad for LeastCost {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        with_costs(|costs| {
            let (bounds, pending) = self.collect_costs(facts, candidates, kept, costs, load);
            self.price_and_score(out, costs, bounds, pending)
        })
    }
}

/// `least-cost` where the document names a `sent.` term: the sum of
/// [`LeastCost`], then the profile's sends. A document that names none is
/// built as [`LeastCost`] alone and reads no count.
#[derive(Debug)]
pub(crate) struct LeastCostWithDispatches {
    cost: LeastCost,
    sent: Arc<DispatchCredit>,
}

impl LeastCostWithDispatches {
    pub(crate) fn new(cost: LeastCost, sent: Arc<DispatchCredit>) -> Self {
        Self { cost, sent }
    }
}

impl Lifecycle for LeastCostWithDispatches {}

impl Scorer for LeastCostWithDispatches {
    fn name(&self) -> &'static str {
        self.cost.name()
    }

    fn refuses_by_reports(&self) -> bool {
        self.cost.refuses_by_reports()
    }

    fn row_wise(&self) -> bool {
        self.cost.row_wise()
    }

    // A report ends the sends before it, so the sends need the reports
    // though no term may read one. A term that prices the sends by their
    // tokens needs each request's size stated: a send of text alone is
    // counted by its text's bytes.
    fn needs(&self) -> Needs {
        Needs {
            loads: true,
            prompt_size: self.cost.sizes_sends(),
            ..Needs::default()
        }
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
}

impl ReadsLoad for LeastCostWithDispatches {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        with_costs(|costs| {
            let (_, pending) = self
                .cost
                .collect_costs(facts, candidates, kept, costs, load);
            // The sends close the sum: their names sort behind every field
            // the walk adds. A row priced over the least kept no sum and
            // takes none. The sends move the ends of the sums, so the ends
            // are taken here and the walk's are dropped.
            let sends = self.sent.pass(facts);
            let mut bounds = CostBounds::EMPTY;
            for (&row, cost) in kept.iter().zip(costs.iter_mut()) {
                if let RowCost::Sum(sum) = cost {
                    let sent = sends.counts(&candidates[row as usize]);
                    self.cost.add_sent(sum, sent);
                    bounds.include(*sum);
                }
            }
            self.cost.price_and_score(out, costs, bounds, pending)
        })
    }
}

/// `least-cost` where the document names a `signal.` term: the sum of
/// [`LeastCost`], then the profile's sends where a term names them, then
/// the signals in name order. A row with no fresh value for a signal a
/// term names is unreported for the scorer: under `over-least` it is
/// priced over the least by its requests in flight, as a row with no
/// report is, and under `missing-as-zero` the term adds nothing. A
/// document that names no signal is built as one of the other two and
/// reads no cell.
#[derive(Debug)]
pub(crate) struct LeastCostSignals {
    cost: LeastCost,
    /// The profile's count of sends, where a term reads it.
    sent: Option<Arc<DispatchCredit>>,
    signals: Box<[SignalTerm]>,
}

impl LeastCostSignals {
    pub(crate) fn new(
        cost: LeastCost,
        sent: Option<Arc<DispatchCredit>>,
        signals: Box<[SignalTerm]>,
    ) -> Self {
        Self {
            cost,
            sent,
            signals,
        }
    }
}

impl Lifecycle for LeastCostSignals {}

impl Scorer for LeastCostSignals {
    fn name(&self) -> &'static str {
        self.cost.name()
    }

    fn refuses_by_reports(&self) -> bool {
        self.cost.refuses_by_reports()
    }

    fn row_wise(&self) -> bool {
        self.cost.row_wise()
    }

    // A report ends the sends before it, so the sends need the reports
    // though no term may read one. A term that prices the sends by their
    // tokens needs each request's size stated. The cells stand beside the
    // rows of a profile whose stage says it reads a signal.
    fn needs(&self) -> Needs {
        Needs {
            loads: self.cost.needs().loads || self.sent.is_some(),
            prompt_size: self.cost.sizes_sends(),
            signal_cells: true,
            ..Needs::default()
        }
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
}

impl ReadsLoad for LeastCostSignals {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        with_costs(|costs| {
            let (_, mut pending) = self
                .cost
                .collect_costs(facts, candidates, kept, costs, load);
            // The sends close the fields' sum and the signals close the
            // sends': `signal.` sorts behind every other name. A row priced
            // over the least kept no sum and takes none. Both move the ends
            // of the sums, so the ends are taken here and the walk's are
            // dropped.
            let sends = self.sent.as_ref().map(|sent| sent.pass(facts));
            // Once a pass: the facts' cells and each signal's oldest
            // instant still read. Once a row: the row's cells.
            let signals = SignalSums::new(facts, &self.signals);
            let mut bounds = CostBounds::EMPTY;
            for (&i, cost) in kept.iter().zip(costs.iter_mut()) {
                let RowCost::Sum(sum) = cost else {
                    continue;
                };
                let row = &candidates[i as usize];
                if let Some(sends) = &sends {
                    let sent = sends.counts(row);
                    self.cost.add_sent(sum, sent);
                }
                let (priced, whole) = signals.add(*sum, signals.row(row));
                *sum = priced;
                if whole {
                    bounds.include(priced);
                    continue;
                }
                // A signal with no fresh value leaves the row incomplete,
                // as no report does in the walk.
                *cost = self.cost.incomplete(priced, || facts.in_flight(row));
                match *cost {
                    RowCost::Sum(sum) => bounds.include(sum),
                    RowCost::OverLeast(_) => pending = true,
                }
            }
            self.cost.price_and_score(out, costs, bounds, pending)
        })
    }
}

#[derive(Clone, Copy)]
enum RowCost {
    Sum(f64),
    /// The price for the row's requests in flight, before adding the least reported cost.
    OverLeast(f64),
}

impl RowCost {
    fn value(self) -> f64 {
        match self {
            Self::Sum(value) | Self::OverLeast(value) => value,
        }
    }
}

/// The ends of the finite costs found while pricing the rows.
struct CostBounds {
    least: f64,
    greatest: f64,
}

impl CostBounds {
    const EMPTY: Self = Self {
        least: f64::INFINITY,
        greatest: f64::NEG_INFINITY,
    };

    fn include(&mut self, cost: f64) {
        if cost < self.least {
            self.least = cost;
        }
        if cost > self.greatest {
            self.greatest = cost;
        }
    }
}

thread_local! {
    /// One slot per row scored. The deciding thread keeps the buffer for reuse.
    static COSTS: Cell<Vec<RowCost>> = const { Cell::new(Vec::new()) };
}

fn with_costs(work: impl FnOnce(&mut Vec<RowCost>) -> TieBand) -> TieBand {
    // Move the buffer out so a cost callback can score another set of rows.
    let mut costs = COSTS.take();
    let band = work(&mut costs);
    costs.clear();
    COSTS.set(costs);
    band
}

/// Writes scores from finite costs with a finite span, lower cost better.
/// Costs within `margin` of the least cost score `1.0`. The returned band
/// is zero, since the tie is already in the scores. Calls `cost` once for
/// each position of `out`, in order.
pub fn score_costs(
    out: &mut [f32],
    margin: CostMargin,
    mut cost: impl FnMut(usize) -> f64,
) -> TieBand {
    with_costs(|costs| {
        costs.extend((0..out.len()).map(|i| RowCost::Sum(cost(i))));
        let mut bounds = CostBounds::EMPTY;
        for &cost in costs.iter() {
            bounds.include(cost.value());
        }
        write_cost_scores(out, margin, costs, bounds)
    })
}

fn write_cost_scores(
    out: &mut [f32],
    margin: CostMargin,
    costs: &[RowCost],
    CostBounds { least, greatest }: CostBounds,
) -> TieBand {
    let span = greatest - least;
    if span == 0.0 {
        out.fill(1.0);
    } else {
        let limit = least + margin.absolute + margin.relative * least.abs();
        for (slot, &cost) in out.iter_mut().zip(costs) {
            let cost = cost.value();
            *slot = if cost <= limit {
                1.0
            } else {
                // Keep rounding to f32 from adding a row to the tie.
                ((1.0 - (cost - least) / span) as f32).min(1.0_f32.next_down())
            };
        }
    }
    TieBand(0.0)
}

#[cfg(test)]
#[path = "../../tests/unit/stages/scorers/tests.rs"]
pub(crate) mod tests;
