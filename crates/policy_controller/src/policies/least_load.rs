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
        in_flight::with_counts,
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
        self.sent.credits.held(worker)
    }

    /// Dispatches no report reflects yet on one rank of `worker`.
    pub fn held_at(&self, worker: WorkerId, dp_rank: u16) -> usize {
        self.sent.credits.held_at(worker, dp_rank)
    }

    /// One rank's held work as a decision reads it: token-work, a send of
    /// no stated size at the mean prefill, and requests.
    #[cfg(test)]
    fn held_word_at(&self, worker: WorkerId, dp_rank: u16) -> (u64, u64) {
        self.sent
            .credits
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
        if let Some(since) = self.sent.credits.tally.load().get(&(worker, dp_rank)) {
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
        with_counts!(facts, |counts| {
            for (slot, &i) in out.iter_mut().zip(kept) {
                if *slot == PENDING {
                    *slot = self.unreported_cost(
                        counts(&candidates[i as usize]),
                        nominal,
                        fleet_has_loads,
                        peer_baseline,
                    );
                }
            }
        });
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
        with_counts!(facts, |counts| {
            for (slot, &i) in out.iter_mut().zip(rows) {
                let c = &candidates[i as usize];
                *slot = match load(c) {
                    Some(report) => {
                        let sent = sends.reported(c);
                        let tokens = sent.tokens + u64::from(sent.unsized_requests) * mean;
                        reporting += 1;
                        self.credit_work(report, counts(c), tokens, u64::from(sent.requests))
                    }
                    None => PENDING,
                };
            }
        });
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
mod tests {
    use super::*;
    use crate::{
        engine::{
            decision::{Pair, Targets},
            framework::Picker,
        },
        stages::{
            pickers::{target_of, Best},
            scorers::{
                take_lane,
                tests::{candidate, kept, report},
                SessionAffinity, RANK_CREDIT_KEPT,
            },
        },
    };

    #[test]
    fn least_load_prices_work_over_drain_and_credits_a_dispatch_until_its_report() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(report(1, 0));
        pool[0].load_sampled_at_micros = 1_000;
        pool[1].load = Some(report(0, 4096));
        pool[1].load_sampled_at_micros = 1_000;
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(
            scores,
            [1.0, 0.0],
            "one running request beats a queue of 4096 tokens"
        );

        // A dispatch after the report's sample stays as credit...
        let lease = Lease {
            issued_micros: 2_000,
            tokens: 8_192,
            ..Lease::default()
        };
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(ll.held(pool[0].id), 1);
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0], "the credited worker now costs more");

        // ...and a report sampled after the dispatch releases it.
        pool[0].load_sampled_at_micros = 3_000;
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(ll.held(pool[0].id), 0);
        assert_eq!(scores, [1.0, 0.0]);

        // A removed worker leaves no tally behind.
        ll.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
        ll.on_membership(&Membership::Removed(pool[1].id));
        assert_eq!(ll.held(pool[1].id), 0);
    }

    #[test]
    fn a_credit_dates_from_the_dispatch_and_not_from_the_decision() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0)];
        pool[0].load = Some(report(0, 0));
        let lease = Lease {
            issued_micros: 100,
            tokens: 10,
            ..Lease::default()
        };
        // Decided at 100, held in a queue, dispatched at 300.
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 200);
        let mut scores = [0.0f32; 1];
        pool[0].load_sampled_at_micros = 200;
        ll.score(&RequestFacts::default(), &pool, &kept(1), &mut scores);
        assert_eq!(
            ll.held(pool[0].id),
            1,
            "a sample before the dispatch keeps the credit"
        );
        pool[0].load_sampled_at_micros = 300;
        ll.score(&RequestFacts::default(), &pool, &kept(1), &mut scores);
        assert_eq!(
            ll.held(pool[0].id),
            0,
            "a sample at the dispatch releases it"
        );
    }

    /// A dispatch whose lease states no size is credited at the mean of
    /// the stage that credits it, whether it enters its rank or finds it,
    /// and so costs what its request cost in flight. A stated size is
    /// credited as stated.
    #[test]
    fn a_dispatch_that_states_no_size_is_credited_at_the_stages_own_mean() {
        let mut pool = [candidate(0, 2), candidate(1, 2)];
        pool[0].load = Some(report(0, 0));
        pool[1].load = Some(report(0, 0));
        let unstated = Lease {
            issued_micros: 2_000,
            ..Lease::default()
        };
        let stated = Lease {
            tokens: 48,
            ..unstated
        };
        let report = pool[0].load.as_ref().unwrap();
        for mean in [512, 2_048] {
            let ll = LeastLoad::new(0.15, mean, 2000.0);
            // Two requests in flight that no report and no credit account
            // for cost the stage's mean each.
            let in_flight = ll.credit_work(report, 2, 0, 0);
            assert_eq!(in_flight, 2.0 * mean as f32);
            // The first dispatch to a rank enters it and the second finds
            // it: each is credited at that mean, so the two cost dispatched
            // what they cost in flight.
            for _ in 0..2 {
                ll.on_dispatched(&unstated, &target_of(&pool[0]), Side::Single, 0);
                ll.on_dispatched(&stated, &target_of(&pool[1]), Side::Single, 0);
            }
            let (tokens, requests) = ll.held_word_at(pool[0].id, 0);
            assert_eq!((tokens, requests), (2 * u64::from(mean), 2));
            assert_eq!(ll.credit_work(report, 2, tokens, requests), in_flight);
            // A stated size is credited as stated.
            assert_eq!(ll.held_word_at(pool[1].id, 0), (96, 2));
        }
    }

    #[test]
    fn a_membership_snapshot_keeps_only_the_tallies_of_its_ids() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let pool = [candidate(0, 0), candidate(1, 0)];
        let lease = Lease::default();
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        ll.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
        ll.on_membership(&Membership::Snapshot(&[pool[0].id]));
        assert_eq!(ll.held(pool[0].id), 1);
        assert_eq!(
            ll.held(pool[1].id),
            0,
            "a worker the snapshot lacks is gone"
        );
    }

    #[test]
    fn a_dark_fleet_scores_by_in_flight_and_a_missing_report_by_the_peer_baseline() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let pool = [candidate(0, 3), candidate(1, 1)];
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0]);
        let mut mixed = pool;
        mixed[0].load = Some(report(0, 0));
        mixed[0].in_flight = 0;
        ll.score(&RequestFacts::default(), &mixed, &kept(2), &mut scores);
        assert_eq!(scores[0], 1.0, "the idle reporting worker wins");
        assert!(
            scores[1] < 1.0,
            "the dark worker starts from the baseline plus its in-flight"
        );
    }

    /// A row without a report drains at the mean rate of the peers that
    /// report one, not at the default. Its peers report 8,000 and 1,000
    /// tokens a second, so its four requests wait 4 x 1,024 / 4,500 =
    /// 0.91 s, ahead of the peer whose one request waits 1.02 s. At the
    /// default of 2,000 they would wait 2.05 s, behind it.
    #[test]
    fn a_row_without_a_report_drains_at_its_peers_mean_rate() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 4), candidate(2, 0)];
        pool[0].load = Some(LoadRow {
            live_rate: 8_000,
            ..report(0, 0)
        });
        pool[2].load = Some(LoadRow {
            live_rate: 1_000,
            ..report(1, 0)
        });
        let mut scores = [0.0f32; 3];
        ll.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert!(scores[0] > scores[1] && scores[1] > scores[2], "{scores:?}");
    }

    /// A row whose report states no rate drains at the mean rate of the
    /// rows that state one, wherever it stands among them: here it is the
    /// first kept row. Its peers state 8,000 and 1,000 tokens a second, so
    /// its four running requests wait 4 x 1,024 / 4,500 = 0.91 s, ahead of
    /// the peer whose one request waits 1.02 s. At the default of 2,000
    /// they would wait 2.05 s, behind it.
    #[test]
    fn a_first_row_without_a_rate_drains_at_the_mean_of_the_rows_behind_it() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        pool[0].load = Some(LoadRow {
            live_rate: 0,
            ..report(4, 0)
        });
        pool[1].load = Some(LoadRow {
            live_rate: 8_000,
            ..report(0, 0)
        });
        pool[2].load = Some(LoadRow {
            live_rate: 1_000,
            ..report(1, 0)
        });
        let mut scores = [0.0f32; 3];
        ll.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert!(scores[1] > scores[0] && scores[0] > scores[2], "{scores:?}");
        assert_eq!(ll.mean_rate(&pool, &kept(3), |c| c.load.as_ref()), 4_500.0);
    }

    /// Reports that state no rate at all drain at the default rate, a
    /// number: the rows rank by their work.
    #[test]
    fn a_fleet_whose_reports_state_no_rate_ranks_by_its_work() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(LoadRow {
            live_rate: 0,
            ..report(2, 0)
        });
        pool[1].load = Some(LoadRow {
            live_rate: 0,
            ..report(1, 0)
        });
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0]);
        assert_eq!(
            ll.mean_rate(&pool, &kept(2), |c| c.load.as_ref()),
            ll.default_throughput
        );
    }

    #[test]
    fn kv_pressure_raises_the_cost_of_an_otherwise_equal_worker() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(report(1, 0));
        pool[1].load = Some(LoadRow {
            kv_usage_permille: 900,
            active_kv_permille: 900,
            ..report(1, 0)
        });
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [1.0, 0.0]);
    }

    /// A full but reclaimable prefix cache is no pressure: the idle worker
    /// whose running requests hold none of its cache beats the busy one
    /// holding half, whatever the raw usage says (the legacy policy's case).
    #[test]
    fn reclaimable_prefix_cache_does_not_make_an_idle_worker_busy() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(LoadRow {
            kv_usage_permille: 1000,
            active_kv_permille: 0,
            ..report(0, 0)
        });
        pool[1].load = Some(LoadRow {
            kv_usage_permille: 500,
            active_kv_permille: 500,
            ..report(1, 0)
        });
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores[0], 1.0, "the idle worker wins");
        assert!(
            scores[1] < 1.0,
            "the busy worker's held share is the pressure"
        );
    }

    #[test]
    fn the_kv_barrier_reads_the_active_share_at_the_legacy_clamp() {
        let scorer = LeastLoad::new(0.15, 1024, 2000.0);
        let row = |active: u16, total: u16, queued: u32| LoadRow {
            active_kv_permille: active,
            kv_usage_permille: total,
            queued,
            live_rate: 2000,
            ..LoadRow::default()
        };
        let facts = RequestFacts::default();
        // A saturated row prices its barrier at 999 times the weight (150 s
        // at 0.15), so a queue of 400 requests at the mean prefill (205 s)
        // loses to it; at a clamp of 0.999999 the barrier would be 150,000 s
        // and the queue would win.
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(row(1000, 1000, 0));
        pool[1].load = Some(row(0, 0, 400));
        let mut scores = [0.0f32; 2];
        scorer.score(&facts, &pool, &kept(2), &mut scores);
        assert!(scores[0] > scores[1], "{scores:?}");
        // The barrier reads the active share: a cache full of evictable
        // blocks (total 1.0, active 0.1) is lighter than one half held by
        // running requests.
        pool[0].load = Some(row(100, 1000, 0));
        pool[1].load = Some(row(500, 500, 0));
        scorer.score(&facts, &pool, &kept(2), &mut scores);
        assert!(scores[0] > scores[1], "{scores:?}");
    }

    /// What the load stage says of the rows it is asked of
    /// ([`Scorer::score_front`]). Where each holds a report it scores them
    /// and says no word: a report without a rate drains at the mean rate
    /// of the kept rows, not of the rows asked and not of the pool. Where
    /// one holds no report it says that it must be asked whole, before it
    /// costs any.
    #[test]
    fn least_load_answers_for_the_rows_asked_unless_one_holds_no_report() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let facts = RequestFacts::default();
        // Five rows, the last two dropped by a filter. The second reports
        // no rate: it drains at 10,000 tokens a second, the mean of the
        // first's 2,000 and the third's 18,000, and waits 0.31 s where the
        // first waits 0.51 s. At the first's rate alone, or at the pool's
        // mean of 5,050, it would wait longer than the first.
        let reports = [(1, 2_000), (3, 0), (2, 18_000), (1, 100), (1, 100)];
        let mut pool: Vec<Candidate> = (0..5).map(|at| candidate(at, 0)).collect();
        for (row, (running, live_rate)) in pool.iter_mut().zip(reports) {
            row.load = Some(LoadRow {
                live_rate,
                ..report(running, 0)
            });
        }
        let (mut said, mut plain) = ([0.0f32; 3], [0.0f32; 3]);
        let band = ll.score(&facts, &pool, &kept(3), &mut plain);
        // Asked of every kept row, it answers as the plain entry.
        assert_eq!(
            ll.score_front(&facts, &pool, &kept(3), &kept(3), &mut said),
            (band, ScoreFront::Unsaid)
        );
        assert_eq!(said, plain);
        // Asked of the first two, it orders them as among the three.
        assert!(plain[1] > plain[0], "{plain:?}");
        let word = ll
            .score_front(&facts, &pool, &kept(3), &[0, 1], &mut said[..2])
            .1;
        assert_eq!((word, &said[..2]), (ScoreFront::Unsaid, &[0.0, 1.0][..]));
        // A row without a report is costed from its reporting peers.
        pool[1].load = None;
        let mut untouched = [-1.0f32; 2];
        let word = ll
            .score_front(&facts, &pool, &kept(3), &[0, 1], &mut untouched)
            .1;
        assert_eq!(word, ScoreFront::Whole);
        assert_eq!(untouched, [-1.0; 2], "nothing is costed");
        // Asked of the rows that hold a report alone, it answers for them.
        let word = ll
            .score_front(&facts, &pool, &kept(3), &[0, 2], &mut said[..2])
            .1;
        assert_eq!(word, ScoreFront::Unsaid);
    }

    #[test]
    fn least_load_ties_within_a_microsecond_of_wait_on_a_spread_pool() {
        // Two workers a quarter of a microsecond apart in expected wait, and a
        // third four seconds behind that stretches the normalized scale: the
        // two still tie, and a pair two microseconds apart still does not.
        // (The band is exact down to the normalized scale's resolution: a
        // difference under about 6e-8 of the spread is one f32 value, so a
        // spread past about 16 s widens a tie beyond the microsecond.)
        let ll = LeastLoad::new(0.0, 1024, 1_000_000.0);
        let near = |tokens: u32| LoadRow {
            waiting_uncached_tokens: tokens,
            live_rate: 4_000_000,
            ..LoadRow::default()
        };
        let mut pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        pool[0].load = Some(near(4_000_000)); // 1.0 s
        pool[1].load = Some(near(4_000_001)); // 1.0 s + 0.25 us
        pool[2].load = Some(near(20_000_000)); // 5 s
        let mut scores = [0.0f32; 3];
        let band = ll.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert!(
            band.0 < TieBand::NORMALIZED.0,
            "a spread pool narrows the band: {band:?}"
        );
        let mut picks = [0usize; 3];
        for seed in 0..200u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
            picks[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
        }
        assert!(picks[0] > 40 && picks[1] > 40 && picks[2] == 0, "{picks:?}");

        pool[1].load = Some(near(4_000_008)); // 1.0 s + 2 us
        let band = ll.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        let mut out = Targets::new();
        for seed in 0..50u64 {
            out.clear();
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
            assert_eq!(
                out.first().map(|t| t.position),
                Some(0),
                "two microseconds apart is no tie"
            );
        }
    }

    #[test]
    fn a_cold_fleet_spreads_its_picks_across_every_worker() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let pool: Vec<Candidate> = (0..128)
            .map(|i| {
                let mut c = candidate(i, 0);
                c.load = Some(report(0, 0));
                c
            })
            .collect();
        let kept = kept(128);
        let mut scores = vec![0.0f32; 128];
        let band = ll.score(&RequestFacts::default(), &pool, &kept, &mut scores);
        let mut hits = vec![0usize; 128];
        for seed in 0..1_000u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &kept, &scores, band, &mut out);
            hits[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
        }
        let idle = hits.iter().filter(|&&n| n == 0).count();
        assert!(
            idle <= 2,
            "{idle} of 128 cold workers never picked in a thousand draws"
        );
        assert!(
            hits.iter().all(|&n| n < 40),
            "a cold worker took {} of a thousand",
            hits.iter().max().unwrap_or(&0)
        );
    }

    #[test]
    fn a_dispatch_credits_the_worker_of_its_side() {
        let scorer = LeastLoad::new(0.15, 1024, 2000.0);
        let lease = Lease {
            issued_micros: 10,
            tokens: 100,
            ..Lease::default()
        };
        let target = Target {
            position: 0,
            id: WorkerId(1),
            dp_rank: 0,
            pair: Some(Pair {
                decode_position: 0,
                decode_id: WorkerId(2),
                ..Pair::default()
            }),
        };
        scorer.on_dispatched(&lease, &target, Side::Prefill, 0);
        scorer.on_dispatched(&lease, &target, Side::Decode, 0);
        assert_eq!(
            scorer.held(WorkerId(1)),
            1,
            "the prefill side credits the prefill"
        );
        assert_eq!(
            scorer.held(WorkerId(2)),
            1,
            "the decode side credits the pair's decode"
        );
        let single = Target {
            pair: None,
            ..target
        };
        scorer.on_dispatched(&lease, &single, Side::Single, 0);
        assert_eq!(scorer.held(WorkerId(1)), 2);
    }

    #[test]
    fn two_ranks_of_one_worker_are_credited_and_released_apart() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        for (rank, c) in pool.iter_mut().enumerate() {
            c.id = WorkerId(7);
            c.dp_rank = rank as u16;
            c.dp_size = 2;
            c.load = Some(report(0, 0));
            c.load_sampled_at_micros = 1_000;
        }
        let lease = Lease {
            issued_micros: 2_000,
            tokens: 8_192,
            ..Lease::default()
        };
        // A dispatch to rank 0 is rank 0's credit alone.
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(
            (ll.held_at(WorkerId(7), 0), ll.held_at(WorkerId(7), 1)),
            (1, 0)
        );
        assert_eq!(ll.held(WorkerId(7)), 1);
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(
            scores,
            [0.0, 1.0],
            "the credited rank costs more than its idle sibling"
        );
        // Rank 1's later report releases nothing of rank 0's.
        pool[1].load_sampled_at_micros = 3_000;
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(
            ll.held_at(WorkerId(7), 0),
            1,
            "a sibling rank's report is not this rank's"
        );
        assert_eq!(scores, [0.0, 1.0]);
        // Rank 0's own report sampled after the dispatch does.
        pool[0].load_sampled_at_micros = 3_000;
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(ll.held(WorkerId(7)), 0);
        assert_eq!(scores, [1.0, 1.0]);
        // The decode side of a pair credits the decode's own rank.
        let pair = Target {
            position: 0,
            id: WorkerId(1),
            dp_rank: 0,
            pair: Some(Pair {
                decode_position: 1,
                decode_id: WorkerId(7),
                decode_dp_rank: 1,
                ..Pair::default()
            }),
        };
        ll.on_dispatched(&lease, &pair, Side::Decode, 0);
        assert_eq!(
            (ll.held_at(WorkerId(7), 0), ll.held_at(WorkerId(7), 1)),
            (0, 1)
        );
        // A resize starts every rank of the worker over; a removal takes them.
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        ll.on_membership(&Membership::Resized(WorkerId(7)));
        assert_eq!(ll.held(WorkerId(7)), 0);
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        ll.on_dispatched(&lease, &pair, Side::Decode, 0);
        ll.on_membership(&Membership::Removed(WorkerId(7)));
        assert_eq!(ll.held(WorkerId(7)), 0);
    }

    #[test]
    fn a_lane_filling_after_another_keeps_its_recent_credits_past_a_partial_release() {
        // One lane books a bound's worth of dispatches, then a second lane
        // books a thousand newer ones: the overflows retire the row's oldest
        // dispatches, in the first lane, never the second lane's own
        // previous ones. A sample past the first batch then leaves the
        // thousand recent credits, as one deque per row did.
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool: Vec<Candidate> = vec![candidate(0, 0)];
        pool[0].load = Some(report(0, 0));
        pool[0].load_sampled_at_micros = 1_000;
        let kept: Vec<u32> = vec![0];
        let book = |lane: usize, from: u64, count: u64, tokens: u32| {
            std::thread::scope(|s| {
                s.spawn(|| {
                    take_lane(lane);
                    for n in 0..count {
                        let lease = Lease {
                            issued_micros: from + n,
                            tokens,
                            ..Lease::default()
                        };
                        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
                    }
                });
            });
        };
        book(0, 2_000, RANK_CREDIT_KEPT, 7);
        book(1, 20_000, 1_000, 70);
        let mut scores = vec![0.0f32; 1];
        let mut sampled = pool.clone();
        sampled[0].load_sampled_at_micros = 15_000;
        ll.score(&RequestFacts::default(), &sampled, &kept, &mut scores);
        assert_eq!(ll.held(pool[0].id), 1_000, "the recent credits");
        assert_eq!(
            ll.held_word_at(pool[0].id, 0),
            (1_000 * 70, 1_000),
            "their token-work"
        );
    }

    #[test]
    fn a_dispatch_past_the_bound_leaves_its_row_level_with_a_peer_at_the_bound() {
        // Two rows hold the bound. A dispatch that lands on the first lets
        // that row's one oldest go, so the row holds the bound still and
        // scores as its peer does: the next send is a tie between them.
        // Dispatches that land on it together from every lane each let one
        // go, and the row holds the bound once they are through.
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool: Vec<Candidate> = vec![candidate(0, 0), candidate(1, 0)];
        for row in &mut pool {
            row.load = Some(report(0, 0));
            row.load_sampled_at_micros = 1_000;
        }
        // On a thread of its own: a thread keeps the lane it took.
        let book = |row: usize, lane: usize, from: u64, count: u64| {
            take_lane(lane);
            for n in 0..count {
                let lease = Lease {
                    issued_micros: from + n,
                    tokens: 7,
                    ..Lease::default()
                };
                ll.on_dispatched(&lease, &target_of(&pool[row]), Side::Single, 0);
            }
        };
        std::thread::scope(|s| {
            for row in 0..2 {
                s.spawn(move || book(row, 0, 2_000, 10));
                s.spawn(move || book(row, 1, 3_000, RANK_CREDIT_KEPT - 10));
            }
        });
        std::thread::scope(|s| {
            s.spawn(|| book(0, 2, 30_000, 1));
        });
        let bound = RANK_CREDIT_KEPT;
        assert_eq!(ll.held(pool[0].id), bound as usize, "the one oldest went");
        assert_eq!(ll.held_word_at(pool[0].id, 0), (bound * 7, bound));
        let mut scores = vec![0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &[0, 1], &mut scores);
        assert_eq!(scores[0], scores[1], "level with its peer at the bound");
        std::thread::scope(|s| {
            for lane in 0..8 {
                s.spawn(move || book(0, lane, 40_000, 500));
            }
        });
        assert_eq!(ll.held(pool[0].id), bound as usize, "each let one go");
        assert_eq!(ll.held_word_at(pool[0].id, 0), (bound * 7, bound));
        // Reports release the row's older dispatches while every lane books
        // past the bound: each one let go is older than these, so the row
        // holds them all once a sample has reached the rest.
        std::thread::scope(|s| {
            for lane in 0..8 {
                s.spawn(move || book(0, lane, 50_000, 500));
            }
            s.spawn(|| {
                let mut sampled = pool.clone();
                for at in (2_000..50_000).step_by(16) {
                    sampled[0].load_sampled_at_micros = at;
                    ll.score(&RequestFacts::default(), &sampled, &[0, 1], &mut [0.0; 2]);
                }
            });
        });
        let mut sampled = pool.clone();
        sampled[0].load_sampled_at_micros = 49_999;
        ll.score(&RequestFacts::default(), &sampled, &[0, 1], &mut scores);
        assert_eq!(ll.held(pool[0].id), 4_000, "none of the newer went");
        assert_eq!(ll.held_word_at(pool[0].id, 0), (4_000 * 7, 4_000));
    }

    #[test]
    fn an_overflow_retires_the_rows_oldest_by_instant_across_interleaved_lanes() {
        // One lane holds one old dispatch and a hundred recent ones, a
        // second the rest of the bound between them; a third lane's
        // dispatch overflows the row. The one that goes is the row's oldest
        // by instant, the old one, so a sample that reaches it alone
        // releases nothing. The front of another lane retired in its place
        // would have left the old one to that sample.
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool: Vec<Candidate> = vec![candidate(0, 0)];
        pool[0].load = Some(report(0, 0));
        pool[0].load_sampled_at_micros = 1_000;
        let kept: Vec<u32> = vec![0];
        let book = |lane: usize, from: u64, count: u64| {
            std::thread::scope(|s| {
                s.spawn(|| {
                    take_lane(lane);
                    for n in 0..count {
                        let lease = Lease {
                            issued_micros: from + n,
                            tokens: 7,
                            ..Lease::default()
                        };
                        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
                    }
                });
            });
        };
        book(1, 2_000, 1);
        book(0, 3_000, RANK_CREDIT_KEPT - 101);
        book(1, 20_000, 100);
        book(2, 30_000, 1);
        let mut scores = vec![0.0f32; 1];
        let mut sampled = pool.clone();
        sampled[0].load_sampled_at_micros = 2_000;
        ll.score(&RequestFacts::default(), &sampled, &kept, &mut scores);
        let left = RANK_CREDIT_KEPT;
        assert_eq!(
            ll.held(pool[0].id),
            left as usize,
            "the old dispatch went ahead of the sample"
        );
        assert_eq!(ll.held_word_at(pool[0].id, 0), (left * 7, left));
    }

    #[test]
    fn a_release_past_the_overflow_check_leaves_the_newer_credits() {
        // A dispatch reads the bound, a load report's release then takes the
        // row's older dispatches, and the retirement runs after it: the row
        // is below the bound, so the newer credits stay.
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool: Vec<Candidate> = vec![candidate(0, 0)];
        pool[0].load = Some(report(0, 0));
        pool[0].load_sampled_at_micros = 1_000;
        let kept: Vec<u32> = vec![0];
        let book = |lane: usize, from: u64, count: u64| {
            std::thread::scope(|s| {
                s.spawn(|| {
                    take_lane(lane);
                    for n in 0..count {
                        let lease = Lease {
                            issued_micros: from + n,
                            tokens: 7,
                            ..Lease::default()
                        };
                        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
                    }
                });
            });
        };
        book(0, 2_000, RANK_CREDIT_KEPT - 1);
        book(1, 20_000, 100);
        let mut scores = vec![0.0f32; 1];
        let mut sampled = pool.clone();
        sampled[0].load_sampled_at_micros = 15_000;
        ll.score(&RequestFacts::default(), &sampled, &kept, &mut scores);
        assert_eq!(ll.held(pool[0].id), 100, "the release past the older lane");
        ll.retire_oldest_of(pool[0].id, 0);
        assert_eq!(
            ll.held(pool[0].id),
            100,
            "a retirement below the bound retires nothing"
        );
        assert_eq!(ll.held_word_at(pool[0].id, 0), (100 * 7, 100));
    }

    #[test]
    fn a_dispatch_landing_behind_a_later_instant_keeps_its_lane_oldest_first() {
        // Two dispatches land on one lane in reverse instant order (a thread
        // that waited past another's landing, or a clock that stepped): the
        // lane holds the older one at its front, so a report's release past
        // it takes that one and leaves the newer.
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool: Vec<Candidate> = vec![candidate(0, 0)];
        pool[0].load = Some(report(0, 0));
        pool[0].load_sampled_at_micros = 1_000;
        let kept: Vec<u32> = vec![0];
        for issued_micros in [20_000, 2_000] {
            let lease = Lease {
                issued_micros,
                tokens: 7,
                ..Lease::default()
            };
            ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        }
        let mut scores = vec![0.0f32; 1];
        let mut sampled = pool.clone();
        sampled[0].load_sampled_at_micros = 15_000;
        ll.score(&RequestFacts::default(), &sampled, &kept, &mut scores);
        assert_eq!(
            ll.held(pool[0].id),
            1,
            "the release through the older dispatch takes it from behind the later one"
        );
        assert_eq!(ll.held_word_at(pool[0].id, 0), (7, 1));
    }

    #[test]
    fn concurrent_dispatches_and_releases_keep_the_tally_exact() {
        // Eight threads, one per lane, each book 500 dispatches at rising
        // instants on their own rank and on one shared rank, and every
        // fifty score over a pool sampled at a midpoint that moves forward,
        // so releases run while other lanes add. A sample at the midpoint
        // then leaves the later half of every lane, exact in the lanes and
        // in the held word a decision reads, and a sample past every
        // dispatch releases them all.
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool: Vec<Candidate> = (0..9).map(|i| candidate(i, 0)).collect();
        for c in &mut pool {
            c.load = Some(report(0, 0));
            c.load_sampled_at_micros = 1_000;
        }
        let kept: Vec<u32> = (0..9).collect();
        let sampled_at = |pool: &[Candidate], at: u64| -> Vec<Candidate> {
            let mut sampled = pool.to_vec();
            for c in &mut sampled {
                c.load_sampled_at_micros = at;
            }
            sampled
        };
        std::thread::scope(|s| {
            for t in 0..8u32 {
                let (ll, pool, kept, sampled_at) = (&ll, &pool, &kept, &sampled_at);
                s.spawn(move || {
                    take_lane(t as usize);
                    let mut scores = vec![0.0f32; 9];
                    for n in 0..500u64 {
                        let lease = Lease {
                            issued_micros: 2_000 + n,
                            tokens: 7,
                            ..Lease::default()
                        };
                        ll.on_dispatched(&lease, &target_of(&pool[t as usize]), Side::Single, 0);
                        ll.on_dispatched(&lease, &target_of(&pool[8]), Side::Single, 0);
                        if n % 50 == 0 {
                            let midpoint = sampled_at(pool, 2_000 + n / 2);
                            ll.score(&RequestFacts::default(), &midpoint, kept, &mut scores);
                        }
                    }
                });
            }
        });
        let mut scores = vec![0.0f32; 9];
        let midpoint = sampled_at(&pool, 2_250);
        ll.score(&RequestFacts::default(), &midpoint, &kept, &mut scores);
        for c in &pool[..8] {
            assert_eq!(ll.held(c.id), 249, "{:?}", c.id);
            assert_eq!(ll.held_word_at(c.id, 0), (249 * 7, 249), "{:?}", c.id);
        }
        assert_eq!(ll.held(pool[8].id), 8 * 249, "the shared rank");
        assert_eq!(
            ll.held_word_at(pool[8].id, 0),
            (8 * 249 * 7, 8 * 249),
            "the shared rank's word"
        );
        let past = sampled_at(&pool, 3_000);
        ll.score(&RequestFacts::default(), &past, &kept, &mut scores);
        assert!(past.iter().all(|c| ll.held(c.id) == 0));
        assert!(past.iter().all(|c| ll.held_word_at(c.id, 0) == (0, 0)));
    }

    #[test]
    fn the_affinity_scorers_read_the_rows_facts() {
        let hashes = [1u64, 2, 3, 4];
        let facts = RequestFacts {
            block_hashes: &hashes,
            ..RequestFacts::default()
        };
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[1].session_hit = true;
        let mut scores = [0.0f32; 2];
        SessionAffinity.score(&facts, &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0]);
        assert!(SessionAffinity.needs().headers);
        assert!(LeastLoad::new(0.15, 1024, 2000.0).needs().loads);
    }
}
