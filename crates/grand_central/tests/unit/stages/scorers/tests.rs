thread_local! {
    /// What a case runs on its own thread between a retirement's two
    /// writes, behind the estimates' and the count's and ahead of the
    /// held word's: the row as a read finds it there.
    pub(crate) static BETWEEN_A_RETIREMENTS_WRITES: Cell<Option<fn(&RankCredit)>> =
        const { Cell::new(None) };
    /// What a case runs on its own thread ahead of a copy's read of a
    /// row's dispatches, with that row: a release that crosses the
    /// copy.
    pub(crate) static AHEAD_OF_A_COPYS_READ: Cell<Option<fn(&RankCredit)>> =
        const { Cell::new(None) };
    /// How many dispatches the case's thread landed ahead of their
    /// lane's end: each was searched a place among the dispatches held.
    pub(crate) static LANDED_AHEAD: Cell<u32> = const { Cell::new(0) };
}

use super::*;
use crate::{
    document::spec::{LeastCostParams, WarmupSliceParams},
    engine::{
        decision::{Pick, Targets},
        framework::{Picker, Shortlist},
        view::{Health, LoadRow},
    },
    policies::cache_aware::{Gates, WarmupSlice},
    stages::pickers::{target_of, Best},
    tracking::lease::{EventKind, TransportClass},
    Draw,
};

pub(crate) fn candidate(position: u32, in_flight: u32) -> Candidate {
    Candidate {
        position,
        id: WorkerId(1_000 + u64::from(position)),
        health: Health {
            healthy: true,
            ..Health::default()
        },
        in_flight,
        ..Candidate::default()
    }
}

pub(crate) fn kept(n: u32) -> Vec<u32> {
    (0..n).collect()
}

#[test]
fn a_report_age_limit_preserves_the_stages_methods() {
    #[derive(Debug, Default)]
    struct ProbeState {
        events: Mutex<Vec<EventKind>>,
    }

    impl DecisionScope for ProbeState {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}
        fn close(&self) {}
    }

    #[derive(Clone, Debug, Default)]
    struct Probe(Arc<ProbeState>);

    impl Lifecycle for Probe {
        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.0.events.lock().push(EventKind::Issued);
        }
        fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
            self.0.events.lock().push(EventKind::Dispatched);
        }
        fn on_completed(&self, _: &Lease, _: &Outcome) {
            self.0.events.lock().push(EventKind::Completed);
        }
        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.0.events.lock().push(EventKind::Released);
        }
        fn on_membership(&self, _: &Membership<'_>) {
            self.0.events.lock().push(EventKind::Membership);
        }
        fn on_signal(&self, _: &Signal<'_>) {
            self.0.events.lock().push(EventKind::Signal);
        }
        fn keeps_loads(&self) -> bool {
            true
        }
        fn needs_lifecycle(&self) -> bool {
            true
        }
    }

    impl Scorer for Probe {
        fn name(&self) -> &'static str {
            "probe"
        }
        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.score_with(facts, candidates, kept, out, |row| row.load.as_ref())
        }
        fn score_front(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            asked: &[u32],
            out: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            self.score_front_with(facts, candidates, kept, asked, out, |row| row.load.as_ref())
        }
        fn row_wise(&self) -> bool {
            true
        }
        fn front_wise(&self) -> bool {
            true
        }
        fn owns(&self) -> bool {
            true
        }
        fn own(
            &self,
            _: &RequestFacts<'_>,
            _: &PoolFacts<'_>,
            candidates: &[Candidate],
            keep: &mut dyn FnMut(&Candidate) -> bool,
            kept: &mut Vec<u32>,
        ) -> bool {
            kept.extend(
                candidates
                    .iter()
                    .filter(|row| keep(row))
                    .map(|row| row.position),
            );
            !kept.is_empty()
        }
        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Levels(1.0)
        }
        fn needs(&self) -> Needs {
            Needs {
                loads: true,
                ..Needs::default()
            }
        }
        fn balances_load_against_affinity(&self) -> bool {
            true
        }
        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self.0.as_ref())
        }
        fn refuses_by_reports(&self) -> bool {
            false
        }
        fn warm_up_rule(&self) -> Option<WarmUpRule> {
            WarmupSlice::new(&WarmupSliceParams::default(), Gates::default()).warm_up_rule()
        }
        fn states_overlap(&self) -> bool {
            true
        }
        fn said(&self, _: &RequestFacts<'_>, _: &[Candidate], row: u32) -> Overlap {
            Overlap {
                blocks: 3,
                of: 5 + row as u16,
                ..Overlap::default()
            }
        }
    }

    impl ReadsLoad for Probe {
        fn score_with(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
            load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
        ) -> TieBand {
            for (slot, &index) in out.iter_mut().zip(kept) {
                *slot = if load(&candidates[index as usize]).is_some() {
                    1.0
                } else {
                    0.0
                };
            }
            TieBand(0.0625)
        }
        fn score_front_with(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            _: &[u32],
            asked: &[u32],
            out: &mut [f32],
            load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
        ) -> (TieBand, ScoreFront) {
            let band = self.score_with(facts, candidates, asked, out, load);
            (band, ScoreFront::of(self.spread(), out))
        }
    }

    let bare = Probe::default();
    let wrapped = with_load_report_age_limit(bare.clone(), Some(37));
    let facts = RequestFacts {
        now_micros: 71_337,
        ..RequestFacts::default()
    };
    let mut rows = [candidate(0, 0), candidate(1, 0)];
    rows[1].load = Some(LoadRow::default());
    rows[1].load_sampled_at_micros = facts.now_micros;
    let score = |stage: &dyn Scorer| {
        let mut out = [0.0; 2];
        let band = stage.score(&facts, &rows, &[0, 1], &mut out);
        (band, out)
    };
    let score_front = |stage: &dyn Scorer| {
        let mut out = [0.0];
        let front = stage.score_front(&facts, &rows, &[0, 1], &[1], &mut out);
        (front, out)
    };
    let own = |stage: &dyn Scorer| {
        let mut kept = Vec::new();
        let owns = stage.own(
            &facts,
            &PoolFacts::default(),
            &rows,
            &mut |row| row.load.is_some(),
            &mut kept,
        );
        (owns, kept)
    };
    assert_eq!(wrapped.name(), bare.name());
    assert_eq!(score(wrapped.as_ref()), score(&bare));
    assert_eq!(score_front(wrapped.as_ref()), score_front(&bare));
    assert_eq!(wrapped.row_wise(), bare.row_wise());
    assert_eq!(wrapped.front_wise(), bare.front_wise());
    assert_eq!(wrapped.owns(), bare.owns());
    assert_eq!(own(wrapped.as_ref()), own(&bare));
    assert_eq!(wrapped.spread(), bare.spread());
    assert_eq!(wrapped.needs(), bare.needs());
    assert!(bare.balances_load_against_affinity());
    assert!(wrapped.balances_load_against_affinity());
    assert!(std::ptr::addr_eq(
        wrapped.scope().expect("the scope"),
        bare.scope().expect("the scope")
    ));
    assert_eq!(wrapped.refuses_by_reports(), bare.refuses_by_reports());
    assert_eq!(wrapped.warm_up_rule(), bare.warm_up_rule());
    assert_eq!(wrapped.states_overlap(), bare.states_overlap());
    assert_eq!(wrapped.said(&facts, &rows, 1), bare.said(&facts, &rows, 1));
    assert_eq!(wrapped.keeps_loads(), bare.keeps_loads());
    assert_eq!(wrapped.needs_lifecycle(), bare.needs_lifecycle());

    let tell = |stage: &dyn Scorer| {
        let lease = Lease::default();
        let target = target_of(&rows[1]);
        stage.on_decision_issued(&lease, &Decision::Pick(Pick::default()));
        stage.on_dispatched(&lease, &target, Side::Single, 23);
        stage.on_completed(
            &lease,
            &Outcome {
                side: Side::Single,
                served: target.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
        );
        stage.on_released(&lease, ReleaseReason::Abandoned);
        stage.on_membership(&Membership::Added(target.id));
        stage.on_signal(&Signal::Health {
            worker: target.id,
            health: Health::default(),
        });
        std::mem::take(&mut *bare.0.events.lock())
    };
    assert_eq!(tell(wrapped.as_ref()), tell(&bare));
}

const COST_TERMS: [(&str, f64); 7] = [
    ("candidate.in_flight", 9.9e14),
    ("load.active_kv_permille", -1.25),
    ("load.kv_usage_permille", 0.75),
    ("load.queued", -9.9e14),
    ("load.resident_tokens", 0.5),
    ("load.running", -0.25),
    ("load.waiting_uncached_tokens", 2.75),
];

fn cost_by_term(terms: &[(CostTerm, f64)], row: &Candidate, price: Option<f64>) -> RowCost {
    let mut sum = 0.0;
    for &(term, coefficient) in terms {
        if let Some(value) = term.read(row.load.as_ref(), row.in_flight) {
            sum += value * coefficient;
        } else if let Some(price) = price {
            return RowCost::OverLeast(f64::from(row.in_flight) * price);
        }
    }
    RowCost::Sum(sum)
}

fn cost_bits(cost: RowCost) -> (bool, u64) {
    (
        matches!(cost, RowCost::OverLeast(_)),
        cost.value().to_bits(),
    )
}

#[test]
fn field_sums_match_the_terms_read_in_name_order() {
    let mut draw = Draw::new(17);
    let mut rows: Vec<_> = (0u32..64)
        .map(|position| {
            let mut row = candidate(position, draw.next_u64() as u32);
            row.load = (!position.is_multiple_of(5)).then(|| LoadRow {
                running: draw.next_u64() as u32,
                queued: draw.next_u64() as u32,
                waiting_uncached_tokens: if position.is_multiple_of(2) {
                    0
                } else {
                    draw.next_u64() as u32
                },
                kv_usage_permille: draw.below(1_001) as u16,
                active_kv_permille: draw.below(1_001) as u16,
                resident_tokens: draw.next_u64() as u32,
                resident_counted: position.is_multiple_of(3),
                ..LoadRow::default()
            });
            row
        })
        .collect();
    // Cancellation makes the smaller terms sensitive to where they were
    // added. An empty report supplies zero waiting tokens.
    rows[0].in_flight = u32::MAX;
    rows[0].load = Some(LoadRow {
        queued: u32::MAX,
        running: 7,
        ..LoadRow::default()
    });
    rows[5] = candidate(5, 0);
    rows[5].load = Some(LoadRow::default());
    let facts = RequestFacts::default();
    let kept = kept(rows.len() as u32);
    let mut costs = Vec::new();
    for mask in 0..1 << COST_TERMS.len() {
        let terms: Box<_> = COST_TERMS
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, &(name, coefficient))| (CostTerm::named(name).unwrap(), coefficient))
            .collect();
        for price in [None, Some(1.5), Some(-0.0)] {
            let rule = price.map_or(Unreported::MissingAsZero, |request_cost| {
                Unreported::OverLeast { request_cost }
            });
            let stage = LeastCost::new(terms.clone(), CostMargin::default(), Some(rule));
            costs.clear();
            stage.collect_costs(&facts, &rows, &kept, &mut costs, |row| row.load.as_ref());
            assert_eq!(costs.len(), rows.len());
            for (row, &cost) in rows.iter().zip(&costs) {
                assert_eq!(
                    cost_bits(cost),
                    cost_bits(cost_by_term(&terms, row, price)),
                    "terms {mask:07b}, {rule:?}, row {}",
                    row.position
                );
            }
        }
    }
}

#[test]
fn a_document_with_one_term_costs_its_field_times_the_coefficient() {
    let mut row = candidate(0, 17);
    row.load = Some(LoadRow {
        active_kv_permille: 19,
        kv_usage_permille: 23,
        queued: 29,
        resident_tokens: 31,
        running: 37,
        waiting_uncached_tokens: 41,
        resident_counted: true,
        ..LoadRow::default()
    });
    let zero = Candidate {
        load: Some(LoadRow {
            resident_counted: true,
            ..LoadRow::default()
        }),
        ..candidate(1, 0)
    };
    let facts = RequestFacts::default();
    let mut costs = Vec::new();
    for ((name, _), value) in COST_TERMS.into_iter().zip([17, 19, 23, 29, 31, 37, 41]) {
        let params: LeastCostParams = serde_yaml::from_str(&format!(
            "terms: {{{name}: -2.5}}\nunreported: missing-as-zero"
        ))
        .unwrap();
        let terms = params
            .terms
            .iter()
            .map(|(name, &coefficient)| (CostTerm::named(name).unwrap(), coefficient))
            .collect();
        let stage = LeastCost::new(terms, params.margin, params.unreported);
        let want = f64::from(value) * -2.5;
        costs.clear();
        stage.collect_costs(
            &facts,
            &[row, zero, candidate(2, 0)],
            &[0, 1, 2],
            &mut costs,
            |row| row.load.as_ref(),
        );
        assert_eq!(cost_bits(costs[0]), (false, want.to_bits()), "{name}");
        assert_eq!(cost_bits(costs[1]), (false, 0.0f64.to_bits()), "{name}");
        assert_eq!(cost_bits(costs[2]), (false, 0.0f64.to_bits()), "{name}");
    }
}

#[test]
fn a_row_with_no_report_is_priced_by_its_count_over_least() {
    let stage = LeastCost::new(
        Box::new([
            (CostTerm::InFlight, 2.0),
            (CostTerm::named("load.queued").unwrap(), 3.0),
        ]),
        CostMargin::default(),
        Some(Unreported::OverLeast { request_cost: 4.0 }),
    );
    let missing = stage.cost_of::<false, false, false>(None, 13);
    assert_eq!(cost_bits(missing), (true, 52.0f64.to_bits()));
}

/// The costs find the same ends by either way: a few rows are compared
/// one by one, many are folded, and a cost that is no number moves
/// neither end. So the same costs score the same bits among a few
/// rows and among many.
#[test]
fn a_few_costs_and_many_find_the_same_ends() {
    // The cost that is no number stands last: no cost behind it could
    // take an end back from it.
    let few = [4.0, 2.0, 3.0, f32::NAN];
    let mut some = few;
    let span = normalize_costs(&mut some);
    assert_eq!(span, 2.0);
    assert_eq!(some[0].to_bits(), 0.0f32.to_bits());
    assert_eq!((some[1], some[2]), (1.0, 0.5));
    assert!(some[3].is_nan());
    let mut many: Vec<f32> = few.iter().copied().chain([3.0; FEW_COSTS]).collect();
    assert!(many.len() > FEW_COSTS);
    assert_eq!(normalize_costs(&mut many), span);
    let bits = |scores: &[f32]| scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&many[..few.len()]), bits(&some));
}

/// The engine's order around a picker: the set it compares, that set's
/// scores from `scores_of`, the pick, then the survivors it passed over
/// as alternates. The positions of the targets come back, in order.
pub(crate) fn picked(
    picker: &dyn Picker,
    facts: &RequestFacts<'_>,
    pool: &[Candidate],
    scores_of: &dyn Fn(&[u32]) -> Vec<f32>,
) -> Vec<u32> {
    picked_within(picker, facts, pool, scores_of, TieBand::NORMALIZED)
}

/// [`picked`] under a tie band of the caller's.
pub(crate) fn picked_within(
    picker: &dyn Picker,
    facts: &RequestFacts<'_>,
    pool: &[Candidate],
    scores_of: &dyn Fn(&[u32]) -> Vec<f32>,
    band: TieBand,
) -> Vec<u32> {
    let mut rows = kept(pool.len() as u32);
    let Shortlist {
        compared,
        alternates,
    } = picker.narrow(facts, pool, &mut rows);
    let scores = scores_of(&rows[..compared]);
    let mut out = Targets::new();
    picker.pick(facts, pool, &rows[..compared], &scores, band, &mut out);
    for &row in &rows[compared..compared + alternates] {
        out.push(target_of(&pool[row as usize]));
    }
    out.iter().map(|t| t.position).collect()
}

#[test]
fn least_requests_ranks_by_in_flight_and_best_orders_alternates() {
    let pool = [candidate(0, 4), candidate(1, 0), candidate(2, 2)];
    let mut scores = [0.0f32; 3];
    let band = LeastRequests.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
    assert_eq!(scores, [0.0, 1.0, 0.5]);
    assert_eq!(band, TieBand::NORMALIZED);
    let mut out = Targets::new();
    Best.pick(
        &RequestFacts::default(),
        &pool,
        &kept(3),
        &scores,
        band,
        &mut out,
    );
    let order: Vec<u32> = out.iter().map(|t| t.position).collect();
    assert_eq!(order, vec![1, 2, 0]);
}

pub(crate) fn report(running: u32, waiting_uncached_tokens: u32) -> LoadRow {
    LoadRow {
        running,
        waiting_uncached_tokens,
        live_rate: 2000,
        ..LoadRow::default()
    }
}

/// A report that counts resident tokens.
fn resident(running: u32, resident_tokens: u32) -> LoadRow {
    LoadRow {
        running,
        resident_tokens,
        resident_counted: true,
        ..LoadRow::default()
    }
}

/// A report of an engine that does not count them.
fn uncounted(running: u32) -> LoadRow {
    LoadRow {
        running,
        ..LoadRow::default()
    }
}

fn resident_scores<const N: usize>(scorer: &ResidentWork, pool: &[Candidate; N]) -> [f32; N] {
    let mut scores = [0.0f32; N];
    scorer.score(&RequestFacts::default(), pool, &kept(N as u32), &mut scores);
    scores
}

/// The request's price is part of the comparison: a worker holding fewer
/// tokens over more requests loses once a request is priced high enough.
#[test]
fn resident_work_is_the_resident_tokens_plus_the_request_price_of_every_request_held() {
    let mut pool = [candidate(0, 2), candidate(1, 1)];
    pool[0].load = Some(resident(2, 6_000));
    pool[1].load = Some(resident(1, 9_000));
    let cheap = ResidentWork::new(1_000);
    assert_eq!(
        cheap.reported_cost(pool[0].in_flight, &resident(2, 6_000), 0, 0),
        8_000.0
    );
    assert_eq!(
        cheap.reported_cost(pool[1].in_flight, &resident(1, 9_000), 0, 0),
        10_000.0
    );
    assert_eq!(resident_scores(&cheap, &pool), [1.0, 0.0]);
    let dear = ResidentWork::new(4_000);
    assert_eq!(
        resident_scores(&dear, &pool),
        [0.0, 1.0],
        "14,000 against 13,000"
    );
    // A queued request is held as a running one is.
    let queued = LoadRow {
        queued: 1,
        ..resident(1, 6_000)
    };
    assert_eq!(
        cheap.reported_cost(pool[0].in_flight, &queued, 0, 0),
        8_000.0
    );
}

/// A dispatch the report does not reflect yet adds its prompt and one
/// request's price; a report sampled after it releases both.
#[test]
fn resident_work_credits_a_dispatch_until_a_report_reflects_it() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 0), candidate(1, 0)];
    pool[0].load = Some(resident(1, 2_000));
    pool[0].load_sampled_at_micros = 1_000;
    pool[1].load = Some(resident(1, 4_000));
    pool[1].load_sampled_at_micros = 1_000;
    assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0]);

    let lease = Lease {
        issued_micros: 2_000,
        tokens: 1_500,
        ..Lease::default()
    };
    rw.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
    assert_eq!(rw.held_at(pool[0].id, 0), 1);
    assert_eq!(
        resident_scores(&rw, &pool),
        [0.0, 1.0],
        "2,000 + 1,500 + 1,000 for each of two requests, against 5,000"
    );

    pool[0].load_sampled_at_micros = 3_000;
    assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0]);
    assert_eq!(rw.held_at(pool[0].id, 0), 0);

    // A removed worker leaves no credit behind.
    rw.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
    rw.on_membership(&Membership::Removed(pool[1].id));
    assert_eq!(rw.held_at(pool[1].id, 0), 0);
}

/// The resident-work stage credits a dispatch of no stated size at its
/// mean prefill, the crate's default: credited at no tokens, the worker
/// would still read cheaper than its peer.
#[test]
fn resident_work_credits_a_dispatch_of_no_stated_size_at_its_mean_prefill() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 0), candidate(1, 0)];
    pool[0].load = Some(resident(1, 2_000));
    pool[0].load_sampled_at_micros = 1_000;
    pool[1].load = Some(resident(1, 4_000));
    pool[1].load_sampled_at_micros = 1_000;
    assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0]);

    let unstated = Lease {
        issued_micros: 2_000,
        ..Lease::default()
    };
    rw.on_dispatched(&unstated, &target_of(&pool[0]), Side::Single, 0);
    assert_eq!(
        resident_scores(&rw, &pool),
        [0.0, 1.0],
        "2,000 + the crate's default 1,024 + 1,000 for each of two requests, against 5,000"
    );
}

/// A dispatch of text alone states no size to the resident-work stage
/// either: it is credited at the stage's mean prefill, and nothing is
/// read of its text's bytes.
#[test]
fn resident_work_credits_a_dispatch_of_text_alone_at_its_mean_prefill() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 0), candidate(1, 0)];
    pool[0].load = Some(resident(1, 2_000));
    pool[0].load_sampled_at_micros = 1_000;
    pool[1].load = Some(resident(1, 4_000));
    pool[1].load_sampled_at_micros = 1_000;
    // 40 bytes of text: 10 tokens by the bytes over four.
    let text = Lease {
        issued_micros: 2_000,
        bytes: 40,
        ..Lease::default()
    };
    rw.on_dispatched(&text, &target_of(&pool[0]), Side::Single, 0);
    assert_eq!(
        resident_scores(&rw, &pool),
        [0.0, 1.0],
        "2,000 + the mean's 1,024 + 1,000 for each of two requests, against 5,000"
    );
}

/// A report releases the dispatches it reflects whether or not the score
/// reads their credit. A rank whose report counts no resident tokens
/// scores from its peers, and holds a dispatch only until a report
/// sampled after it.
#[test]
fn a_report_that_counts_no_resident_tokens_still_releases_its_dispatches() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 1), candidate(1, 1)];
    pool[0].load = Some(uncounted(1));
    pool[0].load_sampled_at_micros = 1_000;
    pool[1].load = Some(resident(1, 4_000));
    pool[1].load_sampled_at_micros = 1_000;
    let lease = Lease {
        issued_micros: 2_000,
        tokens: 1_500,
        ..Lease::default()
    };
    rw.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
    let before = resident_scores(&rw, &pool);
    assert_eq!(
        rw.held_at(pool[0].id, 0),
        1,
        "the report predates the dispatch"
    );

    pool[0].load_sampled_at_micros = 3_000;
    assert_eq!(
        resident_scores(&rw, &pool),
        before,
        "the score reads no credit of that rank"
    );
    assert_eq!(rw.held_at(pool[0].id, 0), 0);
}

/// A lane's dispatches are held by instant and let go through one: one
/// dated earlier lands in front, one made at a held instant behind it,
/// and the oldest goes where it was made at or before the instant.
#[test]
fn a_lanes_dispatches_are_held_by_instant_and_let_go_through_one() {
    let mut dispatches = Dispatches::default();
    assert_eq!(dispatches.oldest(), None);
    // The first held is the oldest. A later one is not.
    assert!(dispatches.hold(20, SendSize::stated(2)));
    assert!(!dispatches.hold(30, SendSize::stated(3)));
    // One dated earlier is the oldest now.
    assert!(dispatches.hold(10, SendSize::stated(1)));
    // One dated between lands between, one at a held instant behind it.
    assert!(!dispatches.hold(25, SendSize::stated(4)));
    assert!(!dispatches.hold(20, SendSize::stated(5)));
    assert_eq!((dispatches.oldest(), dispatches.len()), (Some(10), 5));
    // The oldest stays while the instant is before it and goes at its
    // own instant. The next stays one microsecond short of its own.
    assert_eq!(dispatches.let_go(9), None);
    assert_eq!(dispatches.let_go(10), Some(SendSize::stated(1)));
    assert_eq!(dispatches.let_go(19), None);
    assert_eq!(dispatches.oldest(), Some(20));
    // Oldest first, and of one instant the earlier held first.
    let gone: Vec<u32> = std::iter::from_fn(|| dispatches.let_go(u64::MAX))
        .map(SendSize::tokens)
        .collect();
    assert_eq!(gone, [2, 5, 4, 3]);
    assert_eq!((dispatches.oldest(), dispatches.len()), (None, 0));
}

/// A credit is its rank's: a dispatch to rank 1 of a worker prices that
/// rank's row and leaves rank 0's alone.
#[test]
fn a_dispatch_is_credited_to_the_rank_it_went_to() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 0), candidate(1, 0)];
    pool[1].id = pool[0].id;
    pool[1].dp_rank = 1;
    for c in &mut pool {
        c.load = Some(resident(1, 2_000));
        c.load_sampled_at_micros = 1_000;
    }
    assert_eq!(resident_scores(&rw, &pool), [1.0, 1.0]);

    let lease = Lease {
        issued_micros: 2_000,
        tokens: 1_500,
        ..Lease::default()
    };
    rw.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
    assert_eq!(
        (rw.held_at(pool[0].id, 0), rw.held_at(pool[0].id, 1)),
        (0, 1)
    );
    assert_eq!(
        resident_scores(&rw, &pool),
        [1.0, 0.0],
        "3,000 on rank 0 against 2,000 + 1,500 + 1,000 for each of two requests on rank 1"
    );
}

/// Three ranks whose walks start at one slot are each found, and the two
/// behind are still found once the first is dropped: a walk stops at a
/// free slot, so the table is laid again and no slot is freed in place.
#[test]
fn a_rank_is_found_behind_one_the_table_dropped() {
    let at = |id: u64| RankTable::start(&(WorkerId(id), 0)) & 7;
    let alike: Vec<RankKey> = (1..)
        .filter(|&id| at(id) == at(1))
        .take(3)
        .map(|id| (WorkerId(id), 0))
        .collect();
    let mut table = RankTable::default();
    for &rank in &alike {
        table.insert(rank, Arc::new(RankCredit::default()));
    }
    assert!(alike.iter().all(|rank| table.contains_key(rank)));
    let table = table.retained(|rank, _| *rank != alike[0]);
    assert!(!table.contains_key(&alike[0]));
    assert!(table.contains_key(&alike[1]) && table.contains_key(&alike[2]));
    assert_eq!(table.iter().count(), 2);
}

/// The mean context is the resident tokens over the requests the report
/// holds, the queued with the running: 8,000 tokens over two running and
/// two queued price the one request in transit at 2,000.
#[test]
fn the_mean_context_counts_the_queued_requests_with_the_running() {
    let rw = ResidentWork::new(1_000);
    let busy = candidate(0, 5);
    let row = LoadRow {
        queued: 2,
        ..resident(2, 8_000)
    };
    assert_eq!(
        rw.reported_cost(busy.in_flight, &row, 0, 0),
        8_000.0 + 2_000.0 + 5.0 * 1_000.0
    );
}

/// The band is the scorer's own, on its score in tokens, and not the
/// plain band of a normalized score: with one worker two million tokens
/// behind, two workers a token apart are still not a tie.
#[test]
fn a_token_apart_is_not_a_tie_under_a_wide_spread() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
    pool[0].load = Some(resident(1, 5_000));
    pool[1].load = Some(resident(1, 5_001));
    pool[2].load = Some(resident(1, 2_005_000));
    let mut scores = [0.0f32; 3];
    let band = rw.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
    assert!(scores[0] - scores[1] > band.0, "{scores:?} {band:?}");
    for seed in 0..300u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut out = Targets::new();
        Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
        assert_eq!(out.first().map(|t| t.position), Some(0), "seed {seed}");
    }
}

/// The requests the host holds beyond the report and the credit are in
/// transit: each is priced at the worker's mean context and at the
/// request price; a worker that reports nothing held prices them at the
/// mean prefill.
#[test]
fn resident_work_prices_the_requests_in_transit_at_the_workers_mean_context() {
    let rw = ResidentWork::new(1_000);
    let busy = candidate(0, 5);
    // Two reported requests hold 8,000 tokens, a mean context of 4,000;
    // three more are in transit.
    assert_eq!(
        rw.reported_cost(busy.in_flight, &resident(2, 8_000), 0, 0),
        8_000.0 + 3.0 * 4_000.0 + 5.0 * 1_000.0
    );
    // One of the three is a credited dispatch of 1,500 tokens.
    assert_eq!(
        rw.reported_cost(busy.in_flight, &resident(2, 8_000), 1_500, 1),
        8_000.0 + 1_500.0 + 2.0 * 4_000.0 + 5.0 * 1_000.0
    );
    let idle = candidate(1, 2);
    assert_eq!(
        rw.reported_cost(idle.in_flight, &resident(0, 0), 0, 0),
        2.0 * DEFAULT_MEAN_PREFILL_TOKENS as f32 + 2.0 * 1_000.0
    );
}

/// A worker whose report does not count resident tokens, or that has no
/// report, starts from the best reporting peer and adds its in-flight
/// requests, each at the mean prefill and the request price.
#[test]
fn a_worker_without_resident_tokens_scores_from_the_best_reporting_peer() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 1), candidate(1, 2), candidate(2, 0)];
    pool[0].load = Some(resident(1, 5_000));
    pool[1].load = Some(uncounted(2));
    // 6,000; 6,000 + 2 * 2,024; and the unreported idle worker at 6,000.
    assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0, 1.0]);
}

/// The peer it starts from is the best one, not the worst, and each of
/// its requests in flight costs the mean prefill and the request price:
/// between a peer at 6,000 and a peer at 14,096, two requests in flight
/// put it at 6,000 + 2 * 2,024, the exact middle.
#[test]
fn a_worker_without_resident_tokens_is_priced_between_two_reporting_peers() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 1), candidate(1, 2), candidate(2, 1)];
    pool[0].load = Some(resident(1, 5_000));
    pool[1].load = Some(uncounted(2));
    pool[2].load = Some(resident(1, 13_096));
    assert_eq!(resident_scores(&rw, &pool), [1.0, 0.5, 0.0]);
}

#[test]
fn with_no_survivor_reporting_resident_tokens_the_in_flight_count_decides() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 3), candidate(1, 1), candidate(2, 2)];
    pool[0].load = Some(uncounted(3));
    assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0, 0.5]);
}

/// In a fleet whose engines do not count resident tokens, a report that
/// reads idle is no reporting peer: the in-flight count decides, and the
/// worker the host holds thirty requests on does not take the next one.
#[test]
fn an_idle_report_without_the_count_makes_no_reporting_peer() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 30), candidate(1, 1)];
    pool[0].load = Some(uncounted(0));
    pool[1].load = Some(uncounted(1));
    assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0]);
}

/// A count of zero and no count are two reports. The idle worker of an
/// engine that counts resident tokens scores least, and so does one
/// whose running request has nothing resident yet; the idle worker of an
/// engine that does not count them ties with the best reporting peer.
#[test]
fn a_count_of_zero_is_not_a_report_without_the_count() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 1), candidate(1, 0)];
    pool[0].load = Some(resident(1, 5_000));
    pool[1].load = Some(resident(0, 0));
    assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0]);
    pool[1].load = Some(uncounted(0));
    assert_eq!(resident_scores(&rw, &pool), [1.0, 1.0]);
    pool[1] = candidate(1, 1);
    pool[1].load = Some(resident(1, 0));
    assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0]);
}

/// Equal work is one tie, and a token apart is not: the band is the
/// least-load scorer's, on a score in tokens.
#[test]
fn equal_resident_work_ties_and_a_token_apart_does_not() {
    let rw = ResidentWork::new(1_000);
    let mut pool = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
    pool[0].load = Some(resident(1, 5_000));
    pool[1].load = Some(resident(1, 5_000));
    pool[2].load = Some(resident(1, 5_001));
    let mut scores = [0.0f32; 3];
    let band = rw.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
    assert_eq!(scores, [1.0, 1.0, 0.0]);
    assert!(band.0 < 1.0, "{band:?}");
    let mut picks = [0usize; 3];
    for seed in 0..300u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut out = Targets::new();
        Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
        picks[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
    }
    assert!(picks[0] > 100 && picks[1] > 100, "{picks:?}");
    assert_eq!(picks[2], 0, "{picks:?}");
}

#[test]
fn a_read_across_a_release_counts_no_more_unsized_dispatches_than_dispatches() {
    // One thread releases a row that holds dispatches of no tokens
    // alone, and this one reads the row's two words as they stand. The
    // count of those dispatches goes down ahead of the held word at
    // each one, so no read sees more of them than dispatches. A round
    // whose reads all land beside the release shows nothing: the
    // rounds go on until a read lands inside one, a few at most.
    const HELD: u64 = 6_000;
    for _ in 0..32 {
        let row = RankCredit::default();
        for at in 1..=HELD {
            row.record(at, SendSize::stated(0));
        }
        let crossed = std::thread::scope(|s| {
            let release = s.spawn(|| row.sent_through(HELD));
            let mut crossed = false;
            while !release.is_finished() {
                // An instant ahead of every dispatch releases none.
                let read = row.sent_through(0);
                assert!(
                    read.unsized_requests <= read.requests,
                    "{} dispatches of no tokens beside {} dispatches",
                    read.unsized_requests,
                    read.requests
                );
                crossed |= read.requests != 0 && u64::from(read.requests) != HELD;
            }
            crossed
        });
        if crossed {
            return;
        }
    }
}

/// The estimates' word goes down ahead of the held word at each
/// dispatch a release lets go, so a read of the documents' two numbers
/// that crosses the release adds no estimate of a dispatch the held
/// word has let go. Every dispatch here is an estimate of three
/// tokens: no read finds more than three tokens a dispatch. The rounds
/// go on until a read lands inside a release, a few at most.
#[test]
fn a_read_across_a_release_adds_no_estimate_of_a_dispatch_let_go() {
    const HELD: u64 = 6_000;
    const EACH: u32 = 3;
    for _ in 0..32 {
        let row = RankCredit::default();
        for at in 1..=HELD {
            row.record(at, SendSize::estimated(EACH));
        }
        let crossed = std::thread::scope(|s| {
            let release = s.spawn(|| row.counts_through(HELD));
            let mut crossed = false;
            while !release.is_finished() {
                // An instant ahead of every dispatch releases none.
                let read = row.counts_through(0);
                assert!(
                    read.tokens <= u64::from(EACH) * u64::from(read.requests),
                    "{} estimated tokens beside {} dispatches",
                    read.tokens,
                    read.requests
                );
                crossed |= read.requests != 0 && u64::from(read.requests) != HELD;
            }
            crossed
        });
        if crossed {
            return;
        }
    }
}

/// The estimates' word goes down ahead of the held word at a
/// retirement too. Every dispatch here is an estimate of three tokens,
/// and the dispatch past the row's bound lets the row's oldest go. A
/// read of the documents' two numbers between the retirement's two
/// writes finds the held word still counting that one and no more than
/// three tokens a dispatch: it adds no estimate of a dispatch the held
/// word has let go.
#[test]
fn a_read_between_a_retirements_writes_adds_no_estimate_of_a_dispatch_let_go() {
    const EACH: u32 = 3;
    thread_local!(static READS: Cell<u32> = const { Cell::new(0) });
    fn read(row: &RankCredit) {
        // An instant ahead of every dispatch releases none.
        let read = row.counts_through(0);
        assert!(
            read.tokens <= u64::from(EACH) * u64::from(read.requests),
            "{} estimated tokens beside {} dispatches",
            read.tokens,
            read.requests
        );
        READS.set(READS.get() + 1);
    }
    let row = RankCredit::default();
    for at in 1..=RANK_CREDIT_KEPT {
        row.record(at, SendSize::estimated(EACH));
    }
    BETWEEN_A_RETIREMENTS_WRITES.set(Some(read));
    row.record(RANK_CREDIT_KEPT + 1, SendSize::estimated(EACH));
    BETWEEN_A_RETIREMENTS_WRITES.set(None);
    assert_eq!(READS.get(), 1, "one retirement, read between its writes");
    let kept = RANK_CREDIT_KEPT;
    let words = (0, kept, kept as u16, u64::from(EACH) * kept);
    assert_eq!(row.words_for_test(), words);
}

/// A copy reads every standing row of a rank under that row's locks
/// and takes the longest read. Two tallies hold one dispatch at a
/// rank, and the first lets its own go just ahead of the copy's read
/// of it, as a decision's release does: the copy holds the dispatch
/// the second still holds. A row chosen by the two counts, read with
/// no lock, would be the first, read empty.
#[test]
fn a_copy_holds_the_longest_read_where_a_row_lets_go_ahead_of_its_read() {
    thread_local!(static READS: Cell<u32> = const { Cell::new(0) });
    fn let_the_first_go(row: &RankCredit) {
        let reads = READS.get();
        READS.set(reads + 1);
        if reads == 0 {
            row.release_through(u64::MAX);
        }
    }
    let worker = WorkerId(7);
    let standing = [Credits::default(), Credits::default()];
    for tally in &standing {
        let row = tally.entry((worker, 0)).expect("the rank's entry");
        row.record(91_300, SendSize::stated(317));
    }
    let copy = Credits::default();
    AHEAD_OF_A_COPYS_READ.set(Some(let_the_first_go));
    copy.copy_fullest(standing.iter());
    AHEAD_OF_A_COPYS_READ.set(None);
    assert_eq!(copy.held(worker), 1, "the dispatch the second holds");
    let held: Vec<_> = standing.iter().map(|tally| tally.held(worker)).collect();
    assert_eq!(held, [0, 1], "the first let its own go");
    assert_eq!(READS.get(), 2, "each standing row read once");
}

/// A copy enters a row's dispatches oldest first, so each lands at its
/// lane's end and none is searched a place. The standing row holds its
/// newer dispatches in an earlier lane than its older ones, so a read
/// of its lanes hands the newer first. Entered as read, each of the
/// two older ones would land ahead of the lane's end.
#[test]
fn a_copy_lands_no_dispatch_ahead_of_its_lanes_end() {
    let worker = WorkerId(7);
    let standing = Credits::default();
    let row = standing.entry((worker, 0)).expect("the rank's entry");
    // The first lane takes the three newer dispatches, the second the
    // two older ones.
    take_lane(0);
    for at in [91_310, 91_320, 91_330] {
        row.record(at, SendSize::stated(317));
    }
    take_lane(1);
    for at in [91_300, 91_305] {
        row.record(at, SendSize::stated(317));
    }
    let read: Vec<u64> = row.held_sends().iter().map(|&(at, _)| at).collect();
    assert_eq!(read, [91_310, 91_320, 91_330, 91_300, 91_305]);
    let copy = Credits::default();
    LANDED_AHEAD.set(0);
    copy.copy_fullest(std::iter::once(&standing));
    assert_eq!(LANDED_AHEAD.get(), 0, "each at its lane's end");
    assert_eq!(copy.held(worker), 5, "every dispatch");
}

/// A dispatch that states no tokens states no size: it is held as an
/// estimate of none, among the sends of no stated size.
#[test]
fn a_stated_size_of_no_tokens_is_held_as_a_send_of_no_stated_size() {
    assert_eq!(SendSize::stated(0), SendSize::estimated(0));
    let row = RankCredit::default();
    row.record(1, SendSize::stated(0));
    assert_eq!(row.words_for_test(), (0, 1, 1, 0));
}

/// A size is held under the mark of an estimate, stated or estimated:
/// one at the mark's bit and one past it are held as the largest
/// under it. A stated one stays a stated one. It is not read as an
/// estimate, and its release takes it out of the held word and
/// leaves the estimates' word as it stands.
#[test]
fn a_size_at_the_marks_bit_is_held_under_it_and_in_its_own_word() {
    const MOST: u64 = (1 << 31) - 1;
    for size in [1u32 << 31, u32::MAX] {
        let row = RankCredit::default();
        row.record(1, SendSize::stated(size));
        assert_eq!(row.words_for_test(), (MOST, 1, 0, 0), "stated {size}");
        row.record(2, SendSize::estimated(size));
        assert_eq!(row.words_for_test(), (MOST, 2, 1, MOST), "{size}");
        // A report sampled at the first lets the stated one go alone.
        let read = row.counts_through(1);
        assert_eq!((read.requests, read.tokens), (1, MOST), "{size}");
        assert_eq!(row.words_for_test(), (0, 1, 1, MOST), "{size}");
        row.counts_through(2);
        assert_eq!(row.words_for_test(), (0, 0, 0, 0), "{size}");
    }
}

/// A lane that holds a stated size, an estimate and a stated size
/// reads back to nothing in each of the row's words, by a release and
/// by a retirement: an estimate goes with its dispatch. The
/// documents' read adds the estimate to the stated token-work, and
/// the read of a stage that prices such a send itself counts it and
/// adds none.
#[test]
fn a_mixed_lane_reads_back_to_nothing_by_a_release_and_by_a_retirement() {
    let mixed = [
        SendSize::stated(7),
        SendSize::estimated(101),
        SendSize::stated(5),
    ];
    let row = RankCredit::default();
    for (at, size) in (1..).zip(mixed) {
        row.record(at, size);
    }
    assert_eq!(row.words_for_test(), (12, 3, 1, 101));
    // An instant ahead of every dispatch releases none.
    let read = row.counts_through(0);
    assert_eq!((read.requests, read.tokens), (3, 113));
    let sent = row.sent_through(0);
    assert_eq!((sent.requests, sent.tokens), (3, 12));
    assert_eq!(sent.unsized_requests, 1);
    // A release, one dispatch at a time.
    row.counts_through(1);
    assert_eq!(row.words_for_test(), (5, 2, 1, 101));
    row.counts_through(2);
    assert_eq!(row.words_for_test(), (5, 1, 0, 0));
    row.counts_through(3);
    assert_eq!(row.words_for_test(), (0, 0, 0, 0));

    // A retirement: each of two dispatches that land past the row's
    // bound lets the row's oldest go, a stated size and then an
    // estimate.
    let row = RankCredit::default();
    for at in 0..RANK_CREDIT_KEPT {
        row.record(at + 1, mixed[(at % 3) as usize]);
    }
    assert_eq!(
        row.words_for_test(),
        (2_731 * 7 + 2_730 * 5, 8_192, 2_731, 2_731 * 101)
    );
    for at in RANK_CREDIT_KEPT..RANK_CREDIT_KEPT + 2 {
        row.record(at + 1, mixed[(at % 3) as usize]);
    }
    assert_eq!(
        row.words_for_test(),
        (2_731 * 12, 8_192, 2_730, 2_730 * 101),
        "the oldest two went, each with its own word's share"
    );
    row.counts_through(u64::MAX);
    assert_eq!(row.words_for_test(), (0, 0, 0, 0));
}

/// A read of a row's dispatches whose instant is short of the row's
/// oldest writes nothing of the row. A retirement leaves that instant
/// where it was, below the oldest dispatch the row still holds: a read
/// short of it hands every held dispatch and leaves it there. A read
/// that reaches it runs the release, which lets nothing go here and
/// states the oldest held dispatch's instant.
#[test]
fn a_read_of_a_rows_dispatches_short_of_its_oldest_instant_writes_nothing_of_the_row() {
    let row = RankCredit::default();
    // The dispatch that finds the row at its bound lets the oldest go.
    let (first, last) = (10, 10 + RANK_CREDIT_KEPT);
    for at in first..=last {
        row.record(at, SendSize::stated(1));
    }
    let oldest_held = first + 1;
    let held: Vec<u64> = (oldest_held..=last).collect();
    let oldest = || row.oldest_micros.load(Ordering::Acquire);
    let handed = |through_micros| {
        let mut handed = Vec::new();
        row.each_through(through_micros, |at_micros, _| handed.push(at_micros));
        handed
    };
    assert_eq!(oldest(), first, "a retirement leaves it");
    assert_eq!(handed(first - 1), held);
    assert_eq!(oldest(), first, "a read short of it stores none");
    assert_eq!(handed(first), held);
    assert_eq!(oldest(), oldest_held, "a read that reaches it releases");
}

#[test]
fn the_session_scorer_reads_the_rows_own_bit() {
    let mut pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    let facts = RequestFacts::default();
    let mut scores = [0.0f32; 3];
    SessionAffinity.score(&facts, &pool, &kept(3), &mut scores);
    assert_eq!(scores, [0.0, 0.0, 0.0]);
    pool[1].session_hit = true;
    SessionAffinity.score(&facts, &pool, &kept(3), &mut scores);
    assert_eq!(scores, [0.0, 1.0, 0.0]);
    // The scorer is handed the rows kept: a row is read by its index.
    let mut scores = [0.0f32; 2];
    SessionAffinity.score(&facts, &pool, &[1, 2], &mut scores);
    assert_eq!(scores, [1.0, 0.0]);
}
