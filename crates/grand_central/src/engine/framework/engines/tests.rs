//! The cases of the engine as a pick's unit: the decision held to its rule
//! over drawn pools, the rows it reads, and the pools and pickers that keep
//! the paths they had.

use std::cell::RefCell;

use super::super::{
    Binding, Engine, Filter, Picker, Plan, Profile, Scorer, TieBand, WeightedScorer,
};
use crate::{
    document::spec::{BoundedLoadParams, Needs},
    engine::{
        decision::{Decision, LiftedVeto, Targets, MAX_TARGETS},
        view::{
            fingerprint, Candidate, Constraints, Engines, Health, Identity, PoolFacts,
            RequestFacts, WorkerId,
        },
        Scratch, DRAW_BOUND,
    },
    policies::{
        cache_aware::Gates, consistent_hashing::ConsistentHash, power_of_two::PowerOfTwo,
        random::Random,
    },
    stages::{
        filters::{Healthy, NotOverloaded},
        lifecycle::Lifecycle,
        pickers::{Best, CountFirst, SampledBest},
    },
    tracking::lease::Membership,
    Draw,
};

thread_local! {
    /// The rows the filter of this file was asked about, in order, and the
    /// sets its scorer was handed: a case runs on one thread and reads
    /// what its own decisions wrote.
    static ASKED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static HANDED: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
}

/// What the decisions of this thread read since the last call.
fn read() -> (Vec<u32>, Vec<Vec<u32>>) {
    (ASKED.take(), HANDED.take())
}

/// A row-wise filter that keeps a healthy row and writes down each row it
/// is asked about.
#[derive(Debug)]
struct Asked;

impl Filter for Asked {
    fn name(&self) -> &'static str {
        "asked"
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        ASKED.with_borrow_mut(|asked| asked.push(c.position));
        c.health.healthy
    }
}

/// [`Asked`], as a filter that does not say it answers row by row.
#[derive(Debug)]
struct AskedWhole;

impl Filter for AskedWhole {
    fn name(&self) -> &'static str {
        "asked-whole"
    }

    fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
        Asked.keep(facts, c, cx)
    }
}

/// A row's score under the scorer of this file: exact levels, the fewer
/// requests in flight the better.
fn level(c: &Candidate) -> f32 {
    1.0 - c.in_flight.min(100) as f32 / 100.0
}

/// A row-wise scorer on [`level`] that writes down each set it is handed.
#[derive(Debug)]
struct Handed;

impl Lifecycle for Handed {}

impl Scorer for Handed {
    fn name(&self) -> &'static str {
        "handed"
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        rows: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        HANDED.with_borrow_mut(|handed| handed.push(kept.to_vec()));
        for (slot, &row) in out.iter_mut().zip(kept) {
            *slot = level(&rows[row as usize]);
        }
        TieBand::NORMALIZED
    }
}

/// The filter and the scorer of this file under `picker`.
fn engine_with(picker: Box<dyn Picker>) -> Engine {
    let profile = Profile {
        name: "unit".into(),
        filters: vec![Box::new(Asked)],
        scorers: vec![WeightedScorer {
            scorer: Box::new(Handed),
            weight: 1.0,
        }],
        picker,
    };
    let binding = Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: None,
        decode_shadow: None,
        sessions: None,
        overload: None,
    };
    Engine::new(vec![profile], vec![binding], 1, Needs::default())
}

/// A pool of engines of the sizes `ranks` names, its rows in the list's
/// order, and the list: the rows engine after engine, each engine's end.
/// A row's requests in flight and whether it is healthy are drawn by
/// `shape`: few counts, so equal rows are common, and one row in seven
/// refused.
fn pool_of(ranks: &[u32], shape: u64) -> (Vec<Candidate>, Vec<u32>, Vec<u32>) {
    let mut draw = Draw::new(shape);
    let mut rows = Vec::new();
    let mut ends = Vec::new();
    for (engine, &ranks) in (0_u64..).zip(ranks) {
        for rank in 0..ranks {
            let position = rows.len() as u32;
            rows.push(Candidate {
                position,
                id: WorkerId(1_000 * (engine + 1) + u64::from(rank)),
                dp_rank: rank as u16,
                dp_size: ranks as u16,
                health: Health {
                    healthy: draw.below(7) != 0,
                    ..Health::default()
                },
                in_flight: draw.below(3) as u32,
                ..Candidate::default()
            });
        }
        ends.push(rows.len() as u32);
    }
    let listed = (0..rows.len() as u32).collect();
    (rows, listed, ends)
}

fn decide(engine: &Engine, seed: u64, rows: &[Candidate], engines: Engines<'_>) -> Decision {
    let facts = RequestFacts {
        model: "m",
        seed,
        ..RequestFacts::default()
    };
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        generation: 1,
        engines,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    engine.decide(&facts, &pool, rows, &cx, &mut Scratch::new())
}

fn positions(targets: &Targets) -> Vec<u32> {
    targets.iter().map(|target| target.position).collect()
}

/// What the rule gives: the targets, the rows the filter is asked about,
/// the sets the stages are handed and how many values of the seed's stream
/// are read before the picker's compare.
struct Ruled {
    targets: Vec<u32>,
    asked: Vec<u32>,
    handed: Vec<Vec<u32>>,
    values: u32,
}

/// The rule of a decision by engines, written out; `None` where the whole
/// pass decides: no row is kept, or a pool of several engines yields one
/// engine's row and no second engine's.
fn rule(seed: u64, rows: &[Candidate], engines: Engines<'_>) -> Option<Ruled> {
    let keeps = |row: u32| rows[row as usize].health.healthy;
    let mut draw = Draw::new(seed);
    let mut values = 0;
    let mut refused = 0;
    let mut asked = Vec::new();
    // One engine among the pool's, or among those but one, by one value,
    // and its row by one more where it holds several.
    let mut stand_in = |but: Option<usize>, draw: &mut Draw, values: &mut u32| {
        let left = engines.len() - usize::from(but.is_some());
        while left > 0 && refused < DRAW_BOUND {
            let mut engine = draw.below(left);
            *values += 1;
            if but.is_some_and(|taken| engine >= taken) {
                engine += 1;
            }
            let of = engines.rows(engine);
            let row = if of.len() == 1 {
                of[0]
            } else {
                *values += 1;
                of[draw.below(of.len())]
            };
            asked.push(row);
            if keeps(row) {
                return Some((engine, row));
            }
            refused += 1;
        }
        None
    };
    let (first, second) = if engines.len() > 1 {
        let first = stand_in(None, &mut draw, &mut values)?;
        let second = stand_in(Some(first.0), &mut draw, &mut values);
        (Some(first), second)
    } else {
        (None, None)
    };
    let ties = draw.next_u64();
    values += 1;
    let mut handed = Vec::new();
    let (taken, held, other) = match (first, second) {
        (None, _) => (0, None, None),
        (Some(_), None) => return None,
        (Some((a_engine, a)), Some((b_engine, b))) => {
            handed.push(vec![a, b]);
            // The picker's compare reads the stream past the values above:
            // its first two decide nothing between two rows, its third
            // decides a tie, the lower position leading on a zero.
            let mut compare = Draw::new(draw.state());
            let (of_a, of_b) = (level(&rows[a as usize]), level(&rows[b as usize]));
            let a_leads = if of_a == of_b {
                compare.below(2);
                compare.below(1);
                (compare.below(2) == 0) == (a < b)
            } else {
                of_a > of_b
            };
            if a_leads {
                (a_engine, Some(a), Some(b))
            } else {
                (b_engine, Some(b), Some(a))
            }
        }
    };
    let mut kept = Vec::new();
    for &row in engines.rows(taken) {
        if held == Some(row) {
            kept.push(row);
        } else {
            asked.push(row);
            if keeps(row) {
                kept.push(row);
            }
        }
    }
    if kept.is_empty() {
        return None;
    }
    handed.push(kept.clone());
    // The taken engine's rows best first, equal rows drawn by the stream
    // of the value read for them: the crate's ranked pick over a draw's
    // rows, which has its own cases.
    let scores: Vec<f32> = kept.iter().map(|&row| level(&rows[row as usize])).collect();
    let by_ties = RequestFacts {
        seed: ties,
        ..RequestFacts::default()
    };
    let (mut ranked, band) = (Targets::new(), TieBand::NORMALIZED);
    SampledBest::new(1).order_drawn(&by_ties, rows, &kept, &scores, band, &mut ranked);
    // The taken engine's least rank and then the other engine's row; one
    // engine alone hands its least rank and its next.
    let mut targets = positions(&ranked);
    match other {
        Some(other) => {
            targets.truncate(1);
            targets.push(other);
        }
        None => targets.truncate(2),
    }
    Some(Ruled {
        targets,
        asked,
        handed,
        values,
    })
}

/// The shapes the cases draw over: one engine, engines of one size, sizes
/// mixed with workers alone, and a fleet of many.
const SHAPES: &[&[u32]] = &[
    &[8],
    &[2, 2],
    &[8, 8, 8, 8, 8, 8, 8, 8],
    &[1, 4, 1, 8, 2, 1],
    &[3, 3, 3, 1, 1, 5, 2, 2, 7],
    &[4; 64],
];

#[test]
fn a_decision_by_engines_is_the_rule_row_for_row_over_drawn_pools() {
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut picks, mut most) = (0, 0);
    for (shape, ranks) in (0_u64..).zip(SHAPES) {
        for seed in 0..400_u64 {
            let (rows, listed, ends) = pool_of(ranks, shape * 1_000 + seed / 20);
            let engines = Engines::new(0, &listed, &ends);
            let seed = Draw::new(seed).next_u64();
            let decision = decide(&engine, seed, &rows, engines);
            let (asked, handed) = read();
            let Some(ruled) = rule(seed, &rows, engines) else {
                // The draw decided nothing: the whole pass did, and counted
                // its pool.
                let healthy = rows.iter().filter(|row| row.health.healthy).count() as u32;
                let trace = match &decision {
                    Decision::Pick(pick) => pick.trace,
                    Decision::Shed(shed) => shed.trace,
                    Decision::Queue(queue) => queue.trace,
                };
                assert_eq!(trace.seed, seed, "the trace keeps the request's seed");
                if let Decision::Pick(pick) = &decision {
                    assert_eq!(
                        pick.trace.survivors.get(),
                        Some(healthy),
                        "{ranks:?} {seed}"
                    );
                }
                continue;
            };
            let pick = decision.pick().expect("a row was kept");
            assert_eq!(positions(&pick.targets), ruled.targets, "{ranks:?} {seed}");
            assert_eq!(asked, ruled.asked, "the rows asked about, {ranks:?} {seed}");
            assert_eq!(handed, ruled.handed, "the sets handed, {ranks:?} {seed}");
            assert_eq!(pick.trace.survivors.get(), None, "a part is not counted");
            assert_eq!(pick.trace.seed, seed);
            most = most.max(ruled.values);
            picks += 1;
        }
    }
    assert!(picks > 2_000, "{picks} decisions by the rule");
    // Two values an engine drawn and one for the ties, with every row of
    // the bound refused on the way at the most.
    assert!(most <= 2 * (2 + DRAW_BOUND) + 1, "{most} values");
}

#[test]
fn a_decision_by_engines_reads_no_more_rows_in_a_larger_fleet() {
    let engine = engine_with(Box::new(PowerOfTwo));
    for engines in [8_usize, 64] {
        let ranks = vec![8; engines];
        for seed in 0..200_u64 {
            let (mut rows, listed, ends) = pool_of(&ranks, seed);
            for row in &mut rows {
                row.health.healthy = true;
            }
            let seed = Draw::new(seed).next_u64();
            decide(&engine, seed, &rows, Engines::new(0, &listed, &ends));
            let (asked, handed) = read();
            // The two rows that stand for the engines drawn, then the other
            // rows of the one taken; two sets handed, of two rows and of
            // one engine's.
            assert_eq!(asked.len(), 2 + 7, "{engines} engines, seed {seed}");
            let handed: Vec<usize> = handed.iter().map(Vec::len).collect();
            assert_eq!(handed, [2, 8], "{engines} engines, seed {seed}");
        }
    }
}

#[test]
fn a_pool_of_one_engine_hands_its_least_rank_and_its_next_and_draws_no_engine() {
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[8], 3);
    for (row, in_flight) in rows.iter_mut().zip([5, 3, 9, 1, 7, 2, 8, 6]) {
        row.health.healthy = true;
        row.in_flight = in_flight;
    }
    rows[3].health.healthy = false;
    let decision = decide(&engine, 11, &rows, Engines::new(0, &listed, &ends));
    let pick = decision.pick().expect("a pick");
    assert_eq!(
        positions(&pick.targets),
        [5, 1],
        "the fewest, then the next"
    );
    let (asked, handed) = read();
    assert_eq!(
        asked,
        [0, 1, 2, 3, 4, 5, 6, 7],
        "each row of the engine once"
    );
    assert_eq!(
        handed,
        [vec![0, 1, 2, 4, 5, 6, 7]],
        "one set, the rows kept"
    );
}

#[test]
fn the_alternate_of_a_pick_by_engines_is_another_engines_row() {
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[8, 8], 5);
    for row in &mut rows {
        row.health.healthy = true;
        // The second engine's rows hold more, so the first is taken.
        row.in_flight = if row.position < 8 { row.position } else { 50 };
    }
    for seed in 0..50 {
        let seed = Draw::new(seed).next_u64();
        let decision = decide(&engine, seed, &rows, Engines::new(0, &listed, &ends));
        let targets = positions(&decision.pick().expect("a pick").targets);
        assert_eq!(targets.len(), 2, "{targets:?}");
        assert_eq!(targets[0], 0, "the taken engine's least rank");
        assert!(targets[1] >= 8, "then the other engine's row: {targets:?}");
    }
    // Wherever a pool holds two engines with a row kept in each, over
    // drawn shapes: a commit that loses the first target's engine finds
    // another behind it.
    for (case, &shape) in SHAPES
        .iter()
        .enumerate()
        .filter(|(_, shape)| shape.len() > 1)
    {
        let (mut rows, listed, ends) = pool_of(shape, case as u64);
        for row in &mut rows {
            row.health.healthy = true;
        }
        let engines = Engines::new(0, &listed, &ends);
        let of = |position: u32| {
            (0..engines.len()).find(|&engine| engines.rows(engine).contains(&position))
        };
        for seed in 0..100 {
            let seed = Draw::new(seed ^ 0x51).next_u64();
            let decision = decide(&engine, seed, &rows, engines);
            let targets = positions(&decision.pick().expect("a pick").targets);
            assert_eq!(targets.len(), 2, "shape {case} seed {seed}");
            assert_ne!(of(targets[0]), of(targets[1]), "shape {case} seed {seed}");
        }
    }
}

#[test]
fn a_second_engine_not_drawn_inside_the_bound_leaves_the_pool_to_the_whole_pass() {
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1], 1);
    for row in &mut rows {
        // Two engines stand, the first of them busy and the second idle.
        row.health.healthy = row.position < 2;
        row.in_flight = if row.position == 0 { 8 } else { 0 };
    }
    let engines = Engines::new(0, &listed, &ends);
    for seed in 0..200 {
        let seed = Draw::new(seed).next_u64();
        let decision = decide(&engine, seed, &rows, engines);
        let targets = positions(&decision.pick().expect("a pick").targets);
        assert_eq!(targets[0], 1, "the idle engine leads, seed {seed}");
    }
}

#[test]
fn a_pool_no_row_of_which_is_kept_is_the_whole_passes_to_shed() {
    let engine = engine_with(Box::new(PowerOfTwo));
    for ranks in [&[8][..], &[4, 4, 4]] {
        let (mut rows, listed, ends) = pool_of(ranks, 1);
        for row in &mut rows {
            row.health.healthy = false;
        }
        let decision = decide(&engine, 7, &rows, Engines::new(0, &listed, &ends));
        let Decision::Shed(shed) = decision else {
            panic!("{ranks:?}: {decision:?}");
        };
        // The whole pass ran on the stream continued; the trace keeps the
        // request's seed, from which the decision replays.
        assert_eq!(shed.trace.seed, 7, "{ranks:?}");
    }
}

#[test]
fn a_pick_the_whole_pass_makes_behind_the_unit_keeps_the_requests_seed() {
    let engine = engine_with(Box::new(PowerOfTwo));
    // One row kept in a pool of 64: the draws of most seeds are refused to
    // their bound and the whole pass finds the row.
    let (mut rows, listed, ends) = pool_of(&[4; 16], 1);
    for row in &mut rows {
        row.health.healthy = row.position == 37;
    }
    let mut widened = 0;
    for seed in 1..=40 {
        read();
        let decision = decide(&engine, seed, &rows, Engines::new(0, &listed, &ends));
        let pick = decision.pick().expect("the one row kept");
        assert_eq!(positions(&pick.targets), [37], "seed {seed}");
        assert_eq!(pick.trace.seed, seed);
        // A pass over the pool asked about every row.
        widened += usize::from(read().0.len() >= rows.len());
    }
    assert!(widened >= 20, "{widened} of 40 decisions widened");
}

#[test]
fn a_pool_that_lists_no_engine_and_a_picker_of_rows_keep_their_paths() {
    let by_rows = engine_with(Box::new(Random));
    let engine = engine_with(Box::new(PowerOfTwo));
    for ranks in [&[8, 8][..], &[4; 16]] {
        let (rows, listed, ends) = pool_of(ranks, 9);
        // Under a picker whose unit is a row the list changes nothing.
        for seed in 0..100_u64 {
            let seed = Draw::new(seed).next_u64();
            let with = decide(&by_rows, seed, &rows, Engines::new(0, &listed, &ends));
            assert_eq!(with, decide(&by_rows, seed, &rows, Engines::default()));
        }
        // Under the picker of engines a pool with no list draws rows: two
        // targets, where the list hands out the taken engine's rows.
        let unlisted = decide(&engine, 3, &rows, Engines::default());
        assert!(unlisted.pick().is_some_and(|pick| pick.targets.len() <= 2));
    }
}

#[test]
fn a_list_whose_ends_do_not_cover_its_rows_reads_as_none() {
    assert!(Engines::new(0, &[0, 1, 2], &[2]).is_empty());
    assert!(Engines::new(0, &[0, 1, 2], &[]).is_empty());
    let engines = Engines::new(0, &[2, 0, 1], &[1, 3]);
    assert_eq!(engines.len(), 2);
    assert_eq!((engines.rows(0), engines.rows(1)), (&[2][..], &[0, 1][..]));
    assert_eq!(engines.rows(2), &[] as &[u32], "a place past the list");
}

#[test]
fn an_engine_is_compared_by_one_of_its_rows() {
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[8, 8], 1);
    // How many of 400 decisions take the first engine, each handing out
    // that engine's least row or the second's.
    let first = |rows: &[Candidate]| {
        let lighter = |&seed: &u64| {
            let seed = Draw::new(seed).next_u64();
            let decision = decide(&engine, seed, rows, Engines::new(0, &listed, &ends));
            let pick = positions(&decision.pick().expect("a pick").targets)[0];
            assert!(
                pick == 0 || pick >= 8,
                "the taken engine's least row: {pick}"
            );
            pick == 0
        };
        (0..400_u64).filter(lighter).count()
    };
    // Engines whose rows stand level, 2 against 5: the lighter every time,
    // as the whole pass takes it.
    for row in &mut rows {
        row.health.healthy = true;
        row.in_flight = if row.position == 0 { 1 } else { 2 };
        if row.position >= 8 {
            row.in_flight = 5;
        }
    }
    assert_eq!(first(&rows), 400);
    // The first engine's rows spread from 0 to 7 around the second's 5:
    // the row drawn for it is the lighter five times in eight and equal
    // once, so it is taken eleven times in sixteen where the whole pass
    // takes its least row every time. An engine stands for itself by one
    // row; taking an engine's least row is what keeps its rows level.
    for row in &mut rows[..8] {
        row.in_flight = row.position;
    }
    let taken = first(&rows);
    assert!(
        (240..=310).contains(&taken),
        "{taken} of 400, 275 by the rule"
    );
}

/// The plain ring under `best` and the filter of this file.
fn ring_engine() -> Engine {
    ring_engine_of(ConsistentHash::new(None))
}

/// `ring` under `best` and the filter of this file.
fn ring_engine_of(ring: ConsistentHash) -> Engine {
    let scorer = WeightedScorer {
        scorer: Box::new(ring),
        weight: 1.0,
    };
    engine_of(vec![Box::new(Asked)], vec![scorer])
}

/// One profile of `filters` and `scorers` under `best`.
fn engine_of(filters: Vec<Box<dyn Filter>>, scorers: Vec<WeightedScorer>) -> Engine {
    engine_under(filters, scorers, Box::new(Best))
}

/// One profile of `filters` and `scorers` under `picker`.
fn engine_under(
    filters: Vec<Box<dyn Filter>>,
    scorers: Vec<WeightedScorer>,
    picker: Box<dyn Picker>,
) -> Engine {
    let profile = Profile {
        name: "ring".into(),
        filters,
        scorers,
        picker,
    };
    let binding = Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: None,
        decode_shadow: None,
        sessions: None,
        overload: None,
    };
    Engine::new(vec![profile], vec![binding], 1, Needs::default())
}

/// A pool as [`pool_of`] draws it, each row under a key of its own: where
/// it stands on a ring.
fn fleet(ranks: &[u32], shape: u64) -> (Vec<Candidate>, Vec<u32>, Vec<u32>) {
    let (mut rows, listed, ends) = pool_of(ranks, shape);
    for row in &mut rows {
        row.key = 9_000 + row.position;
    }
    (rows, listed, ends)
}

/// The rows a request with the routing key `key` is handed, by position,
/// under the membership epoch `epoch`.
fn keyed(
    engine: &Engine,
    key: &str,
    rows: &[Candidate],
    engines: Engines<'_>,
    epoch: u64,
) -> Vec<u32> {
    keyed_by(engine, key, rows, engines, epoch).0
}

/// [`keyed`], with whether the walk alone decided.
fn keyed_by(
    engine: &Engine,
    key: &str,
    rows: &[Candidate],
    engines: Engines<'_>,
    epoch: u64,
) -> (Vec<u32>, bool) {
    placed(engine, key, rows, engines, epoch)
        .map(|(rows, _, by_walk)| (rows, by_walk))
        .unwrap_or_default()
}

/// What a request with the routing key `key` is handed: its rows by
/// position, the veto its trace says was lifted, and whether the walk
/// alone decided, whose trace counts no survivors where a whole pass
/// counts them. `None` for a request that is shed.
fn placed(
    engine: &Engine,
    key: &str,
    rows: &[Candidate],
    engines: Engines<'_>,
    epoch: u64,
) -> Option<(Vec<u32>, LiftedVeto, bool)> {
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::plain(key),
        seed: 1,
        ..RequestFacts::default()
    };
    placed_as(engine, &facts, rows, engines, epoch)
}

/// [`placed`] for a request of any facts.
fn placed_as(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    rows: &[Candidate],
    engines: Engines<'_>,
    epoch: u64,
) -> Option<(Vec<u32>, LiftedVeto, bool)> {
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        generation: 1,
        membership_epoch: epoch,
        engines,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    let decision = engine.decide(facts, &pool, rows, &cx, &mut Scratch::new());
    let pick = decision.pick()?;
    Some((
        positions(&pick.targets),
        pick.trace.lifted,
        pick.trace.survivors.get().is_none(),
    ))
}

/// How many of 64 keys a ring decided by its walk alone over `rows` and
/// their list, each key's targets held to the ones `whole`, a ring of its
/// own that is handed no list, gives over the pool read whole.
fn walked(ring: &Engine, whole: &Engine, rows: &[Candidate], listed: &[u32], ends: &[u32]) -> u32 {
    walked_as(ring, whole, rows, Engines::new(0, listed, ends))
}

/// [`walked`] over a list its caller numbered.
fn walked_as(ring: &Engine, whole: &Engine, rows: &[Candidate], engines: Engines<'_>) -> u32 {
    let mut walked = 0;
    for i in 0..64 {
        let key = format!("conv-{i}");
        read();
        let by_pass = keyed(whole, &key, rows, Engines::default(), 1);
        assert!(read().0.len() >= rows.len(), "a pass asks about every row");
        let (targets, by_walk) = keyed_by(ring, &key, rows, engines, 1);
        assert_eq!(targets, by_pass, "{key} over {} rows", rows.len());
        // A walk asks about the rows it hands out and the rows refused on
        // its way, each once and under the bound; past the bound the whole
        // pass asks about every row behind it.
        let asked = read().0.len();
        let bound = MAX_TARGETS + DRAW_BOUND as usize - 1;
        if by_walk {
            assert!(asked <= bound.min(rows.len()), "{asked} rows asked about");
            walked += 1;
        } else {
            let whole = rows.len()..=rows.len() + bound;
            assert!(whole.contains(&asked), "{asked} rows asked about");
        }
    }
    walked
}

#[test]
fn a_keyed_request_by_the_rings_walk_is_the_whole_passes_answer_row_for_row() {
    let shapes: [&[u32]; 5] = [
        &[8; 8],
        &[1, 4, 1, 8, 2, 1],
        &[3, 3, 3, 1, 1, 5, 2, 2, 7],
        &[4; 64],
        &[8; 64],
    ];
    for ranks in shapes {
        for shape in 1..=3 {
            // One row in seven refused: owners and alternates passed over.
            let (rows, listed, ends) = fleet(ranks, shape);
            let walked = walked(&ring_engine(), &ring_engine(), &rows, &listed, &ends);
            assert!(walked >= 48, "{walked} of 64 keys by the walk alone");
        }
    }
}

/// The best rows the trace of a request with the routing key `key` names,
/// each with its score to the bit, and whether the walk alone decided
/// ([`placed`]).
fn traced(
    engine: &Engine,
    key: &str,
    rows: &[Candidate],
    engines: Engines<'_>,
) -> (Vec<(WorkerId, u32)>, bool) {
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::plain(key),
        seed: 1,
        ..RequestFacts::default()
    };
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        generation: 1,
        membership_epoch: 1,
        engines,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    let decision = engine.decide(&facts, &pool, rows, &cx, &mut Scratch::new());
    let trace = decision.trace();
    let best = trace.top[..usize::from(trace.top_len)]
        .iter()
        .map(|&(id, score)| (id, score.to_bits()))
        .collect();
    (best, trace.survivors.get().is_none())
}

/// The trace of a request the ring's walk places names the whole pass's
/// best rows at the whole pass's scores, to the bit. The whole pass folds
/// the stage's share of the profile's weights into each score, and a
/// weight's share of itself need not be one: the walk writes its steps by
/// that share, at a weight whose share is one and at one whose share is
/// not.
#[test]
fn a_walks_trace_scores_its_rows_as_the_whole_pass_scores_them() {
    // The weights stand on both sides: the share of 41 is not one.
    assert_eq!(3.0f32 * (1.0 / 3.0f32), 1.0);
    assert_ne!(41.0f32 * (1.0 / 41.0f32), 1.0);
    let ring_at = |weight: f32| {
        let scorer = WeightedScorer {
            scorer: Box::new(ConsistentHash::new(None)),
            weight,
        };
        engine_of(vec![Box::new(Asked)], vec![scorer])
    };
    let (rows, listed, ends) = fleet(&[8; 8], 2);
    let engines = Engines::new(0, &listed, &ends);
    for weight in [1.0f32, 3.0, 41.0] {
        // The whole pass is a ring of its own that is handed no list.
        let (ring, whole) = (ring_at(weight), ring_at(weight));
        let mut walked = 0;
        for i in 0..64 {
            let key = format!("conv-{i}");
            let (by_pass, alone) = traced(&whole, &key, &rows, Engines::default());
            assert!(!alone, "a pool that lists no engine is read whole");
            let (by_walk, alone) = traced(&ring, &key, &rows, engines);
            if alone {
                assert_eq!(by_walk, by_pass, "{key} at weight {weight}");
                walked += 1;
            }
        }
        assert!(walked >= 48, "{walked} of 64 keys by the walk alone");
    }
}

/// A client's key is walked as a session's key is: the same rows, and
/// where the walk alone decides for the one it decides for the other.
#[test]
fn a_clients_key_is_walked_as_a_sessions_key_is() {
    let ring = ring_engine();
    let (rows, listed, ends) = fleet(&[8; 8], 1);
    let engines = Engines::new(0, &listed, &ends);
    let mut walked = 0;
    for i in 0..64 {
        let key = format!("conv-{i}");
        let by_session = placed(&ring, &key, &rows, engines, 1);
        let client = RequestFacts {
            model: "m",
            client_key: fingerprint(key.as_bytes()),
            seed: 1,
            ..RequestFacts::default()
        };
        assert_eq!(
            placed_as(&ring, &client, &rows, engines, 1),
            by_session,
            "{key}"
        );
        walked += u32::from(by_session.is_some_and(|(_, _, by_walk)| by_walk));
    }
    assert!(walked >= 48, "{walked} of 64 keys by the walk alone");
}

/// A shadow takes the walk where its live decision does: under a session's
/// key and under a client's, the two decisions of one request hand the same
/// rows, and the walk alone decides for both or for neither.
#[test]
fn a_shadow_is_walked_as_its_live_decision_is() {
    let ring = |name: &str| Profile {
        name: name.into(),
        filters: vec![Box::new(Asked)],
        scorers: vec![WeightedScorer {
            scorer: Box::new(ConsistentHash::new(None)),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    };
    let binding = Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: Some(1),
        decode_shadow: None,
        sessions: None,
        overload: None,
    };
    let engine = Engine::new(
        vec![ring("ring"), ring("next")],
        vec![binding],
        1,
        Needs::default(),
    );
    let (rows, listed, ends) = fleet(&[8; 8], 1);
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        generation: 1,
        membership_epoch: 1,
        engines: Engines::new(0, &listed, &ends),
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    let handed = |decision: Option<Decision>| {
        let pick = decision.as_ref().and_then(Decision::pick)?;
        Some((
            positions(&pick.targets),
            pick.trace.survivors.get().is_none(),
        ))
    };
    let mut walked = [0, 0];
    for i in 0..64 {
        let key = format!("conv-{i}");
        let by_session = RequestFacts {
            model: "m",
            session_key: Identity::plain(&key),
            seed: 1,
            ..RequestFacts::default()
        };
        let by_client = RequestFacts {
            model: "m",
            client_key: fingerprint(key.as_bytes()),
            seed: 1,
            ..RequestFacts::default()
        };
        let requests = [("a session's", by_session), ("a client's", by_client)];
        for (walked, (whose, facts)) in walked.iter_mut().zip(requests) {
            let live = engine.decide(&facts, &pool, &rows, &cx, &mut Scratch::new());
            let live = handed(Some(live));
            let shadow = handed(engine.shadow(&facts, &pool, &rows, &cx, &mut Scratch::new()));
            assert_eq!(shadow, live, "{key} as {whose} key");
            *walked += u32::from(live.is_some_and(|(_, by_walk)| by_walk));
        }
    }
    assert!(
        walked.iter().all(|&walked| walked >= 48),
        "{walked:?} of 64 keys by the walk alone"
    );
}

#[test]
fn a_walk_that_meets_refused_rows_to_its_bound_leaves_the_pool_to_the_whole_pass() {
    // Three rows kept of 64: most walks meet eight refused rows first.
    let (mut rows, listed, ends) = fleet(&[8; 8], 1);
    for row in &mut rows {
        row.health.healthy = [5, 23, 60].contains(&row.position);
    }
    let walked = walked(&ring_engine(), &ring_engine(), &rows, &listed, &ends);
    assert!(walked <= 32, "{walked} of 64 keys by the walk alone");
    // And none kept: the whole pass sheds.
    for row in &mut rows {
        row.health.healthy = false;
    }
    let engines = Engines::new(0, &listed, &ends);
    let shed = keyed(&ring_engine(), "conv-0", &rows, engines, 1);
    assert_eq!(shed, Vec::<u32>::new());
}

/// A routing key leads the prefix, so the prefix form places a keyed
/// request by the same walk. The bounded-load rule reads every survivor's
/// count and the pool whole, and where no owner is over its bound it names
/// the plain ring's rows: a key's worker is one under every form.
#[test]
fn a_routing_key_is_placed_alike_under_every_form_of_the_ring() {
    let (rows, listed, ends) = fleet(&[8, 1, 4, 2, 8, 1], 2);
    let engines = Engines::new(0, &listed, &ends);
    let plain = ring_engine();
    let prefixed = ConsistentHash::new(None).keyed_by_prefix(64, Gates::default(), &[]);
    let prefixed = ring_engine_of(prefixed);
    let rule = BoundedLoadParams {
        factor: Some(1.25),
        abs_margin: Some(10),
    };
    let bounded = ring_engine_of(ConsistentHash::new(Some(&rule)));
    for i in 0..64 {
        let key = format!("conv-{i}");
        let targets = keyed(&plain, &key, &rows, engines, 1);
        read();
        assert_eq!(keyed(&prefixed, &key, &rows, engines, 1), targets, "{key}");
        assert!(read().0.len() < rows.len(), "the prefix form walks");
        assert_eq!(keyed(&bounded, &key, &rows, engines, 1), targets, "{key}");
        assert!(
            read().0.len() >= rows.len(),
            "the rule reads the pool whole"
        );
    }
}

#[test]
fn a_pool_whose_host_gives_the_walk_no_word_is_read_whole() {
    let (rows, listed, ends) = fleet(&[8; 8], 1);
    let engines = Engines::new(0, &listed, &ends);
    let (ring, whole) = (ring_engine(), ring_engine());
    let by_pass = keyed(&whole, "conv-3", &rows, Engines::default(), 1);
    // No membership epoch, and no list.
    for (engines, epoch) in [(engines, 0), (Engines::default(), 1)] {
        read();
        assert_eq!(keyed(&ring, "conv-3", &rows, engines, epoch), by_pass);
        assert!(read().0.len() >= rows.len());
    }
    // Rows that do not stand in the order of their ids: a member's row
    // cannot be found by its id, and the pool is the whole pass's at
    // every decision under that epoch.
    let mut turned = rows.clone();
    turned.swap(3, 40);
    for (position, row) in (0..).zip(&mut turned) {
        row.position = position;
    }
    let by_pass = keyed(&ring_engine(), "conv-3", &turned, Engines::default(), 2);
    for _ in 0..2 {
        read();
        assert_eq!(keyed(&ring, "conv-3", &turned, engines, 2), by_pass);
        assert!(read().0.len() >= rows.len());
    }
    // A request with no routing key is the profile's run.
    read();
    decide(&ring, 5, &rows, engines);
    assert!(read().0.len() >= rows.len());
}

#[test]
fn a_row_refused_when_its_pool_was_first_read_owns_its_keys_once_it_is_kept() {
    let (mut rows, listed, ends) = fleet(&[8; 8], 3);
    let engines = Engines::new(0, &listed, &ends);
    for row in &mut rows {
        row.health.healthy = row.position % 4 != 1;
    }
    // Both rings read the pool while a row in four is refused.
    let (ring, whole) = (ring_engine(), ring_engine());
    walked(&ring, &whole, &rows, &listed, &ends);
    // The rows are kept under the same epoch: no membership write shows
    // the ring the pool again, and each owns the keys the whole pass
    // gives it.
    for row in &mut rows {
        row.health.healthy = true;
    }
    let walked = walked(&ring, &whole, &rows, &listed, &ends);
    assert_eq!(walked, 64, "every key by the walk alone");
    let owners = (0..64).map(|i| keyed(&ring, &format!("conv-{i}"), &rows, engines, 1)[0]);
    assert!(
        owners.into_iter().any(|row| row % 4 == 1),
        "a row kept late owns a key"
    );
}

#[test]
fn a_ring_that_holds_another_pools_members_passes_them_by() {
    let (first, first_listed, first_ends) = fleet(&[8; 8], 1);
    let (mut second, second_listed, second_ends) = fleet(&[4; 16], 2);
    for row in &mut second {
        row.id = WorkerId(row.id.0 + 500_000);
        row.key += 500_000;
    }
    // One ring decides for both pools, turn about, each under its own
    // number; each pool's answers are those of a ring that never saw the
    // other.
    let ring = ring_engine();
    for _ in 0..2 {
        let alone = (ring_engine(), ring_engine());
        let engines = Engines::new(0, &first_listed, &first_ends);
        let of_first = walked_as(&ring, &alone.0, &first, engines);
        let engines = Engines::new(1, &second_listed, &second_ends);
        let of_second = walked_as(&ring, &alone.1, &second, engines);
        assert!(of_first >= 48 && of_second >= 48, "{of_first} {of_second}");
    }
}

#[test]
fn a_member_the_feed_takes_away_is_handed_no_key_while_its_row_still_stands() {
    let (rows, listed, ends) = fleet(&[8; 8], 4);
    let engines = Engines::new(0, &listed, &ends);
    let (ring, whole) = (ring_engine(), ring_engine());
    let owner = keyed(&ring, "conv-9", &rows, engines, 1)[0];
    keyed(&whole, "conv-9", &rows, Engines::default(), 1);
    // The feed names the owner as gone before a publication drops its row.
    let gone = Membership::Removed(rows[owner as usize].id);
    ring.on_membership(&gone);
    whole.on_membership(&gone);
    let targets = keyed(&ring, "conv-9", &rows, engines, 1);
    assert!(!targets.contains(&owner), "{targets:?}");
    let by_pass = keyed(&whole, "conv-9", &rows, Engines::default(), 1);
    assert_eq!(targets, by_pass);
}

#[test]
fn a_walk_ends_once_every_row_of_a_small_pool_was_asked() {
    // Fewer rows than the answer has room for: the walk asks about each
    // once and ends by itself, with no turn round the ring.
    for ranks in [2, 3] {
        for refused in [None, Some(1)] {
            let (mut rows, listed, ends) = fleet(&[ranks], 7);
            for row in &mut rows {
                row.health.healthy = Some(row.position) != refused;
            }
            let walked = walked(&ring_engine(), &ring_engine(), &rows, &listed, &ends);
            assert_eq!(walked, 64, "{ranks} rows, {refused:?} refused");
        }
    }
}

#[test]
fn a_row_the_ring_does_not_hold_is_the_whole_passes_to_place() {
    let (mut rows, listed, ends) = fleet(&[3], 4);
    for row in &mut rows {
        row.health.healthy = true;
    }
    let engines = Engines::new(0, &listed, &ends);
    let (ring, whole) = (ring_engine(), ring_engine());
    let owner = keyed(&ring, "conv-9", &rows, engines, 1)[0];
    keyed(&whole, "conv-9", &rows, Engines::default(), 1);
    // The feed names the owner as gone before a publication drops its row:
    // two members are left for three slots, and the whole pass hands the
    // row the ring lost the last of them.
    let gone = Membership::Removed(rows[owner as usize].id);
    ring.on_membership(&gone);
    whole.on_membership(&gone);
    let (targets, by_walk) = keyed_by(&ring, "conv-9", &rows, engines, 1);
    let by_pass = keyed(&whole, "conv-9", &rows, Engines::default(), 1);
    assert_eq!(targets, by_pass);
    assert_eq!(targets.len(), 3, "{targets:?}");
    assert_eq!(targets[2], owner, "the row the ring lost, last");
    assert!(
        !by_walk,
        "a pool shown with a row the ring fenced is not walked"
    );
}

#[test]
fn a_walk_come_round_short_of_the_rows_handed_is_the_whole_passes_to_place() {
    // A host's broken word: a pool shown with two rows is handed a third
    // under the same epoch and number. The ring does not hold it, so a
    // walk comes round having asked about two rows of three.
    let (mut rows, listed, ends) = fleet(&[3], 4);
    for row in &mut rows {
        row.health.healthy = true;
    }
    let (ring, whole) = (ring_engine(), ring_engine());
    let two = Engines::new(0, &listed[..2], &[2]);
    let (_, by_walk) = keyed_by(&ring, "conv-9", &rows[..2], two, 1);
    assert!(by_walk, "two rows of two by the walk");
    let engines = Engines::new(0, &listed, &ends);
    let (targets, by_walk) = keyed_by(&ring, "conv-9", &rows, engines, 1);
    let by_pass = keyed(&whole, "conv-9", &rows, Engines::default(), 1);
    assert_eq!(targets, by_pass);
    assert_eq!(targets.len(), 3, "{targets:?}");
    assert!(!by_walk, "a walk that came round short leaves the pool");
}

#[test]
fn pools_whose_lists_stand_in_one_buffer_are_each_shown_to_the_ring() {
    // A host that builds every pool's list in one buffer: one place in
    // memory under one epoch, and a number a pool.
    let (first, listed, ends) = fleet(&[8; 8], 1);
    let (mut second, ..) = fleet(&[8; 8], 2);
    for row in &mut second {
        row.id = WorkerId(row.id.0 + 500_000);
        row.key += 500_000;
        // A row in four is refused while the ring first reads the pool.
        row.health.healthy = row.position % 4 != 1;
    }
    let ring = ring_engine();
    let alone = (ring_engine(), ring_engine());
    let of_first = walked_as(&ring, &alone.0, &first, Engines::new(0, &listed, &ends));
    let of_second = walked_as(&ring, &alone.1, &second, Engines::new(1, &listed, &ends));
    assert!(of_first >= 48 && of_second >= 32, "{of_first} {of_second}");
    // Kept under the same epoch, each row of the second pool owns the keys
    // a ring of its own gives it.
    for row in &mut second {
        row.health.healthy = true;
    }
    let of_second = walked_as(&ring, &alone.1, &second, Engines::new(1, &listed, &ends));
    assert_eq!(of_second, 64, "every key by the walk alone");
}

#[test]
fn nine_pools_on_one_ring_are_each_walked_turn_about() {
    // Nine models that one fleet serves: nine pools of the same rows, a
    // number each, under one epoch.
    let (mut rows, listed, ends) = fleet(&[4; 4], 1);
    for row in &mut rows {
        row.health.healthy = true;
    }
    let ring = ring_engine();
    for _ in 0..2 {
        for pool in 0..9 {
            let engines = Engines::new(pool, &listed, &ends);
            let walked = walked_as(&ring, &ring_engine(), &rows, engines);
            assert_eq!(walked, 64, "pool {pool}");
        }
    }
}

/// `rows` as a pool of its own: ids and ring keys `by` further on, and
/// every row kept.
fn apart(rows: &mut [Candidate], by: u64) {
    for row in rows {
        row.id = WorkerId(row.id.0 + by);
        row.key += by as u32;
        row.health.healthy = true;
    }
}

#[test]
fn a_pool_that_is_a_small_share_of_its_ring_is_walked_past_the_others_members() {
    // Eight rows and two rows beside 64 of another pool on one ring: a
    // walk for a small pool passes the others' members, each by its mark,
    // and hands the rows a ring of its own gives it.
    let (large, large_listed, large_ends) = fleet(&[8; 8], 1);
    let (mut small, small_listed, small_ends) = fleet(&[8], 2);
    let (mut pair, pair_listed, pair_ends) = fleet(&[2], 3);
    apart(&mut small, 500_000);
    apart(&mut pair, 900_000);
    let ring = ring_engine();
    let alone = (ring_engine(), ring_engine(), ring_engine());
    for turn in 0..2 {
        let engines = Engines::new(0, &large_listed, &large_ends);
        let of_large = walked_as(&ring, &alone.0, &large, engines);
        assert!(of_large >= 48, "{of_large} of 64 keys, turn {turn}");
        let engines = Engines::new(1, &small_listed, &small_ends);
        let of_small = walked_as(&ring, &alone.1, &small, engines);
        let engines = Engines::new(2, &pair_listed, &pair_ends);
        let of_pair = walked_as(&ring, &alone.2, &pair, engines);
        assert_eq!((of_small, of_pair), (64, 64), "turn {turn}");
    }
}

#[test]
fn a_member_two_pools_showed_is_a_row_of_each_to_their_walks() {
    // Two models on fleets that overlap: 16 rows a pool, eight of them
    // the other's too, each pool's rows numbered from its own first.
    let (mut rows, ..) = fleet(&[8; 3], 1);
    apart(&mut rows, 0);
    let (_, listed, ends) = fleet(&[8; 2], 1);
    let one = rows[..16].to_vec();
    let mut two = rows[8..].to_vec();
    for row in &mut two {
        row.position -= 8;
    }
    let ring = ring_engine();
    let alone = (ring_engine(), ring_engine());
    for turn in 0..2 {
        let of_one = walked_as(&ring, &alone.0, &one, Engines::new(0, &listed, &ends));
        let of_two = walked_as(&ring, &alone.1, &two, Engines::new(1, &listed, &ends));
        assert_eq!((of_one, of_two), (64, 64), "turn {turn}");
    }
}

#[test]
fn a_member_no_pool_showed_is_passed_by_a_walk() {
    // A pool whose host lists no engine is read whole, and the ring
    // learns its rows from the pass: members no pool showed, at places
    // past the marks of the pool shown before them.
    let (rows, listed, ends) = fleet(&[8; 2], 1);
    let (mut bare, ..) = fleet(&[8; 2], 2);
    apart(&mut bare, 500_000);
    let ring = ring_engine();
    let alone = (ring_engine(), ring_engine());
    for turn in 0..2 {
        let engines = Engines::new(0, &listed, &ends);
        let of_listed = walked_as(&ring, &alone.0, &rows, engines);
        assert!(of_listed >= 48, "{of_listed} of 64 keys, turn {turn}");
        let of_bare = walked_as(&ring, &alone.1, &bare, Engines::default());
        assert_eq!(of_bare, 0, "turn {turn}: no list, no walk");
    }
}

#[test]
fn a_row_the_fence_left_out_of_a_shown_pool_owns_its_keys_once_the_feed_names_it() {
    let (mut rows, listed, ends) = fleet(&[8; 8], 5);
    for row in &mut rows {
        row.health.healthy = true;
    }
    let engines = Engines::new(0, &listed, &ends);
    let (ring, whole) = (ring_engine(), ring_engine());
    walked(&ring, &whole, &rows, &listed, &ends);
    let owner = keyed(&ring, "conv-9", &rows, engines, 1)[0];
    // The feed takes the owner away, and the next epoch's rows hold it
    // again before the feed says so: the pool is shown with the row inside
    // the fence, and is the whole pass's while the row is left out.
    let id = rows[owner as usize].id;
    for engine in [&ring, &whole] {
        engine.on_membership(&Membership::Removed(id));
    }
    let (targets, by_walk) = keyed_by(&ring, "conv-9", &rows, engines, 2);
    assert_eq!(
        targets,
        keyed(&whole, "conv-9", &rows, Engines::default(), 2)
    );
    assert!(!targets.contains(&owner) && !by_walk, "{targets:?}");
    // The feed names it: the next decision shows the pool with the row,
    // which owns its keys again, and the walk places them from then on.
    for engine in [&ring, &whole] {
        engine.on_membership(&Membership::Added(id));
    }
    for _ in 0..2 {
        let (targets, by_walk) = keyed_by(&ring, "conv-9", &rows, engines, 2);
        assert_eq!(
            targets,
            keyed(&whole, "conv-9", &rows, Engines::default(), 2)
        );
        assert_eq!(targets[0], owner, "the row owns its key again");
        assert!(by_walk);
    }
}

#[test]
fn a_ring_with_a_stage_behind_it_is_its_profiles_run() {
    // The second stage moves the order, so no walk alone places a request.
    let both = || {
        let ring = WeightedScorer {
            scorer: Box::new(ConsistentHash::new(None)),
            weight: 1.0,
        };
        let cost = WeightedScorer {
            scorer: Box::new(Handed),
            weight: 4.0,
        };
        engine_of(vec![Box::new(Asked)], vec![ring, cost])
    };
    let (mut rows, listed, ends) = fleet(&[8; 8], 6);
    for row in &mut rows {
        row.health.healthy = true;
        // Levels far enough apart to outweigh a step of the ring's order.
        row.in_flight = row.position * 37 % 90;
    }
    let engines = Engines::new(0, &listed, &ends);
    let (profile, unlisted, alone) = (both(), both(), ring_engine());
    let mut moved = 0;
    for i in 0..64 {
        let key = format!("conv-{i}");
        read();
        let (targets, by_walk) = keyed_by(&profile, &key, &rows, engines, 1);
        assert!(!by_walk, "{key}");
        assert!(read().0.len() >= rows.len(), "the pool is read whole");
        let by_pass = keyed(&unlisted, &key, &rows, Engines::default(), 1);
        assert_eq!(targets, by_pass, "{key}");
        moved += usize::from(targets != keyed(&alone, &key, &rows, engines, 1));
    }
    assert!(moved > 0, "the second stage moves the order");
}

#[test]
fn a_ring_beside_a_filter_that_reads_its_pool_whole_is_its_profiles_run() {
    let ring = WeightedScorer {
        scorer: Box::new(ConsistentHash::new(None)),
        weight: 1.0,
    };
    let engine = engine_of(vec![Box::new(AskedWhole)], vec![ring]);
    let (mut rows, listed, ends) = fleet(&[8; 8], 6);
    for row in &mut rows {
        row.health.healthy = true;
    }
    let engines = Engines::new(0, &listed, &ends);
    for i in 0..8 {
        let key = format!("conv-{i}");
        read();
        let (targets, by_walk) = keyed_by(&engine, &key, &rows, engines, 1);
        assert!(!by_walk, "{key}");
        assert!(read().0.len() >= rows.len(), "the pool is read whole");
        let by_pass = keyed(&ring_engine(), &key, &rows, Engines::default(), 1);
        assert_eq!(targets, by_pass, "{key}");
    }
}

#[test]
fn a_pick_the_whole_pass_makes_behind_the_unit_reads_the_stream_on() {
    // Three level rows kept in a pool of fifteen, below the size a draw
    // starts at: with no list the same profile is the whole pass.
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[3; 5], 1);
    for row in &mut rows {
        row.health.healthy = [1, 7, 12].contains(&row.position);
        row.in_flight = 0;
    }
    let engines = Engines::new(0, &listed, &ends);
    let pick = |seed, engines| {
        let decision = decide(&engine, seed, &rows, engines);
        positions(&decision.pick().expect("a row is kept").targets)
    };
    let (mut widened, mut past_first, mut apart) = (0, 0, 0);
    for seed in 1..=400 {
        // The unit's draws, as the rule draws them: this case reads the
        // seeds whose draws find no two engines inside the bound.
        let mut draw = Draw::new(seed);
        let mut refused = 0;
        let mut stand_in = |but: Option<usize>, draw: &mut Draw| {
            while refused < DRAW_BOUND {
                let mut engine = draw.below(engines.len() - usize::from(but.is_some()));
                if but.is_some_and(|taken| engine >= taken) {
                    engine += 1;
                }
                let of = engines.rows(engine);
                if rows[of[draw.below(of.len())] as usize].health.healthy {
                    return Some(engine);
                }
                refused += 1;
            }
            None
        };
        let first = stand_in(None, &mut draw);
        let second = first.and_then(|first| stand_in(Some(first), &mut draw));
        if second.is_some() {
            continue;
        }
        // The whole pass picks by the stream where the draws left it, and
        // reads no value of it twice.
        assert_eq!(pick(seed, engines), pick(draw.state(), Engines::default()));
        widened += 1;
        past_first += usize::from(first.is_some());
        apart += usize::from(pick(seed, engines) != pick(seed, Engines::default()));
    }
    // Both misses are read: no engine found, and a first engine with no
    // second.
    assert!(
        widened - past_first >= 20 && past_first >= 20 && apart > 0,
        "{widened} widened, {past_first} of them past a first engine, {apart} apart"
    );
}

#[test]
fn a_second_engine_not_drawn_is_the_stages_pick_under_a_fold() {
    // The health filter folds into one test of a row, and this file's
    // filter does not: its profile widens by the stages. No front row of
    // the pool stands and three rows of two engines do, so most draws
    // find one engine and no second.
    let scored = || WeightedScorer {
        scorer: Box::new(Handed),
        weight: 1.0,
    };
    let folded = engine_under(
        vec![Box::new(Healthy)],
        vec![scored()],
        Box::new(PowerOfTwo),
    );
    let staged = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[3; 5], 1);
    for row in &mut rows {
        row.health.healthy = [7, 12, 13].contains(&row.position);
    }
    let engines = Engines::new(0, &listed, &ends);
    let mut widened = 0;
    for seed in 1..=400 {
        let by_fold = decide(&folded, seed, &rows, engines);
        let by_stages = decide(&staged, seed, &rows, engines);
        let (by_fold, by_stages) = (
            by_fold.pick().expect("a row stands"),
            by_stages.pick().expect("a row stands"),
        );
        assert_eq!(
            positions(&by_fold.targets),
            positions(&by_stages.targets),
            "seed {seed}"
        );
        assert_eq!(by_fold.trace.seed, by_stages.trace.seed, "seed {seed}");
        let counted = by_fold.trace.survivors.get();
        assert_eq!(counted, by_stages.trace.survivors.get(), "seed {seed}");
        widened += usize::from(counted.is_some());
    }
    assert!(widened >= 200, "{widened} of 400 decisions widened");
}

/// The plain ring under `best`, behind the filter of this file and the
/// overload veto, marked as the validator marks a profile the ring alone
/// places: a keyed request keeps its key's worker over the veto.
fn kept_ring() -> Engine {
    let scorer = WeightedScorer {
        scorer: Box::new(ConsistentHash::new(None)),
        weight: 1.0,
    };
    let mut engine = engine_of(vec![Box::new(Asked), Box::new(NotOverloaded)], vec![scorer]);
    engine.places_by_key(0);
    engine
}

/// A key kept over the veto is placed by the ring's walk where its pool
/// lists its engines: the rows handed and the mark of the lifted veto are
/// the whole pass's, with half the rows over their thresholds, with every
/// row but one over, and with every row over. The lift reaches the walk:
/// its rows are the ones it hands where no row is over.
#[test]
fn a_key_kept_over_the_veto_is_placed_by_the_walk_as_the_whole_pass_places_it() {
    let (ring, whole) = (kept_ring(), kept_ring());
    for (shape, under) in [(1, 2), (2, 64), (3, u32::MAX)] {
        let (mut rows, listed, ends) = fleet(&[8; 8], shape);
        rows[1].health.healthy = true;
        let engines = Engines::new(0, &listed, &ends);
        let keys: Vec<String> = (0..64).map(|i| format!("conv-{i}")).collect();
        let calm = keys.iter().map(|key| placed(&ring, key, &rows, engines, 1));
        let calm: Vec<_> = calm.collect();
        for row in &mut rows {
            // One row in `under` stands under its thresholds, or none.
            row.health.overloaded = under == u32::MAX || row.position % under != 1;
        }
        let mut walked = 0;
        for (key, calm) in keys.iter().zip(calm) {
            let by_pass = placed(&whole, key, &rows, Engines::default(), 1);
            let (passed, mark, by_walk) = by_pass.expect("a kept key is placed");
            assert!(!by_walk, "{key}: a pool with no list is read whole");
            let (targets, lifted, by_walk) =
                placed(&ring, key, &rows, engines, 1).expect("a kept key is placed");
            assert_eq!((&targets, lifted), (&passed, mark), "{key}, one in {under}");
            assert_eq!(Some(targets), calm.map(|calm| calm.0), "{key}: the lift");
            // The one row under its thresholds stands without the lift,
            // whether the walk met it or not.
            let stood = if under == u32::MAX {
                LiftedVeto::Overload
            } else {
                LiftedVeto::None
            };
            assert_eq!(lifted, stood, "{key}, one in {under}");
            walked += u32::from(by_walk);
        }
        assert!(walked >= 48, "{walked} of 64 keys by the walk alone");
    }
}

/// Under a binding that sheds the veto stands for a key, and the walk
/// passes a row over its thresholds as it passes any row refused: the rows
/// handed are the whole pass's.
#[test]
fn a_key_under_a_binding_that_sheds_is_walked_with_the_veto_standing() {
    let shedding = || {
        let mut ring = kept_ring();
        ring.shed_when_all_vetoed(0);
        ring
    };
    let (mut rows, listed, ends) = fleet(&[8; 8], 1);
    for row in &mut rows {
        row.health.overloaded = row.position % 5 == 0;
    }
    let walked = walked(&shedding(), &shedding(), &rows, &listed, &ends);
    assert!(walked >= 48, "{walked} of 64 keys by the walk alone");
}

/// The ring under a picker that narrows is its profile's run. Count-first
/// compares the rows that hold the fewest requests alone, so a key's owner
/// yields to a row that holds fewer: no walk hands the owner out, and the
/// rows are the ones a pool that lists nothing is handed. The picker that
/// samples follows no ring either.
#[test]
fn a_ring_under_a_picker_that_narrows_is_its_profiles_run() {
    let ring_under = |picker: Box<dyn Picker>| {
        let scorer = WeightedScorer {
            scorer: Box::new(ConsistentHash::new(None)),
            weight: 1.0,
        };
        engine_under(vec![Box::new(Asked)], vec![scorer], picker)
    };
    let (mut rows, listed, ends) = fleet(&[8; 8], 1);
    for row in &mut rows {
        row.health.healthy = true;
    }
    let engines = Engines::new(0, &listed, &ends);
    let (narrowing, whole) = (
        ring_under(Box::new(CountFirst)),
        ring_under(Box::new(CountFirst)),
    );
    let sampling = ring_under(Box::new(SampledBest::new(2)));
    let mut yielded = 0;
    for i in 0..64 {
        let key = format!("conv-{i}");
        let (targets, by_walk) = keyed_by(&narrowing, &key, &rows, engines, 1);
        assert!(!by_walk, "{key}: count-first narrows");
        let by_pass = keyed(&whole, &key, &rows, Engines::default(), 1);
        assert_eq!(targets, by_pass, "{key}");
        let owner = keyed(&ring_engine(), &key, &rows, Engines::default(), 1)[0];
        yielded += u32::from(targets[0] != owner);
        let (_, by_walk) = keyed_by(&sampling, &key, &rows, engines, 1);
        assert!(!by_walk, "{key}: a sample follows no ring");
    }
    assert!(
        yielded >= 16,
        "{yielded} of 64 owners yield to a row with fewer"
    );
}
