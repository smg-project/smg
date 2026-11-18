//! The cases of the engine as a pick's unit: the decision held to its rule
//! over drawn pools, the rows it reads, and the pools and pickers that keep
//! the paths they had.

use std::cell::RefCell;

use super::super::{Binding, Engine, Filter, Picker, Plan, Profile, Scorer, TieBand, Weighted};
use crate::{
    affinity::Gates,
    builtins::{Best, PowerOfTwo, Random, SampledBest},
    decision::{Decision, Targets, MAX_TARGETS},
    lease::{Lifecycle, Membership},
    ring::ConsistentHash,
    spec::{BoundedLoadParams, Needs},
    view::{Candidate, Constraints, Engines, Health, Identity, PoolFacts, RequestFacts, WorkerId},
    Draw, Scratch, DRAW_BOUND,
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
        scorers: vec![Weighted {
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

/// The rule of a decision by engines, written out; `None` where no row is
/// kept.
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
        (Some((engine, row)), None) => (engine, Some(row), None),
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
    let mut targets = positions(&ranked);
    if let Some(other) = other {
        targets.truncate(MAX_TARGETS - 1);
        targets.push(other);
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
            let engines = Engines::new(&listed, &ends);
            let seed = Draw::new(seed).next_u64();
            let decision = decide(&engine, seed, &rows, engines);
            let (asked, handed) = read();
            let Some(ruled) = rule(seed, &rows, engines) else {
                // Nobody kept: the whole pass decided, and counted its pool.
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
            decide(&engine, seed, &rows, Engines::new(&listed, &ends));
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
fn a_pool_of_one_engine_is_its_rows_best_first_and_draws_no_engine() {
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[8], 3);
    for (row, in_flight) in rows.iter_mut().zip([5, 3, 9, 1, 7, 2, 8, 6]) {
        row.health.healthy = true;
        row.in_flight = in_flight;
    }
    rows[3].health.healthy = false;
    let decision = decide(&engine, 11, &rows, Engines::new(&listed, &ends));
    let pick = decision.pick().expect("a pick");
    assert_eq!(positions(&pick.targets), [5, 1, 0, 7], "the fewest first");
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
fn the_other_engines_row_is_the_last_alternate() {
    let engine = engine_with(Box::new(PowerOfTwo));
    let (mut rows, listed, ends) = pool_of(&[8, 8], 5);
    for row in &mut rows {
        row.health.healthy = true;
        // The second engine's rows hold more, so the first is taken.
        row.in_flight = if row.position < 8 { row.position } else { 50 };
    }
    for seed in 0..50 {
        let seed = Draw::new(seed).next_u64();
        let decision = decide(&engine, seed, &rows, Engines::new(&listed, &ends));
        let targets = positions(&decision.pick().expect("a pick").targets);
        assert_eq!(targets[..3], [0, 1, 2], "the taken engine's best three");
        assert!(targets[3] >= 8, "then the other engine's row: {targets:?}");
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
        let decision = decide(&engine, 7, &rows, Engines::new(&listed, &ends));
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
        let decision = decide(&engine, seed, &rows, Engines::new(&listed, &ends));
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
            let with = decide(&by_rows, seed, &rows, Engines::new(&listed, &ends));
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
    assert!(Engines::new(&[0, 1, 2], &[2]).is_empty());
    assert!(Engines::new(&[0, 1, 2], &[]).is_empty());
    let engines = Engines::new(&[2, 0, 1], &[1, 3]);
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
            let decision = decide(&engine, seed, rows, Engines::new(&listed, &ends));
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
    let profile = Profile {
        name: "ring".into(),
        filters: vec![Box::new(Asked)],
        scorers: vec![Weighted {
            scorer: Box::new(ring),
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
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::plain(key),
        seed: 1,
        ..RequestFacts::default()
    };
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        generation: 1,
        membership_epoch: epoch,
        engines,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    let decision = engine.decide(&facts, &pool, rows, &cx, &mut Scratch::new());
    let pick = decision.pick();
    pick.map(|pick| positions(&pick.targets))
        .unwrap_or_default()
}

/// How many of 64 keys a ring decided by its walk alone over `rows` and
/// their list, each key's targets held to the ones `whole`, a ring of its
/// own that is handed no list, gives over the pool read whole.
fn walked(ring: &Engine, whole: &Engine, rows: &[Candidate], listed: &[u32], ends: &[u32]) -> u32 {
    let engines = Engines::new(listed, ends);
    let mut walked = 0;
    for i in 0..64 {
        let key = format!("conv-{i}");
        read();
        let by_pass = keyed(whole, &key, rows, Engines::default(), 1);
        assert!(read().0.len() >= rows.len(), "a pass asks about every row");
        let targets = keyed(ring, &key, rows, engines, 1);
        assert_eq!(targets, by_pass, "{key} over {} rows", rows.len());
        // A walk asks about the rows it hands out and the rows refused on
        // its way, each once and under the bound; past the bound the whole
        // pass asks about every row behind it.
        let asked = read().0.len();
        let by_walk = MAX_TARGETS + DRAW_BOUND as usize - 1;
        if asked < targets.len() + DRAW_BOUND as usize {
            walked += 1;
        } else {
            let whole = rows.len()..=rows.len() + by_walk;
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
    let engines = Engines::new(&listed, &ends);
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
    let engines = Engines::new(&listed, &ends);
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
    let engines = Engines::new(&listed, &ends);
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
    let engines = Engines::new(&listed, &ends);
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
    // One ring decides for both pools, turn about; each pool's answers
    // are those of a ring that never saw the other.
    let ring = ring_engine();
    for _ in 0..2 {
        let alone = (ring_engine(), ring_engine());
        let of_first = walked(&ring, &alone.0, &first, &first_listed, &first_ends);
        let of_second = walked(&ring, &alone.1, &second, &second_listed, &second_ends);
        assert!(of_first >= 48 && of_second >= 48, "{of_first} {of_second}");
    }
}

#[test]
fn a_member_the_feed_takes_away_is_handed_no_key_while_its_row_still_stands() {
    let (rows, listed, ends) = fleet(&[8; 8], 4);
    let engines = Engines::new(&listed, &ends);
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
