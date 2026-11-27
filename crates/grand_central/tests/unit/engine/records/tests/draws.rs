//! The draw, which writes a worker's record when it asks about the worker:
//! the floor it draws from, and its targets beside the whole write's.

use super::*;

/// Every target of a decision, by position and id; none for a shed.
fn targets(decision: &Decision) -> Vec<(u32, WorkerId)> {
    decision.pick().map_or_else(Vec::new, |pick| {
        pick.targets
            .iter()
            .map(|target| (target.position, target.id))
            .collect()
    })
}

/// A fleet of `workers` whose reports and states differ worker by worker:
/// queued work that falls and rises over the list, one worker in seven not
/// ready, one in eleven overloaded, and a few requests in flight.
fn mixed_fleet(workers: usize) -> (Vec<Arc<dyn Worker>>, Arc<LoadSnapshot>) {
    let fleet = fleet(workers);
    for (at, worker) in fleet.iter().enumerate() {
        if at % 7 == 3 {
            worker.set_status(WorkerStatus::NotReady);
        }
        if at % 11 == 5 {
            worker.set_overloaded(true);
        }
        for _ in 0..at % 4 {
            worker.increment_load();
        }
    }
    let queued: Vec<i32> = (0..workers)
        .map(|at| ((at * 37 + 11) % 97) as i32 * 40)
        .collect();
    let (loads, _) = reports(&queued, Duration::from_millis(5));
    (fleet, loads)
}

/// Asserts that, for `seeds` seeds, the decision that writes a worker's
/// record when its draw asks about the worker ([`decide`]) hands out the
/// targets of the decision over every worker's record written first, under
/// two engines of `document` asked the same requests in the same order.
/// `converted` says whether the drawing side's list holds its snapshot's
/// rows before the draws: a draw then copies a worker's report from them,
/// and looks the report up itself where the list holds none. Returns how
/// many of the decisions were a draw's.
fn assert_drawn_as_written(
    document: &PolicySpec,
    workers: &[Arc<dyn Worker>],
    loads: &Arc<LoadSnapshot>,
    named: Option<AffinityRows<'_>>,
    seeds: u64,
    converted: bool,
) -> usize {
    let (drawing, writing) = (applied_of(document.clone()), applied_of(document.clone()));
    let clock = DecisionClock::new();
    let drawn_pool = PoolRecords::new(workers, &drawing, &clock);
    let written_pool = PoolRecords::new(workers, &writing, &clock);
    let cx = Constraints::default();
    let mut rows = Vec::new();
    if converted {
        let live = Live::at(drawing.engine(), &clock, Instant::now(), Some(loads), None);
        let _ = drawn_pool.write(workers, &live, &mut rows);
    }
    assert_eq!(held(&drawn_pool).is_some(), converted);
    let mut scratch = Scratch::default();
    let mut draws = 0;
    for seed in 0..seeds {
        let now = Instant::now();
        let request = RequestFacts {
            seed: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            named,
            ..single()
        };

        let live = Live::at(drawing.engine(), &clock, now, Some(loads), None);
        let mut facts = request;
        let unsaid = drawn_pool.facts(&live, false);
        draws += usize::from(
            drawing
                .engine()
                .drawn_for(&facts, &unsaid, drawn_pool.len())
                .is_some(),
        );
        let drawn = decide(
            drawing.engine(),
            &drawn_pool,
            workers,
            &live,
            &mut facts,
            &cx,
        );

        let live = Live::at(writing.engine(), &clock, now, Some(loads), None);
        let mut facts = request;
        facts.now_micros = live.now_micros();
        let (written, pool_facts) = written_pool.write(workers, &live, &mut rows);
        let whole = writing
            .engine()
            .decide(&facts, &pool_facts, written, &cx, &mut scratch);

        assert_eq!(
            targets(&drawn),
            targets(&whole),
            "seed {seed}: the draw before the records and the whole write hand out the same targets"
        );
        assert_eq!(
            matches!(drawn, Decision::Shed(_)),
            matches!(whole, Decision::Shed(_))
        );
    }
    // Where every decision was a draw's, the list was converted one time, ahead
    // of them or by them: the targets above are the whole write's either side.
    if draws as u64 == seeds {
        assert_eq!(drawn_pool.conversions(), 1);
    }
    draws
}

#[test]
fn a_draw_before_the_records_hands_out_the_whole_writes_targets() {
    let (workers, loads) = mixed_fleet(64);
    let preset = |name: &str| PolicySpec::preset(name).expect("a built-in preset");
    let mut count_first = preset("power_of_two");
    for profile in count_first.profiles.values_mut() {
        profile.pick = PickSpec::from("power-of-two-count-first");
    }
    for (name, document) in [
        ("power_of_two", preset("power_of_two")),
        ("power-of-two-count-first", count_first),
        ("random", preset("random")),
        ("passthrough", preset("passthrough")),
        ("round_robin", preset("round_robin")),
    ] {
        for converted in [false, true] {
            let draws = assert_drawn_as_written(&document, &workers, &loads, None, 400, converted);
            assert_eq!(draws, 400, "{name}: every decision is its draw's");
        }
    }

    // The sampled best draws for a request that names rows, and the rows
    // named join the rows drawn: their records are written too.
    let holders = [
        AffinityHolder {
            row: 9,
            overlap_blocks: 12,
            predicted_blocks: 0,
        },
        AffinityHolder {
            row: 40,
            overlap_blocks: 3,
            predicted_blocks: 0,
        },
    ];
    let named = AffinityRows {
        holders: &holders,
        warming: &[17],
        ..AffinityRows::default()
    };
    let document = preset("cache_aware");
    for converted in [false, true] {
        let draws =
            assert_drawn_as_written(&document, &workers, &loads, Some(named), 400, converted);
        assert_eq!(
            draws, 400,
            "cache_aware: a request that names rows is drawn for"
        );
    }
    // A request that names none is the whole pass's, and decides the same.
    assert_eq!(
        assert_drawn_as_written(&document, &workers, &loads, None, 50, false),
        0
    );
}

#[test]
fn a_draw_that_keeps_nobody_widens_to_every_record() {
    // One worker in sixty-four is ready: nearly every draw of two lands on
    // refused workers alone, and the whole pass finds the one.
    let (workers, loads) = mixed_fleet(64);
    for (at, worker) in workers.iter().enumerate() {
        if at != 29 {
            worker.set_status(WorkerStatus::NotReady);
        }
    }
    let document = PolicySpec::preset("power_of_two").expect("a built-in preset");
    assert_drawn_as_written(&document, &workers, &loads, None, 200, false);
    assert_drawn_as_written(&document, &workers, &loads, None, 200, true);

    let spec = applied_of(document);
    let clock = DecisionClock::new();
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let mut facts = single();
    let decision = decide(
        spec.engine(),
        &pool,
        &workers,
        &live,
        &mut facts,
        &Constraints::default(),
    );
    assert_eq!(picked(&decision).0, 29);
}

/// Under the floor every profile draws from, a pick of two draws only over
/// a list whose every worker publishes its routing state available. A list
/// with a worker that does not is written whole, as every list of that size
/// was before a pick of two drew there: no refused row costs it a draw, and
/// the decision reads the routing state of every worker. From that floor on
/// a draw is asked of every list, as it was.
#[test]
fn a_pick_of_two_draws_under_the_floor_only_over_a_list_that_stands() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let decided = |workers: &Vec<Arc<dyn Worker>>| {
        let pool = PoolRecords::new(workers, &spec, &clock);
        let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
        decide(
            spec.engine(),
            &pool,
            workers,
            &live,
            &mut single(),
            &Constraints::default(),
        )
    };
    for size in [WRITTEN_PAIR_FROM, DRAW_FROM - 1, DRAW_FROM] {
        let workers = fleet(size);
        let all = decided(&workers);
        assert!(all.trace().drew_first(), "{size} workers, each available");
        assert_eq!(all.pick().expect("a pick").targets.len(), 2);

        workers[size - 1].set_status(WorkerStatus::NotReady);
        let one_out = decided(&workers);
        assert_eq!(
            one_out.trace().drew_first(),
            size >= DRAW_FROM,
            "{size} workers, one not available"
        );
        assert_eq!(one_out.pick().expect("a pick").targets.len(), 2);
    }
}

/// The floor a pick of two draws from, by its number: seven workers are
/// written whole and read by the pass, and of eight the two drawn alone
/// are written.
#[test]
fn a_pick_of_two_draws_over_eight_workers_and_not_over_seven() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let engine = spec.engine();
    let live = Live::at(engine, &clock, Instant::now(), None, None);
    let cx = Constraints::default();
    let mut facts = single();
    facts.now_micros = live.now_micros();

    let seven = fleet(7);
    let pool = PoolRecords::new(&seven, &spec, &clock);
    let unsaid = pool.facts(&live, false);
    assert!(engine.drawn_for(&facts, &unsaid, pool.len()).is_none());
    let whole = decide(engine, &pool, &seven, &live, &mut single(), &cx);
    assert!(!whole.trace().drew_first(), "seven workers: the pass");
    assert_eq!(whole.pick().expect("a pick").targets.len(), 2);

    let eight = fleet(8);
    let pool = PoolRecords::new(&eight, &spec, &clock);
    let unsaid = pool.facts(&live, false);
    let drawn = engine.drawn_for(&facts, &unsaid, pool.len());
    let written = Cell::new(0);
    let mut write = |fixed: &Candidate, row: &mut Candidate| {
        written.set(written.get() + 1);
        *row = *fixed;
        live.write(eight[fixed.position as usize].as_ref(), row);
    };
    let mut rows = vec![Candidate::default(); pool.len()];
    let decision = engine.decide_drawn(
        drawn.expect("a pick of two over eight workers is its draw's"),
        &facts,
        &unsaid,
        pool.fixed(),
        &mut rows,
        &mut write,
        &cx,
        &mut Scratch::default(),
    );
    assert_eq!(decision.pick().expect("a pick").targets.len(), 2);
    assert_eq!(written.get(), 2, "two records of 8");
    let drew = decide(engine, &pool, &eight, &live, &mut single(), &cx);
    assert!(drew.trace().drew_first(), "eight workers: a draw");
}

/// The model the pick-of-two profile is bound to beside another profile
/// ([`pair_alone_and_beside`]).
const PAIR_MODEL: &str = "pair-model";

/// The pick-of-two preset alone, and its profile bound to [`PAIR_MODEL`]
/// beside the least-load preset: under the draw's floor of every profile
/// the second document asks no draw, so its decision is the whole pass's.
fn pair_alone_and_beside() -> (AppliedSpec, AppliedSpec) {
    pair_filtered_by(None)
}

/// [`pair_alone_and_beside`] with `filter` behind the pick-of-two profile's
/// own filters, in both documents.
fn pair_filtered_by(filter: Option<FilterSpec>) -> (AppliedSpec, AppliedSpec) {
    let mut alone = PolicySpec::preset("power_of_two").expect("a built-in preset");
    let mut mixed = PolicySpec::preset("least_load").expect("a built-in preset");
    let mut bound = alone.bindings[0].clone();
    let named = bound.profile.replace("pair".to_owned());
    let pair = alone
        .profiles
        .get_mut(&named.expect("a binding of one profile"))
        .expect("the profile it names");
    pair.filter.extend(filter);
    mixed.profiles.insert("pair".to_owned(), pair.clone());
    PAIR_MODEL.clone_into(&mut bound.matcher.model);
    mixed.bindings.insert(0, bound);
    (applied_of(alone), applied_of(mixed))
}

/// Under the draw's floor of every profile a pick of two is its draw's only
/// where no filter refuses a row. A request that names a worker tried or
/// excluded, carries a session key, is held to places or stands under a
/// veto of its host is written whole, and its targets are the whole pass's
/// for that seed: the ones the same profile hands out beside another kind,
/// where no draw is asked. A request that refuses nothing draws, and hands
/// out the same.
#[test]
fn a_request_that_refuses_a_row_is_the_whole_pass_under_the_floor_to_the_seed() {
    let (alone, mixed) = pair_alone_and_beside();
    let clock = DecisionClock::new();
    for size in [WRITTEN_PAIR_FROM, WRITTEN_PAIR_FROM + 1, DRAW_FROM - 1] {
        let workers = fleet(size);
        let ids: Vec<WorkerId> = (0..2).map(|at| WorkerId::from_address(&url(at))).collect();
        let decided = |spec: &AppliedSpec, facts: RequestFacts<'_>, cx: &Constraints<'_>| {
            let pool = PoolRecords::new(&workers, spec, &clock);
            let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
            let mut facts = facts;
            decide(spec.engine(), &pool, &workers, &live, &mut facts, cx)
        };
        let plain = Constraints::default();
        let excluded = Constraints {
            excluded: &ids[..1],
            ..Constraints::default()
        };
        let places = Constraints {
            places: Some(&[1, 2, 5]),
            ..Constraints::default()
        };
        let vetoed = Constraints {
            stalled: &[0],
            ..Constraints::default()
        };
        for seed in 0..64u64 {
            let fresh = RequestFacts { seed, ..single() };
            let retry = |tried: usize| RequestFacts {
                attempt: 1,
                tried: &ids[..tried],
                ..fresh
            };
            let keyed = RequestFacts {
                session_key: crate::engine::Identity::hashed(seed | 1),
                ..fresh
            };
            let cases = [
                ("nothing refused", fresh, &plain, true),
                ("one tried", retry(1), &plain, false),
                ("two tried", retry(2), &plain, false),
                ("a session key", keyed, &plain, false),
                ("one excluded", fresh, &excluded, false),
                ("held to places", fresh, &places, false),
                ("a veto of the host", fresh, &vetoed, false),
            ];
            for (case, facts, cx, draws) in cases {
                let at = format!("{size} workers, seed {seed}, {case}");
                let got = decided(&alone, facts, cx);
                let beside = RequestFacts {
                    model: PAIR_MODEL,
                    ..facts
                };
                let want = decided(&mixed, beside, cx);
                assert_eq!(got.trace().drew_first(), draws, "{at}: the road");
                assert!(!want.trace().drew_first(), "{at}: the pass alone");
                assert_eq!(targets(&got), targets(&want), "{at}: the pass's pick");
                assert_eq!(targets(&got).len(), 2, "{at}: a pick and its alternate");
            }
        }
    }
}

/// A retry over nine workers that names one of them tried leads evenly
/// among the other eight: the whole pass decides it, by its own law. The
/// case holds that law alone: it passes with the floor of the draw and
/// without it. The road and the pick of such a retry are held by the case
/// of a request that refuses a row.
#[test]
fn a_retry_under_the_floor_leads_evenly_among_the_workers_not_tried() {
    const SEEDS: u32 = 16_384;
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(WRITTEN_PAIR_FROM + 1);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
    let (tried, cx) = ([WorkerId::from_address(&url(0))], Constraints::default());
    let mut leads = [0u32; WRITTEN_PAIR_FROM + 1];
    for seed in 0..u64::from(SEEDS) {
        let mut facts = RequestFacts {
            seed,
            attempt: 1,
            tried: &tried,
            ..single()
        };
        let decision = decide(spec.engine(), &pool, &workers, &live, &mut facts, &cx);
        leads[picked(&decision).0 as usize] += 1;
    }
    assert_eq!(leads[0], 0, "the tried worker leads no decision");
    // An even lead is one in eight: 12.5 percent, here within 0.8.
    let band = SEEDS * 117 / 1000..=SEEDS * 133 / 1000;
    for (at, count) in leads.iter().enumerate().skip(1) {
        assert!(band.contains(count), "worker {at} leads {count} of {SEEDS}");
    }
}

/// Beside a profile of another kind a pick of two keeps the floor every
/// profile draws from: eight workers are written whole and read by the pass
/// under either profile, as before a pick of two drew there, and no list of
/// that size is asked whether it stands. A document of pick-of-two profiles
/// alone draws over them.
#[test]
fn a_pick_of_two_beside_another_profile_draws_from_the_floor_of_every_profile() {
    let ((alone, mixed), clock) = (pair_alone_and_beside(), DecisionClock::new());
    let decided = |spec: &AppliedSpec, model: &'static str, size: usize| {
        let workers = fleet(size);
        let pool = PoolRecords::new(&workers, spec, &clock);
        let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
        let mut facts = RequestFacts { model, ..single() };
        let cx = Constraints::default();
        decide(spec.engine(), &pool, &workers, &live, &mut facts, &cx)
    };
    assert!(decided(&alone, MODEL, 8).trace().drew_first());
    assert!(alone.engine().asks_standing(8));
    for model in [PAIR_MODEL, MODEL] {
        let whole = decided(&mixed, model, 8);
        assert!(!whole.trace().drew_first(), "{model}: the pass");
        assert!(whole.pick().is_some(), "{model}: a pick");
    }
    assert!(!mixed.engine().asks_standing(8));
    assert!(decided(&mixed, PAIR_MODEL, DRAW_FROM).trace().drew_first());
}

/// A pick of two whose profile holds a filter the engine does not fold into
/// its one test of a row, here the room rule, keeps the floor every profile
/// draws from, though every profile of its document draws two. Such a
/// filter may refuse a row of a list that stands under a request that
/// refuses none: the pass draws among the rows it keeps and a draw among
/// all of them, so a draw whose two rows miss the refused one would hand
/// out another pick than the pass. Under that floor the decision is written
/// whole and its targets are the whole pass's for every seed, the ones the
/// same profile hands out beside another kind, and no list of that size is
/// asked whether it stands. From the floor on the profile draws, as before.
#[test]
fn a_pick_of_two_with_a_filter_of_another_kind_draws_from_the_floor_of_every_profile() {
    let rule = FilterSpec::Params(FilterStage::HasRoom(HasRoomParams {
        used: [("load.waiting_uncached_tokens".to_owned(), 1.0)].into(),
        capacity: Capacity::Fixed(1_000.0),
        headroom: 0.0,
    }));
    let ((alone, mixed), clock) = (pair_filtered_by(Some(rule)), DecisionClock::new());
    let decided = |spec: &AppliedSpec,
                   model: &'static str,
                   seed: u64,
                   workers: &Vec<Arc<dyn Worker>>,
                   loads: &Arc<LoadSnapshot>| {
        let pool = PoolRecords::new(workers, spec, &clock);
        let live = Live::at(spec.engine(), &clock, Instant::now(), Some(loads), None);
        let mut facts = RequestFacts {
            model,
            seed,
            ..single()
        };
        let cx = Constraints::default();
        decide(spec.engine(), &pool, workers, &live, &mut facts, &cx)
    };
    for size in [WRITTEN_PAIR_FROM, WRITTEN_PAIR_FROM + 1, DRAW_FROM - 1] {
        let workers = fleet(size);
        // The first worker holds more than the rule's capacity: the one row
        // the rule refuses, of a list whose every worker is available.
        let mut queued = vec![50; size];
        queued[0] = 9_000;
        let (loads, _) = reports(&queued, Duration::from_millis(5));
        let (mut drew, mut differ, mut no_room) = (0, 0, 0);
        for seed in 0..64 {
            let got = decided(&alone, MODEL, seed, &workers, &loads);
            let want = decided(&mixed, PAIR_MODEL, seed, &workers, &loads);
            assert!(!want.trace().drew_first(), "{size} workers: the pass");
            assert_eq!(targets(&want).len(), 2, "{size} workers: two targets");
            drew += usize::from(got.trace().drew_first());
            differ += usize::from(targets(&got) != targets(&want));
            no_room += targets(&got).iter().filter(|target| target.0 == 0).count();
        }
        assert_eq!(
            (drew, differ, no_room),
            (0, 0, 0),
            "{size} workers, of 64 seeds: decisions drawn first, decisions whose targets are \
             not the whole pass's, targets with no room"
        );
    }
    assert!(!alone.engine().asks_standing(WRITTEN_PAIR_FROM));
    let workers = fleet(DRAW_FROM);
    let (loads, _) = reports(&[50; DRAW_FROM], Duration::from_millis(5));
    let from_the_floor = decided(&alone, MODEL, 7, &workers, &loads);
    assert!(from_the_floor.trace().drew_first(), "{DRAW_FROM} workers");
}

/// Under the draw's floor of every profile a pick of two over a list that
/// stands is its draw's whatever its workers report and hold in flight, and
/// the draw hands out the targets of the whole write for every seed: the
/// two rows drawn are compared as the pass compares them.
#[test]
fn a_pick_of_two_under_the_floor_hands_out_the_whole_writes_targets_over_loads_that_differ() {
    let preset = |name: &str| PolicySpec::preset(name).expect("a built-in preset");
    let mut count_first = preset("power_of_two");
    for profile in count_first.profiles.values_mut() {
        profile.pick = PickSpec::from("power-of-two-count-first");
    }
    for size in [WRITTEN_PAIR_FROM, WRITTEN_PAIR_FROM + 1, DRAW_FROM - 1] {
        let workers = fleet(size);
        for (at, worker) in workers.iter().enumerate() {
            for _ in 0..at % 4 {
                worker.increment_load();
            }
        }
        let queued: Vec<i32> = (0..size)
            .map(|at| ((at * 37 + 11) % 97) as i32 * 40)
            .collect();
        let (loads, _) = reports(&queued, Duration::from_millis(5));
        for (name, document) in [
            ("power_of_two", preset("power_of_two")),
            ("power-of-two-count-first", count_first.clone()),
        ] {
            for converted in [false, true] {
                let draws =
                    assert_drawn_as_written(&document, &workers, &loads, None, 400, converted);
                assert_eq!(draws, 400, "{name}, {size} workers, each available");
            }
        }
    }
}

#[test]
fn a_pick_of_two_writes_the_records_of_the_two_workers_it_draws() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(512);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
    let cx = Constraints::default();
    let mut facts = single();
    facts.now_micros = live.now_micros();
    let unsaid = pool.facts(&live, false);
    let drawn = spec
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .expect("a pick of two over a large pool is its draw's");
    let written = Cell::new(0);
    let mut write = |fixed: &Candidate, row: &mut Candidate| {
        written.set(written.get() + 1);
        *row = *fixed;
        live.write(workers[fixed.position as usize].as_ref(), row);
    };
    let mut rows = vec![Candidate::default(); pool.len()];
    let decision = spec.engine().decide_drawn(
        drawn,
        &facts,
        &unsaid,
        pool.fixed(),
        &mut rows,
        &mut write,
        &cx,
        &mut Scratch::default(),
    );
    assert_eq!(decision.pick().expect("a pick").targets.len(), 2);
    assert_eq!(written.get(), 2, "two records of 512");

    // A pick of two draws under the draw's size too, from its own floor
    // on: such a pool's records are written as they are drawn.
    let few = fleet(DRAW_FROM - 1);
    let pool = PoolRecords::new(&few, &spec, &clock);
    let unsaid = pool.facts(&live, false);
    assert!(spec
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .is_some());

    // Under that floor every record is written: the pool is small.
    let fewer = fleet(WRITTEN_PAIR_FROM - 1);
    let pool = PoolRecords::new(&fewer, &spec, &clock);
    let unsaid = pool.facts(&live, false);
    assert!(spec
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .is_none());

    // A profile that scores every worker is no draw's.
    let scores = applied("least_load");
    let pool = PoolRecords::new(&workers, &scores, &clock);
    let unsaid = pool.facts(&live, false);
    assert!(scores
        .engine()
        .drawn_for(&facts, &unsaid, pool.len())
        .is_none());
}
