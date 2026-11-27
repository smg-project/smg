use super::*;

/// The fleets a draw is held against the whole pass over: every row
/// ready, one in seventeen not, every second not, one row ready, none.
const REFUSED: [fn(u32) -> bool; 5] = [
    |_| false,
    |i| i % 17 == 5,
    |i| i % 2 == 1,
    |i| i != 6,
    |_| true,
];

#[test]
fn a_draw_hands_out_only_rows_the_whole_pass_keeps() {
    let mut drew = 0u32;
    let mut widened = 0u32;
    const NAMES: [&str; 5] = [
        "passthrough",
        "random",
        "round_robin",
        "power_of_two",
        "power_of_two/requests",
    ];
    let mut drew_by = [0u32; NAMES.len()];
    for (which, name) in NAMES.into_iter().enumerate() {
        // Round robin turns a rotation at each decision: the pass that
        // answers for it runs on a twin, so neither turns the other's.
        let (engine, twin) = (preset(name), preset(name));
        for rows in [8u32, 9, 17, 64, 257] {
            for (fleet, refused) in REFUSED.into_iter().enumerate() {
                let pool = pool_of(rows, refused);
                for seed in 0..40u64 {
                    // Every third request already tried a row: the
                    // soft filter is asked of the rows drawn too.
                    let tried = [pool[(seed as usize * 7) % pool.len()].id];
                    let facts = RequestFacts {
                        tried: if seed % 3 == 0 { &tried } else { &[] },
                        ..request(seed ^ u64::from(rows) << 32)
                    };
                    let untried = |c: &&Candidate| !facts.tried.contains(&c.id);
                    let ready = || pool.iter().filter(|c| c.health.healthy);
                    // What the pass keeps: the ready rows not tried, or
                    // every ready row when that leaves none.
                    let kept: Vec<u32> = if ready().any(|c| untried(&c)) {
                        ready().filter(untried).map(|c| c.position).collect()
                    } else {
                        ready().map(|c| c.position).collect()
                    };
                    let got = drawn(&engine, &facts, &pool);
                    let want = whole_pass(&twin, &facts, &pool);
                    let at = format!("{name}, {rows} rows, fleet {fleet}, seed {seed}");
                    assert_eq!(got.trace().seed, facts.seed, "{at}: the request's seed");
                    if kept.is_empty() {
                        assert_eq!(got, want, "{at}: a shed is the whole pass's own");
                        continue;
                    }
                    let pick = got.pick().unwrap_or_else(|| panic!("{at}: a pick"));
                    if pick.trace.drew_first() {
                        drew += 1;
                        drew_by[which] += 1;
                        assert_eq!(pick.trace.survivors.get(), None, "{at}: no count");
                    } else {
                        widened += 1;
                        assert_eq!(pick.trace.survivors.get(), Some(kept.len() as u32), "{at}");
                    }
                    let (got, want) = (positions(&got), positions(&want));
                    assert!(
                        got.iter().all(|row| kept.contains(row)),
                        "{at}: {got:?} holds a row a filter refuses (kept {kept:?})"
                    );
                    let mut once = got.clone();
                    once.sort_unstable();
                    once.dedup();
                    assert_eq!(once.len(), got.len(), "{at}: a row handed out twice");
                    assert!(got.len() <= want.len(), "{at}: {got:?} against {want:?}");
                    if fleet == 0 && facts.tried.is_empty() {
                        assert_eq!(got.len(), want.len(), "{at}: every target, no row refused");
                    }
                    if name == "passthrough" {
                        assert_eq!(got, want[..got.len()], "{at}: row for row");
                    }
                }
            }
        }
    }
    assert!(
        drew > 1_000 && widened > 100,
        "{drew} drew, {widened} widened"
    );
    // Each of the five draws, the requests form of `power_of_two` too.
    assert!(
        drew_by.iter().all(|&drew| drew > 100),
        "{drew_by:?} drew, by picker"
    );
}

#[test]
fn a_pool_under_the_draws_size_is_read_whole() {
    for name in ["passthrough", "random", "round_robin", "power_of_two"] {
        let engine = preset(name);
        let facts = request(3);
        let under = pool_of(DRAW_FROM as u32 - 1, |_| false);
        let trace = *drawn(&engine, &facts, &under).trace();
        assert!(!trace.drew_first(), "{name}");
        assert_eq!(trace.survivors.get(), Some(DRAW_FROM as u32 - 1), "{name}");
        let at = pool_of(DRAW_FROM as u32, |_| false);
        let trace = *drawn(&engine, &facts, &at).trace();
        assert!(trace.drew_first(), "{name}");
        assert_eq!(
            trace.survivors.get(),
            None,
            "{name}: a draw counts no survivor"
        );
    }
}

/// A picker that draws two rows is asked to draw before the pool's records
/// are written from its own floor on, under the draw's size, and every
/// other picker of the written road from the draw's size. Over a pool
/// between the two the draw is the whole pass's rule: with every row ready
/// it hands out the pass's targets in the pass's order, writes two records
/// and counts no survivor; with rows refused it hands out rows the pass
/// keeps, as many as the pass, a draw that lands on a refused row gives up
/// at that row, and the pass behind it hands out its own pick of the seed;
/// and a pool that keeps one row or none is the pass's own.
#[cfg(feature = "in-process")]
#[test]
fn a_pick_of_two_draws_over_a_small_pool_before_its_records_are_written() {
    const PAIRS: [&str; 2] = ["power_of_two", "power_of_two/requests"];
    let (shared, cx) = (PoolFacts::default(), Constraints::default());
    let sizes = [
        WRITTEN_PAIR_FROM - 1,
        WRITTEN_PAIR_FROM,
        DRAW_FROM - 1,
        DRAW_FROM,
    ];
    for name in ["passthrough", "random", "round_robin"]
        .into_iter()
        .chain(PAIRS)
    {
        let from = if PAIRS.contains(&name) {
            WRITTEN_PAIR_FROM
        } else {
            DRAW_FROM
        };
        let engine = preset(name);
        for rows in sizes {
            let asked = engine.drawn_for(&request(3), &shared, rows).is_some();
            assert_eq!(asked, rows >= from, "{name} over {rows} rows");
        }
    }
    let (mut same, mut gave_up) = (0u32, 0u32);
    for name in PAIRS {
        let engine = preset(name);
        // The decision over the same rows handed written (the whole pass,
        // under the draw's size), the draw's over rows written as it asks
        // about them, and the records it wrote.
        let both = |facts: &RequestFacts<'_>, pool: &[Candidate]| {
            let want = drawn(&engine, facts, pool);
            let draw = engine.drawn_for(facts, &shared, pool.len());
            let mut written = 0u32;
            let mut write = |fixed: &Candidate, row: &mut Candidate| {
                written += 1;
                *row = *fixed;
            };
            let mut buffer = vec![Candidate::default(); pool.len()];
            let got = engine.decide_drawn(
                draw.expect("a pair's draw"),
                facts,
                &shared,
                pool,
                &mut buffer,
                &mut write,
                &cx,
                &mut Scratch::new(),
            );
            (got, want, written)
        };
        for rows in WRITTEN_PAIR_FROM..DRAW_FROM {
            for (fleet, refused) in REFUSED.into_iter().enumerate() {
                let pool = pool_of(rows as u32, refused);
                // The same fleet with a count of its own on every row: no
                // two rows tie.
                let mut loaded = pool.clone();
                for row in &mut loaded {
                    row.in_flight = row.position;
                }
                for seed in 0..64u64 {
                    let facts = request(seed ^ (rows as u64) << 32);
                    let at = format!("{name}, {rows} rows, fleet {fleet}, seed {seed}");
                    let (got, want, written) = both(&facts, &pool);
                    let kept = pool.iter().filter(|c| c.health.healthy).count();
                    if kept < 2 {
                        assert_eq!(positions(&got), positions(&want), "{at}: the pass's own");
                        assert_eq!(got.pick().is_some(), kept == 1, "{at}");
                        continue;
                    }
                    let (got_rows, want_rows) = (positions(&got), positions(&want));
                    assert_eq!(got_rows.len(), 2, "{at}: the pick and its alternate");
                    assert_eq!(want_rows.len(), 2, "{at}: the pass hands out two");
                    assert_ne!(got_rows[0], got_rows[1], "{at}: a row handed out twice");
                    assert!(
                        got_rows.iter().all(|&row| !refused(row)),
                        "{at}: {got_rows:?} holds a row a filter refuses"
                    );
                    if fleet != 0 {
                        // A draw that stands was refused no row and wrote
                        // its two alone. One that lands on a refused row
                        // gives up there, and the pass behind it reads the
                        // stream as it came: its pick of the seed is the
                        // one it makes with no draw before it.
                        if got.trace().survivors.get().is_none() {
                            assert_eq!(written, 2, "{at}: two records of {rows}");
                        } else {
                            assert_eq!(got_rows, want_rows, "{at}: the pass's own pick");
                            gave_up += 1;
                        }
                        continue;
                    }
                    // Every row ready: the two rows drawn are the pass's
                    // two, in the pass's order. Tied, as here, both order
                    // them by the request's own seed.
                    assert_eq!(written, 2, "{at}: two records of {rows}");
                    assert_eq!(got.trace().survivors.get(), None, "{at}: no count");
                    let counted = want.trace().survivors.get();
                    assert_eq!(counted, Some(rows as u32), "{at}: the pass counts");
                    assert_eq!(got_rows, want_rows, "{at}: the pass's two rows");
                    // And where they do not tie the better leads in both.
                    let (got, want, _) = both(&facts, &loaded);
                    assert_eq!(positions(&got), positions(&want), "{at}: the pass's order");
                    same += 1;
                }
            }
        }
    }
    let sizes = (DRAW_FROM - WRITTEN_PAIR_FROM) as u32;
    assert_eq!(same, 2 * sizes * 64, "the decisions over a ready pool");
    assert!(gave_up > 0, "no draw gave up at a refused row");
}

#[test]
fn a_profile_with_a_stage_that_is_not_row_wise_never_draws() {
    /// A filter that keeps every row and does not say its word on one
    /// row is its whole word: a plugin's filter is such a one.
    #[derive(Debug)]
    struct Whole;
    impl Filter for Whole {
        fn name(&self) -> &'static str {
            "whole"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
    }
    // `least-requests` does not say it may be handed drawn rows.
    let engine = Engine::new(
        vec![Profile {
            name: "two".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            }],
            picker: Box::new(PowerOfTwo),
        }],
        vec![single_binding()],
        1,
        Needs::default(),
    );
    let pool = pool_of(64, |_| false);
    let trace = *drawn(&engine, &request(1), &pool).trace();
    assert!(!trace.drew_first());
    assert_eq!(trace.survivors.get(), Some(64));
    // Nor does a picker that compares every survivor.
    let trace = *drawn(&preset("least_load"), &request(1), &pool).trace();
    assert!(!trace.drew_first());
    // Nor does a filter that is not row-wise, under a picker that draws
    // and at the first size it draws from.
    let engine = Engine::new(
        vec![Profile {
            name: "whole".into(),
            filters: vec![Box::new(Healthy), Box::new(Whole)],
            scorers: Vec::new(),
            picker: Box::new(Passthrough),
        }],
        vec![single_binding()],
        1,
        Needs::default(),
    );
    let pool = pool_of(DRAW_FROM as u32, |_| false);
    let trace = *drawn(&engine, &request(1), &pool).trace();
    assert!(!trace.drew_first());
    assert_eq!(trace.survivors.get(), Some(DRAW_FROM as u32));
}

#[test]
fn the_drawing_pickers_keep_their_law_over_the_rows_the_pass_keeps() {
    // Thirteen rows kept of sixteen, equal loads: the pick is uniform
    // over them, under `random` and under a tie of the two drawn.
    let pool: Vec<Candidate> = pool_of(16, |i| [3, 7, 11].contains(&i))
        .into_iter()
        .map(|c| Candidate { in_flight: 0, ..c })
        .collect();
    const DECISIONS: u32 = 26_000;
    for name in ["random", "power_of_two"] {
        let engine = preset(name);
        let mut scratch = Scratch::new();
        let mut first = [0u32; 16];
        let mut any = [0u32; 16];
        for seed in 0..u64::from(DECISIONS) {
            let decision = engine.decide(
                &request(seed),
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut scratch,
            );
            let rows = positions(&decision);
            first[rows[0] as usize] += 1;
            for row in rows {
                any[row as usize] += 1;
            }
        }
        // 2,000 a row expected, a standard deviation of 43: five of them.
        for row in 0..16usize {
            if [3, 7, 11].contains(&row) {
                assert_eq!(any[row], 0, "{name}: a refused row handed out");
            } else {
                assert!(
                    first[row].abs_diff(DECISIONS / 13) < 220,
                    "{name}: {first:?}"
                );
            }
        }
        let (low, high) = (any.iter().filter(|n| **n > 0).min(), any.iter().max());
        let (low, high) = (*low.expect("a row"), *high.expect("a row"));
        assert!(
            high - low < high / 10,
            "{name}: every target, by row: {any:?}"
        );
    }
}

#[test]
fn round_robin_turns_the_pool_and_shares_the_refused_rows_turns() {
    let engine = preset("round_robin");
    let mut scratch = Scratch::new();
    let mut turn = |pool: &[Candidate], model: &'static str| {
        let facts = RequestFacts {
            model,
            ..request(0)
        };
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut scratch,
        );
        positions(&decision)
    };
    // No row refused: a turn of the pool visits every row once, in the
    // host's order, and the rows behind the pick are its alternates.
    let ready = pool_of(32, |_| false);
    let start = turn(&ready, "a")[0];
    for step in 1..=32u32 {
        let rows = turn(&ready, "a");
        let want: Vec<u32> = (0..4).map(|k| (start + step + k) % 32).collect();
        assert_eq!(rows, want, "step {step}");
    }
    // Another model's pool turns its own rotation: its decisions take
    // no turn of the first.
    let other = turn(&ready, "b")[0];
    assert_eq!(turn(&ready, "a")[0], (start + 33) % 32);
    assert_eq!(turn(&ready, "b")[0], (other + 1) % 32);
    // Three rows refused, twenty turns of the pool: a kept row takes its
    // own twenty turns and an even share of the refused rows' sixty.
    let pool = pool_of(32, |i| [3, 7, 11].contains(&i));
    let mut picks = [0u32; 32];
    for _ in 0..32 * 20 {
        picks[turn(&pool, "c")[0] as usize] += 1;
    }
    for (row, &n) in picks.iter().enumerate() {
        if [3, 7, 11].contains(&row) {
            assert_eq!(n, 0, "a refused row picked");
        } else {
            assert!((22..=23).contains(&n), "row {row}: {picks:?}");
        }
    }
}

/// Under a sessions rule that leaves a new session to the profile
/// (`delegate`), round robin turns one rotation for every request it
/// places. A request with no session value, a new session and a session
/// at its cap take the pool's rows one after the other: under the
/// draw's floor, at it and above it, and under a profile whose plain
/// request is not the draw's, since its scorer reads the rows together.
/// A session's pass counted its survivors, and a turn that lands on a
/// refused row is left to the stages. So is a request its host holds to
/// places, as its plain request draws nothing. Where a soft filter
/// refuses every row and steps aside, the draw keeps nobody for either
/// kind of request, and both take the survivors' rotation. On a fresh
/// engine each, a new session is decided as a request with no session
/// value is: by the draw where the turn's row is kept, its trace the
/// draw's with the pass's count, and by the stages where a filter, a
/// host's filter or the host's veto refuses that row. A row the pass
/// refused is not taken at the turn, whatever its filter says of it
/// there. A host's picker that takes turns and orders nothing of its
/// draw is the stages' too.
/// `random` takes no turn: the stages alone place its new session.
///
/// The classic router asks one function for all three kinds of request,
/// and so steps the same counters
/// (`model_gateway/src/policies/round_robin.rs:142-167`): for a request
/// with no session value at `model_gateway/src/policies/registry.rs:471`,
/// for a new session at `:581` and for a session over its cap at `:552`.
#[test]
fn round_robin_under_delegate_turns_one_rotation_for_every_request() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let scored = || Profile {
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        ..round_robin_profile()
    };
    assert!(draws_first(&round_robin_profile()) && !draws_first(&scored()));
    let sizes = [DRAW_FROM - 1, DRAW_FROM, 64].map(|rows| (rows, false));
    for profile in [round_robin_profile as fn() -> Profile, scored] {
        // The last pool is held to its every place.
        for (rows, held) in sizes.into_iter().chain([(DRAW_FROM, true)]) {
            let engine = session_engine_of(profile(), SESSIONS);
            let pool = pool_of(rows as u32, |_| false);
            let every: Vec<u32> = (0..rows as u32).collect();
            let mut scratch = Scratch::new();
            let mut turn = |pool: &[Candidate], key: Option<u64>| {
                let mut facts = request(0);
                if let Some(key) = key {
                    facts.session_key = Identity::hashed(key);
                }
                let shared = PoolFacts::default();
                let cx = Constraints {
                    places: held.then_some(&every[..]),
                    ..Constraints::default()
                };
                let decision = engine.decide(&facts, &shared, pool, &cx, &mut scratch);
                let pick = decision.pick().expect("a pick");
                let target = *pick.targets.first().expect("a target");
                if key.is_some() {
                    assert!(!pick.trace.drew_first(), "a session's pass counted");
                    let lease = Lease {
                        session: facts.session(),
                        session_verdict: pick.session_verdict,
                        ..Lease::default()
                    };
                    engine.on_dispatched(&lease, &target, Side::Single, 0);
                }
                target.position as usize
            };
            let first = turn(&pool, None);
            let pinned = turn(&pool, Some(1));
            let mut places = vec![first, pinned, turn(&pool, None)];
            // The session's worker carries the rule's cap of its requests.
            let mut capped = pool.clone();
            capped[pinned].session_in_flight = SESSIONS.inflight_cap;
            places.push(turn(&capped, Some(1)));
            places.push(turn(&pool, None));
            places.push(turn(&pool, Some(2)));
            let in_turn: Vec<usize> = (0..6).map(|step| (first + step) % rows).collect();
            assert_eq!(places, in_turn, "{rows} rows, held {held}");
            // A turn that lands on a refused row is left to the stages,
            // and every other turn of the pool is taken where it is asked.
            let ailing = pool_of(rows as u32, |row| row == 3);
            let keys = 10..10 + rows as u64;
            let lands: Vec<usize> = keys.map(|key| turn(&ailing, Some(key))).collect();
            assert!(!lands.contains(&3), "{rows} rows, held {held}: {lands:?}");
            let asked = draws_first(&profile()) && rows >= DRAW_FROM && !held;
            let taken = |(step, &row): (usize, &usize)| {
                !asked || [3, row].contains(&((first + 6 + step) % rows))
            };
            assert!(lands.iter().enumerate().all(taken), "{rows}: {lands:?}");
        }
    }
    // A soft filter that refuses every row steps aside, and the pass
    // keeps them all. A request with no session value is refused at its
    // draw there, which asks every filter of the row it lands on, and
    // falls to the survivors' rotation. A new session falls to it too.
    let roomless = || Profile {
        filters: vec![Box::new(Healthy), Box::new(HasRoom)],
        ..round_robin_profile()
    };
    assert!(draws_first(&roomless()));
    let engine = session_engine_of(roomless(), SESSIONS);
    let pool = pool_of(DRAW_FROM as u32, |_| false);
    let mut scratch = Scratch::new();
    let mut turn = |key: Option<u64>| {
        let mut facts = request(0);
        if let Some(key) = key {
            facts.session_key = Identity::hashed(key);
        }
        let (shared, cx) = (PoolFacts::default(), Constraints::default());
        let decision = engine.decide(&facts, &shared, &pool, &cx, &mut scratch);
        positions(&decision)[0] as usize
    };
    let first = turn(None);
    let keyed = |step: u64| (step % 2 == 1).then_some(step);
    let places: Vec<usize> = (1..6).map(|step| turn(keyed(step))).collect();
    let in_turn: Vec<usize> = (1..6).map(|step| (first + step) % DRAW_FROM).collect();
    assert_eq!(places, in_turn, "a soft filter refuses every row");
    // A fresh engine's request with no session value and a fresh
    // engine's new session are decided alike. Where the turn's row is
    // kept, the session's targets are that request's draw, and its
    // trace is the draw's with the count of the session's own pass.
    // Where the turn's row is refused, the stages place both on the same
    // rows: by a filter of the fold, by a host's filter, by the host's
    // veto, with every row by a soft filter of the fold or of a host,
    // and with every row under the host's veto, which the rule lifts.
    let hosted = || Profile {
        filters: vec![Box::new(Hosted(Box::new(Healthy)))],
        ..round_robin_profile()
    };
    let hosted_roomless = || Profile {
        filters: vec![
            Box::new(Hosted(Box::new(Healthy))),
            Box::new(Hosted(Box::new(HasRoom))),
        ],
        ..round_robin_profile()
    };
    assert!(draws_first(&hosted()) && draws_first(&hosted_roomless()));
    // A host's filter whose answer moves between two asks of one row, as
    // a filter that reads what another decision moves can: it refuses
    // `row` the first time it is asked of it, and keeps it after.
    #[derive(Debug)]
    struct Flips {
        row: u32,
        asked: AtomicBool,
    }
    impl Filter for Flips {
        fn name(&self) -> &'static str {
            "flips"
        }

        fn row_wise(&self) -> bool {
            true
        }

        fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
            c.position != self.row || self.asked.swap(true, Ordering::Relaxed)
        }
    }
    let shared = PoolFacts {
        generation: 7,
        ..PoolFacts::default()
    };
    let decide = |profile: Profile, pool: &[Candidate], cx: &Constraints<'_>, keyed: bool| {
        let engine = session_engine_of(profile, SESSIONS);
        let mut facts = request(5);
        if keyed {
            facts.session_key = Identity::hashed(1);
        }
        let decision = engine.decide(&facts, &shared, pool, cx, &mut Scratch::new());
        (positions(&decision), *decision.trace())
    };
    let free = Constraints::default();
    for rows in [DRAW_FROM as u32, 64] {
        let full = pool_of(rows, |_| false);
        let turn = decide(round_robin_profile(), &full, &free, false).0[0];
        // The turn's row kept, by the fold's test and by a host's filter.
        let apart = pool_of(rows, |row| row == (turn + 8) % rows);
        for profile in [round_robin_profile as fn() -> Profile, hosted] {
            let (plain, drew) = decide(profile(), &apart, &free, false);
            let (keyed, counted) = decide(profile(), &apart, &free, true);
            assert!(drew.drew_first() && plain[0] == turn, "{rows}: {plain:?}");
            assert_eq!(
                (drew.made_under(), drew.seed, drew.generation),
                (Some(0), request(5).seed, 7)
            );
            let pass = Trace {
                survivors: Survivors::counted(rows - 1),
                ..drew
            };
            assert_eq!((keyed, counted), (plain, pass), "{rows} rows");
        }
        let ailing = pool_of(rows, |row| row == turn);
        let stalled = [turn];
        let vetoed = Constraints {
            stalled: &stalled,
            ..Constraints::default()
        };
        let every: Vec<u32> = (0..rows).collect();
        let doubted = Constraints {
            stalled: &every,
            ..Constraints::default()
        };
        // A profile, its pool and what the host holds the request to.
        type State<'a> = (fn() -> Profile, &'a [Candidate], &'a Constraints<'a>);
        let refused: [State<'_>; 6] = [
            (round_robin_profile, &ailing, &free),
            (hosted, &ailing, &free),
            (round_robin_profile, &full, &vetoed),
            (roomless, &full, &free),
            (hosted_roomless, &full, &free),
            (round_robin_profile, &full, &doubted),
        ];
        for (state, (profile, pool, cx)) in refused.into_iter().enumerate() {
            let (plain, whole) = decide(profile(), pool, cx, false);
            let (keyed, session) = decide(profile(), pool, cx, true);
            assert!(!whole.drew_first(), "{rows} rows, state {state}");
            assert!(!plain.is_empty(), "{rows} rows, state {state}: a pick");
            assert_eq!(keyed, plain, "{rows} rows, state {state}");
            // The stages place both requests, by one trace: it says the
            // lift where every row is under the host's veto.
            let lifted = if state == 5 {
                LiftedVeto::Liveness
            } else {
                LiftedVeto::None
            };
            assert_eq!(whole.lifted, lifted, "{rows} rows, state {state}");
            assert_eq!(session, whole, "{rows} rows, state {state}");
        }
        // The row a filter refused at the pass is no survivor, whatever
        // the filter says of it at the turn: the turn does not take it.
        let flips = Profile {
            filters: vec![Box::new(Flips {
                row: turn,
                asked: AtomicBool::new(false),
            })],
            ..round_robin_profile()
        };
        let (keyed, counted) = decide(flips, &full, &free, true);
        assert!(!keyed.contains(&turn), "{rows} rows: {keyed:?}");
        assert_eq!(counted.survivors.get(), Some(rows - 1), "{rows} rows");
    }
    // A host's picker that takes turns and hands out nothing of the
    // row it drew: the stages place its new session, as they place its
    // request with no session value.
    #[derive(Debug)]
    struct Mute;
    impl Lifecycle for Mute {}
    impl Picker for Mute {
        fn name(&self) -> &'static str {
            "mute"
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            if let Some(&row) = kept.last() {
                out.push(target_of(&candidates[row as usize]));
            }
        }

        fn draws_first(&self) -> bool {
            true
        }

        fn takes_turns(&self) -> bool {
            true
        }

        fn draw_first(
            &self,
            _: &RequestFacts<'_>,
            _: &PoolFacts<'_>,
            candidates: &[Candidate],
            keep: &mut dyn FnMut(&Candidate) -> bool,
            kept: &mut Vec<u32>,
            _: &mut Draw,
        ) -> bool {
            kept.extend(
                candidates
                    .iter()
                    .find(|row| keep(row))
                    .map(|row| row.position),
            );
            !kept.is_empty()
        }

        fn order_drawn(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[f32],
            _: TieBand,
            _: &mut Targets,
        ) {
        }
    }
    let mute = || Profile {
        picker: Box::new(Mute),
        ..round_robin_profile()
    };
    assert!(draws_first(&mute()));
    let full = pool_of(DRAW_FROM as u32, |_| false);
    let last = vec![DRAW_FROM as u32 - 1];
    assert_eq!(decide(mute(), &full, &free, false).0, last);
    assert_eq!(decide(mute(), &full, &free, true).0, last);
    // `random` takes no turn. Over a pool with a row refused its draw and
    // its pick over the survivors name other rows for one seed, and its
    // new session lands where the stages alone put it: on the picker's
    // own pick over the survivors, by the request's seed.
    let random = || Profile {
        picker: Box::new(Random),
        ..round_robin_profile()
    };
    let ailing = pool_of(DRAW_FROM as u32, |row| row == 3);
    let survivors: Vec<u32> = (0..DRAW_FROM as u32).filter(|row| *row != 3).collect();
    for seed in 0..8 {
        let facts = RequestFacts {
            session_key: Identity::hashed(1),
            ..request(seed)
        };
        let placed = drawn(&session_engine_of(random(), SESSIONS), &facts, &ailing);
        let mut alone = Targets::new();
        Random.pick(&facts, &ailing, &survivors, &[], TieBand(0.0), &mut alone);
        let alone: Vec<u32> = alone.iter().map(|target| target.position).collect();
        assert_eq!(positions(&placed), alone, "seed {seed}");
    }
}

/// A binding of the single plan decides a request of any phase, and
/// its round robin turns one rotation for both kinds of request at
/// each: a request with no session value draws the pool's turn whatever
/// its phase says, so a new session takes that turn too. A set narrowed
/// to a side of a pair alone takes none.
#[test]
fn round_robin_under_delegate_turns_one_rotation_at_every_phase_of_the_single_plan() {
    let rows = DRAW_FROM + 4;
    let pool = pool_of(rows as u32, |_| false);
    for phase in PHASES {
        let engine = session_engine_of(round_robin_profile(), SESSIONS);
        let mut scratch = Scratch::new();
        let mut turn = |key: Option<u64>| {
            let facts = RequestFacts {
                phase,
                session_key: key.map_or(Identity::NONE, Identity::hashed),
                ..request(0)
            };
            let (shared, cx) = (PoolFacts::default(), Constraints::default());
            let decision = engine.decide(&facts, &shared, &pool, &cx, &mut scratch);
            positions(&decision)[0] as usize
        };
        let first = turn(None);
        let keys = [Some(1), None, Some(2), Some(3), None];
        let places: Vec<usize> = keys.into_iter().map(&mut turn).collect();
        let in_turn: Vec<usize> = (1..6).map(|step| (first + step) % rows).collect();
        assert_eq!(places, in_turn, "{phase:?}");
    }
}

/// A claim another request of the session took between a draw's read of
/// the key and its own claim, as `decide_claimed` settles it: the
/// claimed worker leads the draw's own turn, a worker of that turn at
/// the rule's cap is dropped behind it, and with the claimed worker at
/// the cap the draw answers nothing, so the whole decision hands the
/// key on.
#[cfg(feature = "in-process")]
#[test]
fn a_draw_that_loses_its_claim_leads_with_the_claimed_worker_or_answers_nothing() {
    let rule = SessionRule {
        inflight_cap: 1,
        ..SESSIONS
    };
    let fixed = pool_of(20, |_| false);
    let other = fixed[12].id;
    // The key's requests in flight on the claimed worker, a worker of
    // the turn at the cap, and the places the draw hands out.
    let cases = [
        (0, None, Some(vec![12, 3, 4, 5])),
        (0, Some(5), Some(vec![12, 3, 4])),
        (1, None, None),
    ];
    for (in_flight, capped, placed) in cases {
        let engine = session_engine_of(round_robin_profile(), rule);
        let word = AtomicU64::new(3);
        let pool = PoolFacts {
            word: Some(&word),
            ..PoolFacts::default()
        };
        let facts = RequestFacts {
            session_key: Identity::hashed(41),
            ..request(0)
        };
        let key = facts.session();
        let drawn = engine.drawn_for(&facts, &pool, fixed.len());
        let mut rows = vec![Candidate::default(); fixed.len()];
        let mut asked = 0;
        // The other request claims behind this one's read of the key:
        // at the second row the draw asks about.
        let mut write = |head: &Candidate, row: &mut Candidate| {
            *row = *head;
            if head.id == other {
                row.session_in_flight = in_flight;
            }
            if Some(head.position) == capped {
                row.session_in_flight = 1;
            }
            asked += 1;
            if asked == 2 {
                let claim = engine.sessions.claim(key, [WorkerId(0); 2], other, 0);
                assert_eq!(claim, Claim::Made(other));
            }
        };
        let decision = engine.decide_claimed(
            drawn.expect("a draw that claims"),
            &facts,
            &pool,
            &fixed,
            &mut rows,
            &mut write,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert_eq!(decision.as_ref().map(positions), placed);
        if let Some(Decision::Pick(pick)) = &decision {
            assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        }
    }
}

/// A draw that claims dates its claim by its decision's instant, as the
/// whole decision does. The refusal of a lease issued at another instant
/// leaves the claim, and the refusal of the decision's own lease takes it
/// back.
#[cfg(feature = "in-process")]
#[test]
fn a_draw_that_claims_dates_its_claim_by_its_decisions_instant() {
    let engine = session_engine_of(round_robin_profile(), SESSIONS);
    let fixed = pool_of(20, |_| false);
    let word = AtomicU64::new(3);
    let pool = PoolFacts {
        word: Some(&word),
        ..PoolFacts::default()
    };
    let facts = RequestFacts {
        session_key: Identity::hashed(41),
        now_micros: 500,
        ..request(0)
    };
    let drawn = engine.drawn_for(&facts, &pool, fixed.len());
    let mut rows = vec![Candidate::default(); fixed.len()];
    let decision = engine.decide_claimed(
        drawn.expect("a draw that claims"),
        &facts,
        &pool,
        &fixed,
        &mut rows,
        &mut |head: &Candidate, row: &mut Candidate| *row = *head,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let pick = decision.as_ref().and_then(Decision::pick).expect("a pick");
    let claimed = pick.targets.first().expect("a target").id;
    assert_eq!(pick.session_verdict, SessionVerdict::Claimed(claimed));
    let pins = || engine.sessions().pins(facts.session());
    let lease = |issued_micros: u64| Lease {
        session: facts.session(),
        session_verdict: pick.session_verdict,
        issued_micros,
        ..Lease::default()
    };
    engine.on_released(&lease(501), ReleaseReason::CommitFailed);
    assert_eq!(pins(), Some([claimed, WorkerId(0)]), "another instant");
    engine.on_released(&lease(500), ReleaseReason::CommitFailed);
    assert_eq!(pins(), Some([WorkerId(0); 2]), "its own lease");
}

#[test]
fn a_pick_that_stands_keeps_its_turn_behind_a_run_of_refused_rows() {
    // Row 4 is kept and the twelve rows behind it are not: the walk for
    // alternates meets the bound, and the pick stands with none.
    let pool = pool_of(32, |i| (5..17).contains(&i));
    let engine = preset("round_robin");
    let mut scratch = Scratch::new();
    let mut picks = [0u32; 32];
    let mut alone = 0;
    for _ in 0..32 * 10 {
        let decision = engine.decide(
            &request(0),
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        let rows = positions(&decision);
        picks[rows[0] as usize] += 1;
        if rows == [4] {
            alone += 1;
            assert!(decision.trace().drew_first());
        }
    }
    assert_eq!(alone, 10, "row 4's own turns, one a turn of the pool");
    // Twelve refused turns a turn of the pool, shared by twenty rows.
    for (row, &n) in picks.iter().enumerate() {
        if (5..17).contains(&row) {
            assert_eq!(n, 0);
        } else {
            assert!((16..=16).contains(&n), "row {row}: {picks:?}");
        }
    }
}

#[test]
fn the_pass_behind_a_draw_reads_the_seeds_stream_past_the_draws_values() {
    // Four rows ready of sixty-four: `random` spends its bound on the
    // pick, and where all eight draws were refused the whole pass runs
    // on the stream past those eight values, never from its start.
    let pool = pool_of(64, |i| ![5, 20, 40, 60].contains(&i));
    let (engine, twin) = (preset("random"), preset("random"));
    let mut widened = 0;
    for seed in 0..200 {
        let facts = request(seed);
        let got = drawn(&engine, &facts, &pool);
        if got.trace().drew_first() {
            continue;
        }
        widened += 1;
        let mut stream = Draw::new(facts.seed);
        for _ in 0..DRAW_BOUND {
            stream.next_u64();
        }
        let past = RequestFacts {
            seed: stream.state(),
            ..facts
        };
        let want = whole_pass(&twin, &past, &pool);
        assert_eq!(positions(&got), positions(&want), "seed {seed}");
        assert_eq!(
            got.trace().seed,
            facts.seed,
            "the trace keeps the request's seed"
        );
    }
    assert!(widened > 50, "{widened} of 200 decisions widened");
}

#[test]
fn two_tied_rows_drawn_past_a_refused_one_are_ordered_on_the_stream_past_the_draws() {
    // Every second row ready and every ready row alike: the two rows
    // drawn tie, and the stream orders them from where the draws left
    // it, never from a value a draw already read.
    let pool: Vec<Candidate> = (0..64)
        .map(|i| Candidate {
            health: Health {
                healthy: i % 2 == 0,
                ..Health::default()
            },
            ..candidate(i, 0)
        })
        .collect();
    let engine = preset("power_of_two");
    let (mut past_a_refusal, mut second_first) = (0u32, 0u32);
    for seed in 0..400 {
        let facts = request(seed);
        let got = drawn(&engine, &facts, &pool);
        if !got.trace().drew_first() {
            continue;
        }
        // The two draws, replayed: each value of the stream is one row
        // asked, and the second draw passes over the first's row.
        let mut stream = Draw::new(facts.seed);
        let mut asked = 0u32;
        let mut draw_one = |except: Option<usize>| loop {
            let mut row = stream.below(pool.len() - usize::from(except.is_some()));
            if except.is_some_and(|taken| row >= taken) {
                row += 1;
            }
            asked += 1;
            if pool[row].health.healthy {
                break row;
            }
        };
        let first = draw_one(None);
        let second = draw_one(Some(first));
        if asked == 2 {
            continue;
        }
        past_a_refusal += 1;
        let past = RequestFacts {
            seed: stream.state(),
            ..facts
        };
        let kept = [first as u32, second as u32];
        let mut want = Targets::new();
        PowerOfTwo.pick(
            &past,
            &pool,
            &kept,
            &[0.5, 0.5],
            TieBand::NORMALIZED,
            &mut want,
        );
        let want: Vec<u32> = want.iter().map(|t| t.position).collect();
        assert_eq!(positions(&got), want, "seed {seed}");
        second_first += u32::from(want.first() == Some(&kept[1]));
    }
    assert!(
        second_first > 20 && past_a_refusal - second_first > 20,
        "{past_a_refusal} decisions drew past a refused row, {second_first} put the second first"
    );
}

#[test]
fn the_two_draws_of_one_decision_share_its_bound_of_refused_rows() {
    use std::sync::atomic::{AtomicU32, Ordering};

    static ASKS: AtomicU32 = AtomicU32::new(0);
    static PASSES: AtomicU32 = AtomicU32::new(0);
    /// Keeps the fourth row it is asked about and no other; a whole
    /// pass keeps the pool as it stands and is counted apart.
    #[derive(Debug)]
    struct FourthAsk;
    impl Filter for FourthAsk {
        fn name(&self) -> &'static str {
            "fourth-ask"
        }

        fn row_wise(&self) -> bool {
            true
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            ASKS.fetch_add(1, Ordering::Relaxed) + 1 == 4
        }

        fn retain(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &Constraints<'_>,
            _: &mut Vec<u32>,
        ) {
            PASSES.fetch_add(1, Ordering::Relaxed);
        }
    }
    let engine = Engine::new(
        vec![Profile {
            name: "two".into(),
            filters: vec![Box::new(FourthAsk)],
            scorers: Vec::new(),
            picker: Box::new(PowerOfTwo),
        }],
        vec![single_binding()],
        1,
        Needs::default(),
    );
    let pool = pool_of(64, |_| false);
    let got = drawn(&engine, &request(1), &pool);
    // Three rows refused, the fourth kept, then the second draw's rows
    // refused until the DECISION's bound: one ask more than the bound,
    // not the bound again for the second draw, and the whole pass
    // decides.
    assert_eq!(ASKS.load(Ordering::Relaxed), DRAW_BOUND + 1);
    assert_eq!(PASSES.load(Ordering::Relaxed), 1);
    assert!(!got.trace().drew_first());
    assert!(got.pick().is_some());
}

#[test]
fn a_draw_asks_about_its_targets_and_its_bound_of_refused_rows_and_no_row_more() {
    use std::sync::atomic::{AtomicU32, Ordering};

    static ASKS: AtomicU32 = AtomicU32::new(0);
    /// The health filter's word, with each ask about one row counted;
    /// a whole pass is the health filter's own and is not counted.
    #[derive(Debug)]
    struct Counted;
    impl Filter for Counted {
        fn name(&self) -> &'static str {
            "counted"
        }

        fn row_wise(&self) -> bool {
            true
        }

        fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
            ASKS.fetch_add(1, Ordering::Relaxed);
            Healthy.keep(facts, c, cx)
        }

        fn retain(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            cx: &Constraints<'_>,
            kept: &mut Vec<u32>,
        ) {
            Healthy.retain(facts, candidates, cx, kept);
        }
    }
    let pickers: [(&str, Box<dyn Picker>); 5] = [
        ("passthrough", Box::new(Passthrough)),
        ("random", Box::new(Random)),
        ("round-robin", Box::new(RoundRobin::new())),
        ("power-of-two", Box::new(PowerOfTwo)),
        ("power-of-two-count-first", Box::new(PowerOfTwoCountFirst)),
    ];
    for (name, picker) in pickers {
        let engine = Engine::new(
            vec![Profile {
                name: name.into(),
                filters: vec![Box::new(Counted)],
                scorers: Vec::new(),
                picker,
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        );
        for (fleet, refused) in REFUSED.into_iter().enumerate() {
            let pool = pool_of(64, refused);
            for seed in 0..50 {
                ASKS.store(0, Ordering::Relaxed);
                let decision = drawn(&engine, &request(seed), &pool);
                let asks = ASKS.load(Ordering::Relaxed) as usize;
                // The first fleet is ready whole: a draw there asks
                // about the rows it hands out and about no other.
                let most = if fleet == 0 {
                    assert!(decision.trace().drew_first(), "{name}, seed {seed}");
                    MAX_TARGETS
                } else {
                    MAX_TARGETS + DRAW_BOUND as usize
                };
                assert!(
                    (1..=most).contains(&asks),
                    "{name}, fleet {fleet}, seed {seed}: {asks} asks"
                );
            }
        }
    }
}

#[test]
fn a_pool_under_a_new_membership_epoch_turns_its_own_rotation() {
    let engine = preset("round_robin");
    let ready = pool_of(32, |_| false);
    let mut scratch = Scratch::new();
    let mut turn = |epoch: u64| {
        let pool = PoolFacts {
            membership_epoch: epoch,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let decision = engine.decide(&request(0), &pool, &ready, &cx, &mut scratch);
        positions(&decision)[0]
    };
    let old = turn(1);
    for step in 1..=5 {
        assert_eq!(turn(1), (old + step) % 32, "step {step}");
    }
    // A membership write moves the epoch: another list of rows stands
    // behind the pool's key, and it turns a rotation of its own.
    let new = turn(2);
    for step in 1..=3 {
        assert_eq!(turn(2), (new + step) % 32, "step {step}");
    }
    // The old epoch's rotation stands where its last decision left it.
    assert_eq!(turn(1), (old + 6) % 32);
}

#[test]
fn two_lists_of_one_pool_each_turn_their_own_rotation() {
    // Two lists behind one model and phase, asked through one engine,
    // each pair apart by one word of the key: two callers' lists of one
    // pool that the host names apart, as where a pool holds host-local
    // rows; two lists of two sizes; a pool's list before a membership
    // write and after it.
    let engine = preset("round_robin");
    let mut scratch = Scratch::new();
    for (sizes, reaches, epochs) in [
        ([16u32, 16], [1, 2], [1, 1]),
        ([16, 32], [0, 0], [1, 1]),
        ([16, 16], [0, 0], [2, 3]),
    ] {
        let lists = sizes.map(|rows| pool_of(rows, |_| false));
        let mut drawn = sizes.map(|rows| vec![0u32; rows as usize]);
        let mut ask = |list: usize, drawn: &mut [Vec<u32>; 2]| {
            let pool = PoolFacts {
                membership_epoch: epochs[list],
                reach: reaches[list],
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            let decision = engine.decide(&request(0), &pool, &lists[list], &cx, &mut scratch);
            assert!(decision.trace().drew_first());
            drawn[list][positions(&decision)[0] as usize] += 1;
        };
        // Their asks reach one engine in turn: one rotation turned by
        // both would draw every other row of each list.
        for turn in 0..64 {
            ask(turn % 2, &mut drawn);
        }
        for (rows, drawn) in sizes.iter().zip(&drawn) {
            assert!(
                drawn.iter().all(|&n| n == 32 / rows),
                "{sizes:?}: {drawn:?}"
            );
        }
        // And as a coin falls: each list's turn is still its own, so no
        // two of its rows are more than one decision apart.
        let mut coin = Draw::new(11);
        for _ in 0..320 {
            ask((coin.next_u64() & 1) as usize, &mut drawn);
        }
        for drawn in &drawn {
            let least = drawn.iter().copied().min().unwrap_or(0);
            let most = drawn.iter().copied().max().unwrap_or(0);
            assert!(most - least <= 1, "{sizes:?}: {drawn:?}");
        }
    }
}

#[test]
fn an_encode_item_never_draws() {
    // The `epd` plan's compatibility rule is the whole pass's own: under
    // an encode profile whose picker draws first, an encode item over a
    // pool of the draw's size is still read whole, and the incompatible
    // encoder that leads the pool is handed to nobody.
    let engine = PolicySpec::epd_presets("passthrough", "least_load", "least_load")
        .expect("the presets")
        .validate()
        .expect("the spec validates");
    let pool: Vec<Candidate> = (0..16).map(|i| encode_candidate(i, i != 0, true)).collect();
    let decision = drawn(&engine, &encode_facts(0), &pool);
    assert_eq!(positions(&decision), [1, 2, 3, 4]);
    assert!(!decision.trace().drew_first());
    assert_eq!(decision.trace().survivors.get(), Some(15));
}

#[test]
fn a_drawn_decision_replays_from_its_seed_and_a_widened_one_too() {
    for name in ["random", "power_of_two"] {
        let engine = preset(name);
        for (fleet, refused) in REFUSED.into_iter().enumerate() {
            let pool = pool_of(64, refused);
            for seed in 0..200 {
                let facts = request(seed);
                let first = drawn(&engine, &facts, &pool);
                assert_eq!(
                    first,
                    drawn(&engine, &facts, &pool),
                    "{name}, fleet {fleet}"
                );
            }
        }
    }
}

// ---- a draw the rows a host named join

fn holding(row: u32, overlap_blocks: u32) -> crate::engine::AffinityHolder {
    crate::engine::AffinityHolder {
        row,
        overlap_blocks,
        predicted_blocks: 0,
    }
}

/// A request of sixteen blocks whose host names `holders`.
fn naming(seed: u64, holders: &[crate::engine::AffinityHolder]) -> RequestFacts<'_> {
    static HASHES: [u64; 16] = [0; 16];
    RequestFacts {
        block_hashes: &HASHES,
        prompt_tokens: 256,
        named: Some(AffinityRows {
            holders,
            ..AffinityRows::default()
        }),
        ..request(seed)
    }
}

/// Behind the draw of a `best` that takes a sample, the rows a host
/// named join the rows drawn: the deepest named holder leads whether the
/// draw reached it or not, one a filter refuses leaves the lead to the
/// next, and a named row the draw reached too is handed out once.
#[test]
fn the_rows_a_host_named_join_the_rows_a_sampled_best_drew() {
    let engine = preset("cache_aware");
    assert_eq!(engine.needs().sample, 16);
    let holders = [holding(20, 8), holding(40, 12)];
    view::LOOKS_BEHIND.with(|looks| looks.set(0));
    for (down, lead) in [(64u32, 40u32), (40, 20)] {
        let pool = pool_of(64, |i| i == down);
        for seed in 0..200u64 {
            let got = drawn(&engine, &naming(seed, &holders), &pool);
            assert!(got.trace().drew_first(), "seed {seed}: the decision drew");
            let rows = positions(&got);
            assert_eq!(rows.first(), Some(&lead), "seed {seed}, row {down} down");
            let mut once = rows.clone();
            once.sort_unstable();
            once.dedup();
            assert_eq!(once.len(), rows.len(), "seed {seed}: {rows:?}, a row twice");
        }
    }
    // Behind the draw the stages search the rows named: no walk looked
    // behind its cursor in any of the decisions.
    assert_eq!(view::LOOKS_BEHIND.with(std::cell::Cell::get), 0);
}

/// The resident-work stage answers row by row, as the least-load stage
/// does, so a `best` that takes a sample draws under it: for a request
/// whose host names rows, over a pool of more than twice the sample and
/// of sixteen rows or more, the decision draws, and the stage costs the
/// rows drawn and the rows named among themselves. The row of least work
/// among the rows handed leads, so the pool's least row leads where the
/// sample reached it and not otherwise. A request that names no row
/// reads the pool whole, and there that row leads.
#[test]
fn a_sampled_best_draws_under_the_resident_work_stage() {
    let engine = Engine::new(
        vec![Profile {
            name: "sampled".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(scorers::ResidentWork::new(0)),
                weight: 1.0,
            }],
            picker: Box::new(pickers::SampledBest::new(4)),
        }],
        vec![Binding {
            model: "m".into(),
            profile: 0,
            ..single_binding()
        }],
        1,
        Needs::default(),
    );
    // Every row holds one request and reports its resident tokens, fewer
    // the further its place: no two rows hold the same work, and the
    // last row holds the least of the pool.
    let pool: Vec<Candidate> = pool_of(64, |_| false)
        .into_iter()
        .map(|row| Candidate {
            in_flight: 1,
            load: Some(view::LoadRow {
                running: 1,
                resident_tokens: 1_000 * (64 - row.position),
                resident_counted: true,
                ..view::LoadRow::default()
            }),
            ..row
        })
        .collect();
    let holders = [holding(20, 8)];
    let mut led = 0;
    for seed in 0..64u64 {
        let got = drawn(&engine, &naming(seed, &holders), &pool);
        assert!(got.trace().drew_first(), "seed {seed}: the decision drew");
        let handed = positions(&got);
        assert_eq!(
            handed.first(),
            handed.iter().max(),
            "seed {seed}: the furthest of the rows handed leads"
        );
        led += u32::from(handed.first() == Some(&63));
    }
    assert!(
        led < 32,
        "the pool's least row led {led} of 64 decisions over a sample of four"
    );
    let whole = drawn(&engine, &request(1), &pool);
    assert!(!whole.trace().drew_first(), "no row named: the pool whole");
    assert_eq!(positions(&whole).first(), Some(&63));
}

/// A host caps each list of rows it names at the picker's sample. Past
/// it a decision reads dearer and is never wrong, so a debug build
/// alone stops, by the assertion's own words.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "at or under the profile's sample")]
fn rows_named_past_the_sample_stop_a_debug_build_where_the_picker_draws() {
    let engine = preset("cache_aware");
    let holders: Vec<crate::engine::AffinityHolder> = (0..17).map(|row| holding(row, 12)).collect();
    let decision = drawn(&engine, &naming(1, &holders), &pool_of(64, |_| false));
    panic!("seventeen rows named answered {decision:?}");
}

/// A pool of at most twice the sample is read whole, rows named or not,
/// and so is a larger pool whose host names none: the draw is for the
/// pools a host names rows in.
#[test]
fn a_sampled_best_reads_a_small_pool_and_a_pool_with_no_rows_named_whole() {
    let (engine, twin) = (preset("cache_aware"), preset("cache_aware"));
    let holders = [holding(3, 12)];
    for seed in 0..40u64 {
        // Twice the sample, rows named: the whole pass, row for row.
        let pool = pool_of(32, |i| i % 17 == 5);
        let facts = naming(seed, &holders);
        let got = drawn(&engine, &facts, &pool);
        assert!(!got.trace().drew_first(), "seed {seed}");
        assert_eq!(got, whole_pass(&twin, &facts, &pool), "seed {seed}");
        // One row more: the decision draws.
        let pool = pool_of(33, |i| i % 17 == 5);
        let got = drawn(&engine, &facts, &pool);
        assert!(got.trace().drew_first(), "seed {seed}");
        // No rows named: the whole pass, whatever the pool's size.
        let pool = pool_of(257, |i| i % 17 == 5);
        let facts = request(seed);
        let got = drawn(&engine, &facts, &pool);
        assert!(!got.trace().drew_first(), "seed {seed}");
        assert_eq!(got, whole_pass(&twin, &facts, &pool), "seed {seed}");
    }
}

/// The joined mark stands only beside the draw's: under a filter that is
/// not row-wise a `best` that takes a sample reads its pool whole,
/// whatever rows the host names.
#[test]
fn a_joined_picker_under_a_stage_that_is_not_row_wise_never_draws() {
    /// Keeps every row, and its word on one row is not its whole word.
    #[derive(Debug)]
    struct Whole;
    impl Filter for Whole {
        fn name(&self) -> &'static str {
            "whole"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
    }
    let engine = Engine::new(
        vec![Profile {
            name: "whole".into(),
            filters: vec![Box::new(Healthy), Box::new(Whole)],
            scorers: Vec::new(),
            picker: Box::new(pickers::SampledBest::new(16)),
        }],
        vec![single_binding()],
        1,
        Needs::default(),
    );
    // It says no sample: a stage that is not row-wise reads the pool whole.
    assert_eq!(engine.needs().sample, 0);
    let holders = [holding(3, 12)];
    let pool = pool_of(64, |_| false);
    for seed in 0..20u64 {
        let got = drawn(&engine, &naming(seed, &holders), &pool);
        assert!(!got.trace().drew_first(), "seed {seed}");
        assert_eq!(got.trace().survivors.get(), Some(64), "seed {seed}");
    }
}

/// The mean the spill gate reads behind a draw is the drawn rows': four
/// named holders with seven in flight spill against rows drawn at two,
/// where the mean of the rows handed, the holders among them, would
/// keep each of them in the lead.
#[test]
fn the_rows_named_do_not_move_the_mean_they_are_judged_against() {
    let engine = preset("cache_aware");
    let holders = [
        holding(8, 12),
        holding(24, 12),
        holding(40, 12),
        holding(56, 12),
    ];
    let pool: Vec<Candidate> = (0..64u32)
        .map(|i| candidate(i, if i % 16 == 8 { 7 } else { 2 }))
        .collect();
    let mut led = 0u32;
    for seed in 0..200u64 {
        let got = drawn(&engine, &naming(seed, &holders), &pool);
        assert!(got.trace().drew_first(), "seed {seed}");
        led += u32::from(positions(&got).first().is_some_and(|row| row % 16 == 8));
    }
    // A draw that takes all four holders reads a mean they pass: rare.
    assert!(led < 20, "{led} of 200 decisions were led by a busy holder");
}

/// The mean the join hands on is of the drawn rows' own counts.
#[test]
fn the_join_hands_on_the_mean_of_the_drawn_rows_counts() {
    let pool: Vec<Candidate> = (0..)
        .zip([6, 2, 9, 9])
        .map(|(position, in_flight)| Candidate {
            position,
            in_flight,
            ..Candidate::default()
        })
        .collect();
    let facts = RequestFacts::default();
    let mut kept = vec![0, 1];
    let mean = join_named(
        &facts,
        &AffinityRows::default(),
        &pool,
        &mut |_: &Candidate| true,
        &mut kept,
    );
    assert_eq!(mean, 4.0);
}

/// A named row past the pool names no candidate: the join passes over
/// it in every build and asks no filter about it. A decision of a debug
/// build stops before the join ([`debug_assert_named`]).
#[test]
fn the_join_passes_over_a_row_named_past_the_pool() {
    let pool = pool_of(4, |i| i == 2);
    let holders = [holding(1, 12), holding(2, 12), holding(9, 12)];
    let named = AffinityRows {
        holders: &holders,
        warming: &[3, 4, u32::MAX],
        ..AffinityRows::default()
    };
    let mut asked = Vec::new();
    let mut keep = |row: &Candidate| {
        asked.push(row.position);
        row.health.healthy
    };
    let mut kept = vec![0];
    join_named(
        &RequestFacts::default(),
        &named,
        &pool,
        &mut keep,
        &mut kept,
    );
    assert_eq!(kept, [0, 1, 3], "the pool's rows a filter keeps join");
    assert_eq!(
        asked,
        [1, 2, 3],
        "a filter is asked of the pool's rows alone"
    );
}

/// Two profiles whose `best` takes a sample of four and of sixteen, for
/// the models `narrow` and `wide`.
fn two_samples() -> Engine {
    let bound = |model: &str, profile| Binding {
        model: model.into(),
        profile,
        ..single_binding()
    };
    sampled_under(vec![bound("narrow", 0), bound("wide", 1)])
}

/// Two profiles that draw, under `bindings`: a sample of four (profile
/// 0) and a sample of sixteen (profile 1).
fn sampled_under(bindings: Vec<Binding>) -> Engine {
    let sampled = |name: &str, sample| Profile {
        name: name.into(),
        filters: vec![Box::new(Healthy)],
        scorers: Vec::new(),
        picker: Box::new(pickers::SampledBest::new(sample)),
    };
    Engine::new(
        vec![sampled("narrow", 4), sampled("wide", 16)],
        bindings,
        1,
        Needs::default(),
    )
}

/// A host caps the rows it names by the sample of the profile that
/// decides, which [`Engine::sample_for`] says; the engine's summary
/// holds the largest sample of every profile. The cache-aware preset
/// says sixteen for a model's single decision.
#[test]
fn a_host_reads_the_cap_of_the_rows_it_names_from_the_profile_that_decides() {
    let engine = two_samples();
    assert_eq!(engine.needs().sample, 16);
    let pool = pool_of(64, |_| false);
    let holders: Vec<crate::engine::AffinityHolder> = (0..16).map(|row| holding(row, 12)).collect();
    for (model, sample) in [("narrow", 4usize), ("wide", 16)] {
        let cap = engine.sample_for(model, Phase::Single) as usize;
        assert_eq!(cap, sample, "{model}");
        let facts = RequestFacts {
            model,
            ..naming(1, &holders[..cap])
        };
        let got = drawn(&engine, &facts, &pool);
        assert!(got.trace().drew_first(), "{model}: capped so, it draws");
    }
    let cache_aware = preset("cache_aware");
    assert_eq!(cache_aware.sample_for("m", Phase::Single), 16);
}

/// Capped by the engine's summary instead, the rows a host names under
/// the narrower profile stand past its picker's sample: a debug build
/// stops, by the assertion's own words.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "at or under the profile's sample")]
fn rows_capped_by_the_engines_summary_stop_a_debug_build_under_a_narrower_profile() {
    let engine = two_samples();
    let holders: Vec<crate::engine::AffinityHolder> = (0..engine.needs().sample)
        .map(|row| holding(row, 12))
        .collect();
    let facts = RequestFacts {
        model: "narrow",
        ..naming(1, &holders)
    };
    let decision = drawn(&engine, &facts, &pool_of(64, |_| false));
    panic!("sixteen rows named under a sample of four answered {decision:?}");
}

/// The mean in flight is the engine's to set behind a draw. A host that
/// writes one stops a debug build, by the assertion's own words: on a
/// pool read whole the stages would read its number for their rows'.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "beside an empty column and no mean in flight")]
fn a_mean_in_flight_a_host_wrote_stops_a_debug_build() {
    let engine = preset("cache_aware");
    let holders = [holding(3, 12)];
    let facts = RequestFacts {
        named: Some(AffinityRows {
            holders: &holders,
            mean_in_flight: Some(2.0),
            ..AffinityRows::default()
        }),
        ..request(1)
    };
    let decision = drawn(&engine, &facts, &pool_of(8, |_| false));
    panic!("a mean the host wrote answered {decision:?}");
}

/// A sampled `best` counts no refused rows: under a filter that refuses
/// half of the pool every decision still draws, and none of its targets
/// is a refused row. A pool refused whole is the whole pass's to shed.
#[test]
fn a_sampled_draw_stands_under_a_filter_that_refuses_half_of_the_pool() {
    let engine = preset("cache_aware");
    let holders = [holding(5, 12)];
    let pool = pool_of(64, |i| i % 2 == 0);
    for seed in 0..500u64 {
        let got = drawn(&engine, &naming(seed, &holders), &pool);
        assert!(got.trace().drew_first(), "seed {seed}");
        let rows = positions(&got);
        assert!(rows.iter().all(|row| row % 2 == 1), "seed {seed}: {rows:?}");
    }
    let got = drawn(&engine, &naming(1, &holders), &pool_of(64, |_| true));
    assert!(!got.trace().drew_first());
    assert!(got.pick().is_none());
}

/// A draw's read of the best scores is the whole pass's, slot for slot:
/// ties, scores under zero, scores without an order and every count of
/// rows around the count kept. The keys' own read is held to the pass's
/// at every count, under the count a draw reads by keys too.
#[test]
fn the_best_scores_of_a_draw_are_the_best_scores_of_a_pass() {
    fn bits<const K: usize>(
        read: impl Fn(&[f32], &mut [(usize, f32); K]) -> usize,
        scores: &[f32],
    ) -> ([(usize, u32); K], usize) {
        let mut top = [(usize::MAX, f32::NEG_INFINITY); K];
        let len = read(scores, &mut top);
        (top.map(|(i, score)| (i, score.to_bits())), len)
    }
    fn held<const K: usize>(scores: &[f32]) {
        assert_eq!(
            bits(top_drawn::<K>, scores),
            bits(top_pass::<K>, scores),
            "{scores:?}"
        );
        if !scores.iter().any(|score| score.is_nan()) {
            assert_eq!(
                bits(top_keyed::<K>, scores),
                bits(top_pass::<K>, scores),
                "{scores:?}"
            );
        }
    }
    let values = [
        0.0f32,
        -0.0,
        0.25,
        0.5,
        0.5,
        0.75,
        1.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
    ];
    // Scores under zero alone, a tiny one among them: their order is the
    // keys' turn of every bit under a set sign.
    let under = [
        -0.0f32,
        -1.0e-40,
        -0.25,
        -0.25,
        -1.0,
        -3.5,
        f32::MIN,
        f32::NEG_INFINITY,
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for round in 0..4_000usize {
        let scores: Vec<f32> = (0..round % 41)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let pick = (state >> 33) as usize;
                match round % 4 {
                    // Every fourth list holds scores under zero alone.
                    3 => under[pick % under.len()],
                    // Every second list holds no score without an order.
                    _ => values[pick % (values.len() - round % 2)],
                }
            })
            .collect();
        held::<1>(&scores);
        held::<TRACE_TOP>(&scores);
        held::<5>(&scores);
    }
}

/// A list past the keys' places is read as the pass reads it, and a list
/// that ends at their last place by keys: its last row's place is zero.
#[test]
fn a_list_past_the_keys_places_is_read_as_the_pass_reads_it() {
    let last = 1usize << PLACE_BITS;
    for len in [last - 1, last, last + 1, last + 7] {
        // One score everywhere but at both ends: the best rows stand at
        // the first place and at the last ones, ties among them.
        let mut scores = vec![0.25f32; len];
        scores[0] = 1.0;
        for (back, score) in [1.0f32, 0.5, 0.75, 1.0, 0.5].into_iter().enumerate() {
            scores[len - 1 - back] = score;
        }
        let mut drawn = [(usize::MAX, f32::NEG_INFINITY); 5];
        let mut pass = drawn;
        assert_eq!(top_drawn(&scores, &mut drawn), top_pass(&scores, &mut pass));
        assert_eq!(drawn, pass, "{len} rows");
    }
}

/// A draw's trace names the best of its rows as a pass's trace does,
/// equal scores among them, and does not count its survivors.
#[test]
fn a_draws_trace_names_its_best_rows_as_a_passes_trace_does() {
    let pool = pool_of(60, |_| false);
    let facts = request(7);
    // A draw's order, and scores that tie: twenty rows, which a draw
    // reads by keys.
    let kept: Vec<u32> = (0..60u32).rev().step_by(3).collect();
    assert!(kept.len() >= KEYED_FROM);
    let scores: Vec<f32> = kept.iter().map(|&row| (row % 7) as f32 / 8.0).collect();
    let pass = trace_of(3, &facts, &PoolFacts::default(), &pool, &kept, &scores);
    let draw = trace_drawn(3, &facts, &PoolFacts::default(), &pool, &kept, &scores);
    assert_eq!(usize::from(pass.top_len), TRACE_TOP);
    assert_eq!(draw.survivors, Survivors::NOT_COUNTED);
    assert_eq!(
        Trace {
            survivors: pass.survivors,
            ..draw
        },
        pass
    );
}

/// A decode decision of a model that also pairs is decided by its own
/// profile, and the rows a host names for it are capped by that
/// profile's sample ([`Engine::sample_for`]). A decode worker's needs
/// carry the pair's decode side too ([`Engine::needs_for`]), whose
/// sample is the larger here: a list capped by them stands past the
/// cap.
#[test]
fn a_decode_decision_of_a_model_that_also_pairs_is_capped_by_its_own_profile() {
    let alone = Binding {
        model: "mixed".into(),
        phases: vec![Phase::Decode],
        ..single_binding()
    };
    let pair = Binding {
        model: "mixed".into(),
        profile: 1,
        decode_profile: Some(1),
        ..pair_binding()
    };
    let engine = sampled_under(vec![alone, pair]);
    assert_eq!(engine.needs_for("mixed", Phase::Decode).sample, 16);
    assert_eq!(engine.sample_for("mixed", Phase::Decode), 4);
    assert_eq!(engine.sample_for("nobody", Phase::Decode), 0);
    let holders: Vec<crate::engine::AffinityHolder> = (0..4).map(|row| holding(row, 12)).collect();
    let facts = RequestFacts {
        model: "mixed",
        phase: Phase::Decode,
        ..naming(1, &holders)
    };
    let got = drawn(&engine, &facts, &pool_of(64, |_| false));
    assert!(
        got.trace().drew_first(),
        "capped by its own profile, it draws"
    );
}

/// The engine asks a draw of a pool of [`DRAW_FROM`] rows or more,
/// whatever the sample: under a sample of four a pool of nine to
/// fifteen rows, more than twice the sample, is still read whole.
#[test]
fn a_sample_under_eight_draws_from_sixteen_rows_on() {
    let engine = two_samples();
    let holders = [holding(0, 12)];
    let facts = RequestFacts {
        model: "narrow",
        ..naming(1, &holders)
    };
    for (rows, drew) in [(9, false), (12, false), (15, false), (16, true)] {
        let got = drawn(&engine, &facts, &pool_of(rows, |_| false));
        assert_eq!(got.trace().drew_first(), drew, "{rows} rows");
    }
}
