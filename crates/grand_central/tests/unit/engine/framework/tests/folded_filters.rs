use super::*;
/// An engine of one profile: `filters`, no scorer and `picker`.
fn engine_of(filters: Vec<Box<dyn Filter>>, picker: Box<dyn Picker>) -> Engine {
    Engine::new(
        vec![Profile {
            name: "p".into(),
            filters,
            scorers: Vec::new(),
            picker,
        }],
        vec![single_binding()],
        1,
        Needs::default(),
    )
}
/// The pickers the cases of the folded test decide by.
const PICKERS: [fn() -> Box<dyn Picker>; 4] = [
    || Box::new(Passthrough),
    || Box::new(Random),
    || Box::new(RoundRobin::new()),
    || Box::new(PowerOfTwo),
];

/// The rows the cases of a row's test read, with the host's mask. By
/// position modulo eight: a row each of four bits refuses and four the
/// rows' own bits keep, of which the mask names one.
fn tested_rows() -> (Vec<Candidate>, Vec<u32>) {
    let pool = (0..64u32)
        .map(|i| {
            let mut c = Candidate {
                has_capacity: true,
                has_room: true,
                ..candidate(i, i % 5)
            };
            match i % 8 {
                1 => c.health.healthy = false,
                2 => c.health.overloaded = true,
                3 => c.has_capacity = false,
                4 => c.has_room = false,
                _ => {}
            }
            c
        })
        .collect();
    (pool, (0..64).filter(|i| i % 8 == 7).collect())
}

#[test]
fn a_profile_of_the_engines_own_filters_draws_as_one_asked_a_call_a_filter() {
    use crate::stages::filters::tests::filters_of;
    // A request names one row tried, and the host one excluded.
    let (pool, mask) = tested_rows();
    let (mut drew, mut masked) = (0u32, 0u32);
    for set in 0..32u8 {
        for (which, picker) in PICKERS.into_iter().enumerate() {
            let folded = engine_of(filters_of(set), picker());
            let hosted = filters_of(set)
                .into_iter()
                .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
                .collect();
            let each = engine_of(hosted, picker());
            assert!(folded.row_tests[0].is_some(), "set {set:#07b}: folded");
            // A profile of no filter folds into the empty test whoever built it.
            assert_eq!(
                each.row_tests[0].is_none(),
                set != 0,
                "set {set:#07b}: a host's"
            );
            for seed in 0..24u64 {
                let tried = [pool[(seed as usize * 8 + 5) % pool.len()].id];
                let excluded = [pool[(seed as usize * 8 + 6) % pool.len()].id];
                let facts = RequestFacts {
                    tried: if seed % 2 == 0 { &tried } else { &[] },
                    ..request(seed ^ u64::from(set) << 32)
                };
                let cx = Constraints {
                    excluded: if seed % 3 == 0 { &excluded } else { &[] },
                    stalled: if seed % 4 < 2 { mask.as_slice() } else { &[] },
                    overload_lifted: seed % 5 == 0,
                    ..Constraints::default()
                };
                let decide = |engine: &Engine| {
                    engine.decide(
                        &facts,
                        &PoolFacts::default(),
                        &pool,
                        &cx,
                        &mut Scratch::new(),
                    )
                };
                let (got, want) = (decide(&folded), decide(&each));
                let at = format!("set {set:#07b}, picker {which}, seed {seed}");
                assert_eq!(got, want, "{at}");
                if got.trace().drew_first() {
                    drew += 1;
                    masked += u32::from(set & 1 != 0 && !cx.stalled.is_empty());
                }
                if set & 1 != 0 {
                    assert!(
                        positions(&got).iter().all(|&row| !cx.masks(row)),
                        "{at}: a row the host's mask names"
                    );
                }
            }
        }
    }
    // Most decisions drew, some of them under the health filter and a
    // standing veto, where the mask is asked beside the folded test.
    assert!(drew > 1_000 && masked > 200, "{drew} drew, {masked} masked");
}

/// The test a host asks of a holder before it names it is the draw's
/// own. Over every set of the engine's filters, as the build folds them
/// and as a host's own, the function and a call a filter agree on every
/// row: under a mask, a row tried, a row excluded and the overload veto
/// lifted. Its hard part agrees the same way with a call to each filter
/// that says it is not soft ([`Filter::soft`]), on both roads: the folded
/// test less its soft bits, and a host's filters one by one.
#[test]
fn the_namings_test_is_the_draws_own() {
    use crate::stages::filters::tests::filters_of;
    let (pool, mask) = tested_rows();
    let (mut refused, mut softly) = (0u32, 0u32);
    for set in 0..32u8 {
        let folded = engine_of(filters_of(set), Box::new(Random));
        let hosted = filters_of(set)
            .into_iter()
            .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
            .collect();
        let each = engine_of(hosted, Box::new(Random));
        let asked = filters_of(set);
        for seed in 0..24u64 {
            let tried = [pool[(seed as usize * 8 + 5) % pool.len()].id];
            let excluded = [pool[(seed as usize * 8 + 6) % pool.len()].id];
            let facts = RequestFacts {
                tried: if seed % 2 == 0 { &tried } else { &[] },
                ..request(seed)
            };
            let cx = Constraints {
                excluded: if seed % 3 == 0 { &excluded } else { &[] },
                stalled: if seed % 4 < 2 { mask.as_slice() } else { &[] },
                overload_lifted: seed % 5 == 0,
                ..Constraints::default()
            };
            for (built, engine) in [("folded", &folded), ("a host's", &each)] {
                let keeps = engine.keeps_for(facts.model, facts.phase, &facts, &cx);
                let keeps = keeps.expect("every filter answers row by row");
                for row in &pool {
                    let want = asked.iter().all(|filter| filter.keep(&facts, row, &cx));
                    let at = format!("set {set:#07b}, {built}, seed {seed}");
                    assert_eq!(keeps.keeps(row), want, "{at}, row {}", row.position);
                    refused += u32::from(!want);
                    let mut hard = asked.iter().filter(|filter| !filter.soft());
                    let hard = hard.all(|filter| filter.keep(&facts, row, &cx));
                    assert_eq!(keeps.keeps_hard(row), hard, "{at}, row {}", row.position);
                    softly += u32::from(hard && !want);
                }
            }
        }
    }
    // The spread of the case's own rows, counted by the filters' word:
    // rows the soft filters alone refuse are among them, where the hard
    // part and the whole test part.
    assert!(
        refused > 10_000 && softly > 1_000,
        "{refused} rows refused, {softly} by soft filters alone"
    );
}

/// The function answers none where the pass alone can say which rows
/// stay: under a filter that does not answer row by row, under one
/// that holds a decision's scope, and for a profile under the key's
/// mark, whatever the request carries.
#[test]
fn the_namings_test_is_none_where_the_pass_alone_can_say() {
    use crate::stages::filters::tests::filters_of;
    /// A filter that is shown the whole view: the default word.
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
    /// A filter that answers row by row, out of a decision's scope
    /// where it says it holds one.
    #[derive(Debug)]
    struct Scoped(bool);
    impl Lifecycle for Scoped {}
    impl DecisionScope for Scoped {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

        fn close(&self) {}
    }
    impl Filter for Scoped {
        fn name(&self) -> &'static str {
            "scoped"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn row_wise(&self) -> bool {
            true
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            self.0.then_some(self as &dyn DecisionScope)
        }
    }
    let cx = Constraints::default();
    let plain = request(7);
    let keyed = RequestFacts {
        session_key: Identity::hashed(59),
        ..plain
    };
    let tests = |engine: &Engine, facts: &RequestFacts<'_>| {
        let keeps = engine.keeps_for(facts.model, facts.phase, facts, &cx);
        keeps.is_some()
    };
    let whole = engine_of(vec![Box::new(Whole)], Box::new(Random));
    assert!(!tests(&whole, &plain), "a filter shown the whole view");
    // A scope opens at the decision, behind the naming: a filter that
    // answers row by row out of one is asked by its pass alone.
    let scoped = engine_of(vec![Box::new(Scoped(true))], Box::new(Random));
    assert!(
        !tests(&scoped, &plain),
        "a row-wise filter that holds a scope"
    );
    let unscoped = engine_of(vec![Box::new(Scoped(false))], Box::new(Random));
    assert!(tests(&unscoped, &plain), "the same filter with no scope");
    // The key's mark, set by hand: the validator sets it for the plain
    // ring's shape alone, which is named no row. A keyed request is
    // tested without the mark, and none is under it.
    let mut marked = engine_of(filters_of(0b11), Box::new(Random));
    assert!(tests(&marked, &keyed), "no mark: a key is tested");
    marked.places_by_key(0);
    assert!(!tests(&marked, &keyed), "a key under the mark");
    assert!(!tests(&marked, &plain), "no key under the mark");
}

#[test]
fn a_whole_pass_by_the_folded_test_decides_as_one_asked_a_stage_at_a_time() {
    use crate::stages::filters::tests::filters_of;
    // A row's shape: ready, or refused by one of four bits.
    fn shaped(rows: u32, shape: impl Fn(u32) -> u32) -> Vec<Candidate> {
        (0..rows)
            .map(|i| {
                let mut c = Candidate {
                    has_capacity: true,
                    has_room: true,
                    ..candidate(i, i % 5)
                };
                match shape(i) {
                    1 => c.health.healthy = false,
                    2 => c.health.overloaded = true,
                    3 => c.has_capacity = false,
                    4 => c.has_room = false,
                    _ => {}
                }
                c
            })
            .collect()
    }
    // Pools under the draw's floor, so each decision is a whole pass:
    // rows all of one shape, one ready row among rows of one refusal,
    // and mixed rows.
    let mut pools = Vec::new();
    for rows in [1, 2, 7, DRAW_FROM as u32 - 1] {
        for shape in 0..5 {
            pools.push(shaped(rows, |_| shape));
        }
        for shape in 1..5 {
            pools.push(shaped(rows, |i| if i == rows / 2 { 0 } else { shape }));
        }
        for mix in 0..4 {
            pools.push(shaped(rows, |i| (i * 7 + mix * 3) % 6 % 5));
        }
    }
    let (mut stood, mut fell, mut shed, mut aside) = (0u32, 0u32, 0u32, 0u32);
    for set in 0..32u8 {
        for (which, picker) in PICKERS.into_iter().enumerate() {
            let folded = engine_of(filters_of(set), picker());
            let hosted = filters_of(set)
                .into_iter()
                .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
                .collect();
            let each = engine_of(hosted, picker());
            let filters = filters_of(set);
            for (number, pool) in pools.iter().enumerate() {
                let middle = pool.len() / 2;
                let ids: Vec<WorkerId> = pool.iter().map(|c| c.id).collect();
                let places: Vec<u32> = (0..pool.len() as u32).collect();
                for seed in 0..10u64 {
                    let facts = RequestFacts {
                        tried: match seed % 3 {
                            0 => &[],
                            1 => &ids[middle..=middle],
                            _ => &ids,
                        },
                        ..request(seed ^ u64::from(set) << 32)
                    };
                    let cx = Constraints {
                        excluded: if seed % 4 == 3 {
                            &ids[middle..=middle]
                        } else {
                            &[]
                        },
                        stalled: match seed % 5 {
                            3 => &places[middle..=middle],
                            4 => &places,
                            _ => &[],
                        },
                        overload_lifted: seed % 7 == 0,
                        ..Constraints::default()
                    };
                    let decide = |engine: &Engine| {
                        engine.decide(
                            &facts,
                            &PoolFacts::default(),
                            pool,
                            &cx,
                            &mut Scratch::new(),
                        )
                    };
                    let (got, want) = (decide(&folded), decide(&each));
                    let at = format!("set {set:#07b}, picker {which}, pool {number}, seed {seed}");
                    assert_eq!(got, want, "{at}");
                    assert!(!got.trace().drew_first(), "{at}: a whole pass");
                    // Whether a row stands that every filter keeps:
                    // the folded test decides such a pool, and the
                    // stages one by one any other.
                    let stands = pool
                        .iter()
                        .any(|c| filters.iter().all(|f| f.keep(&facts, c, &cx)));
                    let picked = matches!(got, Decision::Pick(_));
                    stood += u32::from(stands);
                    fell += u32::from(!stands);
                    shed += u32::from(!picked);
                    aside += u32::from(!stands && picked);
                }
            }
        }
    }
    // Both paths decided often: pools the folded test kept rows of,
    // pools a stage emptied and shed, and pools a soft filter or the
    // all-vetoed rule kept rows of behind an empty fold.
    assert!(
        stood > 10_000 && fell > 10_000 && shed > 5_000 && aside > 1_000,
        "{stood} stood, {fell} fell, {shed} shed, {aside} aside"
    );
}

/// `has-room` written bare is the host's bit and folds into the row
/// test. With a rule it is a stage of its own: its profile folds into
/// nothing, and a picker that draws first still draws first.
#[test]
fn has_room_with_a_rule_folds_into_nothing_and_still_draws_first() {
    let engine = |filter: &str| {
        let document: PolicySpec = serde_json::from_str(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":["healthy",{filter}],"pick":"random"}}}}}}"#
            ))
            .expect("a document");
        document.validate().expect("validates")
    };
    let bare = engine(r#""has-room""#);
    assert!(bare.row_tests[0].is_some(), "the host's bit folds");
    let rule = engine(r#"{"has-room":{"used":{"candidate.in_flight":1},"capacity":8}}"#);
    assert!(rule.row_tests[0].is_none(), "the rule is asked as a stage");
    assert!(draws_first(&bare.profiles[0]) && draws_first(&rule.profiles[0]));
}

/// A set that is not the plan's whole pool is asked a stage at a time:
/// an encode item's own rules stand ahead of the profile's filters, and
/// the fold holds none of them.
#[test]
fn a_whole_pass_over_an_encode_items_set_is_asked_a_stage_at_a_time() {
    use crate::stages::filters::tests::filters_of;
    let engine = engine_of(filters_of(0b11111), Box::new(Passthrough));
    assert!(engine.row_tests[0].is_some(), "the profile folds");
    // Two rows every filter keeps. The item's rule refuses the first.
    let pool: Vec<Candidate> = (0..2u32)
        .map(|i| Candidate {
            has_capacity: true,
            has_room: true,
            ec_compatible: i == 1,
            ..candidate(i, 0)
        })
        .collect();
    let facts = request(7);
    let (entry, index, profile) = engine
        .resolve(facts.model, Phase::Single)
        .expect("a binding");
    let item = Set {
        candidates: &pool,
        class: None,
        partnered: None,
        fold: None,
        encode: true,
        vetoed: entry.vetoed,
    };
    let whole = |set: Set<'_>| {
        let (shown, cx) = (PoolFacts::default(), Constraints::default());
        run_whole(
            &engine,
            index,
            &facts,
            &shown,
            set,
            &cx,
            &mut Buffers::default(),
        )
    };
    let (shown, cx) = (PoolFacts::default(), Constraints::default());
    let staged = run_single(
        profile,
        index,
        &facts,
        &shown,
        item,
        &cx,
        &mut Buffers::default(),
    );
    assert!(matches!(staged, Decision::Pick(_)), "the second row stands");
    assert_eq!(whole(item), staged, "the item's set");
    // The rule decides this pool: the plan's whole pool hands the first.
    let plain = Set {
        encode: false,
        ..item
    };
    assert_ne!(whole(plain), staged, "the whole pool");

    // The same under the all-vetoed rule: both rows over their
    // thresholds, so the rule's lifted pass decides. The whole pass
    // hands the rule its fold, and the rule asks it about the whole
    // pool alone: an item's set is lifted a stage at a time, and its
    // own rule still refuses the first row.
    let over: Vec<Candidate> = pool
        .iter()
        .map(|row| Candidate {
            health: Health {
                overloaded: true,
                ..row.health
            },
            ..*row
        })
        .collect();
    let hosted = filters_of(0b11111)
        .into_iter()
        .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
        .collect();
    let each = engine_of(hosted, Box::new(Passthrough));
    let lifted = |engine: &Engine, encode: bool| {
        let (entry, index, _) = engine
            .resolve(facts.model, Phase::Single)
            .expect("a binding");
        let set = Set {
            candidates: &over,
            class: None,
            partnered: None,
            fold: None,
            encode,
            vetoed: entry.vetoed,
        };
        let (shown, cx) = (PoolFacts::default(), Constraints::default());
        run_whole(
            engine,
            index,
            &facts,
            &shown,
            set,
            &cx,
            &mut Buffers::default(),
        )
    };
    let item = lifted(&engine, true);
    assert!(
        matches!(&item, Decision::Pick(pick) if pick.trace.lifted == LiftedVeto::Overload),
        "the rule steers the item"
    );
    assert_eq!(item, lifted(&each, true), "the item's set, lifted");
    assert_eq!(
        lifted(&engine, false),
        lifted(&each, false),
        "the pool, lifted"
    );
    assert_ne!(item, lifted(&engine, false), "the item's rule decides");
}

/// A folded engine beside a staged one over pools of every kind, each
/// on a scratch kept from decision to decision: every decision is
/// equal, and a whole pass's survivors are the stages' by a model of
/// them. A pass that left a list behind for the next decision to read
/// would part the two, so the scratches change hands every seventh
/// decision. The all-vetoed rule's pick is the fold's often.
#[test]
fn a_folded_engine_decides_as_a_staged_one_on_scratches_that_are_kept() {
    use crate::stages::filters::tests::filters_of;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = |n: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % n
    };
    let (mut whole, mut drawn, mut emptied) = (0u32, 0u32, 0u32);
    let (mut steered, mut aside) = (0u32, 0u32);
    for set in 0..32u8 {
        for turn in 0..3usize {
            for (which, picker) in PICKERS.into_iter().enumerate() {
                // The profile's filters in an order of the walk's own.
                let order = |mut filters: Vec<Box<dyn Filter>>| {
                    if !filters.is_empty() {
                        let by = turn % filters.len();
                        filters.rotate_left(by);
                        if turn == 2 {
                            filters.reverse();
                        }
                    }
                    filters
                };
                let folded = engine_of(order(filters_of(set)), picker());
                let hosted: Vec<Box<dyn Filter>> = order(filters_of(set))
                    .into_iter()
                    .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
                    .collect();
                let each = engine_of(hosted, picker());
                let filters = order(filters_of(set));
                // One scratch an engine, kept from decision to decision.
                let (mut ours, mut theirs) = (Scratch::new(), Scratch::new());
                for step in 0..160u32 {
                    let rows = 1 + u32::try_from(next(24)).expect("a small pool");
                    let pool: Vec<Candidate> = (0..rows)
                        .map(|i| {
                            let held = u32::try_from(next(5)).expect("a small count");
                            let mut c = Candidate {
                                has_capacity: next(4) != 0,
                                has_room: next(3) != 0,
                                ..candidate(i, held)
                            };
                            c.health.healthy = next(6) != 0;
                            c.health.overloaded = next(3) == 0;
                            c.health.stalled = next(9) == 0;
                            c.health.draining = next(11) == 0;
                            c.health.breaker_open = next(13) == 0;
                            c
                        })
                        .collect();
                    let (mut tried, mut excluded) = (Vec::new(), Vec::new());
                    let mut stalled = Vec::new();
                    let (t, e, m) = (next(3), next(4), next(4));
                    for c in &pool {
                        if t == 0 && next(3) == 0 {
                            tried.push(c.id);
                        }
                        if e == 0 && next(4) == 0 {
                            excluded.push(c.id);
                        }
                        if m == 0 && next(3) == 0 {
                            stalled.push(c.position);
                        }
                    }
                    let facts = RequestFacts {
                        tried: &tried,
                        ..request(next(1 << 20))
                    };
                    let cx = Constraints {
                        excluded: &excluded,
                        stalled: &stalled,
                        overload_lifted: next(6) == 0,
                        ..Constraints::default()
                    };
                    let shown = PoolFacts::default();
                    let got = folded.decide(&facts, &shown, &pool, &cx, &mut ours);
                    let want = each.decide(&facts, &shown, &pool, &cx, &mut theirs);
                    let at = format!("set {set:#07b}, turn {turn}, picker {which}, step {step}");
                    assert_eq!(got, want, "{at}");
                    if step % 7 == 6 {
                        std::mem::swap(&mut ours, &mut theirs);
                    }
                    // The model of the staged pass: the hard filters,
                    // then each soft filter while it leaves a row.
                    let keeps = |f: &dyn Filter, i: u32| f.keep(&facts, &pool[i as usize], &cx);
                    let mut stand: Vec<u32> = (0..rows).collect();
                    for f in filters.iter().filter(|f| !f.soft()) {
                        stand.retain(|&i| keeps(f.as_ref(), i));
                    }
                    if stand.is_empty() {
                        // Equal to the staged twin, not modelled: a shed,
                        // or the rule's pick behind its lifted pass.
                        emptied += 1;
                        steered += u32::from(got.trace().lifted == LiftedVeto::Overload);
                        continue;
                    }
                    let mut stepped = false;
                    for f in filters.iter().filter(|f| f.soft()) {
                        let left: Vec<u32> = stand
                            .iter()
                            .copied()
                            .filter(|&i| keeps(f.as_ref(), i))
                            .collect();
                        if left.is_empty() {
                            stepped = true;
                        } else {
                            stand = left;
                        }
                    }
                    aside += u32::from(stepped);
                    let Decision::Pick(pick) = &got else {
                        panic!("{at}: no pick over rows that stand: {got:?}")
                    };
                    let named: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
                    assert!(
                        named.iter().all(|row| stand.contains(row)),
                        "{at}: {named:?} of {stand:?}"
                    );
                    if got.trace().drew_first() {
                        drawn += 1;
                    } else {
                        whole += 1;
                        let counted = got.trace().survivors.get().map(|n| n as usize);
                        assert_eq!(counted, Some(stand.len()), "{at}: the survivors counted");
                    }
                }
            }
        }
    }
    // The walk reached every kind of pass: whole passes and draws over
    // rows that stand, pools the hard filters emptied, the rule's
    // picks among them, and passes where a soft filter stepped aside.
    assert!(
        whole > 10_000 && drawn > 10_000 && emptied > 500 && steered > 500 && aside > 500,
        "{whole} whole, {drawn} drawn, {emptied} emptied, {steered} steered, {aside} aside"
    );
}

#[test]
fn a_draw_asks_the_folded_test_of_its_own_profile() {
    // Two profiles of one engine, each folded: the first keeps a row
    // over its ceiling and the second refuses it. A draw under either
    // asks that profile's own test.
    let profile = |name: &str, filters: Vec<Box<dyn Filter>>| Profile {
        name: name.into(),
        filters,
        scorers: Vec::new(),
        picker: Box::new(Random),
    };
    let binding = |model: &str, profile: u16| Binding {
        model: model.into(),
        profile,
        ..single_binding()
    };
    let engine = Engine::new(
        vec![
            profile("lenient", vec![Box::new(Healthy)]),
            profile("strict", vec![Box::new(Healthy), Box::new(NotOverloaded)]),
        ],
        vec![binding("lenient", 0), binding("strict", 1)],
        1,
        Needs::default(),
    );
    assert!(engine.row_tests[0].is_some() && engine.row_tests[1].is_some());
    assert_ne!(engine.row_tests[0], engine.row_tests[1]);
    // Every second row is over its ceiling.
    let pool: Vec<Candidate> = (0..64u32)
        .map(|i| {
            let mut c = candidate(i, 0);
            c.health.overloaded = i % 2 == 0;
            c
        })
        .collect();
    let (mut drew, mut over) = ([0u32; 2], [0usize; 2]);
    for (at, model) in ["lenient", "strict"].into_iter().enumerate() {
        for seed in 0..64u64 {
            let facts = RequestFacts {
                model,
                ..request(seed)
            };
            let decision = engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            drew[at] += u32::from(decision.trace().drew_first());
            let handed = positions(&decision);
            over[at] += handed.iter().filter(|&&row| row % 2 == 0).count();
        }
    }
    assert!(drew[0] == 64 && drew[1] > 48, "{drew:?} drew");
    // The first profile hands such rows out and the second none.
    assert!(over[0] > 0, "the lenient profile handed none");
    assert_eq!(
        over[1], 0,
        "the strict profile handed a row over its ceiling"
    );
}

/// A draw asks the folded test of the profile its binding names: two
/// profiles are bound crosswise, so a binding's place and its profile's
/// index part. The lenient profile hands out overloaded rows and the
/// strict one hands out none.
#[test]
fn a_draw_asks_the_test_of_the_profile_its_binding_names() {
    let profile = |name: &str, filters: Vec<Box<dyn Filter>>| Profile {
        name: name.into(),
        filters,
        scorers: Vec::new(),
        picker: Box::new(Random),
    };
    let binding = |model: &str, profile: u16| Binding {
        model: model.into(),
        profile,
        ..single_binding()
    };
    let engine = Engine::new(
        vec![
            profile("lenient", vec![Box::new(Healthy)]),
            profile("strict", vec![Box::new(Healthy), Box::new(NotOverloaded)]),
        ],
        vec![binding("strict", 1), binding("lenient", 0)],
        1,
        Needs::default(),
    );
    // Every other row is overloaded.
    let pool: Vec<Candidate> = (0..64u32)
        .map(|i| {
            let mut row = candidate(i, 0);
            row.health.overloaded = i % 2 == 0;
            row
        })
        .collect();
    let (mut drew, mut overloaded) = ([0u32; 2], [0usize; 2]);
    for (at, model) in ["lenient", "strict"].into_iter().enumerate() {
        for seed in 0..64u64 {
            let facts = RequestFacts {
                model,
                ..request(seed)
            };
            let decision = drawn(&engine, &facts, &pool);
            drew[at] += u32::from(decision.trace().drew_first());
            overloaded[at] += positions(&decision)
                .iter()
                .filter(|&&row| row % 2 == 0)
                .count();
        }
    }
    assert!(drew[0] == 64 && drew[1] > 48, "{drew:?} decisions drew");
    assert!(
        overloaded[0] > 0 && overloaded[1] == 0,
        "overloaded rows handed out: {overloaded:?}"
    );
}
