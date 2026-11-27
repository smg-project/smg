//! A pair's shadow profiles under the rule: a decode's shadow, a prefill's
//! shadow, and what each reads of the live decision.

use super::*;

/// The shadow of a decode asked again reads the partners its call read
/// by the word that call left in the scratch, never by its pick. Where
/// every partner is doubted the call reads them all, and the rule's
/// liveness tier draws among equals: a draw that lands on a first
/// partner says nothing of the partners read. A twin of the live
/// profile names the live decode under every seed, behind a pair's
/// first decision that was made among first partners and left its own
/// word in the same scratch.
#[test]
fn a_later_decode_shadow_reads_every_partner_its_call_read_past_a_doubt() {
    let engine = pair_engine(false);
    let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let open = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    // Two first partners of the held prefill and one partner that is
    // none, equally loaded and all doubted.
    let decodes = [
        classed(doubted(0, 0), 1),
        classed(doubted(1, 0), 1),
        classed(doubted(2, 0), 2),
    ];
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let held = prefills[1].id;
    let mut among_first = 0;
    for seed in 0..16 {
        let (req, scratch) = (facts(seed), &mut Scratch::new());
        let first_pair = first_pools(&prefills, &open);
        let first = engine.decide_pair(&req, &pool, first_pair, &cx, &cx, scratch);
        assert_eq!(pair_of(&first), (1, 1), "seed {seed}");
        let pools = first_pools(&prefills, &decodes);
        let again = engine.decide_decode(&req, &pool, pools, held, None, &cx, scratch);
        assert_eq!(again.trace().lifted, LiftedVeto::Liveness, "seed {seed}");
        let live = pair_of(&again).1;
        among_first += u32::from(live != 2);
        let later =
            engine.shadow_decode_later(0, &req, &pool, pools, held, None, &cx, scratch, &|_| None);
        assert_eq!(
            later.map(|shadow| order(&shadow)[0]),
            Some(live),
            "seed {seed}"
        );
    }
    // The draws landed on a first partner, and on the partner that is
    // none: the call read every partner.
    assert!((1..16).contains(&among_first), "{among_first} of 16");
}

/// A pair's decode shadow is put to its binding's rule as its live side
/// is, at the pair's first decision and where the decode is asked
/// again. Beside a decode the rule steered among partners over their
/// thresholds, the twin names that decode and the veto the rule lifted
/// for it. Where the decode is asked again past a doubt, the twin
/// follows it there too. A first decision's pair past a doubt is the
/// default's, and the twin follows the default: it names that decode
/// and the liveness veto. Beside a pair no rule placed the twin names
/// the live decode, and no veto.
#[test]
fn a_pairs_decode_shadow_is_put_to_its_bindings_rule() {
    let (engine, prefills) = (pair_engine(false), [classed(free(0, 0), 1)]);
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let shadows = |made: [fn(u32, u32) -> Candidate; 2]| {
        let decodes = [made[0](0, 2), made[1](1, 1)].map(|worker| classed(worker, 1));
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let (held, side, scratch) = (prefills[0].id, PairSide::Decode, &mut Scratch::new());
        // What a twin said beside `live`: its first decode and the veto
        // its trace names. A twin that speaks names the live decode.
        let beside = |live: &Decision, twin: Option<Decision>| {
            twin.map(|twin| {
                assert_eq!(order(&twin)[0], pair_of(live).1);
                (order(&twin)[0], twin.trace().lifted)
            })
        };
        let first = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        let pick = first.pick().expect("a pair");
        let twin = engine.shadow_side(
            0,
            side,
            &req,
            &pool,
            pools,
            &cx,
            &cx,
            pick,
            scratch,
            &|_| None,
        );
        let twin = beside(&first, twin);
        let again = engine.decide_decode(&req, &pool, pools, held, None, &cx, scratch);
        let later =
            engine.shadow_decode_later(0, &req, &pool, pools, held, None, &cx, scratch, &|_| None);
        (twin, beside(&again, later))
    };
    let none = Some((0, LiftedVeto::None));
    assert_eq!(shadows([free, over]), (none, none));
    let steered = Some((1, LiftedVeto::Overload));
    assert_eq!(shadows([over, over]), (steered, steered));
    // Past a doubt the first pair is the default's, and its twin takes
    // the default's decode in the default's own tier.
    let doubt = Some((1, LiftedVeto::Liveness));
    assert_eq!(shadows([doubted, doubted]), (doubt, doubt));
}

/// The binding's word stands for a decode's shadow as for its live
/// side. A live decode side that reads no overload veto takes a busy
/// partner by its own stages. Its twin reads the veto, which leaves
/// the live prefill's partners nobody. Under a binding that steers
/// the rule takes the partner with the fewest requests for the twin,
/// and under one that sheds the twin is shed as saturated: at the
/// pair's first decision and where the decode is asked again.
#[test]
fn a_pairs_decode_shadow_reads_its_bindings_word() {
    let prefills = [classed(free(0, 0), 1)];
    let decodes = [over(0, 2), over(1, 1)].map(|worker| classed(worker, 1));
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let said = |shed: bool| {
        let healthy: Vec<Box<dyn Filter>> = vec![Box::new(Healthy)];
        let engine = pair_engine_of(shed, [vetoes(), healthy]);
        let (held, side, scratch) = (prefills[0].id, PairSide::Decode, &mut Scratch::new());
        let first = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        let live = first.pick().expect("a pair");
        assert_eq!(live.trace.lifted, LiftedVeto::None);
        let twin = engine.shadow_side(
            0,
            side,
            &req,
            &pool,
            pools,
            &cx,
            &cx,
            live,
            scratch,
            &|_| None,
        );
        let again = engine.decide_decode(&req, &pool, pools, held, None, &cx, scratch);
        assert!(again.pick().is_some(), "a decode: {again:?}");
        let later =
            engine.shadow_decode_later(0, &req, &pool, pools, held, None, &cx, scratch, &|_| None);
        [twin, later].map(|shadow| shadow.expect("the decode's shadow"))
    };
    for steered in said(false) {
        let taken = (order(&steered)[0], steered.trace().lifted);
        assert_eq!(taken, (1, LiftedVeto::Overload));
    }
    for shed in said(true) {
        assert_eq!(shed_of(&shed), (ShedReason::Saturated, "not-overloaded"));
    }
}

/// At a pair's first decision the decode's shadow is put to both tiers
/// of its binding's rule. A live decode side that reads no health stage
/// takes a doubted partner by its own stages. Its twin's filters doubt
/// every partner of the live prefill, and the rule's liveness tier
/// steers the twin to the partner that holds the fewest requests.
#[test]
fn a_first_decisions_decode_shadow_is_steered_past_a_doubt() {
    let prefills = [classed(free(0, 0), 1)];
    let decodes = [doubted(0, 2), doubted(1, 1)].map(|worker| classed(worker, 1));
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let loaded: Vec<Box<dyn Filter>> = vec![Box::new(NotOverloaded)];
    let (engine, scratch) = (
        pair_engine_of(false, [vetoes(), loaded]),
        &mut Scratch::new(),
    );
    let first = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
    let live = first.pick().expect("a pair");
    assert_eq!(live.trace.lifted, LiftedVeto::None);
    let side = PairSide::Decode;
    let twin = engine.shadow_side(
        0,
        side,
        &req,
        &pool,
        pools,
        &cx,
        &cx,
        live,
        scratch,
        &|_| None,
    );
    let twin = twin.expect("the decode's shadow");
    let taken = (order(&twin)[0], twin.trace().lifted);
    assert_eq!(taken, (1, LiftedVeto::Liveness));
}

/// Beside a pair the default made a twin follows the default as its
/// live side does. It agrees on both sides, in each tier: its prefill
/// is the prefill of the least pair, and its decode that prefill's
/// usable partner of the fewest requests.
///
/// Beside a pair whose decode side alone the rule steered, both shadows
/// speak as well. The prefill's live side was put to no rule, and the
/// decode's shadow is put to its live side's. One scratch serves the
/// pairs, the default's first.
///
/// Under a binding that sheds the default lifts the liveness veto
/// alone, and never onto a worker over its thresholds. Each twin keeps
/// to the binding's word as its live side does.
#[test]
fn a_twin_follows_the_pairs_default_on_both_sides() {
    let (engine, scratch) = (pair_engine(false), &mut Scratch::new());
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let mut said = |engine: &Engine, prefills: [Candidate; 2], decodes: [Candidate; 2]| {
        let prefills = prefills.map(|worker| classed(worker, 1));
        let decodes = decodes.map(|worker| classed(worker, 1));
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        let pick = live.pick().expect("a pair");
        let twins = PairSide::BOTH.map(|side| {
            let twin = engine.shadow_side(
                0,
                side,
                &req,
                &pool,
                pools,
                &cx,
                &cx,
                pick,
                scratch,
                &|_| None,
            );
            twin.map(|shadow| order(&shadow)[0])
        });
        (pair_of(&live), live.trace().lifted, twins)
    };
    // The default's pair, by load: the second prefill holds the fewer
    // requests, and the second decode none.
    let open = [free(0, 2), free(1, 0)];
    for tier in [LiftedVeto::Overload, LiftedVeto::Liveness] {
        let made = if tier == LiftedVeto::Liveness {
            doubted
        } else {
            over
        };
        let refused = [made(0, 3), made(1, 1)];
        let pair = said(&engine, refused, open);
        assert_eq!(pair, ((1, 1), tier, [Some(1), Some(1)]));
    }
    // The rule's pair: every decode is over, and the prefill side open.
    let busy = [over(0, 2), over(1, 0)];
    let (pair, tier, twins) = said(&engine, [free(0, 0), free(1, 0)], busy);
    assert_eq!((pair.1, tier), (1, LiftedVeto::Overload));
    assert_eq!(twins, [Some(pair.0), Some(1)]);
    // A binding that sheds: every prefill is doubted. On each side the
    // worker over its thresholds holds the fewest requests, and neither
    // the default nor a twin takes it.
    let sheds = pair_engine(true);
    let doubts = [doubted_over(0, 0), doubted(1, 1)];
    let pair = said(&sheds, doubts, [over(0, 0), free(1, 5)]);
    assert_eq!(pair, ((1, 1), LiftedVeto::Liveness, [Some(1), Some(1)]));
}

/// The prefill's shadow leaves what the decode's shadow reads. Beside a
/// pair the profiles made, a prefill's shadow that sheds asks the
/// pair's default, which writes both buffers and their marks. The
/// decode's shadow behind it on the same scratch still takes its ruled
/// run over the live prefill's partners. It names the live decode, the
/// busier one here. The default's own step would take the decode that
/// holds the fewest requests.
#[test]
fn a_decodes_shadow_takes_its_ruled_run_behind_a_prefills_that_asked_the_default() {
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    // The live prefill profile weighs and keeps its busy holder by the
    // lift. Its shadow does not weigh: it is shed, and asks the default.
    let weighs = |weighs| {
        Some(Holds {
            holder: 0,
            reads: true,
            weighs,
        })
    };
    let engine = pair_engine_holding(false, [vetoes(), vetoes()], [weighs(true), weighs(false)]);
    let prefills = [classed(over(0, 3), 1), classed(over(1, 1), 1)];
    let said = |decodes: [Candidate; 2]| {
        let decodes = decodes.map(|worker| classed(worker, 1));
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let scratch = &mut Scratch::new();
        let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        let pick = live.pick().expect("a pair");
        let twins = PairSide::BOTH.map(|side| {
            let twin = engine.shadow_side(
                0,
                side,
                &req,
                &pool,
                pools,
                &cx,
                &cx,
                pick,
                scratch,
                &|_| None,
            );
            order(&twin.expect("the side's shadow"))[0]
        });
        (pair_of(&live), twins)
    };
    // Whichever decode the live profile's picker takes, the other one
    // holds no request: the default's own step would take that one.
    for busier in [0, 1] {
        let mut decodes = [free(0, 0), free(1, 0)];
        decodes[busier].in_flight = 4;
        let ((prefill, decode), twins) = said(decodes);
        assert_eq!(prefill, 0);
        // The prefill's shadow took the default's prefill, and the
        // decode's shadow the live decode.
        assert_eq!(twins, [1, decode]);
    }
}

/// Among first partners a prefill's shadow that sheds over the prefills
/// its live side was shown goes the live decision's way with its own
/// profile. One of those prefills that stands but for its admission
/// limit keeps the request under the limit's stage, and the shed names
/// the prefill side. Else the shadow runs over every open class and
/// takes its own pick there. Where that run sheds too it takes the
/// default's prefill, by load. A default's prefill at its admission
/// limit is the limit's shed, in place of the run's own refusal.
///
/// The default asked for the shadow reads every open class, as the live
/// default does. A prefill of another open class that stands but for
/// its limit makes it answer nothing, and the run's own refusal stands.
#[test]
fn a_prefills_shadow_shed_among_first_partners_goes_the_live_decisions_way() {
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    // The live prefill profile reads the health stage alone, so it
    // takes the prefill with a first partner open whatever its load
    // and its limit. Its shadow reads the vetoes and the limit.
    let healthy = || -> Vec<Box<dyn Filter>> { vec![Box::new(Healthy)] };
    let mut limited = vetoes();
    limited.push(Box::new(HasCapacity));
    let engine = pair_engine_shadowed(false, [healthy(), vetoes(), limited], [None, None]);
    let room = |mut worker: Candidate, class: u32| {
        worker.has_capacity = true;
        classed(worker, class)
    };
    // Place 1 of each side holds the two workers that pair first.
    let open = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let shadow_beside = |prefills: [Candidate; 2], decodes: [Candidate; 2]| {
        let (pools, scratch) = (first_pools(&prefills, &decodes), &mut Scratch::new());
        let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        assert_eq!(pair_of(&live), (1, 1));
        let pick = live.pick().expect("a pair");
        let side = PairSide::Prefill;
        let shadow = engine.shadow_side(
            0,
            side,
            &req,
            &pool,
            pools,
            &cx,
            &cx,
            pick,
            scratch,
            &|_| None,
        );
        shadow.expect("the prefill's shadow")
    };
    // The prefill shown is at its limit and nothing else refuses it: the
    // request waits on the limit, beside a prefill with room.
    let waits = shadow_beside([room(free(0, 0), 2), classed(free(1, 0), 1)], open);
    let limit = (ShedReason::NoEndpoints, "has-capacity");
    assert_eq!(sided(&waits), (limit, Some(PairSide::Prefill)));
    // The prefill shown is over its thresholds: the shadow's profile
    // runs over every open class and takes the free prefill itself.
    let own = shadow_beside([room(free(0, 0), 2), room(over(1, 0), 1)], open);
    assert_eq!((order(&own)[0], own.trace().lifted), (0, LiftedVeto::None));
    // Every prefill is over: that run sheds too, and the shadow takes
    // the default's prefill, the one of the least pair.
    let default = shadow_beside([room(over(0, 1), 2), room(over(1, 5), 1)], open);
    let taken = (order(&default)[0], default.trace().lifted);
    assert_eq!(taken, (0, LiftedVeto::Overload));
    // That prefill is at its limit: the shadow's answer is the limit's
    // shed of the default, which names the prefill side and no lifted
    // veto. The run's own refusal was the overload veto's.
    let full = shadow_beside([classed(over(0, 1), 2), room(over(1, 5), 1)], open);
    assert_eq!(sided(&full), (limit, Some(PairSide::Prefill)));
    assert_eq!(full.trace().lifted, LiftedVeto::None);
    // Every decode is over, so the default would pair past the overload
    // veto. The prefill of the other open class stands but for its
    // limit: the default answers nothing, and the run's own refusal
    // stands. Read over the first partners' classes alone, the default
    // would take the prefill over its thresholds, which holds fewer.
    let busy = [classed(over(0, 0), 2), classed(over(1, 0), 1)];
    let kept = shadow_beside([classed(free(0, 3), 2), room(over(1, 0), 1)], busy);
    assert_eq!(shed_of(&kept), (ShedReason::NoEndpoints, "has-capacity"));
}

/// A picker that takes no row: its profile's run is a decline.
#[derive(Debug)]
struct Declines;

impl Lifecycle for Declines {}

impl Picker for Declines {
    fn name(&self) -> &'static str {
        "declines"
    }

    fn pick(
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

/// A prefill shadow's picker that declines keeps its decline among
/// first partners, as a live prefill's does. The shadow asks no limit,
/// no second run and no default. Taken for a refusal, the decline
/// would turn into the limit's shed: the prefill shown stands but for
/// its limit.
#[test]
fn a_prefill_shadows_decline_passes_through_among_first_partners() {
    let profile_of = |name: &str, picker: Box<dyn Picker>| Profile {
        name: name.into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![],
        picker,
    };
    let engine = Engine::new(
        vec![
            profile_of("prefill", Box::new(Best)),
            profile_of("decode", Box::new(Best)),
            profile_of("prefill-twin", Box::new(Declines)),
        ],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(1),
            shadow: Some(2),
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    // Place 1 of each side holds the two workers that pair first.
    let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let decodes = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let (pools, scratch) = (first_pools(&prefills, &decodes), &mut Scratch::new());
    let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
    assert_eq!(pair_of(&live), (1, 1));
    let pick = live.pick().expect("a pair");
    let side = PairSide::Prefill;
    let shadow = engine.shadow_side(
        0,
        side,
        &req,
        &pool,
        pools,
        &cx,
        &cx,
        pick,
        scratch,
        &|_| None,
    );
    let declined = shadow.expect("the prefill's shadow");
    assert_eq!(shed_of(&declined), (ShedReason::PolicyDeclined, "declines"));
}

/// A prefill's shadow asks the pair's default only under a live decode
/// profile whose hard filters each read a row alone and hold no scope.
/// The default asks those filters behind the decision they stood in,
/// where another filter may answer otherwise. Under such a filter the
/// side says none, and the filter is asked as often as with no shadow.
/// A soft filter is not asked by the default, and stops none.
#[test]
fn a_prefills_shadow_says_none_where_its_default_would_ask_a_decode_filter_again() {
    /// Keeps every row and counts the rows it was asked about. `rows` is
    /// its word that it reads a row alone.
    #[derive(Debug)]
    struct Keeps {
        rows: bool,
        soft: bool,
        scope: Option<Arc<Scope>>,
        asked: Arc<AtomicUsize>,
    }

    impl Filter for Keeps {
        fn name(&self) -> &'static str {
            "keeps"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            self.asked.fetch_add(1, Ordering::Relaxed);
            true
        }
        fn soft(&self) -> bool {
            self.soft
        }
        fn row_wise(&self) -> bool {
            self.rows
        }
        fn scope(&self) -> Option<&dyn DecisionScope> {
            let scope = self.scope.as_deref()?;
            Some(scope)
        }
    }

    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    // Every prefill is over its thresholds. The live prefill profile reads
    // the health stage alone and takes one. Its shadow reads the vetoes:
    // it sheds, and the default takes the prefill of the least pair.
    let prefills = [over(0, 1), over(1, 5)].map(|worker| classed(worker, 1));
    let decodes = [free(0, 0), free(1, 0)].map(|worker| classed(worker, 1));
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    // What the shadow answers under the filter, and whether its run asked
    // the filter again.
    let shadow_under = |rows: bool, soft: bool, scoped: bool| {
        let scope = scoped.then(|| Arc::new(Scope::default()));
        let asked = Arc::new(AtomicUsize::new(0));
        let keeps = Keeps {
            rows,
            soft,
            scope,
            asked: Arc::clone(&asked),
        };
        let healthy: Vec<Box<dyn Filter>> = vec![Box::new(Healthy)];
        let mut decode = vetoes();
        decode.push(Box::new(keeps));
        let engine = pair_engine_shadowed(false, [healthy, decode, vetoes()], [None, None]);
        let scratch = &mut Scratch::new();
        let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        let pick = live.pick().expect("a pair");
        let alone = asked.load(Ordering::Relaxed);
        let side = PairSide::Prefill;
        let shadow = engine.shadow_side(
            0,
            side,
            &req,
            &pool,
            pools,
            &cx,
            &cx,
            pick,
            scratch,
            &|_| None,
        );
        let again = asked.load(Ordering::Relaxed) > alone;
        let taken = shadow.map(|shadow| (order(&shadow)[0], shadow.trace().lifted));
        (taken, again)
    };
    let default = Some((0, LiftedVeto::Overload));
    // A hard filter that reads a row alone and holds no scope: the
    // default asks it.
    assert_eq!(shadow_under(true, false, false), (default, true));
    // One that does not say it reads a row alone.
    assert_eq!(shadow_under(false, false, false), (None, false));
    // One that reads a row alone and holds a scope.
    assert_eq!(shadow_under(true, false, true), (None, false));
    // A soft filter is not asked by the default, whatever it says.
    assert_eq!(shadow_under(false, true, true), (default, false));
}

/// A prefill's shadow that panics inside the pair's default leaves the
/// scratch as the pair's decision left it: the classes it was shown and
/// the mark on the prefill's buffer. A host catches each side's shadow
/// by itself, so the decode's shadow runs on that scratch behind the
/// panic. It answers as it does where no prefill's shadow ran, by its
/// ruled run among the live prefill's first partners. Left with the
/// default's mark, it would take the default's second step, and the
/// decode that holds the fewest requests.
#[test]
fn a_prefills_shadow_that_panics_in_the_default_leaves_the_scratch_as_it_found_it() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Panics where it is asked about a row.
    #[derive(Debug)]
    struct Unwinds;

    impl Filter for Unwinds {
        fn name(&self) -> &'static str {
            "unwinds"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            panic!("this stage unwinds where it is asked");
        }
    }

    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    // The live prefill profile reads no health stage and takes a doubted
    // prefill. Its shadow lists the health stage ahead of the stage that
    // panics. The health stage refuses every prefill in the shadow's own
    // runs and in the default's overload pass, which leaves its mark. The
    // default's liveness pass reads the prefills with no doubt, and asks
    // the stage that panics.
    let loaded: Vec<Box<dyn Filter>> = vec![Box::new(NotOverloaded)];
    let shadow: Vec<Box<dyn Filter>> = vec![Box::new(Healthy), Box::new(Unwinds)];
    let engine = pair_engine_shadowed(false, [loaded, vetoes(), shadow], [None, None]);
    // Place 1 of each side holds the two workers that pair first. The
    // other decode holds fewer requests.
    let prefills = [classed(doubted(0, 0), 2), classed(doubted(1, 0), 1)];
    let decodes = [classed(free(0, 0), 2), classed(free(1, 3), 1)];
    let pools = first_pools(&prefills, &decodes);
    let left = |scratch: &Scratch| {
        let classes = (scratch.classes.clone(), scratch.firsts.clone());
        (classes, scratch.single.lifted)
    };
    // What the decode's shadow says beside the pair, behind a prefill's
    // shadow that panicked or with none.
    let decode_said = |scratch: &mut Scratch, panics: bool| {
        let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        assert_eq!(pair_of(&live), (1, 1));
        let pick = live.pick().expect("a pair");
        if panics {
            // Among first partners the two lists of classes differ.
            let found = left(scratch);
            assert_ne!(found.0 .0, found.0 .1);
            assert_eq!(found.1, LiftedVeto::None);
            let side = PairSide::Prefill;
            let unwound = catch_unwind(AssertUnwindSafe(|| {
                engine.shadow_side(
                    0,
                    side,
                    &req,
                    &pool,
                    pools,
                    &cx,
                    &cx,
                    pick,
                    scratch,
                    &|_| None,
                )
            }));
            assert!(unwound.is_err(), "the stage unwound");
            assert_eq!(left(scratch), found);
        }
        let side = PairSide::Decode;
        let twin = engine.shadow_side(
            0,
            side,
            &req,
            &pool,
            pools,
            &cx,
            &cx,
            pick,
            scratch,
            &|_| None,
        );
        let twin = twin.expect("the decode's shadow");
        (order(&twin), twin.trace().lifted)
    };
    let alone = decode_said(&mut Scratch::new(), false);
    assert_eq!((alone.0[0], alone.1), (1, LiftedVeto::None));
    assert_eq!(decode_said(&mut Scratch::new(), true), alone);
}

/// Beside a pair the default made, a decode's shadow sheds where its own
/// hard filters keep no partner of the live prefill in the pass that
/// paired. The shed is a decode side's with no partner: it stands under
/// `partners`, and names the decode side and no lifted veto.
///
/// Under a binding that sheds, the default lifts the liveness veto
/// alone. The live decode profile reads the health stage alone, and
/// keeps a decode over its thresholds there. Its twin reads the overload
/// veto too, and keeps none.
#[test]
fn a_decodes_shadow_with_no_partner_beside_the_default_sheds_under_partners() {
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let healthy: Vec<Box<dyn Filter>> = vec![Box::new(Healthy)];
    let engine = pair_engine_of(true, [vetoes(), healthy]);
    let prefills = [doubted(0, 0), doubted(1, 1)].map(|worker| classed(worker, 1));
    let decodes = [over(0, 0), over(1, 2)].map(|worker| classed(worker, 1));
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let scratch = &mut Scratch::new();
    let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
    let made = (pair_of(&live), live.trace().lifted);
    assert_eq!(made, ((0, 0), LiftedVeto::Liveness));
    let pick = live.pick().expect("a pair");
    let side = PairSide::Decode;
    let shadow = engine.shadow_side(
        0,
        side,
        &req,
        &pool,
        pools,
        &cx,
        &cx,
        pick,
        scratch,
        &|_| None,
    );
    let shadow = shadow.expect("the decode's shadow");
    let partners = (ShedReason::NoEndpoints, "partners");
    assert_eq!(sided(&shadow), (partners, Some(PairSide::Decode)));
    assert_eq!(shadow.trace().lifted, LiftedVeto::None);
}
