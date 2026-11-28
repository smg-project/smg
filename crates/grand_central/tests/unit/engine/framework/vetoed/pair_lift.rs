//! The lift that a prefill profile's own word gives its run: a profile that
//! weighs load against affinity, and what the pair's default is told of it.

use super::*;

/// The lift rests on a scorer's own word, and on nothing a stage is
/// fed. The busy prefill holds the session's prefix and a free one
/// stands beside it. A scorer that says it weighs load against affinity
/// is lifted and keeps its holder, whether it declares that it reads
/// affinity or not. A scorer that reads affinity and says no such word
/// keeps the veto, and the request moves to the free prefill: a stage
/// that is fed the column has not said that a gate of its own answers
/// a holder loaded too far. So does a profile whose filter declares the
/// need. The gateway lifts on its policy's own word the same way, and
/// on nothing the policy reads
/// (`model_gateway/src/routers/common/placement.rs:2575-2577`, at main
/// `82a135b1b`).
///
/// The default answers for a weighing profile wherever its lifted run
/// sheds. Where every prefill is doubted, the pair goes to the least
/// loaded past the doubt, as under any profile.
#[test]
fn the_lift_rests_on_a_scorers_own_word_and_on_no_need() {
    #[derive(Debug)]
    struct ReadsAffinity;
    impl Filter for ReadsAffinity {
        fn name(&self) -> &'static str {
            "reads-affinity"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
        fn needs(&self) -> Needs {
            Needs {
                affinity: true,
                ..Needs::default()
            }
        }
    }
    let decodes = [classed(free(0, 0), 1)];
    let prefills = [classed(over(0, 3), 1), classed(free(1, 0), 1)];
    let pair = |reads: bool, weighs: bool, filter_reads: bool| {
        let mut filters = vetoes();
        if filter_reads {
            filters.push(Box::new(ReadsAffinity));
        }
        let holder = 0;
        let live = Some(Holds {
            holder,
            reads,
            weighs,
        });
        let engine = pair_engine_holding(false, [filters, vetoes()], [live, None]);
        let placed = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(placed.trace().lifted, LiftedVeto::None);
        pair_of(&placed)
    };
    // The word lifts, beside the need and without it.
    assert_eq!(pair(true, true, false), (0, 0));
    assert_eq!(pair(false, true, false), (0, 0));
    // The need lifts nothing, a scorer's or a filter's.
    assert_eq!(pair(true, false, false), (1, 0));
    assert_eq!(pair(false, false, true), (1, 0));
    let suspect = [classed(doubted(0, 2), 1), classed(doubted(1, 0), 1)];
    let paired = decide_pair(&holding(false, 0, true), &suspect, &decodes);
    assert_eq!(pair_of(&paired), (1, 0));
    assert_eq!(paired.trace().lifted, LiftedVeto::Liveness);
}

/// A pair engine whose prefill profile holds the two vetoes and a stage
/// that prefers the prefill at `holder` and reads affinity ([`Holds`]):
/// `weighs` is that stage's word on a gate of its own. `shed` makes its
/// binding shed.
fn holding(shed: bool, holder: u32, weighs: bool) -> Engine {
    let live = Some(Holds {
        holder,
        reads: true,
        weighs,
    });
    pair_engine_holding(shed, [vetoes(), vetoes()], [live, None])
}

/// Under a binding that steers, a prefill profile that weighs load
/// against affinity keeps a session on the prefill that holds its
/// prefix while that prefill is over its thresholds, beside a free
/// prefill too: its run has the overload veto lifted, and its own
/// stages weigh the load. The pick stands beside a prefill under its
/// thresholds, so the pair's trace names no lifted veto. A binding that
/// sheds keeps the veto, and so does a profile whose stage reads
/// affinity and says no word of a gate: each passes the busy holder
/// over. The gateway's own pair selection lifts the veto so
/// (`model_gateway/src/routers/common/placement.rs:2375-2394`; its case
/// `an_overloaded_prefill_holding_the_prefix_keeps_the_session_under_cache_aware`
/// at `:5881`, at main `55c35100d`).
#[test]
fn a_weighing_prefill_profile_keeps_a_session_on_its_busy_holder() {
    let prefills = [classed(over(0, 3), 1), classed(free(1, 0), 1)];
    let decodes = [classed(free(0, 0), 1)];
    let kept = decide_pair(&holding(false, 0, true), &prefills, &decodes);
    assert_eq!(pair_of(&kept), (0, 0));
    assert_eq!(kept.trace().lifted, LiftedVeto::None);
    for engine in [holding(true, 0, true), holding(false, 0, false)] {
        let moved = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(pair_of(&moved), (1, 0));
        assert_eq!(moved.trace().lifted, LiftedVeto::None);
    }
}

/// Where every prefill is over its thresholds, a weighing profile under
/// a binding that steers places the pair itself: the prefill its stages
/// prefer, not the least loaded. No prefill stood without the lift, so
/// the pair's trace says the lifted overload veto, as a kept key's pick
/// does over a pool vetoed on every worker. A profile whose stage says
/// no word of a gate leaves such a pool to the pair's default, which
/// takes the least loaded, and a binding that sheds sheds it as
/// saturated. The
/// gateway's own selection places such a pool by its prefill policy
/// (`model_gateway/src/routers/common/placement.rs:5853`, the case
/// `an_all_overloaded_prefill_pool_is_placed_by_the_prefill_policy`, at
/// main `55c35100d`).
#[test]
fn a_weighing_prefill_profile_places_a_pool_over_on_every_prefill() {
    let prefills = [classed(over(0, 5), 1), classed(over(1, 1), 1)];
    let decodes = [classed(free(0, 0), 1)];
    let placed = decide_pair(&holding(false, 0, true), &prefills, &decodes);
    assert_eq!(pair_of(&placed), (0, 0));
    assert_eq!(placed.trace().lifted, LiftedVeto::Overload);
    let by_load = decide_pair(&holding(false, 0, false), &prefills, &decodes);
    assert_eq!(pair_of(&by_load), (1, 0));
    assert_eq!(by_load.trace().lifted, LiftedVeto::Overload);
    let shed = decide_pair(&holding(true, 0, true), &prefills, &decodes);
    assert_eq!(shed_of(&shed), (ShedReason::Saturated, "not-overloaded"));
}

/// The lifted constraints of a weighing profile's run are a second
/// value: the pair's default judges the pool as it stands, with no
/// lift. Every prefill is over its thresholds, their partners are over
/// too, and the one free decode partners none of them. No prefill has
/// an open partner, so the prefill's run sheds and the default pairs by
/// load. Judged under the lift, the prefills would stand beside the
/// free decode, and the default would leave the refusal. The gateway
/// judges its default over such a pool "as it was before the lift"
/// (`model_gateway/src/routers/common/placement.rs:2486-2490`, at main
/// `55c35100d`).
#[test]
fn a_weighing_profiles_pool_is_judged_for_the_default_with_no_lift() {
    let prefills = [classed(over(0, 2), 1), classed(over(1, 1), 1)];
    let partners = [classed(over(0, 4), 1), classed(over(1, 0), 1)];
    let decodes = [partners[0], partners[1], classed(free(2, 0), 2)];
    let paired = decide_pair(&holding(false, 0, true), &prefills, &decodes);
    assert_eq!(pair_of(&paired), (1, 1));
    assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
}

/// A weighing shadow's default judges the pool with no lift too, beside
/// a live profile that says no such word. The pool is the one of the
/// case above. No prefill has an open partner, so the shadow's own run
/// is shown none and sheds. Its default reads the constraints as they
/// are stated, not its run's lifted ones: it takes the default's
/// prefill.
#[test]
fn a_weighing_shadows_default_judges_the_pool_with_no_lift() {
    let prefills = [classed(over(0, 2), 1), classed(over(1, 1), 1)];
    let partners = [classed(over(0, 4), 1), classed(over(1, 0), 1)];
    let decodes = [partners[0], partners[1], classed(free(2, 0), 2)];
    let weighs = |weighs| {
        Some(Holds {
            holder: 0,
            reads: true,
            weighs,
        })
    };
    let engine = pair_engine_holding(false, [vetoes(), vetoes()], [weighs(false), weighs(true)]);
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let scratch = &mut Scratch::new();
    let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
    assert_eq!(pair_of(&live), (1, 1));
    let pick = live.pick().expect("a pair");
    let side = PairSide::Prefill;
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
    assert_eq!(twin.map(|shadow| order(&shadow)[0]), Some(1));
}

/// A weighing profile's refusal is its lifted run's own. The one
/// prefill is over its thresholds and at its admission limit: the
/// lifted run ends at the limit, the default leaves that refusal, and
/// the shed names the limit. Named with the veto standing, the stage
/// would be the veto. No request was steered, so the trace names no
/// lifted veto. The gateway refuses a lifted prefill at its limit the
/// same way
/// (`model_gateway/src/routers/common/placement.rs:2590-2595`, at main
/// `55c35100d`).
///
/// The prefills that run could take are counted under the lift too.
/// Where the one prefill is over its thresholds and its only partner
/// is not ready, it is a routable prefill with no decode to go to, and
/// the shed names the decode side's stage. Counted with the veto
/// standing it would be no routable prefill, and the shed would name
/// the veto.
#[test]
fn a_weighing_profiles_refusal_names_the_stage_its_lifted_run_ended_at() {
    let mut filters = vetoes();
    filters.push(Box::new(HasCapacity));
    let live = Some(Holds {
        holder: 0,
        reads: true,
        weighs: true,
    });
    let engine = pair_engine_holding(false, [filters, vetoes()], [live, None]);
    let full = decide_pair(
        &engine,
        &[classed(over(0, 1), 1)],
        &[classed(free(0, 0), 1)],
    );
    assert_eq!(shed_of(&full), (ShedReason::NoEndpoints, "has-capacity"));
    assert_eq!(full.trace().lifted, LiftedVeto::None);
    let mut down = classed(free(0, 0), 1);
    down.health.healthy = false;
    let alone = decide_pair(&holding(false, 0, true), &[classed(over(0, 1), 1)], &[down]);
    assert_eq!(shed_of(&alone), (ShedReason::NoEndpoints, "healthy"));
}

/// The pair's default is told the prefill profile's word, and judges a
/// weighing profile's refusal under the lift its run read. Both
/// prefills are over their thresholds. The busier one stands at its
/// admission limit with an open partner. The other has a slot, and its
/// one partner is over its thresholds beside a free decode, so no rule
/// opened it. A weighing profile's lifted run ends at the limit, and
/// the default leaves that refusal: the shed names the limit, and no
/// veto was lifted for a request. A profile whose scorer says no such
/// word ran with the veto standing, so the default pairs the pool by
/// load, past the prefill at its limit.
#[test]
fn the_pairs_default_is_told_the_prefill_profiles_word() {
    let limited = |weighs: bool| {
        let mut filters = vetoes();
        filters.push(Box::new(HasCapacity));
        let live = Some(Holds {
            holder: 0,
            reads: true,
            weighs,
        });
        pair_engine_holding(false, [filters, vetoes()], [live, None])
    };
    let mut slot = classed(over(1, 0), 2);
    slot.has_capacity = true;
    let prefills = [classed(over(0, 4), 1), slot];
    let decodes = [classed(free(0, 0), 1), classed(over(1, 0), 2)];
    let full = decide_pair(&limited(true), &prefills, &decodes);
    assert_eq!(shed_of(&full), (ShedReason::NoEndpoints, "has-capacity"));
    assert_eq!(full.trace().lifted, LiftedVeto::None);
    let paired = decide_pair(&limited(false), &prefills, &decodes);
    assert_eq!(pair_of(&paired), (1, 1));
    assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
}

/// The prefill's shadow is lifted by its own profile's word, as the
/// live run is by the live profile's: a shadow that weighs load against
/// affinity keeps the busy holder beside a live profile that passes it
/// over, and a shadow whose stage says no such word passes it over
/// beside a live profile that keeps it. The binding's word stands for
/// the shadow as for the live run: under a binding that sheds neither
/// is lifted.
///
/// Over a pool with every prefill over its thresholds each side's
/// shadow follows the pair's default as its live side does. Beside a
/// pair the weighing lift alone placed, a twin is lifted too and
/// agrees. A shadow that does not weigh takes the default's prefill
/// there: it is not said as shed. Beside the default's pair a twin
/// takes the default's prefill and decode. A shadow whose own lifted
/// run keeps a prefill is compared by that pick.
///
/// The decode's shadow runs behind the prefill's on one scratch and
/// names the live decode each time.
#[test]
fn a_prefills_shadow_is_lifted_by_its_own_profiles_word() {
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let decodes = [classed(free(0, 0), 1)];
    let said_under = |shed: bool, prefills: &[Candidate], live: bool, shadow: bool| {
        let sides = [live, shadow].map(|weighs| {
            Some(Holds {
                holder: 0,
                reads: true,
                weighs,
            })
        });
        let engine = pair_engine_holding(shed, [vetoes(), vetoes()], sides);
        let pools = PairPools {
            prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let scratch = &mut Scratch::new();
        let pair = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        let pick = pair.pick().expect("a pair");
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
        (pair_of(&pair).0, pick.trace.lifted, twins)
    };
    let said = |prefills: &[Candidate], live, shadow| said_under(false, prefills, live, shadow);
    // A busy holder beside a free prefill: no pick exists by the lift.
    let beside = [classed(over(0, 3), 1), classed(free(1, 0), 1)];
    let none = LiftedVeto::None;
    assert_eq!(said(&beside, true, true), (0, none, [Some(0), Some(0)]));
    assert_eq!(said(&beside, true, false), (0, none, [Some(1), Some(0)]));
    assert_eq!(said(&beside, false, true), (1, none, [Some(0), Some(0)]));
    assert_eq!(said(&beside, false, false), (1, none, [Some(1), Some(0)]));
    // Under a binding that sheds the live run passes the busy holder
    // over, and so does the shadow's.
    let shed = said_under(true, &beside, true, true);
    assert_eq!(shed, (1, none, [Some(1), Some(0)]));
    // Every prefill over, the holder the busier one. A live profile
    // that weighs keeps the holder by its lift, and one that does not
    // is shed and goes to the default's pair of the least load. Each
    // shadow goes its own profile's way over the same pool.
    let over_all = [classed(over(0, 3), 1), classed(over(1, 1), 1)];
    let lifted = LiftedVeto::Overload;
    assert_eq!(said(&over_all, true, true), (0, lifted, [Some(0), Some(0)]));
    assert_eq!(
        said(&over_all, true, false),
        (0, lifted, [Some(1), Some(0)])
    );
    assert_eq!(
        said(&over_all, false, true),
        (1, lifted, [Some(0), Some(0)])
    );
    assert_eq!(
        said(&over_all, false, false),
        (1, lifted, [Some(1), Some(0)])
    );
}

/// The engine's own affinity stage says the word itself, so its prefill
/// profile is lifted in every engine that holds it: in one built by
/// hand, with no validator in the way, and in one a document built,
/// behind the view's report age limit too, where the stage stands
/// behind the limit's wrapper. The busy prefill holds eight of the
/// request's sixteen blocks and stands under the stage's own gate, and
/// a free prefill holds none: the session keeps its holder.
///
/// A registered scorer answers for itself the same way. One that
/// prefers the holder, reads affinity and says no word is passed over
/// at the busy holder, and beside the affinity stage its profile is
/// lifted by that stage's word.
#[test]
fn the_affinity_stage_lifts_its_prefill_profile_in_every_engine() {
    static HASHES: [u64; 16] = [0; 16];
    let request = RequestFacts {
        block_hashes: &HASHES,
        prompt_tokens: 256,
        ..facts(1)
    };
    let held = |overlap_blocks| AffinityFacts {
        overlap_blocks,
        admitted_age_ms: u32::MAX,
        ..AffinityFacts::default()
    };
    let column = [held(8), held(0)];
    let prefills = [classed(over(0, 0), 1), classed(free(1, 0), 1)];
    let decodes = [classed(free(0, 0), 1)];
    let pair = |engine: &Engine| {
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            prefill_affinity: &column,
            ..PairPools::default()
        };
        let cx = Constraints::default();
        let scratch = &mut Scratch::new();
        let placed = engine.decide_pair(&request, &PoolFacts::default(), pools, &cx, &cx, scratch);
        assert_eq!(placed.trace().lifted, LiftedVeto::None);
        pair_of(&placed)
    };
    let params = PrefixAffinityParams::default();
    let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, None));
    let side = |name: &str, scorers| Profile {
        name: name.into(),
        filters: vetoes(),
        scorers,
        picker: Box::new(Best),
    };
    let weighted = WeightedScorer {
        scorer: Box::new(stage),
        weight: 1.0,
    };
    let by_hand = Engine::new(
        vec![side("prefill", vec![weighted]), side("decode", vec![])],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(1),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    assert_eq!(pair(&by_hand), (0, 0));
    let mut catalog = StageRegistry::builtin();
    let reads_alone = || -> Box<dyn Scorer> {
        Box::new(Holds {
            holder: 0,
            reads: true,
            weighs: false,
        })
    };
    catalog.scorer("holds", reads_alone).expect("a free name");
    let built = |score: &str, view: &str| {
        let stages = r#""filter":["healthy","not-overloaded"]"#;
        let text = format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{view},"profiles":{{"decode":{{{stages},"pick":"best"}},"prefill":{{{stages},"score":[{score}],"pick":"best"}}}},"bindings":[{{"match":{{"model":"*","phase":["prefill","decode"]}},"plan":"pd-pair","profiles":{{"prefill":"prefill","decode":"decode"}}}}]}}"#
        );
        let document: PolicySpec = serde_json::from_str(&text).expect("a well-formed document");
        AppliedSpec::validate_in(&document, &catalog, &[]).expect("the document validates")
    };
    for view in ["{}", r#"{"max_report_age_ms":500}"#] {
        let own = built(r#""prefix-affinity""#, view);
        assert_eq!(pair(own.engine()), (0, 0), "{view}");
        let registered = built(r#""holds""#, view);
        assert_eq!(pair(registered.engine()), (1, 0), "{view}");
        let both = built(r#""holds","prefix-affinity""#, view);
        assert_eq!(pair(both.engine()), (0, 0), "{view}");
    }
}
