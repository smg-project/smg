use super::*;
use crate::document::spec::{PolicySpec, ProfileSpec, ShadowSpec};

const MODEL: &str = "model-a";

fn preset(name: &str) -> PolicySpec {
    PolicySpec::preset(name).expect("a built-in preset")
}

fn engine_of(document: &PolicySpec) -> Engine {
    document.validate().expect("the document validates")
}

/// The one profile of `name`'s preset.
fn profile_of(name: &str) -> ProfileSpec {
    let profile = preset(name).profiles.into_values().next();
    profile.expect("the preset's profile")
}

/// `live`'s preset with the profile of `twin`'s named as its binding's
/// shadow, under a name that sorts behind every preset's own.
fn shadowed_by(live: &str, twin: &str) -> PolicySpec {
    let mut document = preset(live);
    document.profiles.insert("twin".into(), profile_of(twin));
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    document
}

/// `live`'s preset under the sessions rule of the `manual` preset.
fn ruled(live: &str) -> PolicySpec {
    let mut document = preset(live);
    let manual = preset("manual");
    document.view.headers.clone_from(&manual.view.headers);
    let rule = manual.bindings.into_iter().next();
    document.bindings[0].sessions = rule.and_then(|binding| binding.sessions);
    assert!(document.bindings[0].sessions.is_some(), "the preset's rule");
    document
}

/// A binding that names a shadow has each request decided whole, with
/// its shadow profile beside it: one pick would run none. So its place
/// is not one pick's, and both answers name the shadow. The same
/// profile under a binding that names none is picked alone.
#[test]
fn a_place_whose_binding_names_a_shadow_is_not_one_picks() {
    let mut document = shadowed_by("round_robin", "random");
    let mut plain = document.bindings[0].clone();
    plain.matcher.model = "alpha-*".to_owned();
    plain.shadow = None;
    let pattern = document.bindings[0].matcher.model.clone();
    document.bindings.insert(0, plain);
    let engine = engine_of(&document);

    let (profile, in_the_way) = ("round_robin", InTheWay::Shadow);
    let place = engine.place_for(MODEL, Phase::Single);
    assert_eq!(
        place,
        Place::NotAlone {
            profile,
            in_the_way
        }
    );
    let named = NotAlone {
        model: pattern.as_str(),
        phase: Phase::Single,
        profile,
        in_the_way,
    };
    assert_eq!(engine.not_alone(), Some(named));
    assert_eq!(in_the_way.to_string(), "the binding's shadow");

    let unshadowed = engine.place_for("alpha-1", Phase::Single);
    assert!(matches!(unshadowed, Place::Alone(_)), "{unshadowed:?}");
}

/// A sessions rule pins a keyed request ahead of the profile, which one
/// pick would step past. So a place under a rule is not one pick's,
/// whichever picker its profile holds, and both answers name the rule.
#[test]
fn a_place_whose_binding_states_a_sessions_rule_is_not_one_picks() {
    for profile in ["round_robin", "random", "passthrough"] {
        let plain = engine_of(&preset(profile));
        let before = plain.place_for(MODEL, Phase::Single);
        assert!(matches!(before, Place::Alone(_)), "{profile}: {before:?}");

        let engine = engine_of(&ruled(profile));
        let in_the_way = InTheWay::Sessions;
        let place = engine.place_for(MODEL, Phase::Single);
        assert_eq!(
            place,
            Place::NotAlone {
                profile,
                in_the_way
            },
            "{profile}"
        );
        let named = engine
            .not_alone()
            .map(|place| (place.phase, place.in_the_way));
        assert_eq!(named, Some((Phase::Single, in_the_way)), "{profile}");
    }
    let said = InTheWay::Sessions.to_string();
    assert_eq!(said, "the binding's sessions rule");
}

/// A stage of the profile is named ahead of its binding: a place that
/// names a stage says the same word under a sessions rule and under a
/// shadow.
#[test]
fn a_stage_is_named_ahead_of_the_binding() {
    let plain = engine_of(&preset("power_of_two"));
    let before = plain.place_for(MODEL, Phase::Single);
    let in_the_way = InTheWay::Scorer("least-load");
    let profile = "power_of_two";
    assert_eq!(
        before,
        Place::NotAlone {
            profile,
            in_the_way
        }
    );
    for document in [ruled(profile), shadowed_by(profile, "round_robin")] {
        let engine = engine_of(&document);
        assert_eq!(engine.place_for(MODEL, Phase::Single), before);
    }
}

/// The `power_of_two` preset is a place whose whole decision is asked
/// ahead of its request ([`Engine::place_ahead_for`]), though it is
/// not one pick's: its picker reads of a request the seed alone, and
/// its one scorer scores the workers drawn by their load reports and
/// reads no request, which the crate tells by the two stages' names.
/// Each thing that asks more of a decision ends the place alone: a
/// sessions rule, a shadow, another picker, another built-in scorer, and
/// one filter more than a pass holds.
///
/// A host's scorer in the place of least-load ends the place too, though
/// it says of itself what least-load says: a stage of a host stands in
/// no list of the crate and is read as one that reads the request.
#[cfg(feature = "in-process")]
#[test]
fn a_place_decided_ahead_of_its_request_asks_no_more_than_the_facts_stated() {
    use crate::{
        engine::{
            framework::{Scorer, TieBand},
            view::{Candidate, RequestFacts},
        },
        stages::{Lifecycle, StageRegistry},
    };

    /// A host's scorer that scores every worker alike, each row alone,
    /// and needs the load reports alone, as least-load does.
    #[derive(Clone, Copy, Debug)]
    struct Alike;

    impl Lifecycle for Alike {}

    impl Scorer for Alike {
        fn name(&self) -> &'static str {
            "alike"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }

        fn row_wise(&self) -> bool {
            true
        }

        fn needs(&self) -> Needs {
            Needs {
                loads: true,
                ..Needs::default()
            }
        }
    }

    let ahead = |engine: Engine| engine.place_ahead_for(MODEL, Phase::Single);
    let place = Some(AloneProfile(0));
    assert_eq!(ahead(engine_of(&preset("power_of_two"))), place);
    // A limit on the age of a load report wraps least-load, and the
    // wrapper keeps the name of the scorer it wraps.
    let mut aged = preset("power_of_two");
    aged.view.max_report_age_ms = Some(900);
    assert_eq!(ahead(engine_of(&aged)), place, "a limit on a report's age");

    let random = changed(|profile| profile.pick = "random".into());
    let requests = changed(|profile| profile.score = vec!["least-requests".into()]);
    let capacity = changed(|profile| profile.filter.push("has-capacity".into()));
    let more = [
        ("a sessions rule", ruled("power_of_two")),
        ("a shadow", shadowed_by("power_of_two", "random")),
        ("another picker", random),
        ("another scorer", requests),
        ("one more filter", capacity),
    ];
    for (case, document) in more {
        assert_eq!(ahead(engine_of(&document)), None, "{case}");
    }

    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("alike", || Box::new(Alike) as Box<dyn Scorer>);
    added.expect("a free name");
    let document = changed(|profile| profile.score = vec!["alike".into()]);
    let engine = document.validate_in(&stages, &[]);
    let hosted = ahead(engine.expect("the stages hold the scorer"));
    assert_eq!(hosted, None, "a host's own scorer");
}

/// The preset `power_of_two` with its one profile changed by `change`.
#[cfg(feature = "in-process")]
fn changed(change: impl FnOnce(&mut ProfileSpec)) -> PolicySpec {
    let mut document = preset("power_of_two");
    let profile = document.profiles.get_mut("power_of_two");
    change(profile.expect("the preset's profile"));
    document
}

/// The two lists the fill of a list's cell reads name the crate's own
/// stages by what they read of a request ([`PICKS_BY_ITS_SEED`],
/// [`SCORES_BY_ITS_ROWS`]). Every picker and every scorer the crate
/// ships stands in one of two readings: in its list, or among the stages
/// this test names as ones that read the request, so a new stage of the
/// crate is put in one of the two by hand. And each listed stage does
/// what its list says: over rows that hold load reports, at each seed, a
/// request that states the facts below draws the scores and the picks of
/// a request that states its model, its phase, its seed and the instant
/// alone.
///
/// [`PICKS_BY_ITS_SEED`]: super::super::PICKS_BY_ITS_SEED
/// [`SCORES_BY_ITS_ROWS`]: super::super::SCORES_BY_ITS_ROWS
#[cfg(feature = "in-process")]
#[test]
fn the_stages_decided_ahead_are_listed_by_what_they_read_of_a_request() {
    use super::super::{Picker, Scorer, PICKS_BY_ITS_SEED, SCORES_BY_ITS_ROWS};
    use crate::{
        engine::view::{Candidate, Identity, RequestFacts, WorkerId},
        stages::{
            scorers::tests::{candidate, picked, report},
            StageRegistry,
        },
    };

    // The crate's stages that are read as ones that read the request.
    const OTHER_PICKERS: [&str; 8] = [
        "best",
        "round-robin",
        "random",
        "count-first",
        "power-of-two-count-first",
        "passthrough",
        "manual",
        "best-with-temperature",
    ];
    const OTHER_SCORERS: [&str; 10] = [
        "least-requests",
        "fewest-sessions",
        "session-affinity",
        "prefix-affinity",
        "kv-pressure",
        "warmup-slice",
        "bucket",
        "consistent-hash",
        "resident-work",
        "least-cost",
    ];
    let shipped = StageRegistry::builtin();
    let one_reading = |kind: &str, name: &str, listed: &[&str], other: &[&str]| {
        let readings = usize::from(listed.contains(&name)) + usize::from(other.contains(&name));
        assert_eq!(readings, 1, "the {kind} {name} stands in one reading");
    };
    for name in shipped.pickers() {
        one_reading("picker", name, &PICKS_BY_ITS_SEED, &OTHER_PICKERS);
    }
    for name in shipped.scorers() {
        one_reading("scorer", name, &SCORES_BY_ITS_ROWS, &OTHER_SCORERS);
    }
    // No list names a stage the crate does not ship.
    let pickers = PICKS_BY_ITS_SEED.len() + OTHER_PICKERS.len();
    assert_eq!(shipped.pickers().count(), pickers);
    let scorers = SCORES_BY_ITS_ROWS.len() + OTHER_SCORERS.len();
    assert_eq!(shipped.scorers().count(), scorers);

    // Rows that hold load reports of unlike sizes: a score has loads to
    // read, and a pick has scores to tell apart.
    let pool: Vec<Candidate> = (0..8_u32)
        .map(|at| {
            let mut row = candidate(at, at % 3);
            row.load = Some(report(at % 4, 512 * at));
            row.load_sampled_at_micros = 1_000;
            row
        })
        .collect();
    let (tried, hashes) = ([WorkerId(1_001)], [7_u64, 11]);
    let identity = |hash: u64, plain: &'static str| Identity {
        hash,
        plain: Some(plain),
    };
    for picker in PICKS_BY_ITS_SEED {
        for scorer in SCORES_BY_ITS_ROWS {
            let engine = engine_of(&changed(|profile| {
                profile.pick = picker.into();
                profile.score = vec![scorer.into()];
            }));
            let profile = engine.profiles.first().expect("the document's profile");
            let weighted = profile.scorers.first().expect("the profile's scorer");
            let (picks, scores): (&dyn Picker, &dyn Scorer) = (&*profile.picker, &*weighted.scorer);
            assert_eq!((picks.name(), scores.name()), (picker, scorer));
            let drawn = |facts: &RequestFacts<'_>| {
                let scored = |rows: &[u32]| {
                    let mut out = vec![0.0_f32; rows.len()];
                    scores.score(facts, &pool, rows, &mut out);
                    out
                };
                let every: Vec<u32> = (0..8).collect();
                (scored(&every), picked(picks, facts, &pool, &scored))
            };
            for seed in 0..256_u64 {
                let four = RequestFacts {
                    model: MODEL,
                    phase: Phase::Single,
                    seed,
                    now_micros: 2_000,
                    ..RequestFacts::default()
                };
                let more = RequestFacts {
                    namespace_seed: 3,
                    tenant: identity(5, "a tenant"),
                    priority: 9,
                    streaming: true,
                    max_output_tokens: 64,
                    prompt_tokens: 4_096,
                    block_hashes: &hashes,
                    session_key: identity(13, "a session"),
                    attempt: 1,
                    tried: &tried,
                    budget_micros: 100,
                    block_size: 16,
                    prompt_chars: 20_000,
                    prompt_bytes: 21_000,
                    session_counted: true,
                    client_key: 17,
                    ring_positions: &hashes,
                    decode_ring_positions: Some(&hashes),
                    session_key_from_rid: true,
                    ..four
                };
                assert_eq!(
                    drawn(&four),
                    drawn(&more),
                    "{picker}, {scorer}, seed {seed}"
                );
            }
        }
    }
}

/// A pair's binding names a shadow a side. The place of the side that
/// names one is not one pick's, and the other side's place says what
/// it says with no shadow named.
#[test]
fn a_pairs_place_is_not_one_picks_on_the_side_that_names_a_shadow() {
    let pair = PolicySpec::pair_presets("round_robin", "round_robin");
    let pair = pair.expect("the pair presets");
    let mut document = pair.clone();
    let twin = document.profiles["decode"].clone();
    document.profiles.insert("twin".into(), twin);
    let sides = [("decode".to_owned(), "twin".to_owned())];
    document.bindings[0].shadow = Some(ShadowSpec::Sides(sides.into()));
    let (plain, engine) = (engine_of(&pair), engine_of(&document));

    let before = plain.place_for(MODEL, Phase::Decode);
    assert!(matches!(before, Place::Alone(_)), "{before:?}");
    let (profile, in_the_way) = ("decode", InTheWay::Shadow);
    let decode = engine.place_for(MODEL, Phase::Decode);
    assert_eq!(
        decode,
        Place::NotAlone {
            profile,
            in_the_way
        }
    );
    let prefill = engine.place_for(MODEL, Phase::Prefill);
    assert_eq!(prefill, plain.place_for(MODEL, Phase::Prefill));
}

/// A place decided whole runs the shadow its binding names in the same
/// pass, over the same records. So a host has such a document decided
/// whole only where it feeds what the shadow profile's stages declare
/// too. Round robin asks nothing of a host. Under a shadow that reads
/// the workers' load reports it asks for them, and under a shadow that
/// reads a worker's record and no more it still asks nothing.
#[test]
fn a_shadows_needs_are_held_to_what_a_host_feeds() {
    let nothing = Needs::default();
    assert!(engine_of(&preset("round_robin")).single_and_fed(nothing));

    let engine = engine_of(&shadowed_by("round_robin", "least_load"));
    let declared = engine.needs_for(MODEL, Phase::Single);
    assert!(declared.loads, "the shadow's scorer reads reports");
    assert!(!engine.single_and_fed(nothing));
    assert!(engine.single_and_fed(declared));

    let engine = engine_of(&shadowed_by("round_robin", "random"));
    assert!(engine.single_and_fed(nothing));
}
