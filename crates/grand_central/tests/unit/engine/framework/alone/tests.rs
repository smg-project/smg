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
