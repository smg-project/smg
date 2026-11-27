//! The stage registry's cases of a picker added in code: its factory, and
//! the parameters a document writes under its name.

use super::*;

const NTH_PATH: &str = "profiles.default.pick.nth";
const NTH_FIELD: &str = "profiles.default.pick.nth.place";

/// Picks the survivor at `place` among those the filters left, in the
/// list's order.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Nth {
    place: u32,
}

impl Lifecycle for Nth {}

impl Picker for Nth {
    fn name(&self) -> &'static str {
        "nth"
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
        if let Some(&row) = kept.get(self.place as usize) {
            out.push(target_of(&candidates[row as usize]));
        }
    }
}

fn nth_catalog() -> StageRegistry {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker_with("nth", |build| {
            let rule: Nth = build.params()?;
            if rule.place > 3 {
                return Err(crate::stages::StageError::bad(
                    "place",
                    rule.place,
                    "at most 3",
                ));
            }
            Ok(Box::new(rule))
        })
        .expect("a free name");
    catalog
}

fn pick_document(pick: &str) -> PolicySpec {
    let version = crate::document::API_VERSION;
    serde_yaml::from_str(&format!(
        "apiVersion: {version}\nprofiles:\n  default:\n    filter: [healthy]\n    pick: {pick}\n"
    ))
    .expect("a well-formed document")
}

#[test]
fn a_picker_picks_by_its_typed_parameter() {
    let catalog = nth_catalog();
    for place in [0_u64, 2, 3] {
        let engine = pick_document(&format!("{{ nth: {{ place: {place} }} }}"))
            .validate_in(&catalog, &[])
            .expect("the picker reads its place");
        assert_eq!(engine.profiles()[0].picker.name(), "nth");
        assert_eq!(pick(&engine), Some(WorkerId(place + 1)), "place {place}");
    }
}

#[test]
fn a_bare_picker_name_builds_from_an_empty_map() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker_with("worst", |build| {
            assert!(build.written().is_empty());
            let typed: serde_json::Map<String, serde_json::Value> = build.params()?;
            assert!(typed.is_empty());
            Ok(Box::new(Worst))
        })
        .expect("a free name");
    for pick in ["worst", "{ worst: {} }"] {
        let engine = pick_document(pick)
            .validate_in(&catalog, &[])
            .expect("the picker needs no parameters");
        assert_eq!(engine.profiles()[0].picker.name(), "worst");
    }
}

/// A picker's factory reads its parameters in the one order a filter's
/// does, whatever order the document wrote their keys in, at every
/// depth: two files that are one document by their digest hand the
/// factory the same map, whole.
#[test]
fn a_pickers_factory_reads_its_whole_map_in_key_order() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let heard = Arc::clone(&seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker_with("worst", move |build| {
            let written = serde_json::to_string(build.written()).expect("serializes");
            heard.lock().expect("no panic under the lock").push(written);
            Ok(Box::new(Worst))
        })
        .expect("a free name");
    let one = pick_document("{ worst: { weight: 2, place: 500, zones: [{ unit: 17, flat: 3 }] } }");
    let other =
        pick_document("{ worst: { zones: [{ flat: 3, unit: 17 }], place: 500, weight: 2 } }");
    assert_eq!(
        one.digest().expect("digests"),
        other.digest().expect("digests")
    );
    one.validate_in(&catalog, &[]).expect("validates");
    other.validate_in(&catalog, &[]).expect("validates");
    let seen = seen.lock().expect("no panic under the lock");
    assert_eq!(seen[0], seen[1], "two orders of one map");
    assert_eq!(
        seen[0],
        r#"{"place":500,"weight":2,"zones":[{"flat":3,"unit":17}]}"#
    );
}

#[test]
fn whatever_stands_under_a_picker_that_takes_no_parameter_is_refused_by_its_name() {
    // `worst` is registered with no parameter, and the stock's pickers
    // named here take none.
    let catalog = catalog();
    let refused = |pick: &str| {
        pick_document(pick)
            .validate_in(&catalog, &[])
            .expect_err("refused")
    };
    for (pick, path, value) in [
        (
            "{ worst: { place: 2 } }",
            "profiles.default.pick.worst",
            r#"{"place":2}"#,
        ),
        ("{ random: {} }", "profiles.default.pick.random", "{}"),
        (
            "{ round-robin: null }",
            "profiles.default.pick.round-robin",
            "null",
        ),
    ] {
        let error = refused(pick);
        assert_eq!(
            error,
            SpecError::BadField {
                path: path.into(),
                value: value.into(),
                reason: "left out: this picker takes no parameter, write its bare name",
            }
        );
        assert_eq!(error.field_path(), path);
    }
    assert_eq!(
        refused("{ worst: { place: 2 } }").to_string(),
        r#"profiles.default.pick.worst = {"place":2} is not left out: this picker takes no parameter, write its bare name"#
    );
    // A name no catalog holds is unknown, whatever stands under it.
    assert_eq!(
        refused("{ absent: { place: 2 } }"),
        SpecError::UnknownPicker {
            profile: "default".into(),
            name: "absent".into(),
        }
    );
}

#[test]
fn a_pickers_factory_refuses_a_value_at_its_field() {
    let catalog = nth_catalog();
    let refused = |pick: &str| {
        pick_document(pick)
            .validate_in(&catalog, &[])
            .expect_err("the factory refuses the parameters")
    };
    let error = refused("{ nth: { place: 9 } }");
    assert_eq!(
        error,
        SpecError::BadField {
            path: NTH_FIELD.into(),
            value: "9".into(),
            reason: "at most 3",
        }
    );
    assert_eq!(
        error.to_string(),
        format!("{NTH_FIELD} = 9 is not at most 3")
    );
    assert_eq!(error.field_path(), NTH_FIELD);
    // What the factory's type does not read is refused at the entry in
    // the type's own words. A bare name supplies an empty map.
    for (pick, message) in [
        ("{ nth: { place: 2, spare: 17 } }", "unknown field `spare`"),
        ("{ nth: { place: 'far' } }", "invalid type"),
        ("nth", "missing field `place`"),
    ] {
        let SpecError::BadStage { path, reason } = refused(pick) else {
            panic!("{pick}");
        };
        assert_eq!(path, NTH_PATH);
        assert!(reason.contains(message), "{reason}");
    }
    // What stands under the name is a map.
    for (pick, value) in [("{ nth: [17] }", "[17]"), ("{ nth: null }", "null")] {
        assert_eq!(
            refused(pick),
            SpecError::BadField {
                path: NTH_PATH.into(),
                value: value.into(),
                reason: "a map of the picker's parameters",
            }
        );
    }
}

#[test]
fn a_picker_that_answers_another_name_is_refused_at_its_entry() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker_with("nth", |_| Ok(Box::new(Worst)))
        .expect("a free name");
    assert_eq!(
        pick_document("nth")
            .validate_in(&catalog, &[])
            .expect_err("the name disagrees"),
        SpecError::BadStage {
            path: NTH_PATH.into(),
            reason: r#"picker registered as "nth" answers the name "worst""#.into(),
        }
    );
}

#[test]
fn a_pickers_factory_runs_once_a_profile_and_never_under_a_held_name() {
    let mut catalog = nth_catalog();
    let called = Arc::new(AtomicUsize::new(0));
    let counted = |called: &Arc<AtomicUsize>| {
        let calls = Arc::clone(called);
        move |_: &mut crate::stages::StageBuild<'_>| -> Result<Box<dyn Picker>, crate::stages::StageError> {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(Box::new(Worst))
            }
    };
    for (name, held) in [
        ("healthy", StageKind::Filter),
        ("least-requests", StageKind::Scorer),
        ("best", StageKind::Picker),
        ("nth", StageKind::Picker),
    ] {
        assert_eq!(
            catalog.picker_with(name, counted(&called)).map(|_| ()),
            Err(StageRegistryError::Held { name, held })
        );
    }
    assert_eq!(catalog.pickers().count(), PICKERS.len() + 1);
    catalog
        .picker_with("worst", counted(&called))
        .expect("a free name");
    assert_eq!(called.load(Ordering::Relaxed), 0);
    let named = pick_document("worst");
    for expected in [1, 2] {
        named.validate_in(&catalog, &[]).expect("validates");
        assert_eq!(called.load(Ordering::Relaxed), expected);
    }
}

#[test]
fn an_added_pick_entry_is_written_as_it_was_read() {
    let read = pick_document("{ nth: { place: 2 } }");
    let again = serde_json::to_value(&read).expect("serializes");
    assert_eq!(
        again["profiles"]["default"]["pick"].to_string(),
        r#"{"nth":{"place":2}}"#
    );
    assert_eq!(
        serde_json::from_value::<PolicySpec>(again).expect("parses again"),
        read
    );
    // The typed built-ins and a bare name beside it, each through YAML.
    for pick in [
        "{ nth: { place: 2 } }",
        "best",
        "{ best: { sample: 2 } }",
        "random",
    ] {
        let read = pick_document(pick);
        let yaml = serde_yaml::to_string(&read).expect("writes YAML");
        assert_eq!(
            serde_yaml::from_str::<PolicySpec>(&yaml).expect("reads YAML"),
            read,
            "{pick}"
        );
    }
}

/// Only a document built in code holds a typed built-in's name under an
/// added entry: the reader reads that key into its parameter struct.
/// The refusal says what the entry is, as a scorer's does.
#[test]
fn a_typed_built_ins_name_under_an_added_entry_is_refused_by_what_it_is() {
    for (name, written) in [
        ("best", serde_json::json!({ "sample": 2 })),
        (
            "best-with-temperature",
            serde_json::json!({ "temperature": 0.5 }),
        ),
    ] {
        let mut document = pick_document("random");
        let profile = document.profiles.get_mut("default").expect("the profile");
        profile.pick = PickSpec::Added {
            name: name.into(),
            written,
        };
        assert_eq!(
            document
                .validate_in(&StageRegistry::builtin(), &[])
                .expect_err("refused"),
            SpecError::BadField {
                path: format!("profiles.default.pick.{name}"),
                value: name.into(),
                reason: "a picker added in code: a built-in's parameters are typed",
            },
            "{name}"
        );
    }
}
