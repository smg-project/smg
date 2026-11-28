//! The stage registry's cases of a filter added in code: its factory, and
//! the parameters a document writes under its name.

use super::*;

const FREE_PATH: &str = "profiles.default.filter[1].min-free";
const FREE_FIELD: &str = "profiles.default.filter[1].min-free.slots";

/// Keeps the workers with `slots` of their eight request slots free.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MinFree {
    slots: u32,
}

impl Filter for MinFree {
    fn name(&self) -> &'static str {
        "min-free"
    }

    fn keep(&self, _: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        candidate.in_flight.saturating_add(self.slots) <= 8
    }
}

fn free_catalog() -> StageRegistry {
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("min-free", |build| {
            let rule: MinFree = build.params()?;
            if rule.slots == 0 {
                return Err(crate::stages::StageError::bad(
                    "slots",
                    rule.slots,
                    "above zero",
                ));
            }
            Ok(Box::new(rule))
        })
        .expect("a free name");
    catalog
}

fn filter_document(filter: &str) -> PolicySpec {
    let version = crate::document::API_VERSION;
    serde_yaml::from_str(&format!(
        "apiVersion: {version}\nprofiles:\n  default:\n    filter: {filter}\n    pick: random\n"
    ))
    .expect("a well-formed document")
}

#[test]
fn a_filter_keeps_the_rows_its_typed_parameter_says() {
    let catalog = free_catalog();
    let busy = |in_flight| Candidate {
        in_flight,
        ..Candidate::default()
    };
    for (slots, kept, cut) in [(2, 6, 7), (5, 3, 4)] {
        let filter = format!("[healthy, {{ min-free: {{ slots: {slots} }} }}]");
        let engine = filter_document(&filter)
            .validate_in(&catalog, &[])
            .expect("the filter reads its slots");
        let filters = &engine.profiles()[0].filters;
        assert!(filters
            .iter()
            .map(|filter| filter.name())
            .eq(["healthy", "min-free"]));
        let keeps = |row: Candidate| {
            filters[1].keep(&RequestFacts::default(), &row, &Constraints::default())
        };
        assert!(keeps(busy(kept)), "{slots} free at {kept} in flight");
        assert!(!keeps(busy(cut)), "{slots} free at {cut} in flight");
    }
}

#[test]
fn a_bare_filter_name_builds_from_an_empty_map() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("odd-ids", |build| {
            assert!(build.written().is_empty());
            let typed: serde_json::Map<String, serde_json::Value> = build.params()?;
            assert!(typed.is_empty());
            Ok(Box::new(OddIds))
        })
        .expect("a free name");
    for filter in ["[odd-ids]", "[{ odd-ids: {} }]"] {
        let engine = filter_document(filter)
            .validate_in(&catalog, &[])
            .expect("the filter needs no parameters");
        assert_eq!(engine.profiles()[0].filters[0].name(), "odd-ids");
    }
}

/// A filter's factory reads its parameters in the one order a scorer's
/// does, whatever order the document wrote their keys in, at every
/// depth: two files that are one document by their digest hand the
/// factory the same map. A filter has no weight, so the map is handed
/// whole.
#[test]
fn a_filters_factory_reads_its_whole_map_in_key_order() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let heard = Arc::clone(&seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("odd-ids", move |build| {
            let written = serde_json::to_string(build.written()).expect("serializes");
            heard.lock().expect("no panic under the lock").push(written);
            Ok(Box::new(OddIds))
        })
        .expect("a free name");
    let one =
        filter_document("[{ odd-ids: { weight: 2, slots: 500, zones: [{ unit: 17, flat: 3 }] } }]");
    let other =
        filter_document("[{ odd-ids: { zones: [{ flat: 3, unit: 17 }], slots: 500, weight: 2 } }]");
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
        r#"{"slots":500,"weight":2,"zones":[{"flat":3,"unit":17}]}"#
    );
}

#[test]
fn whatever_stands_under_a_filter_that_takes_no_parameter_is_refused_by_its_name() {
    // `odd-ids` is registered with no parameter, and the stock's
    // filters take none.
    let catalog = catalog();
    let refused = |filter: &str| {
        filter_document(filter)
            .validate_in(&catalog, &[])
            .expect_err("refused")
    };
    for (filter, path, value) in [
        (
            "[{ odd-ids: { slots: 2 } }]",
            "profiles.default.filter[0].odd-ids",
            r#"{"slots":2}"#,
        ),
        (
            "[healthy, { healthy: {} }]",
            "profiles.default.filter[1].healthy",
            "{}",
        ),
        (
            "[healthy, odd-ids, { not-tried: null }]",
            "profiles.default.filter[2].not-tried",
            "null",
        ),
    ] {
        let error = refused(filter);
        assert_eq!(
            error,
            SpecError::BadField {
                path: path.into(),
                value: value.into(),
                reason: "left out: this filter takes no parameter, write its bare name",
            }
        );
        assert_eq!(error.field_path(), path);
    }
    assert_eq!(
        refused("[{ odd-ids: { slots: 2 } }]").to_string(),
        r#"profiles.default.filter[0].odd-ids = {"slots":2} is not left out: this filter takes no parameter, write its bare name"#
    );
    // A name no catalog holds is unknown, whatever stands under it.
    assert_eq!(
        refused("[{ absent: { slots: 2 } }]"),
        SpecError::UnknownFilter {
            profile: "default".into(),
            name: "absent".into(),
        }
    );
}

/// A built-in's parameters are its struct's. An entry that carries
/// `has-room` as a filter added in code, which only a document built in
/// code can, is refused at the entry by what it is. The catalog holds
/// the name for its bare form, and the refusal names no bare form.
#[test]
fn a_built_in_rules_name_under_an_added_entry_is_refused_by_what_it_is() {
    let mut built_in_code = filter_document("[healthy]");
    for profile in built_in_code.profiles.values_mut() {
        profile.filter.push(spec::FilterSpec::Added {
            name: "has-room".into(),
            written: serde_json::json!({ "used": { "load.running": 1 }, "capacity": 10 }),
        });
    }
    assert_eq!(
        built_in_code.validate_in(&catalog(), &[]).map(|_| ()),
        Err(SpecError::BadField {
            path: "profiles.default.filter[1]".into(),
            value: "has-room".into(),
            reason: "a filter added in code: a built-in's parameters are typed",
        })
    );
}

#[test]
fn a_filters_factory_refuses_a_value_at_its_field() {
    let catalog = free_catalog();
    let refused = |filter: &str| {
        filter_document(filter)
            .validate_in(&catalog, &[])
            .expect_err("the factory refuses the parameters")
    };
    let error = refused("[healthy, { min-free: { slots: 0 } }]");
    assert_eq!(
        error,
        SpecError::BadField {
            path: FREE_FIELD.into(),
            value: "0".into(),
            reason: "above zero",
        }
    );
    assert_eq!(
        error.to_string(),
        format!("{FREE_FIELD} = 0 is not above zero")
    );
    assert_eq!(error.field_path(), FREE_FIELD);
    // What the factory's type does not read is refused at the entry in
    // the type's own words. A bare name supplies an empty map.
    for (filter, message) in [
        (
            "[healthy, { min-free: { slots: 2, spare: 17 } }]",
            "unknown field `spare`",
        ),
        ("[healthy, { min-free: { slots: 'many' } }]", "invalid type"),
        ("[healthy, min-free]", "missing field `slots`"),
    ] {
        let SpecError::BadStage { path, reason } = refused(filter) else {
            panic!("{filter}");
        };
        assert_eq!(path, FREE_PATH);
        assert!(reason.contains(message), "{reason}");
    }
    // What stands under the name is a map.
    for (filter, value) in [
        ("[healthy, { min-free: [17] }]", "[17]"),
        ("[healthy, { min-free: null }]", "null"),
    ] {
        assert_eq!(
            refused(filter),
            SpecError::BadField {
                path: FREE_PATH.into(),
                value: value.into(),
                reason: "a map of the filter's parameters",
            }
        );
    }
}

#[test]
fn a_filter_that_answers_another_name_is_refused_at_its_entry() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("min-free", |_| Ok(Box::new(OddIds)))
        .expect("a free name");
    assert_eq!(
        filter_document("[healthy, min-free]")
            .validate_in(&catalog, &[])
            .expect_err("the name disagrees"),
        SpecError::BadStage {
            path: FREE_PATH.into(),
            reason: r#"filter registered as "min-free" answers the name "odd-ids""#.into(),
        }
    );
}

#[test]
fn a_filters_factory_runs_once_an_entry_and_never_under_a_held_name() {
    let mut catalog = free_catalog();
    let called = Arc::new(AtomicUsize::new(0));
    let counted = |called: &Arc<AtomicUsize>| {
        let calls = Arc::clone(called);
        move |_: &mut crate::stages::StageBuild<'_>| -> Result<Box<dyn Filter>, crate::stages::StageError> {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(Box::new(OddIds))
            }
    };
    for (name, held) in [
        ("healthy", StageKind::Filter),
        ("least-requests", StageKind::Scorer),
        ("best", StageKind::Picker),
        ("min-free", StageKind::Filter),
    ] {
        assert_eq!(
            catalog.filter_with(name, counted(&called)).map(|_| ()),
            Err(StageRegistryError::Held { name, held })
        );
    }
    assert_eq!(catalog.filters().count(), FILTERS.len() + 1);
    catalog
        .filter_with("odd-ids", counted(&called))
        .expect("a free name");
    assert_eq!(called.load(Ordering::Relaxed), 0);
    let named = filter_document("[odd-ids, healthy, odd-ids]");
    for expected in [2, 4] {
        named.validate_in(&catalog, &[]).expect("validates");
        assert_eq!(called.load(Ordering::Relaxed), expected);
    }
}
