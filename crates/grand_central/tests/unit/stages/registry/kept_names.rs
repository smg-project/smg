//! The stage registry's cases of the names the engine keeps for its own
//! refusals: the list itself and the public test of a name against it,
//! what the registry takes under such a name, what a document's check
//! answers for one and for a name near one, and the check's refusal of
//! a plain factory's build that answers another name than its own.

use std::sync::atomic::AtomicBool;

use super::*;
use crate::engine::refuses_under;

/// A stage of each kind that answers the name it is made with: a host's
/// stage under whatever name a case tries.
#[derive(Debug)]
struct Under(&'static str);

impl Lifecycle for Under {}

impl Filter for Under {
    fn name(&self) -> &'static str {
        self.0
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }
}

impl Scorer for Under {
    fn name(&self) -> &'static str {
        self.0
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        id_shares(candidates, kept, out);
        TieBand::NORMALIZED
    }
}

impl Picker for Under {
    fn name(&self) -> &'static str {
        self.0
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
        if let Some(&i) = kept.first() {
            out.push(target_of(&candidates[i as usize]));
        }
    }
}

/// One of the six ways to register, by its place in the order `filter`,
/// `filter_with`, `scorer`, `scorer_with`, `picker`, `picker_with`: a
/// stage that answers `name` is registered under it. Its factory counts
/// its runs in `built`.
fn by_way(
    way: usize,
    catalog: &mut StageRegistry,
    name: &'static str,
    built: &Arc<AtomicUsize>,
) -> Result<(), StageRegistryError> {
    let built = Arc::clone(built);
    let count = move || built.fetch_add(1, Ordering::Relaxed);
    match way {
        0 => catalog.filter(name, move || {
            count();
            Box::new(Under(name))
        }),
        1 => catalog.filter_with(name, move |_| {
            count();
            Ok(Box::new(Under(name)))
        }),
        2 => catalog.scorer(name, move || {
            count();
            Box::new(Under(name))
        }),
        3 => catalog.scorer_with(name, move |_| {
            count();
            Ok(Box::new(Under(name)))
        }),
        4 => catalog.picker(name, move || {
            count();
            Box::new(Under(name))
        }),
        _ => catalog.picker_with(name, move |_| {
            count();
            Ok(Box::new(Under(name)))
        }),
    }
    .map(|_| ())
}

/// The kind of stage the way at `way` registers ([`by_way`]), a document
/// of one profile that names `name` as a stage of that kind, bare or with
/// a map written under it, and the path a refusal of that stage names.
fn naming(way: usize, name: &str, written: bool) -> (StageKind, PolicySpec, String) {
    let entry = if written {
        format!(r#"{{"{name}":{{"weight":2}}}}"#)
    } else {
        format!(r#""{name}""#)
    };
    match way / 2 {
        0 => (
            StageKind::Filter,
            spec(&format!(
                r#"{{"filter":[{entry}],"score":["least-requests"],"pick":"best"}}"#
            )),
            format!("profiles.p.filter[0].{name}"),
        ),
        1 => (
            StageKind::Scorer,
            spec(&format!(r#"{{"score":[{entry}],"pick":"best"}}"#)),
            format!("profiles.p.score[0].{name}"),
        ),
        _ => (
            StageKind::Picker,
            spec(&format!(r#"{{"score":["least-requests"],"pick":{entry}}}"#)),
            format!("profiles.p.pick.{name}"),
        ),
    }
}

/// The check's refusal of a document that names a host's stage of `kind`
/// under `name`, a name the engine keeps, at `path`.
fn kept_name(path: String, kind: StageKind, name: &str) -> SpecError {
    SpecError::BadStage {
        path,
        reason: format!(
            "{kind} {name:?} bears a name the engine refuses requests under: register the stage under another name"
        ),
    }
}

/// One of the three plain ways to register (`way` 0, 2 or 4: `filter`,
/// `scorer`, `picker`): a factory registered under `name` whose stage
/// answers `name` until the switch returned is set, and `later` at each
/// build while it is.
fn turning(
    way: usize,
    catalog: &mut StageRegistry,
    name: &'static str,
    later: &'static str,
) -> Arc<AtomicBool> {
    let turned = Arc::new(AtomicBool::new(false));
    let switch = Arc::clone(&turned);
    let answers = move || {
        if switch.load(Ordering::Relaxed) {
            later
        } else {
            name
        }
    };
    let added = match way {
        0 => catalog.filter(name, move || Box::new(Under(answers()))),
        2 => catalog.scorer(name, move || Box::new(Under(answers()))),
        _ => catalog.picker(name, move || Box::new(Under(answers()))),
    };
    added.expect("a free name that the registration's build answers");
    turned
}

/// The check's refusal of a document that names a stage of `kind`
/// registered under `name` whose build answers `answers`, at `path`.
fn misnamed(path: String, kind: StageKind, name: &str, answers: &str) -> SpecError {
    SpecError::BadStage {
        path,
        reason: format!("{kind} registered as {name:?} answers the name {answers:?}"),
    }
}

/// The registry takes a host's stage under a name the engine refuses
/// under ([`KEPT`]) as it takes any free name, by each of the six ways:
/// the name's test stands where a document names the stage. A factory
/// that takes no parameter runs once at its registration, for its
/// stage's name.
#[test]
fn each_way_to_register_takes_a_name_the_engine_keeps() {
    for name in KEPT {
        for way in 0..6 {
            let mut catalog = StageRegistry::builtin();
            let built = Arc::new(AtomicUsize::new(0));
            let taken = by_way(way, &mut catalog, name, &built);
            assert_eq!(taken, Ok(()), "{name} by way {way}");
            let last = [
                catalog.filters().last(),
                catalog.scorers().last(),
                catalog.pickers().last(),
            ];
            assert_eq!(last[way / 2], Some(name), "{name} by way {way}");
            assert_eq!(built.load(Ordering::Relaxed), usize::from(way % 2 == 0));
            assert_ne!(catalog.own, 0, "a table of the host's own");
        }
    }
}

/// A document that names a host's stage under a name the engine keeps is
/// refused at its check, and at the name the refusal is by the profile,
/// the stage and why, whichever of the six ways registered the stage and
/// whatever the document writes under the name. No factory runs for the
/// stage there. The same table validates a document that does not name it.
#[test]
fn a_document_that_names_a_stage_under_a_name_the_engine_keeps_is_refused() {
    let free = spec(r#"{"score":["least-requests"],"pick":"best"}"#);
    for name in KEPT {
        for way in 0..6 {
            let mut catalog = StageRegistry::builtin();
            let built = Arc::new(AtomicUsize::new(0));
            by_way(way, &mut catalog, name, &built).expect("the registry takes the name");
            let at_registration = built.load(Ordering::Relaxed);
            for written in [false, true] {
                let (kind, document, path) = naming(way, name, written);
                assert_eq!(
                    document.validate_in(&catalog, &[]).expect_err("refused"),
                    kept_name(path, kind, name),
                    "{name} by way {way}, written {written}"
                );
            }
            assert_eq!(
                built.load(Ordering::Relaxed),
                at_registration,
                "{name} by way {way}: no factory runs at the check"
            );
            assert!(
                free.validate_in(&catalog, &[]).is_ok(),
                "{name} by way {way}: a document that does not name it"
            );
        }
    }
}

/// A name the engine keeps that no stage of the kind holds is unknown, as
/// any other name is: under the stock table, and under a table whose one
/// stage of that name is of another kind.
#[test]
fn a_name_the_engine_keeps_that_no_stage_of_the_kind_holds_is_unknown() {
    let refused = |way: usize, catalog: &StageRegistry| {
        let (_, document, _) = naming(way, BINDING, false);
        document.validate_in(catalog, &[]).expect_err("refused")
    };
    let unknown = |way: usize| {
        let (profile, name) = ("p".to_string(), BINDING.to_string());
        match way / 2 {
            0 => SpecError::UnknownFilter { profile, name },
            1 => SpecError::UnknownScorer { profile, name },
            _ => SpecError::UnknownPicker { profile, name },
        }
    };
    let stock = StageRegistry::builtin();
    for way in [0, 2, 4] {
        assert_eq!(refused(way, &stock), unknown(way), "the stock table");
    }
    // A host's scorer under the name: no filter and no picker bears it.
    let mut scorers = StageRegistry::builtin();
    let built = Arc::new(AtomicUsize::new(0));
    by_way(2, &mut scorers, BINDING, &built).expect("the registry takes the name");
    for way in [0, 4] {
        assert_eq!(refused(way, &scorers), unknown(way), "a scorer's name");
    }
}

/// The check's test of a name is of the whole name: a host's stage under a
/// name that only begins or ends as one the engine keeps is named by a
/// document as any other, by each of the six ways to register.
#[test]
fn a_name_that_only_resembles_one_the_engine_keeps_is_named_as_any_other() {
    let mut catalog = StageRegistry::builtin();
    let added = catalog
        .filter("bindings", || Box::new(Under("bindings")))
        .and_then(|catalog| catalog.filter_with("target", |_| Ok(Box::new(Under("target")))))
        .and_then(|catalog| catalog.scorer("place", || Box::new(Under("place"))))
        .and_then(|catalog| catalog.scorer_with("no-budget", |_| Ok(Box::new(Under("no-budget")))))
        .and_then(|catalog| catalog.picker("partner", || Box::new(Under("partner"))))
        .and_then(|catalog| {
            catalog.picker_with("controllers", |_| Ok(Box::new(Under("controllers"))))
        })
        .map(|_| ());
    assert_eq!(added, Ok(()));
    for pick in ["partner", "controllers"] {
        let document = spec(&format!(
            r#"{{"filter":["bindings","target"],"score":["place","no-budget"],"pick":"{pick}"}}"#
        ));
        let validated = document.validate_in(&catalog, &[]).map(|_| ());
        assert_eq!(validated, Ok(()), "{pick}");
    }
}

/// The engine declines a request under `binding` where its spec binds no
/// plan, and a caller answers by that name. A host's picker of that name
/// is taken by the registry, and a document that names it is refused.
#[test]
fn a_document_that_names_a_hosts_picker_under_binding_is_refused() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker(BINDING, || Box::new(Under(BINDING)))
        .expect("the registry takes the name");
    assert_eq!(catalog.pickers().last(), Some("binding"));
    let refused = spec(r#"{"score":["least-requests"],"pick":"binding"}"#)
        .validate_in(&catalog, &[])
        .expect_err("refused");
    assert_eq!(refused.field_path(), "profiles.p.pick.binding");
    assert_eq!(
        refused.to_string(),
        "profiles.p.pick.binding: picker \"binding\" bears a name the engine refuses requests \
         under: register the stage under another name"
    );
}

/// The list holds each of the engine's own names once, the two a caller
/// answers by among them, and none that a stock stage holds: the public
/// test of a name answers no for each stock stage's.
#[test]
fn the_engines_names_are_distinct_and_no_stock_stage_holds_one() {
    let mut sorted = KEPT;
    sorted.sort_unstable();
    assert!(
        sorted.windows(2).all(|pair| pair[0] != pair[1]),
        "{sorted:?}"
    );
    assert!(KEPT.contains(&BINDING) && KEPT.contains(&TARGET_WORKER));
    let catalog = StageRegistry::builtin();
    let stock = catalog
        .filters()
        .chain(catalog.scorers())
        .chain(catalog.pickers());
    for name in stock {
        assert!(!KEPT.contains(&name) && !refuses_under(name), "{name}");
    }
}

/// The public test of a name answers yes for each name of the list, the
/// two public ones among them. It reads the name whole and by case: it
/// answers no for each name in capitals, for a name that only begins as
/// one, and for one with a space around it.
#[test]
fn the_public_test_of_a_name_answers_yes_for_the_lists_names_alone() {
    for name in KEPT {
        assert!(refuses_under(name), "{name}");
        assert!(!refuses_under(&name.to_uppercase()), "{name} in capitals");
    }
    assert!(refuses_under(BINDING) && refuses_under(TARGET_WORKER));
    let near = [
        "Binding",
        "Target-Worker",
        "bindings",
        "bind",
        " binding",
        "binding ",
        "",
    ];
    for near in near {
        assert!(!refuses_under(near), "{near:?}");
    }
}

/// The check's test of a name is by case too: a host's stage under a name
/// that differs from a kept one in a capital alone is named by a document
/// as any other, by each of the six ways to register.
#[test]
fn a_name_that_differs_from_a_kept_one_by_case_is_named_as_any_other() {
    for way in 0..6 {
        let mut catalog = StageRegistry::builtin();
        let built = Arc::new(AtomicUsize::new(0));
        by_way(way, &mut catalog, "Binding", &built).expect("a free name");
        let (_, document, _) = naming(way, "Binding", false);
        let validated = document.validate_in(&catalog, &[]).map(|_| ());
        assert_eq!(validated, Ok(()), "way {way}");
    }
}

/// The refusal names the entry's own place: a host's filter under a kept
/// name that stands second in its profile's list is refused at that
/// index.
#[test]
fn a_kept_filter_behind_another_is_refused_at_its_own_index() {
    let mut catalog = StageRegistry::builtin();
    let built = Arc::new(AtomicUsize::new(0));
    by_way(0, &mut catalog, BINDING, &built).expect("the registry takes the name");
    let stock = FILTERS[0];
    let document = spec(&format!(
        r#"{{"filter":["{stock}","{BINDING}"],"score":["least-requests"],"pick":"best"}}"#
    ));
    let path = format!("profiles.p.filter[1].{BINDING}");
    assert_eq!(
        document.validate_in(&catalog, &[]).expect_err("refused"),
        kept_name(path, StageKind::Filter, BINDING)
    );
}

/// The name's test stands ahead of the map's: a picker that reads a map,
/// registered under a kept name and written with `null` under it, is
/// refused for its name, and its factory runs at no step.
#[test]
fn a_kept_picker_that_reads_a_map_is_refused_for_its_name_under_null() {
    let mut catalog = StageRegistry::builtin();
    let built = Arc::new(AtomicUsize::new(0));
    by_way(5, &mut catalog, BINDING, &built).expect("the registry takes the name");
    let document = spec(&format!(
        r#"{{"score":["least-requests"],"pick":{{"{BINDING}":null}}}}"#
    ));
    let path = format!("profiles.p.pick.{BINDING}");
    assert_eq!(
        document.validate_in(&catalog, &[]).expect_err("refused"),
        kept_name(path, StageKind::Picker, BINDING)
    );
    assert_eq!(built.load(Ordering::Relaxed), 0, "no factory runs");
}

/// The check holds a plain factory's every build to the name it is
/// registered under, as it holds a build from a document's map. A filter,
/// a scorer or a picker whose factory answers another name at a later
/// build has its document refused at that validation, by both names:
/// where the other name is one the engine keeps, where a stock stage of
/// the kind holds it, where it is free and where it differs from the
/// stage's own in a capital alone, and whether the document writes the
/// name bare or a map under it. The engine an earlier validation built
/// still decides, and the document validates again once the factory
/// answers its name again.
#[test]
fn a_plain_factory_whose_later_build_answers_another_name_is_refused() {
    for way in [0, 2, 4] {
        let stock = [FILTERS[0], SCORERS[0], PICKERS[0]][way / 2];
        for later in [BINDING, stock, "another", "Turns"] {
            let mut catalog = StageRegistry::builtin();
            let turned = turning(way, &mut catalog, "turns", later);
            let (kind, bare, path) = naming(way, "turns", false);
            let standing = bare
                .validate_in(&catalog, &[])
                .expect("the factory answers its name");
            turned.store(true, Ordering::Relaxed);
            for written in [false, true] {
                let (_, document, _) = naming(way, "turns", written);
                assert_eq!(
                    document.validate_in(&catalog, &[]).expect_err("refused"),
                    misnamed(path.clone(), kind, "turns", later),
                    "{kind} then {later}, written {written}"
                );
            }
            assert!(pick(&standing).is_some(), "{kind} then {later}");
            turned.store(false, Ordering::Relaxed);
            let again = bare.validate_in(&catalog, &[]).map(|_| ());
            assert_eq!(again, Ok(()), "{kind} then {later}: its name again");
        }
    }
}

/// The refusal of a build that answers another name names the entry's
/// own place too: a filter and a scorer that stand second in their lists
/// are refused at that index.
#[test]
fn a_build_behind_another_entry_that_answers_another_name_is_refused_at_its_own_index() {
    let mut catalog = StageRegistry::builtin();
    let filter = turning(0, &mut catalog, "sieve", "another");
    let scorer = turning(2, &mut catalog, "weigh", "another");
    let document =
        spec(r#"{"filter":["healthy","sieve"],"score":["least-requests","weigh"],"pick":"best"}"#);
    let validated = document.validate_in(&catalog, &[]).map(|_| ());
    assert_eq!(validated, Ok(()), "both answer their names");
    filter.store(true, Ordering::Relaxed);
    let path = "profiles.p.filter[1].sieve".to_string();
    assert_eq!(
        document.validate_in(&catalog, &[]).expect_err("refused"),
        misnamed(path, StageKind::Filter, "sieve", "another")
    );
    filter.store(false, Ordering::Relaxed);
    scorer.store(true, Ordering::Relaxed);
    let path = "profiles.p.score[1].weigh".to_string();
    assert_eq!(
        document.validate_in(&catalog, &[]).expect_err("refused"),
        misnamed(path, StageKind::Scorer, "weigh", "another")
    );
}
