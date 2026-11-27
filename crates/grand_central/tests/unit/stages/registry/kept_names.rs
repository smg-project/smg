//! The stage registry's cases of the names the engine keeps for its own
//! refusals: the list itself, what the registry takes under such a name,
//! and what a document's check answers for one and for a name near one.

use super::*;

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
/// answers by among them, and none that a stock stage holds.
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
        assert!(!KEPT.contains(&name), "{name}");
    }
}
