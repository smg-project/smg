use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::{Found, StageKind, StageRegistry, StageRegistryError};
use crate::{
    document::{
        spec::{self, PickSpec, ScoreSpec, FILTERS, PICKERS, SCORERS},
        PolicySpec, SpecError,
    },
    engine::{
        decision::KEPT, Candidate, Constraints, Engine, Filter, Health, Picker, PoolFacts,
        RequestFacts, Scorer, Scratch, Targets, TieBand, WorkerId, BINDING, TARGET_WORKER,
    },
    stages::{pickers::target_of, Lifecycle},
};

/// Keeps the workers whose id is odd.
#[derive(Debug)]
struct OddIds;

impl Filter for OddIds {
    fn name(&self) -> &'static str {
        "odd-ids"
    }

    fn keep(&self, _: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        candidate.id.0 % 2 == 1
    }
}

/// The share of the highest id among the rows handed: the highest id
/// scores one.
fn id_shares(candidates: &[Candidate], kept: &[u32], out: &mut [f32]) {
    let top = kept
        .iter()
        .map(|&i| candidates[i as usize].id.0)
        .max()
        .unwrap_or(1);
    for (slot, &i) in out.iter_mut().zip(kept) {
        *slot = candidates[i as usize].id.0 as f32 / top as f32;
    }
}

/// The highest id is the best.
#[derive(Debug)]
struct HighId;

impl Lifecycle for HighId {}

impl Scorer for HighId {
    fn name(&self) -> &'static str {
        "high-id"
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

/// The lowest id is the best.
#[derive(Debug)]
struct LowId;

impl Lifecycle for LowId {}

impl Scorer for LowId {
    fn name(&self) -> &'static str {
        "low-id"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        id_shares(candidates, kept, out);
        for slot in out.iter_mut() {
            *slot = 1.0 - *slot;
        }
        TieBand::NORMALIZED
    }
}

/// Picks the survivor with the lowest score.
#[derive(Debug)]
struct Worst;

impl Lifecycle for Worst {}

impl Picker for Worst {
    fn name(&self) -> &'static str {
        "worst"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let worst = kept
            .iter()
            .zip(scores)
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&i, _)| i);
        if let Some(i) = worst {
            out.push(target_of(&candidates[i as usize]));
        }
    }
}

/// The stock stages and the four of this module.
fn catalog() -> StageRegistry {
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("odd-ids", || Box::new(OddIds))
        .and_then(|catalog| catalog.scorer("high-id", || Box::new(HighId)))
        .and_then(|catalog| catalog.scorer("low-id", || Box::new(LowId)))
        .and_then(|catalog| catalog.picker("worst", || Box::new(Worst)))
        .expect("four free names");
    catalog
}

/// The highest id is the best, by state the lifecycle's calls keep.
#[derive(Debug)]
struct Kept;

impl Lifecycle for Kept {
    fn needs_lifecycle(&self) -> bool {
        true
    }
}

impl Scorer for Kept {
    fn name(&self) -> &'static str {
        "kept"
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

/// Picks as [`Worst`] does, by state the lifecycle's calls keep.
#[derive(Debug)]
struct KeptPick;

impl Lifecycle for KeptPick {
    fn needs_lifecycle(&self) -> bool {
        true
    }
}

impl Picker for KeptPick {
    fn name(&self) -> &'static str {
        "kept-pick"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Worst.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The module's stages and the two that need the lifecycle's calls.
fn kept_catalog() -> StageRegistry {
    let mut catalog = catalog();
    catalog
        .scorer("kept", || Box::new(Kept))
        .and_then(|catalog| catalog.picker("kept-pick", || Box::new(KeptPick)))
        .expect("two free names");
    catalog
}

/// The refusal of a stage that needs the lifecycle's calls, by its field.
fn lacks_lifecycle(path: &str) -> SpecError {
    SpecError::NotInBuild {
        path: path.to_string(),
        what: "the lifecycle this stage needs",
    }
}

/// A document of one profile, `p`, written as JSON.
fn spec(profile: &str) -> PolicySpec {
    serde_json::from_str(&format!(
        r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{profile}}}}}"#
    ))
    .expect("a well-formed document")
}

/// The first target of one decision over healthy workers 1 to 4.
fn pick(engine: &Engine) -> Option<WorkerId> {
    let fleet: Vec<Candidate> = (0..4)
        .map(|position| Candidate {
            position,
            id: WorkerId(u64::from(position) + 1),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        })
        .collect();
    let facts = RequestFacts {
        model: "a-model",
        ..RequestFacts::default()
    };
    let pool = PoolFacts {
        pool_size: 4,
        ..PoolFacts::default()
    };
    let decision = engine.decide(
        &facts,
        &pool,
        &fleet,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    decision
        .pick()
        .and_then(|pick| pick.targets.first())
        .map(|target| target.id)
}

fn pick_of(profile: &str) -> Option<WorkerId> {
    let engine = spec(profile)
        .validate_in(&catalog(), &[])
        .expect("the document names stages the catalog holds");
    pick(&engine)
}

#[test]
fn a_stage_that_needs_the_lifecycle_is_refused_where_the_host_makes_no_call() {
    let unserved = kept_catalog().without_lifecycle();
    for (profile, path) in [
        (
            r#"{"score":["kept"],"pick":"best"}"#,
            "profiles.p.score[0].kept",
        ),
        (
            r#"{"score":["high-id",{"kept":{"weight":2}}],"pick":"best"}"#,
            "profiles.p.score[1].kept",
        ),
        (
            r#"{"score":["high-id"],"pick":"kept-pick"}"#,
            "profiles.p.pick",
        ),
    ] {
        let refused = spec(profile).validate_in(&unserved, &[]).err();
        assert_eq!(refused, Some(lacks_lifecycle(path)), "{profile}");
    }
    // A shape error the walk meets later outranks it, as it outranks
    // any thing a build lacks.
    let shape = spec(r#"{"score":["kept"],"pick":"manual"}"#)
        .validate_in(&unserved, &[])
        .err();
    assert!(
        matches!(&shape, Some(SpecError::PickerRunsNoScorer { .. })),
        "{shape:?}"
    );
    // The host's word refuses no other stage, its own or a built-in.
    let engine =
        spec(r#"{"filter":["odd-ids"],"score":["high-id","least-requests"],"pick":"worst"}"#)
            .validate_in(&unserved, &[])
            .expect("no stage of the document needs the calls");
    assert!(!engine.needs_lifecycle());
}

#[test]
fn a_host_that_makes_the_calls_runs_a_stage_that_needs_them() {
    let served = kept_catalog();
    for profile in [
        r#"{"score":["kept"],"pick":"best"}"#,
        r#"{"score":["high-id"],"pick":"kept-pick"}"#,
    ] {
        let engine = spec(profile)
            .validate_in(&served, &[])
            .expect("the host makes the calls");
        assert!(engine.needs_lifecycle(), "{profile}");
    }
    let plain = spec(r#"{"score":["high-id"],"pick":"worst"}"#)
        .validate_in(&served, &[])
        .expect("stages the catalog holds");
    assert!(!plain.needs_lifecycle());
}

#[test]
fn a_document_names_a_stage_in_code_as_it_names_a_built_in() {
    // The scorer alone: the highest id of the four.
    assert_eq!(
        pick_of(r#"{"filter":["healthy"],"score":["high-id"],"pick":"best"}"#),
        Some(WorkerId(4))
    );
    // The filter drops the even ids before the scorer reads the rest.
    assert_eq!(
        pick_of(r#"{"filter":["healthy","odd-ids"],"score":["high-id"],"pick":"best"}"#),
        Some(WorkerId(3))
    );
    // The picker takes the lowest score of the filter's survivors.
    assert_eq!(
        pick_of(r#"{"filter":["odd-ids"],"score":["high-id"],"pick":"worst"}"#),
        Some(WorkerId(1))
    );
    // A stage in code beside a built-in: at equal load the ids decide.
    assert_eq!(
        pick_of(r#"{"score":["least-requests","low-id"],"pick":"best"}"#),
        Some(WorkerId(1))
    );
}

#[test]
fn the_weight_a_document_gives_a_scorer_in_code_reaches_the_profile() {
    // Two opposite scorers: the heavier one decides.
    assert_eq!(
        pick_of(r#"{"score":[{"high-id":{"weight":3}},{"low-id":{"weight":1}}],"pick":"best"}"#),
        Some(WorkerId(4))
    );
    assert_eq!(
        pick_of(r#"{"score":[{"high-id":{"weight":1}},{"low-id":{"weight":3}}],"pick":"best"}"#),
        Some(WorkerId(1))
    );
    // A bare name and an empty map weigh one: the opposite scorer
    // decides at one and a half, and loses at one half.
    for one in [r#""high-id""#, r#"{"high-id":{}}"#] {
        let against = |weight: &str| {
            pick_of(&format!(
                r#"{{"score":[{one},{{"low-id":{{"weight":{weight}}}}}],"pick":"best"}}"#
            ))
        };
        assert_eq!(against("1.5"), Some(WorkerId(1)), "{one}");
        assert_eq!(against("0.5"), Some(WorkerId(4)), "{one}");
    }
}

#[test]
fn a_yaml_document_names_a_scorer_in_code_with_its_weight() {
    let document = "apiVersion: smg.policy/v1
profiles:
  p:
    filter: [healthy, odd-ids]
    score:
      - high-id: { weight: 1 }
      - low-id: { weight: 3 }
      - least-requests
    pick: best
";
    let read: PolicySpec = serde_yaml::from_str(document).expect("a well-formed document");
    let engine = read.validate_in(&catalog(), &[]).expect("validates");
    // The odd ids survive, and the heavier scorer prefers the lowest.
    assert_eq!(pick(&engine), Some(WorkerId(1)));
}

#[test]
fn a_held_name_is_refused_at_registration_in_every_kind() {
    let mut catalog = catalog();
    let held = |name, held| Err(StageRegistryError::Held { name, held });
    // A built-in's name, its own kind and another.
    assert_eq!(
        catalog
            .scorer("least-load", || Box::new(HighId))
            .map(|_| ()),
        held("least-load", StageKind::Scorer)
    );
    assert_eq!(
        catalog.filter("best", || Box::new(OddIds)).map(|_| ()),
        held("best", StageKind::Picker)
    );
    assert_eq!(
        catalog.picker("healthy", || Box::new(Worst)).map(|_| ()),
        held("healthy", StageKind::Filter)
    );
    // The picker that takes its temperature holds its name too.
    assert_eq!(
        catalog
            .scorer("best-with-temperature", || Box::new(HighId))
            .map(|_| ()),
        held("best-with-temperature", StageKind::Picker)
    );
    // A name registered once, in its kind and across kinds.
    assert_eq!(
        catalog.scorer("high-id", || Box::new(HighId)).map(|_| ()),
        held("high-id", StageKind::Scorer)
    );
    assert_eq!(
        catalog.picker("odd-ids", || Box::new(Worst)).map(|_| ()),
        held("odd-ids", StageKind::Filter)
    );
    // A refusal adds nothing.
    assert_eq!(catalog.scorers().count(), SCORERS.len() + 2);
}

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

#[test]
fn a_stage_registered_under_another_name_than_its_own_is_refused() {
    let mut catalog = StageRegistry::builtin();
    assert_eq!(
        catalog.scorer("by-id", || Box::new(HighId)).map(|_| ()),
        Err(StageRegistryError::Misnamed {
            kind: StageKind::Scorer,
            name: "by-id",
            answers: "high-id",
        })
    );
    assert_eq!(
        catalog.filter("odd", || Box::new(OddIds)).map(|_| ()),
        Err(StageRegistryError::Misnamed {
            kind: StageKind::Filter,
            name: "odd",
            answers: "odd-ids",
        })
    );
    assert_eq!(
        catalog.picker("last", || Box::new(Worst)).map(|_| ()),
        Err(StageRegistryError::Misnamed {
            kind: StageKind::Picker,
            name: "last",
            answers: "worst",
        })
    );
    assert!(catalog.scorers().eq(SCORERS.iter().copied()));
}

#[test]
fn a_name_no_catalog_holds_is_refused_by_profile_and_name_in_both_forms() {
    let unknown = |profile: &str, catalog: &StageRegistry| {
        spec(profile)
            .validate_in(catalog, &[])
            .expect_err("refused")
    };
    let scorer = |name: &str| SpecError::UnknownScorer {
        profile: "p".into(),
        name: name.into(),
    };
    let catalog = catalog();
    assert_eq!(
        unknown(r#"{"score":["by-id"],"pick":"best"}"#, &catalog),
        scorer("by-id")
    );
    assert_eq!(
        unknown(
            r#"{"score":[{"by-id":{"weight":2}}],"pick":"best"}"#,
            &catalog
        ),
        scorer("by-id")
    );
    // A built-in's name misspelt, with a built-in's own keys under it:
    // the name is what is wrong, whatever it carries.
    assert_eq!(
        unknown(
            r#"{"score":[{"kv-presure":{"spread":0.3,"ceiling":0.9}}],"pick":"best"}"#,
            &catalog
        ),
        scorer("kv-presure")
    );
    assert_eq!(
        unknown(r#"{"score":[{"kv-presure":7}],"pick":"best"}"#, &catalog),
        scorer("kv-presure")
    );
    assert_eq!(
        unknown(r#"{"filter":["even-ids"],"pick":"best"}"#, &catalog),
        SpecError::UnknownFilter {
            profile: "p".into(),
            name: "even-ids".into(),
        }
    );
    assert_eq!(
        unknown(r#"{"pick":"first"}"#, &catalog),
        SpecError::UnknownPicker {
            profile: "p".into(),
            name: "first".into(),
        }
    );
    // The stock build holds none of this module's four: the same
    // document is refused by the validator that takes no catalog.
    let named = spec(r#"{"filter":["odd-ids"],"score":["high-id"],"pick":"worst"}"#);
    assert_eq!(
        named.validate().expect_err("refused"),
        SpecError::UnknownFilter {
            profile: "p".into(),
            name: "odd-ids".into(),
        }
    );
    assert!(named.validate_in(&catalog, &[]).is_ok());
    // A stage is of one kind: a scorer's name is no filter and no picker.
    assert_eq!(
        unknown(r#"{"filter":["high-id"],"pick":"best"}"#, &catalog),
        SpecError::UnknownFilter {
            profile: "p".into(),
            name: "high-id".into(),
        }
    );
}

#[test]
fn a_scorer_in_code_takes_its_weight_alone() {
    let bad = |score: &str| {
        let profile = format!(r#"{{"score":[{score}],"pick":"best"}}"#);
        match spec(&profile).validate_in(&catalog(), &[]) {
            Err(SpecError::BadField { path, .. }) => path,
            other => panic!("{score} validated as {other:?}"),
        }
    };
    assert_eq!(
        bad(r#"{"high-id":{"spread":0.3}}"#),
        "profiles.p.score[0].high-id.spread"
    );
    assert_eq!(
        bad(r#"{"high-id":{"weight":2,"spread":0.3}}"#),
        "profiles.p.score[0].high-id.spread"
    );
    for weight in ["0", "-1", "1e60", r#""2""#, "null", "[2]"] {
        assert_eq!(
            bad(&format!(r#"{{"high-id":{{"weight":{weight}}}}}"#)),
            "profiles.p.score[0].high-id.weight",
            "{weight}"
        );
    }
    for written in ["2", "-1", "0.5", "true", "null", r#""heavy""#, "[]"] {
        assert_eq!(
            bad(&format!(r#"{{"high-id":{written}}}"#)),
            "profiles.p.score[0].high-id",
            "{written}"
        );
    }
    // A built-in's parameters are its struct's: an entry that carries
    // one as a scorer in code, which only a document built in code can,
    // is refused at the entry.
    let mut built_in_code = spec(r#"{"score":[],"pick":"best"}"#);
    for profile in built_in_code.profiles.values_mut() {
        profile.score.push(ScoreSpec::Added {
            name: "least-load".into(),
            written: serde_json::json!({ "weight": 2 }),
        });
    }
    assert_eq!(
        built_in_code.validate_in(&catalog(), &[]).map(|_| ()),
        Err(SpecError::BadField {
            path: "profiles.p.score[0]".into(),
            value: "least-load".into(),
            reason: "a scorer added in code: a built-in's parameters are typed",
        })
    );
    // Two keys in one entry is the parser's refusal, as for a built-in.
    let two = serde_json::from_str::<PolicySpec>(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"high-id":{},"low-id":{}}],"pick":"best"}}}"#,
        )
        .expect_err("two keys");
    assert!(two.to_string().contains("single key"), "{two}");
    // A key written twice is the parser's refusal too, in both document
    // forms and in a built-in's words: the map read would keep the last.
    for name in ["high-id", "least-requests"] {
        let json = format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"score":[{{"{name}":{{"weight":0,"weight":2}}}}],"pick":"best"}}}}}}"#
        );
        let twice = serde_json::from_str::<PolicySpec>(&json).expect_err("a key twice");
        assert!(
            twice.to_string().contains("duplicate field `weight`"),
            "{twice}"
        );
        let yaml = format!(
                "apiVersion: smg.policy/v1\nprofiles:\n  p:\n    score:\n      - {name}: {{weight: 0, weight: 2}}\n    pick: best\n"
            );
        let twice = serde_yaml::from_str::<PolicySpec>(&yaml).expect_err("a key twice");
        assert!(
            twice.to_string().contains("duplicate field `weight`"),
            "{twice}"
        );
    }
}

#[test]
fn each_generation_gets_a_stage_of_its_own() {
    let built = Arc::new(AtomicUsize::new(0));
    let mut catalog = StageRegistry::builtin();
    let count = Arc::clone(&built);
    catalog
        .scorer("high-id", move || {
            count.fetch_add(1, Ordering::Relaxed);
            Box::new(HighId)
        })
        .expect("a free name");
    // The registration builds one, to read its name.
    assert_eq!(built.load(Ordering::Relaxed), 1);
    // A registration under a held name builds none: the name is asked
    // first.
    let count = Arc::clone(&built);
    let again = catalog.scorer("high-id", move || {
        count.fetch_add(1, Ordering::Relaxed);
        Box::new(HighId)
    });
    assert!(matches!(again, Err(StageRegistryError::Held { .. })));
    assert_eq!(built.load(Ordering::Relaxed), 1);
    // So does a filter's registration under it, and a picker's.
    let count = Arc::clone(&built);
    let filter = catalog.filter("high-id", move || {
        count.fetch_add(1, Ordering::Relaxed);
        Box::new(OddIds)
    });
    assert!(matches!(filter, Err(StageRegistryError::Held { .. })));
    let count = Arc::clone(&built);
    let picker = catalog.picker("high-id", move || {
        count.fetch_add(1, Ordering::Relaxed);
        Box::new(Worst)
    });
    assert!(matches!(picker, Err(StageRegistryError::Held { .. })));
    assert_eq!(built.load(Ordering::Relaxed), 1);
    let named = spec(r#"{"score":["high-id"],"pick":"best"}"#);
    for generation in 1..=3 {
        let engine = named.validate_in(&catalog, &[]).expect("validates");
        assert_eq!(built.load(Ordering::Relaxed), 1 + generation);
        assert_eq!(pick(&engine), Some(WorkerId(4)));
    }
    // A document that does not name it builds none.
    spec(r#"{"score":["least-requests"],"pick":"best"}"#)
        .validate_in(&catalog, &[])
        .expect("validates");
    assert_eq!(built.load(Ordering::Relaxed), 4);
}

#[test]
fn an_entry_of_a_scorer_in_code_is_written_as_it_was_read() {
    let score = r#"["high-id",{"low-id":{"weight":0.5}}]"#;
    let read = spec(&format!(r#"{{"score":{score},"pick":"worst"}}"#));
    let again = serde_json::to_value(&read).expect("serializes");
    assert_eq!(again["profiles"]["p"]["score"].to_string(), score);
    assert_eq!(
        serde_json::from_value::<PolicySpec>(again).expect("parses again"),
        read
    );
}

#[test]
fn each_stock_stage_answers_the_name_it_is_held_under() {
    let stock = StageRegistry::builtin();
    for name in FILTERS {
        // Every stock filter is built by its factory.
        let Some(Found::Built(filter)) = stock.find_filter(name) else {
            panic!("{name}: a stock filter takes no parameter");
        };
        assert_eq!(filter.name(), name);
    }
    for name in PICKERS {
        match stock.find_picker(name).expect("a stock picker") {
            Found::Built(picker) => assert_eq!(picker.name(), name),
            // The one picker the validator builds from its parameters.
            Found::Typed => assert_eq!(name, "best-with-temperature"),
            Found::With(_) => panic!("{name}: no stock picker reads a map of its own"),
            Found::Kept => panic!("{name}: no stock picker bears a name the engine keeps"),
        }
    }
    // Every scorer is the validator's to build.
    for &name in SCORERS {
        assert!(
            matches!(stock.find_scorer(name), Some(Found::Typed)),
            "{name}"
        );
    }
}

#[test]
fn the_stock_catalog_is_the_lists_the_spec_publishes() {
    let stock = StageRegistry::builtin();
    assert!(stock.filters().eq(FILTERS), "{stock:?}");
    assert!(stock.scorers().eq(SCORERS.iter().copied()), "{stock:?}");
    assert!(stock.pickers().eq(PICKERS), "{stock:?}");
    // Every bare name of the lists is held: whatever else a document
    // that names it is refused for, it is not an unknown name.
    for filter in FILTERS {
        assert!(stock.find_filter(filter).is_some(), "{filter}");
    }
    for name in SCORERS.iter().chain(&PICKERS) {
        let profile = if SCORERS.contains(name) {
            format!(r#"{{"score":["{name}"],"pick":"best"}}"#)
        } else {
            format!(r#"{{"score":["least-requests"],"pick":"{name}"}}"#)
        };
        if let Err(unknown @ (SpecError::UnknownScorer { .. } | SpecError::UnknownPicker { .. })) =
            spec(&profile).validate()
        {
            panic!("{name}: {unknown:?}");
        }
    }
}

const PRICE_PATH: &str = "profiles.default.score[0].queue-price";
const PRICE_FIELD: &str = "profiles.default.score[0].queue-price.per_request";
const BAD_PRICE: &str = "[{ queue-price: { per_request: -11 } }]";

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QueuePrice {
    per_request: i32,
}

impl Lifecycle for QueuePrice {}

impl Scorer for QueuePrice {
    fn name(&self) -> &'static str {
        "queue-price"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&row, score) in kept.iter().zip(out) {
            let cost = candidates[row as usize].in_flight as f32 * self.per_request as f32;
            *score = 1.0 / (1.0 + cost);
        }
        TieBand::NORMALIZED
    }
}

fn price_catalog() -> StageRegistry {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("queue-price", |build| {
            let price: QueuePrice = build.params()?;
            if price.per_request <= 0 {
                return Err(crate::stages::StageError::bad(
                    "per_request",
                    price.per_request,
                    "above zero",
                ));
            }
            Ok(Box::new(price))
        })
        .expect("a free name");
    catalog
}

fn score_document(score: &str) -> PolicySpec {
    let version = crate::document::API_VERSION;
    serde_yaml::from_str(&format!(
        "apiVersion: {version}\nprofiles:\n  default:\n    score: {score}\n    pick: best\n"
    ))
    .expect("a well-formed document")
}

#[test]
fn a_scorer_keeps_its_typed_parameter_and_the_profile_keeps_its_weight() {
    let engine = score_document("[{ queue-price: { weight: 3.5, per_request: 137 } }]")
        .validate_in(&price_catalog(), &[])
        .expect("the scorer reads its price");
    let weighted = &engine.profiles()[0].scorers[0];
    assert_eq!(weighted.weight, 3.5);
    let row = Candidate {
        in_flight: 1,
        ..Candidate::default()
    };
    let mut out = [0.0];
    weighted
        .scorer
        .score(&RequestFacts::default(), &[row], &[0], &mut out);
    assert_eq!(out, [1.0 / 138.0]);
}

#[test]
fn weight_is_absent_from_both_views_of_the_parameters() {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Price {
        per_request: i32,
        flat: u32,
    }

    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("queue-price", |build| {
            let written = build.written();
            assert!(written
                .keys()
                .map(String::as_str)
                .eq(["flat", "per_request"]));
            let price: Price = build.params()?;
            assert_eq!((price.per_request, price.flat), (41, 13));
            Ok(Box::new(QueuePrice {
                per_request: price.per_request,
            }))
        })
        .expect("a free name");
    score_document("[{ queue-price: { weight: 6.25, per_request: 41, flat: 13 } }]")
        .validate_in(&catalog, &[])
        .expect("the weight is outside the parameters");
}

/// A factory reads its parameters in one order whatever order the
/// document wrote their keys in, at every depth: the order of the
/// canonical bytes, which the digest reads. Two files that are one
/// document by their digest hand a factory the same map.
#[test]
fn a_factory_reads_its_parameters_in_key_order() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let heard = Arc::clone(&seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("queue-price", move |build| {
            let written = serde_json::to_string(build.written()).expect("serializes");
            heard.lock().expect("no panic under the lock").push(written);
            Ok(Box::new(QueuePrice { per_request: 0 }))
        })
        .expect("a free name");
    let one = score_document(
        "[{ queue-price: { weight: 2, per_request: 500, costs: [{ unit: 17, flat: 3 }] } }]",
    );
    let other = score_document(
        "[{ queue-price: { costs: [{ flat: 3, unit: 17 }], per_request: 500, weight: 2 } }]",
    );
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
        r#"{"costs":[{"flat":3,"unit":17}],"per_request":500}"#
    );
}

#[test]
fn a_bare_name_builds_from_an_empty_map_at_weight_one() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("high-id", |build| {
            assert!(build.written().is_empty());
            let typed: serde_json::Map<String, serde_json::Value> = build.params()?;
            assert!(typed.is_empty());
            Ok(Box::new(HighId))
        })
        .expect("a free name");
    for score in ["[high-id]", "[{ high-id: {} }]"] {
        let engine = score_document(score)
            .validate_in(&catalog, &[])
            .expect("the scorer needs no parameters");
        assert_eq!(engine.profiles()[0].scorers[0].weight, 1.0);
    }
}

#[test]
fn an_unknown_or_mistyped_parameter_is_refused_at_the_stage() {
    let catalog = price_catalog();
    for (score, message) in [
        ("[{ queue-price: { spare: 17 } }]", "unknown field `spare`"),
        ("[{ queue-price: { per_request: 'many' } }]", "invalid type"),
    ] {
        let error = score_document(score)
            .validate_in(&catalog, &[])
            .expect_err("the parameter is refused");
        let SpecError::BadStage { path, reason } = error else {
            panic!("{error}");
        };
        assert_eq!(path, PRICE_PATH);
        assert!(reason.contains(message), "{reason}");
    }
}

#[test]
fn a_factory_can_refuse_a_value_at_its_field() {
    let error = score_document(BAD_PRICE)
        .validate_in(&price_catalog(), &[])
        .expect_err("a negative price is refused");
    assert_eq!(
        error,
        SpecError::BadField {
            path: PRICE_FIELD.into(),
            value: "-11".into(),
            reason: "above zero",
        }
    );
    assert_eq!(
        error.to_string(),
        format!("{PRICE_FIELD} = -11 is not above zero")
    );
}

#[test]
fn a_factory_refusal_names_the_field_to_edit() {
    let catalog = price_catalog();
    for (score, path) in [("[queue-price]", PRICE_PATH), (BAD_PRICE, PRICE_FIELD)] {
        let error = score_document(score)
            .validate_in(&catalog, &[])
            .expect_err("the factory refuses the parameters");
        assert_eq!(error.field_path(), path);
    }
}

#[test]
fn a_refusal_displays_without_the_stages_path() {
    let field = crate::stages::StageError::bad("per_request", 0, "above zero");
    assert_eq!(field.to_string(), "per_request = 0 is not above zero");
    let other = crate::stages::StageError::other("no price list");
    assert_eq!(other.to_string(), "no price list");
    let error: &dyn std::error::Error = &other;
    assert!(error.source().is_none());
}

#[test]
fn a_held_name_is_refused_without_running_the_factory() {
    let mut catalog = price_catalog();
    let called = Arc::new(AtomicUsize::new(0));
    for (name, held) in [
        ("healthy", StageKind::Filter),
        ("least-requests", StageKind::Scorer),
        ("best", StageKind::Picker),
        ("queue-price", StageKind::Scorer),
    ] {
        let calls = Arc::clone(&called);
        assert_eq!(
            catalog
                .scorer_with(name, move |_| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(Box::new(HighId))
                })
                .map(|_| ()),
            Err(StageRegistryError::Held { name, held })
        );
        assert_eq!(called.load(Ordering::Relaxed), 0);
    }
    assert_eq!(catalog.scorers().count(), SCORERS.len() + 1);
}

#[test]
fn a_factory_runs_only_for_valid_entries_that_name_it() {
    let mut catalog = StageRegistry::builtin();
    let called = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&called);
    catalog
        .scorer_with("high-id", move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(HighId))
        })
        .expect("a free name");
    assert_eq!(called.load(Ordering::Relaxed), 0);
    assert!(catalog.find_scorer("high-id").is_some());
    assert_eq!(called.load(Ordering::Relaxed), 0);
    score_document("[least-requests]")
        .validate_in(&catalog, &[])
        .expect("the document names another scorer");
    assert_eq!(called.load(Ordering::Relaxed), 0);

    let refused = |score: &str| {
        score_document(score)
            .validate_in(&catalog, &[])
            .expect_err("refused")
    };
    assert_eq!(
        refused("[{ absent: [17] }]"),
        SpecError::UnknownScorer {
            profile: "default".into(),
            name: "absent".into(),
        }
    );
    let path = "profiles.default.score[0].high-id";
    for (score, value) in [
        ("[{ high-id: [17] }]", "[17]"),
        ("[{ high-id: null }]", "null"),
    ] {
        assert_eq!(
            refused(score),
            SpecError::BadField {
                path: path.into(),
                value: value.into(),
                reason: "a map of the scorer's weight: { weight }",
            }
        );
    }
    for (weight, value) in [("-7", "-7"), ("9e70", "9e+70"), ("'many'", "\"many\"")] {
        let error = refused(&format!("[{{ high-id: {{ weight: {weight} }} }}]"));
        assert_eq!(
            error,
            SpecError::BadField {
                path: format!("{path}.weight"),
                value: value.into(),
                reason: "a finite number above zero",
            }
        );
    }
    assert_eq!(called.load(Ordering::Relaxed), 0);

    let named = score_document("[high-id, high-id, high-id]");
    for expected in [3, 6] {
        named.validate_in(&catalog, &[]).expect("validates again");
        assert_eq!(called.load(Ordering::Relaxed), expected);
    }
}

#[test]
fn each_build_checks_the_scorers_name() {
    let mut catalog = StageRegistry::builtin();
    let called = AtomicUsize::new(0);
    catalog
        .scorer_with(
            "high-id",
            move |_| -> Result<Box<dyn Scorer>, crate::stages::StageError> {
                if called.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(Box::new(HighId))
                } else {
                    Ok(Box::new(LowId))
                }
            },
        )
        .expect("a free name");
    let document = score_document("[high-id]");
    document
        .validate_in(&catalog, &[])
        .expect("the name agrees");
    let error = document
        .validate_in(&catalog, &[])
        .expect_err("the name disagrees");
    let SpecError::BadStage { path, reason } = error else {
        panic!("{error}");
    };
    assert_eq!(path, "profiles.default.score[0].high-id");
    assert!(
        reason.contains("high-id") && reason.contains("low-id"),
        "{reason}"
    );
}

#[test]
fn a_refused_profile_drops_the_stages_built_for_the_document() {
    #[derive(Debug)]
    struct CountDrop(Arc<AtomicUsize>);

    impl Lifecycle for CountDrop {}

    impl Scorer for CountDrop {
        fn name(&self) -> &'static str {
            "count-drop"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            HighId.score(facts, candidates, kept, out)
        }
    }

    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let mut catalog = StageRegistry::builtin();
    let dropped = Arc::new(AtomicUsize::new(0));
    let drops = Arc::clone(&dropped);
    catalog
        .scorer_with("count-drop", move |_| {
            Ok(Box::new(CountDrop(Arc::clone(&drops))))
        })
        .and_then(|catalog| {
            catalog.scorer_with("refused", |_| {
                Err(crate::stages::StageError::other("no price"))
            })
        })
        .expect("two free names");
    let standing = score_document("[count-drop]")
        .validate_in(&catalog, &[])
        .expect("a valid document");
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    let version = crate::document::API_VERSION;
    let invalid: PolicySpec = serde_yaml::from_str(&format!(
        "apiVersion: {version}
profiles:
  before: {{ score: [count-drop], pick: best }}
  later: {{ score: [refused], pick: best }}
bindings: [{{ profile: before }}]
"
    ))
    .expect("a well-formed document");
    assert_eq!(
        invalid
            .validate_in(&catalog, &[])
            .expect_err("a later profile is refused"),
        SpecError::BadStage {
            path: "profiles.later.score[0].refused".into(),
            reason: "no price".into(),
        }
    );
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(
        standing.profiles()[0].scorers[0].scorer.name(),
        "count-drop"
    );
    drop(standing);
    assert_eq!(dropped.load(Ordering::Relaxed), 2);
}

#[test]
fn a_plain_scorer_reports_that_it_takes_only_weight() {
    let error = score_document("[{ high-id: { weight: 3.5, spread: 17 } }]")
        .validate_in(&catalog(), &[])
        .expect_err("the plain scorer accepts only weight");
    assert!(matches!(&error, SpecError::BadField { .. }));
    assert_eq!(
            error.to_string(),
            "profiles.default.score[0].high-id.spread = 17 is not a parameter of a scorer added in code, which takes its weight alone"
        );
}

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
