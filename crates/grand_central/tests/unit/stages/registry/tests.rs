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

mod filter_parameters;
mod kept_names;
mod picker_parameters;
mod scorer_parameters;

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
