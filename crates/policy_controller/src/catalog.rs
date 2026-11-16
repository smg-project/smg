//! The stages a build knows by name (design 4.5): the stock build's
//! filters, scorers and pickers, and the ones a host adds in code. A
//! document names a stage; the validator resolves every name here, once,
//! when the document becomes an engine. No decision reads the catalog.
//!
//! One namespace covers the three kinds. A built-in whose parameters are a
//! struct of the spec (every scorer, the `best-with-temperature` picker) is
//! held by name and built by the validator from what the document wrote. A
//! stage with no parameters is built by its factory: once when it is
//! registered, to ask the stage its name, and once for each place a document
//! names it, each time a document becomes an engine. An engine owns its
//! stages, and state shared across engines rides the factory's closure.
//!
//! A stage in code is the host's own binary, trusted as a built-in is: no
//! sandbox and no budget. A policy that must change without a rebuild is a
//! plugin.

use std::fmt;

use crate::{
    builtins::{
        Best, CountFirst, HasCapacity, HasRoom, Healthy, Manual, NotOverloaded, NotTried,
        Passthrough, PowerOfTwo, PowerOfTwoCountFirst, Random, RoundRobin,
    },
    framework::{Filter, Picker, Scorer},
    spec::SCORERS,
};

/// Builds one stage: once at its registration, then once for each place a
/// document names it in each engine.
type Factory<T> = Box<dyn Fn() -> Box<T> + Send + Sync>;

/// How a held name becomes a stage.
enum Build<T: ?Sized> {
    /// By the validator, from the parameters the document writes under the
    /// name.
    Typed,
    /// By its factory.
    Factory(Factory<T>),
}

/// What the validator gets for a name the catalog holds.
pub(crate) enum Found<T: ?Sized> {
    /// A built-in with typed parameters: the validator builds it.
    Typed,
    /// The stage, built for the engine under validation.
    Built(Box<T>),
}

/// What a stage is in a profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    Filter,
    Scorer,
    Picker,
}

impl fmt::Display for StageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Filter => "filter",
            Self::Scorer => "scorer",
            Self::Picker => "picker",
        })
    }
}

/// A registration the catalog refuses.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CatalogError {
    /// The name is held already, by a built-in or by an earlier
    /// registration of any kind.
    #[error("stage name {name:?} is already a {held}'s")]
    Held { name: &'static str, held: StageKind },
    /// The stage answers another name than the one it is registered under:
    /// a document, a shed's reason and the stage itself say one name.
    #[error("the {kind} registered as {name:?} answers the name {answers:?}")]
    Misnamed {
        kind: StageKind,
        name: &'static str,
        answers: &'static str,
    },
}

/// The stages a build knows by name. [`Catalog::builtin`] is the stock
/// build's; a host adds its own under new names and hands the catalog to
/// the validator ([`crate::PolicySpec::validate_in`]).
pub struct Catalog {
    filters: Vec<(&'static str, Factory<dyn Filter>)>,
    scorers: Vec<(&'static str, Build<dyn Scorer>)>,
    pickers: Vec<(&'static str, Build<dyn Picker>)>,
}

/// The factory of a built-in that takes no parameter.
fn factory<T: ?Sized + 'static>(build: fn() -> Box<T>) -> Factory<T> {
    Box::new(build)
}

/// A built-in its factory builds.
fn of<T: ?Sized + 'static>(build: fn() -> Box<T>) -> Build<T> {
    Build::Factory(factory(build))
}

impl Catalog {
    /// The stock build's stages, in the order the spec lists them
    /// ([`crate::spec::FILTERS`], [`crate::spec::SCORERS`],
    /// [`crate::spec::PICKERS`]).
    pub fn builtin() -> Self {
        Self {
            filters: vec![
                ("healthy", factory(|| Box::new(Healthy))),
                ("not-overloaded", factory(|| Box::new(NotOverloaded))),
                ("has-capacity", factory(|| Box::new(HasCapacity))),
                ("has-room", factory(|| Box::new(HasRoom))),
                ("not-tried", factory(|| Box::new(NotTried))),
            ],
            // Every scorer takes its weight, and most take more: each is
            // built from its typed parameters.
            scorers: SCORERS.iter().map(|&name| (name, Build::Typed)).collect(),
            pickers: vec![
                ("best", of(|| Box::new(Best))),
                ("round-robin", of(|| Box::new(RoundRobin::new()))),
                ("random", of(|| Box::new(Random))),
                ("power-of-two", of(|| Box::new(PowerOfTwo))),
                ("count-first", of(|| Box::new(CountFirst))),
                (
                    "power-of-two-count-first",
                    of(|| Box::new(PowerOfTwoCountFirst)),
                ),
                ("passthrough", of(|| Box::new(Passthrough))),
                ("manual", of(|| Box::new(Manual))),
                ("best-with-temperature", Build::Typed),
            ],
        }
    }

    /// Adds a filter under `name`: a document names it as it names a
    /// built-in. Refused when the name is held or is not the filter's own.
    pub fn filter<F>(&mut self, name: &'static str, factory: F) -> Result<&mut Self, CatalogError>
    where
        F: Fn() -> Box<dyn Filter> + Send + Sync + 'static,
    {
        self.admit(StageKind::Filter, name, || factory().name())?;
        self.filters.push((name, Box::new(factory)));
        Ok(self)
    }

    /// Adds a scorer under `name`; a document gives it a weight, `{ name:
    /// { weight } }`, or names it bare at weight one. Refused when the name
    /// is held or is not the scorer's own.
    pub fn scorer<F>(&mut self, name: &'static str, factory: F) -> Result<&mut Self, CatalogError>
    where
        F: Fn() -> Box<dyn Scorer> + Send + Sync + 'static,
    {
        self.admit(StageKind::Scorer, name, || factory().name())?;
        self.scorers.push((name, Build::Factory(Box::new(factory))));
        Ok(self)
    }

    /// Adds a picker under `name`. Refused when the name is held or is not
    /// the picker's own.
    pub fn picker<F>(&mut self, name: &'static str, factory: F) -> Result<&mut Self, CatalogError>
    where
        F: Fn() -> Box<dyn Picker> + Send + Sync + 'static,
    {
        self.admit(StageKind::Picker, name, || factory().name())?;
        self.pickers.push((name, Build::Factory(Box::new(factory))));
        Ok(self)
    }

    /// The filters' names: the built-ins in the spec's order, then the
    /// added ones in theirs.
    pub fn filters(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.filters.iter().map(|(name, _)| *name)
    }

    /// The scorers' names, ordered as [`Self::filters`] is.
    pub fn scorers(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.scorers.iter().map(|(name, _)| *name)
    }

    /// The pickers' names, ordered as [`Self::filters`] is.
    pub fn pickers(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.pickers.iter().map(|(name, _)| *name)
    }

    /// The kind that holds `name`, if any does.
    fn held(&self, name: &str) -> Option<StageKind> {
        if self.filters().any(|held| held == name) {
            Some(StageKind::Filter)
        } else if self.scorers().any(|held| held == name) {
            Some(StageKind::Scorer)
        } else if self.pickers().any(|held| held == name) {
            Some(StageKind::Picker)
        } else {
            None
        }
    }

    /// A registration's two checks, in this order: the name is free in every
    /// kind, and it is the name the stage answers. A held name builds no
    /// stage.
    fn admit(
        &self,
        kind: StageKind,
        name: &'static str,
        answers: impl FnOnce() -> &'static str,
    ) -> Result<(), CatalogError> {
        if let Some(held) = self.held(name) {
            return Err(CatalogError::Held { name, held });
        }
        let answers = answers();
        if answers != name {
            return Err(CatalogError::Misnamed {
                kind,
                name,
                answers,
            });
        }
        Ok(())
    }

    /// The filter a document names, built for one engine.
    pub(crate) fn find_filter(&self, name: &str) -> Option<Box<dyn Filter>> {
        self.filters
            .iter()
            .find(|(held, _)| *held == name)
            .map(|(_, factory)| factory())
    }

    /// The scorer a document names: built for one engine, or the validator's
    /// to build from its parameters.
    pub(crate) fn find_scorer(&self, name: &str) -> Option<Found<dyn Scorer>> {
        find(&self.scorers, name)
    }

    /// The picker a document names, as [`Self::find_scorer`].
    pub(crate) fn find_picker(&self, name: &str) -> Option<Found<dyn Picker>> {
        find(&self.pickers, name)
    }
}

fn find<T: ?Sized>(held: &[(&'static str, Build<T>)], name: &str) -> Option<Found<T>> {
    held.iter()
        .find(|(held, _)| *held == name)
        .map(|(_, build)| match build {
            Build::Typed => Found::Typed,
            Build::Factory(factory) => Found::Built(factory()),
        })
}

/// The names by kind; a factory has nothing to print.
impl fmt::Debug for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Catalog")
            .field("filters", &self.filters().collect::<Vec<_>>())
            .field("scorers", &self.scorers().collect::<Vec<_>>())
            .field("pickers", &self.pickers().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::{Catalog, CatalogError, Found, StageKind};
    use crate::{
        builtins::target_of,
        spec::{ScoreSpec, FILTERS, PICKERS, SCORERS},
        Candidate, Constraints, Engine, Filter, Health, Lifecycle, Picker, PolicySpec, PoolFacts,
        RequestFacts, Scorer, Scratch, SpecError, Targets, TieBand, WorkerId,
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
    fn catalog() -> Catalog {
        let mut catalog = Catalog::builtin();
        catalog
            .filter("odd-ids", || Box::new(OddIds))
            .and_then(|catalog| catalog.scorer("high-id", || Box::new(HighId)))
            .and_then(|catalog| catalog.scorer("low-id", || Box::new(LowId)))
            .and_then(|catalog| catalog.picker("worst", || Box::new(Worst)))
            .expect("four free names");
        catalog
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
            pick_of(
                r#"{"score":[{"high-id":{"weight":3}},{"low-id":{"weight":1}}],"pick":"best"}"#
            ),
            Some(WorkerId(4))
        );
        assert_eq!(
            pick_of(
                r#"{"score":[{"high-id":{"weight":1}},{"low-id":{"weight":3}}],"pick":"best"}"#
            ),
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
        let held = |name, held| Err(CatalogError::Held { name, held });
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
        let mut catalog = Catalog::builtin();
        assert_eq!(
            catalog.scorer("by-id", || Box::new(HighId)).map(|_| ()),
            Err(CatalogError::Misnamed {
                kind: StageKind::Scorer,
                name: "by-id",
                answers: "high-id",
            })
        );
        assert_eq!(
            catalog.filter("odd", || Box::new(OddIds)).map(|_| ()),
            Err(CatalogError::Misnamed {
                kind: StageKind::Filter,
                name: "odd",
                answers: "odd-ids",
            })
        );
        assert_eq!(
            catalog.picker("last", || Box::new(Worst)).map(|_| ()),
            Err(CatalogError::Misnamed {
                kind: StageKind::Picker,
                name: "last",
                answers: "worst",
            })
        );
        assert!(catalog.scorers().eq(SCORERS));
    }

    #[test]
    fn a_name_no_catalog_holds_is_refused_by_profile_and_name_in_both_forms() {
        let unknown = |profile: &str, catalog: &Catalog| {
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
        let mut catalog = Catalog::builtin();
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
        assert!(matches!(again, Err(CatalogError::Held { .. })));
        assert_eq!(built.load(Ordering::Relaxed), 1);
        // So does a filter's registration under it, and a picker's.
        let count = Arc::clone(&built);
        let filter = catalog.filter("high-id", move || {
            count.fetch_add(1, Ordering::Relaxed);
            Box::new(OddIds)
        });
        assert!(matches!(filter, Err(CatalogError::Held { .. })));
        let count = Arc::clone(&built);
        let picker = catalog.picker("high-id", move || {
            count.fetch_add(1, Ordering::Relaxed);
            Box::new(Worst)
        });
        assert!(matches!(picker, Err(CatalogError::Held { .. })));
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
        let stock = Catalog::builtin();
        for name in FILTERS {
            let filter = stock.find_filter(name).expect("a stock filter");
            assert_eq!(filter.name(), name);
        }
        for name in PICKERS {
            match stock.find_picker(name).expect("a stock picker") {
                Found::Built(picker) => assert_eq!(picker.name(), name),
                // The one picker the validator builds from its parameters.
                Found::Typed => assert_eq!(name, "best-with-temperature"),
            }
        }
        // Every scorer is the validator's to build.
        for name in SCORERS {
            assert!(
                matches!(stock.find_scorer(name), Some(Found::Typed)),
                "{name}"
            );
        }
    }

    #[test]
    fn the_stock_catalog_is_the_lists_the_spec_publishes() {
        let stock = Catalog::builtin();
        assert!(stock.filters().eq(FILTERS), "{stock:?}");
        assert!(stock.scorers().eq(SCORERS), "{stock:?}");
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
            if let Err(
                unknown @ (SpecError::UnknownScorer { .. } | SpecError::UnknownPicker { .. }),
            ) = spec(&profile).validate()
            {
                panic!("{name}: {unknown:?}");
            }
        }
    }
}
