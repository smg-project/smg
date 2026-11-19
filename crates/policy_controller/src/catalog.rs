//! The stages a build knows by name (design 4.5): the stock build's
//! filters, scorers and pickers, and the ones a host adds in code. A
//! document names a stage; the validator resolves every name here, once,
//! when the document becomes an engine. No decision reads the catalog.
//!
//! One namespace covers the three kinds. A built-in whose parameters are a
//! struct of the spec (every scorer, the `best-with-temperature` picker) is
//! held by name and built by the validator from what the document wrote.
//! A factory that takes no parameters builds a stage at registration to
//! check its name and again for each use during validation. A factory that
//! takes parameters builds during validation from each entry that names it.
//! Each validated policy owns its stages.
//!
//! A stage in code is the host's own binary, trusted as a built-in is: no
//! sandbox and no budget. A policy that must change without a rebuild is a
//! plugin.

use std::{convert::Infallible, fmt, sync::Arc};

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::{
    builtins::{
        Best, CountFirst, HasCapacity, HasRoom, Healthy, Manual, NotOverloaded, NotTried,
        Passthrough, PowerOfTwo, PowerOfTwoCountFirst, Random, RoundRobin,
    },
    framework::{Filter, Picker, Scorer},
    sent::SentCredit,
    spec::{SpecError, SCORERS},
};

/// Builds one stage: once at its registration, then once for each place a
/// document names it in each engine.
type Factory<T> = Box<dyn Fn() -> Box<T> + Send + Sync>;

type ScorerFactory =
    dyn Fn(&mut StageBuild<'_>) -> Result<Box<dyn Scorer>, StageError> + Send + Sync;
type ScorerBuild = Build<dyn Scorer, Box<ScorerFactory>>;

/// How a held name becomes a stage.
enum Build<T: ?Sized, F = Infallible> {
    /// By the validator, from the parameters the document writes under the
    /// name.
    Typed,
    /// By its factory.
    Factory(Factory<T>),
    /// By its factory, once the validator has read the parameters.
    With(F),
}

/// What the validator gets for a name the catalog holds.
pub(crate) enum Found<T: ?Sized, F = Infallible> {
    /// A built-in with typed parameters: the validator builds it.
    Typed,
    /// The stage, built for the engine under validation.
    Built(Box<T>),
    /// The factory the validator calls with the parameters.
    With(F),
}

/// The parameters and profile state available to a stage's factory.
/// The factory borrows the parameters for the call. A stage owns any
/// parameters it keeps.
pub struct StageBuild<'a> {
    written: &'a Map<String, Value>,
    profile: &'a str,
    sent_ttl_ms: Option<u32>,
    sent: &'a mut Option<Arc<SentCredit>>,
}

impl<'a> StageBuild<'a> {
    pub(crate) fn new(
        written: &'a Map<String, Value>,
        profile: &'a str,
        sent_ttl_ms: Option<u32>,
        sent: &'a mut Option<Arc<SentCredit>>,
    ) -> Self {
        Self {
            written,
            profile,
            sent_ttl_ms,
            sent,
        }
    }
}

impl StageBuild<'_> {
    /// Reads the parameters into `T`. A bare name supplies an empty map.
    /// The validator removes the scorer's `weight`, so it appears in
    /// neither `T`'s input nor [`Self::written`]. Derive
    /// `#[serde(deny_unknown_fields)]` on `T`: otherwise a misspelled
    /// parameter is ignored and its default runs.
    pub fn params<T: DeserializeOwned>(&self) -> Result<T, StageError> {
        serde_json::from_value(Value::Object(self.written.clone())).map_err(StageError::from)
    }

    /// The document's parameter map with `weight` removed, the keys of
    /// every map in order: the order of the canonical bytes, whatever
    /// order the document wrote them in.
    pub fn written(&self) -> &Map<String, Value> {
        self.written
    }

    /// The profile's count of sends. The first call for a profile makes
    /// its tally under that profile's `sent_ttl_ms`.
    pub fn sent(&mut self) -> Arc<SentCredit> {
        Arc::clone(
            self.sent
                .get_or_insert_with(|| Arc::new(SentCredit::new(self.profile, self.sent_ttl_ms))),
        )
    }
}

/// Why a factory refuses what the document wrote. It displays as the
/// refusal reads in a document's error, without the stage's path.
#[derive(Debug)]
pub struct StageError(StageRefusal);

#[derive(Debug)]
enum StageRefusal {
    Field {
        field: String,
        value: String,
        reason: &'static str,
    },
    Other(String),
}

impl StageError {
    /// Refuses `value` in `field`, with `reason` describing what it must be.
    pub fn bad(field: &str, value: impl fmt::Display, reason: &'static str) -> Self {
        Self(StageRefusal::Field {
            field: field.to_owned(),
            value: value.to_string(),
            reason,
        })
    }

    /// Refuses the stage's parameters in the factory's own words.
    pub fn other(reason: impl Into<String>) -> Self {
        Self(StageRefusal::Other(reason.into()))
    }

    pub(crate) fn at(self, path: &str) -> SpecError {
        match self.0 {
            StageRefusal::Field {
                field,
                value,
                reason,
            } => SpecError::BadField {
                path: format!("{path}.{field}"),
                value,
                reason,
            },
            StageRefusal::Other(reason) => SpecError::BadStage {
                path: path.to_owned(),
                reason,
            },
        }
    }
}

impl From<serde_json::Error> for StageError {
    fn from(error: serde_json::Error) -> Self {
        Self::other(error.to_string())
    }
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            StageRefusal::Field {
                field,
                value,
                reason,
            } => write!(f, "{field} = {value} is not {reason}"),
            StageRefusal::Other(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for StageError {}

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
    scorers: Vec<(&'static str, ScorerBuild)>,
    pickers: Vec<(&'static str, Build<dyn Picker>)>,
    /// Whether the host makes the lifecycle's calls: a stage that needs
    /// them ([`crate::Lifecycle::needs_lifecycle`]) is in no other build.
    lifecycle: bool,
    /// Whether the host records the prefixes it sent each worker: a
    /// prefix-affinity source that reads them, alone or merged with the
    /// index, is in no other build.
    sent_prefixes: bool,
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
            lifecycle: true,
            sent_prefixes: true,
        }
    }

    /// The catalog of a host that makes some lifecycle calls only: the
    /// validator refuses a stage that needs them, by name, as not in this
    /// build.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn without_lifecycle(mut self) -> Self {
        self.lifecycle = false;
        self
    }

    /// Whether the host makes the lifecycle's calls.
    pub(crate) fn makes_lifecycle_calls(&self) -> bool {
        self.lifecycle
    }

    /// The catalog of a host that records no prefix it sent: the validator
    /// refuses a prefix-affinity source that reads them, by its field, as
    /// not in this build.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn without_sent_prefixes(mut self) -> Self {
        self.sent_prefixes = false;
        self
    }

    /// Whether the host records the prefixes it sent each worker.
    pub(crate) fn records_sent_prefixes(&self) -> bool {
        self.sent_prefixes
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

    /// Adds a scorer that reads parameters beside `weight` under `name`.
    /// A bare name supplies an empty map. Returns [`CatalogError::Held`]
    /// when any stage already holds the name.
    ///
    /// The factory runs once for each place the document names the scorer
    /// at every validation, including a start, an apply, and a dry run.
    /// It must have no effect outside the scorer it returns.
    pub fn scorer_with<F>(
        &mut self,
        name: &'static str,
        build: F,
    ) -> Result<&mut Self, CatalogError>
    where
        F: Fn(&mut StageBuild<'_>) -> Result<Box<dyn Scorer>, StageError> + Send + Sync + 'static,
    {
        if let Some(held) = self.held(name) {
            return Err(CatalogError::Held { name, held });
        }
        self.scorers.push((name, Build::With(Box::new(build))));
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

    /// The scorer a document names, or how the validator builds it.
    pub(crate) fn find_scorer(&self, name: &str) -> Option<Found<dyn Scorer, &ScorerFactory>> {
        find(&self.scorers, name, Box::as_ref)
    }

    /// The picker a document names, as [`Self::find_scorer`].
    pub(crate) fn find_picker(&self, name: &str) -> Option<Found<dyn Picker>> {
        find(&self.pickers, name, |&never| match never {})
    }
}

fn find<'a, T: ?Sized, B, F>(
    held: &'a [(&'static str, Build<T, B>)],
    name: &str,
    factory_ref: impl FnOnce(&'a B) -> F,
) -> Option<Found<T, F>> {
    held.iter()
        .find(|(held, _)| *held == name)
        .map(|(_, build)| match build {
            Build::Typed => Found::Typed,
            Build::Factory(factory) => Found::Built(factory()),
            Build::With(factory) => Found::With(factory_ref(factory)),
        })
}

/// The names by kind and the host's two facts: whether it makes the
/// lifecycle's calls and whether it records the prefixes sent; a factory
/// has nothing to print.
impl fmt::Debug for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Catalog")
            .field("filters", &self.filters().collect::<Vec<_>>())
            .field("scorers", &self.scorers().collect::<Vec<_>>())
            .field("pickers", &self.pickers().collect::<Vec<_>>())
            .field("lifecycle", &self.lifecycle)
            .field("sent_prefixes", &self.sent_prefixes)
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
    fn kept_catalog() -> Catalog {
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
        assert!(catalog.scorers().eq(SCORERS.iter().copied()));
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
        for &name in SCORERS {
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
            if let Err(
                unknown @ (SpecError::UnknownScorer { .. } | SpecError::UnknownPicker { .. }),
            ) = spec(&profile).validate()
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

    fn price_catalog() -> Catalog {
        let mut catalog = Catalog::builtin();
        catalog
            .scorer_with("queue-price", |build| {
                let price: QueuePrice = build.params()?;
                if price.per_request <= 0 {
                    return Err(crate::StageError::bad(
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
        let version = crate::API_VERSION;
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

        let mut catalog = Catalog::builtin();
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
        let mut catalog = Catalog::builtin();
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
        let mut catalog = Catalog::builtin();
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
        let field = crate::StageError::bad("per_request", 0, "above zero");
        assert_eq!(field.to_string(), "per_request = 0 is not above zero");
        let other = crate::StageError::other("no price list");
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
                Err(CatalogError::Held { name, held })
            );
            assert_eq!(called.load(Ordering::Relaxed), 0);
        }
        assert_eq!(catalog.scorers().count(), SCORERS.len() + 1);
    }

    #[test]
    fn a_factory_runs_only_for_valid_entries_that_name_it() {
        let mut catalog = Catalog::builtin();
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
        let mut catalog = Catalog::builtin();
        let called = AtomicUsize::new(0);
        catalog
            .scorer_with(
                "high-id",
                move |_| -> Result<Box<dyn Scorer>, crate::StageError> {
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

        let mut catalog = Catalog::builtin();
        let dropped = Arc::new(AtomicUsize::new(0));
        let drops = Arc::clone(&dropped);
        catalog
            .scorer_with("count-drop", move |_| {
                Ok(Box::new(CountDrop(Arc::clone(&drops))))
            })
            .and_then(|catalog| {
                catalog.scorer_with("refused", |_| Err(crate::StageError::other("no price")))
            })
            .expect("two free names");
        let standing = score_document("[count-drop]")
            .validate_in(&catalog, &[])
            .expect("a valid document");
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        let version = crate::API_VERSION;
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
}
