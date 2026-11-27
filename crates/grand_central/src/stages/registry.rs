//! The stages a build knows by name (design 4.5): the stock build's
//! filters, scorers and pickers, and the ones a host adds in code. A
//! document names a stage; the validator resolves every name here, once,
//! when the document becomes an engine. No decision reads the catalog.
//!
//! One namespace covers the three kinds. The engine keeps the names it
//! refuses under: a host's stage is taken under one of them, and the
//! validator refuses a document that names it. At the name it answers
//! [`SpecError::BadStage`]. A built-in whose parameters are a
//! struct of the spec (every scorer, the `best-with-temperature` picker) is
//! held by name and built by the validator from what the document wrote.
//! A factory that takes no parameters builds a stage at registration to
//! check its name and again for each use during validation. A factory that
//! takes parameters builds during validation from each entry that names it.
//! Under a name the engine refuses under neither builds during validation.
//! Each validated policy owns its stages.
//!
//! A stage in code is the host's own binary, trusted as a built-in is: no
//! sandbox and no budget. A policy that must change without a rebuild is a
//! plugin.

use std::{
    convert::Infallible,
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::{
    document::spec::{Needs, SpecError, SCORERS},
    engine::{
        decision::KEPT,
        framework::{Filter, Picker, Scorer},
    },
    load::{
        load_report_reader::LoadReportReader,
        signals::{column_named, SignalColumn},
    },
    policies::{
        manual::Manual,
        passthrough::Passthrough,
        power_of_two::{PowerOfTwo, PowerOfTwoCountFirst},
        random::Random,
        round_robin::RoundRobin,
    },
    stages::{
        filters::{HasCapacity, HasRoom, Healthy, NotOverloaded, NotTried},
        pickers::{Best, CountFirst},
    },
    tracking::dispatch_credit::{DispatchCredit, SentSettings},
};

/// Builds one stage: once at its registration, then once for each place a
/// document names it in each engine. Under a name the engine refuses under
/// ([`KEPT`]) no place of a document builds it.
type Factory<T> = Box<dyn Fn() -> Box<T> + Send + Sync>;

type ScorerFactory =
    dyn Fn(&mut StageBuild<'_>) -> Result<Box<dyn Scorer>, StageError> + Send + Sync;
type ScorerBuild = Build<dyn Scorer, Box<ScorerFactory>>;
pub(crate) type FilterFactory =
    dyn Fn(&mut StageBuild<'_>) -> Result<Box<dyn Filter>, StageError> + Send + Sync;
type FilterBuild = Build<dyn Filter, Box<FilterFactory>>;
pub(crate) type PickerFactory =
    dyn Fn(&mut StageBuild<'_>) -> Result<Box<dyn Picker>, StageError> + Send + Sync;
type PickerBuild = Build<dyn Picker, Box<PickerFactory>>;

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
    /// A host's stage under a name the engine refuses under ([`KEPT`]):
    /// the validator refuses the document that names it, and the lookup
    /// runs no factory for it.
    Kept,
}

/// The parameters and profile state available to a stage's factory.
/// The factory borrows the parameters for the call. A stage owns any
/// parameters it keeps.
pub struct StageBuild<'a> {
    // No field holds the view's header list, so no factory reads it. The
    // keep of a profile's stages across a new header list rests on that
    // (`AppliedSpec::keep_from`).
    written: &'a Map<String, Value>,
    sent_settings: SentSettings,
    sent: &'a mut SentSlot,
    reports: LoadReportReader,
    /// The document's signals, resolved, in name order.
    signals: &'a [SignalColumn],
    resolved_signal: bool,
}

impl<'a> StageBuild<'a> {
    /// The stage `build` makes under a context of its own, held to its own
    /// word on the signal cells: its needs (`needs`) set them exactly where
    /// its factory resolved a signal ([`Self::signal_column`]). A stage
    /// that kept a column and does not say so would be handed no cells and
    /// read no value, in silence. One that says so and kept none would
    /// have a host hand cells nothing reads. The one maker of a context: a
    /// stage of any kind a factory builds comes through here.
    pub(crate) fn built<S: ?Sized>(
        written: &'a Map<String, Value>,
        sent_settings: SentSettings,
        sent: &'a mut SentSlot,
        reports: LoadReportReader,
        signals: &'a [SignalColumn],
        build: impl FnOnce(&mut Self) -> Result<Box<S>, StageError>,
        needs: impl FnOnce(&S) -> Needs,
    ) -> Result<Box<S>, StageError> {
        let mut context = Self {
            written,
            sent_settings,
            sent,
            reports,
            signals,
            resolved_signal: false,
        };
        let stage = build(&mut context)?;
        Self::held_to_cells(
            context.resolved_signal,
            needs(&stage),
            "resolve one or leave the need unset",
        )?;
        Ok(stage)
    }

    /// A stage's own word on the signal cells, held to what its build
    /// resolved. `resolved` says whether a factory resolved a signal under
    /// its context. It is false for a stage the catalog built with no
    /// context: that stage resolved no column of this document, so it
    /// declares no cells. `remedy` ends the refusal of such a need with
    /// what the stage's author can do, which differs by how the stage is
    /// registered.
    pub(crate) fn held_to_cells(
        resolved: bool,
        needs: Needs,
        remedy: &'static str,
    ) -> Result<(), StageError> {
        match (resolved, needs.signal_cells) {
            (true, false) => Err(StageError::other(
                "the stage keeps a signal's column: its needs must set signal_cells",
            )),
            (false, true) => Err(StageError::other(format!(
                "the stage's needs set signal_cells and its build resolved no signal: {remedy}"
            ))),
            _ => Ok(()),
        }
    }
}

impl StageBuild<'_> {
    /// Reads the parameters into `T`. A bare name supplies an empty map.
    /// The validator removes a scorer's `weight`, so it appears in
    /// neither `T`'s input nor [`Self::written`]; a filter's map and a
    /// picker's are handed whole. Put
    /// `#[serde(deny_unknown_fields)]` on `T`: otherwise a misspelled
    /// parameter is ignored, a field with a default takes it, and a
    /// required field is reported missing.
    pub fn params<T: DeserializeOwned>(&self) -> Result<T, StageError> {
        serde_json::from_value(Value::Object(self.written.clone())).map_err(StageError::from)
    }

    /// The document's parameter map, a scorer's with `weight` removed,
    /// the keys of every map in order: the order of the canonical bytes,
    /// whatever order the document wrote them in.
    pub fn written(&self) -> &Map<String, Value> {
        self.written
    }

    /// The profile's count of sends, under its `sent_ttl_ms` and its
    /// `sent_report_delay_ms`. The profiles of a document that set the same
    /// two read one count: the first call hands an earlier profile's count
    /// where one stands under them, and makes one otherwise. A read by any
    /// of those profiles' decisions lets go the sends the row's report
    /// covers, for all of them.
    pub fn dispatch_credit(&mut self) -> Arc<DispatchCredit> {
        profile_sent(self.sent, self.sent_settings)
    }

    /// The view's reading of a row's load report, for a stage that reads
    /// one: a report older than the document's `view.max_report_age_ms`
    /// reads as none, as it does for the built-in stages. A stage that
    /// reads `row.load` itself sees the report at any age. The value is
    /// `Copy`, and asking for it makes nothing.
    pub fn load_report_reader(&self) -> LoadReportReader {
        self.reports
    }

    /// The column of a signal the document declares, by its name. A name
    /// the document does not declare is refused at the stage's entry. A
    /// stage that keeps a column says so in its needs
    /// ([`Needs::signal_cells`](crate::document::Needs::signal_cells)), and a stage
    /// that keeps none does not: the validator refuses either mismatch, so
    /// a host hands the cells to the profiles that read them and to no
    /// other.
    pub fn signal_column(&mut self, name: &str) -> Result<SignalColumn, StageError> {
        let column = column_named(self.signals, name).ok_or_else(|| {
            StageError::other(format!("{name} is not a signal the document declares"))
        })?;
        self.resolved_signal = true;
        Ok(column.clone())
    }
}

/// A profile's slot for its count of sends while its stages are built.
#[derive(Default)]
pub(crate) struct SentSlot {
    /// The count the profile's stages read, from the first ask on.
    pub(crate) asked: Option<Arc<DispatchCredit>>,
    /// The count of an earlier profile of the document under the same
    /// `sent_ttl_ms` and `sent_report_delay_ms`. The engine records a send
    /// in every count, so two counts under one pair of settings would hold
    /// the same sends, and the one whose profile decides no request would
    /// keep them: only a decision's read lets a send go.
    pub(crate) shared: Option<Arc<DispatchCredit>>,
}

/// The count of sends in a profile's slot: the slot's shared one, or one
/// made at the first ask under the profile's `sent_ttl_ms` and its
/// `sent_report_delay_ms`. A factory's context and a typed stage both ask
/// here, so a profile holds one count.
pub(crate) fn profile_sent(slot: &mut SentSlot, settings: SentSettings) -> Arc<DispatchCredit> {
    let SentSlot { asked, shared } = slot;
    Arc::clone(asked.get_or_insert_with(|| {
        shared
            .clone()
            .unwrap_or_else(|| Arc::new(DispatchCredit::new(settings)))
    }))
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
#[non_exhaustive]
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

/// A registration the registry refuses.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StageRegistryError {
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

/// The stages a build knows by name. [`StageRegistry::builtin`] is the stock
/// build's; a host adds its own under new names and hands the catalog to
/// the validator ([`crate::document::PolicySpec::validate_in`]).
///
/// A host's stage is taken under a name the engine refuses requests under
/// by itself, such as [`BINDING`](crate::engine::BINDING) or
/// [`TARGET_WORKER`](crate::engine::TARGET_WORKER), and validation refuses
/// a document that names it: at the name it answers
/// [`SpecError::BadStage`].
///
/// A running engine keeps a profile's stages from one document to the next
/// (`InProcess::apply`) only where both were validated over the stock
/// table, or over one table of the host's own as it stood at both
/// validations. A host that builds its table anew for each document keeps
/// nothing.
pub struct StageRegistry {
    filters: Vec<(&'static str, FilterBuild)>,
    scorers: Vec<(&'static str, ScorerBuild)>,
    pickers: Vec<(&'static str, PickerBuild)>,
    /// Whether the host makes the lifecycle's calls: a stage that needs
    /// them ([`crate::stages::Lifecycle::needs_lifecycle`]) is in no other
    /// build.
    lifecycle: bool,
    /// Zero while the table holds the stock stages alone. Else a number
    /// no other table of the process has, taken anew at each stage a host
    /// adds. Two tables under one number build the same stage under each
    /// name, so an engine may keep stages built over the other
    /// ([`Engine::keep_from`](crate::engine::Engine::keep_from)).
    pub(crate) own: u64,
}

/// The number the next table that takes a stage of a host's own goes under.
static OWN_TABLES: AtomicU64 = AtomicU64::new(1);

/// A built-in that takes no parameter: its factory builds it.
fn of<T: ?Sized + 'static, F>(build: fn() -> Box<T>) -> Build<T, F> {
    Build::Factory(Box::new(build))
}

impl StageRegistry {
    /// The stock build's stages, in the order the spec lists them
    /// ([`crate::document::FILTERS`], [`crate::document::SCORERS`],
    /// [`crate::document::PICKERS`]).
    pub fn builtin() -> Self {
        Self {
            filters: vec![
                ("healthy", of(|| Box::new(Healthy))),
                ("not-overloaded", of(|| Box::new(NotOverloaded))),
                ("has-capacity", of(|| Box::new(HasCapacity))),
                ("has-room", of(|| Box::new(HasRoom))),
                ("not-tried", of(|| Box::new(NotTried))),
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
            own: 0,
        }
    }

    /// The table with one more stage of a host's own: under a number of
    /// its own from here on.
    fn added(&mut self) -> &mut Self {
        self.own = OWN_TABLES.fetch_add(1, Ordering::Relaxed);
        self
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

    /// Adds a filter under `name`: a document names it as it names a
    /// built-in. It takes no parameter: a map written under its name is
    /// refused. Refused when the name is held or is not the filter's own.
    /// Built with no context, the filter resolves no signal: validation
    /// refuses one whose needs set `signal_cells`
    /// ([`Self::filter_with`] resolves one).
    pub fn filter<F>(
        &mut self,
        name: &'static str,
        factory: F,
    ) -> Result<&mut Self, StageRegistryError>
    where
        F: Fn() -> Box<dyn Filter> + Send + Sync + 'static,
    {
        self.admit(StageKind::Filter, name, || factory().name())?;
        self.filters.push((name, Build::Factory(Box::new(factory))));
        Ok(self.added())
    }

    /// Adds a filter that reads parameters under `name`: a document writes
    /// `{ name: { .. } }`, and a bare name supplies an empty map. Returns
    /// [`StageRegistryError::Held`] when any stage already holds the name.
    ///
    /// The factory runs once for each place the document names the filter
    /// at every validation, including a start, an apply, and a dry run.
    /// It must have no effect outside the filter it returns.
    ///
    /// Under a name the engine refuses requests under the factory runs at
    /// no place: validation refuses a document that names the filter.
    pub fn filter_with<F>(
        &mut self,
        name: &'static str,
        build: F,
    ) -> Result<&mut Self, StageRegistryError>
    where
        F: Fn(&mut StageBuild<'_>) -> Result<Box<dyn Filter>, StageError> + Send + Sync + 'static,
    {
        if let Some(held) = self.held(name) {
            return Err(StageRegistryError::Held { name, held });
        }
        self.filters.push((name, Build::With(Box::new(build))));
        Ok(self.added())
    }

    /// Adds a scorer under `name`; a document gives it a weight, `{ name:
    /// { weight } }`, or names it bare at weight one. Refused when the name
    /// is held or is not the scorer's own. Built with no context, the
    /// scorer resolves no signal: validation refuses one whose needs set
    /// `signal_cells` ([`Self::scorer_with`] resolves one).
    pub fn scorer<F>(
        &mut self,
        name: &'static str,
        factory: F,
    ) -> Result<&mut Self, StageRegistryError>
    where
        F: Fn() -> Box<dyn Scorer> + Send + Sync + 'static,
    {
        self.admit(StageKind::Scorer, name, || factory().name())?;
        self.scorers.push((name, Build::Factory(Box::new(factory))));
        Ok(self.added())
    }

    /// Adds a scorer that reads parameters beside `weight` under `name`.
    /// A bare name supplies an empty map. Returns [`StageRegistryError::Held`]
    /// when any stage already holds the name.
    ///
    /// The factory runs once for each place the document names the scorer
    /// at every validation, including a start, an apply, and a dry run.
    /// It must have no effect outside the scorer it returns.
    ///
    /// Under a name the engine refuses requests under the factory runs at
    /// no place: validation refuses a document that names the scorer.
    pub fn scorer_with<F>(
        &mut self,
        name: &'static str,
        build: F,
    ) -> Result<&mut Self, StageRegistryError>
    where
        F: Fn(&mut StageBuild<'_>) -> Result<Box<dyn Scorer>, StageError> + Send + Sync + 'static,
    {
        if let Some(held) = self.held(name) {
            return Err(StageRegistryError::Held { name, held });
        }
        self.scorers.push((name, Build::With(Box::new(build))));
        Ok(self.added())
    }

    /// Adds a picker under `name`. It takes no parameter: a map written
    /// under its name is refused. Refused when the name is held or is not
    /// the picker's own. Built with no context, the picker resolves no
    /// signal: validation refuses one whose needs set `signal_cells`
    /// ([`Self::picker_with`] resolves one).
    pub fn picker<F>(
        &mut self,
        name: &'static str,
        factory: F,
    ) -> Result<&mut Self, StageRegistryError>
    where
        F: Fn() -> Box<dyn Picker> + Send + Sync + 'static,
    {
        self.admit(StageKind::Picker, name, || factory().name())?;
        self.pickers.push((name, Build::Factory(Box::new(factory))));
        Ok(self.added())
    }

    /// Adds a picker that reads parameters under `name`: a document writes
    /// `pick: { name: { .. } }`, and a bare name supplies an empty map.
    /// Returns [`StageRegistryError::Held`] when any stage already holds the
    /// name.
    ///
    /// The factory runs once for each profile that names the picker at
    /// every validation, including a start, an apply, and a dry run. It
    /// must have no effect outside the picker it returns.
    ///
    /// Under a name the engine refuses requests under the factory runs for
    /// no profile: validation refuses a document that names the picker.
    pub fn picker_with<F>(
        &mut self,
        name: &'static str,
        build: F,
    ) -> Result<&mut Self, StageRegistryError>
    where
        F: Fn(&mut StageBuild<'_>) -> Result<Box<dyn Picker>, StageError> + Send + Sync + 'static,
    {
        if let Some(held) = self.held(name) {
            return Err(StageRegistryError::Held { name, held });
        }
        self.pickers.push((name, Build::With(Box::new(build))));
        Ok(self.added())
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
    ) -> Result<(), StageRegistryError> {
        if let Some(held) = self.held(name) {
            return Err(StageRegistryError::Held { name, held });
        }
        let answers = answers();
        if answers != name {
            return Err(StageRegistryError::Misnamed {
                kind,
                name,
                answers,
            });
        }
        Ok(())
    }

    /// The filter a document names, or how the validator builds it.
    pub(crate) fn find_filter(&self, name: &str) -> Option<Found<dyn Filter, &FilterFactory>> {
        find(&self.filters, name, Box::as_ref)
    }

    /// The scorer a document names, or how the validator builds it.
    pub(crate) fn find_scorer(&self, name: &str) -> Option<Found<dyn Scorer, &ScorerFactory>> {
        find(&self.scorers, name, Box::as_ref)
    }

    /// The picker a document names, or how the validator builds it.
    pub(crate) fn find_picker(&self, name: &str) -> Option<Found<dyn Picker, &PickerFactory>> {
        find(&self.pickers, name, Box::as_ref)
    }
}

/// What `held`, the stages of one kind, holds under `name`: the one lookup
/// behind the three kinds. A name the engine refuses under ([`KEPT`]) that
/// a stage of the kind holds answers [`Found::Kept`], ahead of any build:
/// no factory runs for it. One that no stage of the kind holds answers
/// `None`, as any other name does.
fn find<'a, T: ?Sized, B, F>(
    held: &'a [(&'static str, Build<T, B>)],
    name: &str,
    factory_ref: impl FnOnce(&'a B) -> F,
) -> Option<Found<T, F>> {
    let (_, build) = held.iter().find(|(held, _)| *held == name)?;
    if KEPT.contains(&name) {
        return Some(Found::Kept);
    }
    Some(match build {
        Build::Typed => Found::Typed,
        Build::Factory(factory) => Found::Built(factory()),
        Build::With(factory) => Found::With(factory_ref(factory)),
    })
}

/// The names by kind, the host's one fact, whether it makes the
/// lifecycle's calls, and the table's number; a factory has nothing to
/// print.
impl fmt::Debug for StageRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageRegistry")
            .field("filters", &self.filters().collect::<Vec<_>>())
            .field("scorers", &self.scorers().collect::<Vec<_>>())
            .field("pickers", &self.pickers().collect::<Vec<_>>())
            .field("lifecycle", &self.lifecycle)
            .field("own", &self.own)
            .finish()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/stages/registry/tests.rs"]
mod tests;
