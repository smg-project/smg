//! The document a store publishes its worlds under, as its host decides by
//! it.
//!
//! A host that decides over worker lists decides by the engine's in-process
//! front ([`InProcess`]), which owns the state of its documents' stages.
//! A store tells those stages the words of a lease it files: its issue,
//! its dispatch and its end. So a build that holds that front keeps one
//! engine object for the store's life, and a store's applied document is
//! that engine's value of it ([`Applied`]). The stages that hear of a lease
//! are the ones that decided it, and the document is validated once.
//!
//! A new document is put in force on that engine ([`Applied::next`]), by
//! the call a host that decides in its own process makes
//! ([`InProcess::apply`]). So the session table, the records of sent
//! prefixes and the stages of each profile that did not change stand
//! across it, by that call's own rules and no rule of the store's. Each
//! world holds the document it was published under, and a decision runs
//! under its world's ([`Applied::front`]).
//!
//! The front a store builds runs no task of its own. The store tells it
//! each word of its membership ([`Applied::on_membership`]), and the store's
//! host runs its upkeep on a tick of its own ([`Applied::sweep`]).
//!
//! The engine reads the store's own clock: the store hands it its origin,
//! and a lease is dated where the store reads its file. One instant reads
//! alike under every document and in the store.
//!
//! The engine counts the panics it holds for the store's life
//! ([`EngineFaults`]): a count outlasts the document it rose under, and a
//! panic held under a document that a decision still holds past its apply
//! is counted where the store reads.

#[cfg(feature = "in-process")]
use std::sync::Arc;
use std::{ops::Deref, time::Instant};

#[cfg(feature = "in-process")]
use crate::engine::in_process::{InProcess, Loaded, SpecHandle};
#[cfg(not(feature = "in-process"))]
use crate::stages::Lifecycle;
use crate::{document::spec::AppliedSpec, store::StaleGeneration, tracking::lease::Membership};

/// The panics the engine of one store held under its own catches, as one
/// read of the store answers them ([`MemoryStore::stats`]). A store keeps
/// one engine for its life: they are counted under every document the
/// store applied, and none starts again at a new document. All three are
/// zero in a build without the in-process front, whose store builds no
/// engine.
///
/// [`MemoryStore::stats`]: crate::store::MemoryStore::stats
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EngineFaults {
    /// Whole decisions that panicked, and shadow profiles that did beside
    /// a live decision that stood.
    pub decisions: u64,
    /// Words to the stages that panicked in a stage that heard them, under
    /// the engine's catch: a store's words of membership. A lease's words
    /// are told under their host's catches and counted there: with the
    /// requests that got no decision where a panic unwinds the decision
    /// that told the word, and in
    /// [`MemoryStore::lifecycle_faults`](crate::store::MemoryStore::lifecycle_faults)
    /// at every other word.
    pub events: u64,
    /// Runs of the engine's upkeep that panicked.
    pub upkeep: u64,
}

/// One applied document as its store holds it: the store's engine with its
/// value of the document, where the build holds the in-process front, and
/// the document alone otherwise. It reads as the document ([`Deref`]).
///
/// What keeps a replaced document, with the stages its successor did not
/// keep: each world published under it, and the store's writer until the
/// apply behind its successor's. A thread that writes the store lets go of
/// both: the one that applies, or one that publishes. No deciding thread
/// holds a document but through its world, so none frees a stage.
#[derive(Debug)]
pub(crate) struct Applied {
    /// The store's one engine object: every document the store applies is
    /// put in force on it.
    #[cfg(feature = "in-process")]
    engine: Arc<InProcess>,
    /// The engine's value of this document, as it stood in force: one
    /// counted reference of it.
    #[cfg(feature = "in-process")]
    spec: SpecHandle,
    #[cfg(not(feature = "in-process"))]
    spec: AppliedSpec,
}

impl Applied {
    /// The unbound document ([`AppliedSpec::unbound`]) as a store holds it
    /// at its start. Where the build holds the front, the store's one
    /// engine is built here, on the store's clock: `origin` is that
    /// clock's origin.
    pub(crate) fn unbound(origin: Instant) -> Self {
        let spec = AppliedSpec::unbound();
        #[cfg(feature = "in-process")]
        let applied = {
            let engine = Arc::new(InProcess::with_origin(spec, origin));
            let spec = engine.spec();
            Self { engine, spec }
        };
        #[cfg(not(feature = "in-process"))]
        let applied = {
            // A build without the front keeps no clock beside the store's.
            let _ = origin;
            Self { spec }
        };
        applied
    }

    /// `spec` as the store holds it behind the apply that puts it in place
    /// of this document, the one the store runs. Where the build holds the
    /// front, `spec` is put in force on the store's engine
    /// ([`InProcess::apply`]): what stands across that call stands
    /// here, and a decision that reads the engine's document meets the new
    /// one from this call on. Its caller holds the store's two locks, and
    /// calls it behind the last word that may refuse the generation: a
    /// document in force on the engine is one the store publishes. A build
    /// without the front keeps no stage, and a count of sends takes over
    /// the standing one's
    /// ([`Engine::succeed`](crate::engine::Engine::succeed)).
    ///
    /// # Errors
    ///
    /// [`StaleGeneration`] where `spec`'s number is not above this
    /// document's. Nothing changes then.
    pub(crate) fn next(&self, spec: AppliedSpec) -> Result<Self, StaleGeneration> {
        #[cfg(feature = "in-process")]
        let next = {
            self.engine.apply(spec)?;
            // Behind the store's locks no other apply moves the engine:
            // the document in force is the one this call put there.
            Self {
                engine: Arc::clone(&self.engine),
                spec: self.engine.spec(),
            }
        };
        #[cfg(not(feature = "in-process"))]
        let next = {
            let (standing, offered) = (self.spec.generation(), spec.generation());
            if offered <= standing {
                return Err(StaleGeneration { standing, offered });
            }
            spec.engine().succeed(self.spec.engine());
            Self { spec }
        };
        Ok(next)
    }

    /// Tells a change of membership to what this document built new: the
    /// stages and the counts of sends made under its own generation, and
    /// its session table. What it kept of the standing document heard each
    /// member as it came, and is told nothing again. A store tells the list
    /// of its members through this at the document's apply, while the
    /// document is the one in force. Where the build holds the front, the
    /// front tells it, under its catch of a store's word
    /// ([`InProcess::on_membership_built_new`]): a stage that panics at
    /// the list is held and counted, and the apply goes on. A build without
    /// the front kept nothing: every stage is told, under no catch.
    pub(crate) fn tell_built_new(&self, word: &Membership<'_>) {
        #[cfg(feature = "in-process")]
        self.engine.on_membership_built_new(word);
        #[cfg(not(feature = "in-process"))]
        self.spec.engine().on_membership(word);
    }

    /// The store's engine with this document, as one decision reads it
    /// ([`SpecHandle::under`]). A decision that asks it of its world runs
    /// under the document that world was published under, whatever
    /// document the store applied since. The value borrows the world's
    /// hold of the document: a decision counts nothing for it.
    #[cfg(feature = "in-process")]
    #[cfg_attr(
        not(any(feature = "controller", test)),
        expect(dead_code, reason = "a controller's deciding thread asks it")
    )]
    pub(crate) fn front(&self) -> Loaded<'_, '_> {
        self.spec.under(&self.engine)
    }

    /// The panics the store's engine held under its own catches, as they
    /// stand ([`EngineFaults`]): the one engine counts them, whichever
    /// document this value holds. All zero in a build without the front.
    #[cfg_attr(
        not(feature = "in-process"),
        expect(
            clippy::unused_self,
            reason = "a build without the front holds no count"
        )
    )]
    pub(crate) fn faults(&self) -> EngineFaults {
        #[cfg(feature = "in-process")]
        let [decisions, events, upkeep] = self.engine.held_faults();
        #[cfg(not(feature = "in-process"))]
        let [decisions, events, upkeep] = [0; 3];
        EngineFaults {
            decisions,
            events,
            upkeep,
        }
    }

    /// One word of the store's membership to the engine, said under the
    /// store's lock of its workers. Where the build holds the front, the
    /// front hears it: its records of sent prefixes, then the engine of
    /// the document in force ([`InProcess::on_membership`]), whichever
    /// document this value holds. The front holds a stage's panic at the
    /// word and counts it ([`InProcess::unheard`]): the store's write
    /// goes on. Else this document's engine hears it alone, under no
    /// catch.
    pub(crate) fn on_membership(&self, word: &Membership<'_>) {
        #[cfg(feature = "in-process")]
        self.engine.on_membership(word);
        #[cfg(not(feature = "in-process"))]
        self.spec.engine().on_membership(word);
    }

    /// The upkeep of the store's engine, at `now`, under the engine's own
    /// catch ([`InProcess::upkeep`]): the session keys past their end
    /// leave its table, and its records of sent prefixes take a step of the
    /// work they left waiting. A controller asks it at each pass of its
    /// presence sweep, a tenth of the grace period: once a second, the
    /// period of the engine's own task in a gateway's process.
    #[cfg(feature = "in-process")]
    #[cfg_attr(
        not(feature = "controller"),
        expect(dead_code, reason = "a controller's tick asks it")
    )]
    pub(crate) fn sweep(&self, now: Instant) {
        self.engine.upkeep(now);
    }
}

impl Deref for Applied {
    type Target = AppliedSpec;

    fn deref(&self) -> &AppliedSpec {
        &self.spec
    }
}

// The cases drive a store's engine, which stands under the feature
// `in-process`. The feature's gate is an attribute of its own: the lints
// read a module of cases by its plain `cfg(test)`.
#[cfg(test)]
#[cfg(feature = "in-process")]
#[path = "../../tests/unit/store/applied/tests.rs"]
mod tests;
