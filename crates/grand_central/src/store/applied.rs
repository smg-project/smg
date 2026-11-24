//! The document a store publishes its worlds under, as its host decides by
//! it.
//!
//! A host that decides over worker lists decides by the engine's in-process
//! front ([`InProcess`]), which owns its document and the state of the
//! document's stages. A store tells those stages every word of a lease it
//! files: its issue, its dispatch and its end. So a build that holds that
//! front keeps a store's applied document as that front's engine
//! ([`Applied`]). One engine object stands for a document: the stages that
//! hear of a lease are the ones that decided it, and the document is
//! validated once.
//!
//! A new document is a new engine object, built by the front's constructor.
//! Its session table and its record of sent prompts start empty.
//!
//! The front a store builds runs no task of its own. The store tells it
//! each word of its membership ([`Applied::on_membership`]), and the store's
//! host runs its upkeep on a tick of its own ([`Applied::sweep`]).
//!
//! Every engine a store builds reads the store's own clock: the store hands
//! it its origin. An apply hands a profile's count of sends from the
//! standing engine to the next, and a lease is dated where the store reads
//! its file: one instant reads alike under every document and in the store.

use std::{ops::Deref, time::Instant};

#[cfg(feature = "in-process")]
use crate::engine::in_process::{InProcess, SpecHandle};
#[cfg(not(feature = "in-process"))]
use crate::stages::Lifecycle;
use crate::{document::spec::AppliedSpec, tracking::lease::Membership};

/// One applied document as its store holds it: the engine that decides
/// under it, where the build holds the in-process front, and the document
/// alone otherwise. It reads as the document ([`Deref`]).
#[derive(Debug)]
pub(crate) struct Applied {
    #[cfg(feature = "in-process")]
    engine: InProcess,
    /// The engine's document, held for as long as the engine: a store
    /// puts no second document in force on it.
    #[cfg(feature = "in-process")]
    spec: SpecHandle,
    #[cfg(not(feature = "in-process"))]
    spec: AppliedSpec,
}

impl Applied {
    /// `spec` as a store holds it from its apply on. `origin` is the origin
    /// of the store's clock: the engine that decides under the document
    /// reads that clock.
    pub(crate) fn new(spec: AppliedSpec, origin: Instant) -> Self {
        #[cfg(feature = "in-process")]
        let applied = {
            let engine = InProcess::with_origin(spec, origin);
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

    /// The front that decides under the document: the one engine object
    /// whose stages hear the store's words of a lease.
    #[cfg(feature = "in-process")]
    #[cfg_attr(
        not(any(feature = "controller", test)),
        expect(dead_code, reason = "a controller's deciding thread asks it")
    )]
    pub(crate) fn front(&self) -> &InProcess {
        &self.engine
    }

    /// One word of the store's membership to the engine that decides under
    /// the document, said under the store's lock of its workers. Where the
    /// build holds the front, the front hears it: its records of sent
    /// prefixes, then the document's engine
    /// ([`InProcess::on_membership`]). The front holds a stage's panic
    /// at the word and counts it ([`InProcess::unheard`]): the store's
    /// write goes on. Else the document's engine hears it alone, under no
    /// catch.
    pub(crate) fn on_membership(&self, word: &Membership<'_>) {
        #[cfg(feature = "in-process")]
        self.engine.on_membership(word);
        #[cfg(not(feature = "in-process"))]
        self.spec.engine().on_membership(word);
    }

    /// The upkeep of the engine that decides under the document, at `now`,
    /// under the engine's own catch ([`InProcess::upkeep`]): the session
    /// keys past their end leave its table, and its records of sent prefixes
    /// take a step of the work they left waiting. A controller asks it at
    /// each pass of its presence sweep, a tenth of the grace period: once a
    /// second, the period of the engine's own task in a gateway's process.
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

#[cfg(all(test, feature = "in-process"))]
mod tests {
    use std::time::{Duration, Instant};

    use crate::{
        document::{AppliedSpec, PolicySpec},
        store::MemoryStore,
    };

    /// Every engine a store builds reads the store's own clock: an instant
    /// reads alike in the store and under each document it applies. An apply
    /// hands a profile's count of sends to the next document's engine, which
    /// reads their stamps on the clock they were written on. An engine that
    /// started a clock of its own would read a send of the last second as one
    /// of the future, by the standing document's age.
    #[test]
    fn every_engine_a_store_builds_reads_the_stores_clock() {
        let store = MemoryStore::new();
        let at = Instant::now() + Duration::from_secs(5);
        let engines =
            |store: &MemoryStore| store.deciding(|world| world.front().clock().micros(at));
        for number in 1..=3 {
            let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
            document.generation = number;
            let spec = AppliedSpec::validate(&document).expect("a preset validates");
            store.apply(spec).expect("above the standing one");
            assert_eq!(
                engines(&store),
                store.micros(at),
                "under generation {number}"
            );
        }
    }
}
