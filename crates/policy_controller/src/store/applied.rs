//! The document a store publishes its worlds under, as its host decides by
//! it.
//!
//! A host that decides over worker lists decides by the engine's in-process
//! front ([`PolicyEngine`]), which owns its document and the state of the
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
//! Every engine a store builds reads the store's own clock: the store hands
//! it its origin. An apply hands a profile's count of sends from the
//! standing engine to the next, and a lease is dated where the store reads
//! its file: one instant reads alike under every document and in the store.

use std::{ops::Deref, time::Instant};

use crate::document::spec::AppliedSpec;
#[cfg(feature = "in-process")]
use crate::engine::policy_engine::PolicyEngine;

/// One applied document as its store holds it: the engine that decides
/// under it, where the build holds the in-process front, and the document
/// alone otherwise. It reads as the document ([`Deref`]).
#[derive(Debug)]
pub(crate) struct Applied {
    #[cfg(feature = "in-process")]
    engine: PolicyEngine,
    #[cfg(not(feature = "in-process"))]
    spec: AppliedSpec,
}

impl Applied {
    /// `spec` as a store holds it from its apply on. `origin` is the origin
    /// of the store's clock: the engine that decides under the document
    /// reads that clock.
    pub(crate) fn new(spec: AppliedSpec, origin: Instant) -> Self {
        #[cfg(feature = "in-process")]
        let applied = Self {
            engine: PolicyEngine::with_origin(spec, origin),
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
    pub(crate) fn front(&self) -> &PolicyEngine {
        &self.engine
    }
}

impl Deref for Applied {
    type Target = AppliedSpec;

    fn deref(&self) -> &AppliedSpec {
        #[cfg(feature = "in-process")]
        let spec = self.engine.spec();
        #[cfg(not(feature = "in-process"))]
        let spec = &self.spec;
        spec
    }
}
