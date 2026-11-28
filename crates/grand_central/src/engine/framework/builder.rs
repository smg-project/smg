//! What a document's build gathers for the engine it makes
//! ([`EngineBuilder`], started by [`Engine::builder`]), and the engine made
//! of it ([`Engine::built`]).
//!
//! A build states two things once, where it starts: the age limit of load
//! reports and the number of the stage table its stages are made over.
//! Nothing changes either after. The reading of a report that the build
//! hands its stages is made of the builder's limit, and the engine checks a
//! report's age by the same one. A profile is appended with the count of
//! sends its stages asked for, so the list of counts and each profile's
//! place in it are made together.

use std::sync::Arc;

use super::{Binding, Engine, Profile};
use crate::{
    document::spec::Needs,
    tracking::dispatch_credit::{DispatchCredit, SentSettings},
};

/// The profiles of a document's build in the document's order, with what
/// the engine keeps beside them: the counts of sends their stages read,
/// the age limit of load reports and the number of the stage table.
pub(crate) struct EngineBuilder {
    profiles: Vec<Profile>,
    /// The view's `max_report_age_ms`, as the build started under it.
    max_report_age_ms: Option<u64>,
    /// One count for each two send settings a profile's stages asked
    /// under, in the order the profiles were appended.
    sent_credits: Vec<Arc<DispatchCredit>>,
    /// Beside each profile, the place in `sent_credits` of the count its
    /// stages read, where one of them asked for it.
    sent_of: Vec<Option<usize>>,
    /// The number of the stage table that makes the stages.
    #[cfg(feature = "in-process")]
    stage_table: u64,
}

impl EngineBuilder {
    fn new(max_report_age_ms: Option<u64>, stage_table: u64) -> Self {
        // An engine keeps its table's number under the in-process front
        // alone.
        #[cfg(not(feature = "in-process"))]
        let _ = stage_table;
        Self {
            profiles: Vec::new(),
            max_report_age_ms,
            sent_credits: Vec::new(),
            sent_of: Vec::new(),
            #[cfg(feature = "in-process")]
            stage_table,
        }
    }

    /// The age limit of load reports the build started under: the reading
    /// of a report that its stages are handed is made of it.
    pub(crate) fn max_report_age_ms(&self) -> Option<u64> {
        self.max_report_age_ms
    }

    /// The count of sends that an earlier profile's stages asked for under
    /// `settings`. A later profile under the same two settings reads that
    /// one: the engine records a send in every count, so two counts under
    /// one pair of settings would hold the same sends.
    pub(crate) fn sent_under(&self, settings: SentSettings) -> Option<Arc<DispatchCredit>> {
        let mut counts = self.sent_credits.iter();
        counts.find(|count| count.settings == settings).cloned()
    }

    /// How many profiles the build holds: the place the next one takes.
    pub(crate) fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Appends `profile`, whose stages read the count `asked` where one of
    /// them asked its build for it. A count stands once in the list, under
    /// the first profile that asked for it.
    pub(crate) fn push(&mut self, profile: Profile, asked: Option<Arc<DispatchCredit>>) {
        let read = asked.map(|asked| {
            let listed = |count: &Arc<DispatchCredit>| Arc::ptr_eq(count, &asked);
            match self.sent_credits.iter().position(listed) {
                Some(place) => place,
                None => {
                    self.sent_credits.push(asked);
                    self.sent_credits.len() - 1
                }
            }
        });
        self.sent_of.push(read);
        self.profiles.push(profile);
    }
}

impl Engine {
    /// Starts a document's build of its engine: under the age limit
    /// `max_report_age_ms` of load reports, over the stage table of the
    /// number `stage_table`. A build is told both here, once.
    pub(crate) fn builder(max_report_age_ms: Option<u64>, stage_table: u64) -> EngineBuilder {
        EngineBuilder::new(max_report_age_ms, stage_table)
    }

    /// The engine of what a document's build gathered, under `bindings`:
    /// [`Engine::new`] over the build's profiles, with the counts of sends
    /// their stages read, the build's age limit of load reports and the
    /// number of its stage table. A decision by rank checks a report's age
    /// by that limit, and no call changes it after: an engine made in code
    /// ([`Engine::new`]) checks none. The build's pickers are its stage
    /// table's, and the engine is told so here
    /// ([`Engine::made_by_a_catalog`]): one made in code is not.
    pub(crate) fn built(
        build: EngineBuilder,
        bindings: Vec<Binding>,
        generation: u64,
        needs: Needs,
    ) -> Self {
        let mut engine = Self::new(build.profiles, bindings, generation, needs);
        engine.max_report_age_ms = build.max_report_age_ms;
        engine.sent_credits = build.sent_credits;
        engine.made_by_a_catalog();
        #[cfg(feature = "in-process")]
        {
            engine.sent_of = build.sent_of;
            engine.stage_table = Some(build.stage_table);
        }
        engine
    }
}
