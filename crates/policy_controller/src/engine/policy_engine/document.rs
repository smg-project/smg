//! The document in force: what the engine holds of one policy document, and
//! the call that puts another in its place while the engine runs.
//!
//! [`InForce`] is one value: the document, what is read of it once, and the
//! number a list's value is made under. The engine swaps it whole
//! ([`PolicyEngine::apply`]), so no reader meets one document's stages with
//! another document's word or number. A decision loads the value once, at
//! its entry, and ends under it: what it reads of a document, it reads
//! there.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use arc_swap::Guard;
use smg_worker::runtime::RoutingState;

use super::{pick_by, PolicyEngine, PoolMembers};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision_engine::{Request, RequestMaker},
        framework::{AloneProfile, Picker, SHADOWS},
    },
    stages::Lifecycle,
    store::StaleGeneration,
    tracking::lease::EventSet,
};

/// How many documents the engines of this process took so far.
static APPLIED: AtomicU64 = AtomicU64::new(0);

/// The number of the next document an engine of this process takes: each
/// is drawn here, so no two applied documents share one.
fn next_applied() -> u64 {
    APPLIED.fetch_add(1, Ordering::Relaxed) + 1
}

/// What the engine holds of one policy document: the document, what is
/// read of it once, and its number. An apply swaps the whole value
/// ([`PolicyEngine::apply`]), so no reader meets one document's stages with
/// another document's word or number.
#[derive(Debug)]
pub(in crate::engine) struct InForce {
    pub(super) spec: AppliedSpec,
    /// The number this document drew where the engine took it, which no
    /// other applied document of the process has: a list's value is made
    /// and read under it ([`PolicyEngine::members_of`]).
    pub(in crate::engine) applied: u64,
    /// The kinds of event the document takes ([`Lifecycle::hears`]), read
    /// once, here. A document that takes no word on a lease
    /// ([`EventSet::LEASE`]) costs a request no lease, no id and no word,
    /// and its plain requests are answered a place
    /// ([`Decided::Alone`](crate::engine::Decided::Alone)).
    pub(super) hears: EventSet,
    /// Whether a binding of the document names a shadow profile
    /// ([`SHADOWS`]), read once, here. A whole decision of a single request
    /// pays one test of it, and a place picked alone pays none.
    pub(super) any_shadow: bool,
}

impl InForce {
    /// What the engine holds of `spec`.
    pub(super) fn of(spec: AppliedSpec) -> Self {
        let hears = spec.engine().hears();
        let any_shadow = spec.engine().seat_flags() & SHADOWS != 0;
        Self {
            spec,
            applied: next_applied(),
            hears,
            any_shadow,
        }
    }

    /// One pick under `profile` over `members`, a value the engine made of
    /// a pool ([`PolicyEngine::pool_members`]): the place in the pool's
    /// list of the worker picked. `None` where no worker of the pool
    /// passes, and where this document holds no such profile.
    ///
    /// The picker's rule runs over the pool, whose own word is kept in
    /// `members`, and each worker the rule asks is read once and tested by
    /// `admits`. No row is read, no table looked up, no lock taken and
    /// nothing allocated. Where a rotation's turns all land on refused
    /// workers the pool is passed over once. A rule that asked every worker
    /// and kept none is not followed by that pass.
    ///
    /// `profile` is a position in this document: the caller read it under
    /// the same loaded value.
    #[inline]
    pub(super) fn pick(
        &self,
        profile: AloneProfile,
        members: &PoolMembers,
        admits: impl Fn(RoutingState) -> bool + Copy,
    ) -> Option<usize> {
        let picker = self.picker_of(profile)?;
        pick_by(picker, &members.workers, &members.rotation.0, admits)
    }

    /// The picker of `profile`.
    #[inline]
    pub(super) fn picker_of(&self, profile: AloneProfile) -> Option<&dyn Picker> {
        let profiles = self.spec.engine().profiles();
        Some(profiles.get(usize::from(profile.0))?.picker.as_ref())
    }

    /// The request `maker` makes, as the engine decides it under this
    /// document: with no pins of record lent
    /// ([`Constraints::pins`](crate::engine::view::Constraints::pins)). The
    /// document's session table is the only one it reads, whatever the
    /// entry.
    #[inline(always)]
    pub(super) fn made<'w>(&'w self, maker: &'w impl RequestMaker) -> Request<'w> {
        let mut request = maker.make(&self.spec);
        request.constraints.pins = None;
        request
    }
}

impl PolicyEngine {
    /// The document in force, loaded once for the ask of a list's value
    /// ([`Self::members_of`]): the number the ask compares and the document
    /// a value is made of are that one load's, whatever is applied
    /// meanwhile.
    pub(in crate::engine) fn in_force(&self) -> Guard<Arc<InForce>> {
        self.document.load()
    }

    /// Puts `spec` in force while the engine runs. Every decision that
    /// begins behind this call is decided under it. A decision that began
    /// before it ends under the document it began with, and its lease names
    /// that document's generation.
    ///
    /// The number only rises, as a store's does
    /// ([`MemoryStore::apply`](crate::store::MemoryStore::apply)). A
    /// document whose generation is not above the one in force is refused
    /// by naming the one in force, and nothing changes. So a document that
    /// names no generation (number 0) is never applied.
    ///
    /// The new document's stages start empty. Two things pass to it before
    /// it is shown:
    ///
    /// - A profile's count of sends, which it takes over from the standing
    ///   profile of the same name
    ///   ([`Engine::succeed`](crate::engine::Engine::succeed)).
    /// - The session table, which is the engine's own: the new document's
    ///   engine reads and writes it from here on, and the table takes that
    ///   document's two times. A key's pin outlives the document it was
    ///   made under. The table `spec`'s engine was made with is let go, so
    ///   a key that engine placed before this call is lost.
    ///
    /// No member is told to the new stages: a decision names its stages the
    /// workers of the list its host hands, and a stage that keeps a state
    /// of workers keeps it by the changes told from here on
    /// ([`Self::joined`], [`Self::forget`]).
    ///
    /// A word of a lease is told to the stages of the document the lease
    /// was issued under, and only while that document is in force. A lease
    /// issued before this call is told to no stage from here on: its
    /// dispatch counts no send and pins no session key.
    ///
    /// Nothing a host handed is handed again: the indexes of cache events
    /// and the engine's task stand, and so does a load channel the engine
    /// kept. It kept none where the document in force at the hand-over read
    /// no load report ([`Self::set_loads`]): a host hands that channel
    /// again behind an apply of a document that reads them.
    ///
    /// A list's value is of one document ([`Self::members_of`]). A list
    /// that stands across an apply is asked again under the new document,
    /// and its cell keeps the earlier value beside the new one: one more
    /// value kept and one more step at each read, for each document applied
    /// while the list stands. A program that publishes its lists again
    /// behind an apply starts their cells empty. A value asked before this
    /// call must not be handed to a decision behind it: the new document
    /// would decide over the earlier document's records.
    ///
    /// Applies run one at a time, on the calling thread, and no decision
    /// waits for one. The document replaced is kept until the next apply
    /// and let go there, by the applying thread.
    ///
    /// # Errors
    ///
    /// [`StaleGeneration`]: `spec`'s generation is not above the one in
    /// force.
    pub fn apply(&self, mut spec: AppliedSpec) -> Result<(), StaleGeneration> {
        let mut replaced = self.replaced.lock();
        let standing = self.document.load_full();
        let (number, offered) = (standing.spec.generation(), spec.generation());
        if offered <= number {
            return Err(StaleGeneration {
                standing: number,
                offered,
            });
        }
        spec.engine().succeed(standing.spec.engine());
        spec.share_sessions(Arc::clone(&self.sessions));
        let shown = Arc::new(InForce::of(spec));
        // The document replaced stands until the next apply. The one that
        // stood before it is let go behind the lock.
        let before = replaced.replace(self.document.swap(shown));
        drop(replaced);
        drop(before);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
