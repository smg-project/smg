//! A single request's decision under a binding that names a shadow profile.
//!
//! A binding may name a shadow: a second profile that decides each request
//! of the binding after the live one. Its decision is counted, by the
//! binding's position and by how it compares with the live decision
//! ([`Shadowed::kind`]), and no request is ever sent by it.
//!
//! **The path.** A single request of a binding that names a shadow comes
//! here for its whole decision ([`decide`]). Its list keeps the document's
//! word of the binding, so a request of a binding that names no shadow is
//! decided as it is under any other document, behind one read of that
//! word. A shadow profile runs inside the whole decision, over the records
//! the live decision read and before that decision is returned
//! ([`records::with_shadow`]). So no request of a binding with a shadow is
//! answered by one pick of its picker or by its key's rows: each takes the
//! whole decision.
//!
//! **What a shadow leaves untouched.** The answer is the live profile's,
//! and the one lease issued is that answer's: the stages hear its events
//! and none of a shadow's. A shadow claims and renews no session pin,
//! writes nothing to the record of sent prefixes, which keeps the live
//! pick, and reads no worker a second time. A shadow profile that panics
//! is caught where it ran: the live answer is returned as decided, and the
//! panic is counted with the engine's faults.
//!
//! **The counts.** A shadow decision is counted after its live decision's
//! lease is issued, in the engine's one table ([`ShadowCounts`]).

use std::{
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot};

use super::{
    single::{self, WholeDecision},
    PolicyEngine, PoolMembers,
};
use crate::engine::{
    decision::Decision,
    decision_engine::{DecideOutcome, Request},
    framework::{Engine, ShadowRows},
    records,
    shadow::{Counted, Shadow, ShadowCounts, Shadowed},
    view::{Constraints, RequestFacts},
};

/// The whole decision of a request whose binding names a shadow: the live
/// profile, then the shadow profile ([`records::with_shadow::decide`]). The
/// shadow's outcome is written into `outcome`, for the caller that counts
/// it.
struct WithShadow<'a> {
    /// The binding's position, and how its shadow profile reads the
    /// affinity rows of the live decision ([`Engine::shadow_for`]).
    binding: (u32, ShadowRows),
    outcome: &'a mut Shadow,
}

impl WholeDecision for WithShadow<'_> {
    fn decide(
        self,
        engine: &Engine,
        records: &records::PoolRecords,
        workers: &[Arc<dyn Worker>],
        live: &records::Live<'_>,
        facts: &mut RequestFacts<'_>,
        cx: &Constraints<'_>,
    ) -> Decision {
        let (decided, outcome) =
            records::with_shadow::decide(engine, records, workers, live, facts, cx, self.binding);
        *self.outcome = outcome;
        decided
    }
}

/// The issue of one single request's decision beside which its binding's
/// shadow said `said` ([`single::issued`]), with that word counted behind
/// the issue ([`count`]): a decision whose issue panics counts none.
///
/// Out of line: the path of a binding without a shadow holds no part of
/// this one.
#[inline(never)]
pub(super) fn issued(
    own: &PolicyEngine,
    decision: Decision,
    request: &Request<'_>,
    now: Instant,
    said: Shadowed,
) -> DecideOutcome {
    let decided = single::issued(own, decision, request, now);
    count(own, said);
    decided
}

/// The live decision of one single request, and what its binding's shadow
/// said beside it where one was due: the whole decision followed by the
/// shadow profile's ([`WithShadow`]). `binding` is the document's word of
/// the request's binding ([`Engine::shadow_for`]), as its list keeps it.
///
/// The answer is the live decision's. A request refused ahead of the whole
/// decision has no shadow decision: no profile decided it. Nothing is
/// issued here, and no shadow decision is counted. A shadow's panic is
/// counted here, with the engine's faults, whatever becomes of the lease
/// after it.
///
/// Out of line, as [`issued`] is.
#[inline(never)]
pub(super) fn decide(
    own: &PolicyEngine,
    request: &mut Request<'_>,
    members: &PoolMembers,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    binding: (u32, ShadowRows),
) -> (Decision, Option<Shadowed>) {
    let mut outcome = Shadow::NotDue;
    let whole = WithShadow {
        binding,
        outcome: &mut outcome,
    };
    let decision = single::decide(own, request, members, loads, now, whole);
    let said = match outcome {
        Shadow::Said(said) => Some(said),
        Shadow::NotDue => None,
        Shadow::Faulted => {
            own.faults.fetch_add(1, Ordering::Relaxed);
            None
        }
    };
    (decision, said)
}

/// Counts what the shadow of a single request said, for its binding. The
/// table is allocated at the first count.
fn count(own: &PolicyEngine, said: Shadowed) {
    let counts = own.shadow_counts.get_or_init(Box::<ShadowCounts>::default);
    let count = counts.of(said.binding, Counted::Single, said.kind);
    count.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests;
