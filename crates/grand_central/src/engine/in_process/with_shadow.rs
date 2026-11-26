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
//! **The counts.** A shadow decision is counted in the engine's tables
//! ([`ShadowCounts`]): a single request's in one, and each side of a
//! pair's in its own ([`count`], [`ShadowSide`]). It is
//! counted behind the issue of its live decision: a pick's lease is issued
//! first, and a live shed, which issues none, counts its shadow too. The
//! entry that issues a lease counts by itself ([`issued`]): a stage that
//! panics at that issue is held, so the decision stands and its shadow is
//! counted. The entry that issues none answers what the shadow said, and
//! its caller counts it behind its own filing
//! ([`Loaded::count_shadow`]): a decision whose filing fails counts
//! none. A host reads every count in one call
//! ([`InProcess::shadow_counts`]).
//!
//! **A pair.** A pair's binding names its shadows by side. Its first
//! decision asks them where its entry lends room for what they say
//! (`pair::side_shadows`), and counts them here.

use std::{
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot};

use super::{
    place::Made,
    single::{self, WholeDecision},
    InProcess, Loaded, PoolMembers,
};
use crate::engine::{
    decision::{Decision, PairSide, ShadowKind},
    decision_engine::{DecideOutcome, Request, RequestMaker},
    framework::{Engine, ShadowRows},
    records,
    shadow::{Shadow, ShadowCount, ShadowCounts, ShadowSide, Shadowed, SHADOW_ROWS},
    view::{Constraints, RequestFacts},
};

/// The whole decision of a request whose binding names a shadow: the live
/// profile, then the shadow profile ([`records::with_shadow::decide`]). The
/// shadow's outcome is written into `outcome`, for the caller that counts
/// it.
struct WithShadow<'a> {
    /// The binding's position, how its shadow profile reads the affinity
    /// rows of the live decision ([`Engine::shadow_for`]), and what the
    /// request lends that profile alone ([`RequestMaker::shadow_ring`]).
    binding: (u32, ShadowRows, records::with_shadow::Lends<'a>),
    outcome: &'a mut Shadow,
}

impl WholeDecision for WithShadow<'_> {
    fn decide(
        &mut self,
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
/// the issue ([`count`]). A stage that panics at the issue is held there:
/// the decision stands and its shadow is counted.
///
/// Out of line: the path of a binding without a shadow holds no part of
/// this one.
#[inline(never)]
pub(super) fn issued(
    own: Loaded<'_, '_>,
    decision: Decision,
    request: &Request<'_>,
    now: Instant,
    said: Shadowed,
) -> DecideOutcome {
    let decided = single::issued(own, decision, request, now);
    count(own, said, ShadowSide::Single);
    decided
}

impl Loaded<'_, '_> {
    /// Counts what a binding's shadow said beside a decision of the entry
    /// that issues no lease ([`Self::decide_without_lease`]). It is that
    /// entry's caller's one call, made once its own filing of the
    /// decision's lease stands, as the leasing entry counts behind its own
    /// issue ([`issued`]). A shed files no lease, and its shadow is counted
    /// all the same, as the leasing entry counts a live shed's. A caller
    /// whose filing fails calls nothing, so that decision's shadow counts
    /// none. A decision that had no shadow counts nothing.
    ///
    /// The word is counted under the document this value holds. Its caller
    /// asks it of the value its decision ran under: a word is counted in
    /// the tables of the document whose binding said it, also where
    /// another document was put in force behind the decision.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "a controller's deciding thread calls it")
    )]
    pub(crate) fn count_shadow(self, said: Option<Shadowed>) {
        if let Some(said) = said {
            count(self, said, ShadowSide::Single);
        }
    }
}

impl InProcess {
    /// Tells `each` every count of the shadow decisions made under the
    /// document in force, and answers that document's number, which no
    /// other applied document of the process has. A host that adds up what
    /// a count grew by between two reads takes a count under another
    /// number whole.
    ///
    /// The counts are of each binding that names a shadow, by where its
    /// plan asks one ([`ShadowSide`]) and by each kind, a count of zero
    /// too. A binding of one profile counts beside its single requests. A
    /// pair's binding counts beside each side that names a shadow, and
    /// its decode side beside a decode decided again as well, in a count
    /// of its own. A row is told once, in the order of the rows: the
    /// bindings from the last row on share it ([`SHADOW_ROWS`]). A shadow
    /// that panicked is in no count here: it is one of the engine's faults
    /// ([`DecisionEngine::faults`](crate::engine::DecisionEngine::faults)).
    ///
    /// The document is read once, so every count told is of the number
    /// answered. The call takes no lock and allocates nothing: a count is
    /// one read of its cell, and a document no shadow was counted under
    /// yet reads as zeros.
    pub fn shadow_counts(&self, mut each: impl FnMut(ShadowCount)) -> u64 {
        let document = self.in_force();
        let (engine, counts) = (document.spec.engine(), document.shadow_counts.get());
        // Which sides each row counts by, a side at its place.
        let mut due = [[false; ShadowSide::COUNT]; SHADOW_ROWS];
        for at in (0..).take_while(|&at| engine.binding_at(at).is_some()) {
            let word = engine.shadow_at(at);
            let single = word.is_some_and(|(_, rows)| rows != ShadowRows::Sides);
            let [prefill, decode] = PairSide::BOTH.map(|side| engine.shadows_side(at, side));
            let row = &mut due[(at as usize).min(SHADOW_ROWS - 1)];
            for side in ShadowSide::ALL {
                row[side.index()] |= match side {
                    ShadowSide::Single => single,
                    ShadowSide::Prefill => prefill,
                    ShadowSide::Decode | ShadowSide::DecodeAgain => decode,
                };
            }
        }
        for (binding, due) in (0..).zip(due) {
            for side in ShadowSide::ALL.into_iter().filter(|side| due[side.index()]) {
                for kind in ShadowKind::ALL {
                    let cell = counts.map(|counts| counts.of(binding, side, kind));
                    let count = cell.map_or(0, |cell| cell.load(Ordering::Relaxed));
                    each(ShadowCount {
                        binding,
                        side,
                        kind,
                        count,
                    });
                }
            }
        }
        document.applied
    }
}

/// The live decision of one single request, and what its binding's shadow
/// said beside it where one was due: the whole decision followed by the
/// shadow profile's ([`WithShadow`]). `binding` is the document's word of
/// the request's binding ([`Engine::shadow_for`]), as its list keeps it.
/// `maker` is the request's own, handed on for its entry in a tree
/// its host lends ([`single::decide`]): the live answer is the one the
/// binding would give with no shadow.
///
/// The answer is the live decision's. A request refused ahead of the whole
/// decision has no shadow decision: no profile decided it. Nothing is
/// issued here, and no shadow decision is counted. A shadow's panic is
/// counted here, with the engine's faults, whatever becomes of the lease
/// after it.
///
/// Out of line, as [`issued`] is.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "what the whole decision takes, and the binding's shadow"
)]
pub(super) fn decide(
    own: Loaded<'_, '_>,
    request: &mut Request<'_>,
    members: &PoolMembers,
    maker: &(impl RequestMaker + ?Sized),
    late: Option<&Made<'_, '_>>,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    binding: (u32, ShadowRows),
) -> (Decision, Option<Shadowed>) {
    let mut outcome = Shadow::NotDue;
    let spec = &own.document.spec;
    let lends = |shadow| maker.shadow_ring(spec, shadow);
    let whole = WithShadow {
        binding: (binding.0, binding.1, &lends),
        outcome: &mut outcome,
    };
    let decision = single::decide(own, request, members, maker, late, loads, now, whole);
    let said = match outcome {
        Shadow::Said(said) => Some(said),
        Shadow::NotDue => None,
        Shadow::Faulted => {
            own.held.decisions.fetch_add(1, Ordering::Relaxed);
            None
        }
    };
    (decision, said)
}

/// Counts what a shadow said, for its binding, in the table of the place
/// it was asked: beside a single request, or beside a side of a pair. The
/// tables are allocated at the first count.
pub(super) fn count(own: Loaded<'_, '_>, said: Shadowed, asked: ShadowSide) {
    let counts = &own.document.shadow_counts;
    let counts = counts.get_or_init(Box::<ShadowCounts>::default);
    let count = counts.of(said.binding, asked, said.kind);
    count.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/with_shadow/tests.rs"]
pub(super) mod tests;
