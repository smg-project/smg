//! The decisions the engine keeps to be read later: every Nth pick or shed
//! of each calling thread, under a document that states `explain.sample`.
//!
//! **A sample changes no answer.** A decision is counted behind itself and
//! kept as it stands. A plain request of a place one pick decides is still
//! answered that pick: every Nth of them is kept as the decision that names
//! the worker at its place alone ([`AlonePick::decision`]), a drawn pick's
//! trace with one target and no survivor counted. No whole decision is
//! made for a record, so a request takes one turn of its pool.
//!
//! **The count is the calling thread's own** ([`COUNT`]): no decision
//! writes a word another thread's decision reads. It starts over under
//! another document, by the document's number and its generation, so a
//! thread that decides under two engines in turn starts over at each
//! change. Each thread keeps every Nth of its own decisions, as each
//! decider of a controller does.
//!
//! **A record is found by its request's id** ([`InProcess::explained`]),
//! or where the request states none by the id its host states for it
//! ([`RequestMaker::explain_id`]), and by nothing else. A request that has
//! neither is not counted and not kept. A plain request that is never made
//! is asked its id through its maker ([`RequestMaker::request_id`]), under
//! a document that keeps a sample and under no other.
//!
//! **What is kept**: a single request's decision and a pair's first
//! decision, in the pass that answers both ([`Loaded::decide_into`]). A
//! single's record holds what its binding's shadow said. Not kept: an
//! encode request decided with its items, a decode decided again, and what
//! a pair's shadows said.

use std::{cell::Cell, time::Instant};

use parking_lot::Mutex;

use super::{InProcess, Loaded, PoolMembers};
use crate::{
    engine::{
        decision::Decision,
        decision_engine::{AlonePick, DecideOutcome, Request, RequestMaker},
        explain::{DecisionRecord, SampleCount, Samples},
        records,
        shadow::Shadowed,
        view::{fingerprint, Phase},
    },
    store::GatewayId,
};

/// The decisions kept under one document, newest last: one ring of them
/// behind one lock, which a kept decision takes for its push and a reader
/// for one scan ([`InProcess::explained`]).
pub(super) type Kept = Mutex<Samples>;

thread_local! {
    /// The calling thread's count toward its next sample, under the
    /// document it last counted under. A decision of a document that keeps
    /// no sample does not read it.
    static COUNT: Cell<SampleCount> = const { Cell::new(SampleCount::NEW) };
}

/// Whether the decision of the request `id` names is the calling thread's
/// next sample: the one step of the thread's count a decision takes, under
/// the document's number and its generation. `decision` is the decision
/// where it stands, which counts where it is a pick or a shed
/// ([`SampleCount::keeps`]). A place hands none: it is a pick. A request
/// that is found by no id ([`found_by`]) takes no step.
fn nth(own: Loaded<'_, '_>, id: &[u8], decision: Option<&Decision>) -> bool {
    if id.is_empty() {
        return false;
    }
    let document = own.document;
    let (number, engine) = (document.applied, document.spec.engine());
    COUNT.with(|kept| {
        let mut count = kept.get();
        let nth = match decision {
            Some(decision) => count.keeps(number, engine, decision),
            None => count.steps(number, engine),
        };
        kept.set(count);
        nth
    })
}

/// `decision` pushed onto the document's ring as the decision of the
/// request `id` names, taken at `at`: under the binding a request of
/// `asked`'s model at its phase matches, and with what a single's shadow
/// said.
fn keep(
    own: Loaded<'_, '_>,
    id: &[u8],
    asked: (&str, Phase),
    at: Instant,
    decision: &Decision,
    shadow: Option<Shadowed>,
) {
    let document = own.document;
    let binding = document.spec.engine().binding_position(asked.0, asked.1);
    let taken = (at, GatewayId::LOCAL, id);
    let mut kept = document.kept.lock();
    kept.push(
        document.applied,
        binding,
        taken,
        decision,
        shadow,
        [None; 2],
    );
}

/// The id a decision is kept under: `own_id`, the request's own, or where
/// that is empty the id `maker`'s host states for a kept decision alone
/// ([`RequestMaker::explain_id`]).
fn found_by<'m>(maker: &'m impl RequestMaker, own_id: &'m [u8]) -> &'m [u8] {
    if own_id.is_empty() {
        maker.explain_id()
    } else {
        own_id
    }
}

/// A whole decision's step toward a sample: the decision of `request` at
/// `at`, as the pass wrote it into `written`, is kept where it is the
/// calling thread's next sample ([`nth`]), with what its binding's shadow
/// said, under the id `request` states or else the one `maker`'s host
/// states ([`found_by`]). It is read where it was written, by reference.
/// Under a document that keeps no sample the pass runs this one test of
/// the step.
#[inline(always)]
pub(super) fn whole(
    own: Loaded<'_, '_>,
    maker: &impl RequestMaker,
    request: &Request<'_>,
    at: Instant,
    written: Option<&(DecideOutcome, bool)>,
    shadow: Option<Shadowed>,
) {
    if own.document.samples {
        made_and_counted(own, maker, request, at, written, shadow);
    }
}

/// [`whole`] behind its test of the document. Out of line: no decision of
/// a document that keeps no sample runs it.
#[cold]
#[inline(never)]
fn made_and_counted(
    own: Loaded<'_, '_>,
    maker: &impl RequestMaker,
    request: &Request<'_>,
    at: Instant,
    written: Option<&(DecideOutcome, bool)>,
    shadow: Option<Shadowed>,
) {
    let asked = (request.facts.model, request.facts.phase);
    let id = found_by(maker, request.id);
    counted(own, id, asked, at, written, shadow);
}

/// A noted place's step toward a sample ([`super::single::noted`]): the
/// decision that names the worker at the place, as the pass wrote it into
/// `written`, is kept as a whole decision is ([`whole`]), under the id
/// `maker` states. Under a document that keeps no sample the pass runs
/// this one test of the step.
#[inline(always)]
pub(super) fn noted(
    own: Loaded<'_, '_>,
    maker: &impl RequestMaker,
    at: Instant,
    written: Option<&(DecideOutcome, bool)>,
) {
    if own.document.samples {
        asked_and_counted(own, maker, at, written);
    }
}

/// [`noted`] behind its test of the document: the request's id is asked of
/// `maker` ahead of the count. Out of line, with the asking: a maker that
/// makes the request for its id does so here.
#[cold]
#[inline(never)]
fn asked_and_counted(
    own: Loaded<'_, '_>,
    maker: &impl RequestMaker,
    at: Instant,
    written: Option<&(DecideOutcome, bool)>,
) {
    let id = found_by(maker, maker.request_id(&own.document.spec));
    counted(own, id, (maker.model(), Phase::Single), at, written, None);
}

/// [`whole`] and [`noted`] behind their test of the document. Out of line:
/// no decision of a document that keeps no sample runs it.
#[cold]
#[inline(never)]
fn counted(
    own: Loaded<'_, '_>,
    id: &[u8],
    asked: (&str, Phase),
    at: Instant,
    written: Option<&(DecideOutcome, bool)>,
    shadow: Option<Shadowed>,
) {
    let Some((decided, _)) = written else { return };
    if nth(own, id, Some(&decided.decision)) {
        keep(own, id, asked, at, &decided.decision, shadow);
    }
}

/// A place's step toward a sample, under a document that keeps one and no
/// stage or count of sends of which takes a word on a lease. The place
/// stands as the request's whole answer. Where it is the calling thread's
/// next sample ([`nth`]), it is kept as the decision that names the worker
/// at the place alone ([`AlonePick::decision`]), by the id and the rank a
/// noted pick names that worker by ([`super::single::noted`]), and `clock`
/// is read for that record alone. The request's id is asked of `maker`
/// ahead of the count.
///
/// Out of line: no decision of a document that keeps no sample runs it.
#[cold]
#[inline(never)]
pub(super) fn place(
    own: Loaded<'_, '_>,
    maker: &impl RequestMaker,
    pick: AlonePick,
    members: &PoolMembers,
    clock: impl FnOnce() -> Instant,
) {
    let document = own.document;
    let id = found_by(maker, maker.request_id(&document.spec));
    if !nth(own, id, None) {
        return;
    }
    let position = pick.place as usize;
    let records = members.records(&document.spec, &own.clock);
    let worker = members.workers().get(position);
    if let Some((worker, &named)) = worker.zip(records.ids().get(position)) {
        let decision = pick.decision(named, records::worker_rank(worker.as_ref()));
        keep(
            own,
            id,
            (maker.model(), Phase::Single),
            clock(),
            &decision,
            None,
        );
    }
}

impl InProcess {
    /// The newest decision the engine kept of the request `request_id`
    /// names, where it kept one. Under a document that states
    /// `explain.sample`, each calling thread keeps every Nth pick or shed
    /// among its requests that state an id, in one ring of 1,024 records a
    /// document: a single request's decision and a pair's first decision.
    /// A request that states no id of its own is counted and found by the
    /// id its host states for it
    /// ([`PlaceRequest::explain_id`](crate::engine::PlaceRequest::explain_id)),
    /// and is not counted and not kept where it has neither. A record
    /// names its plan, its profile and that profile's stages through
    /// [`DecisionRecord::names`], read under the document's engine
    /// ([`AppliedSpec::engine`](crate::document::spec::AppliedSpec::engine)).
    ///
    /// The ring is the document's: a document put in force later
    /// ([`Self::apply`]) starts with an empty ring, and this call reads the
    /// ring of the one in force.
    ///
    /// A sample changes no answer. A request one pick of a picker decided
    /// is kept as that pick: its record holds a drawn pick's trace, with
    /// one target and no survivor counted. A record is one decision of the
    /// engine: a call that decides again, past a target it could not send
    /// to, counts each of its decisions.
    #[must_use]
    pub fn explained(&self, request_id: &[u8]) -> Option<DecisionRecord> {
        let document = self.in_force();
        let kept = document.kept.lock();
        kept.find(document.applied, GatewayId::LOCAL, fingerprint(request_id))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/sample/tests.rs"]
mod tests;
