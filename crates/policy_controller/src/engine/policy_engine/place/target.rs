//! A single request that names its worker: the one worker it goes to, by
//! a number its caller's client states.
//!
//! The document says which header names it: the target source of the
//! request's binding ([`KeySourceSpec::target`]). The number counts, from
//! 0 and in the order of the list the caller lends, the workers the
//! request can be sent to when the call reads them: those that meet the
//! request's hard demands ([`PlaceRequest::demands`]), are available by
//! the worker's own test, and are taken by the route's rule
//! ([`PlaceRequest::admits`]). The call reads the header once, ahead of
//! its one decision, counts, and holds that decision to the place of the
//! worker the number reaches ([`Constraints::places`]). The decision then
//! answers as it answers any request held to places: the document's
//! filters read the worker, and a worker an earlier attempt tried is
//! handed again. A request that names a worker is never plain, and goes to
//! no other worker: where the number reaches none, or the worker it
//! reaches cannot take the request after all, the request is shed under a
//! name of its own ([`TARGET_WORKER`]).
//!
//! Where no worker of the list can be sent to, there is no worker to
//! count. The request then leaves its worker's hold and is placed by the
//! document as a request that names none, by the binding's rule for
//! workers that are all vetoed. It carries no routing value there: the
//! worker a request names outranks its routing value, so that value keeps
//! no worker over a veto.
//!
//! [`Constraints::places`]: crate::engine::view::Constraints::places
//! [`KeySourceSpec::target`]: crate::document::spec::KeySourceSpec::target

use std::iter;

use super::{admitted, Call, PlaceRequest, Unplaced, ASKS};
use crate::{
    engine::{decision::ShedReason, list_cell::LentList, policy_engine::Loaded},
    tracking::Placement,
};

/// What a request that names its worker is shed under where that worker
/// cannot take it.
const TARGET_WORKER: &str = "target-worker";

/// The shed of a request whose named worker cannot take it: the request
/// has no worker, and moves to no other.
const MISSED: Unplaced = Unplaced::Shed {
    reason: ShedReason::NoEndpoints,
    emptied_by: TARGET_WORKER,
    retry_after_secs: 1,
};

/// One call for a single request over `list`, under a document one of
/// whose bindings names a target source
/// ([`AppliedSpec::names_target`](crate::document::spec::AppliedSpec::names_target)).
/// A request that names no worker is placed as under any other document
/// ([`super::single`]): its binding has no such source, or it carries no
/// text under the header.
///
/// One that names a worker has the list walked once, as far as the worker
/// its number reaches: each worker on the way is read by its own test of
/// being available, and the route's rule is asked of each that is. A
/// worker that fails a hard demand of the request is passed over ahead of
/// both, by the judgement the decision holds the request to
/// ([`PoolRecords::meeting`](crate::engine::records::PoolRecords::meeting)):
/// the request cannot be sent to it, so it takes no number. The
/// call's one decision is held to that worker's place, and the request
/// takes that worker or none. Where the walk finds no worker at all, the
/// request is placed as one that names no worker and carries no routing
/// value ([`super::single`]).
///
/// # Errors
///
/// A fault ([`DecisionRefused::Faulted`](crate::engine::decision_engine::DecisionRefused::Faulted)),
/// counted as a decision's is, where the count's read of the list's
/// records panicked: a list lent with facts of another count than its
/// workers is refused where its value is made.
///
/// The request's shed with no endpoint, under [`TARGET_WORKER`], where a
/// worker of the list can be sent to: its header's text is no number,
/// fewer workers than the number asks can be sent to, the decision held to
/// the place named no worker because of the worker (its own shed with no
/// endpoint, or for workers over their thresholds), or the worker cannot
/// be sent to when the call comes to it: no longer ready, or refused by
/// the route's rule as a pick. Any other answer of [`super::single`]
/// stands as it is.
#[inline(never)]
pub(super) fn single<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    list: LentList<'_>,
) -> Result<Placement<'a>, Unplaced> {
    let mut call = Call::default();
    let spec = &engine.document.spec;
    let sources = spec.key_sources_for(request.model, request.phase());
    let headers = request.headers;
    let named = sources.and_then(|sources| sources.target(|name| headers.header(name)));
    let Some(named) = named else {
        return super::single(engine, request, Some(list), &mut call);
    };
    // The places of the workers that meet the request's hard demands,
    // ascending, where the demands leave a worker out. A request that
    // states none reads no record. The read may be the list's first, and
    // so the fill of its cell: it stands under the engine's catch, as a
    // decision's does.
    let mut meeting = Vec::new();
    let narrowed = !request.demands.is_none()
        && engine
            .caught(|| {
                let records = engine.members_of(list).records(spec, &engine.clock);
                let unmet = records.meeting(request.demands, None, &mut meeting);
                Ok(unmet.is_some())
            })
            .map_err(Unplaced::Refused)?;
    let mut meets = meeting.iter().copied().peekable();
    let mut taking = (0..).zip(&list.workers[..]).filter_map(|(place, worker)| {
        let met = !narrowed || meets.next_if_eq(&place).is_some();
        let takes = met && worker.is_available() && admitted(request, worker.as_ref());
        takes.then_some(place)
    });
    let Some(first) = taking.next() else {
        call.unkeyed = true;
        return super::single(engine, request, Some(list), &mut call);
    };
    let mut taking = iter::once(first).chain(taking);
    let reached = named.and_then(|number| taking.nth(number as usize));
    call.admitted = Some(vec![reached.ok_or(MISSED)?]);
    // The call's one decision: behind a worker it cannot send to, no
    // second one is made.
    call.spent = ASKS - 1;
    super::single(engine, request, Some(list), &mut call).map_err(|unplaced| match unplaced {
        Unplaced::NoWorkers
        | Unplaced::Shed {
            reason: ShedReason::NoEndpoints | ShedReason::Saturated,
            ..
        } => MISSED,
        other => other,
    })
}
