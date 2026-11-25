//! A request that names its worker: the one worker it goes to, by a number
//! its caller's client states. A single request names its one worker, and
//! a pair one worker a side.
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
//! **A pair** ([`pair`]). A pair's binding names one list of sources for
//! its two sides, and its target source is the side's whose profile is the
//! hash ring keyed by no prefix
//! ([`Engine::names_worker_on`](crate::engine::framework::Engine::names_worker_on)):
//! the one profile whose policy in a gateway reads that header. The
//! prefill's number counts the prefills that can be sent to and have a
//! partner that can, at the places their host states for the side. A
//! worker at a place a standing veto of that host masks
//! ([`PairLists::prefill_stalled`], [`PairLists::decode_stalled`]) cannot
//! be sent to. Where
//! one of them has such a partner among its first partners, it counts
//! those prefills alone, as a pair's decision shows them alone. Where the
//! decode's side is the ring's too, its number counts those partners of
//! the prefill the number reached, and the pair's one decision is held to
//! the two places, a side each ([`PairLists::prefill_places`],
//! [`PairLists::decode_places`]). That decision books the decode with its
//! prefill under every document: a decode asked again behind the prefill
//! would be picked by the pair's routing value, so none is. Where the
//! decode's side is not the ring's, the decision is held to
//! the prefill's place and the decode is its own profile's pick. A pair
//! whose decode side alone is the ring's is read for no target: its
//! prefill is its profile's pick, known inside the decision alone. A pair
//! with no prefill to count leaves its hold with its routing value, which
//! keeps no prefill of a pair over a veto.
//!
//! **A controller's served decision** counts a single request's number by
//! the same function ([`reached`]), over the list its worker objects lend
//! and by tests of its own: a place the asking gateway may take, that none
//! of its standing vetoes holds back, whose worker is available by its own
//! test. Its frame states the number, which the asking host read by the
//! same source. The route's rule is that gateway's own code, and is not
//! asked there.
//!
//! [`Constraints::places`]: crate::engine::view::Constraints::places
//! [`KeySourceSpec::target`]: crate::document::spec::KeySourceSpec::target

use std::{iter, sync::Arc};

use smg_worker::identity::basic::Worker;

use super::{admitted, Call, PlaceRequest, Unplaced, ASKS};
#[cfg(feature = "controller")]
use crate::engine::decision::{Decision, Shed};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::ShedReason, in_process::Loaded, list_cell::LentList, view::Role, PairLists,
        WorkerLists,
    },
    tracking::Placement,
};

/// What a request that names its worker is shed under where that worker
/// cannot take it: by the one call in a host's own process, and by a
/// controller's served decision.
pub(crate) const TARGET_WORKER: &str = "target-worker";

/// The shed of a request whose named worker cannot take it: the request
/// has no worker, and moves to no other.
const MISSED: Unplaced = Unplaced::Shed {
    reason: ShedReason::NoEndpoints,
    emptied_by: TARGET_WORKER,
    retry_after_secs: 1,
};

/// What a decision held to a named worker answers, as the request's
/// answer: where it named no worker because of the worker (no worker for
/// the call's asks, its own shed with no endpoint, or for workers over
/// their thresholds), the request is shed under [`TARGET_WORKER`]. Any
/// other answer stands as it is.
fn held<'a>(placed: Result<Placement<'a>, Unplaced>) -> Result<Placement<'a>, Unplaced> {
    placed.map_err(|unplaced| match unplaced {
        Unplaced::NoWorkers
        | Unplaced::Shed {
            reason: ShedReason::NoEndpoints | ShedReason::Saturated,
            ..
        } => MISSED,
        other => other,
    })
}

/// [`held`] for a controller's served decision, the one decision it held
/// to a named worker's place: the decision's own, but that a shed with no
/// endpoint, or for workers over their thresholds, is the request's shed
/// under [`TARGET_WORKER`]. The decision named no worker because of that
/// worker, and the request moves to no other. The shed keeps the trace
/// of the decision that ran: it names the profile that decision was made
/// under, as any shed made once a profile is resolved does.
#[cfg(feature = "controller")]
pub(crate) fn held_answer(decision: Decision) -> Decision {
    match decision {
        Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints | ShedReason::Saturated,
            trace,
            ..
        }) => Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: TARGET_WORKER,
            retry_after_secs: 1,
            wait_allowed: false,
            trace,
        }),
        other => other,
    }
}

/// Where the number a single request names its worker by reaches
/// ([`reached`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reached {
    /// No worker of the list can be sent to: there is none to count, and
    /// the request leaves its worker's hold.
    NoneToCount,
    /// A worker of the list can be sent to and the number reaches none:
    /// its text is no number, or fewer workers can be sent to.
    NoWorker,
    /// The place of the worker the number reaches.
    Place(u32),
}

/// The count of a single request that names its worker, the one count of
/// both doors: the one call in a host's own process ([`single`]) and a
/// controller's served decision hand it their own tests.
///
/// `number` counts, from 0 and in the list's order, the workers of
/// `workers` the request can be sent to: those `takes` answers for, asked
/// with each worker's place. `None` is a text that is no number, which
/// reaches no worker of any list. `takes` is asked of each worker in turn
/// as far as the worker the number reaches, and of none behind it: with no
/// number, as far as the first worker it answers for.
pub(crate) fn reached(
    workers: &[Arc<dyn Worker>],
    number: Option<u32>,
    mut takes: impl FnMut(u32, &dyn Worker) -> bool,
) -> Reached {
    let mut taking = (0..)
        .zip(workers)
        .filter_map(|(place, worker)| takes(place, worker.as_ref()).then_some(place));
    let Some(first) = taking.next() else {
        return Reached::NoneToCount;
    };
    let mut taking = iter::once(first).chain(taking);
    let reached = number.and_then(|number| taking.nth(number as usize));
    reached.map_or(Reached::NoWorker, Reached::Place)
}

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
    let Some(named) = number(spec, request) else {
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
    let place = match reached(&list.workers[..], named, |place, worker| {
        let met = !narrowed || meets.next_if_eq(&place).is_some();
        met && worker.is_available() && admitted(request, worker)
    }) {
        Reached::NoneToCount => {
            call.unkeyed = true;
            return super::single(engine, request, Some(list), &mut call);
        }
        Reached::NoWorker => return Err(MISSED),
        Reached::Place(place) => place,
    };
    call.admitted = Some(vec![place]);
    // The call's one decision: behind a worker it cannot send to, no
    // second one is made.
    call.spent = ASKS - 1;
    held(super::single(engine, request, Some(list), &mut call))
}

/// The number `request` names its worker by, read once in its headers
/// under the target source its binding has in `spec`: `None` where it
/// names none, and `Some(None)` where its text is no number
/// ([`KeySources::target`](crate::document::KeySources::target)).
fn number(spec: &AppliedSpec, request: &PlaceRequest<'_>) -> Option<Option<u32>> {
    let sources = spec.key_sources_for(request.model, request.phase());
    let headers = request.headers;
    sources.and_then(|sources| sources.target(|name| headers.header(name)))
}

/// The places of `list` whose worker `request` can be sent to at this
/// read, ascending. They are looked for among the places the count reads
/// on the side (`among`, and every place for `None`): those the list's
/// host states, or those of them whose worker meets the request's hard
/// demands. A place a standing veto of that host masks (`masked`,
/// ascending) is none of them. The workers of the rest that pass `state`
/// and that the route's rule takes are. The rule is asked of a worker that
/// passes at a place no veto masks, and of no other.
fn taking(
    request: &PlaceRequest<'_>,
    list: LentList<'_>,
    (among, masked): (Option<&[u32]>, &[u32]),
    state: impl Fn(&dyn Worker) -> bool,
) -> Vec<u32> {
    let takes = |place: u32, worker: &Arc<dyn Worker>| {
        let worker = worker.as_ref();
        masked.binary_search(&place).is_err() && state(worker) && admitted(request, worker)
    };
    match among {
        None => (0..)
            .zip(&list.workers[..])
            .filter_map(|(place, worker)| takes(place, worker).then_some(place))
            .collect(),
        // A place past the list's end names no worker.
        Some(among) => among
            .iter()
            .copied()
            .filter(|&place| {
                let worker = list.workers.get(place as usize);
                worker.is_some_and(|worker| takes(place, worker))
            })
            .collect(),
    }
}

/// One call for a pair over `lists`, under a document one of whose
/// bindings names a target source. A pair that names no worker is placed
/// as under any other document ([`super::pair::place`]): its binding has no such
/// source, it carries no text under the header, or its prefill side is no
/// plain hash ring's
/// ([`Engine::names_worker_on`](crate::engine::framework::Engine::names_worker_on)).
///
/// One that names a worker has its two lists read once: each worker by its
/// own test of being available, and the route's rule asked of each that
/// is, at the places the lists' host states. A worker that fails what its
/// side's role is held to of the request's hard demands
/// ([`Demands::held_by`](crate::engine::Demands::held_by)) is passed over
/// ahead of both, by the judgement the decision holds each side to: it
/// takes no number, as none takes a single request's ([`single`]). Nor
/// does a worker at a place a standing veto of the lists' host masks
/// ([`PairLists::prefill_stalled`], [`PairLists::decode_stalled`]): a host
/// states there a veto its worker objects do not carry, and the count
/// reads the mask wherever it reads a worker's own test. The
/// prefill's number counts
/// the prefills among them that pair with a decode among them
/// ([`PartnerClasses::pairs`](crate::engine::PartnerClasses::pairs)), and
/// those that pair with one first, alone, where there is one
/// ([`PartnerClasses::pairs_first`](crate::engine::PartnerClasses::pairs_first)).
/// Where the decode side is the ring's too, the decode's number counts the
/// partners so counted of the prefill reached, and the call's one decision
/// is held to the two places. It books the decode with its prefill under
/// every document ([`PlaceRequest::decode_again`]), and no ask follows it.
/// Where it is not, the decision is held to the
/// prefill's place, and the call asks again as any pair's does
/// ([`super::pair::asks`]): a rule the request states is then asked of
/// that one place, and of the decodes.
///
/// Where no decode can be sent to, a decode over its overload thresholds
/// alone counts as a prefill's partner, and the decode's number is not
/// read: the binding's rule for decodes that are all over them takes the
/// decode by load. That count reads the decodes' whole mask too
/// ([`PairLists::decode_stalled`], the places of
/// [`PairLists::decode_held`] among them): the lift of the overload veto
/// lifts no standing veto of the host. Under a binding that sheds such a
/// pool no decode counts
/// so, and then no prefill is counted. Where no prefill is counted at all,
/// the pair is placed as one that names no worker
/// ([`super::pair::place`]), and a pool its binding sheds is shed there as
/// it is for any pair. A side masked whole is placed so too: that decision
/// lifts a mask no hold stands on as its last resort.
/// Unlike a single request it keeps its routing value
/// there: under the plain ring that value keeps no prefill of a pair over a
/// veto.
///
/// The gateway's own pair selection counts so: the candidates its
/// prefill's policy is handed
/// (`model_gateway/src/routers/common/placement.rs:2723-2728`,
/// `:2754-2771`, `:2818`), the open partners of its pick (`:2884-2887`),
/// and a decode pool all over its thresholds, where its shedding is off
/// (`:2648`, `:2653-2655`, `:2909-2912`), on main `4b41f39a7`. It
/// narrows both pools of a pinned request to its wire ahead of that count
/// (`:2635-2640`).
///
/// # Errors
///
/// A fault ([`DecisionRefused::Faulted`](crate::engine::decision_engine::DecisionRefused::Faulted)),
/// counted as a decision's is, where the count's read of a list's records
/// panicked: a list lent with facts of another count than its workers is
/// refused where its value is made.
///
/// The pair's shed with no endpoint, under [`TARGET_WORKER`], where a
/// prefill was counted: the header's text is no number, fewer workers than
/// the number asks were counted on a side, or the decision held to the
/// places named no pair that can be sent to ([`held`]). Prefills at the
/// caller's admission limit answer [`Unplaced::PrefillsFull`] as for any
/// pair: the named prefill is waited for. Any other answer of
/// [`super::pair::place`] stands as it is.
#[inline(never)]
pub(super) fn pair<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
) -> Result<Placement<'a>, Unplaced> {
    let mut call = Call::default();
    let spec = &engine.document.spec;
    // The sides are looked up for a request that names a worker alone.
    let sides = |named| (named, spec.engine().names_worker_on(request.model));
    let Some((named, [true, decode_named])) = number(spec, request).map(sides) else {
        return super::pair::place(engine, request, lists, &mut call);
    };
    // The places of each side whose worker meets what its role is held to
    // of the request's hard demands, among the places its host states,
    // where a demand leaves a worker out. A request that states none reads
    // no record. The reads may be the lists' first, and so the fill of
    // their cells: they stand under the engine's catch, as a decision's do.
    let mut meeting = [Vec::new(), Vec::new()];
    let mut stated = [lists.prefill_places, lists.decode_places];
    if !request.demands.is_none() {
        let [prefills, decodes] = &mut meeting;
        let judged = [
            (lists.prefill, Role::Prefill, stated[0], prefills),
            (lists.decode, Role::Decode, stated[1], decodes),
        ];
        let narrowed = engine
            .caught(|| {
                Ok(judged.map(|(list, role, among, kept)| {
                    let records = engine.members_of(list).records(spec, &engine.clock);
                    let demands = request.demands.held_by(role);
                    records.meeting(demands, among, kept).is_some()
                }))
            })
            .map_err(Unplaced::Refused)?;
        for (side, narrowed) in narrowed.into_iter().enumerate() {
            if narrowed {
                stated[side] = Some(&meeting[side]);
            }
        }
    }
    // A place a standing veto of the lists' host masks takes no number
    // either, among the available workers and among the decodes over
    // their thresholds alone: the mask says of a place what a worker
    // object's own test says where the host's objects carry its vetoes.
    // A place a hold of the host stands on is a place of the mask.
    let prefill_places = (stated[0], lists.prefill_stalled);
    let decode_places = (stated[1], lists.decode_stalled);
    let available = |worker: &dyn Worker| worker.is_available();
    let mut decodes = taking(request, lists.decode, decode_places, available);
    let lifted = decodes.is_empty() && spec.engine().steers_all_vetoed(request.model);
    if lifted {
        let but_overloaded =
            |worker: &dyn Worker| worker.routing_state().eligible_ignoring_overload();
        decodes = taking(request, lists.decode, decode_places, but_overloaded);
    }
    let classes = lists.classes;
    let partner = |prefill: u32, decode: u32, first: bool| {
        let (prefill, decode) = (prefill as usize, decode as usize);
        if first {
            classes.pairs_first(prefill, decode)
        } else {
            classes.pairs(prefill, decode)
        }
    };
    let partnered = |prefill: u32, first: bool| {
        decodes
            .iter()
            .any(|&decode| partner(prefill, decode, first))
    };
    let prefills = taking(request, lists.prefill, prefill_places, available);
    // A first partner is looked for where a prefill's class states one:
    // under classes that state none, no decode is read for it.
    let firsts = classes.first();
    let states_first = |prefill: u32| {
        let class = classes.prefills().get(prefill as usize);
        class.is_some_and(|&class| !firsts.of(class).is_empty())
    };
    let first = prefills
        .iter()
        .any(|&prefill| states_first(prefill) && partnered(prefill, true));
    let mut counted = prefills
        .iter()
        .copied()
        .filter(|&prefill| partnered(prefill, first));
    let Some(lead) = counted.next() else {
        return super::pair::place(engine, request, lists, &mut call);
    };
    let mut counted = iter::once(lead).chain(counted);
    let prefill = named
        .and_then(|number| counted.nth(number as usize))
        .ok_or(MISSED)?;
    let decode = if decode_named && !lifted {
        let mut partners = decodes
            .iter()
            .filter(|&&decode| partner(prefill, decode, first));
        let reached = named.and_then(|number| partners.nth(number as usize));
        Some([*reached.ok_or(MISSED)?])
    } else {
        None
    };
    let prefill = [prefill];
    // The pair's lists as the call's asks read them: the prefill's place,
    // and the decode's where the request names one.
    let held_to = PairLists {
        prefill_places: Some(&prefill),
        decode_places: decode
            .as_ref()
            .map_or(lists.decode_places, |decode| Some(decode)),
        ..*lists
    };
    let lent = WorkerLists::Pair(held_to);
    // The pair held on both sides is the call's one decision: it books the
    // decode with its prefill, and behind a worker it cannot send to no
    // second one is made. Held to its prefill alone, the call asks again
    // as any pair's does, and its rule is asked of that one place.
    call.named_pair = decode.is_some();
    held(super::pair::asks(
        engine, request, &held_to, &lent, &mut call,
    ))
}
