//! A request that names its worker: the one worker it goes to, by a number
//! its caller's client states. A single request names its one worker, a
//! pair one worker a side, and an encode request the workers of its pair
//! as a pair does: its number names no encoder.
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
//! with no prefill to count leaves its hold with its routing value, which
//! keeps no prefill of a pair over a veto.
//!
//! Where the decode's side alone is the ring's ([`decode_alone`]), the
//! prefill is its profile's pick, known inside a decision alone. The call
//! then lends the places of the decodes that can be sent to, and each of
//! its decisions counts the decode's number among the partners there of
//! the prefill it picks ([`Constraints::target_worker`]), and books that
//! decode with its prefill.
//!
//! **An encode request** ([`encode`]). Its pair is counted as a pair's
//! is, by the same lines, and its call then asks as an encode request's
//! call does, lent the three lists with the pair held to the places the
//! number reached: three decisions at most, each held so. The count asks
//! no encoder: which workers an encoder serves is the route's rule to
//! say, and under no rule the number can reach a prefill no encoder
//! serves beside one that an encoder does, whose request is then shed
//! under [`TARGET_WORKER`]. The number names no encoder either: where the
//! count reaches a pair, what the call answers for its encoders stands as
//! it is. A request that names a worker and that the encode pass is not
//! laid for is its pair's alone, and answered as a pair is, by the pair's
//! own entry. One that names no worker is placed as under any other
//! document: a target source its binding names costs it one lookup of
//! its header. Where the decode's side alone is the ring's, a request
//! the pass is laid for that names a worker is shed under that name for
//! the decode's side: an encode request's decision counts no number.
//!
//! **A controller's served decision** counts by the same functions, over
//! the lists its worker objects lend: a single request's number by
//! [`reached`], a pair's by [`reached_pair`], and the decodes a pair's
//! decision counts its decode's number among by [`opened`]. Its tests are
//! its own: a place the asking gateway may take, that none of its standing
//! vetoes holds back, whose worker is available by its own test. Its frame
//! states the number, which the asking host read by the same source. The
//! route's rule is that gateway's own code, and is not asked there. It
//! reads no number for an encode request's frame: the two doors differ
//! for such a request.
//!
//! [`Constraints::places`]: crate::engine::view::Constraints::places
//! [`Constraints::target_worker`]: crate::engine::view::Constraints::target_worker
//! [`KeySourceSpec::target`]: crate::document::spec::KeySourceSpec::target

use std::{iter, sync::Arc};

use smg_worker::identity::basic::Worker;

use super::{admitted, hold_to_rule, Call, PlaceRequest, Unplaced, WorkerRule, ASKS};
#[cfg(feature = "controller")]
use crate::engine::decision::{Decision, Shed};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{PairSide, ShedReason, ADMITS, PLACES, TARGET_WORKER},
        framework::{EncodeItems, ShedOver},
        in_process::{pair::shed_over, Loaded},
        list_cell::LentList,
        view::Role,
        Demands, EncodeLists, PairLists, WorkerLists,
    },
    tracking::Placement,
};

/// The shed of a request whose named worker cannot take it: the request
/// has no worker, and moves to no other. `side` is the side of a pair the
/// worker was missed on: the side whose count the number reached nobody of
/// ([`ReachedPair::NoWorker`]), the side the decision held to the worker's
/// place was left with nobody on, or the side the call's last ask could
/// not send to ([`held`]).
const fn missed(side: Option<PairSide>) -> Unplaced {
    Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: TARGET_WORKER,
        retry_after_secs: 1,
        side,
    }
}

/// [`missed`] on no side: a single request's.
const MISSED: Unplaced = missed(None);

/// What a decision held to a named worker answers, as the request's
/// answer ([`held_miss`]), behind a call whose last ask could not send to
/// `unsent`.
fn held<'a>(
    placed: Result<Placement<'a>, Unplaced>,
    unsent: Option<PairSide>,
) -> Result<Placement<'a>, Unplaced> {
    placed.map_err(|unplaced| held_miss(unplaced, unsent))
}

/// What a decision held to a named worker answered in a placement's stead
/// (`unplaced`), as the request's answer, by the one mapping of every
/// call: where it named no worker because of the worker (no worker for
/// the call's asks, its own shed with no endpoint, or for workers over
/// their thresholds), the request is shed under [`TARGET_WORKER`]. The
/// shed is on the side that decision's shed names: none where the route's
/// rule took no worker of a list behind the call's first ask
/// ([`again`](super::pair::again)). Where the call had no worker for its
/// asks it is on `unsent`, the side its last ask could not send to
/// ([`Call::unsent`]): none for a single request, and none for an encode
/// request the pass is laid for, whose walk writes none. Any other answer
/// stands as it is.
fn held_miss(unplaced: Unplaced, unsent: Option<PairSide>) -> Unplaced {
    match unplaced {
        Unplaced::NoWorkers => missed(unsent),
        Unplaced::Shed {
            reason: ShedReason::NoEndpoints | ShedReason::Saturated,
            side,
            ..
        } => missed(side),
        other => other,
    }
}

/// Whether the call of an encode request held to its pair was answered
/// `unplaced` for its encoders or over its lists whole, and not for that
/// pair: the number names no encoder, so such an answer is no miss of the
/// worker it names.
///
/// - No worker for the call's asks, where its last ask was given up for
///   an encoder alone ([`Call::past_encoder`]): that ask's pair could be
///   sent to.
/// - A shed as saturated that its decision made over the encoders, by
///   that decision's own word ([`shed_over`]).
/// - A shed with no endpoint that names no side of the pair, under any
///   name but the places' ([`PLACES`]) and the rule's ([`ADMITS`]). A
///   pair's decision names the side it found nobody on. The encode pass
///   names none: it says its gaps of the encoders and of the pairs'
///   membership over the lists whole, whatever places the pair is held
///   to, and an item's own answer is its encoders'.
/// - A shed under the places' name that its decision made for the
///   encoders, by the same word: places its host states for them hold
///   the request off them.
///
/// The pass's refusal of a place the pair is held to bears the places'
/// name too, with no word of the encoders: it is the pair's. So is the
/// rule's refusal of the worker the request names. Both names are the
/// engine's own, and no document's stage runs under one
/// ([`KEPT`](crate::engine::decision::KEPT)): a stage that bears one all
/// the same is read as the engine's.
fn for_encoders(unplaced: &Unplaced, call: &Call) -> bool {
    // The decision's own word of a shed it made over the encoders, read
    // behind that shed and ahead of anything else the call asks.
    let over_encoders = || shed_over() == Some(ShedOver::Encoders);
    match *unplaced {
        Unplaced::NoWorkers => call.past_encoder,
        Unplaced::Shed {
            reason: ShedReason::Saturated,
            ..
        } => over_encoders(),
        Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by,
            side: None,
            ..
        } => match emptied_by {
            PLACES => over_encoders(),
            ADMITS => false,
            _ => true,
        },
        _ => false,
    }
}

/// [`held`] for a controller's served decision, the one decision it held
/// to a named worker's place: the decision's own, but that a shed with no
/// endpoint, or for workers over their thresholds, is the request's shed
/// under [`TARGET_WORKER`]. The decision named no worker because of that
/// worker, and the request moves to no other. The shed keeps the side and
/// the trace of the decision that ran: the trace names the profile that
/// decision was made under, as any shed made once a profile is resolved
/// does.
#[cfg(feature = "controller")]
pub(crate) fn held_answer(decision: Decision) -> Decision {
    match decision {
        Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints | ShedReason::Saturated,
            side,
            trace,
            ..
        }) => Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: TARGET_WORKER,
            retry_after_secs: 1,
            wait_allowed: false,
            side,
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
    held(super::single(engine, request, Some(list), &mut call), None)
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

/// What holds a pair to the workers its request names: the number
/// ([`number`]), and which sides of the pair read it, the prefill's and
/// then the decode's
/// ([`Engine::names_worker_on`](crate::engine::framework::Engine::names_worker_on)).
/// A request that names no worker has none ([`held_sides`]): its binding
/// has no target source, it carries no text under the header, or neither
/// side of its pair is a plain hash ring's. The sides are looked up for a
/// request that carries the header alone.
type Held = (Option<u32>, [bool; 2]);

/// What holds the pair of `request` to the workers it names ([`Held`]):
/// one read of the header, and one lookup of the sides that read it.
#[inline]
fn held_sides(spec: &AppliedSpec, request: &PlaceRequest<'_>) -> Option<Held> {
    let named = number(spec, request)?;
    let sides = spec.engine().names_worker_on(request.model);
    (sides != [false; 2]).then_some((named, sides))
}

/// What a pair's count reads of one side beside its list: the places it
/// counts among, every place for `None`, and the places a standing veto of
/// the lists' host masks. Both ascending.
pub(crate) type SidePlaces<'a> = (Option<&'a [u32]>, &'a [u32]);

/// Whether `found` takes one of the places of `list` whose worker a
/// request can be sent to at this read: the one walk of a pair's count.
/// The places are handed to `found` ascending, as far as the first it
/// takes. No worker behind that place is read, and the route's rule is
/// asked of none there.
///
/// The places are looked for among those the count reads on the side
/// (`among`, and every place for `None`): those the list's host states, or
/// those of them whose worker meets the request's hard demands. A place a
/// standing veto of that host masks (`masked`, ascending) is none of them.
/// The workers of the rest that pass `state` and that the route's rule
/// takes are (`admits`, and every worker for `None`). The rule is asked of
/// a worker that passes at a place no veto masks, and of no other.
fn any_taking(
    admits: Option<WorkerRule<'_>>,
    list: LentList<'_>,
    (among, masked): SidePlaces<'_>,
    state: impl Fn(&dyn Worker) -> bool,
    mut found: impl FnMut(u32) -> bool,
) -> bool {
    let mut ends = |place: u32, worker: &Arc<dyn Worker>| {
        let worker = worker.as_ref();
        let admitted = || admits.is_none_or(|admits| admits(worker));
        let takes = masked.binary_search(&place).is_err() && state(worker) && admitted();
        takes && found(place)
    };
    match among {
        None => (0..)
            .zip(&list.workers[..])
            .any(|(place, worker)| ends(place, worker)),
        // A place past the list's end names no worker.
        Some(among) => among.iter().any(|&place| {
            let worker = list.workers.get(place as usize);
            worker.is_some_and(|worker| ends(place, worker))
        }),
    }
}

/// The places of `list` whose worker a request can be sent to at this
/// read, ascending: every place of the count's walk ([`any_taking`]),
/// which ends at none of them. They are kept in a list made once for the
/// places the count reads on the side, so that no place grows it.
fn taking(
    admits: Option<WorkerRule<'_>>,
    list: LentList<'_>,
    counted: SidePlaces<'_>,
    state: impl Fn(&dyn Worker) -> bool,
) -> Vec<u32> {
    let mut places = Vec::with_capacity(counted.0.map_or(list.workers.len(), <[u32]>::len));
    any_taking(admits, list, counted, state, |place| {
        places.push(place);
        false
    });
    places
}

/// One call for a pair over `lists`, under a document one of whose
/// bindings names a target source. A pair that names no worker is placed
/// as under any other document ([`super::pair::place`]): its binding has no such
/// source, it carries no text under the header, or neither of its sides
/// is a plain hash ring's
/// ([`Engine::names_worker_on`](crate::engine::framework::Engine::names_worker_on)).
/// One whose decode side alone is such a ring's has its decode counted
/// inside each decision ([`decode_alone`]).
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
/// The count is one function ([`reached_pair`]), which a controller's
/// served decision asks too, over its own two lists.
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
/// `read` is what the request's call read of its header ahead of this
/// function ([`held_sides`]): an encode request's that the encode pass is
/// not laid for, read under the engine's catch ([`encode`]). A pair's own
/// arm states none, and the header is read here.
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
/// places named no pair that can be sent to ([`held`]). The shed names the
/// side the worker was missed on ([`missed`]): the gateway's own pair
/// selection names the role of the policy that found its target nowhere,
/// the prefill's and then the decode's
/// (`model_gateway/src/routers/common/placement.rs:3044`, `:3090`, with
/// the policy's miss at
/// `model_gateway/src/policies/consistent_hashing.rs:187-193`, on main
/// `d43c7b2a9`). Where the decision held to the places says that its
/// prefills are full, the answer is [`Unplaced::PrefillsFull`] as for any
/// pair's decision: the named prefill is waited for. Any other answer of
/// [`super::pair::place`] stands as it is.
#[inline(never)]
pub(super) fn pair<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    read: Option<Held>,
) -> Result<Placement<'a>, Unplaced> {
    let Some(held) = read.or_else(|| held_sides(&engine.document.spec, request)) else {
        return super::pair::place(engine, request, lists, &mut Call::default());
    };
    named(engine, request, lists, held, Alone)
}

/// One call for an encode request over its three `lists`, under a document
/// one of whose bindings names a target source. The number names the
/// pair's sides and no encoder: an item's encoder is picked for its key,
/// among those that serve the prefill its pair is held to.
///
/// A request that names no worker ([`held_sides`]) is placed as under any
/// other document ([`super::encode::three`]), whatever it states of its
/// items. A target source its binding names costs it one lookup of the
/// header, and changes nothing else of its answer.
///
/// One that names a worker and that the encode pass is not laid for
/// ([`super::pair::laid`]) is its pair's alone: it states no items
/// ([`PlaceRequest::encode`]), or none by an empty list, and holds its
/// decode to no images. It is answered as a pair that names that worker
/// is, by the pair's own arm over the pair's two lists ([`pair`]), which
/// is told what was read here. Its entry is the pair's: what the entry
/// for three lists alone declines does not apply to it, and it is
/// observed as a pair's request is.
///
/// For any other the pair is counted as a pair's is, by the same lines
/// ([`named`]): the sides that read the number are the pair's, and the
/// number counts the prefills the request can be sent to that pair with a
/// decode it can ([`reached_pair`]). What an encode request's call does
/// otherwise stands in one place ([`Beside`]). It then asks as an encode
/// request's does ([`super::encode::asks`]), lent the three lists with
/// the pair held to the places the number reached, and a miss is the
/// pair's shed under [`TARGET_WORKER`] ([`held_miss`]). Where no prefill
/// is counted, it is placed as any encode request
/// ([`super::encode::place`]).
///
/// What the call answers for its encoders, or over its lists whole,
/// stands as it is ([`for_encoders`]): a shed its decision made so, and
/// its answer of no worker where its last ask was given up for an encoder
/// alone. Where the rule is asked of the lists and refuses the worker the
/// request names, that refusal is the pair's, whatever the ask before it
/// was given up for.
/// A pair held on both sides is asked again behind an ask given up for an
/// encoder alone, held as it was, and its next decision is told the
/// encoder that could not be sent to: behind a side that cannot be sent
/// to it has no worker, as in a pair's own call. A pair held on its
/// prefill alone asks again as any pair's does, and its decode is each
/// decision's own pick.
///
/// The count asks no encoder. Which prefills an encoder serves is the
/// route's rule to say ([`PlaceRequest::admits`]), as the gateway's own
/// selection is told it by its stage's filter: the count asks that rule of
/// each worker it counts. Under a request that states no rule the number
/// counts as a pair's does. It can reach a prefill no encoder serves
/// beside one that an encoder does, or a decode that takes no images
/// beside one that does, for a request that holds its decode to them: the
/// decision held to that place then places nobody, and the request is
/// shed under [`TARGET_WORKER`]. It takes the worker it names or none.
///
/// Each side of the pair is counted among the workers that meet its part
/// of the request's hard demands, as the request's decision holds that
/// side. Where the request carries an item that is the demands less the
/// media's parts, which the encoders meet
/// ([`Demands::beside_encoders`]), and the whole of them where it
/// carries none.
///
/// A binding whose decode side alone is a plain hash ring's has its number
/// counted inside a pair's decision ([`decode_alone`]). An encode
/// request's decision counts none: a request the pass is laid for that
/// names a worker is shed under [`TARGET_WORKER`], for the decode's side,
/// and placed nowhere.
///
/// The header's read stands under the engine's catch here, for every
/// request with three lists, as the read of such a request's headers did
/// while its decision alone made it: a reader that panics costs its
/// request alone, and is counted as a decision's fault is. The count's
/// own reads of a worker are a pair's ([`reached_pair`]).
///
/// # Errors
///
/// Those of [`pair`], with the answers of [`super::encode::place`] where
/// it names those of [`super::pair::place`]. The count runs ahead of the
/// request's decision, as a pair's does. So a miss is said ahead of
/// whatever that decision would answer: what it would decline, and what
/// it would say of the request's encoders.
#[inline(never)]
pub(super) fn encode<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &EncodeLists<'_>,
) -> Result<Placement<'a>, Unplaced> {
    let spec = &engine.document.spec;
    let read = || Ok(held_sides(spec, request));
    let Some(held) = engine.caught(read).map_err(Unplaced::Refused)? else {
        return super::encode::three(engine, request, lists);
    };
    match request.encode.filter(|_| super::pair::laid(request)) {
        Some(items) => {
            let items = super::encode::ringed(request, items);
            named(engine, request, &lists.pair, held, (lists, items))
        }
        // No item stated, or none the pass is laid for: the request is
        // answered as a pair that names its worker is, by the pair's own
        // arm, which is told what was read here.
        None => pair(engine, request, &lists.pair, Some(held)),
    }
}

/// What stands beside the pair's two lists in the call of a request that
/// names its worker, and with it what that call does at the five places
/// where a pair's own call and an encode request's differ. The lines the
/// two calls share are one function over it ([`named`]).
trait Beside<'a>: Copy {
    /// What each side of the pair is held to of the request's hard
    /// demands, as the request's decision holds that side.
    fn demands(self, request: &PlaceRequest<'a>) -> Demands;

    /// The call of a request whose pair has no prefill to count, over the
    /// pair's `lists` as they were handed.
    fn unnamed(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        lists: &PairLists<'_>,
        call: &mut Call,
    ) -> Result<Placement<'a>, Unplaced>;

    /// The call where the decode's side alone reads the number, behind
    /// what the count reads of each side (`counted`).
    fn decode_alone(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        lists: &PairLists<'_>,
        counted: [SidePlaces<'_>; 2],
        named: Option<u32>,
        call: Call,
    ) -> Result<Placement<'a>, Unplaced>;

    /// The call's asks with its pair held to the places the count reached
    /// (`held_to`).
    fn asks(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        held_to: PairLists<'_>,
        call: &mut Call,
    ) -> Result<Placement<'a>, Unplaced>;

    /// What those asks answered in a placement's stead (`unplaced`), as
    /// the request's answer ([`held_miss`]), behind the asks of `call`.
    /// It is handed no placement: a placement is several hundred bytes,
    /// and every hand of it is a copy.
    fn held(self, unplaced: Unplaced, call: &Call) -> Unplaced;
}

/// A pair's own call: nothing stands beside its two lists. The call of a
/// request with three lists that the encode pass is not laid for is a
/// pair's own too ([`encode`]).
#[derive(Clone, Copy)]
struct Alone;

impl<'a> Beside<'a> for Alone {
    fn demands(self, request: &PlaceRequest<'a>) -> Demands {
        request.demands
    }

    fn unnamed(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        lists: &PairLists<'_>,
        call: &mut Call,
    ) -> Result<Placement<'a>, Unplaced> {
        super::pair::place(engine, request, lists, call)
    }

    fn decode_alone(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        lists: &PairLists<'_>,
        counted: [SidePlaces<'_>; 2],
        named: Option<u32>,
        call: Call,
    ) -> Result<Placement<'a>, Unplaced> {
        decode_alone(engine, request, lists, counted, named, call)
    }

    fn asks(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        held_to: PairLists<'_>,
        call: &mut Call,
    ) -> Result<Placement<'a>, Unplaced> {
        let lent = WorkerLists::Pair(held_to);
        super::pair::asks(engine, request, &held_to, &lent, call)
    }

    fn held(self, unplaced: Unplaced, call: &Call) -> Unplaced {
        held_miss(unplaced, call.unsent)
    }
}

/// What stands beside the pair's two lists in an encode request's call:
/// its three lists, and the items as its call lends them
/// ([`super::encode::ringed`]).
type Three<'l, 'i> = (&'l EncodeLists<'l>, EncodeItems<'i>);

impl<'a> Beside<'a> for Three<'_, '_> {
    /// The hard demands less the media's parts where the request carries
    /// an item: its encoders meet those
    /// ([`Demands::beside_encoders`]). A request that carries no item
    /// takes no encoder, and holds its pair to the whole.
    fn demands(self, request: &PlaceRequest<'a>) -> Demands {
        let (_, items) = self;
        if items.keys.is_empty() {
            request.demands
        } else {
            request.demands.beside_encoders()
        }
    }

    fn unnamed(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        _: &PairLists<'_>,
        call: &mut Call,
    ) -> Result<Placement<'a>, Unplaced> {
        let (lists, items) = self;
        super::encode::place(engine, request, lists, items, call)
    }

    /// An encode request's decision counts no number of its decode side:
    /// the request is refused by name, for that side, and placed nowhere.
    fn decode_alone(
        self,
        _: Loaded<'static, '_>,
        _: &PlaceRequest<'a>,
        _: &PairLists<'_>,
        _: [SidePlaces<'_>; 2],
        _: Option<u32>,
        _: Call,
    ) -> Result<Placement<'a>, Unplaced> {
        Err(missed(Some(PairSide::Decode)))
    }

    /// An encode request's own asks ([`super::encode::asks`]), lent its
    /// three lists with the pair held: the decision places each item's
    /// encoder beside the pair it is held to.
    fn asks(
        self,
        engine: Loaded<'static, '_>,
        request: &PlaceRequest<'a>,
        held_to: PairLists<'_>,
        call: &mut Call,
    ) -> Result<Placement<'a>, Unplaced> {
        let (lists, items) = self;
        let held_to = EncodeLists {
            pair: held_to,
            ..*lists
        };
        let lent = WorkerLists::Encode(held_to);
        super::encode::asks(engine, request, &held_to, items, &lent, call)
    }

    /// The pair's mapping, less what the call answered for its encoders
    /// ([`for_encoders`]): the number names none of them, so that answer
    /// stands as it is. An encode request's walk writes no side
    /// ([`Call::unsent`]), so its miss behind asks that sent to nobody
    /// names none.
    fn held(self, unplaced: Unplaced, call: &Call) -> Unplaced {
        if for_encoders(&unplaced, call) {
            unplaced
        } else {
            held_miss(unplaced, call.unsent)
        }
    }
}

/// The lines of [`pair`] and of [`encode`] for a request that names its
/// worker (`held`: its number, and the sides that read it), over the
/// pair's two `lists`: the demands, the masks and one count. `beside`
/// says whose call it is ([`Beside`]): a pair's own ([`Alone`]), or an
/// encode request's with its three lists and its items ([`Three`]).
fn named<'a, B: Beside<'a>>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    (named, [prefill_named, decode_named]): Held,
    beside: B,
) -> Result<Placement<'a>, Unplaced> {
    let call = Call::default();
    let spec = &engine.document.spec;
    // What each side is held to of the request's hard demands, as the
    // request's decision holds it.
    let demands = beside.demands(request);
    // The places of each side whose worker meets what its role is held to
    // of the request's hard demands, among the places its host states,
    // where a demand leaves a worker out. A request that states none reads
    // no record. The reads may be the lists' first, and so the fill of
    // their cells: they stand under the engine's catch, as a decision's do.
    // A side whose role is held to no part of the demands keeps every
    // place. Its list's value is read all the same, so a list lent wrong
    // is refused where this read makes its value, as before, and its
    // records are not walked.
    let mut meeting = [Vec::new(), Vec::new()];
    let mut stated = [lists.prefill_places, lists.decode_places];
    if !demands.is_none() {
        let [prefills, decodes] = &mut meeting;
        let judged = [
            (lists.prefill, Role::Prefill, stated[0], prefills),
            (lists.decode, Role::Decode, stated[1], decodes),
        ];
        let narrowed = engine
            .caught(|| {
                Ok(judged.map(|(list, role, among, kept)| {
                    let records = engine.members_of(list).records(spec, &engine.clock);
                    let demands = demands.held_by(role);
                    if demands.is_none() {
                        return false;
                    }
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
    let counted = [prefill_places, decode_places];
    if !prefill_named {
        return beside.decode_alone(engine, request, lists, counted, named, call);
    }
    let steers = || spec.engine().steers_all_vetoed(request.model);
    let named = (named, decode_named);
    let reached = reached_pair(lists, counted, named, request.admits, steers);
    held_pair(engine, request, lists, beside, reached, call)
}

/// Where the number a pair names its workers by reaches
/// ([`reached_pair`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReachedPair {
    /// No prefill is counted: there is nobody to name, and the pair leaves
    /// its hold.
    NoneToCount,
    /// A prefill is counted and the number reaches no worker on the side
    /// named: its text is no number, or fewer workers are counted there.
    /// The prefills are counted first.
    NoWorker(PairSide),
    /// The place of the prefill the number reaches, and of the decode it
    /// reaches among that prefill's partners. No decode where the
    /// decode's number is not read: its side is no plain ring's, or a
    /// decode over its thresholds alone counted as a partner.
    Places(u32, Option<u32>),
}

/// The count of a pair that names its worker on its prefill's side, the
/// one count of both doors: the one call in a host's own process
/// ([`pair`]) and a controller's served decision hand it their own two
/// lists, and what each states beside them.
///
/// `counted` is what the count reads of each side, the prefill's and then
/// the decode's. `named` is the number, `None` for a text that is no
/// number, and whether the decode's side reads it too. `admits` is the
/// route's rule, asked as [`taking`] asks it: `None` for a caller that
/// has none, as a controller is. `steers` is whether the binding steers a
/// decode pool that is all over its thresholds, asked where no decode can
/// be sent to and of no other pair.
///
/// The prefill's number counts the prefills that can be sent to and pair
/// with a decode that can, and those that pair with one first, alone,
/// where there is one. The decode's number counts the partners so counted
/// of the prefill reached. Where no decode can be sent to and the binding
/// steers, a decode over its thresholds alone counts as a partner, and
/// the decode's number is not read.
pub(crate) fn reached_pair(
    lists: &PairLists<'_>,
    [prefill_places, decode_places]: [SidePlaces<'_>; 2],
    (named, decode_named): (Option<u32>, bool),
    admits: Option<WorkerRule<'_>>,
    steers: impl FnOnce() -> bool,
) -> ReachedPair {
    let available = |worker: &dyn Worker| worker.is_available();
    let mut decodes = taking(admits, lists.decode, decode_places, available);
    let lifted = decodes.is_empty() && steers();
    if lifted {
        let but_overloaded =
            |worker: &dyn Worker| worker.routing_state().eligible_ignoring_overload();
        decodes = taking(admits, lists.decode, decode_places, but_overloaded);
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
    let prefills = taking(admits, lists.prefill, prefill_places, available);
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
        return ReachedPair::NoneToCount;
    };
    let mut counted = iter::once(lead).chain(counted);
    let Some(prefill) = named.and_then(|number| counted.nth(number as usize)) else {
        return ReachedPair::NoWorker(PairSide::Prefill);
    };
    let decode = if decode_named && !lifted {
        let mut partners = decodes
            .iter()
            .filter(|&&decode| partner(prefill, decode, first));
        let reached = named.and_then(|number| partners.nth(number as usize));
        let Some(&decode) = reached else {
            return ReachedPair::NoWorker(PairSide::Decode);
        };
        Some(decode)
    } else {
        None
    };
    ReachedPair::Places(prefill, decode)
}

/// The asks of a pair's call behind its count ([`reached_pair`]), held to
/// the places the count reached: the prefill's, and the decode's where
/// the count read the decode's number. Where there was no prefill to
/// count, the pair is placed as one that names no worker
/// ([`super::pair::place`]).
///
/// An encode request's call asks as an encode request's does
/// ([`Beside::asks`]), lent its three lists with the pair held to the same
/// places, and each answer of [`super::pair::place`] named here is then
/// the one of [`super::encode::place`]. What it answers for its encoders
/// is no miss of the pair, and stands ([`Beside::held`]).
///
/// # Errors
///
/// The pair's shed with no endpoint, under [`TARGET_WORKER`], where the
/// number reached no worker on a side, and where the decision held to the
/// places named no pair that can be sent to ([`held_miss`]). Any other
/// answer of [`super::pair::place`] stands as it is.
fn held_pair<'a, B: Beside<'a>>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    beside: B,
    reached: ReachedPair,
    mut call: Call,
) -> Result<Placement<'a>, Unplaced> {
    let (prefill, decode) = match reached {
        ReachedPair::NoneToCount => return beside.unnamed(engine, request, lists, &mut call),
        ReachedPair::NoWorker(side) => return Err(missed(Some(side))),
        ReachedPair::Places(prefill, decode) => ([prefill], decode.map(|decode| [decode])),
    };
    // The pair's lists as the call's asks read them: the prefill's place,
    // and the decode's where the request names one.
    let held_to = PairLists {
        prefill_places: Some(&prefill),
        decode_places: decode
            .as_ref()
            .map_or(lists.decode_places, |decode| Some(decode)),
        ..*lists
    };
    // The pair held on both sides stands on one pair of places: its
    // decision books the decode with its prefill, and behind a worker of
    // the pair it cannot send to no second one is made. An encode
    // request's call decides again behind an encoder alone, held as it
    // was. Held to its prefill alone, the call asks again as any pair's
    // does, and its rule is asked of that one place.
    call.named_pair = decode.is_some();
    let placed = beside.asks(engine, request, held_to, &mut call);
    placed.map_err(|unplaced| beside.held(unplaced, &call))
}

/// The call of a pair that names its worker where the decode's side alone
/// reads the header: the prefill is the pick of its own profile, known
/// inside a decision alone, so no place stated ahead says the decode. The
/// decodes are read once here, as [`pair`] reads them, among the places it
/// counts on the decode's side (`counted`, the prefill's side and then the
/// decode's: the places whose worker meets what its role is held to of the
/// request's hard demands, and the mask of its host), and their places are
/// lent beside the lists. Each decision of the call is stated the number
/// ([`Call::named_decode`]) and counts it among those places, over the
/// partners of the prefill it picks: its first partners where the pair is
/// made among those. The call asks again as any pair's does
/// ([`super::pair::asks`]), and each of its asks is lent the same places.
/// Each decision books the decode with its prefill under every document:
/// a decode asked again behind the prefill would count no number.
/// So a rule the request states is asked of both lists here, and of no
/// list behind that: of each decode as it is counted, of the prefills as
/// far as the first that opens the pair, and then of every prefill at the
/// places the caller states, ready or not ([`hold_to_rule`]). A decode the
/// rule refuses as a pick is counted by the call's next ask as it was by
/// its first.
///
/// Where no prefill opens the pair, the header is not read, and the pair
/// is placed as one that names no worker. A prefill opens it where it is
/// read as [`pair`] reads its prefills (the same places, mask, test and
/// rule) and a decode counted here partners it. So no decode that can be
/// sent to opens none: the binding's rule for decodes that are all over
/// their thresholds then takes the decode by load. And a decision's
/// default for a prefill pool its vetoes refuse pairs by load over the
/// decodes at its places: the lent places would hide the others from it.
/// A prefill over its thresholds alone opens the pair too where its
/// profile places among such prefills ([`Engine::lifts_prefills`]). A
/// decision still reaches that default under a number where the fleet
/// changes between this read and its own: it then pairs among the lent
/// decodes.
///
/// The decodes, and whether a prefill opens the pair, are one function's
/// read ([`opened`]), which a controller's served decision asks too.
///
/// The gateway's own pair selection counts the open partners of its
/// prefill policy's pick so
/// (`model_gateway/src/routers/common/placement.rs:2884-2887`,
/// `:2914-2920`), reads no header for a decode pool all over its
/// thresholds (`:2909-2912`), and none where no prefill is open: it
/// answers there as for a pair that names no worker, and pairs by load a
/// pool whose every worker is vetoed (`:2729-2747`, `:2772-2798`). It
/// counts a prefill over its thresholds as open under a policy that
/// weighs load itself, its shedding off (`:2672-2680`). The lines are
/// main `4b41f39a7`'s.
///
/// # Errors
///
/// The pair's shed with no endpoint, under [`TARGET_WORKER`]: the
/// decision's own, where the number reaches no partner of its pick or the
/// decode it reaches is shed, and the call's, where no pair its asks named
/// could be sent to. The shed of a request whose rule takes no prefill:
/// with no endpoint under `admits`, or its first decision's shed as
/// saturated where that stands ([`super::pair::again`]). Any other answer
/// of [`super::pair::place`] stands as it is.
///
/// [`Engine::lifts_prefills`]: crate::engine::framework::Engine::lifts_prefills
#[inline(never)]
fn decode_alone<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    counted: [SidePlaces<'_>; 2],
    named: Option<u32>,
    mut call: Call,
) -> Result<Placement<'a>, Unplaced> {
    let lifts = || engine.document.spec.engine().lifts_prefills(request.model);
    let Some(decodes) = opened(lists, counted, request.admits, lifts) else {
        return super::pair::place(engine, request, lists, &mut call);
    };
    // The rule's prefills, found ahead of the first ask: the asks behind
    // it find no places again, and two asks are the call's.
    let mut prefills = None;
    let stated = lists.prefill_places;
    hold_to_rule(request, Some(lists.prefill), stated, &mut prefills)?;
    if prefills.is_some() {
        call.spent = 1;
    }
    // A text that is no number reaches no decode, where a decision counts.
    call.named_decode = Some(named.unwrap_or(u32::MAX));
    let held_to = PairLists {
        prefill_places: prefills.as_deref().or(stated),
        decode_places: Some(&decodes),
        ..*lists
    };
    let lent = WorkerLists::Pair(held_to);
    let placed = super::pair::asks(engine, request, &held_to, &lent, &mut call);
    // Each pair the asks named held the one decode: the shed is on the
    // side the call's last ask could not send to.
    placed.map_err(|unplaced| match unplaced {
        Unplaced::NoWorkers => missed(call.unsent),
        other => other,
    })
}

/// The decodes a pair's decision counts its decode's number among, where
/// the decode's side alone reads the name: the one read of both doors, the
/// one call in a host's own process ([`decode_alone`]) and a controller's
/// served decision. They are the places of the decodes that can be sent
/// to, among those `counted` states for the decode's side, the second of
/// its two. `admits` is the route's rule, as [`reached_pair`] takes it.
///
/// `None` where no prefill opens the pair, so the number is not read. A
/// prefill opens it where it is read as [`reached_pair`] reads its
/// prefills and one of those decodes partners it. A prefill over its
/// thresholds alone opens it too where its profile places among such
/// prefills: `lifts` says so, and is asked where no available prefill
/// opens the pair and of no other pair.
///
/// The prefills are read in their list's order as far as the first that
/// opens the pair ([`any_taking`]): no prefill behind it is read, and the
/// route's rule is asked of none there. Where none opens the pair, the
/// walk goes on to the last of the prefills' places.
pub(crate) fn opened(
    lists: &PairLists<'_>,
    [prefill_counted, decode_counted]: [SidePlaces<'_>; 2],
    admits: Option<WorkerRule<'_>>,
    lifts: impl FnOnce() -> bool,
) -> Option<Vec<u32>> {
    let available = |worker: &dyn Worker| worker.is_available();
    let decodes = taking(admits, lists.decode, decode_counted, available);
    let classes = lists.classes;
    let opens = |state: &dyn Fn(&dyn Worker) -> bool| {
        let pairs = |prefill: u32, &decode: &u32| classes.pairs(prefill as usize, decode as usize);
        let partnered = |prefill| decodes.iter().any(|decode| pairs(prefill, decode));
        any_taking(admits, lists.prefill, prefill_counted, state, partnered)
    };
    let lifted = |worker: &dyn Worker| worker.routing_state().eligible_ignoring_overload();
    let opened = opens(&available) || (lifts() && opens(&lifted));
    opened.then_some(decodes)
}
