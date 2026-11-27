//! A pair's request placed: a decision over its caller's two lists, and
//! the first of its targets both of whose workers can be sent to. Where no
//! target holds, the call decides again as a single request's does
//! ([`place`]).
//!
//! A pair's target names a prefill by its place in the prefill list and a
//! decode by its place in the decode list ([`PairLists`]). The targets are
//! walked as a single request's are ([`walk`]), over two workers a target:
//! each side is read once, by the one read a single request's worker gets
//! ([`sendable`]): ready, and admitted by the route's own rule. A target
//! one of whose sides cannot be sent to is passed over whole, and the next
//! is asked.
//!
//! Where no target of a call's first decision holds, or that decision
//! sheds the request, as saturated or with no endpoint over lists that
//! pair, and the request
//! states a rule, the rule is asked of the two lists once, the prefills'
//! first, and the call's later decisions are held to the workers it takes
//! on each ([`again`]): they are lent the same two lists with each side's
//! places beside it ([`PairLists::prefill_places`],
//! [`PairLists::decode_places`]). A list the rule takes no worker of ends
//! the call there: behind such prefills the decodes are not asked. The
//! request is then shed by its first decision's shed, where that decision
//! shed it with no endpoint, or as saturated and that shed stands, and
//! else with no endpoint under the rule's name ([`again`]).
//! An encode request that carries items is held by the same routine, over
//! the pair's two of its three lists ([`super::encode`]). One the encode
//! pass is laid for ([`laid`]) is held so behind a shed as saturated
//! alone, as it was before a pair's decision asked its one question, and
//! so is a request that states a hard demand.
//!
//! Which workers pair is the caller's word: the classes its lists carry
//! ([`PartnerClasses`](crate::engine::PartnerClasses)). A decision names no
//! pair across them, and this walk tests no pair again.

use std::{sync::Arc, time::Instant};

use super::{
    decide, hold_to_rule, raise, ready_by, sendable, walk, Call, Made, PlaceRequest, Sent, Sides,
    Unplaced,
};
use crate::{
    engine::{
        decision::{PairSide, ShedReason, Target},
        decision_engine::DecisionRefused,
        framework::ShedOver,
        in_process::{pair::shed_over, Loaded},
        list_cell::LentList,
        prefills_full,
        view::{Phase, WorkerId},
        EncodeLists, PairLists, WorkerLists,
    },
    tracking::Placement,
};

/// The label of the profile that picks a pair's decode side under the
/// decision's document: the decode profile of the binding `model` resolves
/// to, and that binding's one profile where it names no other. The empty
/// label where no binding matches.
pub(super) fn decode_label(engine: Loaded<'_, '_>, model: &str) -> &'static str {
    let binding = engine.document.spec.engine().binding(model, Phase::Prefill);
    let position = binding.map(|binding| binding.decode_profile.unwrap_or(binding.profile));
    position.map_or("", |position| engine.label_at(position))
}

/// One ask of `call` for a pair over `lists`, the two lists its caller
/// handed with their pair classes: one decision told to leave out the
/// workers the call missed so far ([`Call::missed`]), and the first of its
/// targets both of whose workers can be sent to. The decision is lent
/// `lent`: the request's own lists, or those lists with each side's
/// places beside it, the workers a rule takes ([`again`]) or the worker a
/// request names ([`target::pair`](super::target::pair)).
///
/// The placement names the request's routing value, and the profile of its
/// decode side by its label ([`decode_label`]).
///
/// `None` where no target holds. The decision is then given up by name,
/// and each side that could not be sent to joins the call's missed
/// workers by its own id: a prefill's or a decode's, never the pair as a
/// whole. The side that held stays a candidate of the next ask.
///
/// # Errors
///
/// The decision's shed, its queue or its refusal. Prefills with an open
/// partner that are all at the caller's admission limit shed the decision
/// under the capacity filter's name ([`prefills_full`]): that is no
/// refusal, and the answer is [`Unplaced::PrefillsFull`], on which the
/// caller waits for room. A place answered for a pair is the engine's
/// fault ([`DecisionRefused::Faulted`]): no ask after it would leave
/// anything new out.
pub(super) fn ask<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    lent: &WorkerLists<'_>,
    call: &mut Call,
) -> Result<Option<Placement<'a>>, Unplaced> {
    let mut made = Made {
        lists: lent,
        named_decode: call.named_decode,
        ..Made::asked(engine, request, call)
    };
    let mut whole = None;
    let decided = decide(engine, &made, &mut whole);
    // The route's rule panicked inside the decision, at the question of
    // where the prompt goes: the panic is the caller's, as a single's is.
    if let Some(panic) = made.panicked.take() {
        raise(engine.object, panic, whole, decided);
    }
    let (alone, dated) = decided.map_err(Unplaced::Refused)?;
    let (key, kept) = (made.key(), [made.sent.get(), made.decodes.get()]);
    // A pair's decision is whole: no one pick decides a place of two lists.
    let (None, Some((decided, later))) = (alone, whole) else {
        return Err(Unplaced::Refused(DecisionRefused::Faulted));
    };
    if prefills_full(&decided.decision) {
        return Err(Unplaced::PrefillsFull);
    }
    let now = dated.unwrap_or_else(Instant::now);
    let held = |target: &Target| sides(request, lists, target, &kept);
    let placement = walk(engine, decided, later, now, &mut call.missed, held)?;
    Ok(placement.map(|mut placement| {
        placement.routing_key = key;
        placement.decode_profile = Some(decode_label(engine, request.model));
        placement
    }))
}

/// The two workers of `target` in `lists`, where `request` can be sent to
/// both ([`sendable`]): what a walk of a pair's targets asks of each
/// ([`walk`]), over a pair's two lists and over the pair's two of an encode
/// request's three. Else the ids of the sides it cannot be sent to.
///
/// A worker the decision asked of is taken by the answer it was given
/// there (`kept`: the prefills', then the decodes'), and read no second
/// time: the request goes to the pair its prompt was stored for.
pub(super) fn sides(
    request: &PlaceRequest<'_>,
    lists: &PairLists<'_>,
    target: &Target,
    kept: &[Sent; 2],
) -> Result<Sides, [Option<WorkerId>; 2]> {
    let side = |list: LentList<'_>, place: u32, kept: &Sent| {
        let worker = list.workers.get(place as usize)?;
        let sends = kept.answer(place);
        let sends = sends.unwrap_or_else(|| sendable(request, worker.as_ref()));
        sends.then(|| Arc::clone(worker))
    };
    // A target with no decode side is no pair: it names nobody.
    let pair = target.pair.ok_or([None; 2])?;
    let prefill = side(lists.prefill, target.position, &kept[0]);
    let decode = side(lists.decode, pair.decode_position, &kept[1]);
    match (prefill, decode) {
        (Some(prefill), Some(decode)) => Ok((prefill, Some(decode))),
        (prefill, decode) => Err([
            prefill.is_none().then_some(target.id),
            decode.is_none().then_some(pair.decode_id),
        ]),
    }
}

/// The asks of one call for a pair over `lists` ([`asks`]): the first is
/// lent the request's own lists.
///
/// # Errors
///
/// What an ask answers, and [`Unplaced::NoWorkers`] where no target of the
/// call's last ask holds. A rule that takes no worker of one of the two
/// lists sheds the request with no endpoint, under `admits`
/// ([`hold_to_rule`]), or leaves it its first decision's shed ([`again`]).
pub(super) fn place<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    asks(engine, request, lists, &request.lists, call)
}

/// An ask of a call for a pair over `lists`, lent `lent` ([`ask`]), and
/// the asks of the call behind it: where no target of an ask holds, the
/// call's next ask answers in its place ([`again`]), three asks a call at
/// most ([`Call::again`]), and no worker for the request behind the last.
/// An ask that sheds the request goes there too, where a rule may change
/// its answer ([`goes_to_the_rule`]).
///
/// A pair held to the prefill its request names is asked so over its lists
/// with that prefill's place beside them
/// ([`target::pair`](super::target::pair)): a rule the request states is
/// then asked of that place alone. One held to the two workers it names
/// is the call's one decision: no ask follows it ([`again`]).
pub(super) fn asks<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    lent: &WorkerLists<'_>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    let shed = match ask(engine, request, lists, lent, call) {
        Ok(Some(placement)) => return Ok(placement),
        Ok(None) => None,
        Err(shed) if goes_to_the_rule(request, lists, call, &shed) => Some(shed),
        Err(unplaced) => return Err(unplaced),
    };
    let next = |lent: &WorkerLists<'_>, call: &mut Call| asks(engine, request, lists, lent, call);
    again(request, lists, lent, call, shed, &next)
}

/// Whether what an ask of `call` answered in a placement's stead (`shed`)
/// goes to the call's routine behind an ask ([`again`]), which asks the
/// rule the request states: the one test for a pair's asks and for an
/// encode request's, over the pair's two lists (`lists`).
///
/// A shed as saturated goes there, behind whichever ask: the decision
/// judged the overload over the lists it was lent, which a rule may
/// narrow. So does the shed with no endpoint of a call's first ask, where
/// the request states a rule and the shed names a side of the pair: that
/// decision judged the lists whole, and a
/// decision among the rule's workers may pair where it did not. Workers
/// that stand in the whole lists, and that the rule refuses, keep the
/// pair's default from the workers the rule takes: among those alone the
/// default crosses a liveness doubt. The gateway's own pair selection
/// narrows its lists by the route's filter ahead of every judgment of a
/// worker (the `admitted` test of `pair_pass` in
/// `model_gateway/src/routers/common/placement.rs`). Under no rule, and
/// behind a later ask, a shed with no endpoint answers the call: the rule
/// narrows nothing there.
///
/// It answers the call too for a request whose first decision is handed
/// a part (`PairPools::part`) and asks no question of its shed. One
/// states a hard demand ([`PlaceRequest::demands`]): its decision is
/// handed the workers that meet it. The other is an encode request the
/// pass is laid for ([`laid`]). That selection judges another list for
/// both (`admitted_pool` in the same file), which no decision of this
/// call is handed. So such a request is answered as it was before the
/// question stood: the rule is asked behind its shed as saturated alone.
///
/// A shed with no endpoint that names no side answers the call too: a
/// pair's refused ahead of both lists. One an encode request's own pass
/// makes, for its encoders or for an item, is of a request the pass is
/// laid for, and answers the call by the test above.
///
/// So does one made where no prefill of the lists pairs with a decode
/// ([`nothing_pairs`]), with its stage and its side: no rule changes
/// that, and no decision among a rule's workers could pair. The rule is
/// not asked behind it. That selection answers a list with no worker,
/// and lists where nothing pairs, whatever its filter takes: its tests
/// of the lists whole stand first in `pair_pass`.
pub(super) fn goes_to_the_rule(
    request: &PlaceRequest<'_>,
    lists: &PairLists<'_>,
    call: &Call,
    shed: &Unplaced,
) -> bool {
    match shed {
        Unplaced::Shed {
            reason: ShedReason::Saturated,
            ..
        } => true,
        Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            side: Some(_),
            ..
        } => {
            let whole = request.demands.is_none() && !laid(request);
            whole && call.spent == 0 && request.admits.is_some() && !nothing_pairs(lists)
        }
        _ => false,
    }
}

/// Whether the encode pass is laid for `request`: an encode request that
/// states an item, or holds its decode to the ones that take images, by
/// its demand or by references beside pixels
/// (`EncodeItems::lays_a_pass`, the pass's own test). The pass tells its
/// pair's decision that the pools are a part (`PairPools::part`), and
/// the call answers such a request as before a pair's decision asked
/// its one question. The one spelling of it for the call: the test of a
/// shed with no endpoint reads it here ([`goes_to_the_rule`]), and so
/// does the routine behind an ask ([`again`]).
fn laid(request: &PlaceRequest<'_>) -> bool {
    let pixels = request.media.holds_pixels();
    request
        .encode
        .is_some_and(|items| items.lays_a_pass(pixels))
}

/// The reason of a decision's shed, and `None` for every other answer.
const fn reason_of(unplaced: &Unplaced) -> Option<ShedReason> {
    match unplaced {
        Unplaced::Shed { reason, .. } => Some(*reason),
        _ => None,
    }
}

/// `shed` naming `side`, and every other answer as it is.
const fn sided(shed: Unplaced, side: Option<PairSide>) -> Unplaced {
    match shed {
        Unplaced::Shed {
            reason,
            emptied_by,
            retry_after_secs,
            ..
        } => Unplaced::Shed {
            reason,
            emptied_by,
            retry_after_secs,
            side,
        },
        other => other,
    }
}

/// Whether the prefill at `place` of `lists` pairs with a decode, by the
/// class its caller hands for it: the one test of it. The gateway's own
/// pair selection holds no prefill that pairs with no decode
/// (`PdPairIndex::build` in
/// `model_gateway/src/routers/common/pair_index.rs`).
fn pairs(lists: &PairLists<'_>, place: u32) -> bool {
    let classes = lists.classes.prefills();
    classes.get(place as usize).is_some_and(|&class| class != 0)
}

/// Whether no prefill of `lists` pairs with a decode of them ([`pairs`]):
/// a prefill list that holds no worker, a decode list that holds none,
/// or lists whose classes pair nowhere. No decision can pair there,
/// whatever a rule takes. The gateway's own pair selection asks these
/// three of its lists whole, in this order, ahead of any judgment its
/// filter narrows, and refuses for the prefill's role, for the decode's
/// and for the decode's (`pair_pass` in
/// `model_gateway/src/routers/common/placement.rs`): the side a pair's
/// decision names over such lists
/// ([`Shed::side`](crate::engine::decision::Shed::side)).
///
/// Classes numbered for the lists lent give every prefill the class of
/// one that pairs with nobody where the decode list holds no worker
/// ([`PartnerClasses::number`](crate::engine::PartnerClasses::number)),
/// so the test of the prefills says the same there. The test of the decode
/// list holds for a host whose classes were numbered for another list.
fn nothing_pairs(lists: &PairLists<'_>) -> bool {
    let mut prefills = (0..).take(lists.prefill.workers.len());
    lists.decode.workers.is_empty() || !prefills.any(|place| pairs(lists, place))
}

/// The prefills a rule takes that pair with a decode ([`pairs`]), of the
/// prefills at the places `taken`, in the list's order. A rule that takes
/// only prefills which pair with no decode takes none of that selection's
/// prefills. The call answers such a rule as one that takes no prefill
/// behind a shed over the prefills ([`again`]), and reads no such prefill
/// for a shed that stands ([`stands`]). A request the encode pass is laid
/// for keeps the answer it had ([`laid`]): its rule is asked of the
/// decodes behind such prefills too.
fn ruled<'a>(lists: &'a PairLists<'_>, taken: &'a [u32]) -> impl Iterator<Item = u32> + 'a {
    taken
        .iter()
        .copied()
        .filter(move |&place| pairs(lists, place))
}

/// The side a first decision's shed as saturated stands for behind a rule
/// that takes the prefills at the places `taken` and no decode, the
/// decision having said the list `word` ([`again`]): the list the gateway's
/// own pair selection answers its overload shed over there. `None` where
/// that selection answers no overload shed, and the shed does not stand.
///
/// The rule's prefills here are the ones that pair with a decode
/// ([`ruled`]). Where it takes none of those, a shed made over the
/// prefills stands, for their side, and one made over the decodes does
/// not: that selection judges the prefill list whole, which the decision
/// did not shed over. The routine answers a shed made over the prefills
/// ahead of this test. One of a request the encode pass is laid for comes
/// here, as before the question stood ([`laid`]).
///
/// The rule's prefills are read in the list's order, each once and
/// none behind the first that can take the request, with the host's
/// vetoes beside the list. One is ready where it can be sent to ([`ready_by`])
/// and no hold of its host stands on it. One can take the request where
/// it is ready, under its thresholds and under no liveness doubt.
///
/// Where one of them can take the request, that selection judges the
/// decode list, whole behind a rule that takes no decode
/// (`model_gateway/src/routers/common/placement.rs:2909-2936`, `:1831-1852`):
/// the shed stands where the decision said the decodes. Where none can,
/// it judges the rule's prefills (`:2866-2884`): the shed stands where one
/// of them is ready and every ready one is over its thresholds, whichever
/// list the decision said, and it stands for the prefill side: that
/// selection names the prefill's role there. A ready prefill under a doubt
/// is one of that
/// count: under its thresholds it keeps the rule's prefills from being
/// wholly over theirs, and over them it does not
/// (`model_gateway/src/routers/common/overload.rs:58-74`).
#[cold]
#[inline(never)]
fn stands(lists: &PairLists<'_>, taken: &[u32], word: ShedOver) -> Option<PairSide> {
    let mut ruled = ruled(lists, taken).peekable();
    if ruled.peek().is_none() {
        return (word == ShedOver::Prefills).then_some(PairSide::Prefill);
    }
    let stated = |places: &[u32], place: u32| places.binary_search(&place).is_ok();
    let (mut one_ready, mut all_over) = (false, true);
    for place in ruled {
        let Some(worker) = lists.prefill.workers.get(place as usize) else {
            continue;
        };
        let state = worker.routing_state();
        if !ready_by(state) || stated(lists.prefill_held, place) {
            continue;
        }
        one_ready = true;
        if !state.overloaded {
            // This one can take the request: no prefill behind it is read.
            if !state.stalled && !stated(lists.prefill_stalled, place) {
                return (word == ShedOver::Decodes).then_some(PairSide::Decode);
            }
            all_over = false;
        }
    }
    (one_ready && all_over).then_some(PairSide::Prefill)
}

/// The call's next ask (`next`) behind one none of whose targets held, or
/// that shed the request (`shed`: as saturated, or a first ask's shed
/// with no endpoint for a side under a rule, [`goes_to_the_rule`]), which was lent
/// `lent` ([`asks`]). Behind the call's first ask, a rule the request
/// states is asked of the two lists first, while the call holds no
/// decision ([`hold_to_rule`]): of the prefills, then of the decodes,
/// among the places the caller states for each. It is asked whether it
/// refused a side of the first decision or the sides were not ready: a
/// side that is not ready is not asked of it, so a decision none of whose
/// sides is ready learns nothing of the rule.
///
/// The asks behind that one are lent the request's own lists with each
/// side's places set to the workers the rule takes
/// ([`PairLists::prefill_places`], [`PairLists::decode_places`]): the
/// lists stay the caller's, with their cells and their classes. The
/// places live here, beside those asks: a call keeps none of a pair's. A
/// request that states no rule keeps no places, and each of its asks is
/// lent its lists as they stand. Out of line: a pair whose first ask
/// takes a target runs none of it.
///
/// One routine for a pair and for an encode request that carries items
/// ([`super::encode`]): `lists` are the pair's two, of a pair's lists or
/// of an encode request's three, and `next` is the caller's own ask, which
/// this routine is handed and does not name. The rule is asked of the
/// pair's two lists and of no encoder, and the encoders' list is lent as
/// it stands. An encode request comes here too where an encoder could not
/// be sent to: its pair is then decided among the rule's workers as well.
/// A shed as saturated made over its encoders comes here and stands,
/// ahead of the rule: the rule is asked of no encoder, so it narrows
/// nothing that decision judged over. A shed with no endpoint made for
/// them answers the call ahead of this routine ([`goes_to_the_rule`]).
///
/// A decision judges the overload over the lists it is lent: a worker
/// under its thresholds that the rule refuses, and that no decision
/// names, keeps a list from being wholly over them, and the decision
/// sheds the request. Held to the rule's workers, the next decision
/// judges it over those alone. Under a document that sheds a list
/// wholly over its thresholds that decision sheds again, and its shed
/// stands. A shed stands where the rule is not asked here: behind a
/// later ask, and under no rule.
///
/// Where the rule takes no worker of a list, no decision can be made
/// among its workers. The first decision's shed as saturated is then the
/// call's answer in two cases, and in both that decision said a list: its
/// own word of the list it shed over ([`shed_over`]), which it says under
/// a document that sheds a list wholly over its thresholds, where the
/// gateway's own selection would read that list so too. A prefill that
/// pairs with no decode is no prefill of the rule's in either
/// ([`ruled`]). The rule takes no prefill, and the decision shed over the
/// prefills: the decodes are then not asked. Or the rule takes a prefill
/// and no decode, and its prefills let the shed stand ([`stands`]): one
/// of them can take the request and the decision shed over the decodes,
/// or none can, one is ready and every ready one is over its thresholds.
/// The shed that stands names the side of the list that selection
/// judges: the decision's own side, but the prefill side where none of
/// the rule's prefills can take the request, whichever list the decision
/// said. In every other case the request is shed with no endpoint: by
/// the first decision's own shed where that decision shed it so, with no
/// side, and else under `admits`, as it is behind a first decision that
/// picked and behind one that said no list.
///
/// A request the encode pass is laid for keeps the answer it had before
/// the question stood ([`laid`]). Where its rule takes prefills none of
/// which pairs with a decode, behind a shed made over the prefills, the
/// decodes are asked too. The shed stands where the rule takes no decode,
/// and else a decision among the rule's workers answers.
///
/// A first decision's shed with no endpoint stands there, under places
/// its host states too: that decision judged the lists it was handed
/// whole, and the gateway's own selection judges every worker of a list
/// on its wire where its filter takes none of them (`admitted_pool`).
/// The stage it names is the
/// decision's. It names no side: the rule narrowed
/// the lists that decision judged, and the caller judges the rule's
/// lists, as it does behind a shed under `admits`. One made where no
/// prefill of the lists pairs with a decode does not come here: it
/// answers the call with its side, and the rule is not asked behind it
/// ([`goes_to_the_rule`]). Nor does one of a request that states a hard
/// demand, or of an encode request the pass is laid for.
///
/// A request held to a part of a list keeps the shed under `admits` in
/// both cases: one that states a hard demand
/// ([`PlaceRequest::demands`]), and one whose host states places of the
/// prefills or of the decodes ([`PairLists::prefill_places`],
/// [`PairLists::decode_places`]). The demand hands the decision the
/// workers that meet it, and the places the workers at them, so a list
/// the decision counted may be a part. The call acts on no word of the
/// prefills or the decodes for such a request: a word of the list that
/// stayed whole goes too, and so does one of a list whose places name
/// every worker. The gateway's own selection states no places. It holds
/// such a need in its filter, and judges the list on the request's wire
/// whole where its filter takes nobody (`admitted_pool`).
///
/// A shed made over the encoders stands for such a request as for any,
/// ahead of this test. Places of the encoders alone
/// ([`EncodeLists::encoders_places`]) hold an encode request to no part
/// of its pair's two lists. Where the encoders at them leave a prefill or
/// a decode out, the encode pass takes the word back itself. So where the
/// word stays, the decision judged both lists whole, and its shed stands.
///
/// The gateway's own pair selection answers its overload shed in both
/// cases, and over the lists as the decision counted them the shed stands
/// in no case where it answers that it has no worker. It judges one list:
/// the prefills where no prefill the rule takes can take the request, and
/// else the decodes. It judges that list over the rule's workers of it, and
/// whole where the rule takes none of it
/// (`model_gateway/src/routers/common/placement.rs:2866-2884`, `:2909-2936`
/// and `:1831-1852`, on main `56773d28d`). A worker that is not ready is no
/// part of a list's overload there, and one under a liveness doubt is
/// (`model_gateway/src/routers/common/overload.rs:58-74`).
///
/// One read behind the rule is fresh, in the second case: the state of
/// the prefills the rule takes that pair with a decode, in the list's
/// order and up to the first that can take the request ([`stands`]),
/// whether the decision counted them or not. One of them that is ready,
/// under no hold, and came under its thresholds since the decision keeps
/// no shed over the prefills. The decision's count of each list is not
/// read again: a decode, or a prefill the rule does not take, whose state
/// changes behind the decision leaves the answer as it is, though that
/// selection asks its filter first and judges the list behind it
/// (`:2866-2884`). A second read here would only move that instant.
///
/// A pair held to the two workers its request names
/// ([`target::pair`](super::target::pair)) is asked no second time: behind
/// a worker it cannot send to it has no worker, and its shed stands.
#[cold]
#[inline(never)]
pub(super) fn again<'a>(
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    lent: &WorkerLists<'_>,
    call: &mut Call,
    shed: Option<Unplaced>,
    next: &dyn Fn(&WorkerLists<'_>, &mut Call) -> Result<Placement<'a>, Unplaced>,
) -> Result<Placement<'a>, Unplaced> {
    /// `pair` with each side's places beside its list.
    fn held<'w>(pair: PairLists<'w>, prefills: &'w [u32], decodes: &'w [u32]) -> PairLists<'w> {
        PairLists {
            prefill_places: Some(prefills),
            decode_places: Some(decodes),
            ..pair
        }
    }
    if call.named_pair {
        return Err(shed.unwrap_or(Unplaced::NoWorkers));
    }
    let (mut prefills, mut decodes) = (None, None);
    // The rule is asked of a list once a call: behind the first ask.
    if call.spent == 0 {
        // The list the first decision shed the request over as saturated,
        // by that decision's own word, which stands behind such a shed
        // alone. Read ahead of the rule: a rule may place a request on
        // this thread, and that decision writes its own word.
        let saturated = |shed: &Unplaced| reason_of(shed) == Some(ShedReason::Saturated);
        let over = shed.filter(saturated);
        let over = over.and_then(|shed| Some((shed, shed_over()?)));
        // The rule is asked of no encoder: a shed made over the encoders
        // stands, and no rule is asked behind it.
        if let Some((shed, ShedOver::Encoders)) = over {
            return Err(shed);
        }
        // A hard demand hands the decision a part of each list, the
        // workers that meet it, and the places its host states for the
        // prefills or the decodes hand it the workers at them. A list the
        // decision counted may then be a part: no word of the prefills or
        // the decodes is acted on for such a request.
        let whole = request.demands.is_none()
            && lists.prefill_places.is_none()
            && lists.decode_places.is_none();
        let over = over.filter(|_| whole);
        let prefill = lists.prefill;
        // A first decision's shed with no endpoint, which stands where
        // the rule takes no worker of a list, and names no side there: the
        // rule narrowed the lists that decision judged.
        let no_endpoint = shed.filter(|shed| reason_of(shed) == Some(ShedReason::NoEndpoints));
        let no_endpoint = no_endpoint.map(|shed| sided(shed, None));
        let asked = hold_to_rule(request, Some(prefill), lists.prefill_places, &mut prefills);
        if let Err(none) = asked {
            // The rule takes no prefill: a shed made over the prefills
            // stands.
            return Err(match over {
                Some((shed, ShedOver::Prefills)) => shed,
                _ => no_endpoint.unwrap_or(none),
            });
        }
        // The rule takes prefills, and none that pairs with a decode: it
        // is answered as one that takes no prefill ([`ruled`]). A shed made
        // over the prefills stands, and the decodes are not asked. A
        // request the encode pass is laid for is answered as before the
        // question stood: its rule is asked of the decodes too.
        let taken = prefills.as_deref().unwrap_or_default();
        if let Some((shed, ShedOver::Prefills)) = over {
            if !laid(request) && ruled(lists, taken).next().is_none() {
                return Err(shed);
            }
        }
        let asked = hold_to_rule(
            request,
            Some(lists.decode),
            lists.decode_places,
            &mut decodes,
        );
        if let Err(none) = asked {
            // The rule takes a prefill and no decode: the shed stands by
            // what the rule's prefills are, for the side they say
            // ([`stands`]).
            let side = over.and_then(|(_, word)| stands(lists, taken, word));
            return Err(match (over, side) {
                (Some((shed, _)), Some(_)) => sided(shed, side),
                _ => no_endpoint.unwrap_or(none),
            });
        }
    }
    // A shed stands where the rule was not asked here: behind a later ask,
    // and under no rule.
    if let (Some(shed), None) = (shed, &prefills) {
        return Err(shed);
    }
    let lent = match (request.lists, &prefills, &decodes) {
        (WorkerLists::Pair(pair), Some(prefills), Some(decodes)) => {
            WorkerLists::Pair(held(pair, prefills, decodes))
        }
        // An encode request's three lists: the pair's two with their
        // places, the encoders' as it stands. With no item they are
        // decided as their pair, and with items by the request's own ask.
        (WorkerLists::Encode(three), Some(prefills), Some(decodes)) => {
            WorkerLists::Encode(EncodeLists {
                pair: held(three.pair, prefills, decodes),
                ..three
            })
        }
        // No places were found here: the next ask is lent what this one was.
        _ => *lent,
    };
    call.again(|call| next(&lent, call))
}
