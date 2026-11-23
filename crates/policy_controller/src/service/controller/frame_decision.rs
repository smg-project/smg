//! One `Decide` frame decided by the engine's in-process front, and the
//! lease of its pick.
//!
//! A controller decides a single request over the list of its model's
//! single workers, as the replica's worker objects lend it
//! ([`Objects::lent`]). [`decide`] takes the engine and the list of one
//! load of the world, states what the asking gateway may take of the list
//! as places ([`GatewayPlaces::taken`]), makes the frame's request
//! ([`FrameRequest`]), gives it the load snapshot its deciding thread holds
//! ([`FrameLoads`](super::frame_loads::FrameLoads)) where the document
//! reads reports and keeps none, and asks the engine's entry that issues no
//! lease.
//! [`issue`] files a pick's lease in the store, which tells the document's
//! stages of it under its hold of the file's turn.
//!
//! A pair is decided at its prefill, for both sides, over its model's two
//! lists: [`decide_pair`] states each side's places by that side's own
//! list and gives the request the same snapshot, and [`issue_pair`] files
//! a lease a side under one id.
//!
//! The engine that decides and the stages that are told are one object:
//! the world's ([`StoreSnapshot::front`]). A lease is dated on that
//! engine's clock, the one its decisions read.
//!
//! A decode call and an encode request have no decision here yet: their
//! deciding thread answers them `Busy` and reads no list for them.

use std::sync::{atomic::Ordering, Arc};

use smg_worker::load::state::LoadSnapshot;

use super::{
    decision_slots::{DecisionCall, Part},
    frame_request::FrameRequest,
    gateway_places::{Asking, GatewayPlaces, Taken},
    wire::{Checked, Lists},
};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{shed, Decision, Pick, ShedReason, Target},
        decision_engine::{DecisionRefused, WorkerLists},
        policy_engine::PolicyEngine,
        shadow::Shadowed,
        view::{fingerprint, Phase, RequestFacts, Role, Session},
        PairLists,
    },
    stages::lifecycle::Lifecycle,
    store::{
        DecodeSide, LeaseRecord, LeaseRefused, LeaseState, MemoryStore, PickRecord, StoreSnapshot,
    },
    tracking::lease::{Lease, LeaseId, LeaseKind, Side},
    worker_host::{Members, Objects, OpenAdmission},
};

/// A deciding thread's buffers for its frames, kept from frame to frame: a
/// frame's lists in the engine's form, and the places its gateway may take
/// of a single request's list or of a pair's prefills, and of a pair's
/// decodes.
#[derive(Debug, Default)]
pub(crate) struct FrameBuffers {
    lists: Lists,
    places: GatewayPlaces,
    decode_places: GatewayPlaces,
}

/// What a pick's lease takes of its frame, beside the decision: the
/// session and the prompt's tokens as the frame's one reader makes them,
/// and what the binding's shadow said behind the live decision.
#[derive(Debug)]
pub(crate) struct Booked {
    session: Session,
    tokens: u32,
    said: Option<Shadowed>,
}

/// What a pair's two leases take of its frame, beside the decision: each
/// side's session and the prompt's tokens, as the frame's one reader makes
/// them. Each side of a pair keeps a session of its own.
#[derive(Debug)]
pub(crate) struct PairBooked {
    session: Session,
    decode_session: Session,
    tokens: u32,
}

/// The decision of a single request's frame, in the world of one load: the
/// engine and the gateway's vetoes are that world's, and the list is the
/// one the worker objects' registry holds for the frame's model at this
/// read. The load reports are those of `loads`, the newest snapshot the
/// replica's load monitor published when the frame was taken, where the
/// document is given it ([`hands_reports`]). What the
/// decision counts, it counts in `own`, its decider's part:
/// a frame with more block hashes than the document reads, and the workers
/// its gateway's wires took.
///
/// A gateway that dials no worker of the list, and a model the registry
/// holds no worker of, are decided over no worker. A gateway whose wires
/// or whose subset hint of 0 take every worker it dials is shed under that
/// stage, and the engine is asked nothing.
///
/// A profile with a stage that may refuse by a report of leases is decided
/// behind an apply of what waits at the files of leases, where the decider
/// lent its place.
///
/// # Errors
///
/// [`DecisionRefused::Faulted`] where the decision panicked: the engine
/// caught it and counted it.
pub(crate) fn decide(
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    buffers: &mut FrameBuffers,
    loads: &Arc<LoadSnapshot>,
    own: &Part,
) -> Result<(Decision, Booked), DecisionRefused> {
    let DecisionCall {
        request: frame,
        gateway,
        now,
        left,
        place,
        ..
    } = *taken;
    let front = world.front();
    let spec = front.spec();
    let model = frame.model();
    if frame.hashes() > frame.hashes_read(spec) {
        own.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    let (_, _, by_reports) = spec.engine().reads_for(model, Phase::Single);
    if let Some(place) = place.filter(|_| by_reports) {
        place.apply_waiting();
    }
    let FrameBuffers { lists, places, .. } = buffers;
    let read = lists.of(frame);
    let over_none = || FrameRequest::new(frame, &read, left, WorkerLists::None);
    let generation = objects.registry().model_generation(model);
    let (decision, said) = match generation.as_deref() {
        Some(generation) => {
            let lent = objects.lent(generation, Role::Single);
            let members = front.members_of(lent);
            let records = members.records(spec, front.clock());
            let reach = objects.reach(generation, Role::Single);
            let vetoes = world.gateway_vetoes(model, Role::Single, gateway);
            let asking = Asking::of(gateway, frame);
            let decided = match places.taken(reach, records, asking, vetoes.standing_at(now)) {
                Taken::Held(held) => {
                    let list = WorkerLists::Single(lent);
                    let maker = FrameRequest::new(frame, &read, left, list);
                    let maker = maker.held_to(held.places, held.stalled, held.held);
                    let maker = if hands_reports(spec) {
                        maker.reading(loads)
                    } else {
                        maker
                    };
                    front.decide_without_lease(&maker, now)
                }
                Taken::Emptied(stage) => {
                    let (seed, generation) = (frame.seed(), spec.generation());
                    Ok((shed(ShedReason::NoEndpoints, stage, seed, generation), None))
                }
                Taken::Unreached => front.decide_without_lease(&over_none(), now),
            };
            // Counted whatever the decision came to.
            let undialed = places.undialed();
            if undialed > 0 {
                own.undialed.fetch_add(undialed, Ordering::Relaxed);
            }
            decided?
        }
        None => front.decide_without_lease(&over_none(), now)?,
    };
    // What the lease of a pick takes of its request, read where the frame's
    // lists still stand.
    let (session, tokens) = match decision.pick() {
        Some(_) => {
            let facts = frame.facts(&read, 0, 0);
            (facts.session(), facts.prompt_tokens)
        }
        None => (Session::NONE, 0),
    };
    let booked = Booked {
        session,
        tokens,
        said,
    };
    Ok((decision, booked))
}

/// Whether a frame's request is given the load monitor's snapshot under
/// `spec`: its document reads load reports or a named signal, whose values
/// the same reports state, and no stage of it keeps reports.
///
/// A stage that keeps reports is told each one by its engine's own task,
/// ahead of the snapshot that task keeps, and a snapshot a request gives is
/// outside that order ([`RequestMaker::loads`]). A controller runs no such
/// task yet: a document that holds such a stage is given none here, and its
/// decisions read no report.
///
/// [`RequestMaker::loads`]: crate::engine::RequestMaker::loads
fn hands_reports(spec: &AppliedSpec) -> bool {
    let needs = spec.needs();
    (needs.loads || needs.signal_cells) && !spec.engine().keeps_loads()
}

/// The decision of a pair's frame at its prefill, in the world of one load
/// as [`decide`] takes it: over the two lists the worker objects' registry
/// holds for the frame's model, its prefills and its decodes, with their
/// pair classes under the document's pairing. Each side's list is the
/// asking gateway's to take as that side's own reach, wires and vetoes
/// state it ([`GatewayPlaces::taken`], once a side). Both sides read the
/// reports of `loads`, where a single request's decision is given it
/// ([`hands_reports`]). The pair's admission is open: a controller keeps no
/// gate of its own.
///
/// The answers come in one order. A model the registry holds no worker of,
/// and a side with no worker at all, are decided over no worker: one side
/// alone is no pair. Else a side the gateway dials and can take nothing of
/// sheds the frame under that stage. Else a side the gateway dials no
/// worker of is decided over no worker.
///
/// A pair either of whose profiles holds a stage that may refuse by a
/// report of leases is decided behind an apply of what waits at the files
/// of leases, where the decider lent its place. The engine keeps that word
/// by phase, and the decode phase's also carries a decode binding of the
/// model's own: such a binding asks the apply too.
///
/// # Errors
///
/// [`DecisionRefused::Faulted`] where the decision panicked: the engine
/// caught it and counted it.
pub(crate) fn decide_pair(
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    buffers: &mut FrameBuffers,
    loads: &Arc<LoadSnapshot>,
    own: &Part,
) -> Result<(Decision, PairBooked), DecisionRefused> {
    let DecisionCall {
        request: frame,
        gateway,
        now,
        left,
        place,
        ..
    } = *taken;
    let front = world.front();
    let spec = front.spec();
    let model = frame.model();
    if frame.hashes() > frame.hashes_read(spec) {
        own.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    // The decode phase's word carries the pair's decode side.
    let by_reports = [Phase::Prefill, Phase::Decode]
        .into_iter()
        .any(|phase| spec.engine().reads_for(model, phase).2);
    if let Some(place) = place.filter(|_| by_reports) {
        place.apply_waiting();
    }
    let FrameBuffers {
        lists,
        places,
        decode_places,
    } = buffers;
    let read = lists.of(frame);
    let over_none = || FrameRequest::new(frame, &read, left, WorkerLists::None);
    let generation = objects.registry().model_generation(model);
    let (decision, _) = match generation.as_deref() {
        Some(generation) => {
            // One side as the decision reads it: the list its role lends,
            // its records, who may dial its workers, and the gateway's
            // vetoes on them.
            let side = |role| {
                let lent = objects.lent(generation, role);
                let records = front.members_of(lent).records(spec, front.clock());
                let vetoes = world.gateway_vetoes(model, role, gateway);
                (lent, records, objects.reach(generation, role), vetoes)
            };
            let (prefill, prefill_records, prefill_reach, prefill_vetoes) = side(Role::Prefill);
            let (decode, decode_records, decode_reach, decode_vetoes) = side(Role::Decode);
            if prefill_records.is_empty() || decode_records.is_empty() {
                front.decide_without_lease(&over_none(), now)?
            } else {
                let asking = Asking::of(gateway, frame);
                let vetoes = prefill_vetoes.standing_at(now);
                let prefills = places.taken(prefill_reach, prefill_records, asking, vetoes);
                let vetoes = decode_vetoes.standing_at(now);
                let decodes = decode_places.taken(decode_reach, decode_records, asking, vetoes);
                let decided = match (prefills, decodes) {
                    (Taken::Held(prefills), Taken::Held(decodes)) => {
                        let pair = PairLists {
                            prefill,
                            decode,
                            classes: objects.pair_classes(generation, spec.pairing()),
                            admission: &OpenAdmission,
                            prefill_places: prefills.places,
                            decode_places: decodes.places,
                            prefill_stalled: prefills.stalled,
                            prefill_held: prefills.held,
                            decode_stalled: decodes.stalled,
                            decode_held: decodes.held,
                        };
                        let maker = FrameRequest::new(frame, &read, left, WorkerLists::Pair(pair));
                        let maker = if hands_reports(spec) {
                            maker.reading(loads)
                        } else {
                            maker
                        };
                        front.decide_without_lease(&maker, now)
                    }
                    (Taken::Emptied(stage), _) | (_, Taken::Emptied(stage)) => {
                        let (seed, generation) = (frame.seed(), spec.generation());
                        Ok((shed(ShedReason::NoEndpoints, stage, seed, generation), None))
                    }
                    (Taken::Unreached, _) | (_, Taken::Unreached) => {
                        front.decide_without_lease(&over_none(), now)
                    }
                };
                // Counted whatever the decision came to.
                let undialed = places.undialed() + decode_places.undialed();
                if undialed > 0 {
                    own.undialed.fetch_add(undialed, Ordering::Relaxed);
                }
                decided?
            }
        }
        None => front.decide_without_lease(&over_none(), now)?,
    };
    // What the leases of a pick take of its request, read where the frame's
    // lists still stand: the decode side's session is the request's under
    // the decode phase.
    let booked = match decision.pick() {
        Some(_) => {
            let facts = frame.facts(&read, 0, 0);
            let decode = RequestFacts {
                phase: Phase::Decode,
                ..facts
            };
            PairBooked {
                session: facts.session(),
                decode_session: decode.session(),
                tokens: facts.prompt_tokens,
            }
        }
        None => PairBooked {
            session: Session::NONE,
            decode_session: Session::NONE,
            tokens: 0,
        },
    };
    Ok((decision, booked))
}

/// The record of the lease a pick asks on `side`, as the store takes it: a
/// credit on the pick's first target, dated on the engine's own clock at
/// the decision's first read, with the dispatch span that runs from it.
/// The store mints its id, and writes the mark of a pair's lease where it
/// files one.
///
/// Inline in both issues: a single's issue stays the code it was before a
/// pair's shared the record with it.
#[inline(always)]
fn record_of(
    front: &PolicyEngine,
    taken: &DecisionCall<'_, Checked>,
    term: u64,
    (pick, first): (&Pick, &Target),
    side: Side,
    (session, tokens): (Session, u32),
) -> LeaseRecord {
    let lease = Lease {
        id: LeaseId::default(),
        kind: LeaseKind::Credit,
        side,
        target_index: 0,
        booked: first.id,
        generation: pick.trace.generation,
        session,
        session_verdict: pick.session_verdict,
        // The engine's own clock: its decisions and its stages read it.
        issued_micros: front.clock().micros(taken.now),
        tokens,
        // A frame states a count and no text's length.
        chars: 0,
        bytes: 0,
    };
    LeaseRecord {
        lease,
        gateway: taken.gateway,
        request: fingerprint(taken.request.id()),
        target: first.id,
        picked: PickRecord::of(&pick.targets),
        decode: None,
        term,
        issued_at: taken.now,
        dispatch_deadline: taken.due(),
        state: LeaseState::ISSUED,
    }
}

/// Files the lease of `decision`'s pick, one a decision, ahead of the read
/// that lets its answer leave: a credit on the pick's first target, minted
/// and filed by the store under `term`, the fence's at the decision's
/// first read, at that read's instant. The lease books its target's count
/// of the leases on file, and the document's stages are told of it under
/// the store's hold of the file's turn. `None` for a shed, which files
/// none.
///
/// A keyed pick files a lease that counts for its key
/// ([`MemoryStore::issue_keyed`]), under any sessions rule and under
/// none: one request more of its session stands on its target until the
/// lease leaves the file. The store moves that count onto the target's
/// worker object, as a gateway's load guard counts every request that
/// carries a key. A decision reads it there: the key's own requests under
/// an in-flight cap, and the keys with a request in flight where a rule
/// or a scorer places by the fewest.
///
/// The binding's shadow is counted behind a filing that stands, and behind
/// a shed all the same. A filing the store refuses counts none.
///
/// # Errors
///
/// The store's refusal: it stands under another term, or mints no id
/// under this one.
pub(crate) fn issue(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, Checked>,
    term: u64,
    decision: &Decision,
    booked: Booked,
) -> Result<Option<Lease>, LeaseRefused> {
    let front = world.front();
    let Booked {
        session,
        tokens,
        said,
    } = booked;
    let pick = decision.pick();
    let Some((pick, first)) = pick.and_then(|pick| Some((pick, pick.targets.first()?))) else {
        front.count_shadow(said);
        return Ok(None);
    };
    let stages = front.spec().engine();
    let sized = (session, tokens);
    let record = record_of(front, taken, term, (pick, first), Side::Single, sized);
    let told = |lease: &Lease| stages.on_decision_issued(lease, decision);
    // The slot of the pick's count, as the decision's world names it. A
    // member the worker objects list ahead of that world, inside its
    // registration window, is read in the table its write filed it in:
    // its lease books its count as any other's does.
    let booked = world
        .booked_of(first.id)
        .or_else(|| store.filed_slot(first.id));
    // A keyed pick's lease counts for its key, whatever rule placed it and
    // whatever the rule did with the pin: one more of the key's requests
    // on the worker it lands on. A pick with no key reads no session.
    let lease = if session.is_none() {
        store.issue_decided(record, booked, told)
    } else {
        store.issue_keyed(record, booked, told)
    }?;
    front.count_shadow(said);
    Ok(Some(lease))
}

/// Files the two leases of a pair's pick, as [`issue`] files a single's:
/// a lease a side under the one id the store mints, one record and one
/// write of the file. The prefill side's is a single's but for its side,
/// and the decode side's stands beside it on the first target's decode,
/// under the request's session of the decode phase. Each side that is
/// issued books its own worker's count as a single's lease books its
/// target's, and the document's stages are told once a side that is
/// issued, the prefill's first. `Ok` is the prefill side's lease, then the
/// decode side's where it was issued; `None` for a shed, which files none.
///
/// A frame that states that its decode is sent later
/// (`DecideRequest.decode_later`) files the decode side chosen: the side
/// holds the pick's decodes and books nothing, and no stage hears of it
/// until a dispatch names it. Its answer carries the prefill side's lease
/// alone.
///
/// No shadow runs on a pair's side yet, so none is counted here. The
/// decode side's lease states the prefill side's session verdict: the
/// store's decode side keeps none of its own yet.
///
/// # Errors
///
/// Those of [`issue`].
pub(crate) fn issue_pair(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, Checked>,
    term: u64,
    decision: &Decision,
    booked: PairBooked,
) -> Result<Option<(Lease, Option<Lease>)>, LeaseRefused> {
    let front = world.front();
    let PairBooked {
        session,
        decode_session,
        tokens,
    } = booked;
    // Each target of a pair's pick names its decode.
    let named = decision.pick().and_then(|pick| {
        let first = pick.targets.first()?;
        Some((pick, first, first.pair?))
    });
    let Some((pick, first, chosen)) = named else {
        // The engine makes no pick of a pair whose first target names no
        // decode: one that did would file nothing, and is counted as none.
        debug_assert!(decision.pick().is_none(), "a pair's pick: {decision:?}");
        return Ok(None);
    };
    let stages = front.spec().engine();
    let sized = (session, tokens);
    let record = record_of(front, taken, term, (pick, first), Side::Prefill, sized);
    let later = taken.request.decode_later();
    let side = DecodeSide {
        target: chosen.decode_id,
        picked: PickRecord::of_decodes(&pick.targets),
        session: decode_session,
        state: if later {
            LeaseState::Chosen
        } else {
            LeaseState::ISSUED
        },
        tokens,
        target_index: 0,
    };
    // Each side's slot as a single's is found. The store books none for a
    // side filed chosen: the dispatch that names it books its worker.
    let slot = |id| world.booked_of(id).or_else(|| store.filed_slot(id));
    let slots = [slot(first.id), slot(chosen.decode_id)];
    let told = |lease: &Lease| stages.on_decision_issued(lease, decision);
    let lease = store.issue_decided_pair(record, side, slots, told)?;
    Ok(Some((lease, (!later).then(|| side.lease(&lease)))))
}

#[cfg(test)]
mod tests;
