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
//! a lease a side under one id. A decode call behind a pair's held prefill
//! is decided over the same two lists, with the decode side's places alone
//! ([`decide_decode`]), and its pick is issued under the pair's own lease
//! ([`issue_decode`]).
//!
//! The engine that decides and the stages that are told are one object:
//! the world's ([`StoreSnapshot::front`]). A lease is dated on that
//! engine's clock, the one its decisions read.
//!
//! A prefill frame, and a decode frame that names no lease, whose plan
//! pairs no side of its phase stands alone: it is decided as a single
//! request's frame is, over the list of its own phase's role
//! ([`decide_alone`]).
//!
//! An encode request has no decision here yet, and a decode frame that
//! names no lease has none under a plan that pairs its phase: their
//! deciding thread answers them `Busy` and reads no list for them.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{atomic::Ordering, Arc},
};

use smg_worker::load::state::LoadSnapshot;
use tracing::error;

use super::{
    decision_slots::{DecisionCall, Part},
    frame_request::FrameRequest,
    gateway_places::{Asking, GatewayPlaces, Taken},
    lease_words::{issued_later, issuing},
    wire::{Checked, Lists},
};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{shed, Decision, Pair, Pick, ShedReason, Target},
        decision_engine::{DecisionRefused, DecodeRefused, WorkerLists},
        framework::release,
        lease_events,
        policy_engine::PolicyEngine,
        shadow::Shadowed,
        view::{fingerprint, Phase, RequestFacts, Role, Session, WorkerId},
        PairLists,
    },
    stages::lifecycle::Lifecycle,
    store::{
        DecodeIssue, DecodeSide, LeaseRecord, LeaseRefused, LeaseState, MemoryStore, PickRecord,
        StoreSnapshot,
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
    decide_at(Phase::Single, world, objects, taken, buffers, loads, own)
}

/// The decision of a frame that stands alone at a side's phase: a prefill
/// frame, or a decode frame that names no lease, whose plan pairs no side
/// of its phase. It is decided as a single request's frame is ([`decide`]),
/// by the binding of its own phase, over the list the registry holds for
/// the model at that phase's role. The reach and the asking gateway's
/// vetoes are that role's. A frame states no demand yet, so no list is
/// held to one here.
///
/// Out of line: a single request's decision holds no part of it.
///
/// # Errors
///
/// Those of [`decide`].
#[inline(never)]
pub(crate) fn decide_alone(
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    buffers: &mut FrameBuffers,
    loads: &Arc<LoadSnapshot>,
    own: &Part,
) -> Result<(Decision, Booked), DecisionRefused> {
    let phase = taken.request.phase();
    decide_at(phase, world, objects, taken, buffers, loads, own)
}

/// [`decide`] at `phase`, the frame's own: the binding the phase resolves
/// to and the list of the phase's role. In line in both its callers, so a
/// single request's decision stays the code it was.
#[inline(always)]
fn decide_at(
    phase: Phase,
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
    let (_, _, by_reports) = spec.engine().reads_for(model, phase);
    if let Some(place) = place.filter(|_| by_reports) {
        place.apply_waiting();
    }
    let role = phase.role();
    let FrameBuffers { lists, places, .. } = buffers;
    let read = lists.of(frame);
    let over_none = || FrameRequest::new(frame, &read, left, WorkerLists::None);
    let generation = objects.registry().model_generation(model);
    let (decision, said) = match generation.as_deref() {
        Some(generation) => {
            let lent = objects.lent(generation, role);
            let members = front.members_of(lent);
            let records = members.records(spec, front.clock());
            let reach = objects.reach(generation, role);
            let vetoes = world.gateway_vetoes(model, role, gateway);
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
        // A single's issue states the word of a pick its sessions rule
        // keeps from the profile. A pair's record states none.
        untold: false,
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
/// A pick the binding's sessions rule keeps from the profile is filed under
/// that word ([`LeaseRecord::untold`]), read off the decision in hand
/// ([`lease_events::stages_hear`]). No stage hears its issue, and the store
/// reads the word where it tells the lease's dispatch.
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
    // The stages hear of every pick but one the binding's sessions rule
    // keeps from the profile: that one's lease is filed under the word.
    let hears = lease_events::stages_hear(decision);
    let record = LeaseRecord {
        untold: !hears,
        ..record_of(front, taken, term, (pick, first), Side::Single, sized)
    };
    let told = |lease: &Lease| {
        if hears {
            stages.on_decision_issued(lease, decision);
        }
    };
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
/// until a decode call issues it ([`issue_decode`]) or a dispatch names it.
/// Its answer carries the prefill side's lease alone.
///
/// No shadow runs on a pair's side yet, so none is counted here. The
/// decode side's lease states the prefill side's session verdict: the
/// store's decode side keeps none of its own yet. So a document that
/// binds a sessions rule under the pair plan is refused at its apply, as
/// not in this build.
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

/// The decision of a decode call's frame, in the world of one load as
/// [`decide_pair`] takes it: the pair's decode decided again behind the
/// prefill its lease holds. `held` is the store's answer for the lease the
/// call names ([`MemoryStore::held_pair`]): the held prefill's worker, then
/// the decode the lease's first decision chose for the target that
/// prefill's dispatch took. A decode profile that keeps its pick answers
/// that decode again while its filters keep it.
///
/// The decode side's list alone is the asking gateway's to take, by its
/// reach, wires and standing vetoes: the held prefill was sent already, and
/// the engine finds it in its list wherever it stands. The answers come in
/// a pair's order, for the one side. A model the registry holds no worker
/// of, and one with no decode, are decided over no worker. Else a gateway
/// whose wires or whose subset hint of 0 take every decode it dials is shed
/// under that stage. Else a gateway that dials no decode is decided over no
/// worker. Only then is the engine asked, and it alone looks for the held
/// prefill: a call whose prefill left its list is shed under one of the
/// answers above where one holds, and under the engine's own name for it
/// otherwise. The request reads the reports of `loads` where a pair's
/// first decision is given it ([`hands_reports`]).
///
/// A decode profile with a stage that may refuse by a report of leases is
/// decided behind an apply of what waits at the files of leases, where the
/// decider lent its place. The word asked is the decode phase's, which
/// holds the pair's decode profile's: no stage of the prefill's profile
/// runs at the call, so none of its can refuse.
///
/// `Ok` is the decision with what its side's lease takes of the frame: the
/// request's session of the decode phase and the prompt's tokens.
///
/// # Errors
///
/// [`DecisionRefused::Faulted`] where the decision panicked: the engine
/// caught it and counted it.
pub(crate) fn decide_decode(
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    (prefill, first): (WorkerId, Option<(WorkerId, u16)>),
    buffers: &mut FrameBuffers,
    loads: &Arc<LoadSnapshot>,
    own: &Part,
) -> Result<(Decision, (Session, u32)), DecisionRefused> {
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
    let (_, _, by_reports) = spec.engine().reads_for(model, Phase::Decode);
    if let Some(place) = place.filter(|_| by_reports) {
        place.apply_waiting();
    }
    let FrameBuffers {
        lists,
        decode_places,
        ..
    } = buffers;
    let read = lists.of(frame);
    // The target the pair holds: its prefill, and the decode its first
    // decision chose for it.
    let held = Target {
        id: prefill,
        pair: first.map(|(decode_id, decode_dp_rank)| Pair {
            decode_id,
            decode_dp_rank,
            ..Pair::default()
        }),
        ..Target::default()
    };
    let over_none = || FrameRequest::new(frame, &read, left, WorkerLists::None);
    let generation = objects.registry().model_generation(model);
    // The decodes as the decision reads them: the list their role lends,
    // and its records.
    let listed = generation.as_deref().map(|generation| {
        let lent = objects.lent(generation, Role::Decode);
        let records = front.members_of(lent).records(spec, front.clock());
        (generation, lent, records)
    });
    let decision = match listed.filter(|(.., records)| !records.is_empty()) {
        Some((generation, decode, records)) => {
            let reach = objects.reach(generation, Role::Decode);
            let vetoes = world.gateway_vetoes(model, Role::Decode, gateway);
            let asking = Asking::of(gateway, frame);
            let decided = match decode_places.taken(reach, records, asking, vetoes.standing_at(now))
            {
                Taken::Held(decodes) => {
                    let pair = PairLists {
                        prefill: objects.lent(generation, Role::Prefill),
                        decode,
                        classes: objects.pair_classes(generation, spec.pairing()),
                        admission: &OpenAdmission,
                        prefill_places: None,
                        decode_places: decodes.places,
                        prefill_stalled: &[],
                        prefill_held: &[],
                        decode_stalled: decodes.stalled,
                        decode_held: decodes.held,
                    };
                    let maker = FrameRequest::new(frame, &read, left, WorkerLists::Pair(pair));
                    let maker = if hands_reports(spec) {
                        maker.reading(loads)
                    } else {
                        maker
                    };
                    front.decide_decode_without_lease(&maker, &held, now)
                }
                Taken::Emptied(stage) => {
                    let (seed, generation) = (frame.seed(), spec.generation());
                    Ok(shed(ShedReason::NoEndpoints, stage, seed, generation))
                }
                Taken::Unreached => front.decide_decode_without_lease(&over_none(), &held, now),
            };
            // Counted whatever the decision came to.
            let undialed = decode_places.undialed();
            if undialed > 0 {
                own.undialed.fetch_add(undialed, Ordering::Relaxed);
            }
            decided?
        }
        // A model nobody serves, and one with no decode.
        None => front.decide_decode_without_lease(&over_none(), &held, now)?,
    };
    // What the side's lease takes of its request, read where the frame's
    // lists still stand: a decode call's frame is read at the decode phase.
    let booked = match decision.pick() {
        Some(_) => {
            let facts = frame.facts(&read, 0, 0);
            (facts.session(), facts.prompt_tokens)
        }
        None => (Session::NONE, 0),
    };
    Ok((decision, booked))
}

/// Files a decode call's pick under `lease`, the pair's own lease, and
/// tells the engine, under one hold of the file: the decode side on the
/// pick's first decode, under the request's session of the decode phase,
/// booking that decode's count. The side's lease comes with its issue's
/// count, for an answer that does not leave
/// ([`MemoryStore::unissue_decode`]). `None` for a shed, which files none.
///
/// The engine told is the one of the lease's own generation ([`issuing`]):
/// behind a reload the engine of this instant decided the side and hears
/// nothing of the lease. The issue is dated at this call
/// ([`issued_later`]). A stage that panics there is held and counted, as
/// under a report, and the caller still gets the pick the file holds.
///
/// # Errors
///
/// [`DecodeRefused`]: the lease moved since the call's first test.
pub(crate) fn issue_decode(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, Checked>,
    lease: LeaseId,
    decision: &Decision,
    (session, tokens): (Session, u32),
) -> Result<Option<(Lease, DecodeIssue)>, DecodeRefused> {
    // Each target of the decode decided again names its decode.
    let named = decision
        .pick()
        .and_then(|pick| Some((pick, pick.targets.first()?.pair?)));
    let Some((pick, chosen)) = named else {
        // A pick that named no decode would leave with no lease, its side
        // still chosen: the engine makes none.
        debug_assert!(
            decision.pick().is_none(),
            "a decode call's pick: {decision:?}"
        );
        return Ok(None);
    };
    let side = DecodeSide {
        target: chosen.decode_id,
        picked: PickRecord::of_decodes(&pick.targets),
        session,
        state: LeaseState::ISSUED,
        tokens,
        target_index: 0,
    };
    let slot = world
        .booked_of(chosen.decode_id)
        .or_else(|| store.filed_slot(chosen.decode_id));
    let spec = world.spec();
    let told = |lease: &Lease| {
        let Some(engine) = issuing(spec, lease) else {
            return;
        };
        let heard = || issued_later(world, engine, lease, decision, taken.now);
        match catch_unwind(AssertUnwindSafe(heard)) {
            Ok(()) => store.call_returned(),
            Err(payload) => {
                release(payload);
                // The first fault of a streak writes its line, as under a
                // gateway's frame.
                if store.call_faulted() {
                    error!(
                        lease = lease.id.0,
                        "a stage's call panicked at a decode call's issue: held, and the side stands as issued"
                    );
                }
            }
        }
    };
    let issued = store.issue_decided_decode(taken.gateway, lease, (side, slot), taken.now, told);
    issued.map(Some)
}

#[cfg(test)]
mod tests;
