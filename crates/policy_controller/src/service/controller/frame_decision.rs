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
//! The engine that decides and the stages that are told are one object:
//! the world's ([`StoreSnapshot::front`]). A lease is dated on that
//! engine's clock, the one its decisions read.
//!
//! A request of any other phase has no decision here yet: its deciding
//! thread answers it `Busy` before anything is read for it.

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
        decision::{shed, Decision, ShedReason},
        decision_engine::{DecisionRefused, WorkerLists},
        shadow::Shadowed,
        view::{fingerprint, Phase, Role, Session},
    },
    stages::lifecycle::Lifecycle,
    store::{LeaseRecord, LeaseRefused, LeaseState, MemoryStore, PickRecord, StoreSnapshot},
    tracking::lease::{Lease, LeaseId, LeaseKind, Side},
    worker_host::{Members, Objects},
};

/// A deciding thread's buffers for its frames, kept from frame to frame: a
/// frame's lists in the engine's form, and the places its gateway may take.
#[derive(Debug, Default)]
pub(crate) struct FrameBuffers {
    lists: Lists,
    places: GatewayPlaces,
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
    let FrameBuffers { lists, places } = buffers;
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

/// Files the lease of `decision`'s pick, one a decision, ahead of the read
/// that lets its answer leave: a credit on the pick's first target, minted
/// and filed by the store under `term`, the fence's at the decision's
/// first read, at that read's instant. The lease books its target's count
/// of the leases on file, and the document's stages are told of it under
/// the store's hold of the file's turn. `None` for a shed, which files
/// none.
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
    let lease = Lease {
        id: LeaseId::default(),
        kind: LeaseKind::Credit,
        side: Side::Single,
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
    let record = LeaseRecord {
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
    };
    let told = |lease: &Lease| stages.on_decision_issued(lease, decision);
    // The slot of the pick's count, as the decision's world names it. A
    // member the worker objects list ahead of that world, inside its
    // registration window, is read in the table its write filed it in:
    // its lease books its count as any other's does.
    let booked = world
        .booked_of(first.id)
        .or_else(|| store.filed_slot(first.id));
    let lease = store.issue_decided(record, booked, told)?;
    front.count_shadow(said);
    Ok(Some(lease))
}

#[cfg(test)]
mod tests;
