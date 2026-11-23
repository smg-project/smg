//! What is said of a lease on file, told to the engine that issued it.
//!
//! A dispatch, a completion or a release reaches the stages of the
//! generation the lease was issued under, on the lease's record as the
//! store found it. Whoever applies the report calls [`told`]: a decider, a
//! session, the gateways' sweep.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    time::Instant,
};

use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{Decision, Pair, Pick, Target, Targets, Trace},
        decision_slot::DecisionSlot,
        framework::{release, Engine},
        view::WorkerId,
    },
    stages::lifecycle::Lifecycle,
    store::{GatewayId, HealthSink, MemoryStore, Sides, StoreSnapshot},
    tracking::lease::{Dispatch, Lease, Outcome, Said, Side},
};

/// Tells the engine what was said of a lease: one function of the record
/// found on file and the report, whoever calls it. `true` where a stage's
/// call panicked and was held, for the first of a streak: its host may
/// write a line for that one.
///
/// The engine told is the one of the lease's own generation, where
/// `world` holds that generation: a later generation's stages never saw
/// the issue and hear nothing of it. The caller loads `world` for this
/// word, or once for the words it tells in a row: behind a later
/// generation's start, a word told under such a load still reaches the
/// lease's own engine. A dispatch also feeds the counts of sends of the
/// generation `world` holds, whichever generation issued the lease: a
/// send is the host's, whatever document decided it
/// ([`Engine::dispatched_sent`]). A count that does not know the rank
/// the report names enters it only where `listed` says the worker
/// stands at that rank. A served controller answers by its replica's
/// worker objects, the lists its decisions read, so a late report
/// enters no rank behind a removal. A rank a count knows takes the send
/// unasked: a removal forgets the worker's ranks, and a stale rank of a
/// listed worker may take a late send that no row reads.
/// A dispatch is told on the target its report names by index,
/// as the record holds the pick, and the lease told names that index:
/// the record was read before the door wrote it. A report that names no
/// target of the pick tells nothing. A pair's decode side that stood
/// chosen is told its issue first ([`chosen_dispatched`]): no call
/// issued it, and it goes out as the first pick's.
///
/// A pair's lease is told by side: a dispatch on the side its frame names,
/// with that side's lease and on that side's worker, and an end once a
/// side the lease held live, the frame's side or with none every side
/// ([`Sides::ending`]).
///
/// The dispatch of a single request its sessions rule keeps from the
/// profile ([`LeaseRecord::untold`]) is told to the session table alone,
/// where the key's pin moves. No count of sends takes it, under whichever
/// generation `world` holds, and no stage hears it, as under a host that
/// holds its decisions (`lease_events::single`). Its end and its release
/// are told as any lease's.
///
/// Its caller is whoever applies the report, under the turn of the leases'
/// file: what it calls is bounded and takes no door of the leases. It
/// enters as the decider whose file the lease is on, on whichever thread
/// applies.
///
/// A stage's call that panics is held here, whoever applies: a decider
/// inside a decision, as it takes up or leaves its work or behind the
/// give-up of an answer its stream refused, a session, the gateways'
/// sweep, a host's own thread. The lease's event stands on file as
/// applied and the fault is counted on the store
/// ([`MemoryStore::lifecycle_faults`]). It costs no request its decision
/// and ends no thread.
///
/// [`LeaseRecord::untold`]: crate::store::LeaseRecord::untold
pub(crate) fn told(
    store: &MemoryStore,
    world: &StoreSnapshot,
    found: &Sides,
    said: Said,
    by: Option<DecisionSlot>,
    listed: impl Fn(WorkerId, u16) -> bool,
) -> bool {
    // The lease's own decider acts while its event is told, whichever
    // thread applies the report.
    let _acting = by.map(DecisionSlot::enter);
    let told = catch_unwind(AssertUnwindSafe(|| tell(world, found, said, &listed)));
    let Err(payload) = told else {
        store.call_returned();
        return false;
    };
    // A payload's own drop can panic: the engine's helper frees it under a
    // catch of its own.
    release(payload);
    store.call_faulted()
}

/// The engine that hears a lease's words under `spec`: the one of the
/// generation the lease was issued under, where that generation still
/// stands. A later generation's engine heard no issue of the lease and
/// hears nothing of it: no dispatch, no end, and no issue of a side a call
/// decides later.
#[inline]
pub(crate) fn issuing<'a>(spec: &'a AppliedSpec, lease: &Lease) -> Option<&'a Engine> {
    (spec.generation() == lease.generation).then(|| spec.engine())
}

/// What [`told`] tells, under its catch.
fn tell(world: &StoreSnapshot, found: &Sides, said: Said, listed: &impl Fn(WorkerId, u16) -> bool) {
    let spec = world.spec();
    let issuing = || issuing(spec, &found.record.lease);
    match said {
        Said::Dispatched(dispatch, at) => {
            // The side the frame names: its lease, and its worker at
            // the index the frame names.
            let Some((lease, stood)) = found.side(dispatch.side) else {
                return;
            };
            let named = found.target_of(dispatch.side, dispatch.target_index);
            let Some((id, dp_rank)) = named else {
                return;
            };
            let target = Target {
                id,
                dp_rank,
                ..Target::default()
            };
            // The record was read before the door wrote the dispatched
            // index: the lease told carries it, for a stage that reads
            // the index off the lease.
            let lease = Lease {
                target_index: dispatch.target_index,
                ..lease
            };
            // A request its sessions rule keeps from the profile is none
            // of the profile's sends, and no stage hears its dispatch.
            let own = !found.record.untold;
            if own {
                spec.engine().dispatched_sent(
                    &lease,
                    &target,
                    dispatch.side,
                    dispatch.since_decision_micros,
                    || listed(id, dp_rank),
                );
            }
            // The table hears first and the stages last. A stage's call
            // that unwinds there leaves the table's entry written.
            if let Some(engine) = issuing() {
                engine.dispatched_table(&lease, &target, dispatch.side);
                if stood.is_chosen() {
                    // A decode side no call issued goes out as the first
                    // pick's, and this dispatch books it: the stages hear
                    // its issue first, as they hear every side's.
                    chosen_dispatched(world, engine, found, &lease, &target, dispatch, at);
                } else if own {
                    engine.dispatched_stages(
                        &lease,
                        &target,
                        dispatch.side,
                        dispatch.since_decision_micros,
                    );
                }
            }
        }
        Said::Completed(outcome) => {
            if let Some(engine) = issuing() {
                let side = Some(outcome.side);
                found.each_ending(side, |lease| engine.on_completed(lease, &outcome));
            }
        }
        Said::Released(side, reason) => {
            if let Some(engine) = issuing() {
                found.each_ending(side, |lease| engine.on_released(lease, reason));
            }
        }
    }
}

/// Tells `engine`'s stages the issue of a pair's decode side the store
/// files behind the pair's first decision: at its decode call, or at the
/// dispatch that sends it with no call ([`chosen_dispatched`]).
///
/// One rule dates an issue: the instant the store files the side, on the
/// store's own clock. A side filed in its decision bears that decision's
/// instant. A side filed later is told here under its
/// lease dated `at`, the instant of that filing as its caller states it.
/// A served controller files a side later for a pair whose frame stated
/// the later decode: at its decode call, dated by the decider's read at
/// the call, or at the dispatch that sends the side with no call, dated by
/// the read of that dispatch's frame.
/// The lease on file keeps the instant of the pair's first decision for
/// its other words.
///
/// A stage that dates its bookings by its issues reads a date far behind
/// its newest as a clock that went back, and ends what it passed. So a
/// later issue bears neither the instant of that first decision nor the
/// date a gateway gives a dispatch.
pub(crate) fn issued_later(
    world: &StoreSnapshot,
    engine: &Engine,
    lease: &Lease,
    decision: &Decision,
    at: Instant,
) {
    let issued = Lease {
        issued_micros: world.micros(at),
        ..*lease
    };
    engine.on_decision_issued(&issued, decision);
}

/// Tells `engine`'s stages the dispatch of a pair's decode side that no
/// call issued, as the frame read at `at` names it: the side's issue
/// first ([`issued_later`]), then the dispatch's own word. The side goes
/// out as the first pick's and the store books it at this dispatch, so
/// the dispatch is the side's issue too: a stage that books at an issue
/// books the worker that serves, as it does at a decode call's issue.
///
/// The issue names the pair's pick as the lease's record keeps it: every
/// target's two workers and their ranks, with no trace beside the
/// generation. The record keeps no position, and each target bears the
/// default one, 0, as the target of a dispatch's word does: a stage reads
/// the worker by its id. The issue's lease names the target the dispatch
/// took and that target's worker as the one it books. The dispatch's
/// word then keeps the lease's own instant and names the same worker as
/// the one the issue booked: no stage books it twice.
///
/// Out of line: no other dispatch builds a pick, and the teller's frame
/// stays as short as it was.
#[inline(never)]
fn chosen_dispatched(
    world: &StoreSnapshot,
    engine: &Engine,
    found: &Sides,
    lease: &Lease,
    target: &Target,
    dispatch: Dispatch,
    at: Instant,
) {
    let named = |index| {
        let prefill = found.target_of(Side::Prefill, index)?;
        Some((prefill, found.target_of(Side::Decode, index)?))
    };
    let mut targets = Targets::new();
    for ((id, dp_rank), (decode_id, decode_dp_rank)) in (0..=u8::MAX).map_while(named) {
        targets.push(Target {
            id,
            dp_rank,
            pair: Some(Pair {
                decode_id,
                decode_dp_rank,
                ..Pair::default()
            }),
            ..Target::default()
        });
    }
    let decision = Decision::Pick(Pick {
        targets,
        trace: Trace {
            generation: lease.generation,
            ..Trace::default()
        },
        ..Pick::default()
    });
    let sent = Lease {
        booked: target.id,
        ..*lease
    };
    issued_later(world, engine, &sent, &decision, at);
    engine.dispatched_stages(&sent, target, dispatch.side, dispatch.since_decision_micros);
}

/// A completion, said whole: the lease's word goes where `hand` takes it, a
/// session's hand at the file, and the gateway's completion is written on
/// the served worker for the health precedence, whether or not the lease
/// is on file.
pub(crate) fn completed(
    store: &MemoryStore,
    gateway: GatewayId,
    outcome: &Outcome,
    now: Instant,
    hand: impl FnOnce(Said),
) {
    hand(Said::Completed(*outcome));
    // The sink writes for a worker it holds a row for and for no other.
    HealthSink::completed(store, gateway, outcome.served, outcome.class, now);
}
