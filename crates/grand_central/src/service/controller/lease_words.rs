//! What is said of a lease on file, told to the stages that issued it.
//!
//! A dispatch, a completion or a release reaches the stages that stood
//! when the lease was issued, on the lease's record as the store found it:
//! the generation's own, or the ones a later generation kept of them
//! ([`issuing`]). The generation asked is the one in force at the word
//! ([`told`]). Whoever applies the report calls [`told`]: a decider, a
//! session, the gateways' sweep.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    time::Instant,
};

use crate::{
    engine::{
        decision::Decision,
        decision_slot::DecisionSlot,
        framework::{release, Issuing, Stood},
        lease_events::{self, Hearers},
        view::WorkerId,
    },
    store::{MemoryStore, Sides, StoreSnapshot},
    tracking::lease::{Lease, Said},
};

/// Tells the engine what was said of a lease: one function of the record
/// found on file and the report, whoever calls it. `true` where a stage's
/// call panicked and was held, for the first of a streak: its host may
/// write a line for that one.
///
/// **The generation a word is told by** is the one in force at the word:
/// the generation of the store's standing world. The caller loads `world`
/// for this word, or once for the words it tells in a row, so a
/// generation may start behind that load. One atomic load a word says
/// whether `world` still holds the generation in force on the store's
/// engine (`Loaded::still_in_force`). Where it does, the word is told by
/// `world`, and nothing is loaded. Where a generation started since, the
/// store's standing world is loaded for this word and the word is told
/// by it. Between the engine's turn of an apply and the publication of
/// that apply's world, the standing world still holds the generation
/// before: a word said there is told by it. No lease of the new
/// generation exists by then, for a decision reads its generation off a
/// published world.
///
/// The word reads the generation off a world and never off the engine's
/// own cell. A reader's hold of a world is never the last one: the store's
/// writer keeps a replaced world its reader still holds. So no telling
/// frees a replaced generation's stages on its own thread, however many
/// generations start while it runs. A hold of the engine's cell has no
/// such keeper.
///
/// The stages told are the ones of the lease's own generation, where that
/// generation is in force. Under a later generation they are what that
/// generation kept of the stages that stood at the issue ([`issuing`]): a
/// stage built since never saw the issue and hears nothing of it. A lease
/// of a generation above the one in force tells no stage: no stage of
/// this replica booked it. A dispatch also enters the counts of sends of
/// the generation in force, whichever generation issued the lease: a
/// send is the host's, whatever document decided it
/// ([`Engine::dispatched_sent`](crate::engine::Engine::dispatched_sent)).
/// A count that does not know the rank
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
/// chosen is told its issue first: no call issued it, and it goes out as
/// the first pick's, dated by the read of the dispatch's frame on the
/// store's clock.
///
/// A lease's words are told by the rule a host that holds its decisions
/// tells them by ([`lease_events::single`], [`lease_events::pair`]), over
/// the lease as the file lends it ([`Sides::with_held_single`],
/// [`Sides::with_held_pair`]). A pair's lease is told by side: a dispatch
/// on the side its frame names, with that side's lease and on that side's
/// worker, and an end once a side the lease held live, the frame's side
/// or with none every side.
///
/// So the dispatch of a single request its sessions rule keeps from the
/// profile ([`LeaseRecord::untold`]) is told to the session table alone,
/// where the key's pin moves. No count of sends takes it, under whichever
/// generation is in force, and no stage hears it. Its end and its
/// release are told as any lease's.
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
    // By the generation in force at the word: the caller's world where
    // it holds that generation still, and the standing world otherwise.
    let word = || {
        if world.front().still_in_force() {
            tell(world, found, said, &listed);
        } else {
            store.with_snapshot(|standing| tell(standing, found, said, &listed));
        }
    };
    let told = catch_unwind(AssertUnwindSafe(word));
    let Err(payload) = told else {
        store.call_returned();
        return false;
    };
    // A payload's own drop can panic: the engine's helper frees it under a
    // catch of its own.
    release(payload);
    store.call_faulted()
}

/// The engine that hears a lease's words under `world`, read off the
/// world's own hold of its generation (`Loaded::hearing`). For a lease of
/// the world's own generation it is that generation's engine. For a lease
/// of an earlier one it is what the world's generation kept of the stages
/// that stood when the lease was issued, over the session table, which is
/// the store's engine's for every generation: the lease's claim was made
/// in it. A stage the generation built new heard no issue of the lease
/// and hears nothing of it, no dispatch, no end, and no issue of a side a
/// call decides later. Nobody hears a lease of a generation above the
/// world's. On one replica none is on file: a run loads its world behind
/// every issue on its file, and the store's generations only rise.
#[inline]
pub(crate) fn issuing<'a>(world: &'a StoreSnapshot, lease: &Lease) -> Option<Stood<'a>> {
    world.front().hearing(lease.generation)
}

/// What [`told`] tells, under its catch, by the generation `world` holds.
/// A lease's words go by the
/// engine's one rule over the lease as the file lends it: a single's by
/// [`lease_events::single`] over [`Sides::with_held_single`], and a pair's
/// by [`lease_events::pair`] over [`Sides::with_held_pair`], a side at a
/// time. This function says which engines hear, and by which clock the
/// issue of a decode side sent with no call is dated: the store's own, at
/// the read of the dispatch's frame. It says nothing more.
fn tell(world: &StoreSnapshot, found: &Sides, said: Said, listed: &impl Fn(WorkerId, u16) -> bool) {
    let hearers = Hearers {
        issuing: issuing(world, &found.record.lease),
        sends: world.spec().engine(),
        listed,
    };
    if found.decode.is_none() {
        found.with_held_single(said, |held| lease_events::single(&hearers, held, said));
        return;
    }
    let micros = |at| world.micros(at);
    found.with_held_pair(said, |held, said| {
        lease_events::pair(&hearers, held, said, micros);
    });
}

/// Tells `engine`'s stages the issue of a pair's decode side a decode call
/// files behind the pair's first decision.
///
/// One rule dates an issue: the instant the store files the side, on the
/// store's own clock. A side filed in its decision bears that decision's
/// instant. A side filed later is told under its lease dated at that
/// filing. A served controller files a side later for a pair whose frame
/// stated the later decode. At its decode call the issue is told here,
/// dated `at`, the decider's read at the call. At the dispatch that sends
/// the side with no call the engine's rule tells it
/// ([`lease_events::pair`]), dated by the read of that dispatch's frame on
/// the same clock ([`tell`]).
/// The lease on file keeps the instant of the pair's first decision for
/// its other words.
///
/// A stage that dates its bookings by its issues reads a date far behind
/// its newest as a clock that went back, and ends what it passed. So a
/// later issue bears neither the instant of that first decision nor the
/// date a gateway gives a dispatch.
pub(crate) fn issued_later(
    world: &StoreSnapshot,
    engine: Stood<'_>,
    lease: &Lease,
    decision: &Decision,
    at: Instant,
) {
    let issued = Lease {
        issued_micros: world.micros(at),
        ..*lease
    };
    engine.decision_issued(&issued, decision);
}
