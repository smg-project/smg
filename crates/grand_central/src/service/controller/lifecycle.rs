//! The lifecycle frames of a `Session` (design 4.3 and 5.2): `Dispatched`,
//! `Completed` and `Released`, a gateway's report of each request it sent
//! to a worker.
//!
//! The gateway reports every request on every worker, so no arm asks a
//! worker's row what else reports for it. The gateway is the stream's
//! principal: no field of a frame names one, and a lease is keyed by its
//! gateway, so a gateway reaches no lease of another.
//!
//! A frame is read whole before its first write. One that reads wrong is
//! refused by its field's name and writes nothing. A lease the store does
//! not hold is no error: its dispatch deadline may have ended it, or the
//! gateway decided that request itself. Nor is a ticket: this build
//! issues none, so the one a `Released` names is one it does not hold.
//!
//! What a frame says of a lease is handed to the leases' file, never
//! written by its session ([`MemoryStore::hand`]): the file's decider
//! applies it where it is at work or keeps its watch over the file, and
//! the session itself where it does neither, in the order the frames
//! were read. A `Decide` stream hands the release of a decision it stops
//! at its send through a hand of its own.
//!
//! A frame whose lease is on file tells the engine that issued it, the
//! generation's own, what became of the decision: its dispatch, its
//! completion or its release. Behind a later generation's start the
//! stages that one kept are told, and no stage it built new: such a stage
//! never saw the issue. A dispatch
//! alone goes further: it feeds the standing generation's counts of
//! sends, whichever generation issued the lease, where the replica's
//! worker objects still list the worker at the rank the dispatch names.
//!
//! A pair's lease stands under one id: its record is the prefill side,
//! and its decode side is filed beside the record. A frame moves the
//! side it names: the engine is told of that side alone,
//! with that side's lease and on that side's worker, and a `Released`
//! that names no side tells every side that stood. A frame on a side the
//! pair's lease does not hold, or on a side that ended, writes nothing
//! and tells nothing. A single's lease reads no side.

use std::{sync::Arc, time::Instant};

use tonic::Status;
use tracing::error;

use super::lease_words;
use crate::{
    engine::{decision_slot::DecisionSlot, view::WorkerId},
    service::proto::{
        self,
        completed::Class,
        released::{Reason, What},
    },
    store::{Hand, MemoryStore, Sides, StoreSnapshot},
    tracking::lease::{Dispatch, LeaseId, Outcome, ReleaseReason, Said, Side, TransportClass},
    worker_host::{Members, Objects},
};

/// A `Dispatched` frame: the gateway's lease moves to dispatched at `now`.
///
/// # Errors
///
/// The frame names no side, or a target index past a byte.
pub(crate) fn dispatched(
    store: &MemoryStore,
    hand: &Hand,
    frame: &proto::Dispatched,
    now: Instant,
) -> Result<(), Status> {
    let (lease, dispatch) = dispatch(frame)?;
    store.hand(hand, lease, Said::Dispatched(dispatch, now));
    Ok(())
}

/// A `Completed` frame: the lease's side ends with its outcome.
///
/// # Errors
///
/// The frame names no side or no class, or a status past sixteen bits.
pub(crate) fn completed(
    store: &MemoryStore,
    hand: &Hand,
    frame: &proto::Completed,
) -> Result<(), Status> {
    let (lease, outcome) = outcome(frame)?;
    store.hand(hand, lease, Said::Completed(outcome));
    Ok(())
}

/// A `Released` frame: the gateway gives its lease up, on one side or on
/// every side. A ticket is no error and writes nothing: a ticket this
/// build never issued is a ticket not on file.
///
/// # Errors
///
/// The frame names neither a lease nor a ticket, no reason, or a side
/// this version does not know.
pub(crate) fn released(
    store: &MemoryStore,
    hand: &Hand,
    frame: &proto::Released,
) -> Result<(), Status> {
    let (lease, side, reason) = release(frame)?;
    if let Some(lease) = lease {
        store.hand(hand, lease, Said::Released(side, reason));
    }
    Ok(())
}

/// Gives the store the word to the engine of a frame applied to a lease:
/// whoever applies the frame tells with it. The word holds `objects`, the
/// replica's worker objects, and asks them whether the worker a dispatch
/// names stands at its rank ([`Objects::lists_rank`]).
///
/// A replica gives it where its shared state is built, ahead of its
/// deciders' start and so before a session or a `Decide` stream opens a
/// hand. The first word a store is given stays, and the answer says
/// whether this one was kept: a replica's own must be.
pub(crate) fn tells(store: &MemoryStore, objects: &Arc<Objects<Members>>) -> bool {
    let objects = Arc::clone(objects);
    store.tell_through(move |store, world, found, said, by| {
        told(store, world, found, said, by, |id, rank| {
            objects.lists_rank(id, rank)
        });
    })
}

/// Tells the engine what a frame said of a lease, whoever applies the
/// frame: the lease's own word ([`lease_words::told`]), which holds a
/// stage's call that panics and counts it on the store
/// ([`MemoryStore::lifecycle_faults`]). A streak of faults writes one line
/// here. `listed` says whether a worker stands at a rank.
fn told(
    store: &MemoryStore,
    world: &StoreSnapshot,
    found: &Sides,
    said: Said,
    by: Option<DecisionSlot>,
    listed: impl Fn(WorkerId, u16) -> bool,
) {
    if lease_words::told(store, world, found, said, by, listed) {
        error!(
            lease = found.record.lease.id.0,
            "a lease's lifecycle call panicked: held, and the lease's event stands as applied"
        );
    }
}

/// What a `Dispatched` frame says, read whole.
fn dispatch(frame: &proto::Dispatched) -> Result<(LeaseId, Dispatch), Status> {
    Ok((
        LeaseId(frame.lease_id),
        Dispatch {
            target_index: u8::try_from(frame.target_index).map_err(|_| {
                past(
                    "Dispatched.target_index",
                    frame.target_index,
                    u8::MAX.into(),
                )
            })?,
            side: side("Dispatched.side", frame.side)?,
            since_decision_micros: frame.since_decision_micros,
        },
    ))
}

/// What a `Completed` frame says, read whole.
fn outcome(frame: &proto::Completed) -> Result<(LeaseId, Outcome), Status> {
    Ok((
        LeaseId(frame.lease_id),
        Outcome {
            side: side("Completed.side", frame.side)?,
            served: WorkerId(frame.served),
            status: u16::try_from(frame.status)
                .map_err(|_| past("Completed.status", frame.status, u16::MAX.into()))?,
            class: class(frame.class)?,
            ttft_micros: frame.ttft_micros,
            tokens_in: frame.tokens_in,
            tokens_out: frame.tokens_out,
            duration_micros: frame.duration_micros,
        },
    ))
}

/// What a `Released` frame says, read whole: the lease (`None` is a
/// ticket), the side given up (`None` is the frame's unspecified side,
/// every side) and the reason.
fn release(
    frame: &proto::Released,
) -> Result<(Option<LeaseId>, Option<Side>, ReleaseReason), Status> {
    let lease = match frame.what {
        Some(What::LeaseId(lease)) => Some(LeaseId(lease)),
        Some(What::Ticket(_)) => None,
        None => {
            return Err(Status::invalid_argument(
                "Released.what names neither a lease nor a ticket",
            ));
        }
    };
    let side = if frame.side == i32::from(proto::Side::Unspecified) {
        None
    } else {
        Some(side("Released.side", frame.side)?)
    };
    Ok((lease, side, reason(frame.reason)?))
}

/// The side a frame names.
fn side(field: &'static str, number: i32) -> Result<Side, Status> {
    match proto::Side::try_from(number) {
        Ok(proto::Side::Single) => Ok(Side::Single),
        Ok(proto::Side::Prefill) => Ok(Side::Prefill),
        Ok(proto::Side::Decode) => Ok(Side::Decode),
        Ok(proto::Side::Encode) => Ok(Side::Encode),
        Ok(proto::Side::Unspecified) | Err(_) => Err(none(field, number)),
    }
}

/// The transport class a `Completed` names.
fn class(number: i32) -> Result<TransportClass, Status> {
    match Class::try_from(number) {
        Ok(Class::Ok) => Ok(TransportClass::Ok),
        Ok(Class::ClientError) => Ok(TransportClass::ClientError),
        Ok(Class::ServerError) => Ok(TransportClass::ServerError),
        Ok(Class::ConnectFailed) => Ok(TransportClass::ConnectFailed),
        Ok(Class::Timeout) => Ok(TransportClass::Timeout),
        Ok(Class::Cancelled) => Ok(TransportClass::Cancelled),
        Ok(Class::Unspecified) | Err(_) => Err(none("Completed.class", number)),
    }
}

/// The reason a `Released` names.
fn reason(number: i32) -> Result<ReleaseReason, Status> {
    match Reason::try_from(number) {
        Ok(Reason::Abandoned) => Ok(ReleaseReason::Abandoned),
        Ok(Reason::LateDecision) => Ok(ReleaseReason::LateDecision),
        Ok(Reason::ClientGone) => Ok(ReleaseReason::ClientGone),
        Ok(Reason::CommitFailed) => Ok(ReleaseReason::CommitFailed),
        Ok(Reason::Expired) => Ok(ReleaseReason::Expired),
        Ok(Reason::WorkerRemoved) => Ok(ReleaseReason::WorkerRemoved),
        Ok(Reason::Unspecified) | Err(_) => Err(none("Released.reason", number)),
    }
}

/// The refusal of a field whose number names none of its kind in this
/// version.
fn none(field: &'static str, number: i32) -> Status {
    Status::invalid_argument(format!(
        "{field} is {number}, which names none this version knows"
    ))
}

/// The refusal of a number past its field's range.
fn past(field: &'static str, number: u32, most: u32) -> Status {
    Status::invalid_argument(format!("{field} is {number}: {most} is the most it holds"))
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/lifecycle/tests/mod.rs"]
pub(crate) mod tests;
