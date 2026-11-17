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
//! issues none, so the one a `Released` names is one it does not hold. A
//! completion is written for a worker the store holds a row for and for
//! no other, under the stream's own gateway, and a gateway's completions
//! end with its presence: the health table grows with the rows and with
//! who is present, never with what a frame names.

use std::time::Instant;

use tonic::Status;

use crate::{
    lease::{LeaseId, Outcome, ReleaseReason, Side, TransportClass},
    proto::{
        self,
        completed::Class,
        released::{Reason, What},
    },
    store::{Dispatch, GatewayId, HealthSink, LeasesStore, MemoryStore},
    view::WorkerId,
};

/// A `Dispatched` frame: the gateway's lease moves to dispatched at `now`.
///
/// # Errors
///
/// The frame names no side, or a target index past a byte.
pub(crate) fn dispatched(
    store: &MemoryStore,
    gateway: GatewayId,
    frame: &proto::Dispatched,
    now: Instant,
) -> Result<(), Status> {
    let (lease, dispatch) = dispatch(frame)?;
    // A lease not on file is no error: the door's answer is not read.
    let _on_file: bool = LeasesStore::dispatched(store, gateway, lease, dispatch, now);
    Ok(())
}

/// A `Completed` frame: the lease's side ends with its outcome, and the
/// gateway's completion on the worker is written for the health
/// precedence, whether or not the lease is on file.
///
/// # Errors
///
/// The frame names no side or no class, or a status past sixteen bits.
pub(crate) fn completed(
    store: &MemoryStore,
    gateway: GatewayId,
    frame: &proto::Completed,
    now: Instant,
) -> Result<(), Status> {
    let (lease, outcome) = outcome(frame)?;
    let _on_file: bool = LeasesStore::completed(store, gateway, lease, &outcome);
    // The sink writes for a worker it holds a row for and for no other.
    HealthSink::completed(store, gateway, outcome.served, outcome.class, now);
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
    gateway: GatewayId,
    frame: &proto::Released,
) -> Result<(), Status> {
    let (lease, side, reason) = release(frame)?;
    if let Some(lease) = lease {
        let _on_file: bool = LeasesStore::released(store, gateway, lease, side, reason);
    }
    Ok(())
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
pub(crate) mod tests {
    use std::time::{Duration, Instant};

    use tonic::{Code, Status};

    use super::{class, completed, dispatch, dispatched, outcome, reason, release, released, side};
    use crate::{
        controller::gateways::tests::lease,
        lease::{LeaseId, Outcome, ReleaseReason, Side, TransportClass},
        proto::{
            self,
            completed::Class,
            released::{Reason, What},
        },
        store::{
            Dispatch, GatewayId, HealthSink, LeaseState, LeasesStore, MemoryStore, WorkerFacts,
            WorkersStore,
        },
        view::WorkerId,
    };

    const A: GatewayId = GatewayId(11);
    const B: GatewayId = GatewayId(12);

    /// A worker the store holds a row for.
    pub(crate) fn worker(store: &MemoryStore, key: &str) -> WorkerId {
        WorkersStore::upsert(
            store,
            WorkerFacts {
                key: key.to_owned(),
                engine: key.to_owned(),
                incarnation: 1,
                models: vec!["m".to_owned()],
                dp_size: 1,
                ..WorkerFacts::default()
            },
        )
        .id()
    }

    pub(crate) fn a_dispatched(lease: u64) -> proto::Dispatched {
        proto::Dispatched {
            lease_id: lease,
            target_index: 0,
            side: proto::Side::Single.into(),
            since_decision_micros: 250,
        }
    }

    pub(crate) fn a_completed(lease: u64, served: WorkerId) -> proto::Completed {
        proto::Completed {
            lease_id: lease,
            side: proto::Side::Single.into(),
            served: served.0,
            status: 200,
            class: Class::Ok.into(),
            ttft_micros: 1_000,
            tokens_in: 12,
            tokens_out: 34,
            duration_micros: 5_000,
        }
    }

    pub(crate) fn a_released(lease: u64) -> proto::Released {
        proto::Released {
            what: Some(What::LeaseId(lease)),
            reason: Reason::ClientGone.into(),
            side: proto::Side::Unspecified.into(),
        }
    }

    fn state(store: &MemoryStore, gateway: GatewayId, lease: u64) -> Option<LeaseState> {
        LeasesStore::get(store, gateway, LeaseId(lease)).map(|record| record.state)
    }

    #[test]
    fn a_dispatched_frame_moves_its_gateways_issued_lease_to_dispatched() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        let at = t0 + Duration::from_millis(3);
        dispatched(&store, A, &a_dispatched(1), at).expect("the frame reads");
        assert_eq!(state(&store, A, 1), Some(LeaseState::Dispatched { at }));
    }

    #[test]
    fn a_completed_frame_ends_the_lease_and_writes_the_gateways_completion() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let served = worker(&store, "http://a:8000");
        store.issue(lease(A, 1, t0)).expect("a lease");
        completed(&store, A, &a_completed(1, served), t0).expect("the frame reads");
        assert_eq!(state(&store, A, 1), None, "the lease ended");
        assert!(store.completed_ok_since(A, served, t0));
        assert!(
            !store.completed_ok_since(B, served, t0),
            "a completion is its gateway's own"
        );
    }

    /// The gateway reports every request, the ones it decided itself among
    /// them: the completion is a fact of the worker and the gateway.
    #[test]
    fn a_completed_frame_for_a_lease_not_on_file_still_writes_the_completion() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let served = worker(&store, "http://a:8000");
        completed(&store, A, &a_completed(9, served), t0).expect("no error");
        assert!(store.completed_ok_since(A, served, t0));
    }

    /// The health table grows with the rows, never with what a frame names.
    #[test]
    fn a_completion_for_a_worker_on_no_file_writes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let gone = WorkerId(999_999);
        completed(&store, A, &a_completed(9, gone), t0).expect("no error");
        assert!(!store.completed_ok_since(A, gone, t0));
    }

    #[test]
    fn a_released_frame_ends_the_lease() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        released(&store, A, &a_released(1)).expect("the frame reads");
        assert_eq!(state(&store, A, 1), None);
    }

    #[test]
    fn a_lease_not_on_file_is_no_error_and_changes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        dispatched(&store, A, &a_dispatched(2), t0).expect("no error");
        completed(&store, A, &a_completed(2, WorkerId(7)), t0).expect("no error");
        released(&store, A, &a_released(2)).expect("no error");
        assert_eq!(state(&store, A, 1), Some(LeaseState::Issued));
        assert_eq!(store.live(A), 1);
    }

    /// A ticket this build never issued is a ticket not on file: no error
    /// and nothing written, a lease under the ticket's number included.
    /// The frame is read whole all the same.
    #[test]
    fn a_released_ticket_is_no_error_and_writes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        let ticket = proto::Released {
            what: Some(What::Ticket(1)),
            ..a_released(1)
        };
        assert_eq!(
            release(&ticket).expect("the frame reads"),
            (None, None, ReleaseReason::ClientGone)
        );
        released(&store, A, &ticket).expect("no error");
        assert_eq!(state(&store, A, 1), Some(LeaseState::Issued));

        let no_reason = proto::Released {
            reason: 0,
            ..ticket
        };
        let status = released(&store, A, &no_reason).expect_err("the frame is read whole");
        assert_eq!(
            (status.code(), status.message()),
            (
                Code::InvalidArgument,
                "Released.reason is 0, which names none this version knows"
            )
        );
    }

    #[test]
    fn a_gateway_reaches_no_lease_of_another() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(B, 1, t0)).expect("a lease");
        dispatched(&store, A, &a_dispatched(1), t0).expect("no error");
        completed(&store, A, &a_completed(1, WorkerId(7)), t0).expect("no error");
        released(&store, A, &a_released(1)).expect("no error");
        assert_eq!(state(&store, B, 1), Some(LeaseState::Issued));
    }

    #[test]
    fn a_frame_reads_whole_into_what_the_store_is_handed() {
        let read = dispatch(&proto::Dispatched {
            lease_id: 5,
            target_index: 255,
            side: proto::Side::Decode.into(),
            since_decision_micros: 250,
        })
        .expect("the frame reads");
        assert_eq!(
            read,
            (
                LeaseId(5),
                Dispatch {
                    target_index: 255,
                    side: Side::Decode,
                    since_decision_micros: 250,
                }
            )
        );
        let read = outcome(&proto::Completed {
            lease_id: 6,
            side: proto::Side::Prefill.into(),
            served: 7,
            status: 65_535,
            class: Class::Timeout.into(),
            ttft_micros: 1,
            tokens_in: 2,
            tokens_out: 3,
            duration_micros: 4,
        })
        .expect("the frame reads");
        assert_eq!(
            read,
            (
                LeaseId(6),
                Outcome {
                    side: Side::Prefill,
                    served: WorkerId(7),
                    status: 65_535,
                    class: TransportClass::Timeout,
                    ttft_micros: 1,
                    tokens_in: 2,
                    tokens_out: 3,
                    duration_micros: 4,
                }
            )
        );
        // A release of no side names every side; a named side, its own.
        assert_eq!(
            release(&a_released(8)).expect("the frame reads"),
            (Some(LeaseId(8)), None, ReleaseReason::ClientGone)
        );
        let one = proto::Released {
            side: proto::Side::Prefill.into(),
            reason: Reason::Expired.into(),
            ..a_released(8)
        };
        assert_eq!(
            release(&one).expect("the frame reads"),
            (
                Some(LeaseId(8)),
                Some(Side::Prefill),
                ReleaseReason::Expired
            )
        );
    }

    #[test]
    fn every_side_class_and_reason_reads_as_its_own() {
        for (wire, read) in [
            (proto::Side::Single, Side::Single),
            (proto::Side::Prefill, Side::Prefill),
            (proto::Side::Decode, Side::Decode),
            (proto::Side::Encode, Side::Encode),
        ] {
            assert_eq!(side("a side", wire.into()).expect("a side"), read);
        }
        for (wire, read) in [
            (Class::Ok, TransportClass::Ok),
            (Class::ClientError, TransportClass::ClientError),
            (Class::ServerError, TransportClass::ServerError),
            (Class::ConnectFailed, TransportClass::ConnectFailed),
            (Class::Timeout, TransportClass::Timeout),
            (Class::Cancelled, TransportClass::Cancelled),
        ] {
            assert_eq!(class(wire.into()).expect("a class"), read);
        }
        for (wire, read) in [
            (Reason::Abandoned, ReleaseReason::Abandoned),
            (Reason::LateDecision, ReleaseReason::LateDecision),
            (Reason::ClientGone, ReleaseReason::ClientGone),
            (Reason::CommitFailed, ReleaseReason::CommitFailed),
            (Reason::Expired, ReleaseReason::Expired),
            (Reason::WorkerRemoved, ReleaseReason::WorkerRemoved),
        ] {
            assert_eq!(reason(wire.into()).expect("a reason"), read);
        }
    }

    #[test]
    fn a_frame_that_reads_wrong_is_refused_by_its_fields_name_and_writes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let served = worker(&store, "http://a:8000");
        store.issue(lease(A, 1, t0)).expect("a lease");
        let refused = |result: Result<(), Status>, code: Code, message: &str| {
            let status = result.expect_err("the frame is refused");
            assert_eq!((status.code(), status.message()), (code, message));
        };
        let none = |field: &str, number: i32| {
            format!("{field} is {number}, which names none this version knows")
        };
        for number in [0, 99] {
            let frame = proto::Dispatched {
                side: number,
                ..a_dispatched(1)
            };
            refused(
                dispatched(&store, A, &frame, t0),
                Code::InvalidArgument,
                &none("Dispatched.side", number),
            );
            let frame = proto::Completed {
                side: number,
                ..a_completed(1, served)
            };
            refused(
                completed(&store, A, &frame, t0),
                Code::InvalidArgument,
                &none("Completed.side", number),
            );
            let frame = proto::Completed {
                class: number,
                ..a_completed(1, served)
            };
            refused(
                completed(&store, A, &frame, t0),
                Code::InvalidArgument,
                &none("Completed.class", number),
            );
            let frame = proto::Released {
                reason: number,
                ..a_released(1)
            };
            refused(
                released(&store, A, &frame),
                Code::InvalidArgument,
                &none("Released.reason", number),
            );
        }
        let frame = proto::Released {
            side: 99,
            ..a_released(1)
        };
        refused(
            released(&store, A, &frame),
            Code::InvalidArgument,
            &none("Released.side", 99),
        );
        let frame = proto::Dispatched {
            target_index: 256,
            ..a_dispatched(1)
        };
        refused(
            dispatched(&store, A, &frame, t0),
            Code::InvalidArgument,
            "Dispatched.target_index is 256: 255 is the most it holds",
        );
        let frame = proto::Completed {
            status: 65_536,
            ..a_completed(1, served)
        };
        refused(
            completed(&store, A, &frame, t0),
            Code::InvalidArgument,
            "Completed.status is 65536: 65535 is the most it holds",
        );
        let frame = proto::Released {
            what: None,
            ..a_released(1)
        };
        refused(
            released(&store, A, &frame),
            Code::InvalidArgument,
            "Released.what names neither a lease nor a ticket",
        );
        // Nothing was written: the lease stands as issued and no
        // completion is on file.
        assert_eq!(state(&store, A, 1), Some(LeaseState::Issued));
        assert!(!store.completed_ok_since(A, served, t0));
    }
}
