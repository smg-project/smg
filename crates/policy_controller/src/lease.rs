//! The lifecycle (design 4.3): how a stateful stage learns what became of a
//! decision.
//!
//! A lease records that a decision was issued and is expected to be
//! dispatched and completed. A `credit` is soft load accounting; a
//! `reservation` is a prefill slot or a decode room the host holds
//! exclusively. A pair decision carries one lease per side, and the host
//! calls the lifecycle once per side. Every stateful built-in keeps its state
//! behind these events alone.

use crate::{
    decision::{Decision, Target},
    sessions::SessionVerdict,
    view::{Health, LoadRow, WorkerId},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeaseId(pub u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeaseKind {
    /// Soft load accounting. The built-in least-load scorer releases it at the
    /// next report sampled after the dispatch; release at completion and at
    /// the dispatch deadline arrive with the lifecycle step.
    #[default]
    Credit,
    /// A prefill slot or a decode room, released by its side's completion.
    Reservation,
}

/// Which side of a dispatch a lease or an event is about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// The record of one issued decision, or of one side of a pair decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lease {
    pub id: LeaseId,
    pub kind: LeaseKind,
    /// The side this lease books; `Single` outside a pair.
    pub side: Side,
    /// Which target of the pick this lease books (a reservation on a pair
    /// side names its own).
    pub target_index: u8,
    pub generation: u64,
    /// The fingerprint of the session key, so a sticky in-flight cap counts
    /// per key per worker from leases; `0` when none.
    pub session_key: u64,
    /// What the decision did with the session key, copied from its pick, so
    /// the commit moves the key's pins by its own decision alone.
    pub session: SessionVerdict,
    /// The host's clock at issue, microseconds since its origin. In-process
    /// only; the wire carries the dispatch deadline as a duration.
    pub issued_micros: u64,
    /// Token-work the decision books on its target: the prompt, or the
    /// host's mean estimate when the prompt's size is unknown.
    pub tokens: u32,
}

/// The transport outcome of a dispatch, beside its status: what a health
/// rule reads when it weighs a prober's verdict against the gateways'
/// reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportClass {
    Ok,
    ClientError,
    ServerError,
    ConnectFailed,
    Timeout,
    Cancelled,
}

/// What a completed dispatch reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub side: Side,
    pub served: WorkerId,
    /// The HTTP status or the gRPC code the request ended with.
    pub status: u16,
    pub class: TransportClass,
    pub ttft_micros: u32,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub duration_micros: u32,
}

/// Why a lease was released before its completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseReason {
    /// The host gave up before dispatch.
    Abandoned,
    /// A late decision arrived after a fallback dispatch.
    LateDecision,
    /// The client went away.
    ClientGone,
    /// The commit refused every target (a removed worker, a lost room).
    CommitFailed,
}

/// A membership change the host reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Membership<'a> {
    Added(WorkerId),
    Removed(WorkerId),
    /// The same registration key with a new incarnation.
    Replaced {
        old: WorkerId,
        new: WorkerId,
    },
    Relabelled(WorkerId),
    /// The worker's data-parallel width changed.
    Resized(WorkerId),
    /// Every worker of every pool the engine decides over, after a resync;
    /// a host with a prefill and a decode pool sends one snapshot of both.
    Snapshot(&'a [WorkerId]),
}

/// A signal about one worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// A load row for one rank of a worker.
    Load {
        worker: WorkerId,
        dp_rank: u16,
        row: LoadRow,
    },
    Health {
        worker: WorkerId,
        health: Health,
    },
    /// A gateway's standing veto on a worker, until the given instant on the
    /// host's clock. In-process only; the wire carries a duration.
    GatewayVeto {
        worker: WorkerId,
        until_micros: u64,
    },
}

/// The lifecycle a stateful stage implements. Every method defaults to a
/// no-op, so a stateless stage implements nothing.
pub trait Lifecycle: Send + Sync {
    /// The engine issued `decision` under `lease`. In the embedded mode the
    /// commit follows at once and the accounting happens at dispatch; a
    /// remote controller books here.
    fn on_decision_issued(&self, _lease: &Lease, _decision: &Decision) {}

    /// The host dispatched to `target` (one of the decision's targets), on
    /// `side`, `since_decision_micros` after the decision on the host's clock
    /// (the wire's own field): a credit dates from the dispatch, not from a
    /// decision a queue may have held. The untaken targets are released by
    /// the host's commit.
    fn on_dispatched(
        &self,
        _lease: &Lease,
        _target: &Target,
        _side: Side,
        _since_decision_micros: u32,
    ) {
    }

    /// A dispatched side finished, whatever the status.
    fn on_completed(&self, _lease: &Lease, _outcome: &Outcome) {}

    /// The host gave the lease up before its completion.
    fn on_released(&self, _lease: &Lease, _reason: ReleaseReason) {}

    fn on_membership(&self, _change: &Membership<'_>) {}

    fn on_signal(&self, _signal: &Signal) {}
}

/// A boxed stage forwards to the stage it holds.
impl<T: Lifecycle + ?Sized> Lifecycle for Box<T> {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        (**self).on_decision_issued(lease, decision);
    }

    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        (**self).on_dispatched(lease, target, side, since_decision_micros);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        (**self).on_completed(lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        (**self).on_released(lease, reason);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        (**self).on_membership(change);
    }

    fn on_signal(&self, signal: &Signal) {
        (**self).on_signal(signal);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A stage that counts what reached it, to prove the box forwards every
    /// hook and a stateless stage needs none.
    #[derive(Default)]
    struct Counting(AtomicUsize);

    impl Lifecycle for Counting {
        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }

        fn on_signal(&self, _: &Signal) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct Stateless;

    impl Lifecycle for Stateless {}

    #[test]
    fn a_boxed_stage_forwards_its_hooks_and_a_stateless_one_ignores_them() {
        let counting = Box::new(Counting::default());
        let lease = Lease::default();
        Lifecycle::on_released(&counting, &lease, ReleaseReason::ClientGone);
        Lifecycle::on_signal(
            &counting,
            &Signal::Health {
                worker: WorkerId(1),
                health: Health::default(),
            },
        );
        Lifecycle::on_membership(&counting, &Membership::Resized(WorkerId(1)));
        assert_eq!(
            counting.0.load(Ordering::Relaxed),
            2,
            "the two implemented hooks reached the stage through the box; the membership hook is its default no-op"
        );
        let stateless: Box<dyn Lifecycle> = Box::new(Stateless);
        stateless.on_released(&lease, ReleaseReason::Abandoned);
        stateless.on_dispatched(&lease, &Target::default(), Side::Prefill, 0);
    }

    #[test]
    fn a_pair_books_one_lease_per_side_on_its_own_target() {
        let prefill = Lease {
            side: Side::Prefill,
            target_index: 0,
            kind: LeaseKind::Reservation,
            ..Lease::default()
        };
        let decode = Lease {
            side: Side::Decode,
            target_index: 0,
            ..prefill
        };
        assert_ne!(prefill.side, decode.side);
        assert_eq!(prefill.target_index, decode.target_index);
        assert_eq!(decode.kind, LeaseKind::Reservation);
    }
}
