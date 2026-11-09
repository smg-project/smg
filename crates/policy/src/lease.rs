//! The lifecycle: how a stateful stage learns what became of a decision.
//!
//! A lease records that a decision was issued and is expected to be
//! dispatched and completed. `credit` is soft load accounting; `reservation`
//! is a prefill slot or a decode room the host holds exclusively. Every
//! stateful built-in keeps its state behind these events alone.

use crate::{
    decision::{Decision, Target},
    view::{Health, LoadReport, WorkerId},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeaseId(pub u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeaseKind {
    /// Soft load accounting, released by completion, by a report sampled
    /// after the dispatch, or by the dispatch deadline.
    #[default]
    Credit,
    /// A prefill slot or a decode room, released by its side's completion.
    Reservation,
}

/// Which side of a dispatch an event is about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// The record of one issued decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lease {
    pub id: LeaseId,
    pub kind: LeaseKind,
    pub generation: u64,
    /// The session key, hashed, so a sticky in-flight cap counts per key.
    pub session_key: u64,
    /// The host's clock at issue, microseconds since its origin.
    pub issued_micros: u64,
    /// Token-work the decision books on its target: the prompt, or the
    /// host's mean estimate when the prompt's size is unknown.
    pub tokens: u32,
}

/// What a completed dispatch reported.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub side: Side,
    pub served: WorkerId,
    pub status: u16,
    pub ttft_micros: u32,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub duration_micros: u32,
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
    /// The whole pool, after a resync.
    Snapshot(&'a [WorkerId]),
}

/// A signal about one worker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Signal {
    LoadReport {
        worker: WorkerId,
        report: LoadReport,
    },
    Health {
        worker: WorkerId,
        health: Health,
    },
    /// A gateway's standing veto on a worker, until the given instant.
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
    /// remote host books here.
    fn on_decision_issued(&self, _lease: &Lease, _decision: &Decision) {}

    /// The host dispatched to `target` (one of the decision's targets), on
    /// `side`. The untaken targets are released by the host's commit.
    fn on_dispatched(&self, _lease: &Lease, _target: &Target, _side: Side) {}

    /// A dispatched side finished, whatever the status.
    fn on_completed(&self, _lease: &Lease, _outcome: &Outcome) {}

    /// The host gave the lease up before dispatch, or a late decision
    /// arrived after a fallback.
    fn on_released(&self, _lease: &Lease) {}

    fn on_membership(&self, _change: &Membership<'_>) {}

    fn on_signal(&self, _signal: &Signal) {}
}

/// A lifecycle that fans out to several stages.
impl<T: Lifecycle + ?Sized> Lifecycle for Box<T> {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        (**self).on_decision_issued(lease, decision);
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side) {
        (**self).on_dispatched(lease, target, side);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        (**self).on_completed(lease, outcome);
    }

    fn on_released(&self, lease: &Lease) {
        (**self).on_released(lease);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        (**self).on_membership(change);
    }

    fn on_signal(&self, signal: &Signal) {
        (**self).on_signal(signal);
    }
}
