//! The lifecycle of a stage: the calls that tell a stateful stage what became
//! of a decision, of the membership and of a worker's signals. Every call has
//! a default, so a stateless stage writes none.
//!
//! What a call carries (the lease, the outcome, the membership change, the
//! signal) is in `tracking`.

use crate::{
    engine::decision::{Decision, Target},
    tracking::lease::{Lease, Membership, Outcome, ReleaseReason, Side, Signal},
};

/// The lifecycle a stateful stage implements. Every hook defaults to a
/// no-op, so a stateless stage implements nothing.
///
/// A served controller tells a lease's issue, its dispatch, its completion
/// and its release while it holds the turn of the lease's file, and the
/// release of a lease its decision gives up right after that turn: those
/// hooks are bounded, and call no door of the leases' store. A hook that
/// panics at a dispatch, a completion or a release that a frame brings,
/// or at a dispatch deadline's end, is held there: the lease's
/// event stands as applied, and so does one that panics at the issue a
/// dispatch or a decode call brings: that call still hands its host the
/// pick on file. One that panics at another decision's issue, or at the
/// release of a lease its decision gives up, costs that request its
/// decision.
pub trait Lifecycle: Send + Sync {
    /// The engine issued `decision` under `lease`. Its host filed the
    /// lease as it issued it, in process as behind a served controller:
    /// the lease books its first target from here.
    ///
    /// A pair's decode side whose request asked for it later is not issued
    /// with the pair, and no call here names it then. A decode call issues
    /// it and is told here, where the lease's generation still stands. A
    /// side no call issued goes out as the first pick's, and its dispatch
    /// is its issue: a store that files leases tells it here just ahead of
    /// that dispatch's own word. The decision told then is the pair's pick
    /// as the lease's record keeps it: every target's workers and ranks,
    /// with no trace. The record keeps no position, and each target bears
    /// the default one, 0, as the target of a dispatch's word does: a
    /// stage reads the worker by its id. The lease names the target the
    /// dispatch took.
    ///
    /// A lease told here bears the instant its side was filed, on the
    /// clock of every `issued_micros`. A later decode's is the instant of
    /// its decode call, or the instant its dispatch was told: never the
    /// pair's first decision, and never the date a gateway gives the
    /// dispatch. In process the host states that instant with its call.
    /// The lease's later words keep the instant of the pair's first
    /// decision.
    fn on_decision_issued(&self, _lease: &Lease, _decision: &Decision) {}

    /// The host dispatched to `target` (one of the decision's targets), on
    /// `side`, `since_decision_micros` after the decision on the host's clock
    /// (the wire's own field): a credit dates from the dispatch, not from a
    /// decision a queue may have held. The untaken targets are released by
    /// the host's commit. The lease told names the target the dispatch
    /// took: its `target_index` is that target's index in the pick, and its
    /// [`Lease::booked`] is still the worker its issue booked. A decode
    /// side no call issued was told issued just ahead of this word, on the
    /// target it names ([`Self::on_decision_issued`]). That word was dated
    /// at its telling. This one, and every later word of the lease, keeps
    /// the decision's `issued_micros`, so `issued_micros +
    /// since_decision_micros` still dates the dispatch.
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

    /// The host's membership changed. A host may call this under its own
    /// membership lock, as the policy store does: a stage that calls from
    /// here a door of the host that takes that lock waits on itself. A read
    /// that takes no lock does not.
    fn on_membership(&self, _change: &Membership<'_>) {}

    fn on_signal(&self, _signal: &Signal) {}

    /// Whether the stage keeps a number of each rank's report, which a host
    /// tells it before it shows a decision that report ([`Signal::Load`]
    /// with the rank's slot and the report's sample instant). `false` by
    /// default: a host tells no report to an engine none of whose stages
    /// keeps one, and such a stage decides the same untold. The engine asks
    /// once, at its build: the answer is the stage's for the engine's life.
    /// A host tells such a stage under its own locks, as the policy store
    /// does under its membership and load locks: a stage that calls from
    /// that word a door of the host that takes one of them waits on itself.
    fn keeps_loads(&self) -> bool {
        false
    }

    /// Whether the stage decides wrong under a host that makes none of
    /// these calls: `true` for one whose state the calls keep, as the
    /// workers it lists or the bookings it learns from. The standalone
    /// controller makes the membership call and tells a stage of a lease's
    /// issue, of its dispatch, completion and release at its gateway's word
    /// and of its end at the dispatch deadline or where its answer does not
    /// leave, and not yet of the end of a lease whose gateway was let go: it
    /// refuses a document that names such a stage as not in this build
    /// ([`crate::document::SpecError::NotInBuild`]). A host that embeds the engine
    /// makes every call. `false` by default: a stage that takes a call to
    /// shed state early decides the same without it.
    fn needs_lifecycle(&self) -> bool {
        false
    }
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

    fn keeps_loads(&self) -> bool {
        (**self).keeps_loads()
    }

    fn needs_lifecycle(&self) -> bool {
        (**self).needs_lifecycle()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::engine::view::{Health, WorkerId};

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

    /// A stage whose state the calls keep.
    struct Kept;

    impl Lifecycle for Kept {
        fn needs_lifecycle(&self) -> bool {
            true
        }
    }

    #[test]
    fn a_boxed_stage_says_what_its_stage_says_of_the_calls() {
        let kept: Box<dyn Lifecycle> = Box::new(Kept);
        assert!(Lifecycle::needs_lifecycle(&kept));
        let stateless: Box<dyn Lifecycle> = Box::new(Stateless);
        assert!(!Lifecycle::needs_lifecycle(&stateless));
    }

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
}
