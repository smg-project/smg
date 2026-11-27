//! The lifecycle of a stage: the calls that tell a stateful stage what became
//! of a decision, of the membership and of a worker's signals. Every call has
//! a default, so a stateless stage writes none.
//!
//! What a call carries (the lease, the outcome, the membership change, the
//! signal) is in `tracking`.

use crate::{
    engine::decision::{Decision, Target},
    tracking::lease::{EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal},
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
///
/// The engine of a gateway's own process tells every word of a lease
/// under a catch, the issue of a first decision among them. A hook that
/// panics at one is held and counted, the stages behind it do not hear
/// that word, and the request keeps its decision and its lease. A stage
/// may so hear a lease's dispatch and its end with no issue ahead of them:
/// it matches them by the lease's id, as below.
///
/// Under a sessions rule that says so (`rule_placed: untold`), a stage
/// hears neither the issue nor the dispatch of a single request the rule
/// placed alone: a pin of the request's key followed, or the rule's own
/// assignment. No stage placed that request, as a gateway's sticky map
/// keeps a request it places from the gateway's policy. Its completion and
/// its release are told as any request's, so a stage hears an end whose
/// issue and dispatch it did not hear: a stage matches an end by the
/// lease's id ([`Lease::id`]), and moves no count by an end it cannot
/// match. It is so in a gateway's own process and behind a controller,
/// which keeps the rule's word on the lease it files. A side of a pair
/// the rule placed alone is kept from its stages in the same way, each
/// side by its own decision, in a gateway's own process: a controller
/// takes `told` alone under `pd-pair`.
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

    /// The host gave the lease up before its completion. The lease told
    /// names the worker the stages last heard the side on
    /// ([`Lease::booked`]): the one its issue booked, or behind a dispatch
    /// the one the side went to. A stage undoes what it holds of the lease
    /// by that worker. A stage reads no index off a release: behind a
    /// dispatch a single's names the target its dispatch took, and a
    /// pair's decode side is told under index 0, whichever target its
    /// dispatch took. That side states its claim
    /// ([`Lease::session_verdict`]) where it stands on the worker it
    /// claimed, and none behind a dispatch on another worker.
    fn on_released(&self, _lease: &Lease, _reason: ReleaseReason) {}

    /// The host's membership changed. A host may call this under its own
    /// membership lock, as the policy store does: a stage that calls from
    /// here a door of the host that takes that lock waits on itself. A read
    /// that takes no lock does not.
    fn on_membership(&self, _change: &Membership<'_>) {}

    /// A signal about one worker: a load row, a health change or a
    /// gateway's veto. A load row lends its report's named values for
    /// this call alone ([`Signal::Load`]): a stage copies out what it
    /// keeps, and a cell is `Copy`. The values it was lent end with the
    /// call.
    fn on_signal(&self, _signal: &Signal<'_>) {}

    /// Whether the stage keeps a number of each rank's report, which a host
    /// tells it ([`Signal::Load`] with the report's sample instant, and with
    /// the report's named values where the host states them). The engine
    /// over a host's worker lists tells a report that is new before it keeps
    /// the report's snapshot for its decisions, and its own doc names the
    /// reports a decision can read first. `false` by default: a host tells
    /// no report to an engine none of whose stages keeps one, and such a
    /// stage decides the same untold. The engine asks once, at its build:
    /// the answer is the stage's for the engine's life. A host may tell
    /// such a stage under its own locks: a stage that calls from that word a
    /// door of the host that takes one of them waits on itself.
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
    /// makes every call, and under a sessions rule that says so it leaves
    /// out the two the trait's own note names. `false` by
    /// default: a stage that takes a call to shed state early decides the
    /// same without it.
    fn needs_lifecycle(&self) -> bool {
        false
    }

    /// The kinds of event the stage takes ([`EventSet`]). By default a
    /// stage that says it needs the lifecycle's calls
    /// ([`Self::needs_lifecycle`]) takes every word on a lease
    /// ([`EventSet::LEASE`]: the issue, the dispatch, the completion and
    /// the release), and any other stage takes none.
    ///
    /// The engine cannot see an overridden method, and a host that decides
    /// in its own process tells a word on a lease only where a stage of
    /// the document takes it, and keeps nothing for a word nobody takes: a
    /// document none of whose stages takes one costs a request no lease.
    /// So a stage that implements [`Self::on_decision_issued`],
    /// [`Self::on_dispatched`], [`Self::on_completed`] or
    /// [`Self::on_released`] and declares no need names that call's kind
    /// here, or it is not told there. A stage that declares the need may
    /// name the kinds it takes, and is then told no other: each kind left
    /// out is a word its requests do not pay for. The engine asks once, at
    /// its build: the answer is the stage's for the engine's life.
    ///
    /// A change of membership and a signal are told to every stage,
    /// whatever it answers: no host reads those two kinds of the answer
    /// yet. A plugin's stage answers every kind its plugin observes.
    ///
    /// The two answers part for a stage that takes a word only to drop
    /// state early: it decides the same without the call and declares no
    /// need, and it still names the word. Untold, it would keep that
    /// state.
    fn hears(&self) -> EventSet {
        if self.needs_lifecycle() {
            EventSet::LEASE
        } else {
            EventSet::NONE
        }
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

    fn on_signal(&self, signal: &Signal<'_>) {
        (**self).on_signal(signal);
    }

    fn keeps_loads(&self) -> bool {
        (**self).keeps_loads()
    }

    fn needs_lifecycle(&self) -> bool {
        (**self).needs_lifecycle()
    }

    fn hears(&self) -> EventSet {
        (**self).hears()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/stages/lifecycle/tests.rs"]
mod tests;
