//! The words an engine hears, and how it tells them to its stages.
//!
//! The engine implements the lifecycle itself ([`Lifecycle`]): it hears a
//! lease's issue, its dispatch, its end and its release, a change of
//! membership and a signal. Its own entries tell a word whole. An issue
//! and an end are told to the stages that take them ([`Takers`]), in the
//! engine's order. A dispatch ([`Lifecycle::on_dispatched`]) is counted
//! in the tallies of sends, then has the session table's part, then is
//! told to the stages that take it. A release
//! ([`Lifecycle::on_released`]) has the table's part, then the stages'.
//! Each part is a function of its own, and a host may tell a word part
//! by part. A change of membership ([`Lifecycle::on_membership`]) has
//! the session table's part first as well, then the stages' part
//! ([`Engine::membership_stages`]): the tallies of sends and each
//! profile's scorers and its picker hear it, and a host may tell that
//! part alone. A signal is told to each profile's scorers and its picker.
//!
//! Under `in-process`, the trait `Issuing` names what hears a lease's
//! words at the engine that issued it: an engine whole, or what a later
//! engine kept of the stages that stood by the lease's generation. An
//! engine whole is named as the engine itself, or as a `Stood` under no
//! generation's bound; what a later engine kept is a `Stood` too. A
//! host's load reports are told under that feature too, to the stages
//! that keep a number of them (`Engine::tell_loads`).
//!
//! This module is a child of the framework's because it reads private
//! items of that module as they stand: the engine's fields `sessions`,
//! `takers`, `profiles`, `keepers` and `hears`, its method `with_sends`,
//! [`StagePlace`] and its method `stage`, the method `of` of [`Takers`],
//! and under `in-process` the engine's field `born` and the method
//! `profile` of [`StagePlace`].

// `Takers` is named only by doc links of this module.
#[cfg(doc)]
use super::Takers;
use super::{Engine, StagePlace};
#[cfg(feature = "in-process")]
use crate::engine::view::Session;
// `Needs` and `SessionTable` are named only by doc links of this module,
// on functions under `in-process`.
#[cfg(all(doc, feature = "in-process"))]
use crate::{document::spec::Needs, engine::sticky_sessions::SessionTable};
use crate::{
    engine::{
        decision::{Decision, Target},
        sticky_sessions::SessionVerdict,
        view::WorkerId,
    },
    stages::lifecycle::Lifecycle,
    tracking::lease::{
        EventKind, EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal,
    },
};

impl Engine {
    /// The session table's part of a dispatch ([`Lifecycle::on_dispatched`]): the
    /// session pin follows the dispatched worker of the side, the decode
    /// of a pair target on the decode side and the target itself
    /// otherwise. A dispatch is told in parts: the counts of sends, this
    /// table, then the stages ([`Self::dispatched_stages`]), so a stage's
    /// call that unwinds leaves the table's entry written. Every teller of
    /// a dispatch keeps that order, the trait's own entry among them.
    ///
    /// The table is handed the instant the lease was issued under
    /// ([`Lease::issued_micros`]). A dispatch elsewhere withdraws the claim
    /// that lease's decision made, and no later decision's claim on the
    /// same worker.
    pub(crate) fn dispatched_table(&self, lease: &Lease, target: &Target, side: Side) {
        let dispatched = match (side, target.pair) {
            (Side::Decode, Some(pair)) => pair.decode_id,
            _ => target.id,
        };
        if lease.session_verdict != SessionVerdict::None {
            let (verdict, at) = (lease.session_verdict, lease.issued_micros);
            self.sessions
                .dispatched(lease.session, verdict, dispatched, at);
        }
    }

    /// The session table's part of a release ([`Lifecycle::on_released`]): a
    /// commit that refused every target leaves the decision's claim no
    /// worker to move to, and the claim goes with it. A release is told in
    /// parts, as a dispatch is: this table, then the stages
    /// ([`Self::released_stages`]). An engine in its host's process tells
    /// the table each side a pair's release ends ahead of any stage, a
    /// decode stated for later among them: no stage hears that side's end,
    /// and the pair's first decision made its claim.
    ///
    /// The table is handed the instant the lease was issued under
    /// ([`Lease::issued_micros`]). The claim that goes is the one that
    /// lease's decision made: a refusal told late leaves a later decision's
    /// claim on the same worker.
    pub(crate) fn released_table(&self, lease: &Lease, reason: ReleaseReason) {
        if reason == ReleaseReason::CommitFailed {
            if let SessionVerdict::Claimed(claimed) = lease.session_verdict {
                let at = lease.issued_micros;
                self.sessions
                    .move_claim(lease.session, claimed, WorkerId(0), at);
            }
        }
    }

    /// The session table's part of a decode call that states a verdict of
    /// its key ([`SessionTable::told_under`]). The call read `key` at `at`,
    /// its own instant, and tells its words under `issued`, the instant of
    /// the lease its pair's first decision issued. A claim it made, of
    /// `claimed`, takes that date. `first` is the worker that decision
    /// claimed for `key`, `WorkerId(0)` where it claimed none. The pair's
    /// decode entries ask here, so they read no table of the engine's.
    #[cfg(feature = "in-process")]
    #[inline]
    pub(crate) fn told_under(
        &self,
        key: Session,
        claimed: Option<WorkerId>,
        at: u64,
        issued: u64,
        first: WorkerId,
    ) {
        self.sessions.told_under(key, claimed, at, issued, first);
    }

    /// The stages' part of a release, behind the table's
    /// ([`Self::released_table`]).
    pub(crate) fn released_stages(&self, lease: &Lease, reason: ReleaseReason) {
        let stages = self.stages(self.taking(EventKind::Released));
        tell_released(stages, lease, reason);
    }

    /// The places of this engine's stages that take the word `kind`: its
    /// list of them ([`Takers`]), in the engine's order.
    #[inline(always)]
    pub(super) fn taking(&self, kind: EventKind) -> &[StagePlace] {
        self.takers.of(kind)
    }

    /// The stages of `takers`, one of this engine's lists of the stages
    /// that take a word ([`Self::taking`]), in the list's order.
    #[inline(always)]
    fn stages<'a>(&'a self, takers: &'a [StagePlace]) -> impl Iterator<Item = &'a dyn Lifecycle> {
        takers.iter().map(|place| place.stage(&self.profiles))
    }

    /// Records a dispatch in each profile tally that stages read. A tally
    /// asks `present` only where the send would enter a rank it does not
    /// know: a served controller asks its worker objects there, so a late
    /// report enters no rank behind a removal. An engine whose profiles
    /// read no sends asks nothing.
    #[inline]
    pub(crate) fn dispatched_sent(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
        present: impl Fn() -> bool,
    ) {
        match self.sent_credits.as_slice() {
            [] => {}
            // One count is the common document: its record runs in line
            // behind the caller's telling.
            [one] => one.dispatched(lease, target, side, since_decision_micros, present),
            _ => self.credit_dispatch(lease, target, side, since_decision_micros, present),
        }
    }

    /// Out of line: a document of several counts is rare. One count is
    /// told in line ([`Self::dispatched_sent`]).
    #[inline(never)]
    fn credit_dispatch(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
        present: impl Fn() -> bool,
    ) {
        for credit in &self.sent_credits {
            credit.dispatched(lease, target, side, since_decision_micros, &present);
        }
    }

    /// The stages' part of a dispatch: the stages that take it hear it.
    // In line at its callers: as a call it costs a pair's decision 41
    // instructions, counted with Callgrind.
    #[inline(always)]
    pub(crate) fn dispatched_stages(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        let stages = self.stages(self.taking(EventKind::Dispatched));
        tell_dispatched(stages, lease, target, side, since_decision_micros);
    }
}

/// A release told to `stages`, the ones that take it, in their order:
/// the engine's, a profile's scorers ahead of its picker.
#[inline(always)]
fn tell_released<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    reason: ReleaseReason,
) {
    for stage in stages {
        stage.on_released(lease, reason);
    }
}

/// A dispatch told to `stages`, the ones that take it, in their order.
#[inline(always)]
fn tell_dispatched<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    target: &Target,
    side: Side,
    since_decision_micros: u32,
) {
    for stage in stages {
        stage.on_dispatched(lease, target, side, since_decision_micros);
    }
}

/// A lease's issue told to `stages`, the ones that take it, in their
/// order.
#[inline(always)]
fn tell_issued<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    decision: &Decision,
) {
    for stage in stages {
        stage.on_decision_issued(lease, decision);
    }
}

/// An end told to `stages`, the ones that take it, in their order.
#[inline(always)]
fn tell_completed<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    outcome: &Outcome,
) {
    for stage in stages {
        stage.on_completed(lease, outcome);
    }
}

/// What an engine kept of the stages that stood by an earlier generation
/// ([`Engine::stood_by`]): the engine's session table, and each of its
/// profiles whose object was made below a generation. It hears the words
/// of a lease issued under that earlier generation.
///
/// It holds nothing of its own. A word told to the stages walks the
/// engine's list of the stages that take it ([`Takers`]), beside the
/// generations their profiles were made under, and tells the ones whose
/// profile stood. So an engine keeps no list for an earlier generation,
/// whatever the number of generations its profiles were made under: a
/// document that gained one profile at each of a thousand applies holds a
/// thousand profiles once.
///
/// A host that tells leases off a file of its own names the engine whole
/// as one too (`Engine::stands`): every profile stood, under no
/// generation's bound, so a profile made under the last generation a
/// document can bear hears as any other.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stood<'a> {
    pub(super) engine: &'a Engine,
    /// A profile made under a generation below this one stood. With
    /// none, every profile stood: the engine whole.
    pub(super) below: Option<u64>,
}

#[cfg(feature = "in-process")]
impl<'a> Stood<'a> {
    /// The stages of `takers`, one of the engine's lists of the stages
    /// that take a word ([`Engine::taking`]), whose profile stood: in the
    /// list's order.
    #[inline]
    fn stages(self, takers: &'a [StagePlace]) -> impl Iterator<Item = &'a dyn Lifecycle> {
        let Self { engine, below } = self;
        let stood = move |place: &&StagePlace| {
            below.is_none_or(|below| engine.born[place.profile()] < below)
        };
        let stood = takers.iter().filter(stood);
        stood.map(move |place| place.stage(&engine.profiles))
    }
}

/// What hears a lease's words at the engine that issued it: that engine's
/// session table, and its stages. An engine whole is one: the engine in
/// force hears the leases of its own generation. A host that holds its
/// decisions names it as the engine itself, and a host with a file of
/// leases as a [`Stood`] under no bound (`Engine::stands`). What a later
/// engine kept of the stages that stood by the lease's generation is the
/// other (a [`Stood`] by [`Engine::stood_by`]): the same table, and no
/// stage built since. The rule of a lease's words tells either in its one
/// order ([`lease_events`](crate::engine::lease_events)).
#[cfg(feature = "in-process")]
pub(crate) trait Issuing: Copy {
    /// The session table's part of a dispatch
    /// ([`Engine::dispatched_table`]).
    fn dispatched_table(self, lease: &Lease, target: &Target, side: Side);

    /// The stages' part of a dispatch ([`Engine::dispatched_stages`]).
    fn dispatched_stages(self, lease: &Lease, target: &Target, side: Side, since_micros: u32);

    /// The session table's part of a release ([`Engine::released_table`]).
    fn released_table(self, lease: &Lease, reason: ReleaseReason);

    /// The stages' part of a release ([`Engine::released_stages`]).
    fn released_stages(self, lease: &Lease, reason: ReleaseReason);

    /// A lease's issue, told to the stages
    /// ([`Lifecycle::on_decision_issued`]).
    fn decision_issued(self, lease: &Lease, decision: &Decision);

    /// A lease's end, told to the stages ([`Lifecycle::on_completed`]).
    fn completed(self, lease: &Lease, outcome: &Outcome);

    /// A release whole, as [`Lifecycle::on_released`] tells it: the
    /// session table, then the stages.
    #[inline(always)]
    fn released(self, lease: &Lease, reason: ReleaseReason) {
        self.released_table(lease, reason);
        self.released_stages(lease, reason);
    }
}

#[cfg(feature = "in-process")]
impl Issuing for &Engine {
    #[inline(always)]
    fn dispatched_table(self, lease: &Lease, target: &Target, side: Side) {
        Engine::dispatched_table(self, lease, target, side);
    }

    #[inline(always)]
    fn dispatched_stages(self, lease: &Lease, target: &Target, side: Side, since_micros: u32) {
        Engine::dispatched_stages(self, lease, target, side, since_micros);
    }

    #[inline(always)]
    fn released_table(self, lease: &Lease, reason: ReleaseReason) {
        Engine::released_table(self, lease, reason);
    }

    #[inline(always)]
    fn released_stages(self, lease: &Lease, reason: ReleaseReason) {
        Engine::released_stages(self, lease, reason);
    }

    #[inline(always)]
    fn decision_issued(self, lease: &Lease, decision: &Decision) {
        self.on_decision_issued(lease, decision);
    }

    #[inline(always)]
    fn completed(self, lease: &Lease, outcome: &Outcome) {
        self.on_completed(lease, outcome);
    }
}

// The session table is the engine's own, whatever generation a lease is
// of. A word's stages are the ones that take it and whose profile stood.
#[cfg(feature = "in-process")]
impl Issuing for Stood<'_> {
    #[inline]
    fn dispatched_table(self, lease: &Lease, target: &Target, side: Side) {
        self.engine.dispatched_table(lease, target, side);
    }

    fn dispatched_stages(self, lease: &Lease, target: &Target, side: Side, since_micros: u32) {
        let stages = self.stages(self.engine.taking(EventKind::Dispatched));
        tell_dispatched(stages, lease, target, side, since_micros);
    }

    #[inline]
    fn released_table(self, lease: &Lease, reason: ReleaseReason) {
        self.engine.released_table(lease, reason);
    }

    fn released_stages(self, lease: &Lease, reason: ReleaseReason) {
        let stages = self.stages(self.engine.taking(EventKind::Released));
        tell_released(stages, lease, reason);
    }

    fn decision_issued(self, lease: &Lease, decision: &Decision) {
        let stages = self.stages(self.engine.taking(EventKind::Issued));
        tell_issued(stages, lease, decision);
    }

    fn completed(self, lease: &Lease, outcome: &Outcome) {
        let stages = self.stages(self.engine.taking(EventKind::Completed));
        tell_completed(stages, lease, outcome);
    }
}

impl Engine {
    /// Tells a host's reports, each as the host shows it to decisions, to
    /// every stage that keeps a number of the reports
    /// ([`Lifecycle::keeps_loads`]), and to no other stage. `reports` is
    /// run once for each such stage and hands every report to the call it
    /// is given. An engine with no such stage never runs it: a load
    /// snapshot that tells nobody walks no row. A plugin that watches
    /// signals hears of a snapshot's reports no more than it did before a
    /// stage kept any. The engine over worker lists is the one host that
    /// tells reports.
    #[cfg(feature = "in-process")]
    pub(crate) fn tell_loads(&self, reports: impl Fn(&mut dyn FnMut(&Signal<'_>))) {
        for keeper in &self.keepers {
            let stage = keeper.stage(&self.profiles);
            reports(&mut |report| stage.on_signal(report));
        }
    }

    /// Whether a stage that keeps the reports ([`Lifecycle::keeps_loads`])
    /// keeps a signal's column too ([`Needs::signal_cells`]). Such a stage
    /// reads a told report's named values by its column. A host states the
    /// values of a report it tells under this word: every stage that keeps
    /// reports is told them, or none is. It converts none under an engine
    /// that answers no. It asks once, where it makes its value of the
    /// document.
    #[cfg(feature = "in-process")]
    pub(crate) fn keepers_read_signals(&self) -> bool {
        self.keepers.iter().any(|keeper| match *keeper {
            StagePlace::Scorer { profile, stage } => {
                let scorer = &self.profiles[profile].scorers[stage];
                scorer.scorer.needs().signal_cells
            }
            StagePlace::Picker { profile } => {
                let picker = &self.profiles[profile].picker;
                picker.needs().signal_cells
            }
        })
    }

    /// The stages' part of a change of membership
    /// ([`Lifecycle::on_membership`]): the tallies of sends and each
    /// profile's scorers and its picker hear it, and the session table does
    /// not. The engine over a host's worker lists tells this part alone: a
    /// leave there takes no pin away. A session's pin stands on a worker
    /// that is gone until the session's idle time or two other workers end
    /// it, and is a routable pin again once a worker stands under its id.
    pub(crate) fn membership_stages(&self, change: &Membership<'_>) {
        for credit in &self.sent_credits {
            credit.membership(change);
        }
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_membership(change);
            }
            profile.picker.on_membership(change);
        }
    }
}

impl Lifecycle for Engine {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        let stages = self.stages(self.taking(EventKind::Issued));
        tell_issued(stages, lease, decision);
    }

    #[inline(always)]
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.dispatched_sent(lease, target, side, since_decision_micros, || true);
        self.dispatched_table(lease, target, side);
        self.dispatched_stages(lease, target, side, since_decision_micros);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        let stages = self.stages(self.taking(EventKind::Completed));
        tell_completed(stages, lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        // The session table, then the stages. What hears a lease's words
        // spells the same two steps once more, under `in-process`
        // (`Issuing::released`): this entry stands without that feature.
        self.released_table(lease, reason);
        self.released_stages(lease, reason);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            // The pins of a worker that left stand for the absence's
            // time: its key may be registered again, and a host that
            // knows says which way ([`Engine::pins_follow`],
            // [`Engine::pins_forget`]).
            Membership::Removed(worker) => self.sessions.left(*worker),
            Membership::Replaced { old, new } => {
                self.sessions.left(*old);
                self.sessions.returned(*new);
            }
            // A join under an id that left ends its absence, whoever tells
            // it: the worker is back as it was, and its pins stand.
            Membership::Added(worker) => self.sessions.returned(*worker),
            Membership::Snapshot(workers) => self.sessions.returned_all(workers),
            _ => {}
        }
        self.membership_stages(change);
    }

    fn on_signal(&self, signal: &Signal<'_>) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_signal(signal);
            }
            profile.picker.on_signal(signal);
        }
    }

    fn keeps_loads(&self) -> bool {
        !self.keepers.is_empty()
    }

    fn needs_lifecycle(&self) -> bool {
        self.profiles.iter().any(|profile| {
            profile.picker.needs_lifecycle()
                || profile.scorers.iter().any(|w| w.scorer.needs_lifecycle())
        })
    }

    /// The words the document takes: a stage's, the session table's at a
    /// dispatch and at a release under a sessions rule, the record of
    /// dispatched prefixes at an issue, and a count of sends at a
    /// dispatch. Folded at the build: one read here.
    fn hears(&self) -> EventSet {
        self.with_sends(self.hears)
    }
}
