//! Load reports told to the stages that keep them.
//!
//! A stage that keeps a number of each worker's load report says so
//! ([`Lifecycle::keeps_loads`]), and its host tells it the reports
//! ([`Signal::Load`]). Here the host is the
//! engine's own task ([`PolicyEngine::run`]). At each load snapshot the task
//! takes from the channel its host handed ([`PolicyEngine::set_loads`]), it
//! tells such stages the reports that are new in it, and only then keeps the
//! snapshot at hand. So a report told at the first snapshot that holds it
//! is told before a decision can read it in the engine's snapshot. A report
//! whose worker is told as joined behind that snapshot is told at the next
//! one the task takes: a decision can read it first. A decision that runs
//! during a telling reads the snapshot before it, beside what the stages
//! have kept of the new one so far. A request that gives a snapshot of its
//! own ([`RequestMaker::loads`](crate::engine::RequestMaker::loads)) is
//! outside that order: its decision reads what it was given.
//!
//! **Which workers.** Those the task told the stages as joined, by its
//! follow of the registry the host handed ([`PolicyEngine::set_registry`]),
//! each under the id of its join ([`Follow::joined`]). While the follow
//! holds two workers under one id, it names neither. A telling first hears
//! the registry's words that stand ready ([`PolicyEngine::follow_so_far`]):
//! a worker registered ahead of a publication is told its report in it. The
//! telling takes no subscription and tries no failed step of the follow
//! again: those wait for the task's next upkeep, and until then it names
//! the workers the follow holds. What it reads of a worker, the follow read
//! off the worker's object where it took it: the address, the role and the
//! rank. A stage that missed a join, behind the panic of a stage told ahead
//! of it, is told the worker's reports all the same. An engine handed no
//! registry, or no channel, tells no report. A registry handed behind the
//! channel is followed from the task's next upkeep, and its workers are
//! told their reports at the first snapshot behind it.
//!
//! **Which reports.** The follow keeps, with each worker it holds, the
//! sample instant of the report told last ([`Joined::told_at`]). A report is
//! told where its sample instant is not that one: a worker's first report,
//! a report that arrives again with the same counts and a new sample
//! instant, and a sample earlier than the one told last. The report told
//! last is not told again, however many snapshots in a row show it: a
//! store's host tells every record at every publication and leaves a stage
//! to tell new from old, and here a stage is handed the new ones alone. The
//! follow writes what it holds of a worker again at a registration or a
//! replacement it hears of it, and at a whole read of the registry: the
//! worker's report is then told once more, by the role and the rank read
//! there. A snapshot answers by address. So a worker registered at the
//! address of one that left is told the report that stands there, as its
//! record states it, until the monitor's next. And a worker that takes
//! another's address under another id has its report told under the old
//! id until the follow hears of it.
//!
//! **What is told.** The worker under the id of its join, which is the id
//! its lists name it by, its rank, and its report as a list's record states
//! it ([`stated_report`]): the row at rest, its age left for a decision to
//! state, and the sample's instant on the engine's decision clock. A worker
//! object that stands for several ranks is told as its one record: the
//! ranks' sum, under the rank its registration states. The slot of the
//! worker's count is none
//! ([`WorkerSlot::NONE`](crate::tracking::WorkerSlot::NONE)): this engine
//! keeps no table of counts, and a worker's count is its object's.
//!
//! **Its named values.** Under a document a stage of which keeps a signal's
//! column, the report is told with the values it states under the
//! document's names ([`ReportSignals`]), as a list's cells state them
//! ([`stated_signals`]): one conversion, the signal's own unit, a value in
//! range, and each value's own sample instant on the engine's decision
//! clock. A name the report does not state reads none. A report that is
//! not the whole of its worker object states no named value, as a list's
//! cells hold none of it: one that lists fewer ranks than it counts, and
//! one of several rows that does not list as many as the object stands
//! for.
//! No age is tested here: a telling has no request's clock. Under any other
//! document a report states none, and nothing is converted for it. The
//! values stand in one row the task keeps, written again at each report:
//! they are lent for the stage's call, and a stage copies out what it
//! keeps.
//!
//! **What can be passed over.** The host's channel keeps one snapshot, and a
//! snapshot keeps one report a worker. So of two reports of one worker the
//! task took no snapshot between, the earlier is told to no stage. The
//! sample instants it is told do not say whether a report was passed over,
//! and nothing counts them. A report whose counts change under a sample
//! instant that stands is not told again: the worker crate states one
//! instant for a worker object's several ranks. The task tells at a
//! snapshot alone: a worker told as joined behind the last snapshot of a
//! channel that publishes no more is told no report. Of two workers held
//! under one id, the one that stays when the other goes is told from its
//! next new sample on: a report told ahead of the other's join is not told
//! again.
//!
//! **A panic.** Each report's conversion, and each stage's hearing of each
//! report, stands under a catch of its own. A report whose conversion
//! panics is counted ([`PolicyEngine::upkeep_faults`]) and told to no
//! stage, and a stage that panics at a report is counted and keeps what it
//! wrote of the report ahead of its panic. Either report counts as told:
//! the next told of the worker is the next the rule above names. The other
//! reports, the other stages and the task go on, and the snapshot is kept.
//!
//! **What it costs.** A document none of whose stages keeps a report looks
//! up no report and hears no word at a snapshot, and its follow reads
//! nothing of a worker for a telling. A document that reads no load report
//! and no named signal keeps the channel for such a stage alone, and no
//! decision of it reads the engine's snapshot. Where a stage keeps reports,
//! a snapshot is at hand for decisions behind its telling: the registry's
//! ready words, one look-up for each joined worker, one conversion of each
//! new report, its named values where a stage reads them, and the stages'
//! own work at each. A telling allocates nothing.

use smg_worker::load::state::LoadSnapshot;

use super::{
    membership::{Follow, Joined},
    PolicyEngine,
};
use crate::{
    engine::{
        records::{stated_report, stated_signals},
        view::WorkerId,
    },
    load::{ReportSignals, SignalCells},
    stages::Lifecycle,
    tracking::lease::Signal,
};

impl PolicyEngine {
    /// Tells the stages that keep load reports what is new in `newest`, the
    /// snapshot the task is about to keep at hand: for each worker the
    /// follow told them as joined, its report where the sample instant is
    /// not the one told last.
    ///
    /// An engine with no such stage returns at once. Else the registry's
    /// ready words are heard first ([`Self::follow_so_far`]), and nothing
    /// is told while the task follows no registry. Each new report is
    /// found once and told to every such stage, ahead of the next: its
    /// named values stand in `cells`, the one row the task keeps for them.
    /// A stage's panic at a report is held and counted
    /// ([`Self::upkeep_faults`]), each report under a catch of its own:
    /// the stage hears the ones behind it.
    pub(super) fn tell_loads(
        &self,
        members: &mut Option<Follow>,
        cells: &mut SignalCells,
        newest: &LoadSnapshot,
    ) {
        let engine = self.document.spec.engine();
        if !engine.keeps_loads() {
            return;
        }
        self.follow_so_far(members);
        let Some(follow) = members.as_ref() else {
            return;
        };
        for (id, worker) in follow.joined() {
            let Some(report) = self.new_report(id, worker, newest, cells) else {
                continue;
            };
            engine.tell_loads(|tell| {
                self.held(|| tell(&report));
            });
        }
    }

    /// The report of `newest` for `worker`, which the stages know as `id`,
    /// as a stage is told it, where its sample instant is not the one told
    /// last of the worker: one look-up, and one conversion of a report that
    /// is new. The instant counts as told from here on. Its named values
    /// are written into `cells` where a stage of the document keeps a
    /// signal's column, and nothing is converted for them elsewhere. A
    /// conversion that panics is held and counted
    /// ([`Self::upkeep_faults`]), and its report is told to no stage.
    fn new_report<'c>(
        &self,
        id: WorkerId,
        worker: &Joined,
        newest: &LoadSnapshot,
        cells: &'c mut SignalCells,
    ) -> Option<Signal<'c>> {
        let (report, sampled_at) = newest.report(&worker.address)?;
        if worker.told_at.replace(Some(sampled_at)) == Some(sampled_at) {
            return None;
        }
        self.held(move || {
            let clock = &self.clock;
            let (row, sampled_at_micros) = stated_report(report, sampled_at, worker.role, clock);
            let signals = if self.document.signals {
                let columns = self.document.spec.signal_columns();
                stated_signals(report, sampled_at, worker.ranks, columns, clock, cells)
            } else {
                ReportSignals::NONE
            };
            Signal::load(id, worker.dp_rank, row, sampled_at_micros).with_signals(signals)
        })
    }
}

#[cfg(test)]
mod tests;
