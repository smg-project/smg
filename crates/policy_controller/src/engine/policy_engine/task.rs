//! The engine's own task: what a host of the engine no longer does by hand.
//!
//! A host hands the engine its load monitor's channel
//! ([`PolicyEngine::set_loads`]) and its worker registry
//! ([`PolicyEngine::set_registry`]), and spawns [`PolicyEngine::run`] once.
//! The task runs the engine's upkeep once a second and keeps the newest
//! load snapshot at hand, so a decision whose request gives none reads the
//! engine's and never touches the channel. It tells the document's stages
//! a worker that joined or was removed, by the registry's events and under
//! the id the host states of the worker. And it tells the stages that keep
//! load reports ([`Lifecycle::keeps_loads`]) each worker's new report,
//! ahead of keeping the snapshot that holds it.

use std::{
    fmt,
    future::pending,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{atomic::Ordering, Arc},
    time::{Duration, Instant},
};

use arc_swap::{ArcSwapOption, Guard};
use parking_lot::Mutex;
use smg_worker::load::state::{LoadReceiver, LoadSnapshot};
use tokio::time::{self, MissedTickBehavior};

use super::{
    membership::{self, Follow},
    Loaded, PolicyEngine,
};
use crate::{
    engine::{
        decision_engine::{DecisionEngine, RequestMaker},
        framework::release,
    },
    load::SignalCells,
    stages::Lifecycle,
};

/// How often the task runs the engine's upkeep.
const UPKEEP: Duration = Duration::from_secs(1);

/// The engine's own view of its host's load channel.
#[derive(Default)]
pub(super) struct Loads {
    /// The channel the host handed last, with the count of its hand-overs
    /// ([`PolicyEngine::set_loads`]). The task takes the lock once a
    /// second, and no decision takes it.
    feed: Mutex<(u64, Option<LoadReceiver>)>,
    /// The newest snapshot the engine's task kept at hand, behind its
    /// telling to the stages that keep reports. The task alone writes it
    /// ([`PolicyEngine::run`]).
    newest: ArcSwapOption<LoadSnapshot>,
}

impl fmt::Debug for Loads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loads")
            .field("hand_overs", &self.feed.lock().0)
            .field("at_hand", &self.newest.load().is_some())
            .finish()
    }
}

impl Loads {
    /// The newest snapshot the task has at hand: none where the host handed
    /// no channel, the document in force has no use for one, or the task
    /// has not read it yet.
    #[inline]
    pub(super) fn newest(&self) -> Guard<Option<Arc<LoadSnapshot>>> {
        self.newest.load()
    }

    /// Counts one more hand-over of the channel that stands: the task
    /// takes it again at its next step, as behind a new document.
    pub(super) fn handed_again(&self) {
        self.feed.lock().0 += 1;
    }

    /// The channel the host handed last and its count, where the task took
    /// the hand-over of another count: `taken`.
    fn handed_past(&self, taken: u64) -> Option<(u64, Option<LoadReceiver>)> {
        let feed = self.feed.lock();
        (feed.0 != taken).then(|| feed.clone())
    }

    /// Keeps the newest snapshot of `feed` at hand, behind `tell`, which
    /// is handed it first: a decision reads no report of it in the
    /// engine's snapshot until `tell` has ended.
    fn keep(&self, feed: Option<&mut LoadReceiver>, tell: impl FnOnce(&LoadSnapshot)) {
        if let Some(feed) = feed {
            let newest = Arc::clone(&feed.borrow_and_update());
            tell(&newest);
            self.newest.store(Some(newest));
        }
    }
}

/// The snapshot one decision reads, held for as long as the decision reads
/// it ([`Loaded::loads_read`]).
pub(super) enum LoadsRead<'a> {
    /// The snapshot the request gives.
    Given(&'a Arc<LoadSnapshot>),
    /// The engine's newest, read where the request gives none.
    Kept(Option<Guard<Option<Arc<LoadSnapshot>>>>),
}

impl LoadsRead<'_> {
    /// The snapshot: none where the request gives none and the engine
    /// keeps none at hand.
    #[inline]
    pub(super) fn snapshot(&self) -> Option<&Arc<LoadSnapshot>> {
        match self {
            Self::Given(given) => Some(given),
            Self::Kept(kept) => kept.as_ref().and_then(|newest| newest.as_ref()),
        }
    }
}

/// The load monitor's next publication on `feed`: `false` where the monitor
/// ended. Never ready with no channel.
async fn published(feed: Option<&mut LoadReceiver>) -> bool {
    match feed {
        Some(feed) => feed.changed().await.is_ok(),
        None => pending().await,
    }
}

impl PolicyEngine {
    /// Hands the engine its host's load channel: its load monitor's
    /// subscription. A host hands it at its start, and again where it
    /// builds another monitor: a channel ends with its monitor. Of two
    /// hand-overs the newest stands: a later one replaces the one before.
    /// The engine keeps the channel whatever its document, so a document
    /// put in force later reads it ([`Self::apply`]). What the task does
    /// with it is read of the document in force. Under a document that
    /// reads no load report
    /// ([`Needs::loads`](crate::document::spec::Needs::loads)) and no named
    /// signal
    /// ([`Needs::signal_cells`](crate::document::spec::Needs::signal_cells))
    /// the task keeps a snapshot only for the stages that keep reports
    /// ([`Lifecycle::keeps_loads`]), which it tells them ([`Self::run`]),
    /// and none where the document has no such stage. No decision under
    /// such a document reads a report. A named signal's values stand in
    /// the same reports, so a document that reads one reads the channel as
    /// one that reads a report does.
    ///
    /// The engine's task reads the channel ([`Self::run`]). A decision
    /// reads the snapshot its request gives where it gives one
    /// ([`RequestMaker::loads`]), and
    /// the newest the task keeps at hand otherwise: a single request's, a
    /// pair's first, and a pair's decode decided again.
    pub fn set_loads(&self, loads: LoadReceiver) {
        let mut feed = self.loads.feed.lock();
        *feed = (feed.0 + 1, Some(loads));
    }

    /// The engine's own task: what its host no longer does by hand. The
    /// host spawns it once, at its start, on a runtime whose timer is
    /// enabled, and it runs for as long as the engine stands.
    ///
    /// Its first upkeep runs at its first poll, and each later one a
    /// second behind the one before. So what a host handed ahead of the
    /// spawn is taken at that poll: a host relies on the registry it
    /// handed and on its load channel once the task has been polled once.
    ///
    /// - The upkeep once a second ([`DecisionEngine::sweep`]), dated at its
    ///   run by the runtime's clock. A task that was not run for a while
    ///   runs it once for the seconds it missed. A panic in it is held and
    ///   counted ([`Self::upkeep_faults`]): the task goes on.
    /// - The newest load snapshot kept at hand: read from the host's
    ///   channel at the first upkeep behind its hand-over, so within a
    ///   second of it, and swapped when the load monitor publishes one.
    ///   The channel is taken again at the first upkeep behind a new
    ///   document ([`Self::apply`]).
    ///   Until then, and for an engine handed no channel, a decision whose
    ///   request gives no snapshot reads no report. A monitor that ended
    ///   leaves its last snapshot standing until another is handed.
    /// - Each worker's new load report told to the stages that keep them
    ///   ([`Lifecycle::keeps_loads`]), ahead of the snapshot that holds it
    ///   being kept at hand: for the workers the task told them as joined,
    ///   under the id of the join, where the report's sample instant is
    ///   not the one last told of the worker. A telling hears the
    ///   registry's words that stand ready first. A worker told as joined
    ///   behind a snapshot is told its report at the next one. A whole read
    ///   that a panic of the host's own code put off to a later upkeep
    ///   tells its workers the snapshot at hand at that upkeep. The channel
    ///   keeps one snapshot, so of two reports of one worker the task took
    ///   no snapshot between, the earlier is told to no stage. A request
    ///   that gives a snapshot of its own can show its decision a report
    ///   ahead of the telling. A stage's panic at a report, and a report's
    ///   conversion that panics, are held and counted
    ///   ([`Self::upkeep_faults`]): that report counts as told, for that
    ///   stage or for all, and the task goes on. A stage that waits at a
    ///   report holds back the task's other work.
    /// - A worker that joined or was removed told to the document's
    ///   stages, by the events of the registry the host handed
    ///   ([`Self::set_registry`]) and under the id the host states of the
    ///   worker: followed from the first upkeep behind the hand-over, so
    ///   within a second of it, with the workers that stand then told as
    ///   joined. The registry is read whole again where its channel
    ///   dropped events or closed, and at the first upkeep behind a new
    ///   document. A panic of the host's own code in the
    ///   follow is held and counted ([`Self::upkeep_faults`]): the task
    ///   goes on, and reads the registry whole again at its next upkeep.
    /// - The cells of a model's worker lists filled ahead of the model's
    ///   requests, for an engine handed its program's lists
    ///   ([`Self::set_lists`]): behind the registry's words of membership,
    ///   and behind each whole read of the registry. A panic in a fill is
    ///   held and counted ([`Self::upkeep_faults`]): the task goes on to
    ///   the next list, and a list it did not fill is filled at its first
    ///   ask.
    pub async fn run(&self) {
        self.run_with(|now| self.sweep(now)).await;
    }

    /// How many steps of the engine's own task panicked ([`Self::run`]): a
    /// run of the upkeep, a step of the registry's follow in the host's own
    /// code, a stage's hearing of a load report, a report's conversion for
    /// the stages, or a fill of worker lists' cells ahead of their
    /// requests. Each was held, and the task went on. A run of the upkeep
    /// at the tick of a host that spawns no task of the engine is held
    /// and counted with them. While the upkeep's
    /// panics raise the count it does not run whole: session keys past
    /// their end can stay on file, though every request is still answered.
    /// While the follow's raise it, a join whose answer of the host
    /// panicked is lost until a whole read of the registry succeeds. A
    /// removal asks the host nothing ahead of its telling and is still
    /// told. A host whose code panics at one worker every time fails every
    /// whole read. A stage that panics at a load report keeps of it what it
    /// wrote ahead of the panic, and a report whose conversion panics is
    /// told to no stage. A list whose fill panics is left to its first ask,
    /// and a panic of the program's own code where it hands a model's lists
    /// leaves the lists it had not handed to theirs.
    #[must_use]
    pub fn upkeep_faults(&self) -> u64 {
        self.upkeep_faults.load(Ordering::Relaxed)
    }

    /// One step of the engine's task under the task's catch: `None` where
    /// it panicked. The panic is held and counted
    /// ([`Self::upkeep_faults`]), and the task goes on.
    pub(super) fn held<R>(&self, step: impl FnOnce() -> R) -> Option<R> {
        match catch_unwind(AssertUnwindSafe(step)) {
            Ok(done) => Some(done),
            Err(payload) => {
                release(payload);
                self.upkeep_faults.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// The engine's upkeep at `now` under the task's catch
    /// ([`Self::held`]), for a host that spawns no task of the engine and
    /// runs the upkeep at a tick of its own. A panic in it is held and
    /// counted ([`Self::upkeep_faults`]): the host's tick goes on.
    pub(crate) fn upkeep(&self, now: Instant) {
        let _swept = self.held(|| self.sweep(now));
    }

    /// The task's loop. `upkeep` is the step it runs once a second under
    /// a catch of its own: [`Self::run`] hands the engine's own.
    async fn run_with(&self, upkeep: impl Fn(Instant)) {
        // The first upkeep runs at this poll, below, and the first tick is
        // due a second behind it. A tick due at once still waits on the
        // runtime's timer: a host that spawned the task and yielded once
        // would decide ahead of it.
        let mut tick = time::interval_at(time::Instant::now() + UPKEEP, UPKEEP);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // The host's channel while its monitor publishes, and the count of
        // the hand-over it came with.
        let (mut taken, mut feed) = (0, None);
        // The follow of the host's registry, once it is handed.
        let mut members = None;
        // The named values of the report a telling tells, one row.
        let mut told = SignalCells::default();
        self.step(&upkeep, &mut members, &mut told, &mut taken, &mut feed);
        loop {
            tokio::select! {
                _ = tick.tick() => {
                    self.step(&upkeep, &mut members, &mut told, &mut taken, &mut feed);
                }
                went_on = published(feed.as_mut()) => {
                    if went_on {
                        self.keep_loads(&mut members, &mut told, feed.as_mut());
                    } else {
                        feed = None;
                    }
                }
                word = membership::next(members.as_mut()) => self.heard(members.as_mut(), word),
            }
        }
    }

    /// Keeps the channel's newest snapshot and tells it to the stages that
    /// keep reports, where the document in force has a use for it: it
    /// reads a load report or a named signal, or a stage of it keeps
    /// reports. Under any other document the task keeps no snapshot.
    /// `told` is the row a telling states a report's named values in.
    fn keep_loads(
        &self,
        members: &mut Option<Follow>,
        told: &mut SignalCells,
        feed: Option<&mut LoadReceiver>,
    ) {
        let document = self.document.load();
        if document.reads_loads || document.spec.engine().keeps_loads() {
            drop(document);
            let tell = |newest: &_| self.tell_loads(members, told, newest);
            self.loads.keep(feed, tell);
        } else {
            self.loads.newest.store(None);
        }
    }

    /// One upkeep of the task, at its first poll and then once a second:
    /// `upkeep` under a catch of its own, then what the host handed since
    /// the upkeep before. That is the follow of the registry the host
    /// handed ([`Self::follow_registry`]), then the host's load channel,
    /// where it handed one the task has not taken: `taken` counts the
    /// hand-over `feed` came with. The hand-over is read ahead of the
    /// follow and told behind it: a new document owes its whole read of
    /// the registry ahead of its hand-over ([`Self::apply`]), so a
    /// hand-over the step reads finds that read made by its telling,
    /// whenever the document came in. Where a program's own code panicked
    /// in that read, the hand-over was told ahead of it: an upkeep behind
    /// one that owed the read and did not make it tells the snapshot at
    /// hand, so the workers the read tells as joined hear their reports at
    /// its upkeep and not at the monitor's next publication. `told` is the
    /// row the task keeps for the named values of the report a telling
    /// tells.
    fn step(
        &self,
        upkeep: &impl Fn(Instant),
        members: &mut Option<Follow>,
        told: &mut SignalCells,
        taken: &mut u64,
        feed: &mut Option<LoadReceiver>,
    ) {
        // Dated at this run, not at the second its tick was due.
        let now = time::Instant::now().into_std();
        let _swept = self.held(|| upkeep(now));
        let handed = self.loads.handed_past(*taken);
        // Ahead of a hand-over's telling: it tells the reports of the
        // workers the follow told as joined.
        let again = self.follow_registry(members);
        if let Some(handed) = handed {
            (*taken, *feed) = handed;
            self.keep_loads(members, told, feed.as_mut());
        } else if again {
            if let Some(newest) = self.loads.newest.load_full() {
                self.tell_loads(members, told, &newest);
            }
        }
    }
}

impl Loaded<'_, '_> {
    /// The snapshot a made request's decision reads, held for the
    /// decision ([`LoadsRead`]): the one the request gives where it gives
    /// one. Otherwise the newest the task keeps at hand, where the
    /// decision's document reads a load report or a named signal, and none
    /// under any other document. A single request's decision, a pair's
    /// first and a pair's decode decided again all read theirs here.
    #[inline]
    pub(super) fn loads_read<'a>(self, maker: &'a impl RequestMaker) -> LoadsRead<'a> {
        match maker.loads() {
            Some(given) => LoadsRead::Given(given),
            None => LoadsRead::Kept(self.document.reads_loads.then(|| self.loads.newest())),
        }
    }
}

#[cfg(test)]
pub(super) mod tests;
