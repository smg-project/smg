//! The engine's own task: what a host of the engine no longer does by hand.
//!
//! A host hands the engine its load monitor's channel
//! ([`PolicyEngine::set_loads`]) and its worker registry
//! ([`PolicyEngine::set_registry`]), and spawns [`PolicyEngine::run`] once.
//! The task runs the engine's upkeep once a second and keeps the newest
//! load snapshot at hand, so a decision whose request gives none reads the
//! engine's and never touches the channel. It tells the document's stages
//! a worker that joined or was removed, by the registry's events and under
//! the id the host states of the worker.

use std::{
    fmt,
    future::pending,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use arc_swap::{ArcSwapOption, Guard};
use parking_lot::Mutex;
use smg_worker::load::state::{LoadReceiver, LoadSnapshot};
use tokio::time::{self, MissedTickBehavior};

use super::{membership, PolicyEngine};
use crate::engine::{
    decision_engine::{DecisionEngine, RequestMaker, SweepOutcome},
    framework::release,
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
    /// Whether the host ever handed a channel: a decision of an engine
    /// handed none reads this one word.
    handed: AtomicBool,
    /// The newest snapshot the engine's task read from the channel. The
    /// task alone writes it ([`PolicyEngine::run`]).
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
    /// The newest snapshot at hand, for a decision whose request gives
    /// none: none where the host handed no channel, or the task has not
    /// read it yet.
    #[inline]
    pub(super) fn newest(&self) -> Option<Guard<Option<Arc<LoadSnapshot>>>> {
        let handed = self.handed.load(Ordering::Relaxed);
        handed.then(|| self.newest.load())
    }

    /// The channel the host handed last and its count, where the task took
    /// the hand-over of another count: `taken`.
    fn handed_past(&self, taken: u64) -> Option<(u64, Option<LoadReceiver>)> {
        let feed = self.feed.lock();
        (feed.0 != taken).then(|| feed.clone())
    }

    /// Keeps the newest snapshot of `feed` at hand.
    fn keep(&self, feed: Option<&mut LoadReceiver>) {
        if let Some(feed) = feed {
            let newest = Arc::clone(&feed.borrow_and_update());
            self.newest.store(Some(newest));
        }
    }
}

/// The snapshot one decision reads, held for as long as the decision reads
/// it ([`PolicyEngine::loads_read`]).
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
    /// An engine whose document reads no load report
    /// ([`Needs::loads`](crate::document::spec::Needs::loads)) and no named
    /// signal
    /// ([`Needs::signal_cells`](crate::document::spec::Needs::signal_cells))
    /// keeps none, and no decision of it reads a report. A named signal's
    /// values stand in the same reports, so a document that reads one keeps
    /// the channel as one that reads a report does.
    ///
    /// The engine's task reads the channel ([`Self::run`]). A decision
    /// reads the snapshot its request gives where it gives one
    /// ([`RequestMaker::loads`](crate::engine::RequestMaker::loads)), and
    /// the newest the task keeps at hand otherwise: a single request's, a
    /// pair's first, and a pair's decode decided again.
    pub fn set_loads(&self, loads: LoadReceiver) {
        let needs = self.spec.needs();
        if needs.loads || needs.signal_cells {
            let mut feed = self.loads.feed.lock();
            *feed = (feed.0 + 1, Some(loads));
            self.loads.handed.store(true, Ordering::Relaxed);
        }
    }

    /// The snapshot a made request's decision reads, held for the
    /// decision ([`LoadsRead`]): the one the request gives where it gives
    /// one, and the newest the task keeps at hand otherwise. A single
    /// request's decision, a pair's first and a pair's decode decided again
    /// all read theirs here.
    #[inline]
    pub(super) fn loads_read<'a>(&self, maker: &'a impl RequestMaker) -> LoadsRead<'a> {
        match maker.loads() {
            Some(given) => LoadsRead::Given(given),
            None => LoadsRead::Kept(self.loads.newest()),
        }
    }

    /// The engine's own task: what its host no longer does by hand. The
    /// host spawns it once, at its start, on a runtime whose timer is
    /// enabled, and it runs for as long as the engine stands.
    ///
    /// - The upkeep once a second ([`DecisionEngine::sweep`]), dated at its
    ///   run by the runtime's clock. A task that was not run for a while
    ///   runs it once for the seconds it missed. A panic in it is held and
    ///   counted ([`Self::upkeep_faults`]): the task goes on.
    /// - The newest load snapshot kept at hand: read from the host's
    ///   channel at the first upkeep behind its hand-over, so within a
    ///   second of it, and swapped when the load monitor publishes one.
    ///   Until then, and for an engine handed no channel, a decision whose
    ///   request gives no snapshot reads no report. A monitor that ended
    ///   leaves its last snapshot standing until another is handed.
    /// - A worker that joined or was removed told to the document's
    ///   stages, by the events of the registry the host handed
    ///   ([`Self::set_registry`]) and under the id the host states of the
    ///   worker: followed from the first upkeep behind the hand-over, so
    ///   within a second of it, with the workers that stand then told as
    ///   joined. The registry is read whole again where its channel
    ///   dropped events or closed. A panic of the host's own code in the
    ///   follow is held and counted ([`Self::upkeep_faults`]): the task
    ///   goes on, and reads the registry whole again at its next upkeep.
    pub async fn run(&self) {
        self.run_with(|now| self.sweep(now)).await;
    }

    /// How many steps of the engine's own task panicked ([`Self::run`]): a
    /// run of the upkeep, or a step of the registry's follow in the host's
    /// own code. Each was held, and the task went on. While the upkeep's
    /// panics raise the count it does not run whole: session keys past
    /// their end can stay on file, though every request is still answered.
    /// While the follow's raise it, a join whose answer of the host
    /// panicked is lost until a whole read of the registry succeeds. A
    /// removal asks the host nothing and is still told. A host whose code
    /// panics at one worker every time fails every whole read.
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

    /// The task's loop. `upkeep` is the step it runs once a second under
    /// a catch of its own: [`Self::run`] hands the engine's own.
    async fn run_with(&self, upkeep: impl Fn(Instant) -> SweepOutcome) {
        let mut tick = time::interval(UPKEEP);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // The host's channel while its monitor publishes, and the count of
        // the hand-over it came with.
        let (mut taken, mut feed) = (0, None);
        // The follow of the host's registry, once it is handed.
        let mut members = None;
        loop {
            tokio::select! {
                _ = tick.tick() => {
                    // Dated at this run, not at the second its tick was due.
                    let now = time::Instant::now().into_std();
                    let _swept = self.held(|| upkeep(now));
                    if let Some(handed) = self.loads.handed_past(taken) {
                        (taken, feed) = handed;
                        self.loads.keep(feed.as_mut());
                    }
                    self.follow_registry(&mut members);
                }
                went_on = published(feed.as_mut()) => {
                    if went_on {
                        self.loads.keep(feed.as_mut());
                    } else {
                        feed = None;
                    }
                }
                word = membership::next(members.as_mut()) => self.heard(members.as_mut(), word),
            }
        }
    }
}

#[cfg(test)]
mod tests;
