//! The controller's readiness as one state every probe reads: the HTTP
//! `/readiness`, the gRPC health service's overall status and the state
//! gauge all follow it. A controller is ready as the leader and as a
//! follower: both answer `Leader`, and a rollout that adds a replica before
//! it removes one needs the follower's probe to pass. A follower that runs
//! a standby is ready only while it is synced with its leader (design
//! 5.4): before its first frame, and after a silence of its stream as long
//! as the bound, it is `Syncing` and the probes answer not ready, so a
//! rollout waits until the replica that would take over holds the
//! leader's state.
//!
//! The role is never told: it is read from the fence at the instant of the
//! question, so a leader whose renewals stalled reads as a follower from the
//! moment its fence closes, whether or not its election thread has noticed.
//! A waiter is woken at that moment too: it sleeps to the instant the
//! fence's window ends, and to the one a take's window opens.
//!
//! The read handle also answers two reads of the replica's membership log
//! for a host: where the log stands, and the entries behind a place
//! ([`Readiness::log_head`], [`Readiness::log_after`]).

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::watch;

use super::{election::ElectionObservation, standby::Lag};
use crate::{
    service::{
        elector::Fence,
        replication::{Entry, Log, LogPositionLost},
    },
    store::Position,
};

/// Where the controller is between its start and its exit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum State {
    /// Binding, wiring and the first look at the lease; the probes answer
    /// not ready. A controller that cannot reach its lease stays here.
    Starting,
    /// Serving, and the fence is open: this replica leads.
    Leader,
    /// Serving, the fence is closed, and this replica's standby is not
    /// synced with its leader: the probes answer not ready.
    Syncing,
    /// Serving, and the fence is closed: another replica leads, or none.
    /// A replica that runs a standby is synced with its leader.
    Follower,
    /// The shutdown signal arrived: the probes answer not ready while the
    /// listeners settle and drain, then the process exits.
    Draining,
}

impl State {
    /// Every state, in order of a life.
    pub const ALL: [State; 5] = [
        State::Starting,
        State::Leader,
        State::Syncing,
        State::Follower,
        State::Draining,
    ];

    /// The state's name in probe bodies, log lines and the state gauge's
    /// label.
    pub const fn as_str(self) -> &'static str {
        match self {
            State::Starting => "starting",
            State::Leader => "leader",
            State::Syncing => "syncing",
            State::Follower => "follower",
            State::Draining => "draining",
        }
    }

    /// Whether the probes answer ready: as the leader and as a follower,
    /// which a replica that runs a standby is only while it is synced.
    pub const fn is_ready(self) -> bool {
        matches!(self, State::Leader | State::Follower)
    }
}

/// Where the run loop is. Between its two ends the fence says the role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    Starting,
    Serving,
    Draining,
}

/// A read handle on the controller's state: cheap to clone, cheap to read,
/// and awaitable for a transition. A host also reads the replica's
/// membership log through it ([`Self::log_head`], [`Self::log_after`]),
/// and writes nothing.
#[derive(Clone, Debug)]
pub struct Readiness {
    phase: watch::Receiver<Phase>,
    /// The election's steps: a waiter looks again after each.
    steps: watch::Receiver<ElectionObservation>,
    fence: Arc<Fence>,
    /// The standby's lag: a follower that runs one is ready while synced.
    lag: Arc<Lag>,
    /// The lag's turns: a waiter looks again after each.
    turns: watch::Receiver<u64>,
    /// The replica's membership log, for the two reads a host has of it.
    /// The handle hands no log out: a host writes nothing through it.
    log: Arc<Log>,
}

/// One scrape's read of the controller: its state and its standby's lag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sample {
    pub(crate) state: State,
    /// The age of the last frame that left this replica at its leader's
    /// head; `None` while this replica leads and before any such frame.
    pub(crate) lag: Option<Duration>,
    /// Whether this replica does not lead and is synced with its leader.
    pub(crate) synced: bool,
    /// How many times this replica asked its leader to be restored.
    pub(crate) restores: u64,
}

impl From<State> for Sample {
    fn from(state: State) -> Self {
        Self {
            state,
            lag: None,
            synced: false,
            restores: 0,
        }
    }
}

impl Readiness {
    /// The state now: the role from the fence at this instant, and a
    /// follower's sync from its standby's lag at the same one.
    pub fn state(&self) -> State {
        self.read().0
    }

    /// Where the replica's membership log stands: the place of its last
    /// entry ([`Log::head`]), or the default before any. The entry is a
    /// write its store journaled or one of the log's own marks. A host
    /// keeps the place as a mark and reads on behind it
    /// ([`Self::log_after`]).
    pub fn log_head(&self) -> Position {
        self.log.head()
    }

    /// The entries of the replica's membership log after `after`, oldest
    /// first, into `out` (cleared first), as [`Log::read`] answers them.
    /// They are the membership and status writes the replica's store
    /// journaled, in the order written. The log's own marks stand among
    /// them ([`Event::Term`](crate::service::replication::Event::Term),
    /// [`Event::Generation`](crate::service::replication::Event::Generation)).
    ///
    /// A call reads at most `at_most` and never more than [`Log::BATCH`],
    /// all under one term. So a short batch is not the head. A host reads
    /// on from the place of the last entry it took until a read finds
    /// nothing. The read answers [`LogPositionLost`] where the log no
    /// longer holds `after`.
    pub fn log_after(
        &self,
        after: Position,
        at_most: usize,
        out: &mut Vec<Entry>,
    ) -> Result<(), LogPositionLost> {
        self.log.read(after, at_most, out)
    }

    /// The state, the lag and the sync at one instant.
    pub(crate) fn sample(&self) -> Sample {
        self.sample_at(Instant::now()).0
    }

    /// The state now and the instant the clock alone next turns it, by one
    /// read of the clock, of the lag and of the fence: where a follower's
    /// sync lapses with no further frame, and where the fence's window
    /// opens or ends with no further write of the election. A waiter
    /// sleeps to that instant ([`Readiness::moved_by`]), so no such turn
    /// falls between two reads.
    pub(crate) fn read(&self) -> (State, Option<Instant>) {
        let (sample, turn) = self.sample_at(Instant::now());
        (sample.state, turn)
    }

    /// [`Readiness::read`] for the teller of the role: the term this
    /// replica leads at the instant of the read, if it leads one, and the
    /// instant the clock alone next turns the state. The term is the one
    /// the state was read under, by one read of the fence.
    ///
    /// The term is the fence's from the first take on. A take lands ahead
    /// of the serve's turn to serving, and the fence admits a watch and a
    /// write from the take: a serve that is still [`State::Starting`]
    /// leads the term its fence holds. A serve that is
    /// [`State::Draining`] leads none, whatever its fence says: a leader
    /// that leaves says so before its fence closes.
    pub(crate) fn led(&self) -> (Option<u64>, Option<Instant>) {
        let now = Instant::now();
        let term = self.fence.leader_term(now);
        let (sample, turn) = self.sample_under(term, now);
        let leads = matches!(sample.state, State::Starting | State::Leader);
        (term.filter(|_| leads), turn)
    }

    fn sample_at(&self, now: Instant) -> (Sample, Option<Instant>) {
        self.sample_under(self.fence.leader_term(now), now)
    }

    /// The sample at `now` under `term`, the fence's word at that instant.
    fn sample_under(&self, term: Option<u64>, now: Instant) -> (Sample, Option<Instant>) {
        let leads = term.is_some();
        // A leader follows nobody: it has no lag and no sync to lapse.
        let (lag, lapse) = match self.lag.read(now) {
            Some((age, lapse)) if !leads => (Some(age), lapse),
            _ => (None, None),
        };
        let synced = lapse.is_some();
        let state = match *self.phase.borrow() {
            Phase::Starting => State::Starting,
            Phase::Draining => State::Draining,
            Phase::Serving if leads => State::Leader,
            Phase::Serving if self.lag.follows() && !synced => State::Syncing,
            Phase::Serving => State::Follower,
        };
        // The clock alone turns the state at the sync's lapse and at the
        // fence's next turn: the earlier of the two.
        let turn = match (lapse, self.fence.turns_at(now)) {
            (Some(lapse), Some(fence)) => Some(lapse.min(fence)),
            (lapse, fence) => lapse.or(fence),
        };
        let restores = self.lag.asks();
        let sample = Sample {
            state,
            lag,
            synced,
            restores,
        };
        (sample, turn)
    }

    /// Resolves with `wanted` once the state is `wanted`, at once when it
    /// already is; `None` when the controller is gone before that. The
    /// state is read again at every move of the run loop, after every
    /// election step, when a frame turns a standby's sync, when that sync
    /// lapses, and where the fence's window opens or ends: a fence that
    /// closes between two steps, or opens there one renew deadline after a
    /// take, is seen at that instant.
    pub async fn wait_for(&mut self, wanted: State) -> Option<State> {
        loop {
            let (state, turn) = self.read();
            if state == wanted {
                return Some(wanted);
            }
            self.moved_by(turn).await?;
        }
    }

    /// Resolves at the next move of the run loop or step of the election,
    /// when a frame turns a standby's sync, and at `turn`, the instant
    /// its caller read with the state ([`Readiness::read`]): a follower of
    /// the role reads the state again then. An instant already past
    /// resolves at once, so a state the clock turned since that read is
    /// read again. `None` when the controller is gone.
    pub(crate) async fn moved_by(&mut self, turn: Option<Instant>) -> Option<()> {
        tokio::select! {
            Ok(()) = self.turns.changed() => Some(()),
            () = reached(turn) => Some(()),
            moved = self.phase.changed() => moved.ok(),
            stepped = self.steps.changed() => {
                if stepped.is_err() {
                    // The election is gone: the run loop alone moves.
                    self.phase.changed().await.ok()?;
                }
                Some(())
            }
        }
    }
}

/// Resolves at `turn`, at once where it is past; never where there is
/// none.
pub(super) async fn reached(turn: Option<Instant>) {
    match turn {
        Some(turn) => tokio::time::sleep_until(turn.into()).await,
        None => std::future::pending().await,
    }
}

/// The writer the run loop holds and the readers it hands out.
pub(crate) fn channel(
    fence: Arc<Fence>,
    steps: watch::Receiver<ElectionObservation>,
    lag: Arc<Lag>,
    log: Arc<Log>,
) -> (watch::Sender<Phase>, Readiness) {
    let (sender, phase) = watch::channel(Phase::Starting);
    let turns = lag.turns();
    (
        sender,
        Readiness {
            phase,
            steps,
            fence,
            lag,
            turns,
            log,
        },
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/readiness/tests.rs"]
pub(crate) mod tests;
