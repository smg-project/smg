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

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::watch;

use super::{election::ElectionObservation, standby::Lag};
use crate::service::elector::Fence;

/// Where the controller is between its start and its exit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
/// and awaitable for a transition.
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
    pub(crate) fn led(&self) -> (Option<u64>, Option<Instant>) {
        let now = Instant::now();
        let term = self.fence.leader_term(now);
        let (sample, turn) = self.sample_under(term, now);
        (term.filter(|_| sample.state == State::Leader), turn)
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
        },
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };

    use tokio::sync::watch;

    use super::{channel, Phase, Readiness, Sample, State};
    use crate::service::{
        controller::{election::ElectionObservation, standby::Lag},
        elector::{ElectionSettings, Elector, MemoryLease, TickOutcome},
    };

    /// A candidate alone over a lease of its own and a readiness that reads
    /// its fence: the run loop's writer, the election's writer, the reader.
    pub(crate) fn fenced() -> (
        Elector,
        watch::Sender<Phase>,
        watch::Sender<ElectionObservation>,
        Readiness,
    ) {
        over(
            Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
                .expect("the default settings are in order"),
        )
    }

    /// The same over the candidate that is handed in.
    fn over(
        elector: Elector,
    ) -> (
        Elector,
        watch::Sender<Phase>,
        watch::Sender<ElectionObservation>,
        Readiness,
    ) {
        let (steps, stepped) = watch::channel(ElectionObservation::default());
        let (phase, readiness) = channel(elector.fence(), stepped, Arc::new(Lag::new(BOUND)));
        (elector, phase, steps, readiness)
    }

    /// The bound the cases' standby reads synced under: no case waits it
    /// out.
    const BOUND: Duration = Duration::from_secs(60);

    #[test]
    fn a_leader_and_a_follower_are_ready_and_every_state_has_a_name() {
        assert!(State::Leader.is_ready());
        assert!(State::Follower.is_ready());
        assert!(!State::Starting.is_ready());
        assert!(!State::Draining.is_ready());
        assert!(!State::Syncing.is_ready());
        assert_eq!(
            State::ALL.map(State::as_str),
            ["starting", "leader", "syncing", "follower", "draining"]
        );
    }

    #[test]
    fn the_role_is_the_fences_at_the_instant_of_the_read() {
        // A lease that is handed in may have a holder this candidate never
        // saw, so its take waits to lead.
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(100),
            renew_deadline: Duration::from_millis(400),
        };
        let (mut elector, phase, _steps, readiness) = over(
            Elector::new(
                Box::new(MemoryLease::new()),
                "controller-0@127.0.0.1:1",
                settings,
            )
            .expect("the settings are in order"),
        );
        assert_eq!(readiness.state(), State::Starting);

        // Serving with no lease held: a follower.
        phase.send_replace(Phase::Serving);
        assert_eq!(readiness.state(), State::Follower);

        // The take of a lease that held no record opens no fence: still a
        // follower, through the renewals inside the wait. The lead opens one
        // renew deadline after the take's write, between two steps: the
        // leader from that instant, with no step taken and no message sent.
        // The renewals are a chain ahead of the clock, each just short of
        // the deadline the one before it stamped: the fence stays open
        // seconds past the opening, so the read needs no narrow window.
        let taken = Instant::now();
        assert_eq!(
            elector.tick(taken).expect("the take"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(readiness.state(), State::Follower);
        let mut renewed = taken;
        for _ in 0..10 {
            renewed += settings.renew_deadline - Duration::from_millis(10);
            assert_eq!(
                elector.tick(renewed).expect("the renewal"),
                TickOutcome::Renewed { term: 1 }
            );
            assert_eq!(readiness.state(), State::Follower);
        }
        thread::sleep(settings.renew_deadline);
        assert_eq!(readiness.state(), State::Leader);

        // Draining says so whatever the fence says.
        phase.send_replace(Phase::Draining);
        assert_eq!(readiness.state(), State::Draining);
    }

    #[test]
    fn a_fence_past_its_deadline_reads_as_a_follower_though_no_step_said_so() {
        let (mut elector, phase, _steps, readiness) = fenced();
        phase.send_replace(Phase::Serving);

        // A last renewal dated a second past the renew deadline: the holder
        // still holds the record, and its fence is closed at the read.
        let stale = Instant::now()
            .checked_sub(ElectionSettings::default().renew_deadline + Duration::from_secs(1))
            .expect("an instant three seconds back");
        elector.lead_at(stale);
        assert!(elector.is_holder());
        assert_eq!(readiness.state(), State::Follower);

        // The next renewal reopens it.
        assert_eq!(
            elector.tick(Instant::now()).expect("the renewal"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(readiness.state(), State::Leader);
    }

    /// A wait that fails where a state that never comes would hang it.
    async fn within(reader: &mut Readiness, wanted: State) -> Option<State> {
        tokio::time::timeout(Duration::from_secs(10), reader.wait_for(wanted))
            .await
            .expect("the wait ends within ten seconds")
    }

    #[tokio::test]
    async fn a_reader_sees_the_transitions_and_the_writer_leaving() {
        let (mut elector, phase, steps, mut reader) = fenced();
        assert_eq!(
            within(&mut reader, State::Starting).await,
            Some(State::Starting)
        );

        phase.send_replace(Phase::Serving);
        assert_eq!(
            within(&mut reader, State::Follower).await,
            Some(State::Follower)
        );

        // The waiter looks again after an election step.
        let mut waiter = reader.clone();
        let (waited, ()) = tokio::join!(within(&mut waiter, State::Leader), async {
            tokio::task::yield_now().await;
            elector.lead_at(Instant::now());
            steps.send_replace(ElectionObservation::default());
        });
        assert_eq!(waited, Some(State::Leader));
        assert_eq!(reader.clone().state(), State::Leader);

        // The election gone, the run loop's moves still wake the waiter.
        drop(steps);
        let mut waiter = reader.clone();
        let (waited, ()) = tokio::join!(within(&mut waiter, State::Draining), async {
            tokio::task::yield_now().await;
            phase.send_replace(Phase::Draining);
        });
        assert_eq!(waited, Some(State::Draining));

        drop(phase);
        assert_eq!(within(&mut reader, State::Leader).await, None);
    }

    /// The clock alone turns a leader's state where its fence's window
    /// ends: a waiter is woken there, though the election took no step, no
    /// frame came and the run loop did not move.
    #[tokio::test]
    async fn a_waiter_is_woken_where_the_fences_window_ends_with_no_step() {
        let (mut elector, phase, _steps, mut reader) = fenced();
        phase.send_replace(Phase::Serving);
        reader.lag.follow();
        // A take dated so that its window ends a tenth of a second from
        // now: a renewal that stalls from here closes the fence then.
        let now = Instant::now();
        let ends = now + Duration::from_millis(100);
        let taken = ends
            .checked_sub(ElectionSettings::default().renew_deadline)
            .expect("an instant a renew deadline back");
        elector.lead_at(taken);
        let (sample, turn) = reader.sample_at(now);
        assert_eq!((sample.state, turn), (State::Leader, Some(ends)));
        assert_eq!(
            within(&mut reader, State::Syncing).await,
            Some(State::Syncing)
        );
    }

    #[test]
    fn a_follower_that_runs_a_standby_is_ready_only_while_it_is_synced() {
        let (mut elector, phase, _steps, readiness) = fenced();
        phase.send_replace(Phase::Serving);
        let lag = &readiness.lag;
        let read = |at| readiness.sample_at(at).0;
        let sample = |state, lag, synced| Sample {
            state,
            lag,
            synced,
            restores: 0,
        };

        // No standby runs: a follower is ready, as before one existed.
        let start = Instant::now();
        assert_eq!(read(start), sample(State::Follower, None, false));

        // The serve runs one: not ready before its first frame.
        lag.follow();
        assert_eq!(read(start), sample(State::Syncing, None, false));

        // A frame: ready while it is younger than the bound, not at it.
        lag.applied(start);
        // The lag holds the frame's instant at a whole microsecond.
        let frame = start - lag.age(start).expect("a frame applied");
        let under = BOUND - Duration::from_micros(1);
        assert_eq!(
            read(frame + under),
            sample(State::Follower, Some(under), true)
        );
        assert_eq!(
            read(frame + BOUND),
            sample(State::Syncing, Some(BOUND), false)
        );

        // The lapse comes with the state, by the same read of the clock.
        assert_eq!(readiness.sample_at(frame + under).1, Some(frame + BOUND));
        assert_eq!(readiness.sample_at(frame + BOUND).1, None);

        // The next frame ends the silence.
        lag.applied(frame + BOUND);
        assert_eq!(
            read(frame + BOUND),
            sample(State::Follower, Some(Duration::ZERO), true)
        );

        // The fence says leader whatever the lag says, and a leader
        // follows nobody: it has no lag and is not synced.
        let taken = Instant::now();
        assert_eq!(
            elector.tick(taken).expect("the take"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(read(taken), sample(State::Leader, None, false));

        // A leader has no sync to lapse: the clock turns its state where
        // its fence's window ends, one renew deadline past the take.
        let ends = taken + ElectionSettings::default().renew_deadline;
        assert_eq!(readiness.sample_at(taken).1, Some(ends));
        assert_eq!(read(ends), sample(State::Leader, None, false));
        // Past it the fence is closed, and the sync's lapse alone is ahead:
        // the last frame is dated after this instant.
        let past = ends + Duration::from_nanos(1);
        assert_eq!(
            read(past),
            sample(State::Follower, Some(Duration::ZERO), true)
        );
        assert_eq!(readiness.sample_at(past).1, Some(frame + BOUND + BOUND));

        // Draining says so whatever the lag says.
        phase.send_replace(Phase::Draining);
        assert_eq!(readiness.state(), State::Draining);
    }

    #[tokio::test]
    async fn a_waiter_looks_again_when_a_frame_ends_a_silence_and_when_the_sync_lapses() {
        // A frame ends the silence: the lag tells the waiter, with no move
        // of the run loop and no step of the election.
        let (elector, phase, _steps, mut reader) = fenced();
        phase.send_replace(Phase::Serving);
        let lag = Arc::clone(&reader.lag);
        lag.follow();
        assert_eq!(
            within(&mut reader, State::Syncing).await,
            Some(State::Syncing)
        );
        let mut waiter = reader.clone();
        let (waited, ()) = tokio::join!(within(&mut waiter, State::Follower), async {
            tokio::task::yield_now().await;
            lag.applied(Instant::now());
        });
        assert_eq!(waited, Some(State::Follower));

        // The sync lapses by the clock alone, under a bound short enough
        // to wait out: the waiter is at the frame well inside it.
        let short = Arc::new(Lag::new(Duration::from_millis(300)));
        short.follow();
        let (_standing, stepped) = watch::channel(ElectionObservation::default());
        let (phase, mut reader) = channel(elector.fence(), stepped, Arc::clone(&short));
        phase.send_replace(Phase::Serving);
        short.applied(Instant::now());
        assert_eq!(
            within(&mut reader, State::Syncing).await,
            Some(State::Syncing)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_waiter_wakes_at_the_lapse_it_read_and_no_later() {
        const HOUR: Duration = Duration::from_secs(3600);
        // A timer is a millisecond coarse.
        const COARSE: Duration = Duration::from_millis(2);
        let (_elector, _phase, _steps, mut reader) = fenced();
        // The paused clock steps to the next timer: its reading at the
        // wake is the instant the waiter slept to.
        let start = tokio::time::Instant::now();
        let held = Duration::from_millis(300);
        let lapse = start.into_std() + held;
        let woke = tokio::time::timeout(HOUR, reader.moved_by(Some(lapse))).await;
        assert!(matches!(woke, Ok(Some(()))), "{woke:?}");
        let late = start.elapsed().checked_sub(held);
        assert!(late.is_some_and(|late| late < COARSE), "{late:?}");

        // A lapse already past wakes at once: a sync that lapsed since
        // its read is read again.
        let past = tokio::time::Instant::now();
        let woke = tokio::time::timeout(HOUR, reader.moved_by(Some(lapse))).await;
        assert!(matches!(woke, Ok(Some(()))), "{woke:?}");
        assert!(past.elapsed() < COARSE, "{:?}", past.elapsed());

        // With no lapse the waiter sleeps on.
        let woke = tokio::time::timeout(HOUR, reader.moved_by(None)).await;
        assert!(woke.is_err(), "{woke:?}");
    }

    #[test]
    fn a_sample_tells_the_restores_the_standby_asked_whatever_the_state() {
        let (mut elector, phase, _steps, readiness) = fenced();
        phase.send_replace(Phase::Serving);
        let lag = &readiness.lag;
        lag.follow();
        let asked = |at| readiness.sample_at(at).0.restores;
        assert_eq!(asked(Instant::now()), 0);
        lag.asked();
        lag.asked();
        assert_eq!(asked(Instant::now()), 2);
        // The count is the replica's since its start: a take keeps it.
        assert_eq!(
            elector.tick(Instant::now()).expect("the take"),
            TickOutcome::Acquired { term: 1 }
        );
        let sample = readiness.sample_at(Instant::now()).0;
        assert_eq!((sample.state, sample.restores), (State::Leader, 2));
    }
}
