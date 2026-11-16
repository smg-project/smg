//! The controller's readiness as one state every probe reads: the HTTP
//! `/readiness`, the gRPC health service's overall status and the state
//! gauge all follow it. A controller is ready as the leader and as a
//! follower: both answer `Leader`, and a rollout that adds a replica before
//! it removes one needs the follower's probe to pass. "Synced" joins a
//! follower's readiness with the standby (design 5.4).
//!
//! The role is never told: it is read from the fence at the instant of the
//! question, so a leader whose renewals stalled reads as a follower from the
//! moment its fence closes, whether or not its election thread has noticed.

use std::{sync::Arc, time::Instant};

use tokio::sync::watch;

use super::election::Standing;
use crate::elector::Fence;

/// Where the controller is between its start and its exit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    /// Binding, wiring and the first look at the lease; the probes answer
    /// not ready. A controller that cannot reach its lease stays here.
    Starting,
    /// Serving, and the fence is open: this replica leads.
    Leader,
    /// Serving, and the fence is closed: another replica leads, or none.
    Follower,
    /// The shutdown signal arrived: the probes answer not ready while the
    /// listeners settle and drain, then the process exits.
    Draining,
}

impl State {
    /// Every state, in order of a life.
    pub const ALL: [State; 4] = [
        State::Starting,
        State::Leader,
        State::Follower,
        State::Draining,
    ];

    /// The state's name in probe bodies, log lines and the state gauge's
    /// label.
    pub const fn as_str(self) -> &'static str {
        match self {
            State::Starting => "starting",
            State::Leader => "leader",
            State::Follower => "follower",
            State::Draining => "draining",
        }
    }

    /// Whether the probes answer ready: as the leader and as a follower.
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
    steps: watch::Receiver<Standing>,
    fence: Arc<Fence>,
}

impl Readiness {
    /// The state now, the role from the fence at this instant.
    pub fn state(&self) -> State {
        match *self.phase.borrow() {
            Phase::Starting => State::Starting,
            Phase::Draining => State::Draining,
            Phase::Serving => {
                if self.fence.leader_term(Instant::now()).is_some() {
                    State::Leader
                } else {
                    State::Follower
                }
            }
        }
    }

    /// Resolves with `wanted` once the state is `wanted`, at once when it
    /// already is; `None` when the controller is gone before that. The
    /// state is read again at every move of the run loop and after every
    /// election step; a fence that closes between two steps is seen at the
    /// next.
    pub async fn wait_for(&mut self, wanted: State) -> Option<State> {
        loop {
            if self.state() == wanted {
                return Some(wanted);
            }
            tokio::select! {
                moved = self.phase.changed() => moved.ok()?,
                stepped = self.steps.changed() => {
                    if stepped.is_err() {
                        // The election is gone: the run loop alone moves.
                        self.phase.changed().await.ok()?;
                    }
                }
            }
        }
    }
}

/// The writer the run loop holds and the readers it hands out.
pub(crate) fn channel(
    fence: Arc<Fence>,
    steps: watch::Receiver<Standing>,
) -> (watch::Sender<Phase>, Readiness) {
    let (sender, phase) = watch::channel(Phase::Starting);
    (
        sender,
        Readiness {
            phase,
            steps,
            fence,
        },
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use std::time::{Duration, Instant};

    use tokio::sync::watch;

    use super::{channel, Phase, Readiness, State};
    use crate::{
        controller::election::Standing,
        elector::{ElectionSettings, Elector, MemoryLease, Tick},
    };

    /// A candidate over a lease of its own and a readiness that reads its
    /// fence: the run loop's writer, the election's writer, the reader.
    pub(crate) fn fenced() -> (
        Elector,
        watch::Sender<Phase>,
        watch::Sender<Standing>,
        Readiness,
    ) {
        let elector = Elector::new(
            Box::new(MemoryLease::new()),
            "controller-0@127.0.0.1:1",
            ElectionSettings::default(),
        )
        .expect("the default settings are in order");
        let (steps, stepped) = watch::channel(Standing::default());
        let (phase, readiness) = channel(elector.fence(), stepped);
        (elector, phase, steps, readiness)
    }

    #[test]
    fn a_leader_and_a_follower_are_ready_and_every_state_has_a_name() {
        assert!(State::Leader.is_ready());
        assert!(State::Follower.is_ready());
        assert!(!State::Starting.is_ready());
        assert!(!State::Draining.is_ready());
        assert_eq!(
            State::ALL.map(State::as_str),
            ["starting", "leader", "follower", "draining"]
        );
    }

    #[test]
    fn the_role_is_the_fences_at_the_instant_of_the_read() {
        let (mut elector, phase, _steps, readiness) = fenced();
        assert_eq!(readiness.state(), State::Starting);

        // Serving with no lease held: a follower.
        phase.send_replace(Phase::Serving);
        assert_eq!(readiness.state(), State::Follower);

        // The take opens the fence: the leader, with no message sent.
        assert_eq!(
            elector.tick(Instant::now()).expect("the take"),
            Tick::Acquired { term: 1 }
        );
        assert_eq!(readiness.state(), State::Leader);

        // Draining says so whatever the fence says.
        phase.send_replace(Phase::Draining);
        assert_eq!(readiness.state(), State::Draining);
    }

    #[test]
    fn a_fence_past_its_deadline_reads_as_a_follower_though_no_step_said_so() {
        let (mut elector, phase, _steps, readiness) = fenced();
        phase.send_replace(Phase::Serving);

        // A take dated a second past the renew deadline: the holder still
        // holds the record, and its fence is closed at the read.
        let stale = Instant::now()
            .checked_sub(ElectionSettings::default().renew_deadline + Duration::from_secs(1))
            .expect("an instant three seconds back");
        assert_eq!(
            elector.tick(stale).expect("the take"),
            Tick::Acquired { term: 1 }
        );
        assert!(elector.is_holder());
        assert_eq!(readiness.state(), State::Follower);

        // The next renewal reopens it.
        assert_eq!(
            elector.tick(Instant::now()).expect("the renewal"),
            Tick::Renewed { term: 1 }
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
            elector.tick(Instant::now()).expect("the take");
            steps.send_replace(Standing::default());
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
}
