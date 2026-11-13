//! The controller's readiness as one state every probe reads: the HTTP
//! `/readiness`, the gRPC health service's overall status and the state
//! gauge all follow it. From PR 10 on it means "synced and leader" (design
//! 10); here it means "bound and serving".

use tokio::sync::watch;

/// Where the controller is between its start and its exit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    /// Binding and wiring; nothing answers yet.
    Starting,
    /// Every listener serves and the store is built: the probes answer
    /// ready.
    Ready,
    /// The shutdown signal arrived: the probes answer not ready while the
    /// listeners settle and drain, then the process exits.
    Draining,
}

impl State {
    /// Every state, in order of a life.
    pub const ALL: [State; 3] = [State::Starting, State::Ready, State::Draining];

    /// The state's name in probe bodies, log lines and the state gauge's
    /// label.
    pub const fn as_str(self) -> &'static str {
        match self {
            State::Starting => "starting",
            State::Ready => "ready",
            State::Draining => "draining",
        }
    }

    /// Whether the probes answer ready.
    pub const fn is_ready(self) -> bool {
        matches!(self, State::Ready)
    }
}

/// A read handle on the controller's state: cheap to clone, cheap to read,
/// and awaitable for a transition.
#[derive(Clone, Debug)]
pub struct Readiness(watch::Receiver<State>);

impl Readiness {
    /// The state now.
    pub fn state(&self) -> State {
        *self.0.borrow()
    }

    /// Resolves with `wanted` once the state is `wanted`, at once when it
    /// already is; `None` when the controller is gone before that.
    pub async fn wait_for(&mut self, wanted: State) -> Option<State> {
        self.0
            .wait_for(|state| *state == wanted)
            .await
            .ok()
            .map(|state| *state)
    }
}

/// The writer the run loop holds and the readers it hands out.
pub(crate) fn channel() -> (watch::Sender<State>, Readiness) {
    let (sender, receiver) = watch::channel(State::Starting);
    (sender, Readiness(receiver))
}

#[cfg(test)]
mod tests {
    use super::{channel, State};

    #[test]
    fn only_ready_is_ready_and_every_state_has_a_name() {
        assert!(State::Ready.is_ready());
        assert!(!State::Starting.is_ready());
        assert!(!State::Draining.is_ready());
        assert_eq!(
            State::ALL.map(State::as_str),
            ["starting", "ready", "draining"]
        );
    }

    #[tokio::test]
    async fn a_reader_sees_the_transition_and_the_writer_leaving() {
        let (writer, mut reader) = channel();
        assert_eq!(reader.state(), State::Starting);
        assert_eq!(
            reader.wait_for(State::Starting).await,
            Some(State::Starting)
        );

        writer.send_replace(State::Ready);
        assert_eq!(reader.wait_for(State::Ready).await, Some(State::Ready));
        assert_eq!(reader.clone().state(), State::Ready);

        drop(writer);
        assert_eq!(reader.wait_for(State::Draining).await, None);
    }
}
