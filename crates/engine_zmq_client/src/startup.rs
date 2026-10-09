//! The startup handshake's deadline: a bound on silence that signs of life
//! reset, under a ceiling.
//!
//! An engine's start can run for the better part of an hour between its
//! first handshake message and its last: a large checkpoint streams in, the
//! KV cache is sized, graphs are captured, and nothing crosses the ZMQ wire
//! meanwhile. A fixed deadline on that wait kills healthy starts of large
//! models; no deadline waits forever on a wedged one. The budget here does
//! neither: a wait fails after [`StartupBudget::silence`] without a sign of
//! life, where a sign of life is a handshake message or a touch of the
//! [`EngineLiveness`] handle by whoever supervises the engine process, and
//! [`StartupBudget::ceiling`] bounds the whole start however alive the engine
//! is.

use std::{
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use tokio::time::{sleep_until, Instant};

use crate::error::{Error, Result};

/// The record's value before the first touch; a touch stores milliseconds
/// since `base` plus one, so that it is never this.
const NEVER: u64 = 0;

/// `at + after`, or a deadline that never comes when the sum does not fit an
/// `Instant`: a bound of thousands of years means "no bound", not a panic in
/// the background task that owns the handshake.
fn deadline(at: Instant, after: Duration) -> Instant {
    at.checked_add(after).unwrap_or_else(far_future)
}

/// Thirty years out: past any start, within what every platform's `Instant`
/// holds.
fn far_future() -> Instant {
    Instant::now() + Duration::from_secs(30 * 365 * 24 * 60 * 60)
}

/// Signs of life from outside the ZMQ wire. The lifecycle owner that launched
/// the engine process calls [`EngineLiveness::touch`] while it sees the
/// process alive; the handshake's silence bound then counts from the latest
/// sign of life, wire message or touch, instead of from the last message.
/// Clones share one record.
#[derive(Clone, Debug)]
pub struct EngineLiveness {
    base: Instant,
    last_seen_millis: Arc<AtomicU64>,
}

impl Default for EngineLiveness {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineLiveness {
    pub fn new() -> Self {
        Self {
            base: Instant::now(),
            last_seen_millis: Arc::new(AtomicU64::new(NEVER)),
        }
    }

    /// Record a sign of life now.
    pub fn touch(&self) {
        let millis = u64::try_from(self.base.elapsed().as_millis()).unwrap_or(u64::MAX - 1);
        self.last_seen_millis
            .fetch_max(millis + 1, Ordering::Release);
    }

    /// When the engine was last seen alive; `None` before the first touch.
    pub fn last_seen(&self) -> Option<Instant> {
        match self.last_seen_millis.load(Ordering::Acquire) {
            NEVER => None,
            stored => Some(self.base + Duration::from_millis(stored - 1)),
        }
    }
}

/// How long a startup handshake may take.
#[derive(Clone, Debug)]
pub struct StartupBudget {
    /// The longest stretch without a sign of life (a handshake message, an
    /// input registration, or a touch of `liveness`) before the handshake
    /// fails.
    pub silence: Duration,
    /// The longest the whole handshake may take, however alive the engine is;
    /// `None` leaves that to the silence bound and the lifecycle owner.
    pub ceiling: Option<Duration>,
    /// Signs of life from outside the wire; `None` counts wire messages only,
    /// which makes `silence` a plain per-message timeout.
    pub liveness: Option<EngineLiveness>,
}

impl From<Duration> for StartupBudget {
    /// A plain per-message timeout: wire messages only, no ceiling.
    fn from(silence: Duration) -> Self {
        Self {
            silence,
            ceiling: None,
            liveness: None,
        }
    }
}

/// The clock one handshake runs under.
pub(crate) struct StartupClock {
    started: Instant,
    budget: StartupBudget,
}

impl StartupClock {
    pub(crate) fn new(budget: impl Into<StartupBudget>) -> Self {
        Self {
            started: Instant::now(),
            budget: budget.into(),
        }
    }

    /// Await `future` under the budget: the wait fails with
    /// [`Error::HandshakeTimeout`] after `silence` without a sign of life, and
    /// with [`Error::StartupCeiling`] once the ceiling passes. `stage` names
    /// the wait in the error.
    pub(crate) async fn wait<T>(
        &self,
        stage: &'static str,
        future: impl Future<Output = T>,
    ) -> Result<T> {
        tokio::pin!(future);
        // The wire's own latest sign of life: the message before this wait.
        let last_message = Instant::now();
        loop {
            let now = Instant::now();
            if let Some(ceiling) = self.budget.ceiling {
                if now >= deadline(self.started, ceiling) {
                    return Err(Error::StartupCeiling { stage, ceiling });
                }
            }
            let last_sign = self
                .budget
                .liveness
                .as_ref()
                .and_then(EngineLiveness::last_seen)
                .map_or(last_message, |seen| seen.max(last_message));
            let silence_deadline = deadline(last_sign, self.budget.silence);
            if now >= silence_deadline {
                return Err(Error::HandshakeTimeout {
                    stage,
                    timeout: self.budget.silence,
                });
            }
            let next = self.budget.ceiling.map_or(silence_deadline, |ceiling| {
                deadline(self.started, ceiling).min(silence_deadline)
            });
            tokio::select! {
                output = &mut future => return Ok(output),
                () = sleep_until(next) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::pending;

    use tokio::time::sleep;

    use super::*;

    const SILENCE: Duration = Duration::from_millis(80);

    /// An engine that stays alive: a sign of life every quarter silence.
    async fn alive_forever(liveness: EngineLiveness) {
        loop {
            liveness.touch();
            sleep(SILENCE / 4).await;
        }
    }

    #[tokio::test]
    async fn silence_fails_the_wait_at_the_bound() {
        let clock = StartupClock::new(SILENCE);
        let started = Instant::now();
        let error = clock.wait("READY", pending::<()>()).await.unwrap_err();
        assert!(
            matches!(error, Error::HandshakeTimeout { stage: "READY", timeout } if timeout == SILENCE),
            "{error}"
        );
        assert!(started.elapsed() >= SILENCE);
    }

    #[tokio::test]
    async fn signs_of_life_keep_a_silent_wait_alive() {
        let liveness = EngineLiveness::new();
        let clock = StartupClock::new(StartupBudget {
            silence: SILENCE,
            ceiling: None,
            liveness: Some(liveness.clone()),
        });
        // Four silences of loading with a sign of life every quarter silence:
        // a plain timeout would have failed the wait at the first silence.
        let ready = async {
            for _ in 0..16 {
                liveness.touch();
                sleep(SILENCE / 4).await;
            }
            "READY"
        };
        let started = Instant::now();
        assert_eq!(clock.wait("READY", ready).await.unwrap(), "READY");
        assert!(started.elapsed() >= 4 * SILENCE);
    }

    #[tokio::test]
    async fn the_ceiling_bounds_a_wait_however_alive_the_engine_is() {
        let liveness = EngineLiveness::new();
        let ceiling = 3 * SILENCE;
        let clock = StartupClock::new(StartupBudget {
            silence: SILENCE,
            ceiling: Some(ceiling),
            liveness: Some(liveness.clone()),
        });
        let started = Instant::now();
        let error = clock
            .wait("READY", alive_forever(liveness))
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::StartupCeiling { stage: "READY", ceiling: c } if c == ceiling),
            "{error}"
        );
        assert!(started.elapsed() >= ceiling);
    }

    /// A bound too large for an `Instant` is no bound, not a panic: the
    /// handshake runs in a background task whose panic would leave the link
    /// neither connected nor failed.
    #[tokio::test]
    async fn bounds_beyond_the_clock_never_come() {
        let clock = StartupClock::new(StartupBudget {
            silence: Duration::MAX,
            ceiling: Some(Duration::MAX),
            liveness: Some(EngineLiveness::new()),
        });
        let ready = async {
            sleep(SILENCE / 4).await;
            "READY"
        };
        assert_eq!(clock.wait("READY", ready).await.unwrap(), "READY");
    }

    #[test]
    fn liveness_records_the_latest_touch() {
        let liveness = EngineLiveness::new();
        assert!(liveness.last_seen().is_none());
        liveness.touch();
        let seen = liveness.last_seen().expect("touched");
        assert!(seen <= Instant::now());
        assert_eq!(liveness.clone().last_seen(), Some(seen));
    }
}
