//! Helpers the servicer tests share: the configuration they start from and
//! the deadline every wait on the servicer runs under.

use std::{future::Future, panic::Location, path::Path, time::Duration};

use engine_zmq_client::mock_engine::MOCK_DEADLINE;
use tokio::time::timeout;

use crate::ServicerConfig;

/// The servicer configuration every servicer test starts from: an ephemeral
/// port, the engine's data-plane sockets under the test's own directory,
/// one engine, no tokenizer directory, a ten-second silence bound on the
/// start and no ceiling.
pub(crate) fn servicer_config(dir: &Path, handshake: &str) -> ServicerConfig {
    ServicerConfig {
        engine_startup_timeout: Duration::from_secs(10),
        engine_startup_ceiling: None,
        ..ServicerConfig::new(
            "127.0.0.1:0",
            format!("ipc://{}", dir.join("engine").display()),
            handshake,
        )
    }
}

/// A wait on the servicer (a response stream's next message, a watch's next
/// event) under the mock engine's deadline: a servicer that stalls fails the
/// test at the waiting line after thirty seconds instead of hanging the gate.
pub(crate) trait Bounded: Future + Sized {
    #[track_caller]
    fn bounded(self) -> impl Future<Output = Self::Output> {
        let caller = Location::caller();
        async move {
            match timeout(MOCK_DEADLINE, self).await {
                Ok(output) => output,
                Err(_) => panic!("{caller}: nothing from the servicer within {MOCK_DEADLINE:?}"),
            }
        }
    }
}

impl<F: Future> Bounded for F {}
