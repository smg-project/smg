//! The deadline every servicer test's wait on the servicer runs under.

use std::{future::Future, panic::Location};

use engine_zmq_client::mock_engine::MOCK_DEADLINE;
use tokio::time::timeout;

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
