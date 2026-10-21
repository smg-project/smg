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

/// `count` publishers bound on consecutive ports from the returned base, as
/// a data-parallel engine lays its ranks' publishers out (rank `r` on the
/// base port plus `r`).
pub(crate) async fn consecutive_publishers(count: u16) -> (u16, Vec<zeromq::PubSocket>) {
    use zeromq::prelude::*;

    for _ in 0..20 {
        let Some(base) = portpicker::pick_unused_port() else {
            continue;
        };
        let Some(last) = base.checked_add(count.saturating_sub(1)) else {
            continue;
        };
        if (base..=last).any(|port| !portpicker::is_free_tcp(port)) {
            continue;
        }
        let mut publishers = Vec::with_capacity(usize::from(count));
        for port in base..=last {
            let mut publisher = zeromq::PubSocket::new();
            if publisher
                .bind(&format!("tcp://127.0.0.1:{port}"))
                .await
                .is_err()
            {
                break;
            }
            publishers.push(publisher);
        }
        if publishers.len() == usize::from(count) {
            return (base, publishers);
        }
    }
    panic!("no run of {count} consecutive free ports");
}
