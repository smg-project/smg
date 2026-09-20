//! Event bridge binary: see `radix_index::bridge` for the semantics.
//!
//! Usage (`--flag value` and `--flag=value` are both accepted):
//!   radix-index-bridge --workers grpc://127.0.0.1:9000,... \
//!     --index http://127.0.0.1:40000 --model mock-model --block-size 128
//!
//! Flags are validated exactly like the service binary's: an unknown
//! flag or an unparsable value is a startup error. `--block-size` in
//! particular decides the keyspace key — a typo that fell back to the
//! default would feed every event into a keyspace no gateway queries.
//!
//! Exit status: 0 on a signalled stop, 2 on a bad invocation, and
//! failure when every worker subscription has ended — at that point the
//! bridge can never publish again and a supervisor has to restart it.

use radix_index::{
    bridge,
    cli::{parse_flag, parse_list, validate_flags},
    proto,
};
use tokio::sync::mpsc;

const KNOWN_FLAGS: &[&str] = &["--workers", "--index", "--model", "--block-size"];

/// Resolves on SIGTERM or ctrl-c, like the service binary's. Without it
/// the only way out of `main` is worker exhaustion, which must NOT be
/// reported as success — see the exit-status split below.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
    }
    tracing::info!("shutdown signal received; stopping");
}

#[expect(
    clippy::disallowed_methods,
    reason = "worker subscription tasks live for the process lifetime"
)]
#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().collect();
    validate_flags(&args, KNOWN_FLAGS, &[]);
    let workers = parse_list(&args, "--workers");
    let index: String =
        parse_flag(&args, "--index").unwrap_or_else(|| "http://127.0.0.1:40000".to_string());
    let model: String = parse_flag(&args, "--model").unwrap_or_else(|| "mock-model".to_string());
    // Shared default with the gateway's --kv-indexer-block-size: the
    // keyspace key includes block size, so divergent defaults would
    // silently split the fleet into two keyspaces.
    let block_size: u32 =
        parse_flag(&args, "--block-size").unwrap_or(radix_index::DEFAULT_BLOCK_SIZE);
    assert!(block_size > 0, "--block-size must be > 0");
    if workers.is_empty() {
        eprintln!("--workers is required");
        return std::process::ExitCode::from(2);
    }

    let (tx, rx) = mpsc::channel::<proto::Update>(65_536);
    let ledger = bridge::EpochLedger::default();
    for worker in &workers {
        tokio::spawn(bridge::worker_loop(
            worker.clone(),
            model.clone(),
            block_size,
            tx.clone(),
            ledger.clone(),
        ));
    }
    drop(tx);
    tracing::info!(workers = workers.len(), %index, block_size, "bridge running");
    // Two ways out, and they are NOT the same exit status. `run_publisher`
    // returns only when every worker loop has ended and dropped its
    // sender (each one answered Unimplemented, say): the bridge is still
    // up but will never publish again, so the index quietly freezes at
    // whatever it last held while the fleet keeps routing on it. Exiting
    // 0 there reads to a supervisor as a finished job and nothing
    // restarts. A signalled stop is the only success.
    tokio::select! {
        _ = bridge::run_publisher(rx, index, ledger) => {
            tracing::error!("every worker subscription ended; bridge has nothing left to publish");
            std::process::ExitCode::FAILURE
        }
        _ = shutdown_signal() => std::process::ExitCode::SUCCESS,
    }
}
