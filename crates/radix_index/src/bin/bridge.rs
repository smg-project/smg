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

use radix_index::{
    bridge,
    cli::{parse_flag, parse_list, validate_flags},
    proto,
};
use tokio::sync::mpsc;

const KNOWN_FLAGS: &[&str] = &["--workers", "--index", "--model", "--block-size"];

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
    bridge::run_publisher(rx, index, ledger).await;
    std::process::ExitCode::SUCCESS
}
