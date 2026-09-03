//! Event bridge binary: see `radix_index::bridge` for the semantics.
//!
//! Usage:
//!   radix-index-bridge --workers grpc://127.0.0.1:9000,... \
//!     --index http://127.0.0.1:40000 --model mock-model --block-size 128

use radix_index::{bridge, proto};
use tokio::sync::mpsc;

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

#[expect(
    clippy::disallowed_methods,
    reason = "worker subscription tasks live for the process lifetime"
)]
#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().collect();
    let workers: Vec<String> = parse_flag::<String>(&args, "--workers")
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let index: String =
        parse_flag(&args, "--index").unwrap_or_else(|| "http://127.0.0.1:40000".to_string());
    let model: String = parse_flag(&args, "--model").unwrap_or_else(|| "mock-model".to_string());
    // Shared default with the gateway's --kv-indexer-block-size: the
    // keyspace key includes block size, so divergent defaults would
    // silently split the fleet into two keyspaces.
    let block_size: u32 =
        parse_flag(&args, "--block-size").unwrap_or(radix_index::DEFAULT_BLOCK_SIZE);
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
    tracing::info!(workers = workers.len(), %index, "bridge running");
    bridge::run_publisher(rx, index, ledger).await;
    std::process::ExitCode::SUCCESS
}
