//! What the test binaries that start mock engines share, included by path
//! beside `common/net.rs` (the including binary names that module `net`): a
//! mock gRPC engine on an ephemeral port, a bounded wait and the process's
//! CPU time.

use std::{
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use mock_worker::config::Config;
use tokio::{net::TcpListener, sync::oneshot, task::JoinSet};

use crate::net::loopback;

/// The configuration of the mock gRPC engine at `port` with `ranks`
/// data-parallel ranks.
pub fn engine_config(port: u16, ranks: u16) -> Config {
    Config {
        grpc_base_port: port,
        grpc_count: 1,
        grpc_dp_size: ranks,
        grpc_wire: mock_worker::config::SchedulerWire::Vllm,
        loads_like: mock_worker::engine::LoadsLike::Vllm,
        realistic: true,
        ..Config::default()
    }
}

/// A mock gRPC engine of the vLLM wire with `ranks` data-parallel ranks,
/// served until the returned sender drops. Returns its address.
pub async fn mock_engine(ranks: u16) -> Result<(SocketAddr, oneshot::Sender<()>), std::io::Error> {
    let listener = TcpListener::bind(loopback()).await?;
    let addr = listener.local_addr()?;
    let cfg = Arc::new(engine_config(addr.port(), ranks));
    let (stop, stopped) = oneshot::channel();
    let mut served = JoinSet::new();
    served.spawn(mock_worker::grpc::serve_with_listener_until(
        cfg,
        listener,
        async move {
            let _ = stopped.await;
        },
    ));
    served.detach_all();
    Ok((addr, stop))
}

/// Waits until `check` passes or `within` runs out.
pub async fn eventually(within: Duration, mut check: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if check() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    check()
}

/// CPU seconds of this process (user + system), from `/proc/self/stat`.
pub fn cpu_seconds() -> f64 {
    let ticks = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            let after = stat.rsplit_once(')')?.1;
            let fields: Vec<&str> = after.split_whitespace().collect();
            let utime: f64 = fields.get(11)?.parse().ok()?;
            let stime: f64 = fields.get(12)?.parse().ok()?;
            Some(utime + stime)
        })
        .unwrap_or(0.0);
    ticks / 100.0
}
