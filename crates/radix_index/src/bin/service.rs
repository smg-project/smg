//! The radix index service binary.
//!
//! Usage:
//!   radix-index-service --port 40000 [--bind 0.0.0.0]
//!     [--peers http://127.0.0.1:40001,..]
//!     [--bootstrap-from http://127.0.0.1:40001]
//!     [--metrics-port 40100]
//!     [--inferred-ttl-secs 180] [--default-capacity-blocks N]
//!     [--sweep-interval-secs 5]
//!     [--apply-delay-stored-ms 0] [--apply-delay-removed-ms 0]
//!
//! Stops gracefully on SIGTERM or ctrl-c: in-flight streams finish and
//! clients reconnect to a sibling replica. The metrics port serves
//! `/metrics`, `/healthz`, and `/readyz` (503 until the bootstrap pull
//! completes — point the k8s readiness probe there).

use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use radix_index::{
    server::{self, ServiceStats},
    Engine, EngineConfig,
};

const KNOWN_FLAGS: &[&str] = &[
    "--bind",
    "--port",
    "--metrics-port",
    "--peers",
    "--bootstrap-from",
    "--inferred-ttl-secs",
    "--event-ttl-secs",
    "--default-capacity-blocks",
    "--sweep-interval-secs",
    "--apply-delay-stored-ms",
    "--apply-delay-removed-ms",
];

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    let i = args.iter().position(|a| a == flag)?;
    let value = args
        .get(i + 1)
        .unwrap_or_else(|| panic!("flag {flag} is missing its value"));
    Some(
        value
            .parse()
            .unwrap_or_else(|_| panic!("flag {flag} has an unparsable value: {value:?}")),
    )
}

/// Unknown flags and unparsable values are startup errors, never
/// silent fallbacks to defaults (audit finding: a typo'd
/// --bootstrap-from meant silently-cold restarts forever).
fn validate_flags(args: &[String]) {
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") {
            assert!(
                KNOWN_FLAGS.contains(&arg.as_str()),
                "unknown flag {arg}; known flags: {KNOWN_FLAGS:?}"
            );
            i += 2;
        } else {
            panic!("unexpected argument {arg}");
        }
    }
}

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
    tracing::info!("shutdown signal received; stopping gracefully");
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().collect();
    validate_flags(&args);
    let bind: String = parse_flag(&args, "--bind").unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = parse_flag(&args, "--port").unwrap_or(40000);
    let metrics_port: u16 = parse_flag(&args, "--metrics-port").unwrap_or(0);
    let peers: Vec<String> = parse_flag::<String>(&args, "--peers")
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let bootstrap: Option<String> = parse_flag(&args, "--bootstrap-from");
    let cfg = EngineConfig {
        inferred_ttl: Duration::from_secs(parse_flag(&args, "--inferred-ttl-secs").unwrap_or(180)),
        // Liveness backstop for event-fed holders whose gateway-published
        // departure signal was lost; 0 disables. Keep well above the
        // event feed's idle cadence.
        event_ttl: Duration::from_secs(parse_flag(&args, "--event-ttl-secs").unwrap_or(1800)),
        default_capacity_blocks: parse_flag(&args, "--default-capacity-blocks").unwrap_or(u64::MAX),
    };
    let sweep_secs: u64 = parse_flag(&args, "--sweep-interval-secs").unwrap_or(5);
    assert!(sweep_secs > 0, "--sweep-interval-secs must be > 0 (a zero interval panics tokio's timer inside a detached task and silently disables TTL/retire)");
    let sweep = Duration::from_secs(sweep_secs);
    let delay_stored =
        Duration::from_millis(parse_flag(&args, "--apply-delay-stored-ms").unwrap_or(0));
    let delay_removed =
        Duration::from_millis(parse_flag(&args, "--apply-delay-removed-ms").unwrap_or(0));

    tracing::info!(
        inferred_ttl_secs = cfg.inferred_ttl.as_secs(),
        event_ttl_secs = cfg.event_ttl.as_secs(),
        default_capacity_blocks = cfg.default_capacity_blocks,
        "engine config"
    );
    let engine = Arc::new(Engine::new(cfg));
    let stats = Arc::new(ServiceStats::default());

    // Admin plane first so /readyz answers (503) during bootstrap.
    if metrics_port != 0 {
        let admin_addr = format!("{bind}:{metrics_port}")
            .parse()
            .expect("admin addr");
        let admin_engine = Arc::clone(&engine);
        let admin_stats = Arc::clone(&stats);
        #[expect(
            clippy::disallowed_methods,
            reason = "process-lifetime admin listener; dies with the process"
        )]
        tokio::spawn(async move {
            if let Err(error) = server::serve_admin(admin_engine, admin_stats, admin_addr).await {
                tracing::error!(%error, "admin listener exited");
            }
        });
    }

    if let Some(peer) = bootstrap {
        match server::bootstrap_from(&engine, &peer).await {
            Ok(applied) => tracing::info!(peer, applied, "bootstrap pull complete"),
            Err(error) => tracing::warn!(peer, %error, "bootstrap pull failed; starting cold"),
        }
    }
    stats.ready.store(true, Ordering::Relaxed);

    let addr = format!("{bind}:{port}").parse().expect("bind addr");
    tracing::info!(
        %addr,
        peers = peers.len(),
        metrics_port,
        sweep_secs,
        "radix index serving (effective config logged at engine construction)"
    );
    if let Err(error) = server::serve_until(
        engine,
        addr,
        peers,
        sweep,
        delay_stored,
        delay_removed,
        stats,
        shutdown_signal(),
    )
    .await
    {
        tracing::error!(%error, "server exited");
    }
}
