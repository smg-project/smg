//! mock-worker: spin up many mock HTTP and/or gRPC inference workers in one
//! process to scale-test the SMG gateway's routing and runtime behavior.
//!
//! Workers are protocol-accurate stand-ins for vLLM/SGLang engines: HTTP
//! workers answer the probe + chat/generate surface; gRPC workers implement the
//! TokenSpeed scheduler service (the gateway tokenizes and speaks token ids).
//! All responses are canned — there is no real model.

use std::{process::ExitCode, sync::Arc};

use mock_worker::{config::Config, grpc, http, zmq};

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cfg = match Config::from_args() {
        Ok(cfg) => Arc::new(cfg),
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    tracing::info!(
        "mock-worker: {} http from :{}, {} grpc from :{}, model={}",
        cfg.http_count,
        cfg.http_base_port,
        cfg.grpc_count,
        cfg.grpc_base_port,
        cfg.model_id,
    );

    // Spawn each worker as its own task so they run across the whole runtime
    // (join_all would poll them all on a single task) and an unexpected exit is
    // surfaced instead of silently masked.
    let mut workers = tokio::task::JoinSet::new();
    for i in 0..cfg.http_count {
        let Some(port) = cfg.http_base_port.checked_add(i) else {
            tracing::warn!("http port range overflowed u16 at offset {i}; stopping early");
            break;
        };
        workers.spawn(http::serve(cfg.clone(), cfg.host.clone(), port));
    }
    for i in 0..cfg.grpc_count {
        let Some(port) = cfg.grpc_base_port.checked_add(i) else {
            tracing::warn!("grpc port range overflowed u16 at offset {i}; stopping early");
            break;
        };
        workers.spawn(grpc::serve(cfg.clone(), cfg.host.clone(), port));
    }
    if let Some(handshake) = cfg.zmq_handshake.clone() {
        for i in 0..cfg.zmq_count {
            let engine_index = cfg.zmq_start_index + i as u32;
            workers.spawn(zmq::serve(cfg.clone(), handshake.clone(), engine_index));
        }
    }

    tracing::info!("started {} mock workers; ctrl-c to stop", workers.len());

    // A single worker exiting (e.g. a port already in use) is logged but must
    // not tear down the whole fleet — the remaining workers keep serving. Only
    // ctrl-c stops the process. When the set is empty `join_next` yields None,
    // so that select arm is disabled and we wait on the signal.
    loop {
        tokio::select! {
            Some(joined) = workers.join_next() => {
                match joined {
                    Ok(()) => tracing::warn!(
                        "a mock worker stopped (likely a port bind failure); others keep serving"
                    ),
                    Err(e) => tracing::error!("mock worker task panicked: {e}"),
                }
            }
            result = tokio::signal::ctrl_c() => {
                if let Err(e) = result {
                    tracing::error!("failed to listen for ctrl-c: {e}");
                }
                tracing::info!("shutting down");
                break;
            }
        }
    }
    ExitCode::SUCCESS
}
