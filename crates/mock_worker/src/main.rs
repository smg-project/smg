//! mock-worker: spin up many mock HTTP and/or gRPC inference workers in one
//! process to scale-test the SMG gateway's routing and runtime behavior.
//!
//! Workers are protocol-accurate stand-ins for vLLM/SGLang engines: HTTP
//! workers answer the probe + chat/generate surface; gRPC workers serve one
//! engine's scheduler service (`--grpc-wire`; the gateway tokenizes and speaks
//! token ids).
//! All responses are canned — there is no real model.

use std::{io::IsTerminal, pin::pin, process::ExitCode, sync::Arc};

use mock_worker::{admin, config::Config, drain, grpc, http, replay::Capture, zmq};
use tokio::sync::watch;

#[tokio::main]
async fn main() -> ExitCode {
    // The process-level TLS crypto provider, before anything builds a TLS
    // configuration or an HTTP client.
    mock_worker::tls::install_crypto_provider();
    // Coloured on a terminal only: a log file or a pipe holds plain lines, so
    // a launcher that reads the level words of the last lines finds them.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(std::io::stdout().is_terminal())
        .init();

    // Help goes to stdout and succeeds, so `mock-worker --help` also shows
    // that the binary runs.
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "-h" || arg == "--help")
    {
        println!("{}", mock_worker::config::usage());
        return ExitCode::SUCCESS;
    }

    let cfg = match Config::from_args() {
        Ok(cfg) => Arc::new(cfg),
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    // A capture file that cannot be opened is a config error. Each gRPC worker
    // opens it again, and one that cannot would stop while the process kept
    // running, so check it once before any worker starts.
    if let Some(path) = &cfg.replay.capture {
        if let Err(e) = Capture::open(path) {
            eprintln!(
                "mock-worker: cannot open capture file {}: {e}",
                path.display()
            );
            return ExitCode::from(2);
        }
    }

    // A script file that cannot be read or holds a bad line is a config
    // error too: checked once here, loaded for every worker.
    if let Some(path) = &cfg.replay.script {
        match mock_worker::script::load(path) {
            Ok(answers) => tracing::info!(
                "mock-worker: {answers} scripted answers from {}",
                path.display()
            ),
            Err(e) => {
                eprintln!("mock-worker: cannot load script: {e}");
                return ExitCode::from(2);
            }
        }
    }

    tracing::info!(
        "mock-worker: {} http from :{}, {} grpc from :{}, model={}",
        cfg.http_count,
        cfg.http_base_port,
        cfg.grpc_count,
        cfg.grpc_base_port,
        cfg.model_id,
    );

    // The termination signals are listened for before any port is bound: as
    // PID 1 of a container the process would otherwise drop a SIGTERM that
    // arrives once it is reachable, and live on until the SIGKILL at the end
    // of the grace period.
    let mut signals = Signals::install();
    // What tells every worker to stop accepting work.
    let (stop, stopped) = watch::channel(false);

    // Spawn each worker as its own task so they run across the whole runtime
    // (join_all would poll them all on a single task) and an unexpected exit is
    // surfaced instead of silently masked. The admin API is a set of its own:
    // it serves the workers' operators, not the gateway, so it is not counted
    // as a worker and does not keep the process up once none is left.
    let mut workers = tokio::task::JoinSet::new();
    let mut admin = tokio::task::JoinSet::new();
    for i in 0..cfg.http_count {
        let Some(port) = cfg.http_base_port.checked_add(i) else {
            tracing::warn!("http port range overflowed u16 at offset {i}; stopping early");
            break;
        };
        workers.spawn(http::serve_until(
            cfg.clone(),
            cfg.host.clone(),
            port,
            stop_requested(stopped.clone()),
        ));
    }
    for i in 0..cfg.grpc_count {
        let Some(port) = cfg.grpc_base_port.checked_add(i) else {
            tracing::warn!("grpc port range overflowed u16 at offset {i}; stopping early");
            break;
        };
        workers.spawn(grpc::serve_until(
            cfg.clone(),
            cfg.host.clone(),
            port,
            stop_requested(stopped.clone()),
        ));
    }
    if let Some(port) = cfg.admin_port {
        admin.spawn(admin::serve_until(
            cfg.clone(),
            cfg.host.clone(),
            port,
            stop_requested(stopped.clone()),
        ));
    }
    if let Some(handshake) = cfg.zmq_handshake.clone() {
        for i in 0..cfg.zmq_count {
            let engine_index = cfg.zmq_start_index + i as u32;
            workers.spawn(zmq::serve_until(
                cfg.clone(),
                handshake.clone(),
                engine_index,
                stop_requested(stopped.clone()),
            ));
        }
    }

    tracing::info!(
        "started {} mock workers; SIGTERM or ctrl-c to stop",
        workers.len()
    );

    // A worker exiting on its own (a port already in use, a ZMQ rank whose
    // frontend never answered) is logged and must not tear down the fleet:
    // the remaining workers keep serving. The last worker's exit ends the
    // process with a failure status: a fleet that serves nothing is a failed
    // start, and a launcher that waits for it to answer would otherwise wait
    // until its own bound. That line is a warning: the worker's own last
    // line said why it stopped, and a launcher that quotes a log's last
    // error line should quote that one. Otherwise only a termination signal
    // stops the process. When a set is empty `join_next` yields None, so that
    // select arm is disabled and we wait on the signal.
    let signal = {
        let mut next = pin!(signals.next());
        loop {
            tokio::select! {
                Some(joined) = workers.join_next() => {
                    if let Err(e) = joined {
                        tracing::error!("mock worker task panicked: {e}");
                    }
                    if workers.is_empty() {
                        tracing::warn!(
                            "the last mock worker stopped; nothing left to serve, exiting with a failure status"
                        );
                        return ExitCode::FAILURE;
                    }
                    tracing::warn!(
                        "a mock worker stopped (likely a port bind failure); {} still serving",
                        workers.len()
                    );
                }
                Some(joined) = admin.join_next() => {
                    match joined {
                        Ok(()) => tracing::warn!(
                            "the admin API stopped (likely a port bind failure); the workers keep serving"
                        ),
                        Err(e) => tracing::error!("admin API task panicked: {e}"),
                    }
                }
                signal = &mut next => break signal,
            }
        }
    };

    // The drain, as a serving engine does it under SIGTERM: from here on no
    // worker accepts a connection or admits a request; the requests in flight
    // finish, up to the deadline; then the process exits cleanly either way,
    // so the grace period of a pod is an upper bound rather than its lifetime.
    // A second signal ends the wait at once.
    let deadline = cfg.drain_timeout;
    tracing::info!(
        "{signal}: draining {} in-flight request(s), up to {:.1} s",
        drain::in_flight(),
        deadline.as_secs_f64()
    );
    let _ = stop.send(true);
    let drain = async {
        while let Some(joined) = workers.join_next().await {
            if let Err(e) = joined {
                tracing::error!("mock worker task panicked: {e}");
            }
        }
        while let Some(joined) = admin.join_next().await {
            if let Err(e) = joined {
                tracing::error!("admin API task panicked: {e}");
            }
        }
    };
    tokio::select! {
        drained = tokio::time::timeout(deadline, drain) => match drained {
            Ok(()) => tracing::info!("drained; exiting"),
            Err(_) => tracing::warn!(
                "drain deadline of {:.1} s reached with {} request(s) still in flight; exiting",
                deadline.as_secs_f64(),
                drain::in_flight()
            ),
        },
        again = signals.next() => tracing::warn!(
            "{again} during the drain: exiting now with {} request(s) in flight",
            drain::in_flight()
        ),
    }
    ExitCode::SUCCESS
}

/// Resolves once the process asks its workers to stop (or is gone).
async fn stop_requested(mut stopped: watch::Receiver<bool>) {
    let _ = stopped.wait_for(|stop| *stop).await;
}

/// The termination signals: SIGTERM (what a container runtime sends when a
/// pod is deleted) and SIGINT (ctrl-c). The listeners are installed when the
/// value is created, not when it is first polled; `next` names each signal
/// as it arrives, so a second one can cut the drain short.
#[cfg(unix)]
struct Signals {
    terminate: Option<tokio::signal::unix::Signal>,
    interrupt: Option<tokio::signal::unix::Signal>,
}

#[cfg(unix)]
impl Signals {
    fn install() -> Self {
        use tokio::signal::unix::{signal, SignalKind};

        // A listener that could not be installed is logged and never fires;
        // the other one still stops the process.
        let listen = |kind: SignalKind, name: &str| match signal(kind) {
            Ok(signal) => Some(signal),
            Err(e) => {
                tracing::error!("failed to listen for {name}: {e}");
                None
            }
        };
        Self {
            terminate: listen(SignalKind::terminate(), "SIGTERM"),
            interrupt: listen(SignalKind::interrupt(), "SIGINT"),
        }
    }

    /// The next termination signal to arrive, by name.
    async fn next(&mut self) -> &'static str {
        let Self {
            terminate,
            interrupt,
        } = self;
        let terminate = async {
            match terminate {
                Some(terminate) => {
                    terminate.recv().await;
                }
                None => std::future::pending::<()>().await,
            }
        };
        let interrupt = async {
            match interrupt {
                Some(interrupt) => {
                    interrupt.recv().await;
                }
                None => std::future::pending::<()>().await,
            }
        };
        tokio::select! {
            () = terminate => "SIGTERM",
            () = interrupt => "SIGINT",
        }
    }
}

/// Ctrl-c, the one termination signal every platform has.
#[cfg(not(unix))]
struct Signals;

#[cfg(not(unix))]
impl Signals {
    fn install() -> Self {
        Self
    }

    async fn next(&mut self) -> &'static str {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("failed to listen for ctrl-c: {e}");
            std::future::pending::<()>().await;
        }
        "SIGINT"
    }
}
