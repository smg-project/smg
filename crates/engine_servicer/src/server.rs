//! A gRPC server on a dedicated runtime thread with a lifecycle its (Python)
//! owner can drive: bind before serve, bounded drain on stop, last-error
//! reporting, and the tracing subscriber for a servicer-only process.

use std::{
    future::Future,
    net::SocketAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::Duration,
};

use futures::{future::Shared, FutureExt};
use tokio::{net::TcpListener, sync::oneshot};
use tracing::error;
use tracing_subscriber::EnvFilter;

use crate::ServicerError;

/// Time the caller waits for the server thread to bind its listener.
const BIND_TIMEOUT: Duration = Duration::from_secs(5);
/// Part of a `stop` timeout kept back for the thread to wind down after the
/// server future resolves.
const STOP_MARGIN: Duration = Duration::from_secs(1);
/// Least drain grace a server gets, however short the `stop` timeout.
const MIN_GRACE: Duration = Duration::from_millis(100);

/// The shutdown signal a server future awaits. It resolves to the drain
/// grace: how long the server may keep serving open connections before it
/// closes them. Cloneable so one server can both hand it to the graceful
/// shutdown and time the forced close off it.
pub(crate) type Shutdown = Shared<Pin<Box<dyn Future<Output = Duration> + Send>>>;

/// The last fatal error a running server recorded, readable after `start`
/// returned: the only post-start error channel the lifecycle owner has.
pub(crate) type SharedError = Arc<Mutex<Option<String>>>;

pub(crate) fn record_error(slot: &SharedError, message: String) {
    if let Ok(mut slot) = slot.lock() {
        *slot = Some(message);
    }
}

/// Worker threads for a servicer runtime, overridable with
/// `SMG_SERVICER_WORKER_THREADS`. The per-token work here is small (decode a
/// batch, fan it out, encode a frame), and a runtime sized to the machine
/// (tokio's default: one worker per hardware thread, 144 on a large host)
/// turns every engine output batch into a cross-thread wake-up storm: at 512
/// concurrent streams it burned about 18 cores where four threads burn under
/// one, at the same throughput.
const DEFAULT_WORKER_THREADS: usize = 4;
const WORKER_THREADS_ENV: &str = "SMG_SERVICER_WORKER_THREADS";

fn parse_worker_threads(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(DEFAULT_WORKER_THREADS)
}

/// A gRPC server on a dedicated runtime thread: `start` returns once the
/// listener is bound (so the address is known and probes get a refusal, not a
/// hang, from then on), the service future runs until `stop` or exit, and
/// failures land in [`SharedError`].
pub(crate) struct ServerThread {
    address: SocketAddr,
    shutdown: Mutex<Option<oneshot::Sender<Duration>>>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    done: Mutex<Option<Receiver<()>>>,
    running: Arc<AtomicBool>,
    last_error: SharedError,
}

impl ServerThread {
    /// Bind `bind_address` on a new runtime thread named `name`, then run
    /// `serve` with the listener and a shutdown signal. `serve` resolves when
    /// the server exits; an `Err` is recorded as the last error.
    pub(crate) fn start<F, Fut>(
        name: &str,
        bind_address: &str,
        serve: F,
    ) -> Result<Self, ServicerError>
    where
        F: FnOnce(TcpListener, Shutdown, SharedError) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), String>>,
    {
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<Duration>();
        // A dropped sender (the owner went away without `stop`) is an
        // immediate shutdown with no grace.
        let shutdown: Shutdown = shutdown_rx
            .map(|grace| grace.unwrap_or(Duration::ZERO))
            .boxed()
            .shared();
        let running = Arc::new(AtomicBool::new(false));
        let last_error: SharedError = Arc::new(Mutex::new(None));
        let thread_running = Arc::clone(&running);
        let thread_last_error = Arc::clone(&last_error);
        let thread_bind_address = bind_address.to_string();
        let thread = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(parse_worker_threads(
                        std::env::var(WORKER_THREADS_ENV).ok().as_deref(),
                    ))
                    .thread_name("smg-servicer-rt")
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = started_tx.send(Err(format!("failed to create runtime: {error}")));
                        return;
                    }
                };
                let runtime_running = Arc::clone(&thread_running);
                let runtime_last_error = Arc::clone(&thread_last_error);
                runtime.block_on(async move {
                    let listener = match TcpListener::bind(thread_bind_address.as_str()).await {
                        Ok(listener) => listener,
                        Err(error) => {
                            let _ = started_tx.send(Err(format!(
                                "failed to bind {thread_bind_address}: {error}"
                            )));
                            return;
                        }
                    };
                    let address = match listener.local_addr() {
                        Ok(address) => address,
                        Err(error) => {
                            let _ = started_tx
                                .send(Err(format!("failed to read listener address: {error}")));
                            return;
                        }
                    };
                    runtime_running.store(true, Ordering::Release);
                    if started_tx.send(Ok(address)).is_err() {
                        return;
                    }
                    if let Err(message) =
                        serve(listener, shutdown, Arc::clone(&runtime_last_error)).await
                    {
                        error!(%message, "servicer exited with an error");
                        record_error(&runtime_last_error, message);
                    }
                });
                thread_running.store(false, Ordering::Release);
                let _ = done_tx.send(());
            })
            .map_err(|error| {
                ServicerError::Startup(format!("failed to start server thread: {error}"))
            })?;

        let address = started_rx
            .recv_timeout(BIND_TIMEOUT)
            .map_err(|error| match error {
                RecvTimeoutError::Timeout => ServicerError::Startup(format!(
                    "servicer did not bind {bind_address} within {BIND_TIMEOUT:?}"
                )),
                RecvTimeoutError::Disconnected => {
                    ServicerError::Startup("servicer exited during startup".to_string())
                }
            })?
            .map_err(ServicerError::Startup)?;
        Ok(Self {
            address,
            shutdown: Mutex::new(Some(shutdown_tx)),
            thread: Mutex::new(Some(thread)),
            done: Mutex::new(Some(done_rx)),
            running,
            last_error,
        })
    }

    pub(crate) fn address(&self) -> SocketAddr {
        self.address
    }

    pub(crate) fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub(crate) fn last_error(&self) -> Result<Option<String>, ServicerError> {
        Ok(lock(&self.last_error)?.clone())
    }

    /// Signal shutdown and wait up to `timeout` for the server thread. The
    /// server gets most of `timeout` as drain grace for open connections and
    /// then closes them, so a connected client cannot hold the process. On
    /// timeout the server keeps stopping and a later call waits again.
    pub(crate) fn stop(&self, timeout: Duration) -> Result<(), ServicerError> {
        if let Some(shutdown) = lock(&self.shutdown)?.take() {
            let grace = timeout.saturating_sub(STOP_MARGIN).max(MIN_GRACE);
            let _ = shutdown.send(grace);
        }
        let receiver = lock(&self.done)?.take();
        if let Some(receiver) = receiver {
            match receiver.recv_timeout(timeout) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                Err(RecvTimeoutError::Timeout) => {
                    *lock(&self.done)? = Some(receiver);
                    return Err(ServicerError::StopTimeout);
                }
            }
        }
        if let Some(thread) = lock(&self.thread)?.take() {
            thread.join().map_err(|_| ServicerError::ThreadPanicked)?;
        }
        Ok(())
    }
}

impl Drop for ServerThread {
    fn drop(&mut self) {
        if let Ok(shutdown) = self.shutdown.get_mut() {
            if let Some(shutdown) = shutdown.take() {
                let _ = shutdown.send(Duration::ZERO);
            }
        }
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, ServicerError> {
    mutex.lock().map_err(|_| ServicerError::Poisoned)
}

static TRACING: OnceLock<()> = OnceLock::new();

/// Install the Rust tracing subscriber for a process that only hosts a
/// servicer (the Python launcher). `level` overrides `RUST_LOG`; a second
/// call is a no-op.
pub fn init_tracing(level: Option<&str>) -> Result<(), ServicerError> {
    let filter = match level {
        Some(level) => EnvFilter::try_new(level)
            .map_err(|error| ServicerError::InvalidConfig(format!("invalid log level: {error}")))?,
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };
    TRACING.get_or_init(|| {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
    Ok(())
}

#[cfg(test)]
mod worker_threads_tests {
    use super::{parse_worker_threads, DEFAULT_WORKER_THREADS};

    #[test]
    fn worker_threads_default_and_override() {
        assert_eq!(parse_worker_threads(None), DEFAULT_WORKER_THREADS);
        assert_eq!(parse_worker_threads(Some(" 8 ")), 8);
        // Zero and garbage fall back to the default rather than panicking.
        assert_eq!(parse_worker_threads(Some("0")), DEFAULT_WORKER_THREADS);
        assert_eq!(parse_worker_threads(Some("many")), DEFAULT_WORKER_THREADS);
    }
}
