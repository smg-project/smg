//! Token dump: an operator-controlled record of every engine call the gRPC
//! router makes, for debugging, RL data collection and CI replay.
//!
//! `--token-dump-dir` enables it. A session records into one new file,
//! `<dir>/token-dump-<session>.jsonl`: from boot to shutdown under
//! `--token-dump-on-start`, or from `POST /start_token_dump` until
//! `POST /stop_token_dump` or its expiry. One session runs at a time.
//!
//! # Format (v1)
//!
//! JSON Lines, one event per line in the order the gateway saw them; lines of
//! concurrent calls interleave and `call` groups them. Every line has
//! `"v": 1` and a `kind`:
//!
//! - `session`: first line: the session id, start time, model filter, expiry
//!   and size cap.
//! - `request`: an engine call about to be dispatched: model, worker,
//!   runtime, transport (`grpc`/`zmq`), leg (`single`/`prefill`/`decode`),
//!   the engine request id, the client's request id, the full protobuf name
//!   of the message (`type`), its `input_ids` and its base64 protobuf
//!   encoding (`msg`). On `zmq`, `msg` is the request the gateway built
//!   before translating it for the engine.
//! - `response`: each message the engine stream yielded, as the engine sent
//!   it (an error `Complete` included): `seq` from 0, `t_ms` since the call's
//!   request line, `part` (`chunk`/`complete`/`empty`), and the `index`,
//!   `token_ids` and `finish_reason` that message carries, with no
//!   accumulation across messages.
//! - `end`: `status` is `ok`, `error`, `cancelled` (the gateway dropped the
//!   call before it ended) or `start_failed`, with the `error` code and
//!   message for the failures; `responses` counts the call's response lines,
//!   so a reader can tell when some were dropped.
//! - `session_end`: last line: the reason (`stopped`/`expired`/`shutdown`)
//!   and the session's counters. A file without it was cut short and its
//!   last line may be partial.
//!
//! Recording never blocks a request: a line the writer's queue (64 MiB)
//! cannot take, or that would pass the file's size cap, is dropped and
//! counted in `smg_token_dump_lines_dropped_total`. Dump files hold prompts:
//! the directory is created owner-only and each file is mode 0600.

pub mod line;
mod session;

use std::{
    io,
    path::PathBuf,
    sync::{Arc, Mutex, PoisonError, Weak},
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use metrics::{describe_counter, describe_gauge, gauge};
use serde::{Deserialize, Serialize};
use smg_external_router::error;
use tracing::{info, warn};

use self::{
    line::EndReason,
    session::{Limits, QUEUE_BUDGET_BYTES},
};
pub use self::{
    line::{
        CallMeta, Dropped, EndStatus, Leg, Part, RequestEvent, ResponseEvent, Totals, Transport,
    },
    session::{CallRecorder, Session},
};
use crate::config::RouterConfig;

/// Run time of a session `POST /start_token_dump` starts without one.
pub const DEFAULT_DURATION_SECS: u64 = 600;
/// Longest run time `POST /start_token_dump` accepts.
pub const MAX_DURATION_SECS: u64 = 86_400;
/// How long shutdown waits for calls in flight to finish their lines.
const SHUTDOWN_FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// The gateway's token dump: where sessions write, and the one running now.
pub struct TokenDump {
    dir: PathBuf,
    max_bytes: u64,
    current: ArcSwapOption<Session>,
    /// Serializes start, stop, expiry and shutdown; the request path only
    /// loads `current`.
    control: Mutex<()>,
}

/// `POST /start_token_dump` body. There is no path: files go only to the
/// operator's `--token-dump-dir`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartRequest {
    /// Run time in seconds, 1..=86400; 600 when absent.
    #[serde(default)]
    pub duration_secs: Option<u64>,
    /// Models to record (canonical ids); empty records every model.
    #[serde(default)]
    pub models: Vec<String>,
}

/// `POST /start_token_dump` answer.
#[derive(Debug, Serialize)]
pub struct Started {
    pub session: String,
    pub file: PathBuf,
    pub expires_at: Option<String>,
    pub models: Vec<String>,
}

/// `POST /stop_token_dump` answer: the counters as they were at the stop.
/// Calls still in flight finish their lines afterwards; the file's
/// `session_end` line has the final counters.
#[derive(Debug, Serialize)]
pub struct Stopped {
    pub session: String,
    pub file: PathBuf,
    #[serde(flatten)]
    pub totals: Totals,
}

#[derive(Debug)]
pub enum StartError {
    AlreadyRunning,
    BadDuration(u64),
    Open(io::Error),
    Task(String),
}

#[derive(Debug)]
pub enum StopError {
    NotRunning,
}

impl TokenDump {
    pub fn new(dir: PathBuf, max_bytes: u64) -> Self {
        Self {
            dir,
            max_bytes,
            current: ArcSwapOption::empty(),
            control: Mutex::new(()),
        }
    }

    /// The dump `--token-dump-dir` asks for, with its boot session when
    /// `--token-dump-on-start` is set; `None` without a directory. Fails when
    /// the boot session's file cannot be made, so a gateway asked to record
    /// does not run without recording.
    pub fn from_config(config: &RouterConfig) -> Result<Option<Arc<Self>>, String> {
        let Some(dir) = config.token_dump_dir.as_deref() else {
            return Ok(None);
        };
        let dump = Arc::new(Self::new(
            PathBuf::from(dir),
            config.token_dump_max_mb.saturating_mul(1 << 20),
        ));
        if config.token_dump_on_start {
            let session = dump.open_session(Vec::new(), None).map_err(|error| {
                format!("--token-dump-on-start cannot open a dump file in {dir}: {error}")
            })?;
            info!(
                session = %session.id(),
                file = %session.path().display(),
                "token dump started at boot"
            );
            dump.install(session);
        }
        Ok(Some(dump))
    }

    /// The session to record a call to `model` into, if one is running for
    /// it and has not expired.
    pub fn session_for(&self, model: &str) -> Option<Arc<Session>> {
        let current = self.current.load();
        let session = Option::as_ref(&*current)?;
        (session.records_model(model) && !session.expired(Instant::now()))
            .then(|| Arc::clone(session))
    }

    /// Start a session with a time limit. Blocking: it creates the file.
    pub fn start(self: &Arc<Self>, request: StartRequest) -> Result<Started, StartError> {
        let duration_secs = request.duration_secs.unwrap_or(DEFAULT_DURATION_SECS);
        if !(1..=MAX_DURATION_SECS).contains(&duration_secs) {
            return Err(StartError::BadDuration(duration_secs));
        }
        let duration = Duration::from_secs(duration_secs);
        let _control = self.control.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(current) = self.current.load_full() {
            if !current.expired(Instant::now()) {
                return Err(StartError::AlreadyRunning);
            }
            // Its timer has not fired yet; it is over all the same.
            self.retire(EndReason::Expired);
        }
        let session = self
            .open_session(request.models, Some(duration))
            .map_err(StartError::Open)?;
        let started = Started {
            session: session.id().to_string(),
            file: session.path().to_path_buf(),
            expires_at: session.expires_at_text().map(str::to_string),
            models: session.models().to_vec(),
        };
        self.install(session);
        info!(
            session = %started.session,
            file = %started.file.display(),
            models = ?started.models,
            duration_secs,
            "token dump started"
        );
        arm_expiry(Arc::downgrade(self), started.session.clone(), duration);
        Ok(started)
    }

    /// Stop the running session. New calls stop recording at once.
    pub fn stop(&self) -> Result<Stopped, StopError> {
        let _control = self.control.lock().unwrap_or_else(PoisonError::into_inner);
        let session = self
            .retire(EndReason::Stopped)
            .ok_or(StopError::NotRunning)?;
        let stopped = Stopped {
            session: session.id().to_string(),
            file: session.path().to_path_buf(),
            totals: session.totals(),
        };
        info!(session = %stopped.session, calls = stopped.totals.calls, "token dump stopped");
        Ok(stopped)
    }

    /// Stop session `session_id` if it is still the one running.
    pub fn expire(&self, session_id: &str) {
        let _control = self.control.lock().unwrap_or_else(PoisonError::into_inner);
        let running = self
            .current
            .load_full()
            .is_some_and(|session| session.id() == session_id);
        if running {
            self.retire(EndReason::Expired);
            info!(session = session_id, "token dump expired");
        }
    }

    /// Stop the running session and wait, a bounded time, for its file to be
    /// closed with its `session_end` line.
    pub async fn shutdown(&self) {
        let session = {
            let _control = self.control.lock().unwrap_or_else(PoisonError::into_inner);
            self.retire(EndReason::Shutdown)
        };
        let Some(session) = session else {
            return;
        };
        let done = session.take_writer_done();
        drop(session);
        if let Some(done) = done {
            if tokio::time::timeout(SHUTDOWN_FLUSH_TIMEOUT, done)
                .await
                .is_err()
            {
                warn!("token dump: calls still in flight at shutdown; the dump file may lack its session_end line");
            }
        }
    }

    fn open_session(
        &self,
        models: Vec<String>,
        duration: Option<Duration>,
    ) -> io::Result<Arc<Session>> {
        Session::open(
            &self.dir,
            models,
            duration,
            Limits {
                max_bytes: self.max_bytes,
                queue_budget: QUEUE_BUDGET_BYTES,
            },
        )
    }

    /// Make `session` the running one. The caller holds `control`, or owns
    /// the only handle to `self`.
    fn install(&self, session: Arc<Session>) {
        self.current.store(Some(session));
        gauge!("smg_token_dump_active").set(1.0);
    }

    /// Take the running session out, to end with `reason`. The caller holds
    /// `control`.
    fn retire(&self, reason: EndReason) -> Option<Arc<Session>> {
        let session = self.current.swap(None)?;
        session.set_end_reason(reason);
        gauge!("smg_token_dump_active").set(0.0);
        Some(session)
    }
}

/// [`TokenDump::start`] off the async runtime: it creates a file.
pub async fn start_blocking(
    dump: Arc<TokenDump>,
    request: StartRequest,
) -> Result<Started, StartError> {
    tokio::task::spawn_blocking(move || dump.start(request))
        .await
        .unwrap_or_else(|join| Err(StartError::Task(join.to_string())))
}

/// Stop session `session_id` once `after` has passed, if it is still the one
/// running. Nothing waits on this timer: shutdown stops the session itself.
/// Without a Tokio runtime there is no timer; the expiry still holds, since
/// lookups skip an expired session and the next start retires it.
fn arm_expiry(dump: Weak<TokenDump>, session_id: String, after: Duration) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    runtime.spawn(async move {
        tokio::time::sleep(after).await;
        if let Some(dump) = dump.upgrade() {
            dump.expire(&session_id);
        }
    });
}

/// The answer to either route when `--token-dump-dir` is not set.
pub fn not_configured() -> Response {
    error::not_found(
        "token_dump_dir_not_configured",
        "no directory for token dumps: start the gateway with --token-dump-dir <dir>",
    )
}

impl IntoResponse for StartError {
    fn into_response(self) -> Response {
        match self {
            Self::AlreadyRunning => error::create_error(
                StatusCode::CONFLICT,
                "token_dump_already_running",
                "a token dump session is running; POST /stop_token_dump first",
            ),
            Self::BadDuration(secs) => error::bad_request(
                "token_dump_bad_duration",
                format!("duration_secs must be 1..={MAX_DURATION_SECS}, got {secs}"),
            ),
            Self::Open(error) => error::internal_error(
                "token_dump_open_failed",
                format!("cannot open a token dump file: {error}"),
            ),
            Self::Task(reason) => {
                error::internal_error("token_dump_start_failed", format!("start task: {reason}"))
            }
        }
    }
}

impl IntoResponse for StopError {
    fn into_response(self) -> Response {
        match self {
            Self::NotRunning => error::create_error(
                StatusCode::CONFLICT,
                "token_dump_not_running",
                "no token dump session is running",
            ),
        }
    }
}

/// Help text for the token dump series. Call once the recorder is installed.
pub fn init_series() {
    describe_gauge!(
        "smg_token_dump_active",
        "1 while a token dump session records new engine calls"
    );
    describe_counter!(
        "smg_token_dump_calls_total",
        "Engine calls recorded by token dump sessions"
    );
    describe_counter!(
        "smg_token_dump_lines_written_total",
        "Token dump lines written to disk"
    );
    describe_counter!(
        "smg_token_dump_lines_dropped_total",
        "Token dump lines not written, by reason (queue_full, size_cap, write_error)"
    );
    describe_counter!(
        "smg_token_dump_bytes_written_total",
        "Bytes written to token dump files"
    );
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use super::*;

    fn dump_in(dir: &Path) -> Arc<TokenDump> {
        Arc::new(TokenDump::new(dir.to_path_buf(), 1 << 20))
    }

    fn read_lines(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// The writer closes a released session's file on its own thread; wait
    /// for its `session_end` line.
    async fn wait_for_session_end(path: &Path) -> Vec<Value> {
        for _ in 0..100 {
            let lines = read_lines(path);
            if lines
                .last()
                .is_some_and(|line| line["kind"] == "session_end")
            {
                return lines;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("{} never got its session_end line", path.display());
    }

    #[tokio::test]
    async fn one_session_runs_at_a_time() {
        let dir = tempfile::tempdir().unwrap();
        let dump = dump_in(dir.path());
        let started = dump.start(StartRequest::default()).unwrap();
        assert!(started.file.starts_with(dir.path()));
        assert!(started.expires_at.is_some());
        assert!(matches!(
            dump.start(StartRequest::default()),
            Err(StartError::AlreadyRunning)
        ));
        let stopped = dump.stop().unwrap();
        assert_eq!(stopped.session, started.session);
        assert!(matches!(dump.stop(), Err(StopError::NotRunning)));
        assert!(
            dump.start(StartRequest::default()).is_ok(),
            "a stopped session frees the slot"
        );
        let lines = wait_for_session_end(&started.file).await;
        assert_eq!(lines.last().unwrap()["reason"], "stopped");
    }

    #[tokio::test]
    async fn the_duration_must_be_in_range() {
        let dir = tempfile::tempdir().unwrap();
        let dump = dump_in(dir.path());
        for secs in [0, MAX_DURATION_SECS + 1] {
            assert!(matches!(
                dump.start(StartRequest { duration_secs: Some(secs), models: Vec::new() }),
                Err(StartError::BadDuration(rejected)) if rejected == secs
            ));
        }
        assert!(dump.session_for("m").is_none());
    }

    #[tokio::test]
    async fn session_for_follows_the_model_filter_and_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let dump = dump_in(dir.path());
        assert!(dump.session_for("a").is_none());
        dump.start(StartRequest {
            duration_secs: None,
            models: vec!["a".to_string()],
        })
        .unwrap();
        assert!(dump.session_for("a").is_some());
        assert!(dump.session_for("b").is_none());
        dump.stop().unwrap();
        assert!(dump.session_for("a").is_none());
    }

    #[tokio::test]
    async fn expiry_stops_only_its_own_session() {
        let dir = tempfile::tempdir().unwrap();
        let dump = dump_in(dir.path());
        let first = dump.start(StartRequest::default()).unwrap();
        dump.stop().unwrap();
        let second = dump.start(StartRequest::default()).unwrap();
        dump.expire(&first.session);
        assert!(
            dump.session_for("m").is_some(),
            "a stale timer leaves the new session running"
        );
        dump.expire(&second.session);
        assert!(dump.session_for("m").is_none());
        let lines = wait_for_session_end(&second.file).await;
        assert_eq!(lines.last().unwrap()["reason"], "expired");
    }

    #[tokio::test]
    async fn an_expired_session_does_not_block_a_new_start() {
        let dir = tempfile::tempdir().unwrap();
        let dump = dump_in(dir.path());
        let stale = dump.open_session(Vec::new(), Some(Duration::ZERO)).unwrap();
        let stale_path = stale.path().to_path_buf();
        dump.install(stale);
        assert!(
            dump.session_for("m").is_none(),
            "an expired session records no new call"
        );
        dump.start(StartRequest::default()).unwrap();
        let lines = wait_for_session_end(&stale_path).await;
        assert_eq!(lines.last().unwrap()["reason"], "expired");
    }

    #[tokio::test]
    async fn shutdown_closes_the_file_and_waits_for_it() {
        let dir = tempfile::tempdir().unwrap();
        let dump = dump_in(dir.path());
        let started = dump.start(StartRequest::default()).unwrap();
        dump.shutdown().await;
        let lines = read_lines(&started.file);
        assert_eq!(lines.last().unwrap()["kind"], "session_end");
        assert_eq!(lines.last().unwrap()["reason"], "shutdown");
        assert!(dump.session_for("m").is_none());
        dump.shutdown().await; // idle: returns at once
    }

    #[test]
    fn from_config_reads_the_flags() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = RouterConfig::default();
        assert!(TokenDump::from_config(&config).unwrap().is_none());

        config.token_dump_dir = Some(dir.path().display().to_string());
        let idle = TokenDump::from_config(&config).unwrap().unwrap();
        assert!(idle.session_for("m").is_none());

        config.token_dump_on_start = true;
        let boot = TokenDump::from_config(&config).unwrap().unwrap();
        let session = boot.session_for("any-model").unwrap();
        assert!(
            session.expires_at_text().is_none(),
            "a boot session never expires"
        );

        let not_a_dir = dir.path().join("not-a-dir");
        std::fs::write(&not_a_dir, b"").unwrap();
        config.token_dump_dir = Some(not_a_dir.display().to_string());
        assert!(TokenDump::from_config(&config)
            .err()
            .unwrap()
            .contains("--token-dump-on-start"));
    }

    #[test]
    fn errors_answer_with_their_status() {
        assert_eq!(not_configured().status(), StatusCode::NOT_FOUND);
        assert_eq!(
            StartError::AlreadyRunning.into_response().status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            StartError::BadDuration(0).into_response().status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            StartError::Open(io::Error::other("x"))
                .into_response()
                .status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            StopError::NotRunning.into_response().status(),
            StatusCode::CONFLICT
        );
    }

    #[test]
    fn the_start_body_rejects_unknown_fields() {
        assert!(serde_json::from_str::<StartRequest>(r#"{"path":"/etc"}"#).is_err());
        let body: StartRequest =
            serde_json::from_str(r#"{"duration_secs":5,"models":["m"]}"#).unwrap();
        assert_eq!(body.duration_secs, Some(5));
        assert_eq!(body.models, ["m"]);
        let empty: StartRequest = serde_json::from_str("{}").unwrap();
        assert!(empty.duration_secs.is_none() && empty.models.is_empty());
    }
}
