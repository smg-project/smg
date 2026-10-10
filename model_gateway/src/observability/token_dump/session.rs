//! One dump session: its file, the writer thread that owns the file, the
//! limits that keep it from slowing requests or filling the disk, and the
//! recorder of each engine call.
//!
//! The request path never waits on the disk. A line is handed to the
//! session's writer thread over a bounded channel, or counted and dropped
//! when the queue is past its byte budget or the file past its size cap. The
//! writer closes the file, with a `session_end` line, once every handle to
//! the session is gone: the dump's own, and those of calls still in flight.

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::{
    fs::{DirBuilder, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
        Arc, Mutex, PoisonError,
    },
    thread,
    time::{Duration, Instant},
};

use chrono::{DateTime, SecondsFormat, TimeDelta, Utc};
use metrics::{counter, gauge};
use tokio::sync::oneshot;
use tracing::warn;

use super::line::{
    self, CallMeta, Dropped, EndReason, EndStatus, RequestEvent, ResponseEvent, SessionHeader,
    Totals,
};

/// Bytes of lines queued for the writer before new ones are dropped
/// (`queue_full`): a burst of large requests costs at most this much memory.
pub(super) const QUEUE_BUDGET_BYTES: u64 = 64 << 20;
/// Lines the writer queue holds at most, whatever their size.
const QUEUE_LINES: usize = 65_536;
/// File space held back from the cap for the `session_end` line.
const SESSION_END_RESERVE: u64 = 1024;

/// A session's size limits.
pub(super) struct Limits {
    /// The file's size cap.
    pub max_bytes: u64,
    /// Bytes of queued lines the writer may hold.
    pub queue_budget: u64,
}

/// One dump file being written.
pub struct Session {
    id: String,
    path: PathBuf,
    models: Vec<String>,
    expires_at: Option<Instant>,
    expires_at_text: Option<String>,
    tx: SyncSender<Vec<u8>>,
    stats: Arc<Stats>,
    next_call: AtomicU64,
    writer_done: Mutex<Option<oneshot::Receiver<()>>>,
}

/// Why a line was not written.
#[derive(Debug, Clone, Copy)]
enum DropReason {
    QueueFull,
    SizeCap,
    WriteError,
}

impl DropReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::QueueFull => "queue_full",
            Self::SizeCap => "size_cap",
            Self::WriteError => "write_error",
        }
    }
}

/// Counters shared by a session and its writer thread.
struct Stats {
    max_bytes: u64,
    queue_budget: u64,
    /// Bytes of accepted lines, against the file cap.
    file_bytes: AtomicU64,
    /// Bytes of lines queued for the writer and not yet written.
    queued_bytes: AtomicU64,
    calls: AtomicU64,
    lines_written: AtomicU64,
    bytes_written: AtomicU64,
    queue_full: AtomicU64,
    size_cap: AtomicU64,
    write_error: AtomicU64,
    write_error_logged: AtomicBool,
    /// Set by the first line the file cap refused: a full session takes no
    /// new call.
    full: AtomicBool,
    end_reason: AtomicU8,
}

impl Stats {
    fn new(limits: &Limits) -> Self {
        Self {
            max_bytes: limits.max_bytes,
            queue_budget: limits.queue_budget,
            file_bytes: AtomicU64::new(0),
            queued_bytes: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            lines_written: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            queue_full: AtomicU64::new(0),
            size_cap: AtomicU64::new(0),
            write_error: AtomicU64::new(0),
            write_error_logged: AtomicBool::new(false),
            full: AtomicBool::new(false),
            end_reason: AtomicU8::new(EndReason::Shutdown.to_u8()),
        }
    }

    fn dropped(&self, reason: DropReason) {
        let count = match reason {
            DropReason::QueueFull => &self.queue_full,
            DropReason::SizeCap => &self.size_cap,
            DropReason::WriteError => &self.write_error,
        };
        count.fetch_add(1, Ordering::Relaxed);
        counter!("smg_token_dump_lines_dropped_total", "reason" => reason.as_str()).increment(1);
        if matches!(reason, DropReason::SizeCap) && !self.full.swap(true, Ordering::AcqRel) {
            // A full session takes no new call, so it no longer counts as
            // recording.
            gauge!("smg_token_dump_active").set(0.0);
            warn!(
                max_bytes = self.max_bytes,
                "token dump: the dump file reached its size cap; its session records no new call"
            );
        }
    }

    fn written(&self, len: u64) {
        self.lines_written.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(len, Ordering::Relaxed);
        counter!("smg_token_dump_lines_written_total").increment(1);
        counter!("smg_token_dump_bytes_written_total").increment(len);
    }

    fn write_failed(&self, session: &str, error: &io::Error) {
        self.dropped(DropReason::WriteError);
        if !self.write_error_logged.swap(true, Ordering::Relaxed) {
            warn!(
                session,
                %error,
                "token dump: writing the dump file failed; later failures of this session are only counted"
            );
        }
    }

    fn end_reason(&self) -> EndReason {
        EndReason::from_u8(self.end_reason.load(Ordering::Acquire))
    }

    fn totals(&self) -> Totals {
        Totals {
            calls: self.calls.load(Ordering::Relaxed),
            lines_written: self.lines_written.load(Ordering::Relaxed),
            lines_dropped: Dropped {
                queue_full: self.queue_full.load(Ordering::Relaxed),
                size_cap: self.size_cap.load(Ordering::Relaxed),
                write_error: self.write_error.load(Ordering::Relaxed),
            },
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
        }
    }
}

impl Session {
    /// Open a new session file in `dir`, creating the directory (mode 0700)
    /// if it is missing.
    pub(super) fn open(
        dir: &Path,
        models: Vec<String>,
        duration: Option<Duration>,
        limits: Limits,
    ) -> io::Result<Arc<Self>> {
        create_private_dir(dir)?;
        let id = uuid::Uuid::now_v7().to_string();
        let path = dir.join(format!("token-dump-{id}.jsonl"));
        let file = create_private_file(&path)?;
        Self::start(id, path, models, duration, limits, Box::new(file))
    }

    /// Write the `session` line into `sink` and hand `sink` to a new writer
    /// thread.
    pub(super) fn start(
        id: String,
        path: PathBuf,
        models: Vec<String>,
        duration: Option<Duration>,
        limits: Limits,
        mut sink: Box<dyn Write + Send>,
    ) -> io::Result<Arc<Self>> {
        let now = Utc::now();
        let expires_at_text = duration
            .and_then(|duration| TimeDelta::from_std(duration).ok())
            .map(|duration| rfc3339(now + duration));
        let header = line::session_line(&SessionHeader {
            session: &id,
            started_at: &rfc3339(now),
            models: &models,
            expires_at: expires_at_text.as_deref(),
            max_bytes: limits.max_bytes,
        });
        sink.write_all(&header)?;
        let stats = Arc::new(Stats::new(&limits));
        stats
            .file_bytes
            .store(header.len() as u64, Ordering::Release);
        stats.written(header.len() as u64);

        let (tx, rx) = mpsc::sync_channel(QUEUE_LINES);
        let (done_tx, done_rx) = oneshot::channel();
        let writer_stats = Arc::clone(&stats);
        let writer_id = id.clone();
        // Detached: the writer ends on its own once every sender is gone.
        let _writer = thread::Builder::new()
            .name("smg-token-dump".to_string())
            .spawn(move || run_writer(rx, sink, &writer_stats, &writer_id, done_tx))?;

        Ok(Arc::new(Self {
            id,
            path,
            models,
            expires_at: duration.map(|duration| Instant::now() + duration),
            expires_at_text,
            tx,
            stats,
            next_call: AtomicU64::new(0),
            writer_done: Mutex::new(Some(done_rx)),
        }))
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The models this session records; empty means all.
    pub fn models(&self) -> &[String] {
        &self.models
    }

    /// When a runtime session expires (RFC 3339); `None` for a boot session.
    pub fn expires_at_text(&self) -> Option<&str> {
        self.expires_at_text.as_deref()
    }

    pub fn totals(&self) -> Totals {
        self.stats.totals()
    }

    pub(super) fn expired(&self, now: Instant) -> bool {
        self.expires_at.is_some_and(|at| now >= at)
    }

    pub(super) fn records_model(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|listed| listed == model)
    }

    /// The reason the `session_end` line will give.
    pub(super) fn set_end_reason(&self, reason: EndReason) {
        self.stats
            .end_reason
            .store(reason.to_u8(), Ordering::Release);
    }

    /// Resolves once the writer has written `session_end` and closed the
    /// file. Only the first caller gets it.
    pub(super) fn take_writer_done(&self) -> Option<oneshot::Receiver<()>> {
        self.writer_done
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Whether the file cap has refused a line: a full session takes no new
    /// call.
    pub(super) fn is_full(&self) -> bool {
        self.stats.full.load(Ordering::Acquire)
    }

    /// Whether a line at least `min_len` bytes long could still be taken by
    /// the file cap and the writer queue. A check, not a reservation:
    /// [`write`](Self::write) reserves the line's exact size. It lets a
    /// caller skip building a line that cannot fit.
    fn room_for(&self, min_len: usize) -> Result<(), DropReason> {
        let stats = &self.stats;
        let min_len = min_len as u64;
        let file_cap = stats.max_bytes.saturating_sub(SESSION_END_RESERVE);
        if stats
            .file_bytes
            .load(Ordering::Acquire)
            .saturating_add(min_len)
            > file_cap
        {
            return Err(DropReason::SizeCap);
        }
        if stats
            .queued_bytes
            .load(Ordering::Acquire)
            .saturating_add(min_len)
            > stats.queue_budget
        {
            return Err(DropReason::QueueFull);
        }
        Ok(())
    }

    /// Record a new engine call. Its `request` line, at least `min_len` bytes
    /// long, is built by `request` and written now; when even `min_len` bytes
    /// cannot fit, the line is counted dropped without being built.
    pub fn begin_call<'a>(
        self: &Arc<Self>,
        meta: &CallMeta,
        min_len: usize,
        request: impl FnOnce() -> RequestEvent<'a>,
    ) -> CallRecorder {
        let call = self.next_call.fetch_add(1, Ordering::Relaxed);
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        counter!("smg_token_dump_calls_total").increment(1);
        match self.room_for(min_len) {
            Ok(()) => self.write(line::request_line(
                call,
                &rfc3339(Utc::now()),
                meta,
                &request(),
            )),
            Err(reason) => self.stats.dropped(reason),
        }
        CallRecorder {
            session: Arc::clone(self),
            call,
            started: Instant::now(),
            responses: 0,
            completes_expected: 1,
            completes_seen: 0,
            completed_ms: None,
            ended: false,
        }
    }

    /// Queue `line` for the writer, or count it dropped. Never blocks.
    fn write(&self, line: Vec<u8>) {
        if line.is_empty() {
            return;
        }
        let len = line.len() as u64;
        let stats = &self.stats;
        let file_cap = stats.max_bytes.saturating_sub(SESSION_END_RESERVE);
        if !reserve(&stats.file_bytes, len, file_cap) {
            stats.dropped(DropReason::SizeCap);
            return;
        }
        if !reserve(&stats.queued_bytes, len, stats.queue_budget) {
            stats.file_bytes.fetch_sub(len, Ordering::AcqRel);
            stats.dropped(DropReason::QueueFull);
            return;
        }
        if let Err(error) = self.tx.try_send(line) {
            stats.file_bytes.fetch_sub(len, Ordering::AcqRel);
            stats.queued_bytes.fetch_sub(len, Ordering::AcqRel);
            stats.dropped(match error {
                TrySendError::Full(_) => DropReason::QueueFull,
                TrySendError::Disconnected(_) => DropReason::WriteError,
            });
        }
    }
}

/// One engine call in a session. Writes its `request` line when made, a
/// `response` line per engine message, and one `end` line: from
/// [`finish`](Self::finish), or `cancelled` when dropped before that.
pub struct CallRecorder {
    session: Arc<Session>,
    call: u64,
    started: Instant,
    responses: u64,
    /// `Complete` messages the request asks for: one per sample.
    completes_expected: u32,
    completes_seen: u32,
    /// When the last expected `Complete` arrived (ms since the call began).
    completed_ms: Option<u64>,
    ended: bool,
}

impl CallRecorder {
    /// Record one engine message, built by `event` unless its line (at least
    /// `min_len` bytes long) cannot fit. Ignored once the call has ended.
    pub fn response<'a>(&mut self, min_len: usize, event: impl FnOnce() -> ResponseEvent<'a>) {
        if self.ended {
            return;
        }
        let seq = self.responses;
        self.responses += 1;
        match self.session.room_for(min_len) {
            Ok(()) => self.session.write(line::response_line(
                self.call,
                seq,
                self.elapsed_ms(),
                &event(),
            )),
            Err(reason) => self.session.stats.dropped(reason),
        }
    }

    /// The request asks for `n` samples, so the engine finishes it with `n`
    /// `Complete` messages.
    pub fn expect_completes(&mut self, n: u32) {
        self.completes_expected = n.max(1);
    }

    /// The engine sent a (non-error) `Complete`. Once every expected one has
    /// arrived the call is over as far as the engine is concerned: it ends
    /// `ok`, timed at that moment, even if the gateway drops the stream later
    /// without reading on.
    pub fn complete(&mut self) {
        self.completes_seen += 1;
        if self.completes_seen >= self.completes_expected && self.completed_ms.is_none() {
            self.completed_ms = Some(self.elapsed_ms());
        }
    }

    /// Record how the call ended; only the first end counts. A call the
    /// engine completed ends `ok` at its completion time unless it failed.
    pub fn finish(&mut self, status: EndStatus, error: Option<&tonic::Status>) {
        if std::mem::replace(&mut self.ended, true) {
            return;
        }
        let (status, t_ms) = match (status, self.completed_ms) {
            (EndStatus::Ok | EndStatus::Cancelled, Some(completed_ms)) => {
                (EndStatus::Ok, completed_ms)
            }
            (status, _) => (status, self.elapsed_ms()),
        };
        self.session.write(line::end_line(
            self.call,
            t_ms,
            status,
            self.responses,
            error,
        ));
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

impl Drop for CallRecorder {
    fn drop(&mut self) {
        self.finish(EndStatus::Cancelled, None);
    }
}

/// The writer thread: write each queued line, then, once every sender is
/// gone, the `session_end` line, and signal `done`.
fn run_writer(
    rx: Receiver<Vec<u8>>,
    mut sink: Box<dyn Write + Send>,
    stats: &Stats,
    session: &str,
    done: oneshot::Sender<()>,
) {
    while let Ok(line) = rx.recv() {
        let len = line.len() as u64;
        match sink.write_all(&line) {
            Ok(()) => stats.written(len),
            Err(error) => stats.write_failed(session, &error),
        }
        stats.queued_bytes.fetch_sub(len, Ordering::AcqRel);
    }
    let end = line::session_end_line(session, stats.end_reason(), &stats.totals());
    match sink.write_all(&end).and_then(|()| sink.flush()) {
        Ok(()) => stats.written(end.len() as u64),
        Err(error) => stats.write_failed(session, &error),
    }
    // A shutdown that waited on this may have stopped waiting.
    let _ = done.send(());
}

/// Add `len` to `counter` unless that takes it past `limit`.
fn reserve(counter: &AtomicU64, len: u64, limit: u64) -> bool {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
            held.checked_add(len).filter(|&total| total <= limit)
        })
        .is_ok()
}

fn rfc3339(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// The dump directory, made readable by its owner only when this creates it.
fn create_private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(0o700);
    builder.create(dir)
}

/// A new dump file, owner-only. Fails when `path` exists: a session never
/// appends to or overwrites another file.
fn create_private_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        io,
        sync::{Arc, Mutex},
        thread,
        time::Duration,
    };

    use serde_json::{json, Value};

    use super::*;
    use crate::observability::token_dump::line::{Leg, Part, Transport};

    /// A sink tests read back.
    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Shared {
        fn lines(&self) -> Vec<Value> {
            self.0
                .lock()
                .unwrap()
                .split(|&byte| byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect()
        }

        fn len(&self) -> u64 {
            self.0.lock().unwrap().len() as u64
        }
    }

    /// Takes the session header, then fails every write.
    struct FailsAfterHeader {
        wrote_header: bool,
    }

    impl Write for FailsAfterHeader {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.wrote_header {
                return Err(io::Error::other("disk full"));
            }
            self.wrote_header = true;
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn limits(max_bytes: u64, queue_budget: u64) -> Limits {
        Limits {
            max_bytes,
            queue_budget,
        }
    }

    fn open(sink: impl Write + Send + 'static, limits: Limits) -> Arc<Session> {
        Session::start(
            "s1".to_string(),
            PathBuf::from("unused"),
            Vec::new(),
            None,
            limits,
            Box::new(sink),
        )
        .unwrap()
    }

    fn meta() -> CallMeta {
        CallMeta {
            model: "m".to_string(),
            worker: "grpc://w:1".to_string(),
            runtime: "sglang",
            transport: Transport::Grpc,
            leg: Leg::Single,
            root_request_id: None,
        }
    }

    fn request() -> RequestEvent<'static> {
        RequestEvent {
            type_name: "t.Request",
            request_id: "r",
            input_ids: &[1, 2],
            msg: vec![1],
        }
    }

    fn chunk() -> ResponseEvent<'static> {
        ResponseEvent {
            type_name: "t.Response",
            part: Part::Chunk,
            index: 0,
            token_ids: &[9],
            finish_reason: None,
            msg: vec![2],
        }
    }

    /// Release `session` and wait for its writer to write `session_end`.
    fn close(session: Arc<Session>) {
        let done = session.take_writer_done().unwrap();
        drop(session);
        done.blocking_recv().unwrap();
    }

    fn kinds(lines: &[Value]) -> Vec<&str> {
        lines
            .iter()
            .map(|line| line["kind"].as_str().unwrap())
            .collect()
    }

    #[test]
    fn a_session_writes_header_call_lines_and_session_end() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        let mut call = session.begin_call(&meta(), 0, request);
        call.response(0, chunk);
        call.finish(EndStatus::Ok, None);
        drop(call);
        session.set_end_reason(EndReason::Stopped);
        close(session);

        let lines = sink.lines();
        assert_eq!(
            kinds(&lines),
            ["session", "request", "response", "end", "session_end"]
        );
        assert_eq!(lines[1]["call"], 0);
        assert_eq!(lines[2]["seq"], 0);
        assert_eq!(lines[3]["status"], "ok");
        assert_eq!(lines[3]["responses"], 1);
        let end = &lines[4];
        assert_eq!(end["reason"], "stopped");
        assert_eq!(end["calls"], 1);
        assert_eq!(end["lines_written"], 4);
        assert_eq!(
            end["lines_dropped"],
            json!({"queue_full":0,"size_cap":0,"write_error":0})
        );
        let last_line_len = serde_json::to_vec(end).unwrap().len() as u64 + 1;
        assert_eq!(end["bytes_written"], sink.len() - last_line_len);
    }

    #[test]
    fn a_call_dropped_before_it_ends_is_recorded_cancelled() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        drop(session.begin_call(&meta(), 0, request));
        close(session);
        let lines = sink.lines();
        assert_eq!(kinds(&lines), ["session", "request", "end", "session_end"]);
        assert_eq!(lines[2]["status"], "cancelled");
        assert_eq!(lines[2]["responses"], 0);
    }

    #[test]
    fn a_call_dropped_mid_stream_counts_its_responses() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        let mut call = session.begin_call(&meta(), 0, request);
        call.response(0, chunk);
        call.response(0, chunk);
        drop(call);
        close(session);
        let lines = sink.lines();
        assert_eq!(lines[3]["seq"], 1);
        assert_eq!(lines[4]["status"], "cancelled");
        assert_eq!(lines[4]["responses"], 2);
    }

    #[test]
    fn a_call_ends_once() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        let mut call = session.begin_call(&meta(), 0, request);
        call.finish(EndStatus::Error, Some(&tonic::Status::internal("boom")));
        call.finish(EndStatus::Ok, None);
        call.response(0, chunk);
        drop(call);
        close(session);
        let lines = sink.lines();
        assert_eq!(kinds(&lines), ["session", "request", "end", "session_end"]);
        assert_eq!(lines[2]["status"], "error");
        assert_eq!(lines[2]["error"]["message"], "boom");
    }

    #[test]
    fn calls_in_flight_finish_after_the_session_is_released() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        let done = session.take_writer_done().unwrap();
        let mut call = session.begin_call(&meta(), 0, request);
        session.set_end_reason(EndReason::Stopped);
        drop(session);
        call.response(0, chunk);
        call.finish(EndStatus::Ok, None);
        drop(call);
        done.blocking_recv().unwrap();
        assert_eq!(
            kinds(&sink.lines()),
            ["session", "request", "response", "end", "session_end"]
        );
    }

    #[test]
    fn concurrent_calls_get_distinct_ids_and_whole_groups() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let session = Arc::clone(&session);
                thread::spawn(move || {
                    for _ in 0..25 {
                        let mut call = session.begin_call(&meta(), 0, request);
                        call.response(0, chunk);
                        call.finish(EndStatus::Ok, None);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        close(session);

        let mut calls: BTreeMap<u64, Vec<String>> = BTreeMap::new();
        for line in sink.lines() {
            if let Some(call) = line["call"].as_u64() {
                calls
                    .entry(call)
                    .or_default()
                    .push(line["kind"].as_str().unwrap().to_string());
            }
        }
        assert_eq!(calls.len(), 200);
        for kinds in calls.values() {
            assert_eq!(kinds, &["request", "response", "end"]);
        }
    }

    #[test]
    fn lines_past_the_size_cap_are_dropped_and_session_end_still_fits() {
        let sink = Shared::default();
        let max_bytes = 4096;
        let session = open(sink.clone(), limits(max_bytes, 1 << 20));
        let stats = Arc::clone(&session.stats);
        for _ in 0..50 {
            drop(session.begin_call(&meta(), 0, request));
        }
        close(session);

        let lines = sink.lines();
        assert_eq!(lines.last().unwrap()["kind"], "session_end");
        assert!(sink.len() <= max_bytes, "{} bytes", sink.len());
        let dropped = stats.totals().lines_dropped;
        assert!(dropped.size_cap > 0);
        assert_eq!(dropped.queue_full + dropped.write_error, 0);
        assert_eq!(
            lines.last().unwrap()["lines_dropped"]["size_cap"],
            dropped.size_cap
        );
    }

    #[test]
    fn a_full_queue_drops_lines_instead_of_blocking() {
        let sink = Shared::default();
        // Every line is longer than 16 bytes, so none fits the queue.
        let session = open(sink.clone(), limits(1 << 20, 16));
        let stats = Arc::clone(&session.stats);
        for _ in 0..10 {
            drop(session.begin_call(&meta(), 0, request));
        }
        close(session);

        let totals = stats.totals();
        assert_eq!(totals.calls, 10);
        assert_eq!(totals.lines_dropped.queue_full, 20);
        assert_eq!(stats.queued_bytes.load(Ordering::Acquire), 0);
        assert_eq!(kinds(&sink.lines()), ["session", "session_end"]);
    }

    #[test]
    fn a_call_that_got_every_complete_ends_ok_even_when_dropped() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        let mut short = session.begin_call(&meta(), 0, request);
        short.expect_completes(2);
        short.response(0, chunk);
        short.complete();
        drop(short);
        let mut whole = session.begin_call(&meta(), 0, request);
        whole.expect_completes(2);
        whole.complete();
        whole.complete();
        drop(whole);
        close(session);
        let ends: Vec<Value> = sink
            .lines()
            .into_iter()
            .filter(|line| line["kind"] == "end")
            .collect();
        assert_eq!(ends[0]["status"], "cancelled", "one Complete of two");
        assert_eq!(ends[1]["status"], "ok", "every Complete arrived");
    }

    #[test]
    fn an_error_after_every_complete_is_still_an_error() {
        let sink = Shared::default();
        let session = open(sink.clone(), limits(1 << 20, 1 << 20));
        let mut call = session.begin_call(&meta(), 0, request);
        call.complete();
        call.finish(EndStatus::Error, Some(&tonic::Status::internal("late")));
        drop(call);
        close(session);
        assert_eq!(sink.lines()[2]["status"], "error");
    }

    #[test]
    fn a_line_that_cannot_fit_is_never_built() {
        fn never_built() -> RequestEvent<'static> {
            panic!("built a request line that cannot fit")
        }
        fn never_built_response() -> ResponseEvent<'static> {
            panic!("built a response line that cannot fit")
        }
        // Past the file cap: a 1 MiB line in a 4 KiB file.
        let capped = open(io::sink(), limits(4096, 1 << 30));
        let mut call = capped.begin_call(&meta(), 1 << 20, never_built);
        call.response(1 << 20, never_built_response);
        drop(call);
        let totals = capped.totals();
        assert_eq!(totals.lines_dropped.size_cap, 2);
        assert!(capped.is_full());

        // Past the queue budget: a 1 KiB line through a 16-byte queue.
        let queued = open(io::sink(), limits(1 << 20, 16));
        drop(queued.begin_call(&meta(), 1024, never_built));
        assert!(queued.totals().lines_dropped.queue_full >= 1);
        assert!(!queued.is_full(), "a full queue is not a full file");
    }

    #[test]
    fn write_errors_are_counted_not_raised() {
        let session = open(
            FailsAfterHeader {
                wrote_header: false,
            },
            limits(1 << 20, 1 << 20),
        );
        let stats = Arc::clone(&session.stats);
        drop(session.begin_call(&meta(), 0, request));
        close(session);
        // The request, its end and the session_end line all failed.
        assert_eq!(stats.totals().lines_dropped.write_error, 3);
        assert_eq!(stats.totals().lines_written, 1);
    }

    #[test]
    fn the_model_filter_and_expiry_gate_new_calls() {
        let all = Session::start(
            "a".to_string(),
            PathBuf::new(),
            Vec::new(),
            None,
            limits(1 << 20, 1 << 20),
            Box::new(io::sink()),
        )
        .unwrap();
        assert!(all.records_model("anything"));
        assert!(!all.expired(Instant::now()));
        assert!(all.expires_at_text().is_none());

        let some = Session::start(
            "b".to_string(),
            PathBuf::new(),
            vec!["m1".to_string()],
            Some(Duration::ZERO),
            limits(1 << 20, 1 << 20),
            Box::new(io::sink()),
        )
        .unwrap();
        assert!(some.records_model("m1"));
        assert!(!some.records_model("m2"));
        assert!(some.expired(Instant::now()));
        assert!(some.expires_at_text().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn dump_files_are_private_and_never_reused() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("dumps");
        let session = Session::open(&dir, Vec::new(), None, limits(1 << 20, 1 << 20)).unwrap();
        let path = session.path().to_path_buf();
        assert_eq!(
            std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            create_private_file(&path).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("token-dump-") && name.ends_with(".jsonl"));
        close(session);
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .starts_with(r#"{"v":1,"kind":"session","#));
    }
}
