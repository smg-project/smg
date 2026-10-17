//! Logging infrastructure with non-blocking I/O for every sink.
//!
//! Each formatting layer hands its lines to a dedicated writer thread through
//! a bounded, lossy channel ([`tracing_appender::non_blocking`]), so the tokio
//! runtime threads that emit the events never wait for the sink. That matters
//! for stdout as much as for the log file: when stdout is redirected to a
//! regular file, every `write(2)` is subject to the kernel's dirty-page
//! throttling and can sleep for hundreds of milliseconds under writeback
//! pressure, which with a synchronous `std::io::stdout()` writer parked a
//! runtime worker, and every connection it was driving, once per access-log
//! line.
//!
//! The writer threads write in batches ([`Batched`]): the first line after a
//! quiet moment opens a short window, and the lines queued within it leave
//! in one write. Nothing but the writer thread waits for the window; a line
//! is visible a few milliseconds later at most.

use std::{
    io::{self, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Duration,
};

use tracing::{warn, Level};
use tracing_appender::{
    non_blocking::{ErrorCounter, NonBlocking, NonBlockingBuilder, WorkerGuard},
    rolling::{RollingFileAppender, Rotation},
};
use tracing_log::{AsLog, LogTracer};
use tracing_subscriber::{
    filter::{LevelFilter, Targets},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};

use super::{
    log_format::{JsonLine, SpanFields, UtcSeconds},
    otel_trace::get_otel_layer,
};
use crate::{config::TraceConfig, observability::metrics::Metrics};

/// All workspace crate names (hyphens → underscores to match tracing targets).
/// When no explicit log targets are configured, these crates get the configured
/// log level while all external deps default to WARN.
///
/// Update this list when adding a new workspace crate.
const WORKSPACE_CRATES: &[&str] = &[
    "data_connector",
    "kv_index",
    "llm_multimodal",
    "llm_tokenizer",
    "openai_protocol",
    "reasoning_parser",
    "smg",
    "smg_auth",
    "smg_grpc_client",
    "smg_mcp",
    "smg_mesh",
    "smg_wasm",
    "tool_parser",
    "wfaas",
];

#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub level: Level,
    pub json_format: bool,
    pub log_dir: Option<String>,
    pub colorize: bool,
    pub log_file_name: String,
    pub log_targets: Option<Vec<String>>,
}

impl Default for LoggingConfig {
    #[inline]
    fn default() -> Self {
        Self {
            level: Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: true,
            log_file_name: "smg".to_string(),
            log_targets: None, // None = use workspace crate filter
        }
    }
}

/// What `init_logging` hands back. The writer threads live in [`SINKS`] for
/// the process, not in this value: the subscriber is installed once per
/// process and every router started after the first (the Python binding
/// starts them in one process) logs through the same writers, so a guard
/// that died with the first `startup` call would leave every later line
/// queued behind a thread that had exited. Dropping this stops nothing;
/// [`close_logging`] flushes and stops the writers at process exit.
pub struct LogGuard(());

/// One writer thread: its handle, and the count of lines its lossy queue
/// dropped.
struct Sink {
    name: &'static str,
    counter: ErrorCounter,
    /// The count at the last report, for the delta.
    reported: usize,
    _guard: WorkerGuard,
}

/// The writer threads of every sink installed in this process.
struct Sinks {
    sinks: Vec<Sink>,
    reporter_started: bool,
}

static SINKS: Mutex<Sinks> = Mutex::new(Sinks {
    sinks: Vec::new(),
    reporter_started: false,
});

/// Set by [`close_logging`]; the reporter thread ends at its next tick.
static CLOSED: AtomicBool = AtomicBool::new(false);

/// How often the dropped-line counts are published.
const DROPPED_LINES_REPORT_SECS: u64 = 30;

/// How long a writer thread waits, from the first line after a quiet moment,
/// for more lines before it writes: the lines logged within the window leave
/// in one write, and the threads that log them wake nobody. A line is visible
/// this much later at most, and a process that dies without [`close_logging`]
/// loses at most this window of lines (on top of what sat in its queue).
const BATCH_WINDOW: Duration = Duration::from_millis(5);

/// A batch is written as soon as it holds this many bytes, window or not: a
/// backlog (a burst, or a sink that stalled and recovered) drains at full
/// speed in writes of this size, and the buffer stays this small.
const BATCH_BYTES: usize = 64 * 1024;

/// A sink that takes the writer thread's lines in batches.
///
/// [`tracing_appender`]'s writer thread hands its sink one whole line per
/// `write`, drains its queue until the queue is empty and calls `flush`
/// then. On a steady access-log stream the queue is empty again after every
/// line, so each line cost one `write(2)` on the sink and, with the thread
/// parked between lines, one wake-up paid by the thread that logged. This
/// adapter opens a window on the first line after a flush: the writer thread
/// sleeps for [`BATCH_WINDOW`] (the lines logged meanwhile queue up, with
/// nobody to wake), then drains them into one buffer, and the `flush` that
/// ends the drain writes the buffer in one call. Only whole lines enter and
/// leave the buffer, in the order received; a batch that reaches
/// [`BATCH_BYTES`] is written at once and the drain goes on without another
/// wait; the flush at shutdown (the worker guard's drop in
/// [`close_logging`]) writes whatever is buffered, so a clean exit loses
/// nothing.
struct Batched<W> {
    sink: W,
    buf: Vec<u8>,
    window: Duration,
    cap: usize,
    /// A window is open: the next line does not wait for one again.
    open: bool,
}

impl<W: Write> Batched<W> {
    fn new(sink: W) -> Self {
        Self::with(sink, BATCH_WINDOW, BATCH_BYTES)
    }

    fn with(sink: W, window: Duration, cap: usize) -> Self {
        Self {
            sink,
            buf: Vec::with_capacity(cap),
            window,
            cap,
            open: false,
        }
    }

    /// Write the buffered lines in one call. On an error the batch is gone,
    /// as a line whose write failed was before; nothing is written twice.
    fn write_batch(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let written = self.sink.write_all(&self.buf);
        self.buf.clear();
        // A line longer than the cap grew the buffer; let that go again.
        if self.buf.capacity() > 2 * self.cap {
            self.buf.shrink_to(self.cap);
        }
        written
    }
}

impl<W: Write> Write for Batched<W> {
    fn write(&mut self, line: &[u8]) -> io::Result<usize> {
        if !self.open {
            self.open = true;
            if !self.window.is_zero() {
                std::thread::sleep(self.window);
            }
        }
        self.buf.extend_from_slice(line);
        if self.buf.len() >= self.cap {
            self.write_batch()?;
        }
        Ok(line.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.open = false;
        // The worker discards a flush error, and an `Err` from the flush that
        // ends the shutdown drain skips its rendezvous with the guard (which
        // then waits out its full second): the batch is gone either way, as
        // a line whose own write failed was before.
        let _ = self.write_batch();
        self.sink.flush()
    }
}

/// Move `sink` to its own writer thread held for the process, writing in
/// batches, and return the writer for a subscriber layer. The thread is
/// counted among the sinks whose dropped lines the reporter publishes.
fn install<W: Write + Send + 'static>(sink: W, name: &'static str) -> NonBlocking {
    let (writer, guard) = non_blocking(Batched::new(sink), name);
    let mut sinks = SINKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sinks.sinks.push(Sink {
        name,
        counter: writer.error_counter(),
        reported: 0,
        _guard: guard,
    });
    if !sinks.reporter_started {
        sinks.reporter_started = std::thread::Builder::new()
            .name("smg-log-dropped".to_string())
            .spawn(|| {
                let tick = Duration::from_millis(100);
                let ticks_per_report = DROPPED_LINES_REPORT_SECS * 10;
                loop {
                    for _ in 0..ticks_per_report {
                        if CLOSED.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(tick);
                    }
                    report_dropped_lines();
                }
            })
            .is_ok();
    }
    writer
}

/// Lines each writer's queue has dropped since the process started, by sink.
pub fn dropped_lines() -> Vec<(&'static str, usize)> {
    SINKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .sinks
        .iter()
        .map(|sink| (sink.name, sink.counter.dropped_lines()))
        .collect()
}

/// Publish each sink's dropped-line count as `smg_log_dropped_lines{sink}` and
/// warn about the lines dropped since the last report. Returns `(sink, total,
/// since the last report)` per sink.
fn report_dropped_lines() -> Vec<(&'static str, usize, usize)> {
    let mut sinks = SINKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut reports = Vec::with_capacity(sinks.sinks.len());
    for sink in &mut sinks.sinks {
        let total = sink.counter.dropped_lines();
        let since_last = total.saturating_sub(sink.reported);
        sink.reported = total;
        reports.push((sink.name, total, since_last));
    }
    drop(sinks);
    for &(name, total, since_last) in &reports {
        Metrics::set_log_dropped_lines(name, total);
        if since_last > 0 {
            warn!(
                sink = name,
                dropped_total = total,
                dropped_since_last_report = since_last,
                "log lines dropped: the writer queue was full"
            );
        }
    }
    reports
}

/// Flush and stop the writer threads: the last lines queued, and the batch a
/// writer is still gathering, are written before the process exits. Called
/// once, at the end of `main`; nothing logs through the stopped writers
/// afterwards.
pub fn close_logging() {
    CLOSED.store(true, Ordering::Relaxed);
    let sinks = std::mem::take(
        &mut SINKS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .sinks,
    );
    drop(sinks);
}

/// Move `sink` to its own thread behind a bounded, lossy channel.
///
/// Callers never block on the sink: a line is queued for the writer thread, or
/// counted and dropped when the queue is full. A sink that stalls therefore
/// slows nothing but its own thread.
#[inline]
fn non_blocking<W: Write + Send + 'static>(
    sink: W,
    thread_name: &str,
) -> (NonBlocking, WorkerGuard) {
    NonBlockingBuilder::default()
        .lossy(true)
        .thread_name(thread_name)
        .finish(sink)
}

#[inline]
const fn level_to_str(level: Level) -> &'static str {
    match level {
        Level::TRACE => "trace",
        Level::DEBUG => "debug",
        Level::INFO => "info",
        Level::WARN => "warn",
        Level::ERROR => "error",
    }
}

#[inline]
fn build_filter_string(targets: &[String], level_filter: &str) -> String {
    // Exact capacity: sum of target lengths + "=" and level per target + commas between
    let capacity = targets.iter().map(String::len).sum::<usize>()
        + targets.len() * (1 + level_filter.len())
        + targets.len().saturating_sub(1);
    let mut filter_string = String::with_capacity(capacity);

    for (i, target) in targets.iter().enumerate() {
        if i > 0 {
            filter_string.push(',');
        }
        filter_string.push_str(target);
        filter_string.push('=');
        filter_string.push_str(level_filter);
    }

    filter_string
}

/// Build a filter string that sets all workspace crates to the given level
/// and defaults everything else to WARN.
///
/// Example output: `"warn,smg=info,tool_parser=info,kv_index=info,..."`
#[inline]
fn build_workspace_filter(level_filter: &str) -> String {
    // "warn," prefix + each crate entry
    let capacity = 5 + WORKSPACE_CRATES
        .iter()
        .map(|c| c.len() + 1 + level_filter.len() + 1) // "crate=level,"
        .sum::<usize>();
    let mut filter = String::with_capacity(capacity);
    filter.push_str("warn");
    for crate_name in WORKSPACE_CRATES {
        filter.push(',');
        filter.push_str(crate_name);
        filter.push('=');
        filter.push_str(level_filter);
    }
    filter
}

/// The global log filter layer: [`Targets`] when the directives allow it, else [`EnvFilter`].
type FilterLayer = Box<dyn Layer<Registry> + Send + Sync + 'static>;

/// `Targets` for `directives` (`target=level,...`) when it reads them exactly as
/// `EnvFilter` would, else `None`.
///
/// Callers validate the string with `EnvFilter` first, so this only has to rule
/// out the forms the two parsers read differently: `EnvFilter` drops empty
/// items where `Targets` makes an empty target at TRACE that matches everything;
/// span and field syntax (`target[span]=level`, `target[{field}]=level`) is a
/// literal target to `Targets`; and an empty level (`target=`) is TRACE to
/// `EnvFilter` but ERROR to `Targets`.
fn targets_for(directives: &str) -> Option<Targets> {
    let items: Vec<&str> = directives
        .split(',')
        .filter(|item| !item.is_empty())
        .collect();
    if items.is_empty()
        || items
            .iter()
            .any(|item| item.contains(['[', '{']) || item.ends_with('='))
    {
        return None;
    }
    items.join(",").parse::<Targets>().ok()
}

/// Parse `directives` into the cheapest filter that can express them.
///
/// `EnvFilter` decides whether the string is valid, exactly as before: a string
/// it rejects yields `None` and the caller falls back to the configured level.
/// When it accepts, `Targets` is preferred whenever it reads the string the same
/// way (see [`targets_for`]): it is a static lookup with no per-span state, while
/// `EnvFilter` keeps a `RwLock<HashMap<span::Id, _>>` that it reads on *every*
/// span enter and exit, even when no span-matching directive exists. An enabled
/// request span is entered once per polled response-body frame, so under
/// streaming load that single lock becomes a cross-core cache-line ping-pong
/// (61% of gateway CPU in `perf`, ~3.5x the CPU per streamed request).
fn build_filter_layer(directives: &str) -> Option<FilterLayer> {
    let env_filter = EnvFilter::try_new(directives).ok()?;
    if let Some(targets) = targets_for(directives) {
        return Some(Box::new(targets));
    }
    Some(Box::new(env_filter))
}

pub fn init_logging(config: LoggingConfig, otel_layer_config: Option<TraceConfig>) -> LogGuard {
    let level_filter = level_to_str(config.level);

    // RUST_LOG takes precedence (as `EnvFilter::try_from_default_env` did); an unset,
    // empty, or invalid value falls through to the configured level/targets.
    let filter_layer = std::env::var(EnvFilter::DEFAULT_ENV)
        .ok()
        .filter(|directives| !directives.trim().is_empty())
        .and_then(|directives| build_filter_layer(&directives))
        .unwrap_or_else(|| {
            let filter_string = match &config.log_targets {
                Some(targets) if !targets.is_empty() => build_filter_string(targets, level_filter),
                _ => {
                    // Default: external deps at WARN, all workspace crates at configured level.
                    // This ensures logs from imported crates (tool_parser, kv_index, etc.)
                    // are visible while suppressing noisy external deps (hyper, h2, tonic, etc.).
                    build_workspace_filter(level_filter)
                }
            };
            build_filter_layer(&filter_string)
                .unwrap_or_else(|| Box::new(EnvFilter::new(filter_string)))
        });

    // Cap the `log` facade at the filter's most verbose level. A bare
    // `LogTracer::init()` leaves `log::max_level()` at TRACE, so every
    // `log::trace!`/`debug!` in a dependency (the `tokenizers` normalizer and
    // pre-tokenizer emit them on each encode) builds a record and round-trips
    // through the tracing dispatcher only to be dropped there.
    let log_max_level = filter_layer
        .max_level_hint()
        .unwrap_or(LevelFilter::TRACE)
        .as_log();
    let _ = LogTracer::builder().with_max_level(log_max_level).init();

    let mut layers = Vec::with_capacity(3);

    // `std::io::stdout()` itself is a line-buffered writer that performs one
    // synchronous `write(2)` per event on the emitting thread; keep that
    // syscall, and whatever the kernel makes of it, off the runtime threads.
    let stdout_writer = install(io::stdout(), "smg-log-stdout");

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.colorize)
        .with_file(true)
        .with_line_number(true)
        .with_timer(UtcSeconds)
        .with_writer(stdout_writer);

    // The JSON lines are the stock format's, written by `log_format` (the
    // span's fields appended to, never re-parsed per record or per line).
    let stdout_layer = if config.json_format {
        stdout_layer
            .fmt_fields(SpanFields)
            .event_format(JsonLine::new())
            .boxed()
    } else {
        stdout_layer.boxed()
    };

    layers.push(stdout_layer);

    // A start through the CLI or the bindings validated this path before
    // anything bound (`ConfigValidator::validate_log_dir`, over the
    // `RouterConfig` field both fill from the same flag as this one): an
    // existing path is a writable directory, a missing one sits under a
    // writable directory and is created here. A caller that set only this
    // `ServerConfig` field, or bypassed the validation, keeps the console
    // logging on either failure here, reported on stderr, instead of a panic
    // in the appender or a start without any log output.
    if let Some(log_dir) = &config.log_dir {
        let log_dir = PathBuf::from(log_dir);

        let file_appender = std::fs::create_dir_all(&log_dir)
            .map_err(|e| format!("Failed to create log directory {}: {e}", log_dir.display()))
            .and_then(|()| {
                RollingFileAppender::builder()
                    .rotation(Rotation::DAILY)
                    .filename_prefix(&config.log_file_name)
                    .build(&log_dir)
                    .map_err(|e| {
                        format!("Failed to open the log file in {}: {e}", log_dir.display())
                    })
            });

        match file_appender {
            Ok(file_appender) => {
                let file_writer = install(file_appender, "smg-log-file");

                let file_layer = tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_file(true)
                    .with_line_number(true)
                    .with_timer(UtcSeconds)
                    .with_writer(file_writer);

                let file_layer = if config.json_format {
                    file_layer
                        .fmt_fields(SpanFields)
                        .event_format(JsonLine::new())
                        .boxed()
                } else {
                    file_layer.boxed()
                };

                layers.push(file_layer);
            }
            Err(message) => {
                // Logger is not yet initialized; stderr is the only output channel
                #[expect(clippy::print_stderr)]
                {
                    eprintln!("{message}");
                }
            }
        }
    }

    if let Some(otel_layer_config) = &otel_layer_config {
        if otel_layer_config.enable_trace {
            match get_otel_layer() {
                Ok(otel_layer) => {
                    layers.push(otel_layer);
                }
                Err(e) => {
                    // Logger may not be fully initialized yet; use stderr as fallback
                    #[expect(clippy::print_stderr)]
                    {
                        eprintln!("Failed to initialize OpenTelemetry: {e}");
                    }
                }
            }
        }
    }

    let _ = tracing_subscriber::registry()
        .with(filter_layer)
        .with(layers)
        .try_init();

    LogGuard(())
}

#[cfg(test)]
mod writer_tests {
    use std::{
        io,
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc, Mutex,
        },
        thread,
        time::{Duration, Instant},
    };

    use tracing_appender::non_blocking::DEFAULT_BUFFERED_LINES_LIMIT;

    use super::*;

    /// A sink whose first `write` blocks until released: a file write that the
    /// kernel is throttling, as seen from the thread that issued it.
    struct StalledSink {
        gate: Option<mpsc::Receiver<()>>,
    }

    impl Write for StalledSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if let Some(gate) = self.gate.take() {
                let _ = gate.recv();
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A sink that forwards every write on a channel, so a test can see
    /// whether a line reached the writer thread.
    struct ChannelSink(mpsc::Sender<Vec<u8>>);

    impl Write for ChannelSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let _ = self.0.send(buf.to_vec());
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A sink that keeps every write as it was given: a test reads the write
    /// boundaries as well as the bytes.
    #[derive(Clone, Default)]
    struct RecordingSink(Arc<Mutex<Vec<Vec<u8>>>>);

    impl RecordingSink {
        fn writes(&self) -> Vec<Vec<u8>> {
            self.0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }

        /// The writes once they hold `len` bytes, or after ten seconds.
        fn writes_of(&self, len: usize) -> Vec<Vec<u8>> {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let writes = self.writes();
                if writes.iter().map(Vec::len).sum::<usize>() >= len || Instant::now() > deadline {
                    return writes;
                }
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    impl Write for RecordingSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(buf.to_vec());
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn numbered_lines(count: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|i| format!("{i:06} finished processing request\n").into_bytes())
            .collect()
    }

    /// Every write ends at a line boundary, and the writes in their order are
    /// exactly `lines`: no line split, lost, repeated or reordered.
    fn assert_whole_lines_in_order(writes: &[Vec<u8>], lines: &[Vec<u8>]) {
        assert!(
            writes.iter().all(|write| write.ends_with(b"\n")),
            "every write ends at a line boundary"
        );
        let (written, given) = (writes.concat(), lines.concat());
        assert_eq!(written.len(), given.len(), "every byte given is written");
        assert!(written == given, "the lines are written as given, in order");
    }

    /// The writer threads belong to the process, not to the value
    /// `init_logging` returns: a router started after the first in one
    /// process logs through the same subscriber, whose writer must still be
    /// running once the first `startup` has returned and dropped its guard.
    #[test]
    fn the_writer_outlives_the_handle_init_returns() {
        let (tx, rx) = mpsc::channel();
        let mut writer = install(ChannelSink(tx), "test-log-outlives");
        {
            let _handle = LogGuard(());
        }
        writer
            .write_all(b"after the handle dropped\n")
            .expect("a lossy writer reports every line as written");
        let line = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the writer thread still runs after the handle dropped");
        assert_eq!(line, b"after the handle dropped\n");
    }

    /// Lines a full queue drops are counted per sink and reported as the
    /// total and the delta since the last report.
    #[test]
    fn dropped_lines_are_counted_and_reported_per_sink() {
        let (release, gate) = mpsc::channel();
        let mut writer = install(StalledSink { gate: Some(gate) }, "test-log-dropped");
        // More lines than the queue and the writer's batch hold together: the
        // batch takes `BATCH_BYTES / LINE.len() + 1` lines at most before its
        // write stalls on the sink.
        const LINE: &[u8] = b"finished processing request\n";
        for _ in 0..DEFAULT_BUFFERED_LINES_LIMIT + BATCH_BYTES / LINE.len() + 1_000 {
            writer
                .write_all(LINE)
                .expect("a lossy writer reports every line as written");
        }
        let counted = dropped_lines()
            .into_iter()
            .find(|(name, _)| *name == "test-log-dropped")
            .map(|(_, dropped)| dropped)
            .expect("the sink is counted");
        assert!(counted > 0, "overflow is counted");
        let (_, total, since_last) = report_dropped_lines()
            .into_iter()
            .find(|(name, _, _)| *name == "test-log-dropped")
            .expect("the sink is reported");
        assert_eq!(total, counted);
        assert_eq!(since_last, counted, "the first report carries everything");
        let (_, _, since_last) = report_dropped_lines()
            .into_iter()
            .find(|(name, _, _)| *name == "test-log-dropped")
            .expect("the sink is reported");
        assert_eq!(since_last, 0, "a second report carries only new drops");
        let _ = release.send(());
    }

    #[test]
    fn a_stalled_sink_never_blocks_the_emitting_thread() {
        let (release, gate) = mpsc::channel();
        let (mut writer, guard) =
            non_blocking(StalledSink { gate: Some(gate) }, "test-log-stalled");
        let (done, finished) = mpsc::channel();
        thread::spawn(move || {
            // More lines than the channel holds, while the sink accepts none.
            for _ in 0..DEFAULT_BUFFERED_LINES_LIMIT + 1_000 {
                writer
                    .write_all(b"finished processing request\n")
                    .expect("a lossy writer reports every line as written");
            }
            let _ = done.send(writer.error_counter().dropped_lines());
        });
        let dropped = finished.recv_timeout(Duration::from_secs(10));
        // Unblock the writer thread whatever happened, so a failure does not leak it.
        let _ = release.send(());
        let dropped = dropped.expect("writes against a stalled sink must return, not wait for it");
        assert!(
            dropped > 0,
            "the queue is bounded: overflow is dropped, never waited for"
        );
        drop(guard);
    }

    /// Lines that arrive within the window leave in one write, whole and in
    /// order: the write and the wake-up per line are gone.
    #[test]
    fn lines_within_the_window_leave_in_one_write() {
        let sink = RecordingSink::default();
        let (mut writer, guard) = non_blocking(
            Batched::with(sink.clone(), Duration::from_millis(50), BATCH_BYTES),
            "test-log-batched-window",
        );
        let lines = numbered_lines(1_000);
        for line in &lines {
            writer
                .write_all(line)
                .expect("a lossy writer reports every line as written");
        }
        assert_eq!(writer.error_counter().dropped_lines(), 0);
        drop(guard);
        let writes = sink.writes_of(lines.concat().len());
        assert_whole_lines_in_order(&writes, &lines);
        assert!(
            writes.len() <= lines.len() / 100,
            "{} writes for {} lines",
            writes.len(),
            lines.len()
        );
    }

    /// A line with no company still leaves, once the window has passed.
    #[test]
    fn a_lone_line_leaves_after_the_window() {
        let sink = RecordingSink::default();
        let (mut writer, guard) = non_blocking(
            Batched::with(sink.clone(), Duration::from_millis(5), BATCH_BYTES),
            "test-log-batched-lone",
        );
        let line = b"a line on its own\n".to_vec();
        writer
            .write_all(&line)
            .expect("a lossy writer reports every line as written");
        assert_eq!(sink.writes_of(line.len()), vec![line]);
        drop(guard);
    }

    /// A batch that reaches the byte cap is written inside the window, and the
    /// rest of the backlog follows without another wait.
    #[test]
    fn the_byte_cap_writes_a_batch_inside_the_window() {
        let cap = 1_024;
        let sink = RecordingSink::default();
        let (mut writer, guard) = non_blocking(
            Batched::with(sink.clone(), Duration::from_millis(200), cap),
            "test-log-batched-cap",
        );
        // 36 KB in all: some 35 batches of the cap.
        let lines = numbered_lines(1_000);
        let started = Instant::now();
        for line in &lines {
            writer
                .write_all(line)
                .expect("a lossy writer reports every line as written");
        }
        drop(guard);
        let writes = sink.writes_of(lines.concat().len());
        let drained_in = started.elapsed();
        assert_whole_lines_in_order(&writes, &lines);
        let longest = writes.iter().map(Vec::len).max().unwrap_or(0);
        assert!(
            longest < cap + lines[0].len(),
            "a batch is written once it reaches the cap, not later: {longest} bytes"
        );
        assert!(writes.len() >= 30, "{} writes", writes.len());
        assert!(
            drained_in < Duration::from_secs(4),
            "the backlog drained without a window per batch: {drained_in:?}"
        );
    }

    /// The flush at shutdown writes the lines still in the window: a clean
    /// exit loses nothing.
    #[test]
    fn shutdown_writes_the_lines_still_in_the_window() {
        let sink = RecordingSink::default();
        let (mut writer, guard) = non_blocking(
            Batched::with(sink.clone(), Duration::from_millis(20), BATCH_BYTES),
            "test-log-batched-shutdown",
        );
        let lines = numbered_lines(10);
        for line in &lines {
            writer
                .write_all(line)
                .expect("a lossy writer reports every line as written");
        }
        // The writer is inside its window; the guard's drop waits for the flush.
        drop(guard);
        assert_whole_lines_in_order(&sink.writes(), &lines);
    }

    /// A sink that fails its writes (a closed stdout pipe, a full disk) does
    /// not break the shutdown handshake: the writer thread takes the shutdown
    /// message, meets the guard and ends, instead of blocking on its queue for
    /// good while the guard waits out its rendezvous timeout; the batch is
    /// gone, as a line whose own write failed was before.
    #[test]
    fn a_failing_sink_does_not_break_the_shutdown_handshake() {
        /// Refuses every write; sets `ended` when it is dropped, which is when
        /// the writer thread that owns it has ended.
        struct FailingSink {
            ended: Arc<AtomicBool>,
        }

        impl Write for FailingSink {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        impl Drop for FailingSink {
            fn drop(&mut self) {
                self.ended.store(true, Ordering::SeqCst);
            }
        }

        let ended = Arc::new(AtomicBool::new(false));
        let sink = FailingSink {
            ended: ended.clone(),
        };
        // A window: the writer thread sleeps inside its first `write` while
        // the guard's drop queues the shutdown message, so the flush that
        // fails is the one that ends the shutdown drain, every time (with no
        // window the worker may drain the line alone, fail its flush before
        // the shutdown message is queued, and pass against the old flush).
        let (mut writer, guard) = non_blocking(
            Batched::with(sink, Duration::from_millis(50), BATCH_BYTES),
            "test-log-batched-failing",
        );
        writer
            .write_all(b"a line the sink refuses\n")
            .expect("a lossy writer reports every line as written");
        let started = Instant::now();
        drop(guard);
        let dropped_in = started.elapsed();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !ended.load(Ordering::SeqCst) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert!(
            ended.load(Ordering::SeqCst),
            "the writer thread ended after the guard dropped (the drop took {dropped_in:?})"
        );
    }

    /// Writes per line on a steady stream, with the batching and without: a
    /// measurement (`--ignored --nocapture`), not a check.
    #[test]
    #[ignore = "a timing, not a check"]
    #[expect(clippy::print_stdout, reason = "the measurement is the output")]
    fn writes_per_line_on_a_steady_stream() {
        const LINES: usize = 20_000;
        const INTERVAL: Duration = Duration::from_micros(250);
        fn stream<W: Write + Send + 'static>(
            sink: W,
            recording: &RecordingSink,
            name: &str,
        ) -> (usize, Duration) {
            let (mut writer, guard) = non_blocking(sink, name);
            let lines = numbered_lines(LINES);
            let started = Instant::now();
            let mut due = started;
            for line in &lines {
                due += INTERVAL;
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    thread::sleep(wait);
                }
                writer
                    .write_all(line)
                    .expect("a lossy writer reports every line as written");
            }
            let elapsed = started.elapsed();
            drop(guard);
            let writes = recording.writes_of(lines.concat().len());
            assert_whole_lines_in_order(&writes, &lines);
            (writes.len(), elapsed)
        }
        let plain = RecordingSink::default();
        let (writes, elapsed) = stream(plain.clone(), &plain, "test-log-stream-plain");
        println!(
            "plain: {LINES} lines in {elapsed:?} ({:.0} lines/s), {writes} writes, \
             {:.3} writes per line",
            LINES as f64 / elapsed.as_secs_f64(),
            writes as f64 / LINES as f64
        );
        let batched = RecordingSink::default();
        let (writes, elapsed) = stream(
            Batched::new(batched.clone()),
            &batched,
            "test-log-stream-batched",
        );
        println!(
            "batched: {LINES} lines in {elapsed:?} ({:.0} lines/s), {writes} writes, \
             {:.3} writes per line",
            LINES as f64 / elapsed.as_secs_f64(),
            writes as f64 / LINES as f64
        );
    }
}

#[cfg(test)]
mod filter_tests {
    use tracing::Level;

    use super::*;

    fn targets(directives: &str) -> Targets {
        targets_for(directives).unwrap_or_else(|| panic!("{directives:?} must use Targets"))
    }

    fn hint(directives: &str) -> Option<LevelFilter> {
        build_filter_layer(directives).and_then(|layer| layer.max_level_hint())
    }

    #[test]
    fn plain_target_levels_use_targets() {
        let targets = targets("warn,smg=info");
        assert!(targets.would_enable("smg", &Level::INFO));
        assert!(!targets.would_enable("smg", &Level::DEBUG));
        assert!(targets.would_enable("hyper", &Level::WARN));
        assert!(!targets.would_enable("hyper", &Level::INFO));
        assert_eq!(targets.default_level(), Some(LevelFilter::WARN));
    }

    #[test]
    fn env_filter_decides_validity() {
        // Whatever `EnvFilter` rejected before this change is still rejected,
        // whether or not `Targets` would have taken it (`*smg=debug`, ` smg=debug`).
        for directives in [
            "warn,smg=info",
            "warn,",
            "warn, smg=debug",
            "*smg=debug",
            " , ",
            "info",
            "smg[request]=debug",
            "warn,smg=",
        ] {
            assert_eq!(
                build_filter_layer(directives).is_some(),
                EnvFilter::try_new(directives).is_ok(),
                "{directives:?}"
            );
        }
        assert!(targets_for("*smg=debug").is_none() || build_filter_layer("*smg=debug").is_none());
    }

    #[test]
    fn forms_the_parsers_read_differently_stay_with_env_filter() {
        for directives in ["smg[request]=debug", "warn,smg[{model}]=trace", "warn,smg="] {
            assert!(
                targets_for(directives).is_none(),
                "{directives:?} must not be parsed as Targets"
            );
            assert!(
                build_filter_layer(directives).is_some(),
                "{directives:?} must still produce an EnvFilter"
            );
        }
        // `EnvFilter` reads an empty level as TRACE; `Targets` would have read ERROR.
        assert_eq!(hint("warn,smg="), Some(LevelFilter::TRACE));
    }

    #[test]
    fn a_trailing_comma_is_not_an_empty_target() {
        let trailing = targets("warn,");
        assert_eq!(trailing.default_level(), Some(LevelFilter::WARN));
        assert_eq!(
            trailing.iter().count(),
            0,
            "no per-target directive, let alone an empty one"
        );
        assert!(!trailing.would_enable("hyper", &Level::INFO));
    }

    #[test]
    fn log_cap_follows_the_most_verbose_directive() {
        assert_eq!(hint("warn,smg=info"), Some(LevelFilter::INFO));
        assert_eq!(hint("warn,smg=debug,"), Some(LevelFilter::DEBUG));
        assert_eq!(hint("error"), Some(LevelFilter::ERROR));
    }
}
