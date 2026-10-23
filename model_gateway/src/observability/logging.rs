//! Logging infrastructure with non-blocking I/O for every sink.
//!
//! Each formatting layer hands its lines to a dedicated writer thread through
//! a bounded, lossy channel ([`tracing_appender::non_blocking`](mod@tracing_appender::non_blocking)), so the tokio
//! runtime threads that emit the events never wait for the sink. That matters
//! for stdout as much as for the log file: when stdout is redirected to a
//! regular file, every `write(2)` is subject to the kernel's dirty-page
//! throttling and can sleep for hundreds of milliseconds under writeback
//! pressure, which with a synchronous `std::io::stdout()` writer parked a
//! runtime worker, and every connection it was driving, once per access-log
//! line.
//!
//! The writer threads write in batches (`Batched`): the first line after a
//! quiet moment opens a short window, and the lines queued within it leave
//! in one write. Nothing but the writer thread waits for the window; a line
//! is visible a few milliseconds later at most.

use std::{
    ffi::OsStr,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
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
    registry::LookupSpan,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};

use super::{
    log_format::{JsonLine, SpanFields, TextFields, TextLine, UtcSeconds},
    otel_trace::get_otel_layer_at,
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
    /// ANSI color on the console lines; [`color_for_stdout`] is the default
    /// (a terminal yes, a pipe or a file no, `NO_COLOR` and `FORCE_COLOR` /
    /// `CLICOLOR_FORCE` respected). The file sink never colors.
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
            colorize: color_for_stdout(),
            log_file_name: "smg".to_string(),
            log_targets: None, // None = use workspace crate filter
        }
    }
}

/// Whether the console log is colored when nothing asked either way: ANSI
/// escapes go to a terminal, not into a pipe or a file (a redirected start,
/// a container runtime collecting stdout), where they are bytes on every line
/// that nothing renders and a consumer has to strip. `NO_COLOR` set to a
/// non-empty value turns color off on a terminal too; `FORCE_COLOR` or
/// `CLICOLOR_FORCE` set to a non-empty value other than `0` turns it on into
/// a pipe too (the no-color.org and clicolors conventions, where `0` means
/// "do not force"; an explicit ask wins).
pub fn color_for_stdout() -> bool {
    color_for(
        io::stdout().is_terminal(),
        set(std::env::var_os("NO_COLOR").as_deref()),
        truthy(std::env::var_os("FORCE_COLOR").as_deref())
            || truthy(std::env::var_os("CLICOLOR_FORCE").as_deref()),
    )
}

/// The decision behind [`color_for_stdout`]: forced wins over refused, and
/// refused over the terminal.
fn color_for(terminal: bool, refused: bool, forced: bool) -> bool {
    forced || (terminal && !refused)
}

/// Whether an environment variable's value counts as set: present and
/// non-empty (an empty value counts as unset, as the conventions say).
fn set(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty())
}

/// Whether an environment variable's value forces: present, non-empty and
/// not `0`, the clicolors reading of a force variable, where `0` means "do
/// not force" (and what the common `FORCE_COLOR=0` of other tools means).
fn truthy(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

/// What `init_logging` hands back. The writer threads live in `SINKS` for
/// the process, not in this value: the subscriber is installed once per
/// process and every router started after the first (the Python binding
/// starts them in one process) logs through the same writers, so a guard
/// that died with the first `startup` call would leave every later line
/// queued behind a thread that had exited. Dropping this stops nothing;
/// [`close_logging`] flushes and stops the writers at process exit.
pub struct LogGuard(());

/// One writer thread: its handle, the count of lines its lossy queue
/// dropped, and the signal its end raises.
struct Sink {
    name: &'static str,
    counter: ErrorCounter,
    /// The count at the last report, for the delta.
    reported: usize,
    /// Raised when the thread has dropped its `Batched`, its last act.
    ended: Arc<Ended>,
    _guard: WorkerGuard,
}

/// Raised by a writer thread's [`Batched`] when it is dropped, which is the
/// thread's last act after its final flush: whoever waits on it knows the
/// thread has written everything it will ever write.
#[derive(Default)]
struct Ended {
    ended: Mutex<bool>,
    changed: Condvar,
}

impl Ended {
    fn set(&self) {
        *self
            .ended
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        self.changed.notify_all();
    }

    /// Wait until set, or until `deadline`; whether it was set.
    fn wait_until(&self, deadline: Instant) -> bool {
        let mut ended = self
            .ended
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while !*ended {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            ended = self
                .changed
                .wait_timeout(ended, left)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
        *ended
    }
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

/// How long [`close_logging`] waits for the writer threads past the moment
/// each guard's drop gives them (about a second): a thread held in a write
/// the kernel is throttling, or left unscheduled on a loaded host, still
/// finishes its last batch within it. An idle writer ends inside the
/// guard's handshake, and the wait is over the instant the last thread has.
const CLOSE_WAIT: Duration = Duration::from_secs(5);

/// How long the close waits for its own stderr line about a writer that
/// missed [`CLOSE_WAIT`]: long enough for a terminal or a file, short enough
/// that a stderr sharing a stuck pipe does not hold the exit.
const STDERR_WAIT: Duration = Duration::from_millis(100);

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
/// nothing. The writer thread owns the adapter and drops it as its last act,
/// which raises `ended`: [`close_logging`] waits on it.
struct Batched<W> {
    sink: W,
    buf: Vec<u8>,
    window: Duration,
    cap: usize,
    /// A window is open: the next line does not wait for one again.
    open: bool,
    ended: Arc<Ended>,
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
            ended: Arc::default(),
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

impl<W> Drop for Batched<W> {
    fn drop(&mut self) {
        self.ended.set();
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
    let (writer, sink) = start_sink(Batched::new(sink), name);
    let mut sinks = SINKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sinks.sinks.push(sink);
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
    if !close_sinks(sinks) {
        // The writers are closed: stderr is the one channel left. Written off
        // this thread with a short wait: a stderr that shares the stuck
        // stdout pipe (`2>&1`, a journal stream) would block this write too,
        // and the exit behind it; a write error (a reader gone) is not a
        // panic on the way out.
        let (written, wait) = mpsc::channel();
        // `Builder::spawn`, as the reporter thread: a spawn the OS refuses
        // (a pids limit, memory pressure) drops the closure and `written`
        // with it, the wait ends at once and the close goes on without the
        // line, instead of the panic `thread::spawn` would raise here.
        let _ = std::thread::Builder::new()
            .name("smg-log-close".to_string())
            .spawn(move || {
                let _ = writeln!(
                    io::stderr(),
                    "[logging] a log writer was still writing {CLOSE_WAIT:?} after the close; \
                     its last lines may be lost"
                );
                let _ = written.send(());
            });
        let _ = wait.recv_timeout(STDERR_WAIT);
    }
}

/// Stop the writer threads of `sinks` and wait for them. Each guard's drop
/// queues the shutdown message and waits a bounded moment for the flush
/// that answers it; a thread that misses the moment (its write held by the
/// kernel, or itself unscheduled on a loaded host) goes on detached and
/// writes after the drop has returned. So the threads themselves are waited
/// for, up to [`CLOSE_WAIT`] in all, and the lines logged before the close
/// are in their files when this returns. Whether every thread ended in time.
fn close_sinks(sinks: Vec<Sink>) -> bool {
    let ended: Vec<Arc<Ended>> = sinks.iter().map(|sink| Arc::clone(&sink.ended)).collect();
    drop(sinks);
    let deadline = Instant::now() + CLOSE_WAIT;
    ended.iter().all(|ended| ended.wait_until(deadline))
}

/// `batched` on its own writer thread behind a bounded, lossy channel, and
/// the record of that thread: its guard, its dropped-line counter and the
/// signal its end raises, for [`close_sinks`].
fn start_sink<W: Write + Send + 'static>(
    batched: Batched<W>,
    name: &'static str,
) -> (NonBlocking, Sink) {
    let ended = Arc::clone(&batched.ended);
    let (writer, guard) = non_blocking(batched, name);
    let sink = Sink {
        name,
        counter: writer.error_counter(),
        reported: 0,
        ended,
        _guard: guard,
    };
    (writer, sink)
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
    let directives = std::env::var(EnvFilter::DEFAULT_ENV)
        .ok()
        .filter(|directives| {
            !directives.trim().is_empty() && EnvFilter::try_new(directives).is_ok()
        })
        .unwrap_or_else(|| match &config.log_targets {
            Some(targets) if !targets.is_empty() => build_filter_string(targets, level_filter),
            _ => {
                // Default: external deps at WARN, all workspace crates at configured level.
                // This ensures logs from imported crates (tool_parser, kv_index, etc.)
                // are visible while suppressing noisy external deps (hyper, h2, tonic, etc.).
                build_workspace_filter(level_filter)
            }
        });
    let filter_layer =
        build_filter_layer(&directives).unwrap_or_else(|| Box::new(EnvFilter::new(&directives)));

    // Cap the `log` facade at the filter's most verbose level. A bare
    // `LogTracer::init()` leaves `log::max_level()` at TRACE, so every
    // `log::trace!`/`debug!` in a dependency (the `tokenizers` normalizer and
    // pre-tokenizer emit them on each encode) builds a record and round-trips
    // through the tracing dispatcher only to be dropped there.
    let log_level = filter_layer.max_level_hint().unwrap_or(LevelFilter::TRACE);
    let _ = LogTracer::builder()
        .with_max_level(log_level.as_log())
        .init();

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

    // Both line formats are the stock formats' bytes, written by `log_format`
    // (the span's fields appended to, never re-parsed per record or per
    // line; the text line without a styled `Display` per piece).
    let stdout_layer = if config.json_format {
        stdout_layer
            .fmt_fields(SpanFields)
            .event_format(JsonLine::new())
            .boxed()
    } else {
        stdout_layer
            .fmt_fields(TextFields)
            .event_format(TextLine::new())
            .boxed()
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
                    file_layer
                        .fmt_fields(TextFields)
                        .event_format(TextLine::new())
                        .boxed()
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

    let otel_layer: Option<FilterLayer> = otel_layer_config
        .filter(|otel_layer_config| otel_layer_config.enable_trace)
        .and_then(|_| match get_otel_layer_at(log_level) {
            Ok(otel_layer) => Some(otel_layer),
            Err(e) => {
                // Logger may not be fully initialized yet; use stderr as fallback
                #[expect(clippy::print_stderr)]
                {
                    eprintln!("Failed to initialize OpenTelemetry: {e}");
                }
                None
            }
        });

    let _ = match otel_layer {
        // The log filter is the log layers' own, and the trace layer sits behind its own: the
        // tracer's spans export at every log level (one global filter gated them too, and at
        // `--log-level warn`/`error` `--enable-trace` exported nothing).
        Some(otel_layer) => tracing_subscriber::registry()
            .with(otel_layer)
            .with(filtered_by(layers, &directives))
            .try_init(),
        // Without tracing the one global filter, as before: nothing is filtered per layer.
        None => tracing_subscriber::registry()
            .with(filter_layer)
            .with(layers)
            .try_init(),
    };

    LogGuard(())
}

/// The log layers behind the log filter as their own: one filter over the group, read as the
/// global one is (`Targets` when `EnvFilter` accepts the directives and they read the same both
/// ways, else `EnvFilter`, lenient on a string it rejects, as [`init_logging`] falls back).
fn filtered_by<S>(
    layers: Vec<Box<dyn Layer<S> + Send + Sync + 'static>>,
    directives: &str,
) -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a> + Send + Sync + 'static,
{
    match EnvFilter::try_new(directives)
        .ok()
        .and_then(|_| targets_for(directives))
    {
        Some(targets) => Box::new(layers.with_filter(targets)),
        None => Box::new(layers.with_filter(EnvFilter::new(directives))),
    }
}

/// Logging helpers for tests elsewhere in the crate.
#[cfg(test)]
pub(crate) mod test_support {
    use tracing::{subscriber::DefaultGuard, Dispatch};
    use tracing_subscriber::{layer::SubscriberExt, Layer, Registry};

    /// A test's subscriber: this thread's default until the value drops,
    /// seeing every line the thread writes meanwhile.
    ///
    /// `tracing` caches per callsite whether any subscriber wants it, and
    /// while it knows of a single live dispatcher it rebuilds the entry of a
    /// callsite registered for the first time from the registering thread's
    /// default alone. A test running beside this one with no subscriber of
    /// its own, writing a line at such a callsite first, then caches it as
    /// never enabled: the macro skips the line on every thread, this one
    /// included, before any subscriber is asked, until the next dispatcher
    /// registers. The second dispatcher held here keeps every rebuild over
    /// all live dispatchers, the test's among them.
    #[must_use = "dropping the value uninstalls the subscriber"]
    pub(crate) struct TestSubscriber {
        _guard: DefaultGuard,
        _second: Dispatch,
    }

    /// `layer` over a registry as this thread's subscriber until the value
    /// drops.
    pub(crate) fn test_subscriber<L>(layer: L) -> TestSubscriber
    where
        L: Layer<Registry> + Send + Sync + 'static,
    {
        let second = Dispatch::new(Registry::default());
        let guard = tracing::subscriber::set_default(Registry::default().with(layer));
        TestSubscriber {
            _guard: guard,
            _second: second,
        }
    }
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

    /// A sink whose first write takes `delay`: a file write the kernel holds
    /// for longer than the guard's moment.
    struct SlowFirstWrite {
        delay: Option<Duration>,
        sink: RecordingSink,
    }

    impl Write for SlowFirstWrite {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if let Some(delay) = self.delay.take() {
                thread::sleep(delay);
            }
            self.sink.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// The close waits for a writer whose last write outlasts the guard's
    /// moment (about a second): the line logged before the close is in the
    /// sink when the close returns, not written after it.
    #[test]
    fn the_close_waits_for_a_writer_held_past_the_guards_moment() {
        let recording = RecordingSink::default();
        let (mut writer, sink) = start_sink(
            Batched::with(
                SlowFirstWrite {
                    delay: Some(Duration::from_millis(1_500)),
                    sink: recording.clone(),
                },
                Duration::ZERO,
                BATCH_BYTES,
            ),
            "test-log-close-waits",
        );
        let line = b"a line logged before the close\n".to_vec();
        writer
            .write_all(&line)
            .expect("a lossy writer reports every line as written");
        // The writer thread is inside its slow write when the close begins.
        thread::sleep(Duration::from_millis(100));
        let started = Instant::now();
        let all_ended = close_sinks(vec![sink]);
        let took = started.elapsed();
        assert!(
            all_ended,
            "the writer thread ended within the close's wait (the close took {took:?})"
        );
        assert_eq!(
            recording.writes(),
            vec![line],
            "the line is in the sink when the close returns"
        );
    }

    /// A writer with nothing left to write ends inside the guard's handshake:
    /// the close costs no wait after it. The bound is half of [`CLOSE_WAIT`]:
    /// a close that waited the whole of it for a writer already ended would
    /// fail, while a guard handshake that runs out its own second on a busy
    /// runner (the idle writer thread scheduled late) would not. The time is
    /// the output, for the record (`--nocapture`).
    #[test]
    #[expect(clippy::print_stdout, reason = "the measurement is the output")]
    fn the_close_returns_as_soon_as_an_idle_writer_has_ended() {
        let recording = RecordingSink::default();
        let (mut writer, sink) = start_sink(
            Batched::with(recording.clone(), Duration::ZERO, BATCH_BYTES),
            "test-log-close-idle",
        );
        let line = b"a line written long before the close\n".to_vec();
        writer
            .write_all(&line)
            .expect("a lossy writer reports every line as written");
        assert_eq!(recording.writes_of(line.len()), vec![line]);
        let started = Instant::now();
        assert!(close_sinks(vec![sink]), "an idle writer ends in time");
        let took = started.elapsed();
        println!("idle close took {took:?}");
        assert!(
            took < CLOSE_WAIT / 2,
            "an idle writer ends inside the guard's handshake, with no wait after it: {took:?}"
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

#[cfg(test)]
mod color_tests {
    use std::ffi::OsStr;

    use super::{color_for, set, truthy};

    #[test]
    fn a_terminal_is_colored_unless_refused_and_a_pipe_only_when_forced() {
        // (terminal, refused, forced)
        assert!(color_for(true, false, false));
        assert!(!color_for(true, true, false));
        assert!(color_for(true, true, true));
        assert!(!color_for(false, false, false));
        assert!(!color_for(false, true, false));
        assert!(color_for(false, false, true));
        assert!(color_for(false, true, true));
    }

    /// A force variable set to `0` does not force (clicolors); `NO_COLOR` set
    /// to anything non-empty, `0` included, refuses (no-color.org); an empty
    /// value is unset for both. The readers are pure: the process environment
    /// is not touched (the lib tests run in parallel and read it).
    #[test]
    fn a_force_variable_set_to_zero_does_not_force_and_an_empty_one_is_unset() {
        let zero = Some(OsStr::new("0"));
        let empty = Some(OsStr::new(""));
        let one = Some(OsStr::new("1"));
        assert!(set(zero));
        assert!(!truthy(zero));
        assert!(!set(empty));
        assert!(!truthy(empty));
        assert!(set(one));
        assert!(truthy(one));
        assert!(!set(None));
        assert!(!truthy(None));
    }
}
