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
    "smg_service_discovery",
    "smg_wasm",
    "smg_worker",
    "tool_parser",
    "wfaas",
];

/// The directive of the default filter that turns the audio decoder's probe
/// target off: the probe reports a clip it cannot read (`probe reached EOF at
/// N bytes`, `reached probe limit of N bytes`) at ERROR through the library's
/// own target, for bytes a client sent; the stage that refuses the clip
/// writes the refusal's line, with the cause.
const AUDIO_PROBE_OFF: &str = ",symphonia_core::formats::probe=off";

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
/// and defaults everything else to WARN, the audio decoder's probe aside
/// ([`AUDIO_PROBE_OFF`]).
///
/// Example output: `"warn,symphonia_core::formats::probe=off,smg=info,tool_parser=info,..."`
#[inline]
fn build_workspace_filter(level_filter: &str) -> String {
    // "warn," prefix + the probe's directive + each crate entry
    let capacity = 5
        + AUDIO_PROBE_OFF.len()
        + WORKSPACE_CRATES
            .iter()
            .map(|c| c.len() + 1 + level_filter.len() + 1) // "crate=level,"
            .sum::<usize>();
    let mut filter = String::with_capacity(capacity);
    filter.push_str("warn");
    filter.push_str(AUDIO_PROBE_OFF);
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
                #[expect(
                    clippy::print_stderr,
                    reason = "the logger is not up yet; stderr is its only channel"
                )]
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
                #[expect(
                    clippy::print_stderr,
                    reason = "the logger may not be up yet; stderr is the fallback"
                )]
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
#[path = "../../tests/unit/observability/logging/test_support.rs"]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "../../tests/unit/observability/logging/writer_tests.rs"]
mod writer_tests;

#[cfg(test)]
#[path = "../../tests/unit/observability/logging/filter_tests.rs"]
mod filter_tests;

#[cfg(test)]
#[path = "../../tests/unit/observability/logging/color_tests.rs"]
mod color_tests;
