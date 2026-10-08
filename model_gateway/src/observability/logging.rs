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

use std::{
    io::Write,
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
    fmt::time::ChronoUtc,
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};

use super::otel_trace::get_otel_layer;
use crate::{config::TraceConfig, observability::metrics::Metrics};

const TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

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

/// Move `sink` to its own writer thread held for the process, and return
/// the writer for a subscriber layer. The thread is counted among the
/// sinks whose dropped lines the reporter publishes.
fn install<W: Write + Send + 'static>(sink: W, name: &'static str) -> NonBlocking {
    let (writer, guard) = non_blocking(sink, name);
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

/// Flush and stop the writer threads: the last lines queued are written
/// before the process exits. Called once, at the end of `main`; nothing logs
/// through the stopped writers afterwards.
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
    let stdout_writer = install(std::io::stdout(), "smg-log-stdout");

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.colorize)
        .with_file(true)
        .with_line_number(true)
        .with_timer(ChronoUtc::new(TIME_FORMAT.to_string()))
        .with_writer(stdout_writer);

    let stdout_layer = if config.json_format {
        stdout_layer.json().flatten_event(true).boxed()
    } else {
        stdout_layer.boxed()
    };

    layers.push(stdout_layer);

    if let Some(log_dir) = &config.log_dir {
        let log_dir = PathBuf::from(log_dir);

        if !log_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                // Logger is not yet initialized; stderr is the only output channel
                #[expect(clippy::print_stderr)]
                {
                    eprintln!("Failed to create log directory: {e}");
                }
                return LogGuard(());
            }
        }

        let file_appender =
            RollingFileAppender::new(Rotation::DAILY, log_dir, &config.log_file_name);

        let file_writer = install(file_appender, "smg-log-file");

        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_file(true)
            .with_line_number(true)
            .with_timer(ChronoUtc::new(TIME_FORMAT.to_string()))
            .with_writer(file_writer);

        let file_layer = if config.json_format {
            file_layer.json().flatten_event(true).boxed()
        } else {
            file_layer.boxed()
        };

        layers.push(file_layer);
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
    use std::{io, sync::mpsc, thread, time::Duration};

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
        for _ in 0..DEFAULT_BUFFERED_LINES_LIMIT + 1_000 {
            writer
                .write_all(b"finished processing request\n")
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
