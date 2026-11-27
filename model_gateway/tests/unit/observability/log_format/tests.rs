use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use tracing::{info, info_span, warn};
use tracing_subscriber::{
    fmt::{time::ChronoUtc, MakeWriter},
    layer::SubscriberExt,
    Layer, Registry,
};

use super::*;

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).expect("utf-8 lines")
    }
}

impl io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Capture {
    type Writer = Capture;

    fn make_writer(&'a self) -> Capture {
        self.clone()
    }
}

struct FixedTime;

impl FormatTime for FixedTime {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        w.write_str("2026-01-02 03:04:05")
    }
}

#[derive(Debug)]
struct Shape {
    #[expect(dead_code, reason = "read through its Debug form alone")]
    side: u8,
}

/// Every field kind the gateway's spans and events use, from one set of
/// callsites, so two formats see identical metadata; a record bridged
/// from the `log` facade among the events.
fn scenario() {
    let request = info_span!(
        target: "smg::otel-trace",
        "http_request",
        method = "POST",
        path = "/v1/chat/completions",
        version = ?Shape { side: 1 },
        request_id = tracing::field::Empty,
        status_code = tracing::field::Empty,
        latency = tracing::field::Empty,
        module = "smg"
    );
    let entered = request.enter();
    info!(target: "smg::request", "started processing request");
    request.record("request_id", "chatcmpl-a\"b\\c\nd\u{1f}é");
    {
        let upstream = info_span!(
            "grpc_execute",
            request_type = "chat",
            model = %"m1",
            mode = ?Some(2u8),
            r#type = "raw",
            ratio = 0.25f64,
            negative = -4i64,
            flag = true,
            empty = tracing::field::Empty
        );
        let _upstream = upstream.enter();
        warn!(
            count = 2u64,
            text = "t\"q",
            flag = false,
            ratio = 1.5f64,
            negative = -1i64,
            shape = ?Shape { side: 3 },
            error = %"an error",
            "mixed {} fields",
            42
        );
        info!(parent: &request, "explicit parent while another span is current");
        upstream.record("empty", 7u64);
        info!("the recorded field is on the span now");
        // A record bridged from the `log` facade: its own target, file
        // and line, the `log.*` fields flattened into the line.
        let _ = tracing_log::format_trace(
            &tracing_log::log::Record::builder()
                .level(tracing_log::log::Level::Warn)
                .target("hyper::proto")
                .file(Some("src/proto.rs"))
                .line(Some(7))
                .args(format_args!("bridged {}", 1))
                .build(),
        );
    }
    request.record("status_code", 200u64);
    request.record("latency", 1234u64);
    request.record("status_code", 201u64);
    info!(target: "smg::response", "finished processing request");
    drop(entered);
    info!("outside any span");
}

fn stock_lines() -> String {
    let capture = Capture::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_file(true)
        .with_line_number(true)
        .with_timer(FixedTime)
        .with_writer(capture.clone())
        .json()
        .flatten_event(true);
    tracing::subscriber::with_default(Registry::default().with(layer), scenario);
    capture.text()
}

fn fast_lines() -> String {
    let capture = Capture::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(capture.clone())
        .fmt_fields(SpanFields)
        .event_format(JsonLine::with_timer(FixedTime));
    tracing::subscriber::with_default(Registry::default().with(layer), scenario);
    capture.text()
}

#[test]
fn the_lines_are_those_of_the_stock_json_format() {
    let stock = stock_lines();
    let fast = fast_lines();
    assert_eq!(stock.lines().count(), 7, "{stock}");
    for (stock_line, fast_line) in stock.lines().zip(fast.lines()) {
        assert_eq!(fast_line, stock_line);
    }
    assert_eq!(fast, stock);
}

#[test]
fn a_field_recorded_again_keeps_its_last_value_once() {
    let fast = fast_lines();
    let finished = fast
        .lines()
        .find(|line| line.contains("finished processing request"))
        .expect("the response line");
    assert!(finished.contains("\"status_code\":201"), "{finished}");
    assert!(!finished.contains("\"status_code\":200"), "{finished}");
    assert_eq!(
        finished.matches("\"status_code\"").count(),
        2,
        "span and spans: {finished}"
    );
}

#[test]
fn the_timestamp_is_chronos_pattern() {
    for seconds in [
        0i64,
        946_684_800,
        1_700_000_000,
        1_760_000_000,
        4_102_444_799,
    ] {
        let time = DateTime::<Utc>::from_timestamp(seconds, 0).expect("a valid time");
        let mut written = String::new();
        write_utc_seconds(&mut written, time).expect("writes");
        assert_eq!(written, time.format(TIME_FORMAT).to_string());
    }
}

/// The text the formatting machinery gives for `time`: the writer the
/// digit writer replaced, kept to be compared against.
fn through_the_formatter(time: DateTime<Utc>) -> String {
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.minute(),
        time.second()
    )
}

#[test]
fn the_timestamp_is_the_formatters_text_at_every_instant() {
    // Seconds since 1970 and the sub-second part: the turn of a day, of
    // a year and of a century, a leap day, the last second of a month,
    // instants before 1970 and before the year 1, the first year of
    // five digits and a leap second.
    let mut instants: Vec<(i64, u32)> = vec![
        (0, 0),
        (-1, 0),
        (59, 0),
        (60, 0),
        (86_399, 0),
        (86_400, 0),
        (946_684_799, 0),
        (946_684_800, 0),
        (951_782_400, 0),
        (951_868_799, 0),
        (1_709_251_199, 0),
        (1_735_689_599, 0),
        (1_735_689_600, 0),
        (4_102_444_799, 0),
        (253_402_300_799, 0),
        (253_402_300_800, 0),
        (-62_135_596_800, 0),
        (-62_167_219_200, 0),
        (-62_167_219_201, 0),
        (1_483_228_799, 1_500_000_000),
    ];
    // A sub-second part of every width.
    instants.extend((0..9).map(|width| (1_700_000_000, 7 * 10u32.pow(width))));
    // A day, an hour, a minute and a second further at every step, from
    // 1969 to 2026: every month, day, hour, minute and second.
    instants.extend((0..20_000).map(|step| (-31_536_000 + step * 90_061, 0)));
    for (seconds, nanoseconds) in instants {
        let time = DateTime::<Utc>::from_timestamp(seconds, nanoseconds).expect("a valid time");
        let mut written = String::new();
        write_utc_seconds(&mut written, time).expect("writes");
        assert_eq!(
            written,
            through_the_formatter(time),
            "{seconds} {nanoseconds}"
        );
    }
}

/// The cost per access-log line of both formats, for the Numbers table;
/// run with `--ignored --nocapture`.
#[test]
#[ignore = "a timing, not a check"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn time_per_line() {
    fn time<L>(layer: L) -> f64
    where
        L: Layer<Registry> + Send + Sync + 'static,
    {
        let subscriber = Registry::default().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            const LINES: u32 = 100_000;
            let span = info_span!(
                target: "smg::otel-trace",
                "http_request",
                method = "POST",
                path = "/v1/chat/completions",
                version = ?Shape { side: 1 },
                request_id = tracing::field::Empty,
                status_code = tracing::field::Empty,
                latency = tracing::field::Empty,
                module = "smg"
            );
            let _entered = span.enter();
            span.record("request_id", "chatcmpl-0123456789abcdef01234567");
            span.record("status_code", 200u64);
            span.record("latency", 1234u64);
            let start = Instant::now();
            for _ in 0..LINES {
                info!(target: "smg::response", "finished processing request");
            }
            start.elapsed().as_nanos() as f64 / f64::from(LINES)
        })
    }
    let stock = time(
        tracing_subscriber::fmt::layer()
            .with_file(true)
            .with_line_number(true)
            .with_timer(ChronoUtc::new(TIME_FORMAT.to_string()))
            .with_writer(io::sink)
            .json()
            .flatten_event(true),
    );
    let fast = time(
        tracing_subscriber::fmt::layer()
            .with_writer(io::sink)
            .fmt_fields(SpanFields)
            .event_format(JsonLine::new()),
    );
    println!("ns per JSON access-log line: stock {stock:.0}, this format {fast:.0}");
}

/// Beyond `scenario`: the kinds the text format spells out on its own.
/// An error with and without a source, a message and a string with the
/// characters the stock sanitizes or escapes, a span without fields, a
/// span whose first fields are longer than the stack piece, TRACE and
/// DEBUG lines, a raw field name, a field recorded with characters the
/// quick quoted form cannot take.
fn text_scenario() {
    #[derive(Debug)]
    struct Outer(io::Error);
    impl fmt::Display for Outer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("the outer error")
        }
    }
    impl std::error::Error for Outer {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }
    let inner = io::Error::other("the inner one \x1b[31m");
    let outer = Outer(io::Error::other("the \x07 source"));
    let plain = info_span!("plain");
    let plain_entered = plain.enter();
    let long_path = "/v1/".repeat(200);
    let request = info_span!(
        target: "smg::otel-trace",
        "http_request",
        method = "GET",
        path = %long_path,
        r#struct = "raw",
        module = "smg",
        request_id = tracing::field::Empty
    );
    let entered = request.enter();
    tracing::trace!("a trace line");
    tracing::debug!(count = 1u64, "a debug line");
    warn!(
        error = &inner as &(dyn std::error::Error + 'static),
        "without a source"
    );
    tracing::error!(
        error = &outer as &(dyn std::error::Error + 'static),
        "with a source"
    );
    info!(
        text = "quote \" backslash \\ newline \n tab \t unicode é emoji 😀 del \x7f",
        "message with \x1b[2m escape, \x07 bell, \u{85} C1 and \x7f del"
    );
    request.record("request_id", "id with \"quotes\" and é");
    info!("after the record");
    drop(entered);
    drop(plain_entered);
    info!(parent: &plain, "the parent given");
}

fn stock_text_lines(ansi: bool, run: fn()) -> String {
    let capture = Capture::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(ansi)
        .with_file(true)
        .with_line_number(true)
        .with_timer(FixedTime)
        .with_writer(capture.clone());
    tracing::subscriber::with_default(Registry::default().with(layer), run);
    capture.text()
}

fn fast_text_lines(ansi: bool, run: fn()) -> String {
    let capture = Capture::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(ansi)
        .with_writer(capture.clone())
        .fmt_fields(TextFields)
        .event_format(TextLine::with_timer(FixedTime));
    tracing::subscriber::with_default(Registry::default().with(layer), run);
    capture.text()
}

#[test]
fn the_lines_are_those_of_the_stock_text_format() {
    for ansi in [false, true] {
        for (name, run) in [
            ("scenario", scenario as fn()),
            ("text_scenario", text_scenario as fn()),
        ] {
            let stock = stock_text_lines(ansi, run);
            let fast = fast_text_lines(ansi, run);
            assert!(stock.lines().count() >= 7, "{name}: {stock}");
            for (stock_line, fast_line) in stock.lines().zip(fast.lines()) {
                assert_eq!(fast_line, stock_line, "{name}, ansi {ansi}");
            }
            assert_eq!(fast, stock, "{name}, ansi {ansi}");
        }
    }
}

#[test]
fn the_text_lines_carry_the_styles_only_when_ansi_is_on() {
    let styled = fast_text_lines(true, scenario);
    assert!(
        styled.contains(
            "\x1b[32m INFO\x1b[0m \x1b[1mhttp_request\x1b[0m\x1b[1m{\x1b[0m\x1b[3mmethod\x1b[0m\x1b[2m=\x1b[0m\"POST\" "
        ),
        "{styled}"
    );
    let plain = fast_text_lines(false, scenario);
    assert!(
        plain.contains(
            " INFO http_request{method=\"POST\" path=\"/v1/chat/completions\" version=Shape { side: 1 } module=\"smg\"}: smg::request: "
        ),
        "{plain}"
    );
    assert!(!plain.contains('\x1b'), "{plain}");
}

#[test]
fn a_first_field_set_longer_than_the_piece_arrives_whole() {
    use std::fmt::Write as _;
    let mut piece = Piece::new();
    let long = "x".repeat(PIECE_BYTES - 3);
    piece.write_str(&long).expect("fits");
    assert!(piece.heap.is_none());
    piece.write_str("tail é").expect("spills");
    assert!(piece.heap.is_some());
    assert_eq!(piece.as_str().expect("text"), format!("{long}tail é"));
}
