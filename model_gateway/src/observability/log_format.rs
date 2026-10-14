//! The access log's line formats, written without the work the stock
//! formats repeat per line.
//!
//! `tracing_subscriber`'s JSON format keeps a span's fields as a serialized
//! JSON object and, every time a field is recorded after the span opened (a
//! request span records its request id, trace id, status and latency), parses
//! the object back, inserts the field and serializes it again; every event
//! written under the span parses the object once more to copy it into the
//! line's `span` and `spans` members; and the chrono timestamp formatter
//! parses its strftime pattern on every line. [`SpanFields`] keeps a span's
//! fields as entries that are only ever appended to, [`JsonLine`] writes the
//! line from those entries directly, and [`UtcSeconds`] writes the timestamp
//! from the clock's fields. The lines are byte for byte those of the stock
//! JSON format with `flatten_event`, file and line number on (the tests run
//! both formats over the same callsites), so nothing that reads the log
//! changes.

use std::{fmt, io};

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::Serialize;
use tracing::{
    field::{Field, Visit},
    span::Record,
    Event, Subscriber,
};
use tracing_log::NormalizeEvent;
use tracing_subscriber::{
    field::RecordFields,
    fmt::{
        format::Writer, time::FormatTime, FmtContext, FormatEvent, FormatFields, FormattedFields,
    },
    registry::{LookupSpan, SpanRef},
};

/// The timestamp pattern of every line, as chrono spells it.
pub const TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

/// Separates a span's entries in its [`FormattedFields`]. JSON never emits
/// a raw control character (they are escaped), so the byte cannot occur
/// inside an entry.
const ENTRY_SEPARATOR: char = '\x1f';

/// The timestamp of every line: [`TIME_FORMAT`] of the UTC clock, written
/// from the clock's fields instead of through a pattern parsed per line.
#[derive(Clone, Copy, Debug, Default)]
pub struct UtcSeconds;

impl FormatTime for UtcSeconds {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        write_utc_seconds(w, Utc::now())
    }
}

fn write_utc_seconds(w: &mut impl fmt::Write, time: DateTime<Utc>) -> fmt::Result {
    write!(
        w,
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.minute(),
        time.second()
    )
}

/// The span-field format of the JSON layer: a span's entries, each
/// `"name":value` serialized as the stock format serializes it, separated by
/// [`ENTRY_SEPARATOR`] in the order they were recorded. A field recorded
/// again is appended again; [`JsonLine`] keeps the last value and writes the
/// entries in key order, which is what the stock format's map did.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpanFields;

impl<'writer> FormatFields<'writer> for SpanFields {
    fn format_fields<R: RecordFields>(
        &self,
        mut writer: Writer<'writer>,
        fields: R,
    ) -> fmt::Result {
        let mut visitor = SpanEntries::new(&mut writer, true);
        fields.record(&mut visitor);
        visitor.result
    }

    fn add_fields(
        &self,
        current: &'writer mut FormattedFields<Self>,
        fields: &Record<'_>,
    ) -> fmt::Result {
        let first = current.fields.is_empty();
        let mut writer = current.as_writer();
        let mut visitor = SpanEntries::new(&mut writer, first);
        fields.record(&mut visitor);
        visitor.result
    }
}

/// The event format of the JSON layer: the stock JSON line (timestamp,
/// level, the event's fields flattened, target, file, line number, the
/// current span and the span list), with the span members written from
/// [`SpanFields`]' entries.
#[derive(Clone, Copy, Debug, Default)]
pub struct JsonLine<T = UtcSeconds> {
    timer: T,
}

impl JsonLine {
    /// The line format with the UTC timestamp.
    pub fn new() -> Self {
        Self { timer: UtcSeconds }
    }
}

impl<T> JsonLine<T> {
    pub fn with_timer(timer: T) -> Self {
        Self { timer }
    }
}

impl<S, T> FormatEvent<S, SpanFields> for JsonLine<T>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    T: FormatTime,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, SpanFields>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // A record from the `log` facade carries its own target, file and
        // line; the stock format reads them the same way.
        let normalized = event.normalized_metadata();
        let meta = normalized.as_ref().unwrap_or_else(|| event.metadata());

        let mut timestamp = String::with_capacity(32);
        self.timer.format_time(&mut Writer::new(&mut timestamp))?;
        writer.write_str("{\"timestamp\":")?;
        write_json(&mut writer, &timestamp)?;
        writer.write_str(",\"level\":")?;
        write_json(&mut writer, meta.level().as_str())?;

        let mut fields = EventEntries {
            out: &mut writer,
            result: Ok(()),
        };
        event.record(&mut fields);
        fields.result?;

        writer.write_str(",\"target\":")?;
        write_json(&mut writer, meta.target())?;
        if let Some(file) = meta.file() {
            writer.write_str(",\"filename\":")?;
            write_json(&mut writer, file)?;
        }
        if let Some(line) = meta.line() {
            write!(writer, ",\"line_number\":{line}")?;
        }

        // The event's span: its explicit parent, else the current span; the
        // list is the current span's scope, root first (the stock format
        // reads both this way).
        let current = event
            .parent()
            .and_then(|id| ctx.span(id))
            .or_else(|| ctx.lookup_current());
        if let Some(span) = &current {
            writer.write_str(",\"span\":")?;
            write_span(&mut writer, span)?;
            writer.write_str(",\"spans\":[")?;
            if let Some(leaf) = ctx.lookup_current() {
                let mut first = true;
                for span in leaf.scope().from_root() {
                    if !first {
                        writer.write_char(',')?;
                    }
                    first = false;
                    write_span(&mut writer, &span)?;
                }
            }
            writer.write_char(']')?;
        }
        writer.write_str("}\n")
    }
}

/// One span as a JSON object: its entries, the last value of each key in key
/// order, then its name.
fn write_span<S>(writer: &mut Writer<'_>, span: &SpanRef<'_, S>) -> fmt::Result
where
    S: for<'a> LookupSpan<'a>,
{
    let extensions = span.extensions();
    let stored = extensions
        .get::<FormattedFields<SpanFields>>()
        .map_or("", |fields| fields.fields.as_str());
    // Field names are identifiers (the macro spells them), so the key of an
    // entry is the text up to the quote that closes it and sorts as the name
    // sorted in the stock format's map.
    let mut entries: Vec<(&str, &str)> = Vec::with_capacity(8);
    for entry in stored
        .split(ENTRY_SEPARATOR)
        .filter(|entry| !entry.is_empty())
    {
        let key = entry[1..].split('"').next().unwrap_or("");
        match entries.iter_mut().find(|(seen, _)| *seen == key) {
            Some(slot) => slot.1 = entry,
            None => entries.push((key, entry)),
        }
    }
    entries.sort_by_key(|(key, _)| *key);
    writer.write_char('{')?;
    for (_, entry) in &entries {
        writer.write_str(entry)?;
        writer.write_char(',')?;
    }
    writer.write_str("\"name\":")?;
    write_json(writer, span.metadata().name())?;
    writer.write_char('}')
}

/// `value` as JSON, through the same serializer and escaping as the stock
/// format.
fn write_json<T: Serialize + ?Sized>(writer: &mut impl fmt::Write, value: &T) -> fmt::Result {
    serde_json::to_writer(FmtWriter(writer), value).map_err(|_| fmt::Error)
}

/// `std::io::Write` over a `fmt::Write`: serde_json writes whole UTF-8
/// sequences, so every chunk converts.
struct FmtWriter<'a, W: fmt::Write>(&'a mut W);

impl<W: fmt::Write> io::Write for FmtWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let text = std::str::from_utf8(buf)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.0.write_str(text).map_err(io::Error::other)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Writes a span's fields as entries (the stock span visitor's values: a
/// `Debug` field as its debug text, `r#` stripped from a raw name, a
/// `log.*` field of a bridged record skipped).
struct SpanEntries<'a, W: fmt::Write> {
    out: &'a mut W,
    first: bool,
    result: fmt::Result,
}

impl<'a, W: fmt::Write> SpanEntries<'a, W> {
    fn new(out: &'a mut W, first: bool) -> Self {
        Self {
            out,
            first,
            result: Ok(()),
        }
    }

    fn entry<T: Serialize + ?Sized>(&mut self, name: &str, value: &T) {
        if self.result.is_err() {
            return;
        }
        let first = std::mem::replace(&mut self.first, false);
        self.result = (|| {
            if !first {
                self.out.write_char(ENTRY_SEPARATOR)?;
            }
            write_json(self.out, name)?;
            self.out.write_char(':')?;
            write_json(self.out, value)
        })();
    }
}

impl<W: fmt::Write> Visit for SpanEntries<'_, W> {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.entry(field.name(), &value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.entry(field.name(), &value);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.entry(field.name(), &value);
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.entry(field.name(), &value);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.entry(field.name(), value);
    }

    fn record_bytes(&mut self, field: &Field, value: &[u8]) {
        self.entry(field.name(), value);
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        match field.name() {
            name if name.starts_with("log.") => {}
            name if name.starts_with("r#") => self.entry(&name[2..], &format_args!("{value:?}")),
            name => self.entry(name, &format_args!("{value:?}")),
        }
    }
}

/// Writes an event's fields flattened into the line (the stock event
/// visitor's values: a `Debug` field as its debug text, names as spelled).
struct EventEntries<'a, W: fmt::Write> {
    out: &'a mut W,
    result: fmt::Result,
}

impl<W: fmt::Write> EventEntries<'_, W> {
    fn entry<T: Serialize + ?Sized>(&mut self, name: &str, value: &T) {
        if self.result.is_err() {
            return;
        }
        self.result = (|| {
            self.out.write_char(',')?;
            write_json(self.out, name)?;
            self.out.write_char(':')?;
            write_json(self.out, value)
        })();
    }
}

impl<W: fmt::Write> Visit for EventEntries<'_, W> {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.entry(field.name(), &value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.entry(field.name(), &value);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.entry(field.name(), &value);
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.entry(field.name(), &value);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.entry(field.name(), value);
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.entry(field.name(), &format_args!("{value:?}"));
    }
}

#[cfg(test)]
mod tests {
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
}
