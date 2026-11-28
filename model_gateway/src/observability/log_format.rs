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
//!
//! The stock text format writes every styled piece of a line (the span
//! names, the field names, the `=` and `:` signs, the target, the file) as a
//! painted `Display` of its own, each a formatting call with the style's
//! prefix and suffix formatted around it, whether or not ANSI is on, and a
//! span's first fields reach its field string piece by piece, so the string
//! is reallocated at every doubling. [`TextFields`] and [`TextLine`] write
//! the same bytes (the tests run both formats over the same callsites, with
//! ANSI on and off) with direct writes, and a span's first fields in one.

use std::{fmt, fmt::Write as _, io};

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::Serialize;
use tracing::{
    field::{Field, Visit},
    span::Record,
    Event, Level, Subscriber,
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
/// from the clock's fields, digit by digit, instead of through a pattern
/// parsed per line.
#[derive(Clone, Copy, Debug, Default)]
pub struct UtcSeconds;

impl FormatTime for UtcSeconds {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        write_utc_seconds(w, Utc::now())
    }
}

/// The nineteen characters of a timestamp in one write. Padding six numbers
/// through the formatting machinery took a third of a line's work, and every
/// line carries a timestamp. A year outside 0 to 9999 keeps the formatter,
/// whose text this writes for every year inside that range.
fn write_utc_seconds(w: &mut impl fmt::Write, time: DateTime<Utc>) -> fmt::Result {
    let (month, day) = (time.month(), time.day());
    let (hour, minute, second) = (time.hour(), time.minute(), time.second());
    let Ok(year @ 0..=9999) = u32::try_from(time.year()) else {
        let year = time.year();
        return write!(
            w,
            "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}"
        );
    };
    let [y3, y2] = two_digits(year / 100);
    let [y1, y0] = two_digits(year % 100);
    let [m1, m0] = two_digits(month);
    let [d1, d0] = two_digits(day);
    let [h1, h0] = two_digits(hour);
    let [n1, n0] = two_digits(minute);
    let [s1, s0] = two_digits(second);
    let text = [
        y3, y2, y1, y0, b'-', m1, m0, b'-', d1, d0, b' ', h1, h0, b':', n1, n0, b':', s1, s0,
    ];
    // Digits and punctuation are always UTF-8.
    str::from_utf8(&text).map_or(Err(fmt::Error), |text| w.write_str(text))
}

/// The two decimal digits of `value`, which is under 100.
fn two_digits(value: u32) -> [u8; 2] {
    // Both operands are under 10, so the casts keep them.
    [b'0' + (value / 10 % 10) as u8, b'0' + (value % 10) as u8]
}

/// The span-field format of the JSON layer: a span's entries, each
/// `"name":value` serialized as the stock format serializes it, separated by
/// `ENTRY_SEPARATOR` in the order they were recorded. A field recorded
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

// The bytes of the stock text format's styles (nu_ansi_term's for bold,
// dimmed, italic and the reset), written as they are.
const BOLD: &str = "\x1b[1m";
const DIMMED: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";
const RESET: &str = "\x1b[0m";

/// The level column of the stock text format: five characters, and the
/// colour it is painted in when ANSI is on.
fn level_column(level: Level) -> (&'static str, &'static str) {
    match level {
        Level::TRACE => ("\x1b[35m", "TRACE"),
        Level::DEBUG => ("\x1b[34m", "DEBUG"),
        Level::INFO => ("\x1b[32m", " INFO"),
        Level::WARN => ("\x1b[33m", " WARN"),
        Level::ERROR => ("\x1b[31m", "ERROR"),
    }
}

/// `text` under `style` when ANSI is on (the style's bytes, the text, the
/// reset), the text alone otherwise.
fn styled<W: fmt::Write + ?Sized>(out: &mut W, ansi: bool, style: &str, text: &str) -> fmt::Result {
    if ansi {
        out.write_str(style)?;
        out.write_str(text)?;
        out.write_str(RESET)
    } else {
        out.write_str(text)
    }
}

/// [`styled`] as a `Display`, for a piece inside a `format_args!`.
struct Styled<'a>(bool, &'static str, &'a str);

impl fmt::Display for Styled<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        styled(f, self.0, self.1, self.2)
    }
}

/// Writes through to its writer with the characters a terminal could read as
/// the start of an escape sequence spelled out, as the stock format sanitizes
/// a message and an error: ESC, BEL, BS, FF and DEL as `\xNN`, the C1 range
/// as `\u{NN}`.
struct Sanitizing<'a, W: fmt::Write + ?Sized>(&'a mut W);

impl<W: fmt::Write + ?Sized> fmt::Write for Sanitizing<'_, W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // A C1 character is the bytes C2 80..C2 9F in UTF-8.
        if !s
            .bytes()
            .any(|byte| matches!(byte, 0x1b | 0x07 | 0x08 | 0x0c | 0x7f | 0xc2))
        {
            return self.0.write_str(s);
        }
        for ch in s.chars() {
            match ch {
                '\x1b' => self.0.write_str("\\x1b")?,
                '\x07' => self.0.write_str("\\x07")?,
                '\x08' => self.0.write_str("\\x08")?,
                '\x0c' => self.0.write_str("\\x0c")?,
                '\x7f' => self.0.write_str("\\x7f")?,
                ch if (0x80..=0x9f).contains(&(ch as u32)) => {
                    write!(self.0, "\\u{{{:x}}}", ch as u32)?;
                }
                ch => self.0.write_char(ch)?,
            }
        }
        Ok(())
    }
}

/// A value's `Display` text, sanitized when the format sanitizes; as a
/// `Debug` the same text (the stock lists an error's sources this way).
struct Sanitized<T> {
    value: T,
    sanitize: bool,
}

impl<T: fmt::Display> fmt::Display for Sanitized<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sanitize {
            write!(Sanitizing(f), "{}", self.value)
        } else {
            write!(f, "{}", self.value)
        }
    }
}

impl<T: fmt::Display> fmt::Debug for Sanitized<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// An error's sources as the stock format lists them: `[source, its source,
/// ...]`, each its sanitized text.
struct Sources<'a> {
    error: &'a (dyn std::error::Error + 'static),
    sanitize: bool,
}

impl fmt::Display for Sources<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        let mut current = Some(self.error);
        while let Some(error) = current {
            list.entry(&Sanitized {
                value: error,
                sanitize: self.sanitize,
            });
            current = error.source();
        }
        list.finish()
    }
}

/// Whether `str`'s `Debug` form is the string itself between quotes: every
/// character printable ASCII and neither a quote nor a backslash.
fn debug_is_quoted_verbatim(text: &str) -> bool {
    text.bytes()
        .all(|byte| (0x20..=0x7e).contains(&byte) && byte != b'"' && byte != b'\\')
}

/// Writes fields as the stock text format's visitor does: `name=value`
/// entries with a space between them, the name italic and the `=` dimmed
/// when ANSI is on, `r#` stripped from a raw name, a bridged record's `log.*`
/// fields skipped; a `message` as its text (sanitized), a string in its
/// `Debug` form, an error as its text with its sources listed, anything else
/// in its `Debug` form.
struct TextVisitor<'a, W: fmt::Write + ?Sized> {
    out: &'a mut W,
    ansi: bool,
    sanitize: bool,
    first: bool,
    result: fmt::Result,
}

impl<'a, W: fmt::Write + ?Sized> TextVisitor<'a, W> {
    fn new(out: &'a mut W, ansi: bool, sanitize: bool, first: bool) -> Self {
        Self {
            out,
            ansi,
            sanitize,
            first,
            result: Ok(()),
        }
    }

    /// One entry: the space before it (not before the first), then `write`.
    fn entry(&mut self, write: impl FnOnce(&mut W, bool) -> fmt::Result) {
        if self.result.is_err() {
            return;
        }
        let first = std::mem::replace(&mut self.first, false);
        let ansi = self.ansi;
        self.result = (|| {
            if !first {
                self.out.write_char(' ')?;
            }
            write(self.out, ansi)
        })();
    }
}

/// `name=` as the stock writes it.
fn key<W: fmt::Write + ?Sized>(out: &mut W, ansi: bool, name: &str) -> fmt::Result {
    let name = name.strip_prefix("r#").unwrap_or(name);
    styled(out, ansi, ITALIC, name)?;
    styled(out, ansi, DIMMED, "=")
}

impl<W: fmt::Write + ?Sized> Visit for TextVisitor<'_, W> {
    fn record_str(&mut self, field: &Field, value: &str) {
        let name = field.name();
        if name == "message" {
            return self.record_debug(field, &format_args!("{value}"));
        }
        if name.starts_with("log.") {
            return;
        }
        self.entry(|out, ansi| {
            key(out, ansi, name)?;
            if debug_is_quoted_verbatim(value) {
                out.write_char('"')?;
                out.write_str(value)?;
                out.write_char('"')
            } else {
                write!(out, "{value:?}")
            }
        });
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        let (ansi, sanitize) = (self.ansi, self.sanitize);
        let text = Sanitized { value, sanitize };
        match value.source() {
            Some(source) => self.record_debug(
                field,
                &format_args!(
                    "{} {}{}{}{}",
                    text,
                    Styled(ansi, ITALIC, field.name()),
                    Styled(ansi, ITALIC, ".sources"),
                    Styled(ansi, DIMMED, "="),
                    Sources {
                        error: source,
                        sanitize
                    }
                ),
            ),
            None => self.record_debug(field, &format_args!("{text}")),
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let name = field.name();
        if name.starts_with("log.") {
            return;
        }
        let sanitize = self.sanitize;
        self.entry(|out, ansi| {
            if name == "message" {
                if sanitize {
                    write!(Sanitizing(out), "{value:?}")
                } else {
                    write!(out, "{value:?}")
                }
            } else {
                key(out, ansi, name)?;
                write!(out, "{value:?}")
            }
        });
    }
}

/// The room on the stack for a span's first fields.
const PIECE_BYTES: usize = 512;

/// A span's first fields, formatted on the stack so that they reach the
/// span's field string in one write, which sizes the string once (written
/// piece by piece from empty, the string was reallocated at every doubling);
/// a longer set spills to the heap.
struct Piece {
    stack: [u8; PIECE_BYTES],
    len: usize,
    heap: Option<String>,
}

impl Piece {
    fn new() -> Self {
        Self {
            stack: [0; PIECE_BYTES],
            len: 0,
            heap: None,
        }
    }

    fn as_str(&self) -> Result<&str, fmt::Error> {
        match &self.heap {
            Some(heap) => Ok(heap),
            // Whole `str`s were copied in, so the bytes are UTF-8.
            None => std::str::from_utf8(&self.stack[..self.len]).map_err(|_| fmt::Error),
        }
    }
}

impl fmt::Write for Piece {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if let Some(heap) = &mut self.heap {
            heap.push_str(s);
            return Ok(());
        }
        let end = self.len + s.len();
        if end <= PIECE_BYTES {
            self.stack[self.len..end].copy_from_slice(s.as_bytes());
            self.len = end;
            return Ok(());
        }
        let mut heap = String::with_capacity(end * 2);
        heap.push_str(self.as_str()?);
        heap.push_str(s);
        self.heap = Some(heap);
        Ok(())
    }
}

/// The span-field format of the text layer: the entries of the stock
/// format's default field format, byte for byte, written without a styled
/// `Display` per piece, and a span's first fields written in one piece.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextFields;

impl<'writer> FormatFields<'writer> for TextFields {
    fn format_fields<R: RecordFields>(
        &self,
        mut writer: Writer<'writer>,
        fields: R,
    ) -> fmt::Result {
        let mut piece = Piece::new();
        let mut visitor = TextVisitor::new(
            &mut piece,
            writer.has_ansi_escapes(),
            writer.sanitizes_ansi_escapes(),
            true,
        );
        fields.record(&mut visitor);
        visitor.result?;
        writer.write_str(piece.as_str()?)
    }

    fn add_fields(
        &self,
        current: &'writer mut FormattedFields<Self>,
        fields: &Record<'_>,
    ) -> fmt::Result {
        let first = current.fields.is_empty();
        let mut writer = current.as_writer();
        let (ansi, sanitize) = (writer.has_ansi_escapes(), writer.sanitizes_ansi_escapes());
        let mut visitor = TextVisitor::new(&mut writer, ansi, sanitize, first);
        fields.record(&mut visitor);
        visitor.result
    }
}

/// The event format of the text layer: the stock full format's line
/// (timestamp, level, the span scope with each span's fields, target, file
/// and line number, the event's fields) with file and line number on, byte
/// for byte, with ANSI on or off, written with direct writes instead of a
/// styled `Display` per piece.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextLine<T = UtcSeconds> {
    timer: T,
}

impl TextLine {
    /// The line format with the UTC timestamp.
    pub fn new() -> Self {
        Self { timer: UtcSeconds }
    }
}

impl<T> TextLine<T> {
    pub fn with_timer(timer: T) -> Self {
        Self { timer }
    }
}

impl<S, T> FormatEvent<S, TextFields> for TextLine<T>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    T: FormatTime,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, TextFields>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let normalized = event.normalized_metadata();
        let meta = normalized.as_ref().unwrap_or_else(|| event.metadata());
        let ansi = writer.has_ansi_escapes();

        if ansi {
            writer.write_str(DIMMED)?;
        }
        if self.timer.format_time(&mut writer).is_err() {
            writer.write_str("<unknown time>")?;
        }
        if ansi {
            writer.write_str(RESET)?;
        }
        writer.write_char(' ')?;

        let (colour, column) = level_column(*meta.level());
        styled(&mut writer, ansi, colour, column)?;
        writer.write_char(' ')?;

        if let Some(scope) = ctx.event_scope() {
            let mut seen = false;
            for span in scope.from_root() {
                styled(&mut writer, ansi, BOLD, span.metadata().name())?;
                seen = true;
                let extensions = span.extensions();
                if let Some(fields) = extensions.get::<FormattedFields<TextFields>>() {
                    if !fields.fields.is_empty() {
                        styled(&mut writer, ansi, BOLD, "{")?;
                        writer.write_str(&fields.fields)?;
                        styled(&mut writer, ansi, BOLD, "}")?;
                    }
                }
                styled(&mut writer, ansi, DIMMED, ":")?;
            }
            if seen {
                writer.write_char(' ')?;
            }
        }

        styled(&mut writer, ansi, DIMMED, meta.target())?;
        styled(&mut writer, ansi, DIMMED, ":")?;
        writer.write_char(' ')?;

        let line = meta.line();
        if let Some(file) = meta.file() {
            styled(&mut writer, ansi, DIMMED, file)?;
            styled(&mut writer, ansi, DIMMED, ":")?;
            if line.is_none() {
                writer.write_char(' ')?;
            }
        }
        if let Some(line) = line {
            if ansi {
                writer.write_str(DIMMED)?;
            }
            write!(writer, "{line}:")?;
            if ansi {
                writer.write_str(RESET)?;
            }
            writer.write_char(' ')?;
        }

        let sanitize = writer.sanitizes_ansi_escapes();
        let mut fields = TextVisitor::new(&mut writer, ansi, sanitize, true);
        event.record(&mut fields);
        fields.result?;
        writer.write_char('\n')
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/log_format/tests.rs"]
mod tests;
