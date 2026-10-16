//! Re-slicing of streamed chat completion deltas.
//!
//! MiniMax's provider verifier checks the size of every streamed delta:
//! almost none may be shorter than five characters or longer than two
//! hundred. Engines emit one token per event and tool arguments in one
//! piece, so the gateway buffers `content`, `reasoning_content`, `reasoning`
//! and tool-call `arguments` and emits them again in moderate slices. Text is
//! only held back briefly: the relays flush it when the upstream has been
//! quiet for [`IDLE_FLUSH`], except a remainder shorter than `MIN_TAIL_CHARS`
//! (a token or two that arrived just before the engine paused), which waits
//! for the next text or the end of the stream: an event of its own would be
//! exactly the tiny delta the contract forbids.
//!
//! Only single-choice chat completion streams are re-sliced. Everything else
//! is forwarded as it came, after any buffered text so the order never
//! changes: comments, `[DONE]`, malformed frames, events with several
//! choices, events carrying `logprobs` or fields this module does not know.
//!
//! The HTTP relay drives [`SseRechunker`] inline; the gRPC pipeline wraps
//! its own channel in [`rechunk_stream`].
//!
//! Every slice the stage emits, text or tool-call arguments, is shown to a
//! [`DeltaObserver`] first, in the pieces the client will see and on both
//! relays. What the stage forwards whole passes the observer by: events with
//! several choices or a later choice index (the stream is forwarded as it
//! came from there on), events carrying `logprobs` or other choice-level
//! metadata, frames after `[DONE]` or larger than `MAX_FRAME_BYTES`, and
//! malformed frames; and the relays build the stage only for the streams
//! whose provider profile asks for re-slicing. The default, [`NoopObserver`],
//! sees every slice and never objects.
//! An observer that answers [`Verdict::Terminate`] ends the stream in its
//! place: the slice and the text still buffered are dropped, the gateway's
//! error event goes out on a frame of its own, and nothing follows it, not
//! even `[DONE]`.

use std::{borrow::Cow, collections::BTreeMap, fmt, pin::Pin, time::Duration};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::{
    de::{self, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::value::RawValue;

/// Slices hold at most `SLICE_CHARS` characters and are cut once at least
/// `EMIT_THRESHOLD` are buffered. A cut never leaves a tail shorter than
/// `MIN_TAIL_CHARS`, and an idle flush holds back a remainder shorter than
/// it, so only a delta that is tiny in total yields a tiny event.
const SLICE_CHARS: usize = 160;
const EMIT_THRESHOLD: usize = 80;
const MIN_TAIL_CHARS: usize = 5;

/// An unterminated frame longer than this is forwarded as it comes.
const MAX_FRAME_BYTES: usize = 1 << 20;

/// Delta fields whose text is buffered, in the order they are emitted.
const PAYLOAD_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "content"];

/// Buffered text is flushed after this much upstream silence.
pub(crate) const IDLE_FLUSH: Duration = Duration::from_millis(250);

/// Re-slices the deltas of one SSE stream: feed it the body chunks in order
/// and forward what it returns.
pub struct SseRechunker<O = NoopObserver> {
    /// Start of a frame whose delimiter has not arrived yet.
    partial: Vec<u8>,
    /// Bytes of `partial` already searched for a delimiter.
    scanned: usize,
    /// Top-level fields of the latest event except `choices` and `usage`,
    /// serialized as the start of an object (`{"id":"x",`), so synthesized
    /// events look like the upstream ones.
    envelope: Vec<u8>,
    scratch: Vec<u8>,
    fields: [Pending; PAYLOAD_FIELDS.len()],
    tool_args: BTreeMap<Option<u64>, Pending>,
    /// The buffer that received text last. Text for another buffer flushes
    /// it first, so the stream order is kept.
    last: Option<Slot>,
    role_sent: bool,
    /// Forward everything verbatim from here on.
    passthrough: bool,
    forwarded_unknown: usize,
    /// Sees every emitted slice, and may end the stream.
    observer: O,
    /// The stop the observer ended the stream with: nothing is written
    /// after its error event.
    stopped: Option<StreamStop>,
    /// The observer was told the stream ended.
    ended: bool,
}

/// Text waiting to be emitted, with its length in characters kept current
/// so every event does not rescan the buffer.
#[derive(Default)]
struct Pending {
    text: String,
    chars: usize,
}

/// Which buffer a piece of text belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Position in `PAYLOAD_FIELDS`.
    Field(usize),
    /// Tool-call `index`; `None` when the upstream sent none, so the
    /// fragments are emitted without one rather than attributed to call 0.
    ToolArgs(Option<u64>),
}

/// Sees the slices of one stream in the order the client receives them, and
/// may end the stream. One observer per stream; the stage calls it on the
/// relay's task, so a slow observer holds the slice (and the bounded channel
/// behind it) rather than losing it.
pub trait DeltaObserver {
    /// The stream opened: once, before any slice.
    fn open(&mut self) {}

    /// One slice, about to be written. [`Verdict::Continue`] lets it out;
    /// [`Verdict::Terminate`] withholds it and ends the stream.
    fn on_slice(&mut self, slice: DeltaSlice<'_>) -> Verdict;

    /// The stream ended after its last slice: once, from the stage's
    /// `finish`, which the relays call at the upstream's end and after an
    /// upstream error alike. Not called when the observer ended the stream
    /// itself, nor when the client left first: the relays drop the stage
    /// then, without `finish`. An observer that must settle something per
    /// stream does it in its own `Drop`.
    fn end(&mut self) {}
}

/// The observer of a stream nobody watches: its calls compile away.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopObserver;

impl DeltaObserver for NoopObserver {
    fn on_slice(&mut self, _: DeltaSlice<'_>) -> Verdict {
        Verdict::Continue
    }
}

/// One emitted slice: its text and where it belongs in the delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaSlice<'a> {
    pub field: DeltaField,
    pub text: &'a str,
}

/// Where a slice's text belongs in the delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaField {
    /// A text field: `content`, `reasoning_content` or `reasoning`.
    Text(&'static str),
    /// The `arguments` of the tool call with this `index`; `None` when the
    /// upstream sent no index.
    ToolArguments(Option<u64>),
}

/// An observer's answer to a slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The slice goes out as it is.
    Continue,
    /// The slice is withheld and the stream ends with this error.
    Terminate(StreamStop),
}

/// Why an observer ended a stream: the `message` and `type` of the error
/// event that ends it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamStop {
    pub message: String,
    pub error_type: String,
}

impl Default for SseRechunker {
    fn default() -> Self {
        Self::new()
    }
}

impl SseRechunker {
    /// A stage nobody observes.
    pub fn new() -> Self {
        Self::with_observer(NoopObserver)
    }
}

impl<O: DeltaObserver> SseRechunker<O> {
    /// A stage whose slices `observer` sees; it is told the stream opened.
    pub fn with_observer(mut observer: O) -> Self {
        observer.open();
        Self {
            partial: Vec::new(),
            scanned: 0,
            envelope: b"{".to_vec(),
            scratch: Vec::new(),
            fields: Default::default(),
            tool_args: BTreeMap::new(),
            last: None,
            role_sent: false,
            passthrough: false,
            forwarded_unknown: 0,
            observer,
            stopped: None,
            ended: false,
        }
    }

    /// The stop the observer ended the stream with, if it did: the relay
    /// has nothing more to forward then, and letting go of the upstream
    /// cancels the engine's work.
    pub fn stopped(&self) -> Option<&StreamStop> {
        self.stopped.as_ref()
    }

    /// Takes one body chunk and returns the bytes to forward now.
    pub fn feed(&mut self, chunk: Bytes) -> Bytes {
        if self.stopped.is_some() {
            return Bytes::new();
        }
        if self.passthrough {
            return chunk;
        }
        let mut out = Vec::new();
        if self.partial.is_empty() {
            let rest = self.consume(&chunk, 0, &mut out);
            self.partial.extend_from_slice(rest);
        } else {
            self.partial.extend_from_slice(&chunk);
            let mut buffered = std::mem::take(&mut self.partial);
            let consumed = {
                let rest = self.consume(&buffered, self.scanned, &mut out);
                buffered.len() - rest.len()
            };
            buffered.drain(..consumed);
            self.partial = buffered;
        }
        if self.partial.len() > MAX_FRAME_BYTES {
            self.flush(&mut out);
            if self.stopped.is_none() {
                out.append(&mut self.partial);
                self.passthrough = true;
            }
        }
        self.scanned = self.partial.len();
        Bytes::from(out)
    }

    /// Whether text is waiting that an idle flush would emit: a buffer of at
    /// least `MIN_TAIL_CHARS`. A shorter remainder waits for more text.
    pub fn has_pending(&self) -> bool {
        self.fields
            .iter()
            .chain(self.tool_args.values())
            .any(|pending| pending.chars >= MIN_TAIL_CHARS)
    }

    /// Emits the buffered text without waiting for more input, except a
    /// remainder shorter than `MIN_TAIL_CHARS`, which stays for the next
    /// text or [`Self::finish`]: an engine that pauses between tokens must
    /// not turn every token into an event of its own.
    pub fn flush_pending(&mut self) -> Bytes {
        let mut out = Vec::new();
        self.flush_from(MIN_TAIL_CHARS, &mut out);
        Bytes::from(out)
    }

    /// Emits all buffered text, then whatever is left of an unfinished frame.
    pub fn finish(&mut self) -> Bytes {
        let mut out = Vec::new();
        self.flush(&mut out);
        if self.stopped.is_none() {
            out.append(&mut self.partial);
            if !self.ended {
                self.ended = true;
                self.observer.end();
            }
        }
        self.scanned = 0;
        if self.forwarded_unknown > 0 {
            tracing::debug!(
                events = self.forwarded_unknown,
                "SSE events forwarded whole: they carried fields this module does not merge"
            );
        }
        Bytes::from(out)
    }

    /// Handles every complete frame in `input` and returns the unterminated
    /// rest. `searched` bytes at the start are known to hold no delimiter.
    fn consume<'a>(&mut self, input: &'a [u8], searched: usize, out: &mut Vec<u8>) -> &'a [u8] {
        let mut start = 0;
        // A delimiter is up to four bytes long and may straddle the boundary.
        let mut from = searched.saturating_sub(3);
        while let Some((end, delimiter)) = find_frame_end(&input[start..], from) {
            let frame_end = start + end + delimiter;
            self.handle_frame(&input[start..frame_end], out);
            start = frame_end;
            from = 0;
            if self.stopped.is_some() {
                return &[];
            }
            if self.passthrough {
                out.extend_from_slice(&input[start..]);
                return &[];
            }
        }
        &input[start..]
    }

    fn handle_frame(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        let data = match classify_frame(frame) {
            Frame::Data(data) => data,
            Frame::Comment => {
                out.extend_from_slice(frame);
                return;
            }
            Frame::Other => {
                self.forward(frame, out);
                return;
            }
        };
        if data.starts_with(b"[DONE]") {
            self.forward(frame, out);
            self.passthrough = true;
            return;
        }
        let event = std::str::from_utf8(data)
            .ok()
            .and_then(|text| serde_json::from_str::<RawObject>(text).ok());
        match event {
            Some(event) => self.handle_event(&event, frame, out),
            None => self.forward(frame, out),
        }
    }

    /// Writes the buffered text, then the frame as it came.
    fn forward(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        self.flush(out);
        if self.stopped.is_none() {
            out.extend_from_slice(frame);
        }
    }

    /// Writes `variant` of the event, unless the observer ended the stream.
    fn write_structural(
        &self,
        structural: &Structural<'_, '_>,
        variant: Variant,
        out: &mut Vec<u8>,
    ) {
        if self.stopped.is_none() {
            write_event(structural, variant, out);
        }
    }

    fn handle_event(&mut self, event: &RawObject<'_>, frame: &[u8], out: &mut Vec<u8>) {
        let choices = event
            .get("choices")
            .and_then(parse_array)
            .unwrap_or_default();
        let choice = choices.first().and_then(|raw| parse_object(raw));
        let other_choice = choice
            .as_ref()
            .and_then(|choice| choice.get("index"))
            .and_then(parse_u64)
            .is_some_and(|index| index != 0);
        if choices.len() > 1 || other_choice {
            self.forward(frame, out);
            self.passthrough = true;
            return;
        }

        let delta = choice
            .as_ref()
            .and_then(|choice| choice.get("delta"))
            .and_then(parse_object);
        if choice.as_ref().is_some_and(has_choice_metadata) {
            // logprobs and choice-level usage describe this event's own text,
            // so the event stays whole.
            if delta
                .as_ref()
                .and_then(|delta| delta.get("role"))
                .is_some_and(|role| !is_null(role))
            {
                self.role_sent = true;
            }
            self.forward(frame, out);
            self.forwarded_unknown += 1;
            return;
        }

        let mut shape = Shape {
            closes: choice.is_none()
                || delta.is_none()
                || event.get("usage").is_some_and(|usage| !is_null(usage))
                || choice
                    .as_ref()
                    .and_then(|choice| choice.get("finish_reason"))
                    .is_some_and(|finish| !is_null(finish)),
            ..Shape::default()
        };
        if let Some(delta) = &delta {
            self.inspect_delta(delta, &mut shape);
        }
        if choices.len() == 1 {
            self.update_envelope(event);
        }

        let structural = Structural {
            event,
            choice: choice.as_ref(),
            delta: delta.as_ref(),
            shape: &shape,
        };
        // Text already buffered goes out before an event that opens
        // something; an event that closes something goes out after its own
        // text, so nothing is split.
        match (shape.opens, shape.closes) {
            (true, true) => {
                self.flush(out);
                self.write_structural(&structural, Variant::Opening, out);
                self.push_text(&shape, out);
                self.flush(out);
                self.write_structural(&structural, Variant::Closing, out);
            }
            (true, false) => {
                self.flush(out);
                self.write_structural(&structural, Variant::Whole, out);
                self.push_text(&shape, out);
                self.drain_ready(out);
            }
            (false, true) => {
                self.push_text(&shape, out);
                self.flush(out);
                self.write_structural(&structural, Variant::Whole, out);
            }
            (false, false) => {
                self.push_text(&shape, out);
                self.drain_ready(out);
            }
        }
    }

    /// Sorts the delta's fields into buffered text and the parts that must
    /// be forwarded in order.
    fn inspect_delta<'a>(&mut self, delta: &RawObject<'a>, shape: &mut Shape<'a>) {
        let mut unknown = false;
        for (key, value) in &delta.entries {
            if is_null(value) {
                continue;
            }
            match key.as_ref() {
                "role" if self.role_sent => {}
                "role" => {
                    self.role_sent = true;
                    shape.first_role = true;
                    shape.opens = true;
                }
                "tool_calls" => unknown |= !inspect_tool_calls(value, shape),
                key => match PAYLOAD_FIELDS.iter().position(|field| *field == key) {
                    Some(slot) => match parse_text(value) {
                        Some(text) if text.is_empty() => {}
                        Some(text) => shape.texts[slot] = Some(text),
                        None => unknown = true,
                    },
                    None => unknown = true,
                },
            }
        }
        if unknown {
            shape.closes = true;
            self.forwarded_unknown += 1;
        }
    }

    /// Keeps the envelope in step with the latest event.
    fn update_envelope(&mut self, event: &RawObject<'_>) {
        self.scratch.clear();
        self.scratch.push(b'{');
        for (key, value) in &event.entries {
            if key == "choices" || key == "usage" {
                continue;
            }
            write_json(&mut self.scratch, key.as_ref());
            self.scratch.push(b':');
            self.scratch.extend_from_slice(value.get().as_bytes());
            self.scratch.push(b',');
        }
        if self.scratch != self.envelope {
            std::mem::swap(&mut self.scratch, &mut self.envelope);
        }
    }

    fn push_text(&mut self, shape: &Shape<'_>, out: &mut Vec<u8>) {
        for (slot, text) in shape.texts.iter().enumerate() {
            if let Some(text) = text {
                self.push(Slot::Field(slot), text, out);
            }
        }
        for call in &shape.tool_calls {
            if let Some(arguments) = &call.arguments {
                self.push(Slot::ToolArgs(call.index), arguments, out);
            }
        }
    }

    fn push(&mut self, slot: Slot, text: &str, out: &mut Vec<u8>) {
        if self.last.is_some_and(|last| last != slot) {
            self.flush(out);
        }
        if self.stopped.is_some() {
            return;
        }
        self.last = Some(slot);
        let pending = match slot {
            Slot::Field(field) => &mut self.fields[field],
            Slot::ToolArgs(index) => self.tool_args.entry(index).or_default(),
        };
        pending.text.push_str(text);
        pending.chars += text.chars().count();
    }

    /// Emits slices while enough text is buffered.
    fn drain_ready(&mut self, out: &mut Vec<u8>) {
        if let Err(stop) = self.emit_ready(out) {
            self.stop(stop, out);
        }
    }

    fn emit_ready(&mut self, out: &mut Vec<u8>) -> Result<(), StreamStop> {
        for (field, pending) in PAYLOAD_FIELDS.iter().zip(&mut self.fields) {
            while pending.chars >= EMIT_THRESHOLD {
                emit_text(&self.envelope, &mut self.observer, field, pending, out)?;
            }
        }
        for (index, pending) in &mut self.tool_args {
            while pending.chars >= EMIT_THRESHOLD {
                emit_arguments(&self.envelope, &mut self.observer, *index, pending, out)?;
            }
        }
        Ok(())
    }

    /// Emits every buffered slice.
    fn flush(&mut self, out: &mut Vec<u8>) {
        self.flush_from(1, out);
    }

    /// Emits every buffered slice of the buffers holding at least `min_chars`.
    fn flush_from(&mut self, min_chars: usize, out: &mut Vec<u8>) {
        if let Err(stop) = self.emit_from(min_chars, out) {
            self.stop(stop, out);
        }
    }

    fn emit_from(&mut self, min_chars: usize, out: &mut Vec<u8>) -> Result<(), StreamStop> {
        for (field, pending) in PAYLOAD_FIELDS.iter().zip(&mut self.fields) {
            if pending.chars >= min_chars {
                while pending.chars > 0 {
                    emit_text(&self.envelope, &mut self.observer, field, pending, out)?;
                }
            }
        }
        for (index, pending) in &mut self.tool_args {
            if pending.chars >= min_chars {
                while pending.chars > 0 {
                    emit_arguments(&self.envelope, &mut self.observer, *index, pending, out)?;
                }
            }
        }
        Ok(())
    }

    /// Ends the stream on the observer's verdict: the text still buffered is
    /// dropped, the error event is written (the stage writes whole frames
    /// only, so it stands on a frame of its own), and nothing follows it.
    fn stop(&mut self, stop: StreamStop, out: &mut Vec<u8>) {
        for pending in &mut self.fields {
            pending.clear();
        }
        self.tool_args.clear();
        self.last = None;
        write_error_event(&stop, out);
        self.stopped = Some(stop);
    }
}

/// What an event carries besides the text that gets buffered.
#[derive(Default)]
struct Shape<'a> {
    /// Something later text attaches to (a role, a tool-call identity): it
    /// must be written before that text.
    opens: bool,
    /// Something that ends earlier text (finish_reason, usage, a field this
    /// module does not merge): it must be written after that text.
    closes: bool,
    /// The event carries the stream's first `role`; later roles are dropped.
    first_role: bool,
    /// Every tool-call entry was emptied by taking its arguments.
    drop_tool_calls: bool,
    /// Buffered text per `PAYLOAD_FIELDS` position.
    texts: [Option<Text<'a>>; PAYLOAD_FIELDS.len()],
    tool_calls: Vec<ToolCall<'a>>,
}

struct ToolCall<'a> {
    raw: &'a RawValue,
    /// `None` when the entry is not an object; it is then forwarded as is.
    object: Option<RawObject<'a>>,
    function: Option<RawObject<'a>>,
    index: Option<u64>,
    arguments: Option<Text<'a>>,
}

/// Records the tool-call entries of a delta in `shape`. Returns false when
/// the array holds something this module does not merge.
fn inspect_tool_calls<'a>(value: &'a RawValue, shape: &mut Shape<'a>) -> bool {
    let Some(entries) = parse_array(value) else {
        shape.closes = true;
        return false;
    };
    let mut identity = false;
    let mut merged = true;
    for raw in entries {
        let object = parse_object(raw);
        let function = object
            .as_ref()
            .and_then(|object| object.get("function"))
            .and_then(parse_object);
        let arguments = function
            .as_ref()
            .and_then(|function| function.get("arguments"))
            .filter(|arguments| !is_null(arguments))
            .and_then(parse_text)
            .filter(|arguments| !arguments.is_empty());
        identity |= function
            .as_ref()
            .and_then(|function| function.get("name"))
            .is_some_and(|name| !is_null(name));
        identity |= object
            .as_ref()
            .and_then(|object| object.get("id"))
            .is_some_and(|id| !is_null(id));
        merged &= object.is_some();
        shape.tool_calls.push(ToolCall {
            raw,
            index: object
                .as_ref()
                .and_then(|object| object.get("index"))
                .and_then(parse_u64),
            object,
            function,
            arguments,
        });
    }
    if identity {
        shape.opens = true;
    } else if merged && shape.tool_calls.iter().all(is_shell) {
        shape.drop_tool_calls = true;
    } else {
        shape.closes = true;
        return false;
    }
    true
}

/// A tool-call entry left with nothing but its index, type, or an emptied
/// function once its arguments were taken.
fn is_shell(call: &ToolCall<'_>) -> bool {
    call.object.as_ref().is_some_and(|object| {
        object.entries.iter().all(|(key, value)| {
            is_null(value)
                || key == "index"
                || key == "type"
                || (key == "function"
                    && call.function.as_ref().is_some_and(|function| {
                        function
                            .entries
                            .iter()
                            .all(|(key, value)| is_null(value) || key == "arguments")
                    }))
        })
    })
}

/// Whether the choice carries a non-null field beyond index, delta and
/// finish_reason.
fn has_choice_metadata(choice: &RawObject<'_>) -> bool {
    choice.entries.iter().any(|(key, value)| {
        !matches!(key.as_ref(), "index" | "delta" | "finish_reason") && !is_null(value)
    })
}

/// The parts of an event that are written back after its text was taken.
struct Structural<'s, 'a> {
    event: &'s RawObject<'a>,
    choice: Option<&'s RawObject<'a>>,
    delta: Option<&'s RawObject<'a>>,
    shape: &'s Shape<'a>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Whole,
    /// The event minus what ends the stream: `finish_reason` becomes null
    /// and `usage` is left out.
    Opening,
    /// The event with an empty delta.
    Closing,
}

fn write_event(structural: &Structural<'_, '_>, variant: Variant, out: &mut Vec<u8>) {
    out.extend_from_slice(b"data: {");
    let mut comma = Comma::default();
    for (key, value) in &structural.event.entries {
        match (key.as_ref(), structural.choice) {
            ("choices", Some(choice)) => {
                comma.key(out, key);
                out.push(b'[');
                write_choice(choice, structural, variant, out);
                out.push(b']');
            }
            ("usage", _) if variant == Variant::Opening => {}
            _ => comma.raw(out, key, value),
        }
    }
    out.extend_from_slice(b"}\n\n");
}

fn write_choice(
    choice: &RawObject<'_>,
    structural: &Structural<'_, '_>,
    variant: Variant,
    out: &mut Vec<u8>,
) {
    out.push(b'{');
    let mut comma = Comma::default();
    for (key, value) in &choice.entries {
        match (key.as_ref(), structural.delta) {
            ("delta", _) if variant == Variant::Closing => {
                comma.key(out, key);
                out.extend_from_slice(b"{}");
            }
            ("delta", Some(delta)) => {
                comma.key(out, key);
                write_delta(delta, structural.shape, out);
            }
            ("finish_reason", _) if variant == Variant::Opening => {
                comma.key(out, key);
                out.extend_from_slice(b"null");
            }
            _ => comma.raw(out, key, value),
        }
    }
    out.push(b'}');
}

/// The delta without the text that was buffered, null fields, repeated
/// roles and emptied tool-call entries.
fn write_delta(delta: &RawObject<'_>, shape: &Shape<'_>, out: &mut Vec<u8>) {
    out.push(b'{');
    let mut comma = Comma::default();
    for (key, value) in &delta.entries {
        if is_null(value) {
            continue;
        }
        match key.as_ref() {
            "role" if !shape.first_role => {}
            "tool_calls" if shape.drop_tool_calls => {}
            "tool_calls" if !shape.tool_calls.is_empty() => {
                comma.key(out, key);
                write_tool_calls(&shape.tool_calls, out);
            }
            key if PAYLOAD_FIELDS.contains(&key) && value.get().starts_with('"') => {}
            _ => comma.raw(out, key, value),
        }
    }
    out.push(b'}');
}

fn write_tool_calls(calls: &[ToolCall<'_>], out: &mut Vec<u8>) {
    out.push(b'[');
    for (position, call) in calls.iter().enumerate() {
        if position > 0 {
            out.push(b',');
        }
        let Some(object) = &call.object else {
            out.extend_from_slice(call.raw.get().as_bytes());
            continue;
        };
        out.push(b'{');
        let mut comma = Comma::default();
        for (key, value) in &object.entries {
            if is_null(value) {
                continue;
            }
            match (key.as_ref(), &call.function) {
                ("function", Some(function)) => {
                    comma.key(out, key);
                    write_function(function, out);
                }
                _ => comma.raw(out, key, value),
            }
        }
        out.push(b'}');
    }
    out.push(b']');
}

fn write_function(function: &RawObject<'_>, out: &mut Vec<u8>) {
    out.push(b'{');
    let mut comma = Comma::default();
    for (key, value) in &function.entries {
        let taken = key == "arguments" && value.get().starts_with('"');
        if !is_null(value) && !taken {
            comma.raw(out, key, value);
        }
    }
    out.push(b'}');
}

/// Writes one synthesized event with the next slice of `pending` as `field`,
/// unless the observer ends the stream on it.
fn emit_text<O: DeltaObserver>(
    envelope: &[u8],
    observer: &mut O,
    field: &'static str,
    pending: &mut Pending,
    out: &mut Vec<u8>,
) -> Result<(), StreamStop> {
    let (bytes, chars) = pending.next_slice();
    observe(observer, DeltaField::Text(field), &pending.text[..bytes])?;
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(envelope);
    out.extend_from_slice(b"\"choices\":[{\"index\":0,\"delta\":{");
    write_json(out, field);
    out.push(b':');
    write_json(out, &pending.text[..bytes]);
    out.extend_from_slice(b"},\"finish_reason\":null}]}\n\n");
    pending.cut(bytes, chars);
    Ok(())
}

/// Writes one synthesized event with the next slice of `pending` as the
/// arguments of tool call `index`, unless the observer ends the stream on it.
fn emit_arguments<O: DeltaObserver>(
    envelope: &[u8],
    observer: &mut O,
    index: Option<u64>,
    pending: &mut Pending,
    out: &mut Vec<u8>,
) -> Result<(), StreamStop> {
    let (bytes, chars) = pending.next_slice();
    observe(
        observer,
        DeltaField::ToolArguments(index),
        &pending.text[..bytes],
    )?;
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(envelope);
    out.extend_from_slice(b"\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{");
    if let Some(index) = index {
        out.extend_from_slice(b"\"index\":");
        write_json(out, &index);
        out.push(b',');
    }
    out.extend_from_slice(b"\"function\":{\"arguments\":");
    write_json(out, &pending.text[..bytes]);
    out.extend_from_slice(b"}}]},\"finish_reason\":null}]}\n\n");
    pending.cut(bytes, chars);
    Ok(())
}

/// Shows a slice to the observer; the stop it ends the stream with, if so.
fn observe<O: DeltaObserver>(
    observer: &mut O,
    field: DeltaField,
    text: &str,
) -> Result<(), StreamStop> {
    match observer.on_slice(DeltaSlice { field, text }) {
        Verdict::Continue => Ok(()),
        Verdict::Terminate(stop) => Err(stop),
    }
}

/// The event that ends a stream the observer stopped, in the shape of every
/// error event the gateway writes into a stream:
/// `data: {"error":{"message":...,"type":...}}`.
fn write_error_event(stop: &StreamStop, out: &mut Vec<u8>) {
    #[derive(Serialize)]
    struct ErrorEvent<'a> {
        error: ErrorBody<'a>,
    }

    #[derive(Serialize)]
    struct ErrorBody<'a> {
        message: &'a str,
        #[serde(rename = "type")]
        error_type: &'a str,
    }

    out.extend_from_slice(b"data: ");
    write_json(
        out,
        &ErrorEvent {
            error: ErrorBody {
                message: &stop.message,
                error_type: &stop.error_type,
            },
        },
    );
    out.extend_from_slice(b"\n\n");
}

impl Pending {
    fn clear(&mut self) {
        self.text.clear();
        self.chars = 0;
    }

    /// Byte and character length of the next slice: at most `SLICE_CHARS`,
    /// never leaving a tail shorter than `MIN_TAIL_CHARS`.
    fn next_slice(&self) -> (usize, usize) {
        let chars = if self.chars <= SLICE_CHARS {
            self.chars
        } else if self.chars - SLICE_CHARS < MIN_TAIL_CHARS {
            self.chars - MIN_TAIL_CHARS
        } else {
            SLICE_CHARS
        };
        let bytes = self
            .text
            .char_indices()
            .nth(chars)
            .map_or(self.text.len(), |(byte, _)| byte);
        (bytes, chars)
    }

    fn cut(&mut self, bytes: usize, chars: usize) {
        self.text.drain(..bytes);
        self.chars -= chars;
    }
}

/// Puts the commas between the members of a JSON object being written.
#[derive(Default)]
struct Comma {
    started: bool,
}

impl Comma {
    fn key(&mut self, out: &mut Vec<u8>, key: &str) {
        if self.started {
            out.push(b',');
        }
        self.started = true;
        write_json(out, key);
        out.push(b':');
    }

    fn raw(&mut self, out: &mut Vec<u8>, key: &str, value: &RawValue) {
        self.key(out, key);
        out.extend_from_slice(value.get().as_bytes());
    }
}

#[expect(clippy::expect_used, reason = "serializing into a Vec cannot fail")]
fn write_json<T: Serialize + ?Sized>(out: &mut Vec<u8>, value: &T) {
    serde_json::to_writer(&mut *out, value).expect("serialize JSON into a Vec");
}

/// A JSON object as its keys and untouched values, in order.
struct RawObject<'a> {
    entries: Vec<(Cow<'a, str>, &'a RawValue)>,
}

impl<'a> RawObject<'a> {
    fn get(&self, key: &str) -> Option<&'a RawValue> {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| *value)
    }
}

impl<'de> Deserialize<'de> for RawObject<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ObjectVisitor;

        impl<'de> Visitor<'de> for ObjectVisitor {
            type Value = RawObject<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((key, value)) = map.next_entry::<Text<'de>, &'de RawValue>()? {
                    entries.push((key.0, value));
                }
                Ok(RawObject { entries })
            }
        }

        deserializer.deserialize_map(ObjectVisitor)
    }
}

/// A JSON string, borrowed from the input unless it contains escapes.
struct Text<'a>(Cow<'a, str>);

impl std::ops::Deref for Text<'_> {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Text<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TextVisitor;

        impl<'de> Visitor<'de> for TextVisitor {
            type Value = Text<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string")
            }

            fn visit_borrowed_str<E: de::Error>(self, text: &'de str) -> Result<Self::Value, E> {
                Ok(Text(Cow::Borrowed(text)))
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Ok(Text(Cow::Owned(text.to_owned())))
            }

            fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
                Ok(Text(Cow::Owned(text)))
            }
        }

        deserializer.deserialize_str(TextVisitor)
    }
}

fn parse_object(raw: &RawValue) -> Option<RawObject<'_>> {
    serde_json::from_str(raw.get()).ok()
}

fn parse_array(raw: &RawValue) -> Option<Vec<&RawValue>> {
    serde_json::from_str(raw.get()).ok()
}

fn parse_text(raw: &RawValue) -> Option<Text<'_>> {
    raw.get()
        .starts_with('"')
        .then(|| serde_json::from_str(raw.get()).ok())
        .flatten()
}

fn parse_u64(raw: &RawValue) -> Option<u64> {
    raw.get().parse().ok()
}

fn is_null(raw: &RawValue) -> bool {
    raw.get() == "null"
}

enum Frame<'a> {
    /// A frame made of exactly one `data:` line.
    Data(&'a [u8]),
    /// A comment or keep-alive: no `data:` line at all.
    Comment,
    Other,
}

fn classify_frame(frame: &[u8]) -> Frame<'_> {
    let mut data = None;
    let mut data_lines = 0;
    let mut other_lines = false;
    for line in frame.split(|byte| *byte == b'\n' || *byte == b'\r') {
        if line.is_empty() {
            continue;
        }
        match line.strip_prefix(b"data:") {
            Some(rest) => {
                data_lines += 1;
                data = Some(rest.strip_prefix(b" ").unwrap_or(rest));
            }
            None => other_lines = true,
        }
    }
    match (data, data_lines, other_lines) {
        (Some(data), 1, false) => Frame::Data(data),
        (_, 0, _) => Frame::Comment,
        _ => Frame::Other,
    }
}

/// Position and length of the first frame delimiter at or after `from`:
/// `\r\n\r\n`, `\n\n` or `\r\r`.
fn find_frame_end(buf: &[u8], from: usize) -> Option<(usize, usize)> {
    (from..buf.len()).find_map(|i| {
        if buf[i..].starts_with(b"\r\n\r\n") {
            Some((i, 4))
        } else if buf[i..].starts_with(b"\n\n") || buf[i..].starts_with(b"\r\r") {
            Some((i, 2))
        } else {
            None
        }
    })
}

/// Re-chunks a stream of SSE body chunks: every chunk passes through an
/// [`SseRechunker`] watched by `observer`, text held back is flushed once
/// the upstream has been quiet for [`IDLE_FLUSH`], and the tail is flushed
/// when the stream ends. An upstream error still follows whatever text was
/// pending. A stream the observer stops ends with its error event, and the
/// upstream is let go: on this relay, that is what cancels the engine's
/// work.
pub(crate) fn rechunk_stream<S, E, O>(
    inner: S,
    observer: O,
) -> impl Stream<Item = Result<Bytes, E>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: Send + 'static,
    O: DeltaObserver + Send + 'static,
{
    struct State<S, E, O> {
        inner: S,
        rechunker: SseRechunker<O>,
        idle: Pin<Box<tokio::time::Sleep>>,
        /// An upstream error, delivered after the flushed tail.
        deferred: Option<E>,
        ended: bool,
    }

    let state = State {
        inner,
        rechunker: SseRechunker::with_observer(observer),
        idle: Box::pin(tokio::time::sleep(IDLE_FLUSH)),
        deferred: None,
        ended: false,
    };
    futures::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(err) = st.deferred.take() {
                return Some((Err(err), st));
            }
            if st.ended {
                return None;
            }
            tokio::select! {
                chunk = st.inner.next() => match chunk {
                    Some(Ok(bytes)) => {
                        st.idle.as_mut().reset(tokio::time::Instant::now() + IDLE_FLUSH);
                        let out = st.rechunker.feed(bytes);
                        st.ended |= st.rechunker.stopped().is_some();
                        if !out.is_empty() {
                            return Some((Ok(out), st));
                        }
                    }
                    Some(Err(err)) => {
                        st.ended = true;
                        let tail = st.rechunker.finish();
                        if tail.is_empty() {
                            return Some((Err(err), st));
                        }
                        // A veto on the tail ends the stream with the policy's
                        // error frame: the upstream's error is not yielded
                        // after it, or the body would end as an aborted one.
                        if st.rechunker.stopped().is_none() {
                            st.deferred = Some(err);
                        }
                        return Some((Ok(tail), st));
                    }
                    None => {
                        st.ended = true;
                        let tail = st.rechunker.finish();
                        if tail.is_empty() {
                            return None;
                        }
                        return Some((Ok(tail), st));
                    }
                },
                () = st.idle.as_mut(), if st.rechunker.has_pending() => {
                    st.idle.as_mut().reset(tokio::time::Instant::now() + IDLE_FLUSH);
                    let out = st.rechunker.flush_pending();
                    st.ended |= st.rechunker.stopped().is_some();
                    if !out.is_empty() {
                        return Some((Ok(out), st));
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{Arc, Mutex},
    };

    use futures::stream;
    use serde_json::Value;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    use super::*;

    fn bytes(data: impl AsRef<[u8]>) -> Bytes {
        Bytes::copy_from_slice(data.as_ref())
    }

    fn events(bytes: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(bytes)
            .split("\n\n")
            .filter(|f| !f.is_empty())
            .filter_map(|f| f.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap())
            .collect()
    }

    fn content_event(text: &str) -> String {
        format!(
            "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
            serde_json::to_string(text).unwrap()
        )
    }

    fn run(frames: &[&str]) -> Vec<u8> {
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        for frame in frames {
            all.extend_from_slice(r.feed(bytes(frame)).as_ref());
        }
        all.extend_from_slice(r.finish().as_ref());
        all
    }

    fn content_sizes(evs: &[Value]) -> Vec<usize> {
        evs.iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .map(|s| s.chars().count())
            .collect()
    }

    #[test]
    fn tiny_deltas_merge() {
        let frames: Vec<String> = (0..100).map(|_| content_event("ab")).collect();
        let refs: Vec<&str> = frames.iter().map(String::as_str).collect();
        let sizes = content_sizes(&events(&run(&refs)));
        assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 200);
    }

    #[test]
    fn large_delta_splits() {
        let big = "x".repeat(1000);
        let sizes = content_sizes(&events(&run(&[&content_event(&big)])));
        assert!(sizes.iter().all(|&s| s <= 200), "sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 1000);
    }

    #[test]
    fn tail_slices_never_fall_into_the_tiny_band() {
        let sizes = content_sizes(&events(&run(&[&content_event(&"x".repeat(161))])));
        assert_eq!(sizes, vec![156, 5]);
    }

    #[test]
    fn multibyte_payload_is_sliced_by_chars() {
        let all = run(&[&content_event(&"中".repeat(500))]);
        let evs = events(&all);
        for e in &evs {
            let s = e["choices"][0]["delta"]["content"].as_str().unwrap();
            assert!(s.chars().count() <= 200, "chars: {}", s.chars().count());
        }
        assert!(evs
            .iter()
            .any(|e| { e["choices"][0]["delta"]["content"].as_str().unwrap().len() > 200 }));
        assert_eq!(content_sizes(&evs).iter().sum::<usize>(), 500);
    }

    #[test]
    fn escaped_text_is_rebuilt_exactly() {
        let text = "line one\n\t\"quoted\" \\ 中文 \u{1F600}";
        let evs = events(&run(&[&content_event(text)]));
        let joined: String = evs
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn frames_split_across_chunks_are_reassembled() {
        let frame = content_event(&"y".repeat(100));
        let frame = frame.as_bytes();
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        for k in [7usize, 40, 90] {
            all.extend_from_slice(r.feed(bytes(&frame[..k])).as_ref());
            assert!(all.is_empty(), "nothing before the frame is complete");
            all.extend_from_slice(r.feed(bytes(&frame[k..])).as_ref());
            all.extend_from_slice(r.finish().as_ref());
            let sizes = content_sizes(&events(&all));
            assert_eq!(sizes, vec![100], "split at {k}");
            r = SseRechunker::new();
            all.clear();
        }
    }

    #[test]
    fn crlf_frames_are_recognised() {
        let frame = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ab\"}}]}\r\n\r\n";
        let frames = [frame; 50];
        let sizes = content_sizes(&events(&run(&frames)));
        assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn bare_cr_frames_are_recognised() {
        let frame =
            "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ab\"}}]}\r\r";
        let frames = [frame; 50];
        let sizes = content_sizes(&events(&run(&frames)));
        assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn a_delimiter_split_across_feeds_is_found() {
        let frame = content_event(&"z".repeat(100));
        let frame = frame.as_bytes();
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(bytes(&frame[..frame.len() - 1])).as_ref());
        assert!(all.is_empty());
        all.extend_from_slice(r.feed(bytes(&frame[frame.len() - 1..])).as_ref());
        assert_eq!(content_sizes(&events(&all)), vec![100]);
    }

    #[test]
    fn oversized_unterminated_frames_switch_to_passthrough() {
        let mut r = SseRechunker::new();
        assert!(r.feed(bytes(content_event("ab"))).is_empty());
        let fragment = vec![b'q'; 64 * 1024];
        let mut forwarded = Vec::new();
        for _ in 0..17 {
            forwarded.extend_from_slice(r.feed(bytes(&fragment)).as_ref());
        }
        // The pending payload came out first, then every raw byte.
        let text = String::from_utf8_lossy(&forwarded);
        assert!(text.starts_with("data: "), "pending payload flushed first");
        assert!(text.ends_with(&"q".repeat(64 * 1024)));
        assert_eq!(text.matches('q').count(), 17 * 64 * 1024);
        assert_eq!(r.feed(bytes(b"more")).as_ref(), b"more");
    }

    #[test]
    fn passthrough_returns_the_chunk_it_was_given() {
        let mut r = SseRechunker::new();
        r.feed(bytes(b"data: [DONE]\n\n"));
        let chunk = Bytes::from_static(b"trailing bytes");
        let returned = r.feed(chunk.clone());
        assert_eq!(returned.as_ptr(), chunk.as_ptr(), "no copy in passthrough");
    }

    #[test]
    fn choice_level_metadata_is_forwarded_once() {
        let with_logprobs = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"logprobs\":{\"content\":[{\"token\":\"Hi\",\"logprob\":-0.1}]},\"finish_reason\":null}]}\n\n";
        let evs = events(&run(&[
            &content_event("ab"),
            with_logprobs,
            &content_event("cd"),
        ]));
        let logprob_events: Vec<&Value> = evs
            .iter()
            .filter(|e| e["choices"][0]["logprobs"].is_object())
            .collect();
        assert_eq!(logprob_events.len(), 1);
        assert_eq!(
            logprob_events[0]["choices"][0]["delta"]["content"],
            Value::from("Hi")
        );
        // The logprobs stay attached to the content they describe.
        let texts: Vec<&str> = evs
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(texts, vec!["ab", "Hi", "cd"]);
        // A null logprobs field, the common shape when none were requested, still merges.
        let null_logprobs = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ab\"},\"logprobs\":null}]}\n\n";
        let frames = [null_logprobs; 50];
        let sizes = content_sizes(&events(&run(&frames)));
        assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
    }

    #[test]
    fn choice_level_usage_is_forwarded() {
        let with_usage = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"usage\":{\"total_tokens\":3}}]}\n\n";
        let evs = events(&run(&[with_usage]));
        assert!(evs
            .iter()
            .any(|e| e["choices"][0]["usage"]["total_tokens"] == 3));
    }

    #[test]
    fn finish_and_usage_flush_in_order() {
        let fin = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"total_tokens\":5}}\n\n";
        let evs = events(&run(&[&content_event("hello"), fin]));
        assert_eq!(
            evs.last().unwrap()["choices"][0]["finish_reason"],
            Value::from("stop")
        );
        let content_pos = evs
            .iter()
            .position(|e| e["choices"][0]["delta"]["content"].is_string())
            .unwrap();
        assert!(content_pos < evs.len() - 1);
    }

    #[test]
    fn content_and_finish_in_one_event_keep_their_order() {
        let last = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"end\"},\"finish_reason\":\"stop\"}]}\n\n";
        let evs = events(&run(&[&content_event("hello "), last]));
        let text: String = evs
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "hello end");
        let fin = evs.last().unwrap();
        assert_eq!(fin["choices"][0]["finish_reason"], Value::from("stop"));
        assert!(fin["choices"][0]["delta"]["content"].is_null());
    }

    #[test]
    fn a_closing_event_does_not_split_its_own_payload() {
        // 79 pending chars plus a final "!" with finish_reason must leave as
        // one 80-char event, not 79 + 1.
        let last = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n";
        let sizes = content_sizes(&events(&run(&[&content_event(&"y".repeat(79)), last])));
        assert_eq!(sizes, vec![80]);
    }

    #[test]
    fn residual_tool_call_fields_are_forwarded_in_order() {
        let frag = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ab\",\"parsed\":true}}]}}]}\n\n";
        let evs = events(&run(&[frag]));
        let args: String = evs
            .iter()
            .filter_map(|e| {
                e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .collect();
        assert_eq!(args, "ab");
        assert!(evs.iter().any(|e| {
            e["choices"][0]["delta"]["tool_calls"][0]["function"]["parsed"] == Value::Bool(true)
        }));
    }

    #[test]
    fn role_and_content_in_one_event_put_the_role_first() {
        let first = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"}}]}\n\n";
        let evs = events(&run(&[first, &content_event(" there")]));
        assert_eq!(
            evs[0]["choices"][0]["delta"]["role"],
            Value::from("assistant")
        );
        assert!(evs[0]["choices"][0]["delta"]["content"].is_null());
        let text: String = evs
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "Hi there");
    }

    #[test]
    fn repeated_roles_do_not_break_merging() {
        let frame = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ab\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n";
        let frames = [frame; 50];
        let evs = events(&run(&frames));
        let roles = evs
            .iter()
            .filter(|e| e["choices"][0]["delta"]["role"].is_string())
            .count();
        assert_eq!(roles, 1, "only the first role is forwarded");
        let sizes = content_sizes(&evs);
        assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn tool_call_identity_precedes_arguments() {
        let start = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}}]}}]}\n\n";
        let args = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"k\\\":1}\"}}]}}]}\n\n";
        let evs = events(&run(&[start, args]));
        let name_pos = evs
            .iter()
            .position(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["name"].is_string())
            .unwrap();
        let args_pos = evs
            .iter()
            .position(|e| {
                e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty())
            })
            .unwrap();
        assert!(name_pos < args_pos);
    }

    #[test]
    fn identity_and_arguments_in_one_event_stay_ordered() {
        let start = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n";
        let rest = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n";
        let evs = events(&run(&[start, rest]));
        assert_eq!(
            evs[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            Value::from("c1")
        );
        assert!(evs[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].is_null());
        let args: String = evs
            .iter()
            .filter_map(|e| {
                e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .collect();
        assert_eq!(args, "{\"a\":1}");
    }

    #[test]
    fn omitted_tool_index_is_not_invented() {
        let start = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}}]}\n\n";
        let frag = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n";
        let evs = events(&run(&[start, frag]));
        let args_event = evs
            .iter()
            .find(|e| {
                e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].is_string()
            })
            .unwrap();
        assert!(args_event["choices"][0]["delta"]["tool_calls"][0]["index"].is_null());
    }

    #[test]
    fn null_valued_fields_are_not_structural() {
        let frag = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":null,\"refusal\":null,\"tool_calls\":[{\"index\":0,\"function\":{\"name\":null,\"arguments\":\"ab\"}}]}}]}\n\n";
        let frames = [frag; 50];
        let evs = events(&run(&frames));
        let sizes: Vec<usize> = evs
            .iter()
            .filter_map(|e| {
                e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .map(str::len)
            .collect();
        assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 100);
        assert_eq!(
            evs.len(),
            sizes.len(),
            "no event was forwarded structurally"
        );
    }

    #[test]
    fn reasoning_is_flushed_before_content_starts() {
        let reasoning = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n";
        let evs = events(&run(&[reasoning, &content_event(&"c".repeat(100))]));
        let think_pos = evs
            .iter()
            .position(|e| e["choices"][0]["delta"]["reasoning_content"].is_string())
            .unwrap();
        let content_pos = evs
            .iter()
            .position(|e| e["choices"][0]["delta"]["content"].is_string())
            .unwrap();
        assert!(think_pos < content_pos);
    }

    #[test]
    fn keepalive_comments_do_not_flush() {
        let all = run(&[&content_event("ab"), ": ping\n\n", &content_event("cd")]);
        let text = String::from_utf8_lossy(&all);
        assert!(text.contains(": ping\n\n"));
        let sizes = content_sizes(&events(&all));
        assert_eq!(
            sizes,
            vec![4],
            "the two fragments were merged across the comment"
        );
    }

    #[test]
    fn multi_choice_streams_pass_through_verbatim() {
        let two = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}},{\"index\":1,\"delta\":{\"content\":\"b\"}}]}\n\n";
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(bytes(two)).as_ref());
        all.extend_from_slice(r.feed(bytes(two)).as_ref());
        all.extend_from_slice(r.finish().as_ref());
        assert_eq!(all, [two.as_bytes(), two.as_bytes()].concat());
    }

    #[test]
    fn legacy_completion_chunks_pass_through() {
        let text = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"text\":\"hi\",\"finish_reason\":null}]}\n\n";
        let all = run(&[text, text]);
        assert_eq!(all, [text.as_bytes(), text.as_bytes()].concat());
    }

    #[test]
    fn empty_choices_events_pass_through() {
        let opener = "data: {\"id\":\"x\",\"choices\":[],\"prompt_filter_results\":[1]}\n\n";
        let all = run(&[opener, &content_event(&"z".repeat(100))]);
        let evs = events(&all);
        assert_eq!(evs[0]["prompt_filter_results"], serde_json::json!([1]));
        assert!(evs[1]["prompt_filter_results"].is_null());
    }

    #[test]
    fn envelope_follows_the_latest_chunk() {
        let first = "data: {\"id\":\"x\",\"created\":1,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n";
        let second = "data: {\"id\":\"x\",\"created\":2,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}\n\n";
        let evs = events(&run(&[first, second]));
        assert_eq!(evs.last().unwrap()["created"], Value::from(2));
    }

    #[test]
    fn error_frames_do_not_become_the_envelope() {
        let error = "data: {\"error\":{\"message\":\"boom\"}}\n\n";
        let evs = events(&run(&[error, &content_event("hello")]));
        assert_eq!(evs[0]["error"]["message"], Value::from("boom"));
        assert!(evs[1]["error"].is_null());
    }

    #[test]
    fn unknown_delta_keys_are_forwarded_in_order() {
        let audio = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"audio\":{\"id\":\"a\"}}}]}\n\n";
        let evs = events(&run(&[&content_event("ab"), audio, &content_event("cd")]));
        let audio_pos = evs
            .iter()
            .position(|e| e["choices"][0]["delta"]["audio"].is_object())
            .unwrap();
        let texts: Vec<&str> = evs
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(texts, vec!["ab", "cd"]);
        assert_eq!(audio_pos, 1);
    }

    #[test]
    fn flush_pending_emits_what_is_buffered() {
        let mut r = SseRechunker::new();
        assert!(r.feed(bytes(content_event("hello"))).is_empty());
        assert!(r.has_pending());
        let evs = events(&r.flush_pending());
        assert_eq!(
            evs[0]["choices"][0]["delta"]["content"],
            Value::from("hello")
        );
        assert!(!r.has_pending());
    }

    #[test]
    fn flush_pending_holds_a_remainder_shorter_than_the_tiny_band() {
        let mut r = SseRechunker::new();
        assert!(r.feed(bytes(content_event("hi"))).is_empty());
        assert!(!r.has_pending(), "two characters are not worth an event");
        assert!(r.flush_pending().is_empty());
        assert!(r.feed(bytes(content_event(" there"))).is_empty());
        assert!(r.has_pending());
        let evs = events(&r.flush_pending());
        assert_eq!(
            evs[0]["choices"][0]["delta"]["content"],
            Value::from("hi there")
        );
        assert!(!r.has_pending());
        // The end of the stream still releases a tiny remainder.
        assert!(r.feed(bytes(content_event("ok"))).is_empty());
        let evs = events(&r.finish());
        assert_eq!(evs[0]["choices"][0]["delta"]["content"], Value::from("ok"));
    }

    /// The frames of a forced function call: the call's identity, then one
    /// `arguments` delta per decoded token.
    fn tool_identity_event() -> String {
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"functions.write_file:0\",\"type\":\"function\",\"function\":{\"name\":\"write_file\"}}]},\"logprobs\":null,\"finish_reason\":null}]}\n\n".to_string()
    }

    fn arguments_event(text: &str) -> String {
        format!(
            "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{}}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n",
            serde_json::to_string(text).unwrap()
        )
    }

    /// An upstream that repeats null identity fields on every piece.
    fn arguments_event_with_nulls(text: &str) -> String {
        format!(
            "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":null,\"content\":null,\"tool_calls\":[{{\"id\":null,\"index\":0,\"type\":\"function\",\"function\":{{\"name\":null,\"arguments\":{}}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n",
            serde_json::to_string(text).unwrap()
        )
    }

    /// An upstream that sends the identity together with the first piece.
    fn identity_with_arguments_event(text: &str) -> String {
        format!(
            "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{{\"name\":\"write_file\",\"arguments\":{}}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n",
            serde_json::to_string(text).unwrap()
        )
    }

    const FINISH_EVENT: &str = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"tool_calls\"}]}\n\n";

    fn argument_pieces(evs: &[Value]) -> Vec<String> {
        evs.iter()
            .filter_map(|e| {
                e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .map(str::to_owned)
            })
            .collect()
    }

    /// The tokens of `{"path":"a.txt","content":"<hex>"}` as a byte-level
    /// tokenizer cuts them: the JSON punctuation is one to three characters.
    fn forced_call_tokens() -> Vec<String> {
        let mut tokens: Vec<String> = [
            "{\"", "path", "\":\"", "a", ".txt", "\",\"", "content", "\":\"",
        ]
        .iter()
        .map(|t| (*t).to_string())
        .collect();
        tokens.extend((0..40).map(|i| format!("{i:04x}")));
        tokens.push("\"}".to_string());
        tokens
    }

    /// Streams `frames` with the idle timer firing after every one of them,
    /// as it does when the engine pauses between tokens.
    fn run_with_pauses(frames: &[String]) -> Vec<u8> {
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        for frame in frames {
            all.extend_from_slice(r.feed(bytes(frame)).as_ref());
            all.extend_from_slice(r.flush_pending().as_ref());
        }
        all.extend_from_slice(r.feed(bytes(FINISH_EVENT)).as_ref());
        all.extend_from_slice(r.finish().as_ref());
        all
    }

    fn assert_no_tiny_pieces_before_the_tail(all: &[u8], tokens: &[String]) {
        let pieces = argument_pieces(&events(all));
        let sizes: Vec<usize> = pieces.iter().map(|p| p.chars().count()).collect();
        let (tail, body) = sizes.split_last().expect("some argument piece");
        assert!(body.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
        assert!(*tail >= 1);
        assert_eq!(pieces.concat(), tokens.concat());
    }

    #[test]
    fn an_engine_pausing_between_tokens_does_not_make_tiny_argument_events() {
        let tokens = forced_call_tokens();
        let mut frames = vec![tool_identity_event()];
        frames.extend(tokens.iter().map(|t| arguments_event(t)));
        assert_no_tiny_pieces_before_the_tail(&run_with_pauses(&frames), &tokens);
    }

    #[test]
    fn null_identity_fields_on_every_piece_still_merge_across_pauses() {
        let tokens = forced_call_tokens();
        let mut frames = vec![tool_identity_event()];
        frames.extend(tokens.iter().map(|t| arguments_event_with_nulls(t)));
        assert_no_tiny_pieces_before_the_tail(&run_with_pauses(&frames), &tokens);
    }

    #[test]
    fn identity_sent_with_the_first_piece_still_merges_across_pauses() {
        let tokens = forced_call_tokens();
        let mut frames = vec![identity_with_arguments_event(&tokens[0])];
        frames.extend(tokens[1..].iter().map(|t| arguments_event(t)));
        let all = run_with_pauses(&frames);
        let evs = events(&all);
        assert_eq!(
            evs[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            Value::from("write_file")
        );
        assert_no_tiny_pieces_before_the_tail(&all, &tokens);
    }

    #[test]
    fn done_passthrough_after_flush() {
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(bytes(content_event("hi"))).as_ref());
        all.extend_from_slice(r.feed(bytes(b"data: [DONE]\n\n")).as_ref());
        let text = String::from_utf8_lossy(&all).to_string();
        assert!(text.find("\"hi\"").unwrap() < text.find("[DONE]").unwrap());
        assert_eq!(r.feed(bytes(b"trailing")).as_ref(), b"trailing");
    }

    fn chunk(text: &str) -> Bytes {
        Bytes::from(content_event(text))
    }

    fn contents(frames: &[Bytes]) -> Vec<String> {
        let mut all = Vec::new();
        for frame in frames {
            all.extend_from_slice(frame);
        }
        events(&all)
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .map(str::to_string)
            .collect()
    }

    #[tokio::test(start_paused = true)]
    async fn stream_adapter_merges_tiny_deltas_and_flushes_the_tail() {
        let upstream = stream::iter((0..100).map(|_| Ok::<_, io::Error>(chunk("ab"))));
        let frames: Vec<Bytes> = rechunk_stream(upstream, NoopObserver)
            .map(|frame| frame.unwrap())
            .collect()
            .await;
        let sizes: Vec<usize> = contents(&frames)
            .iter()
            .map(|c| c.chars().count())
            .collect();
        assert!(
            sizes.iter().all(|&s| s >= MIN_TAIL_CHARS),
            "sizes: {sizes:?}"
        );
        assert_eq!(sizes.iter().sum::<usize>(), 200);
    }

    #[tokio::test(start_paused = true)]
    async fn stream_adapter_flushes_pending_payload_when_upstream_idles() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
        let mut out = Box::pin(rechunk_stream(ReceiverStream::new(rx), NoopObserver));
        for _ in 0..3 {
            tx.send(Ok(chunk("ab"))).await.unwrap();
        }
        // Below the emit threshold: only the idle timer releases it.
        let frame = out.next().await.unwrap().unwrap();
        assert_eq!(contents(&[frame]), vec!["ababab"]);
        drop(tx);
        assert!(out.next().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn stream_adapter_holds_a_tiny_remainder_across_an_upstream_pause() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
        let mut out = Box::pin(rechunk_stream(ReceiverStream::new(rx), NoopObserver));
        tx.send(Ok(chunk("ab"))).await.unwrap();
        // Quiet for far longer than the idle flush: nothing worth an event yet.
        assert!(tokio::time::timeout(IDLE_FLUSH * 4, out.next())
            .await
            .is_err());
        tx.send(Ok(chunk("cde"))).await.unwrap();
        let frame = out.next().await.unwrap().unwrap();
        assert_eq!(contents(&[frame]), vec!["abcde"]);
        // The end of the stream releases what the idle flush held.
        tx.send(Ok(chunk("f"))).await.unwrap();
        drop(tx);
        let frame = out.next().await.unwrap().unwrap();
        assert_eq!(contents(&[frame]), vec!["f"]);
        assert!(out.next().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn stream_adapter_delivers_the_tail_before_an_upstream_error() {
        let upstream = stream::iter(vec![Ok(chunk("ab")), Err(io::Error::other("boom"))]);
        let mut out = Box::pin(rechunk_stream(upstream, NoopObserver));
        let tail = out.next().await.unwrap().unwrap();
        assert_eq!(contents(&[tail]), vec!["ab"]);
        assert!(out.next().await.unwrap().is_err());
        assert!(out.next().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn stream_adapter_ends_cleanly_when_the_veto_lands_on_the_tail() {
        // The upstream errors with text still held; the observer vetoes the
        // tail's slice, so the policy's error frame is the last thing the
        // client sees and the upstream's error is not yielded after it.
        let upstream = stream::iter(vec![Ok(chunk("ab")), Err(io::Error::other("boom"))]);
        let mut out = Box::pin(rechunk_stream(upstream, Watcher::vetoing_after(0)));
        let frame = out.next().await.unwrap().unwrap();
        assert_eq!(frame.as_ref(), POLICY_ERROR_FRAME.as_bytes());
        assert!(out.next().await.is_none());
    }

    /// What an observer was shown, in order.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Seen {
        Open,
        Slice(DeltaField, String),
        End,
    }

    /// Records what it is shown, and lets `allow` slices out before it ends
    /// the stream on the next one.
    #[derive(Clone)]
    struct Watcher {
        seen: Arc<Mutex<Vec<Seen>>>,
        allow: usize,
    }

    impl Watcher {
        fn recording() -> Self {
            Self::vetoing_after(usize::MAX)
        }

        fn vetoing_after(allow: usize) -> Self {
            Self {
                seen: Arc::default(),
                allow,
            }
        }

        fn seen(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl DeltaObserver for Watcher {
        fn open(&mut self) {
            self.seen.lock().unwrap().push(Seen::Open);
        }

        fn on_slice(&mut self, slice: DeltaSlice<'_>) -> Verdict {
            self.seen
                .lock()
                .unwrap()
                .push(Seen::Slice(slice.field, slice.text.to_owned()));
            if self.allow == 0 {
                return Verdict::Terminate(policy_stop());
            }
            self.allow -= 1;
            Verdict::Continue
        }

        fn end(&mut self) {
            self.seen.lock().unwrap().push(Seen::End);
        }
    }

    fn policy_stop() -> StreamStop {
        StreamStop {
            message: "the response was ended by a policy".to_owned(),
            error_type: "policy_violation".to_owned(),
        }
    }

    const POLICY_ERROR_FRAME: &str = "data: {\"error\":{\"message\":\"the response was ended by a policy\",\"type\":\"policy_violation\"}}\n\n";

    /// The slices of the emitted events, in order: what the client sees.
    fn emitted_slices(all: &[u8]) -> Vec<Seen> {
        events(all)
            .iter()
            .filter_map(|event| {
                let delta = &event["choices"][0]["delta"];
                for field in PAYLOAD_FIELDS {
                    if let Some(text) = delta[field].as_str() {
                        return Some(Seen::Slice(DeltaField::Text(field), text.to_owned()));
                    }
                }
                let call = &delta["tool_calls"][0];
                call["function"]["arguments"]
                    .as_str()
                    .filter(|text| !text.is_empty())
                    .map(|text| {
                        Seen::Slice(
                            DeltaField::ToolArguments(call["index"].as_u64()),
                            text.to_owned(),
                        )
                    })
            })
            .collect()
    }

    /// A stream with every kind of slice: a role, reasoning in tokens, one
    /// large content delta, a tool call's identity and its arguments in
    /// pieces, then the finish.
    fn mixed_stream() -> Vec<String> {
        let mut frames = vec![
            "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"
                .to_string(),
        ];
        frames.extend((0..60).map(|_| {
            "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"th\"}}]}\n\n"
                .to_string()
        }));
        frames.push(content_event(&"c".repeat(400)));
        frames.push(tool_identity_event());
        frames.extend((0..30).map(|i| arguments_event(&format!("{{\"k{i}\":1}}"))));
        frames.push(FINISH_EVENT.to_string());
        frames
    }

    fn run_watched(frames: &[String], watcher: Watcher) -> Vec<u8> {
        let mut r = SseRechunker::with_observer(watcher);
        let mut all = Vec::new();
        for frame in frames {
            all.extend_from_slice(r.feed(bytes(frame)).as_ref());
        }
        all.extend_from_slice(r.finish().as_ref());
        all
    }

    #[test]
    fn the_observer_sees_every_slice_in_the_order_the_client_does() {
        let watcher = Watcher::recording();
        let all = run_watched(&mixed_stream(), watcher.clone());
        let mut expected = vec![Seen::Open];
        expected.extend(emitted_slices(&all));
        expected.push(Seen::End);
        let seen = watcher.seen();
        assert_eq!(seen, expected);
        // Every kind of slice was in it, more than one of each, and the text
        // of a field came through whole.
        let of = |field: DeltaField| -> Vec<&str> {
            seen.iter()
                .filter_map(|s| match s {
                    Seen::Slice(f, text) if *f == field => Some(text.as_str()),
                    _ => None,
                })
                .collect()
        };
        let reasoning = of(DeltaField::Text("reasoning_content"));
        let content = of(DeltaField::Text("content"));
        let arguments = of(DeltaField::ToolArguments(Some(0)));
        assert!(reasoning.len() > 1 && content.len() > 1 && arguments.len() > 1);
        assert_eq!(reasoning.concat(), "th".repeat(60));
        assert_eq!(content.concat(), "c".repeat(400));
        let pieces: String = (0..30).map(|i| format!("{{\"k{i}\":1}}")).collect();
        assert_eq!(arguments.concat(), pieces);
    }

    #[test]
    fn an_observer_that_never_objects_leaves_the_bytes_as_they_are() {
        let frames = mixed_stream();
        let refs: Vec<&str> = frames.iter().map(String::as_str).collect();
        assert_eq!(run_watched(&frames, Watcher::recording()), run(&refs));
    }

    #[test]
    fn a_vetoing_observer_ends_the_stream_with_the_error_event_and_nothing_after_it() {
        let watcher = Watcher::vetoing_after(2);
        let mut r = SseRechunker::with_observer(watcher.clone());
        // 500 characters: three slices are ready at once, the third is vetoed.
        let all = r.feed(bytes(content_event(&"a".repeat(500))));
        let text = String::from_utf8(all.to_vec()).unwrap();
        let evs = events(&all);
        assert_eq!(content_sizes(&evs), vec![160, 160]);
        assert_eq!(evs.len(), 3, "{text}");
        assert_eq!(
            evs[2],
            serde_json::json!({"error": {
                "message": "the response was ended by a policy",
                "type": "policy_violation",
            }})
        );
        assert!(text.ends_with(POLICY_ERROR_FRAME), "{text}");
        assert_eq!(r.stopped(), Some(&policy_stop()));
        // The vetoed slice and the remainder never leave, and neither does
        // anything the upstream sends afterwards: no finish, no [DONE].
        assert!(!r.has_pending());
        assert!(r.flush_pending().is_empty());
        assert!(r.feed(bytes(content_event("more"))).is_empty());
        assert!(r.feed(bytes(FINISH_EVENT)).is_empty());
        assert!(r.feed(bytes(b"data: [DONE]\n\n")).is_empty());
        assert!(r.finish().is_empty());
        // The observer saw the slice it vetoed and nothing after it; the
        // stream was its to end, so there is no end call.
        let seen = watcher.seen();
        assert_eq!(seen.len(), 4, "{seen:?}");
        assert_eq!(seen[0], Seen::Open);
        assert!(matches!(&seen[3], Seen::Slice(DeltaField::Text("content"), t) if t.len() == 160));
    }

    #[test]
    fn a_veto_on_the_tail_replaces_it_with_the_error_event() {
        let watcher = Watcher::vetoing_after(0);
        let mut r = SseRechunker::with_observer(watcher.clone());
        // Held back: too short for a slice until the stream ends.
        assert!(r.feed(bytes(content_event("hi"))).is_empty());
        assert_eq!(r.finish().as_ref(), POLICY_ERROR_FRAME.as_bytes());
        assert_eq!(
            watcher.seen(),
            vec![
                Seen::Open,
                Seen::Slice(DeltaField::Text("content"), "hi".to_owned())
            ]
        );
    }

    #[test]
    fn a_veto_on_arguments_ends_the_stream_too() {
        let mut r = SseRechunker::with_observer(Watcher::vetoing_after(0));
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(bytes(tool_identity_event())).as_ref());
        all.extend_from_slice(r.feed(bytes(arguments_event(&"x".repeat(100)))).as_ref());
        let evs = events(&all);
        assert_eq!(evs.len(), 2, "{}", String::from_utf8_lossy(&all));
        assert_eq!(
            evs[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            Value::from("write_file")
        );
        assert_eq!(evs[1]["error"]["type"], Value::from("policy_violation"));
        assert!(r.feed(bytes(FINISH_EVENT)).is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn stream_adapter_ends_on_the_observers_stop_and_lets_the_upstream_go() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
        let mut out = Box::pin(rechunk_stream(
            ReceiverStream::new(rx),
            Watcher::vetoing_after(0),
        ));
        tx.send(Ok(chunk(&"a".repeat(100)))).await.unwrap();
        let frame = out.next().await.unwrap().unwrap();
        assert_eq!(frame.as_ref(), POLICY_ERROR_FRAME.as_bytes());
        // The stream ends while the upstream is still sending, and ending it
        // lets the upstream go: the sender finds its reader gone, and what it
        // sends next has nowhere to go.
        assert!(out.next().await.is_none());
        assert!(tx.is_closed());
        assert!(tx.send(Ok(chunk("more"))).await.is_err());
    }
}
