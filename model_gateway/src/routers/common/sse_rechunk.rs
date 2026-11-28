//! Re-slicing of streamed chat completion deltas.
//!
//! A wire whose contract bounds the size of every streamed delta (MiniMax's
//! provider verifier does: `routers::grpc::wire::minimax`) hands this stage
//! its numbers as a [`RechunkPolicy`]. Engines emit one token per event and
//! tool arguments in one piece, so the gateway buffers `content`,
//! `reasoning_content`, `reasoning` and tool-call `arguments` and emits them
//! again in slices of the policy's size. Text is only held back briefly: the
//! relays flush it when the upstream has been quiet for the policy's idle
//! flush, except a remainder shorter than its minimum tail (a token or two
//! that arrived just before the engine paused), which waits for the next
//! text or the end of the stream: an event of its own would be exactly the
//! tiny delta the contract forbids. Tool-call arguments keep that tail back
//! on every cut and idle flush as well, until the stream moves on (another
//! call opens, text of another field, the end):
//! a call's closing punctuation is a token of one to four characters that
//! may arrive on its own after a pause, and it rides with the held tail
//! instead of being the only thing left when the next call opens.
//!
//! Only single-choice chat completion streams are re-sliced. Everything else
//! is forwarded as it came, after any buffered text so the order never
//! changes: comments, `[DONE]`, malformed frames, events with several
//! choices, events carrying `logprobs` or fields this module does not know.
//! A `usage: null` placeholder on the latest event (what a stream under
//! `include_usage` writes on every chunk before the usage-only one; a closing
//! event's usage object leaves the earlier text's placeholder in place) rides
//! each synthesized slice, after its choices, so the re-sliced stream keeps
//! the public shape.
//!
//! The HTTP relay drives [`SseRechunker`] inline; the gRPC pipeline wraps
//! its own channel in `rechunk_stream`.
//!
//! Every slice the stage emits, text or tool-call arguments, is shown to a
//! [`DeltaObserver`] first, in the pieces the client will see and on both
//! relays. What the stage forwards whole passes the observer by: events with
//! several choices or a later choice index (the stream is forwarded as it
//! came from there on), events carrying `logprobs` or other choice-level
//! metadata, frames after `[DONE]` or larger than `MAX_FRAME_BYTES`, and
//! malformed frames; and the relays build the stage only for the streams
//! whose wire asks for re-slicing (`routers::grpc::wire::Wire::rechunk_policy`,
//! the wire of the model the client asked for). The default, [`NoopObserver`],
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

/// The numbers of a wire's delta-size contract, owned by the wire that asks
/// for re-slicing (`routers::grpc::wire::Wire::rechunk_policy`) and handed
/// to the stage when it is built. Slices hold at most `slice_chars`
/// characters and are cut once at least `emit_threshold` are buffered. A cut
/// never leaves a tail shorter than `min_tail_chars`, and an idle flush
/// holds back a remainder shorter than it, so only a delta that is tiny in
/// total yields a tiny event. A tool call's arguments keep `min_tail_chars`
/// back on every cut and idle flush as well, until the stream moves on: the
/// call's closer, which may arrive on its own after a pause, then rides with
/// them. Their cut waits for `min_tail_chars` more than the threshold, so it
/// still emits at least `emit_threshold` and the pieces keep their size. The
/// held tail is a whole tail, not one short of it, because the stream may
/// move on with nothing following and release it alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RechunkPolicy {
    pub slice_chars: usize,
    pub emit_threshold: usize,
    pub min_tail_chars: usize,
    /// Buffered text is flushed after this much upstream silence.
    pub idle_flush: Duration,
}

/// An unterminated frame longer than this is forwarded as it comes.
pub(crate) const MAX_FRAME_BYTES: usize = 1 << 20;

/// Delta fields whose text is buffered, in the order they are emitted.
const PAYLOAD_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "content"];

/// The top-level key of the rendered prompt a request asked back
/// (`return_prompt_text`). It is the payload of the event that brings it, not
/// part of the envelope the synthesized events repeat: it goes out once, with
/// that event written whole or with the opening half of that event split.
const PROMPT_TEXT: &str = "prompt_text";

/// Re-slices the deltas of one SSE stream: feed it the body chunks in order
/// and forward what it returns.
pub struct SseRechunker<O = NoopObserver> {
    /// The wire's numbers.
    policy: RechunkPolicy,
    /// Start of a frame whose delimiter has not arrived yet.
    partial: Vec<u8>,
    /// Bytes of `partial` already searched for a delimiter.
    scanned: usize,
    /// Top-level fields of the latest event except `choices`, `usage` and
    /// `prompt_text` (the rendered prompt a request asked back, which the
    /// event that brought it carries alone), serialized as the start of an
    /// object (`{"id":"x",`), so synthesized events look like the upstream
    /// ones.
    envelope: Vec<u8>,
    /// The latest text-bearing event carried the `usage: null` placeholder
    /// (a usage object, which closes the stream, leaves the flag as it was):
    /// every synthesized slice writes the placeholder after its choices.
    usage_null: bool,
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

impl SseRechunker {
    /// A stage nobody observes, re-slicing by `policy`.
    pub fn new(policy: RechunkPolicy) -> Self {
        Self::with_observer(policy, NoopObserver)
    }
}

impl<O: DeltaObserver> SseRechunker<O> {
    /// A stage re-slicing by `policy` whose slices `observer` sees; it is
    /// told the stream opened.
    pub fn with_observer(policy: RechunkPolicy, mut observer: O) -> Self {
        observer.open();
        Self {
            policy,
            partial: Vec::new(),
            scanned: 0,
            envelope: b"{".to_vec(),
            usage_null: false,
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

    /// The numbers this stage re-slices by.
    pub fn policy(&self) -> RechunkPolicy {
        self.policy
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
    /// least the policy's minimum tail, or a tool call's arguments of at
    /// least that beyond the tail they keep. A shorter remainder waits for
    /// more text.
    pub fn has_pending(&self) -> bool {
        let tail = self.policy.min_tail_chars;
        self.fields.iter().any(|pending| pending.chars >= tail)
            || self
                .tool_args
                .values()
                .any(|pending| pending.chars >= 2 * tail)
    }

    /// Emits the buffered text without waiting for more input, except a
    /// remainder shorter than the policy's minimum tail and the tail a tool
    /// call's arguments keep, which stay for the next text or
    /// [`Self::finish`]: an engine that pauses between tokens must not turn
    /// every token into an event of its own.
    pub fn flush_pending(&mut self) -> Bytes {
        let mut out = Vec::new();
        let tail = self.policy.min_tail_chars;
        self.flush_from(tail, tail, &mut out);
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
        let mut from = searched.saturating_sub(DELIMITER_OVERLAP);
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
        let brings_prompt = choices.len() == 1 && self.update_envelope(event);

        let structural = Structural {
            event,
            choice: choice.as_ref(),
            delta: delta.as_ref(),
            shape: &shape,
        };
        // Text already buffered goes out before an event that opens
        // something; an event that closes something goes out after its own
        // text, so nothing is split.
        // An event that brings the rendered prompt and would leave as
        // synthesized text alone goes out as an opening one: an event of
        // its own carries the key, ahead of the event's text.
        let opens = shape.opens || (brings_prompt && !shape.closes);
        match (opens, shape.closes) {
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

    /// Keeps the envelope, and the placeholder flag, in step with the latest
    /// event. A usage object leaves the flag as it was: that event closes the
    /// stream and goes out whole after the flush, so the flag keeps describing
    /// the text buffered before it. Returns whether the event brought the
    /// rendered prompt (`prompt_text`), which stays out of the envelope.
    fn update_envelope(&mut self, event: &RawObject<'_>) -> bool {
        self.scratch.clear();
        self.scratch.push(b'{');
        let mut usage_null = Some(false);
        let mut brings_prompt = false;
        for (key, value) in &event.entries {
            if key == "choices" {
                continue;
            }
            if key == PROMPT_TEXT {
                brings_prompt = true;
                continue;
            }
            if key == "usage" {
                usage_null = is_null(value).then_some(true);
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
        if let Some(usage_null) = usage_null {
            self.usage_null = usage_null;
        }
        brings_prompt
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
        let policy = self.policy;
        for (field, pending) in PAYLOAD_FIELDS.iter().zip(&mut self.fields) {
            while pending.chars >= policy.emit_threshold {
                emit_text(
                    policy,
                    &self.envelope,
                    self.usage_null,
                    &mut self.observer,
                    field,
                    pending,
                    out,
                )?;
            }
        }
        for (index, pending) in &mut self.tool_args {
            // The cut waits for the held tail on top of the threshold, so the
            // pieces of a steady stream keep their size.
            while pending.chars >= policy.emit_threshold + policy.min_tail_chars {
                emit_arguments(
                    policy,
                    &self.envelope,
                    self.usage_null,
                    &mut self.observer,
                    *index,
                    pending,
                    policy.min_tail_chars,
                    out,
                )?;
            }
        }
        Ok(())
    }

    /// Emits every buffered slice: the stream moves on.
    fn flush(&mut self, out: &mut Vec<u8>) {
        self.flush_from(1, 0, out);
    }

    /// Emits the buffers holding at least `min_chars` characters, all but
    /// the last `held` of a tool call's arguments (a buffer with fewer than
    /// `min_chars` beyond them stays whole).
    fn flush_from(&mut self, min_chars: usize, held: usize, out: &mut Vec<u8>) {
        if let Err(stop) = self.emit_from(min_chars, held, out) {
            self.stop(stop, out);
        }
    }

    fn emit_from(
        &mut self,
        min_chars: usize,
        held: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), StreamStop> {
        let policy = self.policy;
        for (field, pending) in PAYLOAD_FIELDS.iter().zip(&mut self.fields) {
            if pending.chars >= min_chars {
                while pending.chars > 0 {
                    emit_text(
                        policy,
                        &self.envelope,
                        self.usage_null,
                        &mut self.observer,
                        field,
                        pending,
                        out,
                    )?;
                }
            }
        }
        for (index, pending) in &mut self.tool_args {
            if pending.chars >= min_chars + held {
                while pending.chars > held {
                    emit_arguments(
                        policy,
                        &self.envelope,
                        self.usage_null,
                        &mut self.observer,
                        *index,
                        pending,
                        held,
                        out,
                    )?;
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
    /// and a usage object is left out (the `usage: null` placeholder stays).
    Opening,
    /// The event with an empty delta, and without the rendered prompt
    /// (`prompt_text`): the opening half carried it.
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
            ("usage", _) if variant == Variant::Opening && !is_null(value) => {}
            (PROMPT_TEXT, _) if variant == Variant::Closing => {}
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
    policy: RechunkPolicy,
    envelope: &[u8],
    usage_null: bool,
    observer: &mut O,
    field: &'static str,
    pending: &mut Pending,
    out: &mut Vec<u8>,
) -> Result<(), StreamStop> {
    let (bytes, chars) = pending.next_slice(policy, 0);
    observe(observer, DeltaField::Text(field), &pending.text[..bytes])?;
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(envelope);
    out.extend_from_slice(b"\"choices\":[{\"index\":0,\"delta\":{");
    write_json(out, field);
    out.push(b':');
    write_json(out, &pending.text[..bytes]);
    out.extend_from_slice(b"},\"finish_reason\":null}]");
    close_synthesized(usage_null, out);
    pending.cut(bytes, chars);
    Ok(())
}

/// Writes one synthesized event with the next slice of `pending` as the
/// arguments of tool call `index`, unless the observer ends the stream on it.
/// The slice leaves `keep` characters in the buffer.
// The stage's numbers and the latest event's frame (envelope, placeholder)
// ride with the observer, the call and the slice's target: one over clippy's
// seven.
#[expect(clippy::too_many_arguments)]
fn emit_arguments<O: DeltaObserver>(
    policy: RechunkPolicy,
    envelope: &[u8],
    usage_null: bool,
    observer: &mut O,
    index: Option<u64>,
    pending: &mut Pending,
    keep: usize,
    out: &mut Vec<u8>,
) -> Result<(), StreamStop> {
    let (bytes, chars) = pending.next_slice(policy, keep);
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
    out.extend_from_slice(b"}}]},\"finish_reason\":null}]");
    close_synthesized(usage_null, out);
    pending.cut(bytes, chars);
    Ok(())
}

/// Ends a synthesized event after its choices: the `usage: null` placeholder
/// when the latest event carried one (see `SseRechunker::usage_null`), then
/// the frame's end.
fn close_synthesized(usage_null: bool, out: &mut Vec<u8>) {
    if usage_null {
        out.extend_from_slice(b",\"usage\":null");
    }
    out.extend_from_slice(b"}\n\n");
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

    /// Byte and character length of the next slice: at most the policy's
    /// slice, leaving `keep` characters (fewer than are buffered) and never
    /// a tail shorter than the policy's minimum beyond them.
    fn next_slice(&self, policy: RechunkPolicy, keep: usize) -> (usize, usize) {
        let free = self.chars - keep;
        let chars = if free <= policy.slice_chars {
            free
        } else if free - policy.slice_chars < policy.min_tail_chars {
            free - policy.min_tail_chars
        } else {
            policy.slice_chars
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

/// Bytes of a frame delimiter that may already sit before the end of a
/// scanned prefix: the four-byte `\r\n\r\n` less its last byte. A scan that
/// resumes this far back finds a delimiter the boundary split.
pub(crate) const DELIMITER_OVERLAP: usize = 3;

/// Position and length of the first frame delimiter at or after `from`:
/// `\r\n\r\n`, `\n\n` or `\r\r`. The decode side's log-probability merge
/// splits its chunks into frames with it too.
pub(crate) fn find_frame_end(buf: &[u8], from: usize) -> Option<(usize, usize)> {
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
/// [`SseRechunker`] re-slicing by `policy` and watched by `observer`, text
/// held back is flushed once the upstream has been quiet for the policy's
/// idle flush, and the tail is flushed
/// when the stream ends. An upstream error still follows whatever text was
/// pending. A stream the observer stops ends with its error event, and the
/// upstream is let go: on this relay, that is what cancels the engine's
/// work.
pub(crate) fn rechunk_stream<S, E, O>(
    policy: RechunkPolicy,
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
        idle_flush: Duration,
        idle: Pin<Box<tokio::time::Sleep>>,
        /// An upstream error, delivered after the flushed tail.
        deferred: Option<E>,
        ended: bool,
    }

    let state = State {
        inner,
        rechunker: SseRechunker::with_observer(policy, observer),
        idle_flush: policy.idle_flush,
        idle: Box::pin(tokio::time::sleep(policy.idle_flush)),
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
                        st.idle.as_mut().reset(tokio::time::Instant::now() + st.idle_flush);
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
                    st.idle.as_mut().reset(tokio::time::Instant::now() + st.idle_flush);
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
#[path = "../../../tests/unit/routers/common/sse_rechunk/tests.rs"]
mod tests;
