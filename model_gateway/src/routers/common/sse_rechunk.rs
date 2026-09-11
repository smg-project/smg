//! SSE delta re-chunking for provider stream-QoS contracts.
//!
//! MiniMax's vendor verifier bounds the per-event payload-size distribution
//! (m3_stream_tests: ≤5–15% events of 1–4 chars, ≤0–2% events over 200 chars,
//! bounded large-event character share). Upstreams routinely violate both
//! directions — token-sized dribbles and multi-kilobyte argument dumps — so
//! the relay absorbs delta payload strings into per-field buffers and
//! re-emits events sized between the tiny and large bounds. Structural events
//! (role, tool-call identity, finish_reason, usage, shapes this module does
//! not merge) flush pending payload first and pass through in order.
//!
//! Re-chunking trades a little time to first token for packet-size
//! compliance; the relay flushes pending payload when the upstream goes idle.
//! Single-choice chat streams only: a second choice, or `[DONE]`, switches
//! the rest of the stream to byte pass-through.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde_json::{Map, Value};

/// Emit buffered payload in slices of this many chars once at least
/// `EMIT_THRESHOLD` chars are pending; both sit safely inside the contract's
/// normal band of 5–200 chars per event.
const SLICE_CHARS: usize = 160;
const EMIT_THRESHOLD: usize = 80;
/// A split never leaves a tail shorter than this, so only a payload that is
/// tiny in total can produce a tiny event.
const MIN_TAIL_CHARS: usize = 5;

/// Delta string fields subject to re-chunking, in emission order.
const PAYLOAD_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "content"];

/// One re-chunked payload buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Buffer {
    Field(&'static str),
    /// `None` when the upstream sent no `index`; the fragment is then emitted
    /// without one rather than attributed to call 0.
    ToolArgs(Option<u64>),
}

/// What one upstream event carried besides its payload.
struct Absorbed {
    /// Announces something later payload attaches to (role, tool identity).
    opens: bool,
    /// Must follow every earlier payload (finish_reason, usage, unknown keys).
    closes: bool,
    payload: Vec<(Buffer, String)>,
}

#[derive(Default)]
pub struct SseRechunker {
    raw: Vec<u8>,
    /// Envelope of the latest chat chunk (top-level fields minus choices/usage).
    envelope: Map<String, Value>,
    /// Buffered payload per delta string field.
    fields: BTreeMap<&'static str, String>,
    /// Buffered tool-call `arguments` per tool index.
    tool_args: BTreeMap<Option<u64>, String>,
    /// The buffer that received payload last; switching flushes it first.
    last: Option<Buffer>,
    role_sent: bool,
    /// Forward bytes verbatim from here on.
    passthrough: bool,
    /// Events forwarded structurally because of delta keys this module does not merge.
    unknown_events: usize,
}

impl SseRechunker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one upstream body chunk, returning the bytes to forward now.
    pub fn feed(&mut self, chunk: &[u8]) -> Bytes {
        if self.passthrough {
            return Bytes::copy_from_slice(chunk);
        }
        self.raw.extend_from_slice(chunk);
        let raw = std::mem::take(&mut self.raw);
        let mut out = Vec::new();
        let mut cursor = 0;
        while let Some((end, delimiter)) = find_frame_end(&raw[cursor..]) {
            let frame_end = cursor + end + delimiter;
            self.handle_frame(&raw[cursor..frame_end], &mut out);
            cursor = frame_end;
            if self.passthrough {
                out.extend_from_slice(&raw[cursor..]);
                cursor = raw.len();
                break;
            }
        }
        self.raw = raw[cursor..].to_vec();
        Bytes::from(out)
    }

    /// Whether payload is waiting for more input before it is emitted.
    pub fn has_pending(&self) -> bool {
        self.fields.values().any(|s| !s.is_empty())
            || self.tool_args.values().any(|s| !s.is_empty())
    }

    /// Emit everything buffered without waiting for more input (idle upstream).
    pub fn flush_pending(&mut self) -> Bytes {
        let mut out = Vec::new();
        self.flush_payload(&mut out);
        Bytes::from(out)
    }

    /// Flush everything buffered (stream end).
    pub fn finish(&mut self) -> Bytes {
        let mut out = Vec::new();
        self.flush_payload(&mut out);
        if !self.raw.is_empty() {
            out.extend_from_slice(&self.raw);
            self.raw.clear();
        }
        if self.unknown_events > 0 {
            tracing::debug!(
                events = self.unknown_events,
                "SSE re-chunking forwarded events with delta keys it does not merge"
            );
        }
        Bytes::from(out)
    }

    fn handle_frame(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        let mut data: Option<&[u8]> = None;
        let mut data_lines = 0usize;
        let mut other_lines = false;
        for line in frame.split(|b| *b == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
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
        let Some(data) = data.filter(|_| data_lines == 1 && !other_lines) else {
            if data_lines == 0 {
                // Comments and keep-alive frames carry no payload: forward, no flush.
                out.extend_from_slice(frame);
            } else {
                // Named events or multi-line data stay in order.
                self.flush_payload(out);
                out.extend_from_slice(frame);
            }
            return;
        };
        if data.starts_with(b"[DONE]") {
            self.flush_payload(out);
            out.extend_from_slice(frame);
            self.passthrough = true;
            return;
        }
        let Ok(Value::Object(mut event)) = serde_json::from_slice::<Value>(data) else {
            self.flush_payload(out);
            out.extend_from_slice(frame);
            return;
        };
        if !is_single_choice(&event) {
            // The buffers are per stream, not per choice.
            self.flush_payload(out);
            out.extend_from_slice(frame);
            self.passthrough = true;
            return;
        }

        let Absorbed {
            opens,
            closes,
            payload,
        } = self.absorb(&mut event);
        if event
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|c| c.len() == 1)
        {
            let mut envelope = event.clone();
            envelope.remove("choices");
            envelope.remove("usage");
            self.envelope = envelope;
        }

        if opens || closes {
            self.flush_payload(out);
        }
        match (opens, closes) {
            (true, true) => {
                let (opening, closing) = split_open_close(event);
                write_event(&Value::Object(opening), out);
                self.push_payload(payload, out);
                self.flush_payload(out);
                write_event(&Value::Object(closing), out);
            }
            (true, false) => {
                write_event(&Value::Object(event), out);
                self.push_payload(payload, out);
                self.drain_ready(out);
            }
            (false, true) => {
                self.push_payload(payload, out);
                self.flush_payload(out);
                write_event(&Value::Object(event), out);
            }
            (false, false) => {
                self.push_payload(payload, out);
                self.drain_ready(out);
            }
        }
    }

    /// Pull payload strings out of the event's delta, leaving the parts that
    /// must be forwarded in order.
    fn absorb(&mut self, event: &mut Map<String, Value>) -> Absorbed {
        let mut absorbed = Absorbed {
            opens: false,
            closes: event.get("usage").is_some_and(|u| !u.is_null()),
            payload: Vec::new(),
        };
        let Some(Value::Array(choices)) = event.get_mut("choices") else {
            absorbed.closes = true;
            return absorbed;
        };
        let Some(Value::Object(choice)) = choices.first_mut() else {
            absorbed.closes = true;
            return absorbed;
        };
        if choice.get("finish_reason").is_some_and(|f| !f.is_null()) {
            absorbed.closes = true;
        }
        let Some(Value::Object(delta)) = choice.get_mut("delta") else {
            // Not a chat chunk (legacy completions `text`, vendor shapes).
            absorbed.closes = true;
            return absorbed;
        };

        for field in PAYLOAD_FIELDS {
            match delta.get(field) {
                Some(Value::String(s)) => {
                    if !s.is_empty() {
                        absorbed.payload.push((Buffer::Field(field), s.clone()));
                    }
                    delta.remove(field);
                }
                Some(Value::Null) => {
                    delta.remove(field);
                }
                _ => {}
            }
        }
        if let Some(Value::Array(tool_calls)) = delta.get_mut("tool_calls") {
            let mut identity = false;
            for tc in tool_calls.iter_mut() {
                let Some(tc_obj) = tc.as_object_mut() else {
                    absorbed.closes = true;
                    continue;
                };
                tc_obj.retain(|_, v| !v.is_null());
                let index = tc_obj.get("index").and_then(Value::as_u64);
                if let Some(Value::Object(function)) = tc_obj.get_mut("function") {
                    function.retain(|_, v| !v.is_null());
                    if let Some(Value::String(args)) = function.get("arguments") {
                        if !args.is_empty() {
                            absorbed
                                .payload
                                .push((Buffer::ToolArgs(index), args.clone()));
                        }
                        function.remove("arguments");
                    }
                    if function.contains_key("name") {
                        identity = true;
                    }
                }
                if tc_obj.contains_key("id") {
                    identity = true;
                }
            }
            if identity {
                absorbed.opens = true;
            } else if tool_calls.iter().all(is_tool_call_shell) {
                delta.remove("tool_calls");
            }
        }
        if delta.contains_key("role") {
            if self.role_sent {
                delta.remove("role");
            } else {
                self.role_sent = true;
                absorbed.opens = true;
            }
        }
        delta.retain(|_, v| !v.is_null());
        if delta.keys().any(|k| k != "role" && k != "tool_calls") {
            absorbed.closes = true;
            self.unknown_events += 1;
        }
        absorbed
    }

    fn push_payload(&mut self, payload: Vec<(Buffer, String)>, out: &mut Vec<u8>) {
        for (buffer, text) in payload {
            if self.last.is_some_and(|last| last != buffer) {
                // A field switch (reasoning to content, one call to the next)
                // keeps stream order.
                self.flush_payload(out);
            }
            self.last = Some(buffer);
            match buffer {
                Buffer::Field(field) => self.fields.entry(field).or_default().push_str(&text),
                Buffer::ToolArgs(index) => self.tool_args.entry(index).or_default().push_str(&text),
            }
        }
    }

    /// Emit while enough payload is buffered.
    fn drain_ready(&mut self, out: &mut Vec<u8>) {
        for field in PAYLOAD_FIELDS {
            while self
                .fields
                .get(field)
                .is_some_and(|s| s.chars().count() >= EMIT_THRESHOLD)
            {
                let slice = self
                    .fields
                    .get_mut(field)
                    .map(take_slice)
                    .unwrap_or_default();
                self.emit_field(field, slice, out);
            }
        }
        let indices: Vec<Option<u64>> = self.tool_args.keys().copied().collect();
        for index in indices {
            while self
                .tool_args
                .get(&index)
                .is_some_and(|s| s.chars().count() >= EMIT_THRESHOLD)
            {
                let slice = self
                    .tool_args
                    .get_mut(&index)
                    .map(take_slice)
                    .unwrap_or_default();
                self.emit_tool_args(index, slice, out);
            }
        }
    }

    /// Emit every remaining buffered slice.
    fn flush_payload(&mut self, out: &mut Vec<u8>) {
        for field in PAYLOAD_FIELDS {
            while self.fields.get(field).is_some_and(|s| !s.is_empty()) {
                let slice = self
                    .fields
                    .get_mut(field)
                    .map(take_slice)
                    .unwrap_or_default();
                self.emit_field(field, slice, out);
            }
        }
        let indices: Vec<Option<u64>> = self.tool_args.keys().copied().collect();
        for index in indices {
            while self.tool_args.get(&index).is_some_and(|s| !s.is_empty()) {
                let slice = self
                    .tool_args
                    .get_mut(&index)
                    .map(take_slice)
                    .unwrap_or_default();
                self.emit_tool_args(index, slice, out);
            }
        }
    }

    fn emit_field(&self, field: &str, slice: String, out: &mut Vec<u8>) {
        let mut delta = Map::new();
        delta.insert(field.to_string(), Value::String(slice));
        self.emit_delta(delta, out);
    }

    fn emit_tool_args(&self, index: Option<u64>, slice: String, out: &mut Vec<u8>) {
        let mut function = Map::new();
        function.insert("arguments".into(), Value::String(slice));
        let mut tc = Map::new();
        if let Some(index) = index {
            tc.insert("index".into(), Value::from(index));
        }
        tc.insert("function".into(), Value::Object(function));
        let mut delta = Map::new();
        delta.insert("tool_calls".into(), Value::Array(vec![Value::Object(tc)]));
        self.emit_delta(delta, out);
    }

    fn emit_delta(&self, delta: Map<String, Value>, out: &mut Vec<u8>) {
        let mut event = self.envelope.clone();
        let mut choice = Map::new();
        choice.insert("index".into(), Value::from(0u64));
        choice.insert("delta".into(), Value::Object(delta));
        choice.insert("finish_reason".into(), Value::Null);
        event.insert("choices".into(), Value::Array(vec![Value::Object(choice)]));
        write_event(&Value::Object(event), out);
    }
}

/// Re-chunking keys its buffers by field, not by choice.
fn is_single_choice(event: &Map<String, Value>) -> bool {
    match event.get("choices").and_then(Value::as_array) {
        Some(choices) if choices.len() > 1 => false,
        Some(choices) => choices.first().is_none_or(|choice| {
            choice
                .get("index")
                .and_then(Value::as_u64)
                .is_none_or(|index| index == 0)
        }),
        None => true,
    }
}

/// A tool-call entry left with nothing but its index, type, or an emptied function.
fn is_tool_call_shell(tc: &Value) -> bool {
    tc.as_object().is_some_and(|o| {
        o.iter().all(|(k, v)| {
            k == "index"
                || k == "type"
                || (k == "function" && v.as_object().is_some_and(Map::is_empty))
        })
    })
}

/// Split an event that both opens (role, identity) and closes (finish, usage)
/// into the opening part and a closing part with an empty delta.
fn split_open_close(mut event: Map<String, Value>) -> (Map<String, Value>, Map<String, Value>) {
    let mut opening = event.clone();
    opening.remove("usage");
    if let Some(choice) = first_choice_mut(&mut opening) {
        choice.insert("finish_reason".into(), Value::Null);
    }
    if let Some(choice) = first_choice_mut(&mut event) {
        choice.insert("delta".into(), Value::Object(Map::new()));
    }
    (opening, event)
}

fn first_choice_mut(event: &mut Map<String, Value>) -> Option<&mut Map<String, Value>> {
    event
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .and_then(|choices| choices.first_mut())
        .and_then(Value::as_object_mut)
}

fn write_event(event: &Value, out: &mut Vec<u8>) {
    out.extend_from_slice(b"data: ");
    #[expect(clippy::expect_used, reason = "serializing a Value cannot fail")]
    out.extend_from_slice(
        serde_json::to_string(event)
            .expect("serialize SSE event")
            .as_bytes(),
    );
    out.extend_from_slice(b"\n\n");
}

/// Split off the next slice: at most `SLICE_CHARS`, never leaving a tail
/// shorter than `MIN_TAIL_CHARS`.
fn take_slice(buf: &mut String) -> String {
    let total = buf.chars().count();
    let take = if total <= SLICE_CHARS {
        total
    } else if total - SLICE_CHARS < MIN_TAIL_CHARS {
        total - MIN_TAIL_CHARS
    } else {
        SLICE_CHARS
    };
    take_chars(buf, take)
}

/// Split off up to `max_chars` characters from the front of `buf`.
fn take_chars(buf: &mut String, max_chars: usize) -> String {
    match buf.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => {
            let rest = buf.split_off(byte_idx);
            std::mem::replace(buf, rest)
        }
        None => std::mem::take(buf),
    }
}

/// Position and length of the first frame delimiter (`\n\n` or `\r\n\r\n`).
fn find_frame_end(raw: &[u8]) -> Option<(usize, usize)> {
    (0..raw.len()).find_map(|i| {
        if raw[i..].starts_with(b"\n\n") {
            Some((i, 2))
        } else if raw[i..].starts_with(b"\r\n\r\n") {
            Some((i, 4))
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
            all.extend_from_slice(r.feed(frame.as_bytes()).as_ref());
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
    fn frames_split_across_chunks_are_reassembled() {
        let frame = content_event(&"y".repeat(100));
        let bytes = frame.as_bytes();
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        for k in [7usize, 40, 90] {
            all.extend_from_slice(r.feed(&bytes[..k]).as_ref());
            assert!(all.is_empty(), "nothing before the frame is complete");
            all.extend_from_slice(r.feed(&bytes[k..]).as_ref());
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
        all.extend_from_slice(r.feed(two.as_bytes()).as_ref());
        all.extend_from_slice(r.feed(two.as_bytes()).as_ref());
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
        assert!(r.feed(content_event("hi").as_bytes()).is_empty());
        assert!(r.has_pending());
        let evs = events(&r.flush_pending());
        assert_eq!(evs[0]["choices"][0]["delta"]["content"], Value::from("hi"));
        assert!(!r.has_pending());
    }

    #[test]
    fn done_passthrough_after_flush() {
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(content_event("hi").as_bytes()).as_ref());
        all.extend_from_slice(r.feed(b"data: [DONE]\n\n").as_ref());
        let text = String::from_utf8_lossy(&all).to_string();
        assert!(text.find("\"hi\"").unwrap() < text.find("[DONE]").unwrap());
        assert_eq!(r.feed(b"trailing").as_ref(), b"trailing");
    }
}
