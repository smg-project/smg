//! SSE delta re-chunking for provider stream-QoS contracts.
//!
//! MiniMax's vendor verifier bounds the per-event payload-size distribution
//! (m3_stream_tests: ≤5–15% events of 1–4 chars, ≤0–2% events over 200 chars,
//! bounded large-event character share). Upstreams routinely violate both
//! directions — token-sized dribbles and multi-kilobyte argument dumps — so
//! the relay absorbs delta payload strings into per-field buffers and
//! re-emits events sized between the tiny and large bounds. Structural events
//! (role, tool-call starts, finish_reason, usage, non-data frames) flush
//! pending payload first and pass through in order.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde_json::{Map, Value};

/// Emit buffered payload in slices of this many chars once at least
/// `EMIT_THRESHOLD` chars are pending; both sit safely inside the contract's
/// normal band of 5–200 chars per event.
const SLICE_CHARS: usize = 160;
const EMIT_THRESHOLD: usize = 80;

/// Delta string fields subject to re-chunking, in emission order.
const PAYLOAD_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "content"];

#[derive(Default)]
pub struct SseRechunker {
    raw: Vec<u8>,
    /// Last-seen event envelope (top-level fields minus choices/usage).
    envelope: Option<Map<String, Value>>,
    /// Buffered payload per delta string field.
    fields: BTreeMap<&'static str, String>,
    /// Buffered tool-call `arguments` per tool index.
    tool_args: BTreeMap<u64, String>,
    role_sent: bool,
    finished: bool,
}

impl SseRechunker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one upstream body chunk, returning the bytes to forward now.
    pub fn feed(&mut self, chunk: &[u8]) -> Bytes {
        self.raw.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = find_frame_end(&self.raw) {
            let frame: Vec<u8> = self.raw.drain(..pos + 2).collect();
            self.handle_frame(&frame, &mut out);
        }
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
        Bytes::from(out)
    }

    fn handle_frame(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        let Some(data) = frame
            .strip_prefix(b"data:")
            .map(|rest| rest.strip_prefix(b" ").unwrap_or(rest))
        else {
            // Comments, event: lines, blank keep-alives — flush and forward.
            self.flush_payload(out);
            out.extend_from_slice(frame);
            return;
        };
        if data.starts_with(b"[DONE]") {
            self.flush_payload(out);
            out.extend_from_slice(frame);
            self.finished = true;
            return;
        }
        let Ok(Value::Object(mut event)) = serde_json::from_slice::<Value>(data) else {
            self.flush_payload(out);
            out.extend_from_slice(frame);
            return;
        };

        let structural = self.absorb(&mut event);
        self.envelope.get_or_insert_with(|| {
            let mut env = event.clone();
            env.remove("choices");
            env.remove("usage");
            env
        });

        if structural {
            self.flush_payload(out);
            write_event(&Value::Object(event), out);
        } else {
            self.drain_ready(out);
        }
    }

    /// Pull payload strings out of the event's deltas into the buffers.
    /// Returns true when the stripped event still carries information that
    /// must be forwarded in order (role, tool-call identity, finish, usage…).
    fn absorb(&mut self, event: &mut Map<String, Value>) -> bool {
        let mut structural = event.get("usage").is_some_and(|u| !u.is_null());
        let mut payload_seen = false;

        if let Some(Value::Array(choices)) = event.get_mut("choices") {
            for choice in choices.iter_mut() {
                let Some(obj) = choice.as_object_mut() else {
                    structural = true;
                    continue;
                };
                if obj.get("finish_reason").is_some_and(|f| !f.is_null()) {
                    structural = true;
                }
                let Some(Value::Object(delta)) = obj.get_mut("delta") else {
                    continue;
                };
                for field in PAYLOAD_FIELDS {
                    if let Some(Value::String(s)) = delta.get(field) {
                        if !s.is_empty() {
                            payload_seen = true;
                            self.fields.entry(field).or_default().push_str(s);
                        }
                        delta.remove(field);
                    }
                }
                if let Some(Value::Array(tool_calls)) = delta.get_mut("tool_calls") {
                    for tc in tool_calls.iter_mut() {
                        let Some(tc_obj) = tc.as_object_mut() else {
                            continue;
                        };
                        let index = tc_obj.get("index").and_then(Value::as_u64).unwrap_or(0);
                        if let Some(Value::Object(function)) = tc_obj.get_mut("function") {
                            if let Some(Value::String(args)) = function.get("arguments") {
                                if !args.is_empty() {
                                    payload_seen = true;
                                    self.tool_args.entry(index).or_default().push_str(args);
                                }
                                function.remove("arguments");
                            }
                            // A name or id announces a new tool call: it must
                            // be ordered after prior payload and before its
                            // own arguments.
                            if function.get("name").is_some() {
                                structural = true;
                            }
                        }
                        if tc_obj.get("id").is_some_and(|v| !v.is_null()) {
                            structural = true;
                        }
                    }
                    if delta
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|tcs| {
                            tcs.iter().all(|tc| {
                                tc.as_object().is_some_and(|o| {
                                    o.keys().all(|k| k == "index")
                                        || o.get("function").is_some_and(|f| {
                                            f.as_object().is_some_and(Map::is_empty)
                                        })
                                })
                            })
                        })
                        && !structural
                    {
                        delta.remove("tool_calls");
                    }
                }
                if delta.get("role").is_some() {
                    if self.role_sent {
                        delta.remove("role");
                    } else {
                        self.role_sent = true;
                        structural = true;
                    }
                }
                if !delta.is_empty() && delta.keys().any(|k| k != "role") {
                    structural = true;
                }
            }
        } else {
            structural = true;
        }

        let _ = payload_seen;
        structural
    }

    /// Emit while enough payload is buffered.
    fn drain_ready(&mut self, out: &mut Vec<u8>) {
        loop {
            let mut emitted = false;
            for field in PAYLOAD_FIELDS {
                let ready = self
                    .fields
                    .get(field)
                    .is_some_and(|s| s.chars().count() >= EMIT_THRESHOLD);
                if ready {
                    if let Some(buf) = self.fields.get_mut(field) {
                        let slice = take_chars(buf, SLICE_CHARS);
                        self.emit_field(field, slice, out);
                        emitted = true;
                    }
                }
            }
            let indices: Vec<u64> = self.tool_args.keys().copied().collect();
            for index in indices {
                let ready = self
                    .tool_args
                    .get(&index)
                    .is_some_and(|s| s.chars().count() >= EMIT_THRESHOLD);
                if ready {
                    if let Some(buf) = self.tool_args.get_mut(&index) {
                        let slice = take_chars(buf, SLICE_CHARS);
                        self.emit_tool_args(index, slice, out);
                        emitted = true;
                    }
                }
            }
            if !emitted {
                break;
            }
        }
    }

    /// Emit every remaining buffered slice, largest-first slicing so only the
    /// final slice per field may fall under the tiny bound.
    fn flush_payload(&mut self, out: &mut Vec<u8>) {
        for field in PAYLOAD_FIELDS {
            while let Some(buf) = self.fields.get_mut(field) {
                if buf.is_empty() {
                    break;
                }
                let slice = take_chars(buf, SLICE_CHARS);
                self.emit_field(field, slice, out);
            }
        }
        let indices: Vec<u64> = self.tool_args.keys().copied().collect();
        for index in indices {
            while let Some(buf) = self.tool_args.get_mut(&index) {
                if buf.is_empty() {
                    break;
                }
                let slice = take_chars(buf, SLICE_CHARS);
                self.emit_tool_args(index, slice, out);
            }
        }
    }

    fn emit_field(&self, field: &str, slice: String, out: &mut Vec<u8>) {
        let mut delta = Map::new();
        delta.insert(field.to_string(), Value::String(slice));
        self.emit_delta(delta, out);
    }

    fn emit_tool_args(&self, index: u64, slice: String, out: &mut Vec<u8>) {
        let mut function = Map::new();
        function.insert("arguments".into(), Value::String(slice));
        let mut tc = Map::new();
        tc.insert("index".into(), Value::from(index));
        tc.insert("function".into(), Value::Object(function));
        let mut delta = Map::new();
        delta.insert("tool_calls".into(), Value::Array(vec![Value::Object(tc)]));
        self.emit_delta(delta, out);
    }

    fn emit_delta(&self, delta: Map<String, Value>, out: &mut Vec<u8>) {
        let mut event = self.envelope.clone().unwrap_or_default();
        let mut choice = Map::new();
        choice.insert("index".into(), Value::from(0u64));
        choice.insert("delta".into(), Value::Object(delta));
        choice.insert("finish_reason".into(), Value::Null);
        event.insert("choices".into(), Value::Array(vec![Value::Object(choice)]));
        write_event(&Value::Object(event), out);
    }
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

fn find_frame_end(raw: &[u8]) -> Option<usize> {
    raw.windows(2).position(|w| w == b"\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(bytes: &Bytes) -> Vec<Value> {
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

    #[test]
    fn tiny_deltas_merge() {
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        for _ in 0..100 {
            all.extend_from_slice(r.feed(content_event("ab").as_bytes()).as_ref());
        }
        all.extend_from_slice(r.finish().as_ref());
        let evs = events(&Bytes::from(all));
        let sizes: Vec<usize> = evs
            .iter()
            .map(|e| e["choices"][0]["delta"]["content"].as_str().unwrap().len())
            .collect();
        assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn large_delta_splits() {
        let mut r = SseRechunker::new();
        let big = "x".repeat(1000);
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(content_event(&big).as_bytes()).as_ref());
        all.extend_from_slice(r.finish().as_ref());
        let evs = events(&Bytes::from(all));
        let sizes: Vec<usize> = evs
            .iter()
            .map(|e| e["choices"][0]["delta"]["content"].as_str().unwrap().len())
            .collect();
        assert!(sizes.iter().all(|&s| s <= 200), "sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 1000);
    }

    #[test]
    fn finish_and_usage_flush_in_order() {
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(content_event("hello").as_bytes()).as_ref());
        let fin = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"total_tokens\":5}}\n\n";
        all.extend_from_slice(r.feed(fin.as_bytes()).as_ref());
        all.extend_from_slice(r.finish().as_ref());
        let evs = events(&Bytes::from(all));
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
    fn tool_call_identity_precedes_arguments() {
        let mut r = SseRechunker::new();
        let start = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}}]}}]}\n\n";
        let args = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"k\\\":1}\"}}]}}]}\n\n";
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(start.as_bytes()).as_ref());
        all.extend_from_slice(r.feed(args.as_bytes()).as_ref());
        all.extend_from_slice(r.finish().as_ref());
        let evs = events(&Bytes::from(all));
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
    fn done_passthrough_after_flush() {
        let mut r = SseRechunker::new();
        let mut all = Vec::new();
        all.extend_from_slice(r.feed(content_event("hi").as_bytes()).as_ref());
        all.extend_from_slice(r.feed(b"data: [DONE]\n\n").as_ref());
        let text = String::from_utf8_lossy(&all).to_string();
        let hi = text.find("\"hi\"").unwrap();
        let done = text.find("[DONE]").unwrap();
        assert!(hi < done);
    }
}
