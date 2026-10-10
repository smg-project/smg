//! The lines of a token dump file (format v1).
//!
//! Every line is one JSON object stamped with `"v": 1` and a `kind`; the
//! encoders here return it with its trailing newline, ready for the session's
//! writer. The format is described in the [module docs](super).

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Serialize;

/// The format version every line carries.
pub const FORMAT_VERSION: u32 = 1;

/// Which leg of a client request an engine call serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Leg {
    /// The only engine call of a request served by one worker.
    Single,
    /// The prefill leg of a disaggregated request.
    Prefill,
    /// The decode leg of a disaggregated request.
    Decode,
}

/// How the gateway reached the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    Grpc,
    /// Direct ZMQ: messages are recorded at the gateway's proto boundary,
    /// before the translation to the engine's own wire format.
    Zmq,
}

/// Which message of the engine's response oneof a `response` line holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Chunk,
    Complete,
    /// The engine sent a response with neither set.
    Empty,
}

/// How an engine call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndStatus {
    /// The stream ended, or the gateway took its last response.
    Ok,
    /// The stream failed, or the engine finished the request with an error.
    Error,
    /// The gateway dropped the call before it ended (e.g. client disconnect).
    Cancelled,
    /// The engine refused or failed to start the call.
    StartFailed,
}

/// Why a session's file was closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    Stopped,
    Expired,
    Shutdown,
}

impl EndReason {
    /// The byte a session keeps its end reason in until the file closes.
    pub(super) const fn to_u8(self) -> u8 {
        match self {
            Self::Shutdown => 0,
            Self::Stopped => 1,
            Self::Expired => 2,
        }
    }

    /// The reason [`to_u8`](Self::to_u8) stored; a session nobody stopped
    /// ends at shutdown.
    pub(super) const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Stopped,
            2 => Self::Expired,
            _ => Self::Shutdown,
        }
    }
}

/// The routing fields of a `request` line.
#[derive(Debug, Clone)]
pub struct CallMeta {
    /// The request's canonical model id.
    pub model: String,
    /// The worker's URL.
    pub worker: String,
    /// The engine runtime (`sglang`, `vllm`, ...).
    pub runtime: &'static str,
    pub transport: Transport,
    pub leg: Leg,
    /// The client request's id, as the request-id middleware assigned it.
    pub root_request_id: Option<String>,
}

/// An engine request, as the gRPC router's proto wrapper exposes it.
#[derive(Debug)]
pub struct RequestEvent<'a> {
    /// Full protobuf name of the message in `msg`.
    pub type_name: &'static str,
    pub request_id: &'a str,
    pub input_ids: &'a [u32],
    /// The message's protobuf encoding.
    pub msg: Vec<u8>,
}

/// One engine response message, as the engine sent it.
#[derive(Debug)]
pub struct ResponseEvent<'a> {
    /// Full protobuf name of the message in `msg`.
    pub type_name: &'static str,
    pub part: Part,
    pub index: u32,
    /// The ids this message carries (a chunk's `token_ids`, a complete's
    /// output ids), with no accumulation across messages.
    pub token_ids: &'a [u32],
    /// Set on a `complete` only.
    pub finish_reason: Option<&'a str>,
    /// The message's protobuf encoding.
    pub msg: Vec<u8>,
}

/// Lines a session did not write, by reason.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Dropped {
    pub queue_full: u64,
    pub size_cap: u64,
    pub write_error: u64,
}

/// A session's counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Totals {
    pub calls: u64,
    pub lines_written: u64,
    pub lines_dropped: Dropped,
    pub bytes_written: u64,
}

/// What a `session` line says about its session.
#[derive(Debug)]
pub struct SessionHeader<'a> {
    pub session: &'a str,
    pub started_at: &'a str,
    pub models: &'a [String],
    pub expires_at: Option<&'a str>,
    pub max_bytes: u64,
}

#[derive(Serialize)]
struct SessionLine<'a> {
    v: u32,
    kind: &'static str,
    session: &'a str,
    started_at: &'a str,
    gateway_version: &'static str,
    models: &'a [String],
    expires_at: Option<&'a str>,
    max_bytes: u64,
}

#[derive(Serialize)]
struct RequestLine<'a> {
    v: u32,
    kind: &'static str,
    call: u64,
    t: &'a str,
    model: &'a str,
    worker: &'a str,
    runtime: &'a str,
    transport: Transport,
    leg: Leg,
    request_id: &'a str,
    root_request_id: Option<&'a str>,
    #[serde(rename = "type")]
    type_name: &'a str,
    input_ids: &'a [u32],
    msg: String,
}

#[derive(Serialize)]
struct ResponseLine<'a> {
    v: u32,
    kind: &'static str,
    call: u64,
    seq: u64,
    t_ms: u64,
    #[serde(rename = "type")]
    type_name: &'a str,
    part: Part,
    index: u32,
    token_ids: &'a [u32],
    finish_reason: Option<&'a str>,
    msg: String,
}

#[derive(Serialize)]
struct EndError {
    code: String,
    message: String,
}

#[derive(Serialize)]
struct EndLine {
    v: u32,
    kind: &'static str,
    call: u64,
    t_ms: u64,
    status: EndStatus,
    responses: u64,
    error: Option<EndError>,
}

#[derive(Serialize)]
struct SessionEndLine<'a> {
    v: u32,
    kind: &'static str,
    session: &'a str,
    reason: EndReason,
    calls: u64,
    lines_written: u64,
    lines_dropped: Dropped,
    bytes_written: u64,
}

/// The `session` line that opens every file.
pub fn session_line(header: &SessionHeader<'_>) -> Vec<u8> {
    to_line(&SessionLine {
        v: FORMAT_VERSION,
        kind: "session",
        session: header.session,
        started_at: header.started_at,
        gateway_version: crate::version::VERSION,
        models: header.models,
        expires_at: header.expires_at,
        max_bytes: header.max_bytes,
    })
}

/// The `request` line of call `call`, written at `t`, just before dispatch.
pub fn request_line(call: u64, t: &str, meta: &CallMeta, event: &RequestEvent<'_>) -> Vec<u8> {
    to_line(&RequestLine {
        v: FORMAT_VERSION,
        kind: "request",
        call,
        t,
        model: &meta.model,
        worker: &meta.worker,
        runtime: meta.runtime,
        transport: meta.transport,
        leg: meta.leg,
        request_id: event.request_id,
        root_request_id: meta.root_request_id.as_deref(),
        type_name: event.type_name,
        input_ids: event.input_ids,
        msg: BASE64.encode(&event.msg),
    })
}

/// Response `seq` of call `call`, `t_ms` after its request line.
pub fn response_line(call: u64, seq: u64, t_ms: u64, event: &ResponseEvent<'_>) -> Vec<u8> {
    to_line(&ResponseLine {
        v: FORMAT_VERSION,
        kind: "response",
        call,
        seq,
        t_ms,
        type_name: event.type_name,
        part: event.part,
        index: event.index,
        token_ids: event.token_ids,
        finish_reason: event.finish_reason,
        msg: BASE64.encode(&event.msg),
    })
}

/// The `end` line of call `call`; `error` is the status it failed with.
pub fn end_line(
    call: u64,
    t_ms: u64,
    status: EndStatus,
    responses: u64,
    error: Option<&tonic::Status>,
) -> Vec<u8> {
    let error = error.map(|status| EndError {
        code: format!("{:?}", status.code()),
        message: status.message().to_string(),
    });
    to_line(&EndLine {
        v: FORMAT_VERSION,
        kind: "end",
        call,
        t_ms,
        status,
        responses,
        error,
    })
}

/// The `session_end` line that closes a file. Its counters cover every
/// line before it.
pub fn session_end_line(session: &str, reason: EndReason, totals: &Totals) -> Vec<u8> {
    to_line(&SessionEndLine {
        v: FORMAT_VERSION,
        kind: "session_end",
        session,
        reason,
        calls: totals.calls,
        lines_written: totals.lines_written,
        lines_dropped: totals.lines_dropped,
        bytes_written: totals.bytes_written,
    })
}

fn to_line<T: Serialize>(value: &T) -> Vec<u8> {
    match serde_json::to_vec(value) {
        Ok(mut line) => {
            line.push(b'\n');
            line
        }
        // Every field is a string, an integer, a list of integers or a unit
        // enum, which always serialize; the writer skips an empty line.
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    fn parse(line: &[u8]) -> Value {
        assert_eq!(line.last(), Some(&b'\n'), "a line ends with its newline");
        assert_eq!(
            line.iter().filter(|&&byte| byte == b'\n').count(),
            1,
            "one line"
        );
        serde_json::from_slice(line).unwrap()
    }

    #[test]
    fn session_line_opens_with_version_and_kind() {
        let models = vec!["m".to_string()];
        let line = session_line(&SessionHeader {
            session: "s1",
            started_at: "2026-10-09T21:00:00.000Z",
            models: &models,
            expires_at: Some("2026-10-09T21:10:00.000Z"),
            max_bytes: 1024,
        });
        assert!(line.starts_with(br#"{"v":1,"kind":"session","#));
        assert_eq!(
            parse(&line),
            json!({"v":1,"kind":"session","session":"s1",
                   "started_at":"2026-10-09T21:00:00.000Z",
                   "gateway_version":crate::version::VERSION,"models":["m"],
                   "expires_at":"2026-10-09T21:10:00.000Z","max_bytes":1024})
        );
    }

    #[test]
    fn request_line_carries_routing_ids_and_message() {
        let meta = CallMeta {
            model: "m".to_string(),
            worker: "grpc://w:1".to_string(),
            runtime: "sglang",
            transport: Transport::Grpc,
            leg: Leg::Prefill,
            root_request_id: Some("root".to_string()),
        };
        let event = RequestEvent {
            type_name: "sglang.grpc.scheduler.GenerateRequest",
            request_id: "r1",
            input_ids: &[1, 2, 3],
            msg: vec![0x0a, 0x01],
        };
        let line = request_line(7, "2026-10-09T21:00:01.456Z", &meta, &event);
        assert!(line.starts_with(br#"{"v":1,"kind":"request","#));
        assert_eq!(
            parse(&line),
            json!({"v":1,"kind":"request","call":7,"t":"2026-10-09T21:00:01.456Z",
                   "model":"m","worker":"grpc://w:1","runtime":"sglang",
                   "transport":"grpc","leg":"prefill","request_id":"r1",
                   "root_request_id":"root",
                   "type":"sglang.grpc.scheduler.GenerateRequest",
                   "input_ids":[1,2,3],"msg":"CgE="})
        );
    }

    #[test]
    fn response_line_carries_the_message_as_sent() {
        let event = ResponseEvent {
            type_name: "vllm.grpc.engine.GenerateResponse",
            part: Part::Complete,
            index: 1,
            token_ids: &[7, 8],
            finish_reason: Some("stop"),
            msg: vec![0xff],
        };
        assert_eq!(
            parse(&response_line(3, 0, 42, &event)),
            json!({"v":1,"kind":"response","call":3,"seq":0,"t_ms":42,
                   "type":"vllm.grpc.engine.GenerateResponse","part":"complete",
                   "index":1,"token_ids":[7,8],"finish_reason":"stop","msg":"/w=="})
        );
    }

    #[test]
    fn end_line_names_the_status_and_its_error() {
        let status = tonic::Status::unavailable("worker gone");
        assert_eq!(
            parse(&end_line(3, 900, EndStatus::StartFailed, 0, Some(&status))),
            json!({"v":1,"kind":"end","call":3,"t_ms":900,"status":"start_failed",
                   "responses":0,"error":{"code":"Unavailable","message":"worker gone"}})
        );
        assert_eq!(
            parse(&end_line(4, 5, EndStatus::Ok, 2, None))["error"],
            Value::Null
        );
    }

    #[test]
    fn session_end_line_reports_totals() {
        let totals = Totals {
            calls: 2,
            lines_written: 7,
            lines_dropped: Dropped {
                queue_full: 1,
                size_cap: 0,
                write_error: 0,
            },
            bytes_written: 900,
        };
        assert_eq!(
            parse(&session_end_line("s1", EndReason::Expired, &totals)),
            json!({"v":1,"kind":"session_end","session":"s1","reason":"expired",
                   "calls":2,"lines_written":7,
                   "lines_dropped":{"queue_full":1,"size_cap":0,"write_error":0},
                   "bytes_written":900})
        );
    }

    #[test]
    fn end_reason_round_trips_through_its_byte() {
        for reason in [EndReason::Stopped, EndReason::Expired, EndReason::Shutdown] {
            assert_eq!(EndReason::from_u8(reason.to_u8()), reason);
        }
    }
}
