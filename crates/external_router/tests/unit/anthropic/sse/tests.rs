use super::*;
use crate::error::HEADER_X_SMG_ERROR_CODE;

#[test]
fn build_sse_response_drops_smg_owned_error_code() {
    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(HEADER_X_SMG_ERROR_CODE, "forged".parse().unwrap());
    upstream_headers.insert("x-upstream-id", "worker-a".parse().unwrap());

    let response = build_sse_response(
        StatusCode::SERVICE_UNAVAILABLE,
        upstream_headers,
        Body::empty(),
    );

    assert!(response.headers().get(HEADER_X_SMG_ERROR_CODE).is_none());
    assert_eq!(response.headers().get("x-upstream-id").unwrap(), "worker-a");
}

#[test]
fn build_sse_response_drops_the_upstream_server_header() {
    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert("server", "upstream/1.2.3".parse().unwrap());
    upstream_headers.insert("keep-alive", "timeout=5".parse().unwrap());
    upstream_headers.insert("x-upstream-id", "worker-a".parse().unwrap());
    upstream_headers.insert("x-request-id", "req-7".parse().unwrap());

    let response = build_sse_response(StatusCode::OK, upstream_headers, Body::empty());

    assert!(response.headers().get("server").is_none());
    assert!(response.headers().get("keep-alive").is_none());
    assert_eq!(response.headers().get("x-upstream-id").unwrap(), "worker-a");
    assert_eq!(response.headers().get("x-request-id").unwrap(), "req-7");
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
}

/// Decode `bytes` through the shared `SseDecoder` and resolve each frame
/// to `(event_type, data)` the way `consume_and_forward` does.
fn decode_events(bytes: &[u8]) -> Vec<(String, String)> {
    let mut decoder = SseDecoder::new();
    decoder.push(bytes).unwrap();
    let mut out = Vec::new();
    while let Some(frame) = decoder.next_frame() {
        if let Some((event_type, data)) = resolve_event(frame.unwrap()) {
            out.push((event_type.into_owned(), data.into_owned()));
        }
    }
    out
}

#[test]
fn test_resolve_event_basic() {
    let frame = SseFrame {
        event_type: Some(Cow::Borrowed("message_start")),
        data: Cow::Borrowed("{\"type\":\"message_start\"}"),
    };
    let (event_type, data) = resolve_event(frame).unwrap();
    assert_eq!(event_type.as_ref(), "message_start");
    assert_eq!(data, "{\"type\":\"message_start\"}");
}

#[test]
fn test_resolve_event_no_event_type_infers() {
    // No `event:` line -> infer the type from the payload's "type" field.
    let frame = SseFrame {
        event_type: None,
        data: Cow::Borrowed("{\"type\":\"ping\"}"),
    };
    let (event_type, _data) = resolve_event(frame).unwrap();
    assert_eq!(event_type.as_ref(), "ping");
}

#[test]
fn test_resolve_event_uninferable_is_none() {
    // No event type and no "type" field in the payload -> skipped.
    let frame = SseFrame {
        event_type: None,
        data: Cow::Borrowed("{\"foo\":1}"),
    };
    assert!(resolve_event(frame).is_none());
}

#[test]
fn test_decode_events_basic() {
    let events = decode_events(
        b"event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n",
    );
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].0, "message_start");
    assert_eq!(events[1].0, "ping");
}

#[test]
fn test_decode_events_content_block() {
    let events = decode_events(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "content_block_start");
    let parsed: Value = serde_json::from_str(&events[0].1).unwrap();
    assert_eq!(parsed["index"], 0);
}

#[test]
fn test_decode_events_infers_event_type() {
    // A `data:`-only frame (no `event:` line) infers its type from the payload.
    let events = decode_events(b"data: {\"type\":\"ping\"}\n\n");
    assert_eq!(
        events,
        vec![("ping".to_string(), "{\"type\":\"ping\"}".to_string())]
    );
}

#[test]
fn test_format_sse_event() {
    let data = serde_json::json!({"type": "ping"});
    let mut enc = SseEncoder::new();
    let bytes = format_sse_event(&mut enc, "ping", &data);
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.starts_with("event: ping\n"));
    assert!(text.contains("data: "));
    assert!(text.ends_with("\n\n"));
}

#[test]
fn test_decode_events_split_across_chunks() {
    // A multi-byte-safe decoder must reassemble frames split mid-stream.
    let mut decoder = SseDecoder::new();
    decoder
        .push(b"event: content_block_delta\ndata: {\"type\":\"content_block_de")
        .unwrap();
    assert!(decoder.next_frame().is_none()); // incomplete
    decoder
        .push(b"lta\",\"index\":1,\"delta\":{\"partial_json\":\"{}\"}}\n\n")
        .unwrap();
    let frame = decoder.next_frame().unwrap().unwrap();
    let (event_type, data) = resolve_event(frame).unwrap();
    assert_eq!(event_type.as_ref(), "content_block_delta");
    let parsed: Value = serde_json::from_str(&data).unwrap();
    assert_eq!(parsed["index"], 1);
}
