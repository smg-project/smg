use super::*;

// --- SseEncoder tests ---

#[test]
fn test_encode_data_simple() {
    let mut enc = SseEncoder::new();
    let val = serde_json::json!({"type": "ping"});
    let bytes = enc.encode_data(&val).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.starts_with("data: "));
    assert!(text.ends_with("\n\n"));
    assert!(text.contains("\"type\":\"ping\""));
}

#[test]
fn test_encode_event_with_type() {
    let mut enc = SseEncoder::new();
    let val = serde_json::json!({"index": 0, "delta": "hi"});
    let bytes = enc.encode_event("content_block_delta", &val).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.starts_with("event: content_block_delta\n"));
    assert!(text.contains("data: "));
    assert!(text.ends_with("\n\n"));
}

#[test]
fn test_encode_done_sentinel() {
    let bytes = SseEncoder::done();
    assert_eq!(&bytes[..], b"data: [DONE]\n\n");
}

/// Twenty small frames are cut from the encoder's first block, one after
/// another, without a copy or a block of their own: each lies inside the
/// first frame's block, and a frame already handed out keeps its bytes
/// while the frames after it are written.
#[test]
fn small_frames_are_cut_from_one_block_and_keep_their_bytes() {
    let mut enc = SseEncoder::new();
    let frames: Vec<Bytes> = (0..20)
        .map(|i| enc.encode_data(&serde_json::json!({"i": i})).unwrap())
        .collect();
    let block = frames[0].as_ptr() as usize;
    for (i, frame) in frames.iter().enumerate() {
        let start = frame.as_ptr() as usize;
        assert!(
            start >= block && start + frame.len() <= block + FRAME_BLOCK,
            "frame {i} lies outside the first block"
        );
        assert_eq!(&frame[..], format!("data: {{\"i\":{i}}}\n\n").as_bytes());
    }
}

/// A frame larger than a block is whole, middle included, and the
/// frames after it are written as before.
#[test]
fn a_frame_larger_than_a_block_is_whole() {
    let mut enc = SseEncoder::new();
    let long = "x".repeat(3 * FRAME_BLOCK);
    let big = enc.encode_data(&serde_json::json!({"text": long})).unwrap();
    assert_eq!(
        &big[..],
        format!("data: {{\"text\":\"{long}\"}}\n\n").as_bytes()
    );
    let next = enc.encode_data(&serde_json::json!({"i": 1})).unwrap();
    assert_eq!(&next[..], b"data: {\"i\":1}\n\n");
}

/// A value whose serialization fails after it has written part of a map.
struct FailsMidway;

impl Serialize for FailsMidway {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeMap};
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("partial", "bytes")?;
        Err(S::Error::custom("boom"))
    }
}

/// A frame whose serialization fails partway leaves nothing in the next
/// frame: the block is emptied at the start of every encode, and a split
/// runs only on success.
#[test]
fn a_failed_frame_leaves_nothing_in_the_next() {
    let mut enc = SseEncoder::new();
    assert!(enc.encode_data(&FailsMidway).is_err());
    let next = enc.encode_data(&serde_json::json!({"i": 1})).unwrap();
    assert_eq!(&next[..], b"data: {\"i\":1}\n\n");
}

/// A large frame sets the next frame's reservation and a small frame
/// clears it: a second large frame is whole, so is the small frame after
/// it, and after that small frame the encoder reserves nothing.
#[test]
fn a_large_frame_sets_the_next_frames_reservation() {
    let mut enc = SseEncoder::new();
    let long = "y".repeat(5 * FRAME_BLOCK);
    let first = enc.encode_data(&serde_json::json!({"text": long})).unwrap();
    assert_eq!(enc.last_len, first.len());
    let second = enc.encode_data(&serde_json::json!({"text": long})).unwrap();
    assert_eq!(&second[..], &first[..]);
    let next = enc.encode_data(&serde_json::json!({"i": 2})).unwrap();
    assert_eq!(&next[..], b"data: {\"i\":2}\n\n");
    assert_eq!(enc.last_len, next.len());
    assert!(enc.last_len <= FRAME_BLOCK);
}

#[test]
fn test_encode_data_matches_format_pattern() {
    // Verify output matches `format!("data: {json}\n\n")` for compatibility.
    let mut enc = SseEncoder::new();
    let val = serde_json::json!({"type":"message_start","message":{"id":"msg_1"}});
    let bytes = enc.encode_data(&val).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();

    let json_str = serde_json::to_string(&val).unwrap();
    let expected = format!("data: {json_str}\n\n");
    assert_eq!(text, expected);
}

#[test]
fn test_encode_event_matches_format_pattern() {
    // Verify output matches `format!("event: {type}\ndata: {json}\n\n")`.
    let mut enc = SseEncoder::new();
    let val = serde_json::json!({"type":"content_block_start","index":0});
    let bytes = enc.encode_event("content_block_start", &val).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();

    let json_str = serde_json::to_string(&val).unwrap();
    let expected = format!("event: content_block_start\ndata: {json_str}\n\n");
    assert_eq!(text, expected);
}

/// The error event of a failed Responses stream: the API's fields in its
/// order, the message escaped as JSON, on a data line alone, no event
/// name and no sentinel.
#[test]
fn a_responses_error_frame_is_the_typed_event_on_a_data_line() {
    let mut enc = SseEncoder::new();
    let frame = enc.encode_responses_error("stream_error", "Stream error: he said \"bye\"\n", 7);
    assert_eq!(
        frame,
        "data: {\"type\":\"error\",\"code\":\"stream_error\",\"message\":\"Stream error: he \
             said \\\"bye\\\"\\n\",\"param\":null,\"sequence_number\":7}\n\n"
            .as_bytes()
    );
}

// --- SseDecoder tests ---

#[test]
fn test_decode_single_data_frame() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: {\"type\":\"ping\"}\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.event_type, None);
    assert_eq!(frame.data, "{\"type\":\"ping\"}");
}

#[test]
fn test_decode_event_and_data() {
    let mut dec = SseDecoder::new();
    dec.push(b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n")
        .unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.event_type.as_deref(), Some("message_start"));
    assert_eq!(frame.data, "{\"type\":\"message_start\"}");
}

#[test]
fn test_decode_multiple_frames() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: first\n\ndata: second\n\n").unwrap();

    let f1 = dec.next_frame().unwrap().unwrap();
    assert_eq!(f1.data, "first");

    let f2 = dec.next_frame().unwrap().unwrap();
    assert_eq!(f2.data, "second");

    assert!(dec.next_frame().is_none());
}

#[test]
fn test_decode_split_across_chunks() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: hel").unwrap();
    assert!(dec.next_frame().is_none()); // incomplete
    dec.push(b"lo\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "hello");
}

#[test]
fn test_decode_multiline_data() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: line1\ndata: line2\ndata: line3\n\n")
        .unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "line1\nline2\nline3");
}

#[test]
fn test_decode_crlf_frame_boundary() {
    // Full CRLF-delimited frame: \r\n\r\n
    let mut dec = SseDecoder::new();
    dec.push(b"event: ping\r\ndata: {}\r\n\r\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.event_type.as_deref(), Some("ping"));
    assert_eq!(frame.data, "{}");
}

#[test]
fn test_decode_crlf_within_frame() {
    // Mixed: CRLF lines but LF-only frame boundary
    let mut dec = SseDecoder::new();
    dec.push(b"event: ping\r\ndata: {}\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.event_type.as_deref(), Some("ping"));
    assert_eq!(frame.data, "{}");
}

#[test]
fn test_decode_cr_cr_frame_boundary() {
    // Standalone CR delimiters: \r\r
    let mut dec = SseDecoder::new();
    dec.push(b"data: hello\r\r").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "hello");
}

#[test]
fn test_decode_mixed_eol_frame_boundary() {
    // Mixed: LF then CRLF (\n\r\n)
    let mut dec = SseDecoder::new();
    dec.push(b"data: mixed\n\r\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "mixed");
}

#[test]
fn test_decode_comments_ignored() {
    let mut dec = SseDecoder::new();
    dec.push(b": this is a comment\ndata: hello\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "hello");
    assert_eq!(frame.event_type, None);
}

#[test]
fn test_decode_event_only_block_skipped() {
    // Per SSE spec, blocks without data lines are control-only and should be skipped
    let mut dec = SseDecoder::new();
    dec.push(b"event: ping\n\n").unwrap();
    assert!(dec.next_frame().is_none());
}

#[test]
fn test_decode_empty_data_value() {
    // A block with an explicit `data:` line (empty value) IS a valid frame
    let mut dec = SseDecoder::new();
    dec.push(b"event: ping\ndata:\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.event_type.as_deref(), Some("ping"));
    assert_eq!(frame.data, "");
}

#[test]
fn test_decode_done_sentinel() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: [DONE]\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert!(frame.is_done());
}

#[test]
fn test_decode_json_deserialization() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: {\"value\":42}\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    let val: serde_json::Value = frame.decode_data().unwrap();
    assert_eq!(val["value"], 42);
}

#[test]
fn test_decode_buffer_overflow() {
    let mut dec = SseDecoder::with_max_size(16);
    let result = dec.push(b"data: this is way too long for the buffer\n\n");
    assert!(matches!(result, Err(SseDecodeError::BufferOverflow)));
}

#[test]
fn test_decode_buffer_overflow_auto_compacts() {
    // Buffer with max_size=32. Push data, drain frames, push more.
    // Without auto-compact, the second push would overflow because
    // consumed bytes still occupy the Vec.
    let mut dec = SseDecoder::with_max_size(32);
    dec.push(b"data: first\n\n").unwrap(); // 13 bytes
    let f = dec.next_frame().unwrap().unwrap();
    assert_eq!(f.data, "first");
    // Without auto-compact: buf.len()=13, consumed=13, unconsumed=0
    // New push of 14 bytes: buf.len()+chunk.len() = 27 <= 32, OK
    // But if we pushed 20 bytes: 13+20 = 33 > 32, auto-compact needed
    dec.push(b"data: second-longer\n\n").unwrap(); // 21 bytes, 13+21=34>32, auto-compacts
    let f = dec.next_frame().unwrap().unwrap();
    assert_eq!(f.data, "second-longer");
}

#[test]
fn test_decode_compact() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: first\n\ndata: second\n\npartial").unwrap();

    let f1 = dec.next_frame().unwrap().unwrap();
    assert_eq!(f1.data, "first");

    let f2 = dec.next_frame().unwrap().unwrap();
    assert_eq!(f2.data, "second");

    assert!(dec.next_frame().is_none());
    assert_eq!(dec.buffered_len(), 7); // "partial"

    dec.compact();
    assert_eq!(dec.consumed, 0);
    assert_eq!(dec.buf, b"partial");
}

#[test]
fn test_decode_flush() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: final").unwrap();
    assert!(dec.next_frame().is_none());
    let frame = dec.flush().unwrap().unwrap();
    assert_eq!(frame.data, "final");
}

#[test]
fn test_decode_flush_empty() {
    let mut dec = SseDecoder::new();
    assert!(dec.flush().is_none());
}

#[test]
fn test_decode_flush_whitespace_only() {
    let mut dec = SseDecoder::new();
    dec.push(b"  \n  ").unwrap();
    // No data: lines, so this is a control-only block — flush returns None
    assert!(dec.flush().is_none());
}

#[test]
fn test_decode_flush_newlines_only() {
    let mut dec = SseDecoder::new();
    dec.push(b"\n\r\n").unwrap();
    // \n\r\n is a frame boundary (LF + CRLF) with empty content.
    // next_frame skips empty frames, so it returns None.
    assert!(dec.next_frame().is_none());
    // After draining, flush also returns None — no meaningful data.
    assert!(dec.flush().is_none());
}

#[test]
fn test_decode_flush_incomplete() {
    let mut dec = SseDecoder::new();
    dec.push(b"data: a\n\ndata: b\n\n").unwrap();
    // Complete frames remain — flush should reject
    let result = dec.flush().unwrap();
    assert!(matches!(result, Err(SseDecodeError::IncompleteFlush)));
}

#[test]
fn test_decode_unknown_fields_ignored() {
    let mut dec = SseDecoder::new();
    dec.push(b"id: 123\nretry: 5000\ndata: hello\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "hello");
    assert_eq!(frame.event_type, None);
}

#[test]
fn test_decode_leading_space_strip() {
    let mut dec = SseDecoder::new();
    // Per spec: strip exactly one leading space after the colon
    dec.push(b"data:  two spaces\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    // Strip one space, leaving " two spaces"
    assert_eq!(frame.data, " two spaces");
}

#[test]
fn test_decode_no_space_after_colon() {
    let mut dec = SseDecoder::new();
    dec.push(b"data:nospace\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "nospace");
}

#[test]
fn test_decode_field_without_colon() {
    // Per spec: a line with no colon is treated as field name with empty value
    let mut dec = SseDecoder::new();
    dec.push(b"data\n\n").unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "");
}

#[test]
fn test_decode_multi_byte_utf8_split() {
    let mut dec = SseDecoder::new();
    // "data: 日本" in UTF-8
    let full = "data: 日本\n\n";
    let bytes = full.as_bytes();
    // Split in the middle of a multi-byte character
    let mid = 8; // "data: 日" is "data: " (6) + 日 (3 bytes) = 9, split at 8
    dec.push(&bytes[..mid]).unwrap();
    assert!(dec.next_frame().is_none()); // incomplete frame
    dec.push(&bytes[mid..]).unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "日本");
}

#[test]
fn test_decode_empty_lines_between_fields() {
    let mut dec = SseDecoder::new();
    dec.push(b"event: test\n\ndata: value\n\n").unwrap();
    // First block (event: test) has no data line — skipped per SSE spec
    // Second block (data: value) is returned as the first frame
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.data, "value");
    assert_eq!(frame.event_type, None);
}

// --- Roundtrip tests ---

#[test]
fn test_roundtrip_data_only() {
    let mut enc = SseEncoder::new();
    let original = serde_json::json!({"type":"ping","id":42});
    let bytes = enc.encode_data(&original).unwrap();

    let mut dec = SseDecoder::new();
    dec.push(&bytes).unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    let decoded: serde_json::Value = frame.decode_data().unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_roundtrip_event_with_type() {
    let mut enc = SseEncoder::new();
    let original = serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}});
    let bytes = enc.encode_event("content_block_delta", &original).unwrap();

    let mut dec = SseDecoder::new();
    dec.push(&bytes).unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.event_type.as_deref(), Some("content_block_delta"));
    let decoded: serde_json::Value = frame.decode_data().unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_roundtrip_done() {
    let bytes = SseEncoder::done();
    let mut dec = SseDecoder::new();
    dec.push(&bytes).unwrap();
    let frame = dec.next_frame().unwrap().unwrap();
    assert!(frame.is_done());
}

#[test]
fn test_roundtrip_multiple_events() {
    let mut enc = SseEncoder::new();
    let events = vec![
        ("message_start", serde_json::json!({"type":"message_start"})),
        (
            "content_block_start",
            serde_json::json!({"type":"content_block_start","index":0}),
        ),
        (
            "content_block_delta",
            serde_json::json!({"type":"content_block_delta","index":0,"delta":{"text":"hi"}}),
        ),
        (
            "content_block_stop",
            serde_json::json!({"type":"content_block_stop","index":0}),
        ),
        ("message_stop", serde_json::json!({"type":"message_stop"})),
    ];

    let mut all_bytes = Vec::new();
    for (event_type, val) in &events {
        let bytes = enc.encode_event(event_type, val).unwrap();
        all_bytes.extend_from_slice(&bytes);
    }

    let mut dec = SseDecoder::new();
    dec.push(&all_bytes).unwrap();

    for (expected_type, expected_val) in &events {
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type.as_deref(), Some(*expected_type));
        let decoded: serde_json::Value = frame.decode_data().unwrap();
        assert_eq!(&decoded, expected_val);
    }

    assert!(dec.next_frame().is_none());
}

// --- parse_frame unit tests ---

#[test]
fn test_parse_frame_basic() {
    let frame = parse_frame("event: message_start\ndata: {\"type\":\"message_start\"}").unwrap();
    assert_eq!(frame.event_type.as_deref(), Some("message_start"));
    assert_eq!(frame.data, "{\"type\":\"message_start\"}");
}

#[test]
fn test_parse_frame_data_only() {
    let frame = parse_frame("data: hello").unwrap();
    assert_eq!(frame.event_type, None);
    assert_eq!(frame.data, "hello");
}

#[test]
fn test_parse_frame_empty() {
    assert!(parse_frame("").is_none());
}

#[test]
fn test_parse_frame_multiline_data() {
    let frame = parse_frame("data: line1\ndata: line2").unwrap();
    assert_eq!(frame.data, "line1\nline2");
}

#[test]
fn test_parse_frame_empty_data_lines() {
    // Empty data: lines must produce newlines per SSE spec
    let frame = parse_frame("data:\ndata: hello").unwrap();
    assert_eq!(frame.data, "\nhello");

    // Two consecutive empty data: lines
    let frame = parse_frame("data:\ndata:").unwrap();
    assert_eq!(frame.data, "\n");
}

#[test]
fn test_parse_frame_comment_only() {
    // Comment-only blocks have no data line — should return None
    assert!(parse_frame(": heartbeat").is_none());
}

#[test]
fn test_parse_frame_field_without_colon() {
    // "data" with no colon = field name "data", empty value — still has a data line
    let frame = parse_frame("data").unwrap();
    assert_eq!(frame.data, "");
}

#[test]
fn test_parse_frame_bare_cr_lines() {
    // Lines separated by bare \r — str::lines() would fail here
    let frame = parse_frame("event: ping\rdata: {}").unwrap();
    assert_eq!(frame.event_type.as_deref(), Some("ping"));
    assert_eq!(frame.data, "{}");
}

// --- split_sse_lines tests ---

#[test]
fn test_split_sse_lines_lf() {
    let lines: Vec<_> = split_sse_lines("a\nb\nc").collect();
    assert_eq!(lines, vec!["a", "b", "c"]);
}

#[test]
fn test_split_sse_lines_crlf() {
    let lines: Vec<_> = split_sse_lines("a\r\nb\r\nc").collect();
    assert_eq!(lines, vec!["a", "b", "c"]);
}

#[test]
fn test_split_sse_lines_bare_cr() {
    let lines: Vec<_> = split_sse_lines("a\rb\rc").collect();
    assert_eq!(lines, vec!["a", "b", "c"]);
}

#[test]
fn test_split_sse_lines_mixed() {
    let lines: Vec<_> = split_sse_lines("a\nb\rc\r\nd").collect();
    assert_eq!(lines, vec!["a", "b", "c", "d"]);
}

// --- find_frame_boundary tests ---

#[test]
fn test_find_frame_boundary_lf_lf() {
    assert_eq!(find_frame_boundary(b"data: x\n\n"), Some((7, 2)));
}

#[test]
fn test_find_frame_boundary_crlf_crlf() {
    assert_eq!(find_frame_boundary(b"data: x\r\n\r\n"), Some((7, 4)));
}

#[test]
fn test_find_frame_boundary_cr_cr() {
    assert_eq!(find_frame_boundary(b"data: x\r\r"), Some((7, 2)));
}

#[test]
fn test_find_frame_boundary_lf_crlf() {
    // Mixed: LF then CRLF
    assert_eq!(find_frame_boundary(b"data: x\n\r\n"), Some((7, 3)));
}

#[test]
fn test_find_frame_boundary_crlf_lf() {
    // Mixed: CRLF then LF
    assert_eq!(find_frame_boundary(b"data: x\r\n\n"), Some((7, 3)));
}

#[test]
fn test_find_frame_boundary_crlf_cr() {
    // Mixed: CRLF then CR
    assert_eq!(find_frame_boundary(b"data: x\r\n\r"), Some((7, 3)));
}

#[test]
fn test_find_frame_boundary_cr_lf_not_crlf_pair() {
    // \r followed by \n is a single CRLF, not two EOLs.
    // Need another EOL after to form a boundary.
    assert_eq!(find_frame_boundary(b"data: x\r\n"), None);
}

#[test]
fn test_find_frame_boundary_none() {
    assert_eq!(find_frame_boundary(b"data: x\n"), None);
}

#[test]
fn test_find_frame_boundary_earliest_wins() {
    // \n\n at pos 7, \r\r later — \n\n should win
    assert_eq!(find_frame_boundary(b"data: x\n\ndata: y\r\r"), Some((7, 2)));
}

#[test]
fn test_find_frame_boundary_empty() {
    assert_eq!(find_frame_boundary(b""), None);
}
