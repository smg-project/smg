use futures::StreamExt;

use super::*;

/// Helper to create a byte stream from string chunks (simulating an HTTP response body).
fn bytes_stream(
    chunks: Vec<&str>,
) -> impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin + 'static {
    let owned: Vec<Result<Bytes, reqwest::Error>> = chunks
        .into_iter()
        .map(|s| Ok(Bytes::from(s.to_string())))
        .collect();
    futures::stream::iter(owned)
}

#[tokio::test]
async fn test_sse_parses_openai_format() {
    let raw = "data: {\"id\":\"1\",\"choices\":[]}\n\ndata: {\"id\":\"2\",\"choices\":[]}\n\ndata: [DONE]\n\n";
    let stream = sse_stream(bytes_stream(vec![raw]));
    let results: Vec<_> = stream.collect().await;
    let events: Vec<SseEvent> = results
        .into_iter()
        .map(|r| r.expect("unexpected SSE parse error"))
        .collect();
    assert_eq!(events.len(), 2);
    assert!(events[0].data.contains("\"id\":\"1\""));
    assert!(events[1].data.contains("\"id\":\"2\""));
    assert!(events[0].event.is_none());
}

#[tokio::test]
async fn test_sse_parses_anthropic_format() {
    let raw = "event: message_start\ndata: {\"message\":{\"id\":\"msg_1\"}}\n\nevent: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n";
    let stream = sse_stream(bytes_stream(vec![raw]));
    let results: Vec<_> = stream.collect().await;
    let events: Vec<SseEvent> = results
        .into_iter()
        .map(|r| r.expect("unexpected SSE parse error"))
        .collect();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event.as_deref(), Some("message_start"));
    assert_eq!(events[1].event.as_deref(), Some("content_block_delta"));
}

#[tokio::test]
async fn test_sse_multi_chunk_split() {
    // Split the payload mid-line to exercise cross-chunk buffering.
    let stream = sse_stream(bytes_stream(vec![
        "data: {\"id\":",
        "\"1\",\"choices\":[]}\n\ndata: [DONE]\n\n",
    ]));
    let results: Vec<_> = stream.collect().await;
    let events: Vec<SseEvent> = results
        .into_iter()
        .map(|r| r.expect("unexpected SSE parse error"))
        .collect();
    assert_eq!(events.len(), 1);
    assert!(events[0].data.contains("\"id\":\"1\""));
}

#[tokio::test]
async fn test_sse_done_terminates_stream() {
    // After [DONE], the stream should end even if the transport hasn't closed.
    let raw = "data: {\"id\":\"1\"}\n\ndata: [DONE]\n\ndata: {\"id\":\"2\"}\n\n";
    let stream = sse_stream(bytes_stream(vec![raw]));
    let results: Vec<_> = stream.collect().await;
    let events: Vec<SseEvent> = results
        .into_iter()
        .map(|r| r.expect("unexpected SSE parse error"))
        .collect();
    // Only the first event should be yielded; [DONE] terminates the stream.
    assert_eq!(events.len(), 1);
    assert!(events[0].data.contains("\"id\":\"1\""));
}

#[tokio::test]
async fn test_sse_stale_event_type_not_leaked() {
    // An event: line followed by an empty boundary (no data) should not
    // leak the event type into the next real event.
    let raw = "event: stale\n\ndata: {\"id\":\"1\"}\n\n";
    let stream = sse_stream(bytes_stream(vec![raw]));
    let results: Vec<_> = stream.collect().await;
    let events: Vec<SseEvent> = results
        .into_iter()
        .map(|r| r.expect("unexpected SSE parse error"))
        .collect();
    assert_eq!(events.len(), 1);
    assert!(events[0].data.contains("\"id\":\"1\""));
    // The stale "stale" event type must NOT be attached to this event.
    assert!(events[0].event.is_none());
}

#[tokio::test]
async fn test_sse_multibyte_utf8_split_across_chunks() {
    // Split a multibyte UTF-8 character (é = 0xC3 0xA9) across two chunks
    // to verify the incremental decoder stitches it back together.
    let prefix = b"data: {\"text\":\"caf";
    let e_acute_byte1: &[u8] = &[0xC3]; // first byte of 'é'
    let e_acute_byte2: &[u8] = &[0xA9]; // second byte of 'é'
    let suffix = b"\"}\n\ndata: [DONE]\n\n";

    let chunk1: Vec<u8> = [prefix.as_slice(), e_acute_byte1].concat();
    let chunk2: Vec<u8> = [e_acute_byte2, suffix.as_slice()].concat();

    let owned: Vec<Result<Bytes, reqwest::Error>> =
        vec![Ok(Bytes::from(chunk1)), Ok(Bytes::from(chunk2))];
    let byte_stream = futures::stream::iter(owned);
    let stream = sse_stream(byte_stream);
    let results: Vec<_> = stream.collect().await;
    let events: Vec<SseEvent> = results
        .into_iter()
        .map(|r| r.expect("unexpected SSE parse error"))
        .collect();
    assert_eq!(events.len(), 1);
    assert!(events[0].data.contains("café"));
}
