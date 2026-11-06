//! The completion stream's first token reaches the connection before the
//! reading runs on into a burst, with the one yield of its hand-off (the
//! stream's first frame is the first token's, so the body's own yield after
//! it makes the two).

use std::pin::Pin;

use futures::{Stream, StreamExt as _};
use serde_json::json;

use super::{
    super::*,
    eof::{chunk, complete, dispatch, processor, scripted_stream, CharacterTokenizer},
};

/// A one-choice streamed completion of up to 16 tokens, no stop.
fn completion_spec() -> CompletionResponseSpec {
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "prompt": "test", "stream": true, "max_tokens": 16
    }))
    .expect("a completion request");
    CompletionResponseSpec::new(&request, Vec::new())
}

/// The first token's frame reaches the connection before the reading runs on
/// into a burst: the stream's first frame is the first token's, so the body's
/// own yield after it and the one yield of the packed read's hand-off give
/// the connection its pending pull, and the pull after that is the burst's
/// frame. A second yield at the hand-off (the Messages stream's, whose first
/// frame is not the first token's) would make that pull pending as well: two
/// polls per request for nothing on this stream.
#[tokio::test]
async fn the_first_tokens_frame_is_pending_to_the_connection_before_the_burst_is_read_on_the_completion_stream(
) {
    // Four chunks ready at once: the first token, then a burst behind it.
    let (stream, server) = scripted_stream(
        vec![
            chunk(0, "a"),
            chunk(0, "b"),
            chunk(0, "c"),
            chunk(0, "d"),
            complete(0, "stop"),
        ],
        "0",
    )
    .await;
    let (tx, rx) = driven_channel();
    let processor = processor(false);
    let spec = completion_spec();
    let reading = async move {
        let result = processor
            .process_completion_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                &spec,
                "",
                0,
                &tx,
            )
            .await;
        assert!(result.is_ok(), "{:?}", result.err());
    };
    let mut body = DrivenBody::new(Box::pin(reading), rx);
    let first = body
        .next()
        .await
        .expect("the body ends only after the first token")
        .expect("a write");
    let first = String::from_utf8(first.to_vec()).expect("text");
    assert!(
        first.contains(r#""text":"a""#) && !first.contains(r#""text":"b""#),
        "the body's first frame is the first token's, alone: {first}"
    );
    // The pull after it: pending (the body's own yield after its first
    // frame), so the connection writes the first token's frame before the
    // reading runs on.
    let pull = std::future::poll_fn(|cx| Poll::Ready(Pin::new(&mut body).poll_next(cx))).await;
    assert!(
        pull.is_pending(),
        "the pull after the first token's frame must be pending, not a frame: {pull:?}"
    );
    // The pull after that: the burst's frame, the reading having yielded
    // once at the hand-off. Pending here would be a second yield.
    let pull = std::future::poll_fn(|cx| Poll::Ready(Pin::new(&mut body).poll_next(cx))).await;
    let Poll::Ready(Some(Ok(burst))) = pull else {
        panic!("the pull after the pending one is the burst's frame: {pull:?}");
    };
    let burst = String::from_utf8(burst.to_vec()).expect("text");
    assert!(
        burst.contains(r#""text":"b""#) && burst.contains(r#""text":"d""#),
        "the chunks ready behind the first token share a frame: {burst}"
    );
    let rest: Vec<_> = body.collect().await;
    assert!(
        rest.iter().all(Result::is_ok),
        "the stream ends without a failed write: {rest:?}"
    );
    server.abort();
}
