//! The completion stream's first token reaches the connection before the
//! reading runs on into a burst, with the one yield of its hand-off (the
//! stream's first frame is the first token's, so the body's own yield after
//! it makes the two); and a chunk the tokenizer cannot decode after the first
//! frame fails the stream as the chat stream does, after the chunks written
//! ahead of it have left: the loop flushes what it wrote since its last frame
//! before it returns the failure; and the finish of the open choices at the
//! engine's end, the stream's third exit, flushes the same way ahead of a
//! later choice's failed finish; and a batched request's later prompt, whose
//! first-token frame is not the body's first frame, gets the two yields of
//! its hand-off, so that frame too reaches the connection before the burst
//! behind it is read.

use std::{pin::Pin, sync::atomic::AtomicBool};

use futures::{future::try_join_all, Stream, StreamExt as _};
use llm_tokenizer::{
    traits::{Decoder, Encoder, Encoding, TokenIdType, Tokenizer},
    SpecialTokens,
};
use openai_protocol::common::StringOrArray;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::sync::oneshot;

use super::{
    super::*,
    eof::{
        chunk, complete, dispatch, gated_stream, processor, scripted_stream, staged_stream,
        CharacterTokenizer,
    },
};
use crate::routers::common::sse::sse_channel;

/// A one-choice streamed completion of up to 16 tokens, no stop.
fn completion_spec() -> CompletionResponseSpec {
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "prompt": "test", "stream": true, "max_tokens": 16
    }))
    .expect("a completion request");
    CompletionResponseSpec::new(&request, Vec::new())
}

/// A chunk of raw token ids: an id that is no character (a surrogate) is one
/// the character tokenizer cannot decode.
fn chunk_of_ids(index: u32, token_ids: Vec<u32>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index,
            completion_tokens: token_ids.len() as u32,
            token_ids,
            ..Default::default()
        })),
    }
}

/// The text of the completion events in the SSE frames, in order.
fn text_of(frames: &[Bytes]) -> String {
    frames
        .iter()
        .flat_map(|frame| {
            std::str::from_utf8(frame)
                .expect("UTF-8 SSE")
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .map(|data| serde_json::from_str::<Value>(data).expect("SSE JSON"))
                .collect::<Vec<Value>>()
        })
        .filter_map(|event| event["choices"][0]["text"].as_str().map(str::to_owned))
        .collect()
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
                None,
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

/// A chunk the tokenizer cannot decode after the first frame fails the
/// stream, and every chunk written ahead of it reaches the client first: the
/// second chunk, buffered behind the first token's frame because the failing
/// chunk was ready in the same poll, is flushed before the loop returns.
#[tokio::test]
async fn a_chunk_that_fails_to_decode_after_the_first_frame_flushes_the_chunks_written_ahead_of_it()
{
    // The first token on its own; then, once its frame left, a second chunk
    // and a chunk no tokenizer decodes, ready together.
    let (gate, gated) = oneshot::channel();
    let (stream, server) = staged_stream(
        vec![
            (None, vec![chunk(0, "hi")]),
            (
                Some(gated),
                vec![
                    chunk(0, " there"),
                    chunk_of_ids(0, vec![0xD800]),
                    complete(0, "stop"),
                ],
            ),
        ],
        "0",
    )
    .await;
    let spec = completion_spec();
    let (tx, mut rx) = sse_channel();
    let processor = processor(false);
    let reading = async {
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
                None,
            )
            .await;
        drop(tx);
        result
    };
    let reader = async {
        let mut frames = Vec::new();
        // The first token's frame: the reader opens the gate only after it
        // left.
        frames.push(rx.recv().await.expect("a frame").expect("a write"));
        gate.send(()).expect("the reader waits at the gate");
        while let Some(frame) = rx.recv().await {
            frames.push(frame.expect("a write"));
        }
        frames
    };
    let (result, frames) = tokio::join!(reading, reader);
    server.abort();
    let Err(failure) = result else {
        panic!("the chunk the tokenizer cannot decode fails the stream");
    };
    assert!(
        format!("{failure:?}").contains("decode"),
        "the failure names the decode: {failure:?}"
    );
    assert_eq!(
        text_of(&frames),
        "hi there",
        "every chunk written ahead of the failure reached the client: {frames:?}"
    );
}

/// The character tokenizer, except that it HOLDS the surrogate id at the
/// step, plain or eager (into the decode window, no text, no error),
/// instead of failing it:
/// the finish then decodes the window, as the production finish does, and
/// fails there. The one way to reach the finish's failure exit with every
/// chunk read, since the character tokenizer fails such an id at the read.
#[derive(Default)]
struct HoldsAnUndecodableId {
    inner: CharacterTokenizer,
}

impl Encoder for HoldsAnUndecodableId {
    fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
        self.inner.encode(text, add_special_tokens)
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        self.inner.encode_batch(texts, add_special_tokens)
    }
}

impl Decoder for HoldsAnUndecodableId {
    fn decode(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner.decode(ids, skip_special_tokens)
    }

    fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> anyhow::Result<Option<String>> {
        if token_id == 0xD800 {
            ids.push(token_id);
            return Ok(None);
        }
        self.inner
            .decode_step(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }

    fn decode_step_eager(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> anyhow::Result<Option<String>> {
        if token_id == 0xD800 {
            ids.push(token_id);
            return Ok(None);
        }
        self.inner
            .decode_step_eager(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }
}

impl Tokenizer for HoldsAnUndecodableId {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The third exit: at the engine's end without a `Complete`, the open
/// choices finish in order and their tails are written into the buffer;
/// a later choice's failed finish returns the failure with the earlier
/// choice's tail already written. The flush ahead of the failure sends it,
/// as the read's two exits send theirs: two choices, choice 0's tail held by
/// a stop-sequence prefix, choice 1 holding an id no tokenizer decodes until
/// its finish, and the client reads choice 0's tail before the error ends
/// the stream.
#[tokio::test]
async fn a_failed_finish_of_a_later_choice_flushes_the_earlier_choice_s_tail_written_ahead_of_it() {
    let (stream, server) = staged_stream(
        vec![(
            None,
            vec![
                chunk(0, "hi"),
                chunk(1, "yo"),
                chunk(0, "q"),
                chunk_of_ids(1, vec![0xD800]),
            ],
        )],
        "0",
    )
    .await;
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "prompt": "test", "stream": true, "max_tokens": 16, "n": 2
    }))
    .expect("a two-choice completion request");
    let spec = CompletionResponseSpec::new(&request, Vec::new());
    let (tx, mut rx) = sse_channel();
    let processor = processor(false);
    let reading = async {
        let result = processor
            .process_completion_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(HoldsAnUndecodableId::default()),
                (
                    Some(StringOrArray::Array(vec!["qzz".to_string()])),
                    None,
                    false,
                    false,
                    false,
                ),
                &spec,
                "",
                0,
                &tx,
                None,
            )
            .await;
        drop(tx);
        result
    };
    let reader = async {
        let mut frames = Vec::new();
        while let Some(frame) = rx.recv().await {
            frames.push(frame.expect("a write"));
        }
        frames
    };
    let (result, frames) = tokio::join!(reading, reader);
    server.abort();
    let Err(failure) = result else {
        panic!("choice 1's failed finish fails the stream");
    };
    assert!(
        failure.to_string().contains("character"),
        "the failure is the held id's decode at the finish: {failure}"
    );
    assert_eq!(
        text_of(&frames),
        "hiyoq",
        "choice 0's held tail (q) reached the client ahead of choice 1's failed finish: {frames:?}"
    );
}

/// A batched request's units share one body, whose own yield follows its
/// first frame only: that frame is the first prompt's first token, so a later
/// prompt's first-token frame is not the body's first and its hand-off yields
/// twice, as the Messages stream's does: the pull after that frame is pending
/// (the connection writes it) and the pull after that is its burst. One yield
/// short, with every unit on the one-yield path, the reading runs on into the
/// burst and the later prompt's first token reaches the client behind it.
#[tokio::test]
async fn a_later_prompts_first_token_is_pending_to_the_connection_before_its_burst_on_a_batched_completion_stream(
) {
    // The first prompt: its first token at once, its end held back.
    let (first_end_tx, first_end) = oneshot::channel();
    let (first_stream, first_server) = gated_stream(
        vec![chunk(0, "a")],
        first_end,
        vec![complete(0, "stop")],
        "0",
    )
    .await;
    // The later prompt: nothing until its gate opens, then its first token
    // with a burst ready behind it.
    let (later_tx, later) = oneshot::channel();
    let (later_stream, later_server) = staged_stream(
        vec![(
            Some(later),
            vec![
                chunk(0, "x"),
                chunk(0, "y"),
                chunk(0, "z"),
                complete(0, "stop"),
            ],
        )],
        "0",
    )
    .await;
    let (tx, rx) = driven_channel();
    let processor = processor(false);
    let spec = completion_spec();
    let reading = async move {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
        // The units of one request share the body's first frame, as the
        // fan-out of a batched request gives it to them.
        let body_first_frame = AtomicBool::new(false);
        let outcomes = try_join_all([
            processor.process_completion_streaming_chunks(
                first_stream,
                dispatch(),
                Arc::clone(&tokenizer),
                (None, None, false, false, false),
                &spec,
                "",
                0,
                &tx,
                Some(&body_first_frame),
            ),
            processor.process_completion_streaming_chunks(
                later_stream,
                dispatch(),
                Arc::clone(&tokenizer),
                (None, None, false, false, false),
                &spec,
                "",
                1,
                &tx,
                Some(&body_first_frame),
            ),
        ])
        .await;
        assert!(outcomes.is_ok(), "{:?}", outcomes.err());
    };
    let mut body = DrivenBody::new(Box::pin(reading), rx);
    let first = body
        .next()
        .await
        .expect("the body ends only after the first prompt's first token")
        .expect("a write");
    let first = String::from_utf8(first.to_vec()).expect("text");
    assert!(
        first.contains(r#""text":"a""#),
        "the body's first frame is the first prompt's first token: {first}"
    );
    // The pull after the body's first frame: pending, the body's own yield.
    let pull = std::future::poll_fn(|cx| Poll::Ready(Pin::new(&mut body).poll_next(cx))).await;
    assert!(
        pull.is_pending(),
        "the pull after the body's first frame must be pending: {pull:?}"
    );
    // The later prompt's messages are ready: its first token's frame leaves
    // alone, the burst behind it unread.
    later_tx.send(()).expect("the later prompt's gate");
    let later_first = body
        .next()
        .await
        .expect("the body ends only after the later prompt's first token")
        .expect("a write");
    let later_first = String::from_utf8(later_first.to_vec()).expect("text");
    assert!(
        later_first.contains(r#""text":"x""#) && !later_first.contains(r#""text":"y""#),
        "the later prompt's first token leaves alone: {later_first}"
    );
    // The pull after it: pending, the second yield of the hand-off (the
    // body's own yield went with its first frame), so the connection writes
    // the frame before the reading runs on into the burst.
    let pull = std::future::poll_fn(|cx| Poll::Ready(Pin::new(&mut body).poll_next(cx))).await;
    assert!(
        pull.is_pending(),
        "the pull after a later prompt's first-token frame must be pending, not its burst: {pull:?}"
    );
    // The pull after that: the burst's frame.
    let pull = std::future::poll_fn(|cx| Poll::Ready(Pin::new(&mut body).poll_next(cx))).await;
    let Poll::Ready(Some(Ok(burst))) = pull else {
        panic!("the pull after the pending one is the burst's frame: {pull:?}");
    };
    let burst = String::from_utf8(burst.to_vec()).expect("text");
    assert!(
        burst.contains(r#""text":"y""#) && burst.contains(r#""text":"z""#),
        "the chunks ready behind the later prompt's first token share a frame: {burst}"
    );
    first_end_tx.send(()).expect("the first prompt's end");
    let rest: Vec<_> = body.collect().await;
    assert!(
        rest.iter().all(Result::is_ok),
        "the stream ends without a failed write: {rest:?}"
    );
    first_server.abort();
    later_server.abort();
}
