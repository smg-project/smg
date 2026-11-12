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
//! behind it is read, whether the body's first frame was an earlier prompt's
//! first token, its finish with no token ahead of it, or the chunk writer's
//! own send of its first chunk past the flush size, whose hand-off the
//! writer's path pays itself: the burst behind that chunk is decoded only
//! once the connection holds the frame.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures::{future::try_join_all, StreamExt as _};
use llm_tokenizer::{
    traits::{Decoder, Encoder, Encoding, TokenIdType, Tokenizer},
    SpecialTokens,
};
use openai_protocol::common::StringOrArray;
use serde_json::{json, Value};
use tokio::{sync::oneshot, task::JoinHandle};

use super::{
    super::*,
    eof::{
        chunk, chunk_of_ids, complete, dispatch, gated_stream, processor, pull, scripted_stream,
        staged_stream, CharacterTokenizer,
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

/// [`completion_spec`] with the prompt echoed in front of the first text.
fn completion_spec_with_echo() -> CompletionResponseSpec {
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "prompt": "test", "stream": true, "max_tokens": 16, "echo": true
    }))
    .expect("a completion request");
    CompletionResponseSpec::new(&request, Vec::new())
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

/// The later prompt of a batched completion: nothing until its gate opens,
/// then its first token with a burst (two more tokens and its finish) ready
/// behind it. Returns the gate's sender, the stream and the server task.
async fn later_prompt_stream() -> (oneshot::Sender<()>, ProtoStream, JoinHandle<()>) {
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
    (later_tx, later_stream, later_server)
}

/// The reading of a batched completion's two prompts, run together as the
/// fan-out of a batched request runs its units: the first prompt under
/// `first_spec` with `prompt_text` as its prompt, the later prompt under the
/// plain spec, the two sharing the body's first frame.
async fn two_prompt_reading(
    first_stream: ProtoStream,
    first_spec: CompletionResponseSpec,
    prompt_text: String,
    later_stream: ProtoStream,
    tokenizer: Arc<dyn Tokenizer>,
    tx: SseSender,
) -> ReadingEnd {
    let processor = processor(false);
    let spec = completion_spec();
    let body_first_frame = AtomicBool::new(false);
    let outcomes = try_join_all([
        processor.process_completion_streaming_chunks(
            first_stream,
            dispatch(),
            Arc::clone(&tokenizer),
            (None, None, false, false, false),
            &first_spec,
            &prompt_text,
            0,
            &tx,
            Some(&body_first_frame),
        ),
        processor.process_completion_streaming_chunks(
            later_stream,
            dispatch(),
            tokenizer,
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
    ReadingEnd::Complete
}

/// Open the gate of [`later_prompt_stream`]'s prompt and read its hand-off on
/// `body`: its first-token frame (`x`) leaves alone, the pull after it is
/// pending (the second yield of the hand-off, so the connection writes the
/// frame before the reading runs on into the burst), and the pull after that
/// is the burst behind it (`y`, `z`) as one frame.
async fn a_later_prompts_first_token_yields_twice(
    body: &mut DrivenBody,
    gate: oneshot::Sender<()>,
) {
    gate.send(()).expect("the later prompt's gate");
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
    let next = pull(body).await;
    assert!(
        next.is_pending(),
        "the pull after a later prompt's first-token frame must be pending, not its burst: {next:?}"
    );
    let next = pull(body).await;
    let Poll::Ready(Some(Ok(burst))) = next else {
        panic!("the pull after the pending one is the burst's frame: {next:?}");
    };
    let burst = String::from_utf8(burst.to_vec()).expect("text");
    assert!(
        burst.contains(r#""text":"y""#) && burst.contains(r#""text":"z""#),
        "the chunks ready behind the later prompt's first token share a frame: {burst}"
    );
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
        ReadingEnd::Complete
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
    let next = pull(&mut body).await;
    assert!(
        next.is_pending(),
        "the pull after the first token's frame must be pending, not a frame: {next:?}"
    );
    // The pull after that: the burst's frame, the reading having yielded
    // once at the hand-off. Pending here would be a second yield.
    let next = pull(&mut body).await;
    let Poll::Ready(Some(Ok(burst))) = next else {
        panic!("the pull after the pending one is the burst's frame: {next:?}");
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
    let (later_tx, later_stream, later_server) = later_prompt_stream().await;
    let (tx, rx) = driven_channel();
    let reading = two_prompt_reading(
        first_stream,
        completion_spec(),
        String::new(),
        later_stream,
        Arc::new(CharacterTokenizer::default()),
        tx,
    );
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
    let next = pull(&mut body).await;
    assert!(
        next.is_pending(),
        "the pull after the body's first frame must be pending: {next:?}"
    );
    // The later prompt's messages are ready: its first token's frame leaves
    // alone and its hand-off yields twice (the body's own yield went with its
    // first frame), so the connection writes the frame before the reading
    // runs on into the burst.
    a_later_prompts_first_token_yields_twice(&mut body, later_tx).await;
    first_end_tx.send(()).expect("the first prompt's end");
    let rest: Vec<_> = body.collect().await;
    assert!(
        rest.iter().all(Result::is_ok),
        "the stream ends without a failed write: {rest:?}"
    );
    first_server.abort();
    later_server.abort();
}

/// An earlier prompt that completes with no token (an immediate stop) hands
/// the body its first frame all the same, the finish's: a later prompt's
/// first-token frame is then not the body's first and its hand-off yields
/// twice, the pull after it pending and the pull after that its burst. The
/// claim of the body's first frame is made at a unit's first flush of any
/// kind, not at a first token's alone.
#[tokio::test]
async fn an_earlier_prompts_token_less_finish_is_the_bodys_first_frame_and_a_later_prompts_first_token_yields_twice(
) {
    // The first prompt: a Complete with no Chunk ahead of it.
    let (first_stream, first_server) = scripted_stream(vec![complete(0, "stop")], "0").await;
    let (later_tx, later_stream, later_server) = later_prompt_stream().await;
    let (tx, rx) = driven_channel();
    let reading = two_prompt_reading(
        first_stream,
        completion_spec(),
        String::new(),
        later_stream,
        Arc::new(CharacterTokenizer::default()),
        tx,
    );
    let mut body = DrivenBody::new(Box::pin(reading), rx);
    let first = body
        .next()
        .await
        .expect("the body ends only after the first prompt's finish")
        .expect("a write");
    let first = String::from_utf8(first.to_vec()).expect("text");
    assert!(
        first.contains(r#""finish_reason":"stop""#) && !first.contains(r#""text":"x""#),
        "the body's first frame is the first prompt's finish, with no token: {first}"
    );
    // The pull after the body's first frame: pending, the body's own yield.
    let next = pull(&mut body).await;
    assert!(
        next.is_pending(),
        "the pull after the body's first frame must be pending: {next:?}"
    );
    // The later prompt's hand-off yields twice, the body's first frame having
    // been the earlier prompt's finish.
    a_later_prompts_first_token_yields_twice(&mut body, later_tx).await;
    let rest: Vec<_> = body.collect().await;
    assert!(
        rest.iter().all(Result::is_ok),
        "the stream ends without a failed write: {rest:?}"
    );
    first_server.abort();
    later_server.abort();
}

/// The character tokenizer, counting the tokens it decoded (`decoded`): a
/// test reads how far the reading got when the body handed a frame out.
#[derive(Default)]
struct CountsDecodedTokens {
    inner: CharacterTokenizer,
    decoded: AtomicUsize,
}

impl Encoder for CountsDecodedTokens {
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

impl Decoder for CountsDecodedTokens {
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
        self.decoded.fetch_add(1, Ordering::Relaxed);
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
        self.decoded.fetch_add(1, Ordering::Relaxed);
        self.inner
            .decode_step_eager(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }
}

impl Tokenizer for CountsDecodedTokens {
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

/// An earlier prompt whose first chunk is past the flush size (the echo of a
/// long prompt in front of its first token) leaves through the chunk writer's
/// own send, with no flush of the read's: that frame is the body's first all
/// the same, so a later prompt's first-token frame is not and its hand-off
/// yields twice, the pull after it pending and the pull after that its burst;
/// the earlier prompt's own first token went with its first frame, so no
/// hand-off is paid on its second. The writer's send pays that frame's
/// hand-off itself: the burst ready behind the first chunk is decoded only
/// once the connection holds the frame, and leaves as the frame after it.
#[tokio::test]
async fn an_earlier_prompts_echo_past_the_flush_size_is_the_bodys_first_frame_and_a_later_prompts_first_token_yields_twice(
) {
    // The first prompt: a long prompt echoed in front of its first token, so
    // the first chunk's text alone fills the writer's buffer past its flush
    // size, with a burst ready behind that chunk; its end held back.
    let prompt_text = "p".repeat(CHUNK_FLUSH_BYTES + 1);
    let (first_end_tx, first_end) = oneshot::channel();
    let (first_stream, first_server) = gated_stream(
        vec![chunk(0, "a"), chunk(0, "b"), chunk(0, "c"), chunk(0, "d")],
        first_end,
        vec![complete(0, "stop")],
        "0",
    )
    .await;
    let (later_tx, later_stream, later_server) = later_prompt_stream().await;
    let (tx, rx) = driven_channel();
    // The tokenizer counts the tokens decoded: how far the reading got when
    // the body handed a frame out.
    let tokenizer = Arc::new(CountsDecodedTokens::default());
    let decoding = Arc::clone(&tokenizer);
    let reading = two_prompt_reading(
        first_stream,
        completion_spec_with_echo(),
        prompt_text,
        later_stream,
        decoding,
        tx,
    );
    let mut body = DrivenBody::new(Box::pin(reading), rx);
    let first = body
        .next()
        .await
        .expect("the body ends only after the first prompt's first chunk")
        .expect("a write");
    let first_len = first.len();
    let first = String::from_utf8(first.to_vec()).expect("text");
    assert!(
        first_len > CHUNK_FLUSH_BYTES
            && first.contains(r#""text":"ppp"#)
            && first.contains(r#"pa""#)
            && !first.contains(r#""text":"b""#),
        "the body's first frame is the first prompt's echo and first token, past the flush size, alone: {first_len} bytes"
    );
    // The writer's send of that frame paid its hand-off: the burst behind the
    // first chunk is not decoded until the connection holds the frame.
    assert_eq!(
        tokenizer.decoded.load(Ordering::Relaxed),
        1,
        "only the first chunk's token is decoded when the body hands the first frame out"
    );
    // The pull after the body's first frame: pending, the body's own yield.
    let next = pull(&mut body).await;
    assert!(
        next.is_pending(),
        "the pull after the body's first frame must be pending: {next:?}"
    );
    // The pull after that: the burst behind the first chunk, as one frame
    // (one yield at the writer's hand-off, the body's own made the second).
    let next = pull(&mut body).await;
    let Poll::Ready(Some(Ok(burst))) = next else {
        panic!("the pull after the pending one is the first prompt's burst: {next:?}");
    };
    let burst = String::from_utf8(burst.to_vec()).expect("text");
    assert!(
        burst.contains(r#""text":"b""#)
            && burst.contains(r#""text":"d""#)
            && !burst.contains(r#""text":"x""#),
        "the chunks ready behind the first prompt's first chunk share a frame: {burst}"
    );
    // The later prompt's hand-off yields twice, the body's first frame having
    // been the earlier prompt's.
    a_later_prompts_first_token_yields_twice(&mut body, later_tx).await;
    first_end_tx.send(()).expect("the first prompt's end");
    let rest: Vec<_> = body.collect().await;
    assert!(
        rest.iter().all(Result::is_ok),
        "the stream ends without a failed write: {rest:?}"
    );
    first_server.abort();
    later_server.abort();
}
