//! Through a prefill/decode pair the client's `cached_tokens` is the prefill
//! leg's prefix-cache hit. The decode leg receives the prompt's KV from the
//! prefill leg and reports what it loaded as cached (the prompt less one
//! token) whatever the cache held, so a deployment reading its hit rate from
//! the decode leg's number saw a full hit on every request.
use std::time::Instant;

use openai_protocol::chat::ChatCompletionRequest;
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::{
    eof_tests::{
        chunk, collect_events, complete_with_prompt, processor, scripted_stream, CharacterTokenizer,
    },
    *,
};
use crate::routers::grpc::{
    common::response_collection::collect_responses,
    context::ExecutionResult,
    proto_wrapper::{PrefillCachedTokens, ProtoGenerateComplete},
};

/// The prompt's length; the decode leg reports all but its last token as
/// cached, the prefill leg the cache's real hit.
const PROMPT_TOKENS: u32 = 2_081;
const PREFILL_HIT: u32 = 1_000;
const DECODE_VIEW: u32 = PROMPT_TOKENS - 1;

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "chatcmpl-pd-usage".to_string(),
        model: "pd-usage-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

fn pd_timing() -> context::PdTiming {
    context::PdTiming {
        prefill_start: Instant::now(),
        decode_sent: Instant::now(),
        runtime: "test",
    }
}

/// A leg's terminal frame reporting `cached` of the prompt's tokens as a
/// prefix-cache hit.
fn complete_with_cached(cached: u32) -> proto::GenerateResponse {
    let mut response = complete_with_prompt(0, "stop", PROMPT_TOKENS);
    if let Some(GenerationEvent::Complete(complete)) = &mut response.response {
        complete.cached_tokens = cached;
    }
    response
}

/// A decode-leg chunk that, like the engine's, repeats the leg's own counts.
fn chunk_with_cached(text: &str, cached: u32) -> proto::GenerateResponse {
    let mut response = chunk(0, text);
    if let Some(GenerationEvent::Chunk(chunk)) = &mut response.response {
        chunk.prompt_tokens = PROMPT_TOKENS;
        chunk.cached_tokens = cached;
    }
    response
}

fn spec(continuous_usage: bool) -> ChatResponseSpec {
    let request = serde_json::json!({
        "model": "pd-usage-test", "messages": [], "stream": true,
        "stream_options": {"include_usage": true, "continuous_usage_stats": continuous_usage}
    });
    ChatResponseSpec::from(
        &serde_json::from_value::<ChatCompletionRequest>(request).expect("chat request"),
    )
}

/// Stream a disaggregated chat answer from the two legs and return its SSE
/// events; the processor must succeed.
async fn stream_pd(
    prefill: ProtoStream,
    decode: ProtoStream,
    continuous_usage: bool,
) -> Vec<Value> {
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_prefill_decode_streaming_chunks(
            prefill,
            decode,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            spec(continuous_usage),
            &tx,
            pd_timing(),
            vec![None],
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    assert!(result.is_ok(), "{result:?}");
    events
}

fn cached_tokens_of(usage: &Value) -> &Value {
    &usage["prompt_tokens_details"]["cached_tokens"]
}

/// The parallel protocol: the prefill leg's completion reaches the streaming
/// layer, which drains it before it reads the decode leg.
#[tokio::test]
async fn a_streamed_pd_answer_reports_the_prefill_legs_cache_hit() {
    for continuous_usage in [false, true] {
        let (prefill, prefill_server) =
            scripted_stream(vec![complete_with_cached(PREFILL_HIT)], "0").await;
        let (decode, decode_server) = scripted_stream(
            vec![
                chunk_with_cached("hi", DECODE_VIEW),
                complete_with_cached(DECODE_VIEW),
            ],
            "0",
        )
        .await;
        let events = stream_pd(prefill, decode, continuous_usage).await;
        prefill_server.abort();
        decode_server.abort();

        let usage = &events.last().expect("the usage event")["usage"];
        assert_eq!(usage["prompt_tokens"], PROMPT_TOKENS, "{usage}");
        assert_eq!(cached_tokens_of(usage), PREFILL_HIT, "{usage}");
        for event in &events[..events.len() - 1] {
            if !event["usage"].is_null() {
                assert_eq!(cached_tokens_of(&event["usage"]), PREFILL_HIT, "{event}");
            }
        }
        assert!(events
            .iter()
            .any(|event| event["choices"][0]["delta"]["content"] == "hi"));
    }
}

/// The sequential protocol: the execution stage drains the prefill leg itself
/// and hands the streaming layer an exhausted prefill stream and a decode leg
/// stamped with the prefill's hit (`ProtoStream::decode_leg`).
#[tokio::test]
async fn a_decode_leg_stamped_by_the_execution_stage_reports_the_prefill_legs_cache_hit() {
    let (prefill, prefill_server) = scripted_stream(vec![], "0").await;
    let (decode, decode_server) = scripted_stream(
        vec![
            chunk_with_cached("hi", DECODE_VIEW),
            complete_with_cached(DECODE_VIEW),
        ],
        "0",
    )
    .await;
    let mut prefill_hit = PrefillCachedTokens::default();
    prefill_hit.note(&ProtoGenerateComplete::Vllm(Box::new(
        proto::GenerateComplete {
            prompt_tokens: PROMPT_TOKENS,
            cached_tokens: PREFILL_HIT,
            ..Default::default()
        },
    )));
    let events = stream_pd(prefill, decode.decode_leg(prefill_hit), true).await;
    prefill_server.abort();
    decode_server.abort();

    for event in &events {
        if !event["usage"].is_null() {
            assert_eq!(cached_tokens_of(&event["usage"]), PREFILL_HIT, "{event}");
        }
    }
    let usage = &events.last().expect("the usage event")["usage"];
    assert_eq!(usage["prompt_tokens"], PROMPT_TOKENS, "{usage}");
}

/// The unary path: `collect_responses` drains the prefill leg and collects
/// the decode leg's completions, which every non-streamed chat, completion,
/// generate and Responses answer builds its usage from.
#[tokio::test]
async fn a_unary_pd_answer_reports_the_prefill_legs_cache_hit() {
    let (prefill, prefill_server) =
        scripted_stream(vec![complete_with_cached(PREFILL_HIT)], "0").await;
    let (decode, decode_server) =
        scripted_stream(vec![complete_with_cached(DECODE_VIEW)], "0").await;
    let result = ExecutionResult::PrefillDecode {
        prefill,
        decode: Box::new(decode),
        prefill_guards: vec![None],
        pd_timing: pd_timing(),
        prefill_input_logprobs: None,
    };
    let completes = collect_responses(result, false)
        .await
        .unwrap_or_else(|_| panic!("the decode leg's completion"));
    prefill_server.abort();
    decode_server.abort();

    assert_eq!(completes.len(), 1);
    assert_eq!(completes[0].prompt_tokens(), PROMPT_TOKENS);
    assert_eq!(completes[0].cached_tokens(), PREFILL_HIT);
}

/// A single engine's answer keeps its own cache hit: the control.
#[tokio::test]
async fn a_single_engines_answer_keeps_its_own_cache_hit() {
    let (stream, server) = scripted_stream(
        vec![
            chunk_with_cached("hi", PREFILL_HIT),
            complete_with_cached(PREFILL_HIT),
        ],
        "0",
    )
    .await;
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            spec(false),
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");

    let usage = &events.last().expect("the usage event")["usage"];
    assert_eq!(usage["prompt_tokens"], PROMPT_TOKENS, "{usage}");
    assert_eq!(cached_tokens_of(usage), PREFILL_HIT, "{usage}");
}

/// A prefill that completed once (one completion for the whole request)
/// covers every sample; one that completed per sample covers each by its
/// index only.
#[test]
fn a_prefill_that_completed_once_covers_every_sample_of_its_request() {
    let complete = |index: u32, cached_tokens: u32| {
        ProtoGenerateComplete::Vllm(Box::new(proto::GenerateComplete {
            index,
            cached_tokens,
            ..Default::default()
        }))
    };
    let mut hit = PrefillCachedTokens::default();
    assert!(hit.is_empty());
    assert_eq!(hit.for_index(0), None);
    hit.note(&complete(0, PREFILL_HIT));
    assert!(!hit.is_empty());
    assert_eq!(hit.for_index(0), Some(PREFILL_HIT));
    assert_eq!(hit.for_index(3), Some(PREFILL_HIT));
    hit.note(&complete(1, 20));
    assert_eq!(hit.for_index(1), Some(20));
    assert_eq!(hit.for_index(3), None);
}
