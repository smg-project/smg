//! Through a prefill/decode pair the client's `cached_tokens` is the prefill
//! side's prefix-cache hit. The decode side receives the prompt's KV from the
//! prefill side and reports what it loaded as cached (the prompt less one
//! token) whatever the cache held, so a deployment reading its hit rate from
//! the decode side's number saw a full hit on every request.
use std::time::Instant;

use openai_protocol::chat::ChatCompletionRequest;
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::{sync::oneshot, task::JoinHandle};

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

/// The prompt's length; the decode side reports all but its last token as
/// cached, the prefill side the cache's real hit.
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
        runtime: "sglang",
    }
}

/// A side's terminal frame reporting `cached` of the prompt's tokens as a
/// prefix-cache hit.
fn complete_with_cached(cached: u32) -> proto::GenerateResponse {
    let mut response = complete_with_prompt(0, "stop", PROMPT_TOKENS);
    if let Some(GenerationEvent::Complete(complete)) = &mut response.response {
        complete.cached_tokens = cached;
    }
    response
}

/// A decode-side chunk that, like the engine's, repeats the side's own counts.
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

/// Stream a disaggregated chat answer from the two sides and return its SSE
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

/// The parallel protocol: the prefill side's completion reaches the streaming
/// layer, which drains it before it reads the decode side.
#[tokio::test]
async fn a_streamed_pd_answer_reports_the_prefill_sides_cache_hit() {
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

/// The sequential protocol: the execution stage drains the prefill side itself
/// and hands the streaming layer an exhausted prefill stream and a decode side
/// stamped with the prefill's hit (`ProtoStream::decode_side`).
#[tokio::test]
async fn a_decode_side_stamped_by_the_execution_stage_reports_the_prefill_sides_cache_hit() {
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
    let events = stream_pd(prefill, decode.decode_side(prefill_hit), true).await;
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

/// The unary path: `collect_responses` drains the prefill side and collects
/// the decode side's completions, which every non-streamed chat, completion,
/// generate and Responses answer builds its usage from.
#[tokio::test]
async fn a_unary_pd_answer_reports_the_prefill_sides_cache_hit() {
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
        .unwrap_or_else(|_| panic!("the decode side's completion"));
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

/// The usage fixtures encoded as real SGLang frames, with a gate between
/// stages. Only fields used by these fixtures are translated.
async fn sglang_gated_stream(
    first: Vec<proto::GenerateResponse>,
    gate: oneshot::Receiver<()>,
    rest: Vec<proto::GenerateResponse>,
    status: &'static str,
) -> (ProtoStream, JoinHandle<()>) {
    use futures::{stream, StreamExt as _};
    use smg_grpc_client::sglang_scheduler::{
        proto as sg, proto::generate_response::Response as Event, AbortOnDropStream,
        SglangSchedulerClient,
    };

    use super::eof_tests::{data_frame, decoded_stream, mock_worker_endpoint, trailers_frame};

    let frames = |responses: Vec<proto::GenerateResponse>| {
        responses
            .into_iter()
            .map(|r| {
                let response = match r.response.expect("fixture event") {
                    GenerationEvent::Chunk(c) => Event::Chunk(sg::GenerateStreamChunk {
                        index: c.index,
                        token_ids: c.token_ids,
                        prompt_tokens: c.prompt_tokens,
                        completion_tokens: c.completion_tokens,
                        cached_tokens: c.cached_tokens,
                        ..Default::default()
                    }),
                    GenerationEvent::Complete(c) => Event::Complete(sg::GenerateComplete {
                        index: c.index,
                        finish_reason: c.finish_reason,
                        prompt_tokens: c.prompt_tokens,
                        completion_tokens: c.completion_tokens,
                        cached_tokens: c.cached_tokens,
                        ..Default::default()
                    }),
                };
                Ok::<_, tonic::Status>(data_frame(&sg::GenerateResponse {
                    response: Some(response),
                    ..Default::default()
                }))
            })
            .collect::<Vec<_>>()
    };
    let body = stream::iter(frames(first))
        .chain(
            stream::once(async move {
                let _ = gate.await;
                stream::iter(frames(rest))
            })
            .flatten(),
        )
        .chain(stream::iter([Ok(trailers_frame(status))]));
    let (url, server) = mock_worker_endpoint().await;
    let client = SglangSchedulerClient::connect(&url)
        .await
        .expect("connect mock");
    let stream = AbortOnDropStream::new(decoded_stream(body), "pd-usage".into(), client);
    stream.mark_completed(); // No generation was dispatched to this mock.
    (ProtoStream::Sglang(stream), server)
}

/// A late Prefill control completion must not hold back real Decode text,
/// but its cache accounting must still precede successful final usage.
#[tokio::test]
async fn sglang_decode_text_precedes_prefill_completion_but_usage_waits() {
    for late_error in [false, true] {
        use std::time::Duration;

        let (release, gate) = oneshot::channel();
        let (prefill, p_server) = sglang_gated_stream(
            vec![],
            gate,
            if late_error {
                vec![]
            } else {
                vec![complete_with_cached(PREFILL_HIT)]
            },
            if late_error { "13" } else { "0" },
        )
        .await;
        let (decode_release, decode_gate) = oneshot::channel();
        decode_release.send(()).expect("open Decode gate");
        let (decode, d_server) = sglang_gated_stream(
            vec![
                chunk_with_cached("hi", DECODE_VIEW),
                complete_with_cached(DECODE_VIEW),
            ],
            decode_gate,
            vec![],
            "0",
        )
        .await;
        let (tx, mut rx) = sse_channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "test joins this task before returning"
        )]
        let task = tokio::spawn(async move {
            let mut timing = pd_timing();
            timing.runtime = "sglang";
            processor(false)
                .process_prefill_decode_streaming_chunks(
                    prefill,
                    decode,
                    dispatch(),
                    Arc::new(CharacterTokenizer::default()),
                    (None, None, false, false, false),
                    spec(false),
                    &tx,
                    timing,
                    vec![None],
                    None,
                )
                .await
        });
        let first = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let frame = rx
                    .recv()
                    .await
                    .expect("SSE remains open")
                    .expect("successful frame");
                let text = std::str::from_utf8(&frame).expect("UTF8");
                if text.contains("hi") {
                    break text.to_owned();
                }
            }
        })
        .await
        .expect("Decode text must not wait for Prefill");
        assert!(first.contains("hi"));
        assert!(
            !task.is_finished(),
            "terminal accounting still waits for Prefill"
        );
        release.send(()).expect("Prefill is still awaited");
        let events = collect_events(rx).await;
        let result = task.await.expect("task");
        if late_error {
            assert!(
                matches!(result, Err(utils::StreamFailure::Upstream { completion_tokens, .. }) if completion_tokens > 0)
            );
            assert!(
                events.iter().all(|e| e["usage"].is_null()),
                "failed stream must not emit successful final usage"
            );
        } else {
            result.expect("successful stream");
            let usage = &events.last().expect("final usage")["usage"];
            assert_eq!(cached_tokens_of(usage), PREFILL_HIT);
        }
        p_server.abort();
        d_server.abort();
    }
}

/// Dropping the HTTP body wakes both silent sides and releases their load,
/// including when Decode has already supplied its terminal response.
#[tokio::test]
async fn disconnect_during_pending_prefill_releases_both_sides() {
    use std::time::Duration;

    use crate::worker::{BasicWorkerBuilder, Worker};

    for decode_complete in [false, true] {
        let p: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p:1").build());
        let d: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://d:1").build());
        let (_p_release, p_gate) = oneshot::channel();
        let (_d_release, d_gate) = oneshot::channel();
        let (prefill, ps) = sglang_gated_stream(vec![], p_gate, vec![], "0").await;
        let mut frames = vec![chunk_with_cached("hi", DECODE_VIEW)];
        if decode_complete {
            frames.push(complete_with_cached(DECODE_VIEW));
        }
        let (decode, ds) = sglang_gated_stream(frames, d_gate, vec![], "0").await;
        let (tx, mut rx) = sse_channel();
        let prefill = prefill
            .tracked(Arc::clone(&p), 0, true)
            .until_client_gone(&tx, "cancel-pd");
        let decode = decode
            .tracked(Arc::clone(&d), 0, true)
            .until_client_gone(&tx, "cancel-pd");
        #[expect(clippy::disallowed_methods, reason = "test joins the task")]
        let task = tokio::spawn(async move {
            processor(false)
                .process_prefill_decode_streaming_chunks(
                    prefill,
                    decode,
                    dispatch(),
                    Arc::new(CharacterTokenizer::default()),
                    (None, None, false, false, false),
                    spec(false),
                    &tx,
                    pd_timing(),
                    vec![None],
                    None,
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            while let Some(frame) = rx.recv().await {
                let frame = frame.expect("successful frame");
                if std::str::from_utf8(&frame).expect("UTF8").contains("hi") {
                    return;
                }
            }
            panic!("Decode text missing");
        })
        .await
        .expect("early Decode text");
        drop(rx);
        let _result = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("disconnect ends the reader without opening either gate")
            .expect("joined reader");
        assert_eq!(p.tracked_load(), 0, "Prefill load released");
        assert_eq!(d.tracked_load(), 0, "Decode load released");
        ps.abort();
        ds.abort();
    }
}

/// Continuous usage must not expose Decode's KV-transfer count in early chunks.
#[tokio::test]
async fn continuous_usage_waits_for_prefill_before_emitting_chunks() {
    use std::time::Duration;

    let (release, gate) = oneshot::channel();
    let (prefill, ps) =
        sglang_gated_stream(vec![], gate, vec![complete_with_cached(PREFILL_HIT)], "0").await;
    let (decode_release, decode_gate) = oneshot::channel();
    decode_release.send(()).expect("open Decode gate");
    let (decode, ds) = sglang_gated_stream(
        vec![
            chunk_with_cached("hi", DECODE_VIEW),
            complete_with_cached(DECODE_VIEW),
        ],
        decode_gate,
        vec![],
        "0",
    )
    .await;
    let (tx, mut rx) = sse_channel();
    #[expect(clippy::disallowed_methods, reason = "test joins the reader")]
    let task = tokio::spawn(async move {
        processor(false)
            .process_prefill_decode_streaming_chunks(
                prefill,
                decode,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                spec(true),
                &tx,
                pd_timing(),
                vec![None],
                None,
            )
            .await
    });
    assert!(
        tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .is_err(),
        "continuous-usage chunks wait for exact Prefill accounting"
    );
    release.send(()).expect("Prefill is awaited");
    let events = collect_events(rx).await;
    task.await
        .expect("joined reader")
        .expect("successful stream");
    let usages: Vec<_> = events
        .iter()
        .filter_map(|event| event.get("usage").filter(|u| !u.is_null()))
        .collect();
    assert!(!usages.is_empty());
    for usage in usages {
        assert_eq!(cached_tokens_of(usage), PREFILL_HIT);
    }
    ps.abort();
    ds.abort();
}
