//! A stop the gateway fires itself (a stop string or stop token id matched in
//! the gateway) ends the client's stream and, when the engine did not see it,
//! the engine's request: the next engine message after the stop decides. The
//! choice's Complete means the engine ended it on that step (an EOS id, a stop
//! the request also gave it): the stream is read to its end as always and the
//! Complete's counts are the usage. A chunk means the engine is generating
//! past the stop: the stream is not read further, and it is released
//! un-marked, which is the abort path a client disconnect takes, so an engine
//! that would have run to `max_tokens` for nobody gives its slot back within a
//! step. The client's semantics stay: finish "stop" with the matched stop, and
//! the usage through the stop's step with the counts of that chunk. Proven on
//! scripted frames whose Complete is held behind a gate that never opens (a
//! stream read past the tail would wait on it), on the mock worker crate's
//! engine over gRPC (its batch empties long before its tail would have ended)
//! and on a mock engine over the ZMQ link (the abort frame arrives before the
//! engine's script runs out; none comes for a stop the engine ended the
//! request on).

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use engine_zmq_client::{
    mock_engine::{connect_to_frontend, default_ready_response, EngineInbound},
    protocol::vllm::output::{
        EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
        StopReason,
    },
    EngineId,
};
use futures::StreamExt as _;
use openai_protocol::worker::RuntimeType;
use serde_json::{json, Value};
use smg_grpc_client::{
    tokenspeed_proto, tokenspeed_scheduler::TokenSpeedSchedulerClient, vllm_proto as vllm,
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle, time::timeout};

use super::{
    eof_tests::{
        chat_spec, chat_text, collect_events, complete, complete_with_prompt, dispatch,
        gated_stream, processor, scripted_stream, CharacterTokenizer,
    },
    *,
};
use crate::routers::{
    common::sse::sse_channel,
    grpc::zmq_client::{EosTokenIds, ZmqEngineClient},
};

/// A chunk of `text` (one token per character) that carries the prompt count
/// every engine chunk carries.
fn counted_chunk(index: u32, text: &str, prompt_tokens: u32) -> vllm::GenerateResponse {
    let token_ids: Vec<_> = text.chars().map(u32::from).collect();
    vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk {
                index,
                prompt_tokens,
                completion_tokens: token_ids.len() as u32,
                token_ids,
                ..Default::default()
            },
        )),
    }
}

/// A single-choice chat spec that asks for the usage chunk.
fn single_choice_spec() -> ChatResponseSpec {
    let mut spec = chat_spec(false);
    spec.separate_reasoning = false;
    spec.expected_choices = 1;
    spec.stream_options =
        Some(serde_json::from_value(json!({"include_usage": true})).expect("stream options"));
    spec
}

/// The stop params of a request with the stop string `stop`.
fn stop_string(stop: &str) -> (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool) {
    (
        Some(StringOrArray::String(stop.to_string())),
        None,
        false,
        false,
        true,
    )
}

fn finish_reasons(events: &[Value]) -> Vec<(u64, String)> {
    events
        .iter()
        .filter_map(|event| {
            let choice = &event["choices"][0];
            choice["finish_reason"]
                .as_str()
                .map(|reason| (choice["index"].as_u64().unwrap_or(0), reason.to_string()))
        })
        .collect()
}

fn usage(events: &[Value]) -> Value {
    events
        .iter()
        .find(|event| event["usage"].is_object())
        .map(|event| event["usage"].clone())
        .expect("the usage chunk")
}

/// A single-choice stream that streams on past the local stop (`ab`, `cSTOPd`,
/// then `zzzz`: a stop the engine did not see); the rest of its tail and its
/// Complete wait behind the returned gate, and a read past the tail's first
/// chunk would wait with them.
async fn stream_past_a_local_stop() -> (oneshot::Sender<()>, ProtoStream, JoinHandle<()>) {
    let (gate_tx, gate) = oneshot::channel::<()>();
    let (stream, server) = gated_stream(
        vec![
            counted_chunk(0, "ab", 7),
            counted_chunk(0, "cSTOPd", 7),
            counted_chunk(0, "zzzz", 7),
        ],
        gate,
        vec![counted_chunk(0, "more", 7), complete(0, "length")],
        "0",
    )
    .await;
    (gate_tx, stream, server)
}

#[tokio::test]
async fn a_local_stop_ends_the_chat_read_and_the_engine_request() {
    let (gate_tx, stream, server) = stream_past_a_local_stop().await;
    let (tx, rx) = sse_channel();
    let result = timeout(
        Duration::from_secs(5),
        processor(false).process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            stop_string("STOP"),
            single_choice_spec(),
            &tx,
            None,
        ),
    )
    .await
    .expect("the stream ends at the stop, without the engine's tail");
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    drop(gate_tx);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "content"), "abc");
    let finishes = finish_reasons(&events);
    assert_eq!(finishes, [(0, "stop".to_string())], "{events:?}");
    let finish = events
        .iter()
        .find(|event| event["choices"][0]["finish_reason"] == "stop")
        .unwrap();
    assert_eq!(finish["choices"][0]["matched_stop"], "STOP");
    // The usage is the choice's through the step the stop matched in, with
    // that chunk's prompt count: what the engine's own server reports; the
    // tail chunk read before the abort is nobody's usage.
    let usage = usage(&events);
    assert_eq!(usage["prompt_tokens"], 7);
    assert_eq!(usage["completion_tokens"], 8, "{usage}");
}

#[tokio::test]
async fn a_batched_request_is_aborted_only_when_every_choice_has_closed() {
    // Choice 0 stops locally on its second chunk while choice 1 goes on: its
    // later chunks (the engine streaming past a stop it did not see) are
    // ignored, not the stream. Choice 1's Complete closes the request; the
    // frames behind the gate are the rest of choice 0's tail, never read.
    let (gate_tx, gate) = oneshot::channel::<()>();
    let (stream, server) = gated_stream(
        vec![
            counted_chunk(0, "ab", 7),
            counted_chunk(0, "cSTOP", 7),
            counted_chunk(1, "xy", 7),
            counted_chunk(0, "tail", 7),
            counted_chunk(1, "z", 7),
            complete_with_prompt(1, "length", 7),
        ],
        gate,
        vec![
            counted_chunk(0, "more", 7),
            complete_with_prompt(0, "length", 7),
        ],
        "0",
    )
    .await;
    let (tx, rx) = sse_channel();
    let mut spec = single_choice_spec();
    spec.expected_choices = 2;
    let result = timeout(
        Duration::from_secs(5),
        processor(false).process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            stop_string("STOP"),
            spec,
            &tx,
            None,
        ),
    )
    .await
    .expect("the stream ends when the last choice closes");
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    drop(gate_tx);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "content"), "abc");
    assert_eq!(chat_text(&events, 1, "content"), "xyz");
    let mut finishes = finish_reasons(&events);
    finishes.sort();
    assert_eq!(
        finishes,
        [(0, "stop".to_string()), (1, "length".to_string())],
        "{events:?}"
    );
    // Choice 0 counts through its stop's step (2 + 5), choice 1 its three
    // tokens; the ignored "tail" chunk is nobody's usage.
    let usage = usage(&events);
    assert_eq!(usage["prompt_tokens"], 7);
    assert_eq!(usage["completion_tokens"], 10, "{usage}");
}

#[tokio::test]
async fn a_local_stop_ends_the_completion_read_and_the_engine_request() {
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "prompt": "test", "stream": true, "ignore_eos": true,
        "max_tokens": 4096, "stop": "STOP"
    }))
    .unwrap();
    let spec = CompletionResponseSpec::from(&request);
    let (gate_tx, stream, server) = stream_past_a_local_stop().await;
    let (tx, rx) = sse_channel();
    let outcome = timeout(
        Duration::from_secs(5),
        processor(false).process_completion_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            stop_string("STOP"),
            &spec,
            "",
            0,
            &tx,
        ),
    )
    .await
    .expect("the unit ends at the stop, without the engine's tail")
    .expect("the unit's outcome");
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    drop(gate_tx);
    let text: String = events
        .iter()
        .filter_map(|event| event["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, "abc");
    let finishes: Vec<_> = events
        .iter()
        .filter_map(|event| event["choices"][0]["finish_reason"].as_str())
        .collect();
    assert_eq!(finishes, ["stop"], "{events:?}");
    // The stop settled the choice's usage as a Complete would have.
    assert!(outcome.saw_complete);
    assert_eq!(outcome.prompt_tokens, 7);
    assert_eq!(outcome.completion_tokens, 8);
}

/// A plain Messages spec: text only, no thinking, no tools.
fn messages_spec() -> MessagesResponseSpec {
    MessagesResponseSpec {
        thinking_requested: false,
        thinking_text_returned: true,
        reasoning_in_prefill: false,
        tool_choice: None,
        has_tools: false,
        history_tool_calls_count: 0,
        chat_tools: Vec::new().into(),
        stop_sequences: None,
        symphony: None,
        first_tool_call_only: false,
    }
}

#[tokio::test]
async fn a_local_stop_ends_the_messages_read_and_the_engine_request() {
    let (gate_tx, stream, server) = stream_past_a_local_stop().await;
    let (tx, rx) = sse_channel();
    let result = timeout(
        Duration::from_secs(5),
        processor(false).process_messages_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            stop_string("STOP"),
            messages_spec(),
            &tx,
            None,
        ),
    )
    .await
    .expect("the message ends at the stop, without the engine's tail");
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    drop(gate_tx);
    assert!(result.is_ok(), "{result:?}");
    let text: String = events
        .iter()
        .filter_map(|event| event["delta"]["text"].as_str())
        .collect();
    assert_eq!(text, "abc");
    let final_delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .expect("message_delta");
    assert_eq!(final_delta["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(final_delta["delta"]["stop_sequence"], "STOP");
    // The input count is the chunk's, as settled as a Complete's.
    assert_eq!(final_delta["usage"]["input_tokens"], 7, "{final_delta}");
    assert_eq!(final_delta["usage"]["output_tokens"], 8, "{final_delta}");
}

/// The mock worker crate's engine on a loopback port, one decode step every
/// `step_ms`: a request is generated token by token until its `max_tokens`
/// or an `Abort` takes it out of the batch.
#[expect(
    clippy::disallowed_methods,
    reason = "bounded test fixture; server task is explicitly aborted"
)]
async fn mock_engine_worker(step_ms: f64) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock worker");
    let port = listener.local_addr().expect("mock worker address").port();
    let config = Arc::new(mock_worker::config::Config {
        admin_port: None,
        host: "127.0.0.1".to_string(),
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        model_id: "local-stop".to_string(),
        tokenizer_path: "local-stop".to_string(),
        output_tokens: 0,
        realistic: true,
        engine: mock_worker::engine::EngineParams {
            timing: mock_worker::engine::TimingModel::Linear {
                prefill_tps: 1.0e6,
                decode_base_ms: step_ms,
                decode_per_req_ms: 0.0,
            },
            ..Default::default()
        },
        ..mock_worker::config::Config::default()
    });
    let server = tokio::spawn(mock_worker::grpc::serve_with_listener(config, listener));
    (format!("http://127.0.0.1:{port}"), server)
}

/// A streamed generation of `max_new` tokens for the engine's request `id`.
fn engine_request(id: &str, max_new: u32) -> tokenspeed_proto::GenerateRequest {
    tokenspeed_proto::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(tokenspeed_proto::TokenizedInput {
            original_text: String::new(),
            input_ids: vec![1, 2, 3, 4],
        }),
        sampling_params: Some(tokenspeed_proto::SamplingParams {
            max_new_tokens: Some(max_new),
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    }
}

/// The engine's batch as its `GetLoads` reports it.
async fn running_requests(client: &TokenSpeedSchedulerClient) -> i32 {
    let loads = client.get_loads(Vec::new()).await.expect("loads");
    loads
        .loads
        .iter()
        .map(|load| load.num_running_reqs + load.num_waiting_reqs)
        .sum()
}

#[tokio::test]
async fn the_engine_is_aborted_at_the_local_stop_over_grpc() {
    const STEP_MS: f64 = 2.0;
    const MAX_TOKENS: u32 = 1000;
    let (url, server) = mock_engine_worker(STEP_MS).await;
    let client = TokenSpeedSchedulerClient::connect(&url)
        .await
        .expect("connect mock worker");

    // The engine's output ids are a function of its request id: a probe run
    // reads the first of them, the run under test stops on the third.
    let mut probe = client
        .generate(engine_request("local-stop-grpc", 6))
        .await
        .expect("probe generation");
    let mut ids = Vec::new();
    while let Some(item) = probe.next().await {
        if let Some(tokenspeed_proto::generate_response::Response::Chunk(chunk)) =
            item.expect("probe item").response
        {
            ids.extend(chunk.token_ids);
        }
    }
    probe.mark_completed();
    assert!(ids.len() >= 3, "{ids:?}");
    let stop_id = ids[2];

    let stream = client
        .generate(engine_request("local-stop-grpc", MAX_TOKENS))
        .await
        .expect("generation");
    let (tx, rx) = sse_channel();
    let started = Instant::now();
    let result = timeout(
        Duration::from_secs(30),
        processor(false).process_streaming_chunks(
            ProtoStream::TokenSpeed(stream),
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, Some(vec![stop_id]), false, false, true),
            single_choice_spec(),
            &tx,
            None,
        ),
    )
    .await
    .expect("the stream ends at the stop");
    let client_done = started.elapsed();
    // The engine's batch empties once the abort reaches it: long before the
    // tail of the generation would have ended it (the exact latency is a
    // loaded runner's scheduler's, so the bound is the tail's, not a step's).
    let idle_after = Instant::now();
    let mut running = running_requests(&client).await;
    while running > 0 && idle_after.elapsed() < Duration::from_secs(5) {
        tokio::time::sleep(Duration::from_millis(1)).await;
        running = running_requests(&client).await;
    }
    let idle_after = idle_after.elapsed();
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();

    assert!(result.is_ok(), "{result:?}");
    let expected: String = ids[..2]
        .iter()
        .map(|id| char::from_u32(*id).expect("a character token"))
        .collect();
    assert_eq!(chat_text(&events, 0, "content"), expected);
    assert_eq!(finish_reasons(&events), [(0, "stop".to_string())]);
    let usage = usage(&events);
    assert_eq!(usage["prompt_tokens"], 4);
    assert_eq!(usage["completion_tokens"], 3, "{usage}");
    assert_eq!(running, 0, "the engine still holds the request");
    let tail = Duration::from_millis((STEP_MS * f64::from(MAX_TOKENS)) as u64);
    assert!(
        client_done < tail / 4,
        "the client's stream took {client_done:?}: the engine's {MAX_TOKENS} steps, not the stop"
    );
    assert!(
        idle_after < tail / 2,
        "the engine kept the request {idle_after:?} after the client's finish (its tail would \
         have run {tail:?})"
    );
}

/// One engine step's output: `token_ids` for `request_id`, not finished.
fn engine_step(request_id: &str, token_ids: Vec<u32>) -> EngineCoreOutputs {
    EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids: token_ids,
            ..Default::default()
        }],
        ..Default::default()
    })
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "the mock engine runs beside the stream under test; its task is awaited"
)]
async fn the_engine_is_aborted_at_the_local_stop_over_the_zmq_link() {
    const STEP: Duration = Duration::from_millis(10);
    let base = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", base.path().join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::Vllm,
            Duration::from_secs(10)
        ),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let client = client.expect("adapter connect");
    let engine = engine.expect("mock engine");
    let stream = client
        .generate_vllm(vllm::GenerateRequest {
            request_id: "local-stop-zmq".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(4096),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        })
        .await
        .expect("generation");

    // The engine: one token per step, "abSTOPxyz..." (the stop completes on
    // the sixth), until the gateway's abort frame arrives; the tokens sent
    // after the stop's are the tail it generated for nobody, and the script
    // has ten of them.
    const SCRIPT: &str = "abSTOPxyzwvutsrq";
    const AFTER_STOP: u32 = (SCRIPT.len() - "abSTOP".len()) as u32;
    let engine_task = tokio::spawn(async move {
        let (mut input, mut output) = engine.split();
        let request = match input.recv().await.expect("the request") {
            EngineInbound::Add(request) => request,
            other => panic!("expected Add, got {other:?}"),
        };
        let mut sent_after_stop = 0u32;
        for (position, character) in SCRIPT.chars().enumerate() {
            output
                .send_outputs(&engine_step(
                    &request.request_id,
                    vec![u32::from(character)],
                ))
                .await
                .expect("output");
            if position > 5 {
                sent_after_stop += 1;
            }
            match timeout(STEP, input.recv()).await {
                Ok(Ok(EngineInbound::Abort(ids))) => {
                    assert_eq!(ids, vec![request.request_id.clone()]);
                    return Some(sent_after_stop);
                }
                Ok(other) => panic!("unexpected inbound {other:?}"),
                Err(_elapsed) => {}
            }
        }
        None
    });

    let (tx, rx) = sse_channel();
    let result = timeout(
        Duration::from_secs(10),
        processor(false).process_streaming_chunks(
            ProtoStream::Zmq(stream),
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            stop_string("STOP"),
            single_choice_spec(),
            &tx,
            None,
        ),
    )
    .await
    .expect("the stream ends at the stop");
    drop(tx);
    let events = collect_events(rx).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "content"), "ab");
    assert_eq!(finish_reasons(&events), [(0, "stop".to_string())]);
    assert_eq!(usage(&events)["completion_tokens"], 6);
    let sent_after_stop = timeout(Duration::from_secs(10), engine_task)
        .await
        .expect("the engine task ends")
        .expect("the engine task")
        .expect("the abort frame reached the engine");
    // The frame came before the script ran out (a loaded runner's scheduler
    // decides how many steps that takes, so the bound is the script's, not a
    // step's).
    assert!(
        sent_after_stop < AFTER_STOP,
        "the abort arrived {sent_after_stop} steps after the stop"
    );
}

/// The frames of a choice the engine ends on the step the gateway's stop
/// matches in (the EOS id `$` in the last chunk, as an engine sends it), its
/// Complete right behind them with a prompt count of its own, to tell its
/// usage from the chunks'.
fn ended_on_the_stop(prompt_of_complete: u32) -> Vec<vllm::GenerateResponse> {
    vec![
        counted_chunk(0, "ab", 7),
        counted_chunk(0, "c$", 7),
        complete_with_prompt(0, "stop", prompt_of_complete),
    ]
}

/// The stop params of a request without stops of its own: the tokenizer's
/// EOS ids are the decoder's stops.
fn no_stops() -> (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool) {
    (None, None, false, false, false)
}

/// A stop the engine saw as well (its EOS id here) ends the choice on the
/// step it matches in: the engine's Complete is the next message, the stream
/// is read to its end as always, nothing is aborted, and the Complete's counts
/// are the usage.
#[tokio::test]
async fn a_stop_the_engine_ends_the_chat_choice_on_is_read_to_the_engines_end() {
    let (stream, server) = scripted_stream(ended_on_the_stop(9), "0").await;
    let (tx, rx) = sse_channel();
    let result = timeout(
        Duration::from_secs(5),
        processor(false).process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::with_eos(vec![u32::from('$')])),
            no_stops(),
            single_choice_spec(),
            &tx,
            None,
        ),
    )
    .await
    .expect("the stream ends");
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "content"), "abc");
    assert_eq!(
        finish_reasons(&events),
        [(0, "stop".to_string())],
        "{events:?}"
    );
    // The Complete was read: its prompt count is the usage's, not the chunk's;
    // the output count has the EOS token in it, as the engine counts it.
    let usage = usage(&events);
    assert_eq!(usage["prompt_tokens"], 9, "{usage}");
    assert_eq!(usage["completion_tokens"], 4, "{usage}");
}

#[tokio::test]
async fn a_stop_the_engine_ends_the_completion_choice_on_is_read_to_the_engines_end() {
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "prompt": "test", "stream": true, "max_tokens": 4096
    }))
    .unwrap();
    let spec = CompletionResponseSpec::from(&request);
    let (stream, server) = scripted_stream(ended_on_the_stop(9), "0").await;
    let (tx, rx) = sse_channel();
    let outcome = timeout(
        Duration::from_secs(5),
        processor(false).process_completion_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::with_eos(vec![u32::from('$')])),
            no_stops(),
            &spec,
            "",
            0,
            &tx,
        ),
    )
    .await
    .expect("the unit ends")
    .expect("the unit's outcome");
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    let text: String = events
        .iter()
        .filter_map(|event| event["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, "abc");
    let finishes: Vec<_> = events
        .iter()
        .filter_map(|event| event["choices"][0]["finish_reason"].as_str())
        .collect();
    assert_eq!(finishes, ["stop"], "{events:?}");
    // The Complete was read: its prompt count is the unit's, not the chunk's;
    // the output count has the EOS token in it, as the engine counts it.
    assert!(outcome.saw_complete);
    assert_eq!(outcome.prompt_tokens, 9);
    assert_eq!(outcome.completion_tokens, 4);
}

#[tokio::test]
async fn a_stop_the_engine_ends_the_message_on_is_read_to_the_engines_end() {
    let (stream, server) = scripted_stream(ended_on_the_stop(9), "0").await;
    let (tx, rx) = sse_channel();
    let result = timeout(
        Duration::from_secs(5),
        processor(false).process_messages_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::with_eos(vec![u32::from('$')])),
            no_stops(),
            messages_spec(),
            &tx,
            None,
        ),
    )
    .await
    .expect("the message ends");
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    let text: String = events
        .iter()
        .filter_map(|event| event["delta"]["text"].as_str())
        .collect();
    assert_eq!(text, "abc");
    let final_delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .expect("message_delta");
    // The Complete was read: its input count is the message's, not the
    // chunk's; the output count has the EOS token in it, as the engine
    // counts it.
    assert_eq!(final_delta["usage"]["input_tokens"], 9, "{final_delta}");
    assert_eq!(final_delta["usage"]["output_tokens"], 4, "{final_delta}");
}

/// The engine's finishing step: `token_ids` for `request_id`, ended on the
/// stop id among them, as an engine ends a request on a stop it was given.
fn finished_step(request_id: &str, token_ids: Vec<u32>, stop_id: u32) -> EngineCoreOutputs {
    EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids: token_ids,
            finish_reason: Some(EngineCoreFinishReason::Stop),
            stop_reason: Some(StopReason::TokenId(stop_id)),
            ..Default::default()
        }],
        ..Default::default()
    })
}

/// A stop id the request gave the engine as well ends the request on the
/// engine's side on the step it matches in: the Complete follows the chunk,
/// the stream ends on its own, and no abort frame goes to the engine.
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "the mock engine runs beside the stream under test; its task is awaited"
)]
async fn the_zmq_link_sends_no_abort_for_a_stop_the_engine_ends_the_request_on() {
    let base = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", base.path().join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::Vllm,
            Duration::from_secs(10)
        ),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let client = client.expect("adapter connect");
    let engine = engine.expect("mock engine");
    let stop_id = u32::from('S');
    let stream = client
        .generate_vllm(vllm::GenerateRequest {
            request_id: "engine-stop-zmq".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(4096),
                stop_token_ids: vec![stop_id],
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        })
        .await
        .expect("generation");

    // The engine: "ab", then the stop id as its finishing step; then it
    // listens for an abort frame that must not come.
    let engine_task = tokio::spawn(async move {
        let (mut input, mut output) = engine.split();
        let request = match input.recv().await.expect("the request") {
            EngineInbound::Add(request) => request,
            other => panic!("expected Add, got {other:?}"),
        };
        for character in ['a', 'b'] {
            output
                .send_outputs(&engine_step(
                    &request.request_id,
                    vec![u32::from(character)],
                ))
                .await
                .expect("output");
        }
        output
            .send_outputs(&finished_step(&request.request_id, vec![stop_id], stop_id))
            .await
            .expect("finishing output");
        match timeout(Duration::from_millis(500), input.recv()).await {
            Ok(Ok(EngineInbound::Abort(ids))) => Some(ids),
            Ok(other) => panic!("unexpected inbound {other:?}"),
            Err(_elapsed) => None,
        }
    });

    let (tx, rx) = sse_channel();
    let result = timeout(
        Duration::from_secs(10),
        processor(false).process_streaming_chunks(
            ProtoStream::Zmq(stream),
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, Some(vec![stop_id]), false, false, true),
            single_choice_spec(),
            &tx,
            None,
        ),
    )
    .await
    .expect("the stream ends at the engine's end");
    drop(tx);
    let events = collect_events(rx).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "content"), "ab");
    assert_eq!(finish_reasons(&events), [(0, "stop".to_string())]);
    assert_eq!(usage(&events)["completion_tokens"], 3);
    let aborted = timeout(Duration::from_secs(10), engine_task)
        .await
        .expect("the engine task ends")
        .expect("the engine task");
    assert_eq!(
        aborted, None,
        "the engine ended the request itself, yet an abort frame came"
    );
}
