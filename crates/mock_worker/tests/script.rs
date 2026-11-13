//! Scripted output on the three wires: what `--script` and
//! `POST /admin/script/{worker}` make a worker answer, frame by frame, and
//! what a request no script covers still gets.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};

use engine_zmq_client::{
    connect_handshake,
    protocol::vllm::{
        logprobs::TokenLogprob,
        output::{EngineCoreFinishReason, StopReason},
        request::EngineCoreRequest,
        sampling::EngineCoreSamplingParams,
    },
    EngineCoreClient,
};
use futures::StreamExt;
use mock_worker::{
    config::{Config, ReplayConfig},
    grpc, http, script, zmq,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto as ts, TokenSpeedSchedulerClient};
use tokio::net::TcpListener;
use ts::generate_response::Response as GenResp;

mod common;
use common::{free_port, generate};

const MODEL: &str = "script-test";

/// A canned worker (three synthetic tokens, `100..103`) with `script` loaded.
fn config(script: Option<PathBuf>) -> Config {
    Config {
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        output_tokens: 3,
        replay: ReplayConfig {
            script,
            ..ReplayConfig::default()
        },
        ..Config::default()
    }
}

/// Write `answers` as a script file, one per line.
fn script_file(dir: &Path, name: &str, answers: &[Value]) -> PathBuf {
    let path = dir.join(name);
    let text: String = answers.iter().map(|a| format!("{a}\n")).collect();
    std::fs::write(&path, text).expect("write the script");
    path
}

// ── gRPC ────────────────────────────────────────────────────────────────────

/// Serve `cfg` as a gRPC worker on a free port: the gateway's client and the
/// worker's name.
async fn start_grpc(cfg: Config) -> (TokenSpeedSchedulerClient, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(grpc::serve_with_listener(Arc::new(cfg), listener));
    let client = TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect to the mock worker");
    (client, format!("grpc:{port}"))
}

fn grpc_request(id: &str, input_ids: &[u32], stream: bool, logprobs: bool) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: input_ids.to_vec(),
            original_text: String::new(),
        }),
        sampling_params: Some(ts::SamplingParams::default()),
        stream,
        return_logprob: logprobs,
        ..Default::default()
    }
}

fn chunk(
    id: &str,
    token_ids: Vec<u32>,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    output_logprobs: Option<ts::OutputLogProbs>,
) -> ts::GenerateResponse {
    ts::GenerateResponse {
        request_id: id.to_string(),
        response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
            token_ids,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            output_logprobs,
            index: 0,
            weight_version: None,
        })),
    }
}

fn logprobs(token_ids: &[u32], values: &[f32]) -> Option<ts::OutputLogProbs> {
    Some(ts::OutputLogProbs {
        token_logprobs: values.to_vec(),
        token_ids: token_ids.to_vec(),
        top_logprobs: Vec::new(),
    })
}

#[tokio::test]
async fn grpc_streams_the_scripted_answer_frame_for_frame() {
    let dir = tempfile::tempdir().expect("temp dir");
    let script = script_file(
        dir.path(),
        "grpc.jsonl",
        &[json!({
            "input_ids": [1, 2, 3],
            "output_ids": [10, 11, 12],
            "finish_reason": "length",
            "matched_stop": "</s>",
            "cached_tokens": 2,
            "prompt_tokens": 7,
            "logprobs": [-0.5, -1.0, -1.5],
        })],
    );
    let (client, _) = start_grpc(config(Some(script))).await;

    let streamed = generate(&client, grpc_request("s", &[1, 2, 3], true, true)).await;
    let complete = ts::GenerateResponse {
        request_id: "s".to_string(),
        response: Some(GenResp::Complete(ts::GenerateComplete {
            output_ids: vec![10, 11, 12],
            finish_reason: "length".to_string(),
            prompt_tokens: 7,
            completion_tokens: 3,
            cached_tokens: 2,
            output_logprobs: logprobs(&[10, 11, 12], &[-0.5, -1.0, -1.5]),
            matched_stop: Some(ts::generate_complete::MatchedStop::MatchedStopStr(
                "</s>".to_string(),
            )),
            index: 0,
            weight_version: None,
            ..Default::default()
        })),
    };
    assert_eq!(
        streamed,
        vec![
            chunk("s", vec![10], 7, 1, 2, logprobs(&[10], &[-0.5])),
            chunk("s", vec![11], 7, 2, 2, logprobs(&[11], &[-1.0])),
            chunk("s", vec![12], 7, 3, 2, logprobs(&[12], &[-1.5])),
            complete.clone(),
        ]
    );

    // Not streamed, no logprobs asked: the completion alone, without them.
    let whole = generate(&client, grpc_request("w", &[1, 2, 3], false, false)).await;
    let mut expected = complete;
    expected.request_id = "w".to_string();
    if let Some(GenResp::Complete(complete)) = &mut expected.response {
        complete.output_logprobs = None;
    }
    assert_eq!(whole, vec![expected]);
}

#[tokio::test]
async fn grpc_answers_a_prompt_the_script_misses_as_before_and_counts_the_miss() {
    let dir = tempfile::tempdir().expect("temp dir");
    let script = script_file(
        dir.path(),
        "miss.jsonl",
        &[json!({"input_ids": [4, 4], "output_ids": [40]})],
    );
    let (client, worker) = start_grpc(config(Some(script))).await;

    let frames = generate(&client, grpc_request("m", &[5, 5], true, false)).await;
    let ids: Vec<u32> = frames
        .iter()
        .filter_map(|f| match &f.response {
            Some(GenResp::Chunk(chunk)) => Some(chunk.token_ids.clone()),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(ids, vec![100, 101, 102], "the canned answer");
    assert!(matches!(
        frames.last().and_then(|f| f.response.as_ref()),
        Some(GenResp::Complete(c)) if c.finish_reason == "stop" && c.output_ids == vec![100, 101, 102]
    ));
    generate(&client, grpc_request("h", &[4, 4], false, false)).await;
    let state = script::status(&worker).expect("a started worker");
    assert_eq!(
        (state["hits"].as_u64(), state["misses"].as_u64()),
        (Some(1), Some(1))
    );
}

#[tokio::test]
async fn grpc_cuts_frames_by_chunk_tokens_and_pauses_between_them() {
    let dir = tempfile::tempdir().expect("temp dir");
    let script = script_file(
        dir.path(),
        "chunks.jsonl",
        &[
            json!({"input_ids": [6], "output_ids": [1, 2, 3, 4, 5], "chunk_tokens": 2, "delay_ms": 40}),
        ],
    );
    let (client, _) = start_grpc(config(Some(script))).await;
    let started = Instant::now();
    let frames = generate(&client, grpc_request("c", &[6], true, false)).await;
    let elapsed = started.elapsed();
    let cuts: Vec<Vec<u32>> = frames
        .iter()
        .filter_map(|f| match &f.response {
            Some(GenResp::Chunk(chunk)) => Some(chunk.token_ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(cuts, vec![vec![1, 2], vec![3, 4], vec![5]]);
    assert!(
        elapsed >= Duration::from_millis(80),
        "two pauses of 40 ms before the second and third frames, took {elapsed:?}"
    );
}

#[tokio::test]
async fn grpc_encodes_a_scripted_text_with_the_workers_tokenizer() {
    // A word-level tokenizer in a directory, as a model directory holds one.
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"the":1,"answer":2,"is":3,"42":4},"unk_token":"[UNK]"},"pre_tokenizer":{"type":"Whitespace"}}"#,
    )
    .expect("write the tokenizer");
    let cfg = Config {
        tokenizer_path: dir.path().display().to_string(),
        ..config(None)
    };
    let (client, worker) = start_grpc(cfg).await;
    let state = script::enqueue(
        &worker,
        vec![script::Answer::parse(
            &json!({"output_text": "the answer is 42", "finish_reason": "stop"}),
        )
        .expect("a text answer")],
    )
    .expect("a started worker");
    assert_eq!(state["queued"], 1);

    let frames = generate(&client, grpc_request("t", &[7, 7], true, false)).await;
    let ids: Vec<u32> = frames
        .iter()
        .filter_map(|f| match &f.response {
            Some(GenResp::Chunk(chunk)) => Some(chunk.token_ids.clone()),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(ids, vec![1, 2, 3, 4], "the text's ids, one chunk each");
    assert!(matches!(
        frames.last().and_then(|f| f.response.as_ref()),
        Some(GenResp::Complete(c)) if c.output_ids == vec![1, 2, 3, 4] && c.completion_tokens == 4
    ));
}

#[tokio::test]
async fn grpc_fails_a_text_answer_it_cannot_encode_instead_of_answering_canned() {
    let cfg = Config {
        tokenizer_path: "/nonexistent/tokenizer-dir".to_string(),
        ..config(None)
    };
    let (client, worker) = start_grpc(cfg).await;
    script::enqueue(
        &worker,
        vec![
            script::Answer::parse(&json!({"output_text": "no tokenizer here"}))
                .expect("a text answer"),
        ],
    )
    .expect("a started worker");
    let error = match client
        .generate(grpc_request("e", &[7, 7], false, false))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("the call must fail before any frame"),
    };
    assert_eq!(error.code(), tonic::Code::FailedPrecondition, "{error}");
    assert!(error.message().contains("tokenizer"), "{error}");
    let state = script::status(&worker).expect("a started worker");
    assert_eq!(
        state["queued"], 0,
        "the answer was taken, and the failure says so"
    );
}

// ── the admin API ───────────────────────────────────────────────────────────

#[tokio::test]
async fn the_admin_api_enqueues_answers_in_order_reads_the_counts_and_clears() {
    let (client, worker) = start_grpc(config(None)).await;
    let admin_port = common::start_admin(config(None)).await;
    let http = common::http_client();
    let url = |worker: &str| format!("http://127.0.0.1:{admin_port}/admin/script/{worker}");

    // Two in order for this worker, one keyed for every worker. (The shared
    // queue, `all`, is not fed here: the tests of this binary run together
    // and any worker's next request would take from it; the store's own
    // tests cover it.)
    let queued = http
        .post(url(&worker))
        .json(&json!([
            {"output_ids": [1], "finish_reason": "stop"},
            {"output_ids": [2], "finish_reason": "length"},
            {"input_ids": [8, 8, 8], "output_ids": [88]},
        ]))
        .send()
        .await
        .expect("enqueue");
    assert_eq!(queued.status(), 200);
    let state: Value = queued.json().await.expect("state json");
    assert_eq!(
        (state["worker"].as_str(), state["queued"].as_u64()),
        (Some(worker.as_str()), Some(2))
    );
    let shared: Value = http
        .get(url("all"))
        .send()
        .await
        .expect("shared status")
        .json()
        .await
        .expect("state json");
    assert_eq!(shared["worker"], "all");
    assert!(shared["keyed"].as_u64().unwrap_or(0) >= 1, "{shared}");

    let finish =
        |frames: &[ts::GenerateResponse]| match frames.last().and_then(|f| f.response.as_ref()) {
            Some(GenResp::Complete(c)) => (c.output_ids.clone(), c.finish_reason.clone()),
            other => panic!("no completion: {other:?}"),
        };
    let first = generate(&client, grpc_request("1", &[9], false, false)).await;
    assert_eq!(finish(&first), (vec![1], "stop".to_string()));
    let second = generate(&client, grpc_request("2", &[9], false, false)).await;
    assert_eq!(finish(&second), (vec![2], "length".to_string()));
    let keyed = generate(&client, grpc_request("3", &[8, 8, 8], false, false)).await;
    assert_eq!(finish(&keyed), (vec![88], "stop".to_string()), "the key");
    let miss = generate(&client, grpc_request("4", &[9], false, false)).await;
    assert_eq!(
        finish(&miss).0,
        vec![100, 101, 102],
        "nothing left: the canned answer"
    );

    let state: Value = http
        .get(url(&worker))
        .send()
        .await
        .expect("status")
        .json()
        .await
        .expect("state json");
    assert_eq!(state["queued"], 0);
    assert_eq!(state["hits"], 3);
    assert_eq!(state["misses"], 1);
    assert!(state["keyed"].as_u64().unwrap_or(0) >= 1);

    // A queue is dropped on DELETE; a bad answer is refused with its reason.
    http.post(url(&worker))
        .json(&json!({"output_ids": [7]}))
        .send()
        .await
        .expect("enqueue");
    let cleared: Value = http
        .delete(url(&worker))
        .send()
        .await
        .expect("clear")
        .json()
        .await
        .expect("state json");
    assert_eq!(cleared["queued"], 0);
    let bad = http
        .post(url(&worker))
        .json(&json!([{"output_ids": [1]}, {"finish_reason": "stop"}]))
        .send()
        .await
        .expect("post");
    assert_eq!(bad.status(), 400);
    let why = bad.text().await.expect("text");
    assert!(why.starts_with("answer 1: "), "{why}");
    assert_eq!(
        script::status(&worker).expect("a started worker")["queued"],
        0,
        "a refused batch enqueues nothing"
    );

    // A name no worker of the process has is refused, as on the fault routes.
    for (method, name) in [("GET", "grpc:1"), ("POST", "grpc1"), ("DELETE", "http:2")] {
        let request = match method {
            "GET" => http.get(url(name)),
            "POST" => http.post(url(name)).json(&json!({"output_ids": [1]})),
            _ => http.delete(url(name)),
        };
        let response = request.send().await.expect(method);
        assert_eq!(response.status(), 404, "{method} {name}");
    }
}

// ── HTTP ────────────────────────────────────────────────────────────────────

/// Serve `cfg` as an HTTP worker on a free port: its base URL and name.
async fn start_http(cfg: Config) -> (String, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(http::serve_with_listener(Arc::new(cfg), listener));
    let base = format!("http://127.0.0.1:{port}");
    // The worker registers itself before it serves: once `/health` answers,
    // the script store and the admin API know it.
    common::http_client()
        .get(format!("{base}/health"))
        .send()
        .await
        .expect("the worker serves");
    (base, format!("http:{port}"))
}

/// The `data:` payloads of an SSE body, in order.
fn sse_data(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            frame
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("not a data frame: {frame:?}"))
                .to_string()
        })
        .collect()
}

#[tokio::test]
async fn http_delivers_the_scripted_text_as_frames_and_as_one_body() {
    let dir = tempfile::tempdir().expect("temp dir");
    let prompt = "hello scripted world";
    let script = script_file(
        dir.path(),
        "http.jsonl",
        &[json!({
            "input_ids": http::synth_token_ids(prompt),
            "output_ids": [20, 21],
            "output_text": ["Hello", " world"],
            "finish_reason": "stop",
            "matched_stop": "\n",
            "cached_tokens": 1,
            "logprobs": [-0.5, -1.0],
        })],
    );
    let (base, worker) = start_http(config(Some(script))).await;
    let http = common::http_client();
    let chat = |stream: bool| {
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "logprobs": true,
            "stream_options": {"include_usage": true},
        })
    };

    let streamed = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat(true))
        .send()
        .await
        .expect("stream");
    assert_eq!(streamed.status(), 200);
    assert!(streamed
        .headers()
        .get("content-type")
        .is_some_and(|v| v.to_str().unwrap_or("").starts_with("text/event-stream")));
    let frames = sse_data(&streamed.text().await.expect("body"));
    let chunk = |choice: Value| json!({"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": MODEL, "choices": [choice]});
    let logprob = |token: &str, value: f64| json!({"content": [{"token": token, "logprob": value, "bytes": null, "top_logprobs": []}]});
    let usage = json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "prompt_tokens_details": {"cached_tokens": 1}});
    let finish =
        chunk(json!({"index": 0, "delta": {}, "finish_reason": "stop", "matched_stop": "\n"}));
    let expected = [
        chunk(
            json!({"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null, "logprobs": logprob("Hello", -0.5)}),
        ),
        chunk(
            json!({"index": 0, "delta": {"content": " world"}, "finish_reason": null, "logprobs": logprob(" world", -1.0)}),
        ),
        finish,
    ];
    assert_eq!(frames.len(), 5, "{frames:?}");
    for (frame, expected) in frames.iter().zip(&expected) {
        let frame: Value = serde_json::from_str(frame).expect("a JSON frame");
        assert_eq!(&frame, expected);
    }
    let usage_chunk: Value = serde_json::from_str(&frames[3]).expect("a JSON frame");
    assert_eq!(
        usage_chunk,
        json!({"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": MODEL, "choices": [], "usage": usage}),
        "the usage rides in a chunk of its own after the finish frame"
    );
    assert_eq!(frames[4], "[DONE]");

    let whole: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat(false))
        .send()
        .await
        .expect("completion")
        .json()
        .await
        .expect("body json");
    assert_eq!(
        whole,
        json!({
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
                "matched_stop": "\n",
                "logprobs": {"content": [
                    {"token": "Hello", "logprob": -0.5, "bytes": null, "top_logprobs": []},
                    {"token": " world", "logprob": -1.0, "bytes": null, "top_logprobs": []},
                ]},
            }],
            "usage": usage,
        })
    );

    // The completions route answers in its own shape, logprobs asked as a count.
    let completion: Value = http
        .post(format!("{base}/v1/completions"))
        .json(&json!({"model": MODEL, "prompt": prompt, "logprobs": 1}))
        .send()
        .await
        .expect("completion")
        .json()
        .await
        .expect("body json");
    assert_eq!(completion["object"], "text_completion");
    assert_eq!(completion["choices"][0]["text"], "Hello world");
    assert_eq!(
        completion["choices"][0]["logprobs"]["tokens"],
        json!(["Hello", " world"])
    );
    assert_eq!(
        completion["choices"][0]["logprobs"]["token_logprobs"],
        json!([-0.5, -1.0])
    );

    // A prompt the script misses gets the canned answer; the miss is counted.
    let canned: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": MODEL, "messages": [{"role": "user", "content": "something else"}]}))
        .send()
        .await
        .expect("completion")
        .json()
        .await
        .expect("body json");
    assert_eq!(canned["choices"][0]["message"]["content"], "mock");
    let state = script::status(&worker).expect("a started worker");
    assert_eq!(
        (state["hits"].as_u64(), state["misses"].as_u64()),
        (Some(3), Some(1))
    );
}

// ── ZMQ ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn zmq_pushes_the_scripted_tokens_with_their_stats_finish_and_stop() {
    let dir = tempfile::tempdir().expect("temp dir");
    let endpoint = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (
        endpoint("hs.sock"),
        endpoint("in.sock"),
        endpoint("out.sock"),
    );
    let script = script_file(
        dir.path(),
        "zmq.jsonl",
        &[json!({
            "input_ids": [4, 5, 6],
            "output_ids": [20, 21],
            "finish_reason": "length",
            "matched_stop": 21,
            "cached_tokens": 1,
            "logprobs": [-0.25, -0.75],
        })],
    );
    // Rank 7 and prompts of this test's own: the store is process-wide.
    let rank = 7;
    #[expect(
        clippy::disallowed_methods,
        reason = "serve loops until the test drops its handles"
    )]
    let _engine = tokio::spawn(zmq::serve(
        Arc::new(config(Some(script))),
        handshake.clone(),
        rank,
    ));
    let transport = connect_handshake(&handshake, 1, &input, &output, Duration::from_secs(10))
        .await
        .expect("frontend handshake");
    let client = EngineCoreClient::new(transport);

    let collect = |request: EngineCoreRequest| async {
        let mut stream = client.submit(request).await.expect("submit");
        let mut outputs = Vec::new();
        while let Some(item) = stream.next().await {
            let output = item.expect("output ok");
            let finished = output.finished();
            outputs.push(output);
            if finished {
                break;
            }
        }
        outputs
    };

    let scripted = collect(EngineCoreRequest {
        request_id: "scripted".to_string(),
        prompt_token_ids: Some(vec![4, 5, 6]),
        sampling_params: Some(EngineCoreSamplingParams {
            logprobs: Some(0),
            ..Default::default()
        }),
        ..Default::default()
    })
    .await;
    assert_eq!(scripted.len(), 2, "{scripted:?}");
    let first = &scripted[0];
    assert_eq!(first.new_token_ids, vec![20]);
    assert_eq!(first.finish_reason, None);
    let prefill = first
        .prefill_stats
        .as_ref()
        .expect("prefill stats on the first batch");
    assert_eq!(
        (prefill.num_prompt_tokens, prefill.num_cached_tokens),
        (3, 1)
    );
    let logprobs = first
        .new_logprobs
        .as_ref()
        .expect("logprobs were asked for");
    assert_eq!(
        logprobs.positions[0].entries,
        vec![TokenLogprob {
            token_id: 20,
            logprob: -0.25,
            rank: 1
        }]
    );
    let last = &scripted[1];
    assert_eq!(last.new_token_ids, vec![21]);
    assert_eq!(last.finish_reason, Some(EngineCoreFinishReason::Length));
    assert_eq!(last.stop_reason, Some(StopReason::TokenId(21)));
    assert!(last.prefill_stats.is_none(), "the stats come once");

    // A prompt the script misses gets the canned tokens; the miss is counted.
    let canned = collect(EngineCoreRequest {
        request_id: "canned".to_string(),
        prompt_token_ids: Some(vec![31, 32, 33]),
        ..Default::default()
    })
    .await;
    let tokens: Vec<u32> = canned
        .iter()
        .flat_map(|o| o.new_token_ids.clone())
        .collect();
    assert_eq!(tokens, vec![100, 101, 102]);
    assert!(canned.iter().all(|o| o.new_logprobs.is_none()));
    let state = script::status(&format!("zmq:{rank}")).expect("a started rank");
    assert_eq!(
        (state["hits"].as_u64(), state["misses"].as_u64()),
        (Some(1), Some(1))
    );
}

// ── the request's token limit ───────────────────────────────────────────────

#[tokio::test]
async fn grpc_cuts_a_scripted_answer_at_the_requests_max_new_tokens() {
    let dir = tempfile::tempdir().expect("temp dir");
    let script = script_file(
        dir.path(),
        "cut.jsonl",
        &[json!({
            "input_ids": [7, 8, 9],
            "output_ids": [30, 31, 32],
            "finish_reason": "stop",
            "matched_stop": 32,
            "logprobs": [-0.1, -0.2, -0.3],
        })],
    );
    let (client, _) = start_grpc(config(Some(script))).await;
    let limited = |limit: u32| {
        let mut request = grpc_request("c", &[7, 8, 9], true, true);
        request.sampling_params = Some(ts::SamplingParams {
            max_new_tokens: Some(limit),
            ..Default::default()
        });
        request
    };

    // Longer than the limit: cut there, the length finish, no matched stop.
    let streamed = generate(&client, limited(2)).await;
    assert_eq!(
        streamed,
        vec![
            chunk("c", vec![30], 3, 1, 0, logprobs(&[30], &[-0.1])),
            chunk("c", vec![31], 3, 2, 0, logprobs(&[31], &[-0.2])),
            ts::GenerateResponse {
                request_id: "c".to_string(),
                response: Some(GenResp::Complete(ts::GenerateComplete {
                    output_ids: vec![30, 31],
                    finish_reason: "length".to_string(),
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    cached_tokens: 0,
                    output_logprobs: logprobs(&[30, 31], &[-0.1, -0.2]),
                    matched_stop: None,
                    index: 0,
                    weight_version: None,
                    ..Default::default()
                })),
            },
        ]
    );

    // Within the limit: the answer as scripted, its stop included.
    let whole = generate(&client, limited(5)).await;
    assert!(matches!(
        whole.last().and_then(|f| f.response.as_ref()),
        Some(GenResp::Complete(c))
            if c.output_ids == vec![30, 31, 32]
                && c.finish_reason == "stop"
                && c.matched_stop == Some(ts::generate_complete::MatchedStop::MatchedTokenId(32))
    ));
}

#[tokio::test]
async fn grpc_answers_an_empty_scripted_text_with_the_completion_alone() {
    // The word-level tokenizer of the encode test: an empty text encodes to no ids.
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"the":1},"unk_token":"[UNK]"},"pre_tokenizer":{"type":"Whitespace"}}"#,
    )
    .expect("write the tokenizer");
    let cfg = Config {
        tokenizer_path: dir.path().display().to_string(),
        ..config(None)
    };
    let (client, worker) = start_grpc(cfg).await;
    script::enqueue(
        &worker,
        vec![
            script::Answer::parse(&json!({"output_text": "", "finish_reason": "stop"}))
                .expect("an empty answer"),
        ],
    )
    .expect("a started worker");

    let frames = generate(&client, grpc_request("z", &[7, 7], true, false)).await;
    assert_eq!(
        frames,
        vec![ts::GenerateResponse {
            request_id: "z".to_string(),
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: Vec::new(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 2,
                completion_tokens: 0,
                cached_tokens: 0,
                output_logprobs: None,
                matched_stop: None,
                index: 0,
                weight_version: None,
                ..Default::default()
            })),
        }],
        "no chunk, a completion of zero tokens"
    );
}

#[tokio::test]
async fn http_cuts_a_scripted_text_at_the_requests_max_tokens() {
    let dir = tempfile::tempdir().expect("temp dir");
    let prompt = "count for me";
    let script = script_file(
        dir.path(),
        "http-cut.jsonl",
        &[json!({
            "input_ids": http::synth_token_ids(prompt),
            "output_text": ["one", " two", " three"],
            "finish_reason": "stop",
            "matched_stop": "\n",
        })],
    );
    let (base, _) = start_http(config(Some(script))).await;
    let http = common::http_client();
    let ask = |max_tokens: u32| {
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        })
    };

    let cut: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&ask(2))
        .send()
        .await
        .expect("response")
        .json()
        .await
        .expect("a JSON body");
    assert_eq!(cut["choices"][0]["message"]["content"], "one two", "{cut}");
    assert_eq!(cut["choices"][0]["finish_reason"], "length", "{cut}");
    assert!(cut["choices"][0]["matched_stop"].is_null(), "{cut}");
    assert_eq!(cut["usage"]["completion_tokens"], 2, "{cut}");

    let whole: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&ask(8))
        .send()
        .await
        .expect("response")
        .json()
        .await
        .expect("a JSON body");
    assert_eq!(
        whole["choices"][0]["message"]["content"], "one two three",
        "{whole}"
    );
    assert_eq!(whole["choices"][0]["finish_reason"], "stop", "{whole}");
    assert_eq!(whole["usage"]["completion_tokens"], 3, "{whole}");
}

#[tokio::test]
async fn zmq_cuts_a_scripted_answer_at_the_requests_max_tokens() {
    let dir = tempfile::tempdir().expect("temp dir");
    let endpoint = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (
        endpoint("hs.sock"),
        endpoint("in.sock"),
        endpoint("out.sock"),
    );
    let script = script_file(
        dir.path(),
        "zmq-cut.jsonl",
        &[json!({
            "input_ids": [14, 15, 16],
            "output_ids": [40, 41, 42],
            "finish_reason": "stop",
            "matched_stop": 42,
        })],
    );
    // Rank 8 and prompts of this test's own: the store is process-wide.
    let rank = 8;
    #[expect(
        clippy::disallowed_methods,
        reason = "serve loops until the test drops its handles"
    )]
    let _engine = tokio::spawn(zmq::serve(
        Arc::new(config(Some(script))),
        handshake.clone(),
        rank,
    ));
    let transport = connect_handshake(&handshake, 1, &input, &output, Duration::from_secs(10))
        .await
        .expect("frontend handshake");
    let client = EngineCoreClient::new(transport);

    let mut stream = client
        .submit(EngineCoreRequest {
            request_id: "cut".to_string(),
            prompt_token_ids: Some(vec![14, 15, 16]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 2,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("submit");
    let mut outputs = Vec::new();
    while let Some(item) = stream.next().await {
        let output = item.expect("output ok");
        let finished = output.finished();
        outputs.push(output);
        if finished {
            break;
        }
    }
    let tokens: Vec<u32> = outputs
        .iter()
        .flat_map(|o| o.new_token_ids.clone())
        .collect();
    assert_eq!(tokens, vec![40, 41], "cut after two tokens");
    let last = outputs.last().expect("a finishing batch");
    assert_eq!(last.finish_reason, Some(EngineCoreFinishReason::Length));
    assert_eq!(last.stop_reason, None, "no matched stop on a cut answer");
}

// ── the binary ──────────────────────────────────────────────────────────────

#[test]
fn the_binary_refuses_a_script_it_cannot_load_before_any_worker_starts() {
    let dir = tempfile::tempdir().expect("temp dir");
    let bad = script_file(dir.path(), "bad.jsonl", &[json!({"output_ids": [1]})]);
    for (path, reason) in [
        (dir.path().join("missing.jsonl"), "No such file"),
        (bad, "needs key or input_ids"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_mock-worker"))
            .args([
                "--grpc-base-port",
                &free_port().to_string(),
                "--grpc-count",
                "1",
            ])
            .arg("--script")
            .arg(&path)
            .output()
            .expect("run the mock worker");
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("cannot load script"), "{stderr}");
        assert!(stderr.contains(reason), "{stderr}");
    }
}

// ── load reports ────────────────────────────────────────────────────────────

/// The gRPC load once its running count reads `running` (within two seconds).
async fn grpc_running_settles_to(
    client: &TokenSpeedSchedulerClient,
    running: i32,
) -> ts::SchedulerLoad {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let load = client
                .get_loads(vec![])
                .await
                .expect("loads")
                .loads
                .remove(0);
            if load.num_running_reqs == running {
                return load;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the running count never read {running}"))
}

#[tokio::test]
async fn grpc_reports_the_scripted_answers_in_flight_as_running() {
    let dir = tempfile::tempdir().expect("temp dir");
    let script = script_file(
        dir.path(),
        "running.jsonl",
        &[json!({
            "input_ids": [8],
            "output_ids": [1, 2, 3, 4, 5],
            "chunk_tokens": 1,
            "delay_ms": 40,
        })],
    );
    let (client, _) = start_grpc(config(Some(script))).await;
    let idle = grpc_running_settles_to(&client, 0).await;
    assert_eq!(idle.num_total_reqs, 0, "{idle:?}");

    // The answer is paced over four pauses: a poll during them reads the one
    // request as running, and as the whole of the load.
    let (frames, busy) = tokio::join!(
        generate(&client, grpc_request("r", &[8], true, false)),
        grpc_running_settles_to(&client, 1)
    );
    assert_eq!(busy.num_total_reqs, 1, "{busy:?}");
    assert_eq!(busy.num_waiting_reqs, 0, "{busy:?}");
    assert_eq!(
        frames.len(),
        6,
        "five chunks and the completion: {frames:?}"
    );

    let idle = grpc_running_settles_to(&client, 0).await;
    assert_eq!(idle.num_total_reqs, 0, "{idle:?}");
}

#[tokio::test]
async fn a_worker_reports_its_own_requests_in_flight_not_the_process_total() {
    let dir = tempfile::tempdir().expect("temp dir");
    let script = script_file(
        dir.path(),
        "own.jsonl",
        &[json!({
            "input_ids": [8],
            "output_ids": [1, 2, 3, 4, 5],
            "chunk_tokens": 1,
            "delay_ms": 40,
        })],
    );
    let (serving, _) = start_grpc(config(Some(script))).await;
    let (other, _) = start_grpc(config(None)).await;
    grpc_running_settles_to(&serving, 0).await;

    // While the paced answer runs on one worker, the other worker of the same
    // process reports none of it: a gateway that polls both sees where the
    // work is.
    let (frames, (busy, idle)) = tokio::join!(
        generate(&serving, grpc_request("r", &[8], true, false)),
        async {
            let busy = grpc_running_settles_to(&serving, 1).await;
            let idle = other
                .get_loads(vec![])
                .await
                .expect("loads")
                .loads
                .remove(0);
            (busy, idle)
        }
    );
    assert_eq!(busy.num_total_reqs, 1, "{busy:?}");
    assert_eq!(
        idle.num_running_reqs, 0,
        "the other worker's own count: {idle:?}"
    );
    assert_eq!(idle.num_total_reqs, 0, "{idle:?}");
    assert_eq!(frames.len(), 6, "{frames:?}");
}

/// The HTTP load once its running count reads `running` (within two seconds).
async fn http_running_settles_to(http: &reqwest::Client, base: &str, running: u64) -> Value {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let body: Value = http
                .get(format!("{base}/v1/loads"))
                .send()
                .await
                .expect("loads")
                .json()
                .await
                .expect("a JSON body");
            if body["loads"][0]["num_running_reqs"].as_u64() == Some(running) {
                return body["loads"][0].clone();
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the running count never read {running}"))
}

#[tokio::test]
async fn http_reports_the_scripted_answers_in_flight_as_running() {
    let dir = tempfile::tempdir().expect("temp dir");
    let prompt = "count me while I run";
    let script = script_file(
        dir.path(),
        "running.jsonl",
        &[json!({
            "input_ids": http::synth_token_ids(prompt),
            "output_ids": [20, 21, 22, 23, 24],
            "output_text": ["a", "b", "c", "d", "e"],
            "chunk_tokens": 1,
            "delay_ms": 40,
        })],
    );
    let (base, _) = start_http(config(Some(script))).await;
    let http = common::http_client();
    let idle = http_running_settles_to(&http, &base, 0).await;
    assert_eq!(idle["num_total_reqs"], 0, "{idle}");

    // The whole answer is paced before its one body: a poll meanwhile reads
    // the one request as running, and as the whole of the load.
    let request = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
    });
    let (response, busy) = tokio::join!(
        async {
            http.post(format!("{base}/v1/chat/completions"))
                .json(&request)
                .send()
                .await
                .expect("chat")
        },
        http_running_settles_to(&http, &base, 1)
    );
    assert_eq!(response.status(), 200);
    assert_eq!(busy["num_total_reqs"], 1, "{busy}");
    assert_eq!(busy["num_waiting_reqs"], 0, "{busy}");
    let whole: Value = response.json().await.expect("a JSON body");
    assert_eq!(
        whole["choices"][0]["message"]["content"], "abcde",
        "{whole}"
    );

    let idle = http_running_settles_to(&http, &base, 0).await;
    assert_eq!(idle["num_total_reqs"], 0, "{idle}");
}

#[tokio::test]
async fn grpc_reports_top_alternatives_from_the_answers_own_tokens() {
    let dir = tempfile::tempdir().expect("temp dir");
    let script = script_file(
        dir.path(),
        "grpc-top.jsonl",
        &[json!({
            "input_ids": [70, 80, 90],
            "output_ids": [10, 11, 12],
            "finish_reason": "stop",
            "logprobs": [-0.5, -1.0, -1.5],
        })],
    );
    // A prompt of this test's own: the store is process-wide.
    let (client, _) = start_grpc(config(Some(script))).await;
    let top = |values: &[f32], token_ids: &[u32]| ts::TopLogProbs {
        values: values.to_vec(),
        token_ids: token_ids.to_vec(),
    };
    let top_of = |responses: &[ts::GenerateResponse]| -> Vec<Vec<ts::TopLogProbs>> {
        responses
            .iter()
            .filter_map(|r| match &r.response {
                Some(GenResp::Complete(c)) => c.output_logprobs.clone(),
                Some(GenResp::Chunk(c)) => c.output_logprobs.clone(),
                None => None,
            })
            .map(|lp| lp.top_logprobs)
            .collect()
    };

    // Two per position: the sampled token first, then the next distinct one of
    // the answer, half a nat less likely; the last position wraps around.
    let mut request = grpc_request("two", &[70, 80, 90], false, true);
    request.top_logprobs_num = 2;
    let whole = generate(&client, request).await;
    assert_eq!(
        top_of(&whole),
        vec![vec![
            top(&[-0.5, -1.0], &[10, 11]),
            top(&[-1.0, -1.5], &[11, 12]),
            top(&[-1.5, -2.0], &[12, 10]),
        ]]
    );

    // More asked than the answer has distinct tokens: as many as it has.
    let mut request = grpc_request("five", &[70, 80, 90], false, true);
    request.top_logprobs_num = 5;
    let whole = generate(&client, request).await;
    assert_eq!(
        top_of(&whole),
        vec![vec![
            top(&[-0.5, -1.0, -1.5], &[10, 11, 12]),
            top(&[-1.0, -1.5, -2.0], &[11, 12, 10]),
            top(&[-1.5, -2.0, -2.5], &[12, 10, 11]),
        ]]
    );

    // Streamed: each chunk carries its own token's alternatives, the
    // completion all of them; none asked, none sent.
    let mut request = grpc_request("stream", &[70, 80, 90], true, true);
    request.top_logprobs_num = 2;
    let streamed = generate(&client, request).await;
    assert_eq!(
        top_of(&streamed),
        vec![
            vec![top(&[-0.5, -1.0], &[10, 11])],
            vec![top(&[-1.0, -1.5], &[11, 12])],
            vec![top(&[-1.5, -2.0], &[12, 10])],
            vec![
                top(&[-0.5, -1.0], &[10, 11]),
                top(&[-1.0, -1.5], &[11, 12]),
                top(&[-1.5, -2.0], &[12, 10]),
            ],
        ]
    );
    let plain = generate(&client, grpc_request("plain", &[70, 80, 90], false, true)).await;
    assert_eq!(top_of(&plain), vec![Vec::<ts::TopLogProbs>::new()]);
}

#[tokio::test]
async fn http_reports_top_alternatives_from_the_answers_own_pieces() {
    let dir = tempfile::tempdir().expect("temp dir");
    let prompt = "top alternatives please";
    let script = script_file(
        dir.path(),
        "http-top.jsonl",
        &[json!({
            "input_ids": http::synth_token_ids(prompt),
            "output_ids": [20, 21, 22],
            "output_text": ["Hello", " world", "!"],
            "finish_reason": "stop",
            "logprobs": [-0.5, -1.0, -1.5],
        })],
    );
    let (base, _worker) = start_http(config(Some(script))).await;
    let http = common::http_client();

    // Chat: `top_logprobs` entries per content item, the piece itself first.
    let chat: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": true,
            "top_logprobs": 2,
        }))
        .send()
        .await
        .expect("completion")
        .json()
        .await
        .expect("body json");
    let entry =
        |token: &str, logprob: f64| json!({"token": token, "logprob": logprob, "bytes": null});
    let content = &chat["choices"][0]["logprobs"]["content"];
    assert_eq!(
        content[0]["top_logprobs"],
        json!([entry("Hello", -0.5), entry(" world", -1.0)])
    );
    assert_eq!(
        content[2]["top_logprobs"],
        json!([entry("!", -1.5), entry("Hello", -2.0)]),
        "the last position wraps around the answer"
    );

    // Completions: `logprobs: n` is the count, as a piece-to-logprob map.
    let completion: Value = http
        .post(format!("{base}/v1/completions"))
        .json(&json!({"model": MODEL, "prompt": prompt, "logprobs": 2}))
        .send()
        .await
        .expect("completion")
        .json()
        .await
        .expect("body json");
    assert_eq!(
        completion["choices"][0]["logprobs"]["top_logprobs"],
        json!([
            {"Hello": -0.5, " world": -1.0},
            {" world": -1.0, "!": -1.5},
            {"!": -1.5, "Hello": -2.0},
        ])
    );

    // Streamed chat: the alternatives ride on each delta's own piece.
    let streamed = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": true,
            "logprobs": true,
            "top_logprobs": 3,
        }))
        .send()
        .await
        .expect("stream");
    let frames = sse_data(&streamed.text().await.expect("body"));
    let second: Value = serde_json::from_str(&frames[1]).expect("a JSON frame");
    assert_eq!(
        second["choices"][0]["logprobs"]["content"][0]["top_logprobs"],
        json!([
            entry(" world", -1.0),
            entry("!", -1.5),
            entry("Hello", -2.0)
        ])
    );
}

#[tokio::test]
async fn zmq_reports_top_alternatives_with_their_ranks() {
    let dir = tempfile::tempdir().expect("temp dir");
    let endpoint = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (
        endpoint("hs.sock"),
        endpoint("in.sock"),
        endpoint("out.sock"),
    );
    let script = script_file(
        dir.path(),
        "zmq-top.jsonl",
        &[json!({
            "input_ids": [44, 55, 66],
            "output_ids": [20, 21],
            "finish_reason": "stop",
            "logprobs": [-0.25, -0.75],
        })],
    );
    // Rank 9 and prompts of this test's own: the store is process-wide and a
    // rank's entry is keyed by its index, so two live ranks of one index would
    // share it and the second one's start would wipe the first's.
    let rank = 9;
    #[expect(
        clippy::disallowed_methods,
        reason = "serve loops until the test drops its handles"
    )]
    let _engine = tokio::spawn(zmq::serve(
        Arc::new(config(Some(script))),
        handshake.clone(),
        rank,
    ));
    let transport = connect_handshake(&handshake, 1, &input, &output, Duration::from_secs(10))
        .await
        .expect("frontend handshake");
    let client = EngineCoreClient::new(transport);
    let mut stream = client
        .submit(EngineCoreRequest {
            request_id: "top".to_string(),
            prompt_token_ids: Some(vec![44, 55, 66]),
            sampling_params: Some(EngineCoreSamplingParams {
                logprobs: Some(2),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("submit");
    let first = stream
        .next()
        .await
        .expect("a first batch")
        .expect("output ok");
    let logprobs = first
        .new_logprobs
        .as_ref()
        .expect("logprobs were asked for");
    // The engine's `n + 1` columns per position: the sampled token's own,
    // then the ranked ones with the sampled token repeated at rank 1.
    assert_eq!(
        logprobs.positions[0].entries,
        vec![
            TokenLogprob {
                token_id: 20,
                logprob: -0.25,
                rank: 1
            },
            TokenLogprob {
                token_id: 20,
                logprob: -0.25,
                rank: 1
            },
            TokenLogprob {
                token_id: 21,
                logprob: -0.75,
                rank: 2
            },
        ],
        "the sampled token's own column, then the two ranked columns"
    );
    // One token per frame: the second position rides the second batch.
    let second = stream
        .next()
        .await
        .expect("a second batch")
        .expect("output ok");
    let logprobs = second
        .new_logprobs
        .as_ref()
        .expect("logprobs ride on every frame");
    assert_eq!(
        logprobs.positions[0].entries,
        vec![
            TokenLogprob {
                token_id: 21,
                logprob: -0.75,
                rank: 1
            },
            TokenLogprob {
                token_id: 21,
                logprob: -0.75,
                rank: 1
            },
            TokenLogprob {
                token_id: 20,
                logprob: -1.25,
                rank: 2
            },
        ],
        "the second token's alternative wraps to the answer's first token"
    );
}

/// `frame` with the choice index an `n > 1` request's later choices carry.
fn at_index(mut frame: ts::GenerateResponse, index: u32) -> ts::GenerateResponse {
    match frame.response.as_mut() {
        Some(GenResp::Chunk(chunk)) => chunk.index = index,
        Some(GenResp::Complete(complete)) => complete.index = index,
        _ => {}
    }
    frame
}

fn complete_at(
    id: &str,
    output_ids: Vec<u32>,
    prompt_tokens: u32,
    index: u32,
) -> ts::GenerateResponse {
    ts::GenerateResponse {
        request_id: id.to_string(),
        response: Some(GenResp::Complete(ts::GenerateComplete {
            completion_tokens: u32::try_from(output_ids.len()).unwrap_or(u32::MAX),
            output_ids,
            finish_reason: "stop".to_string(),
            prompt_tokens,
            cached_tokens: 0,
            output_logprobs: None,
            matched_stop: None,
            index,
            weight_version: None,
            ..Default::default()
        })),
    }
}

fn answers(values: &[Value]) -> Vec<script::Scripted> {
    values
        .iter()
        .map(|value| script::Answer::parse(value).expect("a scripted answer"))
        .collect()
}

#[tokio::test]
async fn grpc_plays_one_scripted_answer_per_choice_in_order() {
    let (client, worker) = start_grpc(config(None)).await;
    // The worker's own queue (the store is shared by the binary's tests): the
    // choices read the enqueued answers in order.
    let sampled = |stream: bool| {
        let mut request = grpc_request("choices", &[51, 52, 53], stream, false);
        request.sampling_params = Some(ts::SamplingParams {
            n: 2,
            ..Default::default()
        });
        request
    };

    script::enqueue(
        &worker,
        answers(&[json!({"output_ids": [30, 31]}), json!({"output_ids": [32]})]),
    )
    .expect("the worker is known");
    let streamed = generate(&client, sampled(true)).await;
    assert_eq!(
        streamed,
        vec![
            chunk("choices", vec![30], 3, 1, 0, None),
            chunk("choices", vec![31], 3, 2, 0, None),
            complete_at("choices", vec![30, 31], 3, 0),
            at_index(chunk("choices", vec![32], 3, 1, 0, None), 1),
            complete_at("choices", vec![32], 3, 1),
        ]
    );

    // Without streaming: the choices' `Complete`s alone, in order.
    script::enqueue(
        &worker,
        answers(&[json!({"output_ids": [40]}), json!({"output_ids": [41, 42]})]),
    )
    .expect("the worker is known");
    let whole = generate(&client, sampled(false)).await;
    assert_eq!(
        whole,
        vec![
            complete_at("choices", vec![40], 3, 0),
            complete_at("choices", vec![41, 42], 3, 1),
        ]
    );
}

#[tokio::test]
async fn grpc_fails_a_request_whose_choices_the_script_does_not_cover() {
    let (client, worker) = start_grpc(config(None)).await;
    script::enqueue(&worker, answers(&[json!({"output_ids": [60]})])).expect("the worker is known");
    let mut request = grpc_request("short", &[61, 62, 63], false, false);
    request.sampling_params = Some(ts::SamplingParams {
        n: 3,
        ..Default::default()
    });

    let status = client
        .generate(request)
        .await
        .err()
        .expect("one answer cannot serve three choices");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert!(
        status
            .message()
            .contains("n = 3 choices need 3 scripted answers for this prompt, 1 queued"),
        "{}",
        status.message()
    );
    // The queue is as it was: the one answer still waits for a request it fits.
    let state = script::status(&worker).expect("the worker is known");
    assert_eq!(state["queued"], 1, "{state}");

    // A request for more choices than the mock holds answers for is refused
    // with nothing taken.
    let mut request = grpc_request("many", &[61, 62, 63], false, false);
    request.sampling_params = Some(ts::SamplingParams {
        n: 129,
        ..Default::default()
    });
    let status = client
        .generate(request)
        .await
        .err()
        .expect("129 choices are above the limit");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert!(
        status
            .message()
            .contains("n = 129 choices is above the mock's limit of 128 per request"),
        "{}",
        status.message()
    );
    let state = script::status(&worker).expect("the worker is known");
    assert_eq!(state["queued"], 1, "{state}");

    // The one answer still serves a request for one choice...
    let one = generate(&client, grpc_request("one", &[61, 62, 63], false, false)).await;
    assert_eq!(one, vec![complete_at("one", vec![60], 3, 0)]);
    // ...and a worker with nothing scripted answers any number of choices in
    // its other mode: the limit binds only a request the script serves.
    let mut request = grpc_request("unscripted", &[61, 62, 63], false, false);
    request.sampling_params = Some(ts::SamplingParams {
        n: 129,
        ..Default::default()
    });
    let canned = generate(&client, request).await;
    assert!(!canned.is_empty(), "the worker's other mode answered");
    let state = script::status(&worker).expect("the worker is known");
    assert_eq!(state["queued"], 0, "{state}");
}

#[tokio::test]
async fn http_fails_a_request_whose_choices_the_script_does_not_cover_without_a_retry() {
    let (base, worker) = start_http(config(None)).await;
    let http = common::http_client();
    let deadline = Instant::now() + Duration::from_secs(5);
    while script::status(&worker).is_none() {
        assert!(
            Instant::now() < deadline,
            "the HTTP worker did not register"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    script::enqueue(&worker, answers(&[json!({"output_text": "one"})]))
        .expect("the worker is known");

    let response = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "pick two of one"}],
            "n": 2,
        }))
        .send()
        .await
        .expect("response");
    // 412, which the gateway does not retry: a retry would find the queue
    // empty and read a canned choice instead of this reason.
    assert_eq!(response.status(), reqwest::StatusCode::PRECONDITION_FAILED);
    let body = response.text().await.expect("the reason");
    assert!(
        body.contains("n = 2 choices need 2 scripted answers for this prompt, 1 queued"),
        "{body}"
    );
    let state = script::status(&worker).expect("the worker is known");
    assert_eq!(state["queued"], 1, "{state}");
}

#[tokio::test]
async fn http_plays_one_scripted_answer_per_choice() {
    let (base, worker) = start_http(config(None)).await;
    let http = common::http_client();
    // The worker registers with the script store as its server task starts.
    let deadline = Instant::now() + Duration::from_secs(5);
    while script::status(&worker).is_none() {
        assert!(
            Instant::now() < deadline,
            "the HTTP worker did not register"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let ask = |stream: bool| {
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "pick two"}],
            "n": 2,
            "stream": stream,
            "stream_options": {"include_usage": true},
        })
    };

    script::enqueue(
        &worker,
        answers(&[
            json!({"output_text": "one"}),
            json!({"output_text": ["two", " more"]}),
        ]),
    )
    .expect("the worker is known");
    let whole: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&ask(false))
        .send()
        .await
        .expect("response")
        .json()
        .await
        .expect("a JSON body");
    assert_eq!(whole["choices"][0]["index"], 0, "{whole}");
    assert_eq!(whole["choices"][0]["message"]["content"], "one", "{whole}");
    assert_eq!(whole["choices"][1]["index"], 1, "{whole}");
    assert_eq!(
        whole["choices"][1]["message"]["content"], "two more",
        "{whole}"
    );
    assert_eq!(
        whole["choices"].as_array().map(Vec::len),
        Some(2),
        "{whole}"
    );
    assert_eq!(whole["usage"]["completion_tokens"], 3, "{whole}");

    script::enqueue(
        &worker,
        answers(&[
            json!({"output_text": "one"}),
            json!({"output_text": ["two", " more"]}),
        ]),
    )
    .expect("the worker is known");
    let body = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&ask(true))
        .send()
        .await
        .expect("response")
        .text()
        .await
        .expect("the stream");
    let frames: Vec<Value> = sse_data(&body)
        .iter()
        .filter(|data| data.as_str() != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("a JSON frame"))
        .collect();
    let choice_frames: Vec<(u64, String, Option<String>)> = frames
        .iter()
        .filter_map(|frame| frame["choices"].get(0))
        .map(|choice| {
            (
                choice["index"].as_u64().expect("an index"),
                choice["delta"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                choice["finish_reason"].as_str().map(str::to_string),
            )
        })
        .collect();
    assert_eq!(
        choice_frames,
        vec![
            (0, "one".to_string(), None),
            (0, String::new(), Some("stop".to_string())),
            (1, "two".to_string(), None),
            (1, " more".to_string(), None),
            (1, String::new(), Some("stop".to_string())),
        ],
        "{body}"
    );
    let usage = frames
        .iter()
        .find(|frame| !frame["usage"].is_null())
        .expect("the usage chunk");
    assert_eq!(usage["usage"]["completion_tokens"], 3, "{body}");
    assert_eq!(usage["choices"], json!([]), "{body}");
}

// ── answers filed under a prompt text ───────────────────────────────────────

#[tokio::test]
async fn grpc_serves_an_answer_filed_under_a_text_to_the_prompt_that_holds_it() {
    // A word-level tokenizer: the prompt's ids decode to the words a test
    // names in `prompt_contains`.
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"say":1,"apricot":2,"bramble":3},"unk_token":"[UNK]"},"pre_tokenizer":{"type":"Whitespace"}}"#,
    )
    .expect("write the tokenizer");
    let cfg = Config {
        tokenizer_path: dir.path().display().to_string(),
        ..config(None)
    };
    let (client, worker) = start_grpc(cfg).await;
    let state = script::enqueue(
        &worker,
        answers(&[
            json!({"prompt_contains": "apricot", "output_ids": [20]}),
            json!({"prompt_contains": "bramble", "output_ids": [30]}),
        ]),
    )
    .expect("a started worker");
    assert_eq!(state["queued"], 0, "{state}");
    assert_eq!(state["marked"], 2, "{state}");

    let output =
        |frames: Vec<ts::GenerateResponse>| match frames.last().and_then(|f| f.response.as_ref()) {
            Some(GenResp::Complete(complete)) => complete.output_ids.clone(),
            other => panic!("a Complete at the end, not {other:?}"),
        };
    // "say bramble" first: the answers match by the prompt's text, not by
    // the order they were filed in.
    assert_eq!(
        output(generate(&client, grpc_request("b", &[1, 3], false, false)).await),
        vec![30]
    );
    assert_eq!(
        output(generate(&client, grpc_request("a", &[1, 2], false, false)).await),
        vec![20]
    );
    // A prompt that holds neither text is a miss, answered canned.
    assert_eq!(
        output(generate(&client, grpc_request("n", &[1, 1], false, false)).await),
        vec![100, 101, 102]
    );
    let state = script::status(&worker).expect("a started worker");
    assert_eq!(
        state["marked"], 0,
        "each filed answer was served once: {state}"
    );
    assert_eq!(
        state["hits"], 2,
        "the two prompts that hold a filed text, not the miss: {state}"
    );
    assert_eq!(state["misses"], 1, "{state}");
}

#[tokio::test]
async fn http_serves_an_answer_filed_under_a_text_by_the_requests_own_text() {
    let (base, worker) = start_http(config(None)).await;
    script::enqueue(
        &worker,
        answers(&[json!({"prompt_contains": "apricot", "output_text": "apricot it is"})]),
    )
    .expect("a started worker");
    let http = common::http_client();
    let ask = |content: &str| {
        http.post(format!("{base}/v1/chat/completions"))
            .json(&json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": content}],
            }))
    };

    // The text wire reads the prompt it was sent: no tokenizer needed.
    let body: Value = ask("Repeat exactly this one word: apricot")
        .send()
        .await
        .expect("chat")
        .json()
        .await
        .expect("json");
    assert_eq!(
        body["choices"][0]["message"]["content"], "apricot it is",
        "{body}"
    );
    let state = script::status(&worker).expect("a started worker");
    assert_eq!(state["marked"], 0, "served once: {state}");
    assert_eq!(state["hits"], 1, "{state}");

    // Served once: the same prompt again reads the canned answer.
    let body: Value = ask("Repeat exactly this one word: apricot")
        .send()
        .await
        .expect("chat")
        .json()
        .await
        .expect("json");
    assert_ne!(
        body["choices"][0]["message"]["content"], "apricot it is",
        "{body}"
    );
    assert_eq!(
        script::status(&worker).expect("a started worker")["misses"],
        1
    );
}

#[tokio::test]
async fn grpc_reads_the_rendered_prompt_beside_the_ids_and_fails_a_prompt_it_cannot_read() {
    // No tokenizer this worker can load: the rendered text the gateway sends
    // beside the ids is what it reads for the answers filed under a text.
    let (client, worker) = start_grpc(config(None)).await;
    script::enqueue(
        &worker,
        answers(&[json!({"prompt_contains": "apricot", "output_ids": [20]})]),
    )
    .expect("a started worker");
    let mut rendered = grpc_request("r", &[1, 2], false, false);
    rendered
        .tokenized
        .as_mut()
        .expect("tokenized input")
        .original_text = "say apricot".to_string();
    let frames = generate(&client, rendered).await;
    match frames.last().and_then(|f| f.response.as_ref()) {
        Some(GenResp::Complete(complete)) => assert_eq!(complete.output_ids, vec![20]),
        other => panic!("a Complete at the end, not {other:?}"),
    }

    // Filed again, and a prompt that carries its ids alone: the request
    // fails with the reason instead of reading canned; the answer stays.
    script::enqueue(
        &worker,
        answers(&[json!({"prompt_contains": "bramble", "output_ids": [30]})]),
    )
    .expect("a started worker");
    let refused = client
        .generate(grpc_request("n", &[1, 3], false, false))
        .await
        .err()
        .expect("a failed request, not a stream");
    assert_eq!(
        refused.code(),
        tonic::Code::FailedPrecondition,
        "{refused:?}"
    );
    assert!(refused.message().contains("cannot be read"), "{refused:?}");
    let state = script::status(&worker).expect("a started worker");
    assert_eq!(state["marked"], 1, "the filed answer stays: {state}");
    assert_eq!(state["hits"], 1, "{state}");
}
