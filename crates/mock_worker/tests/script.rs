//! Scripted output on the three wires: what `--script` and
//! `POST /admin/script/{worker}` make a worker answer, frame by frame, and
//! what a request no script covers still gets.

use std::{
    net::TcpListener as StdTcpListener,
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
    admin,
    config::{Config, ReplayConfig},
    grpc, http, script, zmq,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto as ts, TokenSpeedSchedulerClient};
use tokio::net::TcpListener;
use ts::generate_response::Response as GenResp;

mod common;
use common::generate;

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

/// A port nothing listens on right now (bound and released).
fn free_port() -> u16 {
    StdTcpListener::bind("127.0.0.1:0")
        .expect("bind a free port")
        .local_addr()
        .expect("listener address")
        .port()
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
    let admin_port = free_port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the admin server serves until the test process exits"
    )]
    let _admin = tokio::spawn(admin::serve(
        Arc::new(config(None)),
        "127.0.0.1".to_string(),
        admin_port,
    ));
    let http = common::http_client();
    let url = |worker: &str| format!("http://127.0.0.1:{admin_port}/admin/script/{worker}");
    let deadline = Instant::now() + Duration::from_secs(10);
    while http.get(url(&worker)).send().await.is_err() {
        assert!(Instant::now() < deadline, "the admin API did not come up");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

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
    (format!("http://127.0.0.1:{port}"), format!("http:{port}"))
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
