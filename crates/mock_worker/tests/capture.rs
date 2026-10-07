//! `--capture` writes every gRPC `Generate` request a worker receives to a
//! file, one JSON object per line, before the worker sends any frame. These
//! tests serve the real TokenSpeed service on a free port and drive it with
//! the gateway's own client.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures::StreamExt;
use mock_worker::{
    config::{Config, ReplayConfig},
    engine::{EngineParams, TimingModel},
    grpc::serve_with_listener,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto as ts, TokenSpeedSchedulerClient};
use tokio::net::TcpListener;
use ts::generate_response::Response as GenResp;

/// A canned worker that captures to `capture` when it is set.
fn config(capture: Option<PathBuf>) -> Config {
    Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: 0,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: "capture-test".to_string(),
        tokenizer_path: "capture-test".to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 3,
        realistic: false,
        engine: EngineParams::default(),
        replay: ReplayConfig { capture },
        ..Config::default()
    }
}

/// Serve `cfg` on a free port and connect the gateway's client to it.
async fn start(cfg: Config) -> TokenSpeedSchedulerClient {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect to the mock worker")
}

/// A streamed request with every sampling field unset.
fn request(request_id: &str, input_ids: &[u32], text: &str) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: request_id.to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: input_ids.to_vec(),
            original_text: text.to_string(),
        }),
        sampling_params: Some(ts::SamplingParams::default()),
        stream: true,
        ..Default::default()
    }
}

/// Send `req` and read every frame of the response.
async fn generate(
    client: &TokenSpeedSchedulerClient,
    req: ts::GenerateRequest,
) -> Vec<ts::GenerateResponse> {
    let mut stream = client.generate(req).await.expect("generate");
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        frames.push(frame.expect("a response frame"));
    }
    stream.mark_completed();
    frames
}

/// The capture file's lines, each parsed as JSON.
fn captured(path: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).expect("read the capture file");
    assert!(
        text.is_empty() || text.ends_with('\n'),
        "every line ends with a newline: {text:?}"
    );
    text.lines()
        .map(|line| serde_json::from_str(line).expect("each line is one JSON object"))
        .collect()
}

#[tokio::test]
async fn writes_one_line_per_request_in_order() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("generate.jsonl");
    let client = start(config(Some(path.clone()))).await;

    // A newline inside the text must not split the line; ids span the u32 range.
    let sent: [(&str, Vec<u32>, &str, bool); 3] = [
        (
            "first",
            vec![151644, 872, 198],
            "<|im_start|>user\nhi",
            true,
        ),
        ("second", vec![], "", false),
        (
            "third",
            vec![0, 4_294_967_295],
            "a \"quoted\" word, a \\ backslash,\ta tab and 日本語\n",
            true,
        ),
    ];
    for (request_id, input_ids, text, stream) in &sent {
        let mut req = request(request_id, input_ids, text);
        req.stream = *stream;
        generate(&client, req).await;
    }

    let lines = captured(&path);
    assert_eq!(lines.len(), sent.len(), "one line per request: {lines:?}");
    for (line, (request_id, input_ids, text, stream)) in lines.iter().zip(&sent) {
        assert_eq!(line["request_id"], *request_id);
        assert_eq!(line["input_ids"], json!(input_ids), "{request_id}");
        assert_eq!(line["original_text"], *text, "{request_id}");
        assert_eq!(line["stream"], *stream, "{request_id}");
    }
}

/// The line is in the file once the client holds the response stream,
/// before it reads a frame. The realistic engine's slow decode step holds the
/// first frame back by 200 ms, so a line written while streaming would be
/// missing when the file is read.
#[tokio::test]
async fn line_is_on_disk_before_the_first_frame() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("generate.jsonl");
    let mut cfg = config(Some(path.clone()));
    cfg.realistic = true;
    // The linear timing model with a 200 ms decode base holds the first
    // frame back long enough for the check.
    cfg.engine.timing = TimingModel::Linear {
        prefill_tps: 8000.0,
        decode_base_ms: 200.0,
        decode_per_req_ms: 0.35,
    };
    let client = start(cfg).await;

    let mut req = request("early", &[1, 2, 3], "early");
    if let Some(sampling) = req.sampling_params.as_mut() {
        sampling.max_new_tokens = Some(1);
    }
    let mut stream = client.generate(req).await.expect("generate");

    let lines = captured(&path);
    assert_eq!(lines.len(), 1, "the line is written before any frame");
    assert_eq!(lines[0]["request_id"], "early");

    let first = stream.next().await.expect("a first frame");
    assert_eq!(first.expect("a response frame").request_id, "early");
    while let Some(frame) = stream.next().await {
        frame.expect("a response frame");
    }
    stream.mark_completed();
}

/// With `--capture` on, a canned worker sends exactly the frames it sends
/// without it.
#[tokio::test]
async fn canned_output_is_unchanged_with_capture_on() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("generate.jsonl");
    let with_capture = start(config(Some(path.clone()))).await;
    let without_capture = start(config(None)).await;

    for stream in [true, false] {
        let mut req = request("canned", &[1, 2, 3], "canned");
        req.stream = stream;
        let frames = generate(&with_capture, req.clone()).await;
        assert!(
            matches!(
                frames.last().and_then(|frame| frame.response.as_ref()),
                Some(GenResp::Complete(_))
            ),
            "stream={stream}: the response ends with a complete: {frames:?}"
        );
        assert_eq!(
            frames,
            generate(&without_capture, req).await,
            "stream={stream}"
        );
    }
    assert_eq!(captured(&path).len(), 2, "both requests were captured");
}

/// An optional the client leaves unset is `null`; a field it sets comes back
/// as sent, zeros included. Whole lines are compared, so a missing key fails.
#[tokio::test]
async fn unset_optionals_are_null_and_set_values_come_back_as_sent() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("generate.jsonl");
    let client = start(config(Some(path.clone()))).await;

    generate(&client, request("unset", &[1], "unset")).await;

    let mut set = request("set", &[2], "set");
    set.sampling_params = Some(ts::SamplingParams {
        temperature: Some(0.7),
        top_p: Some(0.95),
        top_k: Some(20),
        min_p: Some(0.0),
        frequency_penalty: Some(0.5),
        presence_penalty: Some(-0.25),
        repetition_penalty: Some(1.1),
        max_new_tokens: Some(128),
        min_new_tokens: 4,
        stop: vec!["</s>".to_string(), "\n\nUser:".to_string()],
        stop_token_ids: vec![151643, 151645],
        ignore_eos: true,
        skip_special_tokens: true,
        spaces_between_special_tokens: true,
        n: 2,
        logit_bias: [("151643".to_string(), -100.0), ("42".to_string(), 0.1)].into(),
        constraint: None,
        no_stop_trim: true,
        custom_params: Some(Default::default()),
        sampling_seed: Some(u64::MAX),
    });
    set.return_logprob = true;
    set.logprob_start_len = Some(0);
    set.top_logprobs_num = 5;
    set.token_ids_logprob = vec![7, 8];
    set.mm_inputs = Some(ts::MultimodalInputs::default());
    set.encode_bootstrap_info = Some(ts::EncodeBootstrapInfo::default());
    set.kv_bootstrap_info = Some(ts::KvBootstrapInfo::default());
    set.data_parallel_rank = Some(0);
    generate(&client, set).await;

    let lines = captured(&path);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(
        lines[0],
        json!({
            "request_id": "unset",
            "input_ids": [1],
            "original_text": "unset",
            "stream": true,
            "temperature": null,
            "top_p": null,
            "top_k": null,
            "min_p": null,
            "repetition_penalty": null,
            "frequency_penalty": null,
            "presence_penalty": null,
            "max_new_tokens": null,
            "min_new_tokens": 0,
            "stop": [],
            "stop_token_ids": [],
            "ignore_eos": false,
            "no_stop_trim": false,
            "skip_special_tokens": false,
            "spaces_between_special_tokens": false,
            "n": 0,
            "sampling_seed": null,
            "logit_bias": {},
            "constraint": null,
            "return_logprob": false,
            "logprob_start_len": null,
            "top_logprobs_num": 0,
            "token_ids_logprob": [],
            "has_custom_params": false,
            "has_mm_inputs": false,
            "has_encode_bootstrap_info": false,
            "has_kv_bootstrap_info": false,
            "has_data_parallel_rank": false,
        })
    );
    assert_eq!(
        lines[1],
        json!({
            "request_id": "set",
            "input_ids": [2],
            "original_text": "set",
            "stream": true,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.25,
            "max_new_tokens": 128,
            "min_new_tokens": 4,
            "stop": ["</s>", "\n\nUser:"],
            "stop_token_ids": [151643, 151645],
            "ignore_eos": true,
            "no_stop_trim": true,
            "skip_special_tokens": true,
            "spaces_between_special_tokens": true,
            "n": 2,
            "sampling_seed": u64::MAX,
            "logit_bias": {"151643": -100.0, "42": 0.1},
            "constraint": null,
            "return_logprob": true,
            "logprob_start_len": 0,
            "top_logprobs_num": 5,
            "token_ids_logprob": [7, 8],
            "has_custom_params": true,
            "has_mm_inputs": true,
            "has_encode_bootstrap_info": true,
            "has_kv_bootstrap_info": true,
            "has_data_parallel_rank": true,
        })
    );
}

/// The constraint is captured as its oneof arm and the string as sent.
#[tokio::test]
async fn constraint_is_captured_with_its_kind_and_value() {
    use ts::sampling_params::Constraint;

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("generate.jsonl");
    let client = start(config(Some(path.clone()))).await;

    /// Builds one arm of the constraint oneof from its value.
    type Arm = fn(String) -> Constraint;
    // The spaces in the JSON values show they are kept as sent, not re-encoded.
    let cases: [(&str, Arm, &str); 4] = [
        ("regex", Constraint::Regex, "[a-z]+"),
        (
            "json_schema",
            Constraint::JsonSchema,
            r#"{"type": "array", "minItems": 1}"#,
        ),
        (
            "ebnf_grammar",
            Constraint::EbnfGrammar,
            r#"root ::= "yes" | "no""#,
        ),
        (
            "structural_tag",
            Constraint::StructuralTag,
            r#"{"type": "structural_tag", "format": {"type": "const_string", "value": "<tool_call>"}}"#,
        ),
    ];
    for (kind, constraint, value) in &cases {
        let mut req = request(kind, &[1], kind);
        if let Some(sampling) = req.sampling_params.as_mut() {
            sampling.constraint = Some(constraint(value.to_string()));
        }
        generate(&client, req).await;
    }

    let lines = captured(&path);
    assert_eq!(lines.len(), cases.len(), "{lines:?}");
    for (line, (kind, _, value)) in lines.iter().zip(&cases) {
        assert_eq!(
            line["constraint"],
            json!({"kind": kind, "value": value}),
            "{kind}"
        );
    }
}

/// A capture file that cannot be opened stops the worker instead of letting
/// it serve requests it cannot record.
#[tokio::test]
async fn unopenable_capture_file_stops_the_worker() {
    let dir = tempfile::tempdir().expect("temp dir");
    // The parent directory does not exist, so the file cannot be created.
    let path = dir.path().join("missing").join("generate.jsonl");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let serving = serve_with_listener(Arc::new(config(Some(path))), listener);
    tokio::time::timeout(Duration::from_secs(5), serving)
        .await
        .expect("the worker stops instead of serving");
}

/// The binary checks the capture file before any worker starts: a path that
/// cannot be opened is a config error (exit 2), not a live process that
/// serves nothing.
#[tokio::test]
async fn unopenable_capture_file_fails_at_startup() {
    let dir = tempfile::tempdir().expect("temp dir");
    // The parent directory does not exist, so the file cannot be created.
    let path = dir.path().join("missing").join("generate.jsonl");
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let port_arg = port.to_string();

    let mut mock_worker = tokio::process::Command::new(env!("CARGO_BIN_EXE_mock-worker"));
    mock_worker
        .args(["--grpc-count", "1", "--grpc-base-port", port_arg.as_str()])
        .arg("--capture")
        .arg(&path)
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(10), mock_worker.output())
        .await
        .expect("mock-worker exits instead of running on")
        .expect("run mock-worker");

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!("cannot open capture file {}", path.display())),
        "stderr names the path: {stderr}"
    );
    assert!(
        std::net::TcpStream::connect(("127.0.0.1", port)).is_err(),
        "nothing is left listening on port {port}"
    );
}

/// The same request gives the same bytes. Decoding puts `logit_bias` in a
/// `HashMap`, whose order changes from one decode to the next, so its keys
/// must be written in a fixed order.
#[tokio::test]
async fn the_same_request_gives_byte_identical_lines() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("generate.jsonl");
    let client = start(config(Some(path.clone()))).await;

    let tokens = [
        0, 1, 5, 7, 9, 13, 31, 42, 64, 77, 100, 256, 2000, 128000, 151643, 151645,
    ];
    let mut req = request("biased", &[1], "biased");
    if let Some(sampling) = req.sampling_params.as_mut() {
        sampling.logit_bias = tokens
            .iter()
            .map(|token| (token.to_string(), -1.5))
            .collect();
    }
    generate(&client, req.clone()).await;
    generate(&client, req).await;

    let text = std::fs::read_to_string(&path).expect("read the capture file");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    assert_eq!(lines[0], lines[1], "the same request gives the same bytes");
    let line: Value = serde_json::from_str(lines[0]).expect("one JSON object");
    let keys: Vec<&String> = line["logit_bias"]
        .as_object()
        .expect("logit_bias is an object")
        .keys()
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "logit_bias keys are sorted");
}
