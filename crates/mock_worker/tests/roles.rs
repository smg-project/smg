//! The workers' roles in a disaggregated fleet (`--worker-type`), on the gRPC
//! and the HTTP wire: what each side answers, what it advertises, and the
//! port a prefill worker binds.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use mock_worker::{
    admin,
    config::{Config, Role, SchedulerWire},
    drain,
    grpc::serve_with_listener,
    http, script,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto as ts, vllm_proto as vllm, TokenSpeedSchedulerClient};
use tokio::{
    io::AsyncReadExt,
    net::{TcpListener, TcpStream},
};
use tonic::transport::Channel;
use ts::generate_response::Response as GenResp;

mod common;
use common::{free_port, generate};

/// Serve one canned worker of `role` on a free port, as worker 0 of its
/// process (its gRPC port is the base port, so its bootstrap port is the
/// base bootstrap port): the gateway's client and the worker's name.
async fn start(
    role: Role,
    bootstrap_base_port: Option<u16>,
    pairing_protocol: Option<&str>,
) -> (TokenSpeedSchedulerClient, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    let cfg = Config {
        model_id: "roles-test".to_string(),
        grpc_base_port: port,
        grpc_count: 1,
        output_tokens: 3,
        role: Some(role),
        bootstrap_base_port,
        pairing_protocol: pairing_protocol.map(str::to_string),
        ..Config::default()
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    let client = TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect to the mock worker");
    (client, format!("grpc:{port}"))
}

fn request(id: &str, input_ids: &[u32], stream: bool) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: input_ids.to_vec(),
            original_text: String::new(),
        }),
        sampling_params: Some(ts::SamplingParams::default()),
        stream,
        ..Default::default()
    }
}

/// A server-arg value as a string: what the gateway flattens into a label.
fn server_arg(info: &ts::GetServerInfoResponse, key: &str) -> Option<String> {
    let value = info.server_args.as_ref()?.fields.get(key)?;
    match value.kind.as_ref()? {
        prost_types::value::Kind::StringValue(s) => Some(s.clone()),
        prost_types::value::Kind::NumberValue(n) => Some(n.to_string()),
        other => Some(format!("{other:?}")),
    }
}

#[tokio::test]
async fn a_prefill_worker_answers_the_prompt_side_alone_and_leaves_the_script_to_decode() {
    let (client, name) = start(Role::Prefill, None, None).await;
    // The answer a test states is the decode side's: a prefill worker given
    // it keeps it where it is.
    let scripted = script::Answer::parse(&json!({"output_ids": [7, 8, 9]})).expect("an answer");
    script::enqueue(&name, vec![scripted]).expect("a known worker");

    for stream in [true, false] {
        let frames = generate(&client, request("pd-1", &[1, 2, 3, 4, 5], stream)).await;
        assert_eq!(
            frames.len(),
            1,
            "the prompt side's completion alone: {frames:?}"
        );
        let Some(GenResp::Complete(complete)) = &frames[0].response else {
            panic!("a Complete, not {:?}", frames[0]);
        };
        assert_eq!(frames[0].request_id, "pd-1");
        assert!(complete.output_ids.is_empty(), "no output: {complete:?}");
        assert_eq!(
            (
                complete.prompt_tokens,
                complete.completion_tokens,
                complete.cached_tokens
            ),
            (5, 0, 0)
        );
        assert_eq!(complete.finish_reason, "stop");
    }
    let state = script::status(&name).expect("a known worker");
    assert_eq!(
        (
            state["queued"].as_u64(),
            state["hits"].as_u64(),
            state["misses"].as_u64()
        ),
        (Some(1), Some(0), Some(0)),
        "the scripted answer waits for the decode side, and no miss is counted: {state}"
    );
}

#[tokio::test]
async fn a_prefill_worker_binds_its_bootstrap_port_and_advertises_its_role() {
    let bootstrap_port = free_port();
    let (client, _) = start(Role::Prefill, Some(bootstrap_port), None).await;

    // The gateway's probe: a connection the port accepts, then closes.
    let mut probe = TcpStream::connect(("127.0.0.1", bootstrap_port))
        .await
        .expect("the bootstrap port accepts a connection");
    let mut buf = [0u8; 1];
    assert_eq!(
        probe.read(&mut buf).await.expect("a clean close"),
        0,
        "the port says nothing and closes"
    );

    let info = client.get_server_info().await.expect("server info");
    assert_eq!(
        server_arg(&info, "disaggregation_mode").as_deref(),
        Some("prefill")
    );
    assert_eq!(
        server_arg(&info, "disaggregation_bootstrap_port"),
        Some(bootstrap_port.to_string())
    );
    assert_eq!(
        server_arg(&info, "pairing_protocol"),
        None,
        "no protocol named, none advertised"
    );
}

#[tokio::test]
async fn a_decode_worker_generates_as_a_regular_worker_and_advertises_its_role() {
    let (client, _) = start(Role::Decode, None, Some("kv-v2")).await;

    let frames = generate(&client, request("pd-2", &[1, 2, 3], true)).await;
    let chunks = frames
        .iter()
        .filter(|frame| matches!(frame.response, Some(GenResp::Chunk(_))))
        .count();
    let Some(GenResp::Complete(complete)) = &frames.last().expect("frames").response else {
        panic!("a Complete at the end, not {:?}", frames.last());
    };
    assert_eq!(chunks, 3, "the canned output, token by token: {frames:?}");
    assert_eq!(complete.output_ids, vec![100, 101, 102]);

    let info = client.get_server_info().await.expect("server info");
    assert_eq!(
        server_arg(&info, "disaggregation_mode").as_deref(),
        Some("decode")
    );
    assert_eq!(
        server_arg(&info, "disaggregation_bootstrap_port"),
        None,
        "a decode worker binds and advertises no bootstrap port"
    );
    assert_eq!(
        server_arg(&info, "pairing_protocol").as_deref(),
        Some("kv-v2"),
        "the pairing protocol the environment names, as the servicers report it"
    );
}

/// A realistic prefill worker on a free port and the admin API beside it:
/// the gateway's client and the worker's fault route.
async fn start_realistic_prefill() -> (TokenSpeedSchedulerClient, reqwest::Client, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    let cfg = Config {
        model_id: "roles-test".to_string(),
        grpc_base_port: port,
        grpc_count: 1,
        realistic: true,
        role: Some(Role::Prefill),
        ..Config::default()
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    let client = TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect to the mock worker");
    let (http, fault) = fault_route(port).await;
    (client, http, fault)
}

/// The admin API beside the worker on gRPC `port`: an HTTP client and the
/// worker's fault route, up.
async fn fault_route(port: u16) -> (reqwest::Client, String) {
    let admin_port = free_port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the admin server serves until the test process exits"
    )]
    let _admin = tokio::spawn(admin::serve(
        Arc::new(Config::default()),
        "127.0.0.1".to_string(),
        admin_port,
    ));
    let http = common::http_client();
    let fault = format!("http://127.0.0.1:{admin_port}/admin/fault/grpc:{port}");
    let deadline = Instant::now() + Duration::from_secs(10);
    while http.get(&fault).send().await.is_err() {
        assert!(Instant::now() < deadline, "the admin API did not come up");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (http, fault)
}

/// A realistic encode worker on the vLLM wire (the wire with an encode role)
/// and the admin API beside it: the gateway's client and the fault route.
async fn start_realistic_encoder() -> (
    vllm::vllm_engine_client::VllmEngineClient<Channel>,
    reqwest::Client,
    String,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    let cfg = Config {
        model_id: "roles-test".to_string(),
        grpc_base_port: port,
        grpc_count: 1,
        grpc_wire: SchedulerWire::Vllm,
        realistic: true,
        role: Some(Role::Encode),
        ..Config::default()
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    let channel = Channel::from_shared(format!("http://127.0.0.1:{port}"))
        .expect("endpoint")
        .connect()
        .await
        .expect("connect to the mock worker");
    let (http, fault) = fault_route(port).await;
    (
        vllm::vllm_engine_client::VllmEngineClient::new(channel),
        http,
        fault,
    )
}

/// What the gateway sends an encoder: a tokenized prompt with one media
/// item under a placeholder and the pull connector's empty parameters.
fn encode_request(id: &str) -> vllm::GenerateRequest {
    vllm::GenerateRequest {
        request_id: id.to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            },
        )),
        sampling_params: Some(vllm::SamplingParams {
            max_tokens: Some(1),
            ..Default::default()
        }),
        mm_inputs: Some(vllm::MultimodalInputs {
            mm_hashes: vec!["roles-image".to_string()],
            mm_placeholders: vec![vllm::PlaceholderRange {
                offset: 1,
                length: 1,
            }],
            ..Default::default()
        }),
        ec_transfer_params_json: Some("{}".to_string()),
        ..Default::default()
    }
}

/// The encoder's one frame: its completion.
async fn encoder_completion(
    client: &mut vllm::vllm_engine_client::VllmEngineClient<Channel>,
    id: &str,
) -> vllm::GenerateComplete {
    let mut stream = client
        .generate(encode_request(id))
        .await
        .expect("the encoder's stream")
        .into_inner();
    let frame = stream
        .message()
        .await
        .expect("a frame")
        .expect("the encoder's completion");
    let Some(vllm::generate_response::Response::Complete(complete)) = frame.response else {
        panic!("a Complete, not {frame:?}");
    };
    complete
}

/// The worker's fault counters (`failed_total`, `cut_total`).
async fn fault_counts(http: &reqwest::Client, fault: &str) -> (u64, u64) {
    let state: serde_json::Value = http
        .get(fault)
        .send()
        .await
        .expect("the fault state")
        .json()
        .await
        .expect("json");
    let worker = &state["workers"][0];
    (
        worker["failed_total"].as_u64().expect("failed_total"),
        worker["cut_total"].as_u64().expect("cut_total"),
    )
}

#[tokio::test]
async fn a_realistic_prefill_worker_honors_an_armed_fault_before_it_answers() {
    let (client, http, fault) = start_realistic_prefill().await;

    // One 503 armed: the prompt side answers it before admission, as any
    // request of the realistic engine would, and the failure is counted.
    let armed = http
        .post(format!("{fault}/fail?status=503&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
    let refused = client
        .generate(request("pd-f1", &[1, 2, 3], false))
        .await
        .err()
        .expect("the armed status, not a stream");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (1, 0));

    // Spent: the next request is the prompt side's completion as before.
    let frames = generate(&client, request("pd-f2", &[1, 2, 3], false)).await;
    let Some(GenResp::Complete(complete)) = &frames[0].response else {
        panic!("a Complete, not {:?}", frames[0]);
    };
    assert_eq!((complete.prompt_tokens, complete.completion_tokens), (3, 0));
    assert_eq!(complete.finish_reason, "stop");
}

#[tokio::test]
async fn a_realistic_prefill_worker_ends_the_prompt_side_with_error_on_an_armed_cut() {
    let (client, http, fault) = start_realistic_prefill().await;

    // `finish=error` wins over the status, as on any worker: the prompt side
    // ends with `error` and the cut is counted, not a failure.
    let armed = http
        .post(format!("{fault}/fail?finish=error&status=503&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
    let frames = generate(&client, request("pd-e1", &[1, 2, 3], false)).await;
    let Some(GenResp::Complete(complete)) = &frames[0].response else {
        panic!("a Complete, not {:?}", frames[0]);
    };
    assert_eq!(complete.finish_reason, "error");
    assert!(complete.output_ids.is_empty(), "no output: {complete:?}");
    assert_eq!(fault_counts(&http, &fault).await, (0, 1));

    // A status cut: with no output to cut after it fires at once, as a cut.
    let armed = http
        .post(format!("{fault}/fail?status=502&after_tokens=4&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
    let refused = client
        .generate(request("pd-e2", &[1, 2, 3], false))
        .await
        .err()
        .expect("the cut's status, not a stream");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (0, 2));
}

#[tokio::test]
async fn a_realistic_encode_worker_takes_the_fault_hooks_and_counts_in_flight() {
    let (mut client, http, fault) = start_realistic_encoder().await;

    // One 503 armed: the encoder refuses before it answers, as the prompt
    // side of a prefill worker does, and the failure is counted.
    let armed = http
        .post(format!("{fault}/fail?status=503&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
    let refused = client
        .generate(encode_request("ec-f1"))
        .await
        .expect_err("the armed status, not a stream");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (1, 0));

    // An armed cut: the encoder's completion ends with `error` and the cut
    // is counted; the handles still come, the gateway reads the reason.
    let armed = http
        .post(format!("{fault}/fail?finish=error&status=503&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
    let complete = encoder_completion(&mut client, "ec-e1").await;
    assert_eq!(complete.finish_reason, "error");
    assert_eq!(fault_counts(&http, &fault).await, (1, 1));

    // A stall: the request counts in flight while it waits, and no longer
    // once its completion is out. Sampled over the stall, so a request of
    // another worker in this process passing by cannot stand in for it.
    let armed = http
        .post(format!("{fault}/fail?status=0&stall_ms=1500&count=1"))
        .send()
        .await
        .expect("arm the stall");
    assert_eq!(armed.status(), 200);
    let (complete, during) = tokio::join!(encoder_completion(&mut client, "ec-s1"), async {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let mut during = usize::MAX;
        for _ in 0..5 {
            let now = drain::in_flight();
            assert!(now >= 1, "the stalled request is not counted in flight");
            during = during.min(now);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        during
    });
    assert_eq!(complete.finish_reason, "stop");
    let deadline = Instant::now() + Duration::from_secs(5);
    while drain::in_flight() >= during {
        assert!(
            Instant::now() < deadline,
            "the completion is out but the request still counts in flight"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_realistic_encode_worker_checks_the_request_before_it_spends_a_fault() {
    let (mut client, http, fault) = start_realistic_encoder().await;
    let armed = http
        .post(format!("{fault}/fail?status=503&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);

    // A request without transfer parameters went to the wrong pool: refused
    // as such, the armed fault and its counters untouched.
    let mut wrong_pool = encode_request("ec-w1");
    wrong_pool.ec_transfer_params_json = None;
    let refused = client
        .generate(wrong_pool)
        .await
        .expect_err("the wrong pool, not a stream");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (0, 0));

    // The fault still waits for the next encode request.
    let refused = client
        .generate(encode_request("ec-w2"))
        .await
        .expect_err("the armed status, not a stream");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (1, 0));
}

/// A canned encode worker on the vLLM wire that holds each request for
/// `gen_delay` before its completion: the gateway's client.
async fn start_canned_encoder(
    gen_delay: Duration,
) -> vllm::vllm_engine_client::VllmEngineClient<Channel> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    let cfg = Config {
        model_id: "roles-test".to_string(),
        grpc_base_port: port,
        grpc_count: 1,
        grpc_wire: SchedulerWire::Vllm,
        role: Some(Role::Encode),
        gen_delay,
        ..Config::default()
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    let channel = Channel::from_shared(format!("http://127.0.0.1:{port}"))
        .expect("endpoint")
        .connect()
        .await
        .expect("connect to the mock worker");
    vllm::vllm_engine_client::VllmEngineClient::new(channel)
}

/// The worker's own running count, as its load report tells the gateway.
async fn running_reqs(client: &mut vllm::vllm_engine_client::VllmEngineClient<Channel>) -> i32 {
    client
        .get_loads(vllm::GetLoadsRequest {
            dp_rank: None,
            ..Default::default()
        })
        .await
        .expect("the load report")
        .into_inner()
        .loads
        .remove(0)
        .num_running_reqs
}

// ── The HTTP wire's roles ───────────────────────────────────────────────────

/// Serve one HTTP worker of `cfg`'s role on a free port, as worker 0 of its
/// process (its port is the base port, so its bootstrap port is the base
/// bootstrap port): the tests' client, the worker's base URL and its name.
async fn start_http(cfg: Config) -> (reqwest::Client, String, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    let cfg = Config {
        model_id: "roles-test".to_string(),
        http_base_port: port,
        http_count: 1,
        output_tokens: 3,
        ..cfg
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(http::serve_with_listener(Arc::new(cfg), listener));
    let client = common::http_client();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(10);
    while client.get(format!("{base}/health")).send().await.is_err() {
        assert!(Instant::now() < deadline, "the HTTP worker did not come up");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (client, base, format!("http:{port}"))
}

/// A realistic HTTP prefill worker on a free port and the admin API beside
/// it: the tests' client, the worker's base URL and its fault route.
async fn start_realistic_http_prefill() -> (reqwest::Client, String, String) {
    let (client, base, name) = start_http(Config {
        realistic: true,
        role: Some(Role::Prefill),
        ..Config::default()
    })
    .await;
    let admin_port = free_port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the admin server serves until the test process exits"
    )]
    let _admin = tokio::spawn(admin::serve(
        Arc::new(Config::default()),
        "127.0.0.1".to_string(),
        admin_port,
    ));
    let fault = format!("http://127.0.0.1:{admin_port}/admin/fault/{name}");
    let deadline = Instant::now() + Duration::from_secs(10);
    while client.get(&fault).send().await.is_err() {
        assert!(Instant::now() < deadline, "the admin API did not come up");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (client, base, fault)
}

/// A chat request as the gateway's disaggregated dispatch sends it to both
/// sides: the prompt, and the rendezvous it injected (`room`, and the
/// prefill side's `port`).
fn disaggregated_chat(room: Option<u64>, port: Option<u16>, stream: bool) -> Value {
    let mut body = json!({
        "model": "roles-test",
        "messages": [{"role": "user", "content": "one two three"}],
        "max_tokens": 8,
        "stream": stream,
    });
    if let Some(room) = room {
        body["bootstrap_host"] = json!("127.0.0.1");
        body["bootstrap_room"] = json!(room);
    }
    if let Some(port) = port {
        body["bootstrap_port"] = json!(port);
    }
    body
}

async fn chat(client: &reqwest::Client, base: &str, body: &Value) -> reqwest::Response {
    client
        .post(format!("{base}/v1/chat/completions"))
        .json(body)
        .send()
        .await
        .expect("the worker answers")
}

async fn server_info(client: &reqwest::Client, base: &str, route: &str) -> Value {
    client
        .get(format!("{base}/{route}"))
        .send()
        .await
        .expect("server info")
        .json()
        .await
        .expect("json")
}

/// The worker's own running count, as its load report tells the gateway.
async fn http_running_reqs(client: &reqwest::Client, base: &str) -> u64 {
    let loads: Value = client
        .get(format!("{base}/v1/loads"))
        .send()
        .await
        .expect("the load report")
        .json()
        .await
        .expect("json");
    loads["loads"][0]["num_running_reqs"]
        .as_u64()
        .expect("num_running_reqs")
}

#[tokio::test]
async fn an_http_prefill_worker_binds_its_bootstrap_port_and_advertises_its_role() {
    let bootstrap_port = free_port();
    let (client, base, name) = start_http(Config {
        role: Some(Role::Prefill),
        bootstrap_base_port: Some(bootstrap_port),
        ..Config::default()
    })
    .await;

    // The gateway's probe: a connection the port accepts, then closes.
    let mut probe = TcpStream::connect(("127.0.0.1", bootstrap_port))
        .await
        .expect("the bootstrap port accepts a connection");
    let mut buf = [0u8; 1];
    assert_eq!(
        probe.read(&mut buf).await.expect("a clean close"),
        0,
        "the port says nothing and closes"
    );

    let info = server_info(&client, &base, "server_info").await;
    assert_eq!(info["disaggregation_mode"], "prefill");
    assert_eq!(info["disaggregation_bootstrap_port"], bootstrap_port);
    assert_eq!(
        server_info(&client, &base, "get_server_info").await,
        info,
        "the older spelling answers the same"
    );

    // The answer a test states is the decode side's: a prefill worker given
    // it keeps it where it is, and answers the prompt side's completion.
    let scripted = script::Answer::parse(&json!({"output_text": "apricot"})).expect("an answer");
    script::enqueue(&name, vec![scripted]).expect("a known worker");
    for stream in [false, true] {
        let response = chat(
            &client,
            &base,
            &disaggregated_chat(Some(7), Some(bootstrap_port), stream),
        )
        .await;
        assert_eq!(response.status(), 200);
        let text = response.text().await.expect("the body");
        if stream {
            assert!(
                text.contains(r#""finish_reason":"stop""#) && text.contains("[DONE]"),
                "the finish frame and the end: {text}"
            );
            assert!(
                !text.contains(r#""content":"x""#),
                "no output on the prompt side: {text}"
            );
        } else {
            let body: Value = serde_json::from_str(&text).expect("json");
            assert_eq!(body["choices"][0]["message"]["content"], "");
            assert_eq!(body["choices"][0]["finish_reason"], "stop");
            assert_eq!(
                (
                    body["usage"]["prompt_tokens"].as_u64(),
                    body["usage"]["completion_tokens"].as_u64()
                ),
                (Some(3), Some(0)),
                "the prompt's words, no output: {body}"
            );
        }
    }
    let state = script::status(&name).expect("a known worker");
    assert_eq!(
        (
            state["queued"].as_u64(),
            state["hits"].as_u64(),
            state["misses"].as_u64()
        ),
        (Some(1), Some(0), Some(0)),
        "the scripted answer waits for the decode side, and no miss is counted: {state}"
    );

    // A request without the rendezvous went to the wrong pool; one naming
    // another port would send the decode partner elsewhere.
    let refused = chat(&client, &base, &disaggregated_chat(None, None, false)).await;
    assert_eq!(refused.status(), 400);
    let refused = chat(
        &client,
        &base,
        &disaggregated_chat(Some(7), Some(bootstrap_port.wrapping_add(1)), false),
    )
    .await;
    assert_eq!(refused.status(), 400);
}

#[tokio::test]
async fn an_http_decode_worker_generates_as_a_regular_worker_and_advertises_its_role() {
    let (client, base, _) = start_http(Config {
        role: Some(Role::Decode),
        ..Config::default()
    })
    .await;

    let info = server_info(&client, &base, "server_info").await;
    assert_eq!(info["disaggregation_mode"], "decode");
    assert!(
        info.get("disaggregation_bootstrap_port").is_none(),
        "a decode worker binds and advertises no bootstrap port: {info}"
    );

    let response = chat(
        &client,
        &base,
        &disaggregated_chat(Some(7), Some(8998), true),
    )
    .await;
    assert_eq!(response.status(), 200);
    let text = response.text().await.expect("the body");
    assert_eq!(
        text.matches(r#""content":"x""#).count(),
        3,
        "the canned output, token by token: {text}"
    );
    assert!(
        text.contains(r#""finish_reason":"stop""#) && text.contains("[DONE]"),
        "{text}"
    );
    let refused = chat(&client, &base, &disaggregated_chat(None, None, false)).await;
    assert_eq!(
        refused.status(),
        400,
        "the decode side needs the rendezvous as the prefill side does"
    );
}

#[tokio::test]
async fn a_realistic_http_prefill_worker_honors_an_armed_fault_before_it_answers() {
    let (client, base, fault) = start_realistic_http_prefill().await;
    let armed = client
        .post(format!("{fault}/fail?status=503&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);

    // A request without the rendezvous is refused before the hooks: the
    // armed fault and its counters wait for a request this side serves.
    let refused = chat(&client, &base, &disaggregated_chat(None, None, false)).await;
    assert_eq!(refused.status(), 400);
    assert_eq!(fault_counts(&client, &fault).await, (0, 0));

    // The prompt side answers the armed status before admission, as any
    // request of the realistic engine would, and the failure is counted.
    let refused = chat(&client, &base, &disaggregated_chat(Some(1), None, false)).await;
    assert_eq!(refused.status(), 503);
    assert_eq!(fault_counts(&client, &fault).await, (1, 0));

    // Spent: the next request is the prompt side's completion as before.
    let body: Value = chat(&client, &base, &disaggregated_chat(Some(2), None, false))
        .await
        .json()
        .await
        .expect("json");
    assert_eq!(
        (
            body["usage"]["prompt_tokens"].as_u64(),
            body["usage"]["completion_tokens"].as_u64()
        ),
        (Some(3), Some(0))
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn a_realistic_http_prefill_worker_ends_the_prompt_side_with_error_on_an_armed_cut() {
    let (client, base, fault) = start_realistic_http_prefill().await;

    // `finish=error` wins over the status, as on any worker: the prompt side
    // ends with `error` and the cut is counted, not a failure.
    let armed = client
        .post(format!("{fault}/fail?finish=error&status=503&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
    let response = chat(&client, &base, &disaggregated_chat(Some(1), None, false)).await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("json");
    assert_eq!(body["choices"][0]["finish_reason"], "error");
    assert_eq!(body["choices"][0]["message"]["content"], "");
    assert_eq!(fault_counts(&client, &fault).await, (0, 1));

    // A status cut: with no output to cut after it fires at once, as a cut.
    let armed = client
        .post(format!("{fault}/fail?status=502&after_tokens=4&count=1"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
    let refused = chat(&client, &base, &disaggregated_chat(Some(2), None, false)).await;
    assert_eq!(refused.status(), 502);
    assert_eq!(fault_counts(&client, &fault).await, (0, 2));
}

#[tokio::test]
async fn a_canned_http_prefill_worker_reports_its_held_request_as_running() {
    let (client, base, _) = start_http(Config {
        role: Some(Role::Prefill),
        gen_delay: Duration::from_millis(1500),
        ..Config::default()
    })
    .await;
    assert_eq!(http_running_reqs(&client, &base).await, 0);

    // Held for the canned delay, the request is the worker's one running
    // request in its own load report, and the report reads idle again once
    // the completion is out.
    let held = disaggregated_chat(Some(1), None, false);
    let (response, during) = tokio::join!(chat(&client, &base, &held), async {
        // Sampled once the request is held, not after a fixed wait: a
        // slow box must not let a sample read the worker before then.
        tokio::time::timeout(Duration::from_secs(5), async {
            while http_running_reqs(&client, &base).await == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the held request never showed in the worker's load report");
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(http_running_reqs(&client, &base).await);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        seen
    });
    assert_eq!(response.status(), 200);
    assert_eq!(
        during, [1; 5],
        "the held request is not the worker's one running request"
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while http_running_reqs(&client, &base).await != 0 {
        assert!(
            Instant::now() < deadline,
            "the completion is out but the worker still reports it running"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_canned_encode_worker_reports_its_held_request_as_running() {
    let mut client = start_canned_encoder(Duration::from_millis(1500)).await;
    let mut poller = client.clone();
    assert_eq!(running_reqs(&mut poller).await, 0);

    // Held for the canned delay, the request is the worker's one running
    // request in its own load report (whatever the process's other workers
    // carry), and the report reads idle again once the completion is out.
    let (complete, during) = tokio::join!(encoder_completion(&mut client, "ec-l1"), async {
        // Sampled once the request is admitted, not after a fixed wait: a
        // slow box must not let a sample read the worker before it holds it.
        tokio::time::timeout(Duration::from_secs(5), async {
            while running_reqs(&mut poller).await == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the held request never showed in the worker's load report");
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(running_reqs(&mut poller).await);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        seen
    });
    assert_eq!(complete.finish_reason, "stop");
    assert_eq!(
        during, [1; 5],
        "the held request is not the worker's one running request"
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while running_reqs(&mut poller).await != 0 {
        assert!(
            Instant::now() < deadline,
            "the completion is out but the worker still reports it running"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
