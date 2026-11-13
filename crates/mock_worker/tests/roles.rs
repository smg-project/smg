//! The workers' roles in a disaggregated fleet (`--worker-type`), on the gRPC
//! and the HTTP wire: what each side answers, what it advertises, and the
//! port a prefill worker binds.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use futures::StreamExt as _;
use mock_worker::{
    config::{Config, Role, SchedulerWire},
    drain,
    grpc::serve_with_listener,
    http, script,
};
use serde_json::{json, Value};
use smg_grpc_client::{
    sglang_proto as sg, tokenspeed_encoder_proto as enc, tokenspeed_proto as ts,
    vllm_proto as vllm, SglangSchedulerClient, TokenSpeedSchedulerClient,
};
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

/// The request with the rendezvous the gateway mints for a disaggregated
/// pair on this wire: the prefill side's bootstrap server and a room.
fn rendezvous(req: ts::GenerateRequest, port: i32) -> ts::GenerateRequest {
    rendezvous_room(req, port, 7)
}

/// The request with a rendezvous naming `room` at the bootstrap `port`.
fn rendezvous_room(mut req: ts::GenerateRequest, port: i32, room: i64) -> ts::GenerateRequest {
    req.kv_bootstrap_info = Some(ts::KvBootstrapInfo {
        bootstrap_host: "127.0.0.1".to_string(),
        bootstrap_port: port,
        bootstrap_room: room,
    });
    req
}

/// Serve one canned worker of `role` on the SGLang wire, on a free port:
/// the gateway's client for that wire.
async fn start_sglang(role: Role) -> SglangSchedulerClient {
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
        grpc_wire: SchedulerWire::Sglang,
        ..Config::default()
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    SglangSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect to the mock worker")
}

/// The SGLang wire's request with a rendezvous naming `room` at the
/// bootstrap port the gateway would name.
fn sglang_request(id: &str, room: i32) -> sg::GenerateRequest {
    sg::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(sg::TokenizedInput {
            input_ids: vec![1, 2, 3],
            original_text: String::new(),
        }),
        sampling_params: Some(sg::SamplingParams::default()),
        disaggregated_params: Some(sg::DisaggregatedParams {
            bootstrap_host: "127.0.0.1".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: room,
        }),
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
        let frames = generate(
            &client,
            rendezvous(request("pd-1", &[1, 2, 3, 4, 5], stream), 8998),
        )
        .await;
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

    let frames = generate(&client, rendezvous(request("pd-2", &[1, 2, 3], true), 8998)).await;
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
/// worker's fault route, on a port the server held before it was read.
async fn fault_route(port: u16) -> (reqwest::Client, String) {
    let admin_port = common::start_admin(Config::default()).await;
    let http = common::http_client();
    let fault = format!("http://127.0.0.1:{admin_port}/admin/fault/grpc:{port}");
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
        .generate(rendezvous(request("pd-f1", &[1, 2, 3], false), 8998))
        .await
        .err()
        .expect("the armed status, not a stream");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (1, 0));

    // Spent: the next request is the prompt side's completion as before.
    let frames = generate(
        &client,
        rendezvous(request("pd-f2", &[1, 2, 3], false), 8998),
    )
    .await;
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
    let frames = generate(
        &client,
        rendezvous(request("pd-e1", &[1, 2, 3], false), 8998),
    )
    .await;
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
        .generate(rendezvous(request("pd-e2", &[1, 2, 3], false), 8998))
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
    let admin_port = common::start_admin(Config::default()).await;
    let fault = format!("http://127.0.0.1:{admin_port}/admin/fault/{name}");
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

#[tokio::test]
async fn a_disaggregated_worker_refuses_a_request_without_its_rendezvous() {
    for role in [Role::Prefill, Role::Decode] {
        let (client, _) = start(role, None, None).await;
        // No rendezvous: the request would pair nowhere, as the engines'
        // disaggregated servers answer, and the refusal names the field.
        let refused = client
            .generate(request("pd-r1", &[1, 2, 3], false))
            .await
            .err()
            .expect("no rendezvous, not a stream");
        assert_eq!(
            refused.code(),
            tonic::Code::InvalidArgument,
            "{role:?}: {refused:?}"
        );
        assert!(
            refused.message().contains("kv_bootstrap_info"),
            "{refused:?}"
        );
        // A rendezvous with a port of 0 names none.
        let empty = client
            .generate(rendezvous(request("pd-r2", &[1, 2, 3], false), 0))
            .await
            .err()
            .expect("a port of 0, not a stream");
        assert_eq!(
            empty.code(),
            tonic::Code::InvalidArgument,
            "{role:?}: {empty:?}"
        );
        // The rendezvous the gateway mints: answered as before.
        let frames = generate(
            &client,
            rendezvous(request("pd-r3", &[1, 2, 3], false), 8998),
        )
        .await;
        assert!(
            matches!(
                frames.last().and_then(|frame| frame.response.as_ref()),
                Some(GenResp::Complete(_))
            ),
            "{role:?}: {frames:?}"
        );
        // A room of 0 is a room the gateway can mint: taken like any other.
        let frames = generate(
            &client,
            rendezvous_room(request("pd-r4", &[1, 2, 3], false), 8998, 0),
        )
        .await;
        assert!(
            matches!(
                frames.last().and_then(|frame| frame.response.as_ref()),
                Some(GenResp::Complete(_))
            ),
            "{role:?}: {frames:?}"
        );
    }
}

/// On the SGLang wire the gateway draws a room from 0 up, so 0 is a room
/// like any other: both disaggregated roles take it.
#[tokio::test]
async fn a_disaggregated_worker_takes_a_room_of_zero_on_the_sglang_wire() {
    for role in [Role::Prefill, Role::Decode] {
        let client = start_sglang(role).await;
        let mut stream = client
            .generate(sglang_request("pd-sg-0", 0))
            .await
            .expect("a room of 0 names a rendezvous");
        let mut last = None;
        while let Some(frame) = stream.next().await {
            last = Some(frame.expect("a response frame"));
        }
        stream.mark_completed();
        assert!(
            matches!(
                last.as_ref().and_then(|frame| frame.response.as_ref()),
                Some(sg::generate_response::Response::Complete(_))
            ),
            "{role:?}: {last:?}"
        );
    }
}

#[tokio::test]
async fn a_prefill_worker_refuses_a_rendezvous_naming_another_port() {
    let bootstrap_port = free_port();
    let (client, _) = start(Role::Prefill, Some(bootstrap_port), None).await;
    // Its decode partner would come to the named port for the handoff: a
    // port other than the one this worker bound is refused with both named.
    let elsewhere = client
        .generate(rendezvous(
            request("pd-p1", &[1, 2, 3], false),
            i32::from(bootstrap_port) + 1,
        ))
        .await
        .err()
        .expect("another port, not a stream");
    assert_eq!(
        elsewhere.code(),
        tonic::Code::InvalidArgument,
        "{elsewhere:?}"
    );
    assert!(
        elsewhere.message().contains(&bootstrap_port.to_string()),
        "{elsewhere:?}"
    );
    let frames = generate(
        &client,
        rendezvous(
            request("pd-p2", &[1, 2, 3], false),
            i32::from(bootstrap_port),
        ),
    )
    .await;
    assert_eq!(
        frames.len(),
        1,
        "the prompt side's completion on its own port: {frames:?}"
    );
}
/// An encode worker on the TokenSpeed wire, canned or realistic, worker 0 of
/// its process with a bootstrap port, and the admin API beside it: the
/// scheduler client, the encoder client, the HTTP client, the fault route and
/// the worker's name.
async fn start_tokenspeed_encoder(
    realistic: bool,
) -> (
    TokenSpeedSchedulerClient,
    enc::token_speed_encoder_client::TokenSpeedEncoderClient<Channel>,
    reqwest::Client,
    String,
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
        grpc_wire: SchedulerWire::TokenSpeed,
        realistic,
        role: Some(Role::Encode),
        bootstrap_base_port: Some(free_port()),
        ..Config::default()
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    let endpoint = format!("http://127.0.0.1:{port}");
    let scheduler = TokenSpeedSchedulerClient::connect(&endpoint)
        .await
        .expect("connect to the mock worker");
    let channel = Channel::from_shared(endpoint)
        .expect("endpoint")
        .connect()
        .await
        .expect("connect to the encoder");
    let (http, fault) = fault_route(port).await;
    (
        scheduler,
        enc::token_speed_encoder_client::TokenSpeedEncoderClient::new(channel),
        http,
        fault,
        format!("grpc:{port}"),
    )
}

/// A media item as the gateway builds it for the TokenSpeed wire: an image
/// with a content hash, an inline feature and one placeholder token at
/// `index` of the prompt (the worker checks the shape before it answers).
fn tokenspeed_media_item(index: usize) -> ts::MultimodalItem {
    ts::MultimodalItem {
        modality: smg_grpc_client::common_proto::Modality::Image as i32,
        content_hash: vec![u8::try_from(index).expect("a small index") + 1; 8],
        encoder_input: Some(ts::TensorData {
            shape: vec![1, 4],
            dtype: "float32".to_string(),
            payload: Some(ts::tensor_data::Payload::Inline(vec![0; 16].into())),
        }),
        placeholders: vec![ts::PlaceholderRange {
            offset: u32::try_from(index).expect("a small index"),
            length: 1,
        }],
        ..Default::default()
    }
}

/// What the gateway sends a TokenSpeed encoder: the request's media items
/// and one rendezvous room per item.
fn tokenspeed_encode_request(id: &str, items: usize, rooms: &[i64]) -> enc::EncodeRequest {
    enc::EncodeRequest {
        request_id: id.to_string(),
        mm_inputs: Some(ts::MultimodalInputs {
            items: (0..items).map(tokenspeed_media_item).collect(),
        }),
        items: rooms
            .iter()
            .map(|&bootstrap_room| enc::EncodeItemAssignment { bootstrap_room })
            .collect(),
    }
}

/// The worker's `encodes` count off the admin API's ops status, as a bed
/// reads it.
async fn encodes_served(http: &reqwest::Client, fault: &str, name: &str) -> u64 {
    let base = fault.split("/admin/fault/").next().expect("the admin base");
    let status: Value = http
        .get(format!("{base}/admin/ops/all"))
        .send()
        .await
        .expect("the ops status")
        .json()
        .await
        .expect("json");
    status["workers"]
        .as_array()
        .expect("workers")
        .iter()
        .find(|worker| worker["worker"] == name)
        .map_or(0, |worker| worker["encodes"].as_u64().unwrap_or(0))
}

async fn arm(http: &reqwest::Client, fault: &str, query: &str) {
    let armed = http
        .post(format!("{fault}/fail?{query}"))
        .send()
        .await
        .expect("arm the fault");
    assert_eq!(armed.status(), 200);
}

#[tokio::test]
async fn a_tokenspeed_encode_worker_accepts_the_media_rooms_counts_them_and_refuses_generate() {
    let (scheduler, mut encoder, http, fault, name) = start_tokenspeed_encoder(false).await;

    // One media item in one room: accepted and counted.
    let before = encodes_served(&http, &fault, &name).await;
    let answer = encoder
        .encode(tokenspeed_encode_request("ts-e1", 1, &[42]))
        .await
        .expect("the encoder's answer")
        .into_inner();
    assert!(answer.accepted);
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);

    // Rooms that do not match the items, no item at all, or a room of 0:
    // refused as malformed and not counted.
    for (items, rooms) in [(2usize, &[7i64][..]), (0, &[][..]), (1, &[0][..])] {
        let refused = encoder
            .encode(tokenspeed_encode_request("ts-e2", items, rooms))
            .await
            .expect_err("a malformed encode request");
        assert_eq!(
            refused.code(),
            tonic::Code::InvalidArgument,
            "{items} item(s), rooms {rooms:?}: {refused:?}"
        );
    }
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);

    // An item without its feature: the encode worker runs the tower on it
    // and has no room to fall back on, so the request is refused and not
    // counted.
    let mut featureless = tokenspeed_encode_request("ts-e3", 1, &[43]);
    featureless.mm_inputs.as_mut().expect("the items").items[0].encoder_input = None;
    let refused = encoder
        .encode(featureless)
        .await
        .expect_err("an item without its feature");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument, "{refused:?}");
    assert!(
        refused.message().contains("media item 0")
            && refused.message().contains("no encoder input"),
        "{refused:?}"
    );
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);

    // A `Generate` went to the wrong pool: this worker takes `Encode`.
    let wrong_pool = scheduler
        .generate(request("ts-g1", &[1, 2, 3], false))
        .await
        .err()
        .expect("the wrong pool, not a stream");
    assert_eq!(
        wrong_pool.code(),
        tonic::Code::FailedPrecondition,
        "{wrong_pool:?}"
    );

    // It advertises its role and the bootstrap port it bound, where the
    // prefill side's engine would come for the embedding.
    let info = scheduler.get_server_info().await.expect("server info");
    assert_eq!(
        server_arg(&info, "disaggregation_mode").as_deref(),
        Some("encode")
    );
    assert!(
        server_arg(&info, "disaggregation_bootstrap_port").is_some(),
        "{info:?}"
    );
}

/// The encoder servicer reads its feature through the generic tensor reader,
/// not the scheduler's feature path: an SHM handle in a dtype the scheduler
/// refuses for its feature is accepted here, with the byte length the
/// generic reader checks.
#[tokio::test]
async fn a_tokenspeed_encode_worker_reads_an_shm_feature_as_the_encoder_servicer_does() {
    let (_scheduler, mut encoder, http, fault, name) = start_tokenspeed_encoder(false).await;
    let before = encodes_served(&http, &fault, &name).await;
    let mut shm = tokenspeed_encode_request("ts-shm", 1, &[45]);
    shm.mm_inputs.as_mut().expect("the items").items[0].encoder_input = Some(ts::TensorData {
        shape: vec![1, 4],
        dtype: "float64".to_string(),
        payload: Some(ts::tensor_data::Payload::Shm(
            smg_grpc_client::common_proto::ShmHandle {
                name: "smg-mm-enc".to_string(),
                nbytes: 32,
                ..Default::default()
            },
        )),
    });
    let answer = encoder
        .encode(shm)
        .await
        .expect("a float64 SHM feature on the encoder")
        .into_inner();
    assert!(answer.accepted);
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);

    let mut short = tokenspeed_encode_request("ts-shm-short", 1, &[46]);
    short.mm_inputs.as_mut().expect("the items").items[0].encoder_input = Some(ts::TensorData {
        shape: vec![1, 4],
        dtype: "float64".to_string(),
        payload: Some(ts::tensor_data::Payload::Shm(
            smg_grpc_client::common_proto::ShmHandle {
                name: "smg-mm-enc".to_string(),
                nbytes: 16,
                ..Default::default()
            },
        )),
    });
    let refused = encoder
        .encode(short)
        .await
        .expect_err("a byte length short of the shape");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument, "{refused:?}");
    assert!(refused.message().contains("byte length 16"), "{refused:?}");
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);
}

/// Both servicers pull a remote handle for the item's feature alone: the
/// encoder accepts a remote feature as the scheduler does, and a remote side
/// tensor is refused, as the generic reader's `tensor_payload_bytes` refuses
/// it on either.
#[tokio::test]
async fn a_tokenspeed_encode_worker_pulls_a_remote_feature_and_refuses_a_remote_side_tensor() {
    let (_scheduler, mut encoder, http, fault, name) = start_tokenspeed_encoder(false).await;
    let before = encodes_served(&http, &fault, &name).await;
    let remote = |nbytes: u64| {
        Some(ts::tensor_data::Payload::Remote(
            smg_grpc_client::common_proto::RemoteTensorHandle {
                transport: "nixl".to_string(),
                nbytes,
                ..Default::default()
            },
        ))
    };
    let mut feature = tokenspeed_encode_request("ts-remote", 1, &[47]);
    feature.mm_inputs.as_mut().expect("the items").items[0].encoder_input = Some(ts::TensorData {
        shape: vec![1, 4],
        dtype: "float16".to_string(),
        payload: remote(8),
    });
    let answer = encoder
        .encode(feature)
        .await
        .expect("a remote feature on the encoder")
        .into_inner();
    assert!(answer.accepted);
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);

    let mut side = tokenspeed_encode_request("ts-remote-side", 1, &[48]);
    side.mm_inputs.as_mut().expect("the items").items[0]
        .model_specific_tensors
        .insert(
            "image_grid_thw".to_string(),
            ts::TensorData {
                shape: vec![1, 3],
                dtype: "int64".to_string(),
                payload: remote(24),
            },
        );
    let refused = encoder
        .encode(side)
        .await
        .expect_err("a remote side tensor the servicer refuses");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument, "{refused:?}");
    assert!(
        refused.message().contains("image_grid_thw")
            && refused.message().contains("remote payload"),
        "{refused:?}"
    );
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);
}

/// The engine's encoder takes exactly one media item under one room per
/// request, named by a request id: the mock refuses the rest as the engine's
/// adapter does, so a gateway that batched items into one `Encode` or dropped
/// the id would not pass the mock row.
#[tokio::test]
async fn a_tokenspeed_encode_worker_takes_one_item_in_one_room_with_a_request_id() {
    let (_scheduler, mut encoder, http, fault, name) = start_tokenspeed_encoder(false).await;
    let before = encodes_served(&http, &fault, &name).await;
    for (id, items, rooms) in [("ts-one-two", 2usize, &[7i64, 8][..]), ("", 1, &[9][..])] {
        let refused = encoder
            .encode(tokenspeed_encode_request(id, items, rooms))
            .await
            .expect_err("refused as the engine's adapter refuses it");
        assert_eq!(
            refused.code(),
            tonic::Code::InvalidArgument,
            "{id:?}, {items} item(s), rooms {rooms:?}: {refused:?}"
        );
    }
    assert_eq!(encodes_served(&http, &fault, &name).await, before);
    let answer = encoder
        .encode(tokenspeed_encode_request("ts-one", 1, &[10]))
        .await
        .expect("one item in one room with an id")
        .into_inner();
    assert!(answer.accepted);
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);
}

#[tokio::test]
async fn a_realistic_tokenspeed_encode_worker_takes_the_fault_hooks() {
    let (_scheduler, mut encoder, http, fault, name) = start_tokenspeed_encoder(true).await;

    // One 503 armed: refused before the answer, as the prompt side of a
    // prefill worker is, and the failure is counted.
    arm(&http, &fault, "status=503&count=1").await;
    let refused = encoder
        .encode(tokenspeed_encode_request("ts-f1", 1, &[1]))
        .await
        .expect_err("the armed status, not an answer");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (1, 0));

    // An armed cut: the request is not accepted, the cut is counted, and the
    // side still ran for the bed that counts it.
    arm(&http, &fault, "finish=error&status=503&count=1").await;
    let before = encodes_served(&http, &fault, &name).await;
    let answer = encoder
        .encode(tokenspeed_encode_request("ts-f2", 1, &[2]))
        .await
        .expect("an answer")
        .into_inner();
    assert!(!answer.accepted, "the armed cut is the engine's refusal");
    assert_eq!(fault_counts(&http, &fault).await, (1, 1));
    assert_eq!(encodes_served(&http, &fault, &name).await, before + 1);

    // A malformed request spends no armed fault: the fault waits for the
    // next well-formed one.
    arm(&http, &fault, "status=503&count=1").await;
    let malformed = encoder
        .encode(tokenspeed_encode_request("ts-f3", 2, &[3]))
        .await
        .expect_err("a malformed encode request");
    assert_eq!(malformed.code(), tonic::Code::InvalidArgument);
    assert_eq!(fault_counts(&http, &fault).await, (1, 1));
    let refused = encoder
        .encode(tokenspeed_encode_request("ts-f4", 1, &[4]))
        .await
        .expect_err("the armed status, not an answer");
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused:?}");
    assert_eq!(fault_counts(&http, &fault).await, (2, 1));
}

#[tokio::test]
async fn a_tokenspeed_prefill_worker_checks_the_rooms_of_its_media_items() {
    let (client, _name) = start(Role::Prefill, None, None).await;
    let room = |item_index: u32, bootstrap_room: i64| ts::EncodeItemBootstrapInfo {
        item_index,
        bootstrap_host: "127.0.0.1".to_string(),
        bootstrap_port: 8998,
        bootstrap_room,
    };
    // The prefill side's own rendezvous rides every request, as on the
    // gateway's wire: the rooms are read at the top of generate, the
    // rendezvous as the frames start, and a request without one ends there.
    let with_media = |id: &str, rooms: Option<Vec<ts::EncodeItemBootstrapInfo>>| {
        let mut req = rendezvous(request(id, &[1, 2, 3], false), 8998);
        req.mm_inputs = Some(ts::MultimodalInputs {
            items: (0..2).map(tokenspeed_media_item).collect(),
        });
        req.encode_bootstrap_info = rooms.map(|items| ts::EncodeBootstrapInfo { items });
        req
    };

    // A room for each item: the prompt side's completion, as for any request.
    let frames = generate(
        &client,
        with_media("ts-p1", Some(vec![room(0, 11), room(1, 12)])),
    )
    .await;
    let Some(GenResp::Complete(complete)) = &frames[0].response else {
        panic!("a Complete, not {:?}", frames[0]);
    };
    assert!(complete.output_ids.is_empty(), "no output: {complete:?}");
    assert_eq!(complete.finish_reason, "stop");

    // An item without a room, a room of 0, an item named twice, an index off
    // the end: refused, as the engine's prefill would wait on that room and
    // fail.
    for rooms in [
        vec![room(0, 11)],
        vec![room(0, 11), room(1, 0)],
        vec![room(0, 11), room(0, 12)],
        vec![room(0, 11), room(2, 12)],
    ] {
        let refused = client
            .generate(with_media("ts-p2", Some(rooms.clone())))
            .await
            .err()
            .expect("a refused handoff, not a stream");
        assert_eq!(
            refused.code(),
            tonic::Code::InvalidArgument,
            "rooms {rooms:?}: {refused:?}"
        );
    }

    // No rooms at all: a request without an encoder, served as before.
    let frames = generate(&client, with_media("ts-p3", None)).await;
    let Some(GenResp::Complete(complete)) = &frames[0].response else {
        panic!("a Complete, not {:?}", frames[0]);
    };
    assert_eq!(complete.finish_reason, "stop");
}
