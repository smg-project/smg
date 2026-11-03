//! The gRPC workers' roles in a disaggregated fleet (`--worker-type`): what
//! each side answers, what it advertises, and the port a prefill worker binds.

use std::{
    net::TcpListener as StdTcpListener,
    sync::Arc,
    time::{Duration, Instant},
};

use mock_worker::{
    admin,
    config::{Config, Role},
    grpc::serve_with_listener,
    script,
};
use serde_json::json;
use smg_grpc_client::{tokenspeed_proto as ts, TokenSpeedSchedulerClient};
use tokio::{
    io::AsyncReadExt,
    net::{TcpListener, TcpStream},
};
use ts::generate_response::Response as GenResp;

mod common;
use common::generate;

/// A port nothing listens on right now (bound and released).
fn free_port() -> u16 {
    StdTcpListener::bind("127.0.0.1:0")
        .expect("bind a free port")
        .local_addr()
        .expect("listener address")
        .port()
}

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
    (client, http, fault)
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
