//! The recorder on the vLLM wire, the model of a recording on another wire,
//! a recorded line posted through the admin API, the metadata a recorder
//! hands on and what it says of a call it cannot hand on. A file of its own:
//! a script is loaded for every worker of a
//! process, so this replay shares none with the tests of `record.rs`.

use std::{sync::Arc, time::Duration};

use anyhow::Context as _;
use futures::StreamExt as _;
use mock_worker::{
    config::{Config, ReplayConfig, SchedulerWire},
    grpc, script,
};
use smg_grpc_client::{vllm_proto as vllm, TokenSpeedSchedulerClient, VllmEngineClient};
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::TcpListener,
};
use tonic_health::pb::{
    health_client::HealthClient,
    health_server::{Health, HealthServer},
};

/// A canned worker of `wire` with no tokenizer files, named `model`.
fn config(model: &str, wire: SchedulerWire, replay: ReplayConfig) -> Config {
    Config {
        model_id: model.to_string(),
        tokenizer_path: "/nonexistent/tokenizer-dir".to_string(),
        output_tokens: 3,
        grpc_wire: wire,
        replay,
        ..Config::default()
    }
}

/// Serve `cfg` as a gRPC worker on a free port: its address.
async fn start(cfg: Config) -> anyhow::Result<String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let port = listener.local_addr().context("listener address")?.port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(grpc::serve_with_listener(Arc::new(cfg), listener));
    let address = format!("http://127.0.0.1:{port}");
    // A connection made, as `start` of `record.rs` makes one: the worker
    // runs by then, and the script store knows it.
    TokenSpeedSchedulerClient::connect(&address)
        .await
        .map_err(anyhow::Error::from_boxed)
        .context("connect to the worker")?;
    Ok(address)
}

/// A vLLM client of the worker at `address`.
async fn vllm_client(address: &str) -> anyhow::Result<VllmEngineClient> {
    VllmEngineClient::connect(address)
        .await
        .map_err(anyhow::Error::from_boxed)
        .context("connect to the worker")
}

/// The frames the worker behind `client` answers `req` with.
async fn frames(
    client: &VllmEngineClient,
    req: vllm::GenerateRequest,
) -> anyhow::Result<Vec<vllm::GenerateResponse>> {
    let mut stream = client.generate(req).await.context("an answer")?;
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        frames.push(*frame.context("a response frame")?);
    }
    stream.mark_completed();
    Ok(frames)
}

/// A call through the recorder on the vLLM wire reads at the replay, frame
/// for frame, what it read at the recording, and the two info calls read the
/// engine's answers; a replay on another wire states what shapes a request
/// (here the vision flag) as the recording has it, under its own name.
#[tokio::test]
async fn a_recording_on_the_vllm_wire_replays_and_shapes_the_model_of_another_wire(
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir().context("temp dir")?;
    let record = dir.path().join("session.jsonl");
    // Ahead of everything else, while the process holds no recorded line: a
    // line that states what its request asked is a recorded line however it
    // came. Posted through the admin API it makes a request that finds
    // nothing fail on the gRPC workers of the process, where a line of a
    // prompt alone leaves it to the engine, and dropping the queue does not
    // take that back.
    let plain = start(config(
        "plain",
        SchedulerWire::Vllm,
        ReplayConfig::default(),
    ))
    .await?;
    let port = plain.rsplit(':').next().context("the port of the worker")?;
    let worker = format!("grpc:{port}");
    let plain = vllm_client(&plain).await?;
    let miss = vllm::GenerateRequest {
        request_id: "a-miss".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                input_ids: vec![99],
                original_text: String::new(),
            },
        )),
        stream: true,
        ..Default::default()
    };
    let line = |line: serde_json::Value| {
        let line = script::Answer::parse(&line).map_err(anyhow::Error::msg)?;
        script::enqueue(&worker, vec![line]).context("a started worker")?;
        anyhow::Ok(())
    };
    line(serde_json::json!({"input_ids": [7], "output_ids": [1]}))?;
    assert!(
        !frames(&plain, miss.clone()).await?.is_empty(),
        "a line of a prompt alone: the engine answers a miss"
    );
    line(serde_json::json!({
        "input_ids": [7], "output_ids": [1], "recorded": {"asked": {"stream": true}},
    }))?;
    for dropped in [false, true] {
        if dropped {
            script::clear(&worker).context("a started worker")?;
        }
        let failed = match plain.generate(miss.clone()).await {
            Err(error) => error,
            Ok(_) => anyhow::bail!("a miss has no answer once a recorded line is held"),
        };
        assert_eq!(failed.code(), tonic::Code::Internal, "{failed}");
    }

    // The engine reads pictures and the replays are started without the
    // flag: the vision flag a replay states is the recording's.
    let engine = start(Config {
        supports_vision: Some(true),
        ..config("the-engine", SchedulerWire::Vllm, ReplayConfig::default())
    })
    .await?;
    // The engine says a scripted answer, as the engine of `record.rs` does:
    // a canned stream states a count of one in every chunk, which no
    // recorded line keeps.
    let answer = script::Answer::parse(&serde_json::json!({"output_ids": [7, 8, 9]}))
        .map_err(anyhow::Error::msg)
        .context("an answer")?;
    let port = engine
        .rsplit(':')
        .next()
        .context("the port of the engine")?;
    script::enqueue(&format!("grpc:{port}"), vec![answer]).context("a started worker")?;
    let recorder = start(config(
        "the-recorder",
        SchedulerWire::Vllm,
        ReplayConfig {
            record_upstream: Some(engine.replacen("http://", "grpc://", 1)),
            record: Some(record.clone()),
            ..ReplayConfig::default()
        },
    ))
    .await?;
    let ask = vllm::GenerateRequest {
        request_id: "on-the-vllm-wire".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                input_ids: vec![11, 12, 13],
                original_text: String::new(),
            },
        )),
        stream: true,
        ..Default::default()
    };
    let through = vllm_client(&recorder).await?;
    let model = through.get_model_info().await.context("the model info")?;
    let server = through.get_server_info().await.context("the server info")?;
    let recorded = frames(&through, ask.clone()).await?;
    assert!(!recorded.is_empty(), "the engine answered");
    let written = || {
        std::fs::read_to_string(&record)
            .unwrap_or_default()
            .lines()
            .count()
    };
    for _ in 0..500 {
        if written() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(written(), 1, "one line for the one choice asked");

    let script = ReplayConfig {
        script: Some(record),
        ..ReplayConfig::default()
    };
    let replay = start(config("the-replay", SchedulerWire::Vllm, script.clone())).await?;
    let replay = vllm_client(&replay).await?;
    assert_eq!(frames(&replay, ask).await?, recorded);
    assert_eq!(
        replay.get_model_info().await.context("the model info")?,
        model
    );
    assert_eq!(
        replay.get_server_info().await.context("the server info")?,
        server
    );

    let other = start(config("the-replay", SchedulerWire::TokenSpeed, script)).await?;
    let other = TokenSpeedSchedulerClient::connect(&other)
        .await
        .map_err(anyhow::Error::from_boxed)
        .context("connect to the worker")?;
    let stated = other.get_model_info().await.context("the model info")?;
    assert!(
        format!("{stated:?}").contains("supports_vision: true"),
        "what shapes a request is the recording's on another wire: {stated:?}"
    );
    Ok(())
}

/// An engine that answers every call with an empty stream and states the
/// `x-asked` header of the call as `x-answered`.
async fn echo(request: axum::extract::Request) -> axum::response::Response {
    let mut response = axum::response::Response::new(axum::body::Body::empty());
    let headers = response.headers_mut();
    let fixed = axum::http::HeaderValue::from_static;
    headers.insert("content-type", fixed("application/grpc"));
    headers.insert("grpc-status", fixed("0"));
    if let Some(asked) = request.headers().get("x-asked") {
        headers.insert("x-answered", asked.clone());
    }
    response
}

/// The recorder hands the metadata of a call on to the engine and the
/// metadata of the engine's answer back to the caller.
#[tokio::test]
async fn a_recorder_hands_the_metadata_of_a_call_both_ways() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let engine = listener.local_addr().context("listener address")?.port();
    let routes = tonic::service::Routes::from(axum::Router::new().fallback(echo));
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _engine = tokio::spawn(
        tonic::transport::Server::builder()
            .add_routes(routes)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
    );
    let dir = tempfile::tempdir().context("temp dir")?;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let recorder = listener.local_addr().context("listener address")?.port();
    let cfg = config(
        "the-recorder",
        SchedulerWire::Vllm,
        ReplayConfig {
            record_upstream: Some(format!("grpc://127.0.0.1:{engine}")),
            record: Some(dir.path().join("session.jsonl")),
            ..ReplayConfig::default()
        },
    );
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _recorder = tokio::spawn(grpc::serve_with_listener(Arc::new(cfg), listener));
    let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{recorder}"))
        .context("the address of the recorder")?
        .connect()
        .await
        .context("connect to the recorder")?;
    let mut call = tonic::Request::new(tonic_health::pb::HealthCheckRequest {
        service: String::new(),
    });
    let asked = tonic::metadata::MetadataValue::from_static("forty-two");
    call.metadata_mut().insert("x-asked", asked);
    let answer = HealthClient::new(channel)
        .watch(call)
        .await
        .context("the engine answers through the recorder")?;
    let answered = answer.metadata().get("x-answered");
    assert_eq!(
        answered.and_then(|value| value.to_str().ok()),
        Some("forty-two"),
        "the engine read the metadata of the call, and the caller that of its answer"
    );
    Ok(())
}

/// A channel to a recorder in front of the engine on `upstream`, its record
/// file in `dir`; `limit` is the size of the largest message it reads where
/// one is given.
async fn recorder(
    upstream: u16,
    limit: Option<usize>,
    dir: &std::path::Path,
) -> anyhow::Result<tonic::transport::Channel> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let port = listener.local_addr().context("listener address")?.port();
    let mut cfg = config(
        "the-recorder",
        SchedulerWire::Vllm,
        ReplayConfig {
            record_upstream: Some(format!("grpc://127.0.0.1:{upstream}")),
            record: Some(dir.join("session.jsonl")),
            ..ReplayConfig::default()
        },
    );
    if let Some(limit) = limit {
        cfg.grpc_max_message_bytes = limit;
    }
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _recorder = tokio::spawn(grpc::serve_with_listener(Arc::new(cfg), listener));
    tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
        .context("the address of the recorder")?
        .connect()
        .await
        .context("connect to the recorder")
}

/// A health check that names `service`.
fn check(service: String) -> tonic_health::pb::HealthCheckRequest {
    tonic_health::pb::HealthCheckRequest { service }
}

/// A request message over the recorder's size limit is answered with the
/// status of that message, as a worker answers it itself, and the recorder
/// serves the next call.
#[test]
fn a_request_message_over_the_size_limit_reads_its_status_through_the_recorder(
) -> anyhow::Result<()> {
    // A runtime of the test's own, let go without waiting for its threads:
    // a recorder that did not end the forwarded request would keep one busy
    // for ever, and the test would not come back to say so.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .context("a runtime")?;
    let told = runtime.block_on(async {
        let engine = start(config(
            "the-engine",
            SchedulerWire::Vllm,
            ReplayConfig::default(),
        ))
        .await?;
        let engine = engine
            .rsplit(':')
            .next()
            .context("the port of the engine")?;
        let dir = tempfile::tempdir().context("temp dir")?;
        let channel = recorder(engine.parse()?, Some(4096), dir.path()).await?;
        let bound = Duration::from_secs(10);
        let mut client = HealthClient::new(channel);
        let large = client.check(check("s".repeat(8192)));
        let refused = match tokio::time::timeout(bound, large).await {
            Err(_) => anyhow::bail!("the recorder answered nothing inside {bound:?}"),
            Ok(Ok(answer)) => anyhow::bail!("a message over the limit was answered: {answer:?}"),
            Ok(Err(status)) => status,
        };
        anyhow::ensure!(
            refused.code() == tonic::Code::OutOfRange,
            "the status of the message that did not read: {refused}"
        );
        let next = client.check(check(String::new()));
        tokio::time::timeout(bound, next)
            .await
            .context("the recorder serves the next call")?
            .context("the engine answers the next call")?;
        anyhow::Ok(())
    });
    runtime.shutdown_background();
    told
}

/// A recorder whose engine does not listen says so at the first call: the
/// failed connect reads as the recorder's upstream, unavailable.
#[tokio::test]
async fn a_recorder_whose_engine_does_not_listen_says_so_at_the_first_call() -> anyhow::Result<()> {
    // A port nothing listens on: bound once and let go.
    let closed = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?
        .local_addr()
        .context("listener address")?
        .port();
    let dir = tempfile::tempdir().context("temp dir")?;
    let channel = recorder(closed, None, dir.path()).await?;
    let Err(refused) = HealthClient::new(channel).check(check(String::new())).await else {
        anyhow::bail!("a call was answered with no engine behind the recorder");
    };
    assert_eq!(refused.code(), tonic::Code::Unavailable, "{refused}");
    assert!(
        refused.message().starts_with("the recorder's upstream"),
        "the failure is the recorder's to name: {refused}"
    );
    Ok(())
}

/// One connection of an engine that resets every call ahead of its first
/// message with `reason`: ahead of its headers, or behind them where it
/// `answers`. It speaks the frames of HTTP/2 by hand: its settings, the
/// acknowledgement of the caller's, and a reset of the stream of each
/// request's headers. It ends with the error of its connection.
async fn resets(listener: &TcpListener, reason: u32, answers: bool) -> std::io::Result<()> {
    let (mut engine, _) = listener.accept().await?;
    // The caller's preface, answered with settings of no entry.
    engine.read_exact(&mut [0; 24]).await?;
    engine.write_all(&[0, 0, 0, 4, 0, 0, 0, 0, 0]).await?;
    loop {
        // A frame: three bytes of length, its kind, its flags, its stream.
        let mut head = [0; 9];
        engine.read_exact(&mut head).await?;
        let [high, mid, low, kind, flags, stream @ ..] = head;
        let length = u32::from_be_bytes([0, high, mid, low]);
        let mut payload = vec![0; usize::try_from(length).unwrap_or(usize::MAX)];
        engine.read_exact(&mut payload).await?;
        match (kind, flags & 1) {
            // Settings that acknowledge nothing: acknowledged.
            (4, 0) => engine.write_all(&[0, 0, 0, 4, 1, 0, 0, 0, 0]).await?,
            // The headers of a request: its stream is reset, behind the
            // headers of an answer where the engine answers.
            (1, _) => {
                let mut frames = Vec::new();
                if answers {
                    // Headers that end no stream, as HPACK writes them:
                    // `:status: 200`, `content-type: application/grpc`.
                    frames.extend_from_slice(&[0, 0, 20, 1, 4]);
                    frames.extend_from_slice(&stream);
                    frames.extend_from_slice(&[0x88, 0x0f, 0x10, 0x10]);
                    frames.extend_from_slice(b"application/grpc");
                }
                frames.extend_from_slice(&[0, 0, 4, 3, 0]);
                frames.extend_from_slice(&stream);
                frames.extend_from_slice(&reason.to_be_bytes());
                engine.write_all(&frames).await?;
            }
            _ => {}
        }
    }
}

/// The status a health check over `channel` reads of the engine behind
/// `listener`, which resets it with `reason`, behind its headers where it
/// `answers`.
async fn reset(
    channel: tonic::transport::Channel,
    listener: &TcpListener,
    reason: u32,
    answers: bool,
) -> anyhow::Result<tonic::Status> {
    let mut client = HealthClient::new(channel);
    let call = client.check(check(String::new()));
    // A bound: were the connection the engine accepts not the one of the
    // call, neither would end.
    let told = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::select! {
            told = call => told.err().context("a call the engine reset was answered"),
            ended = resets(listener, reason, answers) => {
                anyhow::bail!("the engine ended first: {ended:?}")
            }
        }
    });
    told.await
        .context("neither the call nor the engine ended")?
}

/// A health check the engine resets with ENHANCE_YOUR_CALM (the reset of an
/// engine that pushes back on load), behind its headers where it `answers`:
/// read through the recorder, it holds the code a caller of the engine
/// itself reads, the recorder's upstream ahead of the words.
async fn a_reset_reads_its_code_through_the_recorder(answers: bool) -> anyhow::Result<()> {
    const CALM: u32 = 0xb;
    let engine = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let port = engine.local_addr().context("listener address")?.port();
    let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
        .context("the address of the engine")?
        .connect_lazy();
    let direct = reset(channel, &engine, CALM, answers).await?;
    assert_eq!(direct.code(), tonic::Code::ResourceExhausted, "{direct}");
    let dir = tempfile::tempdir().context("temp dir")?;
    let through = recorder(port, None, dir.path()).await?;
    let recorded = reset(through, &engine, CALM, answers).await?;
    assert_eq!(recorded.code(), direct.code(), "{recorded}");
    assert!(
        recorded.message().starts_with("the recorder's upstream"),
        "the failure is the recorder's to name: {recorded}"
    );
    Ok(())
}

/// A stream the engine resets ahead of its headers reads through the
/// recorder with the code a caller of the engine itself reads, the
/// recorder's upstream ahead of the words.
#[tokio::test]
async fn a_call_the_engine_resets_reads_its_code_through_the_recorder() -> anyhow::Result<()> {
    a_reset_reads_its_code_through_the_recorder(false).await
}

/// A stream the engine resets behind its headers, ahead of its first
/// message, reads through the recorder as one reset ahead of them.
#[tokio::test]
async fn a_call_reset_behind_the_engines_headers_reads_alike() -> anyhow::Result<()> {
    a_reset_reads_its_code_through_the_recorder(true).await
}

/// An engine whose `Watch` states one answer and ends well with a trailer
/// of its own, `x-ended`.
struct Trailing;

#[tonic::async_trait]
impl Health for Trailing {
    type WatchStream = futures::stream::Iter<
        std::vec::IntoIter<Result<tonic_health::pb::HealthCheckResponse, tonic::Status>>,
    >;

    async fn check(
        &self,
        _request: tonic::Request<tonic_health::pb::HealthCheckRequest>,
    ) -> Result<tonic::Response<tonic_health::pb::HealthCheckResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented(
            "this engine answers Watch alone",
        ))
    }

    async fn watch(
        &self,
        _request: tonic::Request<tonic_health::pb::HealthCheckRequest>,
    ) -> Result<tonic::Response<Self::WatchStream>, tonic::Status> {
        let mut trailers = tonic::metadata::MetadataMap::new();
        let ended = tonic::metadata::MetadataValue::from_static("well");
        trailers.insert("x-ended", ended);
        // A status of the code OK ends the call well with these trailers.
        let ended = tonic::Status::with_metadata(tonic::Code::Ok, "", trailers);
        let answer = tonic_health::pb::HealthCheckResponse { status: 1 };
        Ok(tonic::Response::new(futures::stream::iter(vec![
            Ok(answer),
            Err(ended),
        ])))
    }
}

/// The trailers an engine sends with a call that ended well reach the caller
/// behind the last message.
#[tokio::test]
async fn a_recorder_hands_on_the_trailers_of_a_call_that_ended_well() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let engine = listener.local_addr().context("listener address")?.port();
    let service = HealthServer::new(Trailing);
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _engine = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
    );
    let dir = tempfile::tempdir().context("temp dir")?;
    let channel = recorder(engine, None, dir.path()).await?;
    let mut answers = HealthClient::new(channel)
        .watch(check(String::new()))
        .await
        .context("the engine answers through the recorder")?
        .into_inner();
    let mut read = 0;
    while answers.message().await.context("an answer")?.is_some() {
        read += 1;
    }
    assert_eq!(read, 1, "the one answer of the engine");
    let trailers = answers.trailers().await.context("the end of the call")?;
    let ended = trailers
        .as_ref()
        .and_then(|trailers| trailers.get("x-ended"));
    assert_eq!(
        ended.and_then(|value| value.to_str().ok()),
        Some("well"),
        "the trailer of the engine behind its last message: {trailers:?}"
    );
    Ok(())
}
