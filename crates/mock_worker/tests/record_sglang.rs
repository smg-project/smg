//! A replay of a recording made on the SGLang wire, started with a tokenizer
//! path of its own. An engine of this wire names its tokenizer in the server
//! arguments as well as in the model info, and a gateway reads the server
//! arguments over the model info. A test binary of its own: the recorded
//! answers stand in the store of the process.

use std::sync::Arc;

use anyhow::Context as _;
use mock_worker::{
    config::{Config, ReplayConfig, SchedulerWire},
    grpc,
};
use smg_grpc_client::{engine_labels::ServerInfo, SglangSchedulerClient};
use tokio::net::TcpListener;

/// Serve `cfg` as a gRPC worker of the SGLang wire on a free port: its
/// client and the port.
async fn start(cfg: Config) -> anyhow::Result<(SglangSchedulerClient, u16)> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let port = listener.local_addr().context("listener address")?.port();
    let cfg = Config {
        grpc_wire: SchedulerWire::Sglang,
        ..cfg
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(grpc::serve_with_listener(Arc::new(cfg), listener));
    let client = SglangSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .map_err(anyhow::Error::from_boxed)
        .context("connect to the worker")?;
    Ok((client, port))
}

/// What `worker` states: the model path and the tokenizer path of its model
/// info, and the tokenizer path a gateway reads off its server arguments.
async fn stated(worker: &SglangSchedulerClient) -> anyhow::Result<[Option<String>; 3]> {
    let model = worker.get_model_info().await.context("the model info")?;
    let server = worker.get_server_info().await.context("the server info")?;
    let argument = ServerInfo::Sglang(Box::new(server))
        .to_labels()
        .remove("tokenizer_path");
    Ok([Some(model.model_path), Some(model.tokenizer_path), argument])
}

#[tokio::test]
async fn a_replay_states_its_tokenizer_path_in_the_server_arguments_too() -> anyhow::Result<()> {
    let dir = tempfile::tempdir().context("temp dir")?;
    let record = dir.path().join("session.jsonl");
    let engine = "/engine/tokenizer";
    let (_, port) = start(Config {
        model_id: "the-engine".to_string(),
        tokenizer_path: engine.to_string(),
        server_args: [("tokenizer_path".to_string(), engine.to_string())].into(),
        ..Config::default()
    })
    .await?;
    let (recorder, _) = start(Config {
        replay: ReplayConfig {
            record_upstream: Some(format!("grpc://127.0.0.1:{port}")),
            record: Some(record.clone()),
            ..ReplayConfig::default()
        },
        ..Config::default()
    })
    .await?;
    let recorded = ["the-engine", engine, engine].map(|said| Some(said.to_string()));
    assert_eq!(stated(&recorder).await?, recorded, "the engine");

    let replay = |tokenizer: &str| Config {
        model_id: "the-replay".to_string(),
        tokenizer_path: tokenizer.to_string(),
        replay: ReplayConfig {
            script: Some(record.clone()),
            ..ReplayConfig::default()
        },
        ..Config::default()
    };
    // No tokenizer of its own, so the model id stands in its place: the
    // replay states what the engine did.
    let (kept, _) = start(replay("the-replay")).await?;
    assert_eq!(stated(&kept).await?, recorded, "no tokenizer of its own");
    // A config built in process may state no tokenizer path at all: the
    // server arguments stay as recorded.
    let (unset, _) = start(replay("")).await?;
    assert_eq!(
        stated(&unset).await?[2].as_deref(),
        Some(engine),
        "no tokenizer path at all"
    );
    // A tokenizer of its own is the one of this host, in both places; the
    // model path stays the one of the engine.
    let own = "/replay/tokenizer";
    let (moved, _) = start(replay(own)).await?;
    assert_eq!(
        stated(&moved).await?,
        ["the-engine", own, own].map(|said| Some(said.to_string())),
        "a tokenizer of its own"
    );
    Ok(())
}
