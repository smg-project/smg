//! A replay served by a config built in process, which states no tokenizer
//! path at all: on each wire its model info keeps the tokenizer path of the
//! recording. A test binary of its own: the recorded answers stand in the
//! store of the process.

use std::sync::Arc;

use anyhow::Context as _;
use mock_worker::{
    config::{Config, ReplayConfig, SchedulerWire},
    grpc,
};
use prost::Message as _;
use smg_grpc_client::{
    sglang_proto, tokenspeed_proto, vllm_proto, SglangSchedulerClient, TokenSpeedSchedulerClient,
    VllmEngineClient,
};
use tokio::net::TcpListener;

/// The tokenizer path the recorded engine states.
const ENGINE: &str = "/engine/tokenizer";
/// The tokenizer path a replay is started with as one of its own.
const REPLAY: &str = "/replay/tokenizer";
/// Each wire with the path of its model info call.
const WIRES: [(SchedulerWire, &str); 3] = [
    (
        SchedulerWire::Vllm,
        "/vllm.grpc.engine.VllmEngine/GetModelInfo",
    ),
    (
        SchedulerWire::TokenSpeed,
        "/tokenspeed.grpc.scheduler.TokenSpeedScheduler/GetModelInfo",
    ),
    (
        SchedulerWire::Sglang,
        "/sglang.grpc.scheduler.SglangScheduler/GetModelInfo",
    ),
];

/// The model info answer of the recorded engine on `wire`.
fn model_info(wire: SchedulerWire) -> Vec<u8> {
    let tokenizer_path = ENGINE.to_string();
    match wire {
        SchedulerWire::Vllm => vllm_proto::GetModelInfoResponse {
            tokenizer_path,
            ..Default::default()
        }
        .encode_to_vec(),
        SchedulerWire::TokenSpeed => tokenspeed_proto::GetModelInfoResponse {
            tokenizer_path,
            ..Default::default()
        }
        .encode_to_vec(),
        SchedulerWire::Sglang => sglang_proto::GetModelInfoResponse {
            tokenizer_path,
            ..Default::default()
        }
        .encode_to_vec(),
    }
}

/// Serve `cfg` as a gRPC worker of `wire` on a free port: its address.
async fn start(wire: SchedulerWire, cfg: Config) -> anyhow::Result<String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let port = listener.local_addr().context("listener address")?.port();
    let cfg = Config {
        grpc_wire: wire,
        ..cfg
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(grpc::serve_with_listener(Arc::new(cfg), listener));
    Ok(format!("http://127.0.0.1:{port}"))
}

/// The tokenizer path of the model info of a client.
macro_rules! stated {
    ($client:ty, $address:expr) => {{
        let client = <$client>::connect($address)
            .await
            .map_err(anyhow::Error::from_boxed)
            .context("connect to the worker")?;
        let info = client.get_model_info().await.context("the model info")?;
        info.tokenizer_path
    }};
}

/// The tokenizer path the worker of `wire` at `address` states in its model
/// info.
async fn stated(wire: SchedulerWire, address: &str) -> anyhow::Result<String> {
    Ok(match wire {
        SchedulerWire::Vllm => stated!(VllmEngineClient, address),
        SchedulerWire::TokenSpeed => stated!(TokenSpeedSchedulerClient, address),
        SchedulerWire::Sglang => stated!(SglangSchedulerClient, address),
    })
}

#[tokio::test]
async fn a_replay_with_no_tokenizer_path_keeps_the_recorded_one() -> anyhow::Result<()> {
    let dir = tempfile::tempdir().context("temp dir")?;
    let record = dir.path().join("session.jsonl");
    std::fs::write(&record, "").context("the record file")?;
    let hex = |bytes: &[u8]| -> String { bytes.iter().map(|byte| format!("{byte:02x}")).collect() };
    let mut info = String::new();
    for (wire, path) in WIRES {
        let line = serde_json::json!({
            "path": path,
            "answer_hex": hex(&model_info(wire)),
        });
        info.push_str(&line.to_string());
        info.push('\n');
    }
    std::fs::write(dir.path().join("session.info.jsonl"), info).context("the info file")?;
    let replay = |tokenizer: &str| Config {
        model_id: "the-replay".to_string(),
        tokenizer_path: tokenizer.to_string(),
        replay: ReplayConfig {
            script: Some(record.clone()),
            ..ReplayConfig::default()
        },
        ..Config::default()
    };

    // What each wire states for a tokenizer path a replay is started with,
    // the wires in one list so that a failure names every wire.
    let mut states = Vec::new();
    for tokenizer in ["the-replay", "", REPLAY] {
        let mut on_wires = Vec::new();
        for (wire, _) in WIRES {
            let address = start(wire, replay(tokenizer)).await?;
            on_wires.push((wire.as_str(), stated(wire, &address).await?));
        }
        states.push(on_wires);
    }
    let on_every_wire = |path: &str| WIRES.map(|(wire, _)| (wire.as_str(), path.to_string()));
    // Started the way the command line does with no `--tokenizer`: the model
    // id stands in the place of the tokenizer path.
    assert_eq!(states[0], on_every_wire(ENGINE), "no tokenizer of its own");
    // Built in process with no tokenizer path at all: the same answer.
    assert_eq!(states[1], on_every_wire(ENGINE), "no tokenizer path at all");
    // Started with a tokenizer path of its own: that path.
    assert_eq!(states[2], on_every_wire(REPLAY), "a tokenizer of its own");
    Ok(())
}
