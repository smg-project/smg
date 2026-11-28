//! The tokenizer fetch of a replay on each wire: the chunks the info file of
//! its record holds, handed back as the engine sent them, unless the replay
//! was started with a tokenizer path of its own. A test binary of its own:
//! the recorded answers stand in the store of the process.

use std::sync::Arc;

use anyhow::Context as _;
use mock_worker::{
    config::{Config, ReplayConfig, SchedulerWire},
    grpc,
};
use prost::Message as _;
use smg_grpc_client::{
    common_proto::GetTokenizerChunk, sglang_proto, tokenspeed_proto, vllm_proto,
    SglangSchedulerClient, TokenSpeedSchedulerClient, VllmEngineClient,
};
use tokio::net::TcpListener;

/// The tokenizer path the recorded engine states.
const ENGINE: &str = "/engine/tokenizer";
/// The data of the two chunks of the recorded fetch.
const DATA: [&[u8]; 2] = [b"the first chunk of a tokenizer, ", b"the last chunk"];
/// The fingerprint of the last chunk: the SHA-256 of the data of both.
const SHA256: &str = "18012ae635f6231e0768caf4067c540382348cec0ff101234f9a90d114c83d56";
/// Each wire with the service its calls are named by.
const WIRES: [(SchedulerWire, &str); 3] = [
    (SchedulerWire::Vllm, "/vllm.grpc.engine.VllmEngine"),
    (
        SchedulerWire::TokenSpeed,
        "/tokenspeed.grpc.scheduler.TokenSpeedScheduler",
    ),
    (
        SchedulerWire::Sglang,
        "/sglang.grpc.scheduler.SglangScheduler",
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

/// The answer of a tokenizer fetch as a recorder keeps it: the chunks one
/// behind the other, each behind its length.
fn fetch() -> anyhow::Result<Vec<u8>> {
    let mut kept = Vec::new();
    for (data, sha256) in [(DATA[0], ""), (DATA[1], SHA256)] {
        let chunk = GetTokenizerChunk {
            data: data.to_vec(),
            sha256: sha256.to_string(),
        };
        chunk
            .encode_length_delimited(&mut kept)
            .context("a chunk behind its length")?;
    }
    Ok(kept)
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

/// What a tokenizer fetch gave: the data of its chunks with the fingerprint,
/// or the failure as its client words it.
type Fetched = Result<(Vec<u8>, String), String>;

/// The tokenizer path of the model info and the tokenizer fetch of a client.
macro_rules! asked {
    ($client:ty, $address:expr) => {{
        let client = <$client>::connect($address)
            .await
            .map_err(anyhow::Error::from_boxed)
            .context("connect to the worker")?;
        let info = client.get_model_info().await.context("the model info")?;
        let fetched: Fetched = client
            .get_tokenizer()
            .await
            .map(|bundle| (bundle.compressed_data, bundle.sha256))
            .map_err(|e| e.to_string());
        (info.tokenizer_path, fetched)
    }};
}

/// The tokenizer path the worker of `wire` at `address` states in its model
/// info, and what its tokenizer fetch gives.
async fn asked(wire: SchedulerWire, address: &str) -> anyhow::Result<(String, Fetched)> {
    Ok(match wire {
        SchedulerWire::Vllm => asked!(VllmEngineClient, address),
        SchedulerWire::TokenSpeed => asked!(TokenSpeedSchedulerClient, address),
        SchedulerWire::Sglang => asked!(SglangSchedulerClient, address),
    })
}

#[tokio::test]
async fn a_replay_hands_back_the_tokenizer_fetch_its_record_holds() -> anyhow::Result<()> {
    let dir = tempfile::tempdir().context("temp dir")?;
    let record = dir.path().join("session.jsonl");
    std::fs::write(&record, "").context("the record file")?;
    let hex = |bytes: &[u8]| -> String { bytes.iter().map(|byte| format!("{byte:02x}")).collect() };
    let mut info = String::new();
    for (wire, service) in WIRES {
        for (call, answer) in [
            ("GetModelInfo", model_info(wire)),
            ("GetTokenizer", fetch()?),
        ] {
            let line = serde_json::json!({
                "path": format!("{service}/{call}"),
                "answer_hex": hex(&answer),
            });
            info.push_str(&line.to_string());
            info.push('\n');
        }
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

    for (wire, _) in WIRES {
        let name = wire.as_str();
        // Started the way the command line does with no `--tokenizer`: the
        // model id stands in the place of the tokenizer path.
        let address = start(wire, replay("the-replay")).await?;
        let (path, fetched) = asked(wire, &address).await?;
        assert_eq!(path, ENGINE, "{name}: the info file is read");
        assert_eq!(
            fetched,
            Ok((DATA.concat(), SHA256.to_string())),
            "{name}: the recorded chunks"
        );
        // Started with a tokenizer path of its own: it states that path and
        // hands back no bundle of another, as a mock with no record.
        let address = start(wire, replay("/replay/tokenizer")).await?;
        let (path, fetched) = asked(wire, &address).await?;
        assert_eq!(path, "/replay/tokenizer", "{name}");
        let refused = fetched.expect_err("no fetch beside a tokenizer path of its own");
        assert!(refused.contains("mock-worker"), "{name}: {refused}");
    }
    Ok(())
}
