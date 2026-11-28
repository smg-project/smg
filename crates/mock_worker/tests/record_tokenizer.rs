//! The tokenizer fetch through a recorder on each wire: a fetch made through
//! it ends as the engine ends it, and the info file of the recorder holds
//! the fetch once, as the engine sent it. A test binary of its own: the
//! answers the engine hands back stand in the store of the process.

use std::sync::Arc;

use anyhow::Context as _;
use mock_worker::{
    config::{Config, ReplayConfig, SchedulerWire},
    grpc,
};
use prost::Message as _;
use smg_grpc_client::{
    common_proto::GetTokenizerChunk, SglangSchedulerClient, TokenSpeedSchedulerClient,
    VllmEngineClient,
};
use tokio::net::TcpListener;

/// The data of the two chunks the engine sends.
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

/// The fetch as an info file holds it, in hex: the chunks one behind the
/// other, each behind its length.
fn fetch() -> anyhow::Result<String> {
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
    Ok(kept.iter().map(|byte| format!("{byte:02x}")).collect())
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

/// The tokenizer fetch of a client.
macro_rules! fetched {
    ($client:ty, $address:expr) => {{
        let client = <$client>::connect($address)
            .await
            .map_err(anyhow::Error::from_boxed)
            .context("connect to the worker")?;
        client
            .get_tokenizer()
            .await
            .map(|bundle| (bundle.compressed_data, bundle.sha256))
            .map_err(|e| e.to_string())
    }};
}

/// What a tokenizer fetch of the worker of `wire` at `address` gives.
async fn fetched(wire: SchedulerWire, address: &str) -> anyhow::Result<Fetched> {
    Ok(match wire {
        SchedulerWire::Vllm => fetched!(VllmEngineClient, address),
        SchedulerWire::TokenSpeed => fetched!(TokenSpeedSchedulerClient, address),
        SchedulerWire::Sglang => fetched!(SglangSchedulerClient, address),
    })
}

/// A wire by its name: what two fetches gave and the fetch lines its info
/// file holds behind them.
type Kept = (&'static str, [Fetched; 2], Vec<(String, String)>);

/// What each wire kept and what is wanted of it: two fetches, one behind the
/// other, through the first and the last of the `workers` gRPC workers of a
/// recorder. The workers of a process serve the same flags: one record path.
async fn two_fetches(workers: usize) -> anyhow::Result<(Vec<Kept>, Vec<Kept>)> {
    let dir = tempfile::tempdir().context("temp dir")?;
    // The engine: a replay whose record holds the fetch of each wire.
    let engine = dir.path().join("engine.jsonl");
    std::fs::write(&engine, "").context("the record file of the engine")?;
    let sent = fetch()?;
    let mut info = String::new();
    for (_, service) in WIRES {
        let line = serde_json::json!({
            "path": format!("{service}/GetTokenizer"),
            "answer_hex": sent,
        });
        info.push_str(&line.to_string());
        info.push('\n');
    }
    std::fs::write(dir.path().join("engine.info.jsonl"), info)
        .context("the info file of the engine")?;

    let mut kept = Vec::new();
    let mut wanted = Vec::new();
    for (wire, service) in WIRES {
        let name = wire.as_str();
        let upstream = start(
            wire,
            Config {
                model_id: "the-engine".to_string(),
                tokenizer_path: "the-engine".to_string(),
                replay: ReplayConfig {
                    script: Some(engine.clone()),
                    ..ReplayConfig::default()
                },
                ..Config::default()
            },
        )
        .await?;
        let record = dir.path().join(format!("{name}.jsonl"));
        let mut recorders = Vec::new();
        for _ in 0..workers {
            let recorder = start(
                wire,
                Config {
                    model_id: "the-recorder".to_string(),
                    replay: ReplayConfig {
                        record_upstream: Some(upstream.replacen("http://", "grpc://", 1)),
                        record: Some(record.clone()),
                        ..ReplayConfig::default()
                    },
                    ..Config::default()
                },
            )
            .await?;
            recorders.push(recorder);
        }
        let first = recorders.first().context("a worker of the recorder")?;
        let last = recorders.last().context("a worker of the recorder")?;
        let fetches = [fetched(wire, first).await?, fetched(wire, last).await?];
        let lines = std::fs::read_to_string(record.with_extension("info.jsonl"))
            .context("the info file of the recorder")?;
        let mut held = Vec::new();
        for line in lines.lines() {
            let line: serde_json::Value =
                serde_json::from_str(line).context("a line of the info file")?;
            let word = |key: &str| {
                line.get(key)
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            if word("path").ends_with("/GetTokenizer") {
                held.push((word("path"), word("answer_hex")));
            }
        }
        kept.push((name, fetches, held));
        let whole: Fetched = Ok((DATA.concat(), SHA256.to_string()));
        wanted.push((
            name,
            [whole.clone(), whole],
            vec![(format!("{service}/GetTokenizer"), sent.clone())],
        ));
    }
    Ok((kept, wanted))
}

#[tokio::test]
async fn a_recorder_keeps_the_tokenizer_fetch_made_through_it_once() -> anyhow::Result<()> {
    let (kept, wanted) = two_fetches(1).await?;
    assert_eq!(
        kept, wanted,
        "each wire: both fetches whole, and one line of the fetch as the engine sent it"
    );
    Ok(())
}

#[tokio::test]
async fn the_workers_of_a_recorder_keep_a_tokenizer_fetch_once() -> anyhow::Result<()> {
    let (kept, wanted) = two_fetches(2).await?;
    assert_eq!(
        kept, wanted,
        "each wire: a fetch whole through each worker, and one line of the fetch between them"
    );
    Ok(())
}
