//! What a gRPC worker reports about its model: the sampling defaults of the
//! model directory's `generation_config.json`, as an engine servicer reports
//! them, for the gateway to complete a request's omitted sampling parameters,
//! and the tokenizer path a replay of a recording states on each wire.

use std::{process::Stdio, time::Duration};

use anyhow::Context as _;
use mock_worker::config::{Config, SchedulerWire};
use smg_grpc_client::{SglangSchedulerClient, TokenSpeedSchedulerClient, VllmEngineClient};
use tokio::process::{Child, Command};

mod common;

#[tokio::test]
async fn grpc_model_info_carries_the_models_sampling_defaults() {
    let client = common::start(Config {
        model_id: "a-model".to_string(),
        default_sampling_params_json: r#"{"temperature":0.7,"top_k":20}"#.to_string(),
        ..Config::default()
    })
    .await
    .expect("the worker starts");
    let info = client.get_model_info().await.expect("model info");
    assert_eq!(info.model_path, "a-model");
    assert_eq!(
        info.default_sampling_params_json,
        r#"{"temperature":0.7,"top_k":20}"#
    );
}

#[tokio::test]
async fn grpc_model_info_reports_no_sampling_defaults_when_the_model_declares_none() {
    let client = common::start(Config::default())
        .await
        .expect("the worker starts");
    let info = client.get_model_info().await.expect("model info");
    assert_eq!(info.default_sampling_params_json, "");
}

/// One gRPC worker of `wire` started from the command line with `flags`:
/// the process, killed when it is dropped, and the address it serves on.
fn worker(wire: SchedulerWire, flags: &[&str]) -> anyhow::Result<(Child, String)> {
    let port = common::free_port()?;
    let child = Command::new(env!("CARGO_BIN_EXE_mock-worker"))
        .args(["--grpc-count", "1", "--grpc-wire", wire.as_str()])
        .args(["--grpc-base-port", &port.to_string()])
        .args(flags)
        .stdout(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("run the mock worker")?;
    Ok((child, format!("http://127.0.0.1:{port}")))
}

/// The model path and the tokenizer path the worker of `wire` at `address`
/// states in its model info: `None` while the process takes no connection
/// yet. Once it takes one, a status it answers the call with is the answer.
async fn stated(wire: SchedulerWire, address: &str) -> Option<Result<[String; 2], tonic::Status>> {
    Some(match wire {
        SchedulerWire::Vllm => {
            let client = VllmEngineClient::connect(address).await.ok()?;
            let info = client.get_model_info().await;
            info.map(|info| [info.model_path, info.tokenizer_path])
        }
        SchedulerWire::TokenSpeed => {
            let client = TokenSpeedSchedulerClient::connect(address).await.ok()?;
            let info = client.get_model_info().await;
            info.map(|info| [info.model_path, info.tokenizer_path])
        }
        SchedulerWire::Sglang => {
            let client = SglangSchedulerClient::connect(address).await.ok()?;
            let info = client.get_model_info().await;
            info.map(|info| [info.model_path, info.tokenizer_path])
        }
    })
}

/// The model path and the tokenizer path the worker `child` of `wire` at
/// `address` states in its model info, asked until the process takes a
/// connection. A worker that exited, one that answers the call with a status
/// and one that gives no answer inside ten seconds are errors that say so.
async fn paths(
    wire: SchedulerWire,
    child: &mut Child,
    address: &str,
) -> anyhow::Result<[String; 2]> {
    let name = wire.as_str();
    let wait = async {
        loop {
            if let Some(status) = child.try_wait().context("try_wait")? {
                anyhow::bail!("the {name} worker at {address} exited at its start: {status}");
            }
            if let Some(answer) = stated(wire, address).await {
                return answer.with_context(|| format!("model info of the {name} worker"));
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), wait)
        .await
        .with_context(|| format!("no model info from the {name} worker at {address}"))?
}

/// A recording keeps the tokenizer path its engine stated. On the wire of the
/// recording a replay started without `--tokenizer`, or with one equal to
/// `--model`, states that path; a `--tokenizer` other than `--model` is the
/// path the replay states. Either way the answer is the recorded one: its
/// model path is the engine's, not the replay's `--model`. Every worker is
/// started from the command line.
#[tokio::test]
async fn a_replay_states_the_recorded_tokenizer_path_unless_started_with_one_of_its_own(
) -> anyhow::Result<()> {
    const ENGINE: &str = "/engine-host/tokenizer";
    const REPLAY: &str = "/replay-host/tokenizer";
    let dir = tempfile::tempdir().context("temp dir")?;
    for wire in [
        SchedulerWire::Vllm,
        SchedulerWire::TokenSpeed,
        SchedulerWire::Sglang,
    ] {
        let name = wire.as_str();
        let (mut engine_process, engine) =
            worker(wire, &["--model", "the-engine", "--tokenizer", ENGINE])?;
        assert_eq!(
            paths(wire, &mut engine_process, &engine).await?,
            ["the-engine", ENGINE],
            "{name}"
        );

        let record = dir.path().join(format!("{name}.jsonl"));
        let record = record.to_str().context("a UTF-8 path")?;
        let upstream = engine.replacen("http://", "grpc://", 1);
        let (mut recorder_process, recorder) =
            worker(wire, &["--record-upstream", &upstream, "--record", record])?;
        assert_eq!(
            paths(wire, &mut recorder_process, &recorder).await?,
            ["the-engine", ENGINE],
            "{name}: the recorder hands on what the engine states"
        );

        for (tokenizer, states) in [
            (None, ENGINE),
            (Some("the-replay"), ENGINE),
            (Some(REPLAY), REPLAY),
        ] {
            let mut flags = vec!["--model", "the-replay", "--script", record];
            if let Some(tokenizer) = tokenizer {
                flags.extend(["--tokenizer", tokenizer]);
            }
            let (mut replay_process, replay) = worker(wire, &flags)?;
            assert_eq!(
                paths(wire, &mut replay_process, &replay).await?,
                ["the-engine", states],
                "{name}: a replay started with --tokenizer {tokenizer:?}"
            );
        }
    }
    Ok(())
}
