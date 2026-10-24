//! Test support shared by the engine modules' tests: one connected client and
//! the mock engine behind it, over a throwaway ipc endpoint triple.

use std::{path::Path, time::Duration};

use engine_zmq_client::{
    mock_engine::{connect_to_frontend, default_ready_response, MockEngine},
    protocol::handshake::EngineCoreReadyResponse,
    EngineId,
};
use openai_protocol::worker::RuntimeType;
use tempfile::TempDir;

use crate::{client::ZmqEngineClient, eos::EosTokenIds};

/// A connected client of `runtime`'s dialect and the mock engine behind it,
/// joined over the three ipc endpoints `hs.sock`, `in.sock` and `out.sock`
/// under `dir`: one engine rank, model `m`, no EOS ids, a ten-second connect
/// timeout. The prologue every end-to-end test of the engine modules opened
/// with.
pub(crate) async fn connected_pair_in(
    dir: &Path,
    runtime: RuntimeType,
) -> (ZmqEngineClient, MockEngine) {
    connected_pair_declaring_in(dir, runtime, default_ready_response()).await
}

/// [`connected_pair_in`] with the mock engine answering the handshake with
/// `ready` (an engine that declares an output layout, a window, ...).
pub(crate) async fn connected_pair_declaring_in(
    dir: &Path,
    runtime: RuntimeType,
    ready: EngineCoreReadyResponse,
) -> (ZmqEngineClient, MockEngine) {
    let ep = |name: &str| format!("ipc://{}", dir.join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            runtime,
            Duration::from_secs(10)
        ),
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
    );
    (
        client.expect("adapter connect"),
        engine.expect("mock engine"),
    )
}

/// [`connected_pair_in`] over a fresh temporary directory, returned so it
/// outlives the sockets for the test's duration.
pub(crate) async fn connected_pair(runtime: RuntimeType) -> (ZmqEngineClient, MockEngine, TempDir) {
    connected_pair_declaring(runtime, default_ready_response()).await
}

/// [`connected_pair`] with the mock engine answering the handshake with `ready`.
pub(crate) async fn connected_pair_declaring(
    runtime: RuntimeType,
    ready: EngineCoreReadyResponse,
) -> (ZmqEngineClient, MockEngine, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let (client, engine) = connected_pair_declaring_in(dir.path(), runtime, ready).await;
    (client, engine, dir)
}
