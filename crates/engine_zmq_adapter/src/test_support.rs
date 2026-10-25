//! Test support shared by the engine modules' tests: one connected client and
//! the mock engine behind it, over a throwaway ipc endpoint triple, and the
//! stream read the typed wires' end-to-end tests share.

use std::{path::Path, time::Duration};

use engine_zmq_client::{
    mock_engine::{connect_to_frontend, default_ready_response, MockEngine},
    protocol::{handshake::EngineCoreReadyResponse, tokenspeed::WireRevision},
    EngineId,
};
use openai_protocol::worker::RuntimeType;
use smg_grpc_client::vllm_proto as vllm;
use tempfile::TempDir;

use crate::{client::ZmqEngineClient, eos::EosTokenIds, stream::ZmqGenerateStream};

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

/// [`connected_pair_declaring_in`] with the connection opened on `wire` (the
/// TokenSpeed dialect's revision the launcher would have read).
pub(crate) async fn connected_pair_declaring_on_wire_in(
    dir: &Path,
    runtime: RuntimeType,
    ready: EngineCoreReadyResponse,
    wire: WireRevision,
) -> (ZmqEngineClient, MockEngine) {
    let (client, engine) = connect_declaring_on_wire_in(dir, runtime, ready, wire).await;
    (
        client.expect("adapter connect"),
        engine.expect("mock engine"),
    )
}

/// [`connected_pair_declaring_on_wire_in`] with the client's verdict left to
/// the caller, so a connect the client refuses after the handshake is read
/// as the error it names; the mock engine's own registration may fail once
/// the refused client closes the transport, so its side comes as an option.
pub(crate) async fn connect_declaring_on_wire_in(
    dir: &Path,
    runtime: RuntimeType,
    ready: EngineCoreReadyResponse,
    wire: WireRevision,
) -> (
    Result<ZmqEngineClient, Box<dyn std::error::Error + Send + Sync>>,
    Option<MockEngine>,
) {
    let ep = |name: &str| format!("ipc://{}", dir.join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect_with_wire(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            runtime,
            Duration::from_secs(10),
            wire,
        ),
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
    );
    (client, engine.ok())
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

/// The stream of a two-token generation with sampled-token logprobs, as the
/// typed wires' end-to-end tests read it: a chunk per token (10 at -0.5, then
/// 11 at -1.25: the finish tick carried a new token, so its delta is emitted
/// as a chunk before the terminal item), then the `Complete` with the
/// cumulative ids and logprobs parallel to `output_ids` (the non-streaming
/// renderer reads only the `Complete`), the `length` finish and two completion
/// tokens, and nothing after it. Returns the first chunk for the wire's own
/// claims on it.
pub(crate) async fn two_logprob_chunks_then_the_complete(
    stream: &mut ZmqGenerateStream,
) -> vllm::GenerateStreamChunk {
    let first = stream.next().await.expect("chunk item").expect("chunk ok");
    let first = match first.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => chunk,
        other => panic!("expected chunk, got {other:?}"),
    };
    assert_eq!(first.token_ids, vec![10]);
    let logprobs = first.output_logprobs.as_ref().expect("chunk logprobs");
    assert_eq!(logprobs.token_logprobs, vec![-0.5]);
    assert_eq!(logprobs.token_ids, vec![10]);

    let second = stream.next().await.expect("chunk item").expect("chunk ok");
    match second.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => {
            assert_eq!(chunk.token_ids, vec![11]);
            let logprobs = chunk.output_logprobs.expect("chunk logprobs");
            assert_eq!(logprobs.token_logprobs, vec![-1.25]);
            assert_eq!(logprobs.token_ids, vec![11]);
        }
        other => panic!("expected chunk, got {other:?}"),
    }

    let third = stream
        .next()
        .await
        .expect("complete item")
        .expect("complete ok");
    match third.response {
        Some(vllm::generate_response::Response::Complete(complete)) => {
            assert_eq!(complete.output_ids, vec![10, 11]);
            assert_eq!(complete.finish_reason, "length");
            assert_eq!(complete.completion_tokens, 2);
            let logprobs = complete.output_logprobs.expect("complete logprobs");
            assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
            assert_eq!(logprobs.token_ids, vec![10, 11]);
        }
        other => panic!("expected complete, got {other:?}"),
    }
    assert!(stream.next().await.is_none());
    first
}

/// [`connected_pair_declaring`] with the connection opened on `wire`.
pub(crate) async fn connected_pair_declaring_on_wire(
    runtime: RuntimeType,
    ready: EngineCoreReadyResponse,
    wire: WireRevision,
) -> (ZmqEngineClient, MockEngine, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let (client, engine) =
        connected_pair_declaring_on_wire_in(dir.path(), runtime, ready, wire).await;
    (client, engine, dir)
}
