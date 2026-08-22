// Test-only mock engine, adapted from the Apache-2.0 reference
// `vllm-engine-core-client` (vllm-project/vllm): mock_engine.rs + test_utils.rs.
//
// Plays the Python `EngineCoreProc` role over ZMQ so the transport and
// connector can be loopback-tested without a GPU: a DEALER completes the
// HELLO/INIT/READY handshake, a DEALER registers on the input socket and
// receives requests, and a PUSH sends outputs back.

use std::time::Duration;

use bytes::Bytes;
use tokio::time::{sleep, timeout};
use zeromq::{
    prelude::{Socket, SocketRecv, SocketSend},
    util::PeerIdentity,
    DealerSocket, PushSocket, SocketOptions, ZmqMessage,
};

use crate::{
    codec::{decode_msgpack, dtype::ModelDtype, encode_msgpack},
    protocol::{
        handshake::{EngineCoreReadyResponse, HandshakeInitMessage, ReadyMessage},
        vllm::{
            output::EngineCoreOutputs,
            request::{EngineCoreRequest, EngineCoreRequestType},
        },
    },
    transport::EngineId,
    Error, Result,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Per-test IPC endpoint namespace backed by a unique temporary directory, so
/// concurrent tests never collide on socket paths.
#[cfg(test)]
pub struct IpcNamespace {
    dir: tempfile::TempDir,
}

#[cfg(test)]
impl IpcNamespace {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            dir: tempfile::TempDir::new()?,
        })
    }

    fn endpoint(&self, name: impl AsRef<std::path::Path>) -> String {
        format!("ipc://{}", self.dir.path().join(name).to_string_lossy())
    }

    pub fn handshake_endpoint(&self) -> String {
        self.endpoint("handshake.sock")
    }

    pub fn input_endpoint(&self) -> String {
        self.endpoint("input.sock")
    }

    pub fn output_endpoint(&self) -> String {
        self.endpoint("output.sock")
    }
}

/// A default post-init ready response for mock engines.
pub fn default_ready_response() -> EngineCoreReadyResponse {
    EngineCoreReadyResponse {
        max_model_len: 1024 * 1024,
        num_gpu_blocks: 0,
        block_size: 16,
        dp_stats_address: None,
        dtype: ModelDtype::Float32,
        vllm_version: "test-vllm-version".to_string(),
        world_size: 1,
        data_parallel_size: 1,
        tensor_parallel_size: 1,
        pipeline_parallel_size: 1,
        decode_context_parallel_size: 1,
        data_parallel_rank: 0,
        max_num_seqs: 256,
        max_num_batched_tokens: 8192,
        instance_id: "test-instance".to_string(),
        kv_cache_size_tokens: None,
        kv_cache_max_concurrency: None,
        kv_events_config: None,
    }
}

/// A typed request received by a mock engine from the frontend.
#[derive(Debug)]
pub enum EngineInbound {
    /// An add-request (`EngineCoreRequestType::Add`). Boxed to keep the enum small.
    Add(Box<EngineCoreRequest>),
    /// An abort for the given request ids (`EngineCoreRequestType::Abort`).
    Abort(Vec<String>),
    /// A lockstep-group wake (`EngineCoreRequestType::StartDpWave`): start this
    /// wave unless this engine is the excluded one (`None` excludes no rank).
    StartDpWave {
        wave: u64,
        exclude_engine_index: Option<u32>,
    },
    /// Any other request type byte (Utility), unhandled here.
    Other(u8),
}

/// The request-receiving half of a mock engine (frontend -> engine).
pub struct MockEngineInput {
    socket: DealerSocket,
}

impl MockEngineInput {
    /// Receive one request's raw frames (`[request_type, payload, aux..]`); the
    /// DEALER identity is already stripped.
    pub async fn recv_frames(&mut self) -> Result<Vec<Bytes>> {
        Ok(self.socket.recv().await?.into_vec())
    }

    /// Receive and classify the next request.
    pub async fn recv(&mut self) -> Result<EngineInbound> {
        let frames = self.recv_frames().await?;
        let Some((type_frame, payload)) = frames.first().zip(frames.get(1)) else {
            return Err(Error::UnexpectedHandshakeMessage {
                message: format!("expected >=2 request frames, got {}", frames.len()),
            });
        };
        match EngineCoreRequestType::from_frame(type_frame) {
            Some(EngineCoreRequestType::Add) => {
                Ok(EngineInbound::Add(Box::new(decode_msgpack(payload)?)))
            }
            Some(EngineCoreRequestType::Abort) => {
                Ok(EngineInbound::Abort(decode_msgpack(payload)?))
            }
            Some(EngineCoreRequestType::StartDpWave) => {
                let (wave, exclude_engine_index) = decode_msgpack(payload)?;
                Ok(EngineInbound::StartDpWave {
                    wave,
                    exclude_engine_index,
                })
            }
            Some(other) => Ok(EngineInbound::Other(other as u8)),
            None => Err(Error::UnexpectedHandshakeMessage {
                message: format!("unknown request type frame {:?}", type_frame.as_ref()),
            }),
        }
    }
}

/// The output-sending half of a mock engine (engine -> frontend).
pub struct MockEngineOutput {
    socket: PushSocket,
}

impl MockEngineOutput {
    /// Push a raw multi-frame output message (frame 0 plus any aux frames).
    pub async fn send_frames(&mut self, frames: Vec<Bytes>) -> Result<()> {
        let mut iter = frames.into_iter();
        let Some(first) = iter.next() else {
            return Err(Error::UnexpectedHandshakeMessage {
                message: "mock engine output needs at least one frame".to_string(),
            });
        };
        let mut message = ZmqMessage::from(first);
        for frame in iter {
            message.push_back(frame);
        }
        self.socket.send(message).await?;
        Ok(())
    }

    /// Encode and push one [`EngineCoreOutputs`] message (text path: single frame).
    pub async fn send_outputs(&mut self, outputs: &EngineCoreOutputs) -> Result<()> {
        self.send_frames(vec![Bytes::from(encode_msgpack(outputs)?)])
            .await
    }
}

/// One mock engine's frontend-facing sockets after a completed handshake.
pub struct MockEngine {
    /// The INIT message the frontend sent during handshake.
    pub init: HandshakeInitMessage,
    input: MockEngineInput,
    output: MockEngineOutput,
}

impl MockEngine {
    /// Split into independently-owned request and output halves (for concurrent
    /// recv/send, e.g. a request loop plus an output writer task).
    pub fn split(self) -> (MockEngineInput, MockEngineOutput) {
        (self.input, self.output)
    }

    /// Receive one request's raw frames. Convenience for sequential test drivers.
    pub async fn recv_request(&mut self) -> Result<Vec<Bytes>> {
        self.input.recv_frames().await
    }

    /// Receive and classify the next request. Convenience for sequential test
    /// drivers that care about the request type rather than the raw frames.
    pub async fn recv(&mut self) -> Result<EngineInbound> {
        self.input.recv().await
    }

    /// Push a raw multi-frame output. Convenience for sequential test drivers.
    pub async fn send_output(&mut self, frames: Vec<Bytes>) -> Result<()> {
        self.output.send_frames(frames).await
    }
}

fn ready_message(status: &str) -> ReadyMessage {
    ReadyMessage {
        status: Some(status.to_string()),
        local: Some(true),
        headless: Some(true),
        parallel_config_hash: None,
    }
}

fn peer_identity(engine_id: &EngineId) -> Result<PeerIdentity> {
    PeerIdentity::try_from(engine_id.clone()).map_err(|error| Error::UnexpectedHandshakeMessage {
        message: format!(
            "invalid mock engine identity {:?}: {error}",
            engine_id.to_vec()
        ),
    })
}

/// Wait for a ZMQ endpoint to become connectable before dialing it.
async fn wait_for_endpoint(endpoint: &str) -> Result<()> {
    let Some(socket_path) = endpoint.strip_prefix("ipc://") else {
        return Ok(());
    };
    timeout(CONNECT_TIMEOUT, async {
        while tokio::net::UnixStream::connect(socket_path).await.is_err() {
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .map_err(|_| Error::HandshakeTimeout {
        stage: "mock engine IPC endpoint",
        timeout: CONNECT_TIMEOUT,
    })
}

/// Connect a mock engine to a frontend-owned handshake endpoint: HELLO → INIT →
/// READY, then register on the input socket with `ready_response` and open the
/// output PUSH socket.
pub async fn connect_to_frontend(
    handshake_address: &str,
    engine_id: impl Into<EngineId>,
    ready_response: EngineCoreReadyResponse,
) -> Result<MockEngine> {
    let engine_id = engine_id.into();
    let identity = peer_identity(&engine_id)?;

    wait_for_endpoint(handshake_address).await?;
    let mut handshake_options = SocketOptions::default();
    handshake_options.peer_identity(identity.clone());
    let mut handshake = DealerSocket::with_options(handshake_options);
    handshake.connect(handshake_address).await?;

    // HELLO -> (INIT) -> READY.
    handshake
        .send(ZmqMessage::from(encode_msgpack(&ready_message("HELLO"))?))
        .await?;
    let init_frames = handshake.recv().await?.into_vec();
    let [init_frame] = init_frames.as_slice() else {
        return Err(Error::UnexpectedHandshakeMessage {
            message: format!("expected one INIT frame, got {}", init_frames.len()),
        });
    };
    let init: HandshakeInitMessage = decode_msgpack(init_frame.as_ref())?;
    handshake
        .send(ZmqMessage::from(encode_msgpack(&ready_message("READY"))?))
        .await?;

    let [input_address] = init.addresses.inputs.as_slice() else {
        return Err(Error::UnexpectedHandshakeMessage {
            message: format!(
                "expected one input address, got {}",
                init.addresses.inputs.len()
            ),
        });
    };
    let [output_address] = init.addresses.outputs.as_slice() else {
        return Err(Error::UnexpectedHandshakeMessage {
            message: format!(
                "expected one output address, got {}",
                init.addresses.outputs.len()
            ),
        });
    };

    // Register on the input socket: [ready_response] with the engine identity.
    wait_for_endpoint(input_address).await?;
    let mut input_options = SocketOptions::default();
    input_options.peer_identity(identity);
    let mut input = DealerSocket::with_options(input_options);
    input.connect(input_address).await?;
    input
        .send(ZmqMessage::from(encode_msgpack(&ready_response)?))
        .await?;

    wait_for_endpoint(output_address).await?;
    let mut output = PushSocket::new();
    output.connect(output_address).await?;

    Ok(MockEngine {
        init,
        input: MockEngineInput { socket: input },
        output: MockEngineOutput { socket: output },
    })
}
