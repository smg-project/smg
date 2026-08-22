// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): transport.rs.
//
// The frontend (SMG) BINDS all sockets; engines CONNECT in. Scoped to a single
// shared transport with no DP coordinator (coordinator lands with DP>1). The
// reference's `expect`/`panic`/`hex`/`thiserror-ext`/`enum-as-inner` are
// replaced with std/error-handled equivalents to satisfy `clippy -D warnings`.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    ops::Deref,
    time::Duration,
};

use bytes::Bytes;
use tokio::{sync::mpsc, time::timeout};
use tracing::{debug, error, info, trace, warn};
use zeromq::{
    prelude::{Socket, SocketRecv, SocketSend},
    util::PeerIdentity,
    PullSocket, RouterSendHalf, RouterSocket, ZmqError, ZmqMessage,
};

use crate::{
    codec::{decode_msgpack, encode_msgpack},
    error::{Error, Result},
    protocol::{
        handshake::{
            EngineCoreReadyResponse, HandshakeAddresses, HandshakeInitMessage, ReadyMessage,
        },
        EngineBatch, EngineProtocol,
    },
};

/// Single-frame sentinel Python `EngineCoreProc` emits on the output socket when
/// the engine dies.
pub const ENGINE_CORE_DEAD_SENTINEL: &[u8] = b"ENGINE_CORE_DEAD";

/// Opaque routing identity of one engine on the frontend transport.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EngineId(Bytes);

impl Debug for EngineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EngineId(")?;
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        write!(f, ")")
    }
}

impl EngineId {
    /// The engine id as a ZMQ frame.
    pub fn to_frame(&self) -> Bytes {
        self.0.clone()
    }

    /// Consume into the underlying ZMQ frame.
    pub fn into_frame(self) -> Bytes {
        self.0
    }

    /// Parse the Python-compatible engine index (two-byte little-endian ROUTER
    /// identity used by `EngineCoreProc`), or `None` if not two bytes.
    pub fn engine_index(&self) -> Option<u32> {
        if self.len() != 2 {
            return None;
        }
        Some(u16::from_le_bytes([self[0], self[1]]) as u32)
    }

    /// Construct an engine id from the Python-compatible engine index encoding.
    /// The identity is two bytes, so an index above [`u16::MAX`] cannot be
    /// represented — it would alias another rank's identity.
    pub fn from_engine_index(value: u32) -> Self {
        debug_assert!(
            value <= u32::from(u16::MAX),
            "engine index {value} does not fit the 2-byte engine identity"
        );
        Self(Bytes::copy_from_slice(&(value as u16).to_le_bytes()))
    }
}

impl Deref for EngineId {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl From<Vec<u8>> for EngineId {
    fn from(value: Vec<u8>) -> Self {
        Self(Bytes::from(value))
    }
}

impl<const N: usize> From<&[u8; N]> for EngineId {
    fn from(value: &[u8; N]) -> Self {
        Self(Bytes::copy_from_slice(value))
    }
}

impl TryFrom<EngineId> for PeerIdentity {
    type Error = ZmqError;

    fn try_from(value: EngineId) -> std::result::Result<Self, Self::Error> {
        PeerIdentity::try_from(value.into_frame())
    }
}

/// Per-engine handshake result collected while bootstrapping one shared
/// transport.
#[derive(Clone, Debug)]
pub struct ConnectedEngine {
    /// The identity of the connected engine.
    pub engine_id: EngineId,
    /// Post-init configuration received on the input socket registration.
    pub ready_response: EngineCoreReadyResponse,
}

/// The connected shared transport plus all registered engines after a
/// successful startup handshake.
pub struct ConnectedTransport {
    /// Local address of the shared input socket engines connect to for requests.
    pub input_address: String,
    /// Local address of the shared output socket engines connect to for outputs.
    pub output_address: String,
    /// All engines connected through the startup handshake.
    pub engines: Vec<ConnectedEngine>,
    /// The sending half of the shared input ROUTER socket.
    pub input_send: RouterSendHalf,
    /// The shared output PULL socket for receiving all engines' responses.
    pub output_socket: PullSocket,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum EngineStartupState {
    HelloReceived,
    ReadyReceived,
}

impl EngineStartupState {
    fn is_ready_received(&self) -> bool {
        matches!(self, Self::ReadyReceived)
    }
}

fn unexpected_handshake(message: impl Into<String>) -> Error {
    Error::UnexpectedHandshakeMessage {
        message: message.into(),
    }
}

/// Connect to one or more engines through the startup handshake protocol,
/// returning the shared data-plane transport plus the registered engines.
///
/// The frontend binds the shared input/output sockets first, then the handshake
/// socket, drives HELLO -> INIT -> READY per engine, and finally waits for each
/// engine to register (its `EngineCoreReadyResponse`) on the input socket.
pub async fn connect_handshake(
    handshake_address: &str,
    engine_count: usize,
    local_input_address: &str,
    local_output_address: &str,
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    if engine_count == 0 {
        return Err(unexpected_handshake("expected engine_count >= 1"));
    }

    info!(
        engine_count,
        handshake_address, "waiting for engines to connect"
    );

    // 1. Bind shared input/output sockets first so every engine receives the
    //    same data-plane addresses during handshake.
    let (input_address, mut input_socket, output_address, output_socket) =
        bind_local_sockets(local_input_address, local_output_address).await?;
    info!(%input_address, %output_address, "bound local transport sockets");

    // 2. Bind the shared handshake socket. All engines connect here with their
    //    own identities; startup order does not matter.
    let mut handshake_socket = RouterSocket::new();
    handshake_socket.bind(handshake_address).await?;

    let mut engines: BTreeMap<EngineId, EngineStartupState> = BTreeMap::new();

    // 3. Receive HELLO from every engine and send a matching INIT.
    while engines.len() < engine_count {
        let message = timeout(ready_timeout, handshake_socket.recv())
            .await
            .map_err(|_| Error::HandshakeTimeout {
                stage: "HELLO",
                timeout: ready_timeout,
            })??;
        let (engine_id, handshake_message) = decode_handshake_message(message)?;
        match handshake_message.status.as_deref() {
            Some("HELLO") => {
                if engines.contains_key(&engine_id) {
                    return Err(unexpected_handshake(format!(
                        "duplicate engine id {engine_id:?} during startup handshake"
                    )));
                }
                debug!(?engine_id, "received HELLO from engine");
                send_init_message(
                    &mut handshake_socket,
                    &engine_id,
                    &input_address,
                    &output_address,
                )
                .await?;
                engines.insert(engine_id, EngineStartupState::HelloReceived);
            }
            Some("READY") => {
                // READY may overlap the HELLO phase.
                match engines.get_mut(&engine_id) {
                    Some(state) if !state.is_ready_received() => {
                        *state = EngineStartupState::ReadyReceived;
                    }
                    _ => {
                        return Err(unexpected_handshake(format!(
                            "READY for unexpected or duplicate engine id {engine_id:?}"
                        )));
                    }
                }
            }
            other => {
                return Err(unexpected_handshake(format!(
                    "unexpected handshake status {other:?}"
                )))
            }
        }
    }

    // 4. Every engine may now send READY (no coordinator gate in this scope).
    while engines.values().any(|state| !state.is_ready_received()) {
        let message = timeout(ready_timeout, handshake_socket.recv())
            .await
            .map_err(|_| Error::HandshakeTimeout {
                stage: "READY",
                timeout: ready_timeout,
            })??;
        let (engine_id, handshake_message) = decode_handshake_message(message)?;
        match handshake_message.status.as_deref() {
            Some("READY") => match engines.get_mut(&engine_id) {
                Some(state) if !state.is_ready_received() => {
                    debug!(?engine_id, "received READY from engine");
                    *state = EngineStartupState::ReadyReceived;
                }
                _ => {
                    return Err(unexpected_handshake(format!(
                        "READY for unexpected or duplicate engine id {engine_id:?}"
                    )));
                }
            },
            Some("HELLO") => {
                return Err(unexpected_handshake(format!(
                    "duplicate HELLO for engine id {engine_id:?} after INIT phase"
                )));
            }
            other => {
                return Err(unexpected_handshake(format!(
                    "unexpected handshake status {other:?}"
                )))
            }
        }
    }

    // 5. Wait for every engine to register on the shared input socket.
    let engines =
        wait_for_input_registrations(&mut input_socket, engines.into_keys(), ready_timeout).await?;
    info!(engine_count = engines.len(), "engines connected");

    let (input_send, _) = input_socket.split();

    Ok(ConnectedTransport {
        input_address,
        output_address,
        engines,
        input_send,
        output_socket,
    })
}

/// Bind the shared input (ROUTER) and output (PULL) sockets to the given
/// addresses (`ipc://` on the same host, `tcp://` otherwise).
async fn bind_local_sockets(
    local_input_address: &str,
    local_output_address: &str,
) -> Result<(String, RouterSocket, String, PullSocket)> {
    let mut input_socket = RouterSocket::new();
    let input_address = input_socket.bind(local_input_address).await?.to_string();

    let mut output_socket = PullSocket::new();
    let output_address = output_socket.bind(local_output_address).await?.to_string();

    Ok((input_address, input_socket, output_address, output_socket))
}

/// Decode a 2-frame handshake message `[engine_id, ready-payload]`.
fn decode_handshake_message(message: ZmqMessage) -> Result<(EngineId, ReadyMessage)> {
    let frames = message.into_vec();
    let [id_frame, payload] = frames.as_slice() else {
        return Err(unexpected_handshake(format!(
            "expected 2 frames, got {}",
            frames.len()
        )));
    };
    let engine_id = EngineId(id_frame.clone());
    let handshake_message: ReadyMessage = decode_msgpack(payload)?;
    Ok((engine_id, handshake_message))
}

/// Send an INIT message giving the engine the frontend's data-plane addresses.
async fn send_init_message(
    handshake_socket: &mut RouterSocket,
    engine_id: &EngineId,
    input_address: &str,
    output_address: &str,
) -> Result<()> {
    let init_message = HandshakeInitMessage {
        addresses: HandshakeAddresses {
            inputs: vec![input_address.to_string()],
            outputs: vec![output_address.to_string()],
            coordinator_input: None,
            coordinator_output: None,
            frontend_stats_publish_address: None,
        },
        parallel_config: BTreeMap::new(),
    };
    let payload = encode_msgpack(&init_message)?;

    let mut message = ZmqMessage::from(engine_id.to_frame());
    message.push_back(Bytes::from(payload));
    handshake_socket.send(message).await?;
    debug!(?engine_id, "sent INIT to engine");
    Ok(())
}

/// Wait for each engine to connect to the shared input socket and register with
/// `[identity, EngineCoreReadyResponse]`.
async fn wait_for_input_registrations(
    input_socket: &mut RouterSocket,
    expected_engines: impl IntoIterator<Item = EngineId>,
    ready_timeout: Duration,
) -> Result<Vec<ConnectedEngine>> {
    let expected_engines = expected_engines.into_iter().collect::<Vec<_>>();
    let mut pending = expected_engines.iter().cloned().collect::<BTreeSet<_>>();
    let mut ready_responses = BTreeMap::new();

    while !pending.is_empty() {
        let registration = timeout(ready_timeout, input_socket.recv())
            .await
            .map_err(|_| Error::InputRegistrationTimeout {
                timeout: ready_timeout,
            })??;

        let frames = registration.into_vec();
        let [id_frame, payload] = frames.as_slice() else {
            return Err(unexpected_handshake(format!(
                "expected 2 frames for engine input registration, got {}",
                frames.len()
            )));
        };
        let engine_id = EngineId(id_frame.clone());
        if !pending.remove(&engine_id) {
            return Err(unexpected_handshake(format!(
                "input registration for unexpected engine id {engine_id:?}"
            )));
        }
        if payload.is_empty() {
            return Err(unexpected_handshake(format!(
                "empty EngineCoreReadyResponse from engine id {engine_id:?}"
            )));
        }
        let ready_response: EngineCoreReadyResponse = decode_msgpack(payload)?;
        debug!(?engine_id, "received input registration from engine");
        ready_responses.insert(engine_id, ready_response);
    }

    expected_engines
        .into_iter()
        .map(|engine_id| {
            let ready_response = ready_responses.remove(&engine_id).ok_or_else(|| {
                unexpected_handshake(format!(
                    "missing ready response for engine id {engine_id:?}"
                ))
            })?;
            Ok(ConnectedEngine {
                engine_id,
                ready_response,
            })
        })
        .collect()
}

/// Send an encoded request to one engine over the shared input ROUTER socket.
/// Frames: `[engine_id, request_type, payload, aux..]`.
pub async fn send_message(
    input_send: &mut RouterSendHalf,
    engine_id: &EngineId,
    request_type: Bytes,
    payload: Bytes,
    aux_frames: Vec<Bytes>,
) -> Result<()> {
    let mut message = ZmqMessage::from(engine_id.to_frame());
    message.push_back(request_type);
    message.push_back(payload);
    for frame in aux_frames {
        message.push_back(frame);
    }
    trace!(
        ?engine_id,
        frame_count = message.len(),
        "sending ZMQ message"
    );
    input_send.send(message).await?;
    Ok(())
}

/// Receive engine outputs from the shared PULL socket, decode them with the
/// engine's protocol `P`, and forward them on `tx`. Terminates on the
/// `ENGINE_CORE_DEAD` sentinel, a transport error, or when the receiver is
/// dropped. Decode errors are forwarded but do not stop the loop.
pub async fn run_output_loop<P: EngineProtocol>(
    mut output_socket: PullSocket,
    tx: mpsc::Sender<Result<EngineBatch<P::Output>>>,
) {
    loop {
        let message = match output_socket.recv().await {
            Ok(message) => message,
            Err(error) => {
                error!(%error, "failed to receive output message");
                let _ = tx.send(Err(Error::Transport(error))).await;
                return;
            }
        };

        let frames = message.into_vec();
        let Some(frame) = frames.first() else {
            warn!("received empty output message; skipping");
            continue;
        };
        if frame.as_ref() == ENGINE_CORE_DEAD_SENTINEL {
            warn!("received ENGINE_CORE_DEAD sentinel from engine");
            let _ = tx.send(Err(Error::EngineCoreDead)).await;
            return;
        }

        let decoded = match P::decode_batch(&frames) {
            Ok(decoded) => Ok(decoded),
            Err(error) => {
                // Keep the loop running so a single bad message doesn't drop the
                // engine connection.
                warn!(%error, "failed to decode output message");
                Err(error)
            }
        };

        if tx.send(decoded).await.is_err() {
            warn!("output loop receiver dropped; shutting down output loop");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codec::encode_msgpack,
        mock_engine::{connect_to_frontend, default_ready_response, IpcNamespace},
        protocol::vllm::{
            output::{
                EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
            },
            request::{EngineCoreRequest, EngineCoreRequestType},
            VllmProtocol,
        },
    };

    const TIMEOUT: Duration = Duration::from_secs(10);

    #[tokio::test]
    async fn bind_local_sockets_supports_ipc() {
        let ns = IpcNamespace::new().unwrap();
        let (input_address, _in, output_address, _out) =
            bind_local_sockets(&ns.input_endpoint(), &ns.output_endpoint())
                .await
                .unwrap();
        assert!(input_address.starts_with("ipc://"));
        assert!(output_address.starts_with("ipc://"));
    }

    #[tokio::test]
    async fn handshake_then_request_and_output_roundtrip() {
        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );

        // Frontend binds + drives handshake; mock engine (index 0) connects in.
        let (transport, engine) = tokio::join!(
            connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let transport = transport.expect("frontend handshake");
        let mut engine = engine.expect("mock engine connect");

        // Registration surfaced the engine + its post-init config.
        assert_eq!(transport.engines.len(), 1);
        assert_eq!(
            transport.engines[0].engine_id,
            EngineId::from_engine_index(0)
        );
        assert_eq!(
            transport.engines[0].ready_response.vllm_version,
            "test-vllm-version"
        );

        let ConnectedTransport {
            mut input_send,
            output_socket,
            engines,
            ..
        } = transport;
        let engine_id = engines[0].engine_id.clone();

        // Send an Add request; the engine receives [request_type, payload].
        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        };
        let payload = encode_msgpack(&request).unwrap();
        send_message(
            &mut input_send,
            &engine_id,
            EngineCoreRequestType::Add.to_frame(),
            Bytes::from(payload),
            Vec::new(),
        )
        .await
        .unwrap();

        let request_frames = engine.recv_request().await.unwrap();
        assert_eq!(request_frames[0].as_ref(), b"\x00"); // Add
        let received: EngineCoreRequest = decode_msgpack(request_frames[1].as_ref()).unwrap();
        assert_eq!(received.request_id, "req-1");
        assert_eq!(received.prompt_token_ids, Some(vec![1, 2, 3]));

        // Engine pushes a terminal output; the output loop decodes + forwards it.
        let (out_tx, mut out_rx) = mpsc::channel(4);
        #[expect(
            clippy::disallowed_methods,
            reason = "test output loop torn down when the test ends"
        )]
        let _output_loop = tokio::spawn(run_output_loop::<VllmProtocol>(output_socket, out_tx));

        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![100, 101],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..EngineCoreOutput::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..RequestBatchOutputs::default()
        });
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
            .await
            .unwrap();

        let batch = out_rx
            .recv()
            .await
            .expect("output delivered")
            .expect("decoded ok");
        assert_eq!(batch.outputs[0].new_token_ids, vec![100, 101]);
        assert_eq!(
            batch.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn dead_sentinel_surfaces_engine_core_dead() {
        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );

        let (transport, engine) = tokio::join!(
            connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let transport = transport.expect("frontend handshake");
        let mut engine = engine.expect("mock engine connect");

        let (out_tx, mut out_rx) = mpsc::channel(4);
        #[expect(
            clippy::disallowed_methods,
            reason = "test output loop torn down when the test ends"
        )]
        let _output_loop = tokio::spawn(run_output_loop::<VllmProtocol>(
            transport.output_socket,
            out_tx,
        ));

        engine
            .send_output(vec![Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
            .await
            .unwrap();
        let received = out_rx.recv().await.expect("sentinel delivered");
        assert!(matches!(received, Err(Error::EngineCoreDead)));
    }
}
