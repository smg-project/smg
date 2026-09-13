// ZMQ backend adapter (gateway glue): presents the vLLM engine surface (the same
// proto request/response types as `VllmEngineClient`) but speaks ZMQ directly to
// a same-host engine (vLLM EngineCore or TokenSpeed) via `engine-zmq-client`,
// bypassing the gRPC Python servicer.
//
// This bridges the gateway's proto request-execution pipeline to the raw ZMQ
// transport, so it lives with the router (which owns `GrpcClient`/`ProtoStream`),
// not in `smg-grpc-client` (pure gRPC) or `engine-zmq-client` (pure transport).
// It consumes the exact `vllm::GenerateRequest` the existing vLLM builders
// produce and emits `vllm::GenerateResponse` built from `EngineCoreOutput`, so
// the request-execution stage is reused unchanged.

use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
    sync::{Arc, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use engine_zmq_client::{
    codec::dtype::ModelDtype,
    connect_handshake,
    connector::{EngineCoreClient, EngineCoreStream, TokenSpeedClient, TokenSpeedStream},
    protocol::{
        tokenspeed::{
            output::TokenSpeedOutput, request::TokenizedGenerateReqInput,
            sampling::SamplingParams as TokenSpeedSamplingParams,
        },
        vllm::{
            logprobs::TokenLogprob,
            output::{EngineCoreFinishReason, EngineCoreOutput, StopReason},
            request::EngineCoreRequest,
            sampling::EngineCoreSamplingParams,
            structured_outputs::StructuredOutputsParams,
        },
        EngineLoad,
    },
    ConnectedEngine,
};
use futures::{stream::SelectAll, Stream, StreamExt};
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use smg_grpc_client::{tokenspeed_proto, vllm_proto as vllm};

use crate::{
    routers::grpc::{
        client::{ModelInfo, ServerInfo},
        proto_wrapper::ProtoGenerateRequest,
        zmq_multimodal,
    },
    worker::RuntimeType,
};

/// Loopback host for the same-host ZMQ transport (TCP handshake and local
/// binds). Shared with the worker-side socket derivation.
pub(crate) const ZMQ_LOOPBACK_HOST: &str = "127.0.0.1";

/// The engine protocol a ZMQ backend speaks — a closed set: the transport has
/// an adapter for exactly these two engines. Resolved once at connect time and
/// exposed by [`ZmqEngineClient::dialect`] so every per-engine dispatch on the
/// ZMQ lane (request building, multimodal, EOS) matches on the same two
/// variants with no unreachable arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZmqDialect {
    /// vLLM EngineCore.
    Vllm,
    /// TokenSpeed.
    TokenSpeed,
}

/// The connected client for a [`ZmqDialect`]. Both share the transport and
/// handshake; only the request/output struct shapes and the translation to/from
/// SMG proto differ.
#[derive(Clone)]
enum ZmqBackend {
    Vllm(Arc<EngineCoreClient>),
    TokenSpeed(Arc<TokenSpeedClient>),
}

/// Per-connection constants of a [`ZmqEngineClient`], fixed at connect time.
struct ZmqConnectionMeta {
    /// Model id advertised for metadata (the engine does not report it on the
    /// wire; it is configured at worker registration).
    model_id: String,
    /// EOS ids attached to every vLLM request (the engine can't stop at EOS
    /// without them).
    eos: EosTokenIds,
    /// Tokenizer-derived EOS ids, adopted once when `eos` came back empty
    /// (the model id is a repo id, not a local directory). Lives here rather
    /// than on the client so every clone shares the one adoption.
    tokenizer_eos: OnceLock<EosTokenIds>,
}

/// The model's EOS stop set, resolved from its local directory. EngineCore
/// has no tokenizer or model config — stopping at EOS is the frontend's job
/// (the ids ride each request), and without them generation only ends at
/// `max_tokens`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EosTokenIds {
    /// Primary EOS id, carried as the request's `_eos_token_id`.
    primary: Option<u32>,
    /// Extra EOS ids (multi-EOS models), merged into `stop_token_ids`.
    extra: Vec<u32>,
}

impl EosTokenIds {
    pub fn new(primary: Option<u32>, extra: Vec<u32>) -> Self {
        Self { primary, extra }
    }

    /// Build from the tokenizer's merged EOS set (same ordering as
    /// [`Self::from_model_dir`]: config first, generation config after).
    fn from_ids(ids: &[u32]) -> Self {
        let mut ids = ids.iter().copied();
        Self {
            primary: ids.next(),
            extra: ids.collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.primary.is_none() && self.extra.is_empty()
    }

    /// Resolve from `config.json` + `generation_config.json` in a local model
    /// directory: primary = the model config's first id, extras = every other
    /// listed id. Missing files or fields degrade to fewer ids.
    pub async fn from_model_dir(dir: &Path) -> Self {
        let model_ids = eos_ids_from_file(&dir.join("config.json")).await;
        let gen_ids = eos_ids_from_file(&dir.join("generation_config.json")).await;
        let primary = (model_ids.first().or_else(|| gen_ids.first())).copied();
        let mut extra = Vec::new();
        for id in model_ids.into_iter().chain(gen_ids) {
            if Some(id) != primary && !extra.contains(&id) {
                extra.push(id);
            }
        }
        Self { primary, extra }
    }
}

/// Read a config file's `eos_token_id`, which is a single id or a list.
///
/// A missing file is expected (a model ships `config.json`,
/// `generation_config.json`, or both), so read errors stay silent. A file
/// that exists but holds corrupt JSON is worth a `warn!`: it runs once at
/// connect time, and losing the EOS ids here silently manifests later as
/// generation running to `max_tokens`.
async fn eos_ids_from_file(path: &Path) -> Vec<u32> {
    let Ok(text) = tokio::fs::read_to_string(path).await else {
        return Vec::new();
    };
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(config) => eos_ids_from_value(config.get("eos_token_id")),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "failed to parse model config for EOS ids");
            Vec::new()
        }
    }
}

fn eos_ids_from_value(value: Option<&serde_json::Value>) -> Vec<u32> {
    let as_id = |v: &serde_json::Value| v.as_u64().and_then(|id| u32::try_from(id).ok());
    match value {
        Some(serde_json::Value::Array(ids)) => ids.iter().filter_map(as_id).collect(),
        Some(id) => as_id(id).into_iter().collect(),
        None => Vec::new(),
    }
}

/// Request-time EOS backstop for the tokenizer-less EngineCore.
///
/// EOS injection has exactly one owner — this file. The connect-time
/// [`EosTokenIds`] model-dir resolution has nothing to read when the worker's
/// model id is a repo id rather than a local path, so the tokenizer's merged
/// EOS set is folded into `stop_token_ids` here as the always-available
/// backstop; without it an uncapped request generates to the full context
/// window. Not needed for TokenSpeed (its scheduler stops at EOS itself), and
/// a TokenSpeed backend builds a TokenSpeed request variant — so the variant
/// match below is the single dispatch point for this policy.
pub(crate) fn fold_tokenizer_eos_backstop(
    request: &mut ProtoGenerateRequest,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) {
    let ProtoGenerateRequest::Vllm(req) = request else {
        return;
    };
    let Some(params) = req.sampling_params.as_mut() else {
        return;
    };
    if params.ignore_eos {
        return;
    }
    if let Some(tokenizer) = tokenizer {
        for &id in tokenizer.eos_token_ids() {
            if !params.stop_token_ids.contains(&id) {
                params.stop_token_ids.push(id);
            }
        }
    }
}

/// Time to wait for a ZMQ engine to complete the startup handshake. Generous:
/// the engine loads the model and profiles KV cache between INIT and READY.
const ZMQ_CONNECT_TIMEOUT: Duration = Duration::from_secs(600);

/// Derive a deterministic TCP handshake port from the ipc data-plane path.
///
/// vLLM's headless engine dials a *TCP* handshake (`--data-parallel-address` +
/// `--data-parallel-rpc-port`); making the port a pure function of the worker
/// URL lets the operator compute the same `--data-parallel-rpc-port` without a
/// side channel. FNV-1a keeps it stable across processes and builds. Mapped
/// into 20000..=29999 to avoid well-known and typical ephemeral ranges.
///
/// `_zmq_handshake_port` in `bindings/python/src/smg/serve.py` mirrors this
/// function — keep them in sync.
fn derive_handshake_port(path: &str) -> u16 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Map into 20000..=29999: below the Linux default ephemeral range
    // (`net.ipv4.ip_local_port_range` = 32768..60999) so an outbound socket
    // can't already hold the port. `hash % 10000` always fits u16.
    20000 + (hash % 10000) as u16
}

/// Derive the ZMQ socket addresses for a worker from its base URL.
///
/// Mirrors vLLM's headless topology: the **handshake is TCP** (the engine dials
/// it, so it matches `vllm serve --headless --data-parallel-rpc-port`), while
/// the **data plane is `ipc://`** for the same-host fast path (SMG chooses these
/// and hands them to the engine during the handshake INIT). The operator gives a
/// single `ipc://<path>` base; SMG binds the ipc input/output at
/// `<path>-in.sock` / `-out.sock` and derives the TCP handshake port from the
/// path. A `WorkerSpec.zmq_handshake_address` override replaces the derived
/// handshake address verbatim (it must be `tcp://`), for engines that dial a
/// fixed, pre-agreed address — e.g. TokenSpeed's default dial target is
/// `tcp://127.0.0.1:30500` (its `--data-parallel-address`/
/// `--data-parallel-rpc-port` defaults, outside the derived 20000..=29999
/// band), so setting the override to that value pairs a bare
/// `ts serve --headless` with a manually registered worker.
/// Returns `(handshake, input, output)`.
///
/// [`zmq_handshake_address`] exposes just the handshake half, for the
/// registration-time validation of that address.
fn zmq_socket_addresses(
    base_url: &str,
    handshake_override: Option<&str>,
) -> Result<(String, String, String), String> {
    let path = base_url
        .strip_prefix("ipc://")
        .ok_or_else(|| format!("ZMQ worker URL must be ipc://<path>, got '{base_url}'"))?;
    let handshake = match handshake_override {
        Some(address) => {
            if !address.starts_with("tcp://") {
                return Err(format!(
                    "zmq_handshake_address must be a tcp:// address \
                     (the engine dials a TCP handshake), got '{address}'"
                ));
            }
            address.to_string()
        }
        None => format!("tcp://{ZMQ_LOOPBACK_HOST}:{}", derive_handshake_port(path)),
    };
    let input = format!("ipc://{path}-in.sock");
    let output = format!("ipc://{path}-out.sock");
    Ok((handshake, input, output))
}

/// The TCP handshake address a ZMQ worker will bind.
///
/// Carries [`zmq_socket_addresses`]'s error verbatim rather than flattening it
/// to "no address": an unusable `ipc://` base or a non-`tcp://` override is a
/// misconfiguration registration must reject, not a value to skip past and
/// rediscover at every later connect attempt.
pub(crate) fn zmq_handshake_address(
    base_url: &str,
    handshake_override: Option<&str>,
) -> Result<String, String> {
    zmq_socket_addresses(base_url, handshake_override).map(|(handshake, _, _)| handshake)
}

/// Create the parent directory for a worker's `ipc://` sockets. Kept off the
/// address computation (which is pure) and async so it doesn't block a runtime
/// thread.
///
/// The ipc:// data-plane sockets SMG binds here carry no authentication, so the
/// directory must be owner-controlled: when this call creates it, it is created
/// 0700 (mode applied at mkdir time — no chmod window); when it already exists,
/// its permissions are left untouched (never chmod a shared dir like `/tmp`)
/// and it is rejected unless it is a real directory owned by the current user.
async fn ensure_ipc_socket_dir(base_url: &str) -> Result<(), String> {
    let path = base_url.strip_prefix("ipc://").unwrap_or(base_url);
    let Some(parent) = Path::new(path).parent() else {
        return Ok(());
    };
    // symlink_metadata: a symlinked parent must not redirect the checks (or the
    // sockets) into a directory we did not verify.
    let meta = match tokio::fs::symlink_metadata(parent).await {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = tokio::fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            builder.mode(0o700);
            builder
                .create(parent)
                .await
                .map_err(|e| format!("failed to create ipc socket dir for {path}: {e}"))?;
            tokio::fs::symlink_metadata(parent)
                .await
                .map_err(|e| format!("failed to stat ipc socket dir for {path}: {e}"))?
        }
        Err(e) => return Err(format!("failed to stat ipc socket dir for {path}: {e}")),
    };
    if !meta.is_dir() {
        return Err(format!(
            "ipc socket dir {} exists but is not a directory",
            parent.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let uid = rustix::process::geteuid().as_raw();
        if meta.uid() != uid {
            return Err(format!(
                "ipc socket dir {} is owned by uid {} (expected {uid}); refusing to bind \
                 unauthenticated ZMQ sockets in a directory owned by another user",
                parent.display(),
                meta.uid()
            ));
        }
    }
    Ok(())
}

/// Remove a stale ipc socket file left by a previous gateway process, if any.
/// Only ever unlinks sockets (never a regular file at the path), inside the
/// owner-verified socket dir.
#[cfg(unix)]
async fn unlink_stale_socket(address: &str) -> Result<(), String> {
    let Some(path) = address.strip_prefix("ipc://") else {
        return Ok(());
    };
    match tokio::fs::symlink_metadata(path).await {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("failed to stat ipc socket path {path}: {e}")),
        Ok(meta) => {
            use std::os::unix::fs::FileTypeExt;
            if !meta.file_type().is_socket() {
                return Err(format!(
                    "ipc socket path {path} exists but is not a socket; refusing to unlink"
                ));
            }
            tracing::info!("Removing stale ipc socket {path} from a previous gateway run");
            tokio::fs::remove_file(path)
                .await
                .map_err(|e| format!("failed to remove stale ipc socket {path}: {e}"))
        }
    }
}

/// Bind the SMG-side ZMQ sockets and complete the handshake with the
/// engine(s): the single connect path for a worker's `ipc://` URL, driven only
/// by the worker's background handshake driver. `model_id` is
/// the config-resolved served model (EngineCore reports none). `engine_count`
/// is the number of DP engines that will dial this worker's sockets (1 for an
/// ungrouped worker). Errors are plain reasons; the worker layer wraps them in
/// its own error type.
pub(crate) async fn connect_for_worker(
    base_url: &str,
    model_id: String,
    runtime: RuntimeType,
    handshake_override: Option<&str>,
    engine_count: usize,
) -> Result<ZmqEngineClient, String> {
    let (handshake, input, output) = zmq_socket_addresses(base_url, handshake_override)?;
    ensure_ipc_socket_dir(base_url).await?;
    // ZMQ refuses to bind over an existing ipc socket file, so leftovers from
    // a dead gateway would fail every reconnect with a bare transport error.
    // The dir is verified owner-only above, and a live gateway can't leave
    // these behind (each worker URL is bound by at most one process), so any
    // existing socket file here is stale by construction.
    #[cfg(unix)]
    {
        unlink_stale_socket(&input).await?;
        unlink_stale_socket(&output).await?;
    }
    // The engine can't stop at EOS on its own (it has no tokenizer or model
    // config); resolve the EOS ids from the local model dir so every request
    // carries them.
    let model_dir = Path::new(&model_id);
    let is_model_dir = tokio::fs::metadata(model_dir)
        .await
        .is_ok_and(|meta| meta.is_dir());
    let eos = if is_model_dir {
        EosTokenIds::from_model_dir(model_dir).await
    } else {
        tracing::warn!(
            "ZMQ worker model id '{model_id}' is not a local model directory; connect-time \
             EOS ids unavailable — relying on the tokenizer's EOS set, folded into stop \
             tokens at request time"
        );
        EosTokenIds::default()
    };
    tracing::info!(
        "Binding ZMQ client for worker {base_url} (handshake={handshake}, engines={engine_count})"
    );
    ZmqEngineClient::connect(
        &handshake,
        &input,
        &output,
        engine_count,
        model_id,
        eos,
        runtime,
        ZMQ_CONNECT_TIMEOUT,
    )
    .await
    .map_err(|e| format!("Failed to connect ZMQ engine: {e}"))
}

/// Direct ZMQ connection to a same-host engine (vLLM EngineCore or TokenSpeed),
/// presented behind the vLLM gRPC client surface.
#[derive(Clone)]
pub struct ZmqEngineClient {
    backend: ZmqBackend,
    /// Connection-constant metadata, shared so cloning the client (once per
    /// request, via `BackendClient`) stays a pointer bump.
    meta: Arc<ZmqConnectionMeta>,
}

impl ZmqEngineClient {
    /// Bind the frontend sockets and complete the handshake with the engine(s),
    /// which must already be running and dialing `handshake_address`.
    ///
    /// `input_address`/`output_address` are the `ipc://` data-plane endpoints the
    /// engines connect to (chosen by SMG). `engine_count` is the number of DP
    /// ranks to await. `runtime` selects the wire protocol spoken over the shared
    /// transport (vLLM EngineCore vs TokenSpeed).
    #[expect(
        clippy::too_many_arguments,
        reason = "transport constructor: endpoints, engine count, and runtime are all irreducible connection inputs"
    )]
    pub async fn connect(
        handshake_address: &str,
        input_address: &str,
        output_address: &str,
        engine_count: usize,
        model_id: String,
        eos: EosTokenIds,
        runtime: RuntimeType,
        timeout: Duration,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Resolve the dialect before the handshake: no silent fallback for a
        // runtime with no ZMQ engine adapter, and no such engine ever dials in,
        // so the handshake would just block for the full timeout.
        let dialect = match runtime {
            // vLLM EngineCore is the default ZMQ wire; an unspecified runtime
            // maps to it for backward compatibility (see `detect_backend`).
            RuntimeType::Vllm | RuntimeType::Unspecified => ZmqDialect::Vllm,
            RuntimeType::TokenSpeed => ZmqDialect::TokenSpeed,
            other => {
                return Err(format!(
                    "ZMQ direct backend has no engine implementation for runtime \
                     {other}; only vllm and tokenspeed are supported"
                )
                .into())
            }
        };

        let transport = connect_handshake(
            handshake_address,
            engine_count,
            input_address,
            output_address,
            timeout,
        )
        .await?;
        let backend = match dialect {
            ZmqDialect::Vllm => ZmqBackend::Vllm(Arc::new(EngineCoreClient::new(transport))),
            ZmqDialect::TokenSpeed => {
                ZmqBackend::TokenSpeed(Arc::new(TokenSpeedClient::new(transport)))
            }
        };
        Ok(Self {
            backend,
            meta: Arc::new(ZmqConnectionMeta {
                model_id,
                eos,
                tokenizer_eos: OnceLock::new(),
            }),
        })
    }

    /// Adopt the tokenizer's EOS ids when the connect-time model-dir lookup
    /// found none (the worker's model id is a repo id, not a local path).
    ///
    /// Without this the primary EOS id would only ride `stop_token_ids`, and
    /// an EOS finish would be reported as `matched_stop = <eos id>` — so the
    /// same model would answer differently depending on whether its files
    /// happen to be local.
    pub(crate) fn adopt_tokenizer_eos(&self, tokenizer: Option<&Arc<dyn Tokenizer>>) {
        if !self.meta.eos.is_empty() || self.meta.tokenizer_eos.get().is_some() {
            return;
        }
        let Some(ids) = tokenizer
            .map(|t| t.eos_token_ids())
            .filter(|ids| !ids.is_empty())
        else {
            return;
        };
        let _ = self.meta.tokenizer_eos.set(EosTokenIds::from_ids(ids));
    }

    /// The EOS set attached to requests: the connect-time set, or the adopted
    /// tokenizer set when that one was empty.
    fn effective_eos(&self) -> &EosTokenIds {
        self.meta.tokenizer_eos.get().unwrap_or(&self.meta.eos)
    }

    /// The wire protocol chosen at connect time.
    pub fn dialect(&self) -> ZmqDialect {
        match &self.backend {
            ZmqBackend::Vllm(_) => ZmqDialect::Vllm,
            ZmqBackend::TokenSpeed(_) => ZmqDialect::TokenSpeed,
        }
    }

    /// The engine runtime behind this connection, widened to the open
    /// [`RuntimeType`] for callers that report it alongside gRPC backends.
    pub fn runtime(&self) -> RuntimeType {
        match self.dialect() {
            ZmqDialect::Vllm => RuntimeType::Vllm,
            ZmqDialect::TokenSpeed => RuntimeType::TokenSpeed,
        }
    }

    /// The engines connected on the shared transport (same handshake for both
    /// protocols).
    fn engines(&self) -> &[ConnectedEngine] {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.engines(),
            ZmqBackend::TokenSpeed(client) => client.engines(),
        }
    }

    /// Submit a generate request and return a stream of vLLM-proto responses.
    /// The request is the engine's native proto (vLLM for a vLLM backend,
    /// TokenSpeed for a TokenSpeed backend — the [`BackendClient`] builders emit
    /// the matching variant per runtime); it is translated into the backend's
    /// wire protocol here.
    ///
    /// Over gRPC the engine-side frontend (e.g. vLLM's AsyncLLM) fans `n` out
    /// itself and multiplexes the choices onto one stream. The raw ZMQ wire has
    /// no such frontend, so `n > 1` is fanned out HERE into `n` independent
    /// single-sample engine requests (see [`fan_out_requests`]); their outputs
    /// are merged back into one stream with each sub tagged via the proto
    /// `index` field, exactly like the gRPC contract.
    ///
    /// [`BackendClient`]: crate::routers::grpc::backend_client::BackendClient
    pub async fn generate(
        &self,
        req: ProtoGenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        // Sub-streams submitted before a mid-loop failure are dropped with the
        // error, which auto-aborts their engine-side requests.
        match &self.backend {
            ZmqBackend::Vllm(client) => {
                let ProtoGenerateRequest::Vllm(req) = req else {
                    return Err(tonic::Status::internal(
                        "vLLM ZMQ backend expects a vLLM generate request",
                    ));
                };
                // EngineCore needs a concrete `max_tokens`; vLLM's OpenAI frontend
                // (which the ZMQ path bypasses) defaults an unset value to
                // `max_model_len - prompt_len`. The context length comes from the
                // engine's ready handshake, so a connected engine is required.
                let (max_model_len, model_dtype) = client
                    .engines()
                    .first()
                    .map(|e| (e.ready_response.max_model_len, e.ready_response.dtype))
                    .ok_or_else(|| tonic::Status::unavailable("no connected ZMQ engine"))?;
                let mut streams = SelectAll::new();
                for (index, sub) in fan_out_requests(*req).into_iter().enumerate() {
                    let request =
                        translate_request(sub, max_model_len, model_dtype, self.effective_eos())
                            .map_err(tonic::Status::invalid_argument)?;
                    // The engine returns the sampled/prompt token's logprob
                    // plus the requested ranked candidates per position; carry
                    // the counts so the stream can shape both `top_logprobs`
                    // lists. The first prompt token is reported with a `null`
                    // logprob (nothing precedes it to condition on).
                    let sampling = request.sampling_params.as_ref();
                    let top_logprobs = ranked_candidate_count(sampling.and_then(|sp| sp.logprobs));
                    let prompt_top_logprobs =
                        ranked_candidate_count(sampling.and_then(|sp| sp.prompt_logprobs));
                    let first_prompt_token = request
                        .prompt_token_ids
                        .as_ref()
                        .and_then(|ids| ids.first().copied());
                    let stream = client.submit(request).await.map_err(zmq_status)?;
                    streams.push(VllmGenerateStream::new(
                        stream,
                        index as u32,
                        top_logprobs,
                        prompt_top_logprobs,
                        first_prompt_token,
                    ));
                }
                Ok(ZmqGenerateStream::Vllm(streams))
            }
            ZmqBackend::TokenSpeed(client) => {
                let ProtoGenerateRequest::TokenSpeed(req) = req else {
                    return Err(tonic::Status::internal(
                        "TokenSpeed ZMQ backend expects a TokenSpeed generate request",
                    ));
                };
                let mut streams = SelectAll::new();
                for (index, sub) in fan_out_tokenspeed_requests(*req).into_iter().enumerate() {
                    let request = translate_request_tokenspeed(sub)
                        .map_err(tonic::Status::invalid_argument)?;
                    let stream = client.submit(request).await.map_err(zmq_status)?;
                    streams.push(TokenSpeedGenerateStream::new(stream, index as u32));
                }
                Ok(ZmqGenerateStream::TokenSpeed(streams))
            }
        }
    }

    /// Local liveness: false once the connection observed `ENGINE_CORE_DEAD` or
    /// a transport failure. No RPC (the raw ZMQ wire has no health RPC).
    pub fn is_alive(&self) -> bool {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.is_alive(),
            ZmqBackend::TokenSpeed(client) => client.is_alive(),
        }
    }

    /// Health as an RPC-shaped response, derived from local liveness.
    pub fn health_check(&self) -> vllm::HealthCheckResponse {
        let alive = self.is_alive();
        vllm::HealthCheckResponse {
            healthy: alive,
            message: if alive {
                "ok".to_string()
            } else {
                "engine core dead".to_string()
            },
        }
    }

    /// Latest per-rank load for one engine index, if the backend carries it.
    /// vLLM piggybacks it on every batch; TokenSpeed does not (always `None`).
    fn engine_load(&self, engine_index: u32) -> Option<EngineLoad> {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.engine_load(engine_index),
            ZmqBackend::TokenSpeed(client) => client.engine_load(engine_index),
        }
    }

    /// Per-rank load from the piggybacked `scheduler_stats` (SMG's DP routing
    /// signal), in the same shape as the gRPC `GetLoads` response. TokenSpeed
    /// carries no piggybacked load, so its response has no per-rank entries.
    pub fn get_loads(&self) -> WorkerLoadResponse {
        let loads: Vec<SchedulerLoadSnapshot> = self
            .engines()
            .iter()
            .filter_map(|engine| {
                let dp_rank = engine.engine_id.engine_index()?;
                let load = self.engine_load(dp_rank)?;
                Some(SchedulerLoadSnapshot {
                    dp_rank: i32::try_from(dp_rank).unwrap_or(i32::MAX),
                    num_running_reqs: i32::try_from(load.num_running).unwrap_or(i32::MAX),
                    num_waiting_reqs: i32::try_from(load.num_waiting).unwrap_or(i32::MAX),
                    token_usage: load.kv_cache_usage,
                    ..Default::default()
                })
            })
            .collect();
        WorkerLoadResponse {
            timestamp: String::new(),
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
            ..Default::default()
        }
    }

    /// Model info derived from the handshake `EngineCoreReadyResponse` plus the
    /// configured model id (the engine does not report tokenizer/vocab metadata,
    /// so those come from worker config). Returned as the runtime's native
    /// metadata variant so the label mapping matches the gRPC path.
    pub fn get_model_info(&self) -> ModelInfo {
        let max_context_length = self
            .engines()
            .first()
            .map(|e| e.ready_response.max_model_len)
            .unwrap_or(0);
        match &self.backend {
            ZmqBackend::Vllm(_) => ModelInfo::Vllm(vllm::GetModelInfoResponse {
                model_path: self.meta.model_id.clone(),
                served_model_name: self.meta.model_id.clone(),
                tokenizer_path: self.meta.model_id.clone(),
                is_generation: true,
                max_context_length: u32::try_from(max_context_length).unwrap_or(u32::MAX),
                ..Default::default()
            }),
            ZmqBackend::TokenSpeed(_) => {
                ModelInfo::TokenSpeed(Box::new(tokenspeed_proto::GetModelInfoResponse {
                    model_path: self.meta.model_id.clone(),
                    served_model_name: self.meta.model_id.clone(),
                    tokenizer_path: self.meta.model_id.clone(),
                    max_context_length: i32::try_from(max_context_length).unwrap_or(i32::MAX),
                    ..Default::default()
                }))
            }
        }
    }

    /// Server info derived from the handshake response, as the runtime's native
    /// metadata variant.
    pub fn get_server_info(&self) -> ServerInfo {
        let data_parallel_size = self
            .engines()
            .first()
            .map(|e| e.ready_response.data_parallel_size)
            .unwrap_or(1);
        match &self.backend {
            ZmqBackend::Vllm(_) => ServerInfo::Vllm(vllm::GetServerInfoResponse {
                data_parallel_size: i32::try_from(data_parallel_size).unwrap_or(i32::MAX),
                server_type: "vllm".to_string(),
                ..Default::default()
            }),
            // TokenSpeed's server-info proto carries no data-parallel size or
            // server-type field; the ZMQ handshake supplies no `server_args`
            // either, so only the fields it does expose are surfaced.
            ZmqBackend::TokenSpeed(_) => {
                ServerInfo::TokenSpeed(Box::<tokenspeed_proto::GetServerInfoResponse>::default())
            }
        }
    }
}

/// Streaming generate output over ZMQ, presented as vLLM-proto
/// `GenerateResponse`. One sub-stream per fanned-out engine request (n>1);
/// they are polled together and yield as ready (interleaved), each tagging its
/// responses with its choice `index`. The merged stream ends when every sub
/// has delivered its terminal `Complete`. Dropping it before that aborts all
/// still-running engine-side sub-requests, so no explicit abort or
/// `mark_completed` is required. Which variant is active is fixed by the
/// backend protocol chosen at connect time.
pub enum ZmqGenerateStream {
    /// vLLM EngineCore outputs.
    Vllm(SelectAll<VllmGenerateStream>),
    /// TokenSpeed outputs.
    TokenSpeed(SelectAll<TokenSpeedGenerateStream>),
}

impl ZmqGenerateStream {
    /// Next vLLM-proto response, or `None` when the stream ends.
    pub async fn next(&mut self) -> Option<Result<vllm::GenerateResponse, tonic::Status>> {
        match self {
            Self::Vllm(streams) => streams.next().await,
            Self::TokenSpeed(streams) => streams.next().await,
        }
    }

    /// No-op: the ZMQ stream aborts natively on drop, so there is nothing to
    /// mark. Present for parity with the tonic abort-on-drop streams.
    #[expect(
        clippy::unused_self,
        reason = "kept for API parity with the tonic abort-on-drop streams"
    )]
    pub fn mark_completed(&mut self) {}
}

/// Ranked candidates to emit per position for a requested logprob count:
/// unset/`0` means the sampled (or prompt) token's own logprob only, and `-1`
/// means every candidate the engine returned.
fn ranked_candidate_count(requested: Option<i32>) -> usize {
    match requested {
        Some(n) if n < 0 => usize::MAX,
        Some(n) => n as usize,
        None => 0,
    }
}

/// Shape one position's wire entries (sampled first, then the engine's ranked
/// candidates) into `top_logprobs`: the sampled entry leads, then ranked
/// candidates fill up to `top_k` entries. The engine leaves the sampled token
/// in its ranked columns, so the ranked entry repeating it is skipped —
/// otherwise the list would carry it twice and drop the last candidate.
fn shape_top_logprobs(entries: &[TokenLogprob], top_k: usize) -> vllm::TopLogProbs {
    let mut top = vllm::TopLogProbs::default();
    let Some((sampled, ranked)) = entries.split_first() else {
        return top;
    };
    if top_k == 0 {
        return top;
    }
    top.values.push(sampled.logprob);
    top.token_ids.push(sampled.token_id);
    for entry in ranked {
        if top.token_ids.len() >= top_k {
            break;
        }
        if entry.token_id == sampled.token_id {
            continue;
        }
        top.values.push(entry.logprob);
        top.token_ids.push(entry.token_id);
    }
    top
}

/// Accumulated per-request token counts shared by both stream mappers.
#[derive(Default)]
struct StreamState {
    output_ids: Vec<u32>,
    completion_tokens: u32,
    prompt_tokens: u32,
    cached_tokens: u32,
    /// Cumulative sampled-token logprobs across ticks. The proto contract is
    /// incremental per streaming chunk but cumulative on the terminal
    /// `Complete`, so accumulate here and drain into `Complete`.
    output_logprobs_val: Vec<f32>,
    output_logprobs_idx: Vec<u32>,
    /// Prompt (input) logprobs, accumulated across chunked-prefill ticks.
    prompt_logprobs: Vec<vllm::InputTokenLogProb>,
    prompt_token_ids: Vec<u32>,
    prompt_top_logprobs: Vec<vllm::TopLogProbs>,
    /// Cumulative per-position ranked candidates (`top_logprobs`), accumulated
    /// alongside the sampled logprobs and drained into the terminal `Complete`.
    output_top_logprobs: Vec<vllm::TopLogProbs>,
}

impl StreamState {
    /// Cumulative sampled-token logprobs for the terminal `Complete`, drained
    /// from the accumulated state (`None` when logprobs were not requested).
    fn take_complete_logprobs(&mut self) -> Option<vllm::OutputLogProbs> {
        (!self.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
            token_logprobs: std::mem::take(&mut self.output_logprobs_val),
            token_ids: std::mem::take(&mut self.output_logprobs_idx),
            top_logprobs: std::mem::take(&mut self.output_top_logprobs),
        })
    }
    /// Emit one engine tick as vLLM-proto responses. On a finish tick the
    /// `Complete` (with the engine-specific finish reason and matched stop) is
    /// returned directly — unless the tick also carried new tokens, in which
    /// case a `Chunk` goes out first and the `Complete` is parked in `pending`
    /// for the next poll. Non-finish ticks emit a plain `Chunk`.
    fn emit_tick(
        &mut self,
        index: u32,
        token_ids: Vec<u32>,
        chunk_logprobs: Option<vllm::OutputLogProbs>,
        finish: Option<(String, Option<vllm::generate_complete::MatchedStop>)>,
        pending: &mut Option<vllm::GenerateResponse>,
    ) -> vllm::GenerateResponse {
        let chunk = |state: &Self, token_ids, chunk_logprobs| {
            vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                token_ids,
                prompt_tokens: state.prompt_tokens,
                completion_tokens: state.completion_tokens,
                cached_tokens: state.cached_tokens,
                output_logprobs: chunk_logprobs,
                index,
                ..Default::default()
            })
        };
        let response = match finish {
            Some((finish_reason, matched_stop)) => {
                let complete = vllm::GenerateResponse {
                    response: Some(vllm::generate_response::Response::Complete(
                        vllm::GenerateComplete {
                            output_ids: std::mem::take(&mut self.output_ids),
                            finish_reason,
                            prompt_tokens: self.prompt_tokens,
                            completion_tokens: self.completion_tokens,
                            cached_tokens: self.cached_tokens,
                            matched_stop,
                            output_logprobs: self.take_complete_logprobs(),
                            index,
                            ..Default::default()
                        },
                    )),
                };
                if token_ids.is_empty() {
                    return complete;
                }
                let chunk = chunk(self, token_ids, chunk_logprobs);
                *pending = Some(complete);
                chunk
            }
            None => chunk(self, token_ids, chunk_logprobs),
        };
        vllm::GenerateResponse {
            response: Some(response),
        }
    }
}

/// The per-dialect half of a ZMQ generate stream: the wire stream, the parked
/// terminal `Complete`, and the mapping from one wire output to a vLLM-proto
/// response. [`poll_mapped`] writes the `Stream` machinery once for every
/// dialect that implements this.
trait MappedGenerateStream {
    /// One tick of engine output on this dialect's wire.
    type Output;
    /// The wire stream carrying those ticks.
    type Inner: Stream<Item = Result<Self::Output, engine_zmq_client::Error>> + Unpin;

    fn inner(&mut self) -> &mut Self::Inner;

    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens; yielded before the wire stream is polled again.
    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse>;

    fn map_output(&mut self, output: Self::Output)
        -> Result<vllm::GenerateResponse, tonic::Status>;
}

/// `Stream::poll_next` for any [`MappedGenerateStream`]: drain the parked
/// `Complete` first, otherwise poll the wire and map the tick.
fn poll_mapped<S: MappedGenerateStream>(
    stream: &mut S,
    cx: &mut std::task::Context<'_>,
) -> std::task::Poll<Option<Result<vllm::GenerateResponse, tonic::Status>>> {
    use std::task::Poll;
    if let Some(pending) = stream.pending().take() {
        return Poll::Ready(Some(Ok(pending)));
    }
    match std::pin::Pin::new(stream.inner()).poll_next(cx) {
        Poll::Ready(Some(Ok(output))) => Poll::Ready(Some(stream.map_output(output))),
        Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(zmq_status(error)))),
        Poll::Ready(None) => Poll::Ready(None),
        Poll::Pending => Poll::Pending,
    }
}

/// Streaming generate output for one vLLM EngineCore sub-request, mapping each
/// `EngineCoreOutput` to a vLLM-proto `GenerateResponse` (chunks until the
/// terminal output, then a complete), tagged with this sub's choice `index`.
pub struct VllmGenerateStream {
    inner: EngineCoreStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1) — the proto field the pipeline demuxes choices by.
    index: u32,
    /// Number of ranked candidates the client requested per position; `0` when
    /// only the sampled logprob (or nothing) was asked for, in which case no
    /// `top_logprobs` are emitted.
    top_logprobs: usize,
    /// Ranked candidates per PROMPT position (`prompt_logprobs`); `0` off.
    prompt_top_logprobs: usize,
    /// First prompt token id; reported with a `null` logprob per the API
    /// contract (nothing precedes it to condition on).
    first_prompt_token: Option<u32>,
    /// Prompt logprobs are attached to the first emitted chunk exactly once.
    input_logprobs_emitted: bool,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: streaming frontends decode text/logprobs from chunks only, so
    /// the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl VllmGenerateStream {
    fn new(
        inner: EngineCoreStream,
        index: u32,
        top_logprobs: usize,
        prompt_top_logprobs: usize,
        first_prompt_token: Option<u32>,
    ) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            top_logprobs,
            prompt_top_logprobs,
            first_prompt_token,
            input_logprobs_emitted: false,
            pending: None,
        }
    }

    /// Attach the accumulated prompt logprobs: once on the first token-bearing
    /// chunk (the proto puts them in the first chunk only) and on every
    /// `Complete`, including one parked in `pending`. Prefill precedes the
    /// first sampled token, so the set is whole by the time a chunk carries
    /// tokens.
    fn attach_input_logprobs(&mut self, response: &mut vllm::GenerateResponse) {
        if self.state.prompt_logprobs.is_empty() {
            return;
        }
        // Built per attachment site rather than up front: with
        // `prompt_logprobs` requested this runs on every decode tick, and the
        // common tick (a later chunk) attaches nothing.
        let state = &self.state;
        let build = || vllm::InputLogProbs {
            token_logprobs: state.prompt_logprobs.clone(),
            token_ids: state.prompt_token_ids.clone(),
            top_logprobs: state.prompt_top_logprobs.clone(),
        };
        if let Some(vllm::generate_response::Response::Complete(parked)) = self
            .pending
            .as_mut()
            .and_then(|pending| pending.response.as_mut())
        {
            parked.input_logprobs = Some(build());
        }
        match response.response.as_mut() {
            Some(vllm::generate_response::Response::Chunk(chunk))
                if !self.input_logprobs_emitted && !chunk.token_ids.is_empty() =>
            {
                chunk.input_logprobs = Some(build());
                self.input_logprobs_emitted = true;
            }
            // Later chunks never repeat them (the proto carries them in the
            // first chunk only), and neither do token-less prefill chunks.
            Some(vllm::generate_response::Response::Chunk(_)) => {}
            Some(vllm::generate_response::Response::Complete(complete)) => {
                complete.input_logprobs = Some(build());
            }
            None => {}
        }
    }
}

impl MappedGenerateStream for VllmGenerateStream {
    type Output = EngineCoreOutput;
    type Inner = EngineCoreStream;

    fn inner(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }

    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse> {
        &mut self.pending
    }

    fn map_output(
        &mut self,
        output: EngineCoreOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        let top_k = self.top_logprobs;
        let state = &mut self.state;
        if let Some(stats) = &output.prefill_stats {
            state.prompt_tokens = stats.num_prompt_tokens;
            state.cached_tokens = stats.num_cached_tokens;
        }
        let token_ids = output.new_token_ids;
        state.completion_tokens += token_ids.len() as u32;
        state.output_ids.extend(token_ids.iter().copied());

        // Sampled-token logprobs (entry 0 per position) plus the requested
        // ranked candidates (`top_logprobs`). Chunks carry this tick's
        // increment; the terminal `Complete` carries the cumulative set, so
        // accumulate into `state` and drain it on finish.
        let mut tick_logprobs_val = Vec::new();
        let mut tick_logprobs_idx = Vec::new();
        let mut tick_top_logprobs = Vec::new();
        if let Some(decoded) = &output.new_logprobs {
            for position in &decoded.positions {
                let Some(sampled) = position.entries.first() else {
                    continue;
                };
                tick_logprobs_val.push(sampled.logprob);
                tick_logprobs_idx.push(sampled.token_id);
                // The entries arrive sampled-first then rank-ordered; shape the
                // requested count so one ranked list lands per sampled token.
                if top_k > 0 {
                    tick_top_logprobs.push(shape_top_logprobs(&position.entries, top_k));
                }
            }
        }
        let chunk_logprobs = (!tick_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
            token_logprobs: tick_logprobs_val.clone(),
            token_ids: tick_logprobs_idx.clone(),
            top_logprobs: tick_top_logprobs.clone(),
        });
        state.output_logprobs_val.extend(tick_logprobs_val);
        state.output_logprobs_idx.extend(tick_logprobs_idx);
        state.output_top_logprobs.extend(tick_top_logprobs);

        // Prompt logprobs accumulate the same way (chunked prefill delivers
        // them incrementally); entry 0 per position is the actual prompt
        // token. The API contract reports the first prompt token with a null
        // logprob, so seed it once before the first scored position.
        if let Some(decoded) = &output.new_prompt_logprobs_tensors {
            if state.prompt_logprobs.is_empty() && !decoded.positions.is_empty() {
                if let Some(first) = self.first_prompt_token {
                    state
                        .prompt_logprobs
                        .push(vllm::InputTokenLogProb::default());
                    state.prompt_token_ids.push(first);
                    if self.prompt_top_logprobs > 0 {
                        state.prompt_top_logprobs.push(vllm::TopLogProbs::default());
                    }
                }
            }
            for position in &decoded.positions {
                let Some(selected) = position.entries.first() else {
                    continue;
                };
                state.prompt_logprobs.push(vllm::InputTokenLogProb {
                    value: Some(selected.logprob),
                });
                state.prompt_token_ids.push(selected.token_id);
                if self.prompt_top_logprobs > 0 {
                    state.prompt_top_logprobs.push(shape_top_logprobs(
                        &position.entries,
                        self.prompt_top_logprobs,
                    ));
                }
            }
        }

        // An engine-side request failure (e.g. grammar compilation) must
        // surface as an error, not as a normal completion with empty output —
        // that would produce a 200 with no content.
        if matches!(output.finish_reason, Some(EngineCoreFinishReason::Error)) {
            return Err(tonic::Status::internal(
                "engine finished the request with an error (see engine logs)",
            ));
        }
        let finish = output.finish_reason.map(|reason| {
            (
                finish_reason_str(reason).to_string(),
                output.stop_reason.map(map_matched_stop),
            )
        });
        let mut response = state.emit_tick(
            self.index,
            token_ids,
            chunk_logprobs,
            finish,
            &mut self.pending,
        );
        self.attach_input_logprobs(&mut response);
        Ok(response)
    }
}

impl Stream for VllmGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// Streaming generate output for one TokenSpeed sub-request, mapping each
/// `TokenSpeedOutput` to a vLLM-proto `GenerateResponse`, tagged with this
/// sub's choice `index`.
pub struct TokenSpeedGenerateStream {
    inner: TokenSpeedStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1) — the proto field the pipeline demuxes choices by.
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: streaming frontends decode text/logprobs from chunks only, so
    /// the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl TokenSpeedGenerateStream {
    fn new(inner: TokenSpeedStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
        }
    }
}

impl MappedGenerateStream for TokenSpeedGenerateStream {
    type Output = TokenSpeedOutput;
    type Inner = TokenSpeedStream;

    fn inner(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }

    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse> {
        &mut self.pending
    }

    fn map_output(
        &mut self,
        output: TokenSpeedOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        let state = &mut self.state;
        // TokenSpeed reports per-request token counts directly (cumulative for
        // completions), rather than vLLM's per-output prefill-stats deltas.
        if output.prompt_tokens > 0 {
            state.prompt_tokens = output.prompt_tokens;
        }
        if output.cached_tokens > 0 {
            state.cached_tokens = output.cached_tokens;
        }
        state.completion_tokens = output.completion_tokens;
        state.output_ids.extend(output.output_ids.iter().copied());

        // Sampled-token logprobs, if requested. The proto column is `float`, so
        // downcast the wire's `f64` values. Chunks carry this tick's increment;
        // the terminal `Complete` carries the cumulative set, so accumulate
        // into `state` and drain it on finish.
        let chunk_logprobs =
            (!output.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
                token_logprobs: output
                    .output_logprobs_val
                    .iter()
                    .map(|&lp| lp as f32)
                    .collect(),
                token_ids: output.output_logprobs_idx.clone(),
                ..Default::default()
            });
        state
            .output_logprobs_val
            .extend(output.output_logprobs_val.iter().map(|&lp| lp as f32));
        state
            .output_logprobs_idx
            .extend(output.output_logprobs_idx.iter().copied());

        // An engine-side failure must surface as an error, not a normal
        // completion with empty output — mirroring the vLLM stream's guard.
        if output.finish_reason.as_deref() == Some("error") {
            return Err(tonic::Status::internal(
                "engine finished the request with an error (see engine logs)",
            ));
        }
        // No matched_stop on this wire: TokenSpeed reports the finish reason
        // only, and the router-side stop machinery owns string matching.
        let finish = output
            .finish_reason
            .map(|reason| (normalize_finish_reason(&reason).to_string(), None));
        Ok(state.emit_tick(
            self.index,
            output.output_ids,
            chunk_logprobs,
            finish,
            &mut self.pending,
        ))
    }
}

impl Stream for TokenSpeedGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// Split an `n > 1` generate request into `n` independent single-sample wire
/// requests (an `n <= 1` request passes through untouched).
///
/// - Rids: sub `i` is `"{request_id}-{i}"` — engine-side rids must be unique,
///   and the pipeline's request id is already unique per request, so the
///   suffixed forms are too. Dropping the merged stream aborts every sub.
/// - Seeds: with no explicit seed each sub keeps `None` — the engine seeds each
///   rid independently, so the samples differ. An explicit seed becomes
///   `seed + i` per sub: a fixed seed with identical params would otherwise
///   make all n samples identical, and deriving distinct per-sample seeds from
///   the request seed is the established engine convention (each of the n
///   sequences gets its own sampler state for exactly this reason) while
///   staying deterministic for repeat runs.
/// - Usage: every sub reports the full `prompt_tokens` on its `Complete` (the
///   subs share one prompt), matching the gRPC engines' n>1 contract — the
///   pipeline de-duplicates (max per prompt), so nothing is counted n times.
fn fan_out_requests(req: vllm::GenerateRequest) -> Vec<vllm::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(sp) = sub.sampling_params.as_mut() {
            sp.n = 1;
            // An explicit seed must still yield distinct samples per sub.
            sp.seed = sp.seed.map(|seed| seed.wrapping_add(i as i32));
        }
    })
}

/// Shared n>1 fan-out scaffolding: clone the request into `n` subs and let
/// `per_sub` apply the engine-specific rid suffix and sampling tweaks. An
/// `n <= 1` request passes through untouched. The last sub reuses `req`
/// itself, so a multimodal payload is copied `n - 1` times, not `n`.
fn fan_out_n<R: Clone>(mut req: R, n: u32, mut per_sub: impl FnMut(&mut R, u32)) -> Vec<R> {
    if n <= 1 {
        return vec![req];
    }
    let mut subs = Vec::with_capacity(n as usize);
    for i in 0..n - 1 {
        let mut sub = req.clone();
        per_sub(&mut sub, i);
        subs.push(sub);
    }
    per_sub(&mut req, n - 1);
    subs.push(req);
    subs
}

/// Split an `n > 1` TokenSpeed proto request into `n` single-sample
/// sub-requests, the TokenSpeed analogue of [`fan_out_requests`] (the wire has
/// no per-sample demux, so `generate` fans out here). An `n <= 1` request passes
/// through untouched. Seed handling matches [`fan_out_requests`]: omission
/// delegates independent seed assignment to the engine, while explicit seed
/// `s` derives deterministic per-sample seeds `s + i`.
fn fan_out_tokenspeed_requests(
    req: tokenspeed_proto::GenerateRequest,
) -> Vec<tokenspeed_proto::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(sp) = sub.sampling_params.as_mut() {
            sp.n = 1;
            sp.sampling_seed = sp.sampling_seed.map(|seed| seed.wrapping_add(u64::from(i)));
        }
    })
}

/// Translate a TokenSpeed proto `GenerateRequest` into the wire
/// `TokenizedGenerateReqInput`. ZMQ mode requires pre-tokenized input (SMG
/// tokenizes upstream).
fn translate_request_tokenspeed(
    req: tokenspeed_proto::GenerateRequest,
) -> Result<TokenizedGenerateReqInput, String> {
    // The TokenSpeed ZMQ wire has no multimodal slot yet; reject loudly rather
    // than silently dropping pixels (assembly also refuses upstream).
    if req.mm_inputs.is_some() {
        return Err(
            "multimodal inputs are not supported over the TokenSpeed ZMQ backend".to_string(),
        );
    }
    let input_ids = match req.tokenized {
        Some(tokenized) => tokenized.input_ids,
        None => {
            return Err("ZMQ mode requires pre-tokenized input; no input provided".to_string());
        }
    };
    // Over the ZMQ wire TokenSpeed returns only the single sampled-token logprob
    // per token: no top-k candidates (`top_logprobs_num > 1`) and no prompt
    // logprobs (`token_ids_logprob`). Reject both rather than silently return
    // fewer than asked. A bare `logprobs: true` (count 0/1) is the plain
    // sampled-token logprob and is wired end-to-end via `return_logprob`.
    if req.top_logprobs_num > 1 {
        return Err("top_logprobs are not supported over the TokenSpeed ZMQ backend".to_string());
    }
    if !req.token_ids_logprob.is_empty() {
        return Err(
            "prompt logprobs are not supported over the TokenSpeed ZMQ backend".to_string(),
        );
    }
    Ok(TokenizedGenerateReqInput {
        rid: req.request_id,
        input_ids,
        sampling_params: req
            .sampling_params
            .map(translate_sampling_tokenspeed)
            .unwrap_or_else(|| {
                let mut params = TokenSpeedSamplingParams::default();
                params.normalize();
                params
            }),
        return_logprob: req.return_logprob,
        stream: req.stream,
        // Every other field keeps its neutral default (the fields after
        // `stream` are not even emitted; the engine fills them from defaults).
        ..TokenizedGenerateReqInput::default()
    })
}

/// Map TokenSpeed proto sampling params onto the wire `SamplingParams`, in the
/// normalized form: the engine skips its decode-time re-derivation once
/// `is_normalized` is set, so [`TokenSpeedSamplingParams::normalize`] resolves
/// the derived fields (top_k sentinel, greedy collapse) before encoding.
///
/// String `stop` sequences are not forwarded — the token-only engine cannot
/// match them; the router-side stop decoder trims them from the text instead.
fn translate_sampling_tokenspeed(sp: tokenspeed_proto::SamplingParams) -> TokenSpeedSamplingParams {
    let mut params = TokenSpeedSamplingParams {
        max_new_tokens: sp.max_new_tokens,
        stop_token_ids: (!sp.stop_token_ids.is_empty()).then_some(sp.stop_token_ids),
        temperature: f64::from(sp.temperature.unwrap_or(1.0)),
        top_p: f64::from(sp.top_p.unwrap_or(1.0)),
        // The proto keeps the API convention `-1` = "all tokens" (and unset);
        // `normalize` resolves it to the engine's disabled sentinel.
        top_k: sp.top_k.unwrap_or(-1),
        min_p: f64::from(sp.min_p.unwrap_or(0.0)),
        frequency_penalty: f64::from(sp.frequency_penalty.unwrap_or(0.0)),
        presence_penalty: f64::from(sp.presence_penalty.unwrap_or(0.0)),
        repetition_penalty: f64::from(sp.repetition_penalty.unwrap_or(1.0)),
        min_new_tokens: sp.min_new_tokens,
        ignore_eos: sp.ignore_eos,
        skip_special_tokens: sp.skip_special_tokens,
        spaces_between_special_tokens: sp.spaces_between_special_tokens,
        no_stop_trim: sp.no_stop_trim,
        seed: sp.sampling_seed,
        // Proto `0` means unspecified; TokenSpeed expects at least one sample.
        // n>1 is fanned out before translation, so this is always 1 on the wire.
        n: sp.n.max(1),
        ..TokenSpeedSamplingParams::default()
    };
    apply_tokenspeed_constraint(&mut params, sp.constraint);
    params.normalize();
    params
}

/// Map the proto structured-output `constraint` oneof onto the wire's dedicated
/// fields. The oneof is single-valued, so at most one field is set; the rest
/// stay `None`.
fn apply_tokenspeed_constraint(
    params: &mut TokenSpeedSamplingParams,
    constraint: Option<tokenspeed_proto::sampling_params::Constraint>,
) {
    use tokenspeed_proto::sampling_params::Constraint;
    match constraint {
        Some(Constraint::JsonSchema(schema)) => params.json_schema = Some(schema),
        Some(Constraint::Regex(regex)) => params.regex = Some(regex),
        Some(Constraint::EbnfGrammar(grammar)) => params.ebnf = Some(grammar),
        Some(Constraint::StructuralTag(tag)) => params.structural_tag = Some(tag),
        None => {}
    }
}

/// Translate a vLLM-proto generate request into an `EngineCoreRequest`. ZMQ mode
/// requires pre-tokenized input (SMG tokenizes upstream).
fn translate_request(
    req: vllm::GenerateRequest,
    max_model_len: u64,
    model_dtype: ModelDtype,
    eos: &EosTokenIds,
) -> Result<EngineCoreRequest, String> {
    let prompt_token_ids = match req.input {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) => Some(tokenized.input_ids),
        Some(vllm::generate_request::Input::Text(_)) => {
            return Err("ZMQ mode requires pre-tokenized input (TokenizedInput)".to_string());
        }
        None => {
            return Err("ZMQ mode requires pre-tokenized input; no input provided".to_string());
        }
    };
    // Per-item mm features: the split the Python servicer performs before the
    // engine happens here instead (the ZMQ path bypasses it).
    let mm_features = req
        .mm_inputs
        .map(|mm| {
            zmq_multimodal::build_mm_features(
                mm,
                prompt_token_ids.as_deref().unwrap_or(&[]),
                model_dtype,
            )
        })
        .transpose()?
        .filter(|features| !features.is_empty());
    let data_parallel_rank = req
        .data_parallel_rank
        .map(|rank| u32::try_from(rank).map_err(|_| format!("invalid data_parallel_rank: {rank}")))
        .transpose()?;
    // vLLM's frontend defaults an unset `max_tokens` to the remaining context
    // (`max_model_len - prompt_len`).
    let prompt_len = prompt_token_ids.as_ref().map_or(0, |ids| ids.len()) as u64;
    let default_max_tokens =
        u32::try_from(max_model_len.saturating_sub(prompt_len)).unwrap_or(u32::MAX);
    Ok(EngineCoreRequest {
        request_id: req.request_id,
        prompt_token_ids,
        mm_features,
        sampling_params: req
            .sampling_params
            .map(|sp| translate_sampling(sp, default_max_tokens, eos)),
        arrival_time: now_secs(),
        data_parallel_rank,
        ..EngineCoreRequest::default()
    })
}

fn translate_sampling(
    sp: vllm::SamplingParams,
    default_max_tokens: u32,
    eos: &EosTokenIds,
) -> EngineCoreSamplingParams {
    // Stopping at EOS is the frontend's duty here: the primary id rides
    // `_eos_token_id`, extra ids merge into `stop_token_ids`, and the union
    // feeds `_all_stop_token_ids` (engine-side `min_tokens` masking, built
    // regardless of `ignore_eos`).
    let mut stop_token_ids = sp.stop_token_ids;
    if !sp.ignore_eos {
        for id in &eos.extra {
            if !stop_token_ids.contains(id) {
                stop_token_ids.push(*id);
            }
        }
    }
    let mut all_stop_token_ids: BTreeSet<u32> = stop_token_ids.iter().copied().collect();
    all_stop_token_ids.extend(eos.primary);
    all_stop_token_ids.extend(eos.extra.iter().copied());
    let logit_bias = if sp.logit_bias.is_empty() {
        None
    } else {
        Some(
            sp.logit_bias
                .into_iter()
                .filter_map(|(token, bias)| match u32::try_from(token) {
                    Ok(t) => Some((t, bias)),
                    Err(_) => {
                        // Don't fold negatives onto key 0 (which would silently
                        // drop all but the last); skip them with a warning.
                        tracing::warn!("dropping negative logit_bias token id {token}");
                        None
                    }
                })
                .collect::<HashMap<_, _>>(),
        )
    };
    EngineCoreSamplingParams {
        temperature: sp.temperature.unwrap_or(1.0),
        top_p: sp.top_p,
        top_k: sp.top_k,
        min_p: sp.min_p,
        frequency_penalty: sp.frequency_penalty,
        presence_penalty: sp.presence_penalty,
        repetition_penalty: sp.repetition_penalty,
        max_tokens: sp.max_tokens.unwrap_or(default_max_tokens),
        min_tokens: sp.min_tokens,
        stop_token_ids,
        eos_token_id: (!sp.ignore_eos).then_some(eos.primary).flatten(),
        all_stop_token_ids,
        seed: sp.seed.map(i64::from),
        logprobs: sp.logprobs,
        prompt_logprobs: sp.prompt_logprobs,
        logit_bias,
        structured_outputs: sp.constraint.and_then(translate_constraint),
        ..EngineCoreSamplingParams::default()
    }
}

/// Map the proto `constraint` oneof onto typed structured-output params. The
/// backend defaults to guidance engine-side; `json_object=false` selects no
/// constraint (the caller opted out), so it maps to `None`.
fn translate_constraint(
    constraint: vllm::sampling_params::Constraint,
) -> Option<StructuredOutputsParams> {
    use vllm::sampling_params::Constraint;
    match constraint {
        Constraint::JsonSchema(schema) => Some(StructuredOutputsParams::json(
            // The engine accepts a JSON schema object or a schema string; parse
            // to preserve object shape, falling back to the raw string.
            serde_json::from_str(&schema).unwrap_or(serde_json::Value::String(schema)),
        )),
        Constraint::Regex(regex) => Some(StructuredOutputsParams::regex(regex)),
        Constraint::Grammar(grammar) => Some(StructuredOutputsParams::grammar(grammar)),
        Constraint::StructuralTag(tag) => Some(StructuredOutputsParams::structural_tag(tag)),
        Constraint::JsonObject(true) => Some(StructuredOutputsParams::json_object()),
        Constraint::JsonObject(false) => None,
        Constraint::Choice(choice) => Some(StructuredOutputsParams::choice(choice.choices)),
    }
}

fn map_matched_stop(reason: StopReason) -> vllm::generate_complete::MatchedStop {
    match reason {
        StopReason::TokenId(id) => vllm::generate_complete::MatchedStop::MatchedTokenId(id),
        StopReason::Text(text) => vllm::generate_complete::MatchedStop::MatchedStopStr(text),
    }
}

fn finish_reason_str(reason: EngineCoreFinishReason) -> &'static str {
    match reason {
        EngineCoreFinishReason::Stop | EngineCoreFinishReason::Repetition => "stop",
        EngineCoreFinishReason::Length => "length",
        EngineCoreFinishReason::Abort => "abort",
        EngineCoreFinishReason::Error => "error",
    }
}

/// Normalize a TokenSpeed wire finish-reason string into the canonical set the
/// gateway's response layer exact-matches (`stop`, `length`, `abort`, `error`) —
/// the same set the vLLM path emits via [`finish_reason_str`]. TokenSpeed emits
/// `stop`/`length`/`abort`; an unknown value falls back to `stop` with a warning
/// so a non-canonical string never mis-renders downstream.
fn normalize_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" => "length",
        "abort" => "abort",
        "error" => "error",
        other => {
            tracing::warn!(
                finish_reason = other,
                "unknown TokenSpeed finish_reason; defaulting to \"stop\""
            );
            "stop"
        }
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn zmq_status(error: engine_zmq_client::Error) -> tonic::Status {
    match error {
        engine_zmq_client::Error::EngineCoreDead => tonic::Status::unavailable(error.to_string()),
        other => tonic::Status::internal(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response, EngineInbound},
        protocol::vllm::{
            logprobs::{Logprobs, PositionLogprobs, TokenLogprob},
            output::{EngineCoreOutputs, RequestBatchOutputs},
        },
        EngineId,
    };
    use llm_tokenizer::mock::MockTokenizer;

    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn unlink_stale_socket_removes_sockets_and_refuses_files() {
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("stale.sock");
        // A bound-then-dropped listener leaves the socket file behind, exactly
        // like a dead gateway does.
        drop(UnixListener::bind(&sock_path).unwrap());
        assert!(sock_path.exists());
        let addr = format!("ipc://{}", sock_path.display());
        unlink_stale_socket(&addr).await.unwrap();
        assert!(!sock_path.exists(), "stale socket must be removed");

        // Missing file: fine.
        unlink_stale_socket(&addr).await.unwrap();

        // A regular file at the path is not ours to delete.
        std::fs::write(&sock_path, b"not a socket").unwrap();
        let err = unlink_stale_socket(&addr).await.unwrap_err();
        assert!(err.contains("not a socket"), "{err}");
        assert!(sock_path.exists(), "regular file must survive");
    }

    #[test]
    fn derive_handshake_port_matches_pinned_vectors() {
        // Fixed vectors shared with `_zmq_handshake_port` in
        // bindings/python/src/smg/serve.py — a change on either side breaks the
        // engine/router port agreement, so these must stay in sync.
        assert_eq!(derive_handshake_port("/tmp/smg-zmq/ts0.ipc"), 25152);
        assert_eq!(derive_handshake_port("/tmp/smg-zmq/engine-31000"), 22714);
        // Range invariant: every path maps into 20000..=29999.
        for p in ["", "a", "/x/y/z.ipc", "very/long/path/with/segments.sock"] {
            let port = derive_handshake_port(p);
            assert!(
                (20000..=29999).contains(&port),
                "port {port} out of band for {p:?}"
            );
        }
    }

    #[test]
    fn zmq_socket_addresses_derive_handshake_by_default() {
        let (handshake, input, output) =
            zmq_socket_addresses("ipc:///tmp/smg-zmq/ts0.ipc", None).unwrap();
        assert_eq!(handshake, "tcp://127.0.0.1:25152");
        assert_eq!(input, "ipc:///tmp/smg-zmq/ts0.ipc-in.sock");
        assert_eq!(output, "ipc:///tmp/smg-zmq/ts0.ipc-out.sock");
    }

    #[test]
    fn zmq_socket_addresses_honor_handshake_override() {
        // TokenSpeed's default dial target — outside the derived band; the
        // override must be bound verbatim while the data plane stays derived.
        let (handshake, input, output) =
            zmq_socket_addresses("ipc:///tmp/smg-zmq/ts0.ipc", Some("tcp://127.0.0.1:30500"))
                .unwrap();
        assert_eq!(handshake, "tcp://127.0.0.1:30500");
        assert_eq!(input, "ipc:///tmp/smg-zmq/ts0.ipc-in.sock");
        assert_eq!(output, "ipc:///tmp/smg-zmq/ts0.ipc-out.sock");
    }

    #[test]
    fn zmq_socket_addresses_reject_non_tcp_override() {
        // The engine dials a TCP handshake; a non-tcp override is a config
        // error and must fail loudly rather than bind something unexpected.
        let err = zmq_socket_addresses("ipc:///tmp/smg-zmq/ts0.ipc", Some("ipc:///tmp/hs.sock"))
            .unwrap_err();
        assert!(
            err.contains("tcp://"),
            "error must name the required scheme: {err}"
        );
    }

    #[tokio::test]
    async fn ensure_ipc_socket_dir_creates_a_private_owner_only_dir() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("sockets");
        let url = format!("ipc://{}/x.ipc", dir.display());
        ensure_ipc_socket_dir(&url).await.unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "created socket dir must be 0700");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ensure_ipc_socket_dir_leaves_an_existing_owned_dir_untouched() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        let url = format!("ipc://{}/x.ipc", dir.path().display());
        ensure_ipc_socket_dir(&url).await.unwrap();
        let mode = std::fs::metadata(dir.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "an existing dir must not be chmod'd");
    }

    #[tokio::test]
    async fn ensure_ipc_socket_dir_rejects_a_non_directory_parent() {
        let base = tempfile::tempdir().unwrap();
        let file = base.path().join("not-a-dir");
        std::fs::write(&file, b"x").unwrap();
        let url = format!("ipc://{}/x.ipc", file.display());
        assert!(ensure_ipc_socket_dir(&url).await.is_err());
    }

    fn eos_request(stop_token_ids: Vec<u32>, ignore_eos: bool) -> ProtoGenerateRequest {
        ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                stop_token_ids,
                ignore_eos,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn eos_stop_ids(req: &ProtoGenerateRequest) -> &[u32] {
        match req {
            ProtoGenerateRequest::Vllm(r) => &r.sampling_params.as_ref().unwrap().stop_token_ids,
            _ => panic!("expected vLLM request"),
        }
    }

    #[test]
    fn eos_backstop_appends_tokenizer_ids_without_duplicates() {
        // MockTokenizer's EOS set is {999}.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());

        let mut req = eos_request(vec![7], false);
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        assert_eq!(eos_stop_ids(&req), &[7, 999]);

        // Already-present EOS ids are not duplicated.
        let mut req = eos_request(vec![999], false);
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        assert_eq!(eos_stop_ids(&req), &[999]);
    }

    /// A connected client over a throwaway ipc endpoint. The mock engine is
    /// dropped on return: these tests only inspect request-side EOS state.
    async fn connected_client(dir: &Path, prefix: &str, eos: EosTokenIds) -> ZmqEngineClient {
        let ep = |name: &str| format!("ipc://{}", dir.join(format!("{prefix}-{name}")).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "org/repo".to_string(),
                eos,
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        engine.expect("mock engine");
        client.expect("adapter connect")
    }

    #[tokio::test]
    async fn tokenizer_eos_is_adopted_when_the_model_dir_is_not_local() {
        // MockTokenizer's EOS set is {999}. With no local model dir the
        // connect-time set is empty, so the primary id must come from the
        // tokenizer — otherwise EOS rides `stop_token_ids` alone and an EOS
        // finish is reported as `matched_stop = 999`.
        let dir = tempfile::tempdir().unwrap();
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());

        let client = connected_client(dir.path(), "empty", EosTokenIds::default()).await;
        assert_eq!(client.effective_eos(), &EosTokenIds::default());
        client.adopt_tokenizer_eos(Some(&tokenizer));
        assert_eq!(client.effective_eos(), &EosTokenIds::new(Some(999), vec![]));

        // A connect-time set resolved from a local model dir wins: adoption is
        // a backstop, not an override.
        let resolved = EosTokenIds::new(Some(5), vec![7]);
        let client = connected_client(dir.path(), "resolved", resolved.clone()).await;
        client.adopt_tokenizer_eos(Some(&tokenizer));
        assert_eq!(client.effective_eos(), &resolved);
    }

    #[test]
    fn eos_backstop_respects_ignore_eos() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let mut req = eos_request(vec![7], true);
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        assert_eq!(eos_stop_ids(&req), &[7]);
    }

    /// The variant match is the only gate on the fold: a TokenSpeed request
    /// (whose scheduler stops at EOS itself) is left untouched.
    #[test]
    fn eos_backstop_leaves_tokenspeed_requests_untouched() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let mut req =
            ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
                sampling_params: Some(tokenspeed_proto::SamplingParams {
                    stop_token_ids: vec![7],
                    ..Default::default()
                }),
                ..Default::default()
            }));
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        let ProtoGenerateRequest::TokenSpeed(req) = req else {
            panic!("expected TokenSpeed request");
        };
        assert_eq!(req.sampling_params.unwrap().stop_token_ids, vec![7]);
    }

    /// The dialect is resolved before the handshake, so a runtime with no ZMQ
    /// adapter fails immediately instead of blocking for the connect timeout
    /// (this test would hang on the generous timeout otherwise).
    #[tokio::test]
    async fn connect_rejects_a_runtime_without_a_zmq_adapter_before_the_handshake() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let Err(error) = ZmqEngineClient::connect(
            &ep("hs.sock"),
            &ep("in.sock"),
            &ep("out.sock"),
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::Sglang,
            ZMQ_CONNECT_TIMEOUT,
        )
        .await
        else {
            panic!("SGLang has no ZMQ engine adapter");
        };
        assert!(
            error.to_string().contains("no engine implementation"),
            "{error}"
        );
    }

    fn batch(
        request_id: &str,
        token: u32,
        logprob: Option<f32>,
        finish: Option<EngineCoreFinishReason>,
    ) -> EngineCoreOutputs {
        let finished = finish.map(|_| BTreeSet::from([request_id.to_string()]));
        let new_logprobs = logprob.map(|lp| Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![TokenLogprob {
                    token_id: token,
                    logprob: lp,
                    rank: 1,
                }],
            }],
        });
        EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: request_id.to_string(),
                new_token_ids: vec![token],
                new_logprobs,
                finish_reason: finish,
                ..Default::default()
            }],
            finished_requests: finished,
            ..Default::default()
        })
    }

    /// End-to-end over ipc://: the adapter translates a vLLM-proto request to
    /// EngineCore, and maps the engine's outputs back to vLLM-proto responses.
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_vllm_proto() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let inbound = input.recv().await.unwrap();
            let request = match inbound {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            assert_eq!(request.request_id, "r1");
            assert_eq!(request.prompt_token_ids, Some(vec![1, 2, 3]));
            assert_eq!(request.sampling_params.as_ref().unwrap().max_tokens, 2);
            output
                .send_outputs(&batch("r1", 10, Some(-0.5), None))
                .await
                .unwrap();
            output
                .send_outputs(&batch(
                    "r1",
                    11,
                    Some(-1.25),
                    Some(EngineCoreFinishReason::Length),
                ))
                .await
                .unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(2),
                logprobs: Some(1),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client
            .generate(ProtoGenerateRequest::Vllm(Box::new(req)))
            .await
            .expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a new token, so its delta is emitted as a
        // chunk before the (cumulative) terminal complete.
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

        engine_task.await.unwrap();
    }

    /// The sampled token also ranks first — the common case under greedy or
    /// low-temperature decoding. The engine repeats it in the ranked columns,
    /// so the shaped list must carry it once and still return `k` candidates.
    #[test]
    fn shape_top_logprobs_dedups_sampled_token_at_rank_one() {
        let entries = vec![
            TokenLogprob {
                token_id: 10,
                logprob: -0.1,
                rank: 1,
            },
            TokenLogprob {
                token_id: 10,
                logprob: -0.1,
                rank: 1,
            },
            TokenLogprob {
                token_id: 20,
                logprob: -0.3,
                rank: 2,
            },
        ];
        assert_eq!(
            shape_top_logprobs(&entries, 2),
            vllm::TopLogProbs {
                values: vec![-0.1, -0.3],
                token_ids: vec![10, 20],
            }
        );
        assert_eq!(
            shape_top_logprobs(&entries, 1),
            vllm::TopLogProbs {
                values: vec![-0.1],
                token_ids: vec![10],
            }
        );
    }

    /// A sampled token outside the top-k leads the list and the ranked
    /// candidates follow in order, truncated to the requested count.
    #[test]
    fn shape_top_logprobs_keeps_sampled_token_outside_top_k() {
        let entries = vec![
            TokenLogprob {
                token_id: 10,
                logprob: -0.5,
                rank: 5,
            },
            TokenLogprob {
                token_id: 20,
                logprob: -0.1,
                rank: 1,
            },
            TokenLogprob {
                token_id: 30,
                logprob: -0.3,
                rank: 2,
            },
        ];
        assert_eq!(
            shape_top_logprobs(&entries, 2),
            vllm::TopLogProbs {
                values: vec![-0.5, -0.1],
                token_ids: vec![10, 20],
            }
        );
    }

    /// With `logprobs=k`, each position's ranked candidates are shaped into
    /// `top_logprobs`, taking the sampled entry plus the leading candidates up
    /// to the requested count (matching the gRPC servicer's `islice` behaviour).
    #[tokio::test]
    async fn generate_shapes_top_logprobs_to_requested_count() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        // One position with the sampled token (actual vocab rank) first, then
        // the engine's ranked candidates. The wire carries `k + 1` entries.
        let position = PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id: 10,
                    logprob: -0.5,
                    rank: 5,
                },
                TokenLogprob {
                    token_id: 20,
                    logprob: -0.1,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: 30,
                    logprob: -0.3,
                    rank: 2,
                },
            ],
        };
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".to_string(),
                new_token_ids: vec![10],
                new_logprobs: Some(Logprobs {
                    positions: vec![position],
                }),
                finish_reason: Some(EngineCoreFinishReason::Length),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["r1".to_string()])),
            ..Default::default()
        });

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let inbound = input.recv().await.unwrap();
            let request = match inbound {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            assert_eq!(request.sampling_params.as_ref().unwrap().logprobs, Some(2));
            output.send_outputs(&outputs).await.unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(1),
                logprobs: Some(2),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client
            .generate(ProtoGenerateRequest::Vllm(Box::new(req)))
            .await
            .expect("generate");

        // The requested count is 2, so `top_logprobs` keeps the sampled entry
        // plus the leading candidate (the third entry is dropped).
        let expected_top = vec![vllm::TopLogProbs {
            values: vec![-0.5, -0.1],
            token_ids: vec![10, 20],
        }];

        // The finish tick carried a token, so the delta streams as a chunk.
        let chunk = stream.next().await.expect("chunk item").expect("chunk ok");
        match chunk.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
                assert_eq!(logprobs.top_logprobs, expected_top);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let complete = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match complete.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
                assert_eq!(logprobs.top_logprobs, expected_top);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    /// Prompt logprobs end to end: the request carries `prompt_logprobs` to the
    /// engine, and the engine's prompt tensors come back as `input_logprobs` on
    /// the first token-bearing chunk (once) and on the terminal `Complete`.
    #[tokio::test]
    async fn generate_streams_prompt_logprobs() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        // Prompt position for input token 2 (the engine scores every prompt
        // token but the first), with one ranked candidate behind it.
        let prompt_tensors = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 2,
                        logprob: -0.5,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 20,
                        logprob: -0.1,
                        rank: 1,
                    },
                ],
            }],
        };
        let prefill = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".to_string(),
                new_token_ids: vec![10],
                new_prompt_logprobs_tensors: Some(prompt_tensors),
                ..Default::default()
            }],
            ..Default::default()
        });
        let decode = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".to_string(),
                new_token_ids: vec![11],
                finish_reason: Some(EngineCoreFinishReason::Length),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["r1".to_string()])),
            ..Default::default()
        });

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let inbound = input.recv().await.unwrap();
            let request = match inbound {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            assert_eq!(
                request.sampling_params.as_ref().unwrap().prompt_logprobs,
                Some(1),
                "prompt_logprobs reaches the engine"
            );
            output.send_outputs(&prefill).await.unwrap();
            output.send_outputs(&decode).await.unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(2),
                prompt_logprobs: Some(1),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client
            .generate(ProtoGenerateRequest::Vllm(Box::new(req)))
            .await
            .expect("generate");

        // Prompt token 1 leads with a null logprob (nothing precedes it); the
        // requested count of 1 keeps the prompt token's own ranked entry.
        let expect_input_logprobs = |logprobs: Option<vllm::InputLogProbs>, whose: &str| {
            let logprobs = logprobs.unwrap_or_else(|| panic!("{whose} input logprobs"));
            assert_eq!(logprobs.token_ids, vec![1, 2]);
            assert_eq!(
                logprobs.token_logprobs,
                vec![
                    vllm::InputTokenLogProb { value: None },
                    vllm::InputTokenLogProb { value: Some(-0.5) },
                ]
            );
            assert_eq!(
                logprobs.top_logprobs,
                vec![
                    vllm::TopLogProbs::default(),
                    vllm::TopLogProbs {
                        values: vec![-0.5],
                        token_ids: vec![2],
                    },
                ]
            );
        };

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                expect_input_logprobs(chunk.input_logprobs, "first chunk");
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a token, so its delta streams as a chunk
        // first — without repeating the prompt logprobs.
        let second = stream.next().await.expect("chunk item").expect("chunk ok");
        match second.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert!(
                    chunk.input_logprobs.is_none(),
                    "prompt logprobs ride the first chunk only"
                );
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let complete = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match complete.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                expect_input_logprobs(complete.input_logprobs, "complete");
            }
            other => panic!("expected complete, got {other:?}"),
        }

        engine_task.await.unwrap();
    }

    /// End-to-end over ipc:// for a TokenSpeed backend: the adapter frames a
    /// tagged `TokenizedGenerateReqInput`, and maps `BatchTokenIDOutSlim`
    /// batches back to vLLM-proto responses. The mock engine speaks the shared
    /// transport with raw frames (it decodes/encodes the TokenSpeed structs
    /// directly).
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::TokenSpeed,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let frames = input.recv_frames().await.unwrap();
            assert_eq!(
                TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                Some(TokenSpeedRequestType::Add)
            );
            let request: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
            assert_eq!(request.rid, "r1");
            assert_eq!(request.input_ids, vec![1, 2, 3]);
            assert_eq!(request.sampling_params.max_new_tokens, Some(2));
            // The adapter always emits the normalized sampling form.
            assert!(request.sampling_params.is_normalized);
            // A plain sampled-token logprob request (logprobs=1) sets the flag.
            assert!(request.return_logprob);

            let chunk = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![10]],
                finished_reasons: vec![String::new()],
                prompt_tokens: vec![3],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-0.5]],
                output_token_logprobs_idx: vec![vec![10]],
                ..Default::default()
            };
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["length".into()],
                prompt_tokens: vec![3],
                completion_tokens: vec![2],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-1.25]],
                output_token_logprobs_idx: vec![vec![11]],
                ..Default::default()
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&chunk).unwrap())])
                .await
                .unwrap();
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
        });

        let req = tokenspeed_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(tokenspeed_proto::SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            // Plain sampled-token logprob; must be wired through.
            return_logprob: true,
            stream: true,
            ..Default::default()
        };
        let mut stream = client
            .generate(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
            .await
            .expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                assert_eq!(chunk.prompt_tokens, 3);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a new token, so its delta is emitted as a
        // chunk before the (cumulative) terminal complete.
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
                // Chunks carry the tick's incremental logprobs; the terminal
                // `Complete` carries the cumulative set, parallel to `output_ids`
                // (the non-streaming renderer reads only the `Complete`).
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
                assert_eq!(logprobs.token_ids, vec![10, 11]);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    #[test]
    fn finish_reasons_map_to_vllm_strings() {
        assert_eq!(finish_reason_str(EngineCoreFinishReason::Length), "length");
        assert_eq!(
            finish_reason_str(EngineCoreFinishReason::Repetition),
            "stop"
        );
        assert_eq!(finish_reason_str(EngineCoreFinishReason::Abort), "abort");
    }

    #[test]
    fn tokenspeed_sampling_maps_top_k_sentinel_and_floors_n() {
        use engine_zmq_client::protocol::tokenspeed::sampling::TOP_K_DISABLED;

        // Unset top_k rides the API convention `-1` ("all tokens") and normalizes
        // to the engine's disabled sentinel; n=0 floors to 1; max_new_tokens
        // forwards.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            top_k: None,
            sampling_seed: Some(1_234),
            n: 0,
            max_new_tokens: Some(8),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, TOP_K_DISABLED);
        assert_eq!(mapped.seed, Some(1_234));
        assert_eq!(mapped.n, 1);
        assert_eq!(mapped.max_new_tokens, Some(8));
        // The wire form is always normalized (the engine skips re-derivation).
        assert!(mapped.is_normalized);

        // An explicit top_k passes through unchanged.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            top_k: Some(40),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, 40);

        // A near-zero temperature collapses to greedy on the wire.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        });
        assert_eq!(mapped.temperature, 1.0);
        assert_eq!(mapped.top_k, 1);

        // Empty stop_token_ids ride as None (the normalized encoding).
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams::default());
        assert_eq!(mapped.stop_token_ids, None);
    }

    fn tokenized_req(sampling: vllm::SamplingParams) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(sampling),
            stream: true,
            ..Default::default()
        }
    }

    fn ts_tokenized_req(
        sampling: tokenspeed_proto::SamplingParams,
    ) -> tokenspeed_proto::GenerateRequest {
        tokenspeed_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(sampling),
            stream: true,
            ..Default::default()
        }
    }

    #[test]
    fn tokenspeed_return_logprob_flag_passes_through() {
        // The request-level `return_logprob` drives the plain sampled-token
        // logprob (count 0/1 in `top_logprobs_num` is the same case).
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.return_logprob = true;
        let wire = translate_request_tokenspeed(req).expect("return_logprob accepted");
        assert!(wire.return_logprob);

        // Unset -> the flag stays false.
        let wire = translate_request_tokenspeed(ts_tokenized_req(
            tokenspeed_proto::SamplingParams::default(),
        ))
        .expect("no logprobs accepted");
        assert!(!wire.return_logprob);
    }

    #[test]
    fn tokenspeed_rejects_top_logprobs_and_prompt_logprobs() {
        // Top-k logprobs (count > 1) cannot be honored over the wire.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.top_logprobs_num = 5;
        assert!(translate_request_tokenspeed(req).is_err());

        // Prompt (input) logprobs cannot be produced.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.token_ids_logprob = vec![1, 2];
        assert!(translate_request_tokenspeed(req).is_err());

        // A bare count of 0/1 is the plain sampled-token case: accepted.
        for count in [0, 1] {
            let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
            req.top_logprobs_num = count;
            assert!(translate_request_tokenspeed(req).is_ok());
        }
    }

    #[test]
    fn tokenspeed_maps_structured_output_constraints() {
        // The `constraint` oneof maps 1:1 onto the wire's dedicated fields; the
        // oneof is single-valued, so the other three stay unset.
        use tokenspeed_proto::sampling_params::Constraint;

        let json = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::JsonSchema("{\"type\":\"object\"}".into())),
            ..Default::default()
        });
        assert_eq!(json.json_schema.as_deref(), Some("{\"type\":\"object\"}"));
        assert_eq!(json.regex, None);
        assert_eq!(json.ebnf, None);
        assert_eq!(json.structural_tag, None);

        let regex = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::Regex("[0-9]+".into())),
            ..Default::default()
        });
        assert_eq!(regex.regex.as_deref(), Some("[0-9]+"));
        assert_eq!(regex.json_schema, None);

        let ebnf = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::EbnfGrammar("root ::= \"a\"".into())),
            ..Default::default()
        });
        assert_eq!(ebnf.ebnf.as_deref(), Some("root ::= \"a\""));

        let tag = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::StructuralTag("<tag>".into())),
            ..Default::default()
        });
        assert_eq!(tag.structural_tag.as_deref(), Some("<tag>"));

        // No constraint leaves all four structured-output fields unset.
        let none = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams::default());
        assert_eq!(none.json_schema, None);
        assert_eq!(none.regex, None);
        assert_eq!(none.ebnf, None);
        assert_eq!(none.structural_tag, None);
    }

    #[test]
    fn tokenspeed_forwards_stop_token_ids_and_drops_stop_strings() {
        // String stops are resolved upstream; any that reach here are dropped
        // (the token-only engine cannot match them) while stop token ids ride
        // through and the router-side decoder trims residual text.
        let req =
            translate_request_tokenspeed(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                stop: vec!["</s>".to_string()],
                stop_token_ids: vec![13],
                ..Default::default()
            }))
            .expect("residual stop strings must not be rejected");
        assert_eq!(req.sampling_params.stop_token_ids, Some(vec![13]));
        assert_eq!(req.sampling_params.stop, None);
    }

    #[test]
    fn tokenspeed_rejects_multimodal_inputs() {
        // The TokenSpeed ZMQ wire has no multimodal slot yet; reject rather than
        // silently drop pixels.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs::default());
        assert!(translate_request_tokenspeed(req).is_err());
    }

    #[test]
    fn vllm_forwards_prompt_logprobs() {
        // Prompt logprobs ride the wire sampling params verbatim.
        let request = translate_request(
            tokenized_req(vllm::SamplingParams {
                prompt_logprobs: Some(2),
                ..Default::default()
            }),
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
        )
        .expect("translated");
        assert_eq!(
            request.sampling_params.as_ref().unwrap().prompt_logprobs,
            Some(2)
        );
    }

    #[test]
    fn ranked_candidate_count_maps_the_sentinels() {
        assert_eq!(ranked_candidate_count(None), 0);
        assert_eq!(ranked_candidate_count(Some(0)), 0);
        assert_eq!(ranked_candidate_count(Some(3)), 3);
        // -1 asks for every candidate the engine returned.
        assert_eq!(ranked_candidate_count(Some(-1)), usize::MAX);
    }

    #[test]
    fn vllm_defaults_unset_max_tokens_to_remaining_context() {
        let max_tokens = |sampling, max_model_len| {
            translate_request(
                tokenized_req(sampling),
                max_model_len,
                ModelDtype::BFloat16,
                &EosTokenIds::default(),
            )
            .expect("request translated")
            .sampling_params
            .expect("sampling params present")
            .max_tokens
        };

        // Unset max_tokens defaults to `max_model_len - prompt_len` (prompt is
        // 3 tokens), mirroring vLLM's bypassed OpenAI frontend.
        assert_eq!(max_tokens(vllm::SamplingParams::default(), 100), 97);
        // An explicit value is always honored.
        assert_eq!(
            max_tokens(
                vllm::SamplingParams {
                    max_tokens: Some(8),
                    ..Default::default()
                },
                100,
            ),
            8,
        );
    }

    #[test]
    fn vllm_attaches_eos_stop_ids() {
        let eos = EosTokenIds::new(Some(5), vec![7]);
        let sampling = |sp| {
            translate_request(tokenized_req(sp), 4096, ModelDtype::BFloat16, &eos)
                .expect("request translated")
                .sampling_params
                .expect("sampling params present")
        };

        // Primary rides `_eos_token_id`, extras merge into `stop_token_ids`
        // without duplicating, and the union lands in `_all_stop_token_ids`.
        let sp = sampling(vllm::SamplingParams {
            stop_token_ids: vec![7, 9],
            ..Default::default()
        });
        assert_eq!(sp.eos_token_id, Some(5));
        assert_eq!(sp.stop_token_ids, vec![7, 9]);
        assert_eq!(sp.all_stop_token_ids, BTreeSet::from([5, 7, 9]));

        // ignore_eos drops the EOS stops from the wire but keeps the
        // bookkeeping set (mirrors the reference frontend).
        let sp = sampling(vllm::SamplingParams {
            stop_token_ids: vec![9],
            ignore_eos: true,
            ..Default::default()
        });
        assert_eq!(sp.eos_token_id, None);
        assert_eq!(sp.stop_token_ids, vec![9]);
        assert_eq!(sp.all_stop_token_ids, BTreeSet::from([5, 7, 9]));
    }

    #[tokio::test]
    async fn eos_token_ids_resolve_from_model_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), r#"{"eos_token_id": 5}"#).unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id": [5, 7, 9]}"#,
        )
        .unwrap();
        assert_eq!(
            EosTokenIds::from_model_dir(dir.path()).await,
            EosTokenIds::new(Some(5), vec![7, 9]),
        );

        // Missing files degrade to no ids, not an error.
        let empty = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            EosTokenIds::from_model_dir(empty.path()).await,
            EosTokenIds::default(),
        );
    }

    #[test]
    fn vllm_translates_structured_output_constraints() {
        use engine_zmq_client::protocol::vllm::structured_outputs::{
            StructuredOutputBackend, StructuredOutputConstraint,
        };

        let translate = |constraint| {
            translate_request(
                tokenized_req(vllm::SamplingParams {
                    constraint: Some(constraint),
                    ..Default::default()
                }),
                4096,
                ModelDtype::BFloat16,
                &EosTokenIds::default(),
            )
            .expect("constraint translated")
            .sampling_params
            .expect("sampling params present")
            .structured_outputs
        };

        // Each constraint mode maps onto its typed counterpart, always lowering
        // to the guidance backend engine-side.
        let json_object = translate(vllm::sampling_params::Constraint::JsonObject(true))
            .expect("json_object translated");
        assert_eq!(
            json_object.constraint,
            StructuredOutputConstraint::JsonObject
        );
        assert_eq!(json_object.backend, StructuredOutputBackend::Guidance);

        let regex = translate(vllm::sampling_params::Constraint::Regex("a.*".to_string()))
            .expect("regex translated");
        assert_eq!(
            regex.constraint,
            StructuredOutputConstraint::Regex("a.*".to_string())
        );

        let choice = translate(vllm::sampling_params::Constraint::Choice(
            vllm::ChoiceConstraint {
                choices: vec!["yes".to_string(), "no".to_string()],
            },
        ))
        .expect("choice translated");
        assert_eq!(
            choice.constraint,
            StructuredOutputConstraint::Choice(vec!["yes".to_string(), "no".to_string()])
        );

        // A JSON schema string is parsed to preserve object shape.
        let json = translate(vllm::sampling_params::Constraint::JsonSchema(
            r#"{"type":"object"}"#.to_string(),
        ))
        .expect("json schema translated");
        assert_eq!(
            json.constraint,
            StructuredOutputConstraint::Json(serde_json::json!({"type": "object"}))
        );

        // json_object=false means the caller opted out: no constraint.
        assert!(translate(vllm::sampling_params::Constraint::JsonObject(false)).is_none());
    }

    /// n=3 fans out into 3 single-sample wire requests with unique sub-rids.
    /// An explicit seed derives per-sub seeds (`seed + i`) so the samples
    /// differ deterministically; no seed stays `None` per sub (the engine
    /// seeds each rid independently).
    #[test]
    fn fan_out_splits_n_into_single_sample_requests() {
        let mut req = tokenized_req(vllm::SamplingParams {
            n: 3,
            seed: Some(7),
            ..Default::default()
        });
        req.request_id = "r1".to_string();

        let subs = fan_out_requests(req);
        assert_eq!(subs.len(), 3);
        let rids: Vec<&str> = subs.iter().map(|sub| sub.request_id.as_str()).collect();
        assert_eq!(rids, vec!["r1-0", "r1-1", "r1-2"]);
        for (i, sub) in subs.iter().enumerate() {
            let sp = sub.sampling_params.as_ref().unwrap();
            assert_eq!(sp.n, 1);
            assert_eq!(sp.seed, Some(7 + i as i32));
            // Everything else is shared verbatim.
            assert_eq!(
                sub.input,
                Some(vllm::generate_request::Input::Tokenized(
                    vllm::TokenizedInput {
                        original_text: String::new(),
                        input_ids: vec![1, 2, 3],
                    }
                ))
            );
        }

        // No explicit seed: every sub keeps None.
        let subs = fan_out_requests(tokenized_req(vllm::SamplingParams {
            n: 2,
            ..Default::default()
        }));
        assert!(subs
            .iter()
            .all(|sub| sub.sampling_params.as_ref().unwrap().seed.is_none()));

        // n<=1 passes through untouched (rid keeps its original form).
        let single = fan_out_requests(tokenized_req(vllm::SamplingParams::default()));
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].request_id, "r1");
    }

    #[test]
    fn tokenspeed_fan_out_derives_explicit_sampling_seeds() {
        let subs =
            fan_out_tokenspeed_requests(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                n: 3,
                sampling_seed: Some(7),
                ..Default::default()
            }));

        assert_eq!(subs.len(), 3);
        for (i, sub) in (0_u64..).zip(&subs) {
            let sampling = sub.sampling_params.as_ref().expect("sampling params");
            assert_eq!(sub.request_id, format!("r1-{i}"));
            assert_eq!(sampling.n, 1);
            assert_eq!(sampling.sampling_seed, Some(7 + i));
        }

        let subs =
            fan_out_tokenspeed_requests(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                n: 2,
                ..Default::default()
            }));
        assert!(subs.iter().all(|sub| sub
            .sampling_params
            .as_ref()
            .expect("sampling params")
            .sampling_seed
            .is_none()));
    }

    /// n=2 over a vLLM EngineCore: two engine-side requests with distinct
    /// sub-rids, and the merged stream yields two `Complete`s tagged with the
    /// proto `index` (0 and 1) the pipeline demuxes choices by.
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_vllm() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let mut rids = Vec::new();
            for _ in 0..2 {
                let request = match input.recv().await.unwrap() {
                    EngineInbound::Add(request) => request,
                    other => panic!("expected Add, got {other:?}"),
                };
                // Each sub is a single-sample request with a derived seed.
                let sp = request.sampling_params.as_ref().unwrap();
                assert_eq!(sp.max_tokens, 4);
                rids.push((request.request_id.clone(), sp.seed));
            }
            assert_eq!(
                rids,
                vec![("r1-0".to_string(), Some(5)), ("r1-1".to_string(), Some(6))],
                "sub-rids must be unique and seeds derived per sub"
            );
            output
                .send_outputs(&batch("r1-0", 10, None, Some(EngineCoreFinishReason::Stop)))
                .await
                .unwrap();
            output
                .send_outputs(&batch("r1-1", 11, None, Some(EngineCoreFinishReason::Stop)))
                .await
                .unwrap();
        });

        let mut req = tokenized_req(vllm::SamplingParams {
            n: 2,
            seed: Some(5),
            max_tokens: Some(4),
            ..Default::default()
        });
        req.request_id = "r1".to_string();
        let mut stream = client
            .generate(ProtoGenerateRequest::Vllm(Box::new(req)))
            .await
            .expect("generate");

        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item").response {
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    completes.push(complete);
                }
                Some(vllm::generate_response::Response::Chunk(_)) | None => {}
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(completes[0].index, 0);
        assert_eq!(completes[0].output_ids, vec![10]);
        assert_eq!(completes[1].index, 1);
        assert_eq!(completes[1].output_ids, vec![11]);

        engine_task.await.unwrap();
    }

    /// n=2 over TokenSpeed: two engine-side requests with distinct sub-rids
    /// (delivered in one wire batch), two indexed `Complete`s, and the shared
    /// prompt reported in full on each (the pipeline de-duplicates via max, so
    /// prompt tokens are not counted n times).
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::TokenSpeed,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let mut rids = Vec::new();
            for _ in 0..2 {
                let frames = input.recv_frames().await.unwrap();
                assert_eq!(
                    TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                    Some(TokenSpeedRequestType::Add)
                );
                let request: TokenizedGenerateReqInput =
                    decode_msgpack(frames[1].as_ref()).unwrap();
                assert_eq!(request.sampling_params.n, 1);
                // TokenSpeed has no seed on the wire; the engine derives one from
                // the (unique) rid so all TP/DP ranks agree.
                assert_eq!(request.sampling_params.seed, None);
                rids.push(request.rid.clone());
            }
            assert_eq!(
                rids,
                vec!["r1-0".to_string(), "r1-1".to_string()],
                "sub-rids must be unique per sub"
            );
            // Both subs finish in one wire batch (the batch demux fans them
            // back out to their sub-streams).
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1-0".into(), "r1-1".into()],
                output_ids: vec![vec![10], vec![11]],
                finished_reasons: vec!["stop".into(), "stop".into()],
                prompt_tokens: vec![3, 3],
                completion_tokens: vec![1, 1],
                cached_tokens: vec![0, 0],
                output_token_logprobs_val: vec![vec![], vec![]],
                output_token_logprobs_idx: vec![vec![], vec![]],
                ..Default::default()
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
        });

        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams {
            n: 2,
            ..Default::default()
        });
        req.request_id = "r1".to_string();
        let mut stream = client
            .generate(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
            .await
            .expect("generate");

        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item").response {
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    completes.push(complete);
                }
                Some(vllm::generate_response::Response::Chunk(_)) | None => {}
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(completes[0].index, 0);
        assert_eq!(completes[0].output_ids, vec![10]);
        assert_eq!(completes[1].index, 1);
        assert_eq!(completes[1].output_ids, vec![11]);
        // Each sub reports the full shared prompt; the pipeline maxes, so
        // usage is not double-counted.
        assert!(completes.iter().all(|complete| complete.prompt_tokens == 3));

        engine_task.await.unwrap();
    }

    /// Dropping the merged stream before completion aborts EVERY fanned-out
    /// engine-side sub-request, not just one.
    #[tokio::test]
    async fn dropping_fanned_out_stream_aborts_all_subs() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");
        let (mut engine_input, _engine_output) = engine.split();

        let stream = client
            .generate(ProtoGenerateRequest::Vllm(Box::new(tokenized_req(
                vllm::SamplingParams {
                    n: 2,
                    ..Default::default()
                },
            ))))
            .await
            .expect("generate");

        // Consume both Adds first so the drop-triggered aborts are the next
        // inbound messages.
        for _ in 0..2 {
            match engine_input.recv().await.unwrap() {
                EngineInbound::Add(_) => {}
                other => panic!("expected Add, got {other:?}"),
            }
        }

        drop(stream); // unfinished -> every sub auto-aborts

        let mut aborted = BTreeSet::new();
        while aborted.len() < 2 {
            match engine_input.recv().await.unwrap() {
                EngineInbound::Abort(rids) => aborted.extend(rids),
                other => panic!("expected Abort, got {other:?}"),
            }
        }
        assert_eq!(
            aborted,
            BTreeSet::from(["r1-0".to_string(), "r1-1".to_string()])
        );
    }

    #[test]
    fn tokenspeed_finish_reason_normalizes() {
        assert_eq!(normalize_finish_reason("stop"), "stop");
        assert_eq!(normalize_finish_reason("length"), "length");
        assert_eq!(normalize_finish_reason("abort"), "abort");
        assert_eq!(normalize_finish_reason("error"), "error");
        // An unknown wire value falls back to "stop".
        assert_eq!(normalize_finish_reason("garbage"), "stop");
    }
}
