//! [`ZmqEngineClient`]: the connected engine behind each engine's gRPC client
//! surface (vLLM and TokenSpeed today), plus the connect entry points and
//! metadata reads.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::{Arc, OnceLock},
    time::Duration,
};

use engine_zmq_client::{
    check_engine_layouts,
    codec::{tensor::encode_with_aux_frames, OpaqueValue},
    connect_handshake,
    connector::{Client, EngineCoreClient, SglangClient, TokenSpeedClient, TrtllmClient},
    protocol::{
        handshake::{EngineCoreReadyResponse, REQUEST_KIND_ENCODE},
        sglang::{request::SglangRequest, SglangProtocol},
        tokenspeed::{
            request::TokenSpeedRequestType, wire_carries_control_calls, TokenSpeedProtocol,
        },
        trtllm::TrtllmProtocol,
        vllm::{request::EngineCoreRequest, sampling::EngineCoreSamplingParams, VllmProtocol},
        EngineLoad, EngineProtocol,
    },
    ConnectedEngine, EngineCount, StartupBudget,
};
use futures::StreamExt;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::worker::{RuntimeType, SchedulerLoadSnapshot, WorkerLoadResponse};
use smg_grpc_client::{sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto as vllm};

#[cfg(unix)]
use crate::sockets::unlink_stale_socket;
use crate::{
    eos::EosTokenIds,
    sglang::{
        aggregate_control, constraint_ref as sglang_constraint_ref, control_outcome,
        encode_media_refs, fan_out_sglang_requests, translate_embed_request_sglang,
        translate_request_sglang, RankPlacement, SglangGenerateStream, SglangProfileStart,
    },
    sglang_encode::{EncodeDispatchRequest, EncodeNotice, SglangEncodeDispatcher},
    sockets::{ensure_ipc_socket_dir, zmq_socket_addresses, Handshake},
    stream::{RequestDeadline, ZmqGenerateStream},
    structured_outputs::{
        refuse_unenforceable_constraint, StructuredOutputsBackendConfig, StructuredOutputsSupport,
    },
    tokenspeed::{
        constraint_ref as tokenspeed_constraint_ref, fan_out_tokenspeed_requests,
        translate_encode_request, translate_request_tokenspeed_for, EncodeRendezvous, GridTensors,
        LogprobColumns, RankIndices, TokenSpeedGenerateStream,
    },
    trtllm::{
        constraint_ref as trtllm_constraint_ref, fan_out_trtllm_requests, translate_request_trtllm,
        TrtllmGenerateStream,
    },
    vllm::{
        fan_out_requests, has_media, kv_transfer_params, now_secs, ranked_candidate_count,
        refuse_greedy_choices, translate_media, translate_request_with_media,
        translated_from_processed, ProcessedMedia, TranslatedMedia, VllmGenerateStream,
    },
};

/// How long the one-time `get_supported_tasks` utility call may take; the
/// engine answers it between scheduler steps.
const SUPPORTED_TASKS_TIMEOUT: Duration = Duration::from_secs(30);

/// What a connection nobody configured enforces of a structured request, by
/// dialect: the wire that stamps a backend on each structured request ran
/// the engine's `auto` (xgrammar, every kind) before this support existed
/// and keeps it, refusing what it refused; the wires that stamp none judged
/// nothing ahead of their engine and keep that too, so a gateway's direct
/// link to an engine on another backend refuses nothing the engine would
/// enforce. The servicer that launched the engine sets the real support.
pub(crate) fn default_structured_outputs_support(dialect: ZmqDialect) -> StructuredOutputsSupport {
    match dialect {
        ZmqDialect::Vllm => StructuredOutputsSupport::default(),
        ZmqDialect::TokenSpeed | ZmqDialect::Sglang | ZmqDialect::Trtllm => {
            StructuredOutputsBackendConfig::Other.into()
        }
    }
}

/// The connector params a refused request must still release: NIXL's
/// `do_remote_prefill` marks a PD decode side whose prefill side holds blocks
/// for it until told otherwise. `None` for every other request.
pub fn kv_transfer_rejection_params(req: &vllm::GenerateRequest) -> Option<serde_json::Value> {
    let params = kv_transfer_params(req).ok().flatten()?;
    params
        .get("do_remote_prefill")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        .then_some(params)
}

/// The engine protocol a ZMQ backend speaks — a closed set: the transport has
/// an adapter for exactly these engines. Resolved once at connect time and
/// exposed by [`ZmqEngineClient::dialect`] so every per-engine dispatch on the
/// ZMQ lane (request building, multimodal, EOS) matches on the same variants
/// with no unreachable arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZmqDialect {
    /// vLLM EngineCore.
    Vllm,
    /// TokenSpeed.
    TokenSpeed,
    /// SGLang's scheduler, behind the SMG plugin that performs the handshake.
    Sglang,
    /// TensorRT-LLM's LLM API, behind the shim that performs the handshake.
    Trtllm,
}

/// The connected client for a [`ZmqDialect`]. All share the transport and
/// handshake; only the request/output struct shapes and the translation to/from
/// SMG proto differ.
#[derive(Clone)]
pub(crate) enum ZmqBackend {
    Vllm(Arc<EngineCoreClient>),
    TokenSpeed(Arc<TokenSpeedClient>),
    Sglang(Arc<SglangClient>),
    Trtllm(Arc<TrtllmClient>),
}

/// Per-connection constants of a [`ZmqEngineClient`], fixed at connect time.
pub(crate) struct ZmqConnectionMeta {
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
    /// What the engine enforces of a structured request, set once by the
    /// caller that knows it (the servicer that launched the engine); unset
    /// is the engine's `auto`, every kind taken
    /// ([`StructuredOutputsSupport::default`]).
    structured_outputs: OnceLock<StructuredOutputsSupport>,
    /// The engine's supported tasks (`get_supported_tasks`), fetched on first
    /// use and kept for the connection, as vLLM's frontend keeps them.
    supported_tasks: tokio::sync::OnceCell<Arc<[String]>>,
    /// Set once by the caller that knows it (the servicer that launched the
    /// engine) when the engine was started to cut a long prompt down to its
    /// own bound instead of refusing it: the prompt admission then applies
    /// no upper bound on this connection
    /// (see [`ZmqEngineClient::note_engine_truncates_long_prompts`]).
    engine_truncates_long_prompts: OnceLock<()>,
    /// Set once by the caller that knows it (the servicer that launched the
    /// engine) when the engine places a disaggregated request, one carrying
    /// a rendezvous room, on the data-parallel rank of the room's residue:
    /// such a request is placed there on this connection, a pin that
    /// disagrees yielding to it
    /// (see [`ZmqEngineClient::note_engine_places_by_rendezvous_room`]).
    engine_places_by_rendezvous_room: OnceLock<()>,
    /// The bound on every generate request this connection opens, from its
    /// dispatch to its last message, set once by the caller that knows it
    /// (see [`ZmqEngineClient::set_request_timeout`]); unset leaves the
    /// streams unbounded.
    request_timeout: OnceLock<Duration>,
}

/// Bind the SMG-side ZMQ sockets and complete the handshake with the
/// engine(s): the single connect path for a worker's `ipc://` or `tcp://` URL
/// (see `zmq_socket_addresses`), driven only by the worker's background
/// handshake driver. `model_id` is
/// the config-resolved served model (EngineCore reports none). `engine_count`
/// is the number of DP engines that will dial this worker's sockets (1 for an
/// ungrouped worker), or [`EngineCount::Declared`] for as many as the engines
/// declare. `startup` is the longest any one stage of the handshake
/// (an engine's HELLO, its READY, its registration on the input socket) may
/// stay silent: the gateway did not launch the engine and sees no sign of
/// life from it beside the wire, so this bound alone must cover the engine's
/// init between its HELLO and its READY. `request_timeout` is the gateway's
/// bound on a request to a worker (`--request-timeout-secs`), which every
/// generate stream this connection opens ends at, counted from its dispatch
/// (see [`ZmqEngineClient::set_request_timeout`]); `None` leaves the
/// streams unbounded. Errors are plain reasons; the worker layer wraps them
/// in its own error type.
pub async fn connect_for_worker(
    base_url: &str,
    model_id: String,
    runtime: RuntimeType,
    handshake_override: Option<&str>,
    engine_count: impl Into<EngineCount>,
    startup: Duration,
    request_timeout: Option<Duration>,
) -> Result<ZmqEngineClient, String> {
    // This dialect is served through its gRPC servicer: the gateway's direct
    // lane has no request builder for it, so a worker registered on it is
    // refused here, at registration, instead of handshaken, counted healthy
    // and failed on every request.
    if matches!(runtime, RuntimeType::Trtllm) {
        return Err(format!(
            "runtime {runtime} is served through its gRPC servicer, not the gateway's direct \
             ZMQ lane"
        ));
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
    let client = connect_with_eos(
        base_url,
        model_id,
        runtime,
        Handshake::registered(handshake_override),
        engine_count,
        eos,
        startup,
    )
    .await?;
    if let Some(timeout) = request_timeout {
        client.set_request_timeout(timeout);
    }
    Ok(client)
}

/// [`connect_for_worker`] with the EOS set supplied by the caller: bind the
/// data-plane sockets under `base_url`, clear stale socket files, and complete
/// the handshake. For a frontend that already knows the model's EOS ids from
/// the engine's own config (the Rust gRPC servicer) and has no model dir to
/// read them from. `startup` bounds the handshake: the gateway's connector
/// passes its configured per-stage bound as a plain [`Duration`]; a servicer
/// that launches its own engine passes a [`StartupBudget`] shaped by what
/// that engine's start may take and by the signs of life it sees from the
/// engine process. `handshake` is
/// the endpoint the engine dials: a registered worker's is tcp-only
/// ([`Handshake::registered`]); a servicer's own link may bind an `ipc://`
/// socket ([`Handshake::TcpOrIpc`]).
pub async fn connect_with_eos(
    base_url: &str,
    model_id: String,
    runtime: RuntimeType,
    handshake: Handshake<'_>,
    engine_count: impl Into<EngineCount>,
    eos: EosTokenIds,
    startup: impl Into<StartupBudget>,
) -> Result<ZmqEngineClient, String> {
    let engine_count = engine_count.into();
    let (handshake, input, output) = zmq_socket_addresses(base_url, handshake)?;
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
    tracing::info!(
        "Binding ZMQ client for worker {base_url} (handshake={handshake}, engines={engine_count:?})"
    );
    ZmqEngineClient::connect(
        &handshake,
        &input,
        &output,
        engine_count,
        model_id,
        eos,
        runtime,
        startup,
    )
    .await
    .map_err(|e| format!("Failed to connect ZMQ engine: {e}"))
}

/// Model metadata as the connected engine's native proto response.
pub enum ZmqModelInfo {
    Vllm(vllm::GetModelInfoResponse),
    TokenSpeed(Box<tokenspeed_proto::GetModelInfoResponse>),
    Sglang(Box<sglang_proto::GetModelInfoResponse>),
    Trtllm(Box<trtllm_proto::GetModelInfoResponse>),
}

/// Server metadata as the connected engine's native proto response.
pub enum ZmqServerInfo {
    Vllm(Box<vllm::GetServerInfoResponse>),
    TokenSpeed(Box<tokenspeed_proto::GetServerInfoResponse>),
    Sglang(Box<sglang_proto::GetServerInfoResponse>),
    Trtllm(Box<trtllm_proto::GetServerInfoResponse>),
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
    /// `input_address`/`output_address` are the data-plane endpoints SMG binds
    /// and the engines connect to (`ipc://` for a same-host engine, `tcp://`
    /// for engines on other hosts). `engine_count` is the number of DP
    /// ranks to await ([`EngineCount`]; a plain count is exact). `runtime` selects the wire protocol spoken over the shared
    /// transport (vLLM EngineCore vs TokenSpeed). `startup` bounds the
    /// handshake (see [`StartupBudget`]; a plain [`Duration`] is a per-message
    /// timeout).
    #[expect(
        clippy::too_many_arguments,
        reason = "transport constructor: endpoints, engine count, and runtime are all irreducible connection inputs"
    )]
    pub async fn connect(
        handshake_address: &str,
        input_address: &str,
        output_address: &str,
        engine_count: impl Into<EngineCount>,
        model_id: String,
        eos: EosTokenIds,
        runtime: RuntimeType,
        startup: impl Into<StartupBudget>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Resolve the dialect before the handshake: no silent fallback for a
        // runtime with no ZMQ engine adapter, and no such engine ever dials in,
        // so the handshake would just block for the full timeout.
        let dialect = match runtime {
            // vLLM EngineCore is the default ZMQ wire; an unspecified runtime
            // maps to it for backward compatibility (see `detect_backend`).
            RuntimeType::Vllm | RuntimeType::Unspecified => ZmqDialect::Vllm,
            RuntimeType::TokenSpeed => ZmqDialect::TokenSpeed,
            RuntimeType::Sglang => ZmqDialect::Sglang,
            RuntimeType::Trtllm => ZmqDialect::Trtllm,
            other => {
                return Err(format!(
                    "ZMQ direct backend has no engine implementation for runtime \
                     {other}; only vllm, tokenspeed, sglang and trtllm are supported"
                )
                .into())
            }
        };

        let transport = connect_handshake(
            handshake_address,
            engine_count,
            input_address,
            output_address,
            startup,
        )
        .await?;
        // Once per engine, before its first request: an engine whose request
        // structs are positional declares their field order at the handshake,
        // and the protocol compares the prefixes its encoders fill against
        // it, so a struct that moved under this client is named here instead
        // of failing every request.
        match dialect {
            ZmqDialect::Vllm => check_engine_layouts::<VllmProtocol>(&transport)?,
            ZmqDialect::TokenSpeed => check_engine_layouts::<TokenSpeedProtocol>(&transport)?,
            ZmqDialect::Sglang => check_engine_layouts::<SglangProtocol>(&transport)?,
            ZmqDialect::Trtllm => check_engine_layouts::<TrtllmProtocol>(&transport)?,
        }
        let backend = match dialect {
            ZmqDialect::Vllm => ZmqBackend::Vllm(Arc::new(EngineCoreClient::new(transport))),
            ZmqDialect::TokenSpeed => {
                ZmqBackend::TokenSpeed(Arc::new(TokenSpeedClient::new(transport)))
            }
            ZmqDialect::Sglang => ZmqBackend::Sglang(Arc::new(SglangClient::new(transport))),
            ZmqDialect::Trtllm => ZmqBackend::Trtllm(Arc::new(TrtllmClient::new(transport))),
        };
        let client = Self {
            backend,
            meta: Arc::new(ZmqConnectionMeta {
                model_id,
                eos,
                tokenizer_eos: OnceLock::new(),
                structured_outputs: OnceLock::new(),
                supported_tasks: tokio::sync::OnceCell::new(),
                engine_truncates_long_prompts: OnceLock::new(),
                engine_places_by_rendezvous_room: OnceLock::new(),
                request_timeout: OnceLock::new(),
            }),
        };
        // Every wire bounds a prompt by the window the engine reported
        // (`crate::context_window`); an engine that reported none leaves
        // prompts unbounded before it, said once here. An encode server runs
        // no language model, takes no prompt and reports no window.
        let takes_prompts = client
            .engines()
            .iter()
            .any(|engine| engine.ready_response.takes_generate());
        if takes_prompts && client.context_window().is_ok_and(|window| window == 0) {
            tracing::warn!(
                "the engine reported no context window at the handshake (max_model_len 0): \
                 prompt lengths are not bounded before the engine"
            );
        }
        Ok(client)
    }

    /// Adopt the tokenizer's EOS ids when the connect-time model-dir lookup
    /// found none (the worker's model id is a repo id, not a local path).
    ///
    /// Without this the primary EOS id would only ride `stop_token_ids`, and
    /// an EOS finish would be reported as `matched_stop = <eos id>` — so the
    /// same model would answer differently depending on whether its files
    /// happen to be local.
    pub fn adopt_tokenizer_eos(&self, tokenizer: Option<&Arc<dyn Tokenizer>>) {
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

    /// Set what the engine enforces of a structured request: its grammar
    /// backend and the constraint kinds that backend compiles, as the
    /// servicer that launched the engine read them off its configuration.
    /// Every wire refuses a constraint outside it before the engine
    /// ([`refuse_unenforceable_constraint`]); the wire that stamps a backend
    /// on each structured request stamps this one, since such an engine keeps
    /// the backend of its first structured request for its life. A connection
    /// nobody configures (the gateway's own worker link) keeps what its wire
    /// ran before this support existed
    /// (`default_structured_outputs_support`). The first call wins.
    pub fn set_structured_outputs_support(&self, support: StructuredOutputsSupport) {
        let _ = self.meta.structured_outputs.set(support);
    }

    /// What the engine enforces of a structured request
    /// ([`Self::set_structured_outputs_support`]); until set, the wire's own
    /// default ([`default_structured_outputs_support`]).
    fn structured_outputs_support(&self) -> StructuredOutputsSupport {
        self.meta
            .structured_outputs
            .get()
            .copied()
            .unwrap_or_else(|| default_structured_outputs_support(self.dialect()))
    }

    /// Bound every generate request this connection opens from here on by
    /// `timeout`, counted from the request's dispatch over the call that
    /// opens its stream and over every message on it: a merged stream
    /// ([`ZmqGenerateStream`]) still waiting on the engine at the deadline
    /// ends there with a `DEADLINE_EXCEEDED` and its engine-side requests
    /// are aborted, as a dropped stream's are.
    /// The gateway's worker link sets its `--request-timeout-secs` here, the
    /// bound its HTTP worker client carries; a connection nobody bounds
    /// (a servicer's own link, whose caller bounds the gRPC call) leaves its
    /// streams open for as long as the caller reads them. Set once: a later
    /// call changes nothing.
    pub fn set_request_timeout(&self, timeout: Duration) {
        let _ = self.meta.request_timeout.set(timeout);
    }

    /// The engine was started to cut a long prompt down to its own bound
    /// instead of refusing it, as the launcher that started it knows: the
    /// prompt admission applies no upper bound on this connection from now
    /// (an empty prompt is still refused; see the `context_window` module).
    /// Set once: a later call changes nothing.
    pub fn note_engine_truncates_long_prompts(&self) {
        let _ = self.meta.engine_truncates_long_prompts.set(());
    }

    /// The engine places a disaggregated request (one carrying a rendezvous
    /// room) on the data-parallel rank of the room's residue, as the
    /// launcher that started it knows: the prefill side of a disaggregated
    /// pair whose engine follows the room, where the engine's KV sender
    /// fails a request that runs elsewhere and the decode side looks for the
    /// KV. A request with a room goes to that rank on this connection from
    /// now, a pin that disagrees yielding to it with a warning; a request
    /// without a room, and every request on a connection nobody notes this
    /// on (an aggregated engine, a decode side), is placed by its pin. Set
    /// once: a later call changes nothing.
    pub fn note_engine_places_by_rendezvous_room(&self) {
        let _ = self.meta.engine_places_by_rendezvous_room.set(());
    }

    /// The window the prompt admission applies on this connection: the
    /// engine's context window, or none (`0`) for an engine that cuts long
    /// prompts down itself ([`Self::note_engine_truncates_long_prompts`]).
    fn prompt_bound(&self) -> Result<u64, tonic::Status> {
        if self.meta.engine_truncates_long_prompts.get().is_some() {
            return Ok(0);
        }
        self.context_window()
    }

    /// The deadline of a generate request dispatched now, when the
    /// connection bounds its requests (see [`Self::set_request_timeout`]).
    fn request_deadline(&self) -> Option<RequestDeadline> {
        self.meta
            .request_timeout
            .get()
            .map(|timeout| RequestDeadline::after(*timeout))
    }

    /// The tasks the engine's model runner serves (vLLM's
    /// `get_supported_tasks` utility call, which its frontend makes once and
    /// validates every request against), fetched on first use.
    pub async fn supported_tasks(&self) -> Result<Arc<[String]>, tonic::Status> {
        let client = self.vllm_client("get_supported_tasks")?;
        let engine_id = client
            .engines()
            .first()
            .map(|engine| engine.engine_id.clone())
            .ok_or_else(|| tonic::Status::unavailable("no connected ZMQ engine"))?;
        self.meta
            .supported_tasks
            .get_or_try_init(|| async {
                let value = client
                    .call_utility(
                        &engine_id,
                        "get_supported_tasks",
                        Vec::new(),
                        SUPPORTED_TASKS_TIMEOUT,
                    )
                    .await
                    .map_err(utility_status)?;
                let tasks = value
                    .as_array()
                    .and_then(|tasks| {
                        tasks
                            .iter()
                            .map(OpaqueValue::as_str)
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or_else(|| {
                        tonic::Status::internal(format!(
                            "get_supported_tasks answered {value}, not a list of task names"
                        ))
                    })?;
                Ok(tasks.into_iter().map(str::to_string).collect())
            })
            .await
            .cloned()
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
            ZmqBackend::Sglang(_) => ZmqDialect::Sglang,
            ZmqBackend::Trtllm(_) => ZmqDialect::Trtllm,
        }
    }

    /// The engine runtime behind this connection, widened to the open
    /// [`RuntimeType`] for callers that report it alongside gRPC backends.
    pub fn runtime(&self) -> RuntimeType {
        match self.dialect() {
            ZmqDialect::Vllm => RuntimeType::Vllm,
            ZmqDialect::TokenSpeed => RuntimeType::TokenSpeed,
            ZmqDialect::Sglang => RuntimeType::Sglang,
            ZmqDialect::Trtllm => RuntimeType::Trtllm,
        }
    }

    /// The engines connected on the shared transport (same handshake for both
    /// protocols), each with the facts its handshake declared.
    pub fn engines(&self) -> &[ConnectedEngine] {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.engines(),
            ZmqBackend::TokenSpeed(client) => client.engines(),
            ZmqBackend::Sglang(client) => client.engines(),
            ZmqBackend::Trtllm(client) => client.engines(),
        }
    }

    /// The vLLM EngineCore client behind this connection, for RPCs only the
    /// vLLM wire defines; `operation` names the RPC in the refusal any other
    /// backend answers.
    pub(crate) fn vllm_client(&self, operation: &str) -> Result<&EngineCoreClient, tonic::Status> {
        match &self.backend {
            ZmqBackend::Vllm(client) => Ok(client),
            ZmqBackend::TokenSpeed(_) | ZmqBackend::Sglang(_) | ZmqBackend::Trtllm(_) => {
                Err(tonic::Status::unimplemented(format!(
                    "{operation} is only available on a vLLM ZMQ backend"
                )))
            }
        }
    }

    /// The SGLang scheduler client behind this connection, for what only the
    /// SGLang wire defines; `operation` names it in the refusal any other
    /// backend answers.
    pub(crate) fn sglang_client(&self, operation: &str) -> Result<&SglangClient, tonic::Status> {
        match &self.backend {
            ZmqBackend::Sglang(client) => Ok(client),
            ZmqBackend::Vllm(_) | ZmqBackend::TokenSpeed(_) | ZmqBackend::Trtllm(_) => {
                Err(tonic::Status::unimplemented(format!(
                    "{operation} is only available on an SGLang ZMQ backend"
                )))
            }
        }
    }

    /// The first connected engine's handshake `READY` response: the
    /// connection-constant engine facts (context length, dtype, parallel
    /// sizes, KV capacity) a frontend re-exposes as metadata.
    pub fn ready_response(&self) -> Option<&EngineCoreReadyResponse> {
        self.engines().first().map(|engine| &engine.ready_response)
    }

    /// The data-parallel ranks this connection serves: one connected engine
    /// per rank (a pinned request is placed on the engine of its index and
    /// refused past this count, `get_loads` names an entry per engine). The
    /// handshake's `data_parallel_size` is not this figure: ranks launched
    /// independently each report a size of one.
    pub fn engine_count(&self) -> usize {
        self.engines().len()
    }

    /// Check the disaggregation role every connected engine declared against
    /// `registered`, the role its worker is registered for (`None` for one
    /// that serves whole requests): a prefill engine registered as a decode
    /// worker, or a whole-request engine as either, fails its connection with
    /// both roles named instead of failing every request it takes. An encode
    /// worker needs the wire's encode request, so an engine that does not
    /// list it (one before the request, or a scheduler) fails the connection
    /// of an `encode` registration whatever else it declares, as the servicer
    /// refuses the same engine at its link. A prefill or encode worker's
    /// requests name its engine's bootstrap rendezvous, which on this lane
    /// only the handshake supplies (the worker's URL names the frontend's own
    /// bind, not the engine's host), so an engine that declares no bootstrap
    /// address fails the connection of either registration, whatever else it
    /// declares, instead of sending every disaggregated request out with an
    /// empty host. An engine that predates the declaration (it lists no
    /// request types) is otherwise not checked.
    pub fn check_declared_role(&self, registered: Option<&str>) -> Result<(), String> {
        for engine in self.engines() {
            let ready = &engine.ready_response;
            if registered == Some("encode") && !ready.lists_request_kind(REQUEST_KIND_ENCODE) {
                return Err(format!(
                    "engine {:?} lists no encode request on its wire (request_types {:?}) but \
                     its worker is registered as an encode worker: a newer engine, or the \
                     Python servicer (--servicer-impl python), serves this worker",
                    engine.engine_id, ready.request_types,
                ));
            }
            if !ready.request_types.is_empty() {
                let declared = ready.disaggregation_role();
                if declared != registered {
                    return Err(format!(
                        "engine {:?} declared disaggregation role {} but its worker is \
                         registered as {}",
                        engine.engine_id,
                        declared.unwrap_or("none (whole requests)"),
                        registered.unwrap_or("a regular worker"),
                    ));
                }
            }
            if let Some(role @ ("prefill" | "encode")) = registered {
                if ready.bootstrap_address().is_none() {
                    return Err(format!(
                        "engine {:?} declares no bootstrap address but its worker is registered \
                         as {role}: a {role} worker's requests name its engine's rendezvous \
                         (bootstrap_host and bootstrap_port in the handshake), which this lane \
                         cannot take from the worker's own address; an engine that declares it \
                         serves this worker",
                        engine.engine_id,
                    ));
                }
            }
        }
        Ok(())
    }

    /// The engine's context window as the ready handshake reported it (`0`
    /// from an engine that reported none), the bound every wire's prompt
    /// admission applies ([`crate::context_window`]); a connected engine is
    /// required.
    fn context_window(&self) -> Result<u64, tonic::Status> {
        self.engines()
            .first()
            .map(|engine| engine.ready_response.max_model_len)
            .ok_or_else(|| tonic::Status::unavailable("no connected ZMQ engine"))
    }

    /// Capture identities from every connected replica, or none if any replica
    /// has no identity. The first engine alone cannot identify a grouped worker.
    pub fn cache_trace_epochs(&self) -> Vec<String> {
        let engines = self.engines();
        if engines
            .iter()
            .any(|engine| engine.ready_response.cache_trace_epochs.is_empty())
        {
            return Vec::new();
        }
        let mut epochs: Vec<String> = engines
            .iter()
            .flat_map(|engine| engine.ready_response.cache_trace_epochs.iter().cloned())
            .collect();
        epochs.sort_unstable();
        epochs.dedup();
        epochs
    }

    /// The unmerged half of [`Self::generate_vllm`], one stream per choice: the
    /// `n > 1` fan-out submitted to the engine, each sub tagged with its proto
    /// `index`, left unmerged so a caller can end one choice without the
    /// others (the gRPC servicer matches string stops per choice). vLLM
    /// dialect only.
    pub async fn generate_vllm_streams(
        &self,
        req: vllm::GenerateRequest,
    ) -> Result<Vec<VllmGenerateStream>, tonic::Status> {
        self.generate_vllm_streams_with_media(req, None, None).await
    }

    /// [`generate_vllm_streams`](Self::generate_vllm_streams) for a request
    /// whose `media_refs` a worker-side processor already turned into engine
    /// features: `processed` is attached in place of the request's own
    /// multimodal batches (it must carry none). `trace_headers` is the
    /// caller's W3C trace context, set on every choice's engine request as
    /// vLLM's own frontend does for its tracer.
    pub async fn generate_vllm_streams_with_media(
        &self,
        mut req: vllm::GenerateRequest,
        processed: Option<ProcessedMedia>,
        trace_headers: Option<BTreeMap<String, String>>,
    ) -> Result<Vec<VllmGenerateStream>, tonic::Status> {
        let ZmqBackend::Vllm(client) = &self.backend else {
            return Err(tonic::Status::internal(
                "per-choice generate streams are only available on a vLLM ZMQ backend",
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
        // Media is translated once for all choices (an SHM payload is read,
        // and unlinked, a single time) and off the runtime: `/dev/shm` reads
        // and the dtype casts of multi-megabyte tensors would otherwise hold a
        // worker thread per request, starving token forwarding and health.
        let media = if let Some(processed) = processed {
            if processed.is_batches() {
                let (returned, media) = tokio::task::spawn_blocking(move || {
                    let media = translated_from_processed(&req, processed, model_dtype);
                    (req, media)
                })
                .await
                .map_err(|error| {
                    tonic::Status::internal(format!("multimodal translation failed: {error}"))
                })?;
                req = returned;
                media.map_err(tonic::Status::invalid_argument)?
            } else {
                translated_from_processed(&req, processed, model_dtype)
                    .map_err(tonic::Status::invalid_argument)?
            }
        } else if has_media(&req) {
            let (returned, media) = tokio::task::spawn_blocking(move || {
                let media = translate_media(&mut req, model_dtype);
                (req, media)
            })
            .await
            .map_err(|error| {
                tonic::Status::internal(format!("multimodal translation failed: {error}"))
            })?;
            req = returned;
            media.map_err(tonic::Status::invalid_argument)?
        } else {
            TranslatedMedia::default()
        };
        let support = self.structured_outputs_support();
        // On the request's own `n`: every sub below carries `n = 1`.
        if let Some(sp) = req.sampling_params.as_ref() {
            refuse_greedy_choices(sp).map_err(tonic::Status::invalid_argument)?;
        }
        let subs = fan_out_requests(req);
        let last = subs.len().saturating_sub(1);
        let mut media = Some(media);
        let mut streams = Vec::new();
        for (index, sub) in subs.into_iter().enumerate() {
            // The last choice takes the translated media; the others share it.
            let sub_media = if index == last {
                media.take()
            } else {
                media.clone()
            }
            .unwrap_or_default();
            // Worker-processed tensors ride as aux frames; each choice's
            // message carries them (shared bytes, not copies).
            let aux_frames = sub_media.aux_frames.clone();
            let mut request = translate_request_with_media(
                sub,
                sub_media,
                max_model_len,
                self.effective_eos(),
                support,
            )
            .map_err(tonic::Status::invalid_argument)?;
            request.trace_headers.clone_from(&trace_headers);
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
            // Sub-streams submitted before a mid-loop failure are dropped with
            // the error, which auto-aborts their engine-side requests.
            let stream = client
                .submit_with_aux(request, aux_frames)
                .await
                .map_err(zmq_status)?;
            streams.push(VllmGenerateStream::new(
                stream,
                index as u32,
                top_logprobs,
                prompt_top_logprobs,
                first_prompt_token,
            ));
        }
        Ok(streams)
    }

    /// vLLM's pre-admission rejection notice
    /// (`AsyncLLM.notify_kv_transfer_request_rejected`): an immediately
    /// aborted one-token request carrying the refused request's connector
    /// params, so the connector's `request_finished` hook runs and the
    /// prefill side frees the blocks it pinned for this decode. Failures are
    /// logged, not returned: the caller is already failing the request.
    pub async fn notify_kv_transfer_rejected(
        &self,
        request_id: &str,
        params: serde_json::Value,
        data_parallel_rank: Option<u32>,
    ) {
        let ZmqBackend::Vllm(client) = &self.backend else {
            return;
        };
        let request = EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(vec![0]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 1,
                extra_args: Some(HashMap::from([("kv_transfer_params".to_string(), params)])),
                ..EngineCoreSamplingParams::default()
            }),
            arrival_time: now_secs(),
            data_parallel_rank,
            abort_immediately: true,
            ..EngineCoreRequest::default()
        };
        // The engine finishes the request by itself; its stream is dropped.
        if let Err(error) = client.submit(request).await {
            tracing::warn!(
                request_id,
                %error,
                "could not notify the engine of a rejected KV-transfer request"
            );
        }
    }

    /// Submit a vLLM-proto generate request to a vLLM backend and return a
    /// stream of vLLM-proto responses; the request is translated into the
    /// EngineCore wire protocol here.
    ///
    /// Over gRPC the engine-side frontend (e.g. vLLM's AsyncLLM) fans `n` out
    /// itself and multiplexes the choices onto one stream. The raw ZMQ wire has
    /// no such frontend, so `n > 1` is fanned out HERE into `n` independent
    /// single-sample engine requests (see `fan_out_requests`); their outputs
    /// are merged back into one stream with each sub tagged via the proto
    /// `index` field, exactly like the gRPC contract.
    pub async fn generate_vllm(
        &self,
        req: vllm::GenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        if !matches!(self.backend, ZmqBackend::Vllm(_)) {
            return Err(tonic::Status::internal(
                "TokenSpeed ZMQ backend expects a TokenSpeed generate request",
            ));
        }
        // The deadline counts from the dispatch: before the submits.
        let deadline = self.request_deadline();
        let subs = self.generate_vllm_streams(req).await?;
        Ok(ZmqGenerateStream::vllm(subs).with_deadline(deadline))
    }

    /// [`Self::generate_vllm`] for a TokenSpeed backend and request.
    pub async fn generate_tokenspeed(
        &self,
        req: tokenspeed_proto::GenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        let deadline = self.request_deadline();
        let subs = self.generate_tokenspeed_streams(req).await?;
        Ok(ZmqGenerateStream::tokenspeed(subs).with_deadline(deadline))
    }

    /// [`Self::generate_vllm_streams`] for a TokenSpeed backend: the `n > 1`
    /// fan-out submitted to the engine, each sub tagged with its proto
    /// `index`, left unmerged so a caller can end one choice without the
    /// others (the gRPC servicer matches string stops per choice).
    /// A multimodal item's grid tensors are read with the model facts the
    /// engine declared at its handshake ([`GridTensors::declared`]); an
    /// engine that declares none keeps the conservative reading, which
    /// refuses an item with a vision grid. The Rust servicer, which has the
    /// launcher's facts, calls [`Self::generate_tokenspeed_streams_for_model`].
    pub async fn generate_tokenspeed_streams(
        &self,
        req: tokenspeed_proto::GenerateRequest,
    ) -> Result<Vec<TokenSpeedGenerateStream>, tonic::Status> {
        let grid_tensors = self
            .ready_response()
            .map_or(GridTensors::Positions, GridTensors::declared);
        self.generate_tokenspeed_streams_for_model(req, grid_tensors)
            .await
    }

    /// [`Self::generate_tokenspeed_streams`] for the model served:
    /// `grid_tensors` says what a multimodal item's grid tensors are (MRoPE
    /// positions, refused on this wire, or a patch layout that passes; see
    /// [`GridTensors`]).
    pub async fn generate_tokenspeed_streams_for_model(
        &self,
        req: tokenspeed_proto::GenerateRequest,
        grid_tensors: GridTensors,
    ) -> Result<Vec<TokenSpeedGenerateStream>, tonic::Status> {
        let ZmqBackend::TokenSpeed(client) = &self.backend else {
            return Err(tonic::Status::internal(
                "vLLM ZMQ backend expects a vLLM generate request",
            ));
        };
        // A disaggregated encode worker lists `encode` alone: it runs no
        // language model, so the gRPC servicer's words for its Generate.
        if !client
            .engines()
            .iter()
            .any(|engine| engine.ready_response.takes_generate())
        {
            return Err(tonic::Status::failed_precondition(
                "Generate is unavailable on TokenSpeed encode workers",
            ));
        }
        // A constraint the engine's grammar backend cannot enforce is refused
        // here, before anything reaches the engine, as on every wire.
        if let Some(sp) = req.sampling_params.as_ref() {
            refuse_unenforceable_constraint(
                tokenspeed_constraint_ref(sp),
                self.structured_outputs_support(),
            )
            .map_err(tonic::Status::invalid_argument)?;
        }
        let max_model_len = self.prompt_bound()?;
        // The ranks as the engine numbers them: a disaggregated request is
        // placed on a prefill engine by its room's residue, a pinned one
        // elsewhere by its pin.
        let ranks = RankIndices::of(client.engines());
        let logprobs = LogprobColumns::of(client.engines());
        // Sub-streams submitted before a mid-loop failure are dropped with the
        // error, which auto-aborts their engine-side requests.
        let mut streams = Vec::new();
        for (index, sub) in fan_out_tokenspeed_requests(req).into_iter().enumerate() {
            let request = translate_request_tokenspeed_for(
                sub,
                grid_tensors,
                max_model_len,
                &ranks,
                logprobs,
            )
            .map_err(tonic::Status::invalid_argument)?;
            let stream = client.submit(request).await.map_err(zmq_status)?;
            streams.push(TokenSpeedGenerateStream::new(stream, index as u32));
        }
        Ok(streams)
    }

    /// Hand an `Encode` RPC to a TokenSpeed encode worker: the request
    /// translated with the gRPC servicer's rules
    /// ([`translate_encode_request`]), sent under the wire's encode request
    /// type to an engine whose handshake lists `encode` (by the room's
    /// residue when several do: the embedding's rendezvous is per room, any
    /// encode worker serves it), with its large tensors as auxiliary frames.
    /// Nothing answers it: the worker ships the embedding to the prefill
    /// peer over the transfer engine, as the gRPC servicer's `Encode` acks
    /// after its send. An engine that lists no encode request (one before
    /// the wire carried it) is refused by name, nothing sent.
    pub async fn encode_tokenspeed(
        &self,
        req: smg_grpc_client::tokenspeed_encoder_proto::EncodeRequest,
        rendezvous: &EncodeRendezvous,
    ) -> Result<(), tonic::Status> {
        let ZmqBackend::TokenSpeed(client) = &self.backend else {
            return Err(tonic::Status::internal(
                "vLLM ZMQ backend expects a vLLM encode request",
            ));
        };
        let encoders: Vec<&ConnectedEngine> = client
            .engines()
            .iter()
            .filter(|engine| {
                engine
                    .ready_response
                    .lists_request_kind(REQUEST_KIND_ENCODE)
            })
            .collect();
        if encoders.is_empty() {
            return Err(tonic::Status::failed_precondition(
                "Encode is not served by this engine: its handshake lists no encode request on \
                 the msgpack wire (an engine before the encode role rode it); a newer engine, \
                 or the Python servicer (--servicer-impl python), serves this worker",
            ));
        }
        let request =
            translate_encode_request(req, rendezvous).map_err(tonic::Status::invalid_argument)?;
        let (payload, aux_frames) = encode_with_aux_frames(&request).map_err(zmq_status)?;
        let residue = request.bootstrap_room.rem_euclid(encoders.len() as i64) as usize;
        client
            .send_typed(
                &encoders[residue].engine_id,
                TokenSpeedRequestType::Encode.to_frame(),
                payload,
                aux_frames,
            )
            .await
            .map_err(zmq_status)
    }

    /// [`Self::generate_vllm`] for an SGLang backend and request.
    pub async fn generate_sglang(
        &self,
        req: sglang_proto::GenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        // This merged direct-ZMQ surface returns vLLM-proto items, which do
        // not carry selected-token scores. The Rust SGLang servicer uses the
        // unmerged streams and their native SGLang response conversion.
        if !req.token_ids_logprob.is_empty() {
            return Err(tonic::Status::unimplemented(
                "selected-token logprobs require the SGLang gRPC servicer",
            ));
        }
        let deadline = self.request_deadline();
        let subs = self.generate_sglang_streams(req).await?;
        Ok(ZmqGenerateStream::sglang(subs).with_deadline(deadline))
    }

    /// [`Self::generate_tokenspeed_streams`] for an SGLang backend: the
    /// `n > 1` fan-out submitted to the scheduler, each sub tagged with its
    /// proto `index`, left unmerged.
    pub async fn generate_sglang_streams(
        &self,
        req: sglang_proto::GenerateRequest,
    ) -> Result<Vec<SglangGenerateStream>, tonic::Status> {
        self.generate_sglang_streams_with_trace(req, None).await
    }

    /// [`generate_sglang_streams`](Self::generate_sglang_streams) with the
    /// caller's W3C trace context: `trace_headers` ride beside every choice's
    /// request (the scheduler's struct has no slot for them; the plugin in
    /// the scheduler starts the request's trace from them), so the engine's
    /// spans join the caller's trace as they do behind SGLang's own frontend.
    pub async fn generate_sglang_streams_with_trace(
        &self,
        req: sglang_proto::GenerateRequest,
        trace_headers: Option<BTreeMap<String, String>>,
    ) -> Result<Vec<SglangGenerateStream>, tonic::Status> {
        self.generate_sglang_streams_dispatching(req, trace_headers, None)
            .await
    }

    /// [`generate_sglang_streams_with_trace`](Self::generate_sglang_streams_with_trace)
    /// on a language-side worker of an encode/prefill-disaggregated
    /// deployment: a choice naming image references (and carrying no
    /// preprocessed tensors) is handed to `dispatcher` before its submit, as
    /// the engine's own frontend hands the media to the encoders, and the
    /// scheduler's request carries what the dispatch set
    /// ([`crate::sglang_encode`]). Without a dispatcher such a request is
    /// refused: this worker runs no encode dispatch. A dispatch that finds no
    /// encoder registered leaves the request without embeddings, which this
    /// worker cannot make itself: refused as well, rather than answered as
    /// text.
    pub async fn generate_sglang_streams_dispatching(
        &self,
        req: sglang_proto::GenerateRequest,
        trace_headers: Option<BTreeMap<String, String>>,
        dispatcher: Option<&Arc<dyn SglangEncodeDispatcher>>,
    ) -> Result<Vec<SglangGenerateStream>, tonic::Status> {
        self.generate_sglang_streams_dispatching_with_mm_hasher(
            req,
            trace_headers,
            dispatcher,
            None,
        )
        .await
    }

    /// As above, with bounded off-runtime CPU feature hashing. The caller must
    /// opt in only for an engine using the SHA256 CPU identity algorithm, not
    /// its GPU hash buffer or skip-hash path. This is a local option, never an
    /// identity supplied by an untrusted RPC caller.
    pub async fn generate_sglang_streams_dispatching_with_mm_hasher(
        &self,
        req: sglang_proto::GenerateRequest,
        trace_headers: Option<BTreeMap<String, String>>,
        dispatcher: Option<&Arc<dyn SglangEncodeDispatcher>>,
        hasher: Option<&crate::SglangCpuMmHasher>,
    ) -> Result<Vec<SglangGenerateStream>, tonic::Status> {
        let ZmqBackend::Sglang(client) = &self.backend else {
            return Err(tonic::Status::internal(
                "this ZMQ backend is not an SGLang scheduler",
            ));
        };
        // A constraint the engine's grammar backend cannot enforce is refused
        // here, before anything reaches the engine, as on every wire.
        if let Some(sp) = req.sampling_params.as_ref() {
            refuse_unenforceable_constraint(
                sglang_constraint_ref(sp),
                self.structured_outputs_support(),
            )
            .map_err(tonic::Status::invalid_argument)?;
        }
        let max_model_len = self.prompt_bound()?;
        // The connected scheduler ranks: a request is placed among them as
        // the engine's own dispatcher would place it, by its data-parallel
        // pin, or on a prefill side that follows the room by its room's
        // residue ([`Self::note_engine_places_by_rendezvous_room`]).
        let ranks = client.engines().len();
        let placement = if self.meta.engine_places_by_rendezvous_room.get().is_some() {
            RankPlacement::RoomResidue
        } else {
            RankPlacement::Pin
        };
        let mut streams = Vec::new();
        for (index, sub) in fan_out_sglang_requests(req).into_iter().enumerate() {
            let image_refs = sub
                .mm_inputs
                .as_ref()
                .map(encode_media_refs)
                .transpose()
                .map_err(tonic::Status::invalid_argument)?
                .flatten();
            let mut request = translate_request_sglang(sub, max_model_len, ranks, placement)
                .map_err(tonic::Status::invalid_argument)?;
            request.trace_headers.clone_from(&trace_headers);
            if let (Some(hasher), Some(mm)) = (hasher, request.mm_inputs.as_mut()) {
                hasher.precompute(mm, &request.rid).await?;
            }
            let dispatched = match (image_refs, dispatcher) {
                (None, _) => None,
                (Some(_), None) => {
                    return Err(tonic::Status::failed_precondition(
                        "multimodal payload carries media references but no preprocessed \
                         tensors, and this worker runs no encode dispatch (not a language-only \
                         worker of an encode/prefill-disaggregated deployment)",
                    ));
                }
                (Some(image_urls), Some(dispatcher)) => {
                    let notice = EncodeNotice::new(Arc::clone(client));
                    let outcome = dispatcher
                        .dispatch(EncodeDispatchRequest {
                            rid: request.rid.clone(),
                            image_urls,
                            on_error: notice.sink(),
                        })
                        .await
                        .map_err(|error| {
                            tonic::Status::internal(format!("encode dispatch failed: {error}"))
                        })?;
                    if !outcome.need_wait_for_mm_inputs {
                        // The engine's frontend would process the media
                        // locally; a language-side worker has no encoder of
                        // its own and this servicer never pretends to.
                        if let Some(sent) = outcome.sent {
                            sent.run().await;
                        }
                        return Err(tonic::Status::failed_precondition(
                            "no encoder is registered with this language-only worker: the \
                             request's media cannot be encoded",
                        ));
                    }
                    request.need_wait_for_mm_inputs = Some(true);
                    request.num_items_assigned = outcome.num_items_assigned;
                    request.encoder_urls = outcome.encoder_urls;
                    Some((notice, outcome.sent))
                }
            };
            let submitted = client.submit(SglangRequest::Generate(request)).await;
            if let Some((notice, sent)) = dispatched {
                // The rank first, then the dispatcher's release: a failure it
                // reports from here on finds the rank that holds the request.
                // A submit that failed releases the dispatcher the same way;
                // its failures are dropped (nothing holds the request).
                if let Ok(stream) = submitted.as_ref() {
                    if let Some(rank) = stream.engine_index() {
                        notice.sent_to(rank);
                    }
                }
                if let Some(sent) = sent {
                    sent.run().await;
                }
            }
            let stream = submitted.map_err(zmq_status)?;
            streams.push(SglangGenerateStream::new(stream, index as u32));
        }
        Ok(streams)
    }

    /// Submit an SGLang-proto embed request to an SGLang scheduler and await
    /// its pooled vector: the scheduler answers an embedding request with one
    /// finished output. A scheduler abort is the caller's error when it
    /// carries SGLang's 400, as a generate abort is.
    pub async fn embed_sglang(
        &self,
        req: sglang_proto::EmbedRequest,
    ) -> Result<sglang_proto::EmbedResponse, tonic::Status> {
        let client = self.sglang_client("Embed")?;
        let request =
            translate_embed_request_sglang(req).map_err(tonic::Status::invalid_argument)?;
        let request_id = request.rid.clone();
        // Dropped before the finishing output (an error below, or the caller
        // giving up), the stream aborts the scheduler-side request.
        let mut stream = client
            .submit(SglangRequest::Embed(request))
            .await
            .map_err(zmq_status)?;
        while let Some(output) = stream.next().await {
            let output = output.map_err(zmq_status)?;
            let Some(reason) = output.finish_reason.as_deref() else {
                continue;
            };
            if reason == "abort" {
                let message = format!(
                    "the scheduler aborted the request: {}",
                    output
                        .finish_message
                        .as_deref()
                        .unwrap_or("no reason reported")
                );
                return Err(crate::abort_status::engine_abort_status(
                    output.finish_status,
                    message,
                ));
            }
            let embedding = output.embedding.unwrap_or_default();
            return Ok(sglang_proto::EmbedResponse {
                embedding_dim: u32::try_from(embedding.len()).unwrap_or(u32::MAX),
                embedding,
                prompt_tokens: output.prompt_tokens,
            });
        }
        Err(tonic::Status::internal(format!(
            "Embed request {request_id} did not produce a result"
        )))
    }

    /// Submit a `TrtllmService` generate request to the engine's shim and
    /// stream back vLLM-proto responses, the `n > 1` fan-out merged.
    pub async fn generate_trtllm(
        &self,
        req: trtllm_proto::GenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        let deadline = self.request_deadline();
        let subs = self.generate_trtllm_streams(req, None).await?;
        Ok(ZmqGenerateStream::trtllm(subs).with_deadline(deadline))
    }

    /// [`Self::generate_sglang_streams_with_trace`] for a TensorRT-LLM
    /// backend: the `n > 1` fan-out submitted to the shim, each sub tagged
    /// with its proto `sequence_index`, left unmerged. `trace_headers` is the
    /// caller's W3C trace context, carried on each sub's request for the
    /// engine's own spans.
    pub async fn generate_trtllm_streams(
        &self,
        req: trtllm_proto::GenerateRequest,
        trace_headers: Option<BTreeMap<String, String>>,
    ) -> Result<Vec<TrtllmGenerateStream>, tonic::Status> {
        let ZmqBackend::Trtllm(client) = &self.backend else {
            return Err(tonic::Status::internal(
                "this ZMQ backend is not a TensorRT-LLM engine",
            ));
        };
        // A constraint the engine's grammar backend cannot enforce is refused
        // here, before anything reaches the engine, as on every wire.
        refuse_unenforceable_constraint(
            trtllm_constraint_ref(&req),
            self.structured_outputs_support(),
        )
        .map_err(tonic::Status::invalid_argument)?;
        let max_model_len = self.prompt_bound()?;
        let mut streams = Vec::new();
        for (index, sub) in fan_out_trtllm_requests(req).into_iter().enumerate() {
            let mut request = translate_request_trtllm(sub, max_model_len)
                .map_err(tonic::Status::invalid_argument)?;
            request.trace_headers.clone_from(&trace_headers);
            let stream = client.submit(request).await.map_err(zmq_status)?;
            streams.push(TrtllmGenerateStream::new(stream, index as u32));
        }
        Ok(streams)
    }

    /// Whether the engine's wire carries control calls (the cache flush and
    /// the profiler): an SGLang scheduler's always does, a TokenSpeed one's
    /// when every rank declared the control reply, a vLLM engine's never.
    pub fn carries_control_calls(&self) -> bool {
        matches!(&self.backend, ZmqBackend::Sglang(_))
            || self.tokenspeed_wire_carries_control_calls()
    }

    /// Whether every connected TokenSpeed rank declared the control reply at
    /// its handshake (the control-wire revision), so the scheduler's control
    /// RPCs can be carried to it; false on a scheduler that declares none and
    /// on the other backends.
    pub fn tokenspeed_wire_carries_control_calls(&self) -> bool {
        matches!(&self.backend, ZmqBackend::TokenSpeed(_))
            && self
                .engines()
                .iter()
                .all(|engine| wire_carries_control_calls(&engine.ready_response.positional_layouts))
    }

    /// One control call (`rpc`, as `method(*args)` on the wire) on every
    /// connected scheduler rank, each rank's `(success, message)` in engine
    /// order: the SGLang plugin, or a TokenSpeed scheduler of the control-wire
    /// revision, dispatches it to the scheduler's own handler. The wait is per
    /// rank (DEADLINE_EXCEEDED); a dead engine is UNAVAILABLE. A TokenSpeed
    /// scheduler whose handshake declares no control reply answers
    /// UNIMPLEMENTED naming the gap, as does a vLLM engine, whose own RPCs
    /// serve its admin.
    async fn control_calls(
        &self,
        rpc: &str,
        method: &str,
        args: Vec<OpaqueValue>,
        wait: Duration,
    ) -> Result<Vec<(bool, String)>, tonic::Status> {
        match &self.backend {
            ZmqBackend::Sglang(client) => control_on_every_rank(client, method, args, wait).await,
            ZmqBackend::TokenSpeed(client) => {
                if !self.tokenspeed_wire_carries_control_calls() {
                    return Err(tonic::Status::unimplemented(format!(
                        "{rpc} is not available through the Rust TokenSpeed servicer: the \
                         scheduler's msgpack wire carries generate and abort only (its \
                         handshake declares no control reply)"
                    )));
                }
                control_on_every_rank(client, method, args, wait).await
            }
            ZmqBackend::Vllm(_) | ZmqBackend::Trtllm(_) => Err(tonic::Status::unimplemented(
                format!("{rpc} is only available on an SGLang or TokenSpeed ZMQ backend"),
            )),
        }
    }

    /// The scheduler's cache flush (`FlushCacheReqInput(timeout_s)` on
    /// SGLang; the control-wire revision's `flush_cache` on TokenSpeed) on
    /// every rank: the scheduler flushes at once when idle, or waits up to
    /// `timeout_s` for idle when positive; `wait` bounds the reply.
    /// Aggregated as the Python servicer aggregates its per-rank communicator
    /// results.
    pub async fn flush_cache_control(
        &self,
        timeout_s: f32,
        wait: Duration,
    ) -> Result<(bool, String), tonic::Status> {
        let results = self
            .control_calls(
                "FlushCache",
                "flush_cache",
                vec![OpaqueValue::F64(f64::from(timeout_s))],
                wait,
            )
            .await?;
        Ok(aggregate_control(&results, "Cache flushed successfully"))
    }

    /// The scheduler's profiler on every rank (`ProfileReq` on SGLang; the
    /// control-wire revision's `start_profile` / `stop_profile` on
    /// TokenSpeed): start with the given options, or stop (`None`) and export
    /// the traces.
    pub async fn profile_control(
        &self,
        start: Option<SglangProfileStart>,
        wait: Duration,
    ) -> Result<(bool, String), tonic::Status> {
        let (rpc, method, args, ok_message) = match start {
            Some(start) => (
                "StartProfile",
                "start_profile",
                vec![start.into_opaque()],
                "Start profiling succeeded",
            ),
            None => (
                "StopProfile",
                "stop_profile",
                Vec::new(),
                "Stop profiling succeeded",
            ),
        };
        let results = self.control_calls(rpc, method, args, wait).await?;
        Ok(aggregate_control(&results, ok_message))
    }

    /// One RL or weight operation on every connected rank: the TokenSpeed
    /// scheduler's control call `method` (`pause_generation`,
    /// `update_weights_from_distributed`, ...), its argument `body`, the JSON
    /// body of the engine's HTTP route of the same name (none when `None`).
    /// Each rank's `(success, message)`, in engine order; `wait` bounds each
    /// rank's reply (DEADLINE_EXCEEDED). Only a TokenSpeed scheduler of the
    /// control-wire revision serves these calls; any other engine answers
    /// UNIMPLEMENTED naming the gap.
    pub async fn rl_control_call(
        &self,
        method: &str,
        body: Option<serde_json::Value>,
        wait: Duration,
    ) -> Result<Vec<(bool, String)>, tonic::Status> {
        if !matches!(&self.backend, ZmqBackend::TokenSpeed(_)) {
            return Err(tonic::Status::unimplemented(format!(
                "{method} is served over the engine connection by a TokenSpeed scheduler only"
            )));
        }
        if !self.tokenspeed_wire_carries_control_calls() {
            return Err(tonic::Status::unimplemented(format!(
                "{method} cannot ride this engine's connection: its handshake declares no \
                 control reply, so the msgpack wire carries generate and abort only"
            )));
        }
        let args = body.map(json_to_msgpack).into_iter().collect();
        self.control_calls(method, method, args, wait).await
    }

    /// Local liveness: false once the connection observed `ENGINE_CORE_DEAD` or
    /// a transport failure. No RPC (the raw ZMQ wire has no health RPC).
    pub fn is_alive(&self) -> bool {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.is_alive(),
            ZmqBackend::TokenSpeed(client) => client.is_alive(),
            ZmqBackend::Sglang(client) => client.is_alive(),
            ZmqBackend::Trtllm(client) => client.is_alive(),
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

    /// vLLM's `reset_prefix_cache(reset_running_requests=False,
    /// reset_connector=False)` on every connected rank, as the Python
    /// servicer's `admin.py` issues it per rank: `true` only when every rank
    /// reset. The wait is bounded by `timeout` (DEADLINE_EXCEEDED); an engine
    /// failure is INTERNAL, a dead engine UNAVAILABLE. TokenSpeed has no such
    /// RPC on the ZMQ wire.
    pub async fn reset_prefix_cache(&self, timeout: Duration) -> Result<bool, tonic::Status> {
        let ZmqBackend::Vllm(client) = &self.backend else {
            return Err(tonic::Status::unimplemented(
                "FlushCache is only available on a vLLM ZMQ backend",
            ));
        };
        let args = vec![OpaqueValue::from(false), OpaqueValue::from(false)];
        let calls = client.engines().iter().map(|engine| {
            client.call_utility(
                &engine.engine_id,
                "reset_prefix_cache",
                args.clone(),
                timeout,
            )
        });
        let results = futures::future::try_join_all(calls)
            .await
            .map_err(utility_status)?;
        Ok(results.iter().all(|result| result.as_bool() == Some(true)))
    }

    /// vLLM's `profile(is_start)` on every connected rank, as the Python
    /// servicer's `AsyncLLM.start_profile` / `stop_profile` issue it: the
    /// engine's own torch profiler, configured at its launch, starts, or
    /// stops and exports its traces. `Ok` once every rank answered; the wait
    /// is bounded by `timeout` (DEADLINE_EXCEEDED); an engine failure (a
    /// profiler the engine was launched without) is INTERNAL with the
    /// engine's words, a dead engine UNAVAILABLE. TokenSpeed and SGLang have
    /// control calls of their own ([`Self::profile_control`]).
    pub async fn profile_vllm(
        &self,
        is_start: bool,
        timeout: Duration,
    ) -> Result<(), tonic::Status> {
        let ZmqBackend::Vllm(client) = &self.backend else {
            return Err(tonic::Status::unimplemented(
                "StartProfile and StopProfile are only available on a vLLM ZMQ backend",
            ));
        };
        let args = vec![OpaqueValue::from(is_start)];
        let calls = client
            .engines()
            .iter()
            .map(|engine| client.call_utility(&engine.engine_id, "profile", args.clone(), timeout));
        futures::future::try_join_all(calls)
            .await
            .map_err(utility_status)?;
        Ok(())
    }

    /// Latest per-rank load for one engine index, if the backend has reported
    /// it: every dialect piggybacks it on its output batches.
    fn engine_load(&self, engine_index: u32) -> Option<EngineLoad> {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.engine_load(engine_index),
            ZmqBackend::TokenSpeed(client) => client.engine_load(engine_index),
            ZmqBackend::Sglang(client) => client.engine_load(engine_index),
            ZmqBackend::Trtllm(client) => client.engine_load(engine_index),
        }
    }

    /// The engine's silence on the requests in flight on it (the ZMQ client's
    /// `output_silence`): the time since its last output of any kind, `None`
    /// while nothing is in flight.
    pub fn output_silence(&self) -> Option<Duration> {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.output_silence(),
            ZmqBackend::TokenSpeed(client) => client.output_silence(),
            ZmqBackend::Sglang(client) => client.output_silence(),
            ZmqBackend::Trtllm(client) => client.output_silence(),
        }
    }

    /// Whether the engine's wire has shown a scheduler step without any
    /// token, so its silence is the scheduler's own (the ZMQ client's
    /// `reports_steps`).
    pub fn reports_steps(&self) -> bool {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.reports_steps(),
            ZmqBackend::TokenSpeed(client) => client.reports_steps(),
            ZmqBackend::Sglang(client) => client.reports_steps(),
            ZmqBackend::Trtllm(client) => client.reports_steps(),
        }
    }

    /// The engine's wire reports every scheduler step, as the launcher that
    /// started the engine knows: `reports_steps` holds from now rather than
    /// from the first stats-only step batch (see the client's own
    /// `note_reports_steps`).
    pub fn note_reports_steps(&self) {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.note_reports_steps(),
            ZmqBackend::TokenSpeed(client) => client.note_reports_steps(),
            ZmqBackend::Sglang(client) => client.note_reports_steps(),
            ZmqBackend::Trtllm(client) => client.note_reports_steps(),
        }
    }

    /// Prompt tokens of the in-flight requests without a first output yet
    /// (the ZMQ client's `prefill_pending_tokens`).
    pub fn prefill_pending_tokens(&self) -> u64 {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.prefill_pending_tokens(),
            ZmqBackend::TokenSpeed(client) => client.prefill_pending_tokens(),
            ZmqBackend::Sglang(client) => client.prefill_pending_tokens(),
            ZmqBackend::Trtllm(client) => client.prefill_pending_tokens(),
        }
    }

    /// Bound how long the engine may hold requests in flight, in total since
    /// its last output, with nothing pending prefill, before it is treated as
    /// dead (the ZMQ client's `set_silence_death_timeout`).
    pub fn set_silence_death_timeout(&self, timeout: Duration) {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.set_silence_death_timeout(timeout),
            ZmqBackend::TokenSpeed(client) => client.set_silence_death_timeout(timeout),
            ZmqBackend::Sglang(client) => client.set_silence_death_timeout(timeout),
            ZmqBackend::Trtllm(client) => client.set_silence_death_timeout(timeout),
        }
    }

    /// The prefill rate the silence bound allows the engine for the prompt
    /// tokens pending prefill; zero allows nothing (the ZMQ client's
    /// `set_prefill_tokens_per_sec`).
    pub fn set_prefill_tokens_per_sec(&self, tokens_per_sec: u64) {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.set_prefill_tokens_per_sec(tokens_per_sec),
            ZmqBackend::TokenSpeed(client) => client.set_prefill_tokens_per_sec(tokens_per_sec),
            ZmqBackend::Sglang(client) => client.set_prefill_tokens_per_sec(tokens_per_sec),
            ZmqBackend::Trtllm(client) => client.set_prefill_tokens_per_sec(tokens_per_sec),
        }
    }

    /// Per-rank load from the piggybacked scheduler stats (SMG's DP routing
    /// signal), in the same shape as the gRPC `GetLoads` response. A rank that
    /// has not reported yet has no entry.
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
    pub fn model_info(&self) -> ZmqModelInfo {
        let max_context_length = self
            .engines()
            .first()
            .map(|e| e.ready_response.max_model_len)
            .unwrap_or(0);
        match &self.backend {
            ZmqBackend::Vllm(_) => ZmqModelInfo::Vllm(vllm::GetModelInfoResponse {
                model_path: self.meta.model_id.clone(),
                served_model_name: self.meta.model_id.clone(),
                tokenizer_path: self.meta.model_id.clone(),
                is_generation: true,
                max_context_length: u32::try_from(max_context_length).unwrap_or(u32::MAX),
                ..Default::default()
            }),
            ZmqBackend::TokenSpeed(_) => {
                ZmqModelInfo::TokenSpeed(Box::new(tokenspeed_proto::GetModelInfoResponse {
                    model_path: self.meta.model_id.clone(),
                    served_model_name: self.meta.model_id.clone(),
                    tokenizer_path: self.meta.model_id.clone(),
                    max_context_length: i32::try_from(max_context_length).unwrap_or(i32::MAX),
                    ..Default::default()
                }))
            }
            ZmqBackend::Sglang(_) => {
                ZmqModelInfo::Sglang(Box::new(sglang_proto::GetModelInfoResponse {
                    model_path: self.meta.model_id.clone(),
                    served_model_name: self.meta.model_id.clone(),
                    tokenizer_path: self.meta.model_id.clone(),
                    is_generation: true,
                    max_context_length: i32::try_from(max_context_length).unwrap_or(i32::MAX),
                    ..Default::default()
                }))
            }
            // The handshake's window bounds the sequence; the proto's input
            // bound follows it here (the servicer's facts refine both).
            ZmqBackend::Trtllm(_) => {
                let max_seq_len = i32::try_from(max_context_length).unwrap_or(i32::MAX);
                ZmqModelInfo::Trtllm(Box::new(trtllm_proto::GetModelInfoResponse {
                    model_id: self.meta.model_id.clone(),
                    max_input_len: max_seq_len,
                    max_seq_len,
                    ..Default::default()
                }))
            }
        }
    }

    /// Server info derived from the handshake response, as the runtime's native
    /// metadata variant.
    pub fn server_info(&self) -> ZmqServerInfo {
        let data_parallel_size = self
            .engines()
            .first()
            .map(|e| e.ready_response.data_parallel_size)
            .unwrap_or(1);
        match &self.backend {
            ZmqBackend::Vllm(_) => ZmqServerInfo::Vllm(Box::new(vllm::GetServerInfoResponse {
                data_parallel_size: i32::try_from(data_parallel_size).unwrap_or(i32::MAX),
                server_type: "vllm".to_string(),
                ..Default::default()
            })),
            // TokenSpeed's server-info proto carries no data-parallel size or
            // server-type field; the ZMQ handshake supplies no `server_args`
            // either, so only the fields it does expose are surfaced.
            ZmqBackend::TokenSpeed(_) => {
                ZmqServerInfo::TokenSpeed(Box::<tokenspeed_proto::GetServerInfoResponse>::default())
            }
            // The handshake carries the version (as `vllm_version`) and the KV
            // capacity; `server_args` has no source on this wire.
            ZmqBackend::Sglang(_) => {
                let ready = self.ready_response();
                ZmqServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
                    // The plugin reports `sglang-<version>` in the shared
                    // field; the gRPC servicer reports the bare version.
                    sglang_version: ready
                        .map(|r| {
                            r.vllm_version
                                .strip_prefix("sglang-")
                                .unwrap_or(&r.vllm_version)
                                .to_string()
                        })
                        .unwrap_or_default(),
                    server_type: "zmq".to_string(),
                    max_total_num_tokens: ready
                        .and_then(|r| r.kv_cache_size_tokens)
                        .and_then(|n| i32::try_from(n).ok())
                        .unwrap_or(0),
                    ..Default::default()
                }))
            }
            // The handshake carries the version (as `vllm_version`, prefixed
            // by the shim) and the parallel sizes.
            ZmqBackend::Trtllm(_) => {
                let ready = self.ready_response();
                ZmqServerInfo::Trtllm(Box::new(trtllm_proto::GetServerInfoResponse {
                    version: ready
                        .map(|r| {
                            r.vllm_version
                                .strip_prefix("trtllm-")
                                .unwrap_or(&r.vllm_version)
                                .to_string()
                        })
                        .unwrap_or_default(),
                    backend: "tensorrt-llm".to_string(),
                    tensor_parallel_size: ready.map_or(1, |r| {
                        i32::try_from(r.tensor_parallel_size).unwrap_or(i32::MAX)
                    }),
                    pipeline_parallel_size: ready.map_or(1, |r| {
                        i32::try_from(r.pipeline_parallel_size).unwrap_or(i32::MAX)
                    }),
                    context_parallel_size: ready.map_or(1, |r| {
                        i32::try_from(r.decode_context_parallel_size).unwrap_or(i32::MAX)
                    }),
                    world_size: ready
                        .map_or(1, |r| i32::try_from(r.world_size).unwrap_or(i32::MAX)),
                    // The handshake does not say which structural-tag
                    // vocabulary the engine's grammar compiler takes.
                    structural_tag_vocabulary: String::new(),
                }))
            }
        }
    }
}

pub(crate) fn zmq_status(error: engine_zmq_client::Error) -> tonic::Status {
    zmq_status_of(&error)
}

/// The status of a connector error, by reference: the connector fans the
/// failure of its engine out to every request in flight as the one error
/// shared (`Error::Shared`), and a shared error answers as the bare one.
fn zmq_status_of(error: &engine_zmq_client::Error) -> tonic::Status {
    match error {
        engine_zmq_client::Error::Shared(inner) => zmq_status_of(inner),
        // The engine is gone, is not taking input right now, or the
        // connector to it is shutting down (its engine died, or the servicer
        // drains): either way this request is better served elsewhere.
        engine_zmq_client::Error::EngineCoreDead
        | engine_zmq_client::Error::EngineInputBlocked { .. }
        | engine_zmq_client::Error::EngineInputFull { .. }
        | engine_zmq_client::Error::ClientClosed { .. } => {
            tonic::Status::unavailable(error.to_string())
        }
        other => tonic::Status::internal(other.to_string()),
    }
}

/// `method(*args)` on every rank of `client`, each rank's `(success,
/// message)` in engine order; the wait is per rank.
async fn control_on_every_rank<P: EngineProtocol>(
    client: &Client<P>,
    method: &str,
    args: Vec<OpaqueValue>,
    wait: Duration,
) -> Result<Vec<(bool, String)>, tonic::Status> {
    let calls = client
        .engines()
        .iter()
        .map(|engine| client.call_utility(&engine.engine_id, method, args.clone(), wait));
    let replies = futures::future::try_join_all(calls)
        .await
        .map_err(utility_status)?;
    Ok(replies.iter().map(control_outcome).collect())
}

/// A JSON value as the msgpack value the engine's control call reads: maps
/// keep their string keys, an integer stays an integer.
fn json_to_msgpack(value: serde_json::Value) -> OpaqueValue {
    match value {
        serde_json::Value::Null => OpaqueValue::Nil,
        serde_json::Value::Bool(value) => OpaqueValue::Boolean(value),
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                OpaqueValue::from(value)
            } else if let Some(value) = number.as_u64() {
                OpaqueValue::from(value)
            } else {
                OpaqueValue::F64(number.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(value) => OpaqueValue::from(value),
        serde_json::Value::Array(items) => {
            OpaqueValue::Array(items.into_iter().map(json_to_msgpack).collect())
        }
        serde_json::Value::Object(fields) => OpaqueValue::Map(
            fields
                .into_iter()
                .map(|(key, value)| (OpaqueValue::from(key), json_to_msgpack(value)))
                .collect(),
        ),
    }
}

/// A utility call's error as a status: the wait expiring is the caller's
/// deadline, everything else is the engine failing.
fn utility_status(error: engine_zmq_client::Error) -> tonic::Status {
    match error {
        engine_zmq_client::Error::UtilityTimeout { .. } => {
            tonic::Status::deadline_exceeded(error.to_string())
        }
        engine_zmq_client::Error::UtilityUnsupported { .. } => {
            tonic::Status::unimplemented(error.to_string())
        }
        other => zmq_status(other),
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Arc, time::Duration};

    use engine_zmq_client::{
        mock_engine::{
            connect_to_frontend, default_ready_response, hello, EngineInbound, MockEngine,
        },
        protocol::vllm::request::UtilityCall,
        EngineId,
    };
    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
    use openai_protocol::worker::RuntimeType;
    use tonic::Code;

    use super::*;
    use crate::{eos::EosTokenIds, test_support::connected_pair};

    /// A route's JSON body becomes the control call's msgpack argument field
    /// by field: integers stay integers (one above `i64` included), floats
    /// floats, strings, booleans and nulls their own kind; arrays and nested
    /// objects recurse.
    #[test]
    fn a_json_body_becomes_a_msgpack_argument_field_by_field() {
        let value = json_to_msgpack(serde_json::json!({
            "steps": 3,
            "big": u64::MAX,
            "neg": -2,
            "ratio": 0.25,
            "name": "w",
            "flag": false,
            "none": null,
            "list": [1, 2.5, "x"],
            "inner": {"k": [true]}
        }));
        let fields = value.as_map().expect("a map");
        let field = |key: &str| {
            fields
                .iter()
                .find(|(name, _)| name.as_str() == Some(key))
                .map(|(_, value)| value)
                .unwrap_or_else(|| panic!("{key}"))
        };
        assert_eq!(*field("steps"), OpaqueValue::from(3i64));
        assert_eq!(*field("big"), OpaqueValue::from(u64::MAX));
        assert_eq!(*field("neg"), OpaqueValue::from(-2i64));
        assert_eq!(*field("ratio"), OpaqueValue::F64(0.25));
        assert_eq!(*field("name"), OpaqueValue::from("w"));
        assert_eq!(*field("flag"), OpaqueValue::Boolean(false));
        assert_eq!(*field("none"), OpaqueValue::Nil);
        assert_eq!(
            *field("list"),
            OpaqueValue::Array(vec![
                OpaqueValue::from(1i64),
                OpaqueValue::F64(2.5),
                OpaqueValue::from("x"),
            ])
        );
        let inner = field("inner").as_map().expect("a nested map");
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].0.as_str(), Some("k"));
        assert_eq!(
            inner[0].1,
            OpaqueValue::Array(vec![OpaqueValue::Boolean(true)])
        );
    }

    /// A request that reaches the connector once it is shutting down (its
    /// engine died and the servicer is on its way out) is answered the way a
    /// dead engine core answers it, UNAVAILABLE, so the gateway retries it on
    /// another worker instead of returning the engine's INTERNAL to the
    /// client; the engine's own failures keep their INTERNAL.
    #[test]
    fn a_closing_connector_answers_unavailable_like_a_dead_engine_core() {
        let closed = zmq_status(engine_zmq_client::Error::ClientClosed {
            message: "client is shutting down".to_string(),
        });
        assert_eq!(closed.code(), Code::Unavailable, "{closed:?}");
        assert!(
            closed.message().contains("client is shutting down"),
            "{closed:?}"
        );
        assert_eq!(
            zmq_status(engine_zmq_client::Error::EngineCoreDead).code(),
            Code::Unavailable
        );
        let engine_failure = zmq_status(engine_zmq_client::Error::RequestStreamClosed {
            request_id: "req-1".to_string(),
        });
        assert_eq!(engine_failure.code(), Code::Internal, "{engine_failure:?}");
    }

    /// The failure the connector fans out to the requests in flight when its
    /// engine dies (`RequestRegistry::fail_all`) reaches them shared; a
    /// request that had streamed nothing is as well served elsewhere as one
    /// that arrived after the close.
    #[test]
    fn a_fanned_out_failure_answers_as_the_bare_one_does() {
        let shared = |error: engine_zmq_client::Error| {
            zmq_status(engine_zmq_client::Error::Shared(Arc::new(error)))
        };
        assert_eq!(
            shared(engine_zmq_client::Error::EngineCoreDead).code(),
            Code::Unavailable
        );
        let ended = shared(engine_zmq_client::Error::ClientClosed {
            message: "engine output stream ended".to_string(),
        });
        assert_eq!(ended.code(), Code::Unavailable, "{ended:?}");
        assert!(
            ended.message().contains("engine output stream ended"),
            "{ended:?}"
        );
        let nested = zmq_status(engine_zmq_client::Error::Shared(Arc::new(
            engine_zmq_client::Error::Shared(Arc::new(
                engine_zmq_client::Error::RequestStreamClosed {
                    request_id: "req-2".to_string(),
                },
            )),
        )));
        assert_eq!(nested.code(), Code::Internal, "{nested:?}");
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

    /// The dialect only its servicer speaks is refused on the gateway's
    /// connect path before any handshake: a worker registered on it would
    /// otherwise connect, count healthy and fail every request. The
    /// servicer's own connect path keeps taking it.
    #[tokio::test]
    async fn the_gateway_connect_path_refuses_the_servicer_only_dialect() {
        let started = std::time::Instant::now();
        let err = match connect_for_worker(
            "ipc:///tmp/no-such-worker",
            "m".to_string(),
            RuntimeType::Trtllm,
            None,
            1,
            Duration::from_secs(5),
            None,
        )
        .await
        {
            Ok(_) => panic!("the gateway's connect path accepted the servicer-only dialect"),
            Err(err) => err,
        };
        assert!(err.contains("direct ZMQ lane"), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "refused before the handshake"
        );
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

    /// The gateway's connect path across a retry: the first handshake greets
    /// the engine and gives up its READY wait while the engine is still
    /// loading (the startup bound is shorter than the engine's init); the
    /// retry binds the same addresses again, and the engine's one READY and
    /// one registration reach it. The retry takes the engine as its own: an
    /// engine says HELLO once, so a wait for another would strand it.
    #[tokio::test]
    async fn a_retried_connect_takes_the_engine_the_first_handshake_greeted() {
        let dir = tempfile::tempdir().unwrap();
        let base_url = format!("ipc://{}", dir.path().join("w").display());
        let handshake = format!("ipc://{}", dir.path().join("hs.sock").display());

        let (gave_up, engine) = tokio::join!(
            connect_with_eos(
                &base_url,
                "m".to_string(),
                RuntimeType::Vllm,
                Handshake::TcpOrIpc(&handshake),
                1,
                EosTokenIds::default(),
                Duration::from_secs(1),
            ),
            hello(&handshake, EngineId::from_engine_index(0)),
        );
        let Err(reason) = gave_up else {
            panic!("the first connect must give up its READY wait");
        };
        assert!(reason.contains("waiting for READY"), "{reason}");
        let engine = engine.expect("HELLO/INIT with the first connect");
        // The dropped listeners' socket files go on their own tasks; the
        // gateway retries a health probe later.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let (client, engine) = tokio::join!(
            connect_with_eos(
                &base_url,
                "m".to_string(),
                RuntimeType::Vllm,
                Handshake::TcpOrIpc(&handshake),
                1,
                EosTokenIds::default(),
                Duration::from_secs(10),
            ),
            async {
                engine
                    .redial(&handshake)
                    .await?
                    .ready(default_ready_response())
                    .await
            },
        );
        let client = client.expect("the retry connects the engine");
        engine.expect("mock engine connected");
        assert_eq!(client.dialect(), ZmqDialect::Vllm);
    }

    /// A `tcp://` worker serves engines on other hosts: they dial its own
    /// address, and the INIT hands them data-plane addresses on the same
    /// host with the ports the gateway bound.
    #[tokio::test]
    async fn a_tcp_worker_connects_its_engines_through_its_own_address() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let base_url = format!("tcp://127.0.0.1:{port}");
        let (client, engine) = tokio::join!(
            connect_with_eos(
                &base_url,
                "m".to_string(),
                RuntimeType::TokenSpeed,
                Handshake::Derived,
                1,
                EosTokenIds::default(),
                Duration::from_secs(10),
            ),
            connect_to_frontend(
                &base_url,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("the tcp worker connects");
        let engine = engine.expect("mock engine connected");
        assert_eq!(client.engine_count(), 1);
        for address in engine
            .init
            .addresses
            .inputs
            .iter()
            .chain(&engine.init.addresses.outputs)
        {
            let bound = address
                .strip_prefix("tcp://127.0.0.1:")
                .unwrap_or_else(|| panic!("a data-plane address on the worker's host: {address}"));
            assert_ne!(bound.parse::<u16>().unwrap(), 0, "{address}");
        }
    }

    /// The gateway's own connection (nothing configures the backend, as
    /// behind `--worker-urls ipc://...`) against an engine that keeps the
    /// backend of its first structured request for its life: whichever
    /// constraint comes first, every later one carries the backend the engine
    /// settled on, and that backend is the one a structural tag compiles on,
    /// so the kind of the first request never decides whether a forced tool
    /// call compiles later; a schema that backend cannot enforce is refused
    /// before the engine instead of being answered off the schema, whether it
    /// comes as a JSON-schema constraint or inside the structural tag of a
    /// forced call (`tool_choice` required or a named function; the tag an
    /// always-constrained worker takes for `auto` alike), and nothing refused
    /// reaches the engine; an `auto` call carries no constraint and passes.
    #[tokio::test]
    async fn the_first_structured_request_never_strands_a_later_one() {
        use engine_zmq_client::protocol::vllm::structured_outputs::{
            StructuredOutputBackend, StructuredOutputConstraint,
        };
        use vllm::sampling_params::Constraint;

        let dir = tempfile::tempdir().unwrap();
        let base_url = format!("ipc://{}", dir.path().join("w").display());
        let handshake = format!("ipc://{}", dir.path().join("hs.sock").display());
        let (client, engine) = tokio::join!(
            connect_with_eos(
                &base_url,
                "m".to_string(),
                RuntimeType::Vllm,
                Handshake::TcpOrIpc(&handshake),
                1,
                EosTokenIds::default(),
                Duration::from_secs(10),
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let mut engine = engine.expect("mock engine");

        // The engine's rule: the first structured request creates its one
        // backend; a later request stamped for another would fail at grammar
        // build (or, for a tag, compile on a backend that has no such format).
        let mut engine_backend: Option<StructuredOutputBackend> = None;
        // A forced call's structural tag as a parser with structural tags
        // sends one: a `tag` per tool, its parameters the `json_schema`
        // content; `at_least_one` for a forced call, off for the tag an
        // always-constrained worker takes under `auto`.
        let call_tag = |tools: &[(&str, serde_json::Value)], at_least_one: bool| {
            let tags: Vec<serde_json::Value> = tools
                .iter()
                .map(|(name, parameters)| {
                    serde_json::json!({
                        "begin": format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": "),
                        "content": {"type": "json_schema", "json_schema": parameters},
                        "end": "}\n</tool_call>",
                    })
                })
                .collect();
            serde_json::json!({"format": {"type": "triggered_tags", "triggers": ["<tool_call>"],
                "tags": tags, "at_least_one": at_least_one}})
            .to_string()
        };
        let plain = (
            "plain",
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        );
        // An array parameter under `uniqueItems`, the shape tool corpora
        // carry; a string with a length bound beside a pattern.
        let tagged = (
            "tagged",
            serde_json::json!({"type": "object", "properties": {"tags": {"type": "array",
                "items": {"type": "string"}, "uniqueItems": true}}, "required": ["tags"]}),
        );
        let bounded = (
            "bounded",
            serde_json::json!({"type": "object", "properties": {"code": {"type": "string",
                "pattern": "^[a-z]+$", "maxLength": 8}}}),
        );
        // Held open: dropping a stream aborts its request on the same socket.
        let mut streams = Vec::new();
        for (id, constraint) in [
            (
                "json-answer-first",
                Some(Constraint::JsonSchema(
                    r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.to_string(),
                )),
            ),
            (
                "forced-call",
                Some(Constraint::StructuralTag(call_tag(
                    std::slice::from_ref(&plain),
                    true,
                ))),
            ),
            ("json-object", Some(Constraint::JsonObject(true))),
            ("regex", Some(Constraint::Regex("[0-9]+".to_string()))),
            (
                "grammar",
                Some(Constraint::Grammar("root ::= \"a\"".to_string())),
            ),
            (
                "choice",
                Some(Constraint::Choice(vllm::ChoiceConstraint {
                    choices: vec!["yes".to_string(), "no".to_string()],
                })),
            ),
            (
                "required-call-unenforceable",
                Some(Constraint::StructuralTag(call_tag(
                    &[plain.clone(), tagged.clone()],
                    true,
                ))),
            ),
            (
                "named-call-unenforceable",
                Some(Constraint::StructuralTag(call_tag(&[bounded], true))),
            ),
            ("auto-call", None),
            (
                "auto-tag-unenforceable",
                Some(Constraint::StructuralTag(call_tag(&[plain, tagged], false))),
            ),
            (
                "schema-xgrammar-lacks",
                Some(Constraint::JsonSchema(
                    r#"{"type":"integer","multipleOf":3}"#.to_string(),
                )),
            ),
        ] {
            let request = vllm::GenerateRequest {
                request_id: id.to_string(),
                input: Some(vllm::generate_request::Input::Tokenized(
                    vllm::TokenizedInput {
                        original_text: String::new(),
                        input_ids: vec![1, 2, 3],
                    },
                )),
                sampling_params: Some(vllm::SamplingParams {
                    max_tokens: Some(8),
                    constraint,
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            };
            let outcome = client.generate_vllm_streams(request).await;
            // Refused before the engine, as vLLM's frontend refuses a schema
            // under `backend=xgrammar`: xgrammar would compile it with the
            // keyword dropped and answer off the schema. The message names
            // the keyword, where it stands and, in a tag, the call's tag by
            // its `begin`, which carries the tool's name.
            let refused_at = match id {
                "required-call-unenforceable" | "auto-tag-unenforceable" => Some((
                    "uniqueItems",
                    "`/format/tags/1/content/json_schema/properties/tags`",
                    Some("tagged"),
                )),
                "named-call-unenforceable" => Some((
                    "minLength/maxLength beside pattern or format",
                    "`/format/tags/0/content/json_schema/properties/code`",
                    Some("bounded"),
                )),
                "schema-xgrammar-lacks" => Some(("multipleOf", "its root", None)),
                _ => None,
            };
            if let Some((keyword, place, tool)) = refused_at {
                let status = outcome.err().expect("refused");
                assert_eq!(status.code(), Code::InvalidArgument, "{id}");
                let message = status.message();
                let call = tool.map_or(String::new(), |tool| {
                    format!(
                        " (inside the tag that begins `<tool_call>\\n{{\"name\": \"{tool}\", \
                         \"arguments\": `)"
                    )
                });
                assert!(
                    message.contains(&format!("`{keyword}` at {place}{call}")),
                    "{id}: {message}"
                );
                continue;
            }
            streams.push(outcome.expect("generate"));
            let EngineInbound::Add(engine_request) = engine.recv().await.expect("inbound") else {
                panic!("{id}: expected an add request");
            };
            // Nothing refused reached the engine: its next request is this one.
            assert_eq!(
                engine_request.request_id, id,
                "{id}: another request reached the engine"
            );
            if id == "auto-call" {
                // `auto` carries no constraint, so nothing to refuse.
                assert!(
                    engine_request
                        .sampling_params
                        .and_then(|sp| sp.structured_outputs)
                        .is_none(),
                    "{id}: an auto call carries no constraint"
                );
                continue;
            }
            let structured = engine_request
                .sampling_params
                .and_then(|sp| sp.structured_outputs)
                .expect("structured outputs");
            let settled = *engine_backend.get_or_insert(structured.backend);
            assert_eq!(
                structured.backend, settled,
                "{id}: stamped for a backend the engine does not run"
            );
            if id == "choice" {
                // xgrammar compiles no choice: it goes as the grammar vLLM's
                // frontend substitutes.
                assert!(
                    matches!(
                        structured.constraint,
                        StructuredOutputConstraint::Grammar(_)
                    ),
                    "{id}: {:?}",
                    structured.constraint
                );
            }
        }
        // The backend the engine settled on is the one the triggered-tags
        // format compiles on.
        assert_eq!(engine_backend, Some(StructuredOutputBackend::Xgrammar));
        drop(streams);
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
            RuntimeType::Mlx,
            Duration::from_secs(600),
        )
        .await
        else {
            panic!("MLX has no ZMQ engine adapter");
        };
        assert!(
            error.to_string().contains("no engine implementation"),
            "{error}"
        );
    }

    /// `count` mock ranks behind one vLLM-dialect client, kept alive so they
    /// can answer utility calls.
    async fn connected_ranks(dir: &Path, count: usize) -> (ZmqEngineClient, Vec<MockEngine>) {
        let ep = |name: &str| format!("ipc://{}", dir.join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let ranks = (0..count as u32).map(|rank| {
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(rank),
                EngineCoreReadyResponse {
                    cache_trace_epochs: vec![format!("capture-{rank}")],
                    ..default_ready_response()
                },
            )
        });
        let (client, engines) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                count,
                "org/repo".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            futures::future::join_all(ranks),
        );
        let engines = engines
            .into_iter()
            .map(|engine| engine.expect("mock engine"))
            .collect();
        (client.expect("adapter connect"), engines)
    }

    /// A TokenSpeed connection whose ranks declared `ready` at their
    /// handshake, one rank per response.
    async fn tokenspeed_ranks(dir: &Path, ready: Vec<EngineCoreReadyResponse>) -> ZmqEngineClient {
        std::fs::create_dir_all(dir).unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let count = ready.len();
        let ranks = ready.into_iter().enumerate().map(|(rank, ready)| {
            connect_to_frontend(&handshake, EngineId::from_engine_index(rank as u32), ready)
        });
        let (client, engines) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                count,
                "org/repo".to_string(),
                EosTokenIds::default(),
                RuntimeType::TokenSpeed,
                Duration::from_secs(10)
            ),
            futures::future::join_all(ranks),
        );
        for engine in engines {
            engine.expect("mock engine");
        }
        client.expect("adapter connect")
    }

    /// A worker registered for a disaggregated role connects only to engines
    /// that declared that role: a prefill rank among a decode worker's, or a
    /// whole-request engine behind a prefill worker, fails the check with both
    /// roles named. An engine that declares no request types predates the
    /// declaration and is taken as registered, except for an encode worker,
    /// which needs the encode request listed and is refused by such an engine.
    /// A prefill or encode worker also needs the engine's bootstrap address:
    /// an engine that declares the role without it, or nothing at all, is
    /// refused with the missing rendezvous named; a decode worker needs none.
    #[tokio::test]
    async fn a_worker_connects_only_to_engines_of_its_registered_role() {
        let declaring = |role: Option<&str>| EngineCoreReadyResponse {
            request_types: vec!["add".into(), "abort".into(), "control".into()],
            disaggregation_mode: role.map(str::to_string),
            ..default_ready_response()
        };
        let rendezvous = |role: Option<&str>| EngineCoreReadyResponse {
            bootstrap_host: Some("10.0.0.7".to_string()),
            bootstrap_port: Some(8998),
            ..declaring(role)
        };
        let dir = tempfile::tempdir().unwrap();
        let prefill = tokenspeed_ranks(
            &dir.path().join("p"),
            vec![rendezvous(Some("prefill")), rendezvous(Some("prefill"))],
        )
        .await;
        prefill
            .check_declared_role(Some("prefill"))
            .expect("prefill engines");
        let error = prefill.check_declared_role(Some("decode")).unwrap_err();
        assert!(
            error.contains("declared disaggregation role prefill") && error.contains("as decode"),
            "{error}"
        );
        let error = prefill.check_declared_role(None).unwrap_err();
        assert!(error.contains("as a regular worker"), "{error}");

        let mixed = tokenspeed_ranks(
            &dir.path().join("m"),
            vec![declaring(Some("decode")), declaring(Some("prefill"))],
        )
        .await;
        let error = mixed.check_declared_role(Some("decode")).unwrap_err();
        assert!(
            error.contains("declared disaggregation role prefill"),
            "{error}"
        );

        let whole = tokenspeed_ranks(&dir.path().join("w"), vec![declaring(None)]).await;
        whole
            .check_declared_role(None)
            .expect("a whole-request engine");
        let error = whole.check_declared_role(Some("prefill")).unwrap_err();
        assert!(error.contains("none (whole requests)"), "{error}");

        // A prefill worker's requests name the engine's rendezvous: an engine
        // that declares the role without a bootstrap address, or declares
        // nothing, is refused with the missing address named; a decode worker
        // needs none, so an engine that declares nothing serves it.
        let unplaced =
            tokenspeed_ranks(&dir.path().join("u"), vec![declaring(Some("prefill"))]).await;
        let error = unplaced.check_declared_role(Some("prefill")).unwrap_err();
        assert!(
            error.contains("declares no bootstrap address") && error.contains("as prefill"),
            "{error}"
        );
        let silent = tokenspeed_ranks(&dir.path().join("s"), vec![default_ready_response()]).await;
        let error = silent.check_declared_role(Some("prefill")).unwrap_err();
        assert!(error.contains("declares no bootstrap address"), "{error}");
        silent
            .check_declared_role(Some("decode"))
            .expect("an engine that declares nothing serves a decode worker");

        // An encode worker needs the wire's encode request: an engine that
        // lists none fails the connection whether it declares nothing (one
        // before the request) or a scheduler's request types, as the servicer
        // refuses the same engine at its link; one that lists it connects
        // when it also declares its bootstrap address.
        let error = silent.check_declared_role(Some("encode")).unwrap_err();
        assert!(
            error.contains("lists no encode request") && error.contains("--servicer-impl python"),
            "{error}"
        );
        let scheduler =
            tokenspeed_ranks(&dir.path().join("e"), vec![declaring(Some("encode"))]).await;
        let error = scheduler.check_declared_role(Some("encode")).unwrap_err();
        assert!(error.contains("lists no encode request"), "{error}");
        let encoder = |bootstrap_host: Option<&str>| EngineCoreReadyResponse {
            request_types: vec!["encode".into()],
            disaggregation_mode: Some("encode".to_string()),
            bootstrap_host: bootstrap_host.map(str::to_string),
            bootstrap_port: Some(8998),
            ..default_ready_response()
        };
        let unplaced_encoder = tokenspeed_ranks(&dir.path().join("g"), vec![encoder(None)]).await;
        let error = unplaced_encoder
            .check_declared_role(Some("encode"))
            .unwrap_err();
        assert!(
            error.contains("declares no bootstrap address") && error.contains("as encode"),
            "{error}"
        );
        let placed_encoder =
            tokenspeed_ranks(&dir.path().join("f"), vec![encoder(Some("10.0.0.9"))]).await;
        placed_encoder
            .check_declared_role(Some("encode"))
            .expect("an encode worker that lists the encode request and declares its address");
    }

    /// An RL operation rides the TokenSpeed control wire as a control call
    /// named after its route, with the route's JSON body as its one argument
    /// (a map, integers kept integers), answered on every rank under the call
    /// id; a call without a body carries no argument.
    #[tokio::test]
    async fn an_rl_operation_rides_the_control_wire_with_its_routes_body() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::{control::ControlReplySlim, tokenspeed::request::TokenSpeedRequestType},
        };

        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let ready = EngineCoreReadyResponse {
            positional_layouts: BTreeMap::from([(
                "ControlReplySlim".to_string(),
                ["call_id", "success", "message", "engine_index"]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )]),
            ..default_ready_response()
        };
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "org/repo".to_string(),
                EosTokenIds::default(),
                RuntimeType::TokenSpeed,
                Duration::from_secs(10)
            ),
            connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
        );
        let client = client.expect("adapter connect");
        let (mut engine_in, mut engine_out) = engine.expect("mock engine").split();
        let wait = Duration::from_secs(10);
        for (method, body, success) in [
            (
                "update_weights_from_mooncake",
                Some(serde_json::json!({"version": 7, "flush_cache": false})),
                true,
            ),
            ("continue_generation", None, false),
        ] {
            let (replies, ()) =
                tokio::join!(client.rl_control_call(method, body.clone(), wait), async {
                    let frames = engine_in.recv_frames().await.unwrap();
                    assert_eq!(
                        TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                        Some(TokenSpeedRequestType::Control)
                    );
                    let (call_id, called, args): (i64, String, Vec<OpaqueValue>) =
                        decode_msgpack(frames[1].as_ref()).unwrap();
                    assert_eq!(called, method);
                    match &body {
                        Some(_) => {
                            let fields = args[0].as_map().expect("the body as a map");
                            assert!(fields.iter().any(|(key, value)| {
                                key.as_str() == Some("version") && *value == OpaqueValue::from(7i64)
                            }));
                            assert!(fields.iter().any(|(key, value)| {
                                key.as_str() == Some("flush_cache")
                                    && *value == OpaqueValue::Boolean(false)
                            }));
                        }
                        None => assert!(args.is_empty()),
                    }
                    let reply = ControlReplySlim {
                        call_id,
                        success,
                        message: (!success).then(|| "not paused".to_string()),
                        engine_index: 0,
                    };
                    engine_out
                        .send_frames(vec![bytes::Bytes::from(encode_msgpack(&reply).unwrap())])
                        .await
                        .unwrap();
                });
            let replies = replies.expect("one reply per rank");
            assert_eq!(replies.len(), 1);
            assert_eq!(replies[0].0, success);
            if !success {
                assert!(replies[0].1.contains("not paused"), "{replies:?}");
            }
        }
    }

    /// An RL operation on a TokenSpeed connection whose ranks declared no
    /// control reply is refused by name before anything is sent: the wire
    /// carries generate and abort only.
    #[tokio::test]
    async fn an_rl_operation_on_a_wire_without_control_calls_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let client = tokenspeed_ranks(&dir.path().join("n"), vec![default_ready_response()]).await;
        let error = client
            .rl_control_call("pause_generation", None, Duration::from_secs(1))
            .await
            .expect_err("refused");
        assert_eq!(error.code(), Code::Unimplemented);
        assert!(
            error.message().contains("declares no control reply"),
            "{}",
            error.message()
        );
    }

    #[tokio::test]
    async fn cache_trace_epochs_include_all_connected_replicas() {
        let dir = tempfile::tempdir().unwrap();
        let (client, _engines) = connected_ranks(dir.path(), 2).await;
        assert_eq!(client.cache_trace_epochs(), vec!["capture-0", "capture-1"]);
    }

    /// A streamed vLLM request over a tokenized prompt, with `constraint` as
    /// its grammar when given and `led` saying the gateway led the grammar for
    /// a first-token start.
    fn vllm_request(
        id: &str,
        constraint: Option<vllm::sampling_params::Constraint>,
        led: bool,
    ) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            request_id: id.to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(4),
                constraint,
                grammar_from_first_token: led,
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        }
    }

    /// The next add-request a mock rank receives.
    async fn next_add(engine: &mut MockEngine) -> EngineCoreRequest {
        let EngineInbound::Add(request) = engine.recv().await.expect("inbound") else {
            panic!("expected an add-request");
        };
        *request
    }

    /// A grammar the gateway led for a first-token start reaches the engine
    /// with the gate behind its reasoning parser taken as passed
    /// (`reasoning_ended = true`); a grammar the gateway did not shape, and a
    /// request without one, carry nothing, and the engine starts where it does.
    #[tokio::test]
    async fn a_grammar_led_for_the_first_token_reaches_the_engine_with_its_gate_passed() {
        let dir = tempfile::tempdir().unwrap();
        let (client, mut engines) = connected_ranks(dir.path(), 1).await;
        let schema = || {
            Some(vllm::sampling_params::Constraint::JsonSchema(
                r#"{"type":"object"}"#.to_string(),
            ))
        };
        // The streams stay alive until the end: a dropped stream aborts its
        // engine-side request, and the abort would be the next inbound.
        let _led = client
            .generate_vllm(vllm_request("r1", schema(), true))
            .await
            .expect("submitted");
        assert_eq!(next_add(&mut engines[0]).await.reasoning_ended, Some(true));
        let _unshaped = client
            .generate_vllm(vllm_request("r2", schema(), false))
            .await
            .expect("submitted");
        assert_eq!(next_add(&mut engines[0]).await.reasoning_ended, None);
        let _free = client
            .generate_vllm(vllm_request("r3", None, true))
            .await
            .expect("submitted");
        assert_eq!(
            next_add(&mut engines[0]).await.reasoning_ended,
            None,
            "a request without a grammar carries nothing"
        );
    }

    /// Answer each rank's next utility call with its entry in `answers`,
    /// returning the calls received.
    async fn answer_reset(
        engines: &mut [MockEngine],
        answers: &[Result<bool, &str>],
    ) -> Vec<UtilityCall> {
        let mut calls = Vec::new();
        for (index, (engine, answer)) in engines.iter_mut().zip(answers).enumerate() {
            let EngineInbound::Utility(call) = engine.recv().await.expect("inbound") else {
                panic!("expected a utility call");
            };
            let outcome = answer.map(OpaqueValue::from).map_err(str::to_string);
            engine
                .send_utility_reply(index as u32, call.call_id, outcome)
                .await
                .expect("reply");
            calls.push(call);
        }
        calls
    }

    /// FlushCache's engine half: `reset_prefix_cache(False, False)` goes to
    /// every rank and succeeds only when every rank reset.
    #[tokio::test]
    async fn reset_prefix_cache_needs_every_rank_to_agree() {
        let dir = tempfile::tempdir().unwrap();
        let (client, mut engines) = connected_ranks(dir.path(), 2).await;
        let (result, calls) = tokio::join!(
            client.reset_prefix_cache(Duration::from_secs(5)),
            answer_reset(&mut engines, &[Ok(true), Ok(false)])
        );
        assert!(!result.unwrap());
        assert_eq!(calls.len(), 2);
        for call in &calls {
            assert_eq!(call.method, "reset_prefix_cache");
            assert_eq!(
                call.args,
                vec![OpaqueValue::from(false), OpaqueValue::from(false)]
            );
        }
        let (result, _) = tokio::join!(
            client.reset_prefix_cache(Duration::from_secs(5)),
            answer_reset(&mut engines, &[Ok(true), Ok(true)])
        );
        assert!(result.unwrap());
    }

    /// The wait expiring is the caller's deadline; an engine failure is the
    /// engine's, with its message.
    #[tokio::test]
    async fn reset_prefix_cache_maps_timeouts_and_failures_to_statuses() {
        let dir = tempfile::tempdir().unwrap();
        let (client, mut engines) = connected_ranks(dir.path(), 1).await;
        let status = client
            .reset_prefix_cache(Duration::from_millis(200))
            .await
            .expect_err("no answer");
        assert_eq!(status.code(), Code::DeadlineExceeded);
        // The unanswered call did reach the engine; drain it.
        assert!(matches!(
            engines[0].recv().await.unwrap(),
            EngineInbound::Utility(_)
        ));

        let (status, _) = tokio::join!(
            client.reset_prefix_cache(Duration::from_secs(5)),
            answer_reset(
                &mut engines,
                &[Err("Call to reset_prefix_cache method failed: boom")]
            )
        );
        let status = status.expect_err("engine failure");
        assert_eq!(status.code(), Code::Internal);
        assert!(status.message().contains("boom"), "{status:?}");
    }

    /// TokenSpeed has no prefix-cache reset on the ZMQ wire.
    #[tokio::test]
    async fn reset_prefix_cache_is_unimplemented_for_tokenspeed() {
        let (client, _engine, _dir) = connected_pair(RuntimeType::TokenSpeed).await;
        let status = client
            .reset_prefix_cache(Duration::from_secs(1))
            .await
            .expect_err("tokenspeed");
        assert_eq!(status.code(), Code::Unimplemented);
    }

    /// Answer each rank's next utility call with its entry in `answers` (the
    /// engine's `profile` returns nothing), returning the calls received.
    async fn answer_profile(
        engines: &mut [MockEngine],
        answers: &[Result<(), &str>],
    ) -> Vec<UtilityCall> {
        let mut calls = Vec::new();
        for (index, (engine, answer)) in engines.iter_mut().zip(answers).enumerate() {
            let EngineInbound::Utility(call) = engine.recv().await.expect("inbound") else {
                panic!("expected a utility call");
            };
            let outcome = answer.map(|()| OpaqueValue::Nil).map_err(str::to_string);
            engine
                .send_utility_reply(index as u32, call.call_id, outcome)
                .await
                .expect("reply");
            calls.push(call);
        }
        calls
    }

    /// StartProfile's and StopProfile's engine half: `profile(is_start)` goes
    /// to every rank, with the start's and the stop's flag; one rank's failure
    /// fails the whole call, in the engine's words.
    #[tokio::test]
    async fn profile_vllm_reaches_every_rank() {
        let dir = tempfile::tempdir().unwrap();
        let (client, mut engines) = connected_ranks(dir.path(), 2).await;
        for is_start in [true, false] {
            let (result, calls) = tokio::join!(
                client.profile_vllm(is_start, Duration::from_secs(5)),
                answer_profile(&mut engines, &[Ok(()), Ok(())])
            );
            result.unwrap();
            assert_eq!(calls.len(), 2);
            for call in &calls {
                assert_eq!(call.method, "profile");
                assert_eq!(call.args, vec![OpaqueValue::from(is_start)]);
            }
        }
        let (status, calls) = tokio::join!(
            client.profile_vllm(true, Duration::from_secs(5)),
            answer_profile(&mut engines, &[Ok(()), Err("Profiler is not enabled.")])
        );
        let status = status.expect_err("one rank failed");
        assert_eq!(status.code(), Code::Internal);
        assert!(
            status.message().contains("Profiler is not enabled."),
            "{status:?}"
        );
        assert_eq!(calls.len(), 2);
    }

    /// The wait expiring is the caller's deadline; an engine failure is the
    /// engine's, with its message.
    #[tokio::test]
    async fn profile_vllm_maps_timeouts_and_failures_to_statuses() {
        let dir = tempfile::tempdir().unwrap();
        let (client, mut engines) = connected_ranks(dir.path(), 1).await;
        let status = client
            .profile_vllm(true, Duration::from_millis(200))
            .await
            .expect_err("no answer");
        assert_eq!(status.code(), Code::DeadlineExceeded);
        // The unanswered call did reach the engine; drain it.
        assert!(matches!(
            engines[0].recv().await.unwrap(),
            EngineInbound::Utility(_)
        ));

        let (status, _) = tokio::join!(
            client.profile_vllm(false, Duration::from_secs(5)),
            answer_profile(&mut engines, &[Err("Call to profile method failed: boom")])
        );
        let status = status.expect_err("engine failure");
        assert_eq!(status.code(), Code::Internal);
        assert!(status.message().contains("boom"), "{status:?}");
    }

    /// TokenSpeed has no profiler call on the ZMQ wire.
    #[tokio::test]
    async fn profile_vllm_is_unimplemented_for_tokenspeed() {
        let (client, _engine, _dir) = connected_pair(RuntimeType::TokenSpeed).await;
        let status = client
            .profile_vllm(true, Duration::from_secs(1))
            .await
            .expect_err("tokenspeed");
        assert_eq!(status.code(), Code::Unimplemented);
    }
}
