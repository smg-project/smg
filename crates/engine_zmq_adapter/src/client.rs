//! [`ZmqEngineClient`]: the connected engine behind each engine's gRPC client
//! surface (vLLM and TokenSpeed today), plus the connect entry points and
//! metadata reads.

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, OnceLock},
    time::Duration,
};

use engine_zmq_client::{
    codec::OpaqueValue,
    connect_handshake,
    connector::{EngineCoreClient, TokenSpeedClient},
    protocol::{
        handshake::EngineCoreReadyResponse,
        vllm::{request::EngineCoreRequest, sampling::EngineCoreSamplingParams},
        EngineLoad,
    },
    ConnectedEngine,
};
use futures::stream::SelectAll;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::worker::{RuntimeType, SchedulerLoadSnapshot, WorkerLoadResponse};
use smg_grpc_client::{tokenspeed_proto, vllm_proto as vllm};

use crate::{
    eos::EosTokenIds,
    sockets::{
        ensure_ipc_socket_dir, unlink_stale_socket, zmq_socket_addresses, ZMQ_CONNECT_TIMEOUT,
    },
    stream::ZmqGenerateStream,
    tokenspeed::{
        fan_out_tokenspeed_requests, translate_request_tokenspeed, TokenSpeedGenerateStream,
    },
    vllm::{
        fan_out_requests, has_media, kv_transfer_params, now_secs, ranked_candidate_count,
        translate_media, translate_request_with_media, translated_from_processed, ProcessedMedia,
        StructuredOutputsBackendConfig, TranslatedMedia, VllmGenerateStream,
    },
};

/// How long the one-time `get_supported_tasks` utility call may take; the
/// engine answers it between scheduler steps.
const SUPPORTED_TASKS_TIMEOUT: Duration = Duration::from_secs(30);

/// The connector params a refused request must still release: NIXL's
/// `do_remote_prefill` marks a PD decode leg whose prefill side holds blocks
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
pub(crate) enum ZmqBackend {
    Vllm(Arc<EngineCoreClient>),
    TokenSpeed(Arc<TokenSpeedClient>),
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
    /// The engine's structured-output backend config, set once by the caller
    /// that knows it (the servicer); unset keeps the translation's
    /// per-constraint default.
    structured_outputs_backend: OnceLock<StructuredOutputsBackendConfig>,
    /// The engine's supported tasks (`get_supported_tasks`), fetched on first
    /// use and kept for the connection, as vLLM's frontend keeps them.
    supported_tasks: tokio::sync::OnceCell<Arc<[String]>>,
}

/// Bind the SMG-side ZMQ sockets and complete the handshake with the
/// engine(s): the single connect path for a worker's `ipc://` URL, driven only
/// by the worker's background handshake driver. `model_id` is
/// the config-resolved served model (EngineCore reports none). `engine_count`
/// is the number of DP engines that will dial this worker's sockets (1 for an
/// ungrouped worker). Errors are plain reasons; the worker layer wraps them in
/// its own error type.
pub async fn connect_for_worker(
    base_url: &str,
    model_id: String,
    runtime: RuntimeType,
    handshake_override: Option<&str>,
    engine_count: usize,
) -> Result<ZmqEngineClient, String> {
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
    connect_with_eos(
        base_url,
        model_id,
        runtime,
        handshake_override,
        engine_count,
        eos,
    )
    .await
}

/// [`connect_for_worker`] with the EOS set supplied by the caller: bind the
/// data-plane sockets under `base_url`, clear stale socket files, and complete
/// the handshake. For a frontend that already knows the model's EOS ids from
/// the engine's own config (the Rust gRPC servicer) and has no model dir to
/// read them from.
pub async fn connect_with_eos(
    base_url: &str,
    model_id: String,
    runtime: RuntimeType,
    handshake_override: Option<&str>,
    engine_count: usize,
    eos: EosTokenIds,
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

/// Model metadata as the connected engine's native proto response.
pub enum ZmqModelInfo {
    Vllm(vllm::GetModelInfoResponse),
    TokenSpeed(Box<tokenspeed_proto::GetModelInfoResponse>),
}

/// Server metadata as the connected engine's native proto response.
pub enum ZmqServerInfo {
    Vllm(Box<vllm::GetServerInfoResponse>),
    TokenSpeed(Box<tokenspeed_proto::GetServerInfoResponse>),
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
                structured_outputs_backend: OnceLock::new(),
                supported_tasks: tokio::sync::OnceCell::new(),
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

    /// Pin the grammar backend for structured-output requests. vLLM's engine
    /// keeps a single backend per process, chosen by the first request that
    /// needs one, so the caller resolves it once from the engine's
    /// `--structured-outputs-config` and every request carries the same one.
    pub fn set_structured_outputs_backend(&self, config: StructuredOutputsBackendConfig) {
        let _ = self.meta.structured_outputs_backend.set(config);
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

    /// The vLLM EngineCore client behind this connection, for RPCs only the
    /// vLLM wire defines; `operation` names the RPC in the refusal a
    /// TokenSpeed backend answers.
    pub(crate) fn vllm_client(&self, operation: &str) -> Result<&EngineCoreClient, tonic::Status> {
        match &self.backend {
            ZmqBackend::Vllm(client) => Ok(client),
            ZmqBackend::TokenSpeed(_) => Err(tonic::Status::unimplemented(format!(
                "{operation} is only available on a vLLM ZMQ backend"
            ))),
        }
    }

    /// The first connected engine's handshake `READY` response: the
    /// connection-constant engine facts (context length, dtype, parallel
    /// sizes, KV capacity) a frontend re-exposes as metadata.
    pub fn ready_response(&self) -> Option<&EngineCoreReadyResponse> {
        self.engines().first().map(|engine| &engine.ready_response)
    }

    /// The vLLM half of [`Self::generate`] as one stream per choice: the
    /// `n > 1` fan-out submitted to the engine, each sub tagged with its proto
    /// `index`, left unmerged so a caller can end one choice without the
    /// others (the gRPC servicer matches string stops per choice). vLLM
    /// dialect only.
    pub async fn generate_vllm_streams(
        &self,
        req: vllm::GenerateRequest,
    ) -> Result<Vec<VllmGenerateStream>, tonic::Status> {
        self.generate_vllm_streams_with_media(req, None).await
    }

    /// [`generate_vllm_streams`](Self::generate_vllm_streams) for a request
    /// whose `media_refs` a worker-side processor already turned into engine
    /// features: `processed` is attached in place of the request's own
    /// multimodal batches (it must carry none).
    pub async fn generate_vllm_streams_with_media(
        &self,
        mut req: vllm::GenerateRequest,
        processed: Option<ProcessedMedia>,
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
            translated_from_processed(&req, processed).map_err(tonic::Status::invalid_argument)?
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
        let structured_backend = self.meta.structured_outputs_backend.get().copied();
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
            let request = translate_request_with_media(
                sub,
                sub_media,
                max_model_len,
                self.effective_eos(),
                structured_backend,
            )
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
    /// single-sample engine requests (see [`fan_out_requests`]); their outputs
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
        let mut streams = SelectAll::new();
        for stream in self.generate_vllm_streams(req).await? {
            streams.push(stream);
        }
        Ok(ZmqGenerateStream::Vllm(streams))
    }

    /// [`Self::generate_vllm`] for a TokenSpeed backend and request.
    pub async fn generate_tokenspeed(
        &self,
        req: tokenspeed_proto::GenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        let ZmqBackend::TokenSpeed(client) = &self.backend else {
            return Err(tonic::Status::internal(
                "vLLM ZMQ backend expects a vLLM generate request",
            ));
        };
        // Sub-streams submitted before a mid-loop failure are dropped with the
        // error, which auto-aborts their engine-side requests.
        let mut streams = SelectAll::new();
        for (index, sub) in fan_out_tokenspeed_requests(req).into_iter().enumerate() {
            let request =
                translate_request_tokenspeed(sub).map_err(tonic::Status::invalid_argument)?;
            let stream = client.submit(request).await.map_err(zmq_status)?;
            streams.push(TokenSpeedGenerateStream::new(stream, index as u32));
        }
        Ok(ZmqGenerateStream::TokenSpeed(streams))
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

    /// vLLM's `reset_prefix_cache(reset_running_requests=False,
    /// reset_connector=False)` on every connected rank, as the Python
    /// servicer's `admin.py` issues it per rank: `true` only when every rank
    /// reset. The wait is bounded by `timeout` (DEADLINE_EXCEEDED); an engine
    /// failure is INTERNAL, a dead engine UNAVAILABLE. TokenSpeed has no such
    /// RPC on the ZMQ wire.
    pub async fn reset_prefix_cache(&self, timeout: Duration) -> Result<bool, tonic::Status> {
        let ZmqBackend::Vllm(client) = &self.backend else {
            return Err(tonic::Status::unimplemented(
                "FlushCache is not available on a TokenSpeed ZMQ backend",
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
        }
    }
}

pub(crate) fn zmq_status(error: engine_zmq_client::Error) -> tonic::Status {
    match error {
        engine_zmq_client::Error::EngineCoreDead => tonic::Status::unavailable(error.to_string()),
        other => tonic::Status::internal(other.to_string()),
    }
}

/// A utility call's error as a status: the wait expiring is the caller's
/// deadline, everything else is the engine failing.
fn utility_status(error: engine_zmq_client::Error) -> tonic::Status {
    match error {
        engine_zmq_client::Error::UtilityTimeout { .. } => {
            tonic::Status::deadline_exceeded(error.to_string())
        }
        other => zmq_status(other),
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Arc, time::Duration};

    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response, EngineInbound, MockEngine},
        protocol::vllm::request::UtilityCall,
        EngineId,
    };
    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
    use openai_protocol::worker::RuntimeType;
    use tonic::Code;

    use super::*;
    use crate::{eos::EosTokenIds, sockets::ZMQ_CONNECT_TIMEOUT};

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

    /// `count` mock ranks behind one vLLM-dialect client, kept alive so they
    /// can answer utility calls.
    async fn connected_ranks(dir: &Path, count: usize) -> (ZmqEngineClient, Vec<MockEngine>) {
        let ep = |name: &str| format!("ipc://{}", dir.join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let ranks = (0..count as u32).map(|rank| {
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(rank),
                default_ready_response(),
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
        let _engine = engine.expect("mock engine");
        let status = client
            .expect("adapter connect")
            .reset_prefix_cache(Duration::from_secs(1))
            .await
            .expect_err("tokenspeed");
        assert_eq!(status.code(), Code::Unimplemented);
    }
}
