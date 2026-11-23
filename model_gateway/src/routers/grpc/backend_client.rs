//! Backend client: the polymorphism point over a worker's transport.
//!
//! A worker's backend is reached either via gRPC (the [`GrpcClient`] multiplexer
//! over SGLang/vLLM/TRT-LLM/MLX/TokenSpeed) or via a direct ZMQ connection to a
//! same-host engine — vLLM EngineCore or TokenSpeed ([`ZmqEngineClient`]).
//! `BackendClient` keeps those
//! first-class siblings — `GrpcClient` stays pure gRPC — while the execution
//! pipeline (which works against `ProtoStream`/`ProtoGenerateRequest`) is
//! shared unchanged.
//!
//! A worker's connection may be one a host made. The request path still
//! sends through a `BackendClient`: the gateway puts the host's connection
//! in front of one ([`Fronted`]).

use std::{
    any::Any,
    collections::HashMap,
    future::Future,
    sync::{Arc, LazyLock},
    time::Duration,
};

use async_trait::async_trait;
use engine_zmq_adapter::admin::{flush_reply_wait, PROFILE_WAIT};
use openai_protocol::{
    chat::ChatCompletionRequest,
    completion::CompletionRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
    worker::{ProfileOptions, WorkerLoadResponse},
};
use smg_grpc_client::{
    common_proto, connect_channel_with_timeout,
    engine_labels::{ModelInfo, ServerInfo},
    sglang_proto,
    tokenizer_bundle::StreamBundle,
    tokenspeed_encoder::{tokenspeed_encoder_proto as tokenspeed_encoder, TokenSpeedEncoderClient},
    tokenspeed_proto, vllm_proto, OutputConstraint, SglangGenerateRequestOptions,
    SglangSchedulerClient, TokenSpeedSchedulerClient, VllmEngineClient,
};
use smg_worker::{
    connection::engine::{
        AdminAnswer, ConnectorDefault, EngineConnection, EngineConnector, HealthAnswer, LoadSource,
        ZmqLink,
    },
    identity::{
        basic::{RuntimeType, Worker},
        builder,
        endpoint::Endpoint,
        error::{WorkerError, WorkerResult},
    },
};
use tokio::sync::OnceCell;

use crate::{
    routers::grpc::{
        client::{
            GenerateRequestBuildOptions, GrpcClient, HealthCheckResponse, TokenSpeedMmBuilder,
            TokenSpeedPlainBuilder, VllmMmBuilder, VllmPlainBuilder,
        },
        common::stages::helpers,
        proto_wrapper::{
            finish_tokenspeed_request, finish_vllm_request, ProtoEmbedComplete, ProtoEmbedRequest,
            ProtoGenerateRequest, ProtoStream,
        },
        utils::tonic_ext::TonicStatusExt,
        zmq_client::{
            self, fold_tokenizer_eos_backstop, EncodeRendezvous, SglangProfileStart, ZmqDialect,
            ZmqEngineClient,
        },
        MultimodalData,
    },
    workflow::steps::classify::CLASSIFY_PROBE_TIMEOUT_SECS,
};

/// A backend connection: gRPC (any engine) or direct ZMQ (vLLM EngineCore,
/// TokenSpeed or SGLang).
#[derive(Clone)]
pub enum BackendClient {
    Grpc(GrpcClient),
    Zmq(ZmqEngineClient),
}

/// The native request builders of the ZMQ dialects for one request kind — the
/// only thing that differs between the ZMQ build surfaces, so the dialect
/// dispatch itself lives once in [`build_zmq_request`]/[`build_zmq_plain_request`].
struct ZmqBuilders<V, T, S> {
    vllm: V,
    tokenspeed: T,
    sglang: S,
}

/// The SGLang counterpart of [`VllmMmBuilder`] without a client; the
/// `_parts` builders take the multimodal inputs and constraints bundled as
/// options.
type SglangMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    SglangGenerateRequestOptions,
) -> Result<sglang_proto::GenerateRequest, String>;

/// The SGLang counterpart of [`VllmPlainBuilder`] without a client.
type SglangPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<sglang_proto::GenerateRequest, String>;

impl BackendClient {
    /// Runtime type backing this client.
    pub fn runtime_type(&self) -> RuntimeType {
        match self {
            Self::Grpc(client) => client.runtime_type(),
            Self::Zmq(client) => client.runtime(),
        }
    }

    /// True if this is a direct-ZMQ backend (the engine receives token ids only
    /// and cannot match string stops itself).
    pub fn is_zmq(&self) -> bool {
        matches!(self, Self::Zmq(_))
    }

    /// How many data-parallel ranks, counted from 0, a request pinned on a
    /// direct-ZMQ connection reaches: a pin below the count is placed, one at
    /// or past it may be refused. TokenSpeed places rank `r` on its `r`-th
    /// rank's engine, so every engine it connected counts (TRT-LLM takes no
    /// pin). vLLM and SGLang place a pin on the engine whose ZMQ index it
    /// names, so the ranks run from 0 to the first index no engine has: none
    /// for a vLLM group launched at a non-zero data-parallel start rank. A
    /// worker given no count (no `dp_size`) has the group its engines
    /// declared. `None` on a gRPC backend, whose servicer checks a pin
    /// against its own engines.
    pub fn pinnable_ranks(&self) -> Option<usize> {
        let Self::Zmq(client) = self else {
            return None;
        };
        Some(match client.dialect() {
            ZmqDialect::Vllm | ZmqDialect::Sglang => (0..)
                .take_while(|&rank| {
                    client
                        .engines()
                        .iter()
                        .any(|engine| engine.engine_id.engine_index() == Some(rank))
                })
                .count(),
            ZmqDialect::TokenSpeed | ZmqDialect::Trtllm => client.engine_count(),
        })
    }

    /// How many engines a direct-ZMQ connection connected, one per
    /// data-parallel rank: the ranks a TokenSpeed engine places a
    /// disaggregated request among, on the rank its bootstrap room's residue
    /// modulo this count names (a rank pin yields to the room). `None` on a
    /// gRPC backend.
    pub fn connected_engines(&self) -> Option<usize> {
        match self {
            Self::Grpc(_) => None,
            Self::Zmq(client) => Some(client.engine_count()),
        }
    }

    /// The bootstrap rendezvous the engine behind a direct-ZMQ connection
    /// declared at its handshake (`host`, `port`): where a prefill or encode
    /// engine's bootstrap server listens. `None` on a gRPC backend, whose
    /// worker's registration carries the rendezvous, and on an engine that
    /// declared none.
    pub fn declared_bootstrap(&self) -> Option<(&str, u16)> {
        match self {
            Self::Grpc(_) => None,
            Self::Zmq(client) => client.ready_response()?.bootstrap_address(),
        }
    }

    /// Finalize a built generate request for this backend's wire: resolve
    /// string `stop`s the engine cannot match (token-only wires and SGLang's
    /// `skip_tokenizer_init` workers) into `stop_token_ids`, folding in EOS
    /// where the frontend owns stopping.
    ///
    /// Returns the router's residual obligation: the stop strings the engine
    /// will never see, which response processing must trim from output text.
    /// Empty when the engine matches stops server-side. This is the client's
    /// own policy — callers need no transport knowledge.
    pub fn finalize_generate_request(
        &self,
        request: &mut ProtoGenerateRequest,
        tokenizer: Option<&Arc<dyn llm_tokenizer::traits::Tokenizer>>,
    ) -> Vec<String> {
        let token_only_wire = self.is_zmq();
        let router_stops = helpers::resolve_string_stops(request, tokenizer, token_only_wire);
        if let Self::Zmq(client) = self {
            // EngineCore has no tokenizer, so stopping at EOS is this
            // frontend's job; TokenSpeed's scheduler stops at EOS itself, and
            // its requests are a different proto variant — the fold's own
            // variant match is the single dispatch point.
            //
            // The primary id must also ride the request's `_eos_token_id`, not
            // just the stop set, or an EOS finish surfaces as a matched_stop
            // token id. That field is filled from the client's own EOS set, so
            // hand it the tokenizer's ids for the deployments where the
            // connect-time model-dir lookup found none (the adopted set is
            // likewise only read by the vLLM translation).
            client.adopt_tokenizer_eos(tokenizer);
            fold_tokenizer_eos_backstop(request, tokenizer);
        }
        router_stops
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.health_check().await,
            Self::Zmq(client) => {
                let resp = client.health_check();
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_model_info().await,
            Self::Zmq(client) => Ok(zmq_client::model_info(client)),
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_server_info().await,
            Self::Zmq(client) => Ok(zmq_client::server_info(client)),
        }
    }

    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Grpc(client) => client.get_tokenizer().await,
            // EngineCore does not serve tokenizer artifacts over ZMQ; the
            // tokenizer is configured at worker registration instead.
            Self::Zmq(_) => Err("ZMQ backend does not serve a tokenizer bundle".into()),
        }
    }

    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match self {
            Self::Grpc(client) => client.generate(req).await,
            // The ZMQ submit future is boxed so that its state (the per-dialect
            // encode dispatch, the trace headers, the sub-stream set-up) does not
            // size this future, and with it every request's: an async block's
            // state holds the largest of its awaited branches inline, and the gRPC
            // backends, which never take this arm, were paying one jemalloc size
            // class (10,240 -> 12,288 B per request) for the ZMQ arm's growth.
            Self::Zmq(client) => Ok(ProtoStream::Zmq(
                Box::pin(zmq_client::generate(client, req)).await?,
            )),
        }
    }

    pub async fn embed(&self, req: ProtoEmbedRequest) -> Result<ProtoEmbedComplete, tonic::Status> {
        match self {
            Self::Grpc(client) => client.embed(req).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "ZMQ backend does not support embedding yet",
            )),
        }
    }

    pub fn build_chat_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_chat_request(request_id, body, processed_text, token_ids, options)
            }
            // A ZMQ backend speaks vLLM EngineCore or TokenSpeed directly; build
            // the native request for its dialect, mirroring the gRPC per-engine
            // dispatch in `GrpcClient::build_chat_request`.
            Self::Zmq(client) => build_zmq_request(
                client.dialect(),
                request_id,
                body,
                processed_text,
                token_ids,
                options,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_chat,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_chat,
                    sglang: SglangSchedulerClient::build_generate_request_from_chat_parts,
                },
            ),
        }
    }

    pub fn build_messages_request(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_messages_request(request_id, body, processed_text, token_ids, options)
            }
            // Mirrors the gRPC per-engine dispatch: build the request natively for
            // the ZMQ backend's dialect (vLLM EngineCore or TokenSpeed).
            Self::Zmq(client) => build_zmq_request(
                client.dialect(),
                request_id,
                body,
                processed_text,
                token_ids,
                options,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_messages,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_messages,
                    sglang: SglangSchedulerClient::build_generate_request_from_messages_parts,
                },
            ),
        }
    }

    pub fn build_completion_request(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_completion_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => build_zmq_plain_request(
                client.dialect(),
                request_id,
                body,
                original_text,
                token_ids,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_completion,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_completion,
                    sglang: SglangSchedulerClient::build_generate_request_from_completion_parts,
                },
            ),
        }
    }

    pub fn build_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_generate_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => build_zmq_plain_request(
                client.dialect(),
                request_id,
                body,
                original_text,
                token_ids,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_plain_generate_request,
                    tokenspeed: TokenSpeedSchedulerClient::build_plain_generate_request,
                    sglang: SglangSchedulerClient::build_plain_generate_request_parts,
                },
            ),
        }
    }
}

/// The worker runtime's view of the client. A call the runtime alone makes
/// has its one body here: the client has no inherent method of its name, so
/// the call cannot resolve to itself.
#[async_trait]
impl EngineConnection for BackendClient {
    /// Whether the backend answers the admin RPCs (cache flush, profiling):
    /// a gRPC servicer does, a ZMQ engine when its wire carries control calls
    /// (an SGLang scheduler, or a TokenSpeed one that declares them).
    fn carries_admin_calls(&self) -> bool {
        match self {
            Self::Grpc(_) => true,
            Self::Zmq(client) => client.carries_control_calls(),
        }
    }

    /// Local liveness. gRPC has no cheap local flag (it uses a health RPC), so
    /// this reports `true` for gRPC; ZMQ reflects its connection liveness.
    fn is_alive(&self) -> bool {
        match self {
            Self::Grpc(_) => true,
            Self::Zmq(client) => client.is_alive(),
        }
    }

    /// The client's own health read, whole, in the worker's words. The
    /// inherent method answers the client's reply, so this path cannot come
    /// back here: without that method the reply has no fields to take.
    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        let HealthCheckResponse { healthy, message } = BackendClient::health_check(self).await?;
        Ok(HealthAnswer { healthy, message })
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_loads().await,
            Self::Zmq(client) => Ok(client.get_loads_by_rank()),
        }
    }

    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        let (success, message) = match self {
            Self::Grpc(client) => {
                let answer = client.flush_cache(timeout_s).await?;
                (answer.success, answer.message)
            }
            // The reply wait is the servicers' rule, read from the adapter.
            Self::Zmq(client) => {
                client
                    .flush_cache_control(timeout_s, flush_reply_wait(timeout_s))
                    .await?
            }
        };
        Ok(AdminAnswer { success, message })
    }

    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        let req = common_proto::StartProfileRequest {
            output_dir: options.output_dir.clone(),
            start_step: options.start_step,
            num_steps: options.num_steps,
            activities: options.activities.clone().unwrap_or_default(),
            with_stack: options.with_stack,
            record_shapes: options.record_shapes,
            profile_by_stage: options.profile_by_stage,
        };
        let (success, message) = match self {
            Self::Grpc(client) => {
                let answer = client.start_profile(req).await?;
                (answer.success, answer.message)
            }
            Self::Zmq(client) => {
                client
                    .profile_control(Some(SglangProfileStart::from_request(req)), PROFILE_WAIT)
                    .await?
            }
        };
        Ok(AdminAnswer { success, message })
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        let (success, message) = match self {
            Self::Grpc(client) => {
                let answer = client.stop_profile().await?;
                (answer.success, answer.message)
            }
            Self::Zmq(client) => client.profile_control(None, PROFILE_WAIT).await?,
        };
        Ok(AdminAnswer { success, message })
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        Ok(BackendClient::get_model_info(self).await?.to_labels())
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        Ok(BackendClient::get_server_info(self).await?.to_labels())
    }

    async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Grpc(client) => client.subscribe_kv_events(start_seq).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents not supported over ZMQ",
            )),
        }
    }

    /// The worker's load records alone, no events; `Unimplemented` over ZMQ
    /// (the engine pushes its scheduler stats on that wire already).
    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Grpc(client) => client.subscribe_load_records().await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents not supported over ZMQ",
            )),
        }
    }
}

/// Makes the gateway's client for a worker. The gateway uses it when the
/// host supplies no connector. A host's connector can delegate to it for
/// workers that use the gateway's client.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendConnector;

impl BackendConnector {
    /// The stock connector, shared in one allocation for the process.
    pub fn shared() -> Arc<dyn EngineConnector> {
        static SHARED: LazyLock<Arc<dyn EngineConnector>> =
            LazyLock::new(|| Arc::new(BackendConnector));
        Arc::clone(&SHARED)
    }
}

impl ConnectorDefault for BackendConnector {
    fn default_connector() -> Option<Arc<dyn EngineConnector>> {
        Some(Self::shared())
    }
}

/// The worker builder as the gateway names it: a worker it builds connects
/// through the gateway's client unless the builder is handed another
/// connector. The builder is the worker crate's; this name for it is the
/// gateway's.
pub type BasicWorkerBuilder = builder::BasicWorkerBuilder<BackendConnector>;

#[async_trait]
impl EngineConnector for BackendConnector {
    async fn connect_grpc(
        &self,
        base_url: &str,
        runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        let client = connect_stock_grpc(base_url, runtime).await?;
        Ok(Arc::new(client))
    }

    async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        let client = zmq_client::connect_for_worker(
            &link.base_url,
            link.model_id,
            link.runtime,
            link.handshake_address.as_deref(),
            link.engine_count.map_or(
                zmq_client::EngineCount::Declared,
                zmq_client::EngineCount::Exactly,
            ),
            link.startup,
            link.request_timeout,
        )
        .await?;
        client.check_declared_role(link.worker_type)?;
        let client = BackendClient::Zmq(client);
        if link.pinned_by_rank {
            check_every_rank_is_pinnable(&client)?;
        }
        Ok(Arc::new(client))
    }

    /// The client crate's own dial: the channel every runtime's client
    /// connects through, so the dial answers as a connect would.
    async fn dial_grpc(&self, base_url: &str, timeout: Duration) -> Result<(), String> {
        connect_channel_with_timeout(base_url, timeout)
            .await
            .map(drop)
            .map_err(|e| e.to_string())
    }
}

/// Refuse a connection whose engines the rank workers it serves cannot all
/// pin: a vLLM or SGLang group whose ZMQ indices do not run from 0 (see
/// [`BackendClient::pinnable_ranks`]), where a request pinned to a rank past
/// the gap would be refused at dispatch.
fn check_every_rank_is_pinnable(client: &BackendClient) -> Result<(), String> {
    let (Some(pinnable), Some(connected)) = (client.pinnable_ranks(), client.connected_engines())
    else {
        return Ok(());
    };
    if pinnable < connected {
        return Err(format!(
            "a rank worker pins a request to its rank, which reaches only {pinnable} of the \
             {connected} engines connected: their ZMQ indices do not run from 0. Launch the \
             group at data-parallel start rank 0, or register it with no engine count, as one \
             grouped worker"
        ));
    }
    Ok(())
}

/// Connect the gateway's client to the gRPC server of an engine of
/// `runtime` at `base_url`: the stock connector's connect, and the one
/// that makes a front's back.
async fn connect_stock_grpc(base_url: &str, runtime: RuntimeType) -> Result<BackendClient, String> {
    match GrpcClient::connect(base_url, &runtime.to_string()).await {
        Ok(client) => Ok(BackendClient::Grpc(client)),
        Err(e) => Err(e.to_string()),
    }
}

/// A host's connection in front of the gateway's client.
///
/// A host's connector can answer the worker runtime on the host's own
/// service. The request path sends through the gateway's client. A front
/// holds both for one engine: the worker runtime's calls (health, loads,
/// labels, events and the admin calls) go to the host's connection, and
/// the gateway's requests go through its own client behind it, the back.
///
/// The gateway puts a front around each gRPC connection a host's connector
/// makes (`Fronting`), and makes the back itself at the first request.
/// A host that holds the gateway's client already may hand it over and
/// keep one channel ([`Fronted::over`]). The front is the worker's
/// connection, so the back ends with the host's connection.
pub struct Fronted {
    front: Arc<dyn EngineConnection>,
    back: Back,
}

/// The gateway's client behind a front.
enum Back {
    /// The host handed it over.
    Handed(Arc<BackendClient>),
    /// The gateway makes it at the first request, for the address and the
    /// runtime the host's connect was asked. The cell holds the client, or
    /// `None` for an engine that does not speak the gateway's wire.
    Made {
        base_url: String,
        runtime: RuntimeType,
        client: OnceCell<Option<Arc<BackendClient>>>,
    },
}

impl Fronted {
    /// `front`, the host's connection, in front of `back`, the gateway's
    /// client for the same engine. One level: a `back` that is not the
    /// gateway's client is refused here, at the connect.
    pub fn over(
        front: Arc<dyn EngineConnection>,
        back: Arc<dyn EngineConnection>,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        let Some(back) = back.downcast_arc::<BackendClient>() else {
            return Err(NOT_A_BACK.to_string());
        };
        let back = Back::Handed(back);
        Ok(Arc::new(Self { front, back }))
    }

    /// The gateway's own front around `front`, the connection a host made
    /// for the engine of `runtime` at `base_url`. Its back is made at the
    /// first request.
    fn around(front: Arc<dyn EngineConnection>, base_url: &str, runtime: RuntimeType) -> Self {
        let back = Back::Made {
            base_url: base_url.to_string(),
            runtime,
            client: OnceCell::new(),
        };
        Self { front, back }
    }

    /// The host's connection, for a host that reads its own type back.
    pub fn front(&self) -> &Arc<dyn EngineConnection> {
        &self.front
    }

    /// The back as it stands: `None` until it is made, and `Some(None)`
    /// for an engine that does not speak the gateway's wire.
    fn standing(&self) -> Option<Option<Arc<BackendClient>>> {
        match &self.back {
            Back::Handed(client) => Some(Some(Arc::clone(client))),
            Back::Made { client, .. } => client.get().cloned(),
        }
    }

    /// The back, made where it is not: the gateway's client connected to
    /// the engine's address and asked its health once ([`asked`]). An
    /// engine that speaks the wire keeps the client. One that does not is
    /// kept too, as `None`, and is not dialed again: so is a runtime the
    /// client has no wire for, which is known ahead of any dial. A connect
    /// that fails, or an ask the channel itself ends, is the error: nothing
    /// is kept, and the next call tries again.
    async fn make(&self) -> Result<Option<Arc<BackendClient>>, String> {
        let Back::Made {
            base_url,
            runtime,
            client,
        } = &self.back
        else {
            return Ok(self.standing().flatten());
        };
        let made = client.get_or_try_init(|| async {
            let url = Endpoint::redacted_str(base_url);
            if !GrpcClient::has_wire(*runtime) {
                tracing::warn!(worker_url = %url, %runtime, "The gateway's client has no wire for the runtime behind a host's connection");
                return Ok(None);
            }
            let stock = connect_stock_grpc(base_url, *runtime).await?;
            match asked(stock.health_check()).await {
                Asked::Speaks => {
                    tracing::info!(worker_url = %url, %runtime, "Connected the gateway's client behind a host's connection");
                    Ok(Some(Arc::new(stock)))
                }
                Asked::NoWire => {
                    tracing::warn!(worker_url = %url, %runtime, "The engine behind a host's connection does not speak the gateway's wire");
                    Ok(None)
                }
                Asked::Lost(reason) => Err(reason),
            }
        });
        made.await.cloned()
    }
}

const NOT_A_BACK: &str = "the back of a fronted connection is not the gateway's backend client";

/// The bound of the back's one health ask: the gateway's bound for asking
/// an engine whether it speaks a wire. A request may carry no deadline, and
/// each first request of a worker waits on the one making.
const ASK_BOUND: Duration = Duration::from_secs(CLASSIFY_PROBE_TIMEOUT_SECS);

/// What the back's one health ask says of an engine.
#[derive(Debug, PartialEq, Eq)]
enum Asked {
    /// It speaks the wire: it answered, with a verdict or an error of its
    /// own, or its handler still ran at the bound. Its health from here on
    /// is the breaker's to read.
    Speaks,
    /// It does not speak the wire: `Unimplemented`.
    NoWire,
    /// No answer of the engine's, with the reason: the status is the
    /// channel's own, as the request path classes it
    /// ([`TonicStatusExt::is_no_answer`]).
    Lost(String),
}

/// Read `ask`, one health call on a front's back, under [`ASK_BOUND`].
///
/// The ask has one verdict to tell, `Unimplemented`, and a server answers
/// that at once, from its router, with no handler run. A health call can
/// be a deep probe that a healthy engine takes longer over than the bound.
/// So an ask still pending at the bound shows the method exists: it reads
/// as an engine that speaks the wire, and the call is dropped there.
async fn asked<T>(ask: impl Future<Output = Result<T, tonic::Status>>) -> Asked {
    match tokio::time::timeout(ASK_BOUND, ask).await {
        Ok(Err(status)) if status.is_no_answer() => {
            let reason = status.client_message();
            Asked::Lost(format!("the engine's health call got no answer: {reason}"))
        }
        Ok(Err(status)) if status.code() == tonic::Code::Unimplemented => Asked::NoWire,
        Ok(_) | Err(_) => Asked::Speaks,
    }
}

// Every call of the worker runtime goes to the front, the two provided
// ones too: a front's own answer always stands.
#[async_trait]
impl EngineConnection for Fronted {
    fn carries_admin_calls(&self) -> bool {
        self.front.carries_admin_calls()
    }

    fn is_alive(&self) -> bool {
        self.front.is_alive()
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        self.front.health_answer().await
    }

    async fn health_check(&self) -> Result<bool, tonic::Status> {
        self.front.health_check().await
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        self.front.get_loads().await
    }

    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        self.front.flush_cache(timeout_s).await
    }

    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        self.front.start_profile(options).await
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        self.front.stop_profile().await
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        self.front.model_labels().await
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        self.front.server_labels().await
    }

    async fn kv_engine_id(&self) -> Result<Option<String>, tonic::Status> {
        self.front.kv_engine_id().await
    }

    async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        self.front.subscribe_kv_events(start_seq).await
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        self.front.subscribe_load_records().await
    }
}

/// The gateway's wrap of a host's connector. Each gRPC connection the host
/// makes is put behind a front ([`Fronted`]), unless it is the gateway's
/// client or a front already. A ZMQ connection, a dial and the ask for an
/// HTTP worker's load source pass through. The host's connector is asked
/// as before, and changes nothing.
///
/// Each call of the connector is forwarded by hand: a call the trait gains
/// with a default body reads that default here until it is forwarded.
#[derive(Debug)]
pub(crate) struct Fronting(Arc<dyn EngineConnector>);

impl Fronting {
    /// `host` wrapped, once, where the gateway's context takes it.
    pub(crate) fn around(host: Arc<dyn EngineConnector>) -> Arc<dyn EngineConnector> {
        Arc::new(Self(host))
    }
}

#[async_trait]
impl EngineConnector for Fronting {
    async fn connect_grpc(
        &self,
        base_url: &str,
        runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        let made = self.0.connect_grpc(base_url, runtime).await?;
        let any: &dyn Any = made.as_ref();
        if any.is::<BackendClient>() || any.is::<Fronted>() {
            return Ok(made);
        }
        Ok(Arc::new(Fronted::around(made, base_url, runtime)))
    }

    async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        self.0.connect_zmq(link).await
    }

    async fn dial_grpc(&self, base_url: &str, timeout: Duration) -> Result<(), String> {
        self.0.dial_grpc(base_url, timeout).await
    }

    async fn http_load_source(
        &self,
        base_url: &str,
        runtime: RuntimeType,
    ) -> Option<Arc<dyn LoadSource>> {
        self.0.http_load_source(base_url, runtime).await
    }
}

/// What stands behind a worker's engine connection for the request path.
enum Behind {
    /// The gateway's client: the connection itself, the same allocation,
    /// or the back of a front.
    Client(Arc<BackendClient>),
    /// A front whose back is not made yet.
    Unmade(Arc<Fronted>),
    /// Neither: a connection a host made with no front, or a front over an
    /// engine that does not speak the gateway's wire.
    NotTheGateways,
}

/// Read what stands behind `connection`. The type is read ahead of the
/// downcast, which consumes its handle: the gateway's own connection
/// clones nothing here.
fn behind(connection: Arc<dyn EngineConnection>) -> Behind {
    let any: &dyn Any = connection.as_ref();
    if any.is::<BackendClient>() {
        let client = connection.downcast_arc();
        return client.map_or(Behind::NotTheGateways, Behind::Client);
    }
    let Some(fronted) = connection.downcast_arc::<Fronted>() else {
        return Behind::NotTheGateways;
    };
    match fronted.standing() {
        Some(Some(client)) => Behind::Client(client),
        Some(None) => Behind::NotTheGateways,
        None => Behind::Unmade(fronted),
    }
}

/// The error for a worker the request path has no client of the gateway's
/// for: its connection is a host's with no front, or its engine does not
/// speak the gateway's wire.
#[cold]
fn not_the_gateways(worker: &dyn Worker) -> WorkerError {
    WorkerError::ConnectionFailed {
        url: worker.url().to_string(),
        reason: "its engine connection is not the gateway's backend client".to_string(),
    }
}

/// Make the back of `fronted`, the front around the connection a host made
/// for `worker`. Boxed: the caller's future holds a pointer where it would
/// hold the connect's whole state, and one call a front runs it.
async fn made(worker: &dyn Worker, fronted: &Fronted) -> WorkerResult<Arc<BackendClient>> {
    match Box::pin(fronted.make()).await {
        Ok(Some(client)) => Ok(client),
        Ok(None) => Err(not_the_gateways(worker)),
        Err(reason) => Err(WorkerError::ConnectionFailed {
            url: worker.url().to_string(),
            reason: format!("the gateway's client behind the host's connection: {reason}"),
        }),
    }
}

/// The gateway's client for `worker`; `None` for an HTTP worker. A
/// connected worker answers from its slot
/// ([`Worker::engine_connection_if_initialized`]): the allocation its
/// connect handed out, read without the boxed future of the trait's async
/// call. The first use, and a worker still connecting, go through
/// [`Worker::engine_connection`].
///
/// Behind a front the client is the front's back, made here by the first
/// call that finds none ([`Fronted`]).
pub async fn get_backend_client(worker: &dyn Worker) -> WorkerResult<Option<Arc<BackendClient>>> {
    let connection = match worker.engine_connection_if_initialized() {
        Some(connection) => connection,
        None => match worker.engine_connection().await? {
            Some(connection) => connection,
            None => return Ok(None),
        },
    };
    match behind(connection) {
        Behind::Client(client) => Ok(Some(client)),
        Behind::Unmade(fronted) => made(worker, &fronted).await.map(Some),
        Behind::NotTheGateways => Err(not_the_gateways(worker)),
    }
}

/// [`get_backend_client`] without the connect
/// ([`Worker::engine_connection_if_initialized`]): `None` for a worker
/// still connecting, for an HTTP worker, and behind a front until its back
/// is made.
pub fn backend_client_if_initialized(
    worker: &dyn Worker,
) -> WorkerResult<Option<Arc<BackendClient>>> {
    match worker.engine_connection_if_initialized().map(behind) {
        Some(Behind::Client(client)) => Ok(Some(client)),
        Some(Behind::Unmade(_)) | None => Ok(None),
        Some(Behind::NotTheGateways) => Err(not_the_gateways(worker)),
    }
}

/// [`backend_client_if_initialized`] for a reader with no error path: a
/// worker the gateway has no client for is said at error level, and the
/// reader goes on as it does before a connect.
pub(crate) fn connected_backend_client(worker: &dyn Worker) -> Option<Arc<BackendClient>> {
    match behind(worker.engine_connection_if_initialized()?) {
        Behind::Client(client) => Some(client),
        Behind::Unmade(_) => None,
        Behind::NotTheGateways => {
            let error = not_the_gateways(worker);
            tracing::error!(%error, "Reading the worker as not connected");
            None
        }
    }
}

// The encode-worker wires of a rendezvous handoff (the EPD encode stage's
// dispatch): the encoder RPC of a gRPC encode worker and the encode frame of
// a TokenSpeed encode server on the direct ZMQ lane.

/// Hand an encode request to a TokenSpeed encode server over its worker's
/// ZMQ connection. Nothing answers it: the server ships the embedding to the
/// prefill side over the transfer engine, as the gRPC encoder acks after its
/// send.
pub(crate) async fn send_tokenspeed_encode_frame(
    worker: &dyn Worker,
    rendezvous: &EncodeRendezvous,
    request: tokenspeed_encoder::EncodeRequest,
) -> Result<(), String> {
    // A failure's text is the dispatch's error line: it names the worker by
    // its masked address.
    let shown = worker.redacted_url();
    let client = get_backend_client(worker)
        .await
        .map_err(|e| format!("encode worker {shown} is not connected: {e}"))?
        .ok_or_else(|| format!("encode worker {shown} has no ZMQ connection"))?;
    let BackendClient::Zmq(client) = client.as_ref() else {
        return Err(format!("encode worker {shown} is not a direct-ZMQ worker"));
    };
    client
        .encode_tokenspeed(request, rendezvous)
        .await
        .map_err(|status| format!("encode request to {shown} failed: {}", status.message()))
}

/// Hand an encode request to a gRPC encode worker over its cached encoder
/// channel. The worker acknowledges after shipping the embedding to the
/// prefill side; a request it did not accept is an error.
pub(crate) async fn send_tokenspeed_encode_rpc(
    endpoint: String,
    request: tokenspeed_encoder::EncodeRequest,
) -> Result<(), String> {
    // The endpoint is the worker's registered address. A failure's text is
    // the dispatch's error line: it names the address masked, rendered only
    // when a send fails.
    let shown = || Endpoint::redacted_url(&endpoint);
    let client = TokenSpeedEncoderClient::connect_cached(&endpoint)
        .await
        .map_err(|e| format!("connect to encode worker {} failed: {e}", shown()))?;
    let response = client
        .encode(request)
        .await
        .map_err(|e| format!("encode RPC to {} failed: {}", shown(), e.message()))?;
    if !response.accepted {
        return Err(format!(
            "encode worker {} did not accept the request",
            shown()
        ));
    }
    Ok(())
}

/// Build a multimodal-carrying request (chat, messages) for a ZMQ backend: one
/// dispatch over the closed [`ZmqDialect`], converting the assembled multimodal
/// data to the dialect's proto and finishing through its SHM-cleanup wrapper.
fn build_zmq_request<B>(
    dialect: ZmqDialect,
    request_id: String,
    body: &B,
    processed_text: String,
    token_ids: Vec<u32>,
    options: GenerateRequestBuildOptions,
    builders: ZmqBuilders<VllmMmBuilder<B>, TokenSpeedMmBuilder<B>, SglangMmBuilder<B>>,
) -> Result<ProtoGenerateRequest, String> {
    let ZmqBuilders {
        vllm,
        tokenspeed,
        sglang,
    } = builders;
    match dialect {
        ZmqDialect::Vllm => {
            let vllm_mm = zmq_vllm_mm(options.multimodal_inputs)?;
            // The ZMQ wire carries one modality batch (see `zmq_vllm_mm`).
            let mut request = finish_vllm_request(vllm_mm.map(|mm| (mm, Vec::new())), |mm| {
                vllm(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    mm,
                    OutputConstraint {
                        tool: options.tool_constraints,
                        set_aside: options.format_set_aside,
                    },
                )
            })?;
            request.set_grammar_from_first_token(options.first_token_grammar);
            Ok(request)
        }
        ZmqDialect::TokenSpeed => {
            let tokenspeed_mm = zmq_tokenspeed_mm(options.multimodal_inputs)?;
            finish_tokenspeed_request(tokenspeed_mm, |mm| {
                tokenspeed(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    mm,
                    OutputConstraint {
                        tool: options.tool_constraints,
                        set_aside: options.format_set_aside,
                    },
                )
            })
        }
        // SGLang tensors ride inline (no SHM handles to release on failure).
        // A grammar led with the thought for the first token needs no
        // reasoning phase in front of it: the engine's grammar manager wraps
        // its reasoner around a request's grammar only when the request asks
        // for reasoning, and the gateway's own link cannot tell whether the
        // engine runs one, so the request does not ask where the tag has the
        // thought already. The engine's own reasoning-token count follows the
        // flag; the gateway's parser fills in the count where it runs.
        ZmqDialect::Sglang => {
            let sglang_mm = zmq_sglang_mm(options.multimodal_inputs)?;
            let (sglang_mm, extras) =
                sglang_mm.map_or((None, Vec::new()), |(first, extras)| (Some(first), extras));
            let mut req = sglang(
                request_id,
                body,
                processed_text,
                token_ids,
                SglangGenerateRequestOptions {
                    multimodal_inputs: sglang_mm,
                    tool_call_constraint: options.tool_constraints,
                    format_set_aside: options.format_set_aside,
                    require_reasoning: options.require_reasoning && !options.first_token_grammar,
                },
            )?;
            req.extra_mm_inputs = extras;
            Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
        }
        ZmqDialect::Trtllm => Err(DIRECT_LANE_NOT_FOR_TRTLLM.to_string()),
    }
}

/// The direct ZMQ lane builds no request for a TensorRT-LLM engine: that
/// engine is reached through its gRPC servicer (the Rust servicer speaks the
/// engine's wire on its node), and the gateway's own link to the wire serves
/// the other dialects only.
pub(crate) const DIRECT_LANE_NOT_FOR_TRTLLM: &str =
    "a TensorRT-LLM engine is served through its gRPC \
                                          servicer, not the gateway's direct ZMQ lane";

/// Build a request kind with no multimodal or tool inputs (completion, plain
/// generate) for a ZMQ backend. `text` is the builders' shared text parameter
/// (`String` for completion, `Option<String>` for plain generate).
fn build_zmq_plain_request<B, T>(
    dialect: ZmqDialect,
    request_id: String,
    body: &B,
    text: T,
    token_ids: Vec<u32>,
    builders: ZmqBuilders<
        VllmPlainBuilder<B, T>,
        TokenSpeedPlainBuilder<B, T>,
        SglangPlainBuilder<B, T>,
    >,
) -> Result<ProtoGenerateRequest, String> {
    let ZmqBuilders {
        vllm,
        tokenspeed,
        sglang,
    } = builders;
    match dialect {
        ZmqDialect::Vllm => Ok(ProtoGenerateRequest::Vllm(Box::new(vllm(
            request_id, body, text, token_ids,
        )?))),
        ZmqDialect::TokenSpeed => Ok(ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed(
            request_id, body, text, token_ids,
        )?))),
        ZmqDialect::Sglang => Ok(ProtoGenerateRequest::Sglang(Box::new(sglang(
            request_id, body, text, token_ids,
        )?))),
        ZmqDialect::Trtllm => Err(DIRECT_LANE_NOT_FOR_TRTLLM.to_string()),
    }
}

/// Convert assembled multimodal data for a vLLM ZMQ backend. A backend/variant
/// mismatch is a gateway bug (the assembly stage should produce the backend's
/// own variant), surfaced as a build error rather than a panic.
fn zmq_vllm_mm(
    inputs: Option<MultimodalData>,
) -> Result<Option<vllm_proto::MultimodalInputs>, String> {
    inputs
        .map(|mm| match mm {
            MultimodalData::Vllm(data) => {
                let (primary, extra) = data.into_protos();
                if !extra.is_empty() {
                    return Err(
                        "the vLLM ZMQ backend takes one modality per request; mixed image and \
                         video requests need the gRPC backend"
                            .to_string(),
                    );
                }
                Ok(primary)
            }
            other => Err(mm_variant_mismatch("vLLM", &other)),
        })
        .transpose()
}

/// Convert assembled multimodal data for a TokenSpeed ZMQ backend. See
/// [`zmq_vllm_mm`] for the mismatch semantics.
fn zmq_tokenspeed_mm(
    inputs: Option<MultimodalData>,
) -> Result<Option<tokenspeed_proto::MultimodalInputs>, String> {
    inputs
        .map(|mm| match mm {
            // No RDMA staging: the ZMQ engine reads tensors inline off the wire
            // and has no puller for `remote` payloads.
            MultimodalData::TokenSpeed(data) => Ok(data.into_proto(false)),
            other => Err(mm_variant_mismatch("TokenSpeed", &other)),
        })
        .transpose()
}

/// Convert assembled multimodal data for an SGLang ZMQ backend. See
/// [`zmq_vllm_mm`] for the mismatch semantics.
fn zmq_sglang_mm(
    inputs: Option<MultimodalData>,
) -> Result<
    Option<(
        sglang_proto::MultimodalInputs,
        Vec<sglang_proto::MultimodalInputs>,
    )>,
    String,
> {
    inputs
        .map(|mm| match mm {
            MultimodalData::Sglang(data) => Ok(data.into_protos()),
            other => Err(mm_variant_mismatch("SGLang", &other)),
        })
        .transpose()
}

/// Name the variant of a mismatched `MultimodalData` without dumping its tensor
/// payloads into the error string.
fn mm_variant_mismatch(expected: &str, got: &MultimodalData) -> String {
    let got = match got {
        MultimodalData::Sglang(_) => "SGLang",
        MultimodalData::Vllm(_) => "vLLM",
        MultimodalData::Trtllm(_) => "TRT-LLM",
        MultimodalData::TokenSpeed(_) => "TokenSpeed",
    };
    format!("multimodal data variant mismatch: {expected} ZMQ backend got {got} data")
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use smg_worker::{
        identity::basic::ConnectionMode,
        test_support::{no_health_check, NoEngine},
    };

    use super::*;
    use crate::routers::grpc::test_engine::mock_engine;

    /// A connected worker's client is read from its slot: the acquisition
    /// after the connect, and the two readers that never connect, hand out
    /// the allocation the connect made.
    #[tokio::test]
    async fn a_connected_workers_client_is_the_slots_own_allocation() {
        let (url, engine) = mock_engine("m").await;
        let worker = BasicWorkerBuilder::new(url.replacen("http://", "grpc://", 1))
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .health_config(no_health_check())
            .build();
        assert!(backend_client_if_initialized(&worker).unwrap().is_none());

        let connected = get_backend_client(&worker).await.unwrap().unwrap();
        let again = get_backend_client(&worker).await.unwrap().unwrap();
        let peeked = backend_client_if_initialized(&worker).unwrap().unwrap();
        assert!(Arc::ptr_eq(&connected, &again));
        assert!(Arc::ptr_eq(&connected, &peeked));
        let read = connected_backend_client(&worker).expect("connected");
        assert!(Arc::ptr_eq(&connected, &read));
        // No front stands in a stock build: the slot holds the client.
        let slot = worker.engine_connection_if_initialized();
        assert!(slot.is_some_and(|slot| slot.downcast_arc::<Fronted>().is_none()));
        engine.abort();
    }

    /// The handle's health read is the engine's verdict, not the arrival of
    /// a reply: an engine that answers unhealthy reads `false`.
    #[tokio::test]
    async fn an_engine_that_answers_unhealthy_reads_unhealthy_through_the_handle() {
        let (url, engine) = mock_engine("m").await;
        let port: u16 = url
            .rsplit(':')
            .next()
            .and_then(|port| port.parse().ok())
            .expect("the mock engine's port");
        let connection = BackendConnector
            .connect_grpc(
                &url.replacen("http://", "grpc://", 1),
                RuntimeType::TokenSpeed,
            )
            .await
            .expect("the mock engine connects");
        assert!(connection.health_check().await.expect("the engine answers"));

        mock_worker::overrides::for_port(port).set_serving(false);
        assert!(!connection.health_check().await.expect("the engine answers"));
        engine.abort();
    }

    /// A worker the gateway builds without naming a connector carries the
    /// gateway's own: the default of the builder as the gateway names it.
    #[tokio::test]
    async fn a_worker_the_gateway_builds_carries_the_gateways_connector() {
        let worker = BasicWorkerBuilder::new("grpc://engine:50051")
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build();

        let connector = worker.engine_connector.as_ref().expect("the default");
        assert_eq!(format!("{connector:?}"), "BackendConnector");
    }

    /// A connection a host made that stands with no front is an error that
    /// says so on both readers, never the `None` of a worker without a
    /// connection. A reader with no error path reads the worker as not
    /// connected.
    #[tokio::test]
    async fn a_connection_the_gateway_did_not_make_is_an_error_that_says_so() {
        let worker = BasicWorkerBuilder::new("grpc://engine:50051")
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .engine_connection(NoEngine)
            .build();

        for refusal in [
            get_backend_client(&worker).await.err(),
            backend_client_if_initialized(&worker).err(),
        ] {
            let refusal = refusal.expect("an error, never None").to_string();
            assert!(
                refusal.contains("not the gateway's backend client"),
                "{refusal}"
            );
        }
        assert!(connected_backend_client(&worker).is_none());
    }

    /// A host's connector that makes a connection of its own for each gRPC
    /// address, as a host that answers on its own service does.
    #[derive(Debug)]
    struct Hosts;

    #[async_trait]
    impl EngineConnector for Hosts {
        async fn connect_grpc(
            &self,
            _: &str,
            _: RuntimeType,
        ) -> Result<Arc<dyn EngineConnection>, String> {
            Ok(Arc::new(NoEngine))
        }

        async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
            BackendConnector.connect_zmq(link).await
        }

        async fn dial_grpc(&self, _: &str, _: Duration) -> Result<(), String> {
            Ok(())
        }
    }

    /// A gRPC worker at `url` whose connections `Hosts` makes, through the
    /// gateway's wrap of it.
    fn hosted(url: &str, runtime: RuntimeType) -> impl Worker {
        hosted_by(Arc::new(Hosts), url, runtime)
    }

    /// A gRPC worker at `url` whose connections `host` makes, through the
    /// gateway's wrap of it.
    fn hosted_by(host: Arc<dyn EngineConnector>, url: &str, runtime: RuntimeType) -> impl Worker {
        BasicWorkerBuilder::new(url)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .health_config(no_health_check())
            .engine_connector(Fronting::around(host))
            .build()
    }

    /// The gateway fronts a connection a host made and sends through a
    /// client of its own behind it: made by the first request, the same
    /// allocation after, and the engine answers through it. The two readers
    /// that do not connect read `None` until then. The worker's connection
    /// is the front, and the host's own is read back from it.
    #[tokio::test]
    async fn a_worker_on_a_hosts_connection_is_reached_through_the_gateways_client() {
        let (url, engine) = mock_engine("m").await;
        let url = url.replacen("http://", "grpc://", 1);
        let worker = hosted(&url, RuntimeType::TokenSpeed);
        let slot = worker.engine_connection().await.unwrap();
        assert!(backend_client_if_initialized(&worker).unwrap().is_none());
        assert!(connected_backend_client(&worker).is_none());

        let client = get_backend_client(&worker).await.unwrap().unwrap();
        let again = get_backend_client(&worker).await.unwrap().unwrap();
        let peeked = backend_client_if_initialized(&worker).unwrap().unwrap();
        let read = connected_backend_client(&worker).expect("made");
        for other in [&again, &peeked, &read] {
            assert!(Arc::ptr_eq(&client, other));
        }
        let health = client.health_check().await.expect("the engine answers");
        assert!(health.healthy);

        let fronted = slot.and_then(|connection| connection.downcast_arc::<Fronted>());
        let front = Arc::clone(fronted.expect("the gateway's front").front());
        assert!(front.downcast_arc::<NoEngine>().is_some());
        engine.abort();
    }

    /// An engine that does not speak the gateway's wire answers the back's
    /// one health ask `Unimplemented`. Its worker is refused by name on
    /// each reader, and the engine is not dialed again: the refusal stands
    /// with the engine gone.
    #[tokio::test]
    async fn an_engine_that_speaks_no_wire_of_the_gateways_is_refused_by_name_and_dialed_once() {
        let (url, engine) = mock_engine("m").await;
        let url = url.replacen("http://", "grpc://", 1);
        let worker = hosted(&url, RuntimeType::Vllm);
        let refused = |refusal: Option<WorkerError>| {
            let refusal = refusal.expect("an error").to_string();
            assert!(
                refusal.contains("not the gateway's backend client"),
                "{refusal}"
            );
        };
        refused(get_backend_client(&worker).await.err());

        engine.abort();
        engine.await.expect_err("the engine is gone");
        refused(get_backend_client(&worker).await.err());
        refused(backend_client_if_initialized(&worker).err());
        assert!(connected_backend_client(&worker).is_none());
    }

    /// Where nothing answers at the engine's address the first request
    /// fails as a connect does, nothing is remembered, and the next request
    /// tries again.
    #[tokio::test]
    async fn an_unreachable_engine_behind_a_front_fails_the_request_and_is_tried_again() {
        let worker = hosted("grpc://127.0.0.1:1", RuntimeType::TokenSpeed);
        for _ in 0..2 {
            let refusal = get_backend_client(&worker).await.err();
            let refusal = refusal.expect("nothing listens").to_string();
            assert!(
                refusal.contains("the gateway's client behind the host's connection"),
                "{refusal}"
            );
            assert!(backend_client_if_initialized(&worker).unwrap().is_none());
        }
    }

    /// A runtime the gateway's client has no wire for is refused by name
    /// ahead of any dial. Nothing listens at this address, and the refusal
    /// is the one of an engine that speaks no wire, not a failed connect.
    /// It is remembered.
    #[tokio::test]
    async fn a_runtime_with_no_wire_behind_a_front_is_refused_by_name_with_no_dial() {
        let worker = hosted("grpc://127.0.0.1:1", RuntimeType::Generic);
        let refusal = get_backend_client(&worker).await.err();
        let refusal = refusal.expect("an error").to_string();
        assert!(
            refusal.contains("not the gateway's backend client"),
            "{refusal}"
        );
        assert!(backend_client_if_initialized(&worker).is_err());
        assert!(connected_backend_client(&worker).is_none());
    }

    /// The back's one health ask tells the engine's answer from the
    /// channel's own. An engine that answers, with either verdict or an
    /// error of its own, speaks the wire. `Unimplemented` says it does not.
    /// A status the channel made is no answer of the engine's: a transport
    /// fault under the call, a dial that failed, a keep-alive that ran out,
    /// a cancelled call.
    #[tokio::test]
    async fn the_backs_health_ask_tells_the_engines_answer_from_the_channels_own() {
        use smg_grpc_client::h2;

        let answer = |status| asked(std::future::ready(Err::<bool, _>(status)));
        assert_eq!(asked(std::future::ready(Ok(true))).await, Asked::Speaks);
        assert_eq!(asked(std::future::ready(Ok(false))).await, Asked::Speaks);
        let own = tonic::Status::unavailable("loading");
        assert_eq!(answer(own).await, Asked::Speaks);
        let no_wire = tonic::Status::unimplemented("no such service");
        assert_eq!(answer(no_wire).await, Asked::NoWire);

        let Err(refused) = smg_grpc_client::connect_channel("grpc://127.0.0.1:1").await else {
            panic!("nothing listens on port 1");
        };
        let reset = h2::Error::from(h2::Reason::PROTOCOL_ERROR);
        for no_answer in [
            tonic::Status::from_error(Box::new(reset)),
            tonic::Status::from_error(refused),
            smg_grpc_client::keep_alive_timeout_status(),
            tonic::Status::cancelled("operation was canceled"),
        ] {
            let lost = answer(no_answer).await;
            assert!(
                matches!(&lost, Asked::Lost(reason) if reason.contains("no answer")),
                "{lost:?}"
            );
        }
    }

    /// An ask still pending at the bound ends there and reads as an engine
    /// that speaks the wire: its handler runs, so the method exists. The
    /// client is kept, and the request goes on through it.
    #[tokio::test(start_paused = true)]
    async fn an_ask_still_pending_at_its_bound_reads_as_an_engine_that_speaks_the_wire() {
        let started = tokio::time::Instant::now();
        let pending = asked(std::future::pending::<Result<bool, tonic::Status>>()).await;
        let waited = started.elapsed();
        assert!(
            waited >= ASK_BOUND && waited < ASK_BOUND + Duration::from_millis(2),
            "{waited:?}"
        );
        assert_eq!(pending, Asked::Speaks);
    }

    /// The back ends with the host's connection: when the worker's
    /// connection goes, the gateway's client behind it goes too.
    #[tokio::test]
    async fn a_fronts_back_ends_with_the_hosts_connection() {
        let (url, engine) = mock_engine("m").await;
        let url = url.replacen("http://", "grpc://", 1);
        let worker = hosted(&url, RuntimeType::TokenSpeed);
        let client = get_backend_client(&worker).await.unwrap().unwrap();
        let back = Arc::downgrade(&client);
        drop(client);
        assert!(back.upgrade().is_some());
        drop(worker);
        assert!(back.upgrade().is_none());
        engine.abort();
    }

    /// A host that holds the gateway's client hands it over: the front's
    /// back is that client, the same allocation, and the gateway dials
    /// nothing. A back that is not the gateway's client is refused at the
    /// connect, and a front is no back.
    #[tokio::test]
    async fn a_host_hands_the_gateways_client_it_holds_over_as_the_back() {
        let (url, engine) = mock_engine("m").await;
        let url = url.replacen("http://", "grpc://", 1);
        let stock = BackendConnector.connect_grpc(&url, RuntimeType::TokenSpeed);
        let stock = stock.await.expect("the mock engine connects");
        let fronted = Fronted::over(Arc::new(NoEngine), Arc::clone(&stock)).expect("a front");
        let Behind::Client(back) = behind(Arc::clone(&fronted)) else {
            panic!("the handed back stands");
        };
        let stock = stock.downcast_arc::<BackendClient>();
        assert!(stock.is_some_and(|stock| Arc::ptr_eq(&stock, &back)));

        let refused = Fronted::over(Arc::new(NoEngine), Arc::new(NoEngine)).err();
        assert_eq!(refused.as_deref(), Some(NOT_A_BACK));
        let refused = Fronted::over(Arc::new(NoEngine), fronted).err();
        assert_eq!(refused.as_deref(), Some(NOT_A_BACK));
        engine.abort();
    }

    /// A host's connector that holds the gateway's client and returns a
    /// front of its own over it.
    #[derive(Debug)]
    struct HandsOver;

    #[async_trait]
    impl EngineConnector for HandsOver {
        async fn connect_grpc(
            &self,
            base_url: &str,
            runtime: RuntimeType,
        ) -> Result<Arc<dyn EngineConnection>, String> {
            let stock = BackendConnector.connect_grpc(base_url, runtime).await?;
            Fronted::over(Arc::new(NoEngine), stock)
        }

        async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
            BackendConnector.connect_zmq(link).await
        }

        async fn dial_grpc(&self, _: &str, _: Duration) -> Result<(), String> {
            Ok(())
        }
    }

    /// A front a host made is the worker's connection as it is: the
    /// gateway's wrap does not front it again. The back stands from the
    /// host's connect, ahead of any request, and the request path reads
    /// the client the host handed over. The gateway dials no second
    /// channel.
    #[tokio::test]
    async fn a_hosts_own_front_is_not_fronted_again() {
        let (url, engine) = mock_engine("m").await;
        let url = url.replacen("http://", "grpc://", 1);
        let worker = hosted_by(Arc::new(HandsOver), &url, RuntimeType::TokenSpeed);
        let slot = worker.engine_connection().await.unwrap();
        let fronted = slot.and_then(|connection| connection.downcast_arc::<Fronted>());
        let fronted = fronted.expect("the host's front");
        let front = Arc::clone(fronted.front());
        assert!(front.downcast_arc::<NoEngine>().is_some());
        let handed = fronted.standing().flatten().expect("the handed back");

        let peeked = backend_client_if_initialized(&worker).unwrap();
        assert!(peeked.is_some_and(|peeked| Arc::ptr_eq(&handed, &peeked)));
        let client = get_backend_client(&worker).await.unwrap();
        assert!(client.is_some_and(|client| Arc::ptr_eq(&handed, &client)));
        engine.abort();
    }

    /// A host's connector that keeps a load source for each HTTP worker.
    struct Sourced(Arc<dyn LoadSource>);

    impl std::fmt::Debug for Sourced {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Sourced")
        }
    }

    struct NoLoads;

    #[async_trait]
    impl LoadSource for NoLoads {
        async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
            Err(tonic::Status::unavailable("no engine"))
        }
    }

    #[async_trait]
    impl EngineConnector for Sourced {
        async fn connect_grpc(
            &self,
            base_url: &str,
            runtime: RuntimeType,
        ) -> Result<Arc<dyn EngineConnection>, String> {
            Hosts.connect_grpc(base_url, runtime).await
        }

        async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
            Hosts.connect_zmq(link).await
        }

        async fn dial_grpc(&self, base_url: &str, timeout: Duration) -> Result<(), String> {
            Hosts.dial_grpc(base_url, timeout).await
        }

        async fn http_load_source(&self, _: &str, _: RuntimeType) -> Option<Arc<dyn LoadSource>> {
            Some(Arc::clone(&self.0))
        }
    }

    /// The gateway's wrap asks the host's connector for an HTTP worker's
    /// load source and hands back the host's own. A host that keeps none
    /// answers none through the wrap too.
    #[tokio::test]
    async fn the_wrap_asks_the_hosts_connector_for_a_load_source() {
        let source: Arc<dyn LoadSource> = Arc::new(NoLoads);
        let wrap = Fronting::around(Arc::new(Sourced(Arc::clone(&source))));
        let url = "http://127.0.0.1:1";
        let asked = wrap.http_load_source(url, RuntimeType::Generic).await;
        assert!(asked.is_some_and(|asked| Arc::ptr_eq(&asked, &source)));

        let wrap = Fronting::around(Arc::new(Hosts));
        let none = wrap.http_load_source(url, RuntimeType::Generic).await;
        assert!(none.is_none());
    }

    /// A host's connection that tells each call it takes, and answers as a
    /// connection with no engine does.
    #[derive(Default)]
    struct Told(std::sync::Mutex<Vec<&'static str>>);

    impl Told {
        fn tell(&self, call: &'static str) {
            self.0.lock().expect("the record").push(call);
        }
    }

    #[async_trait]
    impl EngineConnection for Told {
        fn carries_admin_calls(&self) -> bool {
            self.tell("carries_admin_calls");
            false
        }

        fn is_alive(&self) -> bool {
            self.tell("is_alive");
            true
        }

        async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
            self.tell("health_answer");
            NoEngine.health_answer().await
        }

        async fn health_check(&self) -> Result<bool, tonic::Status> {
            self.tell("health_check");
            Ok(true)
        }

        async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
            self.tell("get_loads");
            NoEngine.get_loads().await
        }

        async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
            self.tell("flush_cache");
            NoEngine.flush_cache(timeout_s).await
        }

        async fn start_profile(
            &self,
            options: &ProfileOptions,
        ) -> Result<AdminAnswer, tonic::Status> {
            self.tell("start_profile");
            NoEngine.start_profile(options).await
        }

        async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
            self.tell("stop_profile");
            NoEngine.stop_profile().await
        }

        async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
            self.tell("model_labels");
            NoEngine.model_labels().await
        }

        async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
            self.tell("server_labels");
            NoEngine.server_labels().await
        }

        async fn kv_engine_id(&self) -> Result<Option<String>, tonic::Status> {
            self.tell("kv_engine_id");
            Ok(None)
        }

        async fn subscribe_kv_events(
            &self,
            start_seq: u64,
        ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
            self.tell("subscribe_kv_events");
            NoEngine.subscribe_kv_events(start_seq).await
        }

        async fn subscribe_load_records(
            &self,
        ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
            self.tell("subscribe_load_records");
            NoEngine.subscribe_load_records().await
        }
    }

    /// Each of the worker runtime's thirteen calls on a front reaches the
    /// host's connection, the two provided ones as themselves. A call the
    /// trait gains later with a default body is not forwarded until the
    /// front forwards it: this case is its guard.
    #[tokio::test]
    async fn each_call_on_a_front_reaches_the_hosts_connection() {
        let told = Arc::new(Told::default());
        let host: Arc<dyn EngineConnection> = told.clone();
        let fronted = Fronted::around(host, "grpc://engine:50051", RuntimeType::TokenSpeed);

        fronted.carries_admin_calls();
        fronted.is_alive();
        fronted.health_answer().await.ok();
        fronted.health_check().await.ok();
        fronted.get_loads().await.ok();
        fronted.flush_cache(1.0).await.ok();
        fronted.start_profile(&ProfileOptions::default()).await.ok();
        fronted.stop_profile().await.ok();
        fronted.model_labels().await.ok();
        fronted.server_labels().await.ok();
        fronted.kv_engine_id().await.ok();
        fronted.subscribe_kv_events(0).await.ok();
        fronted.subscribe_load_records().await.ok();
        assert_eq!(
            *told.0.lock().expect("the record"),
            [
                "carries_admin_calls",
                "is_alive",
                "health_answer",
                "health_check",
                "get_loads",
                "flush_cache",
                "start_profile",
                "stop_profile",
                "model_labels",
                "server_labels",
                "kv_engine_id",
                "subscribe_kv_events",
                "subscribe_load_records",
            ]
        );
    }

    /// A worker without an engine connection answers `None` on both readers.
    #[tokio::test]
    async fn an_http_worker_has_no_backend_client() {
        let worker = BasicWorkerBuilder::new("http://worker:8000")
            .health_config(no_health_check())
            .build();

        assert!(get_backend_client(&worker).await.unwrap().is_none());
        assert!(backend_client_if_initialized(&worker).unwrap().is_none());
    }

    /// Whether `source` names the worker crate's builder. Its module: the
    /// path written out, or `builder` as an item of an `identity::{...}`
    /// group, as a merged `use` writes it. Or the crate's test name for the
    /// builder, which pins no connector either: `BasicWorkerBuilder` of its
    /// `test_support`, written out, as an item of a group, or by a glob of
    /// that module.
    fn names_the_crates_builder(source: &str) -> bool {
        let text: String = source.split_whitespace().collect();
        text.contains("identity::builder")
            || text.contains("builder::BasicWorkerBuilder")
            || text.contains("test_support::BasicWorkerBuilder")
            || text.contains("smg_worker::test_support::*")
            || a_group_holds(&text, "identity::{", is_the_builder)
            || a_group_holds(&text, "test_support::{", is_the_test_builder)
            || a_group_holds(&text, "smg_worker::{", is_the_test_glob)
    }

    /// Whether an item at the top of a group that `opens` starts is one
    /// `wanted` knows. `text` holds no whitespace.
    fn a_group_holds(text: &str, opens: &str, wanted: fn(&str) -> bool) -> bool {
        text.split(opens).skip(1).any(|group| {
            let mut depth = 1_usize;
            let mut item = String::new();
            for c in group.chars() {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    ',' if depth == 1 => {
                        if wanted(&item) {
                            return true;
                        }
                        item.clear();
                    }
                    _ if depth == 1 => item.push(c),
                    _ => {}
                }
                if depth == 0 {
                    break;
                }
            }
            wanted(&item)
        })
    }

    /// Whether one item of an `identity::{...}` group is the builder module,
    /// under that name or another, or a path through it. The item holds no
    /// whitespace, so a renamed module reads `builderas...`.
    fn is_the_builder(item: &str) -> bool {
        item == "builder" || item.starts_with("builder::") || item.starts_with("builderas")
    }

    /// Whether one item of a `test_support::{...}` group is the crate's test
    /// name for the builder, under that name or another.
    fn is_the_test_builder(item: &str) -> bool {
        item.starts_with("BasicWorkerBuilder")
    }

    /// Whether one item of a `smg_worker::{...}` group is a glob of the
    /// crate's test names.
    fn is_the_test_glob(item: &str) -> bool {
        item == "test_support::*"
    }

    /// Every Rust source under `dir`, its directories read through.
    fn sources_under(dir: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                sources_under(&path, found);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                found.push(path);
            }
        }
    }

    /// The worker crate's builder names no connector by default, and a worker
    /// built so refuses a gRPC or ZMQ connect. The gateway names the builder
    /// by [`BasicWorkerBuilder`], which pins the gateway's connector, and
    /// this file alone names the crate's builder module or its test name
    /// for the builder. A file of the gateway, of its test harness or of a
    /// binding that names either is found here, by its path. The walk is held
    /// to one known file of each root it reads.
    #[test]
    fn this_file_alone_names_the_worker_crates_builder() {
        let package = Path::new(env!("CARGO_MANIFEST_DIR"));
        let this_file = package.join("src/routers/grpc/backend_client.rs");
        let mut sources = Vec::new();
        sources_under(package, &mut sources);
        for binding in ["golang", "python"] {
            sources_under(
                &package.join("../bindings").join(binding).join("src"),
                &mut sources,
            );
        }
        assert!(
            sources.contains(&this_file),
            "the walk read {} files and not this one",
            sources.len()
        );
        for known in [
            "tests/api_tests.rs",
            "benches/access_log.rs",
            "test_harness/src/lib.rs",
            "../bindings/golang/src/lib.rs",
            "../bindings/python/src/lib.rs",
        ] {
            assert!(
                sources.contains(&package.join(known)),
                "the walk left out {known}"
            );
        }

        let named_by: Vec<&PathBuf> = sources
            .iter()
            .filter(|path| **path != this_file)
            .filter(|path| {
                fs::read_to_string(path).is_ok_and(|source| names_the_crates_builder(&source))
            })
            .collect();
        assert!(
            named_by.is_empty(),
            "the worker crate's builder defaults to no connector: name the builder by the \
             gateway's `BasicWorkerBuilder` in {named_by:?}"
        );
    }

    /// The forms a path to the crate's builder takes, and six that are not
    /// one: the gateway's own name, the crate's other identity modules, one
    /// of them under another name, the configuration's builder module, the
    /// crate's other test names, and a glob of the gateway's own test
    /// support.
    #[test]
    fn the_crates_builder_is_found_in_each_form_its_path_takes() {
        for named in [
            "use smg_worker::identity::builder;",
            "use smg_worker::identity::builder::BasicWorkerBuilder;",
            "use smg_worker::identity::{basic::Worker, builder};",
            "use smg_worker::identity::{basic::Worker, builder::BasicWorkerBuilder};",
            "use smg_worker::identity::{builder as b, endpoint::Endpoint};",
            "use smg_worker::{\n    identity::{\n        basic::Worker,\n        builder,\n    },\n    registry,\n};",
            "use smg_worker::identity::{builder::*, endpoint::Endpoint};",
            "let worker = identity::builder::BasicWorkerBuilder::new(url).build();",
            "use smg_worker::identity as id;\nlet worker = id::builder::BasicWorkerBuilder::new(url);",
            "use smg_worker::test_support::BasicWorkerBuilder;",
            "use smg_worker::test_support::*;",
            "use smg_worker::{registry::WorkerId, test_support::*};",
            "use smg_worker::test_support::{\n    BasicWorkerBuilder as Bare, RefusedAddress,\n};",
        ] {
            assert!(names_the_crates_builder(named), "{named}");
        }
        for other in [
            "use crate::routers::grpc::backend_client::BasicWorkerBuilder;",
            "use smg_worker::identity::{basic::Worker, endpoint::Endpoint};",
            "use smg_worker::identity::{basic as b, endpoint::Endpoint};",
            "pub use builder::*;",
            "use smg_worker::test_support::{RefusedAddress, WorkerRegistry};",
            "use crate::middleware::test_support::*;",
        ] {
            assert!(!names_the_crates_builder(other), "{other}");
        }
    }
}
