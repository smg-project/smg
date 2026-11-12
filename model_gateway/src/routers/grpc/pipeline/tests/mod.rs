mod alias_pipeline;
mod build_parity;
mod chat_logit_bias;
mod generate_body;
mod glm53;
mod model_label;
mod pretokenized;
mod rate_limit_reserve;

use std::{
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex, PoisonError, Weak,
    },
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures::{Stream, StreamExt};
use llm_tokenizer::{
    traits::{Decoder, Tokenizer},
    HuggingFaceTokenizer, MockTokenizer, TokenizerRegistry,
};
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{
    completion::CompletionRequest, model_card::ModelCard, responses::ResponsesRequest,
    worker::HealthCheckConfig,
};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage, ResponseId,
};
use smg_grpc_client::{
    common_proto as common, constraint_failed_status, tokenspeed_proto as ts, vllm_proto,
};
use smg_mcp::{McpConfig, McpOrchestrator};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request as TonicRequest, Response as TonicResponse, Status};
use tracing::instrument::WithSubscriber;
use tracing_subscriber::prelude::*;
use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

use super::*;
use crate::{
    config::types::PolicyConfig,
    routers::{
        common::{openai_bridge::FormatRegistry, test_support::fast_retry_config},
        grpc::{
            client::ModelInfo,
            common::responses::{
                background::{queued_response, store_record},
                BackgroundResponses, BackgroundStream,
            },
            harmony::serve_harmony_responses_stream,
            multimodal::{
                MultimodalComponents, MultimodalConfigRegistry, MultimodalSettings,
                SUPPORTS_VISION_LABEL,
            },
            proto_wrapper::ENGINE_ERROR_MESSAGE,
            regular::{
                responses::{route_responses, route_responses_background_stream},
                symphony::template::Templates,
            },
            test_engine::serve_on_loopback,
            tokenizer_cache,
            utils::tonic_ext::TonicStatusExt,
        },
    },
    tenant::TenantKey,
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        test_support::select_worker,
        BasicWorkerBuilder, ConnectionMode, RequestCompletionSink, RuntimeType, Worker, WorkerType,
    },
};

const MODEL: &str = "request-release-test-model";
/// A valid 1x1 PNG, as a client sends an inline image.
const ONE_PIXEL_PNG_DATA_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
/// The worker label of an engine without a vision encoder (vLLM
/// `--language-model-only`, or a text-only model).
const LANGUAGE_MODEL_ONLY: &[(&str, &str)] = &[(SUPPORTS_VISION_LABEL, "false")];

/// The labels discovery derives from a servicer's `GetModelInfo`, as
/// `(key, value)` pairs for the worker builders of these tests.
fn labels_of(info: &ModelInfo) -> Vec<(String, String)> {
    let mut labels: Vec<(String, String)> = info.to_labels().into_iter().collect();
    labels.sort();
    labels
}

fn label_pairs(labels: &[(String, String)]) -> Vec<(&str, &str)> {
    labels
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

/// The `(offset, length)` placeholder ranges of one generate call.
type PlaceholderRanges = Vec<(u32, u32)>;
/// A bias map as the stub received it, sorted by token.
type ReceivedBias = Vec<(String, f32)>;
type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
type KvEventStream = Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

/// TokenSpeed stub gated on the parsed request's drop probe: it withholds
/// its tokens (or, with `gate_rpc`, the generate RPC itself) until the
/// probe reaches zero strong references or a deadline passes, recording
/// the outcome in `released`. An ungated stub (no probe) answers
/// immediately -- used for the PD prefill side. `fail_first` makes the
/// first generate call return UNAVAILABLE, for retry-replay tests;
/// `fail_always` fails every call, `fail_internal` answers every call with
/// INTERNAL (an engine that took the request and failed on it),
/// `fail_mid_stream` ends the stream with the engine's own error after
/// its first frame, `fail_constraint` admits the request and ends the
/// stream with the engine's report that it could not continue the
/// request under its constraint, `fail_engine_report` admits it and ends
/// the stream with the engine's own report before any frame, and
/// `answer_after` stalls the generate RPC, standing in for an engine that
/// answers only at its own deadline; `frame_gap` paces the response frames,
/// for a stream that outlasts its dispatch. Every call's input token ids,
/// prompt text, stop sequences, bias map and engine request id are
/// recorded, as is every aborted request id.
#[derive(Clone, Default)]
struct GatedScheduler {
    probe: Option<Weak<CompletionRequest>>,
    gate_rpc: bool,
    released: Arc<AtomicBool>,
    fail_first: bool,
    fail_always: bool,
    fail_internal: bool,
    fail_mid_stream: bool,
    fail_constraint: bool,
    fail_engine_report: bool,
    /// Send the first frame, then hold the stream open with nothing on
    /// it: an engine whose generation wedged after its first token.
    hold_after_first: bool,
    /// Send the first frame, then hold the next until the flag is set:
    /// a generation whose end the test times itself.
    gate_after_first: Option<Arc<AtomicBool>>,
    answer_after: Option<Duration>,
    frame_gap: Option<Duration>,
    calls: Arc<AtomicUsize>,
    seen_input_ids: Arc<Mutex<Vec<Vec<u32>>>>,
    seen_original_texts: Arc<Mutex<Vec<String>>>,
    seen_mm_placeholders: Arc<Mutex<Vec<PlaceholderRanges>>>,
    seen_request_ids: Arc<Mutex<Vec<String>>>,
    seen_stops: Arc<Mutex<Vec<Vec<String>>>>,
    seen_logit_bias: Arc<Mutex<Vec<ReceivedBias>>>,
    aborted_request_ids: Arc<Mutex<Vec<String>>>,
}

impl GatedScheduler {
    async fn await_probe(probe: &Weak<CompletionRequest>, released: &AtomicBool) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while probe.strong_count() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        released.store(probe.strong_count() == 0, Ordering::SeqCst);
    }
}

/// The stub's two frames for `request_id`: one token, then the finish, both counting
/// `prompt_tokens` as the prompt the request carried, as an engine reports it.
fn generate_frames(
    request_id: &str,
    prompt_tokens: u32,
) -> Vec<Result<ts::GenerateResponse, Status>> {
    use ts::generate_response::Response as GenResp;
    vec![
        Ok(ts::GenerateResponse {
            request_id: request_id.to_string(),
            response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                token_ids: vec![100],
                prompt_tokens,
                completion_tokens: 1,
                cached_tokens: 0,
                output_logprobs: None,
                index: 0,
                weight_version: None,
            })),
        }),
        Ok(ts::GenerateResponse {
            request_id: request_id.to_string(),
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: vec![100],
                finish_reason: "stop".to_string(),
                prompt_tokens,
                completion_tokens: 1,
                cached_tokens: 0,
                output_logprobs: None,
                matched_stop: None,
                index: 0,
                ..Default::default()
            })),
        }),
    ]
}

#[tonic::async_trait]
impl TokenSpeedScheduler for GatedScheduler {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    #[expect(
        clippy::disallowed_methods,
        reason = "test stub; the gate task ends at its deadline"
    )]
    async fn generate(
        &self,
        request: TonicRequest<ts::GenerateRequest>,
    ) -> Result<TonicResponse<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        self.seen_mm_placeholders
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(
                request
                    .mm_inputs
                    .as_ref()
                    .map(|mm| {
                        mm.items
                            .iter()
                            .flat_map(|item| item.placeholders.iter().map(|p| (p.offset, p.length)))
                            .collect()
                    })
                    .unwrap_or_default(),
            );
        self.seen_original_texts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(
                request
                    .tokenized
                    .as_ref()
                    .map(|t| t.original_text.clone())
                    .unwrap_or_default(),
            );
        let input_ids = request.tokenized.map(|t| t.input_ids).unwrap_or_default();
        let prompt_tokens = u32::try_from(input_ids.len()).unwrap_or(u32::MAX);
        self.seen_input_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(input_ids);
        self.seen_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request.request_id.clone());
        self.seen_stops
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(
                request
                    .sampling_params
                    .as_ref()
                    .map(|params| params.stop.clone())
                    .unwrap_or_default(),
            );
        self.seen_logit_bias
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(
                request
                    .sampling_params
                    .as_ref()
                    .map(|params| {
                        let mut bias: ReceivedBias = params
                            .logit_bias
                            .iter()
                            .map(|(token, bias)| (token.clone(), *bias))
                            .collect();
                        bias.sort_by(|a, b| a.0.cmp(&b.0));
                        bias
                    })
                    .unwrap_or_default(),
            );
        if self.fail_always || (self.fail_first && self.calls.fetch_add(1, Ordering::SeqCst) == 0) {
            return Err(Status::unavailable("release-test induced failure"));
        }
        if self.fail_internal {
            return Err(Status::internal("release-test induced engine failure"));
        }
        if self.fail_constraint {
            // The engine admitted the request and could not continue it
            // under its constraint: the call succeeds and the stream ends
            // with the report before any frame, as the wires end it.
            let items: Vec<Result<ts::GenerateResponse, Status>> = vec![Err(
                constraint_failed_status("release-test induced grammar failure"),
            )];
            return Ok(TonicResponse::new(Box::pin(futures::stream::iter(items))));
        }
        if self.fail_engine_report {
            // The engine failed the request and named no cause: the
            // report the ZMQ wires end their streams with for an `error`
            // finish, before any frame.
            let items: Vec<Result<ts::GenerateResponse, Status>> =
                vec![Err(Status::internal(ENGINE_ERROR_MESSAGE))];
            return Ok(TonicResponse::new(Box::pin(futures::stream::iter(items))));
        }
        if let Some(delay) = self.answer_after {
            tokio::time::sleep(delay).await;
        }
        if self.gate_rpc {
            if let Some(probe) = &self.probe {
                Self::await_probe(probe, &self.released).await;
            }
        }
        let request_id = request.request_id;
        let (tx, rx) = mpsc::channel(8);
        let probe = (!self.gate_rpc).then(|| self.probe.clone()).flatten();
        let released = Arc::clone(&self.released);
        let frame_gap = self.frame_gap;
        let fail_mid_stream = self.fail_mid_stream;
        let hold_after_first = self.hold_after_first;
        let gate_after_first = self.gate_after_first.clone();
        tokio::spawn(async move {
            if let Some(probe) = probe {
                Self::await_probe(&probe, &released).await;
            }
            for (index, frame) in generate_frames(&request_id, prompt_tokens)
                .into_iter()
                .enumerate()
            {
                if let Some(gap) = frame_gap {
                    tokio::time::sleep(gap).await;
                }
                let frame = if fail_mid_stream && index == 1 {
                    Err(Status::internal(ENGINE_ERROR_MESSAGE))
                } else {
                    frame
                };
                if tx.send(frame).await.is_err() {
                    return;
                }
                if hold_after_first && index == 0 {
                    // The stream stays open, with nothing more on it,
                    // for longer than any wait a test sets.
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    return;
                }
                if let Some(gate) = gate_after_first.as_ref().filter(|_| index == 0) {
                    while !gate.load(Ordering::SeqCst) {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                }
            }
        });
        Ok(TonicResponse::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn health_check(
        &self,
        _request: TonicRequest<ts::HealthCheckRequest>,
    ) -> Result<TonicResponse<ts::HealthCheckResponse>, Status> {
        Ok(TonicResponse::new(ts::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        request: TonicRequest<ts::AbortRequest>,
    ) -> Result<TonicResponse<ts::AbortResponse>, Status> {
        self.aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request.into_inner().request_id);
        Ok(TonicResponse::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: TonicRequest<ts::GetModelInfoRequest>,
    ) -> Result<TonicResponse<ts::GetModelInfoResponse>, Status> {
        Ok(TonicResponse::new(ts::GetModelInfoResponse {
            model_path: MODEL.to_string(),
            tokenizer_path: MODEL.to_string(),
            served_model_name: MODEL.to_string(),
            model_type: "mock".to_string(),
            architectures: vec!["MockForCausalLM".to_string()],
            max_context_length: 32768,
            max_req_input_len: 32768,
            vocab_size: 32000,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            weight_version: "mock".to_string(),
            default_sampling_params_json: String::new(),
            supports_vision: false,
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: TonicRequest<ts::GetServerInfoRequest>,
    ) -> Result<TonicResponse<ts::GetServerInfoResponse>, Status> {
        Ok(TonicResponse::new(ts::GetServerInfoResponse {
            max_total_num_tokens: 1_000_000,
            tokenspeed_version: "mock".to_string(),
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        _request: TonicRequest<ts::GetLoadsRequest>,
    ) -> Result<TonicResponse<ts::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("release-test stub"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: TonicRequest<common::SubscribeKvEventsRequest>,
    ) -> Result<TonicResponse<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("release-test stub"))
    }

    async fn flush_cache(
        &self,
        _request: TonicRequest<common::FlushCacheRequest>,
    ) -> Result<TonicResponse<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("release-test stub"))
    }

    async fn start_profile(
        &self,
        _request: TonicRequest<common::StartProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("release-test stub"))
    }

    async fn stop_profile(
        &self,
        _request: TonicRequest<common::StopProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("release-test stub"))
    }

    async fn get_tokenizer(
        &self,
        _request: TonicRequest<common::GetTokenizerRequest>,
    ) -> Result<TonicResponse<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("release-test stub"))
    }
}

/// A current-thread runtime for a test that drives its pipeline by hand: the
/// metrics recorder is thread-local, so what the pipeline records is read on
/// the thread that ran it.
fn current_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a current-thread runtime")
}

/// The stub engine served on a loopback port bound before it is handed out.
async fn spawn_stub(scheduler: GatedScheduler) -> u16 {
    let router = Server::builder().add_service(TokenSpeedSchedulerServer::new(scheduler));
    serve_on_loopback(router).await.0.port()
}

fn register_worker(
    registry: &WorkerRegistry,
    port: u16,
    worker_type: WorkerType,
) -> Arc<dyn Worker> {
    register_worker_with_labels(registry, port, worker_type, &[])
}

fn register_worker_with_labels(
    registry: &WorkerRegistry,
    port: u16,
    worker_type: WorkerType,
    labels: &[(&str, &str)],
) -> Arc<dyn Worker> {
    register_worker_of_runtime(registry, port, worker_type, RuntimeType::TokenSpeed, labels)
}

/// [`register_worker_with_labels`] for a worker typed as another engine:
/// the capability matrix reads the runtime, and a request refused before
/// dispatch never speaks to the stub.
fn register_worker_of_runtime(
    registry: &WorkerRegistry,
    port: u16,
    worker_type: WorkerType,
    runtime: RuntimeType,
    labels: &[(&str, &str)],
) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
        .worker_type(worker_type)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(runtime)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        });
    for (key, value) in labels {
        builder = builder.label(*key, *value);
    }
    let worker: Arc<dyn Worker> = Arc::new(builder.build());
    registry
        .register(Arc::clone(&worker))
        .expect("register release-test worker");
    worker
}

/// [`register_worker`] with a circuit breaker opening at `failure_threshold`.
fn register_worker_with_breaker(
    registry: &WorkerRegistry,
    port: u16,
    failure_threshold: u32,
) -> Arc<dyn Worker> {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold,
                success_threshold: 1,
                timeout_duration: Duration::from_secs(60),
                window_duration: Duration::from_secs(60),
            })
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    registry
        .register(Arc::clone(&worker))
        .expect("register release-test worker");
    worker
}

async fn components(worker_registry: Arc<WorkerRegistry>) -> Arc<SharedComponents> {
    components_with_multimodal(worker_registry, false).await
}

/// Components over `tokenizer_registry` and `worker_registry` with the
/// legacy parser choice; `with_multimodal` adds the multimodal bundle
/// (default settings) so media reaches the multimodal path instead of the
/// "not available" refusal. The bundle's model config is read from the
/// tokenizer's source, so the caller's tokenizer decides whether there is
/// a model directory.
fn shared_components(
    tokenizer_registry: Arc<TokenizerRegistry>,
    worker_registry: Arc<WorkerRegistry>,
    with_multimodal: bool,
) -> Arc<SharedComponents> {
    let multimodal = with_multimodal.then(|| {
        Arc::new(
            MultimodalComponents::new(
                Arc::new(MultimodalConfigRegistry::new()),
                None,
                None,
                &MultimodalSettings::default(),
            )
            .expect("multimodal components"),
        )
    });
    Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(tokenizer_registry.clone())),
        tokenizer_registry,
        worker_registry,
        tool_parser_factory: ToolParserFactory::default(),
        reasoning_parser_factory: ReasoningParserFactory::default(),
        parser_resolver: utils::ParserResolver::disabled(),
        multimodal,
        tokenizer_autoload: true,
        tokenizer_load_failures: Arc::default(),
        parser: ParserChoice::Legacy,
    })
}

/// [`shared_components`] over the mock tokenizer (no model directory).
async fn components_with_multimodal(
    worker_registry: Arc<WorkerRegistry>,
    with_multimodal: bool,
) -> Arc<SharedComponents> {
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .expect("load mock tokenizer");
    shared_components(tokenizer_registry, worker_registry, with_multimodal)
}

/// A worker whose servicer left `supports_vision` unset (the SGLang and
/// TokenSpeed protos carry the field; only vLLM's servicer sets it on
/// purpose) is not a language-model-only worker: its labels carry no
/// vision verdict, and an image request goes on past the refusal into
/// multimodal processing (here, to the config lookup the mock tokenizer
/// cannot satisfy) instead of a `multimodal_not_supported`.
#[tokio::test]
async fn image_to_a_worker_without_a_vision_verdict_is_not_refused() {
    let labels = labels_of(&ModelInfo::Sglang(Box::default()));
    assert!(
        labels.iter().all(|(key, _)| key != SUPPORTS_VISION_LABEL),
        "{labels:?}"
    );
    let (pipeline, components, _) =
        chat_pipeline_over(RuntimeType::TokenSpeed, &label_pairs(&labels)).await;
    let (status, body) = chat_error_body(&pipeline, components, one_pixel_image_chat()).await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_ne!(body["error"]["code"], "multimodal_not_supported", "{body}");
    assert_eq!(
        body["error"]["code"], "invalid_multimodal_request",
        "{body}"
    );
}

/// An SGLang servicer that carries no media to its engine says so with a
/// deliberate `supports_vision=false` in its `server_args` (its
/// `GetModelInfo` bool is no verdict for this runtime): with the labels
/// discovery merges from both, the worker reads as language model only,
/// and an image is refused at the gate before anything is decoded or
/// dispatched, instead of after the media was fetched.
#[tokio::test]
async fn image_to_an_sglang_worker_that_carries_no_media_is_refused_before_dispatch() {
    use std::collections::BTreeMap;

    use smg_grpc_client::sglang_proto;

    use crate::routers::grpc::client::ServerInfo;

    let mut merged = ModelInfo::Sglang(Box::default()).to_labels();
    merged.extend(
        ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([(
                    "supports_vision".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::BoolValue(false)),
                    },
                )]),
            }),
            ..Default::default()
        }))
        .to_labels(),
    );
    let mut labels: Vec<(String, String)> = merged.into_iter().collect();
    labels.sort();
    assert!(
        labels
            .iter()
            .any(|(key, value)| key == SUPPORTS_VISION_LABEL && value == "false"),
        "{labels:?}"
    );
    let (pipeline, components, seen_ids) =
        chat_pipeline_over(RuntimeType::TokenSpeed, &label_pairs(&labels)).await;
    let (status, body) = chat_error_body(&pipeline, components, one_pixel_image_chat()).await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
    assert!(
        seen_ids.lock().expect("seen ids").is_empty(),
        "nothing must reach the worker"
    );
}

/// An image to a model whose only worker runs without a vision encoder
/// is refused with a prompt 400 that names the modality and the model,
/// before any media is decoded and before anything reaches the worker
/// (an engine without an encoder admits such a request and never
/// schedules it, which the client saw as a hang).
#[tokio::test]
async fn image_to_a_language_model_only_pool_is_refused_before_dispatch() {
    let (pipeline, components, seen_ids) =
        chat_pipeline_over(RuntimeType::TokenSpeed, LANGUAGE_MODEL_ONLY).await;
    let started = Instant::now();
    let (status, body) = chat_error_body(&pipeline, components, one_pixel_image_chat()).await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(
        message.starts_with("image input is not supported by model"),
        "{message}"
    );
    assert!(message.contains("language model only"), "{message}");
    assert!(
        seen_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "nothing must reach the worker"
    );
}

/// The chat request of one text part and the one-pixel image, as the
/// vision tests send it.
fn one_pixel_image_chat() -> serde_json::Value {
    serde_json::json!({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What colour is this pixel?"},
                {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}}
            ]
        }],
        "max_completion_tokens": 32,
    })
}

/// The chat pipeline over one stub worker typed as `runtime` that carries
/// `labels`, and the ids the stub saw.
async fn chat_pipeline_over(
    runtime: RuntimeType,
    labels: &[(&str, &str)],
) -> (
    RequestPipeline,
    Arc<SharedComponents>,
    Arc<Mutex<Vec<Vec<u32>>>>,
) {
    let seen_ids = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        seen_input_ids: Arc::clone(&seen_ids),
        ..Default::default()
    })
    .await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker_of_runtime(&worker_registry, port, WorkerType::Regular, runtime, labels);
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline =
        RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
    let components = components_with_multimodal(worker_registry, true).await;
    (pipeline, components, seen_ids)
}

/// The error body of a chat request the pipeline answered within its
/// budget.
async fn chat_error_body(
    pipeline: &RequestPipeline,
    components: Arc<SharedComponents>,
    request: serde_json::Value,
) -> (http::StatusCode, serde_json::Value) {
    let request: Arc<ChatCompletionRequest> =
        Arc::new(serde_json::from_value(request).expect("chat request"));
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        pipeline.execute_chat(
            request,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
            None,
        ),
    )
    .await
    .expect("the pipeline answers within its budget");
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");
    (
        status,
        serde_json::from_slice(&body).expect("json error body"),
    )
}

/// A video to a model whose only worker runs an engine without video in
/// the capability matrix is refused in preparation, before the model's
/// media config is read and before any byte of the clip is fetched or
/// decoded, with the code and the sentence the selection-time check gave
/// after the decode; with and without a sampling hint on the part (the
/// hint decides between router-side and worker-side processing, both
/// after this refusal). The clip is deliberately undecodable: a request
/// that reached the decoder would be answered differently.
#[tokio::test]
async fn video_to_a_pool_whose_engine_takes_no_video_is_refused_before_any_media_is_touched() {
    for video_url in [
        serde_json::json!({"url": "data:video/mp4;base64,invalid", "fps": 0.5}),
        serde_json::json!({"url": "data:video/mp4;base64,invalid"}),
    ] {
        let (pipeline, components, seen_ids) = chat_pipeline_over(RuntimeType::Trtllm, &[]).await;
        let started = Instant::now();
        let (status, body) = chat_error_body(
            &pipeline,
            components,
            serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video briefly."},
                        {"type": "video_url", "video_url": video_url}
                    ]
                }],
                "max_completion_tokens": 32,
            }),
        )
        .await;
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
        assert_eq!(
            body["error"]["message"], "backend trtllm does not support video inputs",
            "{body}"
        );
        assert!(
            seen_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty(),
            "nothing must reach the worker"
        );
    }
}

/// An image to the same worker is not the matrix's refusal: it goes on
/// into multimodal processing (here, to the config lookup the mock
/// tokenizer cannot satisfy, as for the worker without a vision verdict).
#[tokio::test]
async fn image_to_a_pool_whose_engine_takes_images_goes_on_into_media_processing() {
    let (pipeline, components, _) = chat_pipeline_over(RuntimeType::Sglang, &[]).await;
    let (status, body) = chat_error_body(&pipeline, components, one_pixel_image_chat()).await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_ne!(body["error"]["code"], "multimodal_not_supported", "{body}");
    assert_eq!(
        body["error"]["code"], "invalid_multimodal_request",
        "{body}"
    );
}

/// The Messages route shares the refusal: an image block to a model
/// whose only worker runs an engine without image input in the matrix
/// is refused before any media is touched, with the same sentence; the
/// same block to an engine that takes images goes on.
#[tokio::test]
async fn messages_image_to_a_pool_whose_engine_takes_no_image_is_refused_before_any_media_is_touched(
) {
    let (_, png) = ONE_PIXEL_PNG_DATA_URL
        .split_once("base64,")
        .expect("data url");
    let request = serde_json::json!({
        "model": MODEL,
        "max_tokens": 32,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What colour is this pixel?"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": png
                }}
            ]
        }],
    });
    for (runtime, refused) in [(RuntimeType::Mlx, true), (RuntimeType::Sglang, false)] {
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_of_runtime(&worker_registry, port, WorkerType::Regular, runtime, &[]);
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline = RequestPipeline::build(Endpoint::Messages, Mode::Regular, &deps)
            .expect("messages pipeline");
        let components = components_with_multimodal(worker_registry, true).await;
        let request: Arc<CreateMessageRequest> =
            Arc::new(serde_json::from_value(request.clone()).expect("messages request"));
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            pipeline.execute_messages(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
            ),
        )
        .await
        .expect("the refusal must not wait on anything");
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        if refused {
            assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
            assert_eq!(
                body["error"]["message"], "backend mlx does not support image inputs",
                "{body}"
            );
        } else {
            assert_ne!(body["error"]["code"], "multimodal_not_supported", "{body}");
        }
        assert!(
            seen_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty(),
            "nothing must reach the worker"
        );
    }
}

fn completion_request(stream: bool) -> Arc<CompletionRequest> {
    Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "prompt": "Hello world",
            "stream": stream,
        }))
        .expect("completion request"),
    )
}

fn completion_pipeline(worker_registry: &Arc<WorkerRegistry>, mode: Mode) -> RequestPipeline {
    completion_pipeline_with_admission(worker_registry, mode, None)
}

fn completion_pipeline_with_admission(
    worker_registry: &Arc<WorkerRegistry>,
    mode: Mode,
    prefill_admission: Option<Arc<PrefillAdmission>>,
) -> RequestPipeline {
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        prefill_admission,
        None,
        Duration::from_secs(60),
    );
    RequestPipeline::build(Endpoint::Completion, mode, &deps).expect("completion pipeline")
}

fn completion_pipeline_with_policy(
    worker_registry: &Arc<WorkerRegistry>,
    policy: PolicyConfig,
) -> RequestPipeline {
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(policy)),
        None,
        None,
        Duration::from_secs(60),
    );
    RequestPipeline::build(Endpoint::Completion, Mode::Regular, &deps).expect("completion pipeline")
}

/// A request pinned by a routing key whose engine cannot start it
/// (UNAVAILABLE) is retried on another engine, not replayed on the one
/// that just refused: one call there, one on the other engine, 200.
#[tokio::test]
async fn a_pinned_retry_moves_to_another_engine_after_an_unavailable_start() {
    let seen_refusing = Arc::new(Mutex::new(Vec::new()));
    let seen_serving = Arc::new(Mutex::new(Vec::new()));
    let refusing_port = spawn_stub(GatedScheduler {
        fail_always: true,
        seen_request_ids: Arc::clone(&seen_refusing),
        ..Default::default()
    })
    .await;
    let serving_port = spawn_stub(GatedScheduler {
        seen_request_ids: Arc::clone(&seen_serving),
        ..Default::default()
    })
    .await;
    let calls =
        |seen: &Mutex<Vec<String>>| seen.lock().unwrap_or_else(PoisonError::into_inner).len();

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, refusing_port, WorkerType::Regular);
    register_worker(&worker_registry, serving_port, WorkerType::Regular);
    let pipeline =
        completion_pipeline_with_policy(&worker_registry, PolicyConfig::ConsistentHashing);
    let components = components(worker_registry).await;
    let retry = fast_retry_config(3);

    // Keys hash to either engine; the first one owned by the refusing
    // engine is the case under test (the others succeed in one call).
    for key in (0..64).map(|i| format!("session-{i}")) {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", key.parse().unwrap());
        let refused_before = calls(&seen_refusing);
        let served_before = calls(&seen_serving);
        let response = pipeline
            .execute_completion(
                completion_request(false),
                Some(headers),
                MODEL.to_string(),
                Arc::clone(&components),
                None,
                None,
                Some(&retry),
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK, "key {key}");
        let refused_now = calls(&seen_refusing) - refused_before;
        let served_now = calls(&seen_serving) - served_before;
        if refused_now == 0 {
            assert_eq!(served_now, 1, "a key owned by the serving engine: one call");
            continue;
        }
        assert_eq!(
            refused_now, 1,
            "the refusing engine sees the request once, not once per retry"
        );
        assert_eq!(served_now, 1, "the retry landed on the other engine");
        return;
    }
    panic!("no key hashed to the refusing engine");
}

async fn run_and_drain(
    pipeline: RequestPipeline,
    components: Arc<SharedComponents>,
    request: Arc<CompletionRequest>,
    worker_url: &str,
) -> bytes::Bytes {
    let response = pipeline
        .execute_completion(
            request,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        )
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(response.headers()["x-smg-routed-worker-id"], worker_url);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain SSE body")
}

/// The spans the OpenTelemetry layer exported, as a trace backend sees
/// them: name, start and end.
#[derive(Clone, Debug, Default)]
struct ExportedSpans(Arc<Mutex<Vec<opentelemetry_sdk::trace::SpanData>>>);

impl opentelemetry_sdk::trace::SpanExporter for ExportedSpans {
    fn export(
        &self,
        batch: Vec<opentelemetry_sdk::trace::SpanData>,
    ) -> impl Future<Output = opentelemetry_sdk::error::OTelSdkResult> + Send {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(batch);
        std::future::ready(Ok(()))
    }
}

/// The exported upstream span covers the response stream, not the
/// dispatch alone: with the stub pacing its two frames 100 ms apart,
/// `grpc_execute` ends after the stream's end, not before its first frame.
#[tokio::test]
async fn upstream_span_closes_with_the_response_stream() {
    use opentelemetry::trace::TracerProvider as _;
    const FRAME_GAP: Duration = Duration::from_millis(100);
    let request = completion_request(true);
    let port = spawn_stub(GatedScheduler {
        frame_gap: Some(FRAME_GAP),
        ..Default::default()
    })
    .await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;
    // `tracing` rebuilds a callsite's cached interest from the current
    // thread's default alone while it knows of a single dispatcher; a
    // parallel test dispatching first, under no subscriber, would then
    // cache "never" for `grpc_execute`. A second live dispatcher keeps
    // every rebuild over all of them.
    let _second_dispatcher = tracing::Dispatch::new(tracing_subscriber::registry());
    let exported = ExportedSpans::default();
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_simple_exporter(exported.clone())
        .build();
    // The gateway's exporter setup: the OpenTelemetry layer over the
    // gateway's own spans alone (see `CustomOtelFilter`).
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")))
        .with(tracing_subscriber::filter::Targets::new().with_target(
            "smg::otel-trace",
            tracing::level_filters::LevelFilter::TRACE,
        ));

    let (dispatched, drained) = async {
        // A callsite's interest cached before this subscriber existed
        // is rebuilt over the live dispatchers, this one included.
        tracing::callsite::rebuild_interest_cache();
        let response = pipeline
            .execute_completion(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        let dispatched = Instant::now();
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain SSE body");
        (dispatched, Instant::now())
    }
    .with_subscriber(subscriber)
    .await;

    let upstream_spans = || -> Vec<Duration> {
        exported
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|span| span.name == "grpc_execute")
            .map(|span| {
                span.end_time
                    .duration_since(span.start_time)
                    .unwrap_or_default()
            })
            .collect()
    };
    // The body's drop ends the span; give a slow thread a moment.
    for _ in 0..50 {
        if !upstream_spans().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let lasted = upstream_spans();
    assert_eq!(lasted.len(), 1, "one upstream span per dispatch");
    assert!(
        drained >= dispatched + FRAME_GAP,
        "the stub paced the stream over {:?}",
        drained - dispatched
    );
    assert!(
        lasted[0] >= FRAME_GAP,
        "the exported upstream span lasted {:?}: it ended at dispatch, not with the stream ({:?} later)",
        lasted[0],
        drained - dispatched
    );
}

/// The stream task must run off the response spec: the stub refuses to
/// emit tokens until the parsed request has been freed, so a stream that
/// still pinned it would stall past the stub's deadline.
#[tokio::test]
async fn streaming_releases_parsed_request_before_first_token() {
    let request = completion_request(true);
    let released = Arc::new(AtomicBool::new(false));
    let port = spawn_stub(GatedScheduler {
        probe: Some(Arc::downgrade(&request)),
        released: Arc::clone(&released),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let body = run_and_drain(
        pipeline,
        components,
        request,
        &format!("grpc://127.0.0.1:{port}"),
    )
    .await;

    assert!(
        released.load(Ordering::SeqCst),
        "the parsed request must be freed before the first token"
    );
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("data: [DONE]"), "stream must finish: {body}");
}

/// A text-only request puts its token ids on the wire and not a second
/// copy of the rendered prompt: the worker generates from the ids, and a
/// million-token prompt's text would triple the Generate.
#[tokio::test]
async fn text_only_request_sends_its_ids_without_the_prompt_text() {
    let request = completion_request(false);
    let seen_ids = Arc::new(Mutex::new(Vec::new()));
    let seen_texts = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        seen_input_ids: Arc::clone(&seen_ids),
        seen_original_texts: Arc::clone(&seen_texts),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    run_and_drain(
        pipeline,
        components,
        request,
        &format!("grpc://127.0.0.1:{port}"),
    )
    .await;

    let ids = seen_ids.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(ids.len(), 1, "one Generate: {ids:?}");
    assert!(!ids[0].is_empty(), "the prompt's ids are on the wire");
    let texts = seen_texts.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        texts.as_slice(),
        [String::new()],
        "a text-only request sends no prompt text next to its ids"
    );
}

/// grpc_pd twin: the decode side's stream task must not pin the parsed
/// request either (the prefill stub answers immediately).
#[tokio::test]
async fn pd_streaming_releases_parsed_request_before_first_token() {
    let request = completion_request(true);
    let released = Arc::new(AtomicBool::new(false));
    let prefill_port = spawn_stub(GatedScheduler::default()).await;
    let decode_port = spawn_stub(GatedScheduler {
        probe: Some(Arc::downgrade(&request)),
        released: Arc::clone(&released),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
    let components = components(worker_registry).await;

    let body = run_and_drain(
        pipeline,
        components,
        request,
        &format!("grpc://127.0.0.1:{decode_port}"),
    )
    .await;

    assert!(
        released.load(Ordering::SeqCst),
        "the parsed request must be freed before the first decode token"
    );
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("data: [DONE]"), "stream must finish: {body}");
}

/// The parsed request dies at the build boundary even with the retry
/// window open: only the execution plan is retained for replay. The stub
/// refuses to answer the generate RPC until the probe frees.
#[tokio::test]
async fn buffered_dispatch_releases_parsed_request_even_with_retries_enabled() {
    let request = completion_request(false);
    let released = Arc::new(AtomicBool::new(false));
    let port = spawn_stub(GatedScheduler {
        probe: Some(Arc::downgrade(&request)),
        gate_rpc: true,
        released: Arc::clone(&released),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let retry = fast_retry_config(3);
    let response = pipeline
        .execute_completion(
            request,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            Some(&retry),
        )
        .await;

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        response.headers()["x-smg-routed-worker-id"],
        format!("grpc://127.0.0.1:{port}")
    );
    assert!(
        released.load(Ordering::SeqCst),
        "the parsed request must be freed before the upstream answers"
    );
}

/// grpc_pd twin of the dispatch-release probe: the gated decode side
/// answers only after the parsed request is freed.
#[tokio::test]
async fn pd_buffered_dispatch_releases_parsed_request_before_upstream_responds() {
    let request = completion_request(false);
    let released = Arc::new(AtomicBool::new(false));
    let prefill_port = spawn_stub(GatedScheduler::default()).await;
    let decode_port = spawn_stub(GatedScheduler {
        probe: Some(Arc::downgrade(&request)),
        gate_rpc: true,
        released: Arc::clone(&released),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
    let components = components(worker_registry).await;

    let response = pipeline
        .execute_completion(
            request,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        )
        .await;

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        response.headers()["x-smg-routed-worker-id"],
        format!("grpc://127.0.0.1:{decode_port}")
    );
    assert!(
        released.load(Ordering::SeqCst),
        "the parsed request must be freed before the decode side answers"
    );
}

/// The one `smg_worker_errors_total` sample of the gRPC workers of
/// `worker_type`, if any.
fn side_worker_error_line<'a>(rendered: &'a str, worker_type: &str) -> Option<&'a str> {
    let worker_label = format!("worker_type=\"{worker_type}\"");
    rendered.lines().find(|line| {
        line.starts_with("smg_worker_errors_total{")
            && line.contains(&worker_label)
            && line.contains("connection_mode=\"grpc\"")
    })
}

/// A decode side's stream that fails after its first frame while a
/// non-streamed disaggregated answer is collected is the decode worker's
/// failure, as it is for a streamed answer: one sample of
/// `smg_worker_errors_total` under the side's labels per attempt
/// (`engine_error` for the engine's own report), none for the prefill side
/// that completed.
#[test]
fn a_decode_side_that_fails_mid_way_counts_against_its_worker_for_a_non_streamed_answer() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let attempts = metrics::with_local_recorder(&recorder, || {
        let rt = current_thread_runtime();
        rt.block_on(async {
            let decode_calls = Arc::new(Mutex::new(Vec::new()));
            let prefill_port = spawn_stub(GatedScheduler::default()).await;
            let decode_port = spawn_stub(GatedScheduler {
                fail_mid_stream: true,
                seen_request_ids: Arc::clone(&decode_calls),
                ..Default::default()
            })
            .await;
            let worker_registry = Arc::new(WorkerRegistry::new());
            register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
            register_worker(&worker_registry, decode_port, WorkerType::Decode);
            let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
            let components = components(worker_registry).await;

            let response = run_completion(pipeline, components, &fast_retry_config(3)).await;

            assert_eq!(
                response.status(),
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "the engine's error ends a non-streamed answer with a 500"
            );
            let attempts = calls_seen(&decode_calls);
            assert!(attempts >= 1, "the decode side was dispatched");
            attempts
        })
    });
    let rendered = handle.render();
    let line = side_worker_error_line(&rendered, "decode")
        .unwrap_or_else(|| panic!("the decode worker's failure is counted:\n{rendered}"));
    assert!(line.contains("error_type=\"engine_error\""), "{line}");
    assert!(
        line.ends_with(&format!(" {attempts}")),
        "one sample per attempt ({attempts}): {line}"
    );
    assert!(
        side_worker_error_line(&rendered, "prefill").is_none(),
        "the prefill side completed:\n{rendered}"
    );
}

/// The prefill side's stream, drained before the answer, counts the same
/// way under its own labels when it fails after its first frame; the
/// decode side, dropped unread, is not charged.
#[test]
fn a_prefill_side_that_fails_mid_way_counts_against_its_worker_for_a_non_streamed_answer() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = current_thread_runtime();
        rt.block_on(async {
            let prefill_port = spawn_stub(GatedScheduler {
                fail_mid_stream: true,
                ..Default::default()
            })
            .await;
            let decode_port = spawn_stub(GatedScheduler::default()).await;
            let worker_registry = Arc::new(WorkerRegistry::new());
            register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
            register_worker(&worker_registry, decode_port, WorkerType::Decode);
            let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
            let components = components(worker_registry).await;

            let response = run_completion(pipeline, components, &fast_retry_config(1)).await;

            assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        });
    });
    let rendered = handle.render();
    let line = side_worker_error_line(&rendered, "prefill")
        .unwrap_or_else(|| panic!("the prefill worker's failure is counted:\n{rendered}"));
    assert!(line.contains("error_type=\"engine_error\""), "{line}");
    assert!(line.ends_with(" 1"), "{line}");
    assert!(
        side_worker_error_line(&rendered, "decode").is_none(),
        "the decode side did not fail:\n{rendered}"
    );
}

/// The streamed answer's accounting is unchanged by the sides' attribution:
/// a decode side that fails mid-way is one sample, counted where the stream
/// is written, not a second time where it is read.
#[test]
fn a_decode_side_that_fails_mid_way_is_counted_once_for_a_streamed_answer() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = current_thread_runtime();
        rt.block_on(async {
            let prefill_port = spawn_stub(GatedScheduler::default()).await;
            let decode_port = spawn_stub(GatedScheduler {
                fail_mid_stream: true,
                ..Default::default()
            })
            .await;
            let worker_registry = Arc::new(WorkerRegistry::new());
            register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
            register_worker(&worker_registry, decode_port, WorkerType::Decode);
            let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
            let components = components(worker_registry).await;

            let body = run_and_drain(
                pipeline,
                components,
                completion_request(true),
                &format!("grpc://127.0.0.1:{decode_port}"),
            )
            .await;

            let body = String::from_utf8_lossy(&body);
            assert!(
                body.contains("\"error\""),
                "the stream ends with the error event: {body}"
            );
            assert!(body.contains("data: [DONE]"), "{body}");
        });
    });
    let rendered = handle.render();
    let line = side_worker_error_line(&rendered, "decode")
        .unwrap_or_else(|| panic!("the decode worker's failure is counted:\n{rendered}"));
    assert!(line.contains("error_type=\"engine_error\""), "{line}");
    assert!(line.ends_with(" 1"), "counted once: {line}");
}

/// A failed first dispatch retries from the retained plan: the second
/// attempt must send identical token ids without re-tokenizing, under a
/// fresh per-attempt engine id, and the parsed request must already be
/// gone when the first attempt is sent.
#[tokio::test]
async fn retry_replays_identical_token_ids_from_the_retained_plan() {
    let request = completion_request(false);
    let probe = Arc::downgrade(&request);
    let seen_ids = Arc::new(Mutex::new(Vec::new()));
    let seen_request_ids = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        fail_first: true,
        seen_input_ids: Arc::clone(&seen_ids),
        seen_request_ids: Arc::clone(&seen_request_ids),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let retry = fast_retry_config(2);
    let response = pipeline
        .execute_completion(
            request,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            Some(&retry),
        )
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        probe.strong_count(),
        0,
        "request freed at the build boundary"
    );

    {
        let seen = seen_ids.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(seen.len(), 2, "503 then 200 must mean two attempts");
        assert_eq!(
            seen[0], seen[1],
            "the retry must replay identical input ids"
        );
        assert!(
            !seen[0].is_empty(),
            "attempts must carry the tokenized prompt"
        );
    }
    {
        let ids = seen_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        assert_eq!(ids.len(), 2);
        assert!(ids[0].starts_with("cmpl_") && ids[1].starts_with("cmpl_"));
        assert_ne!(ids[0], ids[1], "each attempt gets a fresh engine id");
        // The header switch alone must yield the header: the capture
        // does not depend on the log switch.
        let trace_header = response.headers().get("x-smg-cache-trace");
        if std::env::var("SMG_CACHE_TRACE_HEADER").is_ok_and(|v| v == "1") {
            let trace: serde_json::Value =
                serde_json::from_str(trace_header.unwrap().to_str().unwrap()).unwrap();
            assert_eq!(trace["attempt"], 1);
            assert_eq!(trace["engine_ids"], serde_json::json!([ids[1]]));
            assert!(trace["selections"][0].get("candidates").is_none());
        } else {
            assert!(trace_header.is_none());
        }
    }
}

/// A request whose first attempt fails retryably and whose retry succeeds
/// is one routed request with no error, whose duration runs from the
/// request's start (the backoff included), as the HTTP routers count a
/// request: the attempts are the retry families' count, not this one's.
#[test]
fn a_retried_request_counts_once_with_its_whole_duration_and_no_error() {
    let rt = current_thread_runtime();
    let backoff = Duration::from_millis(50);
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let port = spawn_stub(GatedScheduler {
                fail_first: true,
                ..Default::default()
            })
            .await;
            let worker_registry = Arc::new(WorkerRegistry::new());
            register_worker(&worker_registry, port, WorkerType::Regular);
            let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
            let components = components(worker_registry).await;
            let retry = RetryConfig {
                max_retries: 2,
                initial_backoff_ms: backoff.as_millis() as u64,
                max_backoff_ms: backoff.as_millis() as u64,
                backoff_multiplier: 1.0,
                jitter_factor: 0.0,
            };
            let response = pipeline
                .execute_completion(
                    completion_request(false),
                    None,
                    MODEL.to_string(),
                    components,
                    None,
                    None,
                    Some(&retry),
                )
                .await;
            assert_eq!(response.status(), http::StatusCode::OK);
        });
    });
    let rendered = handle.render();
    let value = |family: &str| -> Option<f64> {
        rendered
            .lines()
            .find(|line| {
                line.starts_with(&format!("{family}{{"))
                    && line.contains(&format!("model=\"{MODEL}\""))
                    && line.contains(&format!(
                        "endpoint=\"{}\"",
                        metrics_labels::ENDPOINT_COMPLETIONS
                    ))
            })
            .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
    };
    assert_eq!(
        value("smg_router_requests_total"),
        Some(1.0),
        "one request, not one per attempt:\n{rendered}"
    );
    assert_eq!(
        value("smg_router_request_errors_total"),
        None,
        "the retried attempt's failure is not the request's answer:\n{rendered}"
    );
    assert_eq!(
        value("smg_router_request_duration_seconds_count"),
        Some(1.0),
        "one duration sample:\n{rendered}"
    );
    assert!(
        value("smg_router_request_duration_seconds_sum")
            .is_some_and(|sum| sum >= backoff.as_secs_f64()),
        "the duration runs from the request's start, the backoff included:\n{rendered}"
    );
}

/// `x-smg-cache-trace: 1` on a request returns that request's decision
/// record, with the chosen worker, whatever the process switches say.
#[tokio::test]
async fn a_request_header_opts_into_the_decision_record_with_its_worker() {
    let port = spawn_stub(GatedScheduler::default()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
    let response = pipeline
        .execute_completion(
            completion_request(false),
            Some(headers),
            MODEL.to_string(),
            Arc::clone(&components),
            None,
            None,
            None,
        )
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    let trace: serde_json::Value =
        serde_json::from_str(response.headers()["x-smg-cache-trace"].to_str().unwrap()).unwrap();
    assert_eq!(trace["attempt"], 0);
    assert_eq!(trace["selections"][0]["worker"], worker.url());
    assert_eq!(trace["selections"][0]["origin"], "policy");
    assert!(trace["selections"][0].get("candidates").is_none());
    assert_eq!(response.headers()["x-smg-routed-worker-id"], worker.url());

    // Without the request header the process switches decide, as before.
    let response = pipeline
        .execute_completion(
            completion_request(false),
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        )
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    if !std::env::var("SMG_CACHE_TRACE_HEADER").is_ok_and(|v| v == "1") {
        assert!(response.headers().get("x-smg-cache-trace").is_none());
    }
}

async fn run_completion(
    pipeline: RequestPipeline,
    components: Arc<SharedComponents>,
    retry: &RetryConfig,
) -> Response {
    run_completion_of(pipeline, components, retry, completion_request(false)).await
}

async fn run_completion_of(
    pipeline: RequestPipeline,
    components: Arc<SharedComponents>,
    retry: &RetryConfig,
    request: Arc<CompletionRequest>,
) -> Response {
    pipeline
        .execute_completion(
            request,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            Some(retry),
        )
        .await
}

/// The completion with a constraint on its output (a JSON schema).
fn constrained_completion_request() -> Arc<CompletionRequest> {
    Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "prompt": "Hello world",
            "stream": false,
            "json_schema": "{\"type\":\"object\"}",
        }))
        .expect("constrained completion request"),
    )
}

fn calls_seen(ids: &Mutex<Vec<String>>) -> usize {
    ids.lock().unwrap_or_else(PoisonError::into_inner).len()
}

/// Three attempts of one request on a worker that cannot serve it now are
/// one failure for its breaker: the threshold counts failed requests, not
/// attempts.
#[tokio::test]
async fn transient_start_failures_charge_the_breaker_once_per_request() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        fail_always: true,
        seen_request_ids: Arc::clone(&seen),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker_with_breaker(&worker_registry, port, 2);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let response = run_completion(pipeline, components, &fast_retry_config(3)).await;

    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        calls_seen(&seen),
        3,
        "an UNAVAILABLE start is replayed up to the retry limit"
    );
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::Closed,
        "one failed request is one failure, under the threshold of two"
    );
}

/// INTERNAL is the engine's own answer for this payload: it is returned
/// at once rather than replayed on the worker, and the worker is charged
/// once.
#[tokio::test]
async fn an_internal_engine_error_is_not_replayed_on_its_worker() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        fail_internal: true,
        seen_request_ids: Arc::clone(&seen),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker_with_breaker(&worker_registry, port, 2);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        calls_seen(&seen),
        1,
        "the answer is definitive for this payload"
    );
    assert!(!is_retryable_response(&response));
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}

/// A worker's report that the engine could not continue the request under
/// its constraint is the request's answer, a 422 `constraint_failed`: one
/// call on the worker and none elsewhere (the same tokens fail the same
/// way on a replay, each replay a prefill), terminal for the retry layer,
/// and no breaker sample against a worker that is healthy.
#[tokio::test]
async fn a_constraint_failure_is_answered_once_and_never_replayed() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        fail_constraint: true,
        seen_request_ids: Arc::clone(&seen),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker_with_breaker(&worker_registry, port, 1);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

    assert_eq!(response.status(), http::StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "constraint_failed"
    );
    assert_eq!(calls_seen(&seen), 1, "the answer is final for this request");
    assert!(!is_retryable_response(&response));
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::Closed,
        "a healthy worker takes no sample"
    );
}

/// The engine's own report on a constrained request (an `error` finish
/// that names no cause: a failed KV handoff or a pause as often as a
/// grammar rejection) is answered exactly as on a request without a
/// constraint: the same status and code (the engine's error, a 500, never
/// the constraint's 422, which only a worker's own attribution produces),
/// the same calls on the worker (the retry layer replaying it as before)
/// and the same breaker state.
#[tokio::test]
async fn an_engines_unattributed_report_on_a_constrained_request_keeps_its_shape() {
    let mut outcomes = Vec::new();
    for request in [completion_request(false), constrained_completion_request()] {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            fail_engine_report: true,
            seen_request_ids: Arc::clone(&seen),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = register_worker_with_breaker(&worker_registry, port, 1);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let response =
            run_completion_of(pipeline, components, &fast_retry_config(2), request).await;
        outcomes.push((
            response.status(),
            error::extract_error_code_from_response(&response).to_string(),
            calls_seen(&seen),
            worker.circuit_breaker_state(),
        ));
    }

    let (plain, constrained) = (&outcomes[0], &outcomes[1]);
    assert_eq!(plain.0, http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_ne!(
        constrained.1, "constraint_failed",
        "no attribution, no constraint failure"
    );
    assert!(
        plain.2 >= 2,
        "the engine's error is replayed as before ({} calls)",
        plain.2
    );
    assert_eq!(
        constrained, plain,
        "a constraint changes nothing about the engine's own report"
    );
}

/// A smaller engine is not an alternative to the engine that returned a
/// definitive error: keep its 500, without a futile retry or a false 400.
#[tokio::test]
async fn context_ineligible_worker_does_not_extend_the_retry_window() {
    let seen_small = Arc::new(Mutex::new(Vec::new()));
    let seen_large = Arc::new(Mutex::new(Vec::new()));
    let small_port = spawn_stub(GatedScheduler {
        seen_request_ids: Arc::clone(&seen_small),
        ..Default::default()
    })
    .await;
    let large_port = spawn_stub(GatedScheduler {
        fail_internal: true,
        seen_request_ids: Arc::clone(&seen_large),
        ..Default::default()
    })
    .await;
    let registry = Arc::new(WorkerRegistry::new());
    for (port, window) in [(small_port, 1), (large_port, 64)] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .model(ModelCard::new(MODEL).with_context_length(window))
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ))
            .expect("register context-bound worker");
    }
    let pipeline = completion_pipeline(&registry, Mode::Regular);
    let response =
        run_completion(pipeline, components(registry).await, &fast_retry_config(5)).await;
    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!is_retryable_response(&response));
    assert_eq!(calls_seen(&seen_small), 0);
    assert_eq!(calls_seen(&seen_large), 1);
}

/// With two workers the request moves to the other one once; a second
/// INTERNAL answer stands.
#[tokio::test]
async fn an_internal_engine_error_moves_the_request_to_another_worker_once() {
    let seen_a = Arc::new(Mutex::new(Vec::new()));
    let seen_b = Arc::new(Mutex::new(Vec::new()));
    let port_a = spawn_stub(GatedScheduler {
        fail_internal: true,
        seen_request_ids: Arc::clone(&seen_a),
        ..Default::default()
    })
    .await;
    let port_b = spawn_stub(GatedScheduler {
        fail_internal: true,
        seen_request_ids: Arc::clone(&seen_b),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker_a = register_worker_with_breaker(&worker_registry, port_a, 5);
    let worker_b = register_worker_with_breaker(&worker_registry, port_b, 5);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(calls_seen(&seen_a), 1);
    assert_eq!(calls_seen(&seen_b), 1);
    assert!(!is_retryable_response(&response));
    assert_eq!(worker_a.circuit_breaker_state(), CircuitState::Closed);
    assert_eq!(worker_b.circuit_breaker_state(), CircuitState::Closed);
}

/// Once the only worker's breaker opens, the verdict clears at the
/// breaker timeout, not inside a backoff: the 503 ends the retry window.
#[tokio::test]
async fn a_pool_of_open_breakers_ends_the_retry_window_at_once() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        fail_always: true,
        seen_request_ids: Arc::clone(&seen),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker_with_breaker(&worker_registry, port, 1);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(worker_registry).await;

    let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    assert_eq!(calls_seen(&seen), 1, "the open breaker takes no attempt");
    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "no_available_workers"
    );
    assert!(!is_retryable_response(&response));
}

/// A prefill side that cannot start must answer the client immediately
/// rather than waiting out the decode side's own deadline, and the decode
/// room it stranded must be aborted rather than left for the engine to
/// time out. The decode stub here answers only after `DECODE_DELAY`,
/// standing in for an engine whose transfer deadline is what the client
/// used to wait on.
#[tokio::test]
async fn a_failed_prefill_answers_now_and_aborts_the_decode_room() {
    const DECODE_DELAY: Duration = Duration::from_secs(2);

    let prefill_port = spawn_stub(GatedScheduler {
        fail_always: true,
        ..Default::default()
    })
    .await;
    let aborted = Arc::new(Mutex::new(Vec::new()));
    let decode_request_ids = Arc::new(Mutex::new(Vec::new()));
    let decode_port = spawn_stub(GatedScheduler {
        answer_after: Some(DECODE_DELAY),
        aborted_request_ids: Arc::clone(&aborted),
        seen_request_ids: Arc::clone(&decode_request_ids),
        ..Default::default()
    })
    .await;

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
    let components = components(worker_registry).await;

    let started = Instant::now();
    let response = pipeline
        .execute_completion(
            completion_request(false),
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        )
        .await;
    let answered_in = started.elapsed();

    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "prefill_worker_failed_to_start"
    );
    assert!(
        answered_in < DECODE_DELAY,
        "the prefill failure must not wait for the decode side, took {answered_in:?}"
    );

    // The decode side is retired off the request path: once its dispatch
    // lands, its stream drops and that is what sends the abort.
    let deadline = Instant::now() + DECODE_DELAY + Duration::from_secs(8);
    loop {
        if !aborted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the stranded decode room was never aborted"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let aborted = aborted.lock().unwrap_or_else(PoisonError::into_inner);
    let dispatched = decode_request_ids
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        aborted.as_slice(),
        dispatched.as_slice(),
        "the abort must name the decode side's own request id"
    );
}

// ------------------------------------------------------------------
// Request completion on a client cancel. Policies learn that a request
// ended through the worker's completion sink, which the load guard
// drives; these pin that a client leaving mid-request reaches it on the
// gRPC paths, promptly, while the engine still owes its first token.
// ------------------------------------------------------------------

/// Records the worker URLs that reported a request completion.
#[derive(Debug, Default)]
struct CompletionSpy(Mutex<Vec<String>>);

impl RequestCompletionSink for CompletionSpy {
    fn request_completed(&self, worker: &dyn Worker) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(worker.url().to_string());
    }
}

impl CompletionSpy {
    fn urls(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Wait for `expected` completions, well inside the stub's five-second
    /// gate: a completion that only arrives with the engine's tokens is a
    /// failure, not a late pass.
    async fn wait_for(&self, expected: usize) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(2);
        while self.urls().len() < expected {
            assert!(
                Instant::now() < deadline,
                "only {:?} completed in time, expected {expected}",
                self.urls()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.urls()
    }
}

/// Install one spy as the completion sink of every registered worker.
fn install_spy(registry: &WorkerRegistry) -> Arc<CompletionSpy> {
    let spy = Arc::new(CompletionSpy::default());
    for worker in registry.get_all() {
        worker.set_completion_sink(Some(Arc::clone(&spy) as Arc<dyn RequestCompletionSink>));
    }
    spy
}

/// A stub that accepts the generate RPC at once and withholds every token
/// while `hold` lives: the engine has the request and owes its first
/// token, the window a client cancel is hardest to account for.
fn stalled_while(hold: &Arc<CompletionRequest>) -> GatedScheduler {
    GatedScheduler {
        probe: Some(Arc::downgrade(hold)),
        ..Default::default()
    }
}

async fn start_stream(pipeline: &RequestPipeline, components: Arc<SharedComponents>) -> Response {
    let started = Instant::now();
    let response = pipeline
        .execute_completion(
            completion_request(true),
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        )
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the streaming response must open before the engine's first token"
    );
    response
}

/// gRPC regular path: the client drops the stream while the engine is
/// still prefilling. The load guard rides the response body, so the drop
/// releases the worker's load and reports the completion at once.
#[tokio::test]
async fn cancelled_grpc_stream_reports_the_completion_during_prefill() {
    let hold = completion_request(true);
    let port = spawn_stub(stalled_while(&hold)).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let spy = install_spy(&worker_registry);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(Arc::clone(&worker_registry)).await;

    let response = start_stream(&pipeline, components).await;
    assert_eq!(worker.load(), 1, "the dispatch holds the worker's load");
    assert!(
        spy.urls().is_empty(),
        "nothing completes while the client is connected"
    );

    drop(response);

    assert_eq!(spy.wait_for(1).await, [worker.url().to_string()]);
    assert_eq!(worker.load(), 0);
    drop(hold);
}

/// `response`'s body read by a client of its own, as a connection reads it:
/// a streamed body's reading runs when the body is polled (see
/// `DrivenBody`), so a response held unread would hold the engine's stream
/// unread too. Aborting the client's task drops the body, as the client's
/// leaving does.
#[expect(
    clippy::disallowed_methods,
    reason = "a test client reading the body; aborting it is the client's leaving"
)]
fn read_body(response: Response) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut body = response.into_body().into_data_stream();
        while body.next().await.is_some() {}
    })
}

/// Wait until the processor has read the first response of `worker`'s
/// one stream: the stream's output ticket is open from that response on
/// (see `Worker::output_silence`), and the processor waits on the engine
/// for the next one. The test tokenizer renders no text for the stub's
/// token, so the client's body carries no frame to wait for.
async fn first_response_read(worker: &Arc<dyn Worker>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while worker.output_silence().is_none() {
        assert!(
            Instant::now() < deadline,
            "the first response never reached the processor"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Wait for the stub to receive the abort of its one request: within
/// `deadline`, well inside the 30 s it holds a wedged stream open, so an
/// abort that only arrives with the engine's next frame is a failure.
async fn wait_for_abort(scheduler: &GatedScheduler, deadline: Instant) -> Duration {
    let started = Instant::now();
    let request_id = scheduler
        .seen_request_ids
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .first()
        .cloned()
        .expect("the stub saw the request");
    while !scheduler
        .aborted_request_ids
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains(&request_id)
    {
        assert!(
            Instant::now() < deadline,
            "the stub never received the abort for {request_id} after the client left"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    started.elapsed()
}

/// gRPC regular path: the client leaves while the engine's generation
/// wedged after its first token (read by the processor, which then waits
/// on the engine). The abort reaches the engine the moment the body is
/// dropped, not when the engine next speaks, and the worker's load goes
/// with the body.
#[tokio::test]
async fn a_client_that_leaves_a_stalled_stream_aborts_it_on_the_engine_at_once() {
    let scheduler = GatedScheduler {
        hold_after_first: true,
        ..Default::default()
    };
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = components(Arc::clone(&worker_registry)).await;

    let response = start_stream(&pipeline, components).await;
    let client = read_body(response);
    first_response_read(&worker).await;
    assert!(
        scheduler
            .aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "nothing is aborted while the client reads"
    );
    assert_eq!(worker.load(), 1);

    // The client leaves: its task goes, and the body with it.
    client.abort();

    let deadline = Instant::now() + Duration::from_secs(2);
    wait_for_abort(&scheduler, deadline).await;
    while worker.load() != 0 {
        assert!(
            Instant::now() < deadline,
            "the worker's load stayed at {} after the body was dropped",
            worker.load()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// gRPC PD path: the same client leaves while the decode side wedged after
/// its first token. The decode engine receives the abort at once and
/// both sides end with their load released.
#[tokio::test]
async fn a_client_that_leaves_a_stalled_pd_stream_aborts_the_decode_side_at_once() {
    let prefill_port = spawn_stub(GatedScheduler::default()).await;
    let decode_scheduler = GatedScheduler {
        hold_after_first: true,
        ..Default::default()
    };
    let decode_port = spawn_stub(decode_scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let prefill = register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    let decode = register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
    let components = components(Arc::clone(&worker_registry)).await;

    let response = start_stream(&pipeline, components).await;
    let client = read_body(response);
    first_response_read(&decode).await;
    assert!(
        decode_scheduler
            .aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "nothing is aborted while the client reads"
    );

    // The client leaves: its task goes, and the body with it.
    client.abort();

    let deadline = Instant::now() + Duration::from_secs(2);
    wait_for_abort(&decode_scheduler, deadline).await;
    while prefill.load() != 0 || decode.load() != 0 {
        assert!(
            Instant::now() < deadline,
            "the sides' load stayed at {} and {} after the body was dropped",
            prefill.load(),
            decode.load()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// The Responses route over a chat pipeline of `worker_registry`: the
/// in-memory stores, no MCP servers, `background_slots` background
/// responses at once (0: the mode is off).
async fn responses_context(
    worker_registry: &Arc<WorkerRegistry>,
    background_slots: usize,
) -> ResponsesContext {
    responses_context_over(
        Endpoint::Chat,
        Mode::Regular,
        worker_registry,
        background_slots,
    )
    .await
}

/// [`responses_context`] over the `endpoint` pipeline of `worker_registry`
/// in `mode`: the chat pipeline, whose body the Responses relay reads, or
/// the Harmony pipeline, whose engine stream the Harmony reader reads.
async fn responses_context_over(
    endpoint: Endpoint,
    mode: Mode,
    worker_registry: &Arc<WorkerRegistry>,
    background_slots: usize,
) -> ResponsesContext {
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline = RequestPipeline::build(endpoint, mode, &deps)
        .unwrap_or_else(|| panic!("a {endpoint:?} pipeline in {mode:?} mode"));
    ResponsesContext::new(
        Arc::new(pipeline),
        components(Arc::clone(worker_registry)).await,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
        Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("mcp orchestrator"),
        ),
        FormatRegistry::new(),
        None,
        Arc::new(BackgroundResponses::new(background_slots)),
    )
}

/// A streamed Responses request for the stub's model, in the background
/// or not.
fn responses_request(background: bool) -> Arc<ResponsesRequest> {
    Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "input": "hi",
            "stream": true,
            "background": background,
        }))
        .expect("responses request"),
    )
}

/// The caller of these Responses requests: no tenant.
fn anonymous() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("anonymous"))
}

/// The status of the response record stored under `id`; `Null` when
/// none is.
async fn stored_status(ctx: &ResponsesContext, id: &str) -> serde_json::Value {
    ctx.response_storage
        .get_response(&ResponseId::from(id))
        .await
        .expect("the store answers")
        .map_or(serde_json::Value::Null, |stored| {
            stored.raw_response["status"].clone()
        })
}

/// Wait until `worker`'s load is back at 0, within `deadline`.
async fn wait_for_load_released(worker: &Arc<dyn Worker>, deadline: Instant) {
    while worker.load() != 0 {
        assert!(
            Instant::now() < deadline,
            "the worker's load stayed at {} after the body was dropped",
            worker.load()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Wait until the record under `id` reads "completed" (the background
/// generation ran on after its client left), within ten seconds; then prove
/// `scheduler` aborted nothing and `worker`'s load is back at 0.
async fn background_generation_completes_unaborted(
    ctx: &ResponsesContext,
    id: &str,
    scheduler: &GatedScheduler,
    worker: &Arc<dyn Worker>,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while stored_status(ctx, id).await != "completed" {
        assert!(
            Instant::now() < deadline,
            "the background response did not complete after its client left: {}",
            stored_status(ctx, id).await
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        scheduler
            .aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "a background generation is not aborted when its client leaves"
    );
    wait_for_load_released(worker, deadline).await;
}

/// Responses API over the gRPC regular path: the client leaves a
/// foreground stream while the engine's generation wedged after its
/// first token (read by the chat layer, whose body the Responses relay
/// waits on). The relay drops the chat body the moment the client is
/// gone and the abort reaches the engine through the chat layer at once,
/// not when the engine next speaks; the worker's load goes with it.
#[tokio::test]
async fn a_client_that_leaves_a_stalled_responses_stream_aborts_it_on_the_engine_at_once() {
    let scheduler = GatedScheduler {
        hold_after_first: true,
        ..Default::default()
    };
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let ctx = responses_context(&worker_registry, 0).await;

    let response = route_responses(
        &ctx,
        responses_request(false),
        None,
        anonymous(),
        MODEL.to_string(),
    )
    .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    first_response_read(&worker).await;
    assert!(
        scheduler
            .aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "nothing is aborted while the client reads"
    );
    assert_eq!(worker.load(), 1);

    drop(response);

    let deadline = Instant::now() + Duration::from_secs(2);
    wait_for_abort(&scheduler, deadline).await;
    wait_for_load_released(&worker, deadline).await;
}

/// Responses API, `background: true`: the generation outlives its
/// client on purpose. The client leaves the stream once the engine's
/// first token was read; nothing is aborted, the work runs to the
/// engine's end and the record under the response id completes.
#[tokio::test]
async fn a_client_that_leaves_a_background_responses_stream_leaves_its_generation_to_complete() {
    // The engine's first token after a second; its end only once the
    // test releases it, after the client is gone.
    let generation_end = Arc::new(AtomicBool::new(false));
    let scheduler = GatedScheduler {
        frame_gap: Some(Duration::from_secs(1)),
        gate_after_first: Some(Arc::clone(&generation_end)),
        ..Default::default()
    };
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let ctx = responses_context(&worker_registry, 1).await;
    let request = responses_request(true);
    let id = "resp_background_client_leaves";
    let slot = ctx.background.admit(id).expect("a background slot");
    store_record(
        &ctx.response_storage,
        &queued_response(id, &request),
        &request,
        None,
        None,
    )
    .await
    .expect("the queued record is stored");

    let response = route_responses_background_stream(
        &ctx,
        request,
        None,
        anonymous(),
        MODEL.to_string(),
        BackgroundStream::new(slot),
    )
    .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    first_response_read(&worker).await;
    assert_ne!(
        stored_status(&ctx, id).await,
        "completed",
        "the client leaves before the generation ends"
    );

    drop(response);
    generation_end.store(true, Ordering::SeqCst);

    background_generation_completes_unaborted(&ctx, id, &scheduler, &worker).await;
}

/// The id a streamed Responses body announces in its first frame
/// (`response.created`).
fn announced_response_id(frame: &[u8]) -> String {
    let frame = std::str::from_utf8(frame).expect("UTF-8 SSE");
    let data = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("a data line in the first frame");
    let event: serde_json::Value = serde_json::from_str(data).expect("SSE JSON");
    assert_eq!(event["type"], "response.created", "{event}");
    event["response"]["id"]
        .as_str()
        .expect("the response id")
        .to_string()
}

/// Harmony Responses over the gRPC regular path, where the Harmony reader
/// reads the engine stream itself: the client leaves a foreground stream
/// while the engine's generation wedged after its first token. The abort
/// reaches the engine the moment the body is dropped, not when the engine
/// next speaks; the worker's load goes with the body; and nothing of the
/// abandoned stream is stored, not a completed record of its partial
/// output.
#[tokio::test]
async fn a_client_that_leaves_a_stalled_harmony_responses_stream_aborts_it_on_the_engine_at_once() {
    let scheduler = GatedScheduler {
        hold_after_first: true,
        ..Default::default()
    };
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let ctx = responses_context_over(Endpoint::Harmony, Mode::Regular, &worker_registry, 0).await;

    let response = serve_harmony_responses_stream(
        &ctx,
        Arc::unwrap_or_clone(responses_request(false)),
        anonymous(),
        None,
    )
    .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    // The first frame announces the response; the rest of the body is
    // left unread, as a client that stopped reading leaves it.
    let mut body = response.into_body().into_data_stream();
    let first = body
        .next()
        .await
        .expect("the first frame")
        .expect("a readable frame");
    let id = announced_response_id(&first);
    first_response_read(&worker).await;
    assert!(
        scheduler
            .aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "nothing is aborted while the client reads"
    );
    assert_eq!(worker.load(), 1);

    drop(body);

    let deadline = Instant::now() + Duration::from_secs(2);
    wait_for_abort(&scheduler, deadline).await;
    wait_for_load_released(&worker, deadline).await;
    assert_eq!(
        stored_status(&ctx, &id).await,
        serde_json::Value::Null,
        "nothing of an abandoned foreground stream is stored"
    );
}

/// Harmony Responses over the gRPC regular path: the request's `stop`
/// sequences reach the engine on the request the Harmony pipeline built, as
/// a chat request's do; the stub reads them off its sampling parameters.
#[tokio::test]
async fn a_harmony_responses_request_s_stop_sequences_reach_the_engine() {
    let scheduler = GatedScheduler::default();
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let _worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let ctx = responses_context_over(Endpoint::Harmony, Mode::Regular, &worker_registry, 0).await;
    let request: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "input": "hi",
        "stream": true,
        "stop": ["STOP", "\n\n"],
    }))
    .expect("responses request");

    let response = serve_harmony_responses_stream(&ctx, request, anonymous(), None).await;
    assert_eq!(response.status(), http::StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the stream's body");
    assert!(!body.is_empty(), "the stream answers");

    let seen = scheduler
        .seen_stops
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(seen, vec![vec!["STOP".to_string(), "\n\n".to_string()]]);
}

/// Harmony Responses over the gRPC PD path: the same client leaves while
/// the decode side wedged after its first token. The decode engine receives
/// the abort at once and both sides end with their load released.
#[tokio::test]
async fn a_client_that_leaves_a_stalled_harmony_responses_pd_stream_aborts_the_decode_side_at_once()
{
    let prefill_port = spawn_stub(GatedScheduler::default()).await;
    let decode_scheduler = GatedScheduler {
        hold_after_first: true,
        ..Default::default()
    };
    let decode_port = spawn_stub(decode_scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let prefill = register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    let decode = register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let ctx =
        responses_context_over(Endpoint::Harmony, Mode::PrefillDecode, &worker_registry, 0).await;

    let response = serve_harmony_responses_stream(
        &ctx,
        Arc::unwrap_or_clone(responses_request(false)),
        anonymous(),
        None,
    )
    .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    first_response_read(&decode).await;
    assert!(
        decode_scheduler
            .aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "nothing is aborted while the client reads"
    );

    drop(response);

    let deadline = Instant::now() + Duration::from_secs(2);
    wait_for_abort(&decode_scheduler, deadline).await;
    while prefill.load() != 0 || decode.load() != 0 {
        assert!(
            Instant::now() < deadline,
            "the sides' load stayed at {} and {} after the body was dropped",
            prefill.load(),
            decode.load()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Harmony Responses over the gRPC PD path, the client leaving while the
/// prefill side wedged after its first token: the prefill drain is cut by
/// the client's leaving, both sides drop unmarked (the prefill engine
/// receives the abort at once) and both sides' load returns with the body.
#[tokio::test]
async fn a_client_that_leaves_a_stalled_harmony_responses_pd_stream_during_the_prefill_aborts_both_sides_at_once(
) {
    let prefill_scheduler = GatedScheduler {
        hold_after_first: true,
        ..Default::default()
    };
    let prefill_port = spawn_stub(prefill_scheduler.clone()).await;
    let decode_scheduler = GatedScheduler::default();
    let decode_port = spawn_stub(decode_scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let prefill = register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    let decode = register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let ctx =
        responses_context_over(Endpoint::Harmony, Mode::PrefillDecode, &worker_registry, 0).await;

    let response = serve_harmony_responses_stream(
        &ctx,
        Arc::unwrap_or_clone(responses_request(false)),
        anonymous(),
        None,
    )
    .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    first_response_read(&prefill).await;
    assert!(
        prefill_scheduler
            .aborted_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "nothing is aborted while the client reads"
    );

    drop(response);

    let deadline = Instant::now() + Duration::from_secs(2);
    // Both sides drop unmarked: each engine receives the abort.
    wait_for_abort(&prefill_scheduler, deadline).await;
    wait_for_abort(&decode_scheduler, deadline).await;
    while prefill.load() != 0 || decode.load() != 0 {
        assert!(
            Instant::now() < deadline,
            "the sides' load stayed at {} and {} after the body was dropped during the prefill",
            prefill.load(),
            decode.load()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Harmony Responses, `background: true`: the generation outlives its
/// client on purpose. The client leaves the stream once the engine's
/// first token was read; nothing is aborted, the work runs to the
/// engine's end and the record under the response id completes.
#[tokio::test]
async fn a_client_that_leaves_a_background_harmony_responses_stream_leaves_its_generation_to_complete(
) {
    // The engine's first token after a second, its end a second later:
    // the client is gone in between.
    let scheduler = GatedScheduler {
        frame_gap: Some(Duration::from_secs(1)),
        ..Default::default()
    };
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let ctx = responses_context_over(Endpoint::Harmony, Mode::Regular, &worker_registry, 1).await;
    let request = responses_request(true);
    let id = "resp_background_harmony_client_leaves";
    let slot = ctx.background.admit(id).expect("a background slot");
    store_record(
        &ctx.response_storage,
        &queued_response(id, &request),
        &request,
        None,
        None,
    )
    .await
    .expect("the queued record is stored");

    let response = serve_harmony_responses_stream(
        &ctx,
        Arc::unwrap_or_clone(request),
        anonymous(),
        Some(BackgroundStream::new(slot)),
    )
    .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    first_response_read(&worker).await;
    assert_ne!(
        stored_status(&ctx, id).await,
        "completed",
        "the client leaves before the generation ends"
    );

    drop(response);

    background_generation_completes_unaborted(&ctx, id, &scheduler, &worker).await;
}

/// gRPC PD path: the prefill side answers on its own and reports when its
/// phase ends; the decode side is still owed its first token when the
/// client leaves, and reports with the dropped body. Both sides end with
/// their load released, exactly once each.
#[tokio::test]
async fn cancelled_pd_stream_reports_the_prefill_and_decode_completions() {
    let hold = completion_request(true);
    let prefill_port = spawn_stub(GatedScheduler::default()).await;
    let decode_port = spawn_stub(stalled_while(&hold)).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let prefill = register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    let decode = register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let spy = install_spy(&worker_registry);
    let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
    let components = components(Arc::clone(&worker_registry)).await;

    let response = start_stream(&pipeline, components).await;
    assert_eq!(
        decode.load(),
        1,
        "the decode side holds its load until the stream ends"
    );

    drop(response);

    let mut urls = spy.wait_for(2).await;
    urls.sort();
    let mut expected = vec![prefill.url().to_string(), decode.url().to_string()];
    expected.sort();
    assert_eq!(urls, expected, "each side reports exactly once");
    assert_eq!(prefill.load(), 0);
    assert_eq!(decode.load(), 0);
    drop(hold);
}

/// gRPC PD path under prefill admission: the prefill side's load is a
/// reservation in the admission gate rather than a bare guard, and it
/// still reports the completion when the phase ends, so a cancelled
/// request frees its slot for the next one.
#[tokio::test]
async fn cancelled_pd_stream_with_admitted_prefill_reports_and_frees_the_slot() {
    let hold = completion_request(true);
    let prefill_port = spawn_stub(GatedScheduler::default()).await;
    let decode_port = spawn_stub(stalled_while(&hold)).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let prefill = register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
    let decode = register_worker(&worker_registry, decode_port, WorkerType::Decode);
    let spy = install_spy(&worker_registry);
    let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
    let pipeline = completion_pipeline_with_admission(
        &worker_registry,
        Mode::PrefillDecode,
        Some(Arc::clone(&admission)),
    );
    let components = components(Arc::clone(&worker_registry)).await;

    let response = start_stream(&pipeline, components).await;
    drop(response);

    let mut urls = spy.wait_for(2).await;
    urls.sort();
    let mut expected = vec![prefill.url().to_string(), decode.url().to_string()];
    expected.sort();
    assert_eq!(urls, expected, "each side reports exactly once");
    assert_eq!(
        prefill.load(),
        0,
        "the admitted prefill's reservation is released"
    );
    assert_eq!(decode.load(), 0);
    // The one slot is free again: a new admission does not queue.
    let admitted = tokio::time::timeout(
        Duration::from_millis(500),
        admission.admit(None, select_worker(&prefill)),
    )
    .await
    .expect("the freed slot must admit at once")
    .expect("admitted");
    drop(admitted);
    drop(hold);
}

// ------------------------------------------------------------------
// DeepSeek-V4.1 parity through the pipeline with the real checkpoint
// tokenizer: `DEEPSEEK_V41_MODEL_DIR`, else the tokenizer files fetched
// once into the package's cache; skips only when the hub is unreachable,
// like the tokenizer crate's parity test.
// ------------------------------------------------------------------

const V41_RENDER_FIXTURES: &str = include_str!(
    "../../../../../../crates/tokenizer/tests/fixtures/deepseek_v41/render_fixtures.json"
);
const V41_ID_FIXTURES: &str = include_str!(
    "../../../../../../crates/tokenizer/tests/fixtures/deepseek_v41/render_ids_fixtures.json"
);
const V41_CORN_PNG: &[u8] = include_bytes!(
    "../../../../../../crates/multimodal/tests/fixtures/images/deepseek_v41_corn.png"
);
/// `<｜deepseek_image｜>` in the checkpoint's `config.json`.
const V41_IMAGE_TOKEN_ID: u32 = 129264;
/// The reference span for `deepseek_v41_corn.png` (450x308), recorded in
/// the multimodal crate's goldens.
const V41_CORN_TOKENS: usize = 189;

/// The checkpoint's tokenizer directory, fetched on a blocking thread: the
/// download blocks, and must not run on the runtime.
async fn deepseek_v41_model_dir() -> Option<PathBuf> {
    tokio::task::spawn_blocking(tokenizer_cache::ensure_deepseek_v41_cached)
        .await
        .expect("the download thread")
}

#[expect(
    clippy::print_stderr,
    reason = "test diagnostic: says why the parity test did not run"
)]
fn skip_no_tokenizer() {
    eprintln!(
        "skipping: no DeepSeek-V4.1 tokenizer (the hub is unreachable; set {})",
        tokenizer_cache::DEEPSEEK_V41_MODEL_DIR_ENV
    );
}

/// The recorded request (`messages`, kwargs) and reference ids of one
/// tokenizer-crate fixture case.
fn v41_fixture(name: &str) -> (serde_json::Value, Vec<u32>) {
    let render: serde_json::Value =
        serde_json::from_str(V41_RENDER_FIXTURES).expect("render fixtures");
    let ids: serde_json::Value = serde_json::from_str(V41_ID_FIXTURES).expect("id fixtures");
    let find = |doc: &serde_json::Value| {
        doc["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("fixture case {name}"))
            .clone()
    };
    let case = find(&render);
    let ids = find(&ids)["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| id.as_u64().expect("token id") as u32)
        .collect();
    (case, ids)
}

async fn v41_components(
    dir: &Path,
    worker_registry: Arc<WorkerRegistry>,
    with_multimodal: bool,
) -> Arc<SharedComponents> {
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().expect("utf-8 tokenizer path"))
            .expect("load the DeepSeek-V4.1 tokenizer"),
    ) as Arc<dyn Tokenizer>;
    // The tokenizer's source is the checkpoint directory: the multimodal
    // config registry reads `config.json` (image_token_id) from it.
    let source = dir.to_string_lossy().into_owned();
    tokenizer_registry
        .load(
            "deepseek-v41",
            MODEL,
            &source,
            || async move { Ok(tokenizer) },
        )
        .await
        .expect("register the DeepSeek-V4.1 tokenizer");
    shared_components(tokenizer_registry, worker_registry, with_multimodal)
}

/// Status, body, and the input ids and placeholders of every attempt a
/// chat request produced.
type ChatRun = (
    http::StatusCode,
    bytes::Bytes,
    Vec<Vec<u32>>,
    Vec<PlaceholderRanges>,
);

/// Runs one chat request against a recording TokenSpeed stub.
async fn v41_run_chat(dir: &Path, request: serde_json::Value, with_multimodal: bool) -> ChatRun {
    v41_run_chat_on(dir, request, with_multimodal, &[]).await
}

/// [`v41_run_chat`] with `worker_labels` on the stub's worker.
async fn v41_run_chat_on(
    dir: &Path,
    request: serde_json::Value,
    with_multimodal: bool,
    worker_labels: &[(&str, &str)],
) -> ChatRun {
    let seen_ids = Arc::new(Mutex::new(Vec::new()));
    let seen_mm = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_stub(GatedScheduler {
        seen_input_ids: Arc::clone(&seen_ids),
        seen_mm_placeholders: Arc::clone(&seen_mm),
        ..Default::default()
    })
    .await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker_with_labels(&worker_registry, port, WorkerType::Regular, worker_labels);
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline =
        RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
    let components = v41_components(dir, worker_registry, with_multimodal).await;
    let request: Arc<ChatCompletionRequest> =
        Arc::new(serde_json::from_value(request).expect("chat request"));
    let response = pipeline
        .execute_chat(
            request,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
            None,
        )
        .await;
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let ids = seen_ids
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let mm = seen_mm
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    (status, body, ids, mm)
}

/// The prompt ids the worker receives equal the reference encoder's ids
/// for a recorded multi-turn chat-mode case, and a `continue_final_message`
/// request keeps its trailing assistant message and renders it the
/// reference way (no EOS, no generation header) instead of as a prefix.
/// The thinking-mode case is the one recorded at the checkpoint's current
/// revision: the tokenizer directory here carries no encoder, so the
/// renderer writes that revision's effort budget.
#[tokio::test]
async fn deepseek_v41_prompt_ids_match_the_reference_through_the_pipeline() {
    let Some(dir) = deepseek_v41_model_dir().await else {
        skip_no_tokenizer();
        return;
    };

    let (case, expected) = v41_fixture("hf_2");
    let (status, body, ids, _) = v41_run_chat(
        &dir,
        serde_json::json!({
            "model": MODEL,
            "messages": case["messages"],
            "chat_template_kwargs": {"thinking": false, "drop_thinking": true},
            "max_tokens": 1,
        }),
        false,
    )
    .await;
    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        ids,
        vec![expected],
        "hf_2: prompt ids differ from the reference"
    );

    let (case, expected) = v41_fixture("continue_final_message_current");
    let (status, body, ids, _) = v41_run_chat(
        &dir,
        serde_json::json!({
            "model": MODEL,
            "messages": case["messages"],
            "continue_final_message": true,
            "chat_template_kwargs": {"thinking": true, "drop_thinking": true},
            "max_tokens": 1,
        }),
        false,
    )
    .await;
    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        ids,
        vec![expected],
        "continue_final_message: prompt ids differ from the reference's no-EOS rendering"
    );
}

/// A renderer validation error (an effort level the reference rejects) is
/// the client's mistake, so it surfaces as 400, not 500.
#[tokio::test]
async fn deepseek_v41_invalid_effort_is_a_bad_request() {
    let Some(dir) = deepseek_v41_model_dir().await else {
        skip_no_tokenizer();
        return;
    };
    let (status, body, ids, _) = v41_run_chat(
        &dir,
        serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "turbo",
            "max_tokens": 1,
        }),
        false,
    )
    .await;
    let body = String::from_utf8_lossy(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("Invalid reasoning effort"), "{body}");
    assert!(ids.is_empty(), "nothing must reach the worker");
}

/// `medium` is a level of the checkpoint's own chat template (budget
/// 62): the top-level field and the template kwarg both render it
/// instead of refusing the request.
#[tokio::test]
async fn deepseek_v41_medium_effort_renders_the_checkpoints_budget() {
    let Some(dir) = deepseek_v41_model_dir().await else {
        skip_no_tokenizer();
        return;
    };
    let tokenizer = HuggingFaceTokenizer::from_file(
        dir.join("tokenizer.json")
            .to_str()
            .expect("utf-8 tokenizer path"),
    )
    .expect("load the DeepSeek-V4.1 tokenizer");
    for request in [
        serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "medium",
            "max_tokens": 1,
        }),
        serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "chat_template_kwargs": {"reasoning_effort": "medium"},
            "max_tokens": 1,
        }),
    ] {
        let (status, body, ids, _) = v41_run_chat(&dir, request.clone(), false).await;
        assert_eq!(
            status,
            http::StatusCode::OK,
            "{request}: {}",
            String::from_utf8_lossy(&body)
        );
        let prompt = tokenizer.decode(&ids[0], false).expect("decode the prompt");
        assert!(
            prompt.contains("Reasoning Effort: 62 (range"),
            "{request}: {prompt}"
        );
    }
}

/// One image expands to exactly the reference span: `<｜deepseek_image｜>`
/// is inlined at the part's position by the renderer and replaced by
/// `image_token_id` repeated once per span position (189 for the corn
/// example), and the worker is told that span as the placeholder range.
#[tokio::test]
async fn deepseek_v41_image_expands_to_the_reference_span() {
    let Some(dir) = deepseek_v41_model_dir().await else {
        skip_no_tokenizer();
        return;
    };
    let data_url = format!("data:image/png;base64,{}", BASE64.encode(V41_CORN_PNG));
    let (status, body, ids, mm) = v41_run_chat(
        &dir,
        serde_json::json!({
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this picture?"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            "chat_template_kwargs": {"thinking": false},
            "max_tokens": 1,
        }),
        true,
    )
    .await;
    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(ids.len(), 1);
    let prompt = &ids[0];
    let count = prompt
        .iter()
        .filter(|&&id| id == V41_IMAGE_TOKEN_ID)
        .count();
    assert_eq!(count, V41_CORN_TOKENS, "image span length");
    let first = prompt
        .iter()
        .position(|&id| id == V41_IMAGE_TOKEN_ID)
        .expect("the image span is present");
    assert!(
        prompt[first..first + V41_CORN_TOKENS]
            .iter()
            .all(|&id| id == V41_IMAGE_TOKEN_ID),
        "the image span is contiguous"
    );
    assert_eq!(mm, vec![vec![(first as u32, V41_CORN_TOKENS as u32)]]);
}

/// A dispatch the engine answered with RESOURCE_EXHAUSTED is counted as
/// the rate-limit error the client reads in the 429 body, not as an
/// internal error of the router.
#[test]
fn a_dispatch_the_engine_throttled_is_counted_as_a_rate_limit_error() {
    let worker_registry = Arc::new(WorkerRegistry::new());
    // The model is served: its errors are counted under its own id (an
    // unserved name is counted under `other`).
    register_worker(&worker_registry, 1, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let failure = Status::resource_exhausted("kv cache full")
        .to_http_error("engine_busy", "kv cache full".to_string());
    assert_eq!(failure.status(), http::StatusCode::TOO_MANY_REQUESTS);

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        pipeline.record_error(Some(metrics_labels::ENDPOINT_COMPLETIONS), MODEL, &failure);
    });
    let rendered = handle.render();
    let line = rendered
        .lines()
        .find(|line| {
            line.starts_with("smg_router_request_errors_total{")
                && line.contains(&format!("model=\"{MODEL}\""))
                && line.contains(&format!(
                    "endpoint=\"{}\"",
                    metrics_labels::ENDPOINT_COMPLETIONS
                ))
        })
        .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
    assert!(line.contains("error_type=\"rate_limit_error\""), "{line}");
    assert!(line.contains("reason=\"overloaded\""), "{line}");
    assert!(line.ends_with(" 1"), "{line}");
}

/// Two dispatch failures of one class with different causes are two
/// series of the error family, one per `reason`, that sum to the count
/// the family had before the label existed; the gateway's own refusal
/// carries the reason its code names.
#[test]
fn a_router_error_carries_the_reason_behind_its_class() {
    use crate::{observability::metrics::test_support::render_with_recorder, routers::error};

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, 1, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let failed =
        |status: Status| status.to_http_error("start_generation_failed", "failed".to_string());
    let rendered = render_with_recorder(|| {
        for failure in [
            failed(Status::unavailable("draining")),
            failed(Status::cancelled("stream reset")),
            failed(Status::cancelled("connection closed")),
            error::service_unavailable(
                placement::NO_AVAILABLE_WORKERS_CODE,
                "every worker is down",
            ),
        ] {
            pipeline.record_error(Some(metrics_labels::ENDPOINT_COMPLETIONS), MODEL, &failure);
        }
    });
    let samples: Vec<&str> = rendered
        .lines()
        .filter(|line| {
            line.starts_with("smg_router_request_errors_total{")
                && line.contains(&format!("model=\"{MODEL}\""))
        })
        .collect();
    let value = |error_type: &str, reason: &str| -> u64 {
        let line = samples
            .iter()
            .find(|line| {
                line.contains(&format!("error_type=\"{error_type}\""))
                    && line.contains(&format!("reason=\"{reason}\""))
            })
            .unwrap_or_else(|| panic!("a {error_type}/{reason} series:\n{rendered}"));
        line.rsplit(' ').next().unwrap().parse().unwrap()
    };
    assert_eq!(samples.len(), 3, "{rendered}");
    assert_eq!(value("backend_error", "engine_unavailable"), 1);
    assert_eq!(value("backend_error", "transport_disconnect"), 2);
    assert_eq!(value("unavailable", "no_available_workers"), 1);
}

/// The same image to the same model served by a worker without a vision
/// encoder: refused with a 400 in preparation, nothing preprocessed or
/// dispatched (the engine would admit the request and never schedule it).
#[tokio::test]
async fn deepseek_v41_image_to_a_language_model_only_worker_is_refused() {
    let Some(dir) = deepseek_v41_model_dir().await else {
        skip_no_tokenizer();
        return;
    };
    let data_url = format!("data:image/png;base64,{}", BASE64.encode(V41_CORN_PNG));
    let (status, body, ids, mm) = v41_run_chat_on(
        &dir,
        serde_json::json!({
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this picture?"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            "chat_template_kwargs": {"thinking": false},
            "max_tokens": 1,
        }),
        true,
        LANGUAGE_MODEL_ONLY,
    )
    .await;
    let body = String::from_utf8_lossy(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("multimodal_not_supported"), "{body}");
    assert!(body.contains("language model only"), "{body}");
    assert!(ids.is_empty(), "nothing must reach the worker");
    assert!(mm.is_empty(), "nothing must be preprocessed");
}

/// The same image with the worker labels discovery derives from each
/// servicer's `GetModelInfo`: a servicer that left `supports_vision`
/// unset (SGLang, TokenSpeed) gives no verdict, so the image is
/// preprocessed and dispatched as to any vision worker; vLLM's deliberate
/// `false` refuses it.
#[tokio::test]
async fn deepseek_v41_image_follows_the_servicer_vision_verdict() {
    let Some(dir) = deepseek_v41_model_dir().await else {
        skip_no_tokenizer();
        return;
    };
    let request = || {
        serde_json::json!({
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this picture?"},
                    {"type": "image_url", "image_url": {"url": format!(
                        "data:image/png;base64,{}", BASE64.encode(V41_CORN_PNG))}}
                ]
            }],
            "chat_template_kwargs": {"thinking": false},
            "max_tokens": 1,
        })
    };
    for (name, info) in [
        ("sglang", ModelInfo::Sglang(Box::default())),
        ("tokenspeed", ModelInfo::TokenSpeed(Box::default())),
    ] {
        let labels = labels_of(&info);
        let (status, body, ids, mm) =
            v41_run_chat_on(&dir, request(), true, &label_pairs(&labels)).await;
        assert_eq!(
            status,
            http::StatusCode::OK,
            "{name}: {}",
            String::from_utf8_lossy(&body)
        );
        assert_eq!(ids.len(), 1, "{name}: dispatched once");
        assert_eq!(
            mm.iter().map(Vec::len).sum::<usize>(),
            1,
            "{name}: the image span reached the worker"
        );
    }

    let labels = labels_of(&ModelInfo::Vllm(vllm_proto::GetModelInfoResponse {
        supports_vision: false,
        ..Default::default()
    }));
    let (status, body, ids, _) =
        v41_run_chat_on(&dir, request(), true, &label_pairs(&labels)).await;
    let body = String::from_utf8_lossy(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("multimodal_not_supported"), "{body}");
    assert!(ids.is_empty(), "nothing must reach the worker");
}

/// The one `smg_worker_errors_total` sample of the regular gRPC workers.
fn regular_worker_error_line(rendered: &str) -> &str {
    rendered
        .lines()
        .find(|line| {
            line.starts_with("smg_worker_errors_total{")
                && line.contains("worker_type=\"regular\"")
                && line.contains("connection_mode=\"grpc\"")
        })
        .unwrap_or_else(|| panic!("a worker error is counted:\n{rendered}"))
}

/// Every attempt a worker refuses before the first token is one sample of
/// `smg_worker_errors_total` under the worker's labels, however the
/// breaker is charged (once per request): three UNAVAILABLE starts of one
/// request are three worker errors and one breaker failure.
#[test]
fn a_refusal_before_the_first_token_counts_against_the_worker_once_per_attempt() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = current_thread_runtime();
        rt.block_on(async {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let port = spawn_stub(GatedScheduler {
                fail_always: true,
                seen_request_ids: Arc::clone(&seen),
                ..Default::default()
            })
            .await;
            let worker_registry = Arc::new(WorkerRegistry::new());
            let worker = register_worker_with_breaker(&worker_registry, port, 5);
            let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
            let components = components(worker_registry).await;

            let response = run_completion(pipeline, components, &fast_retry_config(3)).await;

            assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(calls_seen(&seen), 3, "three attempts on the one worker");
            assert_eq!(
                worker.circuit_breaker_state(),
                CircuitState::Closed,
                "one failed request is one breaker failure"
            );
        });
    });
    let rendered = handle.render();
    let line = regular_worker_error_line(&rendered);
    assert!(line.contains("error_type=\"backend_error\""), "{line}");
    assert!(line.ends_with(" 3"), "one sample per attempt: {line}");
}

/// A worker stream that fails after its first frame, read into a
/// non-streamed answer, is the worker's failure as it is for a streamed
/// one: one sample of `smg_worker_errors_total`, `engine_error` for the
/// engine's own report.
#[test]
fn a_worker_stream_that_fails_mid_way_counts_against_the_worker_for_a_non_streamed_answer() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = current_thread_runtime();
        rt.block_on(async {
            let port = spawn_stub(GatedScheduler {
                fail_mid_stream: true,
                ..Default::default()
            })
            .await;
            let worker_registry = Arc::new(WorkerRegistry::new());
            register_worker(&worker_registry, port, WorkerType::Regular);
            let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
            let components = components(worker_registry).await;

            let response = run_completion(pipeline, components, &fast_retry_config(1)).await;

            assert_eq!(
                response.status(),
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "the engine's error ends a non-streamed answer with a 500"
            );
        });
    });
    let rendered = handle.render();
    let line = regular_worker_error_line(&rendered);
    assert!(line.contains("error_type=\"engine_error\""), "{line}");
    assert!(line.ends_with(" 1"), "{line}");
}

/// A completion for a model nobody serves and no tokenizer knows is
/// refused by the preparation stage with the 404, before selection: the
/// pipeline reports it once (the response marked, so the stage line and
/// the access log keep quiet) and writes no ERROR event. The one refusal
/// path a stage-level test cannot reach.
#[tokio::test]
async fn a_404_from_the_preparation_stage_is_reported_once_and_marked() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ErrorCounter(Arc<AtomicUsize>);
    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for ErrorCounter {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() == tracing::Level::ERROR {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    let errors = Arc::new(AtomicUsize::new(0));
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ErrorCounter(Arc::clone(&errors))),
    );
    let worker_registry = Arc::new(WorkerRegistry::new());
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(Arc::new(TokenizerRegistry::new()))),
        tokenizer_registry: Arc::new(TokenizerRegistry::new()),
        worker_registry,
        tool_parser_factory: ToolParserFactory::default(),
        reasoning_parser_factory: ReasoningParserFactory::default(),
        parser_resolver: utils::ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: true,
        tokenizer_load_failures: Arc::default(),
        parser: ParserChoice::Legacy,
    });

    let response = pipeline
        .execute_completion(
            completion_request(false),
            None,
            "nobody-serves-this".to_string(),
            components,
            None,
            None,
            Some(&fast_retry_config(1)),
        )
        .await;

    assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        placement::MODEL_NOT_FOUND_CODE
    );
    assert!(
        response.extensions().get::<ReportedRefusal>().is_some(),
        "the pipeline reported the 404, so the stage line and the access log keep quiet"
    );
    assert_eq!(
        errors.load(Ordering::Relaxed),
        0,
        "a model nobody serves is the client's mistake, not an error of the gateway"
    );
}

/// A refusal the gateway answers on its own (every worker of the model
/// unavailable) is counted as `unavailable`, not as a backend's error: no
/// worker was contacted.
#[test]
fn a_refusal_for_a_drained_pool_is_counted_as_unavailable_not_as_a_backend_error() {
    let worker_registry = Arc::new(WorkerRegistry::new());
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let refusal = placement::no_available_workers("every worker is down");
    assert_eq!(refusal.status(), http::StatusCode::SERVICE_UNAVAILABLE);

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        pipeline.record_error(Some(metrics_labels::ENDPOINT_COMPLETIONS), MODEL, &refusal);
    });
    let rendered = handle.render();
    let line = rendered
        .lines()
        .find(|line| {
            // The model label is bounded to the served models (an
            // unregistered one reads "other"), so the line is found by
            // its router, as the HTTP and PD tests find theirs.
            line.starts_with("smg_router_request_errors_total{")
                && line.contains(&format!("router_type=\"{}\"", metrics_labels::ROUTER_GRPC))
        })
        .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
    assert!(line.contains("error_type=\"unavailable\""), "{line}");
    assert!(line.ends_with(" 1"), "{line}");
}

/// The gateway's request timeout on a gRPC worker, in the shapes the
/// issue's reproduction had: a streamed completion whose engine takes
/// the request and says nothing ends at the deadline with the HTTP
/// path's `stream_error` event under the timeout's message and `[DONE]`,
/// the engine receives the abort for that request before the client
/// reads the event, and the worker's load goes with the body.
#[tokio::test]
async fn a_streamed_request_to_a_silent_worker_ends_at_the_request_timeout_and_is_aborted() {
    let (pipeline, worker, scheduler, components) =
        silent_worker_pipeline(SilentShape::Stream, REQUEST_TIMEOUT).await;
    let started = Instant::now();
    let response = pipeline
        .execute_completion(
            completion(true),
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        )
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    let body = tokio::time::timeout(
        CASE_CAP,
        axum::body::to_bytes(response.into_body(), usize::MAX),
    )
    .await
    .expect("the request timeout ends the body: without it the gateway waits on the silent worker as long as the client does")
    .expect("drain body");
    let elapsed = started.elapsed();
    let body = String::from_utf8_lossy(&body);
    assert!(
        elapsed >= REQUEST_TIMEOUT && elapsed < CASE_CAP,
        "ended after {elapsed:?}: {body}"
    );
    assert!(body.contains("\"type\":\"stream_error\""), "{body}");
    assert!(body.contains("request timed out after 300ms"), "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    assert_aborted_and_released(&scheduler, &worker).await;
}

/// The same request non-streamed is the HTTP path's 504 for its client
/// timeout, `call_upstream_timeout`, at the deadline; the engine request
/// is aborted and the slot released as for the streamed one.
#[tokio::test]
async fn a_non_streamed_request_to_a_silent_worker_is_the_504_at_the_request_timeout() {
    let (pipeline, worker, scheduler, components) =
        silent_worker_pipeline(SilentShape::Stream, REQUEST_TIMEOUT).await;
    let started = Instant::now();
    let response = tokio::time::timeout(
        CASE_CAP,
        pipeline.execute_completion(
            completion(false),
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        ),
    )
    .await
    .expect("the request timeout answers: without it the gateway waits on the silent worker as long as the client does");
    let elapsed = started.elapsed();
    assert_timeout_response(response, elapsed).await;
    assert_aborted_and_released(&scheduler, &worker).await;
}

/// A worker that never answers the call that opens the stream (the
/// request never admitted, the servicer wedged before its first frame)
/// is bounded by the same deadline: the call is dropped at the deadline,
/// which cancels it on the worker, and the client reads the 504.
#[tokio::test]
async fn a_call_the_worker_never_answers_is_the_504_at_the_request_timeout() {
    let (pipeline, worker, _scheduler, components) =
        silent_worker_pipeline(SilentShape::Call, REQUEST_TIMEOUT).await;
    let started = Instant::now();
    let response = tokio::time::timeout(
        CASE_CAP,
        pipeline.execute_completion(
            completion(false),
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        ),
    )
    .await
    .expect("the request timeout answers: without it the gateway waits on the call as long as the client does");
    let elapsed = started.elapsed();
    assert_timeout_response(response, elapsed).await;
    assert_eq!(worker.load(), 0, "the slot goes with the answer");
}

/// A worker that answers within the timeout is not touched by it: the
/// completion arrives whole, the engine sees no abort.
#[tokio::test]
async fn a_request_that_completes_before_the_request_timeout_is_unchanged() {
    let scheduler = GatedScheduler::default();
    let aborted = Arc::clone(&scheduler.aborted_request_ids);
    let port = spawn_stub(scheduler).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    )
    .with_request_timeout(Duration::from_secs(30));
    let pipeline = RequestPipeline::build(Endpoint::Completion, Mode::Regular, &deps)
        .expect("completion pipeline");
    let components = components(Arc::clone(&worker_registry)).await;
    for stream in [true, false] {
        let response = pipeline
            .execute_completion(
                completion(stream),
                None,
                MODEL.to_string(),
                Arc::clone(&components),
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
        assert!(!body.contains("timed out"), "{body}");
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        aborted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty(),
        "a completed request sends no abort"
    );
    assert_eq!(worker.load(), 0);
}

/// The request timeout under test and the cap a gateway without it fails at.
const REQUEST_TIMEOUT: Duration = Duration::from_millis(300);
const CASE_CAP: Duration = Duration::from_secs(10);

/// Where the worker falls silent: after it answered the call with an
/// open stream (`Stream`), or before answering the call at all (`Call`).
#[derive(Clone, Copy)]
enum SilentShape {
    Stream,
    Call,
}

/// A completion pipeline with `request_timeout` over one stub worker that
/// takes the request and is silent for longer than any case runs.
async fn silent_worker_pipeline(
    shape: SilentShape,
    request_timeout: Duration,
) -> (
    RequestPipeline,
    Arc<dyn Worker>,
    GatedScheduler,
    Arc<SharedComponents>,
) {
    let silence = Duration::from_secs(3600);
    let scheduler = match shape {
        SilentShape::Stream => GatedScheduler {
            frame_gap: Some(silence),
            ..Default::default()
        },
        SilentShape::Call => GatedScheduler {
            answer_after: Some(silence),
            ..Default::default()
        },
    };
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&worker_registry, port, WorkerType::Regular);
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    )
    .with_request_timeout(request_timeout);
    let pipeline = RequestPipeline::build(Endpoint::Completion, Mode::Regular, &deps)
        .expect("completion pipeline");
    let components = components(Arc::clone(&worker_registry)).await;
    (pipeline, worker, scheduler, components)
}

fn completion(stream: bool) -> Arc<CompletionRequest> {
    Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "prompt": "Hello world",
            "max_tokens": 8,
            "stream": stream,
        }))
        .expect("completion request"),
    )
}

/// The HTTP path's 504 for its client timeout, at the deadline.
async fn assert_timeout_response(response: Response, elapsed: Duration) {
    assert_eq!(response.status(), http::StatusCode::GATEWAY_TIMEOUT);
    assert!(
        elapsed >= REQUEST_TIMEOUT && elapsed < CASE_CAP,
        "answered after {elapsed:?}"
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let error: serde_json::Value = serde_json::from_slice(&body).expect("a JSON body");
    assert_eq!(error["error"]["code"], "call_upstream_timeout", "{error}");
    assert_eq!(error["error"]["type"], "server_error", "{error}");
    assert!(
        error["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("request timed out after 300ms")),
        "{error}"
    );
}

/// The stub received the abort for the one request it saw, and the
/// worker's load is back to zero.
async fn assert_aborted_and_released(scheduler: &GatedScheduler, worker: &Arc<dyn Worker>) {
    let request_id = scheduler
        .seen_request_ids
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .first()
        .cloned()
        .expect("the stub saw the request");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !scheduler
        .aborted_request_ids
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains(&request_id)
    {
        assert!(
            Instant::now() < deadline,
            "the stub never received the abort for {request_id}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    while worker.load() != 0 {
        assert!(
            Instant::now() < deadline,
            "the worker's load never returned to zero"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Media the request got wrong as it sent it is refused under the media
/// stage's own `invalid_multimodal_request`, a 400 with the connector's
/// words, and nothing reaches the worker: a data URL that is not base64, one
/// with no comma, one in the non-base64 form, a payload that is not an
/// image, a URL that does not parse, a host the policy refuses, and a
/// `max_long_side_pixel` out of range. Before, every one of these was a
/// `media_fetch_failed` with nothing fetched.
#[tokio::test]
async fn deepseek_v41_media_refused_as_sent_answers_the_media_stages_code() {
    let Some(dir) = deepseek_v41_model_dir().await else {
        skip_no_tokenizer();
        return;
    };
    let corn = format!("data:image/png;base64,{}", BASE64.encode(V41_CORN_PNG));
    let cases = [
        (
            "not base64",
            serde_json::json!({"url": "data:image/png;base64,!!!"}),
            "base64 decode error",
        ),
        (
            "no comma",
            serde_json::json!({"url": "data:image/png;base64"}),
            "missing comma in data url",
        ),
        (
            "not the base64 form",
            serde_json::json!({"url": "data:image/svg+xml,<svg/>"}),
            "only base64 encoded data URLs are supported",
        ),
        (
            "not an image",
            serde_json::json!({"url": "data:application/pdf;base64,JVBERi0xLjQK"}),
            "image decode error",
        ),
        (
            "not a URL",
            serde_json::json!({"url": "not a url"}),
            "invalid media URL",
        ),
        (
            "a loopback host",
            serde_json::json!({"url": "http://127.0.0.1:9/x.png"}),
            "loopback",
        ),
        (
            "a cap out of range",
            serde_json::json!({"url": corn, "max_long_side_pixel": 0}),
            "max_long_side_pixel",
        ),
    ];
    for (name, image_url, words) in cases {
        let (status, body, ids, _) = v41_run_chat(
            &dir,
            serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this picture?"},
                        {"type": "image_url", "image_url": image_url}
                    ]
                }],
                "chat_template_kwargs": {"thinking": false},
                "max_tokens": 1,
            }),
            true,
        )
        .await;
        let body: serde_json::Value = serde_json::from_slice(&body).expect("a JSON envelope");
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{name}: {body}");
        assert_eq!(
            body["error"]["code"], "invalid_multimodal_request",
            "{name}: {body}"
        );
        assert_eq!(
            body["error"]["type"], "invalid_request_error",
            "{name}: {body}"
        );
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains(words), "{name}: {body}");
        assert!(ids.is_empty(), "{name}: nothing reaches the worker");
    }
}
