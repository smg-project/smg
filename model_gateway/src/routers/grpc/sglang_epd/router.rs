//! Explicit managed chat ingress; ordinary router factories never construct this path.

use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    http::HeaderMap,
    response::{IntoResponse, Response},
    Json,
};
use llm_tokenizer::{
    factory::create_tokenizer_async,
    traits::{PromptEncoding, Tokenizer},
};
use openai_protocol::{chat::ChatCompletionRequest, profile::ProviderProfile};
use prost::Message;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use smg_grpc_client::{
    sglang_managed::{proto, Limits, LAYOUT, VERSION},
    sglang_scheduler::{SglangGenerateRequestOptions, SglangSchedulerClient},
};
use tonic::Status;
use uuid::Uuid;

use super::{
    executor::{AttemptObserver, Executor},
    planner::Planner,
    registry::{wall_ms, Registry},
};
use crate::{
    app_context::AppContext,
    middleware::TenantRequestMeta,
    rate_limit::RateLimitManager,
    routers::{
        common::rate_limit_cell::RateLimitCell,
        error,
        grpc::{
            context::{DispatchMetadata, ExecutionResult},
            proto_wrapper::ProtoStream,
            regular::{processor::ResponseProcessor, streaming::StreamingProcessor},
            spec::ChatResponseSpec,
            utils::{
                create_stop_decoder, process_chat_messages_with_encoding, ParserResolver,
                StreamSource,
            },
        },
        http::pd_types::generate_room_id,
        RouterTrait,
    },
};

pub struct ManagedRouter {
    config: Config,
    tokenizer: Arc<dyn Tokenizer>,
    registry: Arc<Registry>,
    executor: Executor,
    processor: ResponseProcessor,
    streaming: Arc<StreamingProcessor>,
    rate_limit_manager: Option<Arc<RateLimitManager>>,
}

impl fmt::Debug for ManagedRouter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedRouter")
            .field("model", &self.config.model)
            .finish_non_exhaustive()
    }
}

impl ManagedRouter {
    /// Keep probe reads constant-time while managed workers use their own capability registry.
    pub(crate) fn maintain_readiness(
        &self,
        probe: Arc<crate::health::ProbeState>,
    ) -> tokio::task::JoinHandle<()> {
        let registry = Arc::downgrade(&self.registry);
        #[expect(
            clippy::disallowed_methods,
            reason = "one weakly owned readiness monitor per managed router"
        )]
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                let Some(registry) = registry.upgrade() else {
                    break;
                };
                probe.publish_managed(registry.readiness());
            }
        })
    }

    /// Build the independent planner while retaining the gateway's auth and response machinery.
    pub async fn new(ctx: &Arc<AppContext>, path: &str) -> Result<Self, String> {
        let bytes = tokio::fs::read(path).await.map_err(|e| e.to_string())?;
        if bytes.len() > 65536 {
            return Err("managed deployment configuration exceeds 64 KiB".into());
        }
        let config: Config = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let config = config.resolve().map_err(|e| e.to_string())?;
        let model: Value = serde_json::from_slice(
            &tokio::fs::read(std::path::Path::new(&config.model_path).join("config.json"))
                .await
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if model["model_type"].as_str() != Some("qwen3_vl") {
            return Err("managed EPD has no validated layout adapter for this model family".into());
        }
        let tokenizer = create_tokenizer_async(&config.model_path)
            .await
            .map_err(|e| e.to_string())?;
        let tool = ctx
            .tool_parser_factory
            .clone()
            .ok_or("managed router requires tool parser factory")?;
        let reasoning = ctx
            .reasoning_parser_factory
            .clone()
            .ok_or("managed router requires reasoning parser factory")?;
        let resolver = ParserResolver::new(
            ctx.worker_registry.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
            &tool,
            &reasoning,
        );
        let processor = ResponseProcessor::new(tool.clone(), reasoning.clone(), resolver.clone())
            .with_backend("sglang_managed_epd");
        let streaming = Arc::new(StreamingProcessor::new(
            tool,
            reasoning,
            resolver,
            "sglang_managed_epd",
        ));
        let registry = Registry::new(config.clone()).map_err(|e| e.to_string())?;
        let executor = Executor::new(
            config.max_attempts,
            Limits::default(),
            Duration::from_secs(10),
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            config,
            tokenizer,
            registry,
            executor,
            processor,
            streaming,
            rate_limit_manager: ctx.rate_limit_manager.clone(),
        })
    }

    /// Render references and text only: this gateway never fetches or preprocesses media.
    async fn execute_chat(
        &self,
        body: &ChatCompletionRequest,
        tenant: &TenantRequestMeta,
        started: Instant,
        reservation: &RateLimitCell,
    ) -> Result<(ExecutionResult, DispatchMetadata, ChatResponseSpec), Status> {
        let original_expiry = wall_ms()?.saturating_add(self.config.timeout_ms);
        if body.model != self.config.model
            || body.n.unwrap_or(1) != 1
            || body.session_params.is_some()
            || body.declares_tools()
            || body.return_hidden_states
        {
            return Err(Status::invalid_argument(
                "managed EPD requires configured model, n=1, no session/tools/hidden states",
            ));
        }
        let request_bytes =
            serde_json::to_vec(body).map_err(|e| Status::invalid_argument(e.to_string()))?;
        if request_bytes.len() > Limits::default().manifest_bytes {
            return Err(Status::resource_exhausted(
                "managed request exceeds control byte limit",
            ));
        }
        let images = image_references(body)?;
        if images.len() > self.config.max_images {
            return Err(Status::invalid_argument(
                "managed image count exceeds capability",
            ));
        }
        let (processed, encoding) = process_chat_messages_with_encoding(
            body,
            self.tokenizer.as_ref(),
            Some("<|vision_start|><|image_pad|><|vision_end|>"),
        )
        .map_err(Status::invalid_argument)?;
        if !matches!(encoding, PromptEncoding::FromText) {
            return Err(Status::unimplemented(
                "managed EPD requires a flat native prompt layout",
            ));
        }
        let tokens = self
            .tokenizer
            .encode(&processed.text, false)
            .map_err(|e| Status::invalid_argument(e.to_string()))?
            .token_ids()
            .to_vec();
        if tokens.is_empty() || tokens.len() > Limits::default().tokens {
            return Err(Status::invalid_argument(
                "managed prompt length exceeds capability",
            ));
        }
        if let Some(manager) = &self.rate_limit_manager {
            reservation
                .reserve(manager, tenant, &body.model, tokens.len() as u32)
                .await
                .map_err(|response| {
                    let mut status = Status::resource_exhausted("tenant rate limit exceeded");
                    if let Some(value) = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse().ok())
                    {
                        status.metadata_mut().insert("retry-after", value);
                    }
                    status
                })?;
        }
        let attempt = Uuid::now_v7().to_string();
        let request_id = format!("chatcmpl-{}", Uuid::now_v7());
        let (pairs, prefill_capabilities, generation) = self.registry.candidates(&request_id)?;
        let total = Duration::from_millis(self.config.timeout_ms);
        let remaining = total
            .checked_sub(started.elapsed())
            .ok_or_else(|| Status::deadline_exceeded("managed preparation expired"))?;
        let planner = Planner::new(
            self.config.max_pd_candidates,
            self.config.max_encoder_candidates,
        )?;
        let (pair_index, encoder_index, score) = planner.choose(&pairs, images.len(), remaining)?;
        let pair = &pairs[pair_index];
        let prefill_capability = &prefill_capabilities[pair_index];
        let identity = proto::Identity {
            logical_request_id: request_id.clone(),
            attempt_id: attempt.clone(),
            branch_id: 0,
        };
        let request_digest = Sha256::digest(request_bytes).to_vec();
        let encoder = encoder_index.map(|index| &pair.encoders[index]);
        let prefill_rid = encoder.map_or_else(
            || format!("{attempt}-p"),
            |e| format!("{}.{attempt}-p", e.worker.instance.generation),
        );
        let mut binding = proto::InputBinding {
            decode: Some(proto::OperationRef {
                identity: Some(identity.clone()),
                target: Some(pair.decode.instance.clone()),
                operation_id: format!("{attempt}-d"),
                request_digest: request_digest.clone(),
                plan_digest: vec![0; 32],
            }),
            prefill: Some(pair.prefill.instance.clone()),
            prefill_rid,
            bootstrap: Some(proto::Bootstrap {
                host: prefill_capability.bootstrap_host.clone(),
                port: prefill_capability.bootstrap_port,
                room: generate_room_id() as i64,
            }),
            model_revision: self.config.model_revision.clone(),
            processor_contract: self.config.processor_contract.clone(),
            layout_contract: LAYOUT.into(),
            media_routing_labels: vec![],
        };
        let mut hash = Sha256::new();
        hash.update(binding.encode_to_vec());
        hash.update(pair.prefill.control_endpoint.as_bytes());
        hash.update(pair.decode.control_endpoint.as_bytes());
        if let Some(encoder) = encoder {
            hash.update(encoder.worker.instance.encode_to_vec());
            hash.update(encoder.worker.control_endpoint.as_bytes());
            hash.update(encoder.worker.receive_endpoint.as_bytes());
        }
        let plan_digest = hash.finalize().to_vec();
        binding
            .decode
            .as_mut()
            .ok_or_else(|| Status::internal("missing D operation"))?
            .plan_digest
            .clone_from(&plan_digest);
        let plan = Arc::new(planner.freeze(
            pair,
            encoder_index,
            images.len(),
            binding,
            score,
            generation,
        )?);
        let binding = plan.binding();
        let operation = |target: proto::Instance, suffix: &str| proto::OperationRef {
            identity: Some(identity.clone()),
            target: Some(target),
            operation_id: format!("{attempt}-{suffix}"),
            request_digest: request_digest.clone(),
            plan_digest: plan_digest.clone(),
        };
        // Work units are predicted milliseconds, so observed queue work and service priors share a unit.
        let work = encoder.map_or(1, |e| {
            e.cost.preprocess_ms.saturating_add(e.cost.encode_ms).max(1)
        });
        let budget = proto::Budget {
            original_expiry_unix_ms: original_expiry,
            remaining_timeout_ms: remaining.as_millis() as u64,
            estimated_work: work,
            context_bytes: (Limits::default().manifest_bytes * 2) as u64,
        };
        let mut d_generation = SglangSchedulerClient::build_generate_request_from_chat_parts(
            format!("{attempt}-d"),
            body,
            processed.text.clone(),
            tokens,
            SglangGenerateRequestOptions::default(),
        )
        .map_err(Status::invalid_argument)?;
        let mut p_generation = d_generation.clone();
        p_generation.request_id.clone_from(&binding.prefill_rid);
        p_generation
            .sampling_params
            .as_mut()
            .ok_or_else(|| Status::invalid_argument("missing sampling parameters"))?
            .max_new_tokens = Some(1);
        if !images.is_empty() {
            d_generation
                .tokenized
                .as_mut()
                .ok_or_else(|| Status::internal("missing D input"))?
                .input_ids
                .clear();
        }
        let prefill = proto::PrefillStart {
            version: VERSION,
            operation: Some(operation(pair.prefill.instance.clone(), "p")),
            budget: Some(budget),
            binding: Some(binding.clone()),
            generation: Some(p_generation),
            encode_plan: Some(plan.external().clone()),
        };
        let decode = proto::DecodeStart {
            version: VERSION,
            binding: Some(binding.clone()),
            budget: Some(budget),
            generation: Some(d_generation),
            deferred_from_prefill: !images.is_empty(),
        };
        let encode = plan.encoder().map(|e| proto::EncodeRequest {
            version: VERSION,
            operation: Some(operation(e.instance.clone(), "e")),
            budget: Some(budget),
            receiver_binding: Some(binding.clone()),
            plan: Some(plan.external().clone()),
            image_references: images,
            transfer_backend: e.contract.transfer.clone(),
            tenant_id: tenant.tenant_key().as_str().to_owned(),
        });
        let mut operations = vec![
            (operation(pair.prefill.instance.clone(), "p"), false),
            (operation(pair.decode.instance.clone(), "d"), false),
        ];
        if let Some(encoder) = plan.encoder() {
            operations.push((operation(encoder.instance.clone(), "e"), true));
        }
        self.registry.reserve(&operations, work)?;
        metrics::histogram!("smg_managed_epd_predicted_ttft_seconds").record(score as f64 / 1000.0);
        tracing::info!(request_id, attempt, predicted_ttft_ms = score, registry_generation = generation,
            prefill = %pair.prefill.instance.worker_id, decode = %pair.decode.instance.worker_id,
            encoder = plan.encoder().map(|e| e.instance.worker_id.as_str()), "managed EPD plan frozen");
        let observer: Arc<dyn AttemptObserver> = self.registry.clone();
        let remaining = total.saturating_sub(started.elapsed());
        let response = match self
            .executor
            .execute(plan, prefill, decode, encode, remaining, observer)
        {
            Ok(response) => response,
            Err(error) => {
                for (operation, _) in &operations {
                    self.registry.termination(operation, true);
                }
                return Err(error);
            }
        };
        tracing::info!(
            request_id,
            attempt_id = attempt,
            predicted_ttft_ms = score,
            "managed EPD plan dispatched"
        );
        let dispatch = DispatchMetadata {
            request_id,
            model: body.model.clone(),
            created: wall_ms()? / 1000,
            weight_version: Some(self.config.model_revision.clone()),
        };
        let mut spec = ChatResponseSpec::for_request(body, ProviderProfile::for_model(&body.model));
        spec.unbilled_prompt_tokens = processed.unbilled_prompt_tokens;
        Ok((
            ExecutionResult::Single {
                stream: ProtoStream::Managed(Box::new(response)),
                dispatch_start: started,
            },
            dispatch,
            spec,
        ))
    }
}

#[async_trait::async_trait]
impl RouterTrait for ManagedRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn router_type(&self) -> &'static str {
        "sglang_managed_epd"
    }
    fn is_pd_mode(&self) -> bool {
        true
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        tenant: &TenantRequestMeta,
        body: ChatCompletionRequest,
        _model_id: &str,
    ) -> Response {
        let reservation = RateLimitCell::new();
        let (execution, dispatch, spec) = match self
            .execute_chat(&body, tenant, Instant::now(), &reservation)
            .await
        {
            Ok(value) => value,
            Err(status) => {
                reservation.close_unsettled().await;
                let mut response = match status.code() {
                    tonic::Code::InvalidArgument | tonic::Code::Unimplemented => {
                        error::bad_request("managed_capability_refused", status.message())
                    }
                    tonic::Code::ResourceExhausted => {
                        error::too_many_requests("managed_capacity_exhausted", status.message())
                    }
                    tonic::Code::DeadlineExceeded => {
                        error::gateway_timeout("managed_deadline_exceeded", status.message())
                    }
                    tonic::Code::Internal | tonic::Code::DataLoss => {
                        error::internal_error("managed_internal_error", status.message())
                    }
                    _ => error::service_unavailable("managed_epd_unavailable", status.message()),
                };
                if let Some(value) = status
                    .metadata()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                {
                    response.headers_mut().insert("retry-after", value);
                }
                return response;
            }
        };
        if body.stream {
            let source = StreamSource::new(None, &dispatch, "/v1/chat/completions");
            self.streaming
                .clone()
                .process_streaming_response(
                    execution,
                    spec,
                    dispatch,
                    self.tokenizer.clone(),
                    body.skip_special_tokens,
                    reservation.take_for_streaming_handoff(),
                    None,
                    source,
                )
                .await
        } else {
            let mut decoder = create_stop_decoder(
                &self.tokenizer,
                body.stop.as_ref(),
                body.stop_token_ids.as_ref(),
                body.skip_special_tokens,
                body.no_stop_trim,
                body.ignore_eos,
            );
            match self
                .processor
                .process_non_streaming_chat_response(
                    execution,
                    spec,
                    dispatch,
                    self.tokenizer.clone(),
                    &mut decoder,
                    body.logprobs,
                )
                .await
            {
                Ok((response, counts)) => {
                    reservation.settle(counts.settlement()).await;
                    Json(response).into_response()
                }
                Err(response) => {
                    reservation.close_unsettled().await;
                    response
                }
            }
        }
    }
}

/// Extract occurrence order without decoding, fetching, deduplicating, or examining image bytes.
fn image_references(body: &ChatCompletionRequest) -> Result<Vec<String>, Status> {
    let messages = serde_json::to_value(&body.messages)
        .map_err(|e| Status::invalid_argument(e.to_string()))?;
    let mut images = Vec::new();
    for message in messages
        .as_array()
        .ok_or_else(|| Status::invalid_argument("invalid managed messages"))?
    {
        if let Some(parts) = message.get("content").and_then(Value::as_array) {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text" | "refusal") => {}
                    Some("image_url") => {
                        let image = &part["image_url"];
                        if image.get("max_long_side_pixel").is_some()
                            || image
                                .get("detail")
                                .and_then(Value::as_str)
                                .is_some_and(|detail| detail != "auto")
                        {
                            return Err(Status::invalid_argument(
                                "managed image resize/detail overrides are unsupported",
                            ));
                        }
                        images.push(
                            image["url"]
                                .as_str()
                                .filter(|url| !url.is_empty())
                                .ok_or_else(|| {
                                    Status::invalid_argument("managed image URL is missing")
                                })?
                                .to_owned(),
                        );
                    }
                    _ => {
                        return Err(Status::invalid_argument(
                            "managed EPD supports text and images only",
                        ))
                    }
                }
            }
        }
    }
    Ok(images)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderEndpoint {
    pub endpoint: String,
}

#[derive(Debug, Clone)]
pub struct PdEndpoints {
    pub prefill: String,
    pub decode: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub model: String,
    pub model_path: String,
    pub model_revision: String,
    pub processor_contract: String,
    #[serde(skip)]
    pub pairs: Vec<PdEndpoints>,
    pub prefill_urls: Vec<String>,
    pub decode_urls: Vec<String>,
    pub encoders: Vec<EncoderEndpoint>,
    pub max_attempts: usize,
    pub max_records: usize,
    pub timeout_ms: u64,
    pub max_clock_error_ms: u64,
    pub probe_timeout_ms: u64,
    pub snapshot_max_age_ms: u64,
    pub max_images: usize,
    pub max_pd_candidates: usize,
    pub max_encoder_candidates: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: String::new(),
            model_path: String::new(),
            model_revision: String::new(),
            processor_contract: "smg-qwen3-vl-f32-v1".into(),
            pairs: Vec::new(),
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            encoders: Vec::new(),
            max_attempts: 64,
            max_records: 1024,
            timeout_ms: 120_000,
            max_clock_error_ms: 1000,
            probe_timeout_ms: 3000,
            snapshot_max_age_ms: 3000,
            max_images: 32,
            max_pd_candidates: 4,
            max_encoder_candidates: 4,
        }
    }
}

impl Config {
    /// Resolve role pools once; candidate search remains bounded independently of pool size.
    pub fn resolve(mut self) -> Result<Self, Status> {
        if self.prefill_urls.is_empty()
            || self.decode_urls.is_empty()
            || self
                .prefill_urls
                .len()
                .saturating_mul(self.decode_urls.len())
                > 64
        {
            return Err(Status::invalid_argument("invalid managed P/D pools"));
        }
        self.pairs = self
            .prefill_urls
            .iter()
            .flat_map(|prefill| {
                self.decode_urls.iter().map(move |decode| PdEndpoints {
                    prefill: prefill.clone(),
                    decode: decode.clone(),
                })
            })
            .collect();
        self.validate()?;
        Ok(self)
    }

    /// Invalid bounds fail startup; an absent configuration keeps this entire mode disabled.
    pub fn validate(&self) -> Result<(), Status> {
        if self.model.is_empty()
            || self.model_path.is_empty()
            || self.model_revision.is_empty()
            || self.processor_contract.is_empty()
            || self.pairs.is_empty()
            || self.pairs.len() > 64
            || self.encoders.len() > 64
            || !(1..=64).contains(&self.max_pd_candidates)
            || !(1..=64).contains(&self.max_encoder_candidates)
            || self.max_attempts == 0
            || self.max_attempts > 1024
            || self.max_records < self.max_attempts.saturating_mul(3)
            || self.max_records > 65536
            || self.timeout_ms == 0
            || self.timeout_ms > 120_000
            || self.probe_timeout_ms == 0
            || self.max_clock_error_ms == 0
            || self.snapshot_max_age_ms == 0
            || self.max_images == 0
            || self.max_images > 32
        {
            return Err(Status::invalid_argument(
                "invalid managed gateway deployment bounds",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ordinary deployments specify pools and model identity, without scheduler tuning fields.
    #[test]
    fn minimal_pools_derive_bounded_defaults() {
        let config: Config = serde_json::from_value(serde_json::json!({
            "model": "qwen3", "model_path": "/models/qwen3", "model_revision": "weights",
            "prefill_urls": ["http://p1:1", "http://p2:1"],
            "decode_urls": ["http://d1:1", "http://d2:1"],
            "encoders": [{"endpoint": "http://e:1"}]
        }))
        .unwrap();
        let config = config.resolve().unwrap();
        assert_eq!(config.pairs.len(), 4);
        assert_eq!(config.max_pd_candidates, 4);
        assert_eq!(config.max_encoder_candidates, 4);
        assert_eq!(config.processor_contract, "smg-qwen3-vl-f32-v1");
        assert!(config.max_records >= config.max_attempts * 3);
    }
}
