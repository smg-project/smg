//! gRPC client for the TokenSpeed scheduler service.
//!
//! Wire types are TokenSpeed-native end-to-end (`tokenspeed_proto::*`).
//! Sampling-params builders are private `Self::build_*` methods that emit
//! `tokenspeed_proto::SamplingParams` directly. Unary RPC responses
//! (`get_model_info`, `get_server_info`, `get_loads`) also surface as
//! native types — the router consumes them through dedicated
//! `ModelInfo::TokenSpeed` / `ServerInfo::TokenSpeed` enum arms.

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, responses::ResponsesRequest,
    sampling_params::SamplingParams as GenerateSamplingParams,
};
use tonic::{transport::Channel, Request};
use tracing::{debug, warn};

use crate::{
    request_fields::{
        completion_constraint, plain_constraint, response_format_schema, stop_strings,
        ConstraintKind, PlainSampling,
    },
    BoxedTraceInjector,
};

#[expect(clippy::allow_attributes)]
pub mod tokenspeed_proto {
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    tonic::include_proto!("tokenspeed.grpc.scheduler");
    // The client, generated apart with the clients' codec (see build.rs).
    include!(concat!(
        env!("OUT_DIR"),
        "/client/tokenspeed.grpc.scheduler.rs"
    ));
}

/// Streaming `generate()` response that auto-aborts on drop. Concrete
/// alias for the generic `crate::AbortOnDropStream`.
pub type AbortOnDropStream =
    crate::AbortOnDropStream<tokenspeed_proto::GenerateResponse, TokenSpeedSchedulerClient>;

/// gRPC client for the TokenSpeed scheduler.
#[derive(Clone)]
pub struct TokenSpeedSchedulerClient {
    client: tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

crate::impl_abort_for_drop_with_reason!(TokenSpeedSchedulerClient);

impl TokenSpeedSchedulerClient {
    crate::impl_engine_client_basics!(
        tokenspeed_proto,
        tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient<Channel>,
        "TokenSpeed"
    );

    pub async fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> Result<(), tonic::Status> {
        debug!(
            "Sending TokenSpeed abort for {} (reason: {})",
            request_id, reason
        );
        let request = Request::new(tokenspeed_proto::AbortRequest {
            request_id: request_id.clone(),
            reason,
        });
        let mut client = self.client.clone();
        let response = client.abort(request).await?;
        debug!(
            "TokenSpeed abort response for {}: success={}, message={}",
            request_id,
            response.get_ref().success,
            response.get_ref().message
        );
        Ok(())
    }

    pub async fn get_loads(
        &self,
        include: Vec<String>,
    ) -> Result<tokenspeed_proto::GetLoadsResponse, tonic::Status> {
        let request = Request::new(tokenspeed_proto::GetLoadsRequest {
            dp_rank: None,
            include,
        });
        let mut client = self.client.clone();
        let response = client.get_loads(request).await?;
        Ok(response.into_inner())
    }

    crate::impl_get_tokenizer!();
    crate::impl_admin_ops!();
    crate::impl_subscribe_kv_events!();

    // ── Request builders ──────────────────────────────────────────────

    pub fn build_generate_request_from_chat(
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<tokenspeed_proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sampling_params = Self::build_sampling_params_from_chat(body, tool_call_constraint)?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: Some(-1),
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            stream: body.stream,
            mm_inputs: multimodal_inputs,
            ..Default::default()
        })
    }

    pub fn build_plain_generate_request(
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sampling_params =
            Self::build_sampling_params_from_plain(body.sampling_params.as_ref())?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.return_logprob.unwrap_or(false),
            logprob_start_len: Some(body.logprob_start_len.unwrap_or(-1)),
            top_logprobs_num: body.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: body.token_ids_logprob.clone().unwrap_or_default(),
            stream: body.stream,
            mm_inputs: None,
            // EPD bootstrap info injected later by the EPD pipeline, if applicable:
            // `encode_bootstrap_info` (E->P embedding) and
            // `kv_bootstrap_info` (P->D KV).
            encode_bootstrap_info: None,
            kv_bootstrap_info: None,
            // Attention-DP pin injected later via
            // ProtoGenerateRequest::set_data_parallel_rank, if applicable.
            data_parallel_rank: None,
        })
    }

    pub fn build_generate_request_from_responses(
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sampling_params = Self::build_sampling_params_from_responses(body, constraint)?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
        })
    }

    pub fn build_generate_request_from_messages(
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<tokenspeed_proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sampling_params =
            Self::build_sampling_params_from_messages(body, tool_call_constraint)?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            mm_inputs: multimodal_inputs,
            ..Default::default()
        })
    }

    pub fn build_generate_request_from_completion(
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sampling_params = Self::build_sampling_params_from_completion(body)?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs.is_some(),
            logprob_start_len: Some(-1),
            top_logprobs_num: body.logprobs.unwrap_or(0) as i32,
            stream: body.stream,
            ..Default::default()
        })
    }

    // ── Private sampling-params builders ─────────────────────────────
    //
    // TokenSpeed declares every sampling scalar as `optional`. Scalars are
    // wrapped in `Some(_)` so the wire presence bit is set;
    // `helpers::apply_tokenspeed_sampling_defaults` later overwrites them
    // with model-published defaults when a worker advertises them.

    fn build_sampling_params_from_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::SamplingParams, String> {
        let stop_sequences = stop_strings(request.stop.as_ref());

        // Keep skip_special_tokens true; flipping to false on tool calls regresses BFCL.
        Ok(tokenspeed_proto::SamplingParams {
            temperature: Some(request.temperature.unwrap_or(1.0)),
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_k: Some(request.top_k.unwrap_or(-1)),
            min_p: Some(request.min_p.unwrap_or(0.0)),
            frequency_penalty: Some(request.frequency_penalty.unwrap_or(0.0)),
            presence_penalty: Some(request.presence_penalty.unwrap_or(0.0)),
            repetition_penalty: Some(request.repetition_penalty.unwrap_or(1.0)),
            max_new_tokens: request.output_token_cap(),
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ignore_eos: request.ignore_eos,
            no_stop_trim: request.no_stop_trim,
            sampling_seed: Self::chat_sampling_seed(request)?,
            n: request.n.unwrap_or(1),
            constraint: Self::build_constraint_for_chat(request, tool_call_constraint)?,
            ..Default::default()
        })
    }

    /// Used by Harmony models only. Regular models use the Chat API path.
    /// Constraints come from the Harmony preparation stage (`structural_tag`)
    /// or tool handling.
    fn build_sampling_params_from_responses(
        request: &ResponsesRequest,
        constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::SamplingParams, String> {
        Ok(tokenspeed_proto::SamplingParams {
            temperature: Some(request.temperature.unwrap_or(1.0)),
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_k: Some(request.top_k),
            min_p: Some(request.min_p),
            frequency_penalty: Some(request.frequency_penalty.unwrap_or(0.0)),
            presence_penalty: Some(request.presence_penalty.unwrap_or(0.0)),
            repetition_penalty: Some(request.repetition_penalty),
            max_new_tokens: request.max_output_tokens,
            stop: vec![],               // Does not pass through request.stop yet
            stop_token_ids: vec![],     // Handled by Harmony stop tokens
            skip_special_tokens: false, // Keep special tokens for Harmony
            spaces_between_special_tokens: true,
            ignore_eos: false,
            no_stop_trim: false,
            n: 1, // Responses API doesn't support n>1
            constraint: Self::build_constraint_from_pair(constraint)?,
            ..Default::default()
        })
    }

    fn build_sampling_params_from_messages(
        request: &CreateMessageRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::SamplingParams, String> {
        let stop_sequences = request.stop_sequences.clone().unwrap_or_default();

        Ok(tokenspeed_proto::SamplingParams {
            temperature: Some(request.temperature.unwrap_or(1.0) as f32),
            top_p: Some(request.top_p.unwrap_or(1.0) as f32),
            top_k: Some(request.top_k.map(|v| v as i32).unwrap_or(-1)),
            min_p: Some(0.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.0),
            max_new_tokens: Some(request.max_tokens),
            stop: stop_sequences,
            stop_token_ids: vec![],
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ignore_eos: false,
            no_stop_trim: false,
            n: 1,
            constraint: Self::build_constraint_from_pair(tool_call_constraint)?,
            ..Default::default()
        })
    }

    fn build_sampling_params_from_completion(
        request: &CompletionRequest,
    ) -> Result<tokenspeed_proto::SamplingParams, String> {
        let stop_sequences = stop_strings(request.stop.as_ref());

        Ok(tokenspeed_proto::SamplingParams {
            temperature: Some(request.temperature.unwrap_or(1.0)),
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_k: Some(request.top_k.unwrap_or(-1)),
            min_p: Some(request.min_p.unwrap_or(0.0)),
            frequency_penalty: Some(request.frequency_penalty.unwrap_or(0.0)),
            presence_penalty: Some(request.presence_penalty.unwrap_or(0.0)),
            repetition_penalty: Some(request.repetition_penalty.unwrap_or(1.0)),
            max_new_tokens: request.max_tokens,
            min_new_tokens: request.min_tokens.unwrap_or(0),
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens: request.skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: request.ignore_eos,
            no_stop_trim: request.no_stop_trim,
            sampling_seed: Self::completion_sampling_seed(request)?,
            n: request.n.unwrap_or(1),
            constraint: Self::constraint_of(completion_constraint(request)?),
            ..Default::default()
        })
    }

    fn build_sampling_params_from_plain(
        params: Option<&GenerateSamplingParams>,
    ) -> Result<tokenspeed_proto::SamplingParams, String> {
        let plain = PlainSampling::read(params);
        Ok(tokenspeed_proto::SamplingParams {
            temperature: Some(plain.temperature),
            top_p: Some(plain.top_p),
            top_k: Some(plain.top_k.unwrap_or(-1)),
            min_p: plain.min_p,
            frequency_penalty: plain.frequency_penalty,
            presence_penalty: plain.presence_penalty,
            repetition_penalty: Some(plain.repetition_penalty),
            max_new_tokens: plain.max_new_tokens,
            min_new_tokens: plain.min_new_tokens.unwrap_or(0),
            stop: plain.stop,
            stop_token_ids: plain.stop_token_ids,
            skip_special_tokens: plain.skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: plain.ignore_eos,
            no_stop_trim: plain.no_stop_trim,
            sampling_seed: plain.sampling_seed,
            n: plain.n,
            constraint: match params {
                Some(p) => Self::constraint_of(plain_constraint(p)?),
                None => None,
            },
            ..Default::default()
        })
    }

    // ── Constraint helpers ───────────────────────────────────────────

    fn build_constraint_for_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<tokenspeed_proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();

        if let Some(schema) = response_format_schema(request)? {
            constraints.push(tokenspeed_proto::sampling_params::Constraint::JsonSchema(
                schema,
            ));
        }

        if let Some(ebnf) = &request.ebnf {
            constraints.push(tokenspeed_proto::sampling_params::Constraint::EbnfGrammar(
                ebnf.clone(),
            ));
        }
        if let Some(regex) = &request.regex {
            constraints.push(tokenspeed_proto::sampling_params::Constraint::Regex(
                regex.clone(),
            ));
        }

        // response_format wins over tool_call_constraint when both are set.
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if constraints.is_empty() {
                let tool_constraint =
                    Self::constraint_from_pair(constraint_type, constraint_value)?;
                constraints.push(tool_constraint);
            } else {
                warn!(
                    "Constrained decoding is not compatible with tool calls, dropping tool constraint"
                );
            }
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple constraints are not allowed.".to_string()),
        }
    }

    fn build_constraint_from_pair(
        constraint: Option<(String, String)>,
    ) -> Result<Option<tokenspeed_proto::sampling_params::Constraint>, String> {
        if let Some((constraint_type, constraint_value)) = constraint {
            Ok(Some(Self::constraint_from_pair(
                constraint_type,
                constraint_value,
            )?))
        } else {
            Ok(None)
        }
    }

    fn constraint_from_pair(
        constraint_type: String,
        constraint_value: String,
    ) -> Result<tokenspeed_proto::sampling_params::Constraint, String> {
        match constraint_type.as_str() {
            "structural_tag" => {
                Ok(tokenspeed_proto::sampling_params::Constraint::StructuralTag(constraint_value))
            }
            "json_schema" => Ok(tokenspeed_proto::sampling_params::Constraint::JsonSchema(
                constraint_value,
            )),
            "ebnf" => Ok(tokenspeed_proto::sampling_params::Constraint::EbnfGrammar(
                constraint_value,
            )),
            "regex" => Ok(tokenspeed_proto::sampling_params::Constraint::Regex(
                constraint_value,
            )),
            _ => Err(format!("Unknown constraint type: {constraint_type}")),
        }
    }

    /// The engine's variant for the structured constraint a request names.
    fn constraint_of(
        constraint: Option<(ConstraintKind, String)>,
    ) -> Option<tokenspeed_proto::sampling_params::Constraint> {
        constraint.map(|(kind, value)| match kind {
            ConstraintKind::JsonSchema => {
                tokenspeed_proto::sampling_params::Constraint::JsonSchema(value)
            }
            ConstraintKind::Regex => tokenspeed_proto::sampling_params::Constraint::Regex(value),
            ConstraintKind::Ebnf => {
                tokenspeed_proto::sampling_params::Constraint::EbnfGrammar(value)
            }
        })
    }

    fn unsigned_sampling_seed(seed: Option<i64>) -> Result<Option<u64>, String> {
        seed.map(|seed| {
            u64::try_from(seed).map_err(|_| "sampling seed must be an unsigned integer".to_owned())
        })
        .transpose()
    }

    /// Resolve the TokenSpeed sampling seed: the native `sampling_seed` field
    /// wins when set; the OpenAI-compatible `seed` is only the fallback.
    fn resolve_sampling_seed(
        sampling_seed: Option<u64>,
        legacy_seed: Option<i64>,
    ) -> Result<Option<u64>, String> {
        match sampling_seed {
            Some(seed) => Ok(Some(seed)),
            None => Self::unsigned_sampling_seed(legacy_seed),
        }
    }

    #[expect(
        deprecated,
        reason = "TokenSpeed preserves the OpenAI-compatible chat seed on its sampling wire"
    )]
    fn chat_sampling_seed(request: &ChatCompletionRequest) -> Result<Option<u64>, String> {
        Self::resolve_sampling_seed(request.sampling_seed, request.seed)
    }

    fn completion_sampling_seed(request: &CompletionRequest) -> Result<Option<u64>, String> {
        Self::resolve_sampling_seed(request.sampling_seed, request.seed)
    }
}

// ---------------------------------------------------------------------------
// Proto → protocol type conversions
// ---------------------------------------------------------------------------

impl From<tokenspeed_proto::SchedulerLoad> for openai_protocol::worker::SchedulerLoadSnapshot {
    fn from(load: tokenspeed_proto::SchedulerLoad) -> Self {
        let memory =
            load.memory.map(
                |memory| openai_protocol::worker::EngineMemoryMetricsSnapshot {
                    weight_gb: memory.weight_gb,
                    kv_cache_gb: memory.kv_cache_gb,
                    graph_gb: memory.graph_gb,
                    token_capacity: memory.token_capacity,
                },
            );
        let queues =
            load.queues.map(
                |queues| openai_protocol::worker::EngineQueueMetricsSnapshot {
                    waiting: queues.waiting,
                    grammar: queues.grammar,
                    paused: queues.paused,
                    retracted: queues.retracted,
                },
            );
        Self {
            dp_rank: load.dp_rank,
            num_running_reqs: load.num_running_reqs,
            num_waiting_reqs: load.num_waiting_reqs,
            num_waiting_uncached_tokens: load.num_waiting_uncached_tokens,
            num_total_reqs: load.num_total_reqs,
            num_used_tokens: load.num_used_tokens,
            max_total_num_tokens: load.max_total_num_tokens,
            token_usage: load.token_usage,
            active_token_usage: load.active_token_usage,
            gen_throughput: load.gen_throughput,
            cache_hit_rate: load.cache_hit_rate,
            utilization: load.utilization,
            max_running_requests: load.max_running_requests,
            memory,
            queues,
            // TokenSpeed has no disagg section; canonical PD fields stay None.
            ..Default::default()
        }
    }
}

impl From<tokenspeed_proto::GetLoadsResponse> for openai_protocol::worker::WorkerLoadResponse {
    fn from(resp: tokenspeed_proto::GetLoadsResponse) -> Self {
        let aggregate = resp.aggregate.map(|aggregate| {
            openai_protocol::worker::EngineAggregateMetricsSnapshot {
                total_running_reqs: aggregate.total_running_reqs,
                total_waiting_reqs: aggregate.total_waiting_reqs,
                total_reqs: aggregate.total_reqs,
                avg_token_usage: aggregate.avg_token_usage,
                avg_throughput: aggregate.avg_throughput,
                avg_utilization: aggregate.avg_utilization,
            }
        });
        Self {
            timestamp: resp.timestamp,
            version: resp.version,
            dp_rank_count: resp.dp_rank_count,
            loads: resp.loads.into_iter().map(Into::into).collect(),
            aggregate,
            sampled_at: None,
        }
    }
}

#[cfg(test)]
mod connect_tests {
    use super::*;
    use crate::channel::testing::refused_endpoint;

    #[tokio::test]
    async fn connect_fails_on_a_malformed_or_refused_endpoint() {
        let malformed = TokenSpeedSchedulerClient::connect("not an endpoint").await;
        assert!(
            malformed.is_err(),
            "a malformed endpoint fails before the dial"
        );
        let refused = refused_endpoint();
        let result = TokenSpeedSchedulerClient::connect(&refused.url).await;
        assert!(result.is_err(), "a refused dial fails the connect");
    }
}

#[cfg(test)]
mod load_conversion_tests {
    use super::*;

    #[test]
    fn conversion_preserves_version_sections_and_aggregate() {
        let response = tokenspeed_proto::GetLoadsResponse {
            timestamp: "2026-08-09T12:34:56Z".to_owned(),
            version: "dsv4-engine/0.1.0".to_owned(),
            dp_rank_count: 1,
            loads: vec![tokenspeed_proto::SchedulerLoad {
                dp_rank: 0,
                num_running_reqs: 3,
                num_waiting_reqs: 2,
                num_total_reqs: 5,
                num_used_tokens: 98_304,
                max_total_num_tokens: 196_608,
                max_running_requests: 32,
                num_waiting_uncached_tokens: 1_280,
                token_usage: 0.5,
                active_token_usage: Some(0.25),
                gen_throughput: 105.25,
                cache_hit_rate: 0.75,
                utilization: 0.5,
                memory: Some(tokenspeed_proto::MemoryMetrics {
                    weight_gb: 44.0,
                    kv_cache_gb: 12.5,
                    graph_gb: 0.75,
                    token_capacity: 196_608,
                }),
                queues: Some(tokenspeed_proto::QueueMetrics {
                    waiting: 2,
                    grammar: 0,
                    paused: 0,
                    retracted: 0,
                }),
            }],
            aggregate: Some(tokenspeed_proto::AggregateMetrics {
                total_running_reqs: 3,
                total_waiting_reqs: 2,
                total_reqs: 5,
                avg_token_usage: 0.5,
                avg_throughput: 105.25,
                avg_utilization: 0.5,
            }),
        };

        let converted = openai_protocol::worker::WorkerLoadResponse::from(response);
        assert_eq!(converted.timestamp, "2026-08-09T12:34:56Z");
        assert_eq!(converted.version, "dsv4-engine/0.1.0");
        let load = &converted.loads[0];
        assert_eq!(load.active_token_usage, Some(0.25));
        assert_eq!(converted.effective_kv_pressure(), 0.25);
        assert_eq!(converted.effective_token_usage(), 0.5);
        assert!(matches!(
            load.memory,
            Some(ref memory)
                if memory.weight_gb == 44.0
                    && memory.kv_cache_gb == 12.5
                    && memory.graph_gb == 0.75
                    && memory.token_capacity == 196_608
        ));
        assert!(matches!(
            load.queues,
            Some(ref queues)
                if queues.waiting == 2
                    && queues.grammar == 0
                    && queues.paused == 0
                    && queues.retracted == 0
        ));
        assert!(matches!(
            converted.aggregate,
            Some(ref aggregate)
                if aggregate.total_running_reqs == 3
                    && aggregate.total_waiting_reqs == 2
                    && aggregate.total_reqs == 5
                    && aggregate.avg_token_usage == 0.5
                    && aggregate.avg_throughput == 105.25
                    && aggregate.avg_utilization == 0.5
        ));
    }
}

#[cfg(test)]
mod seed_resolution_tests {
    use super::*;

    #[test]
    fn plain_sampling_seed_is_forwarded() {
        let params = GenerateSamplingParams {
            sampling_seed: Some(7),
            ..Default::default()
        };
        let sampling =
            TokenSpeedSchedulerClient::build_sampling_params_from_plain(Some(&params)).unwrap();
        assert_eq!(sampling.sampling_seed, Some(7));
    }

    #[test]
    fn native_sampling_seed_wins_over_legacy_seed() {
        let resolved = TokenSpeedSchedulerClient::resolve_sampling_seed(Some(7), Some(42))
            .expect("native seed should resolve");
        assert_eq!(resolved, Some(7));
    }

    #[test]
    fn legacy_seed_is_the_fallback() {
        let resolved = TokenSpeedSchedulerClient::resolve_sampling_seed(None, Some(42))
            .expect("legacy seed should resolve");
        assert_eq!(resolved, Some(42));
    }

    #[test]
    fn negative_legacy_seed_is_rejected() {
        let err = TokenSpeedSchedulerClient::resolve_sampling_seed(None, Some(-1))
            .expect_err("negative legacy seed must be rejected");
        assert!(err.contains("unsigned integer"), "{err}");
    }

    #[test]
    fn no_seed_stays_none() {
        let resolved = TokenSpeedSchedulerClient::resolve_sampling_seed(None, None)
            .expect("absent seeds should resolve");
        assert_eq!(resolved, None);
    }
}

#[cfg(test)]
mod constraint_mapping_tests {
    use super::*;
    use crate::request_fields::ConstraintKind;

    /// Each constraint kind reaches the engine as its own variant; none stays none.
    #[test]
    fn constraint_of_maps_each_kind_to_the_engines_variant() {
        for (kind, expected) in [
            (
                ConstraintKind::JsonSchema,
                tokenspeed_proto::sampling_params::Constraint::JsonSchema("v".to_string()),
            ),
            (
                ConstraintKind::Regex,
                tokenspeed_proto::sampling_params::Constraint::Regex("v".to_string()),
            ),
            (
                ConstraintKind::Ebnf,
                tokenspeed_proto::sampling_params::Constraint::EbnfGrammar("v".to_string()),
            ),
        ] {
            assert_eq!(
                TokenSpeedSchedulerClient::constraint_of(Some((kind, "v".to_string()))),
                Some(expected)
            );
        }
        assert_eq!(TokenSpeedSchedulerClient::constraint_of(None), None);
    }
}
