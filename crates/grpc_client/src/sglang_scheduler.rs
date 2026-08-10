use std::{future::Future, pin::Pin};

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, StringOrArray},
    completion::CompletionRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
    responses::ResponsesRequest,
    sampling_params::SamplingParams as GenerateSamplingParams,
};
use tonic::{transport::Channel, Request};
use tracing::{debug, warn};

use crate::{AbortOnDropClient, BoxedTraceInjector};

// Include the generated protobuf code
#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("sglang.grpc.scheduler");
}

/// Streaming `generate()` response that auto-aborts on drop. Concrete
/// alias for the generic `crate::AbortOnDropStream`.
pub type AbortOnDropStream =
    crate::AbortOnDropStream<proto::GenerateResponse, SglangSchedulerClient>;

/// gRPC client for SGLang scheduler
#[derive(Clone)]
pub struct SglangSchedulerClient {
    client: proto::sglang_scheduler_client::SglangSchedulerClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

/// Optional request data used when building SGLang generation requests.
#[derive(Default)]
pub struct SglangGenerateRequestOptions {
    pub multimodal_inputs: Option<proto::MultimodalInputs>,
    pub tool_call_constraint: Option<(String, String)>,
    pub require_reasoning: bool,
}

impl AbortOnDropClient for SglangSchedulerClient {
    fn abort_for_drop(
        self,
        request_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>> {
        Box::pin(async move {
            self.abort_request(request_id, "Stream dropped".to_string())
                .await
        })
    }
}

impl SglangSchedulerClient {
    crate::impl_engine_client_basics!(
        proto::sglang_scheduler_client::SglangSchedulerClient<Channel>,
        "SGLang scheduler"
    );

    /// Submit a generation request (returns auto-aborting streaming response)
    ///
    /// The returned stream automatically sends an abort request when dropped,
    /// ensuring proper cleanup even if the HTTP client disconnects or an error occurs.
    /// Call `mark_completed()` on the stream after successful completion to prevent
    /// unnecessary abort RPCs.
    pub async fn generate(
        &self,
        req: proto::GenerateRequest,
    ) -> Result<AbortOnDropStream, tonic::Status> {
        let request_id = req.request_id.clone();
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        // Inject W3C trace context into gRPC metadata for distributed tracing
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.generate(request).await?;

        Ok(AbortOnDropStream::new(
            response.into_inner(),
            request_id,
            self.clone(),
        ))
    }

    /// Submit an embedding request
    pub async fn embed(
        &self,
        req: proto::EmbedRequest,
    ) -> Result<proto::EmbedResponse, tonic::Status> {
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        // Inject W3C trace context into gRPC metadata
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.embed(request).await?;
        Ok(response.into_inner())
    }

    /// Abort a request
    pub async fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> Result<(), tonic::Status> {
        debug!(
            "Sending abort request for {} (reason: {})",
            request_id, reason
        );
        let request = Request::new(proto::AbortRequest {
            request_id: request_id.clone(),
            reason,
        });

        let mut client = self.client.clone();
        let response = client.abort(request).await?;
        debug!(
            "Abort response for {}: success={}, message={}",
            request_id,
            response.get_ref().success,
            response.get_ref().message
        );
        Ok(())
    }

    /// Get load metrics from the scheduler
    pub async fn get_loads(
        &self,
        include: Vec<String>,
    ) -> Result<proto::GetLoadsResponse, tonic::Status> {
        debug!("Requesting load metrics");
        let request = Request::new(proto::GetLoadsRequest {
            dp_rank: None,
            include,
        });

        let mut client = self.client.clone();
        let response = client.get_loads(request).await?;
        debug!("Load metrics response received");
        Ok(response.into_inner())
    }

    crate::impl_get_tokenizer!();
    crate::impl_subscribe_kv_events!();
    crate::impl_admin_ops!();

    /// Build a single SGLang EmbedRequest
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API"
    )]
    pub fn build_embed_request(
        &self,
        request_id: String,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> proto::EmbedRequest {
        proto::EmbedRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            ..Default::default()
        }
    }

    /// Build a single SGLang GenerateRequest from OpenAI ChatCompletionRequest
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API"
    )]
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: SglangGenerateRequestOptions,
    ) -> Result<proto::GenerateRequest, String> {
        Self::build_generate_request_from_chat_parts(
            request_id,
            body,
            processed_text,
            token_ids,
            options,
        )
    }

    fn build_generate_request_from_chat_parts(
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: SglangGenerateRequestOptions,
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling params
        let sampling_params =
            Self::build_grpc_sampling_params_from_chat(body, options.tool_call_constraint)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: options.multimodal_inputs,
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: -1,
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            require_reasoning: options.require_reasoning,
            ..Default::default()
        };

        Ok(grpc_request)
    }

    /// Build a basic GenerateRequest from the SGLang spec GenerateRequest
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API"
    )]
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        Self::build_plain_generate_request_parts(request_id, body, original_text, token_ids)
    }

    fn build_plain_generate_request_parts(
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            Self::build_sampling_params_from_plain(body.sampling_params.as_ref())?;
        let require_reasoning = body
            .other
            .get("require_reasoning")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.return_logprob.unwrap_or(false),
            logprob_start_len: body.logprob_start_len.unwrap_or(-1),
            top_logprobs_num: body.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: body.token_ids_logprob.clone().unwrap_or_default(),
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            log_metrics: body.log_metrics,
            require_reasoning,
            ..Default::default()
        };

        Ok(grpc_request)
    }

    /// Build a GenerateRequest from ResponsesRequest (OpenAI Responses API)
    ///
    /// NOTE: This is used by the Harmony router only. The Regular router uses
    /// responses_to_chat() conversion and goes through the chat pipeline.
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API"
    )]
    pub fn build_generate_request_from_responses(
        &self,
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling params from ResponsesRequest
        let sampling_params = Self::build_grpc_sampling_params_from_responses(body, constraint)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: None, // Responses API doesn't support multimodal yet
            sampling_params: Some(sampling_params),
            // TODO: Logprobs for Responses API is not fully supported yet
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            return_hidden_states: false,
            stream: body.stream.unwrap_or(false),
            require_reasoning: false,
            ..Default::default()
        };

        Ok(grpc_request)
    }

    /// Build gRPC SamplingParams from ChatCompletionRequest
    fn build_grpc_sampling_params_from_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = Self::extract_stop_strings(request);

        let max_new_tokens = request.max_completion_tokens;

        // Hardcode to true: gRPC backends return raw token IDs, not decoded text.
        // Detokenization happens on the SMG Rust side (StopDecoder/Sequence).
        let skip_special_tokens = true;

        Ok(proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(-1),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_new_tokens,
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens,
            spaces_between_special_tokens: true, // Default from Python SamplingParams
            ignore_eos: request.ignore_eos,
            no_stop_trim: request.no_stop_trim,
            n: request.n.unwrap_or(1),
            constraint: Self::build_constraint_for_chat(request, tool_call_constraint)?,
            ..Default::default()
        })
    }

    /// Extract stop strings from request
    fn extract_stop_strings(request: &ChatCompletionRequest) -> Vec<String> {
        match &request.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        }
    }

    /// Build constraint for structured generation
    fn build_constraint_for_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();

        // Handle response_format constraints
        match &request.response_format {
            Some(ResponseFormat::JsonObject) => {
                // json_object mode - constrain to valid JSON object
                let schema = serde_json::json!({"type": "object"});
                let schema_str = serde_json::to_string(&schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema_str = serde_json::to_string(&json_schema.schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::Text) | None => {
                // No constraint for text format
            }
        }

        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
                ebnf.clone(),
            ));
        }

        if let Some(regex) = &request.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }

        // Handle tool call constraint from preparation stage.
        // If response_format already set a constraint, drop the tool constraint
        // (matches SGLang HTTP behavior where response_format takes priority).
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if constraints.is_empty() {
                let tool_constraint = match constraint_type.as_str() {
                    "structural_tag" => {
                        proto::sampling_params::Constraint::StructuralTag(constraint_value)
                    }
                    "json_schema" => {
                        proto::sampling_params::Constraint::JsonSchema(constraint_value)
                    }
                    "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
                    "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                    _ => return Err(format!("Unknown constraint type: {constraint_type}")),
                };
                constraints.push(tool_constraint);
            } else {
                warn!("Constrained decoding is not compatible with tool calls, dropping tool constraint");
            }
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple constraints are not allowed.".to_string()),
        }
    }

    /// Build gRPC SamplingParams from ResponsesRequest
    fn build_grpc_sampling_params_from_responses(
        request: &ResponsesRequest,
        constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        // Used by Harmony models only. Regular models use Chat API path.
        // Constraints come from Harmony preparation stage (structural_tag) or tool handling.

        let max_new_tokens = request.max_output_tokens;

        Ok(proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty,
            max_new_tokens,
            stop: vec![], // Does not pass through request.stop yet (follow-up fix)
            stop_token_ids: vec![], // Handled by Harmony stop tokens
            skip_special_tokens: false, // Keep special tokens for Harmony
            spaces_between_special_tokens: true,
            ignore_eos: false,
            no_stop_trim: false,
            n: 1, // Responses API doesn't support n>1
            constraint: Self::build_constraint_for_responses(constraint)?,
            ..Default::default()
        })
    }

    /// Build constraint for Responses API
    ///
    /// Handles constraints from Harmony preparation stage (structural_tag for Harmony models,
    /// structured output via text field, or tool call constraints).
    ///
    /// Note: Regular gRPC models use Chat API path with response_format, not this function.
    fn build_constraint_for_responses(
        constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        if let Some((constraint_type, constraint_value)) = constraint {
            let parsed_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    // Harmony models: structural tag from preparation stage
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {constraint_type}")),
            };
            Ok(Some(parsed_constraint))
        } else {
            Ok(None)
        }
    }

    /// Build a GenerateRequest from CreateMessageRequest (Anthropic Messages API)
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API"
    )]
    pub fn build_generate_request_from_messages(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: SglangGenerateRequestOptions,
    ) -> Result<proto::GenerateRequest, String> {
        Self::build_generate_request_from_messages_parts(
            request_id,
            body,
            processed_text,
            token_ids,
            options,
        )
    }

    fn build_generate_request_from_messages_parts(
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: SglangGenerateRequestOptions,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            Self::build_grpc_sampling_params_from_messages(body, options.tool_call_constraint)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: options.multimodal_inputs,
            sampling_params: Some(sampling_params),
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            return_hidden_states: false,
            stream: body.stream.unwrap_or(false),
            require_reasoning: options.require_reasoning,
            ..Default::default()
        };

        Ok(grpc_request)
    }

    /// Build gRPC SamplingParams from CreateMessageRequest
    fn build_grpc_sampling_params_from_messages(
        request: &CreateMessageRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = request.stop_sequences.clone().unwrap_or_default();

        // Hardcode to true: gRPC backends return raw token IDs, not decoded text.
        let skip_special_tokens = true;

        Ok(proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0) as f32,
            top_p: request.top_p.unwrap_or(1.0) as f32,
            top_k: request.top_k.map(|v| v as i32).unwrap_or(-1),
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_new_tokens: Some(request.max_tokens),
            stop: stop_sequences,
            stop_token_ids: vec![],
            skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: false,
            no_stop_trim: false,
            n: 1,
            constraint: Self::build_constraint_for_responses(tool_call_constraint)?,
            ..Default::default()
        })
    }

    /// Build a GenerateRequest from CompletionRequest (`/v1/completions`)
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API"
    )]
    pub fn build_generate_request_from_completion(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = Self::build_grpc_sampling_params_from_completion(body)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text,
                input_ids: token_ids,
            }),
            mm_inputs: None,
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs.is_some(),
            logprob_start_len: -1,
            top_logprobs_num: body.logprobs.unwrap_or(0) as i32,
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            require_reasoning: false,
            ..Default::default()
        };

        Ok(grpc_request)
    }

    fn build_grpc_sampling_params_from_completion(
        request: &CompletionRequest,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = match &request.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        };

        let constraint = Self::build_single_constraint_from_completion(request)?;

        Ok(proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(-1),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_new_tokens: request.max_tokens,
            min_new_tokens: request.min_tokens.unwrap_or(0),
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens: request.skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: request.ignore_eos,
            no_stop_trim: request.no_stop_trim,
            n: request.n.unwrap_or(1),
            constraint,
            ..Default::default()
        })
    }

    fn build_single_constraint_from_completion(
        request: &CompletionRequest,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();
        if let Some(json_schema) = &request.json_schema {
            constraints.push(proto::sampling_params::Constraint::JsonSchema(
                json_schema.clone(),
            ));
        }
        if let Some(regex) = &request.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }
        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
                ebnf.clone(),
            ));
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple structured constraints are not allowed".to_string()),
        }
    }

    fn build_single_constraint_from_plain(
        params: &GenerateSamplingParams,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();
        if let Some(json_schema) = &params.json_schema {
            constraints.push(proto::sampling_params::Constraint::JsonSchema(
                json_schema.clone(),
            ));
        }
        if let Some(regex) = &params.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }
        if let Some(ebnf) = &params.ebnf {
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
                ebnf.clone(),
            ));
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple structured constraints are not allowed".to_string()),
        }
    }

    fn build_sampling_params_from_plain(
        params: Option<&GenerateSamplingParams>,
    ) -> Result<proto::SamplingParams, String> {
        let mut sampling = proto::SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            repetition_penalty: 1.0,
            n: 1,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ..Default::default()
        };

        let Some(p) = params else {
            return Ok(sampling);
        };

        // Simple field mappings using a macro
        macro_rules! map_field {
            ($field:ident) => {
                if let Some(val) = p.$field {
                    sampling.$field = val;
                }
            };
        }

        map_field!(temperature);
        map_field!(top_p);
        map_field!(top_k);
        map_field!(frequency_penalty);
        map_field!(presence_penalty);
        map_field!(repetition_penalty);
        map_field!(min_p);
        map_field!(ignore_eos);
        map_field!(skip_special_tokens);
        map_field!(no_stop_trim);

        // Handle stop sequences
        if let Some(stop) = &p.stop {
            match stop {
                StringOrArray::String(s) => sampling.stop.push(s.clone()),
                StringOrArray::Array(arr) => sampling.stop.extend(arr.clone()),
            }
        }

        // Handle stop token IDs
        if let Some(stop_token_ids) = &p.stop_token_ids {
            sampling.stop_token_ids.clone_from(stop_token_ids);
        }

        // Handle max_new_tokens
        sampling.max_new_tokens = p.max_new_tokens;

        // Handle min_new_tokens
        if let Some(min_new_tokens) = p.min_new_tokens {
            sampling.min_new_tokens = min_new_tokens;
        }

        // Handle n
        if let Some(n) = p.n {
            sampling.n = n;
        }

        // Handle constraints (exactly one allowed)
        sampling.constraint = Self::build_single_constraint_from_plain(p)?;

        Ok(sampling)
    }
}

// ---------------------------------------------------------------------------
// Proto → protocol type conversions
// ---------------------------------------------------------------------------

impl From<proto::SchedulerLoad> for openai_protocol::worker::SchedulerLoadSnapshot {
    fn from(load: proto::SchedulerLoad) -> Self {
        // Disaggregation queue depths roll the per-stage SGLang counters into
        // the two canonical totals: prefill = prealloc + inflight, decode =
        // prealloc + transfer + retracted.
        let disagg = load.disaggregation;
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
            gen_throughput: load.gen_throughput,
            cache_hit_rate: load.cache_hit_rate,
            utilization: load.utilization,
            max_running_requests: load.max_running_requests,
            memory,
            queues,
            kv_transfer_latency_ms: disagg.as_ref().map(|d| d.kv_transfer_latency_ms),
            kv_transfer_speed_gb_s: disagg.as_ref().map(|d| d.kv_transfer_speed_gb_s),
            prefill_queue_reqs: disagg.as_ref().map(|d| {
                d.prefill_prealloc_queue_reqs
                    .saturating_add(d.prefill_inflight_queue_reqs)
            }),
            decode_queue_reqs: disagg.as_ref().map(|d| {
                d.decode_prealloc_queue_reqs
                    .saturating_add(d.decode_transfer_queue_reqs)
                    .saturating_add(d.decode_retracted_queue_reqs)
            }),
            disagg_mode: disagg.map(|d| d.mode),
        }
    }
}

impl From<proto::GetLoadsResponse> for openai_protocol::worker::WorkerLoadResponse {
    fn from(resp: proto::GetLoadsResponse) -> Self {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        let _health_req = proto::HealthCheckRequest {};
        // HealthCheckRequest is now empty - no fields to test
    }

    #[test]
    fn test_scheduler_load_maps_disagg_section() {
        let load = proto::SchedulerLoad {
            dp_rank: 0,
            num_running_reqs: 3,
            disaggregation: Some(proto::DisaggregationMetrics {
                mode: "prefill".to_string(),
                prefill_prealloc_queue_reqs: 2,
                prefill_inflight_queue_reqs: 5,
                decode_prealloc_queue_reqs: 1,
                decode_transfer_queue_reqs: 4,
                decode_retracted_queue_reqs: 7,
                kv_transfer_speed_gb_s: 12.5,
                kv_transfer_latency_ms: 3.25,
            }),
            ..Default::default()
        };

        let snap = openai_protocol::worker::SchedulerLoadSnapshot::from(load);

        assert_eq!(snap.disagg_mode.as_deref(), Some("prefill"));
        assert_eq!(snap.kv_transfer_latency_ms, Some(3.25));
        assert_eq!(snap.kv_transfer_speed_gb_s, Some(12.5));
        assert_eq!(snap.prefill_queue_reqs, Some(7)); // 2 + 5
        assert_eq!(snap.decode_queue_reqs, Some(12)); // 1 + 4 + 7
    }

    #[test]
    fn test_scheduler_load_disagg_absent_is_none() {
        let snap = openai_protocol::worker::SchedulerLoadSnapshot::from(proto::SchedulerLoad {
            num_running_reqs: 1,
            ..Default::default()
        });

        assert!(snap.disagg_mode.is_none());
        assert!(snap.kv_transfer_latency_ms.is_none());
        assert!(snap.prefill_queue_reqs.is_none());
        assert!(snap.decode_queue_reqs.is_none());
    }

    #[test]
    fn test_generate_request_construction() {
        let sampling_params = proto::SamplingParams {
            temperature: 0.7,
            max_new_tokens: Some(128),
            top_p: 0.9,
            top_k: 50,
            stop: vec!["</s>".to_string()],
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "test-req-123".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "Hello world".to_string(),
                input_ids: vec![9906, 1917], // Mock token IDs for "Hello world"
            }),
            sampling_params: Some(sampling_params),
            return_logprob: true,
            logprob_start_len: 0,
            top_logprobs_num: 5,
            ..Default::default()
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(ref tokenized) = &gen_req.tokenized {
            assert_eq!(tokenized.original_text, "Hello world");
        }
        assert!(gen_req.return_logprob);
        assert_eq!(gen_req.top_logprobs_num, 5);

        let params = gen_req.sampling_params.unwrap();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.max_new_tokens, Some(128));
        assert_eq!(params.stop, vec!["</s>"]);
    }

    #[test]
    fn test_health_check_request() {
        let _health_req = proto::HealthCheckRequest {};
        // HealthCheckRequest is now empty - server generates its own test internally
    }

    #[test]
    fn test_abort_request_construction() {
        let abort_req = proto::AbortRequest {
            request_id: "req-456".to_string(),
            reason: "User canceled".to_string(),
        };
        assert_eq!(abort_req.request_id, "req-456");
        assert_eq!(abort_req.reason, "User canceled");
    }

    #[test]
    fn test_sampling_params_defaults() {
        let params = proto::SamplingParams::default();
        // Numeric fields have proto defaults (0)
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 0.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repetition_penalty, 0.0);
        assert_eq!(params.n, 0);
        // Bool fields have proto defaults (false)
        assert!(!params.skip_special_tokens);
        assert!(!params.spaces_between_special_tokens);
        assert!(!params.ignore_eos);
        assert!(!params.no_stop_trim);
        // Optional int fields should be None
        assert_eq!(params.max_new_tokens, None);
        assert_eq!(params.stream_interval, None);
        // Other non-optional fields
        assert_eq!(params.min_p, 0.0);
        assert_eq!(params.frequency_penalty, 0.0);
        assert_eq!(params.presence_penalty, 0.0);
        assert!(params.stop.is_empty());
    }

    #[test]
    fn test_multimodal_inputs() {
        let mm_inputs = proto::MultimodalInputs {
            image_urls: vec!["http://example.com/image.jpg".to_string()],
            video_urls: vec![],
            audio_urls: vec![],
            image_data: vec![],
            video_data: vec![],
            audio_data: vec![],
            modalities: vec!["image".to_string()],
            ..Default::default()
        };

        assert_eq!(mm_inputs.image_urls.len(), 1);
        assert_eq!(mm_inputs.image_urls[0], "http://example.com/image.jpg");
        assert_eq!(mm_inputs.modalities[0], "image");
    }

    // TODO: SessionParams not in current proto - skip test

    #[test]
    fn test_responses_sampling_params_are_passed_through() {
        use openai_protocol::responses::ResponsesRequest;

        let request = ResponsesRequest {
            top_k: 40,
            min_p: 0.05,
            repetition_penalty: 1.2,
            frequency_penalty: Some(0.3),
            presence_penalty: Some(-0.4),
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_output_tokens: Some(128),
            ..Default::default()
        };

        let params =
            SglangSchedulerClient::build_grpc_sampling_params_from_responses(&request, None)
                .expect("build sampling params");

        assert_eq!(params.top_k, 40);
        assert!((params.min_p - 0.05).abs() < 1e-6);
        assert!((params.repetition_penalty - 1.2).abs() < 1e-6);
        assert!((params.frequency_penalty - 0.3).abs() < 1e-6);
        assert!((params.presence_penalty - (-0.4)).abs() < 1e-6);

        // Default top_k (-1) passes through as SGLang's disabled sentinel.
        let disabled = ResponsesRequest {
            top_k: -1,
            ..Default::default()
        };
        let disabled_params =
            SglangSchedulerClient::build_grpc_sampling_params_from_responses(&disabled, None)
                .expect("build sampling params");
        assert_eq!(disabled_params.top_k, -1);
    }

    #[test]
    fn test_plain_generate_request_forwards_require_reasoning_bool_only() {
        let enabled: GenerateRequest = serde_json::from_value(json!({
            "text": "hello",
            "require_reasoning": true
        }))
        .expect("parse generate request");
        let enabled_proto = SglangSchedulerClient::build_plain_generate_request_parts(
            "enabled".to_string(),
            &enabled,
            Some("hello".into()),
            vec![1],
        )
        .expect("build request");
        assert!(enabled_proto.require_reasoning);

        let disabled: GenerateRequest = serde_json::from_value(json!({
            "text": "hello",
            "require_reasoning": false
        }))
        .expect("parse generate request");
        let disabled_proto = SglangSchedulerClient::build_plain_generate_request_parts(
            "disabled".to_string(),
            &disabled,
            Some("hello".into()),
            vec![1],
        )
        .expect("build request");
        assert!(!disabled_proto.require_reasoning);

        let non_bool: GenerateRequest = serde_json::from_value(json!({
            "text": "hello",
            "require_reasoning": "true"
        }))
        .expect("parse generate request");
        let non_bool_proto = SglangSchedulerClient::build_plain_generate_request_parts(
            "non-bool".to_string(),
            &non_bool,
            Some("hello".into()),
            vec![1],
        )
        .expect("build request");
        assert!(!non_bool_proto.require_reasoning);
    }

    #[test]
    fn test_chat_options_forward_require_reasoning() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .expect("parse chat request");

        let proto = SglangSchedulerClient::build_generate_request_from_chat_parts(
            "chat".to_string(),
            &request,
            "hello".to_string(),
            vec![1],
            SglangGenerateRequestOptions {
                require_reasoning: true,
                ..Default::default()
            },
        )
        .expect("build request");

        assert!(proto.require_reasoning);
    }

    #[test]
    fn test_messages_options_forward_require_reasoning() {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16
        }))
        .expect("parse messages request");

        let proto = SglangSchedulerClient::build_generate_request_from_messages_parts(
            "messages".to_string(),
            &request,
            "hello".to_string(),
            vec![1],
            SglangGenerateRequestOptions {
                require_reasoning: true,
                ..Default::default()
            },
        )
        .expect("build request");

        assert!(proto.require_reasoning);
    }

    #[test]
    fn test_embed_request() {
        let embed_req = proto::EmbedRequest {
            request_id: "embed-req-202".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "This is a test sentence for embedding".to_string(),
                input_ids: vec![2028, 374, 264, 1296, 11914, 369, 28537], // Mock token IDs
            }),
            data_parallel_rank: 0,
            ..Default::default()
        };

        assert_eq!(embed_req.request_id, "embed-req-202");
        if let Some(ref tokenized) = &embed_req.tokenized {
            assert_eq!(
                tokenized.original_text,
                "This is a test sentence for embedding"
            );
        }
        assert_eq!(embed_req.data_parallel_rank, 0);
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = SglangSchedulerClient::connect("invalid://endpoint").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenized_input() {
        let tokenized = proto::TokenizedInput {
            original_text: "Hello world".to_string(),
            input_ids: vec![1, 15043, 1917, 2],
        };

        assert_eq!(tokenized.original_text, "Hello world");
        assert_eq!(tokenized.input_ids, vec![1, 15043, 1917, 2]);
    }

    #[test]
    fn test_generate_stream_chunk() {
        let chunk = proto::GenerateStreamChunk {
            token_ids: vec![1234, 5678],
            prompt_tokens: 5,
            completion_tokens: 2,
            cached_tokens: 3,
            ..Default::default()
        };

        assert_eq!(chunk.token_ids, vec![1234, 5678]);
        assert_eq!(chunk.prompt_tokens, 5);
        assert_eq!(chunk.completion_tokens, 2);
        assert_eq!(chunk.cached_tokens, 3);
    }

    // TODO: ModelInfo not in current proto - skip test
}
