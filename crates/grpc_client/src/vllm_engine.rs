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
    tonic::include_proto!("vllm.grpc.engine");
}

/// Streaming `generate()` response that auto-aborts on drop. Concrete
/// alias for the generic `crate::AbortOnDropStream`.
pub type AbortOnDropStream = crate::AbortOnDropStream<proto::GenerateResponse, VllmEngineClient>;

/// gRPC client for vLLM scheduler
#[derive(Clone)]
pub struct VllmEngineClient {
    client: proto::vllm_engine_client::VllmEngineClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl AbortOnDropClient for VllmEngineClient {
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

impl VllmEngineClient {
    crate::impl_engine_client_basics!(proto::vllm_engine_client::VllmEngineClient<Channel>, "vLLM");

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

    /// Abort a request
    pub async fn abort_request(
        &self,
        request_id: String,
        _reason: String,
    ) -> Result<(), tonic::Status> {
        debug!("Sending abort request for {}", request_id);
        let request = Request::new(proto::AbortRequest {
            request_ids: vec![request_id.clone()],
        });

        let mut client = self.client.clone();
        let _response = client.abort(request).await?;
        debug!("Abort response received for {}", request_id);
        Ok(())
    }

    crate::impl_get_tokenizer!();
    crate::impl_subscribe_kv_events!();

    /// Get load metrics from the vLLM engine.
    pub async fn get_loads(
        &self,
        include: Vec<String>,
    ) -> Result<proto::GetLoadsResponse, tonic::Status> {
        debug!("Requesting vLLM load metrics");
        let request = Request::new(proto::GetLoadsRequest {
            dp_rank: None,
            include,
        });

        let mut client = self.client.clone();
        let response = client.get_loads(request).await?;
        debug!("vLLM load metrics response received");
        Ok(response.into_inner())
    }

    /// Build a single vLLM GenerateRequest from OpenAI ChatCompletionRequest
    pub fn build_generate_request_from_chat(
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>, // (constraint_type, constraint_value)
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling params
        let sampling_params =
            Self::build_grpc_sampling_params_from_chat(body, tool_call_constraint)?;

        let mm_inputs = multimodal_inputs;

        let grpc_request = proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: processed_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream,
            kv_transfer_params: None,
            kv_transfer_params_json: None,
            data_parallel_rank: None,
            mm_inputs,
        };

        Ok(grpc_request)
    }

    /// Build a basic GenerateRequest from the vLLM spec GenerateRequest
    pub fn build_plain_generate_request(
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            Self::build_sampling_params_from_plain(body.sampling_params.as_ref())?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: original_text.unwrap_or_default(),
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream,
            kv_transfer_params: None,
            kv_transfer_params_json: None,
            data_parallel_rank: None,
            mm_inputs: None,
        };

        Ok(grpc_request)
    }

    /// Build a GenerateRequest from ResponsesRequest (OpenAI Responses API)
    ///
    /// NOTE: This is used by the Harmony router only. The Regular router uses
    /// responses_to_chat() conversion and goes through the chat pipeline.
    pub fn build_generate_request_from_responses(
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
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: processed_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            kv_transfer_params: None,
            kv_transfer_params_json: None,
            data_parallel_rank: None,
            mm_inputs: None,
        };

        Ok(grpc_request)
    }

    /// Build gRPC SamplingParams from ChatCompletionRequest
    #[expect(
        deprecated,
        reason = "ChatCompletionRequest.seed is marked Legacy by openai-protocol, but vLLM still honors it"
    )]
    fn build_grpc_sampling_params_from_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = Self::extract_stop_strings(request);

        let max_tokens = request.max_completion_tokens;

        // Hardcode to true: gRPC backends return raw token IDs, not decoded text.
        // Detokenization happens on the SMG Rust side (StopDecoder/Sequence).
        let skip_special_tokens = true;

        // Map logprobs: if request.logprobs is true, use top_logprobs value (or 1 if not specified)
        // OpenAI API only exposes output logprobs, not prompt logprobs, for chat completions
        let logprobs = if request.logprobs {
            Some(request.top_logprobs.unwrap_or(1).min(20) as i32)
        } else {
            None
        };

        Ok(proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.map(|v| v.max(0) as u32).unwrap_or(0), // 0 means disabled in vLLM
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_tokens,
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens,
            spaces_between_special_tokens: true, // Default from Python SamplingParams
            ignore_eos: request.ignore_eos,
            n: request.n.unwrap_or(1),
            logprobs,
            // Proto seed is i32 (line 48 of generated vllm.grpc.engine.rs);
            // OpenAI request seed is i64. Saturating cast keeps "set" vs
            // "unset" distinction (None stays None — vLLM will pick a random
            // seed itself, matching prior behaviour).
            seed: request
                .seed
                .map(|s| s.clamp(i32::MIN as i64, i32::MAX as i64) as i32),
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

        // vLLM supports: json_schema, regex, grammar, structural_tag, json_object, choice
        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::Grammar(ebnf.clone()));
        }

        if let Some(regex) = &request.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }

        // Handle tool call constraint from preparation stage.
        // If response_format already set a constraint, clear it — tool constraint wins
        // (matches vLLM HTTP behavior where tool calling overrides response_format).
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if !constraints.is_empty() {
                warn!(
                    "Constrained decoding is not compatible with tool calls, using tool constraint"
                );
                constraints.clear();
            }
            let tool_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "grammar" | "ebnf" => proto::sampling_params::Constraint::Grammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {constraint_type}")),
            };
            constraints.push(tool_constraint);
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

        let max_tokens = request.max_output_tokens;

        Ok(proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.max(0) as u32,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty,
            max_tokens,
            stop: vec![], // Does not pass through request.stop yet (follow-up fix)
            stop_token_ids: vec![], // Handled by Harmony stop tokens
            skip_special_tokens: false, // Keep special tokens for Harmony
            spaces_between_special_tokens: true,
            ignore_eos: false,
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
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "grammar" | "ebnf" => proto::sampling_params::Constraint::Grammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {constraint_type}")),
            };
            Ok(Some(parsed_constraint))
        } else {
            Ok(None)
        }
    }

    /// Build a GenerateRequest from CreateMessageRequest (Anthropic Messages API)
    pub fn build_generate_request_from_messages(
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            Self::build_grpc_sampling_params_from_messages(body, tool_call_constraint)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: processed_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            kv_transfer_params: None,
            kv_transfer_params_json: None,
            data_parallel_rank: None,
            mm_inputs: multimodal_inputs,
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
            temperature: Some(request.temperature.unwrap_or(1.0) as f32),
            top_p: request.top_p.unwrap_or(1.0) as f32,
            top_k: request.top_k.unwrap_or(0), // 0 means disabled in vLLM
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_tokens: Some(request.max_tokens),
            stop: stop_sequences,
            stop_token_ids: vec![],
            skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: false,
            n: 1,
            logprobs: None,
            constraint: Self::build_constraint_for_responses(tool_call_constraint)?,
            ..Default::default()
        })
    }

    /// Build a GenerateRequest from CompletionRequest (`/v1/completions`)
    pub fn build_generate_request_from_completion(
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = Self::build_grpc_sampling_params_from_completion(body)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream,
            kv_transfer_params: None,
            kv_transfer_params_json: None,
            data_parallel_rank: None,
            mm_inputs: None,
        };

        Ok(grpc_request)
    }

    /// Build an EmbedRequest for embedding/classify endpoints
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API across gRPC backends"
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
        }
    }

    /// Submit an embedding request
    pub async fn embed(
        &self,
        req: proto::EmbedRequest,
    ) -> Result<proto::EmbedResponse, tonic::Status> {
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.embed(request).await?;
        Ok(response.into_inner())
    }

    fn build_grpc_sampling_params_from_completion(
        request: &CompletionRequest,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = match &request.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        };

        let logprobs = request.logprobs.map(|v| v.min(5) as i32);

        let constraint = Self::build_single_constraint_from_completion(request)?;

        Ok(proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.map(|v| v.max(0) as u32).unwrap_or(0),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_tokens: request.max_tokens,
            min_tokens: request.min_tokens.unwrap_or(0),
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens: request.skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: request.ignore_eos,
            include_stop_str_in_output: request.no_stop_trim,
            n: request.n.unwrap_or(1),
            logprobs,
            seed: request
                .seed
                .map(|s| s.clamp(i32::MIN as i64, i32::MAX as i64) as i32),
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
            constraints.push(proto::sampling_params::Constraint::Grammar(ebnf.clone()));
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
            constraints.push(proto::sampling_params::Constraint::Grammar(ebnf.clone()));
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
            temperature: Some(1.0),
            top_p: 1.0,
            top_k: 0, // 0 means disabled in vLLM
            repetition_penalty: 1.0,
            n: 1,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ..Default::default()
        };

        let Some(p) = params else {
            return Ok(sampling);
        };

        // Handle temperature (now optional)
        if let Some(val) = p.temperature {
            sampling.temperature = Some(val);
        }

        // Simple field mappings
        if let Some(val) = p.top_p {
            sampling.top_p = val;
        }
        if let Some(val) = p.top_k {
            sampling.top_k = val.max(0) as u32; // Clamp negative values to 0 (disabled)
        }
        if let Some(val) = p.frequency_penalty {
            sampling.frequency_penalty = val;
        }
        if let Some(val) = p.presence_penalty {
            sampling.presence_penalty = val;
        }
        if let Some(val) = p.repetition_penalty {
            sampling.repetition_penalty = val;
        }
        if let Some(val) = p.min_p {
            sampling.min_p = val;
        }
        if let Some(val) = p.ignore_eos {
            sampling.ignore_eos = val;
        }
        if let Some(val) = p.skip_special_tokens {
            sampling.skip_special_tokens = val;
        }
        // Note: no_stop_trim not supported in vLLM

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

        // Handle max_tokens (read from internal max_new_tokens)
        if let Some(max_new_tokens) = p.max_new_tokens {
            sampling.max_tokens = Some(max_new_tokens);
        }

        // Handle min_tokens (read from internal min_new_tokens)
        if let Some(min_new_tokens) = p.min_new_tokens {
            sampling.min_tokens = min_new_tokens;
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
// Proto → protocol type conversions (load metrics)
// ---------------------------------------------------------------------------

impl From<proto::SchedulerLoad> for openai_protocol::worker::SchedulerLoadSnapshot {
    fn from(load: proto::SchedulerLoad) -> Self {
        Self {
            dp_rank: load.dp_rank,
            num_running_reqs: load.num_running_reqs,
            num_waiting_reqs: load.num_waiting_reqs,
            // vLLM does not report queued token-work; degrade to 0.
            num_waiting_uncached_tokens: 0,
            num_total_reqs: load.num_total_reqs,
            num_used_tokens: load.num_used_tokens,
            max_total_num_tokens: load.max_total_num_tokens,
            token_usage: load.token_usage,
            gen_throughput: load.gen_throughput,
            cache_hit_rate: load.cache_hit_rate,
            utilization: load.utilization,
            max_running_requests: load.max_running_requests,
            // vLLM disagg mapping is out of scope here; canonical PD fields
            // stay None until the vLLM proto exposes them.
            ..Default::default()
        }
    }
}

impl From<proto::GetLoadsResponse> for openai_protocol::worker::WorkerLoadResponse {
    fn from(resp: proto::GetLoadsResponse) -> Self {
        Self {
            timestamp: resp.timestamp,
            version: resp.version,
            dp_rank_count: resp.dp_rank_count,
            loads: resp.loads.into_iter().map(Into::into).collect(),
            aggregate: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid CompletionRequest for tests that only care about
    /// sampling-param passthrough. CompletionRequest doesn't derive Default,
    /// so we provide the required fields here.
    fn minimal_completion_request() -> CompletionRequest {
        CompletionRequest {
            model: String::new(),
            prompt: StringOrArray::String(String::new()),
            best_of: None,
            echo: false,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            max_tokens: None,
            n: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            stream: false,
            stream_options: None,
            suffix: None,
            temperature: None,
            top_p: None,
            user: None,
            top_k: None,
            min_p: None,
            min_tokens: None,
            repetition_penalty: None,
            regex: None,
            ebnf: None,
            json_schema: None,
            stop_token_ids: None,
            no_stop_trim: false,
            ignore_eos: false,
            skip_special_tokens: true,
            lora_path: None,
            session_params: None,
            return_hidden_states: false,
            sampling_seed: None,
            rid: None,
            other: Default::default(),
        }
    }

    #[test]
    fn test_proto_types_compilation() {
        let _health_req = proto::HealthCheckRequest {};
        // HealthCheckRequest is now empty - no fields to test
    }

    #[test]
    fn test_generate_request_construction() {
        let sampling_params = proto::SamplingParams {
            temperature: Some(0.7),
            max_tokens: Some(128),
            top_p: 0.9,
            top_k: 50,
            stop: vec!["</s>".to_string()],
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "test-req-123".to_string(),
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: "Hello world".to_string(),
                    input_ids: vec![9906, 1917], // Mock token IDs for "Hello world"
                },
            )),
            sampling_params: Some(sampling_params),
            stream: false,
            kv_transfer_params: None,
            kv_transfer_params_json: None,
            data_parallel_rank: None,
            mm_inputs: None,
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(proto::generate_request::Input::Tokenized(ref tokenized)) = gen_req.input {
            assert_eq!(tokenized.original_text, "Hello world");
        }
        // vLLM: logprobs are in SamplingParams, not GenerateRequest

        let params = gen_req.sampling_params.unwrap();
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.max_tokens, Some(128));
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
            request_ids: vec!["req-456".to_string(), "req-789".to_string()],
        };
        assert_eq!(abort_req.request_ids, vec!["req-456", "req-789"]);
    }

    #[test]
    fn test_sampling_params_defaults() {
        let params = proto::SamplingParams::default();
        // Optional float field defaults to None
        assert_eq!(params.temperature, None);
        // Non-optional numeric fields have proto defaults (0)
        assert_eq!(params.top_p, 0.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repetition_penalty, 0.0);
        assert_eq!(params.n, 0);
        // Bool fields have proto defaults (false)
        assert!(!params.skip_special_tokens);
        assert!(!params.spaces_between_special_tokens);
        assert!(!params.ignore_eos);
        assert!(!params.include_stop_str_in_output);
        // Optional fields should be None
        assert_eq!(params.max_tokens, None);
        assert_eq!(params.logprobs, None);
        // Other non-optional fields
        assert_eq!(params.min_p, 0.0);
        assert_eq!(params.frequency_penalty, 0.0);
        assert_eq!(params.presence_penalty, 0.0);
        assert!(params.stop.is_empty());
    }

    // TODO: MultimodalInputs not in vLLM proto - skip test
    // vLLM handles multimodal inputs differently than SGLang

    // TODO: SessionParams not in current proto - skip test

    #[test]
    #[expect(
        deprecated,
        reason = "ChatCompletionRequest.seed is marked Legacy by openai-protocol, but vLLM still honors it"
    )]
    fn test_chat_sampling_params_seed_is_passed_through() {
        // Regression guard: build_grpc_sampling_params_from_chat() previously
        // omitted the seed field, so `..Default::default()` filled proto seed
        // with None even when the request set it. At temp=0 this was inert
        // (vLLM's gumbel sampler gates seed use on temp != 0), but at temp>0
        // it silently broke reproducibility.
        let request = ChatCompletionRequest {
            seed: Some(42),
            ..Default::default()
        };
        let params = VllmEngineClient::build_grpc_sampling_params_from_chat(&request, None)
            .expect("build sampling params");
        assert_eq!(params.seed, Some(42));

        // No seed in the request → proto seed stays None (vLLM picks one).
        let unset = ChatCompletionRequest {
            seed: None,
            ..Default::default()
        };
        let unset_params = VllmEngineClient::build_grpc_sampling_params_from_chat(&unset, None)
            .expect("build sampling params");
        assert_eq!(unset_params.seed, None);

        // Saturating cast: request.seed is i64, proto seed is i32.
        let huge = ChatCompletionRequest {
            seed: Some(i64::MAX),
            ..Default::default()
        };
        let huge_params = VllmEngineClient::build_grpc_sampling_params_from_chat(&huge, None)
            .expect("build sampling params");
        assert_eq!(huge_params.seed, Some(i32::MAX));
    }

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

        let params = VllmEngineClient::build_grpc_sampling_params_from_responses(&request, None)
            .expect("build sampling params");

        assert_eq!(params.top_k, 40);
        assert!((params.min_p - 0.05).abs() < 1e-6);
        assert!((params.repetition_penalty - 1.2).abs() < 1e-6);
        assert!((params.frequency_penalty - 0.3).abs() < 1e-6);
        assert!((params.presence_penalty - (-0.4)).abs() < 1e-6);

        // Negative top_k is clamped to 0 (vLLM's "disabled" sentinel).
        let disabled = ResponsesRequest {
            top_k: -1,
            ..Default::default()
        };
        let disabled_params =
            VllmEngineClient::build_grpc_sampling_params_from_responses(&disabled, None)
                .expect("build sampling params");
        assert_eq!(disabled_params.top_k, 0);
    }

    #[test]
    fn test_completion_sampling_params_seed_is_passed_through() {
        // Regression guard: build_grpc_sampling_params_from_completion()
        // previously omitted the seed field, so `..Default::default()`
        // filled proto seed with None even when the request set it.
        let request = CompletionRequest {
            seed: Some(42),
            ..minimal_completion_request()
        };
        let params = VllmEngineClient::build_grpc_sampling_params_from_completion(&request)
            .expect("build sampling params");
        assert_eq!(params.seed, Some(42));

        // No seed → proto seed stays None.
        let unset = CompletionRequest {
            seed: None,
            ..minimal_completion_request()
        };
        let unset_params = VllmEngineClient::build_grpc_sampling_params_from_completion(&unset)
            .expect("build sampling params");
        assert_eq!(unset_params.seed, None);

        // Saturating cast: request.seed is i64, proto seed is i32.
        let huge = CompletionRequest {
            seed: Some(i64::MAX),
            ..minimal_completion_request()
        };
        let huge_params = VllmEngineClient::build_grpc_sampling_params_from_completion(&huge)
            .expect("build sampling params");
        assert_eq!(huge_params.seed, Some(i32::MAX));
    }

    #[test]
    fn test_embed_request() {
        let embed_req = proto::EmbedRequest {
            request_id: "embed-req-202".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "This is a test sentence for embedding".to_string(),
                input_ids: vec![2028, 374, 264, 1296, 11914, 369, 28537], // Mock token IDs
            }),
        };

        assert_eq!(embed_req.request_id, "embed-req-202");
        if let Some(ref tokenized) = &embed_req.tokenized {
            assert_eq!(
                tokenized.original_text,
                "This is a test sentence for embedding"
            );
        }
        // vLLM: no data_parallel_rank in EmbedRequest
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = VllmEngineClient::connect("invalid://endpoint").await;
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
            output_logprobs: None,
            input_logprobs: None,
            index: 0,
        };

        assert_eq!(chunk.token_ids, vec![1234, 5678]);
        assert_eq!(chunk.prompt_tokens, 5);
        assert_eq!(chunk.completion_tokens, 2);
        assert_eq!(chunk.cached_tokens, 3);
        assert_eq!(chunk.index, 0);
    }

    // TODO: ModelInfo not in current proto - skip test
}
