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

crate::impl_abort_for_drop_with_reason!(VllmEngineClient);

impl VllmEngineClient {
    crate::impl_engine_client_basics!(
        proto,
        proto::vllm_engine_client::VllmEngineClient<Channel>,
        "vLLM"
    );

    /// The request every builder of this client sends: the tokenized input, the
    /// sampling, the stream flag and the multimodal inputs. The fields no
    /// builder sets are named here, once.
    fn tokenized_request(
        request_id: String,
        original_text: String,
        token_ids: Vec<u32>,
        sampling_params: proto::SamplingParams,
        stream: bool,
        mm_inputs: Option<proto::MultimodalInputs>,
    ) -> proto::GenerateRequest {
        proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream,
            kv_transfer_params: None,
            kv_transfer_params_json: None,
            data_parallel_rank: None,
            mm_inputs,
            media_refs: None,
            extra_mm_inputs: Vec::new(),
        }
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

    crate::impl_admin_ops!();
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

        Ok(Self::tokenized_request(
            request_id,
            processed_text,
            token_ids,
            sampling_params,
            body.stream,
            multimodal_inputs,
        ))
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

        Ok(Self::tokenized_request(
            request_id,
            original_text.unwrap_or_default(),
            token_ids,
            sampling_params,
            body.stream,
            None,
        ))
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

        Ok(Self::tokenized_request(
            request_id,
            processed_text,
            token_ids,
            sampling_params,
            body.stream.unwrap_or(false),
            None,
        ))
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
        let stop_sequences = stop_strings(request.stop.as_ref());

        let max_tokens = request.output_token_cap();

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
            include_stop_str_in_output: request.no_stop_trim,
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

    /// Build constraint for structured generation
    fn build_constraint_for_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();

        if let Some(schema) = response_format_schema(request)? {
            constraints.push(proto::sampling_params::Constraint::JsonSchema(schema));
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

        Ok(Self::tokenized_request(
            request_id,
            processed_text,
            token_ids,
            sampling_params,
            body.stream.unwrap_or(false),
            multimodal_inputs,
        ))
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

        Ok(Self::tokenized_request(
            request_id,
            original_text,
            token_ids,
            sampling_params,
            body.stream,
            None,
        ))
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
        let stop_sequences = stop_strings(request.stop.as_ref());

        let logprobs = request.logprobs.map(|v| v.min(20) as i32);

        let constraint = Self::constraint_of(completion_constraint(request)?);

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

    /// The engine's variant for the structured constraint a request names.
    fn constraint_of(
        constraint: Option<(ConstraintKind, String)>,
    ) -> Option<proto::sampling_params::Constraint> {
        constraint.map(|(kind, value)| match kind {
            ConstraintKind::JsonSchema => proto::sampling_params::Constraint::JsonSchema(value),
            ConstraintKind::Regex => proto::sampling_params::Constraint::Regex(value),
            ConstraintKind::Ebnf => proto::sampling_params::Constraint::Grammar(value),
        })
    }

    fn build_sampling_params_from_plain(
        params: Option<&GenerateSamplingParams>,
    ) -> Result<proto::SamplingParams, String> {
        let plain = PlainSampling::read(params);
        Ok(proto::SamplingParams {
            temperature: Some(plain.temperature),
            top_p: plain.top_p,
            top_k: plain.top_k.map_or(0, |k| k.max(0) as u32), // 0 means disabled in vLLM
            min_p: plain.min_p.unwrap_or(0.0),
            frequency_penalty: plain.frequency_penalty.unwrap_or(0.0),
            presence_penalty: plain.presence_penalty.unwrap_or(0.0),
            repetition_penalty: plain.repetition_penalty,
            max_tokens: plain.max_new_tokens,
            min_tokens: plain.min_new_tokens.unwrap_or(0),
            stop: plain.stop,
            stop_token_ids: plain.stop_token_ids,
            skip_special_tokens: plain.skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: plain.ignore_eos,
            include_stop_str_in_output: plain.no_stop_trim,
            n: plain.n,
            // The proto seed is i32; the plain seed is u64. Fold the high half into
            // the low so distinct seeds stay distinct (a saturating cast would send
            // every seed >= 2^31 to one value).
            seed: plain
                .sampling_seed
                .map(|seed| ((seed ^ (seed >> 32)) as u32) as i32),
            constraint: match params {
                Some(p) => Self::constraint_of(plain_constraint(p)?),
                None => None,
            },
            ..Default::default()
        })
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
            num_waiting_uncached_tokens: load.num_waiting_uncached_tokens,
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
            sampled_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::StringOrArray;

    use super::*;
    use crate::channel::testing::refused_endpoint;

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
            media_refs: None,
            extra_mm_inputs: Vec::new(),
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
    fn test_abort_request_construction() {
        let abort_req = proto::AbortRequest {
            request_ids: vec!["req-456".to_string(), "req-789".to_string()],
        };
        assert_eq!(abort_req.request_ids, vec!["req-456", "req-789"]);
    }

    #[test]
    fn plain_sampling_seed_reaches_the_proto_seed_folded() {
        let seeded = |seed| GenerateSamplingParams {
            sampling_seed: seed,
            ..Default::default()
        };
        let seed = |params| {
            VllmEngineClient::build_sampling_params_from_plain(Some(&params))
                .unwrap()
                .seed
        };
        assert_eq!(seed(seeded(Some(7))), Some(7));
        assert_eq!(seed(seeded(Some(1 << 32))), Some(1));
        assert_eq!(seed(seeded(Some(u64::MAX))), Some(0));
        assert_eq!(seed(seeded(None)), None);
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
    fn test_completion_sampling_params_pass_up_to_20_logprobs() {
        // `logprobs` on /v1/completions goes up to 20, as chat's
        // `top_logprobs`, and the backend gets the count as asked.
        for (logprobs, expected) in [(0, 0), (5, 5), (20, 20)] {
            let request = CompletionRequest {
                logprobs: Some(logprobs),
                ..minimal_completion_request()
            };
            let params = VllmEngineClient::build_grpc_sampling_params_from_completion(&request)
                .expect("build sampling params");
            assert_eq!(params.logprobs, Some(expected));
        }
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
    async fn connect_fails_on_a_malformed_or_refused_endpoint() {
        let malformed = VllmEngineClient::connect("not an endpoint").await;
        assert!(
            malformed.is_err(),
            "a malformed endpoint fails before the dial"
        );
        let refused = refused_endpoint();
        let result = VllmEngineClient::connect(&refused.url).await;
        assert!(result.is_err(), "a refused dial fails the connect");
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

    /// Each constraint kind reaches the engine as its own variant; none stays none.
    #[test]
    fn constraint_of_maps_each_kind_to_the_engines_variant() {
        use crate::request_fields::ConstraintKind;
        for (kind, expected) in [
            (
                ConstraintKind::JsonSchema,
                proto::sampling_params::Constraint::JsonSchema("v".to_string()),
            ),
            (
                ConstraintKind::Regex,
                proto::sampling_params::Constraint::Regex("v".to_string()),
            ),
            (
                ConstraintKind::Ebnf,
                proto::sampling_params::Constraint::Grammar("v".to_string()),
            ),
        ] {
            assert_eq!(
                VllmEngineClient::constraint_of(Some((kind, "v".to_string()))),
                Some(expected)
            );
        }
        assert_eq!(VllmEngineClient::constraint_of(None), None);
    }
}
