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
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    tonic::include_proto!("trtllm");
}

/// Streaming `generate()` response that auto-aborts on drop. Concrete
/// alias for the generic `crate::AbortOnDropStream`.
pub type AbortOnDropStream = crate::AbortOnDropStream<proto::GenerateResponse, TrtllmServiceClient>;

/// gRPC client for TensorRT-LLM service
#[derive(Clone)]
pub struct TrtllmServiceClient {
    client: proto::trtllm_service_client::TrtllmServiceClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl AbortOnDropClient for TrtllmServiceClient {
    fn abort_for_drop(
        self,
        request_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>> {
        Box::pin(async move {
            // trtllm's abort returns an `AbortResponse`; collapse to `()`
            // so the wrapper matches the trait signature.
            self.abort_request(request_id).await.map(|_| ())
        })
    }
}

impl TrtllmServiceClient {
    crate::impl_engine_client_basics!(
        proto::trtllm_service_client::TrtllmServiceClient<Channel>,
        "TensorRT-LLM"
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

    /// Abort a request
    pub async fn abort_request(
        &self,
        request_id: String,
    ) -> Result<proto::AbortResponse, tonic::Status> {
        debug!("Sending abort request for {}", request_id);
        let request = Request::new(proto::AbortRequest {
            request_id: request_id.clone(),
        });

        let mut client = self.client.clone();
        let response = client.abort(request).await?;
        debug!(
            "Abort response for {}: success={}, message={}",
            request_id,
            response.get_ref().success,
            response.get_ref().message
        );
        Ok(response.into_inner())
    }

    crate::impl_get_tokenizer!();
    crate::impl_subscribe_kv_events!();

    /// Build a TensorRT-LLM GenerateRequest from OpenAI ChatCompletionRequest
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API across gRPC backends"
    )]
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_input: Option<proto::MultimodalInput>,
        tool_call_constraint: Option<(String, String)>, // (constraint_type, constraint_value)
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling config
        let sampling_config = Self::build_sampling_config_from_chat(body);

        // Build output config
        let output_config = Self::build_output_config_from_chat(body);

        // Build guided decoding params if needed
        let guided_decoding = Self::build_guided_decoding_from_chat(body, tool_call_constraint)?;

        let stop = Self::extract_stop_strings(body.stop.as_ref());

        let max_tokens = body.output_token_cap().unwrap_or(2048);

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens,
            streaming: body.stream,
            stop,
            stop_token_ids: vec![],
            ignore_eos: body.ignore_eos,
            bad: vec![],
            bad_token_ids: vec![],
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
            include_stop_token_in_output: false,
        };

        Ok(grpc_request)
    }

    /// Build a basic GenerateRequest from the GenerateRequest spec
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API across gRPC backends"
    )]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "returns Result for API consistency with sglang/vllm backends which can fail"
    )]
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_config = Self::build_sampling_config_from_plain(body.sampling_params.as_ref());
        let output_config = proto::OutputConfig {
            logprobs: if body.return_logprob.unwrap_or(false) {
                Some(body.top_logprobs_num.unwrap_or(0))
            } else {
                None
            },
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        };

        // Build guided decoding from plain sampling params
        let guided_decoding = if let Some(params) = &body.sampling_params {
            Self::build_guided_decoding_from_plain(params)
        } else {
            None
        };

        let max_tokens = body
            .sampling_params
            .as_ref()
            .and_then(|p| p.max_new_tokens)
            .unwrap_or(2048);

        let stop =
            Self::extract_stop_strings(body.sampling_params.as_ref().and_then(|p| p.stop.as_ref()));

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens,
            streaming: body.stream,
            stop,
            stop_token_ids: vec![],
            ignore_eos: body
                .sampling_params
                .as_ref()
                .and_then(|p| p.ignore_eos)
                .unwrap_or(false),
            bad: vec![],
            bad_token_ids: vec![],
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input: None,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
            include_stop_token_in_output: false,
        };

        Ok(grpc_request)
    }

    /// Build a GenerateRequest from ResponsesRequest (OpenAI Responses API)
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
        let sampling_config = Self::build_sampling_config_from_responses(body);
        let output_config = proto::OutputConfig {
            logprobs: body.top_logprobs.map(|v| v as i32),
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        };

        let guided_decoding = Self::build_guided_decoding_from_responses(constraint)?;

        let max_tokens = body.max_output_tokens.unwrap_or(2048);

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens,
            streaming: body.stream.unwrap_or(false),
            stop: vec![], // Does not pass through body.stop yet (follow-up fix)
            stop_token_ids: vec![],
            ignore_eos: false,
            bad: vec![],
            bad_token_ids: vec![],
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input: None,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
            include_stop_token_in_output: false,
        };

        Ok(grpc_request)
    }

    /// Extract stop strings from an optional StringOrArray
    fn extract_stop_strings(stop: Option<&StringOrArray>) -> Vec<String> {
        match stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        }
    }

    /// Build SamplingConfig from ChatCompletionRequest
    ///
    /// Uses sensible defaults when values are not specified:
    /// - temperature: 1.0 (neutral sampling)
    /// - top_p: 1.0 (no nucleus filtering)
    /// - repetition_penalty: 1.0 (no penalty)
    fn build_sampling_config_from_chat(request: &ChatCompletionRequest) -> proto::SamplingConfig {
        proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: request.n.unwrap_or(1),
            top_k: request.top_k.map(|v| v.max(0)),
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            temperature: Some(request.temperature.unwrap_or(1.0)),
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(request.repetition_penalty.unwrap_or(1.0)),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: request.min_p,
            beam_width_array: vec![],
        }
    }

    /// Build OutputConfig from ChatCompletionRequest
    fn build_output_config_from_chat(request: &ChatCompletionRequest) -> proto::OutputConfig {
        proto::OutputConfig {
            logprobs: if request.logprobs {
                Some(request.top_logprobs.unwrap_or(0) as i32)
            } else {
                None
            },
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        }
    }

    /// Build GuidedDecodingParams from ChatCompletionRequest
    fn build_guided_decoding_from_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::GuidedDecodingParams>, String> {
        // Collect non-tool constraint (response_format, ebnf, regex)
        let mut guided = None;

        match &request.response_format {
            Some(ResponseFormat::JsonObject) => {
                let schema = serde_json::json!({"type": "object"});
                let schema_str = serde_json::to_string(&schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;
                guided = Some(proto::GuidedDecodingParams {
                    guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
                    guide: schema_str,
                });
            }
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema_str = serde_json::to_string(&json_schema.schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;
                guided = Some(proto::GuidedDecodingParams {
                    guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
                    guide: schema_str,
                });
            }
            Some(ResponseFormat::Text) | None => {}
        }

        if guided.is_none() {
            if let Some(ebnf) = &request.ebnf {
                guided = Some(proto::GuidedDecodingParams {
                    guide_type: proto::guided_decoding_params::GuideType::EbnfGrammar as i32,
                    guide: ebnf.clone(),
                });
            }
        }

        if guided.is_none() {
            if let Some(regex) = &request.regex {
                guided = Some(proto::GuidedDecodingParams {
                    guide_type: proto::guided_decoding_params::GuideType::Regex as i32,
                    guide: regex.clone(),
                });
            }
        }

        // Tool call constraint — drop if another constraint already set
        // (matches TRT-LLM HTTP behavior where response_format takes priority)
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if guided.is_some() {
                warn!("Constrained decoding is not compatible with tool calls, dropping tool constraint");
            } else {
                let guide_type = match constraint_type.as_str() {
                    "structural_tag" => proto::guided_decoding_params::GuideType::StructuralTag,
                    "json_schema" => proto::guided_decoding_params::GuideType::JsonSchema,
                    "ebnf" | "grammar" => proto::guided_decoding_params::GuideType::EbnfGrammar,
                    "regex" => proto::guided_decoding_params::GuideType::Regex,
                    _ => return Err(format!("Unknown constraint type: {constraint_type}")),
                };
                guided = Some(proto::GuidedDecodingParams {
                    guide_type: guide_type as i32,
                    guide: constraint_value,
                });
            }
        }

        Ok(guided)
    }

    /// Build SamplingConfig from ResponsesRequest
    fn build_sampling_config_from_responses(request: &ResponsesRequest) -> proto::SamplingConfig {
        proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            top_k: (request.top_k >= 0).then_some(request.top_k),
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            temperature: Some(request.temperature.unwrap_or(1.0)),
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(request.repetition_penalty),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: (request.min_p != 0.0).then_some(request.min_p),
            beam_width_array: vec![],
        }
    }

    /// Build GuidedDecodingParams from ResponsesRequest constraint
    fn build_guided_decoding_from_responses(
        constraint: Option<(String, String)>,
    ) -> Result<Option<proto::GuidedDecodingParams>, String> {
        if let Some((constraint_type, constraint_value)) = constraint {
            let guide_type = match constraint_type.as_str() {
                "structural_tag" => proto::guided_decoding_params::GuideType::StructuralTag,
                "json_schema" => proto::guided_decoding_params::GuideType::JsonSchema,
                "ebnf" | "grammar" => proto::guided_decoding_params::GuideType::EbnfGrammar,
                "regex" => proto::guided_decoding_params::GuideType::Regex,
                _ => return Err(format!("Unknown constraint type: {constraint_type}")),
            };
            Ok(Some(proto::GuidedDecodingParams {
                guide_type: guide_type as i32,
                guide: constraint_value,
            }))
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
        multimodal_input: Option<proto::MultimodalInput>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_config = Self::build_sampling_config_from_messages(body);
        let output_config = proto::OutputConfig {
            logprobs: None,
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        };

        let guided_decoding = Self::build_guided_decoding_from_responses(tool_call_constraint)?;

        let stop = body.stop_sequences.clone().unwrap_or_default();
        let max_tokens = body.max_tokens;

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens,
            streaming: body.stream.unwrap_or(false),
            stop,
            stop_token_ids: vec![],
            ignore_eos: false,
            bad: vec![],
            bad_token_ids: vec![],
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
            include_stop_token_in_output: false,
        };

        Ok(grpc_request)
    }

    /// Build SamplingConfig from CreateMessageRequest
    fn build_sampling_config_from_messages(
        request: &CreateMessageRequest,
    ) -> proto::SamplingConfig {
        proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            top_k: request.top_k.map(|v| v as i32),
            top_p: Some(request.top_p.unwrap_or(1.0) as f32),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            temperature: Some(request.temperature.unwrap_or(1.0) as f32),
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(1.0),
            presence_penalty: None,
            frequency_penalty: None,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: None,
            beam_width_array: vec![],
        }
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
        let sampling_config = Self::build_sampling_config_from_completion(body);
        let output_config = proto::OutputConfig {
            logprobs: body.logprobs.map(|v| v.min(5) as i32),
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        };

        let guided_decoding = Self::build_guided_decoding_from_completion(body)?;

        let stop = match &body.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        };

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text,
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens: body.max_tokens.unwrap_or(16),
            streaming: body.stream,
            stop,
            stop_token_ids: body.stop_token_ids.clone().unwrap_or_default(),
            ignore_eos: body.ignore_eos,
            bad: vec![],
            bad_token_ids: vec![],
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input: None,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
            include_stop_token_in_output: body.no_stop_trim,
        };

        Ok(grpc_request)
    }

    fn build_sampling_config_from_completion(request: &CompletionRequest) -> proto::SamplingConfig {
        proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: request.n.unwrap_or(1),
            top_k: request.top_k.map(|v| v.max(0)),
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: request.seed.map(|s| s as u64),
            temperature: Some(request.temperature.unwrap_or(1.0)),
            min_tokens: request.min_tokens,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(request.repetition_penalty.unwrap_or(1.0)),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: request.min_p,
            beam_width_array: vec![],
        }
    }

    fn build_guided_decoding_from_completion(
        request: &CompletionRequest,
    ) -> Result<Option<proto::GuidedDecodingParams>, String> {
        let mut guides = Vec::new();

        if let Some(json_schema) = &request.json_schema {
            guides.push((
                proto::guided_decoding_params::GuideType::JsonSchema,
                json_schema.clone(),
            ));
        }
        if let Some(regex) = &request.regex {
            guides.push((
                proto::guided_decoding_params::GuideType::Regex,
                regex.clone(),
            ));
        }
        if let Some(ebnf) = &request.ebnf {
            guides.push((
                proto::guided_decoding_params::GuideType::EbnfGrammar,
                ebnf.clone(),
            ));
        }

        match guides.len() {
            0 => Ok(None),
            1 => {
                #[expect(clippy::expect_used, reason = "INVARIANT: checked len == 1 above")]
                let (guide_type, guide) = guides.pop().expect("checked len == 1");
                Ok(Some(proto::GuidedDecodingParams {
                    guide_type: guide_type as i32,
                    guide,
                }))
            }
            _ => Err("Multiple structured constraints are not allowed".to_string()),
        }
    }

    fn build_sampling_config_from_plain(
        params: Option<&GenerateSamplingParams>,
    ) -> proto::SamplingConfig {
        let mut config = proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            top_k: None,
            top_p: Some(1.0),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            temperature: Some(1.0),
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(1.0),
            presence_penalty: None,
            frequency_penalty: None,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: None,
            beam_width_array: vec![],
        };

        let Some(p) = params else {
            return config;
        };

        if let Some(val) = p.temperature {
            config.temperature = Some(val);
        }
        if let Some(val) = p.top_p {
            config.top_p = Some(val);
        }
        if let Some(val) = p.top_k {
            config.top_k = Some(val.max(0));
        }
        if let Some(val) = p.frequency_penalty {
            config.frequency_penalty = Some(val);
        }
        if let Some(val) = p.presence_penalty {
            config.presence_penalty = Some(val);
        }
        if let Some(val) = p.repetition_penalty {
            config.repetition_penalty = Some(val);
        }
        if let Some(val) = p.min_p {
            config.min_p = Some(val);
        }
        config.min_tokens = p.min_new_tokens;
        if let Some(n) = p.n {
            config.num_return_sequences = n;
        }

        config
    }

    fn build_guided_decoding_from_plain(
        params: &GenerateSamplingParams,
    ) -> Option<proto::GuidedDecodingParams> {
        if let Some(json_schema) = &params.json_schema {
            return Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
                guide: json_schema.clone(),
            });
        }
        if let Some(regex) = &params.regex {
            return Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::Regex as i32,
                guide: regex.clone(),
            });
        }
        if let Some(ebnf) = &params.ebnf {
            return Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::EbnfGrammar as i32,
                guide: ebnf.clone(),
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        let _health_req = proto::HealthCheckRequest {};
    }

    #[test]
    fn test_generate_request_construction() {
        let sampling_config = proto::SamplingConfig {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(50),
            beam_width: 1,
            num_return_sequences: 1,
            ..Default::default()
        };

        let output_config = proto::OutputConfig {
            logprobs: Some(5),
            exclude_input_from_output: true,
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "test-req-123".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "Hello world".to_string(),
                input_token_ids: vec![9906, 1917],
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens: 128,
            streaming: false,
            ..Default::default()
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(ref tokenized) = gen_req.tokenized {
            assert_eq!(tokenized.original_text, "Hello world");
            assert_eq!(tokenized.input_token_ids, vec![9906, 1917]);
        }

        let config = gen_req.sampling_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.top_p, Some(0.9));
    }

    #[test]
    fn test_health_check_request() {
        let _health_req = proto::HealthCheckRequest {};
    }

    #[test]
    fn test_abort_request_construction() {
        let abort_req = proto::AbortRequest {
            request_id: "req-456".to_string(),
        };
        assert_eq!(abort_req.request_id, "req-456");
    }

    #[test]
    fn test_sampling_config_defaults() {
        let config = proto::SamplingConfig::default();
        assert_eq!(config.beam_width, 0);
        assert_eq!(config.temperature, None);
        assert_eq!(config.top_p, None);
        assert_eq!(config.top_k, None);
    }

    #[test]
    fn test_responses_sampling_config_is_passed_through() {
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

        let cfg = TrtllmServiceClient::build_sampling_config_from_responses(&request);

        assert_eq!(cfg.top_k, Some(40));
        assert_eq!(cfg.min_p, Some(0.05));
        assert_eq!(cfg.repetition_penalty, Some(1.2));
        assert_eq!(cfg.frequency_penalty, Some(0.3));
        assert_eq!(cfg.presence_penalty, Some(-0.4));

        // Default top_k (-1) maps to None, letting TRT-LLM use its own default.
        let disabled = ResponsesRequest {
            top_k: -1,
            ..Default::default()
        };
        let disabled_cfg = TrtllmServiceClient::build_sampling_config_from_responses(&disabled);
        assert_eq!(disabled_cfg.top_k, None);
        // Default min_p (0.0) maps to None for the same reason.
        assert_eq!(disabled_cfg.min_p, None);
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = TrtllmServiceClient::connect("invalid://endpoint").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenized_input() {
        let tokenized = proto::TokenizedInput {
            original_text: "Hello world".to_string(),
            input_token_ids: vec![1, 15043, 1917, 2],
            query_token_ids: vec![],
        };

        assert_eq!(tokenized.original_text, "Hello world");
        assert_eq!(tokenized.input_token_ids, vec![1, 15043, 1917, 2]);
    }

    #[test]
    fn test_generate_stream_chunk() {
        let chunk = proto::GenerateStreamChunk {
            token_ids: vec![1234, 5678],
            sequence_index: 0,
            prompt_tokens: 5,
            completion_tokens: 2,
            cached_tokens: 3,
            logprobs: vec![],
        };

        assert_eq!(chunk.token_ids, vec![1234, 5678]);
        assert_eq!(chunk.prompt_tokens, 5);
        assert_eq!(chunk.completion_tokens, 2);
        assert_eq!(chunk.cached_tokens, 3);
    }

    #[test]
    fn test_guided_decoding_params() {
        let guided = proto::GuidedDecodingParams {
            guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
            guide: r#"{"type": "object"}"#.to_string(),
        };

        assert_eq!(
            guided.guide_type,
            proto::guided_decoding_params::GuideType::JsonSchema as i32
        );
        assert_eq!(guided.guide, r#"{"type": "object"}"#);
    }
}
