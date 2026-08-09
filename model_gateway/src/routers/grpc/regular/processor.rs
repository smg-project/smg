//! Shared response processing logic for gRPC routers
//!
//! This module contains response processing functions that are shared between
//! the regular router and PD router.

use std::{sync::Arc, time::Instant};

use futures::future::try_join_all;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
    common::{
        FunctionCallResponse, StringOrArray, Tool, ToolCall, ToolChoice, ToolChoiceValue, Usage,
    },
    completion::{CompletionChoice, CompletionRequest, CompletionResponse},
    generate::{GenerateMetaInfo, GenerateRequest, GenerateResponse},
    messages::{self, CreateMessageRequest, Message},
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{error, warn};

use crate::routers::{
    error,
    grpc::{
        common::{response_collection, response_formatting},
        context::{DispatchMetadata, ExecutionResult},
        proto_wrapper::ProtoGenerateComplete,
        utils,
    },
};

/// Unified response processor for both routers
#[derive(Clone)]
pub(crate) struct ResponseProcessor {
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    pub configured_tool_parser: Option<String>,
    pub configured_reasoning_parser: Option<String>,
}

impl ResponseProcessor {
    pub fn new(
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        }
    }

    /// Process a single choice from GenerateComplete response
    #[expect(clippy::too_many_arguments)]
    pub async fn process_single_choice(
        &self,
        complete: &ProtoGenerateComplete,
        index: usize,
        original_request: &ChatCompletionRequest,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
        reasoning_parser_available: bool,
        tool_parser_available: bool,
    ) -> Result<ChatChoice, String> {
        stop_decoder.reset();
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(complete.output_ids())
            .map_err(|e| format!("Failed to process tokens: {e}"))?;

        // Accumulate text with early breaks
        let mut final_text = String::new();
        let mut stopped = false;
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    stopped = true;
                    break;
                }
                SequenceDecoderOutput::Stopped => {
                    stopped = true;
                    break;
                }
                SequenceDecoderOutput::Held => {}
            }
        }

        // Flush remaining text
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            final_text.push_str(&t);
        }

        // Step 1: Handle reasoning content parsing
        let mut reasoning_text: Option<String> = None;
        let mut processed_text = final_text;

        if original_request.separate_reasoning && reasoning_parser_available {
            // Fresh parser per request: non-streaming extraction keeps no state
            // across requests, so avoid serializing on the shared pooled mutex.
            if let Some(mut parser) = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                &original_request.model,
            ) {
                // If the template injected `<think>` in the prefill (thinking toggle
                // is supported and effectively ON), start in reasoning mode.
                if utils::should_mark_reasoning_started(
                    utils::resolve_user_thinking(original_request, tokenizer.as_ref()),
                    tokenizer.as_ref(),
                ) {
                    parser.mark_reasoning_started();
                }

                match parser.detect_and_parse_reasoning(&processed_text) {
                    Ok(result) => {
                        if !result.reasoning_text.is_empty() {
                            reasoning_text = Some(result.reasoning_text);
                        }
                        processed_text = result.normal_text;
                    }
                    Err(e) => {
                        warn!("Reasoning parsing error, skipping parsing: {e}");
                    }
                }
            }
        }

        // Step 2: Handle tool call parsing
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        if tool_choice_enabled && original_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific function or required mode)
            let has_structural_tag = self
                .tool_parser_factory
                .registry()
                .has_structural_tag_for_parser(self.configured_tool_parser.as_deref());
            let used_json_schema = if has_structural_tag {
                false
            } else {
                match &original_request.tool_choice {
                    Some(ToolChoice::Function { .. }) => true,
                    Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
                    Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                    _ => false,
                }
            };

            if used_json_schema {
                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    original_request.tool_choice.as_ref(),
                    &original_request.model,
                    history_tool_calls_count,
                );
            } else if tool_parser_available {
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        &original_request.model,
                        original_request.tools.as_deref().unwrap_or(&[]),
                        history_tool_calls_count,
                    )
                    .await;
            }
        }

        // Step 3: Determine finish reason. A local stop-decoder match takes
        // precedence over the engine's reason (which is "length" when stop
        // strings are enforced gateway-side rather than by the backend).
        let finish_reason_str = if stopped {
            "stop"
        } else {
            complete.finish_reason()
        };

        // Override finish reason if we have tool calls
        let final_finish_reason_str = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason_str
        };

        // When the local decoder matched a stop string, surface it (the engine
        // reports no stop_reason over the ZMQ path); otherwise use the engine's.
        let matched_stop = stop_decoder
            .matched_stop()
            .map(|s| serde_json::Value::String(s.to_string()))
            .or_else(|| complete.matched_stop_json());

        // Step 4: Convert output logprobs if present
        let logprobs = complete.output_logprobs().map(|ref proto_logprobs| {
            utils::convert_proto_to_openai_logprobs(proto_logprobs, tokenizer)
        });

        // Step 5: Build ChatCompletionMessage (proper response message type)
        let chat_message = ChatCompletionMessage {
            role: "assistant".to_string(),
            // Whitespace-only residual (e.g. "\n\n" between </think> and <tool_call>)
            // must be None, not Some("\n\n") — see normalize_assistant_content.
            content: normalize_assistant_content(processed_text),
            tool_calls,
            reasoning_content: reasoning_text,
        };

        // Step 6: Build ChatChoice
        Ok(ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        })
    }

    /// Process non-streaming chat response (collects all responses and builds final response)
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
    ) -> Result<ChatCompletionResponse, axum::response::Response> {
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        let history_tool_calls_count = utils::get_history_tool_calls_count(&chat_request);

        // Check parser availability once upfront (not per choice)
        let reasoning_parser_available = chat_request.separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                &chat_request.model,
            );

        let tool_choice_enabled = !matches!(
            &chat_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        let tool_parser_available = tool_choice_enabled
            && chat_request.tools.is_some()
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_deref(),
                &chat_request.model,
            );

        // Log once per request (not per choice)
        if chat_request.separate_reasoning && !reasoning_parser_available {
            tracing::debug!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                chat_request.model
            );
        }

        if chat_request.tools.is_some() && tool_choice_enabled && !tool_parser_available {
            tracing::debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                chat_request.model
            );
        }

        // Process all choices
        let mut choices = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            match self
                .process_single_choice(
                    complete,
                    index,
                    &chat_request,
                    &tokenizer,
                    stop_decoder,
                    history_tool_calls_count,
                    reasoning_parser_available,
                    tool_parser_available,
                )
                .await
            {
                Ok(choice) => choices.push(choice),
                Err(e) => {
                    return Err(error::internal_error(
                        "process_choice_failed",
                        format!("Failed to process choice {index}: {e}"),
                    ));
                }
            }
        }

        // Build usage from gRPC response counters.
        let usage = response_formatting::build_usage(&all_responses);

        // Build final ChatCompletionResponse
        Ok(
            ChatCompletionResponse::builder(&dispatch.request_id, &dispatch.model)
                .created(dispatch.created)
                .choices(choices)
                .usage(usage)
                .maybe_system_fingerprint(dispatch.weight_version.clone())
                .build(),
        )
    }

    /// Parse tool calls using model-specific parser
    pub async fn parse_tool_calls(
        &self,
        processed_text: &str,
        model: &str,
        tools: &[Tool],
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = utils::get_tool_parser(
            &self.tool_parser_factory,
            self.configured_tool_parser.as_deref(),
            model,
        );

        // Try parsing directly (parser will handle detection internally). Pass the
        // tool schemas so schema-aware parsers coerce argument types by their
        // declared type instead of guessing from the raw text.
        let result = {
            let parser = pooled_parser.lock().await;
            parser
                .parse_complete_with_tools(processed_text, tools)
                .await
            // Lock is dropped here
        };

        match result {
            Ok((normal_text, parsed_tool_calls)) => {
                if parsed_tool_calls.is_empty() {
                    return (None, normal_text);
                }

                let spec_tool_calls = parsed_tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, tc)| {
                        // Generate ID for this tool call
                        let id = utils::generate_tool_call_id(
                            model,
                            &tc.function.name,
                            index,
                            history_tool_calls_count,
                        );
                        ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: tc.function.name,
                                arguments: Some(tc.function.arguments),
                            },
                        }
                    })
                    .collect();
                (Some(spec_tool_calls), normal_text)
            }
            Err(e) => {
                error!("Tool call parsing error: {}", e);
                (None, processed_text.to_string())
            }
        }
    }

    /// Process non-streaming generate response (collects all responses and builds final response array)
    pub async fn process_non_streaming_generate_response(
        &self,
        execution_result: ExecutionResult,
        _generate_request: Arc<GenerateRequest>,
        dispatch: DispatchMetadata,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
        start_time: Instant,
    ) -> Result<Vec<GenerateResponse>, axum::response::Response> {
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        // Process each completion
        let mut result_array = Vec::new();
        for complete in all_responses {
            stop_decoder.reset();

            // Process tokens through stop decoder
            let outputs = match stop_decoder.process_tokens(complete.output_ids()) {
                Ok(outputs) => outputs,
                Err(e) => {
                    return Err(error::internal_error(
                        "process_tokens_failed",
                        format!("Failed to process tokens: {e}"),
                    ))
                }
            };

            // Accumulate text with early breaks
            let mut decoded_text = String::new();
            for output in outputs {
                match output {
                    SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                    SequenceDecoderOutput::StoppedWithText(t) => {
                        decoded_text.push_str(&t);
                        break;
                    }
                    SequenceDecoderOutput::Stopped => break,
                    SequenceDecoderOutput::Held => {}
                }
            }

            // Flush remaining text
            if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
                decoded_text.push_str(&t);
            }

            let output_ids = complete.output_ids().to_vec();
            let finish_reason_str = complete.finish_reason();

            // Parse finish_reason from string to proper type
            let finish_reason =
                utils::parse_finish_reason(finish_reason_str, complete.completion_tokens());

            let matched_stop = complete.matched_stop_json();

            // Extract logprobs if requested (convert proto types to Generate format)
            let input_token_logprobs = if request_logprobs {
                complete
                    .input_logprobs()
                    .as_ref()
                    .map(utils::convert_generate_input_logprobs)
            } else {
                None
            };

            let output_token_logprobs = if request_logprobs {
                complete
                    .output_logprobs()
                    .as_ref()
                    .map(utils::convert_generate_output_logprobs)
            } else {
                None
            };

            // Build GenerateResponse struct
            let meta_info = GenerateMetaInfo {
                id: dispatch.request_id.clone(),
                finish_reason,
                prompt_tokens: complete.prompt_tokens(),
                weight_version: dispatch
                    .weight_version
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                input_token_logprobs,
                output_token_logprobs,
                completion_tokens: complete.completion_tokens(),
                cached_tokens: complete.cached_tokens(),
                reasoning_tokens: Some(complete.reasoning_tokens()),
                e2e_latency: start_time.elapsed().as_secs_f64(),
                matched_stop,
            };

            result_array.push(GenerateResponse {
                text: decoded_text,
                output_ids,
                meta_info,
            });
        }

        Ok(result_array)
    }

    /// Process non-streaming Messages API response
    ///
    /// Collects the single response (Messages always has n=1), decodes tokens,
    /// parses reasoning/tool calls, and builds an Anthropic `Message` response.
    pub async fn process_non_streaming_messages_response(
        &self,
        execution_result: ExecutionResult,
        messages_request: Arc<CreateMessageRequest>,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<Message, axum::response::Response> {
        // Collect all responses (no logprobs for Messages API)
        let all_responses = response_collection::collect_responses(execution_result, false).await?;

        // Messages always has n=1 — enforce the invariant
        if all_responses.is_empty() {
            error!(
                function = "process_non_streaming_messages_response",
                "No responses received"
            );
            return Err(error::internal_error(
                "no_responses",
                "No responses received from backend",
            ));
        }
        if all_responses.len() > 1 {
            error!(
                function = "process_non_streaming_messages_response",
                response_count = all_responses.len(),
                "Messages API expected exactly one response"
            );
            return Err(error::internal_error(
                "unexpected_response_count",
                format!(
                    "Messages API received {} responses, expected exactly one",
                    all_responses.len()
                ),
            ));
        }
        #[expect(clippy::unwrap_used, reason = "safe: checked len == 1 above")]
        let complete = all_responses.into_iter().next().unwrap();

        // Check parser availability. Run parser when thinking is effectively ON —
        // explicitly enabled, or left to a template default of on (DeepSeek V4) —
        // or when the selected parser needs structural special tokens (e.g. Inkling).
        let reasoning_requires_special_tokens = utils::reasoning_parser_requires_special_tokens(
            &self.reasoning_parser_factory,
            self.configured_reasoning_parser.as_deref(),
            &messages_request.model,
        );
        let user_thinking = match &messages_request.thinking {
            Some(
                messages::ThinkingConfig::Enabled { .. }
                | messages::ThinkingConfig::Adaptive { .. },
            ) => Some(true),
            Some(messages::ThinkingConfig::Disabled) => Some(false),
            None => None,
        };
        let separate_reasoning = reasoning_requires_special_tokens
            || user_thinking == Some(true)
            || utils::should_mark_reasoning_started(user_thinking, tokenizer.as_ref());
        let reasoning_parser_available = separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                &messages_request.model,
            );

        let tool_choice_enabled = !matches!(
            &messages_request.tool_choice,
            Some(messages::ToolChoice::None)
        );

        let tool_parser_available = tool_choice_enabled
            && messages_request.tools.is_some()
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_deref(),
                &messages_request.model,
            );

        if separate_reasoning && !reasoning_parser_available {
            tracing::debug!(
                "No reasoning parser found for model '{}', reasoning content will not be separated",
                messages_request.model
            );
        }

        if messages_request.tools.is_some() && tool_choice_enabled && !tool_parser_available {
            tracing::debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                messages_request.model
            );
        }

        // Decode tokens through stop decoder
        stop_decoder.reset();
        let outputs = stop_decoder
            .process_tokens(complete.output_ids())
            .map_err(|e| {
                error!(function = "process_non_streaming_messages_response", error = %e, "Failed to process tokens");
                error::internal_error(
                    "process_tokens_failed",
                    format!("Failed to process tokens: {e}"),
                )
            })?;

        let mut final_text = String::new();
        let mut stopped = false;
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    stopped = true;
                    break;
                }
                SequenceDecoderOutput::Stopped => {
                    stopped = true;
                    break;
                }
                SequenceDecoderOutput::Held => {}
            }
        }
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            final_text.push_str(&t);
        }

        // Step 1: Parse reasoning content
        let mut reasoning_text: Option<String> = None;
        let mut processed_text = final_text;

        if reasoning_parser_available {
            // Fresh parser per request: non-streaming extraction keeps no state
            // across requests, so avoid serializing on the shared pooled mutex.
            if let Some(mut parser) = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                &messages_request.model,
            ) {
                // If thinking is effectively ON and template has a toggle, start in reasoning mode.
                if utils::should_mark_reasoning_started(user_thinking, tokenizer.as_ref()) {
                    parser.mark_reasoning_started();
                }

                match parser.detect_and_parse_reasoning(&processed_text) {
                    Ok(result) => {
                        if !result.reasoning_text.is_empty() {
                            reasoning_text = Some(result.reasoning_text);
                        }
                        processed_text = result.normal_text;
                    }
                    Err(e) => {
                        warn!("Reasoning parsing error, skipping parsing: {e}");
                    }
                }
            }
        }

        // Step 2: Parse tool calls
        let mut tool_calls: Option<Vec<ToolCall>> = None;

        if tool_choice_enabled && messages_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific tool or any/required mode)
            let has_structural_tag = self
                .tool_parser_factory
                .registry()
                .has_structural_tag_for_parser(self.configured_tool_parser.as_deref());
            let used_json_schema = !has_structural_tag
                && matches!(
                    &messages_request.tool_choice,
                    Some(messages::ToolChoice::Tool { .. } | messages::ToolChoice::Any { .. })
                );

            if used_json_schema {
                // Bridge Messages ToolChoice to Chat ToolChoice for reuse
                let chat_tool_choice = messages_request
                    .tool_choice
                    .as_ref()
                    .map(utils::message_utils::convert_message_tool_choice);

                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    chat_tool_choice.as_ref(),
                    &messages_request.model,
                    utils::message_utils::get_history_tool_calls_count_messages(&messages_request),
                );
            } else if tool_parser_available {
                let chat_tools = messages_request
                    .tools
                    .as_deref()
                    .map(utils::message_utils::extract_chat_tools)
                    .unwrap_or_default();
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        &messages_request.model,
                        &chat_tools,
                        utils::message_utils::get_history_tool_calls_count_messages(
                            &messages_request,
                        ),
                    )
                    .await;
            }
        }

        // Step 3: Build content blocks
        let mut content_blocks: Vec<messages::ContentBlock> = Vec::new();

        // Thinking block first (if present)
        if let Some(thinking) = reasoning_text {
            content_blocks.push(messages::ContentBlock::Thinking {
                thinking,
                signature: String::new(),
            });
        }

        // Text block (only if non-whitespace; a bare "\n\n" residual must not become one).
        if !processed_text.trim().is_empty() {
            content_blocks.push(messages::ContentBlock::Text {
                text: processed_text,
                citations: None,
            });
        }

        // Tool use blocks (convert from OpenAI ToolCall format)
        if let Some(calls) = &tool_calls {
            for tc in calls {
                let input = if let Some(args) = tc.function.arguments.as_deref() {
                    serde_json::from_str(args).unwrap_or_else(|e| {
                        warn!(
                            function = "process_non_streaming_messages_response",
                            tool_call_id = %tc.id,
                            error = %e,
                            "Failed to parse tool call arguments, defaulting to empty object"
                        );
                        serde_json::Value::Object(serde_json::Map::new())
                    })
                } else {
                    serde_json::Value::Object(serde_json::Map::new())
                };

                content_blocks.push(messages::ContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    input,
                });
            }
        }

        // Step 4: Determine stop_reason and stop_sequence (derived from same conditions).
        // A local stop-decoder match takes precedence over the engine's reason
        // (the backend has no stop-string detection over ZMQ), surfacing the
        // matched sequence for a StopSequence result.
        let finish_reason_str = if stopped {
            "stop"
        } else {
            complete.finish_reason()
        };
        let stop_sequence = stop_decoder.matched_stop().map(String::from).or_else(|| {
            complete
                .matched_stop_json()
                .and_then(|v| v.as_str().map(String::from))
        });

        let stop_reason = if tool_calls.is_some() || finish_reason_str == "tool_calls" {
            Some(messages::StopReason::ToolUse)
        } else if stop_sequence.is_some() {
            Some(messages::StopReason::StopSequence)
        } else if finish_reason_str == "length" {
            Some(messages::StopReason::MaxTokens)
        } else {
            Some(messages::StopReason::EndTurn)
        };

        // Clear stop_sequence when stop_reason is not StopSequence
        let stop_sequence = if matches!(stop_reason, Some(messages::StopReason::StopSequence)) {
            stop_sequence
        } else {
            None
        };

        // Step 5: Build usage
        let usage = messages::Usage {
            input_tokens: complete.prompt_tokens(),
            output_tokens: complete.completion_tokens(),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cache_creation: None,
            server_tool_use: None,
            service_tier: None,
        };

        // Step 6: Build Message
        Ok(Message {
            id: dispatch.request_id,
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: content_blocks,
            model: dispatch.model,
            stop_reason,
            stop_sequence,
            usage,
        })
    }

    /// Process non-streaming completion response
    ///
    /// Collects all responses (supports n>1 and batched prompts), decodes tokens
    /// through the stop decoder, applies `echo` and `suffix`, and builds one
    /// `CompletionResponse` with prompt-major global choice indices
    /// (`prompt_index * n + choice_index`).
    pub async fn process_non_streaming_completion_response(
        &self,
        execution_result: ExecutionResult,
        completion_req: Arc<CompletionRequest>,
        dispatch: DispatchMetadata,
        _tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<CompletionResponse, axum::response::Response> {
        let request_logprobs = completion_req.logprobs.is_some();
        let per_prompt_results = match execution_result {
            ExecutionResult::Batch { results } => results,
            other => vec![other],
        };
        let choices_per_prompt = completion_req.n.unwrap_or(1).max(1);
        let prompt_texts: Vec<&str> = match &completion_req.prompt {
            StringOrArray::String(text) => vec![text.as_str()],
            StringOrArray::Array(texts) => texts.iter().map(String::as_str).collect(),
        };

        // Drain all sub-streams concurrently; decoding below stays sequential
        // (shared stop decoder).
        let collected = try_join_all(
            per_prompt_results
                .into_iter()
                .map(|result| response_collection::collect_responses(result, request_logprobs)),
        )
        .await?;

        let mut total_prompt = 0u32;
        let mut total_completion = 0u32;
        let mut choices = Vec::new();

        for (prompt_index, all_responses) in collected.into_iter().enumerate() {
            let prompt_text = prompt_texts.get(prompt_index).copied().unwrap_or_default();
            let index_offset = prompt_index as u32 * choices_per_prompt;
            // n>1 choices share one prompt: max within a prompt, summed across prompts.
            let mut prompt_tokens = 0u32;

            // Arrival order, not `complete.index()`: SGLang non-streaming
            // Complete frames carry index 0 for every choice.
            for (i, complete) in all_responses.into_iter().enumerate() {
                stop_decoder.reset();

                let outputs = match stop_decoder.process_tokens(complete.output_ids()) {
                    Ok(outputs) => outputs,
                    Err(e) => {
                        return Err(error::internal_error(
                            "process_tokens_failed",
                            format!("Failed to process tokens: {e}"),
                        ))
                    }
                };

                let mut decoded_text = String::new();
                let mut stopped = false;
                for output in outputs {
                    match output {
                        SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                        SequenceDecoderOutput::StoppedWithText(t) => {
                            decoded_text.push_str(&t);
                            stopped = true;
                            break;
                        }
                        SequenceDecoderOutput::Stopped => {
                            stopped = true;
                            break;
                        }
                        SequenceDecoderOutput::Held => {}
                    }
                }

                if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
                    decoded_text.push_str(&t);
                }

                prompt_tokens = prompt_tokens.max(complete.prompt_tokens());
                total_completion += complete.completion_tokens();

                // A local stop-decoder match takes precedence over the engine's
                // reason (which is "length" when stop strings are enforced
                // gateway-side rather than by the backend).
                let finish_reason = if stopped {
                    Some("stop".to_string())
                } else {
                    let reason = complete.finish_reason();
                    if reason.is_empty() {
                        None
                    } else if reason == "stop" || reason == "length" {
                        Some(reason.to_string())
                    } else if let Ok(json) = serde_json::from_str::<serde_json::Value>(reason) {
                        json.get("type").and_then(|v| v.as_str()).map(|s| match s {
                            "length" => "length".to_string(),
                            "stop" => "stop".to_string(),
                            other => other.to_string(),
                        })
                    } else {
                        Some(reason.to_string())
                    }
                };

                // When the local decoder matched a stop string, surface it (the
                // engine reports no stop_reason over the ZMQ path); otherwise use
                // the engine's.
                let matched_stop = stop_decoder
                    .matched_stop()
                    .map(|s| serde_json::Value::String(s.to_string()))
                    .or_else(|| complete.matched_stop_json());

                let suffix_len = completion_req.suffix.as_ref().map_or(0, |s| s.len());
                let echo_len = if completion_req.echo {
                    prompt_text.len()
                } else {
                    0
                };
                let mut text = String::with_capacity(echo_len + decoded_text.len() + suffix_len);
                if completion_req.echo {
                    text.push_str(prompt_text);
                }
                text.push_str(&decoded_text);
                if let Some(ref sfx) = completion_req.suffix {
                    text.push_str(sfx);
                }

                choices.push(CompletionChoice {
                    text,
                    index: index_offset + i as u32,
                    logprobs: None, // TODO: wire legacy LogProbs from backend token_logprobs
                    finish_reason: finish_reason.or_else(|| Some("stop".to_string())),
                    matched_stop,
                });
            }

            total_prompt += prompt_tokens;
        }

        Ok(CompletionResponse {
            id: dispatch.request_id.clone(),
            object: "text_completion".to_string(),
            created: dispatch.created,
            model: dispatch.model.clone(),
            choices,
            usage: Some(Usage::from_counts(total_prompt, total_completion)),
            system_fingerprint: dispatch.weight_version.clone(),
        })
    }
}

/// Residual assistant text → OpenAI `content`. Whitespace-only (the `"\n\n"` left
/// after reasoning + tool-call extraction) becomes `None`, not `Some("\n\n")`, which
/// would otherwise diverge multi-turn conversations. Real content is kept verbatim.
fn normalize_assistant_content(text: String) -> Option<String> {
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod content_normalization_tests {
    use super::normalize_assistant_content;

    #[test]
    fn whitespace_only_is_none_real_text_kept_verbatim() {
        assert_eq!(normalize_assistant_content("\n\n".to_string()), None);
        assert_eq!(normalize_assistant_content("  \t".to_string()), None);
        assert_eq!(
            normalize_assistant_content("\n\nDone.".to_string()),
            Some("\n\nDone.".to_string())
        );
    }
}
