//! Harmony response processor for non-streaming responses

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::{
    chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
    common::{ChatLogProbs, ToolCall, Usage},
    responses::{
        InputTokensDetails, OutputTokensDetails, ResponseContentPart, ResponseOutputItem,
        ResponseReasoningContent, ResponseStatus, ResponseUsage, ResponsesRequest,
        ResponsesResponse, ResponsesUsage,
    },
};
use tracing::error;

use super::{
    builder::{convert_harmony_logprobs, try_harmony_encoding},
    stop::TextStopScanner,
    HarmonyParserAdapter,
};
use crate::routers::{
    error,
    grpc::{
        common::{response_collection, response_formatting},
        context::{DispatchMetadata, ExecutionResult},
    },
};

/// Processor for non-streaming Harmony responses
///
/// Collects all output tokens from execution and parses them using
/// HarmonyParserAdapter to extract the complete response.
pub(crate) struct HarmonyResponseProcessor;

impl HarmonyResponseProcessor {
    /// Create a new Harmony response processor
    pub fn new() -> Self {
        Self
    }

    /// Process a non-streaming Harmony chat response
    ///
    /// `router_stop_strings` is non-empty only when the router must enforce
    /// string `stop` sequences itself (direct-ZMQ backends: the engine sees
    /// token ids only).
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        router_stop_strings: &[String],
    ) -> Result<ChatCompletionResponse, Response> {
        let request_logprobs = chat_request.logprobs;

        // Collect all completed responses (one per choice)
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;
        if all_responses.is_empty() {
            return Err(error::internal_error(
                "no_responses_from_server",
                "No responses from server",
            ));
        }

        // Build choices by parsing output with HarmonyParserAdapter
        let mut choices: Vec<ChatChoice> = Vec::new();
        let mut total_reasoning_tokens = 0u32;

        for (index, complete) in all_responses.iter().enumerate() {
            let matched_stop = complete.matched_stop_json();

            // Parse Harmony channels with HarmonyParserAdapter
            let mut parser = HarmonyParserAdapter::new().map_err(|e| {
                error!(
                    function = "process_non_streaming_chat_response",
                    error = %e,
                    "Failed to create Harmony parser"
                );
                error::internal_error(
                    "create_harmony_parser_failed",
                    format!("Failed to create Harmony parser: {e}"),
                )
            })?;

            // Parse Harmony channels with finish_reason
            let parsed = parser
                .parse_complete(complete.output_ids(), complete.finish_reason().to_string())
                .map_err(|e| {
                    error!(
                        function = "process_non_streaming_chat_response",
                        error = %e,
                        "Harmony parsing failed on complete response"
                    );
                    error::internal_error(
                        "harmony_parsing_failed",
                        format!("Harmony parsing failed: {e}"),
                    )
                })?;

            // Convert output logprobs if present
            let logprobs: Option<ChatLogProbs> = if request_logprobs {
                let encoding = try_harmony_encoding()
                    .map_err(|e| error::internal_error("harmony_encoding_unavailable", e))?;
                complete
                    .output_logprobs()
                    .map(|lp| convert_harmony_logprobs(encoding, &lp))
            } else {
                None
            };

            let mut analysis = parsed.analysis;
            let mut final_text = parsed.final_text;
            let mut tool_calls = parsed.commentary;
            let mut finish_reason = parsed.finish_reason;
            let mut matched_stop = matched_stop;

            // Router-enforced string stops: scan channel text in emission
            // order (analysis before final), truncate after the first match
            // (retaining the stop as suffix, like the token-forwarding gRPC
            // path), and drop everything generated past it.
            if !router_stop_strings.is_empty() {
                let mut scanner = TextStopScanner::new(router_stop_strings.to_vec());
                if let Some(text) = analysis.take() {
                    let (out, stopped) = scanner.scan_complete(&text);
                    analysis = (!out.is_empty()).then_some(out);
                    if stopped {
                        final_text.clear();
                        tool_calls = None;
                    }
                }
                if scanner.matched().is_none() && !final_text.is_empty() {
                    let (out, stopped) = scanner.scan_complete(&final_text);
                    final_text = out;
                    if stopped {
                        tool_calls = None;
                    }
                }
                if let Some(stop) = scanner.matched() {
                    finish_reason = "stop".to_string();
                    matched_stop = Some(serde_json::Value::String(stop.to_string()));
                }
            }

            // Build response message (assistant)
            let message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: (!final_text.is_empty()).then_some(final_text),
                tool_calls,
                reasoning_content: analysis,
            };

            // Accumulate reasoning tokens across all responses
            total_reasoning_tokens += parsed.reasoning_token_count;

            choices.push(ChatChoice {
                index: index as u32,
                message,
                logprobs,
                finish_reason: Some(finish_reason),
                matched_stop,
                hidden_states: None,
            });
        }

        // Build usage from proto fields
        let usage = response_formatting::build_usage(&all_responses)
            .with_reasoning_tokens(total_reasoning_tokens);

        // Final ChatCompletionResponse
        Ok(
            ChatCompletionResponse::builder(&dispatch.request_id, &chat_request.model)
                .created(dispatch.created)
                .choices(choices)
                .usage(usage)
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build(),
        )
    }
}

impl Default for HarmonyResponseProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing a single Responses API iteration
///
/// Used by the MCP tool loop to determine whether to continue
/// executing tools or return the final response.
pub(crate) enum ResponsesIterationResult {
    /// Tool calls found in commentary channel - continue MCP loop
    ToolCallsFound {
        tool_calls: Vec<ToolCall>,
        analysis: Option<String>, // For streaming emission or reasoning output
        partial_text: String,     // For streaming emission or message output
        usage: Usage,             // Token usage from this iteration
        request_id: String,       // Request ID from dispatch
    },
    /// No tool calls - return final ResponsesResponse
    Completed {
        response: Box<ResponsesResponse>,
        usage: Usage,
    },
}

impl HarmonyResponseProcessor {
    /// Process a single Responses API iteration
    ///
    /// Parses Harmony channels and determines if tool calls are present.
    /// If tool calls found, returns ToolCallsFound for MCP loop to execute.
    /// If no tool calls, builds final ResponsesResponse.
    ///
    /// # Arguments
    ///
    /// * `execution_result` - The execution result from the model
    /// * `responses_request` - The original Responses API request
    /// * `dispatch` - Dispatch metadata for request tracking
    ///
    /// # Returns
    ///
    /// ResponsesIterationResult indicating whether to continue loop or return
    pub async fn process_responses_iteration(
        &self,
        execution_result: ExecutionResult,
        responses_request: Arc<ResponsesRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ResponsesIterationResult, Response> {
        let request_logprobs = responses_request.top_logprobs.is_some();

        // Collect all completed responses
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;
        if all_responses.is_empty() {
            return Err(error::internal_error(
                "no_responses_from_server",
                "No responses from server",
            ));
        }

        // For Responses API, we only process the first response (n=1)
        let complete = all_responses
            .first()
            .ok_or_else(|| error::internal_error("no_complete_response", "No complete response"))?;

        // Parse Harmony channels
        let mut parser = HarmonyParserAdapter::new().map_err(|e| {
            error!(
                function = "process_responses_iteration",
                error = %e,
                "Failed to create Harmony parser"
            );
            error::internal_error(
                "create_harmony_parser_failed",
                format!("Failed to create Harmony parser: {e}"),
            )
        })?;

        let parsed = parser
            .parse_complete(complete.output_ids(), complete.finish_reason().to_string())
            .map_err(|e| {
                error!(
                    function = "process_responses_iteration",
                    error = %e,
                    "Harmony parsing failed on complete response"
                );
                error::internal_error(
                    "harmony_parsing_failed",
                    format!("Harmony parsing failed: {e}"),
                )
            })?;

        // VALIDATION: Check if model incorrectly generated Tool role messages
        // This happens when the model copies the format of tool result messages
        // instead of continuing as assistant. This is a model hallucination bug.
        let messages = parser.get_messages();
        let tool_messages_generated = messages.iter().any(|msg| {
            msg.author.role == openai_harmony::chat::Role::Tool
                && msg.recipient.as_deref() == Some("assistant")
        });

        if tool_messages_generated {
            tracing::warn!(
                "Model generated Tool->Assistant message instead of Assistant message. \
                This is a model hallucination bug where it copies tool result format."
            );
        }

        // Build usage (needed for both ToolCallsFound and Completed)
        let usage = response_formatting::build_usage(std::slice::from_ref(complete))
            .with_reasoning_tokens(parsed.reasoning_token_count);

        // Check for tool calls in commentary channel
        if let Some(tool_calls) = parsed.commentary {
            // Tool calls found - return for MCP loop execution
            return Ok(ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis: parsed.analysis,
                partial_text: parsed.final_text,
                usage,
                request_id: dispatch.request_id.clone(),
            });
        }

        // No tool calls - build final ResponsesResponse
        let mut output: Vec<ResponseOutputItem> = Vec::new();

        // Map analysis channel → ResponseOutputItem::Reasoning
        if let Some(analysis) = parsed.analysis {
            let reasoning_item = ResponseOutputItem::new_reasoning(
                format!("reasoning_{}", dispatch.request_id),
                vec![],
                vec![ResponseReasoningContent::ReasoningText { text: analysis }],
                Some("completed".to_string()),
            );
            output.push(reasoning_item);
        }

        // TODO: Logprobs for Responses API is not fully supported yet.
        // The Chat Completions API logprobs work, but Responses API needs more testing.
        // For now, we disable logprobs in Responses API output.
        let logprobs: Option<ChatLogProbs> = None;
        let _ = request_logprobs; // silence unused variable warning

        // Map final channel → ResponseOutputItem::Message
        if !parsed.final_text.is_empty() {
            let message_item = ResponseOutputItem::Message {
                id: format!("msg_{}", dispatch.request_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: parsed.final_text,
                    annotations: vec![],
                    logprobs,
                }],
                status: "completed".to_string(),
                phase: None,
            };
            output.push(message_item);
        }

        // Build ResponsesResponse with all required fields
        let response = ResponsesResponse::builder(&dispatch.request_id, &responses_request.model)
            .copy_from_request(&responses_request)
            .created_at(dispatch.created as i64)
            .status(ResponseStatus::Completed)
            .output(output)
            .maybe_text(responses_request.text.clone())
            .usage(ResponsesUsage::Modern(ResponseUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                input_tokens_details: usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(InputTokensDetails::from)
                    .unwrap_or_default(),
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: usage
                        .completion_tokens_details
                        .as_ref()
                        .and_then(|d| d.reasoning_tokens)
                        .unwrap_or(0),
                },
            }))
            .build();

        Ok(ResponsesIterationResult::Completed {
            response: Box::new(response),
            usage,
        })
    }
}
