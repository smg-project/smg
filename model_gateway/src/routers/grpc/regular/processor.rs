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
    chat::{ChatChoice, ChatCompletionMessage, ChatCompletionResponse},
    common::{ChatLogProbs, FunctionCallResponse, Tool, ToolCall, ToolChoice, ToolChoiceValue},
    completion::{CompletionChoice, CompletionResponse},
    generate::{GenerateMetaInfo, GenerateResponse},
    messages::{self, Message},
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use symphony::adapt;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{error, warn};

use crate::routers::{
    error,
    grpc::{
        common::{response_collection, response_formatting},
        context::{DispatchMetadata, ExecutionResult},
        proto_wrapper::ProtoGenerateComplete,
        regular::symphony::{content_spans, engine_finish, finished, Assignment, Reader},
        spec::{ChatResponseSpec, CompletionResponseSpec, MessagesResponseSpec},
        utils,
    },
};

/// Unified response processor for both routers
#[derive(Clone)]
pub(crate) struct ResponseProcessor {
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    /// Per-request parser-name resolution (model-card override → configured).
    pub parser_resolver: utils::ParserResolver,
}

impl ResponseProcessor {
    pub fn new(
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        parser_resolver: utils::ParserResolver,
    ) -> Self {
        Self {
            tool_parser_factory,
            reasoning_parser_factory,
            parser_resolver,
        }
    }

    /// Process a single choice from GenerateComplete response
    ///
    /// Returns the choice and the number of output tokens the reasoning
    /// parser routed to `reasoning_content` (0 when it did not run).
    #[expect(clippy::too_many_arguments)]
    pub async fn process_single_choice(
        &self,
        request_id: &str,
        complete: &ProtoGenerateComplete,
        index: usize,
        original_request: &ChatResponseSpec,
        model: &str,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
        reasoning_parser_available: bool,
        tool_parser_available: bool,
        // Resolved once per request by the caller: keeps every choice of one
        // request on the same parser even if the worker registry changes
        // between availability check and parsing.
        reasoning_parser_name: Option<&str>,
        tool_parser_name: Option<&str>,
    ) -> Result<(ChatChoice, u32), String> {
        stop_decoder.reset();
        if let Some(assignment) = &original_request.symphony {
            return Self::read_with_symphony(
                request_id,
                assignment,
                complete,
                index,
                original_request,
                tokenizer,
                stop_decoder,
            );
        }
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(complete.output_ids())
            .map_err(|e| format!("Failed to process tokens: {e}"))?;

        // Accumulate text with early breaks, noting where each token's text
        // ends to place the reasoning span in tokens.
        let mut final_text = String::new();
        let mut stopped = false;
        let mut held = false;
        let mut token_ends = Vec::with_capacity(outputs.len());
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    stopped = true;
                }
                SequenceDecoderOutput::Stopped => stopped = true,
                SequenceDecoderOutput::Held => held = true,
            }
            token_ends.push(final_text.len());
            if stopped {
                break;
            }
        }

        // Complete token decoding, including bytes and a partial stop match.
        // The last token takes the tail so the token spans reach the text end.
        let tail = stop_decoder
            .finish()
            .map_err(|e| format!("Failed to finish token decoding: {e}"))?;
        utils::warn_detokenize_recoveries(request_id, index as u32, stop_decoder);
        if let SequenceDecoderOutput::Text(t) | SequenceDecoderOutput::StoppedWithText(t) = tail {
            final_text.push_str(&t);
            if let Some(last) = token_ends.last_mut() {
                *last = final_text.len();
            }
        }
        if held
            && tokenizer.uses_byte_fallback()
            && original_request.separate_reasoning
            && reasoning_parser_available
        {
            if let Some(ends) = eager_token_ends(
                tokenizer,
                &complete.output_ids()[..token_ends.len()],
                stop_decoder.skip_special_tokens(),
                &final_text,
            ) {
                token_ends = ends;
            }
        }
        stopped |= stop_decoder.is_stopped();

        // The decoded output, kept to place the content's tokens for its
        // logprobs; the content starts after the reasoning span.
        let output_logprobs = complete.output_logprobs();
        let decoded_text = output_logprobs.as_ref().map(|_| final_text.clone());
        let mut content_from = 0;

        // Step 1: Handle reasoning content parsing
        let mut reasoning_text: Option<String> = None;
        let mut reasoning_tokens = 0;
        let mut processed_text = final_text;

        if original_request.separate_reasoning && reasoning_parser_available {
            // Fresh parser per request: non-streaming extraction keeps no state
            // across requests, so avoid serializing on the shared pooled mutex.
            if let Some(mut parser) = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                reasoning_parser_name,
                model,
            ) {
                // Request building decided whether the rendered prompt ends
                // inside the model's thinking block; the parser then starts in
                // reasoning mode.
                if original_request.reasoning_in_prefill {
                    parser.mark_reasoning_started();
                }

                match parser.detect_and_parse_reasoning(&processed_text) {
                    Ok(result) => {
                        if !result.reasoning_text.is_empty() {
                            reasoning_tokens = reasoning_span_tokens(
                                &token_ends,
                                &processed_text,
                                &result.reasoning_text,
                                &result.normal_text,
                            );
                            content_from = reasoning_span(
                                &processed_text,
                                &result.reasoning_text,
                                &result.normal_text,
                            )
                            .1;
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
            let native_tool_format = self
                .tool_parser_factory
                .registry()
                .uses_native_tool_format(tool_parser_name);
            let used_json_schema = if native_tool_format {
                false
            } else {
                original_request
                    .tool_choice
                    .as_ref()
                    .is_some_and(ToolChoice::forces_call)
            };

            if used_json_schema {
                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    original_request.tool_choice.as_ref(),
                    model,
                    history_tool_calls_count,
                );
            } else if tool_parser_available {
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        model,
                        tool_parser_name,
                        original_request.tools.as_deref().unwrap_or(&[]),
                        history_tool_calls_count,
                    )
                    .await;
            }
        }

        if original_request.first_tool_call_only {
            utils::keep_first_tool_call(&mut tool_calls);
        }

        // Step 3: Determine finish reason. A local stop-decoder match takes
        // precedence over the engine's reason (which is "length" when stop
        // strings are enforced gateway-side rather than by the backend).
        let finish_reason_str = if stopped {
            "stop"
        } else {
            complete.finish_reason()
        };

        // Parsed calls do not override an engine truncation or failure.
        let final_finish_reason_str = if tool_calls.is_some()
            && !matches!(finish_reason_str, "length" | "failed" | "error")
        {
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

        // Step 4: Build ChatCompletionMessage (proper response message type)
        let chat_message = ChatCompletionMessage {
            role: "assistant".to_string(),
            // Whitespace-only residual (e.g. "\n\n" between </think> and <tool_call>)
            // must be None, not Some("\n\n") — see normalize_assistant_content.
            content: normalize_assistant_content(processed_text),
            refusal: None,
            annotations: Vec::new(),
            tool_calls,
            reasoning_content: reasoning_text,
        };

        // Step 5: Convert output logprobs if present. They cover the tokens of
        // `message.content` only, as a stream's content deltas do: not the
        // reasoning, the tool calls, the template's markers, or the tokens
        // a stop sequence or the EOS token cut off.
        let logprobs = output_logprobs.map(|ref proto_logprobs| {
            let mut logprobs = utils::convert_proto_to_openai_logprobs(
                proto_logprobs,
                tokenizer,
                original_request.top_logprobs,
            );
            if let (
                ChatLogProbs::Detailed {
                    content: Some(items),
                    ..
                },
                Some(text),
            ) = (&mut logprobs, &decoded_text)
            {
                let ranges = content_ranges(
                    text,
                    content_from,
                    chat_message.content.as_deref().unwrap_or(""),
                );
                let keep = content_tokens(&token_ends, &ranges);
                let mut position = 0;
                items.retain(|_| {
                    position += 1;
                    keep.get(position - 1).copied().unwrap_or(false)
                });
            }
            logprobs
        });

        // Step 6: Build ChatChoice
        let choice = ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        };
        Ok((choice, reasoning_tokens))
    }

    /// The choice as Symphony reads it (see `regular::symphony`): the output as
    /// one delta through the stop decoder, the decoder's last text once decoding
    /// is finished, the engine's finish, and the message the events fold into.
    /// The matched stop is set as on the older path, and the logprobs cover the
    /// tokens of `message.content` only, as the older path's do: the content's
    /// bytes come from the events, each token's from the reader.
    fn read_with_symphony(
        request_id: &str,
        assignment: &Assignment,
        complete: &ProtoGenerateComplete,
        index: usize,
        original_request: &ChatResponseSpec,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<(ChatChoice, u32), String> {
        let tools = original_request.tools.as_deref().unwrap_or(&[]);
        let mut reader =
            Reader::new(assignment, tools)?.first_call_only(original_request.first_tool_call_only);
        let (mut events, stopped) = reader.read(stop_decoder, complete.output_ids())?;
        // Complete token decoding, including bytes and a partial stop match, as
        // on the older path; the text it releases is the last token's.
        let tail = stop_decoder
            .finish()
            .map_err(|e| format!("Failed to finish token decoding: {e}"))?;
        utils::warn_detokenize_recoveries(request_id, index as u32, stop_decoder);
        if let SequenceDecoderOutput::Text(text) | SequenceDecoderOutput::StoppedWithText(text) =
            tail
        {
            events.extend(reader.tail(&text)?);
        }
        // A local stop-decoder match takes precedence over the engine's reason,
        // as on the older path.
        let reason = if stopped || stop_decoder.is_stopped() {
            "stop"
        } else {
            complete.finish_reason()
        };
        events.extend(reader.end(engine_finish(reason))?);
        let reasoning_tokens = finished(&events).map_or(0, |finished| finished.reasoning_tokens);
        let mut choice = adapt::chat::message(index as u32, &events);
        choice.matched_stop = stop_decoder
            .matched_stop()
            .map(|s| serde_json::Value::String(s.to_string()))
            .or_else(|| complete.matched_stop_json());
        // The content's tokens alone keep their logprobs, as on the older path: none when
        // the message has no content (whitespace alone, which `adapt::chat::message` leaves
        // out of the message).
        let has_content = choice.message.content.is_some();
        choice.logprobs = complete.output_logprobs().map(|ref proto_logprobs| {
            let mut logprobs = utils::convert_proto_to_openai_logprobs(
                proto_logprobs,
                tokenizer,
                original_request.top_logprobs,
            );
            if let ChatLogProbs::Detailed {
                content: Some(items),
                ..
            } = &mut logprobs
            {
                let keep = if has_content {
                    content_tokens(reader.token_ends(), &content_spans(&events))
                } else {
                    Vec::new()
                };
                let mut position = 0;
                items.retain(|_| {
                    position += 1;
                    keep.get(position - 1).copied().unwrap_or(false)
                });
            }
            logprobs
        });
        Ok((choice, reasoning_tokens))
    }

    /// The message as Symphony reads it (see `regular::symphony`): the output as one delta
    /// through the stop decoder, the decoder's last text once decoding is finished, the engine's
    /// finish, and the content blocks the events fold into (`adapt::messages::output`). A stop
    /// sequence the decoder or the engine matched is the stop reason, as on the older path; the
    /// thinking tokens are the engine's, or the reasoning text's, when the request asked for
    /// thinking.
    fn read_messages_with_symphony(
        dispatch: &DispatchMetadata,
        assignment: &Assignment,
        complete: &ProtoGenerateComplete,
        request: &MessagesResponseSpec,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<Message, String> {
        let mut reader = Reader::new(assignment, &request.chat_tools)?
            .first_call_only(request.first_tool_call_only);
        stop_decoder.reset();
        let (mut events, stopped) = reader.read(stop_decoder, complete.output_ids())?;
        let tail = stop_decoder
            .finish()
            .map_err(|e| format!("Failed to finish token decoding: {e}"))?;
        utils::warn_detokenize_recoveries(&dispatch.request_id, 0, stop_decoder);
        if let SequenceDecoderOutput::Text(text) | SequenceDecoderOutput::StoppedWithText(text) =
            tail
        {
            events.extend(reader.tail(&text)?);
        }
        let reason = if stopped || stop_decoder.is_stopped() {
            "stop"
        } else {
            complete.finish_reason()
        };
        events.extend(reader.end(engine_finish(reason))?);
        let output = adapt::messages::output(&events);
        let stop_sequence = stop_decoder.matched_stop().map(String::from).or_else(|| {
            complete
                .matched_stop_json()
                .and_then(|v| v.as_str().map(String::from))
        });
        let stop_reason = match output.stop_reason {
            Some(messages::StopReason::EndTurn) if stop_sequence.is_some() => {
                Some(messages::StopReason::StopSequence)
            }
            stop_reason => stop_reason,
        };
        let stop_sequence = stop_sequence
            .filter(|_| matches!(stop_reason, Some(messages::StopReason::StopSequence)));
        let thinking_requested = request.thinking_requested;
        let reasoning_tokens = finished(&events).map_or(0, |finished| finished.reasoning_tokens);
        Ok(Message {
            id: dispatch.request_id.clone(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: output.content,
            model: dispatch.model.clone(),
            stop_reason,
            stop_sequence,
            stop_details: None,
            usage: messages_usage_from_counts(
                complete.prompt_tokens(),
                complete.completion_tokens(),
                thinking_requested.then_some(complete.reasoning_tokens().max(reasoning_tokens)),
            ),
            container: None,
        })
    }

    /// Process non-streaming chat response (collects all responses and builds final response)
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: ChatResponseSpec,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
    ) -> Result<ChatCompletionResponse, axum::response::Response> {
        let model = &dispatch.model;
        let reasoning_parser_name = self.parser_resolver.reasoning_parser(model);
        let tool_parser_name = self.parser_resolver.tool_parser(model);
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        let history_tool_calls_count = chat_request.history_tool_calls_count;

        // Check parser availability once upfront (not per choice)
        let reasoning_parser_available = chat_request.separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                reasoning_parser_name.as_deref(),
                model,
            );

        let tool_choice_enabled = !matches!(
            &chat_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        let tool_parser_available = tool_choice_enabled
            && chat_request.tools.is_some()
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                tool_parser_name.as_deref(),
                model,
            );

        // Log once per request (not per choice)
        if chat_request.separate_reasoning && !reasoning_parser_available {
            tracing::debug!(
                "No reasoning parser found for model '{model}', skipping reasoning parsing"
            );
        }

        if chat_request.tools.is_some() && tool_choice_enabled && !tool_parser_available {
            tracing::debug!("No tool parser found for model '{model}', skipping tool call parsing");
        }

        // Process all choices
        let mut choices = Vec::new();
        let mut parsed_reasoning_tokens = 0;
        for (index, complete) in all_responses.iter().enumerate() {
            match self
                .process_single_choice(
                    &dispatch.request_id,
                    complete,
                    index,
                    &chat_request,
                    model,
                    &tokenizer,
                    stop_decoder,
                    history_tool_calls_count,
                    reasoning_parser_available,
                    tool_parser_available,
                    reasoning_parser_name.as_deref(),
                    tool_parser_name.as_deref(),
                )
                .await
            {
                Ok((choice, reasoning_tokens)) => {
                    choices.push(choice);
                    parsed_reasoning_tokens += reasoning_tokens;
                }
                Err(e) => {
                    return Err(error::internal_error(
                        "process_choice_failed",
                        format!("Failed to process choice {index}: {e}"),
                    ));
                }
            }
        }

        // The usage from the gRPC response counters, built as the streamed
        // answer builds its usage chunk: with the reasoning tokens the
        // gateway's parser split off when it ran (zero included) and every
        // detail counter present as the OpenAI API sends it.
        let usage = response_formatting::chat_usage(
            response_formatting::build_usage(&all_responses),
            (reasoning_parser_available || chat_request.symphony.is_some())
                .then_some(parsed_reasoning_tokens),
            chat_request.unbilled_prompt_tokens,
        );

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
        // Resolved once per request by the caller (see process_single_choice).
        tool_parser_name: Option<&str>,
        tools: &[Tool],
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser =
            utils::get_tool_parser(&self.tool_parser_factory, tool_parser_name, model);

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
            let tail = stop_decoder.finish().map_err(|e| {
                error::internal_error(
                    "process_tokens_failed",
                    format!("Failed to finish token decoding: {e}"),
                )
            })?;
            utils::warn_detokenize_recoveries(&dispatch.request_id, 0, stop_decoder);
            // A stop the decoder found at any token (text streams as soon as its
            // characters are complete, so a string stop can close mid-run) or
            // in the tail is the response's finish reason, as on the other paths.
            let stopped_at_finish = stop_decoder.is_stopped();
            if let SequenceDecoderOutput::Text(t) | SequenceDecoderOutput::StoppedWithText(t) = tail
            {
                decoded_text.push_str(&t);
            }

            let output_ids = complete.output_ids().to_vec();
            let finish_reason_str = complete.finish_reason();

            // Parse finish_reason from string to proper type
            let finish_reason = utils::parse_finish_reason(
                if stopped_at_finish {
                    "stop"
                } else {
                    finish_reason_str
                },
                complete.completion_tokens(),
            );

            // The decoder's string stop when it matched one (the engine does not
            // see string stops it did not apply), else the engine's value: a
            // stop token's id stays reported, as on the other paths.
            let matched_stop = stop_decoder
                .matched_stop()
                .map(|stop| serde_json::Value::String(stop.to_string()))
                .or_else(|| complete.matched_stop_json());

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
                weight_version: response_formatting::effective_weight_version(
                    complete.weight_version(),
                    dispatch.weight_version.as_deref(),
                )
                .to_string(),
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
        messages_request: MessagesResponseSpec,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<Message, axum::response::Response> {
        let model = &dispatch.model;
        let reasoning_parser_name = self.parser_resolver.reasoning_parser(model);
        let tool_parser_name = self.parser_resolver.tool_parser(model);
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
        if let Some(assignment) = &messages_request.symphony {
            return Self::read_messages_with_symphony(
                &dispatch,
                assignment,
                &complete,
                &messages_request,
                stop_decoder,
            )
            .map_err(|e| {
                error!(function = "process_non_streaming_messages_response", error = %e, "Symphony failed to read the message");
                error::internal_error("symphony_read_failed", e)
            });
        }

        // Check parser availability. Run parser when the user explicitly enabled thinking,
        // or when the selected parser needs structural special tokens (e.g. Inkling).
        let reasoning_requires_special_tokens = utils::reasoning_parser_requires_special_tokens(
            &self.reasoning_parser_factory,
            reasoning_parser_name.as_deref(),
            model,
        );
        let separate_reasoning =
            reasoning_requires_special_tokens || messages_request.thinking_requested;
        let reasoning_parser_available = separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                reasoning_parser_name.as_deref(),
                model,
            );

        let tool_choice_enabled = !matches!(
            &messages_request.tool_choice,
            Some(messages::ToolChoice::None)
        );

        let tool_parser_available = tool_choice_enabled
            && messages_request.has_tools
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                tool_parser_name.as_deref(),
                model,
            );

        if separate_reasoning && !reasoning_parser_available {
            tracing::debug!(
                "No reasoning parser found for model '{model}', reasoning content will not be separated"
            );
        }

        if messages_request.has_tools && tool_choice_enabled && !tool_parser_available {
            tracing::debug!("No tool parser found for model '{model}', skipping tool call parsing");
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

        // Accumulate text with early breaks, noting where each token's text
        // ends to place the thinking span in tokens (as the chat path does).
        let mut final_text = String::new();
        let mut stopped = false;
        let mut held = false;
        let mut token_ends = Vec::with_capacity(outputs.len());
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    stopped = true;
                }
                SequenceDecoderOutput::Stopped => stopped = true,
                SequenceDecoderOutput::Held => held = true,
            }
            token_ends.push(final_text.len());
            if stopped {
                break;
            }
        }
        let tail = stop_decoder.finish().map_err(|e| {
            error::internal_error(
                "process_tokens_failed",
                format!("Failed to finish token decoding: {e}"),
            )
        })?;
        utils::warn_detokenize_recoveries(&dispatch.request_id, 0, stop_decoder);
        if let SequenceDecoderOutput::Text(t) | SequenceDecoderOutput::StoppedWithText(t) = tail {
            final_text.push_str(&t);
            if let Some(last) = token_ends.last_mut() {
                *last = final_text.len();
            }
        }
        if held && tokenizer.uses_byte_fallback() && reasoning_parser_available {
            if let Some(ends) = eager_token_ends(
                &tokenizer,
                &complete.output_ids()[..token_ends.len()],
                stop_decoder.skip_special_tokens(),
                &final_text,
            ) {
                token_ends = ends;
            }
        }
        stopped |= stop_decoder.is_stopped();

        // Step 1: Parse reasoning content
        let mut reasoning_text: Option<String> = None;
        let mut thinking_tokens = 0;
        let mut processed_text = final_text;

        if reasoning_parser_available {
            // Fresh parser per request: non-streaming extraction keeps no state
            // across requests, so avoid serializing on the shared pooled mutex.
            if let Some(mut parser) = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                reasoning_parser_name.as_deref(),
                model,
            ) {
                if messages_request.reasoning_in_prefill {
                    parser.mark_reasoning_started();
                }

                match parser.detect_and_parse_reasoning(&processed_text) {
                    Ok(result) => {
                        if !result.reasoning_text.is_empty() {
                            thinking_tokens = reasoning_span_tokens(
                                &token_ends,
                                &processed_text,
                                &result.reasoning_text,
                                &result.normal_text,
                            );
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

        if tool_choice_enabled && messages_request.has_tools {
            // Check if JSON schema constraint was used (specific tool or any/required mode)
            let native_tool_format = self
                .tool_parser_factory
                .registry()
                .uses_native_tool_format(tool_parser_name.as_deref());
            let used_json_schema = !native_tool_format
                && matches!(
                    &messages_request.tool_choice,
                    Some(messages::ToolChoice::Tool { .. } | messages::ToolChoice::Any { .. })
                );

            if used_json_schema {
                // Bridge Messages ToolChoice to Chat ToolChoice for reuse
                let chat_tool_choice = messages_request.tool_choice.as_ref().map(ToolChoice::from);

                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    chat_tool_choice.as_ref(),
                    model,
                    messages_request.history_tool_calls_count,
                );
            } else if tool_parser_available {
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        model,
                        tool_parser_name.as_deref(),
                        &messages_request.chat_tools,
                        messages_request.history_tool_calls_count,
                    )
                    .await;
            }
        }

        if messages_request.first_tool_call_only {
            utils::keep_first_tool_call(&mut tool_calls);
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
                    id: utils::message_utils::anthropic_tool_use_id(&tc.id),
                    name: tc.function.name.clone(),
                    input,
                    caller: messages::ToolUseCaller::Direct,
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

        // Parsed calls do not override a truncation, as in Chat.
        let stop_reason = if finish_reason_str == "length" {
            Some(messages::StopReason::MaxTokens)
        } else if tool_calls.is_some() || finish_reason_str == "tool_calls" {
            Some(messages::StopReason::ToolUse)
        } else if stop_sequence.is_some() {
            Some(messages::StopReason::StopSequence)
        } else {
            Some(messages::StopReason::EndTurn)
        };

        // Clear stop_sequence when stop_reason is not StopSequence
        let stop_sequence = if matches!(stop_reason, Some(messages::StopReason::StopSequence)) {
            stop_sequence
        } else {
            None
        };

        // Step 5: Build usage. A turn that ran with thinking reports its
        // thinking tokens: the engine's own count when it keeps one, else the
        // tokens the parser placed in the thinking span.
        let thinking_requested = messages_request.thinking_requested;
        let usage = messages_usage_from_counts(
            complete.prompt_tokens(),
            complete.completion_tokens(),
            thinking_requested.then_some(complete.reasoning_tokens().max(thinking_tokens)),
        );

        // Step 6: Build Message
        Ok(Message {
            id: dispatch.request_id,
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: content_blocks,
            model: dispatch.model,
            stop_reason,
            stop_sequence,
            stop_details: None,
            usage,
            container: None,
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
        completion_req: CompletionResponseSpec,
        dispatch: DispatchMetadata,
        _tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<CompletionResponse, axum::response::Response> {
        let request_logprobs = completion_req.logprobs;
        let per_prompt_results = match execution_result {
            ExecutionResult::Batch { results } => results,
            other => vec![other],
        };
        let choices_per_prompt = completion_req.choices_per_prompt;
        let prompt_texts = &completion_req.prompt_texts;

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
        let mut total_cached = 0u32;
        let mut total_reasoning = 0u32;
        let mut total_spec_accepted = 0u32;
        let mut total_spec_drafted = 0u32;
        let mut choices = Vec::new();

        for (prompt_index, all_responses) in collected.into_iter().enumerate() {
            let prompt_text = prompt_texts
                .get(prompt_index)
                .map(String::as_str)
                .unwrap_or_default();
            let index_offset = prompt_index as u32 * choices_per_prompt;
            // n>1 choices share one prompt: max within a prompt, summed across prompts.
            let mut prompt_tokens = 0u32;
            let mut cached_tokens = 0u32;

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

                let tail = stop_decoder.finish().map_err(|e| {
                    error::internal_error(
                        "process_tokens_failed",
                        format!("Failed to finish token decoding: {e}"),
                    )
                })?;
                utils::warn_detokenize_recoveries(
                    &dispatch.request_id,
                    index_offset + i as u32,
                    stop_decoder,
                );
                if let SequenceDecoderOutput::Text(t) | SequenceDecoderOutput::StoppedWithText(t) =
                    tail
                {
                    decoded_text.push_str(&t);
                }
                stopped |= stop_decoder.is_stopped();

                prompt_tokens = prompt_tokens.max(complete.prompt_tokens());
                cached_tokens = cached_tokens.max(complete.cached_tokens());
                total_completion += complete.completion_tokens();
                total_reasoning += complete.reasoning_tokens();
                total_spec_accepted += complete.spec_accepted_tokens();
                total_spec_drafted += complete.spec_draft_tokens();

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
            total_cached += cached_tokens;
        }

        Ok(CompletionResponse {
            id: dispatch.request_id.clone(),
            object: "text_completion".to_string(),
            created: dispatch.created,
            model: dispatch.model.clone(),
            choices,
            // The usage object the streamed answer's usage chunk carries.
            usage: Some(
                response_formatting::TokenCounts {
                    prompt: total_prompt,
                    completion: total_completion,
                    cached: total_cached,
                    reasoning: total_reasoning,
                    spec_accepted: total_spec_accepted,
                    spec_drafted: total_spec_drafted,
                }
                .usage(),
            ),
            system_fingerprint: dispatch.weight_version.clone(),
        })
    }
}

// Byte buffering can release several tokens at once. Observe their original
// eager boundaries only for accounting; canonical text and stops stay with the
// stable decoder. A mismatched observation keeps the existing span estimate.
fn eager_token_ends(
    tokenizer: &Arc<dyn Tokenizer>,
    token_ids: &[u32],
    skip_special_tokens: bool,
    text: &str,
) -> Option<Vec<usize>> {
    let mut ids = Vec::new();
    let mut prefix = String::new();
    let mut prefix_index = 0;
    let mut observed = String::new();
    let mut ends = Vec::with_capacity(token_ids.len());
    for &token_id in token_ids {
        if let Some(delta) = tokenizer
            .decode_step_eager(
                token_id,
                &mut ids,
                &mut prefix,
                &mut prefix_index,
                skip_special_tokens,
            )
            .ok()?
        {
            observed.push_str(&delta);
        }
        ends.push(observed.len().min(text.len()));
    }
    observed.starts_with(text).then_some(ends)
}

/// Tokens of the output whose text overlaps the reasoning span, given the
/// cumulative text length after each token. The span is `reasoning` as found
/// in `text`, else (a delimiter inside it was removed) everything before the
/// normal text. Tokens that decoded to nothing count with the token that
/// released their text; the delimiters around the span are not reasoning,
/// like the engines' own counts.
fn reasoning_span_tokens(token_ends: &[usize], text: &str, reasoning: &str, normal: &str) -> u32 {
    let (start, end) = reasoning_span(text, reasoning, normal);
    let mut token_start = 0;
    let mut count = 0;
    for &token_end in token_ends {
        if token_in_span(token_start, token_end, start, end) {
            count += 1;
        }
        token_start = token_end;
    }
    count
}

/// The byte range of `reasoning` in `text`: where it is found, else
/// everything before the normal text.
fn reasoning_span(text: &str, reasoning: &str, normal: &str) -> (usize, usize) {
    match text.find(reasoning) {
        Some(start) => (start, start + reasoning.len()),
        None => (0, text.len().saturating_sub(normal.len())),
    }
}

/// Whether a token whose text is `text[token_start..token_end]` belongs to
/// the span `text[start..end]`. A zero-width token at the span's start held
/// the text that starts it.
fn token_in_span(token_start: usize, token_end: usize, start: usize, end: usize) -> bool {
    token_start < end && (token_end > start || token_start == start)
}

/// The byte ranges of `text`, from `from` on, that `content` was taken from:
/// the first place it appears whole, else its pieces in order, skipping the
/// text a parser cut out between them (a tool call, a marker).
fn content_ranges(text: &str, from: usize, content: &str) -> Vec<(usize, usize)> {
    if content.is_empty() {
        return Vec::new();
    }
    let from = if text.is_char_boundary(from) { from } else { 0 };
    if let Some(at) = text[from..].find(content) {
        return vec![(from + at, from + at + content.len())];
    }
    let (text, content) = (text.as_bytes(), content.as_bytes());
    let (mut at, mut taken) = (from, 0);
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    while at < text.len() && taken < content.len() {
        let common = text[at..]
            .iter()
            .zip(&content[taken..])
            .take_while(|(a, b)| a == b)
            .count();
        if common == 0 {
            at += 1;
            continue;
        }
        ranges.push((at, at + common));
        at += common;
        taken += common;
    }
    ranges
}

/// For each token (by the end of its text, see `reasoning_span_tokens`),
/// whether its text belongs to one of `ranges` (in order, not overlapping).
fn content_tokens(token_ends: &[usize], ranges: &[(usize, usize)]) -> Vec<bool> {
    let mut token_start = 0;
    let mut ranges = ranges.iter().peekable();
    token_ends
        .iter()
        .map(|&token_end| {
            while ranges.next_if(|&&(_, end)| end <= token_start).is_some() {}
            let keep = ranges
                .peek()
                .is_some_and(|&&(start, end)| token_in_span(token_start, token_end, start, end));
            token_start = token_end;
            keep
        })
        .collect()
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

/// Usage for a unary Messages response. Cache counters are integer zeros,
/// never null: the Anthropic wire contract has always-present cache counters
/// (0 when caching is unused) and clients do arithmetic on them; the
/// cache-creation breakdown, service tier and inference geography are always
/// written too, and `output_tokens_details` when the turn ran with thinking.
fn messages_usage_from_counts(
    input_tokens: u32,
    output_tokens: u32,
    thinking_tokens: Option<u32>,
) -> messages::Usage {
    let usage = messages::Usage::from_counts(input_tokens, output_tokens);
    match thinking_tokens {
        Some(thinking_tokens) => usage.with_thinking_tokens(thinking_tokens),
        None => usage,
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

#[cfg(test)]
mod content_logprob_tests {
    use super::{content_ranges, content_tokens};

    /// The tokens of `tokens` whose text makes up `content`.
    fn picked<'a>(tokens: &[&'a str], from: usize, content: &str) -> Vec<&'a str> {
        let text: String = tokens.concat();
        let ends: Vec<usize> = tokens
            .iter()
            .scan(0, |end, token| {
                *end += token.len();
                Some(*end)
            })
            .collect();
        let keep = content_tokens(&ends, &content_ranges(&text, from, content));
        tokens
            .iter()
            .zip(keep)
            .filter_map(|(token, keep)| keep.then_some(*token))
            .collect()
    }

    #[test]
    fn content_tokens_skip_reasoning_markers_and_the_eos() {
        // Harmony-style framing; the EOS decodes to nothing.
        let tokens = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "Hello",
            " there",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "Hello",
            " there",
            "",
        ];
        // The reasoning "Hello there" ends at byte 41; the same text as
        // content is found after it.
        assert_eq!(picked(&tokens, 41, "Hello there"), ["Hello", " there"]);
        // Without a reasoning span the first place it appears is taken.
        assert_eq!(picked(&tokens[8..], 0, "final"), ["final"]);
        assert!(picked(&tokens, 0, "").is_empty());
    }

    #[test]
    fn content_split_by_a_call_takes_its_pieces() {
        let tokens = ["Look", ":", "<tool_call>", "{}", "</tool_call>", " done"];
        assert_eq!(picked(&tokens, 0, "Look: done"), ["Look", ":", " done"]);
        // Stripped content keeps only the tokens it overlaps.
        assert_eq!(picked(&["\n\n", "Hi", "\n"], 0, "Hi"), ["Hi"]);
        // A token held back (zero width) inside the content stays with it.
        assert_eq!(picked(&["a", "", "b", "<eos>"], 0, "ab"), ["a", "", "b"]);
    }
}

#[cfg(test)]
mod reasoning_span_tests {
    use super::reasoning_span_tokens;

    #[test]
    fn reasoning_span_tokens_skips_the_delimiters_and_keeps_held_tokens() {
        // Tokens: "<think>" | "a" | "" (text held back) | "b" | "</think>" | "c"
        let text = "<think>ab</think>c";
        assert_eq!(
            reasoning_span_tokens(&[7, 8, 8, 9, 17, 18], text, "ab", "c"),
            3
        );
        // Reasoning cut off before its end: every token is reasoning.
        assert_eq!(reasoning_span_tokens(&[7, 8, 9], "<think>ab", "ab", ""), 2);
        // The span text is not in the output verbatim: everything before the
        // normal text, delimiters included.
        assert_eq!(
            reasoning_span_tokens(&[7, 8, 9, 17, 18], text, "xy", "c"),
            4
        );
        // Text the stop decoder held back to the end ("ST" of a partial
        // "STOP") went to the last token at the flush: the held tokens before
        // it are zero-width inside the span, or at its start.
        assert_eq!(
            reasoning_span_tokens(&[7, 9, 9, 11], "<think>abST", "abST", ""),
            3
        );
        assert_eq!(reasoning_span_tokens(&[7, 7, 9], "<think>ST", "ST", ""), 2);
        assert_eq!(reasoning_span_tokens(&[0, 2], "ST", "ST", ""), 2);
    }
}

#[cfg(test)]
mod messages_usage_wire_tests {
    use super::messages_usage_from_counts;

    /// The contract is about the serialized JSON, not the Rust struct: cache
    /// counters must be present as integer zeros (a struct-level Some(0) that
    /// a serde attr silently skipped would still fail here).
    #[test]
    fn cache_counters_serialize_as_integer_zeros() {
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, None)).unwrap();
        assert_eq!(v["input_tokens"], 25);
        assert_eq!(v["output_tokens"], 150);
        assert_eq!(v["cache_creation_input_tokens"], 0);
        assert_eq!(v["cache_read_input_tokens"], 0);
    }

    /// The public API writes these keys on every message; a typed client
    /// reads a missing `cache_creation` as a decode error, not as zero.
    #[test]
    fn always_present_usage_keys_are_written_without_thinking() {
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, None)).unwrap();
        assert_eq!(
            v["cache_creation"],
            serde_json::json!({"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0})
        );
        assert_eq!(v["service_tier"], "standard");
        assert_eq!(v["inference_geo"], "not_available");
        let keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        assert!(
            !keys.contains(&"output_tokens_details".to_string()),
            "{keys:?}"
        );
        assert!(!keys.contains(&"server_tool_use".to_string()), "{keys:?}");
    }

    /// A turn that ran with thinking reports its thinking tokens, zero
    /// included: the key says thinking was on, the count says how much.
    #[test]
    fn thinking_tokens_are_written_when_the_turn_ran_with_thinking() {
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, Some(40))).unwrap();
        assert_eq!(v["output_tokens_details"]["thinking_tokens"], 40);
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, Some(0))).unwrap();
        assert_eq!(v["output_tokens_details"]["thinking_tokens"], 0);
    }
}

#[cfg(test)]
mod responses_finish_reason_tests {
    use openai_protocol::chat::ChatCompletionRequest;
    use smg_grpc_client::tokenspeed_proto::GenerateComplete;

    use super::*;

    #[expect(
        dead_code,
        reason = "this shared fixture also supports multi-turn MCP tests"
    )]
    mod scripted_tokenizer {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/scripted_tokenizer.rs"
        ));
    }

    #[tokio::test]
    async fn parsed_tool_call_preserves_engine_length_finish() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(scripted_tokenizer::ScriptedTokenizer::new(
            "<tool_call>\n{\"name\":\"user_tool\",\"arguments\":{}}\n</tool_call>",
        ));
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test-model","messages":[{"role":"user","content":"call the tool"}],
            "tools":[{"type":"function","function":{"name":"user_tool","parameters":{"type":"object","properties":{}}}}]
        })).unwrap();
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        );
        for (finish, expected) in [("stop", "tool_calls"), ("length", "length")] {
            let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
                output_ids: vec![100],
                finish_reason: finish.into(),
                ..Default::default()
            });
            let mut decoder = StopSequenceDecoder::new(
                tokenizer.clone(),
                llm_tokenizer::StopSequenceConfig::default(),
                false,
            );
            let (choice, _) = processor
                .process_single_choice(
                    "test-request",
                    &complete,
                    0,
                    &ChatResponseSpec::from(&request),
                    "test-model",
                    &tokenizer,
                    &mut decoder,
                    0,
                    false,
                    true,
                    None,
                    Some("qwen"),
                )
                .await
                .unwrap();
            assert_eq!(choice.finish_reason.as_deref(), Some(expected));
            assert_eq!(choice.message.tool_calls.as_ref().unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn vllm_json_tool_nonstream_keeps_prefilled_reasoning() {
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        );
        for (tool_choice, output) in [
            (
                serde_json::json!("required"),
                r#"plan</think>[{"name":"lookup","parameters":{"q":"Paris"}}]"#,
            ),
            (
                serde_json::json!({"type":"function", "function":{"name":"lookup"}}),
                r#"plan</think>{"q":"Paris"}"#,
            ),
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model":"test-model", "messages":[], "separate_reasoning":true,
                "tool_choice":tool_choice,
                "tools":[{"type":"function","function":{"name":"lookup",
                    "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]
            }))
            .unwrap();
            let mut spec = ChatResponseSpec::from(&request);
            spec.reasoning_in_prefill = true;
            let tokenizer: Arc<dyn Tokenizer> =
                Arc::new(scripted_tokenizer::ScriptedTokenizer::new(output));
            let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
                output_ids: vec![100],
                finish_reason: "stop".into(),
                ..Default::default()
            });
            let mut decoder = StopSequenceDecoder::new(
                tokenizer.clone(),
                llm_tokenizer::StopSequenceConfig::default(),
                false,
            );
            let (choice, _) = processor
                .process_single_choice(
                    "test-request",
                    &complete,
                    0,
                    &spec,
                    "test-model",
                    &tokenizer,
                    &mut decoder,
                    0,
                    true,
                    true,
                    Some("qwen3"),
                    Some("qwen_coder"),
                )
                .await
                .unwrap();
            assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
            assert!(choice.message.content.is_none());
            assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
            let calls = choice.message.tool_calls.unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].function.name, "lookup");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(
                    calls[0].function.arguments.as_deref().unwrap()
                )
                .unwrap(),
                serde_json::json!({"q":"Paris"})
            );
        }
    }

    #[tokio::test]
    async fn gemma4_nonstream_native_tool_choices_preserve_reasoning_and_arguments() {
        let output = r#"<|channel>thought
plan<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#;
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(scripted_tokenizer::ScriptedTokenizer::new(output));
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        );
        for tool_choice in [
            serde_json::json!("auto"),
            serde_json::json!("required"),
            serde_json::json!({"type":"function", "function":{"name":"get_weather"}}),
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model":"test-model", "messages":[], "separate_reasoning":true,
                "tool_choice":tool_choice,
                "tools":[{"type":"function","function":{"name":"get_weather",
                    "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
            }))
            .unwrap();
            let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
                output_ids: vec![100],
                finish_reason: "stop".into(),
                ..Default::default()
            });
            let mut decoder = StopSequenceDecoder::new(
                tokenizer.clone(),
                llm_tokenizer::StopSequenceConfig::default(),
                false,
            );
            let (choice, _) = processor
                .process_single_choice(
                    "test-request",
                    &complete,
                    0,
                    &ChatResponseSpec::from(&request),
                    "test-model",
                    &tokenizer,
                    &mut decoder,
                    0,
                    true,
                    true,
                    Some("gemma4"),
                    Some("gemma4"),
                )
                .await
                .unwrap();
            assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
            assert!(choice.message.content.is_none());
            assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
            let calls = choice.message.tool_calls.unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].function.name, "get_weather");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(
                    calls[0].function.arguments.as_deref().unwrap()
                )
                .unwrap(),
                serde_json::json!({"city":"Paris"})
            );
        }
    }

    /// `parallel_tool_calls: false` returns the first of the calls a model
    /// writes; unset or true, all of them.
    #[tokio::test]
    async fn parallel_tool_calls_false_returns_the_first_call() {
        let call = |q: u32| {
            format!(
                "<tool_call>\n{{\"name\":\"user_tool\",\"arguments\":{{\"q\":{q}}}}}\n</tool_call>"
            )
        };
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(scripted_tokenizer::ScriptedTokenizer::new(
            &format!("{}\n{}", call(1), call(2)),
        ));
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        );
        for (parallel, calls) in [(None, 2), (Some(true), 2), (Some(false), 1)] {
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model":"test-model","messages":[{"role":"user","content":"call the tool"}],
                "tools":[{"type":"function","function":{"name":"user_tool","parameters":{"type":"object","properties":{"q":{"type":"integer"}}}}}],
                "parallel_tool_calls": parallel
            })).unwrap();
            let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
                output_ids: vec![100],
                finish_reason: "stop".into(),
                ..Default::default()
            });
            let mut decoder = StopSequenceDecoder::new(
                tokenizer.clone(),
                llm_tokenizer::StopSequenceConfig::default(),
                false,
            );
            let (choice, _) = processor
                .process_single_choice(
                    "test-request",
                    &complete,
                    0,
                    &ChatResponseSpec::from(&request),
                    "test-model",
                    &tokenizer,
                    &mut decoder,
                    0,
                    false,
                    true,
                    None,
                    Some("qwen"),
                )
                .await
                .unwrap();
            let got = choice.message.tool_calls.unwrap();
            assert_eq!(got.len(), calls, "{parallel:?}");
            assert_eq!(got[0].function.arguments.as_deref(), Some("{\"q\":1}"));
            assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        }
    }
}
