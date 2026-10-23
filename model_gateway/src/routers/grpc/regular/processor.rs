//! Shared response processing logic for gRPC routers
//!
//! This module contains response processing functions that are shared between
//! the regular router and PD router.

use std::{sync::Arc, time::Instant};

use futures::future::try_join_all;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, SequenceDecoderStep, StopSequenceDecoder},
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
use symphony::{adapt, Events};
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{error, warn};

use crate::routers::{
    error,
    grpc::{
        common::{response_collection, response_formatting},
        context::{DispatchMetadata, ExecutionResult},
        proto_wrapper::ProtoGenerateComplete,
        regular::symphony::{
            engine_finish, finished,
            telemetry::{self, Who},
            Assignment, Delta, Reader, SymphonyError,
        },
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
                model,
                tokenizer,
                stop_decoder,
            );
        }
        let Decoded {
            text: final_text,
            token_ends,
            stopped,
            held: _,
        } = decode_complete(
            stop_decoder,
            complete.output_ids(),
            request_id,
            index as u32,
        )
        .map_err(|e| e.to_string())?
        .with_eager_token_ends(
            original_request.separate_reasoning && reasoning_parser_available,
            tokenizer,
            complete.output_ids(),
            stop_decoder.skip_special_tokens(),
        );

        // The decoded output, kept to place the content's tokens for its
        // logprobs; the content starts after the reasoning span.
        let output_logprobs = complete.output_logprobs();
        let decoded_text = output_logprobs.as_ref().map(|_| final_text.clone());

        // Steps 1 and 2: the reasoning split out, then the tool calls parsed
        // from the content (shared with the Messages reader).
        let ReasoningSplit {
            content: processed_text,
            reasoning: reasoning_text,
            reasoning_tokens,
            content_from,
        } = self.split_reasoning(
            final_text,
            &token_ends,
            ReasoningSource {
                wanted: original_request.separate_reasoning && reasoning_parser_available,
                in_prefill: original_request.reasoning_in_prefill,
                parser_name: reasoning_parser_name,
                model,
            },
        );
        let (tool_calls, processed_text) = self
            .parse_calls(
                processed_text,
                ToolCallSource {
                    tools: original_request.tools.as_deref(),
                    tool_choice: original_request.tool_choice.as_ref(),
                    model,
                    parser_name: tool_parser_name,
                    parser_available: tool_parser_available,
                    history_tool_calls_count,
                    first_call_only: original_request.first_tool_call_only,
                },
            )
            .await;

        // Step 3: Determine finish reason (a local stop beats the engine's).
        let finish_reason_str = finish_after_local_stop(stopped, complete);

        // Parsed calls do not override an engine truncation or failure.
        let final_finish_reason_str = if tool_calls.is_some()
            && !matches!(finish_reason_str, "length" | "failed" | "error")
        {
            "tool_calls"
        } else {
            finish_reason_str
        };

        let matched_stop = matched_stop_value(stop_decoder, complete);

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
    #[expect(clippy::too_many_arguments)]
    fn read_with_symphony(
        request_id: &str,
        assignment: &Assignment,
        complete: &ProtoGenerateComplete,
        index: usize,
        original_request: &ChatResponseSpec,
        model: &str,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<(ChatChoice, u32), String> {
        let tools = original_request.tools.clone().unwrap_or_default();
        // The output as one delta, then the decoder's last text: two deltas the events borrow,
        // both outliving the buffer.
        let mut delta = Delta::default();
        let mut flushed = Delta::default();
        let mut events = Events::new();
        let mut reader = Reader::new(assignment, tools, &mut events)?
            .first_call_only(original_request.first_tool_call_only);
        let stopped = reader.read(stop_decoder, complete.output_ids(), &mut delta, &mut events)?;
        // Complete token decoding, including bytes and a partial stop match, as
        // on the older path; the text it releases is the last token's.
        let tail = stop_decoder
            .finish()
            .map_err(|e| reader.failed(format!("Failed to finish token decoding: {e}")))?;
        utils::warn_detokenize_recoveries(request_id, index as u32, stop_decoder);
        if let SequenceDecoderOutput::Text(text) | SequenceDecoderOutput::StoppedWithText(text) =
            tail
        {
            reader.tail(text, &mut flushed, &mut events)?;
        }
        let reason = finish_after_local_stop(stopped || stop_decoder.is_stopped(), complete);
        let who = Who {
            request_id,
            model,
            choice: index as u32,
        };
        reader.end(engine_finish(reason), who, &mut events)?;
        let reasoning_tokens =
            finished(events.as_slice()).map_or(0, |finished| finished.reasoning_tokens);
        let mut choice = adapt::chat::message(index as u32, &events);
        choice.matched_stop = matched_stop_value(stop_decoder, complete);
        // The content's tokens alone keep their logprobs, as on the older path: none when
        // the message has no content (whitespace alone, which `adapt::chat::message` leaves
        // out of the message); the bytes the reader released at the end are placed where the
        // output wrote them (`Reader::content_spans`).
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
                    content_tokens(
                        reader.token_ends(),
                        &reader.content_spans(events.as_slice()),
                    )
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
    /// thinking. A call whose arguments no `tool_use` input holds refuses the message
    /// (`SymphonyError::InputRefused`, the call named), rather than leave as `{}` without a word;
    /// the caller counts the refusal under the table ([`telemetry::record_refused`]).
    fn read_messages_with_symphony(
        dispatch: &DispatchMetadata,
        assignment: &Assignment,
        complete: &ProtoGenerateComplete,
        request: &MessagesResponseSpec,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<Message, SymphonyError> {
        let mut delta = Delta::default();
        let mut flushed = Delta::default();
        let mut events = Events::new();
        let mut reader = Reader::new(assignment, Arc::clone(&request.chat_tools), &mut events)?
            .first_call_only(request.first_tool_call_only);
        stop_decoder.reset();
        let stopped = reader.read(stop_decoder, complete.output_ids(), &mut delta, &mut events)?;
        let tail = stop_decoder
            .finish()
            .map_err(|e| reader.failed(format!("Failed to finish token decoding: {e}")))?;
        utils::warn_detokenize_recoveries(&dispatch.request_id, 0, stop_decoder);
        if let SequenceDecoderOutput::Text(text) | SequenceDecoderOutput::StoppedWithText(text) =
            tail
        {
            reader.tail(text, &mut flushed, &mut events)?;
        }
        let reason = finish_after_local_stop(stopped || stop_decoder.is_stopped(), complete);
        let who = Who {
            request_id: &dispatch.request_id,
            model: &dispatch.model,
            choice: 0,
        };
        reader.end(engine_finish(reason), who, &mut events)?;
        let mut output = adapt::messages::output(&events);
        // A `tool_use` block carries `{}` only where the parser said the arguments are none;
        // arguments that are JSON no `input` value holds would leave as `{}` with no word, so the
        // message is refused, naming the call (the stream carries such arguments as written).
        if let Some(refused) = output.refused.first() {
            return Err(SymphonyError::InputRefused(format!(
                "the tool_use input of call {} ({}) cannot carry the arguments the model wrote: {}",
                refused.index, refused.name, refused.why
            )));
        }
        if !request.thinking_text_returned {
            utils::message_utils::omit_thinking_text(&mut output.content);
        }
        let stop_sequence = matched_stop_string(stop_decoder, complete);
        let stop_reason = match output.stop_reason {
            Some(messages::StopReason::EndTurn) if stop_sequence.is_some() => {
                Some(messages::StopReason::StopSequence)
            }
            stop_reason => stop_reason,
        };
        let stop_sequence = stop_sequence
            .filter(|_| matches!(stop_reason, Some(messages::StopReason::StopSequence)));
        let thinking_requested = request.thinking_requested;
        let reasoning_tokens =
            finished(events.as_slice()).map_or(0, |finished| finished.reasoning_tokens);
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
    ///
    /// Returns the response with the engine's counters it was built from:
    /// the response's usage drops the unbilled prompt tokens and may carry
    /// the gateway parser's reasoning split; the settlement takes the
    /// counters.
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: ChatResponseSpec,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
    ) -> Result<(ChatCompletionResponse, response_formatting::TokenCounts), axum::response::Response>
    {
        let model = &dispatch.model;
        let reasoning_parser_name = self.parser_resolver.reasoning_parser(model);
        let tool_parser_name = self.parser_resolver.tool_parser(model);
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        let history_tool_calls_count = chat_request.history_tool_calls_count;

        // Check parser availability once upfront (not per choice). Where a Symphony table
        // reads the response its reader takes the thought, and no legacy parser is built to
        // be asked whether one is available.
        let reasoning_parser_available = chat_request.separate_reasoning
            && chat_request.symphony.is_none()
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
        if chat_request.separate_reasoning
            && chat_request.symphony.is_none()
            && !reasoning_parser_available
        {
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
        let counts = response_formatting::TokenCounts::of(&all_responses);
        let usage = response_formatting::chat_usage(
            counts.usage(),
            (reasoning_parser_available || chat_request.symphony.is_some())
                .then_some(parsed_reasoning_tokens),
            chat_request.unbilled_prompt_tokens,
        );

        // Build final ChatCompletionResponse
        let response = ChatCompletionResponse::builder(&dispatch.request_id, &dispatch.model)
            .created(dispatch.created)
            .choices(choices)
            .usage(usage)
            .maybe_system_fingerprint(dispatch.weight_version.clone())
            .build();
        Ok((response, counts))
    }

    /// Split the reasoning out of a decoded choice when the request wants
    /// it, with a fresh parser (non-streaming extraction keeps no state
    /// across requests, so the shared pooled mutex is not serialized on); a
    /// prompt that ends inside the thinking block starts the parser in
    /// reasoning mode. A parse error leaves the text whole, as before.
    fn split_reasoning(
        &self,
        text: String,
        token_ends: &[usize],
        source: ReasoningSource<'_>,
    ) -> ReasoningSplit {
        let mut split = ReasoningSplit {
            content: text,
            reasoning: None,
            reasoning_tokens: 0,
            content_from: 0,
        };
        if !source.wanted {
            return split;
        }
        let Some(mut parser) = utils::create_reasoning_parser(
            &self.reasoning_parser_factory,
            source.parser_name,
            source.model,
        ) else {
            return split;
        };
        if source.in_prefill {
            parser.mark_reasoning_started();
        }
        match parser.detect_and_parse_reasoning(&split.content) {
            Ok(result) => {
                if !result.reasoning_text.is_empty() {
                    split.reasoning_tokens = reasoning_span_tokens(
                        token_ends,
                        &split.content,
                        &result.reasoning_text,
                        &result.normal_text,
                    );
                    split.content_from =
                        reasoning_span(&split.content, &result.reasoning_text, &result.normal_text)
                            .1;
                    split.reasoning = Some(result.reasoning_text);
                }
                split.content = result.normal_text;
            }
            Err(e) => {
                warn!("Reasoning parsing error, skipping parsing: {e}");
            }
        }
        split
    }

    /// Parse the tool calls out of the content when the request offers tools
    /// and its choice allows calls: through the JSON-schema path when the
    /// choice forced a call and the parser's format is not native (the
    /// constraint shaped the output), else through the tool parser when one
    /// is available; the first call alone when the request asked for it.
    async fn parse_calls(
        &self,
        content: String,
        source: ToolCallSource<'_>,
    ) -> (Option<Vec<ToolCall>>, String) {
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let mut content = content;
        let enabled = !matches!(
            source.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );
        if let (true, Some(tools)) = (enabled, source.tools) {
            let native_tool_format = self
                .tool_parser_factory
                .registry()
                .uses_native_tool_format(source.parser_name);
            if forced_call_shapes_output(native_tool_format, source.tool_choice) {
                (tool_calls, content) = utils::parse_json_schema_response(
                    &content,
                    source.tool_choice,
                    source.model,
                    source.history_tool_calls_count,
                );
            } else if source.parser_available {
                (tool_calls, content) = self
                    .parse_tool_calls(
                        &content,
                        source.model,
                        source.parser_name,
                        tools,
                        source.history_tool_calls_count,
                    )
                    .await;
            }
        }
        if source.first_call_only {
            utils::keep_first_tool_call(&mut tool_calls);
        }
        (tool_calls, content)
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
            let decoded =
                decode_complete(stop_decoder, complete.output_ids(), &dispatch.request_id, 0)
                    .map_err(|e| error::internal_error("process_tokens_failed", e.to_string()))?;
            let decoded_text = decoded.text;

            let output_ids = complete.output_ids().to_vec();
            let finish_reason = utils::parse_finish_reason(
                finish_after_local_stop(decoded.stopped, &complete),
                complete.completion_tokens(),
            );
            let matched_stop = matched_stop_value(stop_decoder, &complete);

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
    ///
    /// Returns the message with the engine's counters it was built from: the
    /// Messages usage reports no cache hits and the greater of the engine's
    /// and the parser's thinking count; the settlement takes the counters.
    pub async fn process_non_streaming_messages_response(
        &self,
        execution_result: ExecutionResult,
        messages_request: MessagesResponseSpec,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<(Message, response_formatting::TokenCounts), axum::response::Response> {
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
        // The engine's counters, for the settlement, whichever reader builds the message.
        let counts = response_formatting::TokenCounts::of(std::slice::from_ref(&complete));
        if let Some(assignment) = &messages_request.symphony {
            return Self::read_messages_with_symphony(
                &dispatch,
                assignment,
                &complete,
                &messages_request,
                stop_decoder,
            )
            .map(|message| (message, counts))
            .map_err(|e| match e {
                // The output's own doing: the model wrote arguments no tool_use input holds.
                SymphonyError::InputRefused(why) => {
                    telemetry::record_refused(telemetry::table_label(assignment.format.name()));
                    warn!(function = "process_non_streaming_messages_response", error = %why, "the message's tool_use input cannot carry the model's arguments");
                    error::internal_error("symphony_tool_use_input_refused", why)
                }
                e => {
                    error!(function = "process_non_streaming_messages_response", error = %e, "Symphony failed to read the message");
                    error::internal_error("symphony_read_failed", e.to_string())
                }
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

        let Decoded {
            text: final_text,
            token_ends,
            stopped,
            held: _,
        } = decode_complete(stop_decoder, complete.output_ids(), &dispatch.request_id, 0)
            .map_err(|e| {
                if let DecodeFailure::Process(_) = e {
                    error!(function = "process_non_streaming_messages_response", error = %e.detail(), "Failed to process tokens");
                }
                error::internal_error("process_tokens_failed", e.to_string())
            })?
            .with_eager_token_ends(
                reasoning_parser_available,
                &tokenizer,
                complete.output_ids(),
                stop_decoder.skip_special_tokens(),
            );

        // Steps 1 and 2: the reasoning split out, then the tool calls parsed
        // from the content (shared with the chat reader); the Messages tool
        // choice reads as its chat form, whose `forces_call` is the one rule
        // for the JSON-schema path.
        let ReasoningSplit {
            content: processed_text,
            reasoning: reasoning_text,
            reasoning_tokens: thinking_tokens,
            content_from: _,
        } = self.split_reasoning(
            final_text,
            &token_ends,
            ReasoningSource {
                wanted: reasoning_parser_available,
                in_prefill: messages_request.reasoning_in_prefill,
                parser_name: reasoning_parser_name.as_deref(),
                model,
            },
        );
        let chat_tool_choice = messages_request.tool_choice.as_ref().map(ToolChoice::from);
        let (tool_calls, processed_text) = self
            .parse_calls(
                processed_text,
                ToolCallSource {
                    tools: messages_request
                        .has_tools
                        .then_some(&messages_request.chat_tools[..]),
                    tool_choice: chat_tool_choice.as_ref(),
                    model,
                    parser_name: tool_parser_name.as_deref(),
                    parser_available: tool_parser_available,
                    history_tool_calls_count: messages_request.history_tool_calls_count,
                    first_call_only: messages_request.first_tool_call_only,
                },
            )
            .await;

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
        if !messages_request.thinking_text_returned {
            utils::message_utils::omit_thinking_text(&mut content_blocks);
        }

        // Step 4: Determine stop_reason and stop_sequence (derived from same
        // conditions): a local stop beats the engine's reason, and the matched
        // sequence is surfaced for a StopSequence result.
        let finish_reason_str = finish_after_local_stop(stopped, &complete);
        let stop_sequence = matched_stop_string(stop_decoder, &complete);

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
        let message = Message {
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
        };
        Ok((message, counts))
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
                let decoded = decode_complete(
                    stop_decoder,
                    complete.output_ids(),
                    &dispatch.request_id,
                    index_offset + i as u32,
                )
                .map_err(|e| error::internal_error("process_tokens_failed", e.to_string()))?;
                let decoded_text = decoded.text;
                let stopped = decoded.stopped;

                prompt_tokens = prompt_tokens.max(complete.prompt_tokens());
                cached_tokens = cached_tokens.max(complete.cached_tokens());
                total_completion += complete.completion_tokens();
                total_reasoning += complete.reasoning_tokens();
                total_spec_accepted += complete.spec_accepted_tokens();
                total_spec_drafted += complete.spec_draft_tokens();

                let finish_reason =
                    utils::completion_finish_reason(stopped, complete.finish_reason());

                let matched_stop = matched_stop_value(stop_decoder, &complete);

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
                    finish_reason: Some(finish_reason.to_string()),
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

/// The reasoning split out of a decoded choice: the content after it, the
/// reasoning text, the reasoning's tokens by the token ends, and where the
/// content starts in the decoded text (for the content's logprobs).
struct ReasoningSplit {
    content: String,
    reasoning: Option<String>,
    reasoning_tokens: u32,
    content_from: usize,
}

/// What the reasoning split reads from the request: whether the request
/// wants the split at all (its separate-reasoning switch, with a parser
/// available), whether the rendered prompt ends inside the thinking block,
/// the parser the request resolved to, and the model.
struct ReasoningSource<'a> {
    wanted: bool,
    in_prefill: bool,
    parser_name: Option<&'a str>,
    model: &'a str,
}

/// What a tool-call parse reads from the request: the tools it offered (none
/// when it offered none), its tool choice in the chat form, the model, the
/// parser the request resolved to and whether it is available, how many
/// calls the history already holds, and whether the first call alone is kept.
struct ToolCallSource<'a> {
    tools: Option<&'a [Tool]>,
    tool_choice: Option<&'a ToolChoice>,
    model: &'a str,
    parser_name: Option<&'a str>,
    parser_available: bool,
    history_tool_calls_count: usize,
    first_call_only: bool,
}

/// Whether a forced call shaped the output as JSON the constraint dictated
/// (so the JSON-schema path reads it): the choice forces a call and the
/// parser's format is not native. One rule for the chat choice and the
/// Messages choice read as its chat form.
fn forced_call_shapes_output(native_tool_format: bool, tool_choice: Option<&ToolChoice>) -> bool {
    !native_tool_format && tool_choice.is_some_and(ToolChoice::forces_call)
}

/// A choice's output decoded through the stop decoder, as every non-streaming
/// reader decodes it: the text up to a stop, where each token's text ends in
/// it (the last token taking the decoder's tail), whether a stop fired at any
/// token or in the tail, and whether the decoder held text back for bytes
/// still to come.
#[derive(Debug)]
struct Decoded {
    text: String,
    token_ends: Vec<usize>,
    stopped: bool,
    held: bool,
}

impl Decoded {
    /// The token ends as the tokenizer's eager boundaries when byte buffering
    /// released several tokens at once (see `eager_token_ends`); `wanted` is
    /// the caller's rule for when the spans matter at all.
    fn with_eager_token_ends(
        mut self,
        wanted: bool,
        tokenizer: &Arc<dyn Tokenizer>,
        output_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Self {
        if wanted && self.held && tokenizer.uses_byte_fallback() {
            if let Some(ends) = eager_token_ends(
                tokenizer,
                &output_ids[..self.token_ends.len()],
                skip_special_tokens,
                &self.text,
            ) {
                self.token_ends = ends;
            }
        }
        self
    }
}

/// Why a choice could not be decoded, in the words the readers report.
#[derive(Debug)]
enum DecodeFailure {
    Process(String),
    Finish(String),
}

impl DecodeFailure {
    /// The decoder's own words, without the reader's prefix.
    fn detail(&self) -> &str {
        match self {
            Self::Process(detail) | Self::Finish(detail) => detail,
        }
    }
}

impl std::fmt::Display for DecodeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process(e) => write!(f, "Failed to process tokens: {e}"),
            Self::Finish(e) => write!(f, "Failed to finish token decoding: {e}"),
        }
    }
}

/// Decode a complete choice's `output_ids` through `stop_decoder`: reset it,
/// run the tokens with an early break on a stop, finish the decoding (bytes
/// and a partial stop match; the tail is the last token's), warn about
/// detokenize recoveries, and note a stop the decoder found at any token or
/// in the tail.
fn decode_complete(
    stop_decoder: &mut StopSequenceDecoder,
    output_ids: &[u32],
    request_id: &str,
    index: u32,
) -> Result<Decoded, DecodeFailure> {
    stop_decoder.reset();
    let mut text = String::new();
    let mut stopped = false;
    let mut held = false;
    let mut token_ends = Vec::with_capacity(output_ids.len());
    for &token_id in output_ids {
        match stop_decoder
            .process_token_into(token_id, &mut text)
            .map_err(|e| DecodeFailure::Process(e.to_string()))?
        {
            SequenceDecoderStep::Held => held = true,
            step => stopped |= step.stopped(),
        }
        token_ends.push(text.len());
        if stopped {
            break;
        }
    }

    let tail = stop_decoder
        .finish_into(&mut text)
        .map_err(|e| DecodeFailure::Finish(e.to_string()))?;
    utils::warn_detokenize_recoveries(request_id, index, stop_decoder);
    if matches!(
        tail,
        SequenceDecoderStep::Text | SequenceDecoderStep::StoppedWithText
    ) {
        if let Some(last) = token_ends.last_mut() {
            *last = text.len();
        }
    }
    stopped |= stop_decoder.is_stopped();
    Ok(Decoded {
        text,
        token_ends,
        stopped,
        held,
    })
}

/// The finish reason with a local stop-decoder match taking precedence over
/// the engine's (which is "length" when stop strings are enforced
/// gateway-side rather than by the backend).
fn finish_after_local_stop(stopped: bool, complete: &ProtoGenerateComplete) -> &str {
    if stopped {
        "stop"
    } else {
        complete.finish_reason()
    }
}

/// The string stop the local decoder matched (the engine reports no stop over
/// the ZMQ path), else the engine's value: a stop token's id stays reported.
fn matched_stop_value(
    stop_decoder: &StopSequenceDecoder,
    complete: &ProtoGenerateComplete,
) -> Option<serde_json::Value> {
    stop_decoder
        .matched_stop()
        .map(|s| serde_json::Value::String(s.to_string()))
        .or_else(|| complete.matched_stop_json())
}

/// [`matched_stop_value`] as a Messages stop sequence: the string, or the
/// engine's value when it is one.
fn matched_stop_string(
    stop_decoder: &StopSequenceDecoder,
    complete: &ProtoGenerateComplete,
) -> Option<String> {
    stop_decoder.matched_stop().map(String::from).or_else(|| {
        complete
            .matched_stop_json()
            .and_then(|v| v.as_str().map(String::from))
    })
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
            "/test_harness/src/scripted_tokenizer.rs"
        ));
    }

    /// The Messages tool choice read as its chat form decides the JSON-schema
    /// path exactly as the chat choice does: a named tool and `any` force a
    /// call, `auto` and `none` do not, and a native format never takes it.
    #[test]
    fn a_messages_choice_forces_a_call_as_its_chat_form_does() {
        let named = messages::ToolChoice::Tool {
            name: "lookup".to_string(),
            disable_parallel_tool_use: None,
        };
        let any = messages::ToolChoice::Any {
            disable_parallel_tool_use: None,
        };
        let auto = messages::ToolChoice::Auto {
            disable_parallel_tool_use: None,
        };
        for (choice, forces) in [
            (named, true),
            (any, true),
            (auto, false),
            (messages::ToolChoice::None, false),
        ] {
            let chat = ToolChoice::from(&choice);
            assert_eq!(
                forced_call_shapes_output(false, Some(&chat)),
                forces,
                "{choice:?}"
            );
            assert!(
                !forced_call_shapes_output(true, Some(&chat)),
                "{choice:?} native"
            );
        }
        assert!(!forced_call_shapes_output(false, None));
    }

    /// Without a parser the split leaves the text whole and the content at 0.
    #[test]
    fn splitting_without_a_reasoning_parser_leaves_the_text_whole() {
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        );
        let split = processor.split_reasoning(
            "<think>a</think>b".to_string(),
            &[8, 9, 17],
            ReasoningSource {
                wanted: true,
                in_prefill: false,
                parser_name: None,
                model: "no-such-model",
            },
        );
        assert_eq!(split.content, "<think>a</think>b");
        assert_eq!(
            (split.reasoning, split.reasoning_tokens, split.content_from),
            (None, 0, 0)
        );
    }

    /// One decode for every reader: the text, where each token's text ends,
    /// whether a stop fired; the finish and the matched stop read from it.
    #[test]
    fn decoding_a_choice_reports_its_text_token_ends_and_stop() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(scripted_tokenizer::ScriptedTokenizer::from_chunks(vec![
                "Hel".into(),
                "lo ".into(),
                "world".into(),
                "!".into(),
            ]));
        let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
            finish_reason: "length".into(),
            ..Default::default()
        });

        // Without a stop: every token's text end, the last at the text's end.
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let decoded = decode_complete(&mut decoder, &[100, 101, 102, 103], "req", 0).unwrap();
        assert_eq!(decoded.text, "Hello world!");
        assert_eq!(decoded.token_ends, vec![3, 6, 11, 12]);
        assert!(!decoded.stopped);
        assert_eq!(
            finish_after_local_stop(decoded.stopped, &complete),
            "length"
        );
        assert_eq!(matched_stop_value(&decoder, &complete), None);

        // A string stop: the text ends before it, the stop fired and is the
        // finish and the matched stop, whatever the engine said.
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            llm_tokenizer::StopSequenceConfig {
                stop_sequences: vec!["world".to_string()],
                ..Default::default()
            },
            false,
        );
        let decoded = decode_complete(&mut decoder, &[100, 101, 102, 103], "req", 0).unwrap();
        assert_eq!(decoded.text, "Hello ");
        assert!(decoded.stopped);
        assert_eq!(decoded.token_ends.last(), Some(&decoded.text.len()));
        assert_eq!(finish_after_local_stop(decoded.stopped, &complete), "stop");
        assert_eq!(
            matched_stop_value(&decoder, &complete),
            Some(serde_json::Value::String("world".into()))
        );
        assert_eq!(
            matched_stop_string(&decoder, &complete).as_deref(),
            Some("world")
        );
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
