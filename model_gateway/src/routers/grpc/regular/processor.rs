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
    chat::{ChatChoice, ChatCompletionResponse},
    completion::{CompletionChoice, CompletionResponse},
    generate::{GenerateMetaInfo, GenerateResponse},
    messages::{self, Message},
};
use symphony::{adapt, Events};
use tracing::{error, warn};

use crate::{
    observability::metrics::{metrics_labels, Metrics, StreamingMetricsParams},
    routers::{
        common::retry::mark_non_retryable,
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
            utils::{
                self,
                isolation::{isolated_sync, Panicked, READER_PANICKED_CODE},
            },
        },
    },
};

/// What a reader answers where request building assigned the response no table: every chat
/// and Messages request is assigned one (`regular::stages::reading::decide`), so a spec without
/// one reached the reader by another path.
pub(super) const NO_TABLE: &str = "no table reads the response: request building assigned none";

/// Unified response processor for both routers
#[derive(Clone)]
pub(crate) struct ResponseProcessor {
    /// The `backend_type` label of the answers' token and generation
    /// records, the one the streaming processor beside it carries.
    backend_type: &'static str,
}

impl ResponseProcessor {
    pub fn new() -> Self {
        Self {
            backend_type: metrics_labels::BACKEND_REGULAR,
        }
    }

    /// The processor with its records labeled `backend_type`.
    pub fn with_backend(mut self, backend_type: &'static str) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// One choice of a GenerateComplete response, as the table request
    /// building assigned reads it ([`Self::read_with_symphony`]). Returns the
    /// choice and the number of output tokens the reader routed to
    /// `reasoning_content`.
    pub fn process_single_choice(
        request_id: &str,
        complete: &ProtoGenerateComplete,
        index: usize,
        original_request: &ChatResponseSpec,
        model: &str,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<(ChatChoice, u32), String> {
        stop_decoder.reset();
        let assignment = original_request.symphony.as_ref().ok_or(NO_TABLE)?;
        Self::read_with_symphony(
            request_id,
            assignment,
            complete,
            index,
            original_request,
            model,
            tokenizer,
            stop_decoder,
        )
    }

    /// The choice as Symphony reads it (see `regular::symphony`): the output as
    /// one delta through the stop decoder, the decoder's last text once decoding
    /// is finished, the engine's finish, and the message the events fold into.
    /// The matched stop is the decoder's or the engine's, and the logprobs are the
    /// engine's, one entry per generated token.
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
        // Complete token decoding, including bytes and a partial stop match;
        // the text it releases is the last token's.
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
        // The output's logprobs as the engine reports them, one entry per generated token:
        // a message without content keeps them like any other.
        choice.logprobs = complete.output_logprobs().map(|proto_logprobs| {
            utils::convert_proto_to_openai_logprobs(
                proto_logprobs,
                tokenizer,
                original_request.top_logprobs,
            )
        });
        Ok((choice, reasoning_tokens))
    }

    /// The message as Symphony reads it (see `regular::symphony`): the output as one delta
    /// through the stop decoder, the decoder's last text once decoding is finished, the engine's
    /// finish, and the content blocks the events fold into (`adapt::messages::output`). A stop
    /// sequence the decoder or the engine matched is the stop reason; the
    /// thinking tokens are the engine's, or the reasoning text's, when the request asked for
    /// thinking. A call whose arguments no `tool_use` input holds refuses the message
    /// (`SymphonyError::InputRefused`, the call named), rather than leave as `{}` without a word;
    /// the caller counts the refusal under the table ([`telemetry::record_refused`]). A call whose
    /// arguments are a whole value of another kind than an object (a double-encoded or bare string,
    /// an array) carries `{}` with a word in the log ([`object_input`]).
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
        // Arguments that are a whole value of another kind (a double-encoded or
        // bare string, an array) become `{}` with a word in the log: the request
        // side takes no other `input`.
        object_inputs(&mut output.content, "read_messages_with_symphony");
        if !request.thinking_text_returned {
            utils::message_utils::omit_thinking_text(&mut output.content);
        }
        let stop_sequence = matched_stop_string(stop_decoder, complete);
        let stop_reason = match output.stop_reason {
            Some(messages::StopReason::EndTurn) if stop_sequence.is_some() => {
                Some(messages::StopReason::StopSequence)
            }
            // The reader's `max_tokens` is the engine's `length`: the budget
            // or the window, told apart from the counts.
            Some(messages::StopReason::MaxTokens) => Some(
                request.length_stop_reason(complete.prompt_tokens(), complete.completion_tokens()),
            ),
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
                complete.cached_tokens(),
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
        let started = generation_start(&execution_result);
        let model = &dispatch.model;
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;
        // The table that reads every choice, assigned by request building.
        let table = chat_request
            .symphony
            .as_ref()
            .map(Assignment::table)
            .ok_or_else(|| error::internal_error("no_table_assigned", NO_TABLE))?;

        // Process all choices. Each choice's reading is isolated: a panic in
        // it is this request's 500, not the connection's end (see
        // `utils::isolation`).
        let mut choices = Vec::new();
        let mut parsed_reasoning_tokens = 0;
        for (index, complete) in all_responses.iter().enumerate() {
            let read = isolated_sync(|| {
                Self::process_single_choice(
                    &dispatch.request_id,
                    complete,
                    index,
                    &chat_request,
                    model,
                    &tokenizer,
                    stop_decoder,
                )
            });
            match read {
                Ok(Ok((choice, reasoning_tokens))) => {
                    choices.push(choice);
                    parsed_reasoning_tokens += reasoning_tokens;
                }
                Ok(Err(e)) => {
                    return Err(error::internal_error(
                        "process_choice_failed",
                        format!("Failed to process choice {index}: {e}"),
                    ));
                }
                Err(panicked) => {
                    let panicked = panicked.read_by(Some(table));
                    panicked.record(&dispatch.request_id, model);
                    return Err(reader_panicked(&panicked));
                }
            }
        }

        // The usage from the gRPC response counters, built as the streamed
        // answer builds its usage chunk: with the reasoning tokens the table
        // read (zero included) and every detail counter present as the
        // OpenAI API sends it.
        let counts = response_formatting::TokenCounts::of_collected(
            &all_responses,
            chat_request.expected_choices,
        );
        let usage = response_formatting::chat_usage(
            counts.usage(),
            Some(parsed_reasoning_tokens),
            chat_request.unbilled_prompt_tokens,
            chat_request.image_tokens,
        );

        // Build final ChatCompletionResponse
        let response = ChatCompletionResponse::builder(&dispatch.request_id, &dispatch.model)
            .created(dispatch.created)
            .choices(choices)
            .usage(usage)
            .maybe_system_fingerprint(dispatch.weight_version.clone())
            .build();
        // The answer's token and generation records, the ones a streamed
        // answer makes at its end; a collected answer shows no first token.
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.backend_type,
            model_id: &dispatch.model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: None,
            generation_duration: started.elapsed(),
            input_tokens: Some(u64::from(counts.prompt)),
            cached_tokens: Some(u64::from(counts.cached)),
            output_tokens: u64::from(counts.completion),
        });
        Ok((response, counts))
    }

    /// Process non-streaming generate response (collects all responses and builds final response array)
    ///
    /// `metrics_endpoint` labels the answer's token and generation records:
    /// the endpoint the request's own count carries (the transcription stage
    /// decodes through this processor too).
    pub async fn process_non_streaming_generate_response(
        &self,
        execution_result: ExecutionResult,
        dispatch: DispatchMetadata,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
        start_time: Instant,
        metrics_endpoint: &'static str,
    ) -> Result<Vec<GenerateResponse>, axum::response::Response> {
        let started = generation_start(&execution_result);
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

        // The records a streamed generate answer makes at its end, from the
        // counters the settlement reads: the prompt and its cache hit once
        // per request, the completion tokens of every choice.
        let counts = response_formatting::TokenCounts::of_generated(&result_array);
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.backend_type,
            model_id: &dispatch.model,
            endpoint: metrics_endpoint,
            ttft: None,
            generation_duration: started.elapsed(),
            input_tokens: Some(u64::from(counts.prompt)),
            cached_tokens: Some(u64::from(counts.cached)),
            output_tokens: u64::from(counts.completion),
        });
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
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<(Message, response_formatting::TokenCounts), axum::response::Response> {
        let started = generation_start(&execution_result);
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
        // The engine's counters, for the settlement.
        let counts = response_formatting::TokenCounts::of(std::slice::from_ref(&complete));
        // The records a streamed message makes at its end; a collected one
        // shows no first token. Made once the read succeeded.
        let record = || {
            Metrics::record_streaming_metrics(StreamingMetricsParams {
                router_type: metrics_labels::ROUTER_GRPC,
                backend_type: self.backend_type,
                model_id: &dispatch.model,
                endpoint: metrics_labels::ENDPOINT_MESSAGES,
                ttft: None,
                generation_duration: started.elapsed(),
                input_tokens: Some(u64::from(counts.prompt)),
                cached_tokens: Some(u64::from(counts.cached)),
                output_tokens: u64::from(counts.completion),
            });
        };
        let assignment = messages_request
            .symphony
            .as_ref()
            .ok_or_else(|| error::internal_error("no_table_assigned", NO_TABLE))?;
        // The reading, isolated: a panic in it is this request's 500, not
        // the connection's end (see `utils::isolation`).
        let read = isolated_sync(|| {
            Self::read_messages_with_symphony(
                &dispatch,
                assignment,
                &complete,
                &messages_request,
                stop_decoder,
            )
        });
        match read {
            Ok(Ok(message)) => {
                record();
                Ok((message, counts))
            }
            // The output's own doing: the model wrote arguments no tool_use input holds.
            Ok(Err(SymphonyError::InputRefused(why))) => {
                telemetry::record_refused(telemetry::table_label(assignment.format.name()));
                warn!(function = "process_non_streaming_messages_response", error = %why, "the message's tool_use input cannot carry the model's arguments");
                Err(error::internal_error(
                    "symphony_tool_use_input_refused",
                    why,
                ))
            }
            Ok(Err(e)) => {
                error!(function = "process_non_streaming_messages_response", error = %e, "Symphony failed to read the message");
                Err(error::internal_error("symphony_read_failed", e.to_string()))
            }
            Err(panicked) => {
                let panicked = panicked.read_by(Some(assignment.table()));
                panicked.record(&dispatch.request_id, &dispatch.model);
                Err(reader_panicked(&panicked))
            }
        }
    }

    /// Process non-streaming completion response
    ///
    /// Collects all responses (supports n>1 and batched prompts), decodes tokens
    /// through the stop decoder, applies `echo` and `suffix`, lists each
    /// choice's generated tokens' logprobs when the request asked for them,
    /// and builds one `CompletionResponse` with prompt-major global choice
    /// indices (`prompt_index * n + choice_index`).
    pub async fn process_non_streaming_completion_response(
        &self,
        execution_result: ExecutionResult,
        completion_req: CompletionResponseSpec,
        dispatch: DispatchMetadata,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<CompletionResponse, axum::response::Response> {
        let started = generation_start(&execution_result);
        let request_logprobs = completion_req.logprobs.is_some();
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
        // Membership is per prompt: another prompt's choice cannot stand in
        // for a missing terminal here. Borrow the collected frames so the
        // shared proposal rule needs no intermediate allocation.
        let spec_drafted =
            response_formatting::known_spec_drafted(collected.iter().flatten().map(|complete| {
                (
                    complete.spec_accepted_tokens(),
                    complete.spec_draft_tokens(),
                )
            }))
            .filter(|_| {
                collected.iter().all(|responses| {
                    response_formatting::all_choices_completed(responses, choices_per_prompt)
                })
            });
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

                // The generated tokens' logprobs when the request asked for
                // them, as the engine reports them; their offsets count from
                // the end of the echoed prompt.
                let mut text_offset = if completion_req.echo {
                    u32::try_from(prompt_text.chars().count()).unwrap_or(u32::MAX)
                } else {
                    0
                };
                let logprobs = completion_req.logprobs.and_then(|top_logprobs| {
                    complete.output_logprobs().map(|proto_logprobs| {
                        utils::convert_proto_to_completion_logprobs(
                            proto_logprobs,
                            tokenizer,
                            top_logprobs,
                            &mut text_offset,
                        )
                    })
                });

                choices.push(CompletionChoice {
                    text,
                    index: index_offset + i as u32,
                    logprobs,
                    finish_reason: Some(finish_reason.to_string()),
                    matched_stop,
                });
            }

            total_prompt += prompt_tokens;
            total_cached += cached_tokens;
        }

        // The records a streamed completion makes at its end; a collected
        // one shows no first token.
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.backend_type,
            model_id: &dispatch.model,
            endpoint: metrics_labels::ENDPOINT_COMPLETIONS,
            ttft: None,
            generation_duration: started.elapsed(),
            input_tokens: Some(u64::from(total_prompt)),
            cached_tokens: Some(u64::from(total_cached)),
            output_tokens: u64::from(total_completion),
        });

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
                    spec_drafted,
                }
                .usage(),
            ),
            system_fingerprint: dispatch.weight_version.clone(),
        })
    }
}

/// Where a collected answer's generation clock starts, as its streamed
/// twin's does: at the prefill dispatch of a disaggregated pair (whose
/// prefill may already have run when the answer is collected), or at the
/// single RPC's dispatch. A batch uses its earliest member. The shapes
/// that carry no dispatch instant (an embedding answer, an empty batch)
/// read the clock here, so a generated answer reads it once, at the
/// dispatch.
pub(crate) fn generation_start(execution_result: &ExecutionResult) -> Instant {
    match execution_result {
        ExecutionResult::Single { dispatch_start, .. } => *dispatch_start,
        ExecutionResult::PrefillDecode { pd_timing, .. } => pd_timing.prefill_start,
        ExecutionResult::Batch { results } => results
            .iter()
            .map(generation_start)
            .min()
            .unwrap_or_else(Instant::now),
        ExecutionResult::Embedding { .. } => Instant::now(),
    }
}

/// The 500 a whole response whose reading panicked is answered with: under
/// [`READER_PANICKED_CODE`], and never retried. The panic is the gateway's
/// own, on this output; another attempt would run the engine again for the
/// same reading, so the answer stands (the stream's error event is final by
/// its nature: the response began).
fn reader_panicked(panicked: &Panicked) -> axum::response::Response {
    let mut response = error::internal_error(READER_PANICKED_CODE, panicked.to_string());
    mark_non_retryable(&mut response);
    response
}

/// A choice's output decoded through the stop decoder, as every non-streaming
/// reader decodes it: the text up to a stop, and whether a stop fired at any
/// token or in the tail.
#[derive(Debug)]
struct Decoded {
    text: String,
    stopped: bool,
}

/// Why a choice could not be decoded, in the words the readers report.
#[derive(Debug)]
enum DecodeFailure {
    Process(String),
    Finish(String),
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
/// and a partial stop match), warn about detokenize recoveries, and note a
/// stop the decoder found at any token or in the tail.
fn decode_complete(
    stop_decoder: &mut StopSequenceDecoder,
    output_ids: &[u32],
    request_id: &str,
    index: u32,
) -> Result<Decoded, DecodeFailure> {
    stop_decoder.reset();
    let mut text = String::new();
    let mut stopped = false;
    for &token_id in output_ids {
        let step = stop_decoder
            .process_token_into(token_id, &mut text)
            .map_err(|e| DecodeFailure::Process(e.to_string()))?;
        stopped |= step.stopped();
        if stopped {
            break;
        }
    }
    stop_decoder
        .finish_into(&mut text)
        .map_err(|e| DecodeFailure::Finish(e.to_string()))?;
    utils::warn_detokenize_recoveries(request_id, index, stop_decoder);
    stopped |= stop_decoder.is_stopped();
    Ok(Decoded { text, stopped })
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

/// Usage for a unary Messages response, from the engine's counts: the whole
/// prompt (`prompt_tokens`) and its prefix-cache hit (`cached_tokens`), the
/// counts the settlement and the metrics keep whole. The Anthropic wire splits
/// them: `input_tokens` is the input outside the cache and
/// `cache_read_input_tokens` the hit, their sum the prompt (the HTTP tenant
/// settlement sums them back that way). Cache counters are integers, never
/// null: the wire contract has always-present cache counters and clients do
/// arithmetic on them; `cache_creation_input_tokens` stays zero, as the
/// gateway writes no breakpoint-driven cache entries of its own. The
/// cache-creation breakdown, service tier and inference geography are always
/// written too, and `output_tokens_details` when the turn ran with thinking.
fn messages_usage_from_counts(
    prompt_tokens: u32,
    output_tokens: u32,
    cached_tokens: u32,
    thinking_tokens: Option<u32>,
) -> messages::Usage {
    let usage =
        messages::Usage::from_counts(prompt_tokens.saturating_sub(cached_tokens), output_tokens)
            .with_cache_read_input_tokens(cached_tokens);
    match thinking_tokens {
        Some(thinking_tokens) => usage.with_thinking_tokens(thinking_tokens),
        None => usage,
    }
}

/// A `tool_use` block's `input` as the Messages contract has it: the object
/// itself, or an empty object for a whole value of another kind (a
/// double-encoded or bare string, an array, a number, null), named once in the
/// log under `function`. The request side refuses anything but an object, so a
/// collected answer never carries a value a client could not send back as
/// history. (The stream carries a call's arguments as written.)
fn object_input(
    value: serde_json::Value,
    function: &'static str,
    tool_call_id: &str,
) -> serde_json::Value {
    if value.is_object() {
        return value;
    }
    let kind = match value {
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::Bool(_) => "boolean",
        _ => "null",
    };
    warn!(
        function = function,
        tool_call_id = %tool_call_id,
        error = %format!("the arguments are valid JSON but not an object ({kind})"),
        "Failed to parse tool call arguments, defaulting to empty object"
    );
    serde_json::Value::Object(serde_json::Map::new())
}

/// Every `tool_use` block of a collected answer with its `input` an object
/// ([`object_input`]), in place.
fn object_inputs(content: &mut [messages::ContentBlock], function: &'static str) {
    for block in content {
        if let messages::ContentBlock::ToolUse { id, input, .. } = block {
            if !input.is_object() {
                *input = object_input(std::mem::take(input), function, id);
            }
        }
    }
}

#[cfg(test)]
mod tool_use_input_tests {
    use serde_json::json;
    use symphony::{adapt, Event, Events, FinishReason, Text};

    use super::{messages, object_inputs};

    /// The Symphony reader's collected answer keeps the same contract: the
    /// adapter holds a whole value of any kind as the block's `input` (an
    /// object as it is, `"Paris"` or `[1, 2]` as written, none of them
    /// refused), and the reader leaves every non-object as `{}`.
    #[test]
    fn a_symphony_tool_use_input_is_an_object() {
        for (arguments, expected) in [
            (r#"{"city":"Paris"}"#, json!({"city": "Paris"})),
            (r#""Paris""#, json!({})),
            (r#""{\"city\":\"Paris\"}""#, json!({})),
            ("[1, 2]", json!({})),
        ] {
            let mut events = Events::new();
            events.push(Event::ToolCallStart {
                index: 0,
                id: "call_1".into(),
                name: "f".into(),
                source: Text::new("", 0),
            });
            events.push(Event::ToolCallArguments {
                index: 0,
                json: arguments.into(),
                source: Text::new("", 0),
            });
            events.push(Event::ToolCallEnd {
                index: 0,
                source: Text::new("", 0),
            });
            events.push(Event::Finish {
                reason: FinishReason::ToolCalls,
                tool_calls: 1,
                reasoning_tokens: 0,
            });
            let mut output = adapt::messages::output(&events);
            assert!(
                output.refused.is_empty(),
                "{arguments}: the adapter holds the value"
            );
            object_inputs(&mut output.content, "read_messages_with_symphony");
            let messages::ContentBlock::ToolUse { input, .. } = &output.content[0] else {
                panic!("{arguments}: no tool_use block");
            };
            assert_eq!(input, &expected, "{arguments}");
        }
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
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, None)).unwrap();
        assert_eq!(v["input_tokens"], 25);
        assert_eq!(v["output_tokens"], 150);
        assert_eq!(v["cache_creation_input_tokens"], 0);
        assert_eq!(v["cache_read_input_tokens"], 0);
    }

    /// The engine's whole prompt and its hit become the wire's split: the
    /// input outside the cache beside the hit as the cache-read counter, their
    /// sum the prompt; a miss leaves the whole prompt as input beside an
    /// integer zero; a hit an engine reports above its prompt count leaves no
    /// negative input; the creation counter stays zero throughout.
    #[test]
    fn cache_read_input_tokens_is_the_engines_prefix_cache_hit() {
        let v = serde_json::to_value(messages_usage_from_counts(945, 12, 896, None)).unwrap();
        assert_eq!(v["input_tokens"], 49);
        assert_eq!(v["cache_read_input_tokens"], 896);
        assert_eq!(v["cache_creation_input_tokens"], 0);
        let v = serde_json::to_value(messages_usage_from_counts(945, 12, 0, None)).unwrap();
        assert_eq!(v["input_tokens"], 945);
        assert_eq!(v["cache_read_input_tokens"], 0);
        let v = serde_json::to_value(messages_usage_from_counts(10, 1, 16, None)).unwrap();
        assert_eq!(v["input_tokens"], 0);
        assert_eq!(v["cache_read_input_tokens"], 16);
    }

    /// The public API writes these keys on every message; a typed client
    /// reads a missing `cache_creation` as a decode error, not as zero.
    #[test]
    fn always_present_usage_keys_are_written_without_thinking() {
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, None)).unwrap();
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
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, Some(40))).unwrap();
        assert_eq!(v["output_tokens_details"]["thinking_tokens"], 40);
        let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, Some(0))).unwrap();
        assert_eq!(v["output_tokens_details"]["thinking_tokens"], 0);
    }
}

#[cfg(test)]
mod responses_finish_reason_tests {
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

    /// One decode for every reader: the text and whether a stop fired; the
    /// finish and the matched stop read from it.
    #[test]
    fn decoding_a_choice_reports_its_text_and_stop() {
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

        // Without a stop: the whole text, the engine's finish.
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let decoded = decode_complete(&mut decoder, &[100, 101, 102, 103], "req", 0).unwrap();
        assert_eq!(decoded.text, "Hello world!");
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
}
