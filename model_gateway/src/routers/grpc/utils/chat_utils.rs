//! Chat message processing, tool constraints, and shared utilities for gRPC routers.

use std::{
    collections::HashMap,
    io,
    sync::{Arc, OnceLock},
};

use anyhow::anyhow;
use axum::response::Response;
use bytes::Bytes;
use llm_multimodal::{MediaPartOrder, Modality};
use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    stop::StopSequenceDecoderBuilder,
    traits::{Encoding, Tokenizer},
    StopSequenceDecoder,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::{FunctionCallResponse, StringOrArray, Tool, ToolCall, ToolChoice, ToolChoiceValue},
    generate::GenerateFinishReason,
};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Semaphore};
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{context::RequestContext, multimodal::PlaceholderTokens, ProcessedMessages},
};

/// Type alias for the SSE channel sender used across streaming endpoints.
pub(crate) type SseSender = mpsc::UnboundedSender<Result<Bytes, io::Error>>;

/// Send an SSE error event with a typed error body.
///
/// Produces `data: {"error":{"message":"...","type":"..."}}\n\n` using
/// `serde_json` so that quotes, newlines, and other special characters in the
/// error message are properly escaped.
pub(crate) fn send_error_sse(tx: &SseSender, message: impl ToString, error_type: &str) {
    let chunk = format!(
        "data: {}\n\n",
        json!({
            "error": {
                "message": message.to_string(),
                "type": error_type,
            }
        })
    );
    let _ = tx.send(Ok(Bytes::from(chunk)));
}

/// Resolve tokenizer from registry and cache it in request context.
///
/// This is a helper to avoid duplicating tokenizer resolution logic across
/// preparation stages (chat, generate, embedding).
///
/// Returns the tokenizer Arc, which is also cached in `ctx.state.tokenizer`.
pub(crate) fn resolve_tokenizer(
    ctx: &mut RequestContext,
    stage_name: &str,
) -> Result<Arc<dyn Tokenizer>, Box<Response>> {
    let model_id = ctx.input.model_id.as_str();

    let tokenizer = ctx
        .components
        .tokenizer_registry
        .get(model_id)
        .ok_or_else(|| {
            error!(
                function = %stage_name,
                model = %model_id,
                "Tokenizer not found for model"
            );
            Box::new(error::internal_error(
                "tokenizer_not_found",
                format!("Tokenizer not found for model: {model_id}"),
            ))
        })?;

    // Cache tokenizer in context for reuse in response processing stage
    ctx.state.tokenizer = Some(tokenizer.clone());

    Ok(tokenizer)
}

/// Below this input size (in bytes) the `spawn_blocking` + permit round-trip
/// costs more than the encode itself, so we tokenize inline. Larger prompts —
/// the ones that actually pin a worker thread — are offloaded.
const ENCODE_OFFLOAD_MIN_BYTES: usize = 512;

/// Bounds how many CPU-bound encodes run concurrently on the blocking pool.
/// tokio's blocking pool is otherwise unbounded (grows to 512 threads), so under
/// a burst of large prompts the offloaded encodes would oversubscribe the CPU
/// and starve the very request runtime this offload is meant to protect. Sized
/// to the host's available parallelism.
fn encode_permits() -> &'static Semaphore {
    static SEM: OnceLock<Semaphore> = OnceLock::new();
    SEM.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        Semaphore::new(n)
    })
}

/// Tokenize off the async worker threads so CPU-bound `encode` cannot stall the
/// runtime, bounded by [`encode_permits`] so concurrent offloaded encodes cannot
/// oversubscribe the CPU. Small inputs are encoded inline to avoid the offload
/// round-trip dominating.
pub(crate) async fn encode_blocking(
    tokenizer: Arc<dyn Tokenizer>,
    text: String,
    add_special_tokens: bool,
) -> anyhow::Result<Encoding> {
    if text.len() < ENCODE_OFFLOAD_MIN_BYTES {
        return tokenizer.encode(&text, add_special_tokens);
    }
    let _permit = encode_permits()
        .acquire()
        .await
        .map_err(|e| anyhow!("encode semaphore closed: {e}"))?;
    tokio::task::spawn_blocking(move || tokenizer.encode(&text, add_special_tokens))
        .await
        .map_err(|e| anyhow!("tokenization task failed: {e}"))?
}

/// Process tool call arguments in messages
/// Per Transformers docs, tool call arguments in assistant messages should be dicts
pub(crate) fn process_tool_call_arguments(messages: &mut [Value]) -> Result<(), String> {
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str());
        if role != Some("assistant") {
            continue;
        }

        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) else {
            continue;
        };

        for call in tool_calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let Some(args) = function.get_mut("arguments") else {
                continue;
            };
            let Some(args_str) = args.as_str() else {
                continue;
            };

            // Parse JSON string to object (like Python json.loads)
            match serde_json::from_str::<Value>(args_str) {
                Ok(parsed) => *args = parsed,
                Err(e) => {
                    return Err(format!(
                        "Failed to parse tool call arguments as JSON: '{args_str}'. Error: {e}"
                    ))
                }
            }
        }
    }
    Ok(())
}

/// Process messages based on content format for ANY message type
#[cfg(test)]
pub(crate) fn process_content_format(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
) -> Result<Vec<Value>, String> {
    process_content_format_with_order(
        messages,
        content_format,
        placeholder_tokens,
        MediaPartOrder::MediaFirst,
    )
}

const REASONING_EFFORT_KEY: &str = "reasoning_effort";
const TOOL_CHOICE_KEY: &str = "tool_choice";
const RESPONSE_FORMAT_KEY: &str = "response_format";

/// Merge request-level reasoning, tool, and output-format controls with any
/// request `chat_template_kwargs`, forwarding each verbatim. The chat template
/// owns interpretation (level→value mapping, `tool_choice`/`response_format`
/// rendering, defaulting, and validation); an explicit `chat_template_kwargs`
/// entry wins.
fn build_chat_template_kwargs(request: &ChatCompletionRequest) -> HashMap<String, Value> {
    let kwargs_capacity = 3 + request.chat_template_kwargs.as_ref().map_or(0, |k| k.len());
    let mut combined = HashMap::with_capacity(kwargs_capacity);
    if let Some(reasoning_effort) = &request.reasoning_effort {
        combined.insert(
            REASONING_EFFORT_KEY.to_string(),
            Value::String(reasoning_effort.clone()),
        );
    }
    if let Some(tool_choice) = &request.tool_choice {
        if let Ok(value) = serde_json::to_value(tool_choice) {
            combined.insert(TOOL_CHOICE_KEY.to_string(), value);
        }
    }
    if let Some(response_format) = &request.response_format {
        if let Ok(value) = serde_json::to_value(response_format) {
            combined.insert(RESPONSE_FORMAT_KEY.to_string(), value);
        }
    }
    if let Some(template_kwargs) = &request.chat_template_kwargs {
        combined.extend(template_kwargs.clone());
    }
    combined
}

fn process_content_format_with_order(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| {
            let mut message_json = serde_json::to_value(message)
                .map_err(|e| format!("Failed to serialize message: {e}"))?;

            if let Some(obj) = message_json.as_object_mut() {
                // skip_serializing_none omits content when None; restore it as
                // `null` — the OpenAI-faithful representation for a tool-call-only
                // assistant turn — so the chat template renders it correctly.
                if obj.get("role").and_then(|v| v.as_str()) == Some("assistant")
                    && !obj.contains_key("content")
                {
                    obj.insert("content".to_string(), Value::Null);
                }

                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(
                        content_value,
                        content_format,
                        placeholder_tokens,
                        media_order,
                    )?;
                }
            }

            Ok(message_json)
        })
        .collect()
}

/// Transform a single content field based on content format.
///
/// For `String` templates, every media URL is replaced with its
/// modality-specific structural anchor when placeholders are configured. The
/// public preprocessing bindings do not have model metadata, so their legacy
/// `None` path continues to omit media parts.
///
/// Media parts are emitted before text by default, matching vLLM's
/// `interleave_mm_strings=false` behavior and preserving existing VQA behavior.
/// TML/Inkling opts into authored order because part order is protocol-visible
/// in its canonical renderer.
fn transform_content_field(
    content_value: &mut Value,
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<(), String> {
    let Some(content_array) = content_value.as_array() else {
        return Ok(()); // Not multimodal, keep as-is
    };

    match content_format {
        ChatTemplateContentFormat::String => {
            // Replace media parts with placeholders. The default branch builds
            // separate media/text buckets for vLLM-compatible front placement;
            // the TML branch appends each rendered part as it is encountered.
            let mut media_parts: Vec<String> = Vec::new();
            let mut text_parts: Vec<String> = Vec::new();
            let mut authored_parts: Vec<String> = Vec::new();
            for part in content_array {
                let Some(obj) = part.as_object() else {
                    continue;
                };
                match obj.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
                            match media_order {
                                MediaPartOrder::MediaFirst => text_parts.push(t.to_string()),
                                MediaPartOrder::Authored => authored_parts.push(t.to_string()),
                            }
                        }
                    }
                    Some(type_name @ ("image_url" | "video_url" | "audio_url" | "input_audio")) => {
                        let modality = modality_for_chat_part(type_name).ok_or_else(|| {
                            format!("unsupported media content part type: {type_name}")
                        })?;
                        let Some(tokens) = placeholder_tokens else {
                            continue;
                        };
                        let placeholder = tokens.get(modality).ok_or_else(|| {
                            format!(
                                "missing {modality} placeholder for string-format chat template"
                            )
                        })?;
                        match media_order {
                            MediaPartOrder::MediaFirst => {
                                media_parts.push(placeholder.to_string());
                            }
                            MediaPartOrder::Authored => {
                                authored_parts.push(placeholder.to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }

            let ordered: Vec<String> = match media_order {
                MediaPartOrder::MediaFirst => media_parts.into_iter().chain(text_parts).collect(),
                MediaPartOrder::Authored => authored_parts,
            };
            if !ordered.is_empty() {
                *content_value = Value::String(ordered.join("\n"));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
            // Replace media URLs with simple type placeholders
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| obj.get("type")?.as_str())
                        .and_then(|type_str| match type_str {
                            "image_url" => Some(json!({"type": "image"})),
                            "video_url" => Some(json!({"type": "video"})),
                            "audio_url" | "input_audio" => Some(json!({"type": "audio"})),
                            _ => None,
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            // The default branch places media before the remaining parts,
            // matching vLLM front placement. `partition` is stable, so relative
            // order within media and within text is kept. TML skips partitioning.
            let ordered_parts = match media_order {
                MediaPartOrder::Authored => processed_parts,
                MediaPartOrder::MediaFirst => {
                    let (mut media, rest): (Vec<Value>, Vec<Value>) =
                        processed_parts.into_iter().partition(|p| {
                            matches!(
                                p.get("type").and_then(|t| t.as_str()),
                                Some("image") | Some("video") | Some("audio")
                            )
                        });
                    media.extend(rest);
                    media
                }
            };

            *content_value = Value::Array(ordered_parts);
        }
    }

    Ok(())
}

fn modality_for_chat_part(type_name: &str) -> Option<Modality> {
    match type_name {
        "image_url" | "image" => Some(Modality::Image),
        "audio_url" | "input_audio" | "audio" => Some(Modality::Audio),
        "video_url" | "video" => Some(Modality::Video),
        _ => None,
    }
}

/// Filter tools based on tool_choice (generic helper)
///
/// Returns filtered tools if filtering is needed, otherwise returns None.
/// Used by both Chat API and Responses API (Harmony) for constraint generation.
pub(crate) fn filter_tools_by_tool_choice(
    tools: &[Tool],
    tool_choice: Option<&ToolChoice>,
) -> Option<Vec<Tool>> {
    match tool_choice {
        Some(ToolChoice::AllowedTools { tools: allowed, .. }) => {
            let allowed_names: std::collections::HashSet<&str> =
                allowed.iter().filter_map(|t| t.function_name()).collect();
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| allowed_names.contains(t.function.name.as_str()))
                .cloned()
                .collect();
            Some(filtered)
        }
        Some(ToolChoice::Function { function, .. }) => {
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| t.function.name == function.name)
                .cloned()
                .collect();
            Some(filtered)
        }
        _ => None, // No filtering needed
    }
}

/// Filter ChatCompletionRequest by tool_choice
///
/// Returns a reference to the original request if no filtering needed,
/// otherwise returns a cloned request with filtered tools.
///
/// Note: Tool existence is validated earlier in ChatCompletionRequest::validate(),
/// so this function assumes tool_choice references valid tools.
pub(crate) fn filter_chat_request_by_tool_choice(
    body: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    if let Some(tools) = &body.tools {
        if let Some(filtered_tools) = filter_tools_by_tool_choice(tools, body.tool_choice.as_ref())
        {
            let mut filtered_body = body.clone();
            filtered_body.tools = Some(filtered_tools);
            return std::borrow::Cow::Owned(filtered_body);
        }
    }

    // No filtering needed - return original request
    std::borrow::Cow::Borrowed(body)
}

/// Process chat messages and apply template (shared by both routers)
/// Requires HuggingFace tokenizer with chat template support
pub fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
    image_placeholder: Option<&str>,
) -> Result<ProcessedMessages, String> {
    let placeholder_tokens = image_placeholder.map(|token| {
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, token.to_string());
        placeholders
    });
    process_chat_messages_with_placeholders(
        request,
        tokenizer,
        placeholder_tokens.as_ref(),
        MediaPartOrder::MediaFirst,
    )
}

pub(crate) fn process_chat_messages_with_placeholders(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<ProcessedMessages, String> {
    let formatted_text = {
        // Get content format and transform messages accordingly
        let content_format = tokenizer.chat_template_content_format();
        let mut transformed_messages = process_content_format_with_order(
            &request.messages,
            content_format,
            placeholder_tokens,
            media_order,
        )?;

        // Process tool call arguments in assistant messages
        process_tool_call_arguments(&mut transformed_messages)?;

        // Convert tools to JSON values for template processing
        let tools_json: Option<Vec<Value>> = request
            .tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()
            .map_err(|e| format!("Failed to serialize tools: {e}"))?;

        let combined_template_kwargs = build_chat_template_kwargs(request);

        let final_template_kwargs = if combined_template_kwargs.is_empty() {
            None
        } else {
            Some(&combined_template_kwargs)
        };

        let params = ChatTemplateParams {
            add_generation_prompt: true,
            tools: tools_json.as_deref(),
            template_kwargs: final_template_kwargs,
            // Project the protocol-level DeepSeek `thinking` field and OpenAI
            // compatibility values onto the model's thinking toggle.
            // An explicit chat_template_kwargs toggle still wins (in apply).
            thinking: request.thinking_preference(),
            ..Default::default()
        };

        // Handle assistant prefix for continue_final_message
        let assistant_prefix = if request.continue_final_message
            && !transformed_messages.is_empty()
            && transformed_messages
                .last()
                .and_then(|msg| msg.get("role"))
                .and_then(|v| v.as_str())
                == Some("assistant")
        {
            // Pop the last message to handle it separately — guarded by !is_empty() check above
            let Some(last_msg) = transformed_messages.pop() else {
                return Ok(ProcessedMessages {
                    text: String::new(),
                    stop_sequences: request.stop.clone(),
                });
            };
            last_msg
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

        // Apply chat template with the (now possibly shorter) list of messages
        let rendered = tokenizer
            .apply_chat_template(&transformed_messages, params)
            .map_err(|e| format!("Failed to apply chat template: {e}"))?;

        // Append assistant prefix if we have one
        if let Some(prefix) = assistant_prefix {
            format!("{rendered}{prefix}")
        } else {
            rendered
        }
    };

    Ok(ProcessedMessages {
        text: formatted_text,
        stop_sequences: request.stop.clone(),
    })
}

/// Create a StopSequenceDecoder from stop parameters
pub fn create_stop_decoder(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    skip_special_tokens: bool,
    no_stop_trim: bool,
    ignore_eos: bool,
) -> StopSequenceDecoder {
    // Extract stop sequences
    let stop_sequences: Vec<String> = match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };

    // Build stop sequence decoder
    let mut builder =
        StopSequenceDecoderBuilder::new(tokenizer.clone()).skip_special_tokens(skip_special_tokens);

    // Add stop sequences (visible if no_stop_trim is true, hidden otherwise)
    for seq in stop_sequences {
        builder = if no_stop_trim {
            builder.visible_stop_sequence(seq)
        } else {
            builder.stop_sequence(seq)
        };
    }

    // Collect stop token IDs: EOS from tokenizer (unless ignore_eos) + user-provided.
    // EOS tokens come from generation_config.json and are stripped at the token ID
    // level before decoding, matching vllm/sglang behavior.
    // When ignore_eos=true, EOS tokens are not added — the backend continues past EOS.
    let eos_ids = if ignore_eos {
        &[] as &[u32]
    } else {
        tokenizer.eos_token_ids()
    };
    for &token_id in eos_ids
        .iter()
        .chain(stop_token_ids.map(|ids| ids.as_slice()).unwrap_or_default())
    {
        builder = if no_stop_trim {
            builder.visible_stop_token(token_id)
        } else {
            builder.stop_token(token_id)
        };
    }

    builder.build()
}

/// Parse tool calls from JSON schema constrained response
pub(crate) fn parse_json_schema_response(
    processed_text: &str,
    tool_choice: Option<&ToolChoice>,
    model: &str,
    history_tool_calls_count: usize,
) -> (Option<Vec<ToolCall>>, String) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            // Specific function: Parse parameters directly
            match serde_json::from_str::<Value>(processed_text) {
                Ok(params) => {
                    let tool_call = ToolCall {
                        id: generate_tool_call_id(
                            model,
                            &function.name,
                            0,
                            history_tool_calls_count,
                        ),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: function.name.clone(),
                            arguments: Some(
                                serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
                            ),
                        },
                    };
                    (Some(vec![tool_call]), String::new())
                }
                Err(e) => {
                    error!("Failed to parse specific function parameters: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required))
        | Some(ToolChoice::AllowedTools { .. }) => {
            // Required mode: Parse array of tool calls
            match serde_json::from_str::<Vec<Value>>(processed_text) {
                Ok(parsed_array) => {
                    let spec_tool_calls: Vec<ToolCall> = parsed_array
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, item)| {
                            let obj = item.as_object()?;
                            let name = obj.get("name")?.as_str()?.to_string();
                            let parameters = obj.get("parameters")?;

                            Some(ToolCall {
                                id: generate_tool_call_id(
                                    model,
                                    &name,
                                    i,
                                    history_tool_calls_count,
                                ),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name,
                                    arguments: Some(
                                        serde_json::to_string(parameters)
                                            .unwrap_or_else(|_| "{}".to_string()),
                                    ),
                                },
                            })
                        })
                        .collect();
                    (Some(spec_tool_calls), String::new())
                }
                Err(e) => {
                    error!("Failed to parse required tool call array: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        _ => (None, processed_text.to_string()),
    }
}

/// Count the number of tool calls in the request message history
/// This is used for KimiK2 format which needs globally unique indices
pub(crate) fn get_history_tool_calls_count(request: &ChatCompletionRequest) -> usize {
    request
        .messages
        .iter()
        .filter_map(|msg| {
            if let ChatMessage::Assistant { tool_calls, .. } = msg {
                tool_calls.as_ref().map(|calls| calls.len())
            } else {
                None
            }
        })
        .sum()
}

/// Generate a tool call ID based on model format
///
/// # Arguments
/// * `model` - Model name to determine ID format
/// * `tool_name` - Name of the tool being called
/// * `tool_index` - Index of this tool call within the current message
/// * `history_count` - Number of tool calls in previous messages
///
/// # Returns
/// A unique ID string:
/// - Kimi-K3 (XTML): `{name}:{tool_index}` — an opaque, per-message zero-based
///   ordinal. K3 never renders the id into the prompt and matches tool results
///   back to calls by opaque id equality scoped to the most recent assistant
///   message, so the id carries no `functions.` prefix and no history offset.
///   Mirrors the K3 reference decode parser (`{tool_name}:{xtml_index - 1}`).
/// - Kimi-K2: `functions.{name}:{history_count + tool_index}` (globally unique).
/// - others: `call_{24-char-uuid}`.
pub(crate) fn generate_tool_call_id(
    model: &str,
    tool_name: &str,
    tool_index: usize,
    history_count: usize,
) -> String {
    // Case-insensitive substring checks without allocation.
    let is_kimi = model
        .as_bytes()
        .windows(4) // "kimi".len()
        .any(|window| window.eq_ignore_ascii_case(b"kimi"));

    if !is_kimi {
        // Standard OpenAI format: call_{24-char-uuid}
        return format!("call_{}", &Uuid::now_v7().simple().to_string()[..24]);
    }

    let is_k3 = model
        .as_bytes()
        .windows(2) // "k3".len()
        .any(|window| window.eq_ignore_ascii_case(b"k3"));

    if is_k3 {
        // Kimi-K3 (XTML) opaque format: {name}:{per-message zero-based ordinal}.
        format!("{tool_name}:{tool_index}")
    } else {
        // Kimi-K2 format: functions.{name}:{global_index}.
        format!("functions.{}:{}", tool_name, history_count + tool_index)
    }
}

/// Parse finish_reason string into GenerateFinishReason enum
///
/// Uses serde to deserialize the finish_reason, which handles all tagged variants automatically.
/// The GenerateFinishReason enum is tagged with `#[serde(tag = "type", rename_all = "lowercase")]`,
/// so it expects JSON objects like:
/// - `{"type":"stop"}` -> Stop
/// - `{"type":"length","length":100}` -> Length { length: 100 }
/// - Any other JSON -> Other(...)
///
/// For backward compatibility, also handles simple string "stop" -> Stop
pub(crate) fn parse_finish_reason(
    reason_str: &str,
    completion_tokens: u32,
) -> GenerateFinishReason {
    if reason_str == "stop" {
        return GenerateFinishReason::Stop {
            finish_type: openai_protocol::generate::GenerateFinishType::Stop,
        };
    }

    if reason_str == "length" {
        return GenerateFinishReason::Length {
            finish_type: openai_protocol::generate::GenerateFinishType::Length,
            length: completion_tokens,
        };
    }

    match serde_json::from_str::<GenerateFinishReason>(reason_str) {
        Ok(finish_reason) => finish_reason,
        Err(_) => match serde_json::from_str::<Value>(reason_str) {
            Ok(json_value) => GenerateFinishReason::Other(json_value),
            Err(_) => GenerateFinishReason::Other(Value::String(reason_str.to_string())),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::fs as test_fs;

    use llm_tokenizer::{chat_template::ChatTemplateContentFormat, HuggingFaceTokenizer};
    use openai_protocol::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        common::{AudioUrl, ContentPart, ImageUrl, InputAudio, VideoUrl},
    };
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    fn placeholders(entries: &[(Modality, &str)]) -> PlaceholderTokens {
        let mut tokens = PlaceholderTokens::default();
        for (modality, token) in entries {
            tokens.insert(*modality, (*token).to_string());
        }
        tokens
    }

    fn deepseek_v4_tokenizer() -> (TempDir, HuggingFaceTokenizer) {
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": { "hello": 0, "<s>": 1, "</s>": 2 },
                "merges": []
            }
        }"#;

        let dir = TempDir::new().unwrap();
        let tokenizer_path = dir.path().join("tokenizer.json");
        test_fs::write(&tokenizer_path, TOKENIZER_JSON).unwrap();
        test_fs::write(
            dir.path().join("config.json"),
            r#"{"architectures":["DeepseekV4ForCausalLM"]}"#,
        )
        .unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
        (dir, tokenizer)
    }

    fn deepseek_request(extra: Value) -> ChatCompletionRequest {
        let mut request = json!({
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hello"}],
        });
        request.as_object_mut().unwrap().extend(
            extra
                .as_object()
                .expect("extra request fields must be an object")
                .clone(),
        );
        serde_json::from_value(request).unwrap()
    }

    #[test]
    fn deepseek_v4_official_thinking_request_controls_prompt() {
        let (_dir, tokenizer) = deepseek_v4_tokenizer();

        let default_prompt = process_chat_messages(&deepseek_request(json!({})), &tokenizer, None)
            .unwrap()
            .text;
        assert!(default_prompt.ends_with("<think>"));

        let disabled_prompt = process_chat_messages(
            &deepseek_request(json!({"thinking": {"type": "disabled"}})),
            &tokenizer,
            None,
        )
        .unwrap()
        .text;
        assert!(disabled_prompt.ends_with("</think>"));

        let max_prompt = process_chat_messages(
            &deepseek_request(json!({"reasoning_effort": "xhigh"})),
            &tokenizer,
            None,
        )
        .unwrap()
        .text;
        assert!(max_prompt.contains("Reasoning Effort: Absolute maximum"));
        assert!(max_prompt.ends_with("<think>"));
    }

    #[test]
    fn test_transform_messages_string_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Hello".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "World".to_string(),
                },
            ]),
            name: None,
        }];

        let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Media is hoisted and uses the modality-specific placeholder.
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "<|image|>\nHello\nWorld"
        );
        assert_eq!(transformed_message["role"].as_str().unwrap(), "user");
    }

    #[test]
    fn test_transform_messages_string_format_without_placeholders_omits_media() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, None).unwrap();

        assert_eq!(result[0]["content"], "Describe this");
    }

    #[test]
    fn test_transform_messages_string_format_with_video_placeholder() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Watch this".to_string(),
                },
                ContentPart::VideoUrl {
                    video_url: VideoUrl {
                        url: "https://example.com/video.mp4".to_string(),
                    },
                },
            ]),
            name: None,
        }];

        let tokens = placeholders(&[(Modality::Video, "<|video|>")]);
        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();

        // Media placeholder is emitted before the text (vLLM front placement).
        assert_eq!(
            result[0]["content"].as_str().unwrap(),
            "<|video|>\nWatch this"
        );
    }

    #[test]
    fn test_transform_messages_string_format_uses_per_modality_placeholders() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe and transcribe".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "image".to_string(),
                        detail: None,
                    },
                },
                ContentPart::AudioUrl {
                    audio_url: AudioUrl {
                        url: "audio".to_string(),
                    },
                },
            ]),
            name: None,
        }];
        let tokens = placeholders(&[(Modality::Image, "<image>"), (Modality::Audio, "<audio>")]);

        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();

        assert_eq!(
            result[0]["content"],
            "<image>\n<audio>\nDescribe and transcribe"
        );

        let openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI, Some(&tokens))
                .unwrap();
        assert_eq!(
            openai[0]["content"],
            json!([
                {"type": "image"},
                {"type": "audio"},
                {"type": "text", "text": "Describe and transcribe"}
            ])
        );
    }

    #[test]
    fn test_transform_messages_input_audio_uses_audio_placeholder() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Transcribe this".to_string(),
                },
                ContentPart::InputAudio {
                    input_audio: InputAudio {
                        data: "UklGRg==".to_string(),
                        format: "wav".to_string(),
                    },
                },
            ]),
            name: None,
        }];
        let tokens = placeholders(&[(Modality::Audio, "<audio>")]);

        let string =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();
        assert_eq!(string[0]["content"], "<audio>\nTranscribe this");

        let openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI, Some(&tokens))
                .unwrap();
        assert_eq!(
            openai[0]["content"],
            json!([
                {"type": "audio"},
                {"type": "text", "text": "Transcribe this"}
            ])
        );
    }

    #[test]
    fn test_transform_messages_string_format_rejects_missing_modality_placeholder() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::AudioUrl {
                audio_url: AudioUrl {
                    url: "audio".to_string(),
                },
            }]),
            name: None,
        }];
        let tokens = placeholders(&[(Modality::Image, "<image>")]);

        let error =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap_err();

        assert!(error.contains("missing audio placeholder"));
    }

    #[test]
    fn test_transform_messages_openai_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this image:".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let result =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Media URLs replaced with simple type placeholders, and the image is
        // hoisted before the text (vLLM front placement; image-after-question
        // degrades VQA accuracy).
        let content_array = transformed_message["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);

        // Image part comes first now.
        assert_eq!(content_array[0], json!({"type": "image"}));

        // Text part follows, unchanged.
        assert_eq!(content_array[1]["type"], "text");
        assert_eq!(content_array[1]["text"], "Describe this image:");
    }

    #[test]
    fn test_transform_messages_simple_string_content() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Text("Simple text message".to_string()),
            name: None,
        }];

        let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Simple string content should remain unchanged
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Simple text message"
        );
    }

    #[test]
    fn test_transform_messages_multiple_messages() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("System prompt".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "User message".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: None,
                        },
                    },
                ]),
                name: None,
            },
        ];

        let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();

        assert_eq!(result.len(), 2);

        // System message should remain unchanged
        assert_eq!(result[0]["role"].as_str().unwrap(), "system");
        assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");

        // User message retains the media anchor.
        assert_eq!(result[1]["role"].as_str().unwrap(), "user");
        assert_eq!(
            result[1]["content"].as_str().unwrap(),
            "<|image|>\nUser message"
        );
    }

    #[test]
    fn test_transform_messages_empty_text_parts() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            }]),
            name: None,
        }];

        let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        assert_eq!(transformed_message["content"], "<|image|>");
    }

    #[test]
    fn test_transform_messages_mixed_content_types() {
        let messages = vec![
            ChatMessage::User {
                content: MessageContent::Text("Plain text".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "With image".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
        let result_string =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();

        assert_eq!(result_string.len(), 2);
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(
            result_string[1]["content"].as_str().unwrap(),
            "<|image|>\nWith image"
        );

        let result_openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();

        assert_eq!(result_openai.len(), 2);
        assert_eq!(result_openai[0]["content"].as_str().unwrap(), "Plain text");

        let content_array = result_openai[1]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        // Image hoisted before text.
        assert_eq!(content_array[0], json!({"type": "image"}));
        assert_eq!(content_array[1]["type"], "text");
    }

    #[test]
    fn test_media_hoisted_before_text_openai() {
        // Real MMBench shape: [question text, image] must render image-first.
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Question: ...\nAnswer with only the option letter.".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/jpeg;base64,XXX".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        let result =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();
        let arr = result[0]["content"].as_array().unwrap();
        assert_eq!(arr[0], json!({"type": "image"}));
        assert_eq!(arr[1]["type"], "text");
    }

    #[test]
    fn test_media_hoisted_before_text_string() {
        // String-format template: placeholder prepended, matching vLLM exactly.
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Question?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/jpeg;base64,XXX".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        let tokens = placeholders(&[(Modality::Image, "<|image_pad|>")]);
        let result =
            process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
                .unwrap();
        assert_eq!(
            result[0]["content"].as_str().unwrap(),
            "<|image_pad|>\nQuestion?"
        );
    }

    #[test]
    fn test_media_first_stable_and_multi() {
        // Multiple media + text keep relative order within each group, media first.
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "a".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "i1".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "b".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "i2".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        let result =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();
        let arr = result[0]["content"].as_array().unwrap();
        assert_eq!(arr[0], json!({"type": "image"}));
        assert_eq!(arr[1], json!({"type": "image"}));
        assert_eq!(arr[2]["text"], "a");
        assert_eq!(arr[3]["text"], "b");
    }

    #[test]
    fn test_tml_preserves_authored_multipart_order_openai() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "question".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "image".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        let expected = json!([
            {"type": "text", "text": "question"},
            {"type": "image"}
        ]);
        let result = process_content_format_with_order(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::Authored,
        )
        .unwrap();
        assert_eq!(result[0]["content"], expected);
    }

    #[test]
    fn test_absent_assistant_content_renders_null() {
        let messages = vec![ChatMessage::Assistant {
            content: None,
            name: None,
            tool_calls: None,
            reasoning_content: None,
        }];

        let result = process_content_format_with_order(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert!(result[0]["content"].is_null());
    }

    #[test]
    fn test_tml_preserves_authored_multipart_order_string() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "question".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "image".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];
        let tokens = placeholders(&[(Modality::Image, "<image>")]);

        let result = process_content_format_with_order(
            &messages,
            ChatTemplateContentFormat::String,
            Some(&tokens),
            MediaPartOrder::Authored,
        )
        .unwrap();

        assert_eq!(result[0]["content"], "question\n<image>");
    }

    fn effort_request(reasoning_effort: Option<&str>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "inkling-chat".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hello".to_string()),
                name: None,
            }],
            reasoning_effort: reasoning_effort.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn reasoning_effort_is_forwarded_verbatim() {
        // The chat template owns level->value mapping, defaulting, and validation;
        // the router forwards the string and omits it when absent so the template
        // applies its own default.
        let kwargs = build_chat_template_kwargs(&effort_request(Some("high")));
        assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("high")));

        let kwargs = build_chat_template_kwargs(&effort_request(None));
        assert!(!kwargs.contains_key(REASONING_EFFORT_KEY));
    }

    #[test]
    fn chat_template_kwargs_override_top_level_effort() {
        let mut request = effort_request(Some("high"));
        request.chat_template_kwargs = Some(HashMap::from([
            (REASONING_EFFORT_KEY.to_string(), json!("low")),
            ("custom".to_string(), json!(true)),
        ]));
        let kwargs = build_chat_template_kwargs(&request);
        assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("low")));
        assert_eq!(kwargs.get("custom"), Some(&Value::Bool(true)));
    }

    #[test]
    fn tool_and_output_controls_are_forwarded_to_renderer() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "required",
            "response_format": {"type": "json_object"}
        }))
        .unwrap();
        let kwargs = build_chat_template_kwargs(&request);
        assert_eq!(kwargs.get(TOOL_CHOICE_KEY), Some(&json!("required")));
        assert_eq!(
            kwargs.get(RESPONSE_FORMAT_KEY),
            Some(&json!({"type": "json_object"}))
        );

        // Absent both, neither key is forwarded.
        let bare: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let kwargs = build_chat_template_kwargs(&bare);
        assert!(!kwargs.contains_key(TOOL_CHOICE_KEY));
        assert!(!kwargs.contains_key(RESPONSE_FORMAT_KEY));
    }

    /// End-to-end: run a real MMBench-shaped `[text, image]` message through the
    /// full SMG pipeline (`process_content_format` + the actual model chat
    /// template) and assert the rendered prompt places the image BEFORE the
    /// question. Proves the fix end-to-end without needing the GPU worker.
    /// Ignored by default (needs the real template); run with:
    ///   QWEN35_CHAT_TEMPLATE=/path/chat_template.jinja cargo test -p smg \
    ///     render_image_before_question -- --ignored --nocapture
    #[test]
    #[ignore = "needs real chat_template.jinja via QWEN35_CHAT_TEMPLATE"]
    fn test_render_image_before_question_real_template() {
        use llm_tokenizer::chat_template::{
            detect_chat_template_content_format, ChatTemplateParams, ChatTemplateProcessor,
        };

        let Ok(path) = std::env::var("QWEN35_CHAT_TEMPLATE") else {
            return; // skip when not provided
        };
        let template = std::fs::read_to_string(&path).expect("read template");
        let format = detect_chat_template_content_format(&template);

        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Question: Which description is correct?\n\
                           Answer with only the option letter (A/B/C/D)."
                        .to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/jpeg;base64,XXX".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        let tokens = placeholders(&[(Modality::Image, "<|image_pad|>")]);
        let transformed = process_content_format(&messages, format, Some(&tokens)).unwrap();

        let mut kwargs = HashMap::new();
        kwargs.insert("enable_thinking".to_string(), json!(false));
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let rendered = ChatTemplateProcessor::new(template)
            .unwrap()
            .apply_chat_template(&transformed, params)
            .unwrap();

        let vstart = rendered
            .find("<|vision_start|>")
            .expect("rendered prompt has <|vision_start|>");
        let qpos = rendered
            .find("Question:")
            .expect("rendered prompt has the question");
        assert!(
            vstart < qpos,
            "image must precede the question (vstart={vstart}, qpos={qpos}).\n--- rendered ---\n{rendered}"
        );
    }

    #[test]
    fn test_generate_tool_call_id_kimi_k3_opaque_format() {
        // K3: `{name}:{per-message zero-based ordinal}` — no `functions.` prefix,
        // no history offset (matches the K3 reference decode parser).
        assert_eq!(
            generate_tool_call_id("moonshotai/Kimi-K3", "get_weather", 0, 0),
            "get_weather:0"
        );
        assert_eq!(
            generate_tool_call_id("kimi_k3", "get_weather", 1, 5),
            "get_weather:1"
        );
    }

    #[test]
    fn test_generate_tool_call_id_kimi_k2_unchanged() {
        // K2 keeps the globally unique `functions.{name}:{history + index}` form.
        assert_eq!(
            generate_tool_call_id("moonshotai/Kimi-K2-Instruct", "get_weather", 0, 0),
            "functions.get_weather:0"
        );
        assert_eq!(
            generate_tool_call_id("Kimi-K2", "get_weather", 1, 2),
            "functions.get_weather:3"
        );
    }

    #[test]
    fn test_generate_tool_call_id_non_kimi_uses_uuid() {
        let id = generate_tool_call_id("gpt-4o", "get_weather", 0, 0);
        assert!(id.starts_with("call_"), "got: {id}");
        assert!(!id.contains("get_weather"), "got: {id}");
    }
}
