//! Message API utilities for converting Anthropic Messages API types
//! into the internal chat template format.
//!
//! Parallel to `chat_utils.rs` but works with `CreateMessageRequest` / `InputMessage`
//! instead of `ChatCompletionRequest` / `ChatMessage`.
#![allow(dead_code)] // wired in follow-up PR (pipeline factory)

use llm_multimodal::{MediaPartOrder, Modality};
use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    traits::Tokenizer,
};
use openai_protocol::{
    common::{self, StringOrArray, Tool as ChatTool, ToolChoice as ChatToolChoice},
    messages::{
        self, CreateMessageRequest, InputContent, InputContentBlock, InputMessage, SystemContent,
        ThinkingConfig, ToolResultContent,
    },
};
use serde_json::{json, Value};

use super::chat_utils;
use crate::routers::grpc::{multimodal::PlaceholderTokens, ProcessedMessages};

// ============================================================================
// Top-level processing function
// ============================================================================

/// Process messages from a CreateMessageRequest and apply the chat template.
///
/// Parallel to `process_chat_messages()` in chat_utils, but works with
/// Anthropic Messages API types. Converts InputMessages to JSON values
/// that the chat template expects, then applies the template.
pub fn process_messages(
    request: &CreateMessageRequest,
    tokenizer: &dyn Tokenizer,
    chat_tools: Option<&[ChatTool]>,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<ProcessedMessages, String> {
    let content_format = tokenizer.chat_template_content_format();

    // Step 1: Convert InputMessages to chat template JSON values
    let mut transformed_messages = process_message_content_format(
        &request.messages,
        content_format,
        placeholder_tokens,
        media_order,
    )?;

    // Step 2: Prepend system message if present
    if let Some(system) = &request.system {
        let system_text = match system {
            SystemContent::String(s) => s.clone(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| {
                    let messages::SystemContentBlock::Text(tb) = b;
                    tb.text.as_str()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };
        transformed_messages.insert(0, json!({"role": "system", "content": system_text}));
    }

    // Step 3: Process tool call arguments in assistant messages (reuse from chat_utils)
    chat_utils::process_tool_call_arguments(&mut transformed_messages)?;

    // Step 4: Serialize tools to JSON values for template processing
    let tools_json: Option<Vec<Value>> = chat_tools
        .map(|tools| {
            tools
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|e| format!("Failed to serialize tools: {e}"))?;

    // Step 5: Project the Anthropic ThinkingConfig onto a thinking on/off
    // preference. Adaptive is treated as "thinking on"; the model decides
    // whether to actually emit it. The tokenizer applies this under the model's
    // own toggle key (`enable_thinking`/`thinking`) in `apply`.
    let thinking = match &request.thinking {
        Some(ThinkingConfig::Enabled { .. } | ThinkingConfig::Adaptive { .. }) => Some(true),
        Some(ThinkingConfig::Disabled) => Some(false),
        None => None, // Let template use its default behavior
    };

    // Step 6: Apply chat template
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: tools_json.as_deref(),
        thinking,
        ..Default::default()
    };

    let formatted_text = tokenizer
        .apply_chat_template(&transformed_messages, params)
        .map_err(|e| format!("Failed to apply chat template: {e}"))?;

    // Step 7: Build ProcessedMessages
    let stop_sequences = request
        .stop_sequences
        .as_ref()
        .map(|seqs| StringOrArray::Array(seqs.clone()));

    Ok(ProcessedMessages {
        text: formatted_text,
        stop_sequences,
    })
}

// ============================================================================
// InputMessage → JSON conversion
// ============================================================================

/// Convert InputMessage array to JSON Values for the chat template.
///
/// Mirrors `process_content_format()` in chat_utils but works with
/// `InputMessage` instead of `ChatMessage`.
///
/// Key conversion rules:
/// - User messages with String content → `{"role": "user", "content": "text"}`
/// - User messages with Blocks → text/image blocks stay as user content,
///   ToolResult blocks become separate `{"role": "tool", ...}` messages
/// - Assistant messages → `{"role": "assistant", "content": ..., "tool_calls": [...], "reasoning_content": ...}`
pub(crate) fn process_message_content_format(
    messages: &[InputMessage],
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<Vec<Value>, String> {
    messages.iter().try_fold(Vec::new(), |mut result, message| {
        match message.role {
            messages::Role::User => {
                convert_user_message(
                    &message.content,
                    content_format,
                    placeholder_tokens,
                    media_order,
                    &mut result,
                );
            }
            messages::Role::Assistant => {
                result.push(convert_assistant_message(&message.content));
            }
            // A `system`-role message in `messages[]` (e.g. from Claude Code) is
            // forwarded in place, preserving its position in the conversation so
            // inline-`system` chat templates render it where it was sent.
            // See https://github.com/smg-project/smg/issues/1795
            messages::Role::System => {
                result.push(convert_system_message(&message.content));
            }
        }
        Ok(result)
    })
}

/// Convert a `system`-role message's content to a chat-template JSON value,
/// preserving its position in `messages[]`. System content is text; text blocks
/// are concatenated.
fn convert_system_message(content: &InputContent) -> Value {
    let text = match content {
        InputContent::String(text) => text.clone(),
        InputContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                InputContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    json!({"role": "system", "content": text})
}

/// Convert a user message content to JSON values.
///
/// User messages may contain mixed content: text, images, and tool results.
/// Tool results are split into separate "tool" role messages (the chat template
/// expects tool results as their own messages, not embedded in user content).
fn convert_user_message(
    content: &InputContent,
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
    result: &mut Vec<Value>,
) {
    match content {
        InputContent::String(text) => {
            result.push(json!({"role": "user", "content": text}));
        }
        InputContent::Blocks(blocks) => {
            let (user_parts, tool_msgs) = blocks.iter().fold(
                (Vec::new(), Vec::new()),
                |(mut user_parts, mut tool_msgs), block| {
                    match block {
                        InputContentBlock::Text(t) => {
                            user_parts.push(json!({"type": "text", "text": t.text}));
                        }
                        InputContentBlock::Image(_) => {
                            user_parts.push(json!({"type": "image"}));
                        }
                        InputContentBlock::Document(_) => {
                            user_parts.push(json!({"type": "document"}));
                        }
                        InputContentBlock::ToolResult(tr) => {
                            tool_msgs.push(json!({
                                "role": "tool",
                                "tool_call_id": tr.tool_use_id,
                                "content": extract_tool_result_text(tr)
                            }));
                        }
                        _ => {}
                    }
                    (user_parts, tool_msgs)
                },
            );

            if !user_parts.is_empty() {
                let content = format_content_parts(
                    user_parts,
                    content_format,
                    placeholder_tokens,
                    media_order,
                );
                result.push(json!({"role": "user", "content": content}));
            }
            result.extend(tool_msgs);
        }
    }
}

/// Extract text content from a ToolResult block.
fn extract_tool_result_text(tool_result: &messages::ToolResultBlock) -> String {
    match &tool_result.content {
        Some(ToolResultContent::String(s)) => s.clone(),
        Some(ToolResultContent::Blocks(blocks)) => blocks
            .iter()
            .filter_map(|b| match b {
                messages::ToolResultContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    }
}

/// Convert an assistant message content to a single JSON value.
///
/// Extracts text content, tool calls, and reasoning/thinking into the
/// appropriate JSON fields that the chat template expects.
fn convert_assistant_message(content: &InputContent) -> Value {
    match content {
        InputContent::String(text) => json!({"role": "assistant", "content": text}),
        InputContent::Blocks(blocks) => {
            let (text_parts, tool_calls, thinking_parts) = blocks.iter().fold(
                (
                    Vec::<String>::new(),
                    Vec::<Value>::new(),
                    Vec::<String>::new(),
                ),
                |(mut texts, mut tools, mut thinking), block| {
                    match block {
                        InputContentBlock::Text(t) => texts.push(t.text.clone()),
                        InputContentBlock::ToolUse(tu) => tools.push(json!({
                            "id": tu.id,
                            "type": "function",
                            "function": {
                                "name": tu.name,
                                "arguments": serde_json::to_string(&tu.input)
                                    .unwrap_or_else(|_| "{}".to_string())
                            }
                        })),
                        InputContentBlock::Thinking(t) => thinking.push(t.thinking.clone()),
                        _ => {}
                    }
                    (texts, tools, thinking)
                },
            );

            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Value::String("assistant".into()));

            // With no text blocks (e.g. tool-calls-only), render content as
            // `null` — the OpenAI-faithful representation the chat template
            // expects — rather than an empty text frame.
            let content = if text_parts.is_empty() {
                Value::Null
            } else {
                Value::String(text_parts.join(""))
            };
            obj.insert("content".into(), content);
            if !tool_calls.is_empty() {
                obj.insert("tool_calls".into(), Value::Array(tool_calls));
            }
            if !thinking_parts.is_empty() {
                obj.insert(
                    "reasoning_content".into(),
                    Value::String(thinking_parts.join("\n")),
                );
            }

            Value::Object(obj)
        }
    }
}

/// Format content parts based on the template's content format preference.
///
/// - `String` format: join text parts into a single string
/// - `OpenAI` format: keep as array of typed parts
fn format_content_parts(
    parts: Vec<Value>,
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Value {
    let ordered = order_media_parts(parts, media_order);
    let image_placeholder = placeholder_tokens.and_then(|tokens| tokens.get(Modality::Image));
    match content_format {
        ChatTemplateContentFormat::String => {
            // Extract text parts; optionally replace image parts with placeholders
            let text: String = ordered
                .iter()
                .filter_map(|p| {
                    let obj = p.as_object()?;
                    let type_str = obj.get("type")?.as_str()?;
                    match type_str {
                        "text" => obj.get("text")?.as_str().map(String::from),
                        "image" => image_placeholder.map(String::from),
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            Value::String(text)
        }
        ChatTemplateContentFormat::OpenAI => Value::Array(ordered),
    }
}

/// Hoist media parts before text for `MediaFirst`, matching vLLM front
/// placement; `Authored` keeps request order. `partition` is stable so relative
/// order within each group is preserved.
fn order_media_parts(parts: Vec<Value>, media_order: MediaPartOrder) -> Vec<Value> {
    match media_order {
        MediaPartOrder::Authored => parts,
        MediaPartOrder::MediaFirst => {
            let (mut media, rest): (Vec<Value>, Vec<Value>) = parts.into_iter().partition(|p| {
                matches!(
                    p.get("type").and_then(|t| t.as_str()),
                    Some("image") | Some("video") | Some("audio") | Some("document")
                )
            });
            media.extend(rest);
            media
        }
    }
}

// ============================================================================
// Type adapters: Messages API → Chat API types
// ============================================================================

/// Convert a Messages API CustomTool to a Chat API Tool.
///
/// Maps `CustomTool { name, description, input_schema }` to
/// `ChatTool { type: "function", function: Function { name, description, parameters } }`
pub(crate) fn custom_tool_to_chat_tool(tool: &messages::CustomTool) -> ChatTool {
    // Convert InputSchema to a JSON Value for Function.parameters
    let parameters = input_schema_to_value(&tool.input_schema);

    ChatTool {
        tool_type: "function".to_string(),
        function: common::Function {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters,
            strict: None,
        },
    }
}

/// Convert InputSchema struct to a serde_json::Value.
fn input_schema_to_value(schema: &messages::InputSchema) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "type".to_string(),
        Value::String(schema.schema_type.clone()),
    );

    if let Some(properties) = &schema.properties {
        let props: serde_json::Map<String, Value> = properties
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        obj.insert("properties".to_string(), Value::Object(props));
    }

    if let Some(required) = &schema.required {
        obj.insert(
            "required".to_string(),
            Value::Array(required.iter().map(|s| Value::String(s.clone())).collect()),
        );
    }

    // Include any additional schema fields
    for (key, value) in &schema.additional {
        obj.insert(key.clone(), value.clone());
    }

    Value::Object(obj)
}

/// Convert a Messages API ToolChoice to a Chat API ToolChoice.
///
/// Mapping:
/// - `Auto { .. }`    → `Value(Auto)`
/// - `Any { .. }`     → `Value(Required)`
/// - `Tool { name }`  → `Function { name }`
/// - `None`           → `Value(None)`
pub(crate) fn convert_message_tool_choice(tc: &messages::ToolChoice) -> ChatToolChoice {
    match tc {
        messages::ToolChoice::Auto { .. } => ChatToolChoice::Value(common::ToolChoiceValue::Auto),
        messages::ToolChoice::Any { .. } => {
            ChatToolChoice::Value(common::ToolChoiceValue::Required)
        }
        messages::ToolChoice::Tool { name, .. } => ChatToolChoice::Function {
            tool_type: "function".to_string(),
            function: common::FunctionChoice { name: name.clone() },
        },
        messages::ToolChoice::None => ChatToolChoice::Value(common::ToolChoiceValue::None),
    }
}

/// Extract Custom tools from Messages API tool list and convert to ChatTool.
///
/// Only `Tool::Custom` is supported in gRPC mode. Other tool types
/// (McpToolset, Bash, TextEditor, WebSearch, ToolSearch) are ignored
/// since they require runtime capabilities not available in the gRPC pipeline.
pub(crate) fn extract_chat_tools(tools: &[messages::Tool]) -> Vec<ChatTool> {
    tools
        .iter()
        .filter_map(|t| match t {
            messages::Tool::Custom(custom) => Some(custom_tool_to_chat_tool(custom)),
            _ => None,
        })
        .collect()
}

/// Count the number of tool use blocks in assistant messages of the request history.
///
/// Parallel to `get_history_tool_calls_count` in chat_utils, but works with
/// Messages API `InputMessage` types. Used for generating globally unique
/// tool call IDs (e.g. KimiK2 format).
pub(crate) fn get_history_tool_calls_count_messages(request: &CreateMessageRequest) -> usize {
    request
        .messages
        .iter()
        .filter(|msg| msg.role == messages::Role::Assistant)
        .flat_map(|msg| match &msg.content {
            InputContent::Blocks(blocks) => blocks.as_slice(),
            InputContent::String(_) => &[],
        })
        .filter(|b| matches!(b, InputContentBlock::ToolUse(_)))
        .count()
}

/// Anthropic-native id for a `tool_use` content block.
///
/// The Messages surface should expose `toolu_`-prefixed ids — strict Anthropic
/// SDKs and ecosystem tooling pattern-match the prefix. Parsed tool calls carry
/// the standard OpenAI `call_{uuid}` id, so swap the prefix and keep the
/// suffix: deterministic, so the non-streaming builder and every streaming
/// `content_block_start` site emit the same id for the same call, and clients
/// echo it back opaquely just as before.
///
/// Any other id shape passes through unchanged — model families whose id
/// format is load-bearing in the prompt (rendered into history or correlated
/// verbatim by the reference server) must keep their native shape.
pub(crate) fn anthropic_tool_use_id(id: &str) -> String {
    match id.strip_prefix("call_") {
        Some(suffix) => format!("toolu_{suffix}"),
        None => id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use messages::{InputMessage, Role, TextBlock};

    use super::*;

    #[test]
    fn anthropic_tool_use_id_swaps_standard_prefix() {
        assert_eq!(
            anthropic_tool_use_id("call_0123456789abcdef01234567"),
            "toolu_0123456789abcdef01234567"
        );
        // Deterministic: same input, same output.
        assert_eq!(
            anthropic_tool_use_id("call_0123456789abcdef01234567"),
            anthropic_tool_use_id("call_0123456789abcdef01234567"),
        );
    }

    #[test]
    fn anthropic_tool_use_id_passes_other_shapes_through() {
        // Prompt-visible / reference-correlated id formats keep their shape.
        assert_eq!(anthropic_tool_use_id("Bash_3"), "Bash_3");
        assert_eq!(
            anthropic_tool_use_id("functions.get_weather:2"),
            "functions.get_weather:2"
        );
        // Already-native ids are untouched.
        assert_eq!(
            anthropic_tool_use_id("toolu_0123456789abcdef01234567"),
            "toolu_0123456789abcdef01234567"
        );
    }

    #[test]
    fn test_simple_user_message() {
        let messages = vec![InputMessage {
            role: Role::User,
            content: InputContent::String("Hello".to_string()),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "Hello");
    }

    #[test]
    fn test_assistant_with_text() {
        let messages = vec![InputMessage {
            role: Role::Assistant,
            content: InputContent::Blocks(vec![InputContentBlock::Text(TextBlock {
                text: "Hi there".to_string(),
                cache_control: None,
                citations: None,
            })]),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "assistant");
        assert_eq!(result[0]["content"], "Hi there");
    }

    #[test]
    fn test_assistant_with_tool_use() {
        let messages = vec![InputMessage {
            role: Role::Assistant,
            content: InputContent::Blocks(vec![
                InputContentBlock::Text(TextBlock {
                    text: "Let me check.".to_string(),
                    cache_control: None,
                    citations: None,
                }),
                InputContentBlock::ToolUse(messages::ToolUseBlock {
                    id: "tu_1".to_string(),
                    name: "calculator".to_string(),
                    input: json!({"expr": "2+2"}),
                    cache_control: None,
                }),
            ]),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "assistant");
        assert_eq!(result[0]["content"], "Let me check.");
        let tool_calls = result[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["function"]["name"], "calculator");
    }

    #[test]
    fn test_absent_assistant_content_renders_null() {
        let messages = vec![InputMessage {
            role: Role::Assistant,
            content: InputContent::Blocks(vec![InputContentBlock::ToolUse(
                messages::ToolUseBlock {
                    id: "tu_1".to_string(),
                    name: "calc".to_string(),
                    input: json!({"x": 1}),
                    cache_control: None,
                },
            )]),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert!(result[0]["content"].is_null());
    }

    #[test]
    fn test_user_media_order_follows_contract() {
        let messages = vec![InputMessage {
            role: Role::User,
            content: InputContent::Blocks(vec![
                InputContentBlock::Text(TextBlock {
                    text: "question".to_string(),
                    cache_control: None,
                    citations: None,
                }),
                InputContentBlock::Image(messages::ImageBlock {
                    source: messages::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "AAAA".to_string(),
                    },
                    cache_control: None,
                }),
            ]),
        }];

        let media_first = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        let arr = media_first[0]["content"].as_array().unwrap();
        assert_eq!(arr[0], json!({"type": "image"}));
        assert_eq!(arr[1]["text"], "question");

        let authored = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::Authored,
        )
        .unwrap();
        let arr = authored[0]["content"].as_array().unwrap();
        assert_eq!(arr[0]["text"], "question");
        assert_eq!(arr[1], json!({"type": "image"}));
    }

    #[test]
    fn test_user_with_tool_result_splits() {
        let messages = vec![InputMessage {
            role: Role::User,
            content: InputContent::Blocks(vec![InputContentBlock::ToolResult(
                messages::ToolResultBlock {
                    tool_use_id: "tu_1".to_string(),
                    content: Some(ToolResultContent::String("4".to_string())),
                    is_error: None,
                    cache_control: None,
                },
            )]),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        // Tool result becomes a "tool" role message, not a "user" message
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "tool");
        assert_eq!(result[0]["tool_call_id"], "tu_1");
        assert_eq!(result[0]["content"], "4");
    }

    #[test]
    fn test_assistant_with_thinking() {
        // Single thinking block
        let messages = vec![InputMessage {
            role: Role::Assistant,
            content: InputContent::Blocks(vec![
                InputContentBlock::Thinking(messages::ThinkingBlock {
                    thinking: "Let me reason...".to_string(),
                    signature: "sig123".to_string(),
                }),
                InputContentBlock::Text(TextBlock {
                    text: "The answer is 42.".to_string(),
                    cache_control: None,
                    citations: None,
                }),
            ]),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "assistant");
        assert_eq!(result[0]["content"], "The answer is 42.");
        assert_eq!(result[0]["reasoning_content"], "Let me reason...");

        // Multiple thinking blocks are concatenated
        let messages = vec![InputMessage {
            role: Role::Assistant,
            content: InputContent::Blocks(vec![
                InputContentBlock::Thinking(messages::ThinkingBlock {
                    thinking: "First thought.".to_string(),
                    signature: "sig1".to_string(),
                }),
                InputContentBlock::Thinking(messages::ThinkingBlock {
                    thinking: "Second thought.".to_string(),
                    signature: "sig2".to_string(),
                }),
                InputContentBlock::Text(TextBlock {
                    text: "Combined answer.".to_string(),
                    cache_control: None,
                    citations: None,
                }),
            ]),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert_eq!(
            result[0]["reasoning_content"],
            "First thought.\nSecond thought."
        );
    }

    #[test]
    fn test_tool_choice_conversion() {
        assert!(matches!(
            convert_message_tool_choice(&messages::ToolChoice::Auto {
                disable_parallel_tool_use: None
            }),
            ChatToolChoice::Value(common::ToolChoiceValue::Auto)
        ));
        assert!(matches!(
            convert_message_tool_choice(&messages::ToolChoice::Any {
                disable_parallel_tool_use: None
            }),
            ChatToolChoice::Value(common::ToolChoiceValue::Required)
        ));
        assert!(matches!(
            convert_message_tool_choice(&messages::ToolChoice::None),
            ChatToolChoice::Value(common::ToolChoiceValue::None)
        ));

        let tc = convert_message_tool_choice(&messages::ToolChoice::Tool {
            name: "calc".to_string(),
            disable_parallel_tool_use: None,
        });
        assert!(matches!(tc, ChatToolChoice::Function { .. }));
    }

    #[test]
    fn test_custom_tool_conversion() {
        let custom = messages::CustomTool {
            name: "weather".to_string(),
            tool_type: None,
            description: Some("Get weather".to_string()),
            input_schema: messages::InputSchema {
                schema_type: "object".to_string(),
                properties: Some(
                    [("city".to_string(), json!({"type": "string"}))]
                        .into_iter()
                        .collect(),
                ),
                required: Some(vec!["city".to_string()]),
                additional: Default::default(),
            },
            defer_loading: None,
            cache_control: None,
        };

        let chat_tool = custom_tool_to_chat_tool(&custom);
        assert_eq!(chat_tool.function.name, "weather");
        assert_eq!(
            chat_tool.function.description,
            Some("Get weather".to_string())
        );
        assert_eq!(chat_tool.function.parameters["type"], "object");
        assert!(chat_tool.function.parameters["properties"]["city"].is_object());
    }

    #[test]
    fn test_get_history_tool_calls_count_messages() {
        // No tool calls
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![InputMessage {
                role: Role::User,
                content: InputContent::String("Hello".to_string()),
            }],
            max_tokens: 100,
            metadata: None,
            service_tier: None,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            container: None,
            mcp_servers: None,
            rid: None,
            other: serde_json::Map::new(),
        };
        assert_eq!(get_history_tool_calls_count_messages(&request), 0);

        // With tool calls in assistant message
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![
                InputMessage {
                    role: Role::User,
                    content: InputContent::String("Hello".to_string()),
                },
                InputMessage {
                    role: Role::Assistant,
                    content: InputContent::Blocks(vec![
                        InputContentBlock::Text(TextBlock {
                            text: "Let me check.".to_string(),
                            cache_control: None,
                            citations: None,
                        }),
                        InputContentBlock::ToolUse(messages::ToolUseBlock {
                            id: "tu_1".to_string(),
                            name: "calc".to_string(),
                            input: json!({"x": 1}),
                            cache_control: None,
                        }),
                        InputContentBlock::ToolUse(messages::ToolUseBlock {
                            id: "tu_2".to_string(),
                            name: "search".to_string(),
                            input: json!({"q": "test"}),
                            cache_control: None,
                        }),
                    ]),
                },
            ],
            max_tokens: 100,
            metadata: None,
            service_tier: None,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            container: None,
            mcp_servers: None,
            rid: None,
            other: serde_json::Map::new(),
        };
        assert_eq!(get_history_tool_calls_count_messages(&request), 2);
    }
}
