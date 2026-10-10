//! Message API utilities for converting Anthropic Messages API types
//! into the internal chat template format.
//!
//! Parallel to `chat_utils.rs` but works with `CreateMessageRequest` / `InputMessage`
//! instead of `ChatCompletionRequest` / `ChatMessage`.
#![allow(dead_code)] // wired in follow-up PR (pipeline factory)

use std::{fmt::Display, sync::Arc};

use axum::response::Response;
use llm_multimodal::{MediaPartOrder, Modality};
use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    traits::{PromptEncoding, Tokenizer},
};
use openai_protocol::{
    common::{self, StringOrArray, Tool as ChatTool, ToolChoice as ChatToolChoice},
    messages::{
        self, CountMessageTokensRequest, CreateMessageRequest, InputContent, InputContentBlock,
        InputMessage, SystemContent, ThinkingConfig, ToolResultContent,
    },
};
use serde_json::{json, Value};
use tracing::error;

use super::chat_utils;
use crate::routers::{
    error,
    grpc::{
        context::SharedComponents,
        multimodal::{self, MediaPlan, MultimodalComponents, PlaceholderTokens},
        ProcessedMessages,
    },
};

// ============================================================================
// Top-level processing function
// ============================================================================

/// The parts of a Messages request its prompt is rendered from, so that
/// `/v1/messages` and `/v1/messages/count_tokens` render through one path.
pub(crate) struct MessagesPrompt<'a> {
    pub messages: &'a [InputMessage],
    pub system: Option<&'a SystemContent>,
    pub thinking: Option<&'a ThinkingConfig>,
    pub stop_sequences: Option<&'a [String]>,
}

impl<'a> From<&'a CreateMessageRequest> for MessagesPrompt<'a> {
    fn from(request: &'a CreateMessageRequest) -> Self {
        Self {
            messages: &request.messages,
            system: request.system.as_ref(),
            thinking: request.thinking.as_ref(),
            stop_sequences: request.stop_sequences.as_deref(),
        }
    }
}

impl<'a> From<&'a CountMessageTokensRequest> for MessagesPrompt<'a> {
    fn from(request: &'a CountMessageTokensRequest) -> Self {
        Self {
            messages: &request.messages,
            system: request.system.as_ref(),
            thinking: request.thinking.as_ref(),
            stop_sequences: None,
        }
    }
}

/// Process messages from a CreateMessageRequest and apply the chat template.
///
/// Parallel to `process_chat_messages()` in chat_utils, but works with
/// Anthropic Messages API types. Converts InputMessages to JSON values
/// that the chat template expects, then applies the template. The second
/// element says how the tokenize step must encode the prompt.
pub fn process_messages(
    request: &CreateMessageRequest,
    tokenizer: &dyn Tokenizer,
    chat_tools: Option<&[ChatTool]>,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<(ProcessedMessages, PromptEncoding), String> {
    process_messages_prompt(
        MessagesPrompt::from(request),
        tokenizer,
        chat_tools,
        placeholder_tokens,
        media_order,
    )
}

/// [`process_messages`] over the prompt parts alone.
pub(crate) fn process_messages_prompt(
    request: MessagesPrompt<'_>,
    tokenizer: &dyn Tokenizer,
    chat_tools: Option<&[ChatTool]>,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<(ProcessedMessages, PromptEncoding), String> {
    let content_format = tokenizer.chat_template_content_format();

    // Step 1: Convert InputMessages to chat template JSON values
    let mut transformed_messages = process_message_content_format(
        request.messages,
        content_format,
        placeholder_tokens,
        media_order,
    )?;

    // Step 2: Prepend system message if present
    if let Some(system) = request.system {
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

    // Step 3: Process tool call arguments in assistant messages (reuse from
    // chat_utils), unless the renderer parses them itself as written.
    if !tokenizer.renderer_capabilities().raw_tool_call_arguments {
        chat_utils::process_tool_call_arguments(&mut transformed_messages)?;
    }

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
    let thinking = match request.thinking {
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

    let rendered = tokenizer
        .apply_chat_template_with_encoding(&transformed_messages, params, None)
        .map_err(|e| format!("Failed to apply chat template: {e}"))?;

    // Step 7: Build ProcessedMessages
    let stop_sequences = request
        .stop_sequences
        .map(|seqs| StringOrArray::Array(seqs.to_vec()));

    Ok((
        ProcessedMessages {
            text: rendered.text,
            stop_sequences,
            unbilled_prompt_tokens: rendered.unbilled_prompt_tokens,
        },
        rendered.encoding,
    ))
}

/// The tools the template renders for a request: the custom tools as chat
/// tools, narrowed by `tool_choice` the way the chat path narrows them (a
/// forced tool renders alone). Shared by the preparation stage and
/// count_tokens so both render the same prompt.
pub(crate) fn template_tools(
    tools: Option<&[messages::Tool]>,
    tool_choice: Option<&messages::ToolChoice>,
) -> Vec<ChatTool> {
    let Some(tools) = tools else {
        return Vec::new();
    };
    let chat_tools = extract_chat_tools(tools);
    match tool_choice.map(convert_message_tool_choice) {
        Some(tc) => {
            chat_utils::filter_tools_by_tool_choice(&chat_tools, Some(&tc)).unwrap_or(chat_tools)
        }
        None => chat_tools,
    }
}

/// The media context of a Messages request, resolved from the registries
/// before the template renders, as the preparation stage resolves it:
/// nothing is fetched or preprocessed here.
pub(crate) struct MessagesMedia {
    /// Where the model's template wants media parts relative to text.
    pub(crate) media_order: MediaPartOrder,
    /// The request's media and the placeholders the template writes for it,
    /// when the request carries media.
    pub(crate) media: Option<MediaContext>,
}

impl MessagesMedia {
    pub(crate) fn placeholders(&self) -> Option<&PlaceholderTokens> {
        self.media.as_ref().map(|media| &media.placeholders)
    }
}

/// A request's media as the preparation stage processes it after rendering.
pub(crate) struct MediaContext {
    pub(crate) plan: MediaPlan,
    pub(crate) placeholders: PlaceholderTokens,
    pub(crate) components: Arc<MultimodalComponents>,
    pub(crate) tokenizer_id: String,
    pub(crate) tokenizer_source: String,
}

/// The 400 for media the request carries that cannot be prepared.
pub(crate) fn invalid_multimodal_request(error: impl Display) -> Response {
    error::bad_request(
        "invalid_multimodal_request",
        format!("Invalid multimodal request: {error}"),
    )
}

/// Resolve what a Messages request's media needs before the template renders:
/// the model's media-part order and, for a request with media, the plan and
/// the placeholder tokens (the engine's own limits applied). The refusals are
/// the preparation stage's, so count_tokens answers a request the way
/// `/v1/messages` would.
pub(crate) async fn resolve_messages_media(
    components: &SharedComponents,
    model_id: &str,
    tokenizer: &dyn Tokenizer,
    messages: &[InputMessage],
) -> Result<MessagesMedia, Response> {
    let tokenizer_entry = components
        .tokenizer_registry
        .get_by_name(model_id)
        .or_else(|| components.tokenizer_registry.get_by_id(model_id));
    // The media-part order comes from the model registry so /v1/messages
    // renders each model consistently with /v1/chat/completions.
    let media_order = match (components.multimodal.as_ref(), tokenizer_entry.as_ref()) {
        (Some(mm_components), Some(entry)) => {
            multimodal::resolve_media_part_order(
                model_id,
                tokenizer,
                mm_components,
                &entry.id,
                &entry.source,
            )
            .await
        }
        _ => MediaPartOrder::MediaFirst,
    };

    let plan = multimodal::media_plan_messages(messages);
    if plan.is_empty() {
        return Ok(MessagesMedia {
            media_order,
            media: None,
        });
    }
    let Some(mm_components) = components.multimodal.as_ref() else {
        error!(
            function = "resolve_messages_media",
            "Multimodal content detected but multimodal components not initialized"
        );
        return Err(error::bad_request(
            "multimodal_not_supported",
            "Multimodal content detected but multimodal processing is not available",
        ));
    };
    let Some(entry) = tokenizer_entry else {
        error!(
            function = "resolve_messages_media",
            model = %model_id,
            "Tokenizer entry not found for multimodal processing"
        );
        return Err(error::bad_request(
            "multimodal_config_missing",
            format!("Tokenizer not found for model: {model_id}"),
        ));
    };

    // The per-request media limits the model's workers advertise for their
    // engine, which the plan is held to before any fetch.
    let engine_limits = multimodal::engine_item_limits(&components.worker_registry, model_id);
    let placeholders = multimodal::prepare_placeholder_tokens(
        &plan,
        model_id,
        tokenizer,
        mm_components,
        &entry.id,
        &entry.source,
        &engine_limits,
    )
    .await
    .map_err(|e| {
        error!(
            function = "resolve_messages_media",
            model = %model_id,
            error = %e,
            "Failed to resolve multimodal placeholder token"
        );
        invalid_multimodal_request(e)
    })?;

    Ok(MessagesMedia {
        media_order,
        media: Some(MediaContext {
            plan,
            placeholders,
            components: Arc::clone(mm_components),
            tokenizer_id: entry.id,
            tokenizer_source: entry.source,
        }),
    })
}

/// Why a count could not be produced: the request does not render (the
/// client's mistake, a 400 on `/v1/messages`) or the rendered prompt does
/// not encode (a gateway fault, a 500 there).
#[derive(Debug)]
pub(crate) enum CountError {
    Render(String),
    Encode(anyhow::Error),
}

/// The prompt tokens `/v1/messages` would send for `request`: the same tool
/// set, media placeholders, template rendering and encoding as the
/// preparation stage, without media processing (an image counts as the
/// placeholder the template writes for it; the image itself is not fetched).
/// A text or content-source document counts as its rendered text; a source
/// with no text is refused, as `/v1/messages` refuses it.
pub(crate) async fn count_input_tokens(
    request: &CountMessageTokensRequest,
    tokenizer: Arc<dyn Tokenizer>,
    tools: &[ChatTool],
    placeholders: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<u32, CountError> {
    let (processed, encoding) = process_messages_prompt(
        MessagesPrompt::from(request),
        &*tokenizer,
        (!tools.is_empty()).then_some(tools),
        placeholders,
        media_order,
    )
    .map_err(CountError::Render)?;
    let encoded = chat_utils::encode_prompt_blocking(tokenizer, &processed.text, encoding)
        .await
        .map_err(CountError::Encode)?;
    u32::try_from(encoded.token_ids().len()).map_err(|_| {
        CountError::Render("the prompt has more tokens than a count can report".to_owned())
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
    messages
        .iter()
        .enumerate()
        .try_fold(Vec::new(), |mut result, (index, message)| {
            match message.role {
                messages::Role::User => {
                    convert_user_message(
                        index,
                        &message.content,
                        content_format,
                        placeholder_tokens,
                        media_order,
                        &mut result,
                    )?;
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
    index: usize,
    content: &InputContent,
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
    result: &mut Vec<Value>,
) -> Result<(), String> {
    match content {
        InputContent::String(text) => {
            result.push(json!({"role": "user", "content": text}));
        }
        InputContent::Blocks(blocks) => {
            let mut user_parts = Vec::new();
            let mut tool_msgs = Vec::new();
            for (position, block) in blocks.iter().enumerate() {
                match block {
                    InputContentBlock::Text(t) => {
                        user_parts.push(json!({"type": "text", "text": t.text}));
                    }
                    InputContentBlock::Image(_) => {
                        user_parts.push(json!({"type": "image"}));
                    }
                    // A document is prompt text for every template, not a
                    // media part the template may not know.
                    InputContentBlock::Document(document) => {
                        let text = render_document(
                            document,
                            &format!("messages.{index}.content.{position}"),
                        )?;
                        user_parts.push(json!({"type": "text", "text": text}));
                    }
                    InputContentBlock::ToolResult(tr) => {
                        tool_msgs.push(json!({
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": extract_tool_result_text(tr)
                        }));
                        // A tool result may carry images (a screenshot a
                        // browser or shell tool returned). They join the
                        // user content here, in block order, so the model
                        // sees them; `media_plan_messages` lists them at
                        // the same position, which keeps the plan aligned
                        // with the placeholders this message renders.
                        user_parts
                            .extend(tool_result_image_blocks(tr).map(|_| json!({"type": "image"})));
                    }
                    _ => {}
                }
            }

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
    Ok(())
}

/// A document block as prompt text, so that any template reads it: the
/// title and context the request gave, then the document's text, in the
/// tagged layout the public API documents for documents in a prompt. Only
/// text renders: a text source, or a content source made of text blocks. A
/// PDF or URL source needs a document reader this gateway has no path for
/// and is refused naming the block (`at`), not handed to the template.
pub(crate) fn render_document(
    document: &messages::DocumentBlock,
    at: &str,
) -> Result<String, String> {
    let text = match &document.source {
        messages::DocumentSource::Text { data } => data.as_str(),
        messages::DocumentSource::Content { content } => {
            let mut texts = Vec::with_capacity(content.len());
            for block in content {
                match block {
                    InputContentBlock::Text(t) => texts.push(t.text.as_str()),
                    _ => {
                        return Err(format!(
                            "{at}: a document content source may hold text blocks only"
                        ))
                    }
                }
            }
            return Ok(document_text(document, &texts.join("\n")));
        }
        messages::DocumentSource::Base64 { media_type, .. } => {
            return Err(format!(
                "{at}: document source type 'base64' ({media_type}) is not supported; \
                 send the document as a text or content source"
            ))
        }
        messages::DocumentSource::Url { .. } => {
            return Err(format!(
                "{at}: document source type 'url' is not supported; send the document \
                 as a text or content source"
            ))
        }
    };
    Ok(document_text(document, text))
}

fn document_text(document: &messages::DocumentBlock, text: &str) -> String {
    let mut out = String::from("<document>\n");
    if let Some(title) = &document.title {
        out.push_str(&format!("<source>{title}</source>\n"));
    }
    if let Some(context) = &document.context {
        out.push_str(&format!("<document_context>{context}</document_context>\n"));
    }
    out.push_str("<document_content>\n");
    out.push_str(text);
    out.push_str("\n</document_content>\n</document>");
    out
}

/// The image blocks inside a ToolResult block's content, in order.
///
/// Shared with media detection so the plan and the rendered content agree on
/// which images a tool result contributes and where they stand.
pub(crate) fn tool_result_image_blocks(
    tool_result: &messages::ToolResultBlock,
) -> impl Iterator<Item = &messages::ImageBlock> {
    let blocks = match &tool_result.content {
        Some(ToolResultContent::Blocks(blocks)) => blocks.as_slice(),
        Some(ToolResultContent::String(_)) | None => &[],
    };
    blocks.iter().filter_map(|block| match block {
        messages::ToolResultContentBlock::Image(image) => Some(image),
        _ => None,
    })
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
                    Some("image") | Some("video") | Some("audio")
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
            extra: Default::default(),
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

    #[test]
    fn test_tool_result_images_join_user_content_in_block_order() {
        let messages = vec![InputMessage {
            role: Role::User,
            content: InputContent::Blocks(vec![
                InputContentBlock::ToolResult(messages::ToolResultBlock {
                    tool_use_id: "tu_1".to_string(),
                    content: Some(ToolResultContent::Blocks(vec![
                        messages::ToolResultContentBlock::Text(TextBlock {
                            text: "screenshot taken".to_string(),
                            cache_control: None,
                            citations: None,
                        }),
                        messages::ToolResultContentBlock::Image(messages::ImageBlock {
                            source: messages::ImageSource::Base64 {
                                media_type: "image/png".to_string(),
                                data: "AAAA".to_string(),
                            },
                            cache_control: None,
                        }),
                    ])),
                    is_error: None,
                    cache_control: None,
                }),
                InputContentBlock::Text(TextBlock {
                    text: "what do you see".to_string(),
                    cache_control: None,
                    citations: None,
                }),
            ]),
        }];

        let result = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::Authored,
        )
        .unwrap();

        // The user content carries the tool result's image ahead of the text
        // that followed it; the tool message keeps the result's text.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["role"], "user");
        let parts = result[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], json!({"type": "image"}));
        assert_eq!(parts[1]["text"], "what do you see");
        assert_eq!(result[1]["role"], "tool");
        assert_eq!(result[1]["tool_call_id"], "tu_1");
        assert_eq!(result[1]["content"], "screenshot taken");
    }

    #[test]
    fn test_tool_result_image_blocks_skips_text_only_results() {
        let text_only = messages::ToolResultBlock {
            tool_use_id: "tu_1".to_string(),
            content: Some(ToolResultContent::String("4".to_string())),
            is_error: None,
            cache_control: None,
        };
        assert_eq!(tool_result_image_blocks(&text_only).count(), 0);

        let empty = messages::ToolResultBlock {
            tool_use_id: "tu_2".to_string(),
            content: None,
            is_error: None,
            cache_control: None,
        };
        assert_eq!(tool_result_image_blocks(&empty).count(), 0);
    }

    fn document(
        source: messages::DocumentSource,
        title: Option<&str>,
        context: Option<&str>,
    ) -> Vec<InputMessage> {
        vec![InputMessage {
            role: Role::User,
            content: InputContent::Blocks(vec![
                InputContentBlock::Document(messages::DocumentBlock {
                    source,
                    cache_control: None,
                    title: title.map(str::to_owned),
                    context: context.map(str::to_owned),
                    citations: Some(messages::CitationsConfig {
                        enabled: Some(true),
                    }),
                }),
                InputContentBlock::Text(TextBlock {
                    text: "What is the powerhouse of the cell?".to_string(),
                    cache_control: None,
                    citations: None,
                }),
            ]),
        }]
    }

    /// A text-source document is prompt text with its title and context,
    /// kept in the order the request wrote it, for both content formats.
    #[test]
    fn text_document_renders_as_prompt_text_in_authored_order() {
        let messages = document(
            messages::DocumentSource::Text {
                data: "The mitochondria is the powerhouse of the cell.".to_string(),
            },
            Some("Biology notes"),
            Some("From a textbook"),
        );
        let expected = "<document>\n<source>Biology notes</source>\n<document_context>From a textbook</document_context>\n<document_content>\nThe mitochondria is the powerhouse of the cell.\n</document_content>\n</document>";

        let parts = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        let arr = parts[0]["content"].as_array().unwrap();
        assert_eq!(arr[0], json!({"type": "text", "text": expected}));
        assert_eq!(arr[1]["text"], "What is the powerhouse of the cell?");

        let flat = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap();
        assert_eq!(
            flat[0]["content"],
            json!(format!("{expected}\nWhat is the powerhouse of the cell?"))
        );
    }

    /// A content-source document joins its text blocks; one holding anything
    /// else is refused naming the block.
    #[test]
    fn content_document_joins_its_text_blocks() {
        let text = |t: &str| {
            InputContentBlock::Text(TextBlock {
                text: t.to_string(),
                cache_control: None,
                citations: None,
            })
        };
        let messages = document(
            messages::DocumentSource::Content {
                content: vec![text("First paragraph."), text("Second paragraph.")],
            },
            None,
            None,
        );
        let parts = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::Authored,
        )
        .unwrap();
        assert_eq!(
            parts[0]["content"][0]["text"],
            "<document>\n<document_content>\nFirst paragraph.\nSecond paragraph.\n</document_content>\n</document>"
        );

        let messages = document(
            messages::DocumentSource::Content {
                content: vec![InputContentBlock::Image(messages::ImageBlock {
                    source: messages::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "AAAA".to_string(),
                    },
                    cache_control: None,
                })],
            },
            None,
            None,
        );
        let err = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::Authored,
        )
        .unwrap_err();
        assert!(
            err.starts_with(
                "messages.0.content.0: a document content source may hold text blocks only"
            ),
            "{err}"
        );
    }

    /// A PDF or URL document has no reader here: refused with a message
    /// naming the block, before any template sees it.
    #[test]
    fn pdf_and_url_documents_are_refused_naming_the_block() {
        for (source, needle) in [
            (
                messages::DocumentSource::Base64 {
                    media_type: "application/pdf".to_string(),
                    data: "JVBERi0=".to_string(),
                },
                "messages.0.content.0: document source type 'base64' (application/pdf) is not supported",
            ),
            (
                messages::DocumentSource::Url {
                    url: "https://example.com/paper.pdf".to_string(),
                },
                "messages.0.content.0: document source type 'url' is not supported",
            ),
        ] {
            let err = process_message_content_format(
                &document(source, None, None),
                ChatTemplateContentFormat::OpenAI,
                None,
                MediaPartOrder::Authored,
            )
            .unwrap_err();
            assert!(err.starts_with(needle), "{err}");
        }
    }

    /// count_tokens renders and encodes through the same path as a message
    /// request: with a renderer that encodes itself, the count is the length
    /// of the ids it hands back, not of a re-tokenization of its text.
    #[tokio::test]
    async fn count_input_tokens_reports_the_rendered_prompt_length() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(
            llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![7, 8, 9, 10, 11]),
        );
        let request: CountMessageTokensRequest = serde_json::from_value(json!({
            "model": "m",
            "system": "You are terse.",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{
                "name": "get_weather",
                "description": "Weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "Weather in Paris?"}]
        }))
        .unwrap();
        let tools = template_tools(request.tools.as_deref(), request.tool_choice.as_ref());
        assert_eq!(tools.len(), 1, "a forced tool renders alone");

        let counted = count_input_tokens(
            &request,
            tokenizer,
            &tools,
            None,
            MediaPartOrder::MediaFirst,
        )
        .await
        .unwrap();
        assert_eq!(counted, 5);
    }

    /// With a string-format template an image is the placeholder the model's
    /// registry names; the count carries it, as the prompt `/v1/messages`
    /// sends carries it. (The mock vocabulary knows "token", "Hello", "world".)
    #[tokio::test]
    async fn count_input_tokens_counts_the_image_placeholder_the_template_writes() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(
            llm_tokenizer::mock::MockTokenizer::new()
                .with_content_format(ChatTemplateContentFormat::String),
        );
        let request: CountMessageTokensRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Hello world"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
            ]}]
        }))
        .unwrap();
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "token".to_string());

        let without = count_input_tokens(
            &request,
            Arc::clone(&tokenizer),
            &[],
            None,
            MediaPartOrder::MediaFirst,
        )
        .await
        .unwrap();
        let with = count_input_tokens(
            &request,
            tokenizer,
            &[],
            Some(&placeholders),
            MediaPartOrder::MediaFirst,
        )
        .await
        .unwrap();
        assert_eq!(without, 2, "the text alone: Hello, world");
        assert_eq!(with, 3, "the placeholder counts once");
    }

    /// A tokenizer whose encode fails is a gateway fault, told apart from a
    /// request that does not render, so the router can answer 500 and 400 as
    /// `/v1/messages` does.
    #[tokio::test]
    async fn count_input_tokens_tells_an_encode_failure_from_a_render_failure() {
        struct EncodeFails(llm_tokenizer::mock::MockTokenizer);
        impl llm_tokenizer::traits::Encoder for EncodeFails {
            fn encode(&self, _: &str, _: bool) -> anyhow::Result<llm_tokenizer::traits::Encoding> {
                Err(anyhow::anyhow!("the tokenizer files are unreadable"))
            }
            fn encode_batch(
                &self,
                _: &[&str],
                _: bool,
            ) -> anyhow::Result<Vec<llm_tokenizer::traits::Encoding>> {
                Err(anyhow::anyhow!("the tokenizer files are unreadable"))
            }
        }
        impl llm_tokenizer::traits::Decoder for EncodeFails {
            fn decode(&self, ids: &[u32], s: bool) -> anyhow::Result<String> {
                self.0.decode(ids, s)
            }
        }
        impl Tokenizer for EncodeFails {
            fn vocab_size(&self) -> usize {
                self.0.vocab_size()
            }
            fn get_special_tokens(&self) -> &llm_tokenizer::traits::SpecialTokens {
                self.0.get_special_tokens()
            }
            fn token_to_id(&self, t: &str) -> Option<u32> {
                self.0.token_to_id(t)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.0.id_to_token(id)
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn apply_chat_template(
                &self,
                messages: &[Value],
                params: ChatTemplateParams,
            ) -> anyhow::Result<String> {
                self.0.apply_chat_template(messages, params)
            }
        }
        let request: CountMessageTokensRequest = serde_json::from_value(json!({
            "model": "m", "messages": [{"role": "user", "content": "Hello world"}]
        }))
        .unwrap();

        let failure = count_input_tokens(
            &request,
            Arc::new(EncodeFails(llm_tokenizer::mock::MockTokenizer::new())),
            &[],
            None,
            MediaPartOrder::MediaFirst,
        )
        .await
        .unwrap_err();
        assert!(matches!(failure, CountError::Encode(_)), "{failure:?}");
    }
}
