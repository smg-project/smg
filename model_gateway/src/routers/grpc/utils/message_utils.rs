//! Message API utilities for converting Anthropic Messages API types
//! into the internal chat template format.
//!
//! Parallel to `chat_utils.rs` but works with `CreateMessageRequest` / `InputMessage`
//! instead of `ChatCompletionRequest` / `ChatMessage`.
#![allow(dead_code)] // wired in follow-up PR (pipeline factory)

use std::{borrow::Cow, collections::HashMap, fmt::Display, sync::Arc};

use axum::response::Response;
use llm_multimodal::{MediaPartOrder, Modality};
use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    traits::{MediaHandling, PromptEncoding, RendererCapabilities, Tokenizer},
};
use openai_protocol::{
    common::{Tool as ChatTool, ToolChoice as ChatToolChoice},
    messages::{
        self, CountMessageTokensRequest, CreateMessageRequest, InputContent, InputContentBlock,
        InputMessage, SystemContent, ToolResultContent,
    },
};
use serde_json::{json, Value};
use tracing::error;

use super::chat_utils;
use crate::routers::{
    error,
    grpc::{
        context::SharedComponents,
        multimodal::{
            self, anchor_rendered_markers, MediaPlan, MultimodalComponents, PlaceholderTokens,
        },
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
    /// The thinking preference the prompt is rendered with
    /// (`CreateMessageRequest::template_thinking`).
    pub thinking: Option<bool>,
    /// `output_config.effort`, rendered as the `reasoning_effort` kwarg.
    pub reasoning_effort: Option<&'a str>,
    /// The request's tool choice, told to the template as the chat path
    /// tells it; a renderer that declares the tools under `none` writes its
    /// instruction not to call them from it.
    pub tool_choice: Option<&'a messages::ToolChoice>,
}

impl<'a> From<&'a CreateMessageRequest> for MessagesPrompt<'a> {
    fn from(request: &'a CreateMessageRequest) -> Self {
        Self {
            messages: &request.messages,
            system: request.system.as_ref(),
            thinking: request.template_thinking(),
            reasoning_effort: request.reasoning_effort(),
            tool_choice: request.tool_choice.as_ref(),
        }
    }
}

impl<'a> From<&'a CountMessageTokensRequest> for MessagesPrompt<'a> {
    fn from(request: &'a CountMessageTokensRequest) -> Self {
        Self {
            messages: &request.messages,
            system: request.system.as_ref(),
            thinking: request.template_thinking(),
            reasoning_effort: request.reasoning_effort(),
            tool_choice: request.tool_choice.as_ref(),
        }
    }
}

/// Process messages from a CreateMessageRequest and apply the chat template.
///
/// Parallel to `process_chat_messages()` in chat_utils, but works with
/// Anthropic Messages API types. Converts InputMessages to JSON values
/// that the chat template expects, then applies the template, with
/// `reasoning_effort` as the template kwarg of that name when set. The
/// second element says how the tokenize step must encode the prompt.
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

/// [`process_messages`] for a request without media: the entry point for
/// callers outside the pipeline (bindings, tests), the Messages-side twin
/// of `process_chat_messages_with_encoding`.
pub fn process_messages_with_encoding(
    request: &CreateMessageRequest,
    tokenizer: &dyn Tokenizer,
    chat_tools: Option<&[ChatTool]>,
) -> Result<(ProcessedMessages, PromptEncoding), String> {
    process_messages(
        request,
        tokenizer,
        chat_tools,
        None,
        MediaPartOrder::MediaFirst,
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

    // Step 2b: a renderer whose reference writes no turn for a system
    // message past the leading run gets the ones sent inside `messages[]`
    // after it as user turns at their positions; the leading run (the
    // top-level system prompt and any system messages that open the array)
    // is the system prompt and stays.
    let capabilities = tokenizer.renderer_capabilities();
    if capabilities.leading_system_only {
        system_messages_after_the_first_as_user_turns(&mut transformed_messages);
    }

    // Step 3: Process tool call arguments in assistant messages (reuse from
    // chat_utils), unless the renderer parses them itself as written.
    if !capabilities.raw_tool_call_arguments {
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

    // Step 5: the thinking preference the request states, or its provider
    // profile reads from the effort, as for Chat.
    let thinking = request.thinking;

    // Step 6: Apply chat template. The template learns the tool choice the
    // way the chat path tells it (the chat value under the same kwarg): a
    // renderer that declares the tools under `none` writes its instruction
    // not to call them from this key. The effort rides under the chat
    // path's `reasoning_effort` kwarg as well.
    let mut kwargs = HashMap::with_capacity(2);
    if let Some(tool_choice) = request.tool_choice {
        chat_utils::insert_tool_choice_kwarg(&mut kwargs, &ChatToolChoice::from(tool_choice));
    }
    if let Some(effort) = request.reasoning_effort {
        kwargs.insert(
            chat_utils::REASONING_EFFORT_KEY.to_string(),
            Value::String(effort.to_string()),
        );
    }
    let template_kwargs = (!kwargs.is_empty()).then_some(kwargs);
    // A trailing assistant message with text is a prefill the response
    // continues. As for chat `continue_final_message`, a renderer that
    // continues it natively keeps it; any other gets it popped and its text
    // appended after the generation prompt. Only the text survives that, so
    // a message that also has thinking stays a closed turn there.
    let continues_final_assistant = messages::continues_final_assistant(request.messages);
    let native_continuation = continues_final_assistant
        && tokenizer
            .renderer_capabilities()
            .native_assistant_continuation;
    let text_only = transformed_messages
        .last()
        .and_then(Value::as_object)
        .is_some_and(|message| message.keys().all(|key| key == "role" || key == "content"));
    let assistant_prefix = if continues_final_assistant && !native_continuation && text_only {
        transformed_messages.pop().and_then(|mut message| {
            match message.get_mut("content").map(Value::take) {
                Some(Value::String(text)) => Some(text),
                _ => None,
            }
        })
    } else {
        None
    };
    let params = ChatTemplateParams {
        add_generation_prompt: !native_continuation,
        continue_final_message: native_continuation,
        tools: tools_json.as_deref(),
        template_kwargs: template_kwargs.as_ref(),
        thinking,
        ..Default::default()
    };

    let rendered = tokenizer
        .apply_chat_template_with_encoding(
            &transformed_messages,
            params,
            assistant_prefix.as_deref(),
        )
        .map_err(|e| format!("Failed to apply chat template: {e}"))?;

    // Step 7: Build ProcessedMessages
    let text = anchor_rendered_markers(placeholder_tokens, &rendered.encoding, rendered.text);
    Ok((
        ProcessedMessages {
            text,
            unbilled_prompt_tokens: rendered.unbilled_prompt_tokens,
            continued_final_message: rendered.continued_final_message,
        },
        rendered.encoding,
    ))
}

/// The tools the template renders for a request: `chat_tools` (the custom
/// tools as chat tools) narrowed by `tool_choice` the way the chat path
/// narrows them (a forced tool renders alone, and `none` renders none unless
/// the renderer declares the tools under `none` itself, see
/// [`RendererCapabilities::declares_tools_under_tool_choice_none`]); borrowed
/// when the choice does not narrow. Shared by the preparation stage and
/// count_tokens so both render the same prompt.
pub fn template_tools<'a>(
    chat_tools: &'a [ChatTool],
    tool_choice: Option<&messages::ToolChoice>,
    capabilities: RendererCapabilities,
) -> Cow<'a, [ChatTool]> {
    if matches!(tool_choice, Some(messages::ToolChoice::None))
        && !capabilities.declares_tools_under_tool_choice_none
    {
        return Cow::Borrowed(&[]);
    }
    match tool_choice
        .map(ChatToolChoice::from)
        .and_then(|choice| choice.narrow_tools(chat_tools))
    {
        Some(narrowed) => Cow::Owned(narrowed),
        None => Cow::Borrowed(chat_tools),
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
    if tokenizer.renderer_capabilities().media_handling == MediaHandling::TemplateText {
        return Ok(MessagesMedia {
            media_order: MediaPartOrder::Authored,
            media: None,
        });
    }
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
    // Same refusal as the chat path: no images or video to a pool whose
    // every worker is language model only (see chat/preparation.rs); the
    // count answers it as `/v1/messages` does.
    multimodal::ensure_model_accepts_media(&components.worker_registry, model_id, &plan)?;
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
        |format| multimodal::fleet_reads_format(&components.worker_registry, model_id, format),
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

/// The prompt tokens `/v1/messages` would send for `prompt` (a message
/// request's or a count_tokens request's): the same tool set, media
/// placeholders, reasoning effort, template rendering and encoding as the
/// preparation stage, without media processing (an image counts as the
/// placeholder the template writes for it; the image itself is not fetched). A text or content-source
/// document counts as its rendered text; a source with no text is refused,
/// as `/v1/messages` refuses it.
pub(crate) async fn count_input_tokens(
    prompt: MessagesPrompt<'_>,
    tokenizer: Arc<dyn Tokenizer>,
    tools: &[ChatTool],
    placeholders: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<u32, CountError> {
    let (processed, encoding) = process_messages_prompt(
        prompt,
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
                // See https://github.com/smg-project/smg/issues/1795. A renderer
                // that writes no turn for a system message past the leading run
                // gets it as a user turn there instead, once the system prompt is
                // in (`system_messages_after_the_first_as_user_turns`).
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

/// For a renderer whose reference writes no turn for a system message after
/// the first ([`RendererCapabilities::leading_system_only`]): every `system`
/// message past the leading run becomes a user turn at its position. A
/// coding agent sends its environment block and its notes as `system`-role
/// messages inside `messages[]`, after tool results too; such a renderer
/// would write the message as bare text with no role marker and, for one
/// that ends the conversation, no assistant header, and the model answers
/// an empty turn. As a user turn the text keeps its place and the header
/// follows it, the way the family's newer encoder renders a mid-conversation
/// system message. The leading run stays: it is the system prompt, which
/// every renderer writes.
fn system_messages_after_the_first_as_user_turns(messages: &mut [Value]) {
    let mut leading_run = true;
    for message in messages.iter_mut() {
        let is_system = message.get("role").and_then(Value::as_str) == Some("system");
        if is_system && !leading_run {
            message["role"] = Value::String("user".to_owned());
        }
        leading_run &= is_system;
    }
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
    fn tool_schemas_render_in_the_clients_order() {
        use llm_tokenizer::chat_template::ChatTemplateProcessor;

        // More keys than a hash map would keep in order by chance.
        let names = [
            "zeta", "alpha", "mu", "beta", "omega", "delta", "kappa", "gamma",
        ];
        let properties: serde_json::Map<String, Value> = names
            .iter()
            .map(|name| (name.to_string(), json!({"type": "string"})))
            .collect();
        let body = json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "lookup", "input_schema": {
                "type": "object",
                "properties": properties,
                "required": ["mu", "alpha"],
                "additionalProperties": false
            }}]
        });
        let template = "{% for tool in tools %}{{ tool | tojson }}{% endfor %}";
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        // The tools of a fresh request, rendered as `process_messages` does.
        let render = || {
            let request: CreateMessageRequest = serde_json::from_value(body.clone()).unwrap();
            let tools: Vec<Value> = request
                .chat_tools()
                .iter()
                .map(|tool| serde_json::to_value(tool).unwrap())
                .collect();
            let params = ChatTemplateParams {
                tools: Some(&tools),
                ..Default::default()
            };
            processor.apply_chat_template(&[], params).unwrap()
        };

        // The same request renders the same prompt, so its prefix stays cached.
        let prompt = render();
        assert_eq!(prompt, render());
        let tool: Value = serde_json::from_str(&prompt).unwrap();
        let parameters = tool["function"]["parameters"].as_object().unwrap();
        let keys: Vec<&str> = parameters["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, names);
        let keys: Vec<&str> = parameters.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["type", "properties", "required", "additionalProperties"]
        );
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
        let chat_tools = request.chat_tools();
        let tools = template_tools(
            &chat_tools,
            request.tool_choice.as_ref(),
            RendererCapabilities::default(),
        );
        assert_eq!(tools.len(), 1, "a forced tool renders alone");

        let counted = count_input_tokens(
            MessagesPrompt::from(&request),
            tokenizer,
            &tools,
            None,
            MediaPartOrder::MediaFirst,
        )
        .await
        .unwrap();
        assert_eq!(counted, 5);
    }

    /// `tool_choice: {"type": "none"}` renders no tool, as on the chat path,
    /// unless the renderer declares the tools under `none` itself; `auto`
    /// renders them all.
    #[test]
    fn tool_choice_none_renders_no_tool() {
        let request = |tool_choice: Value| -> CountMessageTokensRequest {
            serde_json::from_value(json!({
                "model": "m",
                "tools": [{
                    "name": "get_weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
                }],
                "tool_choice": tool_choice,
                "messages": [{"role": "user", "content": "Weather in Paris?"}]
            }))
            .unwrap()
        };
        let plain = RendererCapabilities::default();
        let declaring = RendererCapabilities {
            declares_tools_under_tool_choice_none: true,
            ..RendererCapabilities::default()
        };
        let none = request(json!({"type": "none"}));
        assert!(template_tools(&none.chat_tools(), none.tool_choice.as_ref(), plain).is_empty());
        assert_eq!(
            template_tools(&none.chat_tools(), none.tool_choice.as_ref(), declaring).len(),
            1,
            "the renderer that declares the tools under none gets them"
        );
        let auto = request(json!({"type": "auto"}));
        assert_eq!(
            template_tools(&auto.chat_tools(), auto.tool_choice.as_ref(), plain).len(),
            1
        );
    }

    /// The Messages path tells the template the request's tool choice as the
    /// chat path tells it, whatever the renderer: the chat value under the
    /// chat path's kwarg (`any` as `required`, a named tool as its function
    /// object, `none` as `none` with the tools narrowed away), and no kwargs
    /// when the request names no choice.
    #[test]
    fn the_messages_path_tells_the_template_the_tool_choice() {
        let tok = llm_tokenizer::MockTokenizer::new().with_json_chat_template();
        let rendered = |tool_choice: Option<Value>| -> Value {
            let mut body = json!({
                "model": "m",
                "max_tokens": 16,
                "tools": [{
                    "name": "get_weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
                }],
                "messages": [{"role": "user", "content": "Weather in Paris?"}]
            });
            if let Some(tool_choice) = tool_choice {
                body["tool_choice"] = tool_choice;
            }
            let request: CreateMessageRequest = serde_json::from_value(body).unwrap();
            let chat_tools = request.chat_tools();
            let tools = template_tools(
                &chat_tools,
                request.tool_choice.as_ref(),
                tok.renderer_capabilities(),
            );
            let tools_for_template = if tools.is_empty() {
                None
            } else {
                Some(&*tools)
            };
            let (processed, _) =
                process_messages_with_encoding(&request, &tok, tools_for_template).unwrap();
            serde_json::from_str(&processed.text).unwrap()
        };
        let any = rendered(Some(json!({"type": "any"})));
        assert_eq!(any["template_kwargs"]["tool_choice"], json!("required"));
        assert_eq!(any["tools"].as_array().map(Vec::len), Some(1));
        let named = rendered(Some(json!({"type": "tool", "name": "get_weather"})));
        assert_eq!(
            named["template_kwargs"]["tool_choice"],
            json!({"type": "function", "function": {"name": "get_weather"}})
        );
        let none = rendered(Some(json!({"type": "none"})));
        assert_eq!(none["template_kwargs"]["tool_choice"], json!("none"));
        assert!(
            none.get("tools").is_none(),
            "a plain renderer gets no tool under none: {none}"
        );
        let unset = rendered(None);
        assert!(
            unset.get("template_kwargs").is_none(),
            "no choice, no kwargs: {unset}"
        );
    }

    /// The K3 stand-in vocabulary of the kit test (`tests/kimi_k3_tool_choice_none.rs`):
    /// byte-level ranks plus the K3 control tokens, under the config that
    /// selects the renderer, so the rendering reads back as text.
    fn k3_stand_in_dir() -> tempfile::TempDir {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let model: String = (0u32..256)
            .map(|b| format!("{} {b}\n", STANDARD.encode([b as u8])))
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("tiktoken.model"), model).expect("tiktoken.model");
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {
                "300": {"content": "<|open|>", "special": false},
                "301": {"content": "<|close|>", "special": false},
                "302": {"content": "<|sep|>", "special": false},
                "303": {"content": "<|end_of_msg|>", "special": true}
            }}"#,
        )
        .expect("tokenizer_config.json");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#,
        )
        .expect("config.json");
        dir
    }

    /// On the Messages path the K3 profile renders `tool_choice: {"type":
    /// "none"}` as its reference does and as the chat path renders it: the
    /// tool declaration before the conversation, the instruction not to call
    /// it after (the template learns the choice from the chat path's kwarg).
    #[test]
    fn k3_declares_the_tool_and_forbids_the_call_on_the_messages_path() {
        let dir = k3_stand_in_dir();
        let tok = llm_tokenizer::create_tokenizer(dir.path().to_str().expect("utf-8 path"))
            .expect("tokenizer");
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 64,
            "tools": [{
                "name": "get_weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
        }))
        .expect("the request");
        let chat_tools = request.chat_tools();
        let tools = template_tools(
            &chat_tools,
            request.tool_choice.as_ref(),
            tok.renderer_capabilities(),
        );
        let (processed, _) = process_messages(
            &request,
            tok.as_ref(),
            Some(&*tools),
            None,
            MediaPartOrder::MediaFirst,
        )
        .expect("render");
        let text = &processed.text;
        let declaration = text
            .find("Here are the available tools, described in JSONSchema.")
            .unwrap_or_else(|| panic!("no tool declaration in {text:?}"));
        assert!(text.contains(r#""name":"get_weather""#), "{text:?}");
        let user = text
            .find("What is the weather in Paris?")
            .unwrap_or_else(|| panic!("no user turn in {text:?}"));
        let forbids = text
            .find("You MUST NOT call any tools in the next message.")
            .unwrap_or_else(|| panic!("no tool_choice=none message in {text:?}"));
        assert!(
            declaration < user && user < forbids,
            "declaration at {declaration}, user turn at {user}, instruction at {forbids}"
        );
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
            MessagesPrompt::from(&request),
            Arc::clone(&tokenizer),
            &[],
            None,
            MediaPartOrder::MediaFirst,
        )
        .await
        .unwrap();
        let with = count_input_tokens(
            MessagesPrompt::from(&request),
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
            MessagesPrompt::from(&request),
            Arc::new(EncodeFails(llm_tokenizer::mock::MockTokenizer::new())),
            &[],
            None,
            MediaPartOrder::MediaFirst,
        )
        .await
        .unwrap_err();
        assert!(matches!(failure, CountError::Encode(_)), "{failure:?}");
    }

    /// A user turn, then `assistant` as the last message.
    fn ending_with(assistant: Value) -> CreateMessageRequest {
        serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "Hello"}, assistant]
        }))
        .unwrap()
    }

    fn render(request: &CreateMessageRequest, tokenizer: &dyn Tokenizer) -> String {
        process_messages(request, tokenizer, None, None, MediaPartOrder::MediaFirst)
            .unwrap()
            .0
            .text
    }

    /// A trailing assistant message with text is a prefill: a Jinja template
    /// continues it inside the turn it renders for it, header included,
    /// instead of closing it and opening a new turn.
    #[test]
    fn trailing_assistant_text_is_continued_inside_its_turn() {
        let mut tokenizer =
            llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
                .unwrap();
        tokenizer
            .set_chat_template(
                r"
{%- for m in messages -%}
{%- if m.role == 'assistant' -%}{{- '<|turn|>assistant<|to|>user<|body|>' + m.content + '<|end|>' -}}
{%- else -%}{{- '<|turn|>' + m.role + '<|body|>' + m.content + '<|end|>' -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|turn|>assistant' -}}{%- endif -%}"
                    .to_string(),
            )
            .unwrap();
        let request = ending_with(json!({"role": "assistant", "content": [
            {"type": "text", "text": "Sure"}
        ]}));
        assert_eq!(
            render(&request, &tokenizer),
            "<|turn|>user<|body|>Hello<|end|><|turn|>assistant<|to|>user<|body|>Sure"
        );
    }

    /// A renderer without native continuation gets the prefill appended
    /// after the generation prompt, as chat `continue_final_message` does.
    #[test]
    fn trailing_assistant_text_follows_the_generation_prompt_elsewhere() {
        let tokenizer = llm_tokenizer::MockTokenizer::new();
        let request = ending_with(json!({"role": "assistant", "content": "Sure"}));
        assert_eq!(render(&request, &tokenizer), "user: Hello\nassistant: Sure");
    }

    /// A trailing assistant message without text (tool calls only) is not a
    /// prefill: it is rendered as a closed turn before a new one.
    #[test]
    fn trailing_tool_use_still_opens_a_new_turn() {
        let tokenizer = llm_tokenizer::MockTokenizer::new();
        let request = ending_with(json!({"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
        ]}));
        assert_eq!(
            render(&request, &tokenizer),
            "user: Hello\nassistant: \nassistant: "
        );
    }

    /// The coding agent's shape: its environment block after the first user
    /// turn and a note after every tool result, both `system`-role messages
    /// inside `messages[]`, beside a top-level system prompt.
    fn coding_agent_request() -> CreateMessageRequest {
        serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "system": "You are a coding assistant.",
            "messages": [
                {"role": "user", "content": "Fix the failing test."},
                {"role": "system", "content": "# Environment\nworking directory: /work"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Run the suite first.", "signature": "sig"},
                    {"type": "tool_use", "id": "toolu_01", "name": "Bash",
                     "input": {"command": "pytest -q"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "1 failed, 14 passed"}
                ]},
                {"role": "system", "content": [
                    {"type": "text", "text": "<total_tokens>14994851 tokens left</total_tokens>",
                     "cache_control": {"type": "ephemeral"}}
                ]}
            ]
        }))
        .expect("the request")
    }

    /// A `system`-role message inside `messages[]` is forwarded in place
    /// unless the renderer writes no turn for a system message after the
    /// first: then every one past the leading run is a user turn at its
    /// position, so the prompt keeps its assistant header after the note
    /// that ends the conversation and before the historical assistant turn.
    #[test]
    fn system_messages_after_the_first_are_user_turns_for_a_leading_system_only_renderer() {
        let request = coding_agent_request();
        let in_place = llm_tokenizer::MockTokenizer::new();
        assert_eq!(
            render(&request, &in_place),
            "system: You are a coding assistant.\n\
             user: Fix the failing test.\n\
             system: # Environment\nworking directory: /work\n\
             assistant: \n\
             tool: 1 failed, 14 passed\n\
             system: <total_tokens>14994851 tokens left</total_tokens>\n\
             assistant: "
        );
        let leading_system_only =
            llm_tokenizer::MockTokenizer::new().with_renderer_capabilities(RendererCapabilities {
                leading_system_only: true,
                ..RendererCapabilities::default()
            });
        assert_eq!(
            render(&request, &leading_system_only),
            "system: You are a coding assistant.\n\
             user: Fix the failing test.\n\
             user: # Environment\nworking directory: /work\n\
             assistant: \n\
             tool: 1 failed, 14 passed\n\
             user: <total_tokens>14994851 tokens left</total_tokens>\n\
             assistant: "
        );
    }

    /// Without a top-level system prompt a leading `system` message is the
    /// system prompt, which every renderer writes: it stays where it is.
    #[test]
    fn a_leading_system_message_stays_the_system_prompt_for_every_renderer() {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Note"}
            ]
        }))
        .expect("the request");
        let leading_system_only =
            llm_tokenizer::MockTokenizer::new().with_renderer_capabilities(RendererCapabilities {
                leading_system_only: true,
                ..RendererCapabilities::default()
            });
        assert_eq!(
            render(&request, &leading_system_only),
            "system: Be terse.\nuser: Hi\nuser: Note\nassistant: "
        );
    }

    /// With a top-level system prompt and a `system` message opening the
    /// array, both are the leading run: the system prompt, which stays as it
    /// is for every renderer; only the system messages past it become user
    /// turns for a leading-system-only renderer.
    #[test]
    fn a_top_level_system_and_a_leading_system_message_are_both_the_system_prompt() {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 8,
            "system": "Be terse.",
            "messages": [
                {"role": "system", "content": "# Environment"},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Note"}
            ]
        }))
        .expect("the request");
        let leading_system_only =
            llm_tokenizer::MockTokenizer::new().with_renderer_capabilities(RendererCapabilities {
                leading_system_only: true,
                ..RendererCapabilities::default()
            });
        assert_eq!(
            render(&request, &leading_system_only),
            "system: Be terse.\nsystem: # Environment\nuser: Hi\nuser: Note\nassistant: "
        );
        assert_eq!(
            render(&request, &llm_tokenizer::MockTokenizer::new()),
            "system: Be terse.\nsystem: # Environment\nuser: Hi\nsystem: Note\nassistant: "
        );
    }

    /// A model directory the DeepSeek V4 renderer is chosen for, with a
    /// tokenizer that loads (the render never encodes).
    fn deepseek_v4_stand_in_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("tokenizer.json"),
            r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null, "decoder": null,
                "model": {"type": "BPE", "vocab": {"hello": 0}, "merges": []}}"#,
        )
        .expect("tokenizer.json");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}"#,
        )
        .expect("config.json");
        dir
    }

    /// Through the DeepSeek V4 renderer the coding agent's notes are user
    /// turns the encoder merges into the turn before them: the prompt ends
    /// with the note, the assistant header and the open thought, where the
    /// in-place system message left it ending with the bare note; and the
    /// historical assistant turn has its header after the environment block.
    /// The prompt is rendered before the parser is chosen, so every parser
    /// reads this one.
    #[test]
    fn the_deepseek_v4_renderer_gets_the_coding_agents_notes_as_user_turns() {
        let dir = deepseek_v4_stand_in_dir();
        let tok = llm_tokenizer::create_tokenizer(dir.path().to_str().expect("utf-8 path"))
            .expect("tokenizer");
        assert!(tok.renderer_capabilities().leading_system_only);
        let text = render(&coding_agent_request(), tok.as_ref());
        assert!(
            text.ends_with(
                "<tool_result>1 failed, 14 passed</tool_result>\n\n\
                 <total_tokens>14994851 tokens left</total_tokens><｜Assistant｜><think>"
            ),
            "{text:?}"
        );
        assert!(
            text.contains(
                "<｜User｜>Fix the failing test.\n\n# Environment\nworking directory: /work<｜Assistant｜>"
            ),
            "{text:?}"
        );
    }

    /// An assistant turn as Qwen3 renders one: reasoning before the text,
    /// tool calls after it.
    const REASONING_TEXT_CALLS: &str = r"
{%- for m in messages -%}
{{- '<|im_start|>' + m.role + '\n' -}}
{%- if m.role == 'assistant' -%}
{%- if m.reasoning_content -%}{{- '<think>' + m.reasoning_content + '</think>' -}}{%- endif -%}
{{- m.content or '' -}}
{%- for tc in m.tool_calls or [] -%}{{- '<tool_call>' + tc.function.name + '</tool_call>' -}}{%- endfor -%}
{%- else -%}{{- m.content -}}{%- endif -%}
{{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}";

    fn jinja(template: &str) -> llm_tokenizer::TiktokenTokenizer {
        let mut tokenizer =
            llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
                .unwrap();
        tokenizer.set_chat_template(template.to_string()).unwrap();
        tokenizer
    }

    /// Text followed by a tool call is a closed turn: continuing after the
    /// text would drop the call, since transformers' `continue_final_message`
    /// cuts the prompt there.
    #[test]
    fn trailing_text_with_tool_use_is_a_closed_turn() {
        let request = ending_with(json!({"role": "assistant", "content": [
            {"type": "text", "text": "Sure"},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
        ]}));
        assert_eq!(
            render(&request, &jinja(REASONING_TEXT_CALLS)),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n\
             Sure<tool_call>f</tool_call><|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// Thinking then text is continued after the text, its reasoning kept, as
    /// transformers continues it.
    #[test]
    fn trailing_thinking_and_text_is_continued_with_its_reasoning() {
        let request = ending_with(json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "s"},
            {"type": "text", "text": "Sure"}
        ]}));
        assert_eq!(
            render(&request, &jinja(REASONING_TEXT_CALLS)),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>plan</think>Sure"
        );
    }

    /// Without native continuation only the text would follow the generation
    /// prompt, so a message that also has thinking or a tool call is a
    /// closed turn and keeps them.
    #[test]
    fn mixed_trailing_assistant_keeps_its_fields_elsewhere() {
        let tokenizer = llm_tokenizer::MockTokenizer::new().with_json_chat_template();
        let thinking = json!({"type": "thinking", "thinking": "plan", "signature": "s"});
        let text = json!({"type": "text", "text": "Sure"});
        let tool_use = json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}});
        let cases = [
            (
                json!([thinking, text]),
                json!({"role": "assistant", "content": "Sure", "reasoning_content": "plan"}),
            ),
            (
                json!([text, tool_use]),
                json!({"role": "assistant", "content": "Sure", "tool_calls": [
                    {"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": {}}}
                ]}),
            ),
        ];
        for (content, kept) in cases {
            let request = ending_with(json!({"role": "assistant", "content": content}));
            let rendered = render(&request, &tokenizer);
            let rendered: Value = serde_json::from_str(&rendered)
                .unwrap_or_else(|_| panic!("text appended after the prompt: {rendered}"));
            assert_eq!(rendered["add_generation_prompt"], json!(true));
            assert_eq!(rendered["messages"][1], kept);
        }
    }

    /// `output_config.effort` reads as a Chat `reasoning_effort` does, and
    /// reaches the chat template as the `reasoning_effort` kwarg.
    #[test]
    fn output_config_effort_is_rendered_as_the_reasoning_effort() {
        let request = |output_config: Value| -> CreateMessageRequest {
            serde_json::from_value(json!({
                "model": "m", "max_tokens": 8, "output_config": output_config,
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .unwrap()
        };
        let mut tokenizer =
            llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
                .unwrap();
        tokenizer
            .set_chat_template(
                "{%- for m in messages -%}{{ m.role }}: {{ m.content }} {% endfor -%}\
                 effort={{ reasoning_effort | default('unset') }}"
                    .to_string(),
            )
            .unwrap();
        let rendered = |output_config| {
            process_messages(
                &request(output_config),
                &tokenizer,
                None,
                None,
                MediaPartOrder::MediaFirst,
            )
            .unwrap()
            .0
            .text
        };
        assert_eq!(
            rendered(json!({"effort": "high"})),
            "user: Hello effort=high"
        );
        assert_eq!(rendered(json!({"effort": 64})), "user: Hello effort=64");
        assert_eq!(
            rendered(json!({"effort": null})),
            "user: Hello effort=unset"
        );
        assert_eq!(rendered(json!({})), "user: Hello effort=unset");
    }

    /// The reasoning parser is armed from the effort the template renders: on
    /// a template that switches thinking through `reasoning_effort`, an
    /// `output_config.effort` that closes the think block leaves the parser
    /// unarmed, as the same effort does for a Chat request. Without a thinking
    /// block, "none" turns a toggle template's thinking off, as for Chat.
    #[test]
    fn output_config_effort_arms_the_reasoning_parser_as_rendered() {
        use crate::routers::grpc::utils::{
            chat_reasoning_starts_in_prefill, messages_reasoning_starts_in_prefill,
        };
        let mut tokenizer =
            llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
                .unwrap();
        tokenizer
            .set_chat_template(
                "{# think_begin_token #}{%- for m in messages -%}{{ m.role }}: {{ m.content }}\n{% endfor -%}\
                 {%- if reasoning_effort is defined and reasoning_effort == 'no_think' -%}assistant: <think></think>\
                 {%- else -%}assistant: <think>{%- endif -%}"
                    .to_string(),
            )
            .unwrap();
        for (effort, prompt_end, armed) in [
            (json!("no_think"), "<think></think>", false),
            (json!("high"), "<think>", true),
            (Value::Null, "<think>", true),
        ] {
            let request: CreateMessageRequest = serde_json::from_value(json!({
                "model": "m", "max_tokens": 8, "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .unwrap();
            let rendered =
                process_messages(&request, &tokenizer, None, None, MediaPartOrder::MediaFirst)
                    .unwrap()
                    .0
                    .text;
            assert_eq!(rendered, format!("user: Hello\nassistant: {prompt_end}"));
            assert_eq!(
                messages_reasoning_starts_in_prefill(&request, false, &tokenizer),
                armed,
                "{effort}"
            );
            let chat: openai_protocol::chat::ChatCompletionRequest =
                serde_json::from_value(json!({
                    "model": "m", "reasoning_effort": effort,
                    "messages": [{"role": "user", "content": "Hello"}]
                }))
                .unwrap();
            assert_eq!(
                chat_reasoning_starts_in_prefill(&chat, &tokenizer),
                armed,
                "chat {effort}"
            );
        }

        tokenizer
            .set_chat_template(
                "{%- for m in messages -%}{{ m.role }}: {{ m.content }} {% endfor -%}\
                 thinking={{ enable_thinking | default('unset') }}"
                    .to_string(),
            )
            .unwrap();
        let rendered = |effort: &str| {
            let request: CreateMessageRequest = serde_json::from_value(json!({
                "model": "m", "max_tokens": 8, "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .unwrap();
            process_messages(&request, &tokenizer, None, None, MediaPartOrder::MediaFirst)
                .unwrap()
                .0
                .text
        };
        assert_eq!(rendered("none"), "user: Hello thinking=False");
        assert_eq!(rendered("high"), "user: Hello thinking=unset");
    }
}
