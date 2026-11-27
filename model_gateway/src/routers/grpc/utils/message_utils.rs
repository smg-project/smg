//! Message API utilities for converting Anthropic Messages API types
//! into the internal chat template format.
//!
//! Parallel to `chat_utils.rs` but works with `CreateMessageRequest` / `InputMessage`
//! instead of `ChatCompletionRequest` / `ChatMessage`.

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use axum::response::Response;
use llm_multimodal::{MediaPartOrder, Modality};
use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams, PreparedTools},
    traits::{PromptEncoding, Tokenizer},
};
use openai_protocol::{
    common::{Tool as ChatTool, ToolChoice as ChatToolChoice},
    messages::{
        self, CountMessageTokensRequest, CreateMessageRequest, InputContent, InputContentBlock,
        InputMessage, SystemContent, ToolResultContent,
    },
    profile::ProviderProfile,
};
use serde_json::{json, Value};

use super::{
    chat_utils::{self, PromptError},
    prepared_tools::prepare_tools,
};
use crate::routers::grpc::{
    context::SharedComponents,
    multimodal::{self, anchor_rendered_markers, resolve_media, PlaceholderTokens, ResolvedMedia},
    ProcessedMessages,
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
    /// `output_config.effort` as the provider profile hands it to the
    /// template (`CreateMessageRequest::template_reasoning_effort_for`),
    /// rendered as the `reasoning_effort` kwarg.
    pub reasoning_effort: Option<&'a str>,
    /// The request's tool choice, told to the template as the chat path
    /// tells it; a renderer that declares the tools under `none` writes its
    /// instruction not to call them from it.
    pub tool_choice: Option<&'a messages::ToolChoice>,
}

impl<'a> MessagesPrompt<'a> {
    /// The prompt parts of `request`, with the model's provider profile the
    /// caller derived already (the preparation stage derives it once per
    /// request); `From` derives it for a caller without one.
    pub(crate) fn for_request(
        request: &'a CreateMessageRequest,
        provider: ProviderProfile,
    ) -> Self {
        Self {
            messages: &request.messages,
            system: request.system.as_ref(),
            thinking: request.template_thinking_for(provider),
            reasoning_effort: request.template_reasoning_effort_for(provider),
            tool_choice: request.tool_choice.as_ref(),
        }
    }
}

impl<'a> From<&'a CreateMessageRequest> for MessagesPrompt<'a> {
    fn from(request: &'a CreateMessageRequest) -> Self {
        Self::for_request(request, ProviderProfile::for_model(&request.model))
    }
}

impl<'a> From<&'a CountMessageTokensRequest> for MessagesPrompt<'a> {
    fn from(request: &'a CountMessageTokensRequest) -> Self {
        let provider = ProviderProfile::for_model(&request.model);
        Self {
            messages: &request.messages,
            system: request.system.as_ref(),
            thinking: request.template_thinking_for(provider),
            reasoning_effort: request.template_reasoning_effort_for(provider),
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
    provider: ProviderProfile,
) -> Result<(ProcessedMessages, PromptEncoding), PromptError> {
    process_messages_prompt(
        MessagesPrompt::for_request(request, provider),
        tokenizer,
        chat_tools,
        placeholder_tokens,
        media_order,
    )
}

/// `process_messages` for a request without media: the entry point for
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
        ProviderProfile::for_model(&request.model),
    )
    .map_err(String::from)
}

/// [`process_messages`] over the prompt parts alone.
fn process_messages_prompt(
    request: MessagesPrompt<'_>,
    tokenizer: &dyn Tokenizer,
    chat_tools: Option<&[ChatTool]>,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<(ProcessedMessages, PromptEncoding), PromptError> {
    let content_format = tokenizer.chat_template_content_format_for_request(chat_tools.is_some());

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
        chat_utils::process_tool_call_arguments(&mut transformed_messages);
    }

    // Step 4: the tools as the template reads them, prepared once per
    // distinct tool set and shared by every request that carries the same
    // tools.
    let prepared_tools = chat_tools.map(prepare_tools).transpose()?;

    // Step 5: the thinking preference the request states, or its provider
    // profile reads from the effort, as for Chat.
    let thinking = request.thinking;

    // Step 6: Apply chat template. The template learns the tool choice the
    // way the chat path tells it (the chat value under the same kwarg): a
    // renderer that declares the tools under `none` writes its instruction
    // not to call them from this key; a renderer that never reads the key
    // is not handed it. The effort rides under the chat path's
    // `reasoning_effort` kwarg as well.
    let mut kwargs = HashMap::with_capacity(2);
    if let Some(tool_choice) = request
        .tool_choice
        .filter(|_| tokenizer.template_reads(chat_utils::TOOL_CHOICE_KEY))
    {
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
        tools: prepared_tools.as_deref().map(PreparedTools::tools),
        prepared_tools: prepared_tools.as_deref(),
        template_kwargs: template_kwargs.as_ref(),
        thinking,
        ..Default::default()
    };

    let rendered = tokenizer
        .apply_chat_template_with_encoding_owned(
            transformed_messages,
            params,
            assistant_prefix.as_deref(),
        )
        .map_err(|e| PromptError::of_render(&e))?;

    // Step 7: Build ProcessedMessages
    let text = anchor_rendered_markers(placeholder_tokens, &rendered.encoding, rendered.text);
    Ok((
        ProcessedMessages {
            text,
            unbilled_prompt_tokens: rendered.unbilled_prompt_tokens,
            continued_final_message: rendered.continued_final_message,
            generation_prompt: rendered.generation_prompt,
        },
        rendered.encoding,
    ))
}

/// The tools the template renders for a request: `chat_tools` (the custom
/// tools as chat tools) narrowed by `tool_choice` the way the chat path
/// narrows them (a forced tool renders alone; `none` renders them all, as
/// the chat path and the engine's own server render them, the choice
/// reaching the template as a kwarg); borrowed when the choice does not
/// narrow. Shared by the preparation stage and count_tokens so both render
/// the same prompt.
pub fn template_tools<'a>(
    chat_tools: &'a [ChatTool],
    tool_choice: Option<&messages::ToolChoice>,
) -> Cow<'a, [ChatTool]> {
    match tool_choice
        .map(ChatToolChoice::from)
        .and_then(|choice| choice.narrow_tools(chat_tools))
    {
        Some(narrowed) => Cow::Owned(narrowed),
        None => Cow::Borrowed(chat_tools),
    }
}

/// Resolve what a Messages request's media needs before the template renders
/// (see [`resolve_media`]). The refusals are the preparation stage's, so
/// count_tokens answers a request the way `/v1/messages` would.
pub(crate) async fn resolve_messages_media(
    components: &SharedComponents,
    model_id: &str,
    tokenizer: &dyn Tokenizer,
    messages: &[InputMessage],
    record_media: bool,
    has_tools: bool,
) -> Result<ResolvedMedia, Response> {
    resolve_media(
        components,
        model_id,
        tokenizer,
        "resolve_messages_media",
        || multimodal::media_plan_messages(messages),
        record_media,
        has_tools,
    )
    .await
}

/// Why a count could not be produced: the request does not render (the
/// client's mistake, a 400 on `/v1/messages`, or the template's own failure,
/// a 500 there; the [`PromptError`] says which) or the rendered prompt does
/// not encode (a gateway fault, a 500 there).
#[derive(Debug)]
pub(crate) enum CountError {
    Render(PromptError),
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
        CountError::Render(PromptError::Failed(
            "the prompt has more tokens than a count can report".to_owned(),
        ))
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
fn process_message_content_format(
    messages: &[InputMessage],
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<Vec<Value>, PromptError> {
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
                    result.push(convert_assistant_message(
                        index,
                        &message.content,
                        content_format,
                    )?);
                }
                // A `system`-role message in `messages[]` (e.g. from Claude Code) is
                // forwarded in place, preserving its position in the conversation so
                // inline-`system` chat templates render it where it was sent.
                // See https://github.com/smg-project/smg/issues/1795. A renderer
                // that writes no turn for a system message past the leading run
                // gets it as a user turn there instead, once the system prompt is
                // in (`system_messages_after_the_first_as_user_turns`).
                messages::Role::System => {
                    result.push(convert_system_message(index, &message.content)?);
                }
            }
            Ok(result)
        })
}

/// Convert a `system`-role message's content to a chat-template JSON value,
/// preserving its position in `messages[]`. System content is text; text blocks
/// are concatenated. An image, document, server tool result, or container upload
/// is refused by name; the other non-text blocks are dropped.
fn convert_system_message(index: usize, content: &InputContent) -> Result<Value, PromptError> {
    let text = match content {
        InputContent::String(text) => text.clone(),
        InputContent::Blocks(blocks) => {
            let mut texts = Vec::<&str>::new();
            for (position, block) in blocks.iter().enumerate() {
                match block {
                    InputContentBlock::Text(t) => texts.push(t.text.as_str()),
                    InputContentBlock::Image(_) => {
                        return Err(user_turn_only_refusal("image", index, position));
                    }
                    InputContentBlock::Document(_) => {
                        return Err(user_turn_only_refusal("document", index, position));
                    }
                    InputContentBlock::WebFetchToolResult(_)
                    | InputContentBlock::CodeExecutionToolResult(_)
                    | InputContentBlock::BashCodeExecutionToolResult(_)
                    | InputContentBlock::TextEditorCodeExecutionToolResult(_)
                    | InputContentBlock::ContainerUpload(_) => {
                        return Err(server_tool_block_refusal(block, index, position));
                    }
                    _ => {}
                }
            }
            texts.join("\n")
        }
    };
    Ok(json!({"role": "system", "content": text}))
}

/// For a renderer whose reference writes no turn for a system message after
/// the first ([`llm_tokenizer::traits::RendererCapabilities::leading_system_only`]): every `system`
/// message past the leading run becomes a user turn at its position. A
/// coding agent sends its environment block and its notes as `system`-role
/// messages inside `messages[]`, after tool results too; such a renderer
/// writes the message as bare text with no role marker, glued to the text
/// before it (the assistant header after it follows the reference's own
/// rule, not the message's role). As a user turn the text keeps its place
/// and is set off from the text before it as the user's words, the way the
/// family's newer encoder gives a mid-conversation system message a turn of
/// its own. The leading run stays: it is the system prompt, which every
/// renderer writes.
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
) -> Result<(), PromptError> {
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
                    InputContentBlock::Image(image) => {
                        file_image_refusal(image, index, position)?;
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
                        for image in tool_result_image_blocks(tr) {
                            file_image_refusal(image, index, position)?;
                            user_parts.push(json!({"type": "image"}));
                        }
                    }
                    // An MCP exchange belongs to the MCP loop, which no engine
                    // route runs: replayed here it is refused by name, as a
                    // document this gateway cannot render is, never dropped.
                    InputContentBlock::McpToolUse(_) | InputContentBlock::McpToolResult(_) => {
                        return Err(mcp_block_refusal(block, index, position));
                    }
                    // A server tool's result or a container upload is the
                    // public API's own server tools at work, which no engine
                    // route runs: refused by name, not dropped. The older
                    // server tool blocks (`server_tool_use`,
                    // `web_search_tool_result`, `tool_search_tool_result`)
                    // are still dropped, as before: refusing them would turn
                    // requests that are answered today into 400s.
                    InputContentBlock::WebFetchToolResult(_)
                    | InputContentBlock::CodeExecutionToolResult(_)
                    | InputContentBlock::BashCodeExecutionToolResult(_)
                    | InputContentBlock::TextEditorCodeExecutionToolResult(_)
                    | InputContentBlock::ContainerUpload(_) => {
                        return Err(server_tool_block_refusal(block, index, position));
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
                )?;
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
/// text renders: a text source, or a content source given as a string or
/// made of text blocks. A PDF or URL source needs a document reader this
/// gateway has no path for and is refused naming the block (`at`), not
/// handed to the template: the request's own, as every refusal here is.
fn render_document(document: &messages::DocumentBlock, at: &str) -> Result<String, PromptError> {
    let text = match &document.source {
        messages::DocumentSource::Text { data }
        | messages::DocumentSource::Content {
            content: messages::DocumentContent::String(data),
        } => data.as_str(),
        messages::DocumentSource::Content {
            content: messages::DocumentContent::Blocks(content),
        } => {
            let mut texts = Vec::with_capacity(content.len());
            for block in content {
                match block {
                    InputContentBlock::Text(t) => texts.push(t.text.as_str()),
                    _ => {
                        return Err(PromptError::Refused(format!(
                            "{at}: a document content source may hold text blocks only"
                        )))
                    }
                }
            }
            return Ok(document_text(document, &texts.join("\n")));
        }
        messages::DocumentSource::Base64 { media_type, .. } => {
            return Err(PromptError::Refused(format!(
                "{at}: document source type 'base64' ({media_type}) is not supported; \
                 send the document as a text or content source"
            )))
        }
        messages::DocumentSource::Url { .. } => {
            return Err(PromptError::Refused(format!(
                "{at}: document source type 'url' is not supported; send the document \
                 as a text or content source"
            )))
        }
        messages::DocumentSource::File { .. } => {
            return Err(PromptError::Refused(format!(
                "{at}: document source type 'file' is not supported; send the document \
                 as a text or content source"
            )))
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

/// An image whose source is a file of the public API's Files API, which no
/// engine route can read, refused naming the block that holds it, as a
/// document source this gateway cannot render is; any other image passes.
fn file_image_refusal(
    image: &messages::ImageBlock,
    index: usize,
    position: usize,
) -> Result<(), PromptError> {
    match image.source {
        messages::ImageSource::File { .. } => Err(PromptError::Refused(format!(
            "messages.{index}.content.{position}: image source type 'file' is not supported; \
             send the image as a base64 or url source"
        ))),
        _ => Ok(()),
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

/// The refusal of a replayed MCP block (`mcp_tool_use`, `mcp_tool_result`)
/// on an engine route, naming the block and its place. The exchange belongs
/// to the MCP loop, which no engine route runs: the prompt cannot carry it
/// faithfully, and dropping it in silence would answer without the call.
fn mcp_block_refusal(block: &InputContentBlock, index: usize, position: usize) -> PromptError {
    let kind = match block {
        InputContentBlock::McpToolResult(_) => "mcp_tool_result",
        _ => "mcp_tool_use",
    };
    PromptError::Refused(format!(
        "messages.{index}.content.{position}: block type '{kind}' is not supported on an engine \
         route, which runs no MCP loop; send the turn to a provider route or drop the exchange"
    ))
}

/// The public API accepts image and document blocks only in user turns.
fn user_turn_only_refusal(kind: &str, index: usize, position: usize) -> PromptError {
    PromptError::Refused(format!(
        "messages.{index}.content.{position}: {kind} blocks are supported in user messages only"
    ))
}

/// The refusal of a block of the public API's server tools (a web fetch, code
/// execution, bash or text editor result, a container upload) on an engine
/// route, naming the block and its place. No engine route runs those tools, so
/// the prompt cannot carry the block faithfully, and dropping it in silence
/// would answer a turn whose input it threw away (an upload-only user turn
/// would leave no user turn at all). The server tool blocks the request read
/// before these (`server_tool_use`, `web_search_tool_result`,
/// `tool_search_tool_result`) are still dropped, as they were: refusing them
/// is a separate choice, which would turn requests answered today into 400s.
fn server_tool_block_refusal(
    block: &InputContentBlock,
    index: usize,
    position: usize,
) -> PromptError {
    let kind = match block {
        InputContentBlock::WebFetchToolResult(_) => "web_fetch_tool_result",
        InputContentBlock::CodeExecutionToolResult(_) => "code_execution_tool_result",
        InputContentBlock::BashCodeExecutionToolResult(_) => "bash_code_execution_tool_result",
        InputContentBlock::TextEditorCodeExecutionToolResult(_) => {
            "text_editor_code_execution_tool_result"
        }
        _ => "container_upload",
    };
    PromptError::Refused(format!(
        "messages.{index}.content.{position}: block type '{kind}' is not supported on an engine \
         route, which runs no server tools; send the turn to a provider route or drop the block"
    ))
}

/// Convert an assistant message content to a single JSON value.
///
/// Extracts text content, tool calls, and reasoning/thinking into the
/// appropriate JSON fields that the chat template expects. An image, a
/// document, or a replayed MCP block is refused by name rather than dropped.
fn convert_assistant_message(
    index: usize,
    content: &InputContent,
    content_format: ChatTemplateContentFormat,
) -> Result<Value, PromptError> {
    let blocks = match content {
        InputContent::String(text) => return Ok(json!({"role": "assistant", "content": text})),
        InputContent::Blocks(blocks) => blocks,
    };
    let mut text_parts = Vec::<String>::new();
    let mut tool_calls = Vec::<Value>::new();
    let mut thinking_parts = Vec::<String>::new();
    for (position, block) in blocks.iter().enumerate() {
        match block {
            InputContentBlock::Text(t) => text_parts.push(t.text.clone()),
            InputContentBlock::ToolUse(tu) => tool_calls.push(json!({
                "id": tu.id,
                "type": "function",
                "function": {
                    "name": tu.name,
                    "arguments": serde_json::to_string(&tu.input)
                        .unwrap_or_else(|_| "{}".to_string())
                }
            })),
            InputContentBlock::Thinking(t) => thinking_parts.push(t.thinking.clone()),
            InputContentBlock::Image(_) => {
                return Err(user_turn_only_refusal("image", index, position));
            }
            InputContentBlock::Document(_) => {
                return Err(user_turn_only_refusal("document", index, position));
            }
            // The exchange the public API puts in the assistant turn:
            // refused by name, never dropped from the prompt.
            InputContentBlock::McpToolUse(_) | InputContentBlock::McpToolResult(_) => {
                return Err(mcp_block_refusal(block, index, position));
            }
            // A server tool's result, as the user turn's, the older
            // server tool blocks still dropped.
            InputContentBlock::WebFetchToolResult(_)
            | InputContentBlock::CodeExecutionToolResult(_)
            | InputContentBlock::BashCodeExecutionToolResult(_)
            | InputContentBlock::TextEditorCodeExecutionToolResult(_)
            | InputContentBlock::ContainerUpload(_) => {
                return Err(server_tool_block_refusal(block, index, position));
            }
            _ => {}
        }
    }

    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), Value::String("assistant".into()));

    // With no text blocks (a tool-use-only turn) the content is the
    // empty content the engine's chat utils hand a template in its
    // format (`""` or `[]`), never null, which a template that tests
    // or concatenates every assistant message's content cannot take.
    let content = if text_parts.is_empty() {
        chat_utils::empty_content(content_format)
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

    Ok(Value::Object(obj))
}

/// Format content parts based on the template's content format preference.
///
/// - `String` format: join text parts into a single string, the media parts
///   placed as `media_order` says (the template cannot place them itself)
/// - `OpenAI` format: keep as array of typed parts, in the request's order:
///   the template walks them and writes each anchor where its part sits, as
///   the engine's own server hands them to such a template
fn format_content_parts(
    parts: Vec<Value>,
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<Value, PromptError> {
    match content_format {
        ChatTemplateContentFormat::String => {
            let ordered = order_media_parts(parts, media_order);
            // Text parts as they are, an image part as its placeholder. An
            // image the placeholders hold no token for is refused, as the
            // chat renderer refuses it, and as the request's own: left out,
            // the prompt would carry no anchor for media that still travels
            // with the request. A caller without model metadata passes no
            // placeholders and its media parts are left out.
            let text: String = ordered
                .iter()
                .filter_map(|p| {
                    let obj = p.as_object()?;
                    let type_str = obj.get("type")?.as_str()?;
                    match type_str {
                        "text" => obj.get("text")?.as_str().map(|text| Ok(String::from(text))),
                        "image" => placeholder_tokens.map(|tokens| {
                            tokens
                                .get(Modality::Image)
                                .map(String::from)
                                .ok_or_else(|| {
                                    PromptError::Refused(
                                        "missing image placeholder for string-format chat template"
                                            .to_owned(),
                                    )
                                })
                        }),
                        _ => None,
                    }
                })
                .collect::<Result<Vec<_>, PromptError>>()?
                .join("\n");
            Ok(Value::String(text))
        }
        ChatTemplateContentFormat::OpenAI => Ok(Value::Array(parts)),
    }
}

/// For a template that takes one string per message: hoist media parts
/// before text under `MediaFirst` (the engine's own server's placement for
/// such a template); `Authored` keeps the request's order. `partition` is
/// stable so relative order within each group is preserved.
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

/// Empties the text of the thinking blocks in `content`, as a
/// `thinking.display` of `omitted` returns them: the blocks and their
/// signature stay where they are, the reasoning does not leave the gateway
/// (see `ThinkingDisplay::returns_text`).
pub(crate) fn omit_thinking_text(content: &mut [messages::ContentBlock]) {
    for block in content {
        if let messages::ContentBlock::Thinking { thinking, .. } = block {
            thinking.clear();
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/utils/message_utils/tests.rs"]
mod tests;
