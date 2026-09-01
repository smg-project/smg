//! Harmony request builder
//!
//! Handles encoding of Chat/Responses requests into Harmony format using openai-harmony library.

use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

use chrono::Local;
use openai_harmony::{
    chat::{
        Author, ChannelConfig, Content, Conversation, DeveloperContent, Message as HarmonyMessage,
        ReasoningEffort, Role, SystemContent, TextContent, ToolDescription,
    },
    HarmonyEncoding, HarmonyEncodingName,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{ChatLogProbs, ContentPart, Tool},
    responses::{
        ReasoningEffort as ResponsesReasoningEffort, ResponseContentPart, ResponseInput,
        ResponseInputOutputItem, ResponseReasoningContent, ResponseTool, ResponsesRequest,
        StringOrContentParts,
    },
};
use serde_json::json;
use tracing::{debug, trace, warn};

use super::types::HarmonyBuildOutput;
use crate::routers::grpc::{proto_wrapper::ProtoOutputLogProbs, utils};

/// Global Harmony encoding (lazy-initialized)
static HARMONY_ENCODING: OnceLock<HarmonyEncoding> = OnceLock::new();

fn reject_chat_audio(messages: &[ChatMessage]) -> Result<(), String> {
    let has_audio = messages.iter().any(|message| match message {
        ChatMessage::System { content, .. }
        | ChatMessage::User { content, .. }
        | ChatMessage::Tool { content, .. }
        | ChatMessage::Developer { content, .. }
        | ChatMessage::Root { content, .. } => content_contains_audio(content),
        ChatMessage::Assistant { content, .. } => {
            content.as_ref().is_some_and(content_contains_audio)
        }
        ChatMessage::Function { .. } => false,
    });

    if has_audio {
        Err(
            "Harmony does not support audio content parts; use the regular multimodal path"
                .to_string(),
        )
    } else {
        Ok(())
    }
}

fn content_contains_audio(content: &MessageContent) -> bool {
    matches!(
        content,
        MessageContent::Parts(parts)
            if parts
                .iter()
                .any(|part| matches!(part, ContentPart::AudioUrl { .. } | ContentPart::InputAudio { .. }))
    )
}

/// Serializes first-load attempts; a failed load leaves the cell empty so a
/// later attempt (e.g. the next gpt-oss worker registration) can retry.
static HARMONY_ENCODING_INIT: Mutex<()> = Mutex::new(());

/// Get the Harmony encoding, loading it on first use.
///
/// The vocab may come from disk (`TIKTOKEN_ENCODINGS_BASE`) or a cached
/// download; loading is triggered at gpt-oss worker registration
/// (`EnsureHarmonyEncodingStep`), so request paths hit the loaded fast path.
pub(crate) fn try_harmony_encoding() -> Result<&'static HarmonyEncoding, String> {
    if let Some(encoding) = HARMONY_ENCODING.get() {
        return Ok(encoding);
    }
    let _init = HARMONY_ENCODING_INIT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(encoding) = HARMONY_ENCODING.get() {
        return Ok(encoding);
    }
    let encoding = openai_harmony::load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
        .map_err(|e| format!("Failed to load Harmony encoding: {e:#}"))?;
    Ok(HARMONY_ENCODING.get_or_init(|| encoding))
}

/// Convert ProtoOutputLogProbs to OpenAI ChatLogProbs format using Harmony's tokenizer
///
/// Delegates to the shared `convert_proto_logprobs` with Harmony's built-in tokenizer
/// for token ID decoding.
pub(crate) fn convert_harmony_logprobs(
    encoding: &HarmonyEncoding,
    proto_logprobs: &ProtoOutputLogProbs,
) -> ChatLogProbs {
    let tokenizer = encoding.tokenizer();
    utils::convert_proto_logprobs(proto_logprobs, |token_id| {
        tokenizer
            .decode_utf8([token_id])
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    })
}

/// Built-in tools that are advertised in the gpt-oss system message.
///
/// Scoped to the hosted tools gpt-oss was trained to emit directly as
/// channel-tagged tool calls (per the openai-harmony spec):
/// `web_search_preview`, `web_search`, `code_interpreter` / `container`,
/// `file_search`.
///
/// Hosted tools outside this set — notably `image_generation` — were
/// *not* part of gpt-oss training. Advertising them here would render
/// them into the builtin-tools preamble, but the model has never been
/// trained to emit the corresponding `image_generation_call` channel
/// tag, so the result is undefined behavior (hallucinated malformed
/// call, ignored advertisement, or garbled output). Instead, the
/// [`ToolLike`] impl for [`ResponseTool::ImageGeneration`] renders the
/// hosted tool as a *function tool* in the developer-message
/// custom-tool section: gpt-oss sees `image_generation` as a callable
/// function, emits a plain function call with a `prompt` argument, and
/// the downstream MCP dispatch path — keyed on the exposed function-tool
/// name — routes the call to the registered `image_generation` MCP
/// server and materializes the response as an `image_generation_call`
/// output item.
const BUILTIN_TOOLS: &[&str] = &[
    "web_search_preview",
    "web_search",
    "code_interpreter",
    "container",
    "file_search",
    "shell",
];

/// Trait for tool-like objects that can be converted to Harmony ToolDescription
trait ToolLike {
    /// Check if this is a built-in tool (should be skipped in developer message).
    ///
    /// Only exercised by tests today; once per-worker hosted-tool
    /// capability flags land (R0 follow-up), production code will dispatch
    /// on this to gate advertisement per model.
    #[cfg_attr(not(test), expect(dead_code, reason = "reserved for R0 follow-up"))]
    fn is_builtin(&self) -> bool;

    /// Check if this is a custom tool (function or MCP)
    fn is_custom(&self) -> bool;

    /// Convert to ToolDescription
    fn to_tool_description(&self) -> Option<ToolDescription>;
}

/// Implement ToolLike for Chat Completion Tool
impl ToolLike for Tool {
    fn is_builtin(&self) -> bool {
        matches!(
            self.tool_type.as_str(),
            "web_search_preview" | "code_interpreter" | "container"
        )
    }

    fn is_custom(&self) -> bool {
        matches!(self.tool_type.as_str(), "function")
    }

    fn to_tool_description(&self) -> Option<ToolDescription> {
        Some(ToolDescription::new(
            self.function.name.clone(),
            self.function.description.clone().unwrap_or_default(),
            Some(self.function.parameters.clone()),
        ))
    }
}

/// Implement ToolLike for Responses API Tool
impl ToolLike for ResponseTool {
    fn is_builtin(&self) -> bool {
        matches!(
            self,
            ResponseTool::WebSearchPreview(_)
                | ResponseTool::WebSearch(_)
                | ResponseTool::CodeInterpreter(_)
        )
    }

    fn is_custom(&self) -> bool {
        // `ImageGeneration` is rendered as a *function tool* because gpt-oss
        // was not trained to emit `image_generation_call` as a builtin
        // channel tag; instead, the gateway advertises it in the custom-tool
        // section and routes the resulting function call through MCP
        // dispatch. See [`BUILTIN_TOOLS`] above.
        matches!(
            self,
            ResponseTool::Function(_) | ResponseTool::ImageGeneration(_)
        )
    }

    fn to_tool_description(&self) -> Option<ToolDescription> {
        match self {
            ResponseTool::Function(ft) => Some(ToolDescription::new(
                ft.function.name.clone(),
                ft.function.description.clone().unwrap_or_default(),
                Some(ft.function.parameters.clone()),
            )),
            ResponseTool::ImageGeneration(_) => Some(image_generation_tool_description()),
            _ => None,
        }
    }
}

/// Synthesize a function-tool description for the hosted `image_generation`
/// tool when targeting gpt-oss via the harmony pipeline.
///
/// gpt-oss was not trained to emit `image_generation_call` as a native
/// hosted-tool channel tag (see [`BUILTIN_TOOLS`]). This helper produces a
/// JSON-schema that exposes the same tool surface the OpenAI spec documents
/// (`prompt`, plus pass-through configuration fields mirrored from
/// [`openai_protocol::responses::ImageGenerationTool`]) as a plain function
/// tool. gpt-oss then emits `{"name": "image_generation", "arguments": {…}}`
/// on the commentary channel, and the shared MCP dispatch path — keyed on
/// the function name against the registered `image_generation` MCP server —
/// materializes the response into a proper `image_generation_call` output
/// item.
///
/// The schema deliberately stays a superset of what the spec documents on
/// the tool-level configuration so the model can choose to override
/// caller-supplied defaults when the prompt demands it (e.g. switching
/// `size` for portrait vs. landscape output). The caller's original
/// tool-level configuration continues to round-trip in
/// `ResponseTool::ImageGeneration(cfg)` for downstream routing / tests.
fn image_generation_tool_description() -> ToolDescription {
    let parameters = json!({
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Natural-language description of the image to generate. Required."
            },
            "background": {
                "type": "string",
                "enum": ["transparent", "opaque", "auto"],
                "description": "Background handling for the generated image."
            },
            "model": {
                "type": "string",
                "description": "Image-generation model identifier (e.g. gpt-image-1, gpt-image-1-mini, gpt-image-1.5)."
            },
            "output_format": {
                "type": "string",
                "enum": ["png", "webp", "jpeg"],
                "description": "Encoding for the returned image."
            },
            "quality": {
                "type": "string",
                "enum": ["low", "medium", "high", "auto"],
                "description": "Quality tier requested from the image model."
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1024x1536", "1536x1024", "auto"],
                "description": "Output resolution. Use 'auto' to let the model pick."
            }
        },
        "required": ["prompt"],
        "additionalProperties": false
    });

    ToolDescription::new(
        "image_generation",
        "Generate an image from a natural-language prompt. Use this when the user asks for a picture, illustration, or other visual content. The call is routed through the gateway's image_generation MCP server and the result is returned as a base64-encoded image in an image_generation_call output item.",
        Some(parameters),
    )
}

fn has_custom_tools(tool_types: &[&str]) -> bool {
    !tool_types.iter().all(|t| BUILTIN_TOOLS.contains(t))
}

/// Harmony request builder
///
/// Converts OpenAI-format requests into Harmony-encoded format with input_ids,
/// stop tokens, and selection text for worker routing. The encoding itself is
/// fetched per build via [`try_harmony_encoding`] (loaded at gpt-oss worker
/// registration).
pub(crate) struct HarmonyBuilder;

impl HarmonyBuilder {
    /// Create a new Harmony builder
    pub fn new() -> Self {
        Self
    }

    /// Build Harmony request from Chat Completion request
    ///
    /// # Arguments
    ///
    /// * `request` - The ChatCompletionRequest to encode
    ///
    /// # Returns
    ///
    /// HarmonyBuildOutput containing input_ids, stop_token_ids, selection_text, and messages
    pub fn build_from_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<HarmonyBuildOutput, String> {
        reject_chat_audio(&request.messages)?;
        let encoding = try_harmony_encoding()?;

        let mut all_messages = Vec::new();

        let sys_msg = self.build_system_message_from_chat(request);
        all_messages.push(sys_msg);

        let dev_msg = self.build_developer_message_from_chat(request.tools.as_ref());
        all_messages.push(dev_msg);

        let mut user_messages = self.convert_chat_messages(&request.messages);
        all_messages.append(&mut user_messages);

        let conversation = Conversation::from_messages(all_messages.clone());
        let token_ids = encoding
            .render_conversation_for_completion(&conversation, Role::Assistant, None)
            .map_err(|e| format!("Failed to encode Harmony conversation: {e}"))?;

        let selection_text = self.extract_selection_text(&all_messages);

        // Get stop tokens for Harmony assistant actions (<|return|> and <|call|>)
        let stop_token_ids: Vec<u32> = encoding
            .stop_tokens_for_assistant_actions()
            .into_iter()
            .flat_map(|set| set.into_iter())
            .collect();

        Ok(HarmonyBuildOutput {
            input_ids: token_ids,
            stop_token_ids,
            selection_text,
            harmony_messages: all_messages
                .into_iter()
                .map(super::types::HarmonyMessage::from_openai_harmony)
                .collect(),
        })
    }

    /// Build Harmony request from Responses request
    ///
    /// # Arguments
    ///
    /// * `request` - The ResponsesRequest to encode
    ///
    /// # Returns
    ///
    /// HarmonyBuildOutput containing input_ids, stop_token_ids, selection_text, and messages
    pub fn build_from_responses(
        &self,
        request: &ResponsesRequest,
    ) -> Result<HarmonyBuildOutput, String> {
        let encoding = try_harmony_encoding()?;
        let all_messages = self.construct_input_messages_with_harmony(request)?;

        let conversation = Conversation::from_messages(all_messages.clone());
        let token_ids = encoding
            .render_conversation_for_completion(&conversation, Role::Assistant, None)
            .map_err(|e| format!("Failed to encode Harmony conversation: {e}"))?;

        let selection_text = self.extract_selection_text(&all_messages);

        // Get stop tokens for Harmony assistant actions (<|return|> and <|call|>)
        let stop_token_ids: Vec<u32> = encoding
            .stop_tokens_for_assistant_actions()
            .into_iter()
            .flat_map(|set| set.into_iter())
            .collect();

        // Decode tokens to see what the model actually receives
        let decoded_text = encoding
            .tokenizer()
            .decode_utf8(&token_ids)
            .unwrap_or_else(|_| "<decode error>".to_string());
        trace!(
            token_count = token_ids.len(),
            token_preview = ?&token_ids[..token_ids.len().min(20)],
            decoded_length = decoded_text.len(),
            "Encoded conversation to tokens - decoded text follows:"
        );
        trace!("DECODED_TEXT_START\n{}\nDECODED_TEXT_END", decoded_text);

        Ok(HarmonyBuildOutput {
            input_ids: token_ids,
            stop_token_ids,
            selection_text,
            harmony_messages: all_messages
                .into_iter()
                .map(super::types::HarmonyMessage::from_openai_harmony)
                .collect(),
        })
    }

    /// Build system message with common logic
    ///
    /// # Arguments
    /// * `reasoning_effort` - Optional reasoning effort level
    /// * `has_tools` - Whether custom tools are present
    #[expect(
        clippy::unused_self,
        reason = "method on HarmonyBuilder for logical grouping"
    )]
    fn build_system_message(
        &self,
        reasoning_effort: Option<ReasoningEffort>,
        has_tools: bool,
    ) -> HarmonyMessage {
        let mut sys_content = SystemContent::new();

        // Add reasoning_effort if provided
        if let Some(effort) = reasoning_effort {
            sys_content = sys_content.with_reasoning_effort(effort);
        }

        // Set conversation start date (always current date)
        sys_content =
            sys_content.with_conversation_start_date(Local::now().format("%Y-%m-%d").to_string());

        // If no tools, remove "commentary" from valid channels
        if !has_tools {
            if let Some(channel_config) = &sys_content.channel_config {
                let valid_channels: Vec<String> = channel_config
                    .valid_channels
                    .iter()
                    .filter(|c| c.as_str() != "commentary")
                    .cloned()
                    .collect();
                sys_content = sys_content
                    .with_channel_config(ChannelConfig::require_channels(valid_channels));
            }
        }

        HarmonyMessage::from_role_and_content(Role::System, sys_content)
    }

    fn build_system_message_from_chat(&self, request: &ChatCompletionRequest) -> HarmonyMessage {
        let reasoning_effort = request
            .reasoning_effort
            .as_deref()
            .map(|effort| match effort {
                "high" => ReasoningEffort::High,
                "medium" => ReasoningEffort::Medium,
                "low" => ReasoningEffort::Low,
                // Harmony does not support minimal reasoning effort
                "minimal" => ReasoningEffort::Low,
                _ => ReasoningEffort::Medium,
            });

        let has_tools = request.tools.is_some();
        self.build_system_message(reasoning_effort, has_tools)
    }

    /// Build system message from ResponsesRequest
    ///
    /// # Arguments
    /// * `request` - The ResponsesRequest
    /// * `with_custom_tools` - Whether custom tools (beyond built-ins) are present
    fn build_system_message_from_responses(
        &self,
        request: &ResponsesRequest,
        with_custom_tools: bool,
    ) -> HarmonyMessage {
        let reasoning_effort = request
            .reasoning
            .as_ref()
            .and_then(|r| r.effort.as_ref())
            .map(|effort| match effort {
                ResponsesReasoningEffort::High => ReasoningEffort::High,
                ResponsesReasoningEffort::Medium => ReasoningEffort::Medium,
                ResponsesReasoningEffort::Low => ReasoningEffort::Low,
                ResponsesReasoningEffort::Minimal => ReasoningEffort::Low,
            });

        self.build_system_message(reasoning_effort, with_custom_tools)
    }

    /// Build developer message with common logic
    ///
    /// Filters out built-in tools and converts custom tools to ToolDescription
    ///
    /// # Arguments
    /// * `tools` - Optional list of tools
    /// * `instructions` - Optional instructions (Responses API only)
    #[expect(
        clippy::unused_self,
        reason = "method on HarmonyBuilder for logical grouping"
    )]
    fn build_developer_message<T: ToolLike>(
        &self,
        tools: Option<&Vec<T>>,
        instructions: Option<&str>,
    ) -> HarmonyMessage {
        let mut dev_content = DeveloperContent::new();

        // Add instructions if provided (Responses API only)
        if let Some(instructions) = instructions {
            dev_content = dev_content.with_instructions(instructions.to_string());
        }

        // Early return if no tools
        let Some(tools) = tools else {
            return HarmonyMessage::from_role_and_content(Role::Developer, dev_content);
        };

        // Filter to custom tools, convert to ToolDescription, and
        // deduplicate by name.
        //
        // Deduplication matters in the Responses-API path because the
        // upstream MCP loop extends `request.tools` with a
        // `ResponseTool::Function` entry for every tool exposed by a
        // registered MCP server (see `execute_with_mcp_loop` in
        // `harmony/responses/non_streaming.rs`). When the caller's
        // original request also declared a hosted tool that the MCP
        // server resolves by the same name — most notably
        // `image_generation` via the synthesized schema — we would
        // otherwise emit two identically-named entries inside
        // `namespace functions { … }` and confuse gpt-oss about which
        // signature to follow. Keeping the first occurrence (the
        // caller's original / synthesized tool) yields a stable
        // developer-message shape even when the MCP loop injects a
        // same-name function tool on a later iteration.
        let mut seen_names = HashSet::<String>::new();
        let tool_descriptions: Vec<ToolDescription> = tools
            .iter()
            .filter(|t| t.is_custom())
            .filter_map(|t| t.to_tool_description())
            .filter(|td| seen_names.insert(td.name.clone()))
            .collect();

        // Add function tools to developer content
        if !tool_descriptions.is_empty() {
            dev_content = dev_content.with_function_tools(tool_descriptions);
        }

        HarmonyMessage::from_role_and_content(Role::Developer, dev_content)
    }

    fn build_developer_message_from_chat(&self, tools: Option<&Vec<Tool>>) -> HarmonyMessage {
        self.build_developer_message(tools, None)
    }

    /// Build developer message from Responses request
    ///
    /// # Arguments
    /// * `instructions` - Optional instructions (Responses API specific)
    /// * `tools` - Optional list of tools
    fn build_developer_message_from_responses(
        &self,
        instructions: Option<&str>,
        tools: Option<&Vec<ResponseTool>>,
    ) -> HarmonyMessage {
        self.build_developer_message(tools, instructions)
    }

    /// Construct input messages for Responses API with Harmony
    ///
    /// Handles both new conversations and continuations of previous responses.
    ///
    /// This handles:
    /// - New conversation: system message, developer message, and user input
    /// - Continuing conversation: loads previous messages, cleans up chain-of-thoughts
    /// - MCP tool allowlisting for special tool types
    /// - Complex response input parsing with function call tracking
    ///
    /// # Arguments
    /// * `request` - The ResponsesRequest
    /// * `prev_response` - Optional previous response to continue from
    fn construct_input_messages_with_harmony(
        &self,
        request: &ResponsesRequest,
    ) -> Result<Vec<HarmonyMessage>, String> {
        let mut all_messages = Vec::new();

        // Handle new vs continuing conversation
        if request.previous_response_id.is_none() {
            // New conversation

            let tool_types: Vec<&str> = request
                .tools
                .as_ref()
                .map(|tools| tools.iter().map(ResponseTool::as_str).collect())
                .unwrap_or_default();

            let with_custom_tools = has_custom_tools(&tool_types);

            // Add system message
            let sys_msg = self.build_system_message_from_responses(request, with_custom_tools);
            all_messages.push(sys_msg);

            // Add developer message if we have custom tools or instructions
            if with_custom_tools || request.instructions.is_some() {
                let dev_msg = self.build_developer_message_from_responses(
                    request.instructions.as_deref(),
                    request.tools.as_ref(),
                );
                all_messages.push(dev_msg);
            }
        } else {
            // Continue the previous conversation
            // NOTE: Previous messages are loaded by serve_harmony_responses() before calling this method.
            // The request.input will already contain the conversation history when previous_response_id was set.
            // We just proceed with parsing the input items as normal.
            debug!("Continuing conversation (history already loaded in request.input)");
        }

        // Append the new input
        // Responses API supports simple text inputs without chat format
        match &request.input {
            ResponseInput::Text(text) => {
                let user_msg = HarmonyMessage {
                    author: Author {
                        role: Role::User,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text: text.clone() })],
                    channel: None,
                    content_type: None,
                };
                all_messages.push(user_msg);
            }
            ResponseInput::Items(items) => {
                // Track function calls for looking up call_id → name mapping
                let mut prev_outputs: Vec<&ResponseInputOutputItem> = Vec::new();

                for item in items {
                    let msg = self.parse_response_item_to_harmony_message(item, &prev_outputs)?;
                    all_messages.push(msg);

                    // Track function tool calls so that function_call_output can find the name
                    if matches!(item, ResponseInputOutputItem::FunctionToolCall { .. }) {
                        prev_outputs.push(item);
                    }
                }
            }
        }

        debug!(
            message_count = all_messages.len(),
            "Constructed Harmony messages for Responses API"
        );
        Ok(all_messages)
    }

    /// Parse a ResponseInputOutputItem into a HarmonyMessage
    ///
    /// Handles conversion of various response item types (messages, function calls, reasoning, etc.)
    /// to Harmony message format.
    ///
    /// # Arguments
    /// * `item` - The ResponseInputOutputItem to parse
    /// * `prev_outputs` - Previous items for looking up function call names (for function_call_output)
    #[expect(
        clippy::unused_self,
        reason = "method on HarmonyBuilder for logical grouping"
    )]
    fn parse_response_item_to_harmony_message(
        &self,
        item: &ResponseInputOutputItem,
        prev_outputs: &[&ResponseInputOutputItem],
    ) -> Result<HarmonyMessage, String> {
        match item {
            // Regular message (user or assistant)
            ResponseInputOutputItem::Message { role, content, .. } => {
                let harmony_role = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "system" => Role::System,
                    _ => Role::User, // Default to user for unknown roles
                };

                // Extract text from content parts. `Refusal` is losslessly
                // representable as text and is preserved verbatim. Image /
                // file parts are currently dropped (R1/R2/R3 will implement
                // full media handling).
                let text_parts: Vec<String> = content
                    .iter()
                    .filter_map(|part| match part {
                        ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                        ResponseContentPart::InputText { text } => Some(text.clone()),
                        ResponseContentPart::Refusal { refusal } => Some(refusal.clone()),
                        // R1/R2/R3 will implement full media handling
                        ResponseContentPart::InputImage { .. }
                        | ResponseContentPart::InputFile { .. } => None,
                    })
                    .collect();

                let text = text_parts.join("\n");

                Ok(HarmonyMessage {
                    author: Author {
                        role: harmony_role,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text })],
                    channel: None,
                    content_type: None,
                })
            }

            // Reasoning content (chain-of-thought)
            ResponseInputOutputItem::Reasoning { content, .. } => {
                // Extract reasoning text
                let reasoning_texts: Vec<String> = content
                    .iter()
                    .map(|rc| match rc {
                        ResponseReasoningContent::ReasoningText { text } => text.clone(),
                    })
                    .collect();

                let text = reasoning_texts.join("\n");

                // Reasoning goes in the "analysis" channel for Harmony
                Ok(HarmonyMessage {
                    author: Author {
                        role: Role::Assistant,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text })],
                    channel: Some("analysis".to_string()),
                    content_type: None,
                })
            }

            // Function tool call (with optional output)
            ResponseInputOutputItem::FunctionToolCall {
                name,
                arguments,
                output,
                ..
            } => {
                // If there's an output, this represents the tool result
                // Otherwise, it's the tool call itself
                if let Some(output_str) = output {
                    // Tool result - use Tool role with "functions.{name}" as author name
                    // IMPORTANT: Must include recipient="assistant" for parser to recognize it.
                    // We keep channel=None to minimize what the model might copy.
                    let author_name = format!("functions.{name}");
                    debug!(
                        tool_name = %name,
                        author_name = %author_name,
                        output_preview = %output_str.chars().take(100).collect::<String>(),
                        "Building tool result message with Tool role (recipient=assistant, no channel)"
                    );
                    Ok(HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(author_name),
                        },
                        recipient: Some("assistant".to_string()),
                        content: vec![Content::Text(TextContent {
                            text: output_str.clone(),
                        })],
                        channel: None,
                        content_type: None,
                    })
                } else {
                    // Tool call - assistant message in commentary channel with recipient
                    // msg.with_channel("commentary").with_recipient(f"functions.{name}")
                    let recipient = format!("functions.{name}");
                    debug!(
                        tool_name = %name,
                        recipient = %recipient,
                        "Building tool call message with recipient"
                    );
                    Ok(HarmonyMessage {
                        author: Author {
                            role: Role::Assistant,
                            name: None,
                        },
                        recipient: Some(recipient),
                        content: vec![Content::Text(TextContent {
                            text: arguments.clone(),
                        })],
                        channel: Some("commentary".to_string()),
                        content_type: Some("json".to_string()),
                    })
                }
            }

            // Function call output (separate from call) - requires looking up the original call
            ResponseInputOutputItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                // Search prev_outputs in reverse order to find the matching function call
                let call = prev_outputs
                    .iter()
                    .rev()
                    .find_map(|item| match item {
                        ResponseInputOutputItem::FunctionToolCall {
                            call_id: item_call_id,
                            name,
                            ..
                        } if item_call_id == call_id => Some(name.clone()),
                        _ => None,
                    })
                    .ok_or_else(|| format!("No function call found for call_id: {call_id}"))?;

                // Create Tool message with "functions.{name}" prefix
                // IMPORTANT: Must include recipient="assistant" for parser to recognize it.
                // We keep channel=None to minimize what the model might copy.
                Ok(HarmonyMessage {
                    author: Author {
                        role: Role::Tool,
                        name: Some(format!("functions.{call}")),
                    },
                    recipient: Some("assistant".to_string()),
                    content: vec![Content::Text(TextContent {
                        text: output.clone(),
                    })],
                    channel: None,
                    content_type: None,
                })
            }

            // Simple input message (usually user message)
            ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                let harmony_role = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "system" => Role::System,
                    _ => Role::User,
                };

                let text = match content {
                    StringOrContentParts::String(s) => s.clone(),
                    StringOrContentParts::Array(parts) => {
                        // Extract text from content parts. `Refusal` is
                        // losslessly representable as text and is preserved
                        // verbatim. Image / file parts are currently dropped
                        // (R1/R2/R3 will implement full media handling).
                        parts
                            .iter()
                            .filter_map(|part| match part {
                                ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                                ResponseContentPart::InputText { text } => Some(text.clone()),
                                ResponseContentPart::Refusal { refusal } => Some(refusal.clone()),
                                // R1/R2/R3 will implement full media handling
                                ResponseContentPart::InputImage { .. }
                                | ResponseContentPart::InputFile { .. } => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };

                Ok(HarmonyMessage {
                    author: Author {
                        role: harmony_role,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text })],
                    channel: None,
                    content_type: None,
                })
            }

            ResponseInputOutputItem::McpApprovalResponse { .. }
            | ResponseInputOutputItem::McpApprovalRequest { .. }
            | ResponseInputOutputItem::ComputerCall { .. }
            | ResponseInputOutputItem::ComputerCallOutput { .. }
            | ResponseInputOutputItem::McpCall { .. }
            | ResponseInputOutputItem::McpListTools { .. } => {
                warn!(
                    function = "parse_response_item_to_harmony_message",
                    "Approval item reached Harmony conversion"
                );
                Err("Unsupported input item type".to_string())
            }

            ResponseInputOutputItem::ImageGenerationCall { .. } => {
                warn!(
                    function = "parse_response_item_to_harmony_message",
                    "image_generation_call input item reached Harmony conversion"
                );
                Err("Unsupported input item type".to_string())
            }

            ResponseInputOutputItem::Compaction { .. }
            | ResponseInputOutputItem::ItemReference { .. } => {
                Err("Unsupported input item type".to_string())
            }

            ResponseInputOutputItem::CustomToolCall { .. }
            | ResponseInputOutputItem::CustomToolCallOutput { .. } => {
                warn!(
                    function = "parse_response_item_to_harmony_message",
                    "Custom tool item reached Harmony conversion"
                );
                Err("Unsupported input item type".to_string())
            }

            ResponseInputOutputItem::ShellCall { .. }
            | ResponseInputOutputItem::ShellCallOutput { .. } => {
                warn!(
                    function = "parse_response_item_to_harmony_message",
                    "Shell tool item reached Harmony conversion"
                );
                Err("Unsupported input item type".to_string())
            }

            ResponseInputOutputItem::ApplyPatchCall { .. }
            | ResponseInputOutputItem::ApplyPatchCallOutput { .. } => {
                warn!(
                    function = "parse_response_item_to_harmony_message",
                    "apply_patch item reached Harmony conversion"
                );
                Err("Unsupported input item type".to_string())
            }
            // T5 schema-only: forced-cascade arm, no behavior.
            ResponseInputOutputItem::LocalShellCall { .. }
            | ResponseInputOutputItem::LocalShellCallOutput { .. } => {
                Err("Unsupported input item type".to_string())
            }
        }
    }

    /// Convert OpenAI ChatMessage format to Harmony messages
    ///
    /// - Assistant messages with tool_calls create multiple messages (one per tool call)
    /// - Tool role messages use Role::Tool with proper author
    /// - Tool-related messages use channel="commentary"
    #[expect(
        clippy::unused_self,
        reason = "method on HarmonyBuilder for logical grouping"
    )]
    fn convert_chat_messages(&self, messages: &[ChatMessage]) -> Vec<HarmonyMessage> {
        let mut harmony_messages = Vec::new();

        // Build a map of tool_call_id -> function_name for tool responses
        let mut tool_call_map = std::collections::HashMap::new();
        for msg in messages {
            if let ChatMessage::Assistant {
                tool_calls: Some(calls),
                ..
            } = msg
            {
                for call in calls {
                    tool_call_map.insert(call.id.clone(), call.function.name.clone());
                }
            }
        }

        for msg in messages {
            match msg {
                ChatMessage::System { content, name, .. } | ChatMessage::Root { content, name } => {
                    // System messages stay as-is
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::System,
                            name: name.clone(),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent {
                            text: content.to_simple_string(),
                        })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }
                ChatMessage::Developer {
                    content,
                    name,
                    tools: _,
                } => {
                    // Developer messages stay as-is
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::Developer,
                            name: name.clone(),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent {
                            text: content.to_simple_string(),
                        })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }

                ChatMessage::User { content, name, .. } => {
                    // Extract text from user content
                    let text = match content {
                        MessageContent::Text(text) => text.clone(),
                        MessageContent::Parts(parts) => {
                            // For multimodal content, extract text parts
                            parts
                                .iter()
                                .filter_map(|part| {
                                    if let ContentPart::Text { text } = part {
                                        Some(text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    };

                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::User,
                            name: name.clone(),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent { text })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }

                ChatMessage::Assistant {
                    content,
                    name,
                    tool_calls,
                    reasoning_content,
                    ..
                } => {
                    if let Some(calls) = tool_calls.as_ref().filter(|c| !c.is_empty()) {
                        // Per Harmony spec: when tool calls are present, include
                        // previous reasoning as a separate analysis channel message
                        // because the model calls tools as part of its chain-of-thought.
                        if let Some(reasoning) = reasoning_content {
                            if !reasoning.is_empty() {
                                let analysis_msg = HarmonyMessage {
                                    author: Author {
                                        role: Role::Assistant,
                                        name: name.clone(),
                                    },
                                    recipient: None,
                                    content: vec![Content::Text(TextContent {
                                        text: reasoning.clone(),
                                    })],
                                    channel: Some("analysis".to_string()),
                                    content_type: None,
                                };
                                harmony_messages.push(analysis_msg);
                            }
                        }

                        // Tool calls go to commentary channel
                        for call in calls {
                            let function_name = &call.function.name;
                            let arguments = call.function.arguments.clone().unwrap_or_default();

                            let tool_call_msg = HarmonyMessage {
                                author: Author {
                                    role: Role::Assistant,
                                    name: name.clone(),
                                },
                                recipient: Some(format!("functions.{function_name}")),
                                content: vec![Content::Text(TextContent { text: arguments })],
                                channel: Some("commentary".to_string()),
                                content_type: Some("json".to_string()),
                            };
                            harmony_messages.push(tool_call_msg);
                        }
                    } else {
                        // Per Harmony spec: drop previous CoT/analysis content on
                        // subsequent turns when the response ended with a final
                        // channel message. Only emit the final content.
                        let text = content
                            .as_ref()
                            .map(|c| c.to_simple_string())
                            .unwrap_or_default();

                        let harmony_msg = HarmonyMessage {
                            author: Author {
                                role: Role::Assistant,
                                name: name.clone(),
                            },
                            recipient: None,
                            content: vec![Content::Text(TextContent { text })],
                            channel: Some("final".to_string()),
                            content_type: None,
                        };
                        harmony_messages.push(harmony_msg);
                    }
                }

                ChatMessage::Tool {
                    content,
                    tool_call_id,
                } => {
                    // Look up the function name from the tool_call_id
                    let function_name = tool_call_map
                        .get(tool_call_id)
                        .cloned()
                        .unwrap_or_else(|| tool_call_id.clone());

                    // Tool result - Must include recipient="assistant" for parser to recognize it.
                    // We keep channel=None to minimize what the model might copy.
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(format!("functions.{function_name}")),
                        },
                        recipient: Some("assistant".to_string()),
                        content: vec![Content::Text(TextContent {
                            text: content.to_simple_string(),
                        })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }

                ChatMessage::Function { content, name } => {
                    // Function messages also use Role::Tool
                    // Tool result - Must include recipient="assistant" for parser to recognize it.
                    // We keep channel=None to minimize what the model might copy.
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(format!("functions.{name}")),
                        },
                        recipient: Some("assistant".to_string()),
                        content: vec![Content::Text(TextContent {
                            text: content.clone(),
                        })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }
            }
        }

        harmony_messages
    }

    /// Extract selection text for worker routing
    ///
    /// Uses the last user message for load balancing
    #[expect(
        clippy::unused_self,
        reason = "method on HarmonyBuilder for logical grouping"
    )]
    fn extract_selection_text(&self, messages: &[HarmonyMessage]) -> String {
        // Find the last user message
        if let Some(last_user_msg) = messages.iter().rev().find(|m| m.author.role == Role::User) {
            // Extract full text from content
            return last_user_msg
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::Text(tc) => Some(tc.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
        }

        // Fallback: concatenate all text
        messages
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::Text(tc) => Some(tc.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for HarmonyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    //! Regression coverage for the `image_generation` → function-tool
    //! translation gpt-oss needs via the harmony pipeline.
    //!
    //! Classifying `ResponseTool::ImageGeneration` as a builtin made the
    //! harmony builder silently drop it — gpt-oss was never trained to
    //! emit `image_generation_call` as a builtin channel tag, so the
    //! model saw no tool at all, emitted a reasoning + message pair, and
    //! the registered `image_generation` MCP server received zero
    //! dispatches.
    //!
    //! These tests lock in the contract:
    //!   1. `image_generation` is NOT in `BUILTIN_TOOLS`.
    //!   2. `ResponseTool::ImageGeneration` is treated as a *custom* tool
    //!      by the harmony `ToolLike` impl.
    //!   3. `to_tool_description()` synthesizes a JSON-schema whose
    //!      `required` set includes `prompt`.
    //!   4. The encoded harmony conversation contains the function-tool
    //!      signature (`type image_generation = (_: …) => any;`) that
    //!      gpt-oss is trained to emit calls against.

    use openai_protocol::{
        common::{AudioUrl, InputAudio},
        responses::{ImageGenerationTool, ResponseInput, ResponseTool, ResponsesRequest},
    };

    use super::*;

    #[test]
    fn chat_audio_is_explicitly_rejected() {
        let audio_parts = [
            ContentPart::AudioUrl {
                audio_url: AudioUrl {
                    url: "https://example.com/audio.wav".to_string(),
                },
            },
            ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "UklGRg==".to_string(),
                    format: "wav".to_string(),
                },
            },
        ];

        for part in audio_parts {
            let messages = vec![ChatMessage::User {
                ext: Default::default(),
                content: MessageContent::Parts(vec![part]),
                name: None,
            }];
            let error = reject_chat_audio(&messages).expect_err("Harmony must reject chat audio");
            assert!(error.contains("audio content parts"));
            assert!(error.contains("regular multimodal path"));
        }
    }

    /// Invariant: `image_generation` must never be advertised as a
    /// gpt-oss native builtin tool. If a future change re-adds it,
    /// gpt-oss's behavior becomes undefined (hallucinated tool call
    /// shape, ignored advertisement, or garbled output) because the
    /// model was not trained on that channel tag. Keep this guard until
    /// per-worker hosted-tool capability flags exist.
    #[test]
    fn image_generation_is_not_a_builtin_tool() {
        assert!(
            !BUILTIN_TOOLS.contains(&"image_generation"),
            "image_generation must not be advertised as a gpt-oss builtin; \
             it is rendered as a function tool instead",
        );
    }

    /// The [`ToolLike`] custom-tool classifier drives two pieces of the
    /// harmony prompt assembly: whether to include the `commentary`
    /// channel in the system message and whether the developer message
    /// is emitted with a tools section. Both are required for gpt-oss
    /// to emit a function call, so `ImageGeneration` must report as
    /// custom.
    #[test]
    fn response_tool_image_generation_is_custom() {
        let tool = ResponseTool::ImageGeneration(ImageGenerationTool::default());
        assert!(
            tool.is_custom(),
            "ImageGeneration must be is_custom() == true so the harmony \
             pipeline renders it in the developer-message tools section",
        );
        assert!(
            !tool.is_builtin(),
            "ImageGeneration must not be classified as a gpt-oss native builtin",
        );
    }

    /// Exercise the synthesized JSON-schema the harmony builder hands to
    /// gpt-oss. `prompt` is the only required field (the rest mirror
    /// [`ImageGenerationTool`] caller-side configuration knobs). If the
    /// schema shape drifts, the model either stops emitting valid
    /// `image_generation` calls or regresses on caller-driven overrides.
    #[test]
    fn image_generation_tool_description_exposes_required_prompt() {
        let tool = ResponseTool::ImageGeneration(ImageGenerationTool::default());
        let description = tool
            .to_tool_description()
            .expect("ImageGeneration must produce a synthesized ToolDescription");

        assert_eq!(
            description.name, "image_generation",
            "function-tool name must match what the MCP session exposes so \
             dispatch routes correctly",
        );
        assert!(
            !description.description.is_empty(),
            "description should be non-empty so gpt-oss understands when to call",
        );

        let parameters = description
            .parameters
            .as_ref()
            .expect("parameters JSON-schema must be present");
        assert_eq!(parameters["type"], "object", "schema must be an object");

        let required = parameters["required"]
            .as_array()
            .expect("`required` array must be present");
        assert!(
            required.iter().any(|v| v.as_str() == Some("prompt")),
            "`prompt` must be a required parameter, got: {required:?}",
        );

        let properties = parameters["properties"]
            .as_object()
            .expect("`properties` object must be present");
        for expected in ["prompt", "size", "quality", "background", "output_format"] {
            assert!(
                properties.contains_key(expected),
                "parameters.properties must include `{expected}`; got: {:?}",
                properties.keys().collect::<Vec<_>>(),
            );
        }

        // `prompt` must be typed as a string so gpt-oss renders a
        // free-form text slot rather than a structured object.
        assert_eq!(
            properties["prompt"]["type"], "string",
            "`prompt` must be a string parameter",
        );
    }

    /// End-to-end assertion on the encoded harmony prompt: for a request
    /// that declares only an `image_generation` tool, the rendered
    /// conversation MUST contain the function-tool signature gpt-oss
    /// looks for on the commentary channel. This catches regressions
    /// where the tool is silently dropped from the developer message.
    #[test]
    fn build_from_responses_renders_image_generation_as_function_tool() {
        let builder = HarmonyBuilder::new();
        let request = ResponsesRequest {
            model: "gpt-oss-120b".to_string(),
            input: ResponseInput::Text("draw a cat".to_string()),
            tools: Some(vec![ResponseTool::ImageGeneration(
                ImageGenerationTool::default(),
            )]),
            ..Default::default()
        };

        let output = builder
            .build_from_responses(&request)
            .expect("harmony build must succeed");

        let decoded = try_harmony_encoding()
            .expect("harmony encoding must load")
            .tokenizer()
            .decode_utf8(&output.input_ids)
            .expect("decode harmony tokens back to UTF-8");

        assert!(
            decoded.contains("type image_generation = ("),
            "harmony prompt must advertise image_generation as a function \
             tool to gpt-oss; decoded prompt: {decoded}",
        );
        assert!(
            decoded.contains("namespace functions"),
            "function-tool section (namespace functions) must be present; \
             decoded prompt: {decoded}",
        );
        assert!(
            decoded.contains("prompt"),
            "synthesized schema must expose `prompt`; decoded prompt: {decoded}",
        );
    }

    /// Guard against future regressions where callers attach an
    /// `image_generation` configuration but `has_custom_tools()` still
    /// returns false (which would skip the developer-message tools
    /// section entirely and leave gpt-oss unable to emit the call).
    #[test]
    fn has_custom_tools_true_for_image_generation_only_request() {
        let tool_types = ["image_generation"];
        assert!(
            has_custom_tools(&tool_types),
            "a request whose only tool is image_generation must still \
             trigger the custom-tool path in the harmony prompt",
        );
    }

    /// Simulate the MCP-loop side effect where the router appends a
    /// `ResponseTool::Function` copy of each MCP-exposed tool onto the
    /// request (see `execute_with_mcp_loop`). For
    /// `image_generation` the MCP server's tool name collides with the
    /// synthesized function-tool name, so the harmony developer
    /// message would otherwise carry two identically-named entries
    /// inside `namespace functions { … }` and confuse gpt-oss. The
    /// builder deduplicates by name, keeping the caller's
    /// original/synthesized entry.
    #[test]
    fn dedupes_duplicate_function_tool_names_from_mcp_loop() {
        use openai_protocol::{common::Function, responses::FunctionTool};

        let builder = HarmonyBuilder::new();
        let request = ResponsesRequest {
            model: "gpt-oss-120b".to_string(),
            input: ResponseInput::Text("draw a cat".to_string()),
            tools: Some(vec![
                ResponseTool::ImageGeneration(ImageGenerationTool::default()),
                // What `convert_mcp_tools_to_response_tools` would
                // append after the MCP session exposes
                // `image_generation` — a function tool with the same
                // name but a schema reflecting the MCP server's side.
                ResponseTool::Function(FunctionTool {
                    function: Function {
                        name: "image_generation".to_string(),
                        description: Some("mcp-exposed duplicate".to_string()),
                        parameters: json!({
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"}
                            },
                            "required": ["prompt"]
                        }),
                        strict: None,
                    },
                }),
            ]),
            ..Default::default()
        };

        let output = builder
            .build_from_responses(&request)
            .expect("harmony build must succeed");

        let decoded = try_harmony_encoding()
            .expect("harmony encoding must load")
            .tokenizer()
            .decode_utf8(&output.input_ids)
            .expect("decode harmony tokens back to UTF-8");

        let occurrences = decoded.matches("type image_generation = (").count();
        assert_eq!(
            occurrences, 1,
            "image_generation must appear exactly once in the rendered \
             function-tools namespace; found {occurrences} occurrences. \
             Decoded prompt:\n{decoded}",
        );
    }
}
