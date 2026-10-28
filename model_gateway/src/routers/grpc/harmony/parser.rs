//! Harmony response parser
//!
//! Adapter for openai_harmony::StreamableParser that handles channel-based parsing.

use openai_harmony::{chat::Role, HarmonyEncoding, StreamableParser};
use openai_protocol::common::{FunctionCallResponse, ToolCall};
use uuid::Uuid;

use super::types::{HarmonyChannelDelta, HarmonyChannelOutput};

/// Get the global Harmony encoding
///
/// References the same encoding used by the builder for consistency; loaded
/// at gpt-oss worker registration, so parser construction hits the fast path.
fn try_harmony_encoding() -> Result<&'static HarmonyEncoding, String> {
    super::builder::try_harmony_encoding()
}

/// Harmony parser adapter
///
/// Wraps openai_harmony::StreamableParser and provides methods for parsing
/// complete responses and streaming chunks.
pub(crate) struct HarmonyParserAdapter {
    parser: StreamableParser,
    prev_recipient: Option<String>,
    reasoning_token_count: u32,
}

impl HarmonyParserAdapter {
    /// Create a new Harmony parser
    pub fn new() -> Result<Self, String> {
        let encoding = try_harmony_encoding()?;
        let parser = StreamableParser::new(encoding.clone(), Some(Role::Assistant))
            .map_err(|e| format!("Failed to create StreamableParser: {e}"))?;

        Ok(Self {
            parser,
            prev_recipient: None,
            reasoning_token_count: 0,
        })
    }

    /// Extract text from message content (private helper)
    ///
    /// Filters text content from a message's content array and joins them into a single string.
    ///
    /// # Arguments
    ///
    /// * `content` - The content array from a Harmony message
    ///
    /// # Returns
    ///
    /// Joined text string from all text content items
    fn extract_text_from_content(content: &[openai_harmony::chat::Content]) -> String {
        content
            .iter()
            .filter_map(|c| match c {
                openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Handle incomplete content from parser state (private helper)
    ///
    /// Checks for any remaining incomplete content in the parser and appends it
    /// to the appropriate channel (analysis or final_text).
    ///
    /// # Arguments
    ///
    /// * `parser` - Reference to the StreamableParser
    /// * `analysis` - Mutable reference to analysis content
    /// * `final_text` - Mutable reference to final text content
    fn handle_incomplete_content(
        parser: &StreamableParser,
        analysis: &mut Option<String>,
        final_text: &mut String,
    ) {
        if let Ok(current_content) = parser.current_content() {
            if !current_content.is_empty() {
                let current_channel = parser.current_channel();
                match current_channel.as_deref() {
                    Some("analysis") => {
                        *analysis = Some(current_content);
                    }
                    Some("final") | None => {
                        final_text.push_str(&current_content);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Parse messages into channel outputs (private helper)
    ///
    /// Extracts analysis, commentary (tool calls), and final text from Harmony messages.
    /// This is the core parsing logic shared by both parse_complete and finalize.
    ///
    /// # Arguments
    ///
    /// * `messages` - The messages to parse from the Harmony parser
    ///
    /// # Returns
    ///
    /// Tuple of (analysis, commentary, final_text)
    pub fn parse_messages(
        messages: &[openai_harmony::chat::Message],
    ) -> (Option<String>, Option<Vec<ToolCall>>, String) {
        let mut analysis: Option<String> = None;
        let mut commentary: Option<Vec<ToolCall>> = None;
        let mut final_text = String::new();

        for msg in messages {
            // Filter: Only process assistant messages
            if msg.author.role != Role::Assistant {
                continue;
            }

            let channel = msg.channel.as_deref().unwrap_or("");
            let recipient = msg.recipient.as_deref();

            // IMPORTANT: Check recipient FIRST before channel
            // The model sometimes generates tool calls with channel="analysis" + recipient="functions.*"
            // instead of channel="commentary" + recipient="functions.*"
            // We should trust the recipient field to determine if this is a tool call
            if let Some(recipient_str) = recipient {
                if let Some(function_name) = recipient_str.strip_prefix("functions.") {
                    // This is a tool call, regardless of channel

                    // Process each content item separately
                    for content in &msg.content {
                        if let openai_harmony::chat::Content::Text(tc) = content {
                            let call_id = format!("call_{}", Uuid::now_v7());
                            let tool_call = ToolCall {
                                id: call_id,
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: function_name.to_string(),
                                    arguments: Some(tc.text.clone()),
                                },
                            };

                            match commentary.as_mut() {
                                Some(calls) => calls.push(tool_call),
                                None => commentary = Some(vec![tool_call]),
                            }
                        }
                    }
                    // Skip further channel processing for this message
                    continue;
                } else if recipient_str.starts_with("python")
                    || recipient_str.starts_with("browser")
                    || recipient_str.starts_with("container")
                {
                    // Built-in tools → treat as reasoning
                    // For Chat API, we add to analysis content
                    let text = Self::extract_text_from_content(&msg.content);

                    if !text.is_empty() {
                        // Append to analysis (built-in tools are reasoning)
                        match analysis.as_mut() {
                            Some(existing) => {
                                existing.push('\n');
                                existing.push_str(&text);
                            }
                            None => analysis = Some(text),
                        }
                    }
                    // Skip further channel processing
                    continue;
                }
            }

            // Now process by channel (only if not already handled by recipient)
            match channel {
                "analysis" => {
                    // Process each content item
                    // For Chat API, we join them into a single reasoning_content
                    let text = Self::extract_text_from_content(&msg.content);

                    if !text.is_empty() {
                        analysis = Some(text);
                    }
                }
                "commentary" => {
                    // If we reach here, recipient was not "functions.*" or built-in tools
                    // Commentary channel should always have a recipient
                    // This is likely a model bug - log warning and treat as reasoning
                    tracing::warn!(
                        channel = "commentary",
                        recipient = ?recipient,
                        "Commentary message without valid recipient, treating as reasoning"
                    );

                    let text = Self::extract_text_from_content(&msg.content);

                    if !text.is_empty() {
                        match analysis.as_mut() {
                            Some(existing) => {
                                existing.push('\n');
                                existing.push_str(&text);
                            }
                            None => analysis = Some(text),
                        }
                    }
                }
                "final" => {
                    // Process final channel content
                    let text = Self::extract_text_from_content(&msg.content);
                    final_text.push_str(&text);
                }
                _ => {
                    // Unknown channel, append to final text as fallback
                    let text = Self::extract_text_from_content(&msg.content);
                    final_text.push_str(&text);
                }
            }
        }

        (analysis, commentary, final_text)
    }

    /// Parse complete response
    ///
    /// Parses all output token IDs and returns the complete channel output
    /// containing analysis, commentary (tool calls), and final text.
    pub fn parse_complete(
        &mut self,
        output_ids: &[u32],
        finish_reason: String,
    ) -> Result<HarmonyChannelOutput, String> {
        let mut reasoning_token_count = 0u32;

        // Feed all tokens to the parser
        for &token_id in output_ids {
            self.parser
                .process(token_id)
                .map_err(|e| format!("Failed to process token {token_id}: {e}"))?;

            // Count reasoning tokens (analysis + commentary channels)
            if let Some(channel) = self.parser.current_channel() {
                if channel == "analysis" || channel == "commentary" {
                    reasoning_token_count += 1;
                }
            }
        }

        // Extract all completed messages from the parser
        let messages = self.parser.messages();

        // Parse messages into channel outputs using shared helper
        let (mut analysis, commentary, mut final_text) = Self::parse_messages(messages);

        // Check for incomplete content in parser state
        Self::handle_incomplete_content(&self.parser, &mut analysis, &mut final_text);

        // Determine finish reason: override to "tool_calls" if commentary has tool calls
        let final_finish_reason = if commentary.is_some() {
            "tool_calls".to_string()
        } else {
            finish_reason.clone()
        };

        Ok(HarmonyChannelOutput {
            analysis,
            commentary,
            final_text,
            finish_reason: final_finish_reason,
            reasoning_token_count,
        })
    }

    /// Get all messages from the parser
    ///
    /// Returns the raw messages extracted by the Harmony parser.
    /// Used for validation checks.
    pub fn get_messages(&self) -> Vec<openai_harmony::chat::Message> {
        self.parser.messages().to_vec()
    }

    /// Extract incomplete commentary content from parser state
    ///
    /// When the stream ends, there may be incomplete commentary content in the parser
    /// that hasn't been finalized into a completed message. This method extracts
    /// such content and converts it to tool calls.
    ///
    /// # Returns
    ///
    /// Optional vector of ToolCall if incomplete commentary is found
    pub fn extract_incomplete_commentary(&self) -> Option<Vec<ToolCall>> {
        // Check if current channel is commentary
        let current_channel = self.parser.current_channel();
        if current_channel.as_deref() != Some("commentary") {
            return None;
        }

        // Get current recipient (should be "functions.{name}")
        let recipient = self.parser.current_recipient()?;
        if !recipient.starts_with("functions.") {
            return None;
        }

        // Get current incomplete content
        let content = self.parser.current_content().ok()?;
        if content.is_empty() {
            return None;
        }

        // Extract function name from recipient
        let Some(function_name) = recipient.strip_prefix("functions.") else {
            // guarded by starts_with("functions.") check above
            return None;
        };

        // Create tool call from incomplete content
        let call_id = format!("call_{}", Uuid::now_v7());
        let tool_call = ToolCall {
            id: call_id,
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: function_name.to_string(),
                arguments: Some(content),
            },
        };

        Some(vec![tool_call])
    }

    /// Parse streaming chunk
    ///
    /// Parses incremental token IDs and returns a delta with any new content
    /// from the analysis, commentary, or final channels.
    ///
    /// # Arguments
    ///
    /// * `chunk_ids` - New token IDs from the current chunk
    ///
    /// # Returns
    ///
    /// Optional HarmonyChannelDelta if there's new content to emit
    pub fn parse_chunk(
        &mut self,
        chunk_ids: &[u32],
    ) -> Result<Option<HarmonyChannelDelta>, String> {
        let mut analysis_delta: Option<String> = None;
        let mut commentary_delta = None;
        let mut final_delta: Option<String> = None;

        // Accumulate delta text for commentary channel
        let mut accumulated_delta = String::new();

        // Process each token
        for &token_id in chunk_ids {
            self.parser
                .process(token_id)
                .map_err(|e| format!("Failed to process token {token_id}: {e}"))?;

            // Count reasoning tokens (analysis + commentary channels)
            if let Some(channel) = self.parser.current_channel() {
                if channel == "analysis" || channel == "commentary" {
                    self.reasoning_token_count += 1;
                }
            }

            // Check for content delta
            if let Ok(Some(delta_text)) = self.parser.last_content_delta() {
                // Determine which channel this delta belongs to
                let channel = self.parser.current_channel();
                match channel.as_deref() {
                    Some("analysis") => {
                        analysis_delta
                            .get_or_insert_with(String::new)
                            .push_str(&delta_text);
                    }
                    Some("final") | None => {
                        final_delta
                            .get_or_insert_with(String::new)
                            .push_str(&delta_text);
                    }
                    Some("commentary") => {
                        // Accumulate delta for commentary
                        accumulated_delta.push_str(&delta_text);
                    }
                    _ => {}
                }
            }
        }

        // Handle commentary channel tool call deltas
        if self.parser.current_channel().as_deref() == Some("commentary") {
            if let Some(cur_recipient) = self.parser.current_recipient() {
                if let Some(tool_name) = cur_recipient.strip_prefix("functions.") {
                    // Count completed tool calls for index
                    let base_index = self
                        .parser
                        .messages()
                        .iter()
                        .filter(|msg| {
                            msg.channel.as_deref() == Some("commentary")
                                && msg
                                    .recipient
                                    .as_deref()
                                    .is_some_and(|r| r.starts_with("functions."))
                        })
                        .count();

                    // Check if recipient changed (new tool call)
                    let recipient_changed = self.prev_recipient.as_deref() != Some(&cur_recipient);

                    if recipient_changed {
                        // NEW tool call: emit name + id
                        let call_id = format!("call_{}", Uuid::now_v7());

                        commentary_delta = Some(super::types::ToolCallDelta {
                            index: base_index,
                            id: Some(call_id),
                            function: Some(super::types::FunctionDelta {
                                name: Some(tool_name.to_string()),
                                arguments: Some(String::new()),
                            }),
                        });

                        // Update prev_recipient
                        self.prev_recipient = Some(cur_recipient);
                    } else if !accumulated_delta.is_empty() {
                        // CONTINUING tool call: emit arguments delta
                        commentary_delta = Some(super::types::ToolCallDelta {
                            index: base_index,
                            id: None,
                            function: Some(super::types::FunctionDelta {
                                name: None,
                                arguments: Some(accumulated_delta),
                            }),
                        });
                    }
                }
            }
        }

        // Only emit a delta if at least one channel produced output
        let has_delta =
            analysis_delta.is_some() || commentary_delta.is_some() || final_delta.is_some();

        if has_delta {
            Ok(Some(HarmonyChannelDelta {
                analysis_delta,
                commentary_delta,
                final_delta,
            }))
        } else {
            Ok(None)
        }
    }

    /// Finalize parsing
    ///
    /// Called at the end of streaming to get the final state and any
    /// remaining content.
    ///
    /// # Arguments
    ///
    /// * `finish_reason` - The finish reason from GenerateComplete ("stop", "length", etc.)
    /// # Returns
    ///
    /// Final HarmonyChannelOutput with complete parsed content
    pub fn finalize(&mut self, finish_reason: String) -> HarmonyChannelOutput {
        // Extract all completed messages
        let messages = self.parser.messages();

        // Parse messages into channel outputs using shared helper
        let (mut analysis, commentary, mut final_text) = Self::parse_messages(messages);

        // Check for remaining incomplete content
        Self::handle_incomplete_content(&self.parser, &mut analysis, &mut final_text);

        // Determine finish reason: override to "tool_calls" if commentary has tool calls
        let final_finish_reason = if commentary.is_some() {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        HarmonyChannelOutput {
            analysis,
            commentary,
            final_text,
            finish_reason: final_finish_reason,
            reasoning_token_count: self.reasoning_token_count,
        }
    }
}
