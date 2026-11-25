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

fn call_id() -> String {
    format!("call_{}", Uuid::now_v7())
}

fn function_call(name: &str, arguments: String) -> ToolCall {
    ToolCall {
        id: call_id(),
        tool_type: "function".to_string(),
        function: FunctionCallResponse {
            name: name.to_string(),
            arguments: Some(arguments),
        },
    }
}

fn append_reasoning(analysis: &mut Option<String>, text: String) {
    if text.is_empty() {
        return;
    }
    match analysis.as_mut() {
        Some(existing) => {
            existing.push('\n');
            existing.push_str(&text);
        }
        None => *analysis = Some(text),
    }
}

/// Harmony parser adapter
///
/// Wraps openai_harmony::StreamableParser and provides methods for parsing
/// complete responses and streaming chunks.
pub(crate) struct HarmonyParserAdapter {
    parser: StreamableParser,
    action_stop_ids: Vec<u32>,
    prev_recipient: Option<String>,
    reasoning_token_count: u32,
}

impl HarmonyParserAdapter {
    /// Create a new Harmony parser
    pub fn new(require_action: bool) -> Result<Self, String> {
        let encoding = try_harmony_encoding()?;
        let parser = StreamableParser::new(encoding.clone(), Some(Role::Assistant))
            .map_err(|e| format!("Failed to create StreamableParser: {e}"))?;

        let action_stop_ids = if require_action {
            encoding
                .stop_tokens_for_assistant_actions()
                .map_err(|e| format!("Failed to load Harmony action stops: {e}"))?
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            parser,
            action_stop_ids,
            prev_recipient: None,
            reasoning_token_count: 0,
        })
    }

    fn process_token(&mut self, token_id: u32) -> Result<(), String> {
        self.parser
            .process(token_id)
            .map_err(|e| format!("Failed to process token {token_id}: {e}"))?;

        // String-based grammars also accept ordinary BPE tokens spelling
        // <|message|>. Only the special token closes a Harmony header. The
        // strict parser reports an unfinished header on EOS, so check at a
        // real action stop; a length limit or a user stop can cut any header.
        if self.action_stop_ids.contains(&token_id) {
            self.parser
                .process_eos()
                .map_err(|e| format!("Invalid Harmony message at action stop: {e}"))?;
            // A byte-spelled transition after real analysis is ordinary message
            // text to Harmony. The generated grammar still requires a real
            // assistant output before the action stop.
            let valid_output = self.parser.messages().last().is_some_and(|message| {
                message.author.role == Role::Assistant
                    && (message.channel.as_deref() == Some("final")
                        || message
                            .recipient
                            .as_deref()
                            .is_some_and(|r| r.starts_with("functions.")))
            });
            if !valid_output {
                return Err(
                    "Constrained Harmony turn ended without a final message or function call"
                        .into(),
                );
            }
        }
        Ok(())
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
                            commentary
                                .get_or_insert_with(Vec::new)
                                .push(function_call(function_name, tc.text.clone()));
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
                    append_reasoning(&mut analysis, text);
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
                    append_reasoning(&mut analysis, text);
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
            self.process_token(token_id)?;

            if self.in_reasoning_channel() {
                reasoning_token_count += 1;
            }
        }
        Ok(self.output(finish_reason, reasoning_token_count))
    }

    fn in_reasoning_channel(&self) -> bool {
        matches!(
            self.parser.current_channel().as_deref(),
            Some("analysis" | "commentary")
        )
    }

    fn output(&self, finish_reason: String, reasoning_token_count: u32) -> HarmonyChannelOutput {
        let messages = self.parser.messages();
        let (mut analysis, commentary, mut final_text) = Self::parse_messages(messages);
        Self::handle_incomplete_content(&self.parser, &mut analysis, &mut final_text);
        let finish_reason = if commentary.is_some() {
            "tool_calls".to_string()
        } else {
            finish_reason
        };
        HarmonyChannelOutput {
            analysis,
            commentary,
            final_text,
            finish_reason,
            reasoning_token_count,
        }
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

        // Extract function name from recipient
        let Some(function_name) = recipient.strip_prefix("functions.") else {
            // guarded by starts_with("functions.") check above
            return None;
        };

        Some(vec![function_call(function_name, content)])
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
            self.process_token(token_id)?;

            if self.in_reasoning_channel() {
                self.reasoning_token_count += 1;
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
                        commentary_delta = Some(super::types::ToolCallDelta {
                            index: base_index,
                            id: Some(call_id()),
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
        self.output(finish_reason, self.reasoning_token_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_spelled_headers_fail_at_the_action_stop_collected_and_streamed() {
        let tokenizer = try_harmony_encoding()
            .expect("Harmony encoding")
            .tokenizer();
        for (header, stop) in [
            ("<|channel|>final<|message|>", "<|return|>"),
            (
                "<|channel|>commentary to=functions.lookup<|message|>",
                "<|call|>",
            ),
        ] {
            let prefix = tokenizer.encode_ordinary(&format!("{header}{{}}"));
            let stop_ids = tokenizer.encode_with_special_tokens(stop);
            let turn = [prefix.as_slice(), &stop_ids].concat();

            let mut collected = HarmonyParserAdapter::new(true).expect("Harmony parser");
            let error = collected
                .parse_complete(&turn, "stop".to_string())
                .expect_err("a natural stop cannot complete a byte-spelled header");
            assert!(error.contains("message header"), "{error}");

            let mut streamed = HarmonyParserAdapter::new(true).expect("Harmony parser");
            for token in &prefix {
                assert!(streamed
                    .parse_chunk(std::slice::from_ref(token))
                    .expect("the header is still open")
                    .is_none());
            }
            assert!(streamed
                .parse_chunk(&stop_ids)
                .expect_err("the terminal chunk must report the malformed header")
                .contains("message header"));

            for reason in ["length", "stop"] {
                let mut cut = HarmonyParserAdapter::new(true).expect("Harmony parser");
                assert!(
                    cut.parse_complete(&prefix, reason.to_string()).is_ok(),
                    "a length limit or explicit stop can cut off the header"
                );
            }
        }
    }

    #[test]
    fn constrained_analysis_must_be_followed_by_a_real_output() {
        let tokenizer = try_harmony_encoding()
            .expect("Harmony encoding")
            .tokenizer();
        let analysis = tokenizer.encode_with_special_tokens("<|channel|>analysis<|message|>Think.");
        let stop = tokenizer.encode_with_special_tokens("<|return|>");
        for header in ["final", "commentary to=functions.lookup"] {
            let transition = format!("<|end|><|start|>assistant<|channel|>{header}<|message|>{{}}");
            for real_markers in [false, true] {
                let mut prefix = analysis.clone();
                prefix.extend(if real_markers {
                    tokenizer.encode_with_special_tokens(&transition)
                } else {
                    tokenizer.encode_ordinary(&transition)
                });
                let turn = [prefix.as_slice(), &stop].concat();
                let mut collected = HarmonyParserAdapter::new(true).unwrap();
                let collected = collected.parse_complete(&turn, "stop".into());
                let mut streamed = HarmonyParserAdapter::new(true).unwrap();
                for chunk in prefix.chunks(2) {
                    streamed.parse_chunk(chunk).unwrap();
                }
                let terminal = streamed.parse_chunk(&stop);
                if real_markers {
                    let collected = collected.unwrap();
                    terminal.unwrap();
                    let streamed = streamed.finalize("stop".into());
                    for result in [collected, streamed] {
                        assert_eq!(result.analysis.as_deref(), Some("Think."));
                        if header == "final" {
                            assert_eq!(result.final_text, "{}");
                        } else {
                            assert_eq!(result.commentary.unwrap()[0].function.name, "lookup");
                        }
                    }
                } else {
                    assert!(collected.unwrap_err().contains("without a final message"));
                    assert!(terminal.unwrap_err().contains("without a final message"));
                    // An explicit cut before the action token and unconstrained
                    // output have no final-message requirement.
                    HarmonyParserAdapter::new(true)
                        .unwrap()
                        .parse_complete(&prefix, "length".into())
                        .unwrap();
                    HarmonyParserAdapter::new(false)
                        .unwrap()
                        .parse_complete(&turn, "stop".into())
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn byte_spelled_markers_remain_literal_in_a_valid_message() {
        let tokenizer = try_harmony_encoding()
            .expect("Harmony encoding")
            .tokenizer();
        let mut turn = tokenizer.encode_with_special_tokens("<|channel|>final<|message|>");
        turn.extend(tokenizer.encode_ordinary("The literal marker is <|message|>."));
        turn.extend(tokenizer.encode_with_special_tokens("<|return|>"));
        let mut parser = HarmonyParserAdapter::new(true).expect("Harmony parser");
        let result = parser
            .parse_complete(&turn, "stop".to_string())
            .expect("valid message");
        assert_eq!(result.final_text, "The literal marker is <|message|>.");
    }
}
