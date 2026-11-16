//! Harmony response parser
//!
//! Adapter for openai_harmony::StreamableParser that handles channel-based parsing.

use openai_harmony::{chat::Role, HarmonyEncoding, StreamableParser};
use openai_protocol::common::{FunctionCallResponse, ToolCall};
use uuid::Uuid;

use super::types::{HarmonyChannelDelta, HarmonyChannelOutput};
use crate::routers::grpc::common::responses::utils::cut_call;

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
    /// The call the stream announced last: the closed messages when it opened, and its place
    /// among the turn's calls.
    announced_call: Option<(usize, usize)>,
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
            announced_call: None,
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
        // A message addressed to a function is a call whatever its channel: its open content
        // is neither reasoning nor answer text (`extract_incomplete_commentary` reads it).
        if parser
            .current_recipient()
            .is_some_and(|r| r.starts_with("functions."))
        {
            return;
        }
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
            self.parser
                .process(token_id)
                .map_err(|e| format!("Failed to process token {token_id}: {e}"))?;

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
        let (mut analysis, closed, mut final_text) = Self::parse_messages(messages);
        Self::handle_incomplete_content(&self.parser, &mut analysis, &mut final_text);
        // A turn none of whose calls closed returns the call it ended inside, as far as it
        // came: every read takes it as the stream's recovery does.
        let open = if closed.is_none() {
            self.extract_incomplete_commentary()
        } else {
            None
        };
        // A turn ended on its calls when one closed, or when it stopped inside a whole one:
        // a worker may stop at a call's closing token without delivering it. A turn the
        // token limit or a failure ended, or one that stopped inside a call's arguments,
        // keeps the engine's finish, and the Responses loops run none of its server calls.
        let stopped_on_a_whole_call = finish_reason == "stop"
            && open.as_deref().is_some_and(|calls| {
                let arguments = calls
                    .iter()
                    .filter_map(|call| call.function.arguments.as_deref());
                !cut_call(None, arguments)
            });
        let finish_reason = if closed.is_some() || stopped_on_a_whole_call {
            "tool_calls".to_string()
        } else {
            finish_reason
        };
        HarmonyChannelOutput {
            analysis,
            commentary: closed.or(open),
            final_text,
            finish_reason,
            reasoning_token_count,
        }
    }

    /// Get all messages from the parser
    ///
    /// Returns the raw messages extracted by the Harmony parser.
    /// Used for validation checks.
    pub fn get_messages(&self) -> &[openai_harmony::chat::Message] {
        self.parser.messages()
    }

    /// The call the turn ended in: the open message addressed to a function, whatever its
    /// channel, as `parse_messages` reads a closed one. A worker may stop at a call's closing
    /// token without delivering it, so a turn's last call is often still open.
    pub fn extract_incomplete_commentary(&self) -> Option<Vec<ToolCall>> {
        let recipient = self.parser.current_recipient()?;
        let function_name = recipient.strip_prefix("functions.")?;
        let content = self.parser.current_content().ok()?;
        if content.is_empty() {
            return None;
        }
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
        let mut commentary_deltas: Vec<super::types::ToolCallDelta> = Vec::new();
        let mut final_delta: Option<String> = None;

        // Process each token
        for &token_id in chunk_ids {
            self.parser
                .process(token_id)
                .map_err(|e| format!("Failed to process token {token_id}: {e}"))?;

            if self.in_reasoning_channel() {
                self.reasoning_token_count += 1;
            }
            let text = self.parser.last_content_delta().ok().flatten();

            // A message addressed to a function is a tool call whatever its channel, as
            // `parse_messages` reads it. Every call of the chunk leaves its own delta, in the
            // turn's order: a chunk may hold a whole call, or the end of one and the start
            // of the next. A call is new by its place among the turn's calls, not by its
            // recipient: two calls in a row may name one tool. The closed messages do not
            // grow inside a call, so its recipient is read and its place counted once, when
            // it opens, not on every token of it.
            let closed = self.parser.messages().len();
            let mut call = match self.announced_call {
                Some((at, index)) if at == closed => Some(index),
                _ => None,
            };
            if call.is_none() {
                let recipient = self.parser.current_recipient();
                if let Some(tool_name) = recipient
                    .as_deref()
                    .and_then(|recipient| recipient.strip_prefix("functions."))
                {
                    let index = self
                        .parser
                        .messages()
                        .iter()
                        .filter(|msg| {
                            msg.recipient
                                .as_deref()
                                .is_some_and(|r| r.starts_with("functions."))
                        })
                        .count();
                    self.announced_call = Some((closed, index));
                    commentary_deltas.push(super::types::ToolCallDelta {
                        index,
                        id: Some(call_id()),
                        function: Some(super::types::FunctionDelta {
                            name: Some(tool_name.to_string()),
                            arguments: Some(String::new()),
                        }),
                    });
                    call = Some(index);
                }
            }
            if let Some(index) = call {
                if let Some(text) = text.filter(|text| !text.is_empty()) {
                    let open = commentary_deltas
                        .last_mut()
                        .filter(|delta| delta.index == index)
                        .and_then(|delta| delta.function.as_mut()?.arguments.as_mut());
                    match open {
                        Some(arguments) => arguments.push_str(&text),
                        None => commentary_deltas.push(super::types::ToolCallDelta {
                            index,
                            id: None,
                            function: Some(super::types::FunctionDelta {
                                name: None,
                                arguments: Some(text),
                            }),
                        }),
                    }
                }
                continue;
            }

            if let Some(text) = text {
                match self.parser.current_channel().as_deref() {
                    Some("analysis") => {
                        analysis_delta
                            .get_or_insert_with(String::new)
                            .push_str(&text);
                    }
                    Some("final") | None => {
                        final_delta.get_or_insert_with(String::new).push_str(&text);
                    }
                    _ => {}
                }
            }
        }

        // Only emit a delta if at least one channel produced output
        let has_delta =
            analysis_delta.is_some() || !commentary_deltas.is_empty() || final_delta.is_some();

        if has_delta {
            Ok(Some(HarmonyChannelDelta {
                analysis_delta,
                commentary_deltas,
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
    use crate::routers::grpc::harmony::types::ToolCallDelta;

    const CALL: &str = "<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{\"city\": \"Paris\"}<|call|>";

    fn ids(text: &str) -> Vec<u32> {
        try_harmony_encoding()
            .expect("harmony encoding must load")
            .tokenizer()
            .encode_with_special_tokens(text)
    }

    /// The call deltas a turn leaves when its tokens come `size` at a time.
    fn call_deltas(turn: &str, size: usize) -> Vec<ToolCallDelta> {
        let mut parser = HarmonyParserAdapter::new().expect("a parser");
        ids(turn)
            .chunks(size)
            .filter_map(|chunk| parser.parse_chunk(chunk).expect("a parsed chunk"))
            .flat_map(|delta| delta.commentary_deltas)
            .collect()
    }

    #[test]
    fn each_call_is_named_once_and_its_argument_deltas_add_up_however_the_chunks_fall() {
        // Two calls in a row to one tool: the second is new by its place, not its recipient.
        let turn = format!("{CALL}<|start|>assistant{CALL}");
        // The last size holds the whole turn in one chunk.
        for size in (1..=8).chain([usize::MAX]) {
            let deltas = call_deltas(&turn, size);
            assert!(
                deltas.iter().all(|delta| delta.index < 2),
                "chunks of {size}: {deltas:?}"
            );
            for index in 0..2 {
                let call: Vec<&ToolCallDelta> =
                    deltas.iter().filter(|delta| delta.index == index).collect();
                let names: Vec<&str> = call
                    .iter()
                    .filter_map(|delta| delta.function.as_ref()?.name.as_deref())
                    .collect();
                let arguments: String = call
                    .iter()
                    .filter_map(|delta| delta.function.as_ref()?.arguments.as_deref())
                    .collect();
                assert_eq!(names, ["get_time"], "call {index}, chunks of {size}");
                assert!(call[0].id.is_some(), "call {index}, chunks of {size}");
                assert_eq!(
                    arguments, r#"{"city": "Paris"}"#,
                    "call {index}, chunks of {size}"
                );
            }
        }
    }

    #[test]
    fn a_call_on_the_analysis_channel_is_a_call_as_the_final_list_reads_it() {
        let turn = CALL.replace("commentary", "analysis");
        let mut parser = HarmonyParserAdapter::new().expect("a parser");
        let delta = parser
            .parse_chunk(&ids(&turn))
            .expect("a parsed chunk")
            .expect("a delta");
        assert_eq!(delta.analysis_delta, None);
        let [call] = delta.commentary_deltas.as_slice() else {
            panic!("one call delta: {:?}", delta.commentary_deltas);
        };
        let function = call.function.as_ref().expect("the call's function");
        assert_eq!(function.name.as_deref(), Some("get_time"));
        assert_eq!(function.arguments.as_deref(), Some(r#"{"city": "Paris"}"#));
    }

    #[test]
    fn a_call_cut_before_its_closing_token_is_read_on_any_channel() {
        for channel in ["commentary", "analysis"] {
            let turn = CALL.replace("commentary", channel);
            let cut = turn
                .strip_suffix("<|call|>")
                .expect("the call's closing token");
            let mut parser = HarmonyParserAdapter::new().expect("a parser");
            parser.parse_chunk(&ids(cut)).expect("a parsed chunk");

            let calls = parser
                .extract_incomplete_commentary()
                .expect("the open call");
            let [call] = calls.as_slice() else {
                panic!("{channel}: one call, not {}", calls.len());
            };
            assert_eq!(call.function.name, "get_time", "{channel}");
            assert_eq!(
                call.function.arguments.as_deref(),
                Some(r#"{"city": "Paris"}"#),
                "{channel}"
            );
        }
    }
}
