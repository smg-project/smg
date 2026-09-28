//! Reasoning parser for tokenizer-declared response templates.

use std::sync::Arc;

use smg_response_template::{
    consume, find_close, partial_close_len, CompiledTemplate, Field, Opener,
};

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

/// Splits output into the `thinking` and `content` fields of a template.
///
/// Field framing is removed. `tool_calls` blocks stay in the normal text
/// verbatim, framing included, for the template tool parser, or are dropped
/// when the request does not parse tool calls. Whitespace
/// between blocks is dropped and any other text outside a block is normal
/// text. A block without a close ends with the input, because a close that
/// is an EOS token never reaches the parser.
pub struct TemplateReasoningParser {
    template: Arc<CompiledTemplate>,
    buffer: String,
    /// Bytes at the start of `buffer` kept only as look-behind context.
    context: usize,
    /// Offset in `buffer` before which no opener can start.
    checked: usize,
    block: Option<Field>,
    tool_calls_enabled: bool,
    max_buffer_size: usize,
}

impl TemplateReasoningParser {
    pub fn new(template: Arc<CompiledTemplate>) -> Self {
        Self {
            template,
            buffer: String::new(),
            context: 0,
            checked: 0,
            block: None,
            tool_calls_enabled: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
        }
    }

    /// Drop `buffer[..end]`, keeping look-behind context.
    fn advance(&mut self, end: usize) {
        self.context = consume(&mut self.buffer, end);
        self.checked = 0;
    }

    fn parse(&mut self, eof: bool) -> ParserResult {
        let mut result = ParserResult::default();
        loop {
            let Some(field) = self.block else {
                if self.parse_outside(eof, &mut result) {
                    continue;
                }
                return result;
            };
            let text = &self.buffer[self.context..];
            let closes = self.template.closes(field);
            let (text_end, end, closed) = match find_close(text, closes) {
                Some((start, end)) => (start, end, true),
                None if eof => (text.len(), text.len(), true),
                None => {
                    let end = text.len() - partial_close_len(text, closes);
                    (end, end, false)
                }
            };
            match field {
                Field::Thinking => result.reasoning_text.push_str(&text[..text_end]),
                Field::Content => result.normal_text.push_str(&text[..text_end]),
                Field::ToolCalls if self.tool_calls_enabled => {
                    result.normal_text.push_str(&text[..end]);
                }
                Field::ToolCalls => {}
            }
            self.advance(self.context + end);
            if !closed {
                return result;
            }
            self.block = None;
        }
    }

    /// Handle text between blocks. Returns whether a block was opened.
    fn parse_outside(&mut self, eof: bool, result: &mut ParserResult) -> bool {
        let from = self.context.max(self.checked);
        let mut found = self.template.scan(&self.buffer, from, &Field::ALL, eof);
        if found.is_none() && eof {
            // A partial opener at the end of input is dropped, not emitted.
            found = self.template.scan(&self.buffer, from, &Field::ALL, false);
        }
        let (text_end, opener) = match found {
            Some(Opener::Complete { field, start, end }) => (start, Some((field, end))),
            Some(Opener::Pending { start }) => (start, None),
            None => (self.buffer.len(), None),
        };
        let text = &self.buffer[self.context..text_end];
        // Whitespace before an opener or the end of input is layout. Hold
        // trailing whitespace until the next text decides which it is.
        let kept = text.trim_end().len();
        result.normal_text.push_str(&text[..kept]);
        let Some((field, end)) = opener else {
            if eof {
                self.advance(self.buffer.len());
            } else {
                let consumed = self.context + kept;
                self.advance(consumed);
                self.checked = text_end - (consumed - self.context);
            }
            return false;
        };
        // Never report normal text while in reasoning: open the thinking
        // block on the next call instead.
        if field == Field::Thinking && !eof && !result.normal_text.is_empty() {
            self.advance(text_end);
            return false;
        }
        if field == Field::ToolCalls && self.tool_calls_enabled {
            result.normal_text.push_str(&self.buffer[text_end..end]);
        }
        self.advance(end);
        self.block = Some(field);
        true
    }
}

impl ReasoningParser for TemplateReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        if text.len() > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(text.len()));
        }
        // Complete parsing is independent of any streaming state.
        let mut parser = Self::new(self.template.clone());
        parser.tool_calls_enabled = self.tool_calls_enabled;
        parser.buffer.push_str(text);
        Ok(parser.parse(true))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        let size = self.buffer.len() + text.len();
        if size > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(size));
        }
        self.buffer.push_str(text);
        Ok(self.parse(false))
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        Ok(self.parse(true))
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.context = 0;
        self.checked = 0;
        self.block = None;
    }

    fn model_type(&self) -> &str {
        "response_template"
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }

    fn set_tool_calls_enabled(&mut self, enabled: bool) {
        self.tool_calls_enabled = enabled;
    }

    fn is_in_reasoning(&self) -> bool {
        self.block == Some(Field::Thinking)
    }

    fn mark_reasoning_started(&mut self) {
        // The template decides where thinking starts; a prompt that ends
        // inside a thinking block is not supported.
    }

    fn mark_think_start_stripped(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParserFactory;

    const TOOL_BLOCK: &str =
        "<|turn|>bot<|tool|>get_weather<|args|><arg name=\"city\">Paris</arg><|eos|>";
    const OUTPUT: &str = concat!(
        "<|think|>check the map<|done|>",
        "<|turn|>bot<|say|>Paris it is.<|done|>",
        "<|turn|>bot<|tool|>get_weather<|args|><arg name=\"city\">Paris</arg><|eos|>",
    );

    fn template() -> Arc<CompiledTemplate> {
        let value = serde_json::json!({
            "start_anchor_pattern": "<\\|turn\\|>bot",
            "fields": {
                "thinking": {
                    "open_pattern": "(?:<\\|turn\\|>bot)?<\\|think\\|>",
                    "close": "<|done|>",
                    "content": "text"
                },
                "content": {
                    "open_pattern": "(?:<\\|turn\\|>bot)?<\\|say\\|>",
                    "close": ["<|done|>", "<|eos|>"],
                    "content": "text"
                },
                "tool_calls": {
                    "open_pattern": "(?:<\\|turn\\|>bot)?<\\|tool\\|>(?P<name>[A-Za-z_][A-Za-z0-9_.]*)<\\|args\\|>",
                    "close": ["<|done|>", "<|eos|>"],
                    "repeats": true,
                    "content": "xml-inline",
                    "content_args": {
                        "tag_pattern": "<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>",
                        "value_parser": {"name": "text"}
                    },
                    "transform": {"name": "{name}", "arguments": "{content}"}
                }
            }
        });
        Arc::new(CompiledTemplate::from_value(&value).unwrap())
    }

    fn parser() -> TemplateReasoningParser {
        TemplateReasoningParser::new(template())
    }

    fn whole(text: &str) -> ParserResult {
        parser().detect_and_parse_reasoning(text).unwrap()
    }

    /// Feed `chunks`, flush, and return every call's result.
    fn stream<'a>(chunks: impl IntoIterator<Item = &'a str>) -> Vec<ParserResult> {
        let mut parser = parser();
        let mut results: Vec<_> = chunks
            .into_iter()
            .map(|chunk| parser.parse_reasoning_streaming_incremental(chunk).unwrap())
            .collect();
        results.push(parser.flush().unwrap());
        assert!(!parser.is_in_reasoning());
        results
    }

    fn joined(results: &[ParserResult]) -> ParserResult {
        let mut all = ParserResult::default();
        for result in results {
            all.normal_text.push_str(&result.normal_text);
            all.reasoning_text.push_str(&result.reasoning_text);
        }
        all
    }

    #[test]
    fn separates_fields_and_keeps_tool_blocks_verbatim() {
        let result = whole(OUTPUT);
        assert_eq!(result.reasoning_text, "check the map");
        assert_eq!(result.normal_text, format!("Paris it is.{TOOL_BLOCK}"));
        let parser = parser();
        assert!(parser.requires_special_tokens());
        assert_eq!(parser.model_type(), "response_template");
    }

    #[test]
    fn any_chunking_matches_the_whole_input() {
        let eos_hidden = OUTPUT.strip_suffix("<|eos|>").unwrap();
        let outside = " {\"a\": 1}\n<|say|>x<|done|>\n\n<|think|>y<|done|> tail \n";
        for text in [OUTPUT, eos_hidden, outside] {
            let expected = whole(text);
            let bounds: Vec<_> = text.char_indices().map(|(at, _)| at).collect();
            for &at in &bounds {
                let (a, b) = text.split_at(at);
                assert_eq!(joined(&stream([a, b])), expected, "{text:?} split at {at}");
            }
            let chars: Vec<String> = text.chars().map(String::from).collect();
            assert_eq!(joined(&stream(chars.iter().map(String::as_str))), expected);
        }
        // Only whitespace before an opener or the end of input is dropped.
        assert_eq!(whole(outside).normal_text, " {\"a\": 1}x tail");
        assert_eq!(whole(outside).reasoning_text, "y");
    }

    #[test]
    fn streams_each_token_as_it_arrives() {
        let chunks = [
            "<|think|>",
            "check",
            " the",
            " map",
            "<|done|>",
            "<|say|>",
            "Paris",
            " it",
            "<|do",
            "ne|>",
        ];
        let results = stream(chunks);
        let reasoning: Vec<_> = results.iter().map(|r| r.reasoning_text.as_str()).collect();
        let normal: Vec<_> = results.iter().map(|r| r.normal_text.as_str()).collect();
        assert_eq!(reasoning[1..4], ["check", " the", " map"]);
        assert_eq!(normal[6..], ["Paris", " it", "", "", ""]);
    }

    #[test]
    fn never_reports_content_while_in_reasoning() {
        let mut parser = parser();
        let first = parser
            .parse_reasoning_streaming_incremental("<|say|>hi<|done|><|think|>deep")
            .unwrap();
        assert_eq!(first, ParserResult::normal("hi".to_string()));
        assert!(!parser.is_in_reasoning());
        let second = parser.parse_reasoning_streaming_incremental("er").unwrap();
        assert_eq!(second, ParserResult::reasoning("deeper".to_string()));
        assert!(parser.is_in_reasoning());
    }

    #[test]
    fn end_of_input_closes_the_open_block() {
        // A close that is an EOS token is removed before the parser sees it.
        assert_eq!(joined(&stream(["<|say|>answer"])).normal_text, "answer");
        assert_eq!(joined(&stream(["<|say|>"])), ParserResult::default());
        assert_eq!(
            joined(&stream(["<|think|>unfinished"])),
            ParserResult::reasoning("unfinished".to_string())
        );
        let tool = "<|tool|>get_time<|args|>";
        assert_eq!(joined(&stream([tool])).normal_text, tool);
        // A partial close in an open block is text; a partial opener outside
        // a block is dropped.
        assert_eq!(joined(&stream(["<|say|>part<|do"])).normal_text, "part<|do");
        assert_eq!(
            joined(&stream(["<|say|>hi<|done|><|turn|>bo"])).normal_text,
            "hi"
        );
    }

    #[test]
    fn whitespace_between_tokens_of_unframed_text_is_kept() {
        let json = "{\"city\": \"Paris\", \"days\": [1, 2]}";
        let tokens = [
            "{\"city\":",
            " ",
            "\"Paris\",",
            " ",
            "\"days\":",
            " ",
            "[1,",
            " 2]}",
        ];
        assert_eq!(joined(&stream(tokens)).normal_text, json);
        assert_eq!(whole(json).normal_text, json);
    }

    #[test]
    fn drops_tool_blocks_when_tool_calls_are_off() {
        let mut parser = parser();
        parser.set_tool_calls_enabled(false);
        let expected = ParserResult::new("Paris it is.".to_string(), "check the map".to_string());
        assert_eq!(parser.detect_and_parse_reasoning(OUTPUT).unwrap(), expected);
        let mut streamed = ParserResult::default();
        for chunk in OUTPUT.split_inclusive("|>") {
            let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            streamed.normal_text.push_str(&result.normal_text);
            streamed.reasoning_text.push_str(&result.reasoning_text);
        }
        assert_eq!(streamed, expected);
    }

    #[test]
    fn ignores_reasoning_prefill_hints_and_resets() {
        let mut parser = parser();
        parser.mark_reasoning_started();
        parser.mark_think_start_stripped();
        assert!(!parser.is_in_reasoning());
        parser
            .parse_reasoning_streaming_incremental("<|think|>x<|do")
            .unwrap();
        assert!(parser.is_in_reasoning());
        parser.reset();
        let result = parser
            .parse_reasoning_streaming_incremental("plain")
            .unwrap();
        assert_eq!(result, ParserResult::normal("plain".to_string()));

        parser.max_buffer_size = 4;
        assert!(parser
            .parse_reasoning_streaming_incremental("12345")
            .is_err());
        assert!(parser.detect_and_parse_reasoning("12345").is_err());
    }

    #[test]
    fn factory_registers_the_template_once() {
        let factory = ParserFactory::new();
        let template = template();
        let name = factory.register_response_template(template.clone());
        assert_eq!(name, template.parser_name());
        assert_eq!(factory.register_response_template(template), name);
        let parser = factory.registry().create_parser(&name).unwrap();
        assert_eq!(parser.model_type(), "response_template");
    }
}
