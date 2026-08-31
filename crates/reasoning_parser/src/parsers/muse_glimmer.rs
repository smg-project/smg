//! Muse-Glimmer segment-channel reasoning parser.
//!
//! Muse-Glimmer frames every assistant message as
//! `<|start|>assistant to=<recipient><|message|><body><terminator>` where the
//! terminator is `<|eom|>` (another message follows) or `<|eot|>` (end of turn).
//! The recipient selects the channel: `self` is chain-of-thought, `user` (or an
//! absent recipient) is the user-facing answer, and any other value addresses a
//! tool by name.
//!
//! Tool segments are re-emitted verbatim into the normal output so the
//! tool-call parser can consume them after reasoning separation, mirroring the
//! Inkling parser's contract.

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";

/// Recipient of a chain-of-thought segment.
const SELF_RECIPIENT: &str = "self";
/// Recipient of a user-facing answer segment.
const USER_RECIPIENT: &str = "user";

/// Header re-synthesized in front of a tool segment, so the tool parser sees
/// one uniform grammar regardless of where in the turn the segment appeared.
const CANONICAL_TOOL_HEADER_PREFIX: &str = "<|start|>assistant to=";

// Keep this in sync with the Muse-Glimmer tokenizer's added special tokens.
const CONTROL_TOKENS: &[&str] = &[
    START,
    MESSAGE,
    EOM,
    EOT,
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|patch|>",
    "<|image|>",
    "<|video|>",
    "<|image_start|>",
    "<|image_end|>",
    "<|vid_start|>",
    "<|vid_end|>",
    "<|vid_frame_separator|>",
];

/// A header that never closes is malformed; flush it rather than buffer forever.
const MAX_HEADER_LEN: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Stream start. The generation prompt already emitted `<|start|>assistant`,
    /// so the model's first bytes are the rest of a header: ` to=<r><|message|>`.
    LeadingHeader,
    /// After a literal `<|start|>`: role word, optional ` to=<r>`, `<|message|>`.
    Header,
    /// Inside a `to=self` body.
    Reasoning,
    /// Inside a `to=user` body, or one whose header carried no recipient.
    Content,
    /// Inside a tool-addressed body, which is re-emitted verbatim.
    Tool,
    /// Between a terminator and the next `<|start|>`.
    Idle,
}

#[derive(Debug, Clone, Copy)]
enum ControlCandidate {
    Complete { start: usize, token: &'static str },
    Partial { start: usize },
}

/// Parser for Muse-Glimmer's channel-scoped message segments.
#[derive(Debug, Clone)]
pub struct MuseGlimmerParser {
    state: State,
    /// Header bytes seen so far. Protocol metadata: never emitted as content
    /// unless the header turns out not to be a header at all.
    header: String,
    buffer: String,
    max_buffer_size: usize,
}

impl MuseGlimmerParser {
    pub fn new() -> Self {
        Self {
            // Unlike block-delimited formats, a Muse-Glimmer stream begins in
            // the middle of a header: the generation prompt supplied the
            // `<|start|>assistant` prefix and the model continues from ` to=`.
            state: State::LeadingHeader,
            header: String::new(),
            buffer: String::new(),
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
        }
    }

    fn find_control_candidate(text: &str) -> Option<ControlCandidate> {
        for (start, _) in text.match_indices('<') {
            let suffix = &text[start..];

            if let Some(token) = CONTROL_TOKENS
                .iter()
                .copied()
                .find(|token| suffix.starts_with(token))
            {
                return Some(ControlCandidate::Complete { start, token });
            }

            if CONTROL_TOKENS.iter().any(|token| token.starts_with(suffix)) {
                return Some(ControlCandidate::Partial { start });
            }
        }

        None
    }

    fn in_header(&self) -> bool {
        matches!(self.state, State::LeadingHeader | State::Header)
    }

    /// The recipient of a segment, taken from the first `to=` token in its
    /// header. One rule serves every segment: the leading header (` to=self`),
    /// a subsequent header (`assistant to=self`), a header with no recipient at
    /// all (`assistant`), and dotted tool names.
    ///
    /// Comparison is exact — `to=Self` names a tool called `Self`, not the
    /// reasoning channel.
    fn recipient_from_header(header: &str) -> Option<&str> {
        header
            .split_whitespace()
            .find_map(|token| token.strip_prefix("to="))
            .filter(|recipient| !recipient.is_empty())
    }

    /// Whether accumulated bytes can still become a header.
    ///
    /// The leading-header state is entered optimistically, so it must be able to
    /// recognize that it is not looking at a header at all — markers stripped by
    /// an unexpected `skip_special_tokens`, a debug parse of plain text, or a
    /// passthrough-shaped fine-tune. Worst case, text beginning "tomorrow" is
    /// held for two bytes and then flushed intact.
    fn leading_header_viable(header: &str) -> bool {
        if header.len() > MAX_HEADER_LEN {
            return false;
        }
        let (prefix, marker) = match header.find('<') {
            Some(index) => (&header[..index], &header[index..]),
            None => (header, ""),
        };
        if !marker.is_empty() && !MESSAGE.starts_with(marker) && !START.starts_with(marker) {
            return false;
        }
        let prefix = prefix.trim_start_matches(|c: char| c.is_ascii_whitespace());
        if prefix.is_empty() || "to=".starts_with(prefix) {
            return true;
        }
        match prefix.strip_prefix("to=") {
            Some(recipient) => !recipient.chars().any(char::is_whitespace),
            None => false,
        }
    }

    /// Viability once a literal `<|start|>` has been seen. Relaxed relative to
    /// the leading header because the role word (`assistant`) precedes `to=`.
    fn header_viable(header: &str) -> bool {
        if header.len() > MAX_HEADER_LEN {
            return false;
        }
        match header.find('<') {
            Some(index) => {
                let marker = &header[index..];
                MESSAGE.starts_with(marker) || START.starts_with(marker)
            }
            None => true,
        }
    }

    fn push_header(&mut self, text: &str, result: &mut ParserResult) {
        self.header.push_str(text);
        let viable = match self.state {
            State::LeadingHeader => Self::leading_header_viable(&self.header),
            _ => Self::header_viable(&self.header),
        };
        if !viable {
            // Not a header: surface everything accumulated, verbatim. Nothing
            // was emitted before now, so streamed deltas stay consistent.
            result.normal_text.push_str(&self.header);
            self.header.clear();
            self.state = State::Content;
        }
    }

    fn push_text(&mut self, text: &str, result: &mut ParserResult) {
        if text.is_empty() {
            return;
        }
        match self.state {
            State::LeadingHeader | State::Header => self.push_header(text, result),
            State::Reasoning => result.reasoning_text.push_str(text),
            State::Content | State::Tool | State::Idle => result.normal_text.push_str(text),
        }
    }

    fn open_segment(&mut self, result: &mut ParserResult) {
        let recipient = Self::recipient_from_header(&self.header);
        match recipient {
            Some(SELF_RECIPIENT) => self.state = State::Reasoning,
            Some(USER_RECIPIENT) | None => self.state = State::Content,
            Some(tool) => {
                // Re-synthesize a canonical header so the tool parser sees the
                // same grammar for the leading segment (whose `<|start|>` came
                // from the prompt) as for every later one. Role-word spacing is
                // normalized; the recipient is reproduced exactly.
                result.normal_text.push_str(CANONICAL_TOOL_HEADER_PREFIX);
                result.normal_text.push_str(tool);
                result.normal_text.push_str(MESSAGE);
                self.state = State::Tool;
            }
        }
        self.header.clear();
    }

    fn handle_control(&mut self, token: &str, result: &mut ParserResult) {
        if token == START {
            // Opens a new header regardless of what preceded it. A tool segment
            // whose terminator never arrived ends here rather than absorbing the
            // next segment — but the tool parser downstream only sees the bytes
            // we emit, and an unterminated segment there swallows whatever
            // follows it, so close the segment explicitly on the way out.
            // Without this a truncated call is followed by the model's answer
            // being absorbed into the tool body and dropped entirely.
            if self.state == State::Tool {
                result.normal_text.push_str(EOM);
            }
            self.header.clear();
            self.state = State::Header;
            return;
        }

        match self.state {
            State::LeadingHeader | State::Header => {
                if token == MESSAGE {
                    self.open_segment(result);
                }
                // Any other control token inside a header is protocol noise.
            }
            State::Tool => {
                // The tool parser needs the body's framing intact, including a
                // stray media token inside a parameter value.
                result.normal_text.push_str(token);
                if token == EOM || token == EOT {
                    self.state = State::Idle;
                }
            }
            State::Reasoning | State::Content | State::Idle => {
                if token == EOM || token == EOT {
                    self.state = State::Idle;
                }
                // Media placeholders are protocol data, not assistant text.
            }
        }
    }

    fn parse_buffer(&mut self, finalize: bool) -> ParserResult {
        let text = std::mem::take(&mut self.buffer);
        let mut result = ParserResult::default();
        let mut pos = 0;
        // Offset of the `<|start|>` that opened the header being parsed, used
        // to rewind when a reasoning segment must be deferred (see below).
        let mut segment_start = 0usize;

        while pos < text.len() {
            let remaining = &text[pos..];
            match Self::find_control_candidate(remaining) {
                Some(ControlCandidate::Complete { start, token }) => {
                    self.push_text(&remaining[..start], &mut result);
                    let absolute = pos + start;
                    if token == START {
                        segment_start = absolute;
                    }
                    // The gateway reads `is_in_reasoning()` after each chunk and
                    // withholds normal text from the tool parser while it is
                    // true, with no replay. So a single call must never both
                    // produce normal text and finish inside reasoning: defer the
                    // segment and re-parse it on the next call instead.
                    if !finalize
                        && token == MESSAGE
                        && self.in_header()
                        && !result.normal_text.is_empty()
                        && Self::recipient_from_header(&self.header) == Some(SELF_RECIPIENT)
                    {
                        self.header.clear();
                        self.state = State::Idle;
                        self.buffer.push_str(&text[segment_start..]);
                        return result;
                    }
                    self.handle_control(token, &mut result);
                    pos = absolute + token.len();
                }
                Some(ControlCandidate::Partial { start }) => {
                    self.push_text(&remaining[..start], &mut result);
                    let tail = &remaining[start..];
                    if finalize {
                        self.push_text(tail, &mut result);
                    } else {
                        // A partial marker always goes to the re-scan buffer,
                        // including mid-header: a `<|message|>` split across two
                        // chunks must rejoin, and `header` is never re-scanned.
                        self.buffer.push_str(tail);
                    }
                    break;
                }
                None => {
                    self.push_text(remaining, &mut result);
                    break;
                }
            }
        }

        if finalize && self.in_header() {
            // A header truncated by end-of-generation is protocol metadata.
            self.header.clear();
        }

        result
    }
}

impl Default for MuseGlimmerParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MuseGlimmerParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        if text.len() > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(text.len()));
        }

        // Complete parsing is independent of any prior streaming state.
        let mut parser = Self::new();
        parser.max_buffer_size = self.max_buffer_size;
        parser.buffer.push_str(text);
        Ok(parser.parse_buffer(true))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        let buffered_size = self.buffer.len() + text.len();
        if buffered_size > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(buffered_size));
        }

        self.buffer.push_str(text);
        Ok(self.parse_buffer(false))
    }

    fn reset(&mut self) {
        self.state = State::LeadingHeader;
        self.header.clear();
        self.buffer.clear();
    }

    fn model_type(&self) -> &str {
        "muse_glimmer"
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }

    fn is_in_reasoning(&self) -> bool {
        self.state == State::Reasoning
    }

    fn mark_reasoning_started(&mut self) {
        // Deliberately inert. Reasoning is opened by a `to=self` header, never
        // by an injected prefill marker; forcing the state here would route the
        // leading header's own bytes into reasoning as literal text.
    }

    fn mark_think_start_stripped(&mut self) {
        // Deliberately inert, same reason.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leading(recipient: &str, body: &str, terminator: &str) -> String {
        format!(" to={recipient}{MESSAGE}{body}{terminator}")
    }

    fn segment(recipient: &str, body: &str, terminator: &str) -> String {
        format!("{START}assistant to={recipient}{MESSAGE}{body}{terminator}")
    }

    const CALL_BODY: &str = concat!(
        "<atem:function_calls>",
        r#"<atem:invoke name="get_weather">"#,
        r#"<atem:parameter name="city">Paris</atem:parameter>"#,
        "</atem:invoke>",
        "</atem:function_calls>"
    );

    #[test]
    fn leading_reasoning_then_answer() {
        let output = format!(
            "{}{}",
            leading(SELF_RECIPIENT, "weighing options", EOM),
            segment(USER_RECIPIENT, "Here is the answer.", EOT)
        );

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.reasoning_text, "weighing options");
        assert_eq!(result.normal_text, "Here is the answer.");
    }

    #[test]
    fn leading_tool_segment_gets_a_synthesized_start_header() {
        // The prompt supplied `<|start|>assistant`, so the wire form of the
        // first segment has no start marker; the tool parser must still see one.
        let output = leading("get_weather", CALL_BODY, EOM);

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, segment("get_weather", CALL_BODY, EOM));
    }

    #[test]
    fn tool_segment_framing_is_preserved_verbatim() {
        let output = segment("search", CALL_BODY, EOT);

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.normal_text, output);
    }

    #[test]
    fn header_without_recipient_is_content() {
        let output = format!("{START}assistant{MESSAGE}plain answer{EOT}");

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.normal_text, "plain answer");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn dotted_tool_recipient_round_trips() {
        let output = leading("weather.get_current", CALL_BODY, EOM);

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(
            result.normal_text,
            segment("weather.get_current", CALL_BODY, EOM)
        );
    }

    #[test]
    fn recipient_matching_is_case_sensitive() {
        // `to=Self` names a tool, not the reasoning channel.
        let output = leading("Self", CALL_BODY, EOM);

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, segment("Self", CALL_BODY, EOM));
    }

    #[test]
    fn multiple_reasoning_segments_concatenate_in_order() {
        let output = format!(
            "{}{}{}",
            leading(SELF_RECIPIENT, "first thought", EOM),
            segment(SELF_RECIPIENT, "second thought", EOM),
            segment(USER_RECIPIENT, "done", EOT)
        );

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.reasoning_text, "first thoughtsecond thought");
        assert_eq!(result.normal_text, "done");
    }

    #[test]
    fn plain_text_without_framing_is_content() {
        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning("Hello world").unwrap();

        assert_eq!(result.normal_text, "Hello world");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn header_lookalike_prefix_is_flushed_intact() {
        // "tomorrow" starts like a `to=` header; the viability valve must give
        // the bytes back rather than swallowing them.
        let mut parser = MuseGlimmerParser::new();
        let result = parser
            .detect_and_parse_reasoning("tomorrow we ship")
            .unwrap();

        assert_eq!(result.normal_text, "tomorrow we ship");
    }

    #[test]
    fn whitespace_inside_bodies_is_preserved() {
        let output = format!(
            "{}{}",
            leading(SELF_RECIPIENT, "  padded thought \n", EOM),
            segment(USER_RECIPIENT, "\n  spaced answer  ", EOT)
        );

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.reasoning_text, "  padded thought \n");
        assert_eq!(result.normal_text, "\n  spaced answer  ");
    }

    #[test]
    fn media_token_inside_a_tool_body_survives() {
        let body = concat!(
            "<atem:function_calls>",
            r#"<atem:invoke name="describe">"#,
            r#"<atem:parameter name="img"><|patch|></atem:parameter>"#,
            "</atem:invoke>",
            "</atem:function_calls>"
        );
        let output = leading("describe", body, EOM);

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert!(result.normal_text.contains("<|patch|>"));
    }

    #[test]
    fn unterminated_reasoning_is_flushed_on_complete_parse() {
        let output = " to=self<|message|>cut off mid-thought";

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(output).unwrap();

        assert_eq!(result.reasoning_text, "cut off mid-thought");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn truncated_header_is_dropped_on_complete_parse() {
        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(" to=self").unwrap();

        assert_eq!(result, ParserResult::default());
    }

    #[test]
    fn missing_terminator_does_not_absorb_the_next_segment() {
        // The model is known to abandon a reasoning block without `<|eom|>` and
        // open the next channel directly.
        let output = format!(
            " to=self{MESSAGE}abandoned{}",
            segment("get_weather", CALL_BODY, EOM)
        );

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(result.reasoning_text, "abandoned");
        assert_eq!(result.normal_text, segment("get_weather", CALL_BODY, EOM));
    }

    #[test]
    fn streaming_matches_one_shot_at_every_chunk_boundary() {
        let output = format!(
            "{}{}{}",
            leading(SELF_RECIPIENT, "check the map", EOM),
            segment("get_weather", CALL_BODY, EOM),
            segment(USER_RECIPIENT, "It is sunny.", EOT)
        );
        let expected_normal = format!(
            "{}{}",
            segment("get_weather", CALL_BODY, EOM),
            "It is sunny."
        );

        for split in output
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(output.len()))
        {
            let mut parser = MuseGlimmerParser::new();
            let first = parser
                .parse_reasoning_streaming_incremental(&output[..split])
                .unwrap();
            let second = parser
                .parse_reasoning_streaming_incremental(&output[split..])
                .unwrap();

            assert_eq!(
                format!("{}{}", first.reasoning_text, second.reasoning_text),
                "check the map",
                "reasoning mismatch at split {split}"
            );
            assert_eq!(
                format!("{}{}", first.normal_text, second.normal_text),
                expected_normal,
                "content mismatch at split {split}"
            );
        }
    }

    #[test]
    fn streaming_holds_a_partial_terminator() {
        let mut parser = MuseGlimmerParser::new();
        parser
            .parse_reasoning_streaming_incremental(" to=self<|message|>think")
            .unwrap();

        let held = parser
            .parse_reasoning_streaming_incremental("<|eo")
            .unwrap();
        assert_eq!(held, ParserResult::default());

        let closed = parser.parse_reasoning_streaming_incremental("m|>").unwrap();
        assert_eq!(closed, ParserResult::default());
        assert!(!parser.is_in_reasoning());
    }

    #[test]
    fn streaming_holds_a_partial_leading_header() {
        let mut parser = MuseGlimmerParser::new();

        let held = parser
            .parse_reasoning_streaming_incremental(" to=se")
            .unwrap();
        assert_eq!(held, ParserResult::default());

        let opened = parser
            .parse_reasoning_streaming_incremental("lf<|message|>hi")
            .unwrap();
        assert_eq!(opened.reasoning_text, "hi");
        assert_eq!(opened.normal_text, "");
    }

    #[test]
    fn streaming_defers_reasoning_opened_after_normal_text() {
        // Regression: the gateway gates the tool parser on `!is_in_reasoning()`
        // and never replays withheld normal text, so a chunk that emitted a tool
        // segment must not also end inside reasoning.
        let chunk = format!(
            "{}{START}assistant to=self{MESSAGE}later",
            segment("get_weather", CALL_BODY, EOM)
        );

        let mut parser = MuseGlimmerParser::new();
        let first = parser
            .parse_reasoning_streaming_incremental(&chunk)
            .unwrap();

        assert_eq!(first.normal_text, segment("get_weather", CALL_BODY, EOM));
        assert_eq!(first.reasoning_text, "");
        assert!(!parser.is_in_reasoning());

        let second = parser
            .parse_reasoning_streaming_incremental(&format!(" thought{EOM}"))
            .unwrap();
        assert_eq!(second.reasoning_text, "later thought");
    }

    #[test]
    fn is_in_reasoning_tracks_only_the_self_channel() {
        let mut parser = MuseGlimmerParser::new();
        assert!(!parser.is_in_reasoning());

        parser
            .parse_reasoning_streaming_incremental(" to=self<|message|>a")
            .unwrap();
        assert!(parser.is_in_reasoning());

        parser.parse_reasoning_streaming_incremental(EOM).unwrap();
        assert!(!parser.is_in_reasoning());
    }

    #[test]
    fn mark_hooks_are_inert() {
        let mut parser = MuseGlimmerParser::new();
        parser.mark_reasoning_started();
        parser.mark_think_start_stripped();

        let result = parser
            .parse_reasoning_streaming_incremental(&leading(SELF_RECIPIENT, "x", EOM))
            .unwrap();

        assert_eq!(result.reasoning_text, "x");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn reset_restores_the_leading_header_state() {
        let mut parser = MuseGlimmerParser::new();
        parser
            .parse_reasoning_streaming_incremental(&leading(SELF_RECIPIENT, "old", EOM))
            .unwrap();

        parser.reset();
        let result = parser
            .parse_reasoning_streaming_incremental(&leading(SELF_RECIPIENT, "new", EOM))
            .unwrap();

        assert_eq!(result.reasoning_text, "new");
    }

    #[test]
    fn model_type_and_special_token_requirement() {
        let parser = MuseGlimmerParser::new();
        assert_eq!(parser.model_type(), "muse_glimmer");
        assert!(parser.requires_special_tokens());
    }

    /// Regression: a tool segment interrupted by the next `<|start|>` — the
    /// model abandoned the call without its terminator — must still be closed
    /// on the way out. The tool parser only sees what we emit, and an
    /// unterminated segment there absorbs the following answer and drops it.
    #[test]
    fn interrupted_tool_segment_is_closed_before_the_next_one() {
        let output = format!(
            " to=get_weather{MESSAGE}{CALL_BODY}{START}assistant to=user{MESSAGE}It is sunny.{EOT}"
        );

        let mut parser = MuseGlimmerParser::new();
        let result = parser.detect_and_parse_reasoning(&output).unwrap();

        assert_eq!(
            result.normal_text,
            format!("{}It is sunny.", segment("get_weather", CALL_BODY, EOM)),
            "the tool segment must carry a terminator before the answer"
        );
    }

    #[test]
    fn buffer_overflow_is_reported() {
        let mut parser = MuseGlimmerParser::new();
        parser.max_buffer_size = 8;

        let oversized = "x".repeat(9);
        assert!(matches!(
            parser.detect_and_parse_reasoning(&oversized),
            Err(ParseError::BufferOverflow(size)) if size == 9
        ));
        assert!(matches!(
            parser.parse_reasoning_streaming_incremental(&oversized),
            Err(ParseError::BufferOverflow(size)) if size == 9
        ));
    }
}
