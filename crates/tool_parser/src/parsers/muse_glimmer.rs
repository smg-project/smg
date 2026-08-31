use std::sync::LazyLock;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::{
    errors::ParserResult,
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Muse-Glimmer ATEM tool-call parser.
///
/// Muse-Glimmer frames every assistant message as
/// `<|start|>assistant to=<recipient><|message|><body><terminator>`. A recipient
/// of `self` is chain-of-thought and `user` is the answer; any other recipient
/// addresses a tool, and that segment's body carries the call:
///
/// ```text
/// <atem:function_calls>
/// <atem:invoke name="get_weather">
/// <atem:parameter name="city">Paris</atem:parameter>
/// </atem:invoke>
/// </atem:function_calls>
/// ```
///
/// Channel scoping is load-bearing: the model quotes this markup inside its own
/// reasoning, and an invoke echoed in a `to=self` or `to=user` body must never
/// become a real call. The parser therefore segments the stream first and only
/// then extracts from tool-addressed bodies.
///
/// Reference: the model's published chat template, whose `render_atem` macro
/// emits the markup above and whose tool-definition prose states that the output
/// "is not expected to be valid XML" and that "spaces for string values are not
/// stripped".
pub struct MuseGlimmerParser {
    /// Streaming buffer; the whole thing is re-scanned on every chunk.
    buffer: String,
    /// Bytes of derived normal text already returned to the caller.
    emitted_normal_len: usize,
    /// Complete calls already emitted, in emission order.
    sent_tool_call_count: usize,
}

const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";
const FUNCTION_CALLS_OPEN: &str = "<atem:function_calls>";
const INVOKE_OPEN: &str = "<atem:invoke";

#[expect(
    clippy::expect_used,
    reason = "regex pattern is a compile-time string literal"
)]
static INVOKE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<atem:invoke\b[^>]*?\bname="([^"]*)"[^>]*?>(.*?)</atem:invoke>"#)
        .expect("valid ATEM invoke pattern")
});

#[expect(
    clippy::expect_used,
    reason = "regex pattern is a compile-time string literal"
)]
static PARAM_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    // `[^"]*` rather than `[^"]+` so a malformed empty name still matches and
    // the scan advances past it instead of stalling.
    Regex::new(r#"(?s)<atem:parameter\b[^>]*?\bname="([^"]*)"[^>]*?>(.*?)</atem:parameter>"#)
        .expect("valid ATEM parameter pattern")
});

const SELF_RECIPIENT: &str = "self";
const USER_RECIPIENT: &str = "user";

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
    LeadingHeader,
    Header,
    /// Inside a `to=self` body, which this parser discards.
    Reasoning,
    Content,
    Tool,
    Idle,
}

#[derive(Debug, Clone, Copy)]
enum ControlCandidate {
    Complete { start: usize, token: &'static str },
    Partial { start: usize },
}

#[derive(Debug, Default)]
struct Scan {
    normal_text: String,
    calls: Vec<ToolCall>,
}

/// Decode a parameter value written by the template's `render_atem` macro.
///
/// The macro writes booleans as bare `true`/`false`, `None` as bare `null`,
/// mappings and non-string sequences as JSON, and every other scalar verbatim
/// and unquoted. The value is therefore never trimmed: its bytes are the value.
fn infer_atem_value(raw: &str) -> Value {
    match raw {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" => return Value::Null,
        _ => {}
    }

    // Containers and bare numbers round-trip through JSON; anything else is a
    // verbatim string, spaces and newlines included. Python-style `True`/`None`
    // are deliberately NOT special-cased: the macro does not emit them, so a
    // value that literally reads "True" is a string.
    if let Ok(value) = serde_json::from_str::<Value>(raw) {
        if value.is_object() || value.is_array() || value.is_number() {
            return value;
        }
    }

    Value::String(raw.to_string())
}

fn coerce_atem_value(raw: &str, declared_type: Option<&str>) -> Value {
    match declared_type {
        // Values are rendered verbatim and unquoted, so a declared string is
        // exactly its bytes. `coerce_by_schema_type`'s string arm would first
        // try `from_str::<String>`, which unwraps a value that happens to read
        // `"hello"` (quotes included) down to `hello`.
        Some("string") => Value::String(raw.to_string()),
        Some(declared) => helpers::coerce_by_schema_type(raw, Some(declared))
            .unwrap_or_else(|| infer_atem_value(raw)),
        None => infer_atem_value(raw),
    }
}

impl MuseGlimmerParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            emitted_normal_len: 0,
            sent_tool_call_count: 0,
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

    /// The recipient of a segment, from the first `to=` token in its header.
    /// Compared exactly: `to=Self` names a tool, not the reasoning channel.
    fn recipient_from_header(header: &str) -> Option<&str> {
        header
            .split_whitespace()
            .find_map(|token| token.strip_prefix("to="))
            .filter(|recipient| !recipient.is_empty())
    }

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

    /// Map an emitted invoke name back onto a registered tool.
    ///
    /// When a client registers a bare name the template advertises the valid
    /// recipient as `NAME.*`, and the model answers with `NAME.NAME`. Collapsing
    /// that doubled form is safe because both halves are identical and the
    /// result is registered. Matching on the trailing segment alone would not
    /// be: an emitted `weather.get` against a registered `calendar.get` has a
    /// unique leaf match and would dispatch the wrong tool.
    fn normalize_name(emitted: &str, tools: &[Tool]) -> String {
        if tools.is_empty() || tools.iter().any(|tool| tool.function.name == emitted) {
            return emitted.to_string();
        }
        if let Some((head, tail)) = emitted.split_once('.') {
            if head == tail && tools.iter().any(|tool| tool.function.name == head) {
                return head.to_string();
            }
        }
        emitted.to_string()
    }

    /// Extract every complete `<atem:invoke>` from a tool-channel body.
    fn extract_calls(body: &str, tools: &[Tool]) -> (Vec<ToolCall>, Option<usize>) {
        let mut calls = Vec::new();
        let mut last_call_end = None;

        for capture in INVOKE_PATTERN.captures_iter(body) {
            let (Some(full_match), Some(name_match), Some(inner_match)) =
                (capture.get(0), capture.get(1), capture.get(2))
            else {
                continue;
            };
            let name = Self::normalize_name(name_match.as_str(), tools);
            if name.is_empty() {
                continue;
            }

            let param_types = helpers::param_types_for_function(tools, &name);
            let mut parameters = serde_json::Map::new();
            for param in PARAM_PATTERN.captures_iter(inner_match.as_str()) {
                if let (Some(key), Some(value)) = (param.get(1), param.get(2)) {
                    let key = key.as_str().to_string();
                    let declared = param_types.get(&key).map(String::as_str);
                    parameters.insert(key, coerce_atem_value(value.as_str(), declared));
                }
            }

            let arguments = serde_json::to_string(&parameters).unwrap_or_else(|_| "{}".to_string());
            calls.push(ToolCall {
                function: FunctionCall { name, arguments },
            });
            last_call_end = Some(full_match.end());
        }

        (calls, last_call_end)
    }

    /// Close a tool segment: harvest its calls, and if it produced none while
    /// being complete, give the body back as normal text rather than dropping
    /// it or leaking its framing.
    fn close_tool_segment(body: &mut String, closed: bool, tools: &[Tool], scan: &mut Scan) {
        if body.is_empty() {
            return;
        }
        let (calls, last_call_end) = Self::extract_calls(body, tools);
        if calls.is_empty() {
            if closed {
                scan.normal_text.push_str(body);
                body.clear();
            }
            // An open body that has not produced a call yet is held: it may
            // still close into one, and emitting it now would un-emit later.
            return;
        }
        scan.calls.extend(calls);
        if closed {
            // A valid call must not make a malformed call after it disappear.
            // Drop completed-call framing, but surface the unclosed invoke as
            // normal text just as we do when it is the only invoke in a body.
            if let Some(tail) = last_call_end.and_then(|end| body.get(end..)) {
                if let Some(invoke_start) = tail.find(INVOKE_OPEN) {
                    scan.normal_text.push_str(&tail[invoke_start..]);
                }
            }
            body.clear();
        }
    }

    /// Accumulate header bytes, flushing them as content if they turn out not
    /// to be a header at all (framing stripped upstream, or a debug parse of
    /// plain text).
    fn push_header_text(header: &mut String, state: &mut State, text: &str, scan: &mut Scan) {
        header.push_str(text);
        let viable = match *state {
            State::LeadingHeader => Self::leading_header_viable(header),
            _ => Self::header_viable(header),
        };
        if !viable {
            scan.normal_text.push_str(header);
            header.clear();
            *state = State::Content;
        }
    }

    /// Segment `text` and derive normal text plus complete calls.
    ///
    /// Pure: pooled parsers are shared across requests, and the streaming path
    /// re-runs this over the whole buffer on every chunk, so the derived normal
    /// text must only ever grow.
    fn scan(text: &str, tools: &[Tool], finalize: bool) -> Scan {
        let mut scan = Scan::default();
        let mut state = State::LeadingHeader;
        let mut header = String::new();
        let mut tool_body = String::new();
        let mut pos = 0;

        while pos < text.len() {
            let remaining = &text[pos..];
            let (chunk_end, control) = match Self::find_control_candidate(remaining) {
                Some(ControlCandidate::Complete { start, token }) => (start, Some((start, token))),
                // Hold a partial marker back unless this is the final parse,
                // where it can only ever be literal text.
                Some(ControlCandidate::Partial { start }) if !finalize => (start, None),
                _ => (remaining.len(), None),
            };

            let chunk = &remaining[..chunk_end];
            if !chunk.is_empty() {
                match state {
                    State::LeadingHeader | State::Header => {
                        Self::push_header_text(&mut header, &mut state, chunk, &mut scan);
                    }
                    State::Reasoning => {}
                    State::Content | State::Idle => scan.normal_text.push_str(chunk),
                    State::Tool => tool_body.push_str(chunk),
                }
            }

            let Some((start, token)) = control else {
                break;
            };

            if token == START {
                // Opens a new header whatever preceded it, so a tool segment
                // whose terminator never arrived ends here.
                Self::close_tool_segment(&mut tool_body, true, tools, &mut scan);
                header.clear();
                state = State::Header;
            } else if matches!(state, State::LeadingHeader | State::Header) {
                if token == MESSAGE {
                    match Self::recipient_from_header(&header) {
                        Some(SELF_RECIPIENT) => state = State::Reasoning,
                        Some(USER_RECIPIENT) | None => state = State::Content,
                        Some(_) => state = State::Tool,
                    }
                    header.clear();
                }
            } else if token == EOM || token == EOT {
                if state == State::Tool {
                    Self::close_tool_segment(&mut tool_body, true, tools, &mut scan);
                }
                state = State::Idle;
            } else if state == State::Tool {
                // Media placeholders can appear inside a parameter value.
                tool_body.push_str(token);
            }

            pos += start + token.len();
        }

        // A tool segment still open at end-of-input can already have produced
        // complete calls; surface those without giving up its body.
        Self::close_tool_segment(&mut tool_body, finalize, tools, &mut scan);

        scan
    }

    /// Segmenting never fails: unparsable input degrades to normal text, so
    /// this returns the pair directly and the trait methods wrap it.
    ///
    /// Always segments, even with no ATEM markup in sight. Short-circuiting on
    /// `has_tool_markers` would return the raw bytes — framing and any `to=self`
    /// body included — for a turn that reasoned and answered without calling a
    /// tool, and would disagree with `parse_incremental`, which has no such
    /// shortcut. Unframed plain text still round-trips unchanged: the scanner's
    /// leading-header valve flushes it verbatim.
    fn parse_complete_inner(text: &str, tools: &[Tool]) -> (String, Vec<ToolCall>) {
        let scan = Self::scan(text, tools, true);
        (scan.normal_text, scan.calls)
    }
}

impl Default for MuseGlimmerParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MuseGlimmerParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(Self::parse_complete_inner(output, &[]))
    }

    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(Self::parse_complete_inner(output, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        let scan = Self::scan(&self.buffer, tools, false);

        // The derived text only grows, so the delta is its unsent suffix. On the
        // impossible mismatch emit nothing rather than double-emit.
        let normal_text = if scan.normal_text.len() >= self.emitted_normal_len
            && scan.normal_text.is_char_boundary(self.emitted_normal_len)
        {
            scan.normal_text[self.emitted_normal_len..].to_string()
        } else {
            String::new()
        };
        self.emitted_normal_len = scan.normal_text.len();

        // Arguments are opaque until the invoke closes, so each call is emitted
        // once, whole. `tool_index` continues across segments.
        let mut calls = Vec::new();
        for (offset, call) in scan
            .calls
            .iter()
            .enumerate()
            .skip(self.sent_tool_call_count)
        {
            calls.push(ToolCallItem {
                tool_index: offset,
                name: Some(call.function.name.clone()),
                parameters: call.function.arguments.clone(),
            });
        }
        self.sent_tool_call_count = scan.calls.len();

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(FUNCTION_CALLS_OPEN) || text.contains(INVOKE_OPEN)
    }

    fn take_unstreamed_normal_text(&mut self) -> String {
        // The incremental scan intentionally holds an open tool body because a
        // later chunk may complete its invoke. At end of stream that ambiguity
        // is gone: finalize the same scanner and return only normal text that
        // was not already emitted. This also keeps truncated framing out of the
        // response while preserving the malformed body for diagnosis.
        let finalized = Self::scan(&self.buffer, &[], true).normal_text;
        let tail = finalized
            .get(self.emitted_normal_len..)
            .unwrap_or_default()
            .to_string();

        self.buffer.clear();
        self.emitted_normal_len = 0;
        self.sent_tool_call_count = 0;
        tail
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.emitted_normal_len = 0;
        self.sent_tool_call_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;

    use super::*;

    fn segment(recipient: &str, body: &str, terminator: &str) -> String {
        format!("{START}assistant to={recipient}{MESSAGE}{body}{terminator}")
    }

    fn invoke(name: &str, params: &str) -> String {
        format!("<atem:invoke name=\"{name}\">{params}</atem:invoke>")
    }

    fn param(key: &str, value: &str) -> String {
        format!("<atem:parameter name=\"{key}\">{value}</atem:parameter>")
    }

    fn calls_block(invokes: &str) -> String {
        format!("{FUNCTION_CALLS_OPEN}{invokes}</atem:function_calls>")
    }

    fn tools_with(props: Value) -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": props}),
                strict: None,
            },
        }]
    }

    fn args_of(call: &ToolCall) -> Value {
        serde_json::from_str(&call.function.arguments).unwrap()
    }

    #[tokio::test]
    async fn parses_a_single_call() {
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &param("city", "Paris"))),
            EOM,
        );

        let (normal, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert_eq!(normal, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(args_of(&calls[0]), serde_json::json!({"city": "Paris"}));
    }

    #[tokio::test]
    async fn parses_parallel_invokes_in_one_block() {
        let block = calls_block(&format!(
            "{}{}",
            invoke("get_weather", &param("city", "Paris")),
            invoke("get_weather", &param("city", "Berlin"))
        ));
        let text = segment("get_weather", &block, EOM);

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(args_of(&calls[1]), serde_json::json!({"city": "Berlin"}));
    }

    #[tokio::test]
    async fn reasoning_channel_invokes_are_never_calls() {
        // The model quotes ATEM markup inside its own chain of thought.
        let quoted = calls_block(&invoke("get_weather", &param("city", "Paris")));
        let text = format!(
            "{}{}",
            segment("self", &format!("I could write {quoted}"), EOM),
            segment("user", "Let me know which city.", EOT)
        );

        let (normal, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert!(calls.is_empty(), "reasoning markup must not become a call");
        assert_eq!(normal, "Let me know which city.");
    }

    #[tokio::test]
    async fn answer_channel_invokes_are_never_calls() {
        let quoted = calls_block(&invoke("get_weather", &param("city", "Paris")));
        let text = segment("user", &format!("You would write {quoted}"), EOT);

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn answer_text_precedes_a_call_without_framing() {
        let text = format!(
            "{}{}",
            segment("user", "Checking now.", EOM),
            segment(
                "get_weather",
                &calls_block(&invoke("get_weather", &param("city", "Paris"))),
                EOM
            )
        );

        let (normal, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert_eq!(normal, "Checking now.");
        assert!(!normal.contains("<|"), "framing must never surface");
        assert_eq!(calls.len(), 1);
    }

    #[tokio::test]
    async fn leading_headerless_tool_segment_parses() {
        // The generation prompt supplies `<|start|>assistant`, so a first-segment
        // call arrives without it.
        let text = format!(
            " to=get_weather{MESSAGE}{}{EOM}",
            calls_block(&invoke("get_weather", &param("city", "Paris")))
        );

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[tokio::test]
    async fn empty_invoke_yields_empty_arguments() {
        let text = segment("get_weather", &calls_block(&invoke("get_weather", "")), EOM);

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert_eq!(args_of(&calls[0]), serde_json::json!({}));
    }

    #[tokio::test]
    async fn text_without_markers_is_returned_unchanged() {
        let (normal, calls) = MuseGlimmerParser::new()
            .parse_complete("just an answer")
            .await
            .unwrap();

        assert_eq!(normal, "just an answer");
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn malformed_body_keeps_its_text_and_drops_framing() {
        let broken = format!("{FUNCTION_CALLS_OPEN}<atem:invoke name=\"get_weather\">");
        let text = segment("get_weather", &broken, EOM);

        let (normal, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert!(calls.is_empty());
        assert!(normal.contains(FUNCTION_CALLS_OPEN));
        assert!(!normal.contains("<|start|>"));
        assert!(!normal.contains(MESSAGE));
    }

    #[tokio::test]
    async fn doubled_namespaced_name_collapses_to_the_registered_tool() {
        // A bare registered name is advertised as `NAME.*`, so the model answers
        // `NAME.NAME`.
        let text = segment(
            "get_weather.get_weather",
            &calls_block(&invoke("get_weather.get_weather", &param("city", "Paris"))),
            EOM,
        );
        let tools = tools_with(serde_json::json!({"city": {"type": "string"}}));

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();

        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[tokio::test]
    async fn unrelated_namespaced_name_is_forwarded_unchanged() {
        let text = segment(
            "weather.get",
            &calls_block(&invoke("weather.get", &param("city", "Paris"))),
            EOM,
        );
        let tools = tools_with(serde_json::json!({"city": {"type": "string"}}));

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();

        assert_eq!(calls[0].function.name, "weather.get");
    }

    #[tokio::test]
    async fn bare_literals_and_containers_decode_by_inference() {
        let params = format!(
            "{}{}{}{}{}",
            param("flag", "true"),
            param("off", "false"),
            param("nothing", "null"),
            param("count", "7"),
            param("coords", "[60,30]")
        );
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &params)),
            EOM,
        );

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();
        let args = args_of(&calls[0]);

        assert_eq!(args["flag"], Value::Bool(true));
        assert_eq!(args["off"], Value::Bool(false));
        assert_eq!(args["nothing"], Value::Null);
        assert_eq!(args["count"], serde_json::json!(7));
        assert_eq!(args["coords"], serde_json::json!([60, 30]));
    }

    #[tokio::test]
    async fn scalar_values_keep_their_spaces_verbatim() {
        // The template's own prose: "spaces for string values are not stripped".
        let params = format!(
            "{}{}",
            param("note", "  padded  "),
            param("multi", "line one\nline two")
        );
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &params)),
            EOM,
        );

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();
        let args = args_of(&calls[0]);

        assert_eq!(args["note"], Value::String("  padded  ".to_string()));
        assert_eq!(
            args["multi"],
            Value::String("line one\nline two".to_string())
        );
    }

    #[tokio::test]
    async fn python_literals_are_plain_strings_without_a_schema() {
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &param("flag", "True"))),
            EOM,
        );

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert_eq!(
            args_of(&calls[0])["flag"],
            Value::String("True".to_string())
        );
    }

    #[tokio::test]
    async fn declared_string_keeps_numeric_and_quoted_values_verbatim() {
        let params = format!("{}{}", param("city", "007"), param("note", "\"hello\""));
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &params)),
            EOM,
        );
        let tools = tools_with(serde_json::json!({
            "city": {"type": "string"},
            "note": {"type": "string"},
        }));

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        let args = args_of(&calls[0]);

        assert_eq!(args["city"], Value::String("007".to_string()));
        // Not unwrapped to `hello`: the bytes are the value.
        assert_eq!(args["note"], Value::String("\"hello\"".to_string()));
    }

    #[tokio::test]
    async fn declared_types_coerce_when_inference_would_not() {
        let params = format!("{}{}", param("flag", "True"), param("count", "42"));
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &params)),
            EOM,
        );
        let tools = tools_with(serde_json::json!({
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
        }));

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        let args = args_of(&calls[0]);

        assert_eq!(args["flag"], Value::Bool(true));
        assert_eq!(args["count"], serde_json::json!(42));
    }

    #[tokio::test]
    async fn value_with_angle_bracket_is_not_truncated() {
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &param("expr", "a < b"))),
            EOM,
        );

        let (_, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert_eq!(
            args_of(&calls[0])["expr"],
            Value::String("a < b".to_string())
        );
    }

    #[tokio::test]
    async fn calls_continue_indexing_across_segments() {
        let text = format!(
            "{}{}",
            segment(
                "get_weather",
                &calls_block(&invoke("get_weather", &param("city", "Paris"))),
                EOM
            ),
            segment(
                "get_weather",
                &calls_block(&invoke("get_weather", &param("city", "Berlin"))),
                EOT
            )
        );

        let mut parser = MuseGlimmerParser::new();
        let result = parser.parse_incremental(&text, &[]).await.unwrap();

        let indices: Vec<usize> = result.calls.iter().map(|c| c.tool_index).collect();
        assert_eq!(indices, vec![0, 1]);
    }

    #[tokio::test]
    async fn streaming_matches_one_shot_at_every_chunk_boundary() {
        let text = format!(
            "{}{}{}",
            segment("self", "thinking it through", EOM),
            segment(
                "get_weather",
                &calls_block(&invoke("get_weather", &param("city", "Paris"))),
                EOM
            ),
            segment("user", "It is sunny.", EOT)
        );

        for split in text
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(text.len()))
        {
            let mut parser = MuseGlimmerParser::new();
            let first = parser.parse_incremental(&text[..split], &[]).await.unwrap();
            let second = parser.parse_incremental(&text[split..], &[]).await.unwrap();

            let normal = format!("{}{}", first.normal_text, second.normal_text);
            assert_eq!(normal, "It is sunny.", "content mismatch at split {split}");
            assert!(
                !normal.contains("<|"),
                "framing leaked at split {split}: {normal}"
            );

            let names: Vec<String> = first
                .calls
                .iter()
                .chain(second.calls.iter())
                .filter_map(|c| c.name.clone())
                .collect();
            assert_eq!(
                names,
                vec!["get_weather".to_string()],
                "call mismatch at split {split}"
            );
        }
    }

    #[tokio::test]
    async fn streaming_emits_each_call_once_with_whole_arguments() {
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &param("city", "Paris"))),
            EOM,
        );

        let mut parser = MuseGlimmerParser::new();
        let mut emitted = Vec::new();
        // Feed one byte at a time; the call must surface exactly once.
        for index in 0..text.len() {
            if !text.is_char_boundary(index) || !text.is_char_boundary(index + 1) {
                continue;
            }
            let result = parser
                .parse_incremental(&text[index..index + 1], &[])
                .await
                .unwrap();
            emitted.extend(result.calls);
        }

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].name.as_deref(), Some("get_weather"));
        let args: Value = serde_json::from_str(&emitted[0].parameters).unwrap();
        assert_eq!(args, serde_json::json!({"city": "Paris"}));
    }

    #[tokio::test]
    async fn streaming_holds_an_incomplete_invoke() {
        let opening = format!(
            "{START}assistant to=get_weather{MESSAGE}{FUNCTION_CALLS_OPEN}<atem:invoke name=\"get_weather\">"
        );

        let mut parser = MuseGlimmerParser::new();
        let result = parser.parse_incremental(&opening, &[]).await.unwrap();

        // The invoke has not closed, so there is nothing to emit yet — and the
        // body must not leak as content while it may still become a call.
        assert!(result.calls.is_empty());
        assert!(result.normal_text.is_empty());
    }

    #[tokio::test]
    async fn end_of_stream_flushes_an_incomplete_invoke_as_content() {
        let body = format!("{FUNCTION_CALLS_OPEN}<atem:invoke name=\"get_weather\">");
        let opening = format!("{START}assistant to=get_weather{MESSAGE}{body}");

        let mut parser = MuseGlimmerParser::new();
        let result = parser.parse_incremental(&opening, &[]).await.unwrap();

        assert!(result.calls.is_empty());
        assert!(result.normal_text.is_empty());
        assert_eq!(parser.take_unstreamed_normal_text(), body);
        assert_eq!(parser.take_unstreamed_normal_text(), "");
    }

    #[tokio::test]
    async fn end_of_stream_does_not_reemit_a_completed_open_invoke() {
        let body = calls_block(&invoke("get_weather", &param("city", "Paris")));
        let opening = format!("{START}assistant to=get_weather{MESSAGE}{body}");

        let mut parser = MuseGlimmerParser::new();
        let result = parser.parse_incremental(&opening, &[]).await.unwrap();

        assert_eq!(result.calls.len(), 1);
        assert!(result.normal_text.is_empty());
        assert_eq!(parser.take_unstreamed_normal_text(), "");
    }

    #[tokio::test]
    async fn end_of_stream_flushes_a_truncated_invoke_after_a_completed_invoke() {
        let completed = invoke("get_weather", &param("city", "Paris"));
        let truncated = "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Ber";
        let body = format!("{FUNCTION_CALLS_OPEN}{completed}{truncated}");
        let opening = format!("{START}assistant to=get_weather{MESSAGE}{body}");

        let mut parser = MuseGlimmerParser::new();
        let result = parser.parse_incremental(&opening, &[]).await.unwrap();

        assert_eq!(result.calls.len(), 1);
        assert!(result.normal_text.is_empty());
        assert_eq!(parser.take_unstreamed_normal_text(), truncated);
        assert_eq!(parser.take_unstreamed_normal_text(), "");
    }

    #[tokio::test]
    async fn unframed_atem_markup_stays_content() {
        // Channel framing is what authorizes a call. Without it there is no way
        // to tell a real invocation from one the model quoted inside its own
        // reasoning, so the markup is surfaced verbatim rather than executed:
        // visible markup is a debuggable symptom, a fabricated tool call is not.
        let raw = calls_block(&invoke("get_weather", &param("city", "Paris")));

        let mut parser = MuseGlimmerParser::new();
        let result = parser.parse_incremental(&raw, &[]).await.unwrap();

        assert!(result.calls.is_empty());
        assert_eq!(result.normal_text, raw);
    }

    #[tokio::test]
    async fn reset_allows_reuse() {
        let text = segment(
            "get_weather",
            &calls_block(&invoke("get_weather", &param("city", "Paris"))),
            EOM,
        );

        let mut parser = MuseGlimmerParser::new();
        parser.parse_incremental(&text, &[]).await.unwrap();
        parser.reset();

        let result = parser.parse_incremental(&text, &[]).await.unwrap();
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
    }

    /// A turn that reasoned and answered without calling a tool still has full
    /// framing. Returning it verbatim would leak `<|start|>` and the private
    /// chain-of-thought to the client, and would differ from the streaming path.
    #[tokio::test]
    async fn framed_turn_without_calls_is_still_segmented() {
        let text = format!(
            "{}{}",
            segment("self", "No tool applies here.", EOM),
            segment("user", "Paris is the capital of France.", EOT)
        );

        let (normal, calls) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();

        assert!(calls.is_empty());
        assert_eq!(normal, "Paris is the capital of France.");
        assert!(!normal.contains("<|"), "framing must not leak: {normal:?}");
        assert!(
            !normal.contains("No tool applies"),
            "reasoning must not surface"
        );
    }

    /// The complete and streaming paths must derive the same normal text.
    #[tokio::test]
    async fn complete_and_streaming_agree_on_a_callless_framed_turn() {
        let text = format!(
            "{}{}",
            segment("self", "thinking", EOM),
            segment("user", "the answer", EOT)
        );

        let (complete, _) = MuseGlimmerParser::new()
            .parse_complete(&text)
            .await
            .unwrap();
        let streamed = MuseGlimmerParser::new()
            .parse_incremental(&text, &[])
            .await
            .unwrap()
            .normal_text;

        assert_eq!(complete, streamed);
    }

    #[test]
    fn has_tool_markers_keys_on_the_body() {
        let parser = MuseGlimmerParser::new();
        assert!(parser.has_tool_markers(FUNCTION_CALLS_OPEN));
        assert!(parser.has_tool_markers("<atem:invoke name=\"x\">"));
        assert!(!parser.has_tool_markers("plain answer"));
        assert!(!parser.has_tool_markers(&segment("user", "hello", EOT)));
    }
}
