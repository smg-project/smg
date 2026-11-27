// Kimi K3 reasoning parser for the XTML tag format.
//
// K3 uses structural special tokens <|open|>, <|close|>, <|sep|> to delimit
// channels. This parser extracts the `think` channel into `reasoning_text` and
// delivers the remainder (with `response`/`message` wrapper tokens removed) as
// `normal_text`. The `tools` channel is preserved verbatim for the tool parser.
//
// A tag the end of the output cut short (`<|close|>message` with no `<|sep|>`
// after it, when `max_tokens` falls inside the turn's closing structure) is
// dropped at the end of the output on both paths: the structural tokens are
// atomic, so a tail that begins with a whole one is never the model's text.
//
// Ported from the Kimi-K3 reference reasoning parser (XTML `think` channel).

use regex::Regex;

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

/// Literal marker strings (used for streaming overlap detection).
const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const MESSAGE_OPEN: &str = "<|open|>message<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
/// The structural tokens every XTML tag begins with. Each is one token of the
/// model's vocabulary, so it reaches the parser whole or not at all.
const STRUCTURAL_TOKENS: [&str; 2] = ["<|open|>", "<|close|>"];
/// Every tag this parser reads, for the check at the end of an output.
const TAGS: [&str; 7] = [
    THINK_OPEN,
    THINK_CLOSE,
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    MESSAGE_OPEN,
    MESSAGE_CLOSE,
    TOOLS_OPEN,
];

/// `text` less a tag the end of the output cut short: a tail that begins with a
/// whole structural token and is a proper prefix of one of [`TAGS`], such as the
/// `<|close|>message` left when `max_tokens` falls inside the turn's closing
/// structure. The tokens are atomic, so such a tail cannot be the model's text,
/// and it would reach the client as content otherwise. A tail shorter than a
/// token (`<|clo`) could be text, and stays, as every parser keeps a partial
/// marker at the end of the stream.
fn without_cut_tag(text: &str) -> &str {
    let Some(at) = STRUCTURAL_TOKENS
        .iter()
        .filter_map(|token| text.rfind(token))
        .max()
    else {
        return text;
    };
    let tail = &text[at..];
    if TAGS
        .iter()
        .any(|tag| tag.len() > tail.len() && tag.starts_with(tail))
    {
        &text[..at]
    } else {
        text
    }
}

/// Reasoning parser for the Kimi K3 (XTML) chat format.
///
/// Extracts the `think` channel into `reasoning_text` and strips the
/// `response`/`message` wrapper tokens from `normal_text` by substitution,
/// preserving any `tools` channel for downstream parsing.
#[derive(Debug, Clone)]
pub struct KimiK3Parser {
    think_open_re: Regex,
    think_close_re: Regex,
    response_open_re: Regex,
    response_close_re: Regex,
    message_open_re: Regex,
    message_close_re: Regex,
    tools_open_re: Regex,

    // Shared state
    in_reasoning: bool,

    // Streaming accumulation
    buffer: String,
    reason_phase_ended: bool,
    /// Byte offset in `buffer` where post-think content begins.
    content_tail_start: usize,
    /// Safe-to-emit reasoning already returned to the caller.
    emitted_reasoning: String,
    /// Safe-to-emit content already returned to the caller.
    emitted_content: String,
}

impl KimiK3Parser {
    /// Create a new `KimiK3Parser` with pre-compiled tolerant XTML regexes.
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are hardcoded valid literals; failure indicates a programming error"
    )]
    pub fn new() -> Self {
        let open = r"<\|open\|>";
        let close = r"<\|close\|>";
        let sep = r"<\|sep\|>";

        Self {
            think_open_re: Regex::new(&format!(r"{open}\s*think\s*{sep}"))
                .expect("valid think-open regex"),
            think_close_re: Regex::new(&format!(r"{close}\s*think\s*{sep}"))
                .expect("valid think-close regex"),
            response_open_re: Regex::new(&format!(r"{open}\s*response\s*{sep}"))
                .expect("valid response-open regex"),
            response_close_re: Regex::new(&format!(r"{close}\s*response\s*{sep}"))
                .expect("valid response-close regex"),
            message_open_re: Regex::new(&format!(r"{open}\s*message\s*{sep}"))
                .expect("valid message-open regex"),
            message_close_re: Regex::new(&format!(r"{close}\s*message\s*{sep}"))
                .expect("valid message-close regex"),
            tools_open_re: Regex::new(&format!(r"{open}\s*tools\s*{sep}"))
                .expect("valid tools-open regex"),

            in_reasoning: false,
            buffer: String::new(),
            reason_phase_ended: false,
            content_tail_start: 0,
            emitted_reasoning: String::new(),
            emitted_content: String::new(),
        }
    }

    /// Strip the `response` and `message` wrapper markers
    /// (`<|open|>response<|sep|>`, `<|close|>response<|sep|>`,
    /// `<|open|>message<|sep|>`, `<|close|>message<|sep|>`) from `text` via
    /// substitution.
    ///
    /// Unlike a structured extraction that pulls only the response body,
    /// this approach removes only the wrapper markers, preserving any
    /// `<|open|>tools<|sep|>…` channel that follows for the tool parser. A tag
    /// the end of the output cut short goes too ([`without_cut_tag`]).
    fn strip_content_wrapper(&self, text: &str) -> String {
        let text = self.response_open_re.replace_all(text, "");
        let text = self.response_close_re.replace_all(&text, "");
        let text = self.message_open_re.replace_all(&text, "");
        let text = self.message_close_re.replace_all(&text, "");
        without_cut_tag(&text).to_owned()
    }

    /// Locate the end of the think channel in `text`, searching from `from`.
    ///
    /// Returns `(reasoning_end, content_start)` for the earliest terminator, or
    /// `None` when there is none. Three kinds, each spanning differently:
    ///
    /// - think-close: consumed, so content resumes after it.
    /// - sibling `response`/`tools` *opener*: left in place for downstream
    ///   parsing, since a forced tool call can jump straight from the prefilled
    ///   `<|open|>think<|sep|>` into `<|open|>tools<|sep|>`. `message` is
    ///   excluded — it opens *before* the think channel.
    /// - sibling `response`/`message` *closer*: spans `(from, from)`, all answer
    ///   and no reasoning, and only when `!think_observed`.
    ///
    /// `think_observed` says whether `text` holds a literal `<|open|>think<|sep|>`,
    /// as opposed to the channel being *assumed* open because the caller
    /// pre-marked reasoning for a prefilled prompt. Only when assumed does a
    /// sibling closer mean anything: it proves the model ignored the prefill and
    /// answered directly, so `strip_content_wrapper` clears the stray markers.
    /// With the channel observed, the same closer is malformed output and must
    /// not retroactively empty the reasoning span.
    fn think_channel_end(
        &self,
        text: &str,
        from: usize,
        think_observed: bool,
    ) -> Option<(usize, usize)> {
        let close = self
            .think_close_re
            .find_at(text, from)
            .map(|m| (m.start(), (m.start(), m.end())));
        let sibling_open = [&self.response_open_re, &self.tools_open_re]
            .into_iter()
            .filter_map(|re| re.find_at(text, from).map(|m| m.start()))
            .min()
            .map(|start| (start, (start, start)));
        let sibling_close = (!think_observed)
            .then(|| {
                [&self.response_close_re, &self.message_close_re]
                    .into_iter()
                    .filter_map(|re| re.find_at(text, from).map(|m| m.start()))
                    .min()
            })
            .flatten()
            .map(|start| (start, (from, from)));
        // Keyed on marker position, not on the returned span, or `(from, from)`
        // would always win. Ties favour `close` (listed first), though no two
        // markers share a literal prefix so ties cannot arise.
        [close, sibling_open, sibling_close]
            .into_iter()
            .flatten()
            .min_by_key(|&(marker_start, _)| marker_start)
            .map(|(_, span)| span)
    }

    /// Return the prefix of accumulated reasoning text that is safe to stream.
    ///
    /// Skips past the think-open marker if present, then holds back any partial
    /// marker suffix that could complete into a think marker or a sibling opener.
    fn reasoning_text_ready_to_emit<'a>(&self, text: &'a str) -> &'a str {
        let after_open = if let Some(m) = self.think_open_re.find(text) {
            &text[m.end()..]
        } else {
            text
        };

        // Hold back any tail that could complete into a channel terminator.
        // Sibling closers need listing individually despite sharing the
        // `<|close|>` prefix with `THINK_CLOSE`: `compute_overlap` matches whole
        // suffixes, so `<|close|>res` would otherwise leak the `res`.
        let overlap = Self::compute_overlap(
            after_open,
            &[
                THINK_OPEN,
                THINK_CLOSE,
                RESPONSE_OPEN,
                RESPONSE_CLOSE,
                MESSAGE_CLOSE,
                TOOLS_OPEN,
            ],
        );
        if overlap > 0 {
            &after_open[..after_open.len() - overlap]
        } else {
            after_open
        }
    }

    /// Return the prefix of post-reasoning content that is safe to stream.
    ///
    /// Strips the response-open prefix, removes complete response-close and
    /// message-close markers, then holds back any partial marker suffix; on the
    /// final chunk a tag the end of the output cut short is dropped instead.
    fn content_ready_to_emit(&self, text: &str, final_chunk: bool) -> String {
        // Strip the response-open marker as a prefix (first occurrence only).
        // In the K3 grammar the response channel opens exactly once and its
        // opening marker is a prefix of the content tail, so a slice from its
        // end is correct and a second occurrence cannot legitimately appear.
        // The response-close / message-open / message-close markers, by
        // contrast, are terminators that may sit anywhere in the tail, so they
        // are removed globally below. This asymmetry is deliberate.
        let text = if let Some(m) = self.response_open_re.find(text) {
            &text[m.end()..]
        } else {
            text
        };

        // Remove all complete response-close, message-open, and message-close markers
        let text = self.response_close_re.replace_all(text, "");
        let text = self.message_open_re.replace_all(&text, "");
        let text = self.message_close_re.replace_all(&text, "");

        if final_chunk {
            return without_cut_tag(&text).to_owned();
        }

        // Hold a possible marker prefix until the next chunk or EOF.
        let text_str: &str = &text;
        let overlap = Self::compute_overlap(
            text_str,
            &[RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_OPEN, MESSAGE_CLOSE],
        );
        if overlap > 0 {
            text_str[..text_str.len() - overlap].to_owned()
        } else {
            text_str.to_owned()
        }
    }

    /// Compute the maximum length of a suffix of `text` that is also a
    /// non-empty prefix of any marker in `markers`.
    fn compute_overlap(text: &str, markers: &[&str]) -> usize {
        let mut overlap = 0usize;
        for marker in markers {
            let max_check = marker.len().saturating_sub(1).min(text.len());
            for n in (1..=max_check).rev() {
                if text.ends_with(&marker[..n]) {
                    overlap = overlap.max(n);
                    break;
                }
            }
        }
        overlap
    }
}

impl Default for KimiK3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for KimiK3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        // Guard against oversized single-shot input, mirroring the streaming
        // entry point (`BaseReasoningParser` and `InklingParser` apply the same
        // guard to their non-streaming path).
        if text.len() > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(text.len()));
        }

        let m_open = self.think_open_re.find(text);
        let reasoning_start = m_open.map(|m| m.end()).unwrap_or(0);

        // A think channel exists if the marker is present, or the caller
        // pre-marked reasoning because the prompt prefilled it.
        let has_think_channel = m_open.is_some() || self.in_reasoning;

        // Plain content: strip the wrappers and hand everything downstream,
        // tools channel included.
        if !has_think_channel && self.think_close_re.find(text).is_none() {
            return Ok(ParserResult::normal(self.strip_content_wrapper(text)));
        }

        let Some((reasoning_end, content_start)) =
            self.think_channel_end(text, reasoning_start, m_open.is_some())
        else {
            // Opened but never closed — e.g. output truncated mid-thought.
            // Classify the remainder as reasoning rather than lose it, less a
            // tag the truncation cut short.
            self.in_reasoning = true;
            return Ok(ParserResult::reasoning(
                without_cut_tag(&text[reasoning_start..]).to_owned(),
            ));
        };

        let reasoning = text[reasoning_start..reasoning_end].to_owned();
        let normal = self.strip_content_wrapper(&text[content_start..]);
        self.in_reasoning = false;
        Ok(ParserResult::new(normal, reasoning))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        // Guard against unbounded buffer growth, mirroring `BaseReasoningParser`.
        let buffered_size = self.buffer.len() + text.len();
        if buffered_size > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(buffered_size));
        }

        self.buffer.push_str(text);

        // Without a prefilled think channel, the stream begins in normal content.
        // Hold only a possible split think-open marker; otherwise route the
        // accumulated text downstream so structural tool tokens reach the tool parser.
        if !self.in_reasoning
            && !self.reason_phase_ended
            && self.think_open_re.find(&self.buffer).is_none()
        {
            if THINK_OPEN.starts_with(self.buffer.as_str()) {
                return Ok(ParserResult::default());
            }
            self.reason_phase_ended = true;
            self.content_tail_start = 0;
        }

        if self.reason_phase_ended {
            // Content phase: emit new content from the post-think tail.
            let content_tail = &self.buffer[self.content_tail_start..];
            let current_safe = self.content_ready_to_emit(content_tail, false);
            // Invariant: `emitted_content` is always a prefix of `current_safe`;
            // on the impossible mismatch, emit nothing rather than double-emit.
            let new_content = if current_safe.starts_with(self.emitted_content.as_str()) {
                current_safe[self.emitted_content.len()..].to_owned()
            } else {
                String::new()
            };
            self.emitted_content = current_safe;
            return Ok(ParserResult {
                normal_text: new_content,
                reasoning_text: String::new(),
            });
        }

        // Has the reasoning phase ended in the buffer yet? Accepting a sibling
        // opener as well as the think-close marker is what lets a forced tools
        // channel reach the content phase instead of streaming as reasoning
        // indefinitely.
        //
        // A sibling closer can also end it — see `think_channel_end`. Recovery is
        // partial while streaming, since reasoning deltas for what turns out to be
        // the answer cannot be recalled and so repeat in `reasoning_text`; what it
        // buys is a complete `normal_text` and a cleared `in_reasoning`, which
        // gates the tool parser. Non-streaming is exact.
        let think_open = self.think_open_re.find(&self.buffer);
        let think_observed = think_open.is_some();
        let r_start = think_open.map(|m| m.end()).unwrap_or(0);

        if let Some((reasoning_end, content_start)) =
            self.think_channel_end(&self.buffer, r_start, think_observed)
        {
            // Transition: reasoning phase ends.
            self.reason_phase_ended = true;
            self.content_tail_start = content_start;
            self.in_reasoning = false;

            // Final reasoning delta.
            let full_reasoning = &self.buffer[r_start..reasoning_end];
            // Invariant: `emitted_reasoning` is always a prefix of `full_reasoning`;
            // on the impossible mismatch, emit nothing rather than double-emit.
            let reasoning_delta = if full_reasoning.starts_with(self.emitted_reasoning.as_str()) {
                full_reasoning[self.emitted_reasoning.len()..].to_owned()
            } else {
                String::new()
            };

            // Initial content delta (safe prefix of the content tail).
            let content_tail = &self.buffer[content_start..];
            let content_safe = self.content_ready_to_emit(content_tail, false);
            self.emitted_content.clone_from(&content_safe);

            Ok(ParserResult {
                normal_text: content_safe,
                reasoning_text: reasoning_delta,
            })
        } else {
            // Still in reasoning — emit the safe prefix of accumulated reasoning.
            // The tool parser is gated on `is_in_reasoning()`, so the flag must
            // stay set until the think-close transition above clears it.
            self.in_reasoning = true;
            let current_safe = self.reasoning_text_ready_to_emit(&self.buffer).to_owned();
            // Invariant: `emitted_reasoning` is always a prefix of `current_safe`;
            // on the impossible mismatch, emit nothing rather than double-emit.
            let reasoning_delta = if current_safe.starts_with(self.emitted_reasoning.as_str()) {
                current_safe[self.emitted_reasoning.len()..].to_owned()
            } else {
                String::new()
            };
            self.emitted_reasoning = current_safe;
            Ok(ParserResult {
                normal_text: String::new(),
                reasoning_text: reasoning_delta,
            })
        }
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        if self.buffer.is_empty() {
            return Ok(ParserResult::default());
        }
        let result = if self.reason_phase_ended {
            let content = self.content_ready_to_emit(&self.buffer[self.content_tail_start..], true);
            ParserResult::normal(
                content
                    .strip_prefix(&self.emitted_content)
                    .unwrap_or("")
                    .to_owned(),
            )
        } else if self.in_reasoning {
            let start = self.think_open_re.find(&self.buffer).map_or(0, |m| m.end());
            let text = without_cut_tag(&self.buffer[start..]);
            ParserResult::reasoning(
                text.strip_prefix(&self.emitted_reasoning)
                    .unwrap_or("")
                    .to_owned(),
            )
        } else {
            ParserResult::normal(self.buffer.clone())
        };
        self.buffer.clear();
        Ok(result)
    }

    fn reset(&mut self) {
        self.in_reasoning = false;
        self.buffer.clear();
        self.reason_phase_ended = false;
        self.content_tail_start = 0;
        self.emitted_reasoning.clear();
        self.emitted_content.clear();
    }

    fn model_type(&self) -> &str {
        "kimi_k3"
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }

    fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    fn mark_reasoning_started(&mut self) {
        self.in_reasoning = true;
    }

    fn mark_think_start_stripped(&mut self) {
        // For K3 streaming, the absence of a think-open marker in the buffer
        // is already handled by treating reasoning_start as 0. No additional
        // state is needed.
    }
}

#[cfg(test)]
#[path = "../../tests/unit/parsers/kimi_k3/tests.rs"]
mod tests;
