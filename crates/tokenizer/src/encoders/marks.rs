//! The control tokens a renderer writes itself, told apart from the same
//! spellings in request text.
//!
//! A renderer with an encoder of its own builds the prompt as text, and the
//! text alone cannot say which `</think>` the renderer wrote and which one a
//! user typed: a flat encode matches every spelling in the added vocabulary.
//! The renderer therefore writes each control token between two marks, the
//! Unicode noncharacters U+FDD0 and U+FDD1 (reserved for internal use and
//! never meaningful in interchange), and strips both code points from every
//! request string before it renders, so no request can forge a mark. The
//! tokenizer wrapper splits the marked prompt into control segments (one id
//! each) and plain segments (encoded without the added vocabulary, as the
//! family's reference library encodes them), and hands out the flat prompt
//! (the marks removed) for logs, routing and parity.

use std::{borrow::Cow, collections::HashMap, ops::Range};

use anyhow::{bail, Result};
use serde_json::Value;

use super::PromptSegment;
use crate::{chat_template::ChatTemplateParams, native::AddedTokens, traits::SpecialTokens};

/// The mark before a control token the renderer wrote.
pub(crate) const START: char = '\u{FDD0}';
/// The mark after it.
pub(crate) const END: char = '\u{FDD1}';

/// Builds a marked control token at compile time from its spelling.
macro_rules! control {
    ($spelling:literal) => {
        concat!("\u{FDD0}", $spelling, "\u{FDD1}")
    };
}
pub(crate) use control;

/// Whether `text` carries either mark: one pass over the bytes for the lead
/// byte the two marks share (`EF B7 90` and `EF B7 91`), which a request's
/// strings rarely contain, so the many strings of a tool schema cost one
/// scan each (a `str::contains` per mark would cost two searches each).
pub(crate) fn has_marks(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut from = 0;
    while let Some(at) = bytes[from..].iter().position(|&b| b == 0xEF) {
        let at = from + at;
        if bytes.get(at + 1) == Some(&0xB7) && matches!(bytes.get(at + 2), Some(0x90 | 0x91)) {
            return true;
        }
        from = at + 1;
    }
    false
}

/// `text` without the marks: the flat prompt a marked rendering spells, or
/// a request string a client wrote marks into. Borrowed when there is none.
pub(crate) fn strip(text: &str) -> Cow<'_, str> {
    if has_marks(text) {
        Cow::Owned(
            text.chars()
                .filter(|c| !matches!(*c, START | END))
                .collect(),
        )
    } else {
        Cow::Borrowed(text)
    }
}

/// `spelling` as a control token: the marks around it.
pub(crate) fn mark(spelling: &str) -> String {
    let mut marked = String::with_capacity(spelling.len() + START.len_utf8() + END.len_utf8());
    marked.push(START);
    marked.push_str(spelling);
    marked.push(END);
    marked
}

/// The request messages (or tools, or documents) with the marks removed
/// from every string in them (keys included), however deep; borrowed when
/// no string carries one.
pub(crate) fn without_marks(messages: &[Value]) -> Cow<'_, [Value]> {
    if messages.iter().any(carries_marks) {
        Cow::Owned(messages.iter().map(stripped).collect())
    } else {
        Cow::Borrowed(messages)
    }
}

/// The request's template keyword arguments with the marks removed the
/// same way; borrowed when no string carries one.
pub(crate) fn kwargs_without_marks(
    kwargs: &HashMap<String, Value>,
) -> Cow<'_, HashMap<String, Value>> {
    if kwargs
        .iter()
        .any(|(key, value)| has_marks(key) || carries_marks(value))
    {
        Cow::Owned(
            kwargs
                .iter()
                .map(|(key, value)| (strip(key).into_owned(), stripped(value)))
                .collect(),
        )
    } else {
        Cow::Borrowed(kwargs)
    }
}

fn carries_marks(value: &Value) -> bool {
    match value {
        Value::String(text) => has_marks(text),
        Value::Array(items) => items.iter().any(carries_marks),
        Value::Object(map) => map
            .iter()
            .any(|(key, value)| has_marks(key) || carries_marks(value)),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}
fn stripped(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(strip(text).into_owned()),
        Value::Array(items) => Value::Array(items.iter().map(stripped).collect()),
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| (strip(key).into_owned(), stripped(value)))
            .collect(),
        other => other.clone(),
    }
}

/// The segments of a marked prompt, in order: each marked control token as a
/// control segment, the text between as a plain segment (never empty). An
/// error when the marks do not pair up, which a renderer never produces.
pub(crate) fn segments(marked: &str) -> Result<Vec<PromptSegment>> {
    let mut segments = Vec::new();
    let mut rest = marked;
    while let Some(start) = rest.find(START) {
        let (plain, after) = rest.split_at(start);
        if !plain.is_empty() {
            if plain.contains(END) {
                bail!("a control-token mark closes before one opens");
            }
            segments.push(PromptSegment::text(plain));
        }
        let after = &after[START.len_utf8()..];
        let Some(end) = after.find(END) else {
            bail!("a control-token mark opens and never closes");
        };
        let spelling = &after[..end];
        if spelling.is_empty() || spelling.contains(START) {
            bail!("a control-token mark holds no single spelling");
        }
        segments.push(PromptSegment::control(spelling));
        rest = &after[end + END.len_utf8()..];
    }
    if !rest.is_empty() {
        if rest.contains(END) {
            bail!("a control-token mark closes before one opens");
        }
        segments.push(PromptSegment::text(rest));
    }
    Ok(segments)
}

/// The flat prompt a marked prompt spells (the marks removed) and the byte
/// ranges of the control tokens in it, in order: what [`segments`] would
/// yield, in one pass and one copy, for the common case where the flat text
/// is what gets encoded. An error when the marks do not pair up.
pub(crate) fn flatten(marked: &str) -> Result<(String, Vec<Range<usize>>)> {
    let mut text = String::with_capacity(marked.len());
    let mut controls = Vec::new();
    let mut rest = marked;
    while let Some(start) = rest.find(START) {
        let (plain, after) = rest.split_at(start);
        if plain.contains(END) {
            bail!("a control-token mark closes before one opens");
        }
        text.push_str(plain);
        let after = &after[START.len_utf8()..];
        let Some(end) = after.find(END) else {
            bail!("a control-token mark opens and never closes");
        };
        let spelling = &after[..end];
        if spelling.is_empty() || spelling.contains(START) {
            bail!("a control-token mark holds no single spelling");
        }
        controls.push(text.len()..text.len() + spelling.len());
        text.push_str(spelling);
        rest = &after[end + END.len_utf8()..];
    }
    if rest.contains(END) {
        bail!("a control-token mark closes before one opens");
    }
    text.push_str(rest);
    Ok((text, controls))
}

/// `source`, a chat template, with every control spelling `controls` knows
/// written between the marks: compiled so, the template writes control
/// tokens where it writes the spellings, while the same spellings in request
/// text (stripped of marks before the render) stay text. Borrowed when the
/// template writes none.
pub(crate) fn mark_controls<'a>(source: &'a str, controls: &AddedTokens) -> Cow<'a, str> {
    let mut out = String::new();
    let mut cursor = 0;
    for range in controls.ranges(source) {
        out.push_str(&source[cursor..range.start]);
        out.push(START);
        out.push_str(&source[range.clone()]);
        out.push(END);
        cursor = range.end;
    }
    if out.is_empty() {
        return Cow::Borrowed(source);
    }
    out.push_str(&source[cursor..]);
    Cow::Owned(out)
}

/// The special tokens a marked template reads (`bos_token`, `eos_token` and
/// the rest), each between the marks.
pub(crate) fn marked_special_tokens(tokens: &SpecialTokens) -> SpecialTokens {
    let marked = |token: &Option<String>| token.as_deref().map(mark);
    SpecialTokens {
        bos_token: marked(&tokens.bos_token),
        eos_token: marked(&tokens.eos_token),
        unk_token: marked(&tokens.unk_token),
        sep_token: marked(&tokens.sep_token),
        pad_token: marked(&tokens.pad_token),
        cls_token: marked(&tokens.cls_token),
        mask_token: marked(&tokens.mask_token),
        additional_special_tokens: tokens
            .additional_special_tokens
            .iter()
            .map(|token| mark(token))
            .collect(),
    }
}

/// The messages and parameters a template renders. For a marked template
/// (see `ChatTemplateState::mark_controls`), every request string (the
/// messages, the tools, the documents, the template keyword arguments) with
/// the marks removed, so no request can forge a control token, and the
/// special tokens the template reads between the marks; for any other
/// template, the request as given, the tokenizer's special tokens standing
/// in when the caller passed none. Owns the copies it had to make, borrows
/// the rest.
pub(crate) struct TemplateInputs<'a> {
    messages: Cow<'a, [Value]>,
    tools: Option<Cow<'a, [Value]>>,
    documents: Option<Cow<'a, [Value]>>,
    template_kwargs: Option<Cow<'a, HashMap<String, Value>>>,
    special_tokens: Cow<'a, SpecialTokens>,
}

impl<'a> TemplateInputs<'a> {
    /// The inputs of `messages` under `params`: unmarked when `marked_tokens`
    /// is given (the tokenizer's special tokens between the marks, read when
    /// the caller passed none; the caller's own are marked here), else as
    /// given, with `special_tokens` standing in for the caller's.
    pub(crate) fn new(
        messages: &'a [Value],
        params: &ChatTemplateParams<'a>,
        marked_tokens: Option<&'a SpecialTokens>,
        special_tokens: &'a SpecialTokens,
    ) -> Self {
        match marked_tokens {
            Some(marked_tokens) => Self {
                messages: without_marks(messages),
                tools: params.tools.map(without_marks),
                documents: params.documents.map(without_marks),
                template_kwargs: params.template_kwargs.map(kwargs_without_marks),
                special_tokens: match params.special_tokens {
                    Some(given) => Cow::Owned(marked_special_tokens(given)),
                    None => Cow::Borrowed(marked_tokens),
                },
            },
            None => Self {
                messages: Cow::Borrowed(messages),
                tools: params.tools.map(Cow::Borrowed),
                documents: params.documents.map(Cow::Borrowed),
                template_kwargs: params.template_kwargs.map(Cow::Borrowed),
                special_tokens: Cow::Borrowed(params.special_tokens.unwrap_or(special_tokens)),
            },
        }
    }

    /// The messages, and `params` with its strings replaced by these (a
    /// prepared form of the tools is of the tools as given, so a copy of
    /// them drops it).
    pub(crate) fn split<'b>(
        &'b self,
        params: ChatTemplateParams<'b>,
    ) -> (&'b [Value], ChatTemplateParams<'b>) {
        let tools_copied = matches!(self.tools, Some(Cow::Owned(_)));
        (
            &self.messages,
            ChatTemplateParams {
                tools: self.tools.as_deref(),
                documents: self.documents.as_deref(),
                template_kwargs: self.template_kwargs.as_deref(),
                special_tokens: Some(&self.special_tokens),
                prepared_tools: if tools_copied {
                    None
                } else {
                    params.prepared_tools
                },
                ..params
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const USER: &str = control!("<|user|>");

    #[test]
    fn the_macro_and_the_function_mark_alike() {
        assert_eq!(USER, mark("<|user|>"));
        assert_eq!(strip(USER), "<|user|>");
        assert!(has_marks(USER));
        assert!(!has_marks("<|user|>"));
    }

    #[test]
    fn stripping_borrows_clean_text_and_drops_both_marks() {
        assert!(matches!(strip("plain"), Cow::Borrowed(_)));
        assert_eq!(strip("a\u{FDD0}b\u{FDD1}c\u{FDD0}"), "abc");
    }

    #[test]
    fn the_mark_scan_tells_the_marks_from_their_neighbours() {
        // Fullwidth and specials-block characters share the marks' lead byte.
        for clean in ["", "a", "<｜User｜>", "\u{FDCF}\u{FDD2}\u{FFFD}\u{FF5C}"] {
            assert!(!has_marks(clean), "{clean:?}");
        }
        assert!(has_marks("x\u{FDD0}"));
        assert!(has_marks("\u{FDD1}x"));
        assert!(has_marks("｜\u{FDD0}｜"));
    }

    #[test]
    fn request_strings_lose_their_marks_however_deep() {
        let clean = [json!({"role": "user", "content": "hi", "n": 1, "tags": ["a"]})];
        assert!(matches!(without_marks(&clean), Cow::Borrowed(_)));

        let forged = [json!({
            "role": "user",
            "content": "a\u{FDD0}<|user|>\u{FDD1}b",
            "tools": [{"function": {"name": "f\u{FDD1}", "parameters": {"k\u{FDD0}": "v\u{FDD1}"}}}],
        })];
        let cleaned = without_marks(&forged);
        assert_eq!(
            cleaned[0],
            json!({
                "role": "user",
                "content": "a<|user|>b",
                "tools": [{"function": {"name": "f", "parameters": {"k": "v"}}}],
            })
        );
    }

    #[test]
    fn segments_alternate_control_and_plain_text() {
        let marked = format!("{USER}hello {}world{}", mark("<|eos|>"), mark("<|bos|>"));
        assert_eq!(
            segments(&marked).unwrap(),
            vec![
                PromptSegment::control("<|user|>"),
                PromptSegment::text("hello "),
                PromptSegment::control("<|eos|>"),
                PromptSegment::text("world"),
                PromptSegment::control("<|bos|>"),
            ]
        );
        assert_eq!(segments("").unwrap(), vec![]);
        assert_eq!(
            segments("plain").unwrap(),
            vec![PromptSegment::text("plain")]
        );
    }

    #[test]
    fn flattening_strips_the_marks_and_locates_the_controls() {
        let marked = format!("{USER}hi{}", mark("<|eos|>"));
        let (text, controls) = flatten(&marked).unwrap();
        assert_eq!(text, "<|user|>hi<|eos|>");
        assert_eq!(controls, vec![0..8, 10..17]);
        assert_eq!(text, strip(&marked));
        assert_eq!(flatten("").unwrap(), (String::new(), vec![]));
        assert_eq!(flatten("plain").unwrap(), ("plain".to_owned(), vec![]));
    }

    #[test]
    fn unpaired_marks_are_an_error() {
        for malformed in [
            "\u{FDD0}<|a|>",
            "<|a|>\u{FDD1}",
            "\u{FDD0}\u{FDD1}",
            "\u{FDD0}x\u{FDD0}y\u{FDD1}",
        ] {
            assert!(segments(malformed).is_err(), "{malformed:?}");
            assert!(flatten(malformed).is_err(), "{malformed:?}");
        }
    }
}
