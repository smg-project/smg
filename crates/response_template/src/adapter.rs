//! Glue for smg's reasoning and tool parsers (feature `adapter`); not part of
//! transformers. A [`ResponseParserState`] holds the [`ResponseParser`] of one
//! generated choice, and the choice's reasoning and tool parsers share it. They
//! read the fields `transformers serve` reads: `thinking` (here also
//! `reasoning_content`), `content` and `tool_calls`. A stream reads the text of
//! their regions and each closed tool-call region after the prompt; a complete
//! output reads the parsed message. A stream's tool parser also reads the name
//! of a call once its region is open, when the template's opener captures it.
//! A region that fails to parse is content.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde_json::Value;

use crate::{
    content_parsers::{self, ContentParser},
    error::{ParseError, PyErrorKind, Taint},
    py,
    response_parser::{Event, ResponseParser},
    response_templates::ResponseTemplate,
};

const REASONING: [&str; 2] = ["thinking", "reasoning_content"];
const CONTENT: &str = "content";
const TOOL_CALLS: &str = "tool_calls";

/// Why smg's parsers do not use a template.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Unsuitable(&'static str);

/// Whether smg's parsers can use `template`: it has a reasoning or a
/// `tool_calls` field, and its reasoning and `content` values are strings.
pub fn check(template: &ResponseTemplate) -> Result<(), Unsuitable> {
    let spec = &template.0;
    let field = |name: &str| spec.fields.iter().find(|f| f.name == name);
    if REASONING.iter().all(|name| field(name).is_some()) {
        return Err(Unsuitable(
            "both a `thinking` and a `reasoning_content` field",
        ));
    }
    if field(TOOL_CALLS).is_none() && REASONING.iter().all(|name| field(name).is_none()) {
        return Err(Unsuitable(
            "no `thinking`, `reasoning_content` or `tool_calls` field",
        ));
    }
    for name in [REASONING[0], REASONING[1], CONTENT] {
        let text = field(name).is_none_or(|f| {
            f.content.parser == ContentParser::Text
                && f.transform.is_none()
                && (!f.repeats || f.join.is_some())
        });
        if !text
            || spec
                .defaults
                .get(name)
                .is_some_and(|v| !v.is_string() && !v.is_null())
        {
            return Err(Unsuitable(
                "a `thinking`, `reasoning_content` or `content` value that is not a string",
            ));
        }
    }
    Ok(())
}

/// Whether `template` reads content only inside a region it opens: output
/// that opens no region, such as output constrained to JSON from its first
/// token, is then not content. False when `content` is the field without an
/// opener.
pub fn content_needs_opener(template: &ResponseTemplate) -> bool {
    let spec = &template.0;
    !spec
        .implicit
        .and_then(|i| spec.fields.get(i))
        .is_some_and(|field| field.name == CONTENT)
}

/// A call read from a `tool_calls` value: its `function`'s `name` and
/// `arguments`, as `transformers serve` reads them.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// What the tool parser of a stream reads, in output order.
#[derive(Debug, Clone, PartialEq)]
pub enum CallItem {
    /// A call read whole when its region closed.
    Call(ToolCall),
    /// The name of the call in the open tool-call region, which the
    /// template's opener captured.
    Name(String),
    /// The arguments of the call last named, read when its region closed,
    /// before any output that followed the close.
    Arguments(Value),
}

/// Output read from the parser's events.
enum Piece {
    Reasoning(String),
    Content(String),
    Item(CallItem),
}

/// The opener group that names every call of the template's `tool_calls`
/// field: its transform is one dict whose `function.name` is `{group}`, with a
/// `function.arguments`. A region of the field that closes without an error
/// then holds one call, named with the opener's capture.
fn name_group(template: &ResponseTemplate) -> Option<String> {
    let field = template.0.fields.iter().find(|f| f.name == TOOL_CALLS)?;
    if field.transform_each || field.join.is_some() {
        return None;
    }
    let open = field.open.as_ref()?;
    let function = field.transform.as_ref()?.get("function")?;
    function.get("arguments")?;
    let group = content_parsers::placeholder(function.get("name")?.as_str()?)?;
    let named = group != CONTENT
        && !group.contains('.')
        && open.pattern.group_names().any(|name| name == group);
    named.then(|| group.to_owned())
}

/// The calls of a `tool_calls` value: one call, or a list of them.
fn read_calls(value: &Value) -> Result<Vec<ToolCall>, ParseError> {
    let mut calls = Vec::new();
    each_call(value, &mut |name, arguments| {
        calls.push(ToolCall {
            name: name.to_owned(),
            arguments: arguments.clone(),
        });
    })?;
    Ok(calls)
}

/// Visit the `function.name` and `function.arguments` of each call.
fn each_call<'v>(
    value: &'v Value,
    visit: &mut impl FnMut(&'v str, &'v Value),
) -> Result<(), ParseError> {
    if let Value::Array(items) = value {
        return items.iter().try_for_each(|item| each_call(item, visit));
    }
    let function = value.get("function");
    let name = function.and_then(|f| f.get("name")).and_then(Value::as_str);
    let arguments = function.and_then(|f| f.get("arguments"));
    let (Some(name), Some(arguments)) = (name, arguments) else {
        return Err(ParseError::Content {
            field: TOOL_CALLS.to_owned(),
            kind: PyErrorKind::Key,
            message: "a tool call needs a string function.name and function.arguments".to_owned(),
        });
    };
    visit(name, arguments);
    Ok(())
}

/// A text field's value as `transformers serve` reads it.
fn message_text(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// The reasoning, content and calls of a parsed message, as `transformers
/// serve` reads them; an error where its read fails (calls it cannot read, or
/// a value JSON cannot hold).
fn read_message(parser: &ResponseParser) -> Result<(String, String, Vec<ToolCall>), ParseError> {
    if let Some(why) = parser.taint_of(&[REASONING[0], REASONING[1], CONTENT, TOOL_CALLS]) {
        return Err(ParseError::Unrepresentable(why));
    }
    let message = parser.message();
    let reasoning = message_text(REASONING.iter().find_map(|name| message.get(*name)));
    let calls = match message.get(TOOL_CALLS).filter(|v| py::truthy(v)) {
        Some(calls) => read_calls(calls)?,
        None => Vec::new(),
    };
    Ok((reasoning, message_text(message.get(CONTENT)), calls))
}

/// The adapter's check of each region before its value is stored: a value
/// JSON cannot hold, or a `tool_calls` value whose calls cannot be read,
/// fails the region. An empty value holds no calls, as in a complete read.
fn check_region(field: &str, value: &Value, taint: Taint) -> Result<(), ParseError> {
    if let Some(why) = taint {
        return Err(ParseError::Unrepresentable(why));
    }
    if field == TOOL_CALLS && py::truthy(value) {
        each_call(value, &mut |_, _| {})?;
    }
    Ok(())
}

/// The most text a stream holds back while it could still begin a delimiter.
/// transformers holds it all and searches it again on every chunk, so a
/// stream would cost time quadratic in its length.
const HOLD_LIMIT: usize = 8 * 1024;

/// The response parser of one generated choice, shared by the choice's
/// reasoning and tool parsers (clones share it). A region that fails to parse
/// is content, as is the rest of the output after it: its text from the
/// opener on is returned as it was generated (from the start of the output
/// when the region began in the prompt), and the output after it passes
/// through unparsed. A stream holds back at most [`HOLD_LIMIT`] bytes that
/// could still begin a delimiter; past that, the text goes to the current
/// region, and only a delimiter that starts in its last half is waited for.
#[derive(Clone)]
pub struct ResponseParserState(Arc<Mutex<Inner>>);

struct Inner {
    /// `None` once the output ended, or after a region failed.
    parser: Option<ResponseParser>,
    /// Why parsing stopped, until [`ResponseParserState::take_error`].
    error: Option<ParseError>,
    continuation: bool,
    /// The reasoning parser feeds the parser; the tool parser only takes calls.
    reasoning_feeds: bool,
    /// Tool-call items the tool parser has not taken.
    calls: Vec<CallItem>,
    /// See [`name_group`].
    name_group: Option<String>,
    /// The name read for the open tool-call region.
    named: Option<String>,
    /// Output that followed the close of the named call's region, deferred
    /// to the next call so that the call's arguments come first.
    deferred: Vec<Piece>,
}

#[derive(Default)]
struct Text {
    reasoning: String,
    content: String,
}

impl ResponseParserState {
    /// The state of one output. `prompt_tail` is the prompt, decoded from the
    /// token ids the model was given, after the template's last start anchor
    /// (transformers' `prefix`, truncated), and `tools` cast tool-call
    /// arguments. With `continuation` (the prompt ends
    /// inside the assistant message), a complete output returns what was
    /// generated, as a stream does, instead of the parsed message. When the
    /// prompt fails to parse, the whole output is content.
    pub fn new(
        template: &ResponseTemplate,
        prompt_tail: &str,
        tools: &[Value],
        continuation: bool,
    ) -> Self {
        let parsed = ResponseParser::for_adapter(template, prompt_tail, tools, check_region);
        let (parser, error) = match parsed {
            Ok(parser) => (Some(parser), None),
            Err(error) => (None, Some(error)),
        };
        Self(Arc::new(Mutex::new(Inner {
            parser,
            error,
            continuation,
            reasoning_feeds: false,
            calls: Vec::new(),
            name_group: name_group(template),
            named: None,
            deferred: Vec::new(),
        })))
    }

    fn state(&self) -> MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// For the reasoning parser: feed streamed output (`None` ends it) and
    /// return the `(reasoning, content)` text it completes. Tool calls wait
    /// for the tool parser.
    pub fn reasoning(&self, output: Option<&str>) -> (String, String) {
        let mut state = self.state();
        state.reasoning_feeds = true;
        let mut text = Text::default();
        state.stream(output, false, &mut text);
        (text.reasoning, text.content)
    }

    /// For the reasoning parser: parse a complete output and return the
    /// message's `(reasoning, content)`. Tool calls wait for the tool parser.
    /// Where `transformers serve` reads the parsed message, this is its read;
    /// where that fails, a region failed: the message parsed before it is
    /// read, and the output from that region on is added to its content.
    pub fn reasoning_complete(&self, output: &str) -> (String, String) {
        let mut state = self.state();
        state.reasoning_feeds = true;
        let parser = if state.continuation {
            None
        } else {
            state.parser.take()
        };
        let Some(mut parser) = parser else {
            let mut text = Text::default();
            state.stream(Some(output), false, &mut text);
            state.stream(None, false, &mut text);
            return (text.reasoning, text.content);
        };
        let mut plain = parser.unchecked();
        let mut events = Vec::new();
        let parsed = plain
            .feed_into(output, &mut events, None)
            .and_then(|()| plain.finish_into(&mut events));
        if let Ok((reasoning, content, calls)) = parsed.and_then(|()| read_message(&plain)) {
            state.calls.extend(calls.into_iter().map(CallItem::Call));
            return (reasoning, content);
        }
        events.clear();
        let parsed = parser
            .feed_into(output, &mut events, None)
            .and_then(|()| parser.finish_into(&mut events));
        let message = parser.message();
        let reasoning = message_text(REASONING.iter().find_map(|name| message.get(*name)));
        let mut content = message_text(message.get(CONTENT));
        // A closed region's calls were checked; only a template's default can fail here.
        match message
            .get(TOOL_CALLS)
            .filter(|v| py::truthy(v))
            .map(read_calls)
        {
            Some(Ok(calls)) => state.calls.extend(calls.into_iter().map(CallItem::Call)),
            Some(Err(error)) => state.fail(error),
            None => {}
        }
        if let Err(error) = parsed {
            content.push_str(parser.unparsed());
            state.fail(error);
        }
        (reasoning, content)
    }

    /// Whether the output is in a reasoning region, with no tool call waiting
    /// for the tool parser.
    pub fn in_reasoning(&self) -> bool {
        let state = self.state();
        let field = state
            .parser
            .as_ref()
            .and_then(ResponseParser::current_field);
        state.calls.is_empty() && field.is_some_and(|field| REASONING.contains(&field))
    }

    /// For the tool parser: the content and tool calls of streamed `text`
    /// (`None` ends the output). When the reasoning parser feeds this state,
    /// `text` is its content and passes through with the calls it closed;
    /// otherwise this state is fed `text`, and reasoning is content.
    pub fn tools(&self, text: Option<&str>) -> (String, Vec<ToolCall>) {
        let mut state = self.state();
        let content = state.tool_content(text);
        // Only `tool_items` names calls.
        let calls = std::mem::take(&mut state.calls).into_iter();
        let calls = calls.filter_map(|item| match item {
            CallItem::Call(call) => Some(call),
            CallItem::Name(_) | CallItem::Arguments(_) => None,
        });
        (content, calls.collect())
    }

    /// For the tool parser of a stream: [`tools`](Self::tools) as items, and a
    /// [`Name`](CallItem::Name) when the output is in a tool-call region whose
    /// opener captured the name of its call (the template's transform is one
    /// dict whose `function.name` is a group of the open pattern, with a
    /// `function.arguments`). A regex opener that ends at the edge of the text
    /// is only committed with the next text, and the name with it. The call's
    /// [`Arguments`](CallItem::Arguments) come when its region closes, with no
    /// text: output that followed the close in the same text is deferred to
    /// the next call. A call whose region fails keeps only its name, and the
    /// region is content.
    pub fn tool_items(&self, text: Option<&str>) -> (String, Vec<CallItem>) {
        let mut state = self.state();
        let content = state.tool_content(text);
        let mut items = std::mem::take(&mut state.calls);
        if let Some(name) = state.open_call_name() {
            items.push(CallItem::Name(name.clone()));
            state.named = Some(name);
        }
        (content, items)
    }

    /// Why parsing stopped, once: the region that failed, or the prompt. The
    /// output from there on was returned as content; this is for a log.
    pub fn take_error(&self) -> Option<ParseError> {
        self.state().error.take()
    }
}

impl Inner {
    /// Keep the first reason parsing stopped.
    fn fail(&mut self, error: ParseError) {
        self.parser = None;
        self.named = None;
        self.error.get_or_insert(error);
    }

    /// The tool parser's content of `text`: `text` itself when the reasoning
    /// parser feeds the parser, else what feeding it gives.
    fn tool_content(&mut self, text: Option<&str>) -> String {
        let mut out = Text::default();
        if self.reasoning_feeds {
            out.content.push_str(text.unwrap_or_default());
        } else {
            self.stream(text, true, &mut out);
        }
        out.content
    }

    /// The name of the call in the open tool-call region, when none is read
    /// yet and no output is deferred.
    fn open_call_name(&self) -> Option<String> {
        if self.named.is_some() || !self.deferred.is_empty() {
            return None;
        }
        let group = self.name_group.as_deref()?;
        let (field, captures) = self.parser.as_ref()?.open_region()?;
        let (_, name) = captures
            .iter()
            .find(|(key, _)| key == group)
            .filter(|_| field == TOOL_CALLS)?;
        (!name.is_empty()).then(|| name.clone())
    }

    /// Feed `output` (`None` ends it) after the deferred output, and read the
    /// region events, reasoning into the content when `merge`. Without a
    /// parser, `output` passes through.
    fn stream(&mut self, output: Option<&str>, merge: bool, text: &mut Text) {
        let deferred = std::mem::take(&mut self.deferred);
        self.apply(deferred, merge, text);
        let Some(mut parser) = self.parser.take() else {
            text.content.push_str(output.unwrap_or_default());
            return;
        };
        let mut events = Vec::new();
        let parsed = match output {
            Some(output) => parser.feed_into(output, &mut events, Some(HOLD_LIMIT)),
            None => parser.finish_into(&mut events),
        };
        // The events before an error come from the regions before the one
        // that failed, whose text this does not read.
        let (mut pieces, cut) = self.read(events);
        match parsed {
            Ok(()) if output.is_some() => self.parser = Some(parser),
            Ok(()) => {}
            Err(error) => {
                debug_assert!(parser
                    .current_field()
                    .is_none_or(|field| field != CONTENT && !REASONING.contains(&field)));
                pieces.push(Piece::Content(parser.unparsed().to_owned()));
                self.fail(error);
            }
        }
        if let Some(cut) = cut.filter(|_| output.is_some()) {
            self.deferred = pieces.split_off(cut);
        }
        self.apply(pieces, merge, text);
    }

    /// The output of `events`, and where what follows the close of the named
    /// call's region starts: that close gives the call's arguments.
    fn read(&mut self, events: Vec<Event>) -> (Vec<Piece>, Option<usize>) {
        let (mut pieces, mut cut) = (Vec::new(), None);
        for event in events {
            match event {
                Event::RegionChunk { field, text, .. } if REASONING.contains(&field.as_str()) => {
                    pieces.push(Piece::Reasoning(text));
                }
                Event::RegionChunk { field, text, .. } if field == CONTENT => {
                    pieces.push(Piece::Content(text));
                }
                Event::RegionClose { field, value }
                    if field == TOOL_CALLS && py::truthy(&value) =>
                {
                    // The region check read these calls before the close.
                    let mut calls = read_calls(&value).unwrap_or_default().into_iter();
                    if let Some(name) = self.named.take() {
                        // One call, named with the opener's capture (`name_group`).
                        if let Some(call) = calls.next() {
                            debug_assert_eq!(call.name, name);
                            pieces.push(Piece::Item(CallItem::Arguments(call.arguments)));
                            cut = Some(pieces.len());
                        }
                    }
                    pieces.extend(calls.map(|call| Piece::Item(CallItem::Call(call))));
                }
                _ => {}
            }
        }
        (pieces, cut)
    }

    /// Add the text of `pieces`, reasoning into the content when `merge`, and
    /// queue their items.
    fn apply(&mut self, pieces: Vec<Piece>, merge: bool, text: &mut Text) {
        for piece in pieces {
            match piece {
                Piece::Reasoning(t) if !merge => text.reasoning.push_str(&t),
                Piece::Reasoning(t) | Piece::Content(t) => text.content.push_str(&t),
                Piece::Item(item) => self.calls.push(item),
            }
        }
    }
}
