//! Glue for smg's reasoning and tool parsers (feature `adapter`); not part of
//! transformers. A [`ResponseParserState`] holds the [`ResponseParser`] of one
//! generated choice, and the choice's reasoning and tool parsers share it. They
//! read the fields `transformers serve` reads: `thinking` (here also
//! `reasoning_content`), `content` and `tool_calls`. A stream reads the text of
//! their regions and each closed tool-call region after the prompt; a complete
//! output reads the parsed message.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde_json::Value;

use crate::{
    content_parsers::ContentParser,
    error::{ParseError, PyErrorKind},
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

/// A call read from a `tool_calls` value: its `function`'s `name` and
/// `arguments`, as `transformers serve` reads them.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Append the calls of a `tool_calls` value: one call, or a list of them.
fn read_calls(value: &Value, calls: &mut Vec<ToolCall>) -> Result<(), ParseError> {
    if let Value::Array(items) = value {
        return items.iter().try_for_each(|item| read_calls(item, calls));
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
    calls.push(ToolCall {
        name: name.to_owned(),
        arguments: arguments.clone(),
    });
    Ok(())
}

/// The response parser of one generated choice, shared by the choice's
/// reasoning and tool parsers (clones share it). After an error, text passes
/// through unparsed.
#[derive(Clone)]
pub struct ResponseParserState(Arc<Mutex<Inner>>);

struct Inner {
    /// `None` once the output ended, or after an error.
    parser: Option<ResponseParser>,
    /// What the prompt raised, for the first call.
    error: Option<ParseError>,
    continuation: bool,
    /// The reasoning parser feeds the parser; the tool parser only takes calls.
    reasoning_feeds: bool,
    /// Closed tool calls the tool parser has not taken.
    calls: Vec<ToolCall>,
}

#[derive(Default)]
struct Text {
    reasoning: String,
    content: String,
}

impl ResponseParserState {
    /// The state of one output. `prompt_tail` is the rendered prompt after the
    /// template's last start anchor (transformers' `prefix`, truncated), and
    /// `tools` cast tool-call arguments. With `continuation` (the prompt ends
    /// inside the assistant message), a complete output returns what was
    /// generated, as a stream does, instead of the parsed message.
    pub fn new(
        template: &ResponseTemplate,
        prompt_tail: &str,
        tools: &[Value],
        continuation: bool,
    ) -> Self {
        let (parser, error) = match ResponseParser::from_truncated(template, prompt_tail, tools) {
            Ok(parser) => (Some(parser), None),
            Err(error) => (None, Some(error)),
        };
        Self(Arc::new(Mutex::new(Inner {
            parser,
            error,
            continuation,
            reasoning_feeds: false,
            calls: Vec::new(),
        })))
    }

    fn state(&self) -> MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// For the reasoning parser: feed streamed output (`None` ends it) and
    /// return the `(reasoning, content)` text it completes. Tool calls wait
    /// for the tool parser.
    pub fn reasoning(&self, output: Option<&str>) -> Result<(String, String), ParseError> {
        let mut state = self.state();
        state.reasoning_feeds = true;
        let mut text = Text::default();
        state.stream(output, false, &mut text)?;
        Ok((text.reasoning, text.content))
    }

    /// For the reasoning parser: parse a complete output and return the
    /// message's `(reasoning, content)`. Tool calls wait for the tool parser.
    pub fn reasoning_complete(&self, output: &str) -> Result<(String, String), ParseError> {
        let mut state = self.state();
        state.reasoning_feeds = true;
        let parser = if state.continuation {
            None
        } else {
            state.parser.take()
        };
        let Some(mut parser) = parser else {
            let mut text = Text::default();
            state.stream(Some(output), false, &mut text)?;
            state.stream(None, false, &mut text)?;
            return Ok((text.reasoning, text.content));
        };
        let (message, _) = parser.feed(output).and_then(|_| parser.finalize())?;
        let text =
            |value: Option<&Value>| value.and_then(Value::as_str).unwrap_or_default().to_owned();
        let reasoning = text(REASONING.iter().find_map(|name| message.get(*name)));
        if let Some(calls) = message.get(TOOL_CALLS).filter(|v| py::truthy(v)) {
            read_calls(calls, &mut state.calls)?;
        }
        Ok((reasoning, text(message.get(CONTENT))))
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
    pub fn tools(&self, text: Option<&str>) -> Result<(String, Vec<ToolCall>), ParseError> {
        let mut state = self.state();
        let mut out = Text::default();
        if state.reasoning_feeds {
            out.content.push_str(text.unwrap_or_default());
        } else {
            state.stream(text, true, &mut out)?;
        }
        Ok((out.content, std::mem::take(&mut state.calls)))
    }
}

impl Inner {
    /// Feed `output` (`None` ends it) and read the region events, reasoning
    /// into the content when `merge`. Without a parser, `output` passes
    /// through.
    fn stream(
        &mut self,
        output: Option<&str>,
        merge: bool,
        text: &mut Text,
    ) -> Result<(), ParseError> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let events = match output {
            Some(output) => self.parser.as_mut().map(|parser| parser.feed(output)),
            None => self
                .parser
                .take()
                .map(|parser| parser.finalize().map(|(_, events)| events)),
        };
        let Some(events) = events else {
            text.content.push_str(output.unwrap_or_default());
            return Ok(());
        };
        let queued = self.calls.len();
        let read = events.and_then(|events| self.read(&events, merge, text));
        if read.is_err() {
            // The caller passes this output through as it is.
            self.parser = None;
            self.calls.truncate(queued);
        }
        read
    }

    fn read(&mut self, events: &[Event], merge: bool, text: &mut Text) -> Result<(), ParseError> {
        for event in events {
            match event {
                Event::RegionChunk { field, text: t, .. }
                    if REASONING.contains(&field.as_str()) =>
                {
                    if merge {
                        &mut text.content
                    } else {
                        &mut text.reasoning
                    }
                    .push_str(t);
                }
                Event::RegionChunk { field, text: t, .. } if field == CONTENT => {
                    text.content.push_str(t);
                }
                Event::RegionClose { field, value } if field == TOOL_CALLS => {
                    read_calls(value, &mut self.calls)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}
