//! The streaming parser, ported from transformers' `response_parser.py`.
//! Methods keep transformers' names (without the leading underscore) and
//! statement order, so an error surfaces at the same point and leaves the
//! same state behind.

use std::{cmp::Reverse, collections::HashMap};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    content_parsers::process_field,
    error::{ParseError, PyError, PyErrorKind, PyResult, Taint},
    py::{self, JsonError},
    pyre::{Found, Match, Memo},
    response_templates::{Delimiter, ResponseTemplate, ResponseTemplateField},
};

/// transformers: the parsed message dict; keys keep Python's insertion order.
pub type Message = Map<String, Value>;

/// transformers: the event dicts; serializes to the same keys in the same order.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "the variants are transformers' event types"
)]
pub enum Event {
    /// A region (message field) opened.
    RegionOpen { field: String },
    /// Text routed into the open region. `dirty` marks structured regions
    /// (`json`, `xml-inline`, `kv-lines`) whose value only exists on close.
    RegionChunk {
        field: String,
        text: String,
        dirty: bool,
    },
    /// The region closed with its parsed value.
    RegionClose { field: String, value: Value },
}

/// transformers: `_schema_types`, the JSON-schema types of a tool parameter.
pub(crate) fn schema_types(schema: &Value) -> PyResult<Vec<String>> {
    let Value::Object(schema) = schema else {
        return Ok(Vec::new());
    };
    let mut types: Vec<String> = match schema.get("type") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|t| t.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    };
    // `for choice in schema.get("anyOf") or []`: a dict yields its keys and a
    // string its characters, neither of which has types.
    match schema.get("anyOf") {
        Some(Value::Array(choices)) => {
            for choice in choices {
                types.extend(schema_types(choice)?);
            }
        }
        Some(Value::Bool(true) | Value::Number(_))
            if schema.get("anyOf").is_some_and(py::truthy) =>
        {
            return Err(PyError(PyErrorKind::Type, "'anyOf' is not iterable".into()));
        }
        _ => {}
    }
    if schema.get("nullable").is_some_and(py::truthy) && !types.iter().any(|t| t == "null") {
        types.push("null".to_owned());
    }
    Ok(types)
}

/// transformers: `_coerce`, cast a string argument to the first schema type
/// it fits; failed casts keep the string.
pub(crate) fn coerce(raw: &str, types: &[String], taint: &mut Taint) -> PyResult<Value> {
    for type_name in types {
        match type_name.as_str() {
            "integer" => {
                if let Some((value, t)) = py::int(raw) {
                    *taint = t.or(*taint);
                    return Ok(value);
                }
            }
            "number" => match py::float(raw) {
                Some(number) if number.is_finite() => {
                    if number.fract() == 0.0 && !raw.contains('.') {
                        let (value, t) =
                            py::int_value(number < 0.0, &format!("{:.0}", number.abs()));
                        *taint = t.or(*taint);
                        return Ok(value);
                    }
                    return Ok(py::float_value(number).0);
                }
                _ => {}
            },
            "boolean" => match py::strip(raw).to_lowercase().as_str() {
                "true" | "1" => return Ok(Value::Bool(true)),
                "false" | "0" => return Ok(Value::Bool(false)),
                _ => {}
            },
            "null" if matches!(py::strip(raw), "null" | "None") => return Ok(Value::Null),
            "object" | "array" => {
                let mut t = None;
                match py::json_loads(raw, &mut t) {
                    Ok(value @ Value::Object(_)) if type_name == "object" => {
                        *taint = t.or(*taint);
                        return Ok(value);
                    }
                    Ok(value @ Value::Array(_)) if type_name == "array" => {
                        *taint = t.or(*taint);
                        return Ok(value);
                    }
                    Ok(_) | Err(JsonError::Decode | JsonError::Digits) => {}
                    Err(JsonError::Recursion) => {
                        return Err(PyError(
                            PyErrorKind::Recursion,
                            "maximum recursion depth exceeded".into(),
                        ))
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Value::String(raw.to_owned()))
}

/// transformers: `ResponseParser`, a streaming parser with a response template.
///
/// Feed model output with [`feed`](Self::feed) and finish with
/// [`finalize`](Self::finalize); each returns the region events it produced.
/// Events from the prompt prefix are in [`initial_events`](Self::initial_events).
#[derive(Debug)]
pub struct ResponseParser {
    spec: ResponseTemplate,
    /// Tool name to the `properties` of its parameters schema.
    tool_params: HashMap<String, Map<String, Value>>,
    buffer: String,
    pos: usize,
    output: Message,
    /// Output fields holding a value JSON cannot represent, and why.
    tainted: HashMap<String, &'static str>,
    current: Option<usize>,
    captures: Vec<(String, String)>,
    body: String,
    opened: bool,
    initial_events: Vec<Event>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Open,
    Close,
}

impl ResponseParser {
    /// transformers: `ResponseParser(template, prefix=prefix, tools=tools)`.
    /// `prefix` is the chat prompt sent before generation; pass `""` to opt out.
    pub fn new(
        template: &ResponseTemplate,
        prefix: &str,
        tools: &[Value],
    ) -> Result<Self, ParseError> {
        Self::from_truncated(template, template.truncate_past_last_anchor(prefix), tools)
    }

    /// [`new`](Self::new) with the prefix already truncated past its last
    /// start anchor; truncating it again could cut more.
    pub(crate) fn from_truncated(
        template: &ResponseTemplate,
        truncated: &str,
        tools: &[Value],
    ) -> Result<Self, ParseError> {
        let mut tool_params = HashMap::new();
        for tool in tools {
            let function = match tool {
                Value::Object(map) => map.get("function").unwrap_or(tool),
                _ => continue,
            };
            let Some(Value::String(name)) = function.get("name") else {
                continue;
            };
            let properties = function
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(Value::as_object);
            tool_params.insert(name.clone(), properties.cloned().unwrap_or_default());
        }
        let mut parser = Self {
            spec: template.clone(),
            tool_params,
            buffer: String::new(),
            pos: 0,
            output: template.0.defaults.clone(),
            tainted: HashMap::new(),
            current: template.0.implicit,
            captures: Vec::new(),
            body: String::new(),
            opened: false,
            initial_events: Vec::new(),
        };
        if !truncated.is_empty() {
            let mut events = Vec::new();
            parser.consume_prefix(truncated, &mut events)?;
            parser.initial_events = events;
        }
        Ok(parser)
    }

    /// Events produced while consuming the prefix, to replay before the output.
    pub fn initial_events(&self) -> &[Event] {
        &self.initial_events
    }

    /// The field of the current region: the open explicit region, else the
    /// implicit field.
    #[cfg(feature = "adapter")]
    pub(crate) fn current_field(&self) -> Option<&str> {
        self.current.map(|i| self.spec.0.fields[i].name.as_str())
    }

    /// transformers: `_consume_prefix`, from the truncation on.
    fn consume_prefix(
        &mut self,
        truncated: &str,
        events: &mut Vec<Event>,
    ) -> Result<(), ParseError> {
        truncated.clone_into(&mut self.buffer);
        let mut taint = None;
        self.process(events, false, &mut taint)?;
        taint.map_or(Ok(()), |t| Err(ParseError::Unrepresentable(t)))
    }

    /// transformers: `feed`. Feed more model output; returns the events it produced.
    pub fn feed(&mut self, text: &str) -> Result<Vec<Event>, ParseError> {
        self.buffer.push_str(text);
        let mut events = Vec::new();
        let mut taint = None;
        self.process(&mut events, false, &mut taint)?;
        taint.map_or(Ok(events), |t| Err(ParseError::Unrepresentable(t)))
    }

    /// transformers: `finalize`. Close the stream; returns the message and the
    /// final events. Consumes the parser: feeding or finalizing again does not compile.
    pub fn finalize(mut self) -> Result<(Message, Vec<Event>), ParseError> {
        let mut events = Vec::new();
        let mut taint = None;
        self.process(&mut events, true, &mut taint)?;
        let missing: Vec<String> = self
            .spec
            .0
            .fields
            .iter()
            .filter(|f| !f.optional && !self.output.contains_key(&f.name))
            .map(|f| f.name.clone())
            .collect();
        if !missing.is_empty() {
            return Err(ParseError::MissingRequired(missing));
        }
        let defaults = &self.spec.0.defaults;
        self.output
            .retain(|k, v| defaults.contains_key(k) || !is_empty(v));
        if let Some(t) = taint.or_else(|| {
            let mut kept = self
                .tainted
                .iter()
                .filter(|(k, _)| self.output.contains_key(*k));
            kept.next().map(|(_, t)| *t)
        }) {
            return Err(ParseError::Unrepresentable(t));
        }
        Ok((self.output, events))
    }

    /// transformers: `_process`.
    fn process(
        &mut self,
        events: &mut Vec<Event>,
        eos: bool,
        taint: &mut Taint,
    ) -> Result<(), ParseError> {
        // The buffer does not change here, so searches of it are remembered.
        let mut memos = vec![[Memo::default(), Memo::default()]; self.spec.0.fields.len()];
        loop {
            let watch = self.watchlist();
            let (best, hold_start) = self.scan(&watch, eos, &mut memos);
            if let Some((kind, field, m)) = best {
                if m.start > self.pos {
                    let text = self.buffer[self.pos..m.start].to_owned();
                    self.accumulate(events, text);
                }
                self.pos = m.end;
                if kind == Kind::Open {
                    self.close_current(events, taint)?;
                    self.open_explicit(events, field, &m);
                } else {
                    let had_content = self.opened;
                    self.close_current(events, taint)?;
                    // A zero-width close on an empty region would fire again; stop.
                    if !had_content && m.start == m.end {
                        break;
                    }
                }
                continue;
            }
            if eos {
                if self.pos < self.buffer.len() {
                    let text = self.buffer[self.pos..].to_owned();
                    self.accumulate(events, text);
                    self.pos = self.buffer.len();
                }
                self.close_current(events, taint)?;
                break;
            }
            if hold_start > self.pos {
                let text = self.buffer[self.pos..hold_start].to_owned();
                self.accumulate(events, text);
                self.pos = hold_start;
            }
            break;
        }
        Ok(())
    }

    /// transformers: `_watchlist`.
    fn watchlist(&self) -> Vec<(Kind, usize)> {
        let fields = &self.spec.0.fields;
        if let Some(current) = self.current.filter(|&c| Some(c) != self.spec.0.implicit) {
            return match fields[current].close {
                Some(_) => vec![(Kind::Close, current)],
                None => Vec::new(),
            };
        }
        let mut watch: Vec<(Kind, usize)> = (0..fields.len())
            .filter(|&i| fields[i].open.is_some())
            .map(|i| (Kind::Open, i))
            .collect();
        if let Some(implicit) = self.spec.0.implicit.filter(|&i| fields[i].close.is_some()) {
            watch.push((Kind::Close, implicit));
        }
        watch
    }

    fn delimiter(&self, kind: Kind, field: usize) -> Option<&Delimiter> {
        let field = &self.spec.0.fields[field];
        match kind {
            Kind::Open => field.open.as_ref(),
            Kind::Close => field.close.as_ref(),
        }
    }

    /// transformers: `_scan`. The earliest delimiter that can be committed now
    /// (longest on ties, opens before closes, then by field name), and the
    /// start of the earliest delimiter still pending.
    fn scan(
        &self,
        watch: &[(Kind, usize)],
        eos: bool,
        memos: &mut [[Memo; 2]],
    ) -> (Option<(Kind, usize, Match)>, usize) {
        let mut best: Option<(Kind, usize, Match)> = None;
        let mut hold_start = self.buffer.len();
        for &(kind, field) in watch {
            let Some(delimiter) = self.delimiter(kind, field) else {
                continue;
            };
            let memo = &mut memos[field][usize::from(kind == Kind::Close)];
            let found = if eos {
                delimiter
                    .pattern
                    .search(&self.buffer, self.pos, memo)
                    .map(Found::Complete)
            } else {
                delimiter
                    .pattern
                    .search_partial(&self.buffer, self.pos, memo)
            };
            let m = match found {
                None => continue,
                Some(Found::Partial(start)) => {
                    hold_start = hold_start.min(start);
                    continue;
                }
                Some(Found::Complete(m)) if !eos && self.can_grow(delimiter, &m) => {
                    hold_start = hold_start.min(m.start);
                    continue;
                }
                Some(Found::Complete(m)) => m,
            };
            let key = |kind: Kind, field: usize, m: &Match| {
                (
                    m.start,
                    Reverse(m.end - m.start),
                    kind == Kind::Close,
                    &self.spec.0.fields[field].name,
                )
            };
            if best
                .as_ref()
                .is_none_or(|(k, f, b)| key(kind, field, &m) < key(*k, *f, b))
            {
                best = Some((kind, field, m));
            }
        }
        if best.as_ref().is_some_and(|(_, _, m)| m.start >= hold_start) {
            best = None;
        }
        (best, hold_start)
    }

    /// transformers: `_can_grow`. Whether a complete match at the buffer edge
    /// could still change with more input.
    fn can_grow(&self, delimiter: &Delimiter, m: &Match) -> bool {
        if m.end != self.buffer.len() {
            return false;
        }
        m.start == m.end || !delimiter.literals || delimiter.can_extend
    }

    /// transformers: `_accumulate`. Route text into the current region; the
    /// null sink (no implicit field, no open region) drops it.
    fn accumulate(&mut self, events: &mut Vec<Event>, text: String) {
        let Some(current) = self.current.filter(|_| !text.is_empty()) else {
            return;
        };
        let field = &self.spec.0.fields[current];
        if !self.opened {
            events.push(Event::RegionOpen {
                field: field.name.clone(),
            });
            self.opened = true;
        }
        self.body.push_str(&text);
        events.push(Event::RegionChunk {
            field: field.name.clone(),
            text,
            dirty: !field.content.parser.streamable(),
        });
    }

    /// transformers: `_open_explicit`.
    fn open_explicit(&mut self, events: &mut Vec<Event>, field: usize, m: &Match) {
        let spec = self.spec.clone();
        let f = &spec.0.fields[field];
        self.current = Some(field);
        self.captures = match &f.open {
            Some(open) => open
                .pattern
                .groupdict(&self.buffer, m)
                .into_iter()
                .filter_map(|(name, value)| Some((name.to_owned(), value?.to_owned())))
                .collect(),
            None => Vec::new(),
        };
        self.body.clear();
        self.opened = true;
        events.push(Event::RegionOpen {
            field: f.name.clone(),
        });
    }

    /// transformers: `_close_current`. Parse and store the current region, then
    /// return to the implicit region. An error leaves the region as it was.
    fn close_current(
        &mut self,
        events: &mut Vec<Event>,
        taint: &mut Taint,
    ) -> Result<(), ParseError> {
        let Some(current) = self.current.filter(|_| self.opened) else {
            self.reset_to_implicit();
            return Ok(());
        };
        let spec = self.spec.clone();
        let field = &spec.0.fields[current];
        let fail = |e: PyError| e.in_field(&field.name);
        let (value, value_taint) =
            py::unless_unreached(|| self.field_value(field)).map_err(fail)?;
        let name = &field.name;
        if let Some(join) = &field.join {
            let Value::String(part) = &value else {
                return Err(fail(PyError::value(format!(
                    "Field '{name}': 'join' requires each match to parse to a string, got {}.",
                    py::type_name(&value)
                ))));
            };
            let joined = match self.output.get(name) {
                None | Some(Value::Null) => part.clone(),
                Some(Value::String(previous)) => format!("{previous}{join}{part}"),
                Some(previous) => {
                    return Err(fail(PyError(
                        PyErrorKind::Type,
                        format!(
                            "can only concatenate str (not \"{}\")",
                            py::type_name(previous)
                        ),
                    )))
                }
            };
            self.output.insert(name.clone(), Value::String(joined));
        } else if field.repeats {
            match self.output.get_mut(name) {
                None => {
                    self.output
                        .insert(name.clone(), Value::Array(vec![value.clone()]));
                }
                Some(Value::Array(items)) => items.push(value.clone()),
                Some(other) => {
                    return Err(fail(PyError(
                        PyErrorKind::Attribute,
                        format!(
                            "'{}' object has no attribute 'append'",
                            py::type_name(other)
                        ),
                    )))
                }
            }
        } else {
            self.output.insert(name.clone(), value.clone());
            self.tainted.remove(name);
        }
        if let Some(t) = value_taint {
            self.tainted.insert(name.clone(), t);
            *taint = Some(t);
        }
        events.push(Event::RegionClose {
            field: name.clone(),
            value,
        });
        self.reset_to_implicit();
        Ok(())
    }

    /// The value of the current region: `process_field`, then
    /// `_coerce_tool_calls` when tools are given.
    fn field_value(&self, field: &ResponseTemplateField) -> PyResult<(Value, Taint)> {
        let mut taint = None;
        let mut value = process_field(&self.body, field, &self.captures, &mut taint)?;
        if !self.tool_params.is_empty() {
            self.coerce_tool_calls(&mut value, &mut taint)?;
        }
        Ok((value, taint))
    }

    /// transformers: `_coerce_tool_calls`. Casts string arguments of a known
    /// tool to their schema types, in place.
    pub(crate) fn coerce_tool_calls(&self, value: &mut Value, taint: &mut Taint) -> PyResult<()> {
        if let Value::Array(items) = value {
            return items
                .iter_mut()
                .try_for_each(|item| self.coerce_tool_calls(item, taint));
        }
        let Some(Value::Object(function)) = value.get_mut("function") else {
            return Ok(());
        };
        let Some(Value::String(name)) = function.get("name") else {
            return Ok(());
        };
        let Some(properties) = self.tool_params.get(name).filter(|p| !p.is_empty()) else {
            return Ok(());
        };
        let Some(Value::Object(arguments)) = function.get_mut("arguments") else {
            return Ok(());
        };
        for (key, argument) in arguments.iter_mut() {
            let Some(schema) = properties.get(key) else {
                continue;
            };
            let types = schema_types(schema)?;
            if types.is_empty() {
                continue;
            }
            match argument {
                Value::String(s) => *argument = coerce(s, &types, taint)?,
                Value::Array(items) => {
                    for item in items {
                        if let Value::String(s) = item {
                            *item = coerce(s, &types, taint)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// transformers: `_reset_to_implicit`.
    fn reset_to_implicit(&mut self) {
        self.current = self.spec.0.implicit;
        self.captures.clear();
        self.body.clear();
        self.opened = false;
    }
}

/// `_is_empty` in `finalize`: `None`, or an empty list, dict or str.
fn is_empty(value: &Value) -> bool {
    matches!(value, Value::Null)
        || matches!(value, Value::String(_) | Value::Array(_) | Value::Object(_))
            && !py::truthy(value)
}

/// transformers: `parse_response`. Parse a complete generation.
pub fn parse_response(
    text: &str,
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
) -> Result<Message, ParseError> {
    let mut parser = ResponseParser::new(template, prefix, tools)?;
    parser.feed(text)?;
    parser.finalize().map(|(message, _)| message)
}
