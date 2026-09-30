//! Content parsers and transforms, ported from transformers'
//! `content_parsers.py`. `content_args` are checked when the template loads:
//! shapes transformers would only fail on while parsing, or read through
//! Python's dynamic typing, are refused as `Unsupported::ContentArgs`.

use serde_json::{Map, Value};

use crate::{
    error::{LoadError, PyError, PyErrorKind, PyResult, Taint, Unsupported},
    py::{self, JsonError},
    pyre::{self, Pattern, Role},
    response_templates::ResponseTemplateField,
};

/// transformers: the keys of `CONTENT_PARSERS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContentParser {
    Text,
    Int,
    Float,
    Bool,
    Json,
    XmlInline,
    KvLines,
}

impl ContentParser {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "text" => Self::Text,
            "int" => Self::Int,
            "float" => Self::Float,
            "bool" => Self::Bool,
            "json" => Self::Json,
            "xml-inline" => Self::XmlInline,
            "kv-lines" => Self::KvLines,
            _ => return None,
        })
    }

    /// transformers: `STREAMABLE_PARSERS`; chunks of other regions are `dirty`.
    pub(crate) fn streamable(self) -> bool {
        matches!(self, Self::Text | Self::Int | Self::Float | Self::Bool)
    }
}

/// A content parser with the `content_args` it reads.
#[derive(Debug)]
pub(crate) struct Content {
    pub(crate) parser: ContentParser,
    strip: bool,
    string_delims: Vec<(String, String)>,
    unquoted_keys: bool,
    allow_non_json: bool,
    /// `None`: transformers raises `ValueError` on every close.
    tag_pattern: Option<Pattern>,
    value_parser: Option<Box<Content>>,
    merge_duplicates: bool,
    line_sep: String,
    kv_sep: String,
}

impl Content {
    /// Check the `content_args` the parser reads (`None` when absent).
    pub(crate) fn new(parser: ContentParser, args: Option<&Value>) -> Result<Self, Unsupported> {
        let empty = Map::new();
        let args = match args {
            None => &empty,
            Some(Value::Object(args)) => args,
            Some(_) => return Err(Unsupported::ContentArgs("not a dict")),
        };
        let flag = |key: &str, default: bool| args.get(key).map_or(default, py::truthy);
        let separator = |key: &'static str, default: &str| match args.get(key) {
            None => Ok(default.to_owned()),
            Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
            Some(_) => Err(Unsupported::ContentArgs(key)),
        };
        let mut content = Self {
            parser,
            strip: flag("strip", true),
            string_delims: Vec::new(),
            unquoted_keys: false,
            allow_non_json: false,
            tag_pattern: None,
            value_parser: None,
            merge_duplicates: false,
            line_sep: String::new(),
            kv_sep: String::new(),
        };
        match parser {
            ContentParser::Json => {
                content.unquoted_keys = flag("unquoted_keys", false);
                content.allow_non_json = flag("allow_non_json", false);
                if let Some(delims) = args.get("string_delims") {
                    content.string_delims =
                        string_delims(delims).ok_or(Unsupported::ContentArgs("string_delims"))?;
                }
            }
            ContentParser::XmlInline => {
                content.merge_duplicates = flag("merge_duplicates", false);
                content.tag_pattern = match args.get("tag_pattern") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(source)) => Some(tag_pattern(source)?),
                    Some(_) => return Err(Unsupported::ContentArgs("tag_pattern")),
                };
                content.value_parser = value_parser(args.get("value_parser"))?;
            }
            ContentParser::KvLines => {
                content.line_sep = separator("line_sep", "\n")?;
                content.kv_sep = separator("kv_sep", ":")?;
                content.value_parser = value_parser(args.get("value_parser"))?;
            }
            _ => {}
        }
        Ok(content)
    }

    /// transformers: `parse_content`.
    pub(crate) fn parse(&self, text: &str, taint: &mut Taint) -> PyResult<Value> {
        let tainted = |(value, t): (Value, Taint), taint: &mut Taint| {
            if t.is_some() {
                *taint = t;
            }
            value
        };
        match self.parser {
            ContentParser::Text => Ok(Value::String(self.text(text).to_owned())),
            ContentParser::Int => py::int(self.text(text))
                .map(|v| tainted(v, taint))
                .ok_or_else(|| PyError::value("invalid literal for int() with base 10")),
            ContentParser::Float => py::float(self.text(text))
                .map(|f| tainted(py::float_value(f), taint))
                .ok_or_else(|| PyError::value("could not convert string to float")),
            ContentParser::Bool => Ok(Value::Bool(matches!(
                self.text(text).to_lowercase().as_str(),
                "true" | "1"
            ))),
            ContentParser::Json => self.json(text, taint),
            ContentParser::XmlInline => self.xml_inline(text, taint),
            ContentParser::KvLines => self.kv_lines(text, taint),
        }
    }

    /// transformers: `_text`.
    fn text<'a>(&self, text: &'a str) -> &'a str {
        if self.strip {
            py::strip(text)
        } else {
            text
        }
    }

    /// transformers: `_json`. `string_delims` spans are cut out first
    /// (`re.sub` of `open(.*?)close`), bare keys are quoted, and the spans
    /// come back as JSON strings.
    fn json(&self, text: &str, taint: &mut Taint) -> PyResult<Value> {
        if !self.string_delims.is_empty() && text.contains(['\x01', '\x02']) {
            return Err(PyError::value(
                "json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.",
            ));
        }
        let mut working = text.to_owned();
        let mut captured = Vec::new();
        for (open, close) in &self.string_delims {
            working = cut_delimited(&working, open, close, &mut captured);
        }
        if self.unquoted_keys {
            working = quote_keys(&working);
        }
        for (i, span) in captured.iter().enumerate() {
            let dumped = serde_json::to_string(span).unwrap_or_default();
            working = working.replace(&format!("\x01{i}\x02"), &dumped);
        }
        match py::json_loads(&working, taint) {
            Ok(value) => Ok(value),
            Err(JsonError::Decode) if self.allow_non_json => {
                Ok(Value::String(self.text(text).to_owned()))
            }
            Err(JsonError::Decode) => Err(PyError::value(format!(
                "json parser could not parse region as JSON.\nContent: {text:?}"
            ))),
            Err(JsonError::Digits) => Err(PyError::value(
                "Exceeds the limit (4300 digits) for integer string conversion",
            )),
            Err(JsonError::Recursion) => Err(PyError(
                PyErrorKind::Recursion,
                "maximum recursion depth exceeded while decoding JSON".into(),
            )),
        }
    }

    /// transformers: `_xml_inline`.
    fn xml_inline(&self, text: &str, taint: &mut Taint) -> PyResult<Value> {
        let Some(pattern) = &self.tag_pattern else {
            return Err(PyError::value(
                "xml-inline: 'tag_pattern' content_arg is required",
            ));
        };
        let mut out = Map::new();
        for m in pattern.finditer(text) {
            let groups = pattern.groupdict(text, &m);
            let group = |name: &str| {
                groups
                    .iter()
                    .find(|(n, _)| *n == name)
                    .and_then(|(_, v)| *v)
            };
            let Some(key) = group("key") else {
                return Err(PyError::value(
                    "xml-inline: tag_pattern must have a named group 'key'",
                ));
            };
            let value = sub_parse(
                group("value").unwrap_or(""),
                self.value_parser.as_deref(),
                taint,
            )?;
            match out.get_mut(key) {
                Some(Value::Array(items)) if self.merge_duplicates => items.push(value),
                Some(previous) if self.merge_duplicates => {
                    *previous = Value::Array(vec![previous.take(), value]);
                }
                _ => {
                    out.insert(key.to_owned(), value);
                }
            }
        }
        Ok(Value::Object(out))
    }

    /// transformers: `_kv_lines`.
    fn kv_lines(&self, text: &str, taint: &mut Taint) -> PyResult<Value> {
        let mut out = Map::new();
        for line in text.split(self.line_sep.as_str()) {
            let line = self.text(line);
            let Some((key, value)) = line.split_once(self.kv_sep.as_str()) else {
                continue;
            };
            let value = sub_parse(self.text(value), self.value_parser.as_deref(), taint)?;
            out.insert(self.text(key).to_owned(), value);
        }
        Ok(Value::Object(out))
    }
}

/// transformers: `_sub_parse`.
fn sub_parse(raw: &str, value_parser: Option<&Content>, taint: &mut Taint) -> PyResult<Value> {
    match value_parser {
        None => Ok(Value::String(raw.to_owned())),
        Some(content) => content.parse(raw, taint),
    }
}

/// `string_delims` as `(open, close)` pairs of non-empty strings.
fn string_delims(value: &Value) -> Option<Vec<(String, String)>> {
    value
        .as_array()?
        .iter()
        .map(|pair| match pair.as_array()?.as_slice() {
            [Value::String(open), Value::String(close)]
                if !open.is_empty() && !close.is_empty() =>
            {
                Some((open.clone(), close.clone()))
            }
            _ => None,
        })
        .collect()
}

/// A tag pattern whose `key` and `value` groups every match sets.
fn tag_pattern(source: &str) -> Result<Pattern, Unsupported> {
    let pattern = Pattern::new(source, Role::Finditer)?;
    if ["key", "value"]
        .iter()
        .any(|g| pattern.always_sets(g) == Some(false))
    {
        return Err(Unsupported::ContentArgs(
            "tag_pattern group that can be unset",
        ));
    }
    Ok(pattern)
}

/// `value_parser`: a dict with a known `name` and dict `args`.
fn value_parser(value: Option<&Value>) -> Result<Option<Box<Content>>, Unsupported> {
    let spec = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(spec)) => spec,
        Some(_) => return Err(Unsupported::ContentArgs("value_parser")),
    };
    let parser = match spec.get("name") {
        None => ContentParser::Text,
        Some(Value::String(name)) => {
            ContentParser::from_name(name).ok_or(Unsupported::ContentArgs("value_parser"))?
        }
        Some(_) => return Err(Unsupported::ContentArgs("value_parser")),
    };
    Content::new(parser, spec.get("args")).map(|c| Some(Box::new(c)))
}

/// `re.sub(re.escape(open) + "(.*?)" + re.escape(close), ...)`: each span
/// becomes the sentinel `\x01<index>\x02` and its body is kept in `captured`.
fn cut_delimited(text: &str, open: &str, close: &str, captured: &mut Vec<String>) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find(open) {
        let body = at + open.len();
        let Some(len) = rest[body..].find(close) else {
            break;
        };
        out.push_str(&rest[..at]);
        out.push_str(&format!("\x01{}\x02", captured.len()));
        captured.push(rest[body..body + len].to_owned());
        rest = &rest[body + len + close.len()..];
    }
    out.push_str(rest);
    out
}

/// `re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)`.
fn quote_keys(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev = None;
    let mut at = 0;
    while let Some(c) = text[at..].chars().next() {
        if matches!(prev, Some('{' | ',')) {
            let run = text[at..]
                .find(|c: char| !pyre::is_word(c))
                .unwrap_or(text.len() - at);
            if run > 0 && text[at + run..].starts_with(':') {
                out.push('"');
                out.push_str(&text[at..at + run]);
                out.push_str("\":");
                at += run + 1;
                prev = Some(':');
                continue;
            }
        }
        out.push(c);
        prev = Some(c);
        at += c.len_utf8();
    }
    out
}

/// `_PLACEHOLDER.fullmatch(s)` for `\{(\w+(?:\.\w+)*)\}`: the dotted path.
fn placeholder(s: &str) -> Option<&str> {
    let path = s.strip_prefix('{')?.strip_suffix('}')?;
    is_path(path).then_some(path)
}

fn is_path(path: &str) -> bool {
    path.split('.')
        .all(|key| !key.is_empty() && key.chars().all(pyre::is_word))
}

/// `_PLACEHOLDER.search(s)`.
fn has_placeholder(s: &str) -> bool {
    s.match_indices('{').any(|(at, _)| {
        let rest = &s[at + 1..];
        rest.find('}').is_some_and(|end| is_path(&rest[..end]))
    })
}

/// transformers: `validate_transform_strings`.
pub(crate) fn validate_transform_strings(scope: &str, transform: &Value) -> Result<(), LoadError> {
    match transform {
        Value::Object(map) => map
            .values()
            .try_for_each(|v| validate_transform_strings(scope, v)),
        Value::Array(items) => items
            .iter()
            .try_for_each(|v| validate_transform_strings(scope, v)),
        Value::String(s) if has_placeholder(s) && placeholder(s).is_none() => {
            Err(LoadError::Invalid {
                scope: scope.to_owned(),
                message: format!(
                    "transform string {s:?} mixes a {{placeholder}} with literal text"
                ),
            })
        }
        _ => Ok(()),
    }
}

/// Whether `transform` would put one value that transformers may later
/// change in place (tool-call arguments) at two paths: two placeholders whose
/// paths are equal or one extends the other. Captured groups are strings, so
/// only paths into parsed content count.
pub(crate) fn shares_values(transform: &Value, each: bool, parser: ContentParser) -> bool {
    fn paths<'a>(t: &'a Value, out: &mut Vec<Vec<&'a str>>) {
        match t {
            Value::Object(map) => map.values().for_each(|v| paths(v, out)),
            Value::Array(items) => items.iter().for_each(|v| paths(v, out)),
            Value::String(s) => out.extend(placeholder(s).map(|p| p.split('.').collect())),
            _ => {}
        }
    }
    if !each && parser.streamable() {
        return false;
    }
    let mut all = Vec::new();
    paths(transform, &mut all);
    all.retain(|p| each || p[0] == "content");
    all.iter().enumerate().any(|(i, a)| {
        all[i + 1..]
            .iter()
            .any(|b| a.starts_with(b) || b.starts_with(a))
    })
}

/// transformers: `_apply_transform`.
fn apply_transform(transform: &Value, scope: &Map<String, Value>) -> PyResult<Value> {
    match transform {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| Ok((k.clone(), apply_transform(v, scope)?)))
            .collect::<PyResult<Map<_, _>>>()
            .map(Value::Object),
        Value::Array(items) => items
            .iter()
            .map(|v| apply_transform(v, scope))
            .collect::<PyResult<_>>()
            .map(Value::Array),
        Value::String(s) => {
            let Some(path) = placeholder(s) else {
                return Ok(transform.clone());
            };
            let mut keys = path.split('.');
            let root = keys.next().unwrap_or_default();
            let mut value = scope.get(root).ok_or_else(|| {
                PyError(
                    PyErrorKind::Key,
                    format!("transform placeholder '{{{path}}}' is not defined"),
                )
            })?;
            for key in keys {
                let Value::Object(map) = value else {
                    return Err(PyError::value(format!(
                        "transform placeholder '{{{path}}}' cannot index into {} at '{key}'",
                        py::type_name(value)
                    )));
                };
                value = map.get(key).ok_or_else(|| {
                    PyError::value(format!(
                        "transform placeholder '{{{path}}}' is missing key '{key}'"
                    ))
                })?;
            }
            Ok(value.clone())
        }
        _ => Ok(transform.clone()),
    }
}

/// transformers: `process_field`.
pub(crate) fn process_field(
    body: &str,
    field: &ResponseTemplateField,
    captures: &[(String, String)],
    taint: &mut Taint,
) -> PyResult<Value> {
    let value = field.content.parse(body, taint)?;
    let Some(transform) = &field.transform else {
        return Ok(value);
    };
    let mut scope: Map<String, Value> = captures
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    if !field.transform_each {
        scope.insert("content".to_owned(), value);
        return apply_transform(transform, &scope);
    }
    let Value::Array(items) = value else {
        return Err(PyError::value(format!(
            "Field '{}': transform_each requires the parsed content to be a list, got {}.",
            field.name,
            py::type_name(&value)
        )));
    };
    items
        .into_iter()
        .map(|item| {
            let Value::Object(item) = item else {
                return Err(PyError::value(format!(
                    "Field '{}': transform_each requires each list element to be a dict, got {}.",
                    field.name,
                    py::type_name(&item)
                )));
            };
            let mut scope = scope.clone();
            scope.extend(item);
            apply_transform(transform, &scope)
        })
        .collect::<PyResult<_>>()
        .map(Value::Array)
}
