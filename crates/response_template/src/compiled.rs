//! Validation of the supported template subset.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use regex::{Captures, Regex, RegexBuilder};
use regex_syntax::hir::{Hir, HirKind};
use serde_json::{json, Value};

use crate::{
    error::{invalid, TemplateError},
    scan::{Opener, Openers},
    schema::{ContentArgs, FieldTemplate, ResponseTemplate},
};

/// A template field, in the sorted order of the template's field names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Field {
    Content,
    Thinking,
    ToolCalls,
}

impl Field {
    /// Every field, in scan tie-break order.
    pub const ALL: [Field; 3] = [Field::Content, Field::Thinking, Field::ToolCalls];

    fn name(self) -> &'static str {
        match self {
            Field::Content => "content",
            Field::Thinking => "thinking",
            Field::ToolCalls => "tool_calls",
        }
    }
}

/// The argument tag of the `tool_calls` field.
#[derive(Debug)]
pub struct Tag {
    /// Matches one argument, with named `key` and `value` captures.
    pub regex: Regex,
    /// The literal every tag ends with, after its value.
    pub close: String,
    /// Trim surrounding whitespace from each value.
    pub strip: bool,
}

/// A validated response template.
///
/// The supported subset: `thinking` and `content` are non-repeating `text`
/// fields; `tool_calls` is a repeating `xml-inline` field whose opener
/// captures `name`, whose tag captures `key` and `value` and ends with a
/// literal, whose values use the `text` parser, and whose transform is the
/// OpenAI function-call shape. Closes are literals. Anything else is an error.
/// In every pattern `.` also matches a newline.
#[derive(Debug)]
pub struct CompiledTemplate {
    parser_name: String,
    openers: Openers,
    closes: [Vec<String>; 3],
    tag: Tag,
}

impl CompiledTemplate {
    /// Validate and compile a raw `response_template` value.
    pub fn from_value(value: &Value) -> Result<Self, TemplateError> {
        let template: ResponseTemplate = serde_json::from_value(value.clone())
            .map_err(|error| invalid("response_template", error.to_string()))?;
        if template.fields.len() != Field::ALL.len()
            || Field::ALL
                .iter()
                .any(|field| !template.fields.contains_key(field.name()))
        {
            return Err(invalid(
                "fields",
                "exactly thinking, content and tool_calls are required",
            ));
        }
        compile("start_anchor_pattern", &template.start_anchor_pattern)?;

        let mut regexes = Vec::with_capacity(Field::ALL.len());
        let mut closes: [Vec<String>; 3] = Default::default();
        let mut tag = None;
        for field in Field::ALL {
            let name = field.name();
            let spec = &template.fields[name];
            let args = check_capability(field, spec)?;
            let regex = compile(name, &spec.open_pattern)?;
            if field == Field::ToolCalls && regex.capture_names().all(|n| n != Some("name")) {
                return Err(invalid(name, "open_pattern requires a `name` capture"));
            }
            regexes.push(regex);
            let close = spec.close.as_slice();
            if close.is_empty() || close.iter().any(String::is_empty) {
                return Err(invalid(name, "close delimiters must be non-empty"));
            }
            closes[field as usize] = close.to_vec();
            if let Some(args) = args {
                tag = Some(compile_tag(args)?);
            }
        }
        let tag = tag.ok_or_else(|| invalid("tool_calls", "content_args are required"))?;
        let patterns: Vec<&str> = Field::ALL
            .iter()
            .map(|field| template.fields[field.name()].open_pattern.as_str())
            .collect();
        let openers = Openers::new(&patterns, regexes)
            .map_err(|error| invalid("open_pattern", format!("cannot build matcher: {error}")))?;
        Ok(Self {
            parser_name: format!("response_template_{:016x}", fingerprint(value)),
            openers,
            closes,
            tag,
        })
    }

    /// Parser registry name. It is derived from the template contents, so
    /// equal templates share it and a changed template gets a new one.
    pub fn parser_name(&self) -> &str {
        &self.parser_name
    }

    /// The earliest opener of one of `fields` (in [`Field::ALL`] order) at
    /// or after `from`; `hay[..from]` is look-behind context only. With
    /// `eof`, an opener at the end of `hay` is complete if it matches.
    pub fn scan(&self, hay: &str, from: usize, fields: &[Field], eof: bool) -> Option<Opener> {
        self.openers.scan(hay, from, fields, eof)
    }

    /// The captures of `field`'s opener that starts at `start`.
    pub fn opener_captures<'h>(
        &self,
        field: Field,
        hay: &'h str,
        start: usize,
    ) -> Option<Captures<'h>> {
        self.openers.captures(field, hay, start)
    }

    /// The literal closes of `field`.
    pub fn closes(&self, field: Field) -> &[String] {
        &self.closes[field as usize]
    }

    /// The argument tag of `tool_calls`.
    pub fn tag(&self) -> &Tag {
        &self.tag
    }
}

fn check_capability(
    field: Field,
    spec: &FieldTemplate,
) -> Result<Option<&ContentArgs>, TemplateError> {
    let supported = match field {
        Field::Thinking | Field::Content => {
            spec.content == "text"
                && spec.content_args.is_none()
                && !spec.repeats
                && spec.transform.is_none()
        }
        Field::ToolCalls => {
            spec.content == "xml-inline"
                && spec.repeats
                && spec
                    .transform
                    .as_ref()
                    .is_some_and(|transform| is_function_transform(&transform.0))
        }
    };
    if supported {
        Ok(spec.content_args.as_ref())
    } else {
        Err(invalid(field.name(), "unsupported field capabilities"))
    }
}

fn is_function_transform(value: &Value) -> bool {
    let call = json!({"name": "{name}", "arguments": "{content}"});
    *value == call || *value == json!({"type": "function", "function": call})
}

/// Template patterns let `.` match a newline, so an argument value can span
/// lines.
fn compile(field: &str, pattern: &str) -> Result<Regex, TemplateError> {
    let regex = RegexBuilder::new(pattern)
        .dot_matches_new_line(true)
        .build()
        .map_err(|error| invalid(field, format!("invalid regex: {error}")))?;
    if regex.is_match("") {
        return Err(invalid(field, "pattern matches the empty string"));
    }
    Ok(regex)
}

fn compile_tag(args: &ContentArgs) -> Result<Tag, TemplateError> {
    if args.value_parser.name != "text" {
        return Err(invalid("tool_calls", "value_parser must be text"));
    }
    let regex = compile("tool_calls", &args.tag_pattern)?;
    let names: Vec<_> = regex.capture_names().flatten().collect();
    if !names.contains(&"key") || !names.contains(&"value") {
        return Err(invalid(
            "tool_calls",
            "tag_pattern requires `key` and `value` captures",
        ));
    }
    let close = tag_close_literal(&args.tag_pattern).ok_or_else(|| {
        invalid(
            "tool_calls",
            "tag_pattern must end with a literal that follows the `value` capture",
        )
    })?;
    Ok(Tag {
        regex,
        close,
        strip: args.value_parser.args.strip,
    })
}

/// The literal that ends `pattern`, if the `value` capture precedes it.
fn tag_close_literal(pattern: &str) -> Option<String> {
    let hir = regex_syntax::parse(pattern).ok()?;
    let HirKind::Concat(parts) = hir.kind() else {
        return None;
    };
    let last = parts
        .iter()
        .rposition(|part| !matches!(part.kind(), HirKind::Literal(_)))?;
    let mut literal = Vec::new();
    for part in &parts[last + 1..] {
        if let HirKind::Literal(bytes) = part.kind() {
            literal.extend_from_slice(&bytes.0);
        }
    }
    if literal.is_empty() || !parts[..=last].iter().any(captures_value) {
        return None;
    }
    String::from_utf8(literal).ok()
}

fn captures_value(hir: &Hir) -> bool {
    match hir.kind() {
        HirKind::Capture(capture) => {
            capture.name.as_deref() == Some("value") || captures_value(&capture.sub)
        }
        HirKind::Concat(parts) | HirKind::Alternation(parts) => parts.iter().any(captures_value),
        HirKind::Repetition(repetition) => captures_value(&repetition.sub),
        _ => false,
    }
}

/// Hash of the template with object keys sorted, so key order does not matter.
fn fingerprint(value: &Value) -> u64 {
    fn canonical(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by_key(|(key, _)| *key);
                Value::Object(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key.clone(), canonical(value)))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
            other => other.clone(),
        }
    }
    let mut hasher = DefaultHasher::new();
    canonical(value).to_string().hash(&mut hasher);
    hasher.finish()
}
