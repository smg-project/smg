use std::collections::{BTreeMap, BTreeSet};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    config::ParserConfig,
    error::ResponseTemplateError,
    schema::{FieldTemplate, ResponseTemplate, Transform},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "max_bytes", rename_all = "snake_case")]
pub enum DelimiterBound {
    Bounded(usize),
    Unbounded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelimiterMetadata {
    pub field: String,
    pub role: String,
    pub bound: DelimiterBound,
}

#[derive(Debug)]
pub(crate) struct CompiledTemplate {
    pub(crate) model_name: String,
    pub(crate) defaults: BTreeMap<String, Value>,
    pub(crate) start_anchor: Regex,
    pub(crate) fields: Vec<CompiledField>,
    pub(crate) metadata: Vec<DelimiterMetadata>,
}

#[derive(Debug)]
pub(crate) struct CompiledField {
    pub(crate) name: String,
    pub(crate) open: Regex,
    pub(crate) open_bound: DelimiterBound,
    pub(crate) closes: Vec<String>,
    pub(crate) content: ContentKind,
    pub(crate) repeats: bool,
    pub(crate) tag: Option<CompiledTag>,
    pub(crate) transform: Option<Transform>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContentKind {
    Text,
    XmlInline,
}

#[derive(Debug)]
pub(crate) struct CompiledTag {
    pub(crate) regex: Regex,
    pub(crate) bound: DelimiterBound,
    pub(crate) strip: bool,
    /// Matches an unfinished key capture and exposes it as `partial_key`.
    pub(crate) partial_key: Regex,
    /// Matches an unfinished value capture and exposes it as `partial_value`.
    pub(crate) partial_value: Regex,
    /// Literal terminal close following the value capture.
    pub(crate) value_close: String,
}

impl CompiledTemplate {
    pub(crate) fn new(
        model_name: String,
        template: ResponseTemplate,
        config: ParserConfig,
    ) -> Result<Self, ResponseTemplateError> {
        for (field, value) in [
            ("max_pending_bytes", config.max_pending_bytes),
            (
                "max_structured_field_bytes",
                config.max_structured_field_bytes,
            ),
            ("max_body_bytes", config.max_body_bytes),
        ] {
            if value == 0 {
                return Err(invalid(&model_name, field, value, "limit must be non-zero"));
            }
        }
        if template.fields.len() != 3
            || !template.fields.contains_key("thinking")
            || !template.fields.contains_key("content")
            || !template.fields.contains_key("tool_calls")
        {
            return Err(invalid(
                &model_name,
                "fields",
                0,
                "exactly thinking, content and tool_calls fields are required",
            ));
        }

        for (name, field) in &template.fields {
            validate_field_capability(&model_name, name, field)?;
        }
        for name in ["thinking", "content"] {
            if template
                .defaults
                .get(name)
                .is_some_and(|value| !value.is_string())
            {
                return Err(invalid(
                    &model_name,
                    name,
                    0,
                    "text-field default must be a string",
                ));
            }
        }
        if template
            .defaults
            .get("tool_calls")
            .is_some_and(|value| value.as_array().is_none_or(|calls| !calls.is_empty()))
        {
            return Err(invalid(
                &model_name,
                "tool_calls",
                0,
                "tool_calls default must be an empty array",
            ));
        }

        let ResponseTemplate {
            defaults,
            start_anchor_pattern,
            fields: template_fields,
        } = template;

        let (start_anchor, start_bound) =
            compile_delimiter(&model_name, "start_anchor_pattern", &start_anchor_pattern)?;
        let mut metadata = vec![DelimiterMetadata {
            field: "start_anchor_pattern".to_string(),
            role: "start_anchor".to_string(),
            bound: start_bound,
        }];
        let mut fields = Vec::with_capacity(template_fields.len());
        for (name, field) in template_fields {
            fields.push(compile_field(&model_name, name, field, &mut metadata)?);
        }
        Ok(Self {
            model_name,
            defaults,
            start_anchor,
            fields,
            metadata,
        })
    }
}

fn validate_field_capability(
    model_name: &str,
    name: &str,
    field: &FieldTemplate,
) -> Result<(), ResponseTemplateError> {
    let valid = match name {
        "thinking" | "content" => {
            field.content == "text"
                && field.content_args.is_none()
                && !field.repeats
                && field.transform.is_none()
        }
        "tool_calls" => {
            field.content == "xml-inline"
                && field.content_args.is_some()
                && field.repeats
                && field.transform.is_some()
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(
            model_name,
            name,
            0,
            "unsupported field/content capability combination",
        ))
    }
}

fn compile_field(
    model_name: &str,
    name: String,
    field: FieldTemplate,
    metadata: &mut Vec<DelimiterMetadata>,
) -> Result<CompiledField, ResponseTemplateError> {
    let (open, open_bound) = compile_delimiter(model_name, &name, &field.open_pattern)?;
    metadata.push(DelimiterMetadata {
        field: name.clone(),
        role: "open".to_string(),
        bound: open_bound,
    });
    let closes = field.close.as_slice().to_vec();
    if closes.is_empty() || closes.iter().any(String::is_empty) {
        return Err(invalid(
            model_name,
            &name,
            0,
            "close delimiters must be non-empty",
        ));
    }
    for close in &closes {
        metadata.push(DelimiterMetadata {
            field: name.clone(),
            role: "close".to_string(),
            bound: DelimiterBound::Bounded(close.len()),
        });
    }

    let content = match field.content.as_str() {
        "text" => ContentKind::Text,
        "xml-inline" => ContentKind::XmlInline,
        other => {
            return Err(invalid(
                model_name,
                &name,
                0,
                format!("unsupported content parser {other:?}"),
            ));
        }
    };
    let tag = match (content, field.content_args) {
        (ContentKind::Text, None) => None,
        (ContentKind::Text, Some(_)) => {
            return Err(invalid(
                model_name,
                &name,
                0,
                "text fields cannot have content_args",
            ));
        }
        (ContentKind::XmlInline, None) => {
            return Err(invalid(
                model_name,
                &name,
                0,
                "xml-inline fields require content_args",
            ));
        }
        (ContentKind::XmlInline, Some(args)) => {
            if args.value_parser.name != "text" {
                return Err(invalid(
                    model_name,
                    &name,
                    0,
                    "xml-inline value_parser must be text",
                ));
            }
            let (regex, bound) = compile_delimiter(model_name, &name, &args.tag_pattern)?;
            let captures: BTreeSet<_> = regex.capture_names().flatten().collect();
            if !captures.contains("key") || !captures.contains("value") {
                return Err(invalid(
                    model_name,
                    &name,
                    0,
                    "tag_pattern requires named key and value captures",
                ));
            }
            metadata.push(DelimiterMetadata {
                field: name.clone(),
                role: "tag".to_string(),
                bound,
            });
            let partial = derive_partial_tag_matchers(model_name, &name, &args.tag_pattern)?;
            Some(CompiledTag {
                regex,
                bound,
                strip: args.value_parser.args.strip,
                partial_key: partial.partial_key,
                partial_value: partial.partial_value,
                value_close: partial.value_close,
            })
        }
    };
    if content == ContentKind::XmlInline && open.capture_names().all(|item| item != Some("name")) {
        return Err(invalid(
            model_name,
            &name,
            0,
            "xml-inline open_pattern requires a named name capture",
        ));
    }

    Ok(CompiledField {
        name,
        open,
        open_bound,
        closes,
        content,
        repeats: field.repeats,
        tag,
        transform: field.transform,
    })
}

struct PartialTagMatchers {
    partial_key: Regex,
    partial_value: Regex,
    value_close: String,
}

fn derive_partial_tag_matchers(
    model_name: &str,
    field: &str,
    pattern: &str,
) -> Result<PartialTagMatchers, ResponseTemplateError> {
    let key = named_capture_span(pattern, "key")
        .ok_or_else(|| invalid(model_name, field, 0, "cannot derive key capture boundaries"))?;
    let value = named_capture_span(pattern, "value").ok_or_else(|| {
        invalid(
            model_name,
            field,
            0,
            "cannot derive value capture boundaries",
        )
    })?;
    if key.whole_end > value.whole_start {
        return Err(invalid(
            model_name,
            field,
            0,
            "key capture must precede value capture",
        ));
    }

    let prefix = &pattern[..key.whole_start];
    let key_expression = &pattern[key.inner_start..key.inner_end];
    let between = &pattern[key.whole_end..value.whole_start];
    let suffix = &pattern[value.whole_end..];
    let value_close = terminal_close_literal(suffix).ok_or_else(|| {
        invalid(
            model_name,
            field,
            0,
            "tag_pattern must end in a literal closing tag",
        )
    })?;

    let partial_key_pattern = format!(r"(?s)(?:{prefix})(?P<partial_key>{key_expression})?$");
    let partial_value_pattern =
        format!(r"(?s)(?:{prefix})(?:{key_expression})(?:{between})(?P<partial_value>.*)$");
    let partial_key = Regex::new(&partial_key_pattern).map_err(|error| {
        invalid(
            model_name,
            field,
            0,
            format!("cannot derive partial key matcher: {error}"),
        )
    })?;
    let partial_value = Regex::new(&partial_value_pattern).map_err(|error| {
        invalid(
            model_name,
            field,
            0,
            format!("cannot derive partial value matcher: {error}"),
        )
    })?;
    Ok(PartialTagMatchers {
        partial_key,
        partial_value,
        value_close,
    })
}

#[derive(Clone, Copy)]
struct CaptureSpan {
    whole_start: usize,
    inner_start: usize,
    inner_end: usize,
    whole_end: usize,
}

fn named_capture_span(pattern: &str, name: &str) -> Option<CaptureSpan> {
    let long_marker = format!("(?P<{name}>");
    let short_marker = format!("(?<{name}>");
    let (whole_start, marker_len) = pattern
        .find(&long_marker)
        .map(|start| (start, long_marker.len()))
        .or_else(|| {
            pattern
                .find(&short_marker)
                .map(|start| (start, short_marker.len()))
        })?;
    let inner_start = whole_start + marker_len;
    let mut depth = 1usize;
    let mut escaped = false;
    let mut in_class = false;
    for (relative, character) in pattern[inner_start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => depth += 1,
            ')' if !in_class => {
                depth -= 1;
                if depth == 0 {
                    let inner_end = inner_start + relative;
                    return Some(CaptureSpan {
                        whole_start,
                        inner_start,
                        inner_end,
                        whole_end: inner_end + 1,
                    });
                }
            }
            _ => {}
        }
    }
    None
}

fn terminal_close_literal(suffix: &str) -> Option<String> {
    let start = suffix.rfind("</")?;
    let candidate = suffix[start..]
        .strip_suffix('$')
        .unwrap_or_else(|| &suffix[start..]);
    let mut literal = String::with_capacity(candidate.len());
    let mut escaped = false;
    for character in candidate.chars() {
        if escaped {
            match character {
                '<' | '>' | '/' | ':' | '_' | '-' | '.' | '"' | '\'' | '\\' => {
                    literal.push(character);
                }
                _ => return None,
            }
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if matches!(
            character,
            '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
        ) {
            return None;
        } else {
            literal.push(character);
        }
    }
    if escaped || !literal.starts_with("</") || !literal.ends_with('>') {
        None
    } else {
        Some(literal)
    }
}

fn compile_delimiter(
    model_name: &str,
    field: &str,
    pattern: &str,
) -> Result<(Regex, DelimiterBound), ResponseTemplateError> {
    let regex = Regex::new(pattern)
        .map_err(|error| invalid(model_name, field, 0, format!("invalid regex: {error}")))?;
    if regex.is_match("") {
        return Err(invalid(
            model_name,
            field,
            0,
            "delimiter pattern matches the empty string",
        ));
    }
    let hir = regex_syntax::parse(pattern)
        .map_err(|error| invalid(model_name, field, 0, format!("invalid regex: {error}")))?;
    let bound = match hir.properties().maximum_len() {
        Some(maximum) => DelimiterBound::Bounded(maximum),
        None => DelimiterBound::Unbounded,
    };
    Ok((regex, bound))
}

pub(crate) fn invalid(
    model_name: &str,
    field: &str,
    limit: usize,
    reason: impl Into<String>,
) -> ResponseTemplateError {
    ResponseTemplateError::InvalidTemplate {
        model_name: model_name.to_string(),
        field: field.to_string(),
        limit,
        reason: reason.into(),
    }
}
